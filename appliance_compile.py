#!/usr/bin/env python3
"""
appliance_compile.py — Compile the FMM-2.5D WSE-3 kernel for the Cerebras
Wafer-Scale Cluster (appliance mode) on ALCF.

Submits the compile job to the Kubernetes-managed appliance via
cerebras.sdk.client.SdkCompiler, then organizes all compile artifacts into
./compiled/<LABEL>/.

The domain is C^3 leaf cells (C = cells_per_side). Every plane column (x, y)
of C cells along Z is cut into C / ZP chunks of ZP cells, one per PE, laid out
by pe_map.py: rectangle (C * BW) x (C * BH) PEs. --zp C (the default) is one PE
per column, the C x C mesh.

Usage:
    python3 appliance_compile.py CELLS_PER_SIDE MAX_BODIES_PE [--P ORDER] [--zp ZP] --label LABEL

Examples (C=16, up to 8 bodies/cell, P=4 / C=32, MB4, P=4 / C=16, P=5 / 32^3
with 8 cells per PE = 64 x 64 PEs):
    python3 appliance_compile.py 16 8 --label C16_MB8
    python3 appliance_compile.py 32 4 --label C32_MB4
    python3 appliance_compile.py 16 8 --P 5 --label C16_MB8_P5
    python3 appliance_compile.py 32 4 --zp 8 --label C32_MB4_Z8

Output (in ./compiled/<LABEL>/):
    artifact_path.json, cs_<hash>.tar.gz, out/, compile.log, compile_meta.json

The compile runs from a STAGING directory (<staging-dir>/<label>/) holding only
the *.csl files of --source-dir: the SDK uploads the whole app directory, so
compiling from the repository root would also ship compiled/, runs/...
"""
import argparse
import datetime
import glob
import io
import json
import logging
import os
import shutil
import socket
import sys

from cerebras.sdk.client import SdkCompiler


WSE3_FABRIC_DIMS = os.environ.get("CSL_FABRIC_DIMS", "762,1172")
WSE3_FABRIC_OFFSETS = "4,1"     # default; --offsets X,Y moves the rectangle


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


def main():
    parser = argparse.ArgumentParser(
        description="Compile FMM-2.5D WSE-3 kernel for appliance hardware."
    )
    parser.add_argument("cells_per_side", type=int,
                        help="C: domain side in leaf cells (power of 2).")
    parser.add_argument("max_bodies_pe", type=int,
                        help="Max bodies per leaf cell.")
    parser.add_argument("--P", type=int, default=4, help="expansion order (default 4)")
    parser.add_argument("--block-lbw", type=int, default=None,
                        help="log2 of the width of the block of PEs of a column "
                             "(default: square-ish, pe_map.block_shape; 0 = 1 x NK, "
                             "narrowest rectangle)")
    parser.add_argument("--offsets", default=WSE3_FABRIC_OFFSETS,
                        help=f"fabric offsets X,Y of the program rectangle "
                             f"(default {WSE3_FABRIC_OFFSETS})")
    parser.add_argument("--zp", type=int, default=None,
                        help="leaf cells per PE along z, a power of two dividing C "
                             "(default C: one PE per column)")
    parser.add_argument("--label", required=True,
                        help="Config label (e.g. C16_MB8). Output -> ./compiled/<label>/.")
    parser.add_argument("--aliases", default="")
    parser.add_argument("--source-dir", default=".",
                        help="Directory containing layout.csl and pe.csl.")
    parser.add_argument("--top-file", default="layout.csl")
    parser.add_argument("--compiled-dir", default="compiled")
    parser.add_argument("--staging-dir", default="staging")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    C = args.cells_per_side
    assert C >= 8 and (C & (C - 1)) == 0, "cells_per_side must be a power of 2 >= 8"
    tree_h = C.bit_length() - 1
    zp = C if args.zp is None else args.zp
    assert 1 <= zp <= C and (zp & (zp - 1)) == 0, "--zp: a power of two dividing C"
    import pe_map
    pmap = pe_map.PeMap(C, C, C, zp, args.block_lbw)
    total_pe = pmap.width * pmap.height
    fab_w, fab_h = (int(v) for v in WSE3_FABRIC_DIMS.split(","))
    off_x, off_y = (int(v) for v in args.offsets.split(","))
    assert off_x + pmap.width <= fab_w and off_y + pmap.height <= fab_h, \
        f"rectangle {pmap.width}x{pmap.height} at {args.offsets} outside the fabric {WSE3_FABRIC_DIMS}"

    out_dir = os.path.join(args.compiled_dir, args.label)
    os.makedirs(out_dir, exist_ok=True)

    # Staging: only the .csl sources (see the module docstring).
    stage = os.path.abspath(os.path.join(args.staging_dir, args.label))
    if os.path.exists(stage):
        shutil.rmtree(stage)
    os.makedirs(stage)
    csl_files = sorted(glob.glob(os.path.join(args.source_dir, "*.csl")))
    assert csl_files, f"no .csl in {args.source_dir}"
    for f in csl_files:
        shutil.copy(f, stage)

    log_buf = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_buf)

    # FMM-2.5D compile-time params: the runner reads them back from out.json
    # (cells_per_side, order_p, zp_log, ...). prob_range is a const in the CSL
    # (512.0).
    # Operator table sizes for order P, and the size of the transport table:
    # compile-time parameters, computed by the same host modules the runner uses.
    import fmm_passes
    import fmm_tables
    import zmp_tables
    zmp_tables.configure(C, args.P, zp, args.block_lbw)
    fmm_tables.set_order(args.P)
    tsz = fmm_tables.table_sizes()
    mp_ent_cap = zmp_tables.stats(args.max_bodies_pe)[1]

    cslc_opts = (
        f"--arch=wse3 "
        f"--fabric-dims={WSE3_FABRIC_DIMS} "
        f"--fabric-offsets={args.offsets} "
        f"--params=cells_per_side:{C} "
        f"--params=max_bodies_pe:{args.max_bodies_pe} "
        f"--params=tree_h:{tree_h} "
        f"--params=order_p:{args.P} "
        f"--params=m2m_n_entries:{tsz['m2m']} "
        f"--params=m2l_n_entries:{tsz['m2l']} "
        f"--params=l2l_n_entries:{tsz['l2l']} "
        f"--params=mp_ent_cap:{mp_ent_cap} "
        f"--params=zp_log:{pmap.lzp} "
        f"--params=blk_lbw:{pmap.lbw} "
        f"--params=blk_lbh:{pmap.lbh} "
        f"--params=MEMCPY_H2D_ID:0 "
        f"--params=MEMCPY_D2H_ID:1 "
        # Loop-unroll budget. 500 fully unrolls the big FMM loops (M2L 375,
        # M2M/L2L 92) and blows past per-PE code memory; 1 is too low for the
        # system libraries (simprint's inline byte loop needs a few). 16 lets the
        # short library loops unroll while the large FMM loops (>16) stay as real
        # loops, keeping .text/.data.hi small.
        f"--max-inlined-iterations 16 "
        f"-o out --memcpy --channels 1"
    )

    print("=" * 64)
    print("FMM-2.5D WSE-3 APPLIANCE COMPILE")
    print("=" * 64)
    print(f"  Date          : {datetime.datetime.now().isoformat()}")
    print(f"  Host          : {socket.gethostname()}")
    print(f"  Label         : {args.label}")
    if args.aliases:
        print(f"  Aliases       : {args.aliases}")
    print(f"  Source dir    : {args.source_dir}")
    print(f"  Staging dir   : {stage}  ({', '.join(os.path.basename(f) for f in csl_files)})")
    print(f"  Top file      : {args.top_file}")
    print(f"  Arch          : wse3 (hardware)")
    print(f"  Fabric        : {WSE3_FABRIC_DIMS}")
    print(f"  cells_per_side: {C}  (domain {C}^3, tree height {tree_h})")
    print(f"  z split       : ZP {zp} cells per PE, {pmap.nk} PE per column "
          f"(block {pmap.bw}x{pmap.bh}), rectangle {pmap.width}x{pmap.height} = {total_pe} PE "
          f"at fabric offsets {args.offsets}")
    print(f"  max_bodies_pe : {args.max_bodies_pe}")
    print(f"  P (order)     : {args.P}   operator tables M2M {tsz['m2m']} / "
          f"M2L {tsz['m2l']} / L2L {tsz['l2l']} entries")
    print(f"  passes        : {fmm_passes.NPASS}  "
          f"({' '.join(p[0] for p in fmm_passes.PASSES)})")
    print(f"  mp_ent_cap    : {mp_ent_cap}")
    print(f"  cslc opts     : {cslc_opts}")
    print(f"  Out dir       : {out_dir}")
    print()
    print("[appliance_compile] Submitting compile job to appliance...")

    with SdkCompiler(
        disable_version_check=True,
        resource_mem=256 << 30,
        resource_cpu=48000,
    ) as compiler:
        artifact_path = compiler.compile(
            stage,
            args.top_file,
            cslc_opts,
            stage,
        )

    print()
    print(f"[appliance_compile] Compile complete.")
    print(f"[appliance_compile] artifact_path = {artifact_path}")

    # Move artifacts into out_dir.
    artifact_basename = os.path.basename(artifact_path)
    local_artifact = os.path.join(stage, artifact_basename)
    dst = os.path.join(out_dir, artifact_basename)
    if os.path.exists(local_artifact):
        shutil.move(local_artifact, dst)
        print(f"[appliance_compile] Moved {artifact_basename} -> {dst}")
    elif os.path.exists(artifact_path):
        shutil.copy(artifact_path, dst)
        print(f"[appliance_compile] Copied {artifact_path} -> {dst}")
    else:
        raise FileNotFoundError(f"new artifact not found ({artifact_path}); "
                                f"compiled/{args.label}/ left unchanged")
    # Only now that the new artifact is in place: a recompilation with the same
    # label replaces the previous one (appliance_run.py expects exactly one).
    for old in glob.glob(os.path.join(out_dir, "cs_*.tar.gz")):
        if os.path.basename(old) != artifact_basename:
            os.remove(old)
            print(f"[appliance_compile] Removed previous artifact {os.path.basename(old)}")

    src_out = os.path.join(stage, "out")
    dst_out = os.path.join(out_dir, "out")
    if os.path.exists(src_out) and os.path.abspath(src_out) != os.path.abspath(dst_out):
        if os.path.exists(dst_out):
            shutil.rmtree(dst_out)
        shutil.move(src_out, dst_out)
        print(f"[appliance_compile] Moved out/ -> {dst_out}")

    apj = os.path.join(out_dir, "artifact_path.json")
    with open(apj, "w", encoding="utf8") as f:
        json.dump({"artifact_path": artifact_path}, f)
    print(f"[appliance_compile] Wrote {apj}")

    meta = {
        "label": args.label,
        "aliases": args.aliases.split(",") if args.aliases else [],
        "date": datetime.datetime.now().isoformat(),
        "host": socket.gethostname(),
        "config": {
            "cells_per_side": C,
            "max_bodies_pe": args.max_bodies_pe,
            "P": args.P,
            "tree_h": tree_h,
            "zp": zp,
            "block_lbw": pmap.lbw,
            "offsets": args.offsets,
            "mp_ent_cap": mp_ent_cap,
            "total_pe": total_pe,
        },
        "cslc_options": cslc_opts,
        "artifact_path": artifact_path,
    }
    meta_path = os.path.join(out_dir, "compile_meta.json")
    with open(meta_path, "w", encoding="utf8") as f:
        json.dump(meta, f, indent=2)
    print(f"[appliance_compile] Wrote {meta_path}")

    sys.stdout = original_stdout
    log_path = os.path.join(out_dir, "compile.log")
    with open(log_path, "w", encoding="utf8") as f:
        f.write(log_buf.getvalue())
    print(f"[appliance_compile] Wrote {log_path}")
    shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    main()
