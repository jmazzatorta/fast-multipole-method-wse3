#!/usr/bin/env python3
"""
appliance_compile.py — Compile the FMM-2.5D WSE-3 kernel for the Cerebras
Wafer-Scale Cluster (appliance mode) on ALCF.

Submits the compile job to the Kubernetes-managed appliance via
cerebras.sdk.client.SdkCompiler, then organizes all compile artifacts into
./compiled/<LABEL>/.

The FMM-2.5D mesh is cells_per_side x cells_per_side (direct cartesian plane,
NO Morton). Each PE hosts a column of cells_per_side cells along Z.

Usage:
    python3 appliance_compile.py CELLS_PER_SIDE MAX_BODIES_PE --label LABEL

Example (C=16, up to 8 bodies/cell):
    python3 appliance_compile.py 16 8 --label C16_MB8

Output (in ./compiled/<LABEL>/):
    artifact_path.json, cs_<hash>.tar.gz, out/, compile.log, compile_meta.json
"""
import argparse
import datetime
import io
import json
import logging
import os
import shutil
import socket
import sys

from cerebras.sdk.client import SdkCompiler


WSE3_FABRIC_DIMS = os.environ.get("CSL_FABRIC_DIMS", "762,1172")
WSE3_FABRIC_OFFSETS = "4,1"


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
                        help="C: domain side in cells = PE mesh side (power of 2).")
    parser.add_argument("max_bodies_pe", type=int,
                        help="Max bodies per leaf cell.")
    parser.add_argument("--label", required=True,
                        help="Config label (e.g. C16_MB8). Output -> ./compiled/<label>/.")
    parser.add_argument("--aliases", default="")
    parser.add_argument("--source-dir", default=".",
                        help="Directory containing layout.csl and pe.csl.")
    parser.add_argument("--top-file", default="layout.csl")
    parser.add_argument("--compiled-dir", default="compiled")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    C = args.cells_per_side
    assert (C & (C - 1)) == 0, "cells_per_side must be a power of 2"
    total_pe = C * C

    out_dir = os.path.join(args.compiled_dir, args.label)
    os.makedirs(out_dir, exist_ok=True)

    log_buf = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_buf)

    # FMM-2.5D compile-time params. cells_per_side is a param so the host can
    # read `side` from out.json (mesh side == cells_per_side). prob_range is a
    # const in the CSL (512.0); max_bodies_pe and memcpy ids are params.
    cslc_opts = (
        f"--arch=wse3 "
        f"--fabric-dims={WSE3_FABRIC_DIMS} "
        f"--fabric-offsets={WSE3_FABRIC_OFFSETS} "
        f"--params=cells_per_side:{C} "
        f"--params=max_bodies_pe:{args.max_bodies_pe} "
        f"--params=MEMCPY_H2D_ID:0 "
        f"--params=MEMCPY_D2H_ID:1 "
        f"--max-inlined-iterations 500 "
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
    print(f"  Top file      : {args.top_file}")
    print(f"  Arch          : wse3 (hardware)")
    print(f"  Fabric        : {WSE3_FABRIC_DIMS}")
    print(f"  cells_per_side: {C}  (mesh {C}x{C} = {total_pe} PE)")
    print(f"  max_bodies_pe : {args.max_bodies_pe}")
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
            args.source_dir,
            args.top_file,
            cslc_opts,
            args.source_dir,
        )

    print()
    print(f"[appliance_compile] Compile complete.")
    print(f"[appliance_compile] artifact_path = {artifact_path}")

    # Move artifacts into out_dir.
    artifact_basename = os.path.basename(artifact_path)
    local_artifact = os.path.join(args.source_dir, artifact_basename)
    if os.path.exists(local_artifact):
        dst = os.path.join(out_dir, artifact_basename)
        if os.path.abspath(local_artifact) != os.path.abspath(dst):
            shutil.move(local_artifact, dst)
            print(f"[appliance_compile] Moved {artifact_basename} -> {dst}")

    src_out = os.path.join(args.source_dir, "out")
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


if __name__ == "__main__":
    main()
