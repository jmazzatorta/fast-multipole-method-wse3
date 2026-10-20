#!/usr/bin/env python3
"""
appliance_run.py — Runner for the FMM-2.5D benchmark on CS-3 (appliance mode).

Flow:
  1. Resolve artifact from compiled/<LABEL>/, extract to extracted/<LABEL>/.
  2. Read side (= cells_per_side) and max_bodies_pe from out/out.json.
  3. Build the operator tables (fmm_tables.py) and the serpentine pass table
     (serp_tables.py).
  4. Bin bodies into the C x C column layout (cartesian, NO Morton).
  5. Push math tables (broadcast) and bodies (per-PE).
  6. Launch the FMM steps in sequence. Each PE writes the step id into
     fmm_done on TRUE completion; after each launch the host polls fmm_done
     until all PEs report the step (or a timeout): the global barrier.
        run_p2m (1) -> run_pass x8 (2..9: M2M43 M2M32 M2L4 M2L3 M2L2 L2L23
        L2L34 P2P) -> run_l2p (10)
     After every pass the per-PE counters are checked against the pass table;
     l_lvl4 is checked against fmm_ref after M2L4 and after L2L34.
  7. Pull forces_col, reassemble, compare to the direct N-body sum.

Usage:
    python appliance_run.py --label C16_MB8 [--simulator] [--bodies file.npy]

Must run with the appliance venv python.
"""
import argparse
import datetime
import glob
import io
import json
import os
import re
import socket
import sys
import tarfile
import time

import numpy as np

from cerebras.sdk.client import SdkRuntime
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

FLOATS_PER_BODY = 4


class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, s):
        for st in self.streams:
            st.write(s); st.flush()
    def flush(self):
        for st in self.streams:
            st.flush()


# ===========================================================================
# Memcpy helpers (appliance SDK)
# ===========================================================================
def broadcast(side, arr):
    """Replicate a 1D array to every PE: shape (side, side, len)."""
    n = arr.shape[0]
    return np.broadcast_to(arr, (side, side, n)).copy()

def h2d_bcast_u16(runner, sym, arr_u16, side):
    """u16 device buffer, broadcast. Host tensor is u32 zero-extended,
    pushed as MEMCPY_16BIT."""
    n = arr_u16.shape[0]
    t = broadcast(side, arr_u16.astype(np.uint32))
    runner.memcpy_h2d(sym, t.ravel(), 0, 0, side, side, n,
                      streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                      nonblock=False, order=MemcpyOrder.ROW_MAJOR)

def h2d_bcast_u32(runner, sym, arr_u32, side):
    """u32 device buffer, broadcast (e.g. packed signs)."""
    n = arr_u32.shape[0]
    t = broadcast(side, arr_u32.astype(np.uint32))
    runner.memcpy_h2d(sym, t.ravel(), 0, 0, side, side, n,
                      streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False, order=MemcpyOrder.ROW_MAJOR)

def h2d_per_pe_u32(runner, sym, tensor, side, words_per_pe):
    runner.memcpy_h2d(sym, tensor.ravel(), 0, 0, side, side, words_per_pe,
                      streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False, order=MemcpyOrder.ROW_MAJOR)

def h2d_per_pe_u16(runner, sym, tensor_u16, side, words_per_pe):
    """u16 device buffer, per-PE data. Host tensor (side, side, n) is u32
    zero-extended, pushed as MEMCPY_16BIT."""
    t = np.ascontiguousarray(tensor_u16.astype(np.uint32))
    runner.memcpy_h2d(sym, t.ravel(), 0, 0, side, side, words_per_pe,
                      streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
                      nonblock=False, order=MemcpyOrder.ROW_MAJOR)

def d2h_per_pe_u32(runner, sym, side, words_per_pe):
    out = np.zeros(side * side * words_per_pe, dtype=np.uint32)
    runner.memcpy_d2h(out, sym, 0, 0, side, side, words_per_pe,
                      streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False, order=MemcpyOrder.ROW_MAJOR)
    return out.reshape(side, side, words_per_pe)

def d2h_per_pe_f32(runner, sym, side, words_per_pe):
    out = np.zeros(side * side * words_per_pe, dtype=np.float32)
    runner.memcpy_d2h(out, sym, 0, 0, side, side, words_per_pe,
                      streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                      nonblock=False, order=MemcpyOrder.ROW_MAJOR)
    return out.reshape(side, side, words_per_pe)


# ===========================================================================
# Artifact resolution (mirror the reference appliance_run)
# ===========================================================================
def resolve_artifact(compiled_dir, label):
    pattern = os.path.join(compiled_dir, label, "cs_*.tar.gz")
    cands = glob.glob(pattern)
    if not cands:
        raise FileNotFoundError(f"No artifact at {pattern}. Run appliance_compile first.")
    if len(cands) > 1:
        raise RuntimeError(f"Multiple artifacts in compiled/{label}/: {cands}")
    return cands[0]

def ensure_extracted(artifact, extracted_dir, label):
    label_dir = os.path.join(extracted_dir, label)
    os.makedirs(label_dir, exist_ok=True)
    inner = glob.glob(os.path.join(label_dir, "cs_*", "out"))
    if inner:
        return inner[0]
    # Extract only <cs_hash>/out/ (out.json + binaries): the artifact also
    # carries a copy of the whole source directory, which the runner never needs
    # (and which, in older artifacts, nests every previous build).
    with tarfile.open(artifact, "r:gz") as tar:
        members = [m for m in tar.getmembers()
                   if len(m.name.split("/")) >= 2 and m.name.split("/")[1] == "out"]
        tar.extractall(path=label_dir, members=members)
    inner = glob.glob(os.path.join(label_dir, "cs_*", "out"))
    if not inner:
        raise RuntimeError(f"No cs_*/out after extracting into {label_dir}")
    return inner[0]


# ===========================================================================
# Body binning: cartesian column layout (NO Morton)
# ===========================================================================
def build_bodies_tensor(bodies, C, max_bodies_pe):
    """
    bodies: (N,4) float32 (x,y,z,q) in physical domain [0, prob_range).
    Returns per-PE flat tensor shaped (C, C, CELL_BLOCK*C) where each PE holds a
    column of C cells, each cell block = [count | body0(x,y,z,q) | ...].
    A body at cell (cx,cy,cz) lives on PE(cx,cy) at column index cz.
    """
    CELL_BLOCK = 1 + max_bodies_pe * FLOATS_PER_BODY
    col_len = C * CELL_BLOCK
    flat = np.zeros((C, C, col_len), dtype=np.uint32)
    # per-cell running counts
    counts = {}
    # Cell index from physical coordinate: cell = floor(coord / cell_size),
    # cell_size = prob_range / C. prob_range is 512.0 (const in the CSL).
    PROB_RANGE = 512.0
    cell_size = PROB_RANGE / C
    for (x, y, z, q) in bodies:
        cx = min(max(int(x / cell_size), 0), C - 1)
        cy = min(max(int(y / cell_size), 0), C - 1)
        cz = min(max(int(z / cell_size), 0), C - 1)
        key = (cx, cy, cz)
        k = counts.get(key, 0)
        if k >= max_bodies_pe:
            raise ValueError(f"cell {key} exceeds max_bodies_pe={max_bodies_pe}")
        base = cz * CELL_BLOCK
        off = base + 1 + k * FLOATS_PER_BODY
        flat[cy, cx, off + 0] = np.float32(x).view(np.uint32)
        flat[cy, cx, off + 1] = np.float32(y).view(np.uint32)
        flat[cy, cx, off + 2] = np.float32(z).view(np.uint32)
        flat[cy, cx, off + 3] = np.float32(q).view(np.uint32)
        counts[key] = k + 1
    # write counts into each cell block slot 0
    for (cx, cy, cz), k in counts.items():
        base = cz * CELL_BLOCK
        flat[cy, cx, base] = np.uint32(k)
    return flat, col_len, CELL_BLOCK


# ===========================================================================
# Direct N-body reference
# ===========================================================================
def direct_nbody(bodies):
    N = len(bodies)
    pos = bodies[:, :3].astype(np.float64)
    q = bodies[:, 3].astype(np.float64)
    F = np.zeros((N, 3)); pot = np.zeros(N)
    for i in range(N):
        d = pos - pos[i]
        r2 = (d * d).sum(axis=1)
        r2[i] = np.inf
        rinv = 1.0 / np.sqrt(r2)
        rinv3 = rinv ** 3
        F[i] = (q[:, None] * rinv3[:, None] * d).sum(axis=0)
        pot[i] = (q * rinv).sum()
    return F, pot


# ===========================================================================
# Serpentine pass verification
# ===========================================================================
def expected_m2l_translations(C, level):
    """Per target PE: number of (target box, source box) pairs of the FMM
    interaction list at `level` (parents adjacent, boxes not adjacent), incl.
    the own column. Only the level's representatives have any. This is what
    serp_stats[4] must count in an M2L pass."""
    s = C >> level                   # PEs per box side (TREE_H = log2 C)
    nb = 1 << level                  # boxes per side
    exp = np.zeros((C, C), dtype=np.int64)
    for ty in range(nb):
        for tx in range(nb):
            n = 0
            for sy in range(max(0, ty - 3), min(nb, ty + 4)):
                for sx in range(max(0, tx - 3), min(nb, tx + 4)):
                    if abs(tx // 2 - sx // 2) > 1 or abs(ty // 2 - sy // 2) > 1:
                        continue
                    for tz in range(nb):
                        for sz in range(nb):
                            if abs(tz // 2 - sz // 2) > 1:
                                continue
                            if max(abs(tx - sx), abs(ty - sy), abs(tz - sz)) < 2:
                                continue
                            n += 1
            exp[ty * s, tx * s] = n
    return exp


def expected_work(C, k):
    """serp_stats[4] per PE for pass k: M2L translations, boxes added/copied, or
    (P2P) (target cell, source cell) pairs."""
    import serp_tables as T
    name = T.PASSES[k][0]
    if name.startswith("M2L"):
        return expected_m2l_translations(C, int(name[3]))
    exp = np.zeros((C, C), dtype=np.int64)
    if name == "P2P":
        # own + neighbour columns, each source cell acts on the target cells at
        # |dz| <= 1 (3C - 2 pairs per column)
        for y in range(C):
            for x in range(C):
                ncol = (min(x + 1, C - 1) - max(x - 1, 0) + 1) * \
                       (min(y + 1, C - 1) - max(y - 1, 0) + 1)
                exp[y, x] = ncol * (3 * C - 2)
        return exp
    for y in range(C):
        for x in range(C):
            if name == "M2M43" and T.is_rep(x, y, 3):
                exp[y, x] = 3 * 8            # 3 block mates x 8 boxes (m_lvl3)
            elif name == "M2M32" and T.is_rep(x, y, 2):
                exp[y, x] = 3 * 4            # 3 level-3 reps x 4 boxes (m_lvl2)
            elif name == "L2L23" and T.is_rep(x, y, 3) and not T.is_rep(x, y, 2):
                exp[y, x] = 4                # l_lvl2: 4 boxes
            elif name == "L2L34" and not T.is_rep(x, y, 3):
                exp[y, x] = 8                # l_lvl3: 8 boxes
    return exp


def compare_l4(l4_dev, ref, C, what):
    """Compare device l_lvl4 with the reference.

    PASS criterion: max |device - ref| / (max |ref| of that PE) <= 1e-5. f32
    summation in a different order gives ~1e-7..1e-6 on this scale, while a
    lost, duplicated or corrupted message changes a whole M2L contribution and
    shows up at >= 1e-3 of the PE max.
    Also printed (informational): the error per coefficient, each scaled by its
    own max over z. With the current (wrong) M2L operator the terms reach 1e16
    and cancel down to ~1e8 in some coefficients, so that view is dominated by
    f32 cancellation and is not a pass criterion until the math is fixed."""
    d4 = l4_dev.astype(np.float64).reshape(C, C, C, 30)
    r4 = ref.astype(np.float64).reshape(C, C, C, 30)
    if not np.all(np.isfinite(r4)):
        print(f"  l_lvl4 {what}: reference not finite, comparison skipped")
        return True
    pe_max = np.abs(r4).max(axis=(2, 3), keepdims=True) + 1e-30
    rel_pe = (np.abs(d4 - r4) / pe_max).max(axis=(2, 3))
    ok = bool(rel_pe.max() <= 1e-5)
    wy, wx = np.unravel_index(int(np.argmax(rel_pe)), rel_pe.shape)
    print(f"  l_lvl4 {what}: error / PE max: max {rel_pe.max():.3e} at PE(px,py)=({wx},{wy}), "
          f"median {np.median(rel_pe):.3e}; PEs above 1e-5: {int((rel_pe > 1e-5).sum())}"
          f"  -> {'OK' if ok else 'MISMATCH'}")
    scale = np.maximum(np.abs(r4).max(axis=2, keepdims=True), 1e-6 * pe_max)
    relx = np.abs(d4 - r4) / scale
    py, px, z, k = np.unravel_index(int(np.argmax(relx)), relx.shape)
    print(f"      (info) per-coefficient rel err max {relx.max():.3e} at PE({px},{py}) "
          f"z={z} coeff={k}: device {d4[py, px, z, k]:.4e} ref {r4[py, px, z, k]:.4e}")
    return ok


def physics_check_m2l4(l4_dev, bodies, C, max_bodies_pe):
    """Physical check of the device level-4 M2L, independent of fmm_ref's M2L:
    potential from L2P of the DEVICE l_lvl4 (after pass M2L4, so it holds only
    level-4 interactions) vs the direct sum over the bodies of the level-4
    interaction-list cells (not adjacent, parents adjacent). Expected for P=4:
    median ~1e-4, max ~1e-3."""
    import fmm_ref
    binned = fmm_ref.bin_bodies(bodies, max_bodies=max_bodies_pe)
    where = binned[2]
    # device layout (py, px, z*30+k) -> fmm_ref layout [x, y, z, k]
    l4 = np.transpose(l4_dev.reshape(C, C, C, 30), (1, 0, 2, 3)).astype(np.float32)
    out = fmm_ref.l2p(binned, l4)
    pot_dev = np.array([out[w[0], w[1], w[2], w[3], 3] for w in where], dtype=np.float64)
    cell = 512.0 / C
    ci = (bodies[:, :3] // cell).astype(int).clip(0, C - 1)
    pos = bodies[:, :3].astype(np.float64)
    pot_dir = np.zeros(len(bodies))
    for i in range(len(bodies)):
        m = (np.max(np.abs(ci - ci[i]), axis=1) >= 2) & \
            (np.max(np.abs(ci // 2 - ci[i] // 2), axis=1) <= 1)
        pot_dir[i] = (bodies[m, 3] / np.linalg.norm(pos[m] - pos[i], axis=1)).sum()
    rel = np.abs(pot_dev - pot_dir) / np.abs(pot_dir)
    ok = bool(np.median(rel) < 1e-3 and rel.max() < 1e-2)
    print(f"  physics M2L4 (device l_lvl4 -> L2P vs direct sum on the level-4 list): "
          f"potential rel err median {np.median(rel):.2e} max {rel.max():.2e}"
          f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def device_forces_per_body(forces_3d, bodies, C, max_bodies_pe):
    """(N,4) [fx,fy,fz,pot] from forces_col, in the input order of `bodies`
    (same slot assignment as build_bodies_tensor)."""
    out = np.zeros((len(bodies), 4))
    counts = {}
    cell = 512.0 / C
    for bi, (x, y, z, q) in enumerate(bodies):
        cx = min(max(int(x / cell), 0), C - 1)
        cy = min(max(int(y / cell), 0), C - 1)
        cz = min(max(int(z / cell), 0), C - 1)
        slot = counts.get((cx, cy, cz), 0); counts[(cx, cy, cz)] = slot + 1
        fb = cz * (max_bodies_pe * FLOATS_PER_BODY) + slot * FLOATS_PER_BODY
        out[bi] = forces_3d[cy, cx, fb:fb + 4]
    return out


def physics_check_forces(forces_3d, bodies, C, max_bodies_pe):
    """Device forces/potentials (complete FMM: far field + P2P near field) vs
    the direct N-body sum. Expected for P=4: force rel err median ~1e-3."""
    dev = device_forces_per_body(forces_3d, bodies, C, max_bodies_pe)
    Fr, pr = direct_nbody(bodies)
    ferr = np.linalg.norm(dev[:, :3] - Fr, axis=1) / (np.linalg.norm(Fr, axis=1) + 1e-30)
    perr = np.abs(dev[:, 3] - pr) / (np.abs(pr) + 1e-30)
    ok = bool(np.median(ferr) < 1e-2 and np.median(perr) < 1e-3)
    print(f"  physics forces (device FMM vs direct N-body): force rel err "
          f"median {np.median(ferr):.2e} rms {np.sqrt((ferr**2).mean()):.2e} max {ferr.max():.2e}; "
          f"pot median {np.median(perr):.2e} max {perr.max():.2e}  -> {'OK' if ok else 'FAIL'}")
    return ok


def verify_passes(pass_stats, l4_checkpoints, serp_tab, bodies, C, max_bodies_pe):
    import serp_tables as T
    print("\n--- VERIFICATION: serpentine passes ---")
    allok = True
    for k in sorted(pass_stats):
        st = pass_stats[k]
        row = serp_tab[:, :, k * T.NF:(k + 1) * T.NF].astype(np.int64)
        bad = []
        for i, col in enumerate((4, 6, 5, 7)):      # drains F/B, emits F/B
            n = int((st[:, :, i] != row[:, :, col]).sum())
            if n:
                bad.append(f"{['drainsF', 'drainsB', 'emitsF', 'emitsB'][i]}:{n} PEs")
        ndone = int((st[:, :, 6] == 1).sum())
        stray = int(st[:, :, 5].sum()); perr = int(st[:, :, 7].sum())
        wexp = expected_work(C, k)
        nwork = int((st[:, :, 4] != wexp).sum())
        ok = not bad and ndone == C * C and stray == 0 and perr == 0 and nwork == 0
        allok &= ok
        print(f"  pass {k} {T.PASSES[k][0]:6s}: done {ndone}/{C*C}  arrivals "
              f"{int(st[:, :, 0].sum() + st[:, :, 1].sum()):6d}/"
              f"{int(row[:, :, 4].sum() + row[:, :, 6].sum()):6d}  work "
              f"{int(st[:, :, 4].sum()):6d}/{int(wexp.sum()):6d}  stray {stray}  "
              f"proto-err {perr}  {'OK' if ok else 'FAIL ' + ', '.join(bad) + (f' work:{nwork} PEs' if nwork else '')}")
    try:
        import fmm_ref
        binned = fmm_ref.bin_bodies(bodies, max_bodies=max_bodies_pe)
        if 2 in l4_checkpoints:
            # device math: M2L corrected (irregular harmonics), see fmm_tables
            ref = fmm_ref.m2l_level4_only(binned, ops="fixed")
            allok &= compare_l4(l4_checkpoints[2], ref, C, "after M2L4 ")
            allok &= physics_check_m2l4(l4_checkpoints[2], bodies, C, max_bodies_pe)
        if 6 in l4_checkpoints:
            # Whole pipeline (M2M, M2L at 3 levels, L2L) vs the reference with
            # the device math as written.
            mlev = fmm_ref.upward(fmm_ref.p2m(binned))
            # device math: corrected M2L and L2L
            ref = fmm_ref.to_device_layout(
                fmm_ref.downward(mlev, m2l_ops="fixed", l2l_ops="fixed")[4])
            allok &= compare_l4(l4_checkpoints[6], ref, C, "after L2L34")
    except Exception as e:
        allok = False
        print(f"  l_lvl4 reference check FAILED to run: {e!r}")
    print("  RESULT:", "PASS" if allok else "FAIL")
    return allok


# ===========================================================================
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Run FMM-2.5D on CS-3.")
    parser.add_argument("--label", default=None)
    parser.add_argument("--bodies", default=None,
                        help="(N,4) float32 .npy (x,y,z,q). Default: a small test set.")
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--compiled-dir", default="compiled")
    parser.add_argument("--extracted-dir", default="extracted")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--phase-timeout", type=float, default=60.0,
                        help="seconds to wait for fmm_done after each phase")
    parser.add_argument("--nsteps", type=int, default=10,
                        help="run only the first N steps of the sequence (1..10)")
    parser.add_argument("--nbodies", type=int, default=2000,
                        help="random bodies when --bodies is not given")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--verify-only", default=None, metavar="RUN_DIR",
                        help="re-run the verification on RUN_DIR/device_data.npz "
                             "(no device needed)")
    args = parser.parse_args()
    if args.verify_only:
        d = np.load(os.path.join(args.verify_only, "device_data.npz"))
        ps = {int(k[6:]): d[k] for k in d.files if k.startswith("stats_")}
        cp = {int(k[3:]): d[k] for k in d.files if k.startswith("l4_")}
        ok = verify_passes(ps, cp, d["serp_tab"], d["bodies"], int(d["C"]),
                           int(d["max_bodies_pe"]))
        if "forces" in d.files:
            ok &= physics_check_forces(d["forces"], d["bodies"], int(d["C"]),
                                       int(d["max_bodies_pe"]))
        sys.exit(0 if ok else 1)
    if not args.label:
        parser.error("--label is required (unless --verify-only)")

    artifact = resolve_artifact(args.compiled_dir, args.label)
    out_dir_ex = ensure_extracted(artifact, args.extracted_dir, args.label)
    with open(os.path.join(out_dir_ex, "out.json")) as f:
        compile_data = json.load(f)
    C = int(compile_data["params"]["cells_per_side"])
    max_bodies_pe = int(compile_data["params"]["max_bodies_pe"])
    side = C
    total_pe = side * side

    # bodies
    if args.bodies and os.path.exists(args.bodies):
        bodies = np.load(args.bodies).astype(np.float32)
    else:
        # Random bodies in the domain (prob_range=512); re-draw any body whose
        # cell is already full (max_bodies_pe per cell).
        rng = np.random.default_rng(args.seed)
        cell = 512.0 / C
        fill = {}
        rows = []
        while len(rows) < args.nbodies:
            p = rng.uniform(0.0, 512.0, 3)
            key = tuple(min(int(v / cell), C - 1) for v in p)
            if fill.get(key, 0) >= max_bodies_pe:
                continue
            fill[key] = fill.get(key, 0) + 1
            rows.append([p[0], p[1], p[2], rng.uniform(0.5, 1.5)])
        bodies = np.array(rows, dtype=np.float32)
    N = len(bodies)
    import serp_tables
    # (step id, exported fn, label); pass k is step 2+k.
    STEPS = [(1, "run_p2m", "P2M")]
    STEPS += [(2 + k, "run_pass", f"pass {k} {serp_tables.PASSES[k][0]}")
              for k in range(serp_tables.NPASS)]
    STEP_L2P = 2 + serp_tables.NPASS            # 10
    STEPS += [(STEP_L2P, "run_l2p", "L2P")]
    STEPS = STEPS[:args.nsteps]

    # run dir
    if args.rep is None:
        rep = 1
        if os.path.isdir(args.runs_dir):
            pat = re.compile(rf"^{re.escape(args.label)}_run(\d+)$")
            used = [int(m.group(1)) for n in os.listdir(args.runs_dir)
                    if (m := pat.match(n))]
            rep = max(used) + 1 if used else 1
    else:
        rep = args.rep
    run_dir = os.path.join(args.runs_dir, f"{args.label}_run{rep:02d}")
    os.makedirs(run_dir, exist_ok=True)

    log_buf = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_buf)

    print("=" * 64)
    print("FMM-2.5D RUN")
    print(f"  Label  : {args.label}  Rep: {rep:02d}")
    print(f"  Target : {'simulator' if args.simulator else 'CS-3 hardware'}")
    print(f"  Mesh   : {side} x {side} = {total_pe} PE   (C={C})")
    print(f"  max_bodies_pe : {max_bodies_pe}")
    print(f"  Bodies : {N}")
    print(f"  Steps  : {' -> '.join(lbl for _, _, lbl in STEPS)}")
    print("=" * 64)

    # operator tables (generated on the fly, see fmm_tables.py)
    import fmm_tables
    math = fmm_tables.device_tables()
    for nm in ("m2m", "m2l", "l2l"):
        s = math[nm]["signs"]
        assert s.dtype == np.uint32 and s.max() <= 15, f"{nm} signs bad"

    print("[1/4] Binning bodies into column layout...")
    bodies_flat, col_len, CELL_BLOCK = build_bodies_tensor(bodies, C, max_bodies_pe)

    serp_tab = serp_tables.device_table(max_bodies_pe)   # (C, C, NPASS*10) u16
    print(f"      P2P: {serp_tables.p2p_cells(max_bodies_pe)} cells per message, "
          f"{serp_tables.pass_nmsgs(serp_tables.NPASS - 1, max_bodies_pe)} messages per column")
    assert serp_tab.shape == (C, C, serp_tables.NPASS * serp_tables.NF)

    err = None
    forces_3d = None
    pass_stats = {}          # pass index -> (C, C, 8) u32
    l4_checkpoints = {}      # pass index -> l_lvl4 after that pass
    completed = []
    try:
        with SdkRuntime(artifact, simulator=args.simulator,
                        disable_version_check=True) as runner:
            sym = {
                "bodies_col": runner.get_id("ptr_bodies_col"),
                "forces_col": runner.get_id("ptr_forces_col"),
                "fmm_done":   runner.get_id("ptr_fmm_done"),
                "m2m_offsets": runner.get_id("ptr_m2m_offsets"),
                "m2m_child_indices": runner.get_id("ptr_m2m_child_indices"),
                "m2m_v_indices": runner.get_id("ptr_m2m_v_indices"),
                "m2m_signs": runner.get_id("ptr_m2m_signs"),
                "m2l_offsets": runner.get_id("ptr_m2l_offsets"),
                "m2l_source_indices": runner.get_id("ptr_m2l_source_indices"),
                "m2l_v_indices": runner.get_id("ptr_m2l_v_indices"),
                "m2l_signs": runner.get_id("ptr_m2l_signs"),
                "l2l_offsets": runner.get_id("ptr_l2l_offsets"),
                "l2l_source_indices": runner.get_id("ptr_l2l_source_indices"),
                "l2l_v_indices": runner.get_id("ptr_l2l_v_indices"),
                "l2l_signs": runner.get_id("ptr_l2l_signs"),
                "serp_tab": runner.get_id("ptr_serp_tab"),
                "serp_stats": runner.get_id("ptr_serp_stats"),
                "serp_stats_all": runner.get_id("ptr_serp_stats_all"),
                "l_lvl4": runner.get_id("ptr_l_lvl4"),
            }
            print("[3/4] Pushing data...")
            # math tables (broadcast)
            for nm, keys in (("m2m", ("offsets","child_indices","v_indices")),
                             ("m2l", ("offsets","source_indices","v_indices")),
                             ("l2l", ("offsets","source_indices","v_indices"))):
                h2d_bcast_u16(runner, sym[f"{nm}_offsets"], math[nm]["offsets"], side)
                idxkey = "child_indices" if nm == "m2m" else "source_indices"
                h2d_bcast_u16(runner, sym[f"{nm}_{idxkey}"], math[nm][idxkey], side)
                h2d_bcast_u16(runner, sym[f"{nm}_v_indices"], math[nm]["v_indices"], side)
                h2d_bcast_u32(runner, sym[f"{nm}_signs"], math[nm]["signs"], side)

            # bodies (per-PE)
            h2d_per_pe_u32(runner, sym["bodies_col"], bodies_flat, side, col_len)
            # serpentine pass table (per-PE, host-built in serp_tables.py)
            h2d_per_pe_u16(runner, sym["serp_tab"], serp_tab, side, serp_tab.shape[2])

            sym["fmm_done"] = runner.get_id("ptr_fmm_done")

            # PROBE 0: read fmm_done BEFORE any launch. The device is freshly
            # initialized; fmm_done is @zeros so this should return all zeros
            # quickly. If THIS d2h hangs, the problem is the memcpy/RPC layer
            # itself (or a version mismatch), NOT our FMM phase code.
            print("[probe] d2h fmm_done BEFORE any launch...", flush=True)
            tp = time.time()
            pre = np.zeros(side * side, dtype=np.uint32)
            runner.memcpy_d2h(pre, sym["fmm_done"], 0, 0, side, side, 1,
                              streaming=False,
                              data_type=MemcpyDataType.MEMCPY_32BIT,
                              nonblock=False, order=MemcpyOrder.ROW_MAJOR)
            print(f"[probe] pre-launch d2h OK in {time.time()-tp:.3f} s  "
                  f"(sum={int(pre.sum())}, expect 0)", flush=True)

            print("[4/4] Launching FMM phases (host-barriered)...")
            for pid, phase, label in STEPS:
                t0 = time.time()
                runner.launch(phase, nonblock=False)
                dt = time.time() - t0
                print(f"      {label:18s} {dt:.3f} s", flush=True)
                # Poll the completion flag: each PE writes the phase id into
                # fmm_done on TRUE completion (a phase unblocks the host before
                # it is done, so the launch return alone proves nothing).
                t1 = time.time()
                done = np.zeros(side * side, dtype=np.uint32)
                while True:
                    runner.memcpy_d2h(done, sym["fmm_done"], 0, 0, side, side, 1,
                                      streaming=False,
                                      data_type=MemcpyDataType.MEMCPY_32BIT,
                                      nonblock=False, order=MemcpyOrder.ROW_MAJOR)
                    ndone = int((done >= pid).sum())
                    if ndone == side * side or time.time() - t1 > args.phase_timeout:
                        break
                    time.sleep(0.5)
                print(f"        fmm_done = {ndone}/{side*side} PEs  "
                      f"(waited {time.time()-t1:.3f} s)", flush=True)
                if ndone != side * side:
                    missing = [(int(i % side), int(i // side))
                               for i in np.where(done < pid)[0][:12]]
                    print(f"        NOT done (first 12, as (px,py)): {missing}",
                          flush=True)
                    print(f"        stopping: {label} did not complete on every PE",
                          flush=True)
                    err = f"{label} incomplete ({ndone}/{side*side})"
                    if phase == "run_pass":
                        # live counters of the pass that did not complete
                        pass_stats[pid - 2] = d2h_per_pe_u32(runner, sym["serp_stats"], side, 8)
                    break
                completed.append(pid)
                if phase == "run_pass":
                    k = pid - 2
                    if serp_tables.PASSES[k][0] in ("M2L4", "L2L34"):
                        l4_checkpoints[k] = d2h_per_pe_f32(runner, sym["l_lvl4"], side, C * 30)

            # Per-pass counters, copied by each PE when it completed the pass.
            done_passes = [pid - 2 for pid in completed if 2 <= pid < STEP_L2P]
            if done_passes:
                allst = d2h_per_pe_u32(runner, sym["serp_stats_all"], side,
                                       serp_tables.NPASS * 8)
                for k in done_passes:
                    pass_stats[k] = allst[:, :, k * 8:(k + 1) * 8]

            # pull forces
            print("      pulling forces (d2h forces_col)...", flush=True)
            tf = time.time()
            forces_3d = d2h_per_pe_f32(runner, sym["forces_col"],
                                       side, C * max_bodies_pe * FLOATS_PER_BODY)
            print(f"      forces d2h {time.time()-tf:.3f} s", flush=True)
    except Exception as e:
        err = repr(e)
        print(f"\n[appliance_run] ERROR: {err}")

    if forces_3d is not None and STEP_L2P in completed:
        print("\n--- physics: device forces (complete FMM) vs direct N-body ---")
        if not physics_check_forces(forces_3d, bodies, C, max_bodies_pe) and err is None:
            err = "force physics check failed"

    if pass_stats:
        # Save what came off the device, so the verification can be re-run
        # offline: python appliance_run.py --verify-only runs/<label>_runNN
        np.savez(os.path.join(run_dir, "device_data.npz"), bodies=bodies,
                 C=C, max_bodies_pe=max_bodies_pe, serp_tab=serp_tab,
                 **{f"stats_{k}": v for k, v in pass_stats.items()},
                 **{f"l4_{k}": v for k, v in l4_checkpoints.items()},
                 **({"forces": forces_3d} if forces_3d is not None and STEP_L2P in completed else {}))
        ok = verify_passes(pass_stats, l4_checkpoints, serp_tab, bodies, C, max_bodies_pe)
        if not ok and err is None:
            err = "verification failed"

    if forces_3d is not None:
        np.save(os.path.join(run_dir, "forces_device.npy"), forces_3d)

    sys.stdout = original_stdout
    with open(os.path.join(run_dir, "run.log"), "w") as f:
        f.write(log_buf.getvalue())
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
