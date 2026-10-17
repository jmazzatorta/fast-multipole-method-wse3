#!/usr/bin/env python3
"""
appliance_run.py — Runner for the FMM-2.5D benchmark on CS-3 (appliance mode).

Flow:
  1. Resolve artifact from compiled/<LABEL>/, extract to extracted/<LABEL>/.
  2. Read side (= cells_per_side) and max_bodies_pe from out/out.json.
  3. Load fmm25d_data.pkl: operator math tables + per-PE M2L agendas/quotas.
  4. Bin bodies into the C x C column layout (cartesian, NO Morton).
  5. Push math tables (broadcast), agendas/quotas/bodies (per-PE).
  6. Launch the six FMM phases in sequence; each launch is nonblock=False, which
     returns only after ALL PEs finish that phase -> host-mediated global barrier:
        run_p2m -> run_m2m_upward -> run_m2l -> run_l2l_downward -> run_l2p -> run_p2p
  7. Pull forces_col, reassemble, compare to a direct N-body reference.

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
import pickle
import re
import socket
import sys
import tarfile
import time

import numpy as np

from cerebras.sdk.client import SdkRuntime
from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

DATA_FILE = "fmm25d_data.pkl"
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
# Header field layout (must match constants.csl)
# ===========================================================================
SHIFT_TGT_X = 28; SHIFT_TGT_Y = 24; SHIFT_MSG_TYPE = 22
SHIFT_LEVEL = 20; SHIFT_TAG = 11; SHIFT_ZCOUNT = 8; SHIFT_ZBASE = 4

def pack_agenda_entry(tgt_x, tgt_y, tag, zbase, zcount):
    """Agenda entry: same field positions as the runtime header, with
    msg_type and level left ZERO (the PE ORs them in at send time)."""
    h = (int(tgt_x) & 0xF) << SHIFT_TGT_X
    h |= (int(tgt_y) & 0xF) << SHIFT_TGT_Y
    h |= (int(tag) & 0x1FF) << SHIFT_TAG
    h |= (int(zcount) & 0x7) << SHIFT_ZCOUNT
    h |= (int(zbase) & 0xF) << SHIFT_ZBASE
    return np.uint32(h)


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
    with tarfile.open(artifact, "r:gz") as tar:
        tar.extractall(path=label_dir)
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
# Pack per-PE M2L agendas and quotas from the pickle routing dict
# ===========================================================================
def pack_m2l_level(routing, C, level, max_agenda):
    agenda = routing["m2l_agenda"]      # (px,py)->level->[(tx,ty,tag,zbase,zcount)]
    send_e = routing["send_expected"]
    recv_e = routing["recv_expected"]
    fwd_e  = routing["fwd_expected"]

    a3d = np.zeros((C, C, max_agenda), dtype=np.uint32)
    nsend = np.zeros((C, C, 1), dtype=np.uint32)
    nrecv = np.zeros((C, C, 1), dtype=np.uint32)
    nfwd  = np.zeros((C, C, 1), dtype=np.uint32)

    for (px, py), lvls in agenda.items():
        if level in lvls:
            msgs = lvls[level]
            assert len(msgs) <= max_agenda, \
                f"agenda ({px},{py}) lvl {level}: {len(msgs)} > MAX_AGENDA {max_agenda}"
            for i, (tx, ty, tag, zbase, zcount) in enumerate(msgs):
                a3d[py, px, i] = pack_agenda_entry(tx, ty, tag, zbase, zcount)
            nsend[py, px, 0] = len(msgs)
    for (px, py), lvls in send_e.items():
        if level in lvls: nsend[py, px, 0] = lvls[level]
    for (px, py), lvls in recv_e.items():
        if level in lvls: nrecv[py, px, 0] = lvls[level]
    for (px, py), lvls in fwd_e.items():
        if level in lvls: nfwd[py, px, 0] = lvls[level]
    return a3d, nsend, nrecv, nfwd


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
# Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description="Run FMM-2.5D on CS-3.")
    parser.add_argument("--label", required=True)
    parser.add_argument("--bodies", default=None,
                        help="(N,4) float32 .npy (x,y,z,q). Default: a small test set.")
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--compiled-dir", default="compiled")
    parser.add_argument("--extracted-dir", default="extracted")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--data-file", default=DATA_FILE)
    parser.add_argument("--max-agenda", type=int, default=256)
    args = parser.parse_args()

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
        # default tiny test set inside the domain (prob_range=512, C cells)
        cell = 512.0 / C
        bodies = np.array([
            [1.5 * cell, 1.5 * cell, 1.5 * cell, 1.0],
            [5.5 * cell, 2.5 * cell, 3.5 * cell, 2.0],
            [8.5 * cell, 8.5 * cell, 8.5 * cell, 1.5],
        ], dtype=np.float32)
    N = len(bodies)

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
    print("=" * 64)

    # load tables + routing
    with open(args.data_file, "rb") as f:
        data = pickle.load(f)
    math = data["math"]
    routing = data["routing"]
    for nm in ("m2m", "m2l", "l2l"):
        s = math[nm]["signs"]
        assert s.dtype == np.uint32 and s.max() <= 15, f"{nm} signs bad"

    print("[1/4] Binning bodies into column layout...")
    bodies_flat, col_len, CELL_BLOCK = build_bodies_tensor(bodies, C, max_bodies_pe)

    print("[2/4] Packing M2L agendas/quotas per level...")
    levels = [2, 3, 4]
    # Per-level agenda bound, matching constants.csl exactly:
    #   bound(L) = M2L_MAX_DESTS * ceil( (C / 2**(TREE_H-L)) / K_CUBES )
    # i.e. 48 * ceil(nz_at_level / K). TREE_H = log2(C). For C=16,K=4:
    #   L4 -> 192, L3 -> 96, L2 -> 48.
    K_CUBES = int(d["params"]["K_CUBES"])
    M2L_MAX_DESTS = 48
    TREE_H = C.bit_length() - 1  # log2(C)
    def agenda_bound(level):
        nz = C // (2 ** (TREE_H - level))   # cubes along Z in a box at this level
        msgs = (nz + K_CUBES - 1) // K_CUBES
        return M2L_MAX_DESTS * msgs
    agenda_sz = {lv: agenda_bound(lv) for lv in levels}
    print(f"       per-level agenda bounds: {agenda_sz}")
    packed = {lv: pack_m2l_level(routing, C, lv, agenda_sz[lv]) for lv in levels}

    err = None
    forces_3d = None
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
            }
            for lv in levels:
                sym[f"agenda{lv}"]   = runner.get_id(f"ptr_m2l_agenda_lvl{lv}")
                sym[f"nsend{lv}"]    = runner.get_id(f"ptr_num_send_lvl{lv}")
                sym[f"nrecv{lv}"]    = runner.get_id(f"ptr_recv_expected_lvl{lv}")
                sym[f"nfwd{lv}"]     = runner.get_id(f"ptr_fwd_expected_lvl{lv}")

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

            # agendas + quotas (per-PE)
            for lv in levels:
                a3d, nsend, nrecv, nfwd = packed[lv]
                h2d_per_pe_u32(runner, sym[f"agenda{lv}"], a3d, side, agenda_sz[lv])
                h2d_per_pe_u32(runner, sym[f"nsend{lv}"], nsend, side, 1)
                h2d_per_pe_u32(runner, sym[f"nrecv{lv}"], nrecv, side, 1)
                h2d_per_pe_u32(runner, sym[f"nfwd{lv}"],  nfwd,  side, 1)

            # bodies (per-PE)
            h2d_per_pe_u32(runner, sym["bodies_col"], bodies_flat, side, col_len)

            print("[4/4] Launching FMM phases (host-barriered)...")
            # Each launch returns only after ALL PEs finish -> global barrier.
            for phase in ("run_p2m", "run_m2m_upward", "run_m2l",
                          "run_l2l_downward", "run_l2p", "run_p2p"):
                t0 = time.time()
                runner.launch(phase, nonblock=False)
                print(f"      {phase:18s} {time.time()-t0:.3f} s")

            # pull forces
            forces_3d = d2h_per_pe_f32(runner, sym["forces_col"],
                                       side, C * max_bodies_pe * FLOATS_PER_BODY)
    except Exception as e:
        err = repr(e)
        print(f"\n[appliance_run] ERROR: {err}")

    if forces_3d is not None:
        # reassemble device forces per body and compare to direct
        print("\n--- VERIFICATION vs direct N-body ---")
        Fref, potref = direct_nbody(bodies)
        # map each body back to (pe_x=cx, pe_y=cy, cell=cz, slot)
        counts = {}
        maxferr = 0.0; maxperr = 0.0
        cell_size_v = 512.0 / C
        for bi, (x, y, z, q) in enumerate(bodies):
            cx = min(max(int(x / cell_size_v), 0), C - 1)
            cy = min(max(int(y / cell_size_v), 0), C - 1)
            cz = min(max(int(z / cell_size_v), 0), C - 1)
            slot = counts.get((cx, cy, cz), 0); counts[(cx, cy, cz)] = slot + 1
            fbase = cz * (max_bodies_pe * FLOATS_PER_BODY) + slot * FLOATS_PER_BODY
            fx, fy, fz, pot = forces_3d[cy, cx, fbase:fbase + 4]
            fe = np.linalg.norm([fx - Fref[bi, 0], fy - Fref[bi, 1], fz - Fref[bi, 2]])
            pe = abs(pot - potref[bi])
            maxferr = max(maxferr, fe); maxperr = max(maxperr, pe)
            print(f"  body {bi}: F_dev=({fx:.4e},{fy:.4e},{fz:.4e}) "
                  f"F_ref=({Fref[bi,0]:.4e},{Fref[bi,1]:.4e},{Fref[bi,2]:.4e})  |dF|={fe:.3e}")
        print(f"\n  max |F_dev - F_ref| = {maxferr:.3e}")
        print(f"  max |pot_dev - pot_ref| = {maxperr:.3e}")
        np.save(os.path.join(run_dir, "forces_device.npy"), forces_3d)

    sys.stdout = original_stdout
    with open(os.path.join(run_dir, "run.log"), "w") as f:
        f.write(log_buf.getvalue())
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
