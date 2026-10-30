#!/usr/bin/env python3
"""
appliance_run.py — Runner for the FMM-2.5D benchmark on CS-3 (appliance mode).

Flow:
  1. Resolve artifact from compiled/<LABEL>/, extract to extracted/<LABEL>/.
  2. Read C (= cells_per_side), the expansion order P (order_p), ZP (2**zp_log
     leaf cells per PE) and max_bodies_pe from out/out.json and configure
     zmp_tables (and fmm_passes), fmm_tables and fmm_ref.
  3. Build the operator tables (fmm_tables.py) and the all-static
     transport tables (zmp_tables.py; brute-force checked on small meshes).
  4. Bin bodies per PE: cell (x, y, z) -> PE of pe_map.py (ZP cells per PE).
  5. Push math tables (broadcast), bodies and transport tables (per-PE).
  6. Launch the FMM steps in sequence. Each PE writes the step id into
     fmm_done on TRUE completion; after each launch the host polls fmm_done
     until all PEs report the step (or a timeout): the global barrier.
        run_p2m (1) -> run_pass x NPASS (2..NPASS+1) -> run_l2p (NPASS+2)
     NPASS = 3H - 4 with H = log2 C, pass list in fmm_passes.py (C = 16:
     M2M43 M2M32 M2L4 M2L3 M2L2 L2L23 L2L34 P2P).
     After the run the per-PE counters of every pass are checked against the
     transport tables (messages received / sent, relays, port arrivals) and the expected work; the leaf
     local buffer l_leaf is checked against fmm_ref after the leaf M2L pass
     (M2L{H}) and after the last L2L pass (L2L{H-1}{H}).
  7. Pull forces_col, reassemble, compare to the direct N-body sum.

Usage:
    python appliance_run.py --label C16_MB8 [--simulator] [--bodies file.npy]
                            [--cell-size EDGE]   (physical units, see --help)

Must run with the appliance venv python.
"""
import argparse
import datetime
import glob
import io
import json
import os
import re
import shutil
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
# rect = (W, H): the program rectangle in PEs. Host tensors are (H, W, n),
# indexed [fy, fx] (ROW_MAJOR). Every transfer is split into bands of whole
# PE rows of at most MEMCPY_MAX_BYTES each: one memcpy message must stay
# below the gRPC limit of 2 GiB (an mp_ent push at 512x512 PEs is 2.4 GB).
MEMCPY_MAX_BYTES = 256 << 20

def _row_bands(rect, n):
    """(y0, rows) bands of whole PE rows, each <= MEMCPY_MAX_BYTES (4-byte
    host words, n per PE)."""
    rows = max(1, MEMCPY_MAX_BYTES // (4 * n * rect[0]))
    return [(y, min(rows, rect[1] - y)) for y in range(0, rect[1], rows)]

def _h2d(runner, sym, band, rect, n, data_type):
    """band(y0, rows) -> (rows, W, n) u32 host data of those PE rows."""
    for y0, h in _row_bands(rect, n):
        t = np.ascontiguousarray(band(y0, h), dtype=np.uint32)
        runner.memcpy_h2d(sym, t.ravel(), 0, y0, rect[0], h, n,
                          streaming=False, data_type=data_type,
                          nonblock=False, order=MemcpyOrder.ROW_MAJOR)

def _d2h(runner, sym, rect, n, dtype):
    out = np.zeros((rect[1], rect[0], n), dtype=dtype)
    for y0, h in _row_bands(rect, n):
        part = np.zeros(h * rect[0] * n, dtype=dtype)
        runner.memcpy_d2h(part, sym, 0, y0, rect[0], h, n,
                          streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                          nonblock=False, order=MemcpyOrder.ROW_MAJOR)
        out[y0:y0 + h] = part.reshape(h, rect[0], n)
    return out

def h2d_bcast_u16(runner, sym, arr_u16, rect):
    """u16 device buffer, broadcast to every PE. Host words are u32
    zero-extended, pushed as MEMCPY_16BIT."""
    a = arr_u16.astype(np.uint32)
    _h2d(runner, sym, lambda y0, h: np.broadcast_to(a, (h, rect[0], len(a))),
         rect, len(a), MemcpyDataType.MEMCPY_16BIT)

def h2d_bcast_u32(runner, sym, arr_u32, rect):
    """u32 device buffer, broadcast (e.g. packed signs)."""
    a = arr_u32.astype(np.uint32)
    _h2d(runner, sym, lambda y0, h: np.broadcast_to(a, (h, rect[0], len(a))),
         rect, len(a), MemcpyDataType.MEMCPY_32BIT)

def h2d_per_pe_u32(runner, sym, tensor, rect, words_per_pe):
    t = tensor.reshape(rect[1], rect[0], words_per_pe)
    _h2d(runner, sym, lambda y0, h: t[y0:y0 + h], rect, words_per_pe,
         MemcpyDataType.MEMCPY_32BIT)

def h2d_per_pe_u16(runner, sym, tensor_u16, rect, words_per_pe):
    """u16 device buffer, per-PE data. Host words are u32 zero-extended,
    pushed as MEMCPY_16BIT."""
    t = tensor_u16.reshape(rect[1], rect[0], words_per_pe)
    _h2d(runner, sym, lambda y0, h: t[y0:y0 + h], rect, words_per_pe,
         MemcpyDataType.MEMCPY_16BIT)

def d2h_per_pe_u32(runner, sym, rect, words_per_pe):
    return _d2h(runner, sym, rect, words_per_pe, np.uint32)

def d2h_per_pe_f32(runner, sym, rect, words_per_pe):
    return _d2h(runner, sym, rect, words_per_pe, np.float32)


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
    # Reuse an extraction only if it comes from THIS artifact (same cs_<hash>):
    # after a recompilation with the same label the old one is stale.
    stem = os.path.basename(artifact)[:-len(".tar.gz")]
    cached = os.path.join(label_dir, stem, "out")
    if os.path.isdir(cached):
        return cached
    if os.path.isdir(label_dir):
        shutil.rmtree(label_dir)
    os.makedirs(label_dir)
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
# Body binning: cell (x, y, z) -> PE of the z-split mapping (pe_map.py)
# ===========================================================================
def cell_of(p, C):
    """Leaf cell of a position in the domain [0, 512): floor(p / cell_size)."""
    cell = 512.0 / C
    return tuple(min(max(int(v / cell), 0), C - 1) for v in p)


def build_bodies_tensor(bodies, C, max_bodies_pe):
    """
    bodies: (N,4) float32 (x,y,z,q) in the domain [0, 512).
    Returns the per-PE tensor (H, W, ZP*CELL_BLOCK) on the program rectangle:
    each PE holds its ZP leaf cells, each cell block = [count | body0(x,y,z,q)
    | ...]. A body in cell (cx,cy,cz) lives on PE cell_to_pe(cx, cy, cz) at
    local cell index cz mod ZP. Slots in input order (device_forces_per_body
    relies on it).
    """
    import zmp_tables as Z
    m = Z.MAP
    CELL_BLOCK = 1 + max_bodies_pe * FLOATS_PER_BODY
    col_len = m.zp * CELL_BLOCK
    flat = np.zeros((m.height, m.width, col_len), dtype=np.uint32)
    counts = {}
    for (x, y, z, q) in bodies:
        key = cell_of((x, y, z), C)
        k = counts.get(key, 0)
        if k >= max_bodies_pe:
            raise ValueError(f"cell {key} exceeds max_bodies_pe={max_bodies_pe}")
        fx, fy, zl = m.cell_to_pe(*key)
        off = zl * CELL_BLOCK + 1 + k * FLOATS_PER_BODY
        flat[fy, fx, off:off + 4] = np.array([x, y, z, q], dtype=np.float32).view(np.uint32)
        counts[key] = k + 1
    for key, k in counts.items():
        fx, fy, zl = m.cell_to_pe(*key)
        flat[fy, fx, zl * CELL_BLOCK] = np.uint32(k)
    return flat, col_len, CELL_BLOCK

def check_targets(n, sample):
    """Indices of the target bodies of the direct-sum checks: all n when
    sample is 0 or >= n, else `sample` of them drawn with a fixed seed."""
    if sample <= 0 or sample >= n:
        return np.arange(n)
    return np.sort(np.random.default_rng(12345).choice(n, sample, replace=False))


def _target_blocks(targets, n):
    """Split `targets` into blocks of about 2**24 / n targets, so a block of
    target x source f64 arrays stays near 128 MiB."""
    step = max(1, (1 << 24) // max(n, 1))
    return [targets[a:a + step] for a in range(0, len(targets), step)]


def direct_nbody(bodies, targets):
    """Direct sum (f64) of the force and potential on bodies[targets] from all
    other bodies. Returns F (len(targets), 3) and pot (len(targets),)."""
    pos = bodies[:, :3].astype(np.float64)
    q = bodies[:, 3].astype(np.float64)
    F = np.zeros((len(targets), 3)); pot = np.zeros(len(targets))
    a = 0
    for tb in _target_blocks(targets, len(bodies)):
        d = pos[None, :, :] - pos[tb, None, :]
        r2 = (d * d).sum(axis=2)
        r2[np.arange(len(tb)), tb] = np.inf
        rinv = 1.0 / np.sqrt(r2)
        F[a:a + len(tb)] = np.einsum("tn,tnc->tc", q * rinv ** 3, d)
        pot[a:a + len(tb)] = (q * rinv).sum(axis=1)
        a += len(tb)
    return F, pot


# ===========================================================================
# Pass verification
# ===========================================================================
def _n_near(nb):
    """(nb,) number of indices j in [0, nb) with |i - j| <= 1, for each i."""
    i = np.arange(nb)
    return np.minimum(i + 1, nb - 1) - np.maximum(i - 1, 0) + 1


def _n_parent_adj(nb):
    """(nb,) number of j in [0, nb) whose parent is adjacent to i's parent."""
    p = np.arange(nb) // 2
    np_ = nb // 2
    return 2 * (np.minimum(p + 1, np_ - 1) - np.maximum(p - 1, 0) + 1)


def _held(level):
    """PE -> (bx, by, [bz...]) of the level boxes it OWNS (zmp_tables)."""
    import zmp_tables as Z
    import fmm_passes as T
    st = T.stride(level)
    out = {}
    for bz in range(1 << level):
        k = Z.chunk_of(level, bz)
        for by in range(1 << level):
            for bx in range(1 << level):
                pe = Z.pe_of((bx * st, by * st), k)
                out.setdefault(pe, (bx, by, []))[2].append(bz)
    return out


def expected_work(C, k, max_bodies_pe):
    """(H, W) pass_stats[4] per PE for pass k:
      M2L  translations = sum over the target boxes the PE owns of the size of
           their interaction list (parents adjacent minus adjacent, separable:
           Px Py Pz - Nx Ny Nz);
      M2M  boxes added, L2L boxes copied = boxes addressed to the PE (tables);
      P2P  (target cell, source cell) pairs = sum over its cells of Nx Ny Nz
           (|d| <= 1 per axis, the cell itself included)."""
    import fmm_passes as T
    import zmp_tables as Z
    name, lv = T.PASSES[k]
    exp = np.zeros((Z.MAP.height, Z.MAP.width), dtype=np.int64)
    nb = 1 << lv
    if name.startswith("M2L"):
        pa, nn = _n_parent_adj(nb), _n_near(nb)
        for (fx, fy), (bx, by, zs) in _held(lv).items():
            exp[fy, fx] = sum(int(pa[bx] * pa[by] * pa[z] - nn[bx] * nn[by] * nn[z])
                              for z in zs)
    elif name == "P2P":
        nn = _n_near(nb)
        for (fx, fy), (bx, by, zs) in _held(lv).items():
            exp[fy, fx] = sum(int(nn[bx] * nn[by] * nn[z]) for z in zs)
    else:
        for _, lst in Z.pass_entries(k).items():
            for (dx, dy, zb, zn) in lst:
                exp[dy, dx] += zn
    return exp


def checkpoint_passes():
    """{pass index: label} of the passes after which l_leaf is read back: the
    leaf M2L (l_leaf holds only the leaf-level interactions) and the last L2L
    (complete local expansions)."""
    import fmm_passes as T
    h = T.TREE_H
    names = (f"M2L{h}", f"L2L{h - 1}{h}")
    return {k: p[0] for k, p in enumerate(T.PASSES) if p[0] in names}


def assemble_leaf(l_dev, C):
    """Per-PE l_leaf (H, W, ZP*C_BUF_SIZE) -> column layout (C, C, C*C_BUF_SIZE)
    indexed (py, px, z*C_BUF_SIZE+k), the layout of fmm_ref.to_device_layout."""
    import zmp_tables as Z
    m = Z.MAP
    n = l_dev.shape[2]                       # ZP * C_BUF_SIZE
    out = np.zeros((C, C, C * (n // m.zp)), dtype=l_dev.dtype)
    for fy in range(m.height):
        for fx in range(m.width):
            x, y, k, _ = m.pe_to_column(fx, fy)
            out[y, x, k * n:(k + 1) * n] = l_dev[fy, fx]
    return out


def compare_leaf(l_dev, ref, C, what):
    """Compare device l_leaf with the reference.

    PASS criterion: max |device - ref| / (max |ref| of that PE) <= 1e-5. f32
    summation in a different order gives ~1e-7..1e-6 on this scale, while a
    lost, duplicated or corrupted message changes a whole M2L contribution and
    shows up at >= 1e-3 of the PE max.
    Also printed (informational): the error per coefficient, each scaled by its
    own max over z. With the current (wrong) M2L operator the terms reach 1e16
    and cancel down to ~1e8 in some coefficients, so that view is dominated by
    f32 cancellation and is not a pass criterion until the math is fixed."""
    import fmm_ref
    nb = fmm_ref.C_BUF_SIZE
    d4 = l_dev.astype(np.float64).reshape(C, C, C, nb)
    r4 = ref.astype(np.float64).reshape(C, C, C, nb)
    if not np.all(np.isfinite(r4)):
        print(f"  l_leaf {what}: reference not finite, comparison skipped")
        return True
    pe_max = np.abs(r4).max(axis=(2, 3), keepdims=True) + 1e-30
    rel_pe = (np.abs(d4 - r4) / pe_max).max(axis=(2, 3))
    ok = bool(rel_pe.max() <= 1e-5)
    wy, wx = np.unravel_index(int(np.argmax(rel_pe)), rel_pe.shape)
    print(f"  l_leaf {what}: error / column max: max {rel_pe.max():.3e} at column (x,y)=({wx},{wy}), "
          f"median {np.median(rel_pe):.3e}; PEs above 1e-5: {int((rel_pe > 1e-5).sum())}"
          f"  -> {'OK' if ok else 'MISMATCH'}")
    scale = np.maximum(np.abs(r4).max(axis=2, keepdims=True), 1e-6 * pe_max)
    relx = np.abs(d4 - r4) / scale
    py, px, z, k = np.unravel_index(int(np.argmax(relx)), relx.shape)
    print(f"      (info) per-coefficient rel err max {relx.max():.3e} at column ({px},{py}) "
          f"z={z} coeff={k}: device {d4[py, px, z, k]:.4e} ref {r4[py, px, z, k]:.4e}")
    return ok


def physics_check_m2l_leaf(l_dev, bodies, C, max_bodies_pe, sample):
    """Physical check of the device leaf-level M2L, independent of fmm_ref's
    M2L: potential from L2P of the DEVICE l_leaf (after the leaf M2L pass, so
    it holds only leaf-level interactions) vs the direct sum over the bodies of
    the leaf interaction-list cells (not adjacent, parents adjacent), on the
    target bodies of check_targets(sample). Measured at C=16, P=4: median
    ~4e-5, max ~2.5e-3."""
    import fmm_ref
    binned = fmm_ref.bin_bodies(bodies, max_bodies=max_bodies_pe)
    where = binned[2]
    # device layout (py, px, z*C_BUF_SIZE+k) -> fmm_ref layout [x, y, z, k]
    l4 = np.transpose(l_dev.reshape(C, C, C, fmm_ref.C_BUF_SIZE),
                      (1, 0, 2, 3)).astype(np.float32)
    out = fmm_ref.l2p(binned, l4)
    tg = check_targets(len(bodies), sample)
    w = where[tg]
    pot_dev = out[w[:, 0], w[:, 1], w[:, 2], w[:, 3], 3].astype(np.float64)
    cell = 512.0 / C
    ci = (bodies[:, :3] // cell).astype(int).clip(0, C - 1)
    pos = bodies[:, :3].astype(np.float64)
    q = bodies[:, 3].astype(np.float64)
    pot_dir = np.zeros(len(tg))
    a = 0
    for tb in _target_blocks(tg, len(bodies)):
        dc = np.abs(ci[None, :, :] - ci[tb, None, :]).max(axis=2)
        dp = np.abs(ci[None, :, :] // 2 - ci[tb, None, :] // 2).max(axis=2)
        r = np.linalg.norm(pos[None, :, :] - pos[tb, None, :], axis=2)
        pot_dir[a:a + len(tb)] = np.where((dc >= 2) & (dp <= 1),
                                          q / np.where(dc >= 2, r, 1.0), 0.0).sum(axis=1)
        a += len(tb)
    rel = np.abs(pot_dev - pot_dir) / np.abs(pot_dir)
    ok = bool(np.median(rel) < 1e-3 and rel.max() < 1e-2)
    print(f"  physics M2L{fmm_ref.TREE_H} (device l_leaf -> L2P vs direct sum on the leaf list, "
          f"{len(tg)}/{len(bodies)} target bodies): "
          f"potential rel err median {np.median(rel):.2e} max {rel.max():.2e}"
          f"  -> {'OK' if ok else 'FAIL'}")
    return ok


def device_forces_per_body(forces_3d, bodies, C, max_bodies_pe):
    """(N,4) [fx,fy,fz,pot] from the per-PE forces_col (H, W, ZP*MB*4), in the
    input order of `bodies` (same slot assignment as build_bodies_tensor)."""
    import zmp_tables as Z
    out = np.zeros((len(bodies), 4))
    counts = {}
    for bi, (x, y, z, q) in enumerate(bodies):
        key = cell_of((x, y, z), C)
        slot = counts.get(key, 0); counts[key] = slot + 1
        fx, fy, zl = Z.MAP.cell_to_pe(*key)
        fb = zl * (max_bodies_pe * FLOATS_PER_BODY) + slot * FLOATS_PER_BODY
        out[bi] = forces_3d[fy, fx, fb:fb + 4]
    return out


def physics_check_forces(forces_3d, bodies, C, max_bodies_pe, sample):
    """Device forces/potentials (complete FMM: far field + P2P near field) vs
    the direct N-body sum, on the target bodies of check_targets(sample).
    Expected for P=4: force rel err median ~1e-3."""
    tg = check_targets(len(bodies), sample)
    dev = device_forces_per_body(forces_3d, bodies, C, max_bodies_pe)[tg]
    Fr, pr = direct_nbody(bodies, tg)
    ferr = np.linalg.norm(dev[:, :3] - Fr, axis=1) / (np.linalg.norm(Fr, axis=1) + 1e-30)
    perr = np.abs(dev[:, 3] - pr) / (np.abs(pr) + 1e-30)
    ok = bool(np.median(ferr) < 1e-2 and np.median(perr) < 1e-3)
    print(f"  physics forces (device FMM vs direct N-body, {len(tg)}/{len(bodies)} "
          f"target bodies): force rel err "
          f"median {np.median(ferr):.2e} rms {np.sqrt((ferr**2).mean()):.2e} max {ferr.max():.2e}; "
          f"pot median {np.median(perr):.2e} max {perr.max():.2e}  -> {'OK' if ok else 'FAIL'}")
    return ok


def expected_transport(k, max_bodies_pe):
    """(H, W, 4) expected pass_stats[0..3] of pass k (pe.csl TRANSPORT, all-
    static transport): messages consumed (slot 0), relays = straight forwards + turns
    (slot 1), own messages sent (slot 2), port arrivals incl. credits, E + W +
    N + S (slot 3); from zmp_tables."""
    import zmp_tables as Z
    n_rx, n_tx = Z.pass_counts(k, max_bodies_pe)
    lc = Z.lane_counts(k, max_bodies_pe)
    exp = np.zeros((Z.MAP.height, Z.MAP.width, 4), dtype=np.int64)
    for slot, counts in ((0, n_rx), (2, n_tx), (1, lc["fwd"]), (1, lc["turn"]),
                         (3, lc["p_e"]), (3, lc["p_w"]), (3, lc["p_n"]), (3, lc["p_s"])):
        for (x, y), v in counts.items():
            exp[y, x, slot] += v
    return exp, ["rx", "relays", "tx", "port-arr"]


def verify_passes(pass_stats, leaf_checkpoints, bodies, C, max_bodies_pe, sample):
    """pass_stats: pass -> (H, W, 8); leaf_checkpoints: pass -> per-PE l_leaf
    (H, W, ZP*C_BUF_SIZE); sample: target bodies of the direct-sum check
    (check_targets)."""
    import fmm_passes as T
    print("\n--- VERIFICATION: passes (all-static transport) ---")
    allok = True
    for k in sorted(pass_stats):
        st = pass_stats[k]
        exp, names = expected_transport(k, max_bodies_pe)
        bad = []
        for i in range(4):
            n = int((st[:, :, i] != exp[:, :, i]).sum())
            if n:
                bad.append(f"{names[i]}:{n} PEs")
        npe = st.shape[0] * st.shape[1]
        ndone = int((st[:, :, 6] == 1).sum())
        stray = int(st[:, :, 5].sum()); perr = int(st[:, :, 7].sum())
        wexp = expected_work(C, k, max_bodies_pe)
        nwork = int((st[:, :, 4] != wexp).sum())
        ok = not bad and ndone == npe and stray == 0 and perr == 0 and nwork == 0
        allok &= ok
        print(f"  pass {k} {T.PASSES[k][0]:6s}: done {ndone}/{npe}  arrivals "
              f"{int(st[:, :, 0].sum()):6d}/{int(exp[:, :, 0].sum()):6d}  relays "
              f"{int(st[:, :, 1].sum()):6d}/{int(exp[:, :, 1].sum()):6d}  work "
              f"{int(st[:, :, 4].sum()):6d}/{int(wexp.sum()):6d}  stray {stray}  "
              f"proto-err {perr}  {'OK' if ok else 'FAIL ' + ', '.join(bad) + (f' work:{nwork} PEs' if nwork else '')}")
    try:
        import fmm_ref
        binned = fmm_ref.bin_bodies(bodies, max_bodies=max_bodies_pe)
        for k, name in checkpoint_passes().items():
            if k not in leaf_checkpoints:
                continue
            leaf = assemble_leaf(leaf_checkpoints[k], C)
            if name.startswith("M2L"):
                # device math: M2L corrected (irregular harmonics), see fmm_tables
                ref = fmm_ref.m2l_leaf_only(binned, ops="fixed")
                allok &= compare_leaf(leaf, ref, C, f"after {name}")
                allok &= physics_check_m2l_leaf(leaf, bodies, C, max_bodies_pe, sample)
            else:
                # Whole pipeline (M2M, M2L at every level, L2L) vs the
                # reference with the device math (corrected M2L and L2L).
                mlev = fmm_ref.upward(fmm_ref.p2m(binned))
                ref = fmm_ref.to_device_layout(
                    fmm_ref.downward(mlev, m2l_ops="fixed",
                                     l2l_ops="fixed")[fmm_ref.TREE_H])
                allok &= compare_leaf(leaf, ref, C, f"after {name}")
    except Exception as e:
        allok = False
        print(f"  l_leaf reference check FAILED to run: {e!r}")
    print("  RESULT:", "PASS" if allok else "FAIL")
    return allok


# ===========================================================================
# Cycle counter (pe.csl CYCLE COUNTER)
# ===========================================================================
def tsc_step_names():
    """Name of every tsc step: P2M, the passes, L2P."""
    import fmm_passes
    return ["P2M"] + [p[0] for p in fmm_passes.PASSES] + ["L2P"]


def print_tsc(tsc):
    """tsc: (H, W, NSTEP*9) u32 from tsc_log. Per step and per PE:
    t0 launch reached the PE, t1 first event (0 = never), t2 step complete.
    Prints min / median / max over the PEs of
      dur  = t2 - min(t0, t1)   (the PE's whole step, including work it did
                                  before its own launch arrived)
      wait = t2 - t0            (from the launch arrival to completion)
    Per-PE counters are not synchronized: only per-PE differences are valid.
    Returns {step name: (dur min, median, max)}."""
    names = tsc_step_names()
    nstep = tsc.shape[2] // 9
    assert nstep == len(names), (nstep, len(names))
    w = tsc.reshape(tsc.shape[0], tsc.shape[1], nstep, 3, 3).astype(np.uint64)
    t = w[..., 0] | (w[..., 1] << np.uint64(16)) | (w[..., 2] << np.uint64(32))
    out = {}
    print("\n--- cycle counter: cycles per PE (min / median / max over the PEs) ---")
    print(f"  {'step':8s} {'dur min':>11s} {'dur median':>11s} {'dur max':>11s}   "
          f"{'wait median':>11s} {'wait max':>11s}   {'PEs with t2':>11s}")
    for s, nm in enumerate(names):
        t0, t1, t2 = t[:, :, s, 0], t[:, :, s, 1], t[:, :, s, 2]
        have = t2 > 0
        if not have.any():
            print(f"  {nm:8s}   (no completion recorded)")
            continue
        start = np.where((t1 > 0) & (t1 < t0), t1, t0)
        dur = (t2 - start).astype(np.int64)[have]
        wait = (t2.astype(np.int64) - t0.astype(np.int64))[have]
        out[nm] = (int(dur.min()), int(np.median(dur)), int(dur.max()))
        print(f"  {nm:8s} {int(dur.min()):11d} {int(np.median(dur)):11d} "
              f"{int(dur.max()):11d}   {int(np.median(wait)):11d} {int(wait.max()):11d}   "
              f"{int(have.sum()):11d}")
    return out


# ===========================================================================
# Main
# ===========================================================================
def configure_order(C, P, zp, lbw=None):
    """Configure the host modules for mesh side C, expansion order P, ZP cells
    per PE and block width 2**lbw (the values the program was compiled with)."""
    import fmm_ref
    import zmp_tables
    zmp_tables.configure(C, P, zp, lbw)  # also fmm_passes
    fmm_ref.set_order(P)          # also sets fmm_tables' order
    fmm_ref.set_mesh(C)


def main():
    parser = argparse.ArgumentParser(description="Run FMM-2.5D on CS-3.")
    parser.add_argument("--label", default=None)
    parser.add_argument("--bodies", default=None,
                        help="(N,4) .npy (x,y,z,q): internal units [0, 512) or, with "
                             "--cell-size, physical units [0, C * cell size). "
                             "Default: --nbodies random bodies")
    parser.add_argument("--cell-size", type=float, default=None,
                        help="physical edge of a leaf cell (any length unit, e.g. 10 "
                             "for 10 km = 1000 km^3 cells): --bodies is read in that "
                             "unit and forces_phys.npy is written in it")
    parser.add_argument("--rep", type=int, default=None)
    parser.add_argument("--simulator", action="store_true")
    parser.add_argument("--compiled-dir", default="compiled")
    parser.add_argument("--extracted-dir", default="extracted")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--phase-timeout", type=float, default=60.0,
                        help="seconds to wait for fmm_done after each phase")
    parser.add_argument("--nsteps", type=int, default=None,
                        help="run only the first N steps of the sequence "
                             "(default: all NPASS + 2)")
    parser.add_argument("--nbodies", type=int, default=2000,
                        help="random bodies when --bodies is not given")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--memcpy-max-kib", type=int, default=256 << 10,
                        help="largest memcpy transfer; bigger ones are split into "
                             "bands of PE rows (gRPC refuses messages >= 2 GiB)")
    parser.add_argument("--check-sample", type=int, default=4000,
                        help="target bodies of the direct-sum physics checks, drawn "
                             "with a fixed seed (0 = all bodies; the cost is "
                             "O(sample * N))")
    parser.add_argument("--verify-only", default=None, metavar="RUN_DIR",
                        help="re-run the verification on RUN_DIR/device_data.npz "
                             "(no device needed)")
    args = parser.parse_args()
    global MEMCPY_MAX_BYTES
    MEMCPY_MAX_BYTES = args.memcpy_max_kib << 10
    if args.verify_only:
        d = np.load(os.path.join(args.verify_only, "device_data.npz"))
        # runs saved before GEN have no order_p (P = 4), before G no zp (ZP = C)
        configure_order(int(d["C"]), int(d["order_p"]) if "order_p" in d.files else 4,
                        int(d["zp"]) if "zp" in d.files else int(d["C"]),
                        int(d["blk_lbw"]) if "blk_lbw" in d.files else None)
        ps = {int(k[6:]): d[k] for k in d.files if k.startswith("stats_")}
        # leaf_<k>: l_leaf after pass k (l4_<k> in runs saved before GEN)
        cp = {int(k.split("_")[1]): d[k] for k in d.files
              if k.startswith("leaf_") or k.startswith("l4_")}
        ok = verify_passes(ps, cp, d["bodies"], int(d["C"]), int(d["max_bodies_pe"]),
                           args.check_sample)
        if "forces" in d.files:
            ok &= physics_check_forces(d["forces"], d["bodies"], int(d["C"]),
                                       int(d["max_bodies_pe"]), args.check_sample)
        if "tsc" in d.files:
            print_tsc(d["tsc"])
        sys.exit(0 if ok else 1)
    if not args.label:
        parser.error("--label is required (unless --verify-only)")

    artifact = resolve_artifact(args.compiled_dir, args.label)
    out_dir_ex = ensure_extracted(artifact, args.extracted_dir, args.label)
    with open(os.path.join(out_dir_ex, "out.json")) as f:
        compile_data = json.load(f)
    C = int(compile_data["params"]["cells_per_side"])
    P = int(compile_data["params"]["order_p"])
    zp = 1 << int(compile_data["params"]["zp_log"])
    blk_lbw = int(compile_data["params"]["blk_lbw"])
    configure_order(C, P, zp, blk_lbw)
    import zmp_tables
    pmap = zmp_tables.MAP
    assert pmap.lbh == int(compile_data["params"]["blk_lbh"])
    max_bodies_pe = int(compile_data["params"]["max_bodies_pe"])
    mp_ent_cap = int(compile_data["params"]["mp_ent_cap"])
    rect = (pmap.width, pmap.height)          # program rectangle (W, H) in PEs
    total_pe = rect[0] * rect[1]

    # Physical units: the device always works in internal units (domain edge
    # 512, cell edge 512 / C, well conditioned in f32). The kernel 1/r is
    # homogeneous: with lengths scaled by s (physical units per internal unit),
    # potentials scale by 1/s and forces by 1/s^2; charges / masses are
    # unchanged. Relative FMM errors do not depend on s.
    s_phys = None
    if args.cell_size is not None:
        assert args.cell_size > 0, "--cell-size must be > 0"
        s_phys = args.cell_size / (512.0 / C)

    # bodies
    if args.bodies:
        bodies = np.load(args.bodies).astype(np.float64)
        assert bodies.ndim == 2 and bodies.shape[1] == 4, "--bodies: (N, 4) x, y, z, q"
        if s_phys is not None:
            bodies[:, :3] /= s_phys
        assert bodies[:, :3].min() >= 0.0 and bodies[:, :3].max() < 512.0, \
            "--bodies outside the domain [0, C * cell size)"
        bodies = bodies.astype(np.float32)
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
    import fmm_passes
    import fmm_ref
    # (step id, exported fn, label); pass k is step 2+k.
    STEPS = [(1, "run_p2m", "P2M")]
    STEPS += [(2 + k, "run_pass", f"pass {k} {fmm_passes.PASSES[k][0]}")
              for k in range(fmm_passes.NPASS)]
    STEP_L2P = 2 + fmm_passes.NPASS
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
    print(f"  Domain : {C}^3 cells (C={C}, tree height {fmm_passes.TREE_H})")
    print(f"  PEs    : ZP {zp} cells per PE, {pmap.nk} per column (block {pmap.bw}x{pmap.bh}), "
          f"rectangle {rect[0]} x {rect[1]} = {total_pe} PE")
    print(f"  Order  : P = {P}")
    print(f"  max_bodies_pe : {max_bodies_pe}")
    print(f"  Bodies : {N}")
    print(f"  Steps  : {' -> '.join(lbl for _, _, lbl in STEPS)}")
    print("=" * 64)

    # operator tables (generated on the fly, see fmm_tables.py)
    import fmm_tables
    math = fmm_tables.device_tables()
    for nm, n in fmm_tables.table_sizes().items():
        nc = int(compile_data["params"][f"{nm}_n_entries"])
        assert n == nc, f"{nm} table {n} entries != compiled {nc}: recompile"
    for nm in ("m2m", "m2l", "l2l"):
        s = math[nm]["signs"]
        assert s.dtype == np.uint32 and s.max() <= 15, f"{nm} signs bad"

    print("[1/4] Binning bodies per PE...")
    bodies_flat, col_len, CELL_BLOCK = build_bodies_tensor(bodies, C, max_bodies_pe)

    print(f"      P2P: {fmm_passes.p2p_cells(max_bodies_pe)} cells per message")
    if total_pe <= 4096:          # box-by-box brute force: ~15 s at 4096 PE
        zmp_tables.check()
        print("      transport tables: brute-force check OK")
    mp_hdr, mp_ent, cap = zmp_tables.device_tables(max_bodies_pe)
    assert cap == mp_ent_cap, f"mp_ent cap {cap} != compiled {mp_ent_cap}: recompile"
    print(f"      transport tables: mp_hdr {mp_hdr.shape[2]} u16, mp_ent cap {cap} entries per PE")

    err = None
    forces_3d = None
    tsc = None               # (H, W, NSTEP*9) u32 cycle counter log
    pass_stats = {}          # pass index -> (H, W, 8) u32
    stuck_done = None        # fmm_done per PE when a step did not complete
    leaf_checkpoints = {}    # pass index -> l_leaf after that pass
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
                "mp_hdr": runner.get_id("ptr_mp_hdr"),
                "mp_ent": runner.get_id("ptr_mp_ent"),
                "pass_stats": runner.get_id("ptr_pass_stats"),
                "pass_stats_all": runner.get_id("ptr_pass_stats_all"),
                "l_leaf": runner.get_id("ptr_l_leaf"),
                "tsc_log": runner.get_id("ptr_tsc_log"),
            }
            print("[3/4] Pushing data...")
            # math tables (broadcast)
            for nm, keys in (("m2m", ("offsets","child_indices","v_indices")),
                             ("m2l", ("offsets","source_indices","v_indices")),
                             ("l2l", ("offsets","source_indices","v_indices"))):
                h2d_bcast_u16(runner, sym[f"{nm}_offsets"], math[nm]["offsets"], rect)
                idxkey = "child_indices" if nm == "m2m" else "source_indices"
                h2d_bcast_u16(runner, sym[f"{nm}_{idxkey}"], math[nm][idxkey], rect)
                h2d_bcast_u16(runner, sym[f"{nm}_v_indices"], math[nm]["v_indices"], rect)
                h2d_bcast_u32(runner, sym[f"{nm}_signs"], math[nm]["signs"], rect)

            # bodies (per-PE)
            h2d_per_pe_u32(runner, sym["bodies_col"], bodies_flat, rect, col_len)
            # transport tables (per-PE, host-built in zmp_tables.py)
            h2d_per_pe_u16(runner, sym["mp_hdr"], mp_hdr, rect, mp_hdr.shape[2])
            h2d_per_pe_u16(runner, sym["mp_ent"], mp_ent, rect, mp_ent.shape[2])

            sym["fmm_done"] = runner.get_id("ptr_fmm_done")

            # PROBE 0: read fmm_done BEFORE any launch. The device is freshly
            # initialized; fmm_done is @zeros so this should return all zeros
            # quickly. If THIS d2h hangs, the problem is the memcpy/RPC layer
            # itself (or a version mismatch), NOT our FMM phase code.
            print("[probe] d2h fmm_done BEFORE any launch...", flush=True)
            tp = time.time()
            pre = np.zeros(total_pe, dtype=np.uint32)
            runner.memcpy_d2h(pre, sym["fmm_done"], 0, 0, rect[0], rect[1], 1,
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
                done = np.zeros(total_pe, dtype=np.uint32)
                while True:
                    runner.memcpy_d2h(done, sym["fmm_done"], 0, 0, rect[0], rect[1], 1,
                                      streaming=False,
                                      data_type=MemcpyDataType.MEMCPY_32BIT,
                                      nonblock=False, order=MemcpyOrder.ROW_MAJOR)
                    ndone = int((done >= pid).sum())
                    if ndone == total_pe or time.time() - t1 > args.phase_timeout:
                        break
                    time.sleep(0.5)
                print(f"        fmm_done = {ndone}/{total_pe} PEs  "
                      f"(waited {time.time()-t1:.3f} s)", flush=True)
                if ndone != total_pe:
                    missing = [pmap.pe_to_column(int(i % rect[0]), int(i // rect[0]))[:3]
                               for i in np.where(done < pid)[0][:12]]
                    print(f"        NOT done (first 12, as column (x,y) chunk k): {missing}",
                          flush=True)
                    print(f"        stopping: {label} did not complete on every PE",
                          flush=True)
                    err = f"{label} incomplete ({ndone}/{total_pe})"
                    if phase == "run_pass":
                        # Counters of the pass that did not complete: the PEs
                        # that completed it have already prepared the next pass
                        # (live pass_stats reset), their copy is in
                        # pass_stats_all; the others: live counters.
                        k = pid - 2
                        live = d2h_per_pe_u32(runner, sym["pass_stats"], rect, 8)
                        allst = d2h_per_pe_u32(runner, sym["pass_stats_all"], rect,
                                               fmm_passes.NPASS * 8)
                        fin = (done >= pid).reshape(rect[1], rect[0])
                        pass_stats[k] = np.where(fin[:, :, None],
                                                 allst[:, :, k * 8:(k + 1) * 8], live)
                        stuck_done = done.reshape(rect[1], rect[0]).copy()
                    break
                completed.append(pid)
                if phase == "run_pass":
                    k = pid - 2
                    if k in checkpoint_passes():
                        leaf_checkpoints[k] = d2h_per_pe_f32(
                            runner, sym["l_leaf"], rect, zp * fmm_ref.C_BUF_SIZE)

            # Per-pass counters, copied by each PE when it completed the pass.
            done_passes = [pid - 2 for pid in completed if 2 <= pid < STEP_L2P]
            if done_passes:
                allst = d2h_per_pe_u32(runner, sym["pass_stats_all"], rect,
                                       fmm_passes.NPASS * 8)
                for k in done_passes:
                    pass_stats[k] = allst[:, :, k * 8:(k + 1) * 8]

            # per-PE cycle counter log first (small; read even if a step did not
            # complete, and kept even if the forces d2h fails)
            tsc = d2h_per_pe_u32(runner, sym["tsc_log"], rect,
                                 len(tsc_step_names()) * 9)

            # pull forces
            print("      pulling forces (d2h forces_col)...", flush=True)
            tf = time.time()
            forces_3d = d2h_per_pe_f32(runner, sym["forces_col"],
                                       rect, zp * max_bodies_pe * FLOATS_PER_BODY)
            print(f"      forces d2h {time.time()-tf:.3f} s", flush=True)
    except Exception as e:
        err = repr(e)
        print(f"\n[appliance_run] ERROR: {err}")

    if tsc is not None:
        print_tsc(tsc)

    if forces_3d is not None and STEP_L2P in completed:
        print("\n--- physics: device forces (complete FMM) vs direct N-body ---")
        if not physics_check_forces(forces_3d, bodies, C, max_bodies_pe,
                                    args.check_sample) and err is None:
            err = "force physics check failed"

    if pass_stats:
        # Save what came off the device, so the verification can be re-run
        # offline: python appliance_run.py --verify-only runs/<label>_runNN
        np.savez(os.path.join(run_dir, "device_data.npz"), bodies=bodies,
                 C=C, order_p=P, zp=zp, blk_lbw=blk_lbw, max_bodies_pe=max_bodies_pe,
                 **{f"stats_{k}": v for k, v in pass_stats.items()},
                 **{f"leaf_{k}": v for k, v in leaf_checkpoints.items()},
                 **({"forces": forces_3d} if forces_3d is not None and STEP_L2P in completed else {}),
                 **({"tsc": tsc} if tsc is not None else {}),
                 **({"fmm_done": stuck_done} if stuck_done is not None else {}))
        ok = verify_passes(pass_stats, leaf_checkpoints, bodies, C, max_bodies_pe,
                          args.check_sample)
        if not ok and err is None:
            err = "verification failed"

    if forces_3d is not None:
        np.save(os.path.join(run_dir, "forces_device.npy"), forces_3d)
        if s_phys is not None and STEP_L2P in completed:
            # (N, 4) fx, fy, fz, pot per body in the --bodies order, physical
            # length unit; the physical constant (G, k_e) is not applied.
            per = device_forces_per_body(forces_3d, bodies, C, max_bodies_pe)
            per[:, :3] /= s_phys ** 2
            per[:, 3] /= s_phys
            np.save(os.path.join(run_dir, "forces_phys.npy"), per)
            print(f"[appliance_run] forces_phys.npy: cell edge {args.cell_size} "
                  f"(scale {s_phys:.6g} per internal unit)")

    sys.stdout = original_stdout
    with open(os.path.join(run_dir, "run.log"), "w") as f:
        f.write(log_buf.getvalue())
    if err:
        sys.exit(1)


if __name__ == "__main__":
    main()
