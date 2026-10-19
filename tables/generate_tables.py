#!/usr/bin/env python3
# gen_agendas25d.py
#
# Generator for the 2.5D routing agendas and quotas, plus the reused operator
# math tables. Replaces the Morton-based routing of the 3D generator with the
# cartesian XY-routing model verified during design.
#
# Output (pickle) consumed by the host loader -> memcpy'd per PE:
#   math:  m2m / m2l / l2l  (offsets, indices, v_indices, signs)  [reused]
#   routing (per PE (px,py), per level):
#       m2l_agenda   : list of (tgt_x, tgt_y, tag, zbase, zcount) messages
#       recv_expected: # messages this PE will RECEIVE (own) at this level
#       fwd_expected : # messages this PE will FORWARD at this level
#       send_expected: # messages this PE will ORIGINATE at this level
#   p2p_agenda: the <=8 plane neighbours (1 hop)
#
# Geometry: cartesian, no Morton. PE(px,py) hosts a column of C cells along Z.
# Message granularity K_CUBES: a plane M2L message carries up to K source Z-cube
# multipoles bound for one destination plane column.

import numpy as np
import pickle

# ---------------------------------------------------------------------------
# Parameters (must match constants.csl)
# ---------------------------------------------------------------------------
P = 4
C = 16                 # cells_per_side
TREE_H = 4
K_CUBES = 4            # must match constants.csl K_CUBES
N_COEFFS   = (P + 1) * (P + 2) // 2
N_COEFFS_V = (2*P + 1) * (2*P + 2) // 2


# ===========================================================================
# OPERATOR MATH TABLES  (reused verbatim from the 3D generator; geometry-free)
# ===========================================================================
def get_mem_idx(l, m):
    pos = 0
    for mm in range(m):
        pos += (P + 1 - mm)
    pos += (l - m)
    return pos * 2

def get_v_mem_idx(l, m):
    pos = 0
    for mm in range(m):
        pos += (2*P + 1 - mm)
    pos += (l - m)
    return pos * 2

def parity_sign(k):
    return 1.0 if (abs(k) % 2 == 0) else -1.0

def pack_signs(s_rr, s_ri, s_ir, s_ii):
    bits = 0
    if s_rr < 0: bits |= 1
    if s_ri < 0: bits |= 2
    if s_ir < 0: bits |= 4
    if s_ii < 0: bits |= 8
    return bits

def gen_m2m_tables():
    father = {(l, m): [] for l in range(P+1) for m in range(l+1)}
    for n in range(P + 1):
        for k in range(-n, n + 1):
            ci = get_mem_idx(n, abs(k))
            if k < 0: csr, csi = parity_sign(k), -parity_sign(k)
            else:     csr, csi = 1.0, 1.0
            for v_l in range(P - n + 1):
                for v_m in range(-v_l, v_l + 1):
                    l_f, m_f = n + v_l, k + v_m
                    if m_f < 0: continue
                    vi = get_mem_idx(v_l, abs(v_m))
                    if v_m < 0: vsr, vsi = parity_sign(v_m), -parity_sign(v_m)
                    else:       vsr, vsi = 1.0, 1.0
                    father[(l_f, m_f)].append((ci, vi,
                        pack_signs(csr*vsr, -(csi*vsi), csr*vsi, csi*vsr)))
    return _flatten(father)

def gen_m2l_tables():
    target = {(l, m): [] for l in range(P+1) for m in range(l+1)}
    for n in range(P + 1):
        for k in range(-n, n + 1):
            si = get_mem_idx(n, abs(k))
            if k < 0: csr, csi = parity_sign(k), -parity_sign(k)
            else:     csr, csi = 1.0, 1.0
            for v_l in range(2*P + 1):
                for v_m in range(-v_l, v_l + 1):
                    l_f, m_f = v_l - n, k + v_m
                    if l_f < 0 or l_f > P: continue
                    if m_f < 0 or m_f > l_f: continue
                    vi = get_v_mem_idx(v_l, abs(v_m))
                    if v_m < 0: vsr, vsi = parity_sign(v_m), -parity_sign(v_m)
                    else:       vsr, vsi = 1.0, 1.0
                    target[(l_f, m_f)].append((si, vi,
                        pack_signs(csr*vsr, -(csi*vsi), csr*vsi, csi*vsr)))
    return _flatten(target)

def gen_l2l_tables():
    return gen_m2m_tables()

def _flatten(entries):
    offsets, idxA, idxV, signs = [], [], [], []
    curr = 0
    for m in range(P + 1):
        for l in range(m, P + 1):
            offsets.append(curr)
            for (a, v, s) in entries[(l, m)]:
                idxA.append(a); idxV.append(v); signs.append(s)
            curr += len(entries[(l, m)])
    offsets.append(curr)
    return offsets, idxA, idxV, signs


# ===========================================================================
# 2.5D GEOMETRY  (cartesian; mirror of geometry25d.py)
# ===========================================================================
def cells_per_box(level):
    return 1 << (TREE_H - level)

def grid_boxes(level):
    return 1 << level

def spatial_tag(dx, dy, dz):
    return (dx + 3) + (dy + 3) * 7 + (dz + 3) * 49


# ===========================================================================
# XY DIMENSION-ORDER ROUTING  (verified acyclic)
# ===========================================================================
def xy_route(sx, sy, tx, ty):
    """List of PEs on the XY path from src to dst, inclusive of both ends."""
    pts = [(sx, sy)]
    x, y = sx, sy
    while x != tx:
        x += 1 if tx > x else -1
        pts.append((x, y))
    while y != ty:
        y += 1 if ty > y else -1
        pts.append((x, y))
    return pts


# ===========================================================================
# M2L INTERACTION LIST per level, split into K-cube messages
# ===========================================================================
# At a given level, a source plane column (reprsented by its representative PE)
# interacts with destination plane columns at planar offsets (dx,dy) with
# 2 <= max(|dx|,|dy|,|dz|) <= 3. The source sends its Z-cube multipoles for that
# level (nz = grid_boxes(level) of them) toward each destination column, split
# into ceil(nz/K) messages of up to K cubes each.
#
# For routing/quotas we work in PLANE-PE coordinates. A level-`level` box's
# representative PE is at (bx*s, by*s) with s = cells_per_box(level). Two boxes
# at planar offset (dx,dy) have representatives s*dx, s*dy PE apart.

def il_planar_offsets():
    """Distinct planar (dx,dy) offsets in the M2L interaction list (excl 0,0)."""
    offs = set()
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                if 2 <= max(abs(dx), abs(dy), abs(dz)) <= 3 and (dx, dy) != (0, 0):
                    offs.add((dx, dy))
    return sorted(offs)


def representative_pes(level):
    """All representative PEs (plane positions) of boxes at this level."""
    s = cells_per_box(level)
    gb = grid_boxes(level)
    return [(bx * s, by * s) for bx in range(gb) for by in range(gb)]


def gen_m2l_routing():
    """
    Build, per PE and per level:
      - m2l_agenda: list of (tgt_x, tgt_y, tag, zbase, zcount)
      - send_expected, recv_expected, fwd_expected
    """
    MSG_TYPE_M2L = 1
    # per-PE per-level dicts
    agenda = {}          # (px,py) -> level -> list of message tuples
    send_exp = {}        # (px,py) -> level -> int
    recv_exp = {}        # (px,py) -> level -> int
    fwd_exp = {}         # (px,py) -> level -> int

    def ensure(px, py, level):
        agenda.setdefault((px, py), {}).setdefault(level, [])
        send_exp.setdefault((px, py), {}).setdefault(level, 0)
        recv_exp.setdefault((px, py), {}).setdefault(level, 0)
        fwd_exp.setdefault((px, py), {}).setdefault(level, 0)

    OFF = il_planar_offsets()

    for level in range(2, TREE_H + 1):
        s = cells_per_box(level)
        nz = grid_boxes(level)                 # cubes along Z at this level
        n_msgs_per_dest = (nz + K_CUBES - 1) // K_CUBES
        reps = representative_pes(level)
        rep_set = set(reps)

        for (sx, sy) in reps:
            # source box plane coords
            sbx, sby = sx // s, sy // s
            for (dx, dy) in OFF:
                tbx, tby = sbx + dx, sby + dy
                # destination box must be inside the grid at this level
                if not (0 <= tbx < grid_boxes(level) and 0 <= tby < grid_boxes(level)):
                    continue
                tx, ty = tbx * s, tby * s
                # split the nz source cubes into K-sized messages
                for mi in range(n_msgs_per_dest):
                    zbase = mi * K_CUBES
                    zcount = min(K_CUBES, nz - zbase)
                    # tag encodes the planar translation; dz handled per-cube at
                    # the receiver, so we store the planar (dx,dy) with dz=0 as a
                    # representative tag (the receiver reconstructs per-cube dz).
                    tag = spatial_tag(dx, dy, 0)
                    ensure(sx, sy, level)
                    agenda[(sx, sy)][level].append((tx, ty, tag, zbase, zcount))
                    send_exp[(sx, sy)][level] += 1
                    # receiver quota
                    ensure(tx, ty, level)
                    recv_exp[(tx, ty)][level] += 1
                    # forwarders quota: every intermediate PE on the XY path
                    path = xy_route(sx, sy, tx, ty)
                    for pe in path[1:-1]:
                        ensure(pe[0], pe[1], level)
                        fwd_exp[pe][level] += 1

    return agenda, send_exp, recv_exp, fwd_exp


# ===========================================================================
# P2P agenda: the <=8 plane neighbours at 1 hop
# ===========================================================================
def gen_p2p_agendas():
    sched = {}
    for px in range(C):
        for py in range(C):
            nbrs = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    tx, ty = px + dx, py + dy
                    if 0 <= tx < C and 0 <= ty < C:
                        nbrs.append((tx, ty))
            sched[(px, py)] = nbrs
    return sched


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print("Generating 2.5D FMM data (cartesian, K=%d cubes/msg)...\n" % K_CUBES)

    print("[1/3] operator math tables (reused)...")
    m2m = gen_m2m_tables()
    m2l = gen_m2l_tables()
    l2l = gen_l2l_tables()
    print(f"      M2M={len(m2m[1])}, M2L={len(m2l[1])}, L2L={len(l2l[1])}")
    assert len(m2m[1]) == 92 and len(m2l[1]) == 375 and len(l2l[1]) == 92

    print("[2/3] M2L routing agendas + quotas (XY dimension-order)...")
    agenda, send_exp, recv_exp, fwd_exp = gen_m2l_routing()
    p2p = gen_p2p_agendas()

    # sanity: global conservation per level (sum send == sum recv)
    for level in range(2, TREE_H + 1):
        s_tot = sum(send_exp.get(pe, {}).get(level, 0) for pe in send_exp)
        r_tot = sum(recv_exp.get(pe, {}).get(level, 0) for pe in recv_exp)
        f_tot = sum(fwd_exp.get(pe, {}).get(level, 0) for pe in fwd_exp)
        assert s_tot == r_tot, f"level {level}: send {s_tot} != recv {r_tot}"
        print(f"      level {level}: sent==recv={s_tot}, forwards={f_tot}")

    print("[3/3] packing blob...")
    data = {
        "math": {
            "m2m": {"offsets": np.array(m2m[0], np.uint16),
                    "child_indices": np.array(m2m[1], np.uint16),
                    "v_indices": np.array(m2m[2], np.uint16),
                    "signs": np.array(m2m[3], np.uint32)},
            "m2l": {"offsets": np.array(m2l[0], np.uint16),
                    "source_indices": np.array(m2l[1], np.uint16),
                    "v_indices": np.array(m2l[2], np.uint16),
                    "signs": np.array(m2l[3], np.uint32)},
            "l2l": {"offsets": np.array(l2l[0], np.uint16),
                    "source_indices": np.array(l2l[1], np.uint16),
                    "v_indices": np.array(l2l[2], np.uint16),
                    "signs": np.array(l2l[3], np.uint32)},
        },
        "routing": {
            "m2l_agenda": agenda,
            "send_expected": send_exp,
            "recv_expected": recv_exp,
            "fwd_expected": fwd_exp,
            "p2p_agenda": p2p,
        },
        "params": {"P": P, "C": C, "TREE_H": TREE_H, "K_CUBES": K_CUBES},
    }
    with open("fmm25d_data.pkl", "wb") as f:
        pickle.dump(data, f)
    print("\nWrote fmm25d_data.pkl")

    # report the busiest PE
    busiest = max(fwd_exp, key=lambda pe: sum(fwd_exp[pe].values()))
    print(f"  busiest forwarder: PE{busiest} total forwards = "
          f"{sum(fwd_exp[busiest].values())}")


if __name__ == "__main__":
    main()
