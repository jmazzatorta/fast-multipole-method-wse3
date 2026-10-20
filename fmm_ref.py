#!/usr/bin/env python3
"""fmm_ref.py - float32 NumPy reference of the 2.5D FMM that runs on the WSE-3.

Mirrors the device math in kernels.csl / pe.csl / geometry.csl:

  kernels.csl  compute_solid_harmonics      (order P,  30 floats)
  kernels.csl  compute_solid_harmonics_2p   (order 2P, 90 floats)
  kernels.csl  translate                    (CSR loop, packed 4-bit signs)
  kernels.csl  evaluate_l2p                 (potential and force = -grad phi)
  pe.csl       compute_p2m_cell             (multipole about the leaf centre)
  pe.csl       m2m_z_local + plane combine  (upward pass, levels 4 -> 3 -> 2)
  pe.csl       apply_m2l_message            (M2L, physical vector d * box_size)
  pe.csl       l2l_z_local + plane copy     (downward pass, levels 2 -> 3 -> 4)
  pe.csl       run_l2p, apply_p2p_message / p2p_self_and_zlocal

Every kernel is vectorised over a leading batch axis, but each batch element
goes through exactly the scalar f32 operation sequence of the CSL code (same
loop order, same left-to-right association, one rounding per operation).
NumPy >= 2 (NEP 50) keeps np.float32 arrays/scalars in float32 when combined
with Python scalars; all constants are additionally wrapped in np.float32.

Internal storage of expansions is box-indexed: an array of shape
(n, n, n, 30) indexed [bx, by, bz, k] at a level with n = 2**L boxes per side.
The device layout of a level buffer on PE(px, py) is m_lvlL[z*30 + k]; the
host tensor convention (appliance_run.py) is (py, px, ...). to_device_layout()
converts.
"""
import os

import numpy as np

f32 = np.float32

# --------------------------------------------------------------------------
# Constants (constants.csl / layout)
# --------------------------------------------------------------------------
P = 4
N_COEFFS = (P + 1) * (P + 2) // 2                 # 15
C_BUF_SIZE = 2 * N_COEFFS                          # 30
N_COEFFS_M2L = (2 * P + 1) * (2 * P + 2) // 2      # 45
C_BUF_M2L_SIZE = 2 * N_COEFFS_M2L                  # 90

PROB_RANGE = f32(512.0)
C = 16                       # cells_per_side
TREE_H = 4
CELL_SIZE = f32(PROB_RANGE / f32(C))               # 32.0
BOX_SIZE_AT_LEVEL = [f32(PROB_RANGE / f32(2.0 ** L)) for L in range(TREE_H + 1)]
MAX_BODIES_PE = 8

import fmm_tables   # device operator tables (generated, no data file)


# --------------------------------------------------------------------------
# Operator tables
# --------------------------------------------------------------------------
class OpTable:
    """One CSR operator table (m2m, m2l or l2l, from fmm_tables), with the
    packed 4-bit signs pre-decoded exactly as kernels.csl decode_sign."""

    def __init__(self, d, idx_key):
        self.offsets = np.asarray(d["offsets"], dtype=np.uint16).astype(np.int64)
        self.src_indices = np.asarray(d[idx_key], dtype=np.uint16).astype(np.int64)
        self.v_indices = np.asarray(d["v_indices"], dtype=np.uint16).astype(np.int64)
        signs = np.asarray(d["signs"], dtype=np.uint32)
        s16 = (signs & np.uint32(0xFFFF)).astype(np.int64)
        # decode_sign: +1.0 if the bit is 0, -1.0 if the bit is 1
        self.s = [np.where((s16 >> b) & 1, f32(-1.0), f32(1.0)).astype(f32)
                  for b in range(4)]   # rr, ri, ir, ii
        # Pre-build per-entry python tuples for the hot loop.
        self.entries = []
        for fi in range(N_COEFFS):
            lst = []
            for i in range(int(self.offsets[fi]), int(self.offsets[fi + 1])):
                lst.append((int(self.src_indices[i]), int(self.v_indices[i]),
                            f32(self.s[0][i]), f32(self.s[1][i]),
                            f32(self.s[2][i]), f32(self.s[3][i])))
            self.entries.append(lst)


_TABLES = None


def load_tables():
    """Return dict op -> OpTable (cached):
      'm2m', 'm2l', 'l2l'       : the device tables (fmm_tables.py)
      'm2l_fixed', 'l2l_fixed'  : corrected tables (same CSR / packed-sign
                                  format, see gen_*_fixed below)."""
    global _TABLES
    if _TABLES is None:
        m = fmm_tables.device_tables()
        # 'm2l' (ops='csl') = the FORMER device table (regular harmonics), to
        # reproduce older runs; 'm2l_fixed' = the current device table.
        _TABLES = {"m2m": OpTable(m["m2m"], "child_indices"),
                   "m2l": OpTable(fmm_tables._pack(fmm_tables.gen_m2l_tables_regular(),
                                                   "source_indices"), "source_indices"),
                   # 'l2l' (ops='csl') = the FORMER device table (= M2M table)
                   "l2l": OpTable(fmm_tables._pack(fmm_tables.gen_m2m_tables(),
                                                   "source_indices"), "source_indices"),
                   "m2l_fixed": OpTable(gen_m2l_table_fixed(), "source_indices"),
                   "l2l_fixed": OpTable(gen_l2l_table_fixed(), "source_indices")}
    return _TABLES


# --------------------------------------------------------------------------
# Corrected operator tables (diagnostic / proposed fix), same format as the
# generator in gen_agendas25d.py: CSR over target (l,m>=0) in m-outer/l-inner
# order, entries (source float idx, v float idx, packed signs).
#
# Normalisation (that of the device recurrences, verified numerically):
#   R_l^m = r^l P_l^m(cos t) e^{i m phi} / (l+m)!      (regular, P2M/M2M/L2P)
#   I_l^m = (l-m)! P_l^m(cos t) e^{i m phi} / r^{l+1}  (irregular)
#   X^{-m} = (-1)^m conj(X^m) for both.
#   M_n^k = sum q R_n^k(x - c);  1/|u - a| = sum conj(R_n^k(a)) I_n^k(u);
#   phi(y) = sum_{j,p} L_j^p R_j^p(y)  (what evaluate_l2p computes).
# Then:
#   M2L: L_j^p = (-1)^j sum_{n,k} M_n^k conj(I_{n+j}^{p+k}(v)), v = c_t - c_s
#   L2L: Lc_n^k = sum_{j>=n,p} Lp_j^p R_{j-n}^{p-k}(w),           w = c_c - c_p
# --------------------------------------------------------------------------
def _gen_mem_idx(l, m, order):
    pos = 0
    for mm in range(m):
        pos += order + 1 - mm
    pos += l - m
    return pos * 2


def _pack_signs(s_rr, s_ri, s_ir, s_ii):
    bits = 0
    if s_rr < 0: bits |= 1
    if s_ri < 0: bits |= 2
    if s_ir < 0: bits |= 4
    if s_ii < 0: bits |= 8
    return bits


def _par(k):
    return 1.0 if abs(k) % 2 == 0 else -1.0


def _flatten(target):
    offsets, idxA, idxV, signs = [], [], [], []
    cur = 0
    for m in range(P + 1):
        for l in range(m, P + 1):
            offsets.append(cur)
            for (a, v, s) in target[(l, m)]:
                idxA.append(a); idxV.append(v); signs.append(s)
            cur += len(target[(l, m)])
    offsets.append(cur)
    return {"offsets": np.array(offsets, np.uint16),
            "source_indices": np.array(idxA, np.uint16),
            "v_indices": np.array(idxV, np.uint16),
            "signs": np.array(signs, np.uint32)}


def _src_signs(k):
    """(sign_re, sign_im) to read X^k from the stored X^{|k|}."""
    return (_par(k), -_par(k)) if k < 0 else (1.0, 1.0)


def gen_m2l_table_fixed():
    """Corrected M2L table for v_buf = irregular harmonics I(v) of order 2P
    (compute_irregular_harmonics_2p). 375 entries, like the device table."""
    target = {(l, m): [] for l in range(P + 1) for m in range(l + 1)}
    for j in range(P + 1):
        for p in range(j + 1):
            g = -1.0 if j % 2 else 1.0
            for n in range(P + 1):
                for k in range(-n, n + 1):
                    csr, csi = _src_signs(k)
                    vl, vm = n + j, p + k
                    # conj(I^vm) from stored I^{|vm|}
                    if vm >= 0:
                        vsr, vsi = 1.0, -1.0
                    else:
                        vsr, vsi = _par(vm), _par(vm)
                    target[(j, p)].append((
                        _gen_mem_idx(n, abs(k), P),
                        _gen_mem_idx(vl, abs(vm), 2 * P),
                        _pack_signs(g * csr * vsr, -g * csi * vsi,
                                    g * csr * vsi, g * csi * vsr)))
    return _flatten(target)


def gen_l2l_table_fixed():
    """Corrected L2L table for v_buf = regular harmonics R(w) of order P,
    w = child centre - parent centre (the device's negated
    get_translation_vector)."""
    target = {(l, m): [] for l in range(P + 1) for m in range(l + 1)}
    for n in range(P + 1):
        for k in range(n + 1):
            for j in range(n, P + 1):
                for p in range(-j, j + 1):
                    vl, vm = j - n, p - k
                    if abs(vm) > vl:
                        continue
                    csr, csi = _src_signs(p)
                    vsr, vsi = _src_signs(vm)
                    target[(n, k)].append((
                        _gen_mem_idx(j, abs(p), P),
                        _gen_mem_idx(vl, abs(vm), P),
                        _pack_signs(csr * vsr, -(csi * vsi),
                                    csr * vsi, csi * vsr)))
    return _flatten(target)


# --------------------------------------------------------------------------
# kernels.csl: solid harmonics (Cartesian recurrence), batched
# --------------------------------------------------------------------------
def _solid_harmonics(dx, dy, dz, order):
    """Port of compute_solid_harmonics{,_2p}. dx,dy,dz: f32 arrays of equal
    shape (...). Returns (..., 2*(order+1)(order+2)/2) f32, interleaved re/im,
    ordered m-outer / l-inner."""
    dx = np.asarray(dx, dtype=f32)
    dy = np.asarray(dy, dtype=f32)
    dz = np.asarray(dz, dtype=f32)
    nbuf = (order + 1) * (order + 2)
    buf = np.zeros(dx.shape + (nbuf,), dtype=f32)

    # r2 = dx*dx + dy*dy + dz*dz  (left to right)
    r2 = (dx * dx + dy * dy) + dz * dz
    mm_r = np.ones_like(dx)
    mm_c = np.zeros_like(dx)
    i_m = 0
    for m in range(order + 1):
        if m > 0:
            k_diag = f32(f32(-1.0) / (f32(2.0) * f32(m)))
            temp_r = k_diag * (dx * mm_r - dy * mm_c)
            temp_c = k_diag * (dy * mm_r + dx * mm_c)
            mm_r = temp_r
            mm_c = temp_c
        buf[..., i_m] = mm_r
        buf[..., i_m + 1] = mm_c
        i_m += 2
        prev2_r = np.zeros_like(dx)
        prev2_c = np.zeros_like(dx)
        prev1_r = mm_r
        prev1_c = mm_c
        for l in range(m + 1, order + 1):
            k_lm = f32(f32(1.0) / f32(l * l - m * m))
            z_k = (f32(2.0) * f32(l) - f32(1.0)) * dz
            cur_r = k_lm * (z_k * prev1_r - r2 * prev2_r)
            cur_c = k_lm * (z_k * prev1_c - r2 * prev2_c)
            buf[..., i_m] = cur_r
            buf[..., i_m + 1] = cur_c
            i_m += 2
            prev2_r, prev2_c = prev1_r, prev1_c
            prev1_r, prev1_c = cur_r, cur_c
    return buf


def compute_solid_harmonics(dx, dy, dz):
    """kernels.csl compute_solid_harmonics: order P -> (..., 30) f32."""
    return _solid_harmonics(dx, dy, dz, P)


def compute_solid_harmonics_2p(dx, dy, dz):
    """kernels.csl compute_solid_harmonics_2p: order 2P -> (..., 90) f32."""
    return _solid_harmonics(dx, dy, dz, 2 * P)


def compute_irregular_harmonics_2p(dx, dy, dz):
    """PROPOSED device kernel (not in the CSL): irregular solid harmonics
    I_l^m(v), order 2P, same buffer layout as compute_solid_harmonics_2p.
    Trig-free Cartesian recurrence, one sqrt and one division:
      I_0^0 = 1/r
      I_m^m = -(2m-1) (x + i y) / r^2 * I_{m-1}^{m-1}
      I_l^m = ((2l-1) z I_{l-1}^m - ((l-1)^2 - m^2) I_{l-2}^m) / r^2
    """
    dx = np.asarray(dx, dtype=f32)
    dy = np.asarray(dy, dtype=f32)
    dz = np.asarray(dz, dtype=f32)
    order = 2 * P
    buf = np.zeros(dx.shape + ((order + 1) * (order + 2),), dtype=f32)
    r2 = (dx * dx + dy * dy) + dz * dz
    inv_r2 = f32(1.0) / r2
    mm_r = f32(1.0) / np.sqrt(r2)
    mm_c = np.zeros_like(dx)
    i_m = 0
    for m in range(order + 1):
        if m > 0:
            k_diag = f32(-(2 * m - 1)) * inv_r2
            temp_r = k_diag * (dx * mm_r - dy * mm_c)
            temp_c = k_diag * (dy * mm_r + dx * mm_c)
            mm_r, mm_c = temp_r, temp_c
        buf[..., i_m] = mm_r
        buf[..., i_m + 1] = mm_c
        i_m += 2
        prev2_r = np.zeros_like(dx); prev2_c = np.zeros_like(dx)
        prev1_r = mm_r; prev1_c = mm_c
        for l in range(m + 1, order + 1):
            a = f32(2 * l - 1) * dz
            b = f32((l - 1) * (l - 1) - m * m)
            cur_r = (a * prev1_r - b * prev2_r) * inv_r2
            cur_c = (a * prev1_c - b * prev2_c) * inv_r2
            buf[..., i_m] = cur_r
            buf[..., i_m + 1] = cur_c
            i_m += 2
            prev2_r, prev2_c = prev1_r, prev1_c
            prev1_r, prev1_c = cur_r, cur_c
    return buf


# --------------------------------------------------------------------------
# kernels.csl: translate (unified M2M / M2L / L2L), batched
# --------------------------------------------------------------------------
def translate(v, src, tgt, table):
    """Port of kernels.csl translate. ACCUMULATES into tgt (in place, +=).

    v:   (K, 30|90) or (30|90,) translation-vector harmonics (broadcast)
    src: (K, 30) source expansions
    tgt: (K, 30) target expansions, modified in place and returned
    """
    v = np.asarray(v, dtype=f32)
    src = np.asarray(src, dtype=f32)
    if v.ndim == 1:
        v = v[None, :]
    K = src.shape[0]
    for fi in range(N_COEFFS):
        ar = np.zeros(K, dtype=f32)
        ai = np.zeros(K, dtype=f32)
        for (ci, vi, s_rr, s_ri, s_ir, s_ii) in table.entries[fi]:
            c_r = src[:, ci]
            c_i = src[:, ci + 1]
            v_r = v[:, vi]
            v_i = v[:, vi + 1]
            # accum_r += s_rr*c_r*v_r + s_ri*c_i*v_i ; accum_i likewise
            ar = ar + ((s_rr * c_r) * v_r + (s_ri * c_i) * v_i)
            ai = ai + ((s_ir * c_r) * v_i + (s_ii * c_i) * v_r)
        tgt[:, 2 * fi] = tgt[:, 2 * fi] + ar
        tgt[:, 2 * fi + 1] = tgt[:, 2 * fi + 1] + ai
    return tgt


# --------------------------------------------------------------------------
# kernels.csl: lm_index / R_signed / evaluate_l2p, batched over bodies
# --------------------------------------------------------------------------
def lm_index(l, m):
    """Float index of Re R_l^m in an order-P buffer (kernels.csl lm_index)."""
    pos = 0
    for mm in range(m):
        pos += P + 1 - mm
    pos += l - m
    return pos * 2


def _R_signed(r_buf, l, m):
    """kernels.csl R_signed: (re, im) arrays of R_l^m, zero if out of range,
    with R_l^{-m} = (-1)^m conj(R_l^m)."""
    am = -m if m < 0 else m
    K = r_buf.shape[0]
    if l < 0 or am > l:
        z = np.zeros(K, dtype=f32)
        return z, z
    base = lm_index(l, am)
    rr = r_buf[:, base]
    ri = r_buf[:, base + 1]
    if m >= 0:
        return rr, ri
    sgn = f32(1.0) if am % 2 == 0 else f32(-1.0)
    return sgn * rr, -sgn * ri


def evaluate_l2p(dx, dy, dz, l_coeffs):
    """Port of kernels.csl evaluate_l2p, batched over K evaluation points.
    dx,dy,dz: (K,) f32 offsets point - expansion centre; l_coeffs: (K, 30).
    Returns pot, fx, fy, fz, each (K,) f32. Force = -grad(phi)."""
    l_coeffs = np.asarray(l_coeffs, dtype=f32)
    r_buf = compute_solid_harmonics(dx, dy, dz)
    K = r_buf.shape[0]
    p_acc = np.zeros(K, dtype=f32)
    gx = np.zeros(K, dtype=f32)
    gy = np.zeros(K, dtype=f32)
    gz = np.zeros(K, dtype=f32)
    half = f32(0.5)
    idx = 0
    for m in range(P + 1):
        w = f32(1.0) if m == 0 else f32(2.0)
        for l in range(m, P + 1):
            l_r = l_coeffs[:, idx * 2]
            l_i = l_coeffs[:, idx * 2 + 1]
            r_r = r_buf[:, idx * 2]
            r_i = r_buf[:, idx * 2 + 1]
            p_acc = p_acc + w * (l_r * r_r - l_i * r_i)
            b1_r, b1_i = _R_signed(r_buf, l - 1, m - 1)
            b2_r, b2_i = _R_signed(r_buf, l - 1, m + 1)
            bz_r, bz_i = _R_signed(r_buf, l - 1, m)
            dRx_r = f32(-0.5) * b1_r + half * b2_r
            dRx_i = f32(-0.5) * b1_i + half * b2_i
            sum_r = b1_r + b2_r
            sum_i = b1_i + b2_i
            dRy_r = half * sum_i
            dRy_i = f32(-0.5) * sum_r
            dRz_r = bz_r
            dRz_i = bz_i
            gx = gx + w * (l_r * dRx_r - l_i * dRx_i)
            gy = gy + w * (l_r * dRy_r - l_i * dRy_i)
            gz = gz + w * (l_r * dRz_r - l_i * dRz_i)
            idx += 1
    return p_acc, -gx, -gy, -gz


# --------------------------------------------------------------------------
# geometry.csl
# --------------------------------------------------------------------------
def get_translation_vector(child_id, l):
    """geometry.csl get_translation_vector: +half_child_size where the
    child_id bit is 0, -half_child_size where it is 1 (bit0=X, 1=Y, 2=Z).
    child_id may be an int array; returns three f32 arrays."""
    child_id = np.asarray(child_id)
    power_of_2 = f32(1 << l)
    half_child = f32(f32(PROB_RANGE / power_of_2) * f32(0.5))
    vx = np.where((child_id & 1) == 0, half_child, -half_child).astype(f32)
    vy = np.where((child_id & 2) == 0, half_child, -half_child).astype(f32)
    vz = np.where((child_id & 4) == 0, half_child, -half_child).astype(f32)
    return vx, vy, vz


def get_physical_distances(dx_idx, dy_idx, dz_idx, box_size):
    """geometry.csl get_physical_distances."""
    bs = f32(box_size)
    return (f32(dx_idx) * bs, f32(dy_idx) * bs, f32(dz_idx) * bs)


# --------------------------------------------------------------------------
# Bodies: binning exactly as appliance_run.build_bodies_tensor
# --------------------------------------------------------------------------
def bin_bodies(bodies, max_bodies=MAX_BODIES_PE):
    """bodies (N,4) f32 (x,y,z,q). Returns
      cells  : (C,C,C,max_bodies,4) f32 indexed [cx,cy,cz,slot]
      counts : (C,C,C) int
      where  : (N,4) int array (cx,cy,cz,slot) per body (input order)
    Slot order within a cell = input order (build_bodies_tensor)."""
    bodies = np.asarray(bodies, dtype=f32)
    cell_size = 512.0 / C   # python float, as in build_bodies_tensor
    cells = np.zeros((C, C, C, max_bodies, 4), dtype=f32)
    counts = np.zeros((C, C, C), dtype=np.int64)
    where = np.zeros((len(bodies), 4), dtype=np.int64)
    for i, (x, y, z, q) in enumerate(bodies):
        cx = min(max(int(x / cell_size), 0), C - 1)
        cy = min(max(int(y / cell_size), 0), C - 1)
        cz = min(max(int(z / cell_size), 0), C - 1)
        k = counts[cx, cy, cz]
        if k >= max_bodies:
            raise ValueError(f"cell {(cx, cy, cz)} exceeds max_bodies={max_bodies}")
        cells[cx, cy, cz, k] = (x, y, z, q)
        counts[cx, cy, cz] = k + 1
        where[i] = (cx, cy, cz, k)
    return cells, counts, where


def _cell_centres():
    """f32 cell centres (C,) per axis: (i + 0.5) * cell_size (pe.csl
    init_position / cell_center_z)."""
    return (np.arange(C, dtype=f32) + f32(0.5)) * CELL_SIZE


# --------------------------------------------------------------------------
# P2M  (pe.csl compute_p2m_cell)
# --------------------------------------------------------------------------
def p2m(bodies_or_binned):
    """Leaf multipoles, (C,C,C,30) f32 indexed [x,y,z]. Accepts raw bodies
    (N,4) or the (cells, counts, where) tuple from bin_bodies."""
    if isinstance(bodies_or_binned, tuple):
        cells, counts, _ = bodies_or_binned
    else:
        cells, counts, _ = bin_bodies(bodies_or_binned)
    cc = _cell_centres()
    cx = cc[:, None, None]
    cy = cc[None, :, None]
    cz = cc[None, None, :]
    m = np.zeros((C, C, C, C_BUF_SIZE), dtype=f32)
    for b in range(cells.shape[3]):
        valid = counts > b
        if not valid.any():
            break
        bx = cells[..., b, 0]
        by = cells[..., b, 1]
        bz = cells[..., b, 2]
        q = cells[..., b, 3]
        contrib = compute_solid_harmonics(bx - cx, by - cy, bz - cz)
        # @fmacs(m, m, contrib, q): m = m + contrib*q (modelled unfused)
        upd = m + contrib * q[..., None]
        m = np.where(valid[..., None], upd, m)
    return m


# --------------------------------------------------------------------------
# M2M  (pe.csl m2m_z_local + m2m_plane_send_up / m2m_plane_accumulate)
# --------------------------------------------------------------------------
def _child_vec_harmonics(child_level):
    """Harmonics of -get_translation_vector(cid, child_level) for cid 0..7
    (pe.csl m2m_z_local / l2l_z_local). Returns (8, 30) f32."""
    cid = np.arange(8)
    vx, vy, vz = get_translation_vector(cid, child_level)
    return compute_solid_harmonics(-vx, -vy, -vz)


# Plane-combine arrival order assumed at the representative (the true order
# depends on the router; it only affects f32 rounding).
_PLANE_ORDER = [(0, 0), (1, 0), (0, 1), (1, 1)]


def m2m_level(m_child, child_level, tables=None):
    """One upward step child_level -> child_level-1. m_child: (n,n,n,30)
    indexed [x,y,z] at child_level. Returns parent (n/2,n/2,n/2,30).

    Device sequence per PE column (bits bx,by = position in the 2x2 plane
    block): partial[pz] = 0; partial += T(cid0) M[2pz]; partial += T(cid1)
    M[2pz+1]; then the parent representative sums the four partials (plain
    add, all are about the parent centre)."""
    tb = (tables or load_tables())["m2m"]
    vh = _child_vec_harmonics(child_level)
    n = m_child.shape[0]
    h = n // 2
    partial = {}
    for (bx, by) in _PLANE_ORDER:
        sub = m_child[bx::2, by::2, :, :]          # (h, h, n, 30) columns
        par = np.zeros((h, h, h, C_BUF_SIZE), dtype=f32)
        for bz in (0, 1):
            cid = bx | (by << 1) | (bz << 2)
            src = sub[:, :, bz::2, :].reshape(-1, C_BUF_SIZE)
            tgt = par.reshape(-1, C_BUF_SIZE)
            translate(vh[cid], src, tgt, tb)
            par = tgt.reshape(h, h, h, C_BUF_SIZE)
        partial[(bx, by)] = par
    out = partial[_PLANE_ORDER[0]].copy()
    for key in _PLANE_ORDER[1:]:
        out = out + partial[key]
    return out


def upward(m4, tables=None, device_race=False):
    """Returns {4: m4, 3: m3, 2: m2} (levels 1, 0 not needed).

    device_race=True models what pe.csl run_m2m_upward (lines 1563-1585)
    actually computes: the loop runs m2m_z_local(3) on a level-3
    representative BEFORE the plane partials from the other 3 PEs of its 2x2
    block can be accumulated (they are processed by rx tasks, which only run
    after the kernel function returns). m3 ends up complete, but m2 is built
    only from the representatives' own-column partials of m3."""
    m3 = m2m_level(m4, 4, tables)
    if device_race:
        # own-column partial of each level-3 rep = z-local M2M of its column
        tb = (tables or load_tables())["m2m"]
        vh = _child_vec_harmonics(4)
        h = C // 2
        own = np.zeros((h, h, h, C_BUF_SIZE), dtype=f32)
        for bz in (0, 1):
            src = m4[0::2, 0::2, bz::2, :].reshape(-1, C_BUF_SIZE)
            tgt = own.reshape(-1, C_BUF_SIZE)
            translate(vh[bz << 2], src, tgt, tb)
            own = tgt.reshape(h, h, h, C_BUF_SIZE)
        m2 = m2m_level(own, 3, tables)
    else:
        m2 = m2m_level(m3, 3, tables)
    return {4: m4, 3: m3, 2: m2}


# --------------------------------------------------------------------------
# M2L  (pe.csl apply_m2l_message)
# --------------------------------------------------------------------------
def interaction_offsets(level=None):
    """All candidate M2L offsets d = target - source (box units), in
    lexicographic order: 2 <= max(|dx|,|dy|,|dz|) <= 3. Under the correct
    list (parents adjacent) |d| can never exceed 3, so this is a superset of
    both lists; _m2l_mask() selects, per offset, the target boxes for which
    the pair is in the list:
      mode='correct' : parents adjacent (Chebyshev <= 1) and boxes not adjacent
      mode='csl'     : 2 <= max(|dx|,|dy|,|dz|) <= 3 only (pe.csl:722 guard)
    """
    out = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                if max(abs(dx), abs(dy), abs(dz)) < 2:
                    continue
                out.append((dx, dy, dz))
    return out


def _m2l_mask(level, d, mode):
    """Boolean (n,n,n) mask over TARGET boxes t for which source s = t - d
    exists and (t, s) is in the interaction list."""
    n = 1 << level
    idx = np.arange(n)
    mask = np.ones((n, n, n), dtype=bool)
    for ax, dd in enumerate(d):
        t = idx
        s = idx - dd
        ok = (s >= 0) & (s < n)
        if mode == "correct":
            ok &= np.abs((t >> 1) - (s >> 1)) <= 1
        shape = [1, 1, 1]
        shape[ax] = n
        mask &= ok.reshape(shape)
    if mode == "csl":
        mx = max(abs(x) for x in d)
        if mx < 2 or mx > 3:
            mask[:] = False
    elif mode == "correct":
        if max(abs(x) for x in d) < 2:
            mask[:] = False
    else:
        raise ValueError(mode)
    return mask


def m2l_level(m, level, l_out=None, mode="correct", sign=+1, ops="csl",
              tables=None):
    """Apply M2L at `level` into l_out (zeros if None) and return it.

    m: (n,n,n,30) multipoles at `level`. For every interacting (target t,
    source s) the device translation vector is d * box_size_at_level[level]
    with d = sign * (t - s) (sign=+1 is the device convention: dz = tz - sz in
    apply_m2l_message). Accumulation order into a target: offsets in the
    lexicographic order of interaction_offsets().

    ops='csl'  : device as written (compute_solid_harmonics_2p + device m2l table)
    ops='fixed': irregular harmonics + corrected table (gen_m2l_table_fixed)"""
    tabs = tables or load_tables()
    if ops == "csl":
        tb, harm = tabs["m2l"], compute_solid_harmonics_2p
    elif ops == "fixed":
        tb, harm = tabs["m2l_fixed"], compute_irregular_harmonics_2p
    else:
        raise ValueError(ops)
    n = 1 << level
    if l_out is None:
        l_out = np.zeros((n, n, n, C_BUF_SIZE), dtype=f32)
    bs = BOX_SIZE_AT_LEVEL[level]
    offs = interaction_offsets(level)
    # Gather every (target, source) pair of every offset into one batch.
    t_list, s_list, v_list, seg = [], [], [], []
    for d in offs:
        mask = _m2l_mask(level, d, mode)
        if not mask.any():
            continue
        t = np.argwhere(mask)
        s = t - np.asarray(d)[None, :]
        dd = [sign * x for x in d]
        fx, fy, fz = get_physical_distances(dd[0], dd[1], dd[2], bs)
        v = harm(np.array([fx], f32), np.array([fy], f32),
                 np.array([fz], f32))[0]
        t_list.append(t)
        s_list.append(s)
        v_list.append(np.broadcast_to(v, (len(t), C_BUF_M2L_SIZE)))
        seg.append(len(t))
    if not t_list:
        return l_out
    T = np.concatenate(t_list)
    S = np.concatenate(s_list)
    V = np.concatenate(v_list)
    src = m[S[:, 0], S[:, 1], S[:, 2]]
    acc = np.zeros((len(T), C_BUF_SIZE), dtype=f32)
    # translate() adds its per-coefficient accumulator into the target; doing
    # it on a zero buffer then adding into l_out reproduces "tgt += accum".
    translate(V, src, acc, tb)
    start = 0
    for ln in seg:
        tt = T[start:start + ln]
        l_out[tt[:, 0], tt[:, 1], tt[:, 2]] += acc[start:start + ln]
        start += ln
    return l_out


def m2l_level4_only(bodies, mode="correct", sign=+1, ops="csl", tables=None):
    """P2M then ONLY the level-4 M2L (correct list) into a zeroed l_lvl4.
    Returns the device layout (C, C, 16*30) f32 indexed (py, px, z*30+k).
    ops='csl' (default) reproduces the device kernels/tables as they are
    today, which is what the device will output; ops='fixed' uses the
    corrected M2L operator."""
    m4 = p2m(bodies)
    l4 = m2l_level(m4, 4, None, mode=mode, sign=sign, ops=ops, tables=tables)
    return to_device_layout(l4)


# --------------------------------------------------------------------------
# L2L  (pe.csl l2l_z_local + l2l_plane_receive)
# --------------------------------------------------------------------------
def l2l_level(l_parent, l_child, parent_level, ops="csl", tables=None):
    """Downward step parent_level -> parent_level+1, ACCUMULATING into
    l_child (which already holds the child-level M2L result) and returning it.

    Plane copy: every PE of a 2x2 block holds the parent representative's
    local expansion; each then applies the Z-local L2L with its own plane
    bits and the z bit, vector = -get_translation_vector(cid, child_level).
    ops='csl' uses the device l2l table (== the M2M table), ops='fixed' the
    corrected gen_l2l_table_fixed()."""
    tb = (tables or load_tables())["l2l" if ops == "csl" else "l2l_fixed"]
    child_level = parent_level + 1
    vh = _child_vec_harmonics(child_level)
    n = l_child.shape[0]
    for bx in (0, 1):
        for by in (0, 1):
            for bz in (0, 1):
                cid = bx | (by << 1) | (bz << 2)
                src = l_parent.reshape(-1, C_BUF_SIZE)
                sub = l_child[bx::2, by::2, bz::2, :].copy()
                tgt = sub.reshape(-1, C_BUF_SIZE)
                translate(vh[cid], src, tgt, tb)
                l_child[bx::2, by::2, bz::2, :] = tgt.reshape(sub.shape)
    return l_child


# --------------------------------------------------------------------------
# L2P  (pe.csl run_l2p)  and  P2P  (pe.csl p2p_self_and_zlocal /
# apply_p2p_message)
# --------------------------------------------------------------------------
def l2p(binned, l4, force_sign=+1):
    """Far-field (pot, fx, fy, fz) per slot: returns (C,C,C,maxb,4) f32
    [fx,fy,fz,pot] indexed [cx,cy,cz,slot]."""
    cells, counts, _ = binned
    cc = _cell_centres()
    out = np.zeros(cells.shape[:4] + (4,), dtype=f32)
    nb = cells.shape[3]
    for b in range(nb):
        valid = counts > b
        if not valid.any():
            break
        ix = np.argwhere(valid)
        bod = cells[ix[:, 0], ix[:, 1], ix[:, 2], b]
        ddx = bod[:, 0] - cc[ix[:, 0]]
        ddy = bod[:, 1] - cc[ix[:, 1]]
        ddz = bod[:, 2] - cc[ix[:, 2]]
        L = l4[ix[:, 0], ix[:, 1], ix[:, 2]]
        pot, fx, fy, fz = evaluate_l2p(ddx, ddy, ddz, L)
        if force_sign < 0:          # diagnostic: use +grad(phi) instead
            fx, fy, fz = -fx, -fy, -fz
        out[ix[:, 0], ix[:, 1], ix[:, 2], b] = np.stack([fx, fy, fz, pot], 1)
    return out


def _pair_partial(tgt, src, src_valid):
    """Direct sum over the source bodies of one cell (sequential over source
    slots, as the CSL inner loop) for all target slots. tgt: (K, nt, 3),
    src: (K, ns, 4), src_valid: (K, ns). Returns (K, nt, 4) f32
    [fx,fy,fz,pot] with d = source - target, f += q*d/r^3, pot += q/r,
    skipping r2 <= 1e-10."""
    K, nt = tgt.shape[:2]
    acc = np.zeros((K, nt, 4), dtype=f32)
    one = f32(1.0)
    for ri in range(src.shape[1]):
        rx = src[:, ri, 0][:, None]
        ry = src[:, ri, 1][:, None]
        rz = src[:, ri, 2][:, None]
        rq = src[:, ri, 3][:, None]
        ddx = rx - tgt[..., 0]
        ddy = ry - tgt[..., 1]
        ddz = rz - tgt[..., 2]
        r2 = (ddx * ddx + ddy * ddy) + ddz * ddz
        ok = (r2 > f32(1e-10)) & src_valid[:, ri][:, None]
        r2s = np.where(ok, r2, one)
        rinv = one / np.sqrt(r2s)
        rinv3 = (rinv * rinv) * rinv
        ff = rq * rinv3
        acc[..., 0] = np.where(ok, acc[..., 0] + ff * ddx, acc[..., 0])
        acc[..., 1] = np.where(ok, acc[..., 1] + ff * ddy, acc[..., 1])
        acc[..., 2] = np.where(ok, acc[..., 2] + ff * ddz, acc[..., 2])
        acc[..., 3] = np.where(ok, acc[..., 3] + rq * rinv, acc[..., 3])
    return acc


def _p2p_dir_offsets():
    """pe.csl p2p_dir_offset order: (-1,-1),(-1,0),(-1,1),(0,-1),(0,1),
    (1,-1),(1,0),(1,1)."""
    out = []
    for d8 in range(8):
        k = d8 + 1 if d8 >= 4 else d8
        out.append((k // 3 - 1, k % 3 - 1))
    return out


def p2p(binned, forces):
    """Near field, accumulated into forces (C,C,C,maxb,4) in place.
    First the self + Z-local part (one partial per target over source cells
    tz-1, tz, tz+1 of the same column), then one message per plane neighbour
    (dir8 order) with one partial per (remote source cell, target cell)."""
    cells, counts, _ = binned
    nb = cells.shape[3]
    slot = np.arange(nb)
    valid = counts[..., None] > slot          # (C,C,C,nb)
    tpos = cells[..., :3]

    def shifted(arr, ox, oy, oz, fill):
        out = np.full_like(arr, fill)
        xs_t = slice(max(0, ox), C + min(0, ox))
        xs_s = slice(max(0, -ox), C + min(0, -ox))
        ys_t = slice(max(0, oy), C + min(0, oy))
        ys_s = slice(max(0, -oy), C + min(0, -oy))
        zs_t = slice(max(0, oz), C + min(0, oz))
        zs_s = slice(max(0, -oz), C + min(0, -oz))
        # out[t] = arr[t - o]   (source cell = target cell - o ... see below)
        out[xs_t, ys_t, zs_t] = arr[xs_s, ys_s, zs_s]
        return out

    def partial_from(dx, dy, dz):
        # source cell = target cell + (dx,dy,dz)
        src = shifted(cells, -dx, -dy, -dz, 0)
        sv = shifted(valid, -dx, -dy, -dz, False)
        acc = _pair_partial(tpos.reshape(-1, nb, 3), src.reshape(-1, nb, 4),
                            sv.reshape(-1, nb))
        return acc.reshape(C, C, C, nb, 4)

    # self + Z-local: one running partial per target over sz = tz-1..tz+1
    part = None
    for dz in (-1, 0, 1):
        src = shifted(cells, 0, 0, -dz, 0)
        sv = shifted(valid, 0, 0, -dz, False)
        K = C * C * C
        tg = tpos.reshape(K, nb, 3)
        if part is None:
            part = np.zeros((K, nb, 4), dtype=f32)
        # continue the running accumulation over the next source cell
        acc = part
        one = f32(1.0)
        s4 = src.reshape(K, nb, 4)
        svv = sv.reshape(K, nb)
        for ri in range(nb):
            rx = s4[:, ri, 0][:, None]; ry = s4[:, ri, 1][:, None]
            rz = s4[:, ri, 2][:, None]; rq = s4[:, ri, 3][:, None]
            ddx = rx - tg[..., 0]; ddy = ry - tg[..., 1]; ddz = rz - tg[..., 2]
            r2 = (ddx * ddx + ddy * ddy) + ddz * ddz
            ok = (r2 > f32(1e-10)) & svv[:, ri][:, None]
            rinv = one / np.sqrt(np.where(ok, r2, one))
            rinv3 = (rinv * rinv) * rinv
            ff = rq * rinv3
            acc[..., 0] = np.where(ok, acc[..., 0] + ff * ddx, acc[..., 0])
            acc[..., 1] = np.where(ok, acc[..., 1] + ff * ddy, acc[..., 1])
            acc[..., 2] = np.where(ok, acc[..., 2] + ff * ddz, acc[..., 2])
            acc[..., 3] = np.where(ok, acc[..., 3] + rq * rinv, acc[..., 3])
        part = acc
    forces += np.where(valid[..., None], part.reshape(C, C, C, nb, 4), f32(0.0))

    # plane neighbours, dir8 order; per remote source cell sz, targets |dz|<=1
    for (dxp, dyp) in _p2p_dir_offsets():
        for dz in (-1, 0, 1):
            acc = partial_from(dxp, dyp, dz)
            forces += np.where(valid[..., None], acc, f32(0.0))
    return forces


# --------------------------------------------------------------------------
# Full pipeline
# --------------------------------------------------------------------------
def downward(mlev, mode="correct", sign=+1, m2l_ops="csl", l2l_ops="csl",
             tables=None):
    """M2L at levels 2,3,4 then L2L 2->3->4. Returns {2: l2, 3: l3, 4: l4}."""
    kw = dict(mode=mode, sign=sign, ops=m2l_ops, tables=tables)
    l2 = m2l_level(mlev[2], 2, None, **kw)
    l3 = m2l_level(mlev[3], 3, None, **kw)
    l4 = m2l_level(mlev[4], 4, None, **kw)
    l2l_level(l2, l3, 2, l2l_ops, tables)
    l2l_level(l3, l4, 3, l2l_ops, tables)
    return {2: l2, 3: l3, 4: l4}


def fmm(bodies, mode="correct", sign=+1, m2l_ops="csl", l2l_ops="csl",
        l2p_force_sign=+1, m2m_race=False, far=True, near=True, tables=None,
        return_parts=False):
    """Full 2.5D FMM. Returns F (N,3) and pot (N,) f32 in input-body order.
    forces = L2P (far) + P2P (near), in the device order (L2P first).

    Defaults reproduce the device as written (m2l_ops='csl', l2l_ops='csl',
    l2p_force_sign=+1 i.e. evaluate_l2p's -grad(phi)). The corrected
    pipeline is m2l_ops='fixed', l2l_ops='fixed', l2p_force_sign=-1."""
    binned = bin_bodies(bodies)
    cells, counts, where = binned
    forces = np.zeros(cells.shape[:4] + (4,), dtype=f32)
    levs = None
    if far:
        m4 = p2m(binned)
        mlev = upward(m4, tables, device_race=m2m_race)
        levs = downward(mlev, mode=mode, sign=sign, m2l_ops=m2l_ops,
                        l2l_ops=l2l_ops, tables=tables)
        forces += l2p(binned, levs[4], l2p_force_sign)
    if near:
        p2p(binned, forces)
    per = forces[where[:, 0], where[:, 1], where[:, 2], where[:, 3]]
    F, pot = per[:, :3], per[:, 3]
    if return_parts:
        return F, pot, levs
    return F, pot


def to_device_layout(lev4):
    """(C,C,C,30) [x,y,z,k] -> (C, C, C*30) [py, px, z*30+k]."""
    return np.ascontiguousarray(
        np.transpose(lev4, (1, 0, 2, 3)).reshape(lev4.shape[1], lev4.shape[0], -1))
