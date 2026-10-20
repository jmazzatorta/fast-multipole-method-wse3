#!/usr/bin/env python3
"""FMM operator tables (M2M / M2L / L2L), generated on the fly.

Replaces fmm25d_data.pkl: the device and the NumPy reference (fmm_ref.py) both
get their tables from device_tables(). The generators are the ones of the
former gen_agendas25d.py (math part only; the routing agendas are obsolete).

Format of each table (the contract with kernels.csl `translate`):
  offsets     u16[N_COEFFS+1]  CSR over target coefficients (m outer, l inner)
  *_indices   u16[n]           float index of the source coefficient (re part)
  v_indices   u16[n]           float index of the translation-vector harmonic
  signs       u32[n]           4 packed sign bits (rr, ri, ir, ii; 1 = negative)

M2L (irregular harmonics) and L2L (its own table, 105 entries) are the
corrected operators, validated against direct N-body in fmm_ref.py.
"""
import numpy as np

P = 4
N_COEFFS = (P + 1) * (P + 2) // 2          # 15
N_COEFFS_V = (2 * P + 1) * (2 * P + 2) // 2  # 45


def get_mem_idx(l, m):
    pos = 0
    for mm in range(m):
        pos += (P + 1 - mm)
    pos += (l - m)
    return pos * 2


def get_v_mem_idx(l, m):
    pos = 0
    for mm in range(m):
        pos += (2 * P + 1 - mm)
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


def gen_m2m_tables():
    father = {(l, m): [] for l in range(P + 1) for m in range(l + 1)}
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
                        pack_signs(csr * vsr, -(csi * vsi), csr * vsi, csi * vsr)))
    return _flatten(father)


def _src_signs(k):
    """(sign_re, sign_im) to read X^k from the stored X^{|k|}:
    X^{-m} = (-1)^m conj(X^m)."""
    return (parity_sign(k), -parity_sign(k)) if k < 0 else (1.0, 1.0)


def gen_m2l_tables():
    """M2L for v_buf = IRREGULAR solid harmonics I(v) of order 2P
    (kernels.csl compute_irregular_harmonics_2p), v = c_target - c_source:
        L_j^p = (-1)^j sum_{n,k} M_n^k conj(I_{n+j}^{p+k}(v))
    in the normalisation of the device recurrences (see fmm_ref.py). 375
    entries. Validated end to end against direct N-body in fmm_ref.py."""
    target = {(l, m): [] for l in range(P + 1) for m in range(l + 1)}
    for j in range(P + 1):
        for p in range(j + 1):
            g = -1.0 if j % 2 else 1.0
            for n in range(P + 1):
                for k in range(-n, n + 1):
                    csr, csi = _src_signs(k)
                    vl, vm = n + j, p + k
                    # conj(I^vm) read from the stored I^{|vm|}
                    if vm >= 0:
                        vsr, vsi = 1.0, -1.0
                    else:
                        vsr, vsi = parity_sign(vm), parity_sign(vm)
                    target[(j, p)].append((
                        get_mem_idx(n, abs(k)), get_v_mem_idx(vl, abs(vm)),
                        pack_signs(g * csr * vsr, -g * csi * vsi,
                                   g * csr * vsi, g * csi * vsr)))
    return _flatten(target)


def gen_m2l_tables_regular():
    """FORMER device M2L table (regular harmonics): mathematically wrong, kept
    only so fmm_ref.py can reproduce older runs (ops='csl')."""
    target = {(l, m): [] for l in range(P + 1) for m in range(l + 1)}
    for n in range(P + 1):
        for k in range(-n, n + 1):
            si = get_mem_idx(n, abs(k))
            if k < 0: csr, csi = parity_sign(k), -parity_sign(k)
            else:     csr, csi = 1.0, 1.0
            for v_l in range(2 * P + 1):
                for v_m in range(-v_l, v_l + 1):
                    l_f, m_f = v_l - n, k + v_m
                    if l_f < 0 or l_f > P: continue
                    if m_f < 0 or m_f > l_f: continue
                    vi = get_v_mem_idx(v_l, abs(v_m))
                    if v_m < 0: vsr, vsi = parity_sign(v_m), -parity_sign(v_m)
                    else:       vsr, vsi = 1.0, 1.0
                    target[(l_f, m_f)].append((si, vi,
                        pack_signs(csr * vsr, -(csi * vsi), csr * vsi, csi * vsr)))
    return _flatten(target)


def gen_l2l_tables():
    """L2L for v_buf = REGULAR solid harmonics R(w) of order P, w = child centre
    - parent centre (the negated get_translation_vector used by l2l_z_local):
        Lc_n^k = sum_{j>=n, p} Lp_j^p R_{j-n}^{p-k}(w)
    105 entries. Validated end to end against direct N-body in fmm_ref.py.
    (The former table was the M2M one, which maps to HIGHER orders.)"""
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
                        get_mem_idx(j, abs(p)), get_mem_idx(vl, abs(vm)),
                        pack_signs(csr * vsr, -(csi * vsi), csr * vsi, csi * vsr)))
    return _flatten(target)


def _pack(t, idx_key):
    return {"offsets": np.array(t[0], np.uint16),
            idx_key: np.array(t[1], np.uint16),
            "v_indices": np.array(t[2], np.uint16),
            "signs": np.array(t[3], np.uint32)}


_CACHE = None


def device_tables():
    """dict op -> table dict, op in m2m (key child_indices), m2l and l2l (key
    source_indices). Same layout the former pickle had under data['math']."""
    global _CACHE
    if _CACHE is None:
        _CACHE = {"m2m": _pack(gen_m2m_tables(), "child_indices"),
                  "m2l": _pack(gen_m2l_tables(), "source_indices"),
                  "l2l": _pack(gen_l2l_tables(), "source_indices")}
        n = {k: len(v["v_indices"]) for k, v in _CACHE.items()}
        # must match M2M_N_ENTRIES / M2L_N_ENTRIES / L2L_N_ENTRIES (constants.csl)
        assert n == {"m2m": 92, "m2l": 375, "l2l": 105}, n
    return _CACHE


if __name__ == "__main__":
    for k, v in device_tables().items():
        print(k, {kk: (vv.dtype, vv.shape) for kk, vv in v.items()})
