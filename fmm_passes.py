#!/usr/bin/env python3
"""The FMM passes and the plane geometry (host side, transport-independent).

Mirror of the PASSES section of pe.csl: the list of passes and the level of
the data each one sends. Who sends what to whom: zmp_tables.py.

Parametric in the mesh side C = 2**H and the expansion order P: call
configure(C, P) first (default: C = 16, P = 4). The module globals below (W,
TREE_H, P, PASSES, NPASS, MSG_PAYLOAD_MAX) are updated in place, so users must
read them as fmm_passes.<name> at call time, never copy them at import.

Plane: a representative ("rep") of level L is the column (x, y) with x and y
multiples of s = 2**(H - L); a level-L box spans s x s columns on the plane.

PASSES, in this order (one host launch of run_pass each), NPASS = 3H - 4:
  M2M c->c-1, c = H..3   data level c-1 (partials of the parents)
  M2L L,     L = H..2    data level L
  L2L p->p+1, p = 2..H-1 data level p (parents)
  P2P                    leaf cell bodies (level H)
C = 16: M2M43 M2M32 M2L4 M2L3 M2L2 L2L23 L2L34 P2P (8 passes).
"""

K_CUBES = 4               # boxes per message (constants.csl)

# Set by configure(): see the module docstring.
W = 0                     # mesh side C
TREE_H = 0                # log2(C)
P = 0                     # expansion order
MSG_PAYLOAD_MAX = 0       # K_CUBES * C_BUF_SIZE words (constants.csl)
PASSES = []               # (name, data level)
NPASS = 0


def stride(level):
    return 1 << (TREE_H - level)


def is_rep(x, y, level):
    s = stride(level)
    return x % s == 0 and y % s == 0


def rep_of(x, y, level):
    s = stride(level)
    return ((x // s) * s, (y // s) * s)


def p2p_cells(max_bodies_pe):
    """Leaf cells per P2P message (pe.csl P2P_CELLS)."""
    k = min(7, MSG_PAYLOAD_MAX // (1 + 4 * max_bodies_pe))
    assert k >= 1, "max_bodies_pe too large for one cell per message"
    return k


def _build_passes(h):
    out = [(f"M2M{c}{c - 1}", c - 1) for c in range(h, 2, -1)]
    out += [(f"M2L{lv}", lv) for lv in range(h, 1, -1)]
    out += [(f"L2L{p}{p + 1}", p) for p in range(2, h)]
    out.append(("P2P", h))
    return out


def configure(c, p=4):
    """Set the mesh side C (power of two, 8..128) and the expansion order P.
    C <= 128: mp_ent packs zbase | zcount << 8 with zcount up to C."""
    global W, TREE_H, P, MSG_PAYLOAD_MAX, PASSES, NPASS
    h = c.bit_length() - 1
    assert c == 1 << h and 3 <= h <= 7, f"C = {c}: power of two in 8..128"
    assert p >= 1
    W = c
    TREE_H = h
    P = p
    MSG_PAYLOAD_MAX = K_CUBES * (p + 1) * (p + 2)
    PASSES = _build_passes(h)
    NPASS = len(PASSES)
    assert NPASS == 3 * h - 4


configure(16, 4)
