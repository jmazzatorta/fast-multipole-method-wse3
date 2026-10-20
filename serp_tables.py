#!/usr/bin/env python3
"""Serpentine pass tables for the FMM software-routed transport (host side).

Single source of truth for paths, directions and message counts. Used by the
event simulator (sim_serp.py) and by appliance_run.py, which pushes the table to
every PE. The device never computes a route: it reads its row of the table.

PASSES (one host launch of run_pass each, in this order):
  pass  name    path  senders              msgs  receivers need ...
  0     M2M43   L4    non level-3 reps     2     level-3 rep <- its 3 block mates
  1     M2M32   L3    level-3, non lvl-2   1     level-2 rep <- its 3 level-3 reps
  2     M2L4    L4    all PEs (lvl-4 reps) 4     M2L interaction partners, level 4
  3     M2L3    L3    level-3 reps         2     M2L interaction partners, level 3
  4     M2L2    L2    level-2 reps         1     M2L interaction partners, level 2
  5     L2L23   L3    level-2 reps         1     level-3 rep <- its level-2 rep
  6     L2L34   L4    level-3 reps         2     PE <- its level-3 rep
  7     P2P     L4    all PEs              *     PE <- its 8 plane neighbours
"msgs" = ceil(nz / K) messages of K=4 z-boxes, nz = boxes along z of the data
level (m_lvl3 / l_lvl3: 8 -> 2, m_lvl2 / l_lvl2: 4 -> 1, m_lvl4: 16 -> 4).
(*) P2P sends the leaf cell BODY blocks of the column (1 + 4*max_bodies_pe
words each), p2p_cells(max_bodies_pe) cells per message (as many as fit in the
120-word payload, at most 7 = the 3-bit zcount): ceil(16 / that) messages,
e.g. max_bodies_pe 2 -> 7 cells, 3 msgs; 8 -> 3 cells, 6 msgs.

PATH of level L (stride s = 2**(TREE_H - L) PEs = box side at level L):
  rows y with y % s == 0 are "path rows"; path row r = y // s runs East if r is
  even, West if r is odd (row-by-row serpentine, as in the N-body reference);
  between path rows r and r+1 the path goes South along the end column
  (x = W-1 after an even row, x = 0 after an odd row) through the s-1 skipped
  rows ("turn PEs"). Every other PE is idle in that pass.

STREAMS: FWD runs along increasing path position, BWD along decreasing. A
sender originates its messages on both streams. Every PE on the path forwards a
message while hop < HMAX of the pass and consumes it (the receiver filters).
HMAX = the largest path distance over the (receiver, sender) pairs that must be
delivered, so every needed message arrives.

DIRECTION CODES (as in pe.csl):
  tx code: direction the PE sends to         0=E 1=W 2=N 3=S, 4 = none
  rx code: direction of TRAVEL of the traffic 0=eastbound(from W) 1=westbound
           (from E) 2=northbound(from S) 3=southbound(from N), 4 = none
  (an rx code equals the tx code of the neighbour that sends to us)

TABLE ROW per (PE, pass), u16 fields (index -> name):
  0 fwd_in   1 fwd_out   2 bwd_in   3 bwd_out
  4 n_rx_fwd 5 n_tx_fwd  6 n_rx_bwd 7 n_tx_bwd
  8 n_own    9 hmax
"""
import numpy as np

W = H = 16
TREE_H = 4
K_CUBES = 4
R = 3                     # M2L interaction radius in boxes (2:1 octree)
E, Wd, N, S, NONE = 0, 1, 2, 3, 4
FIELDS = ["fwd_in", "fwd_out", "bwd_in", "bwd_out",
          "n_rx_fwd", "n_tx_fwd", "n_rx_bwd", "n_tx_bwd", "n_own", "hmax"]
NF = len(FIELDS)
STEP = {E: (1, 0), Wd: (-1, 0), N: (0, -1), S: (0, 1)}


def stride(level):
    return 1 << (TREE_H - level)


def is_rep(x, y, level):
    s = stride(level)
    return x % s == 0 and y % s == 0


def rep_of(x, y, level):
    s = stride(level)
    return ((x // s) * s, (y // s) * s)


def n_msgs(data_level):
    nz = 1 << data_level
    return -(-nz // K_CUBES)


MSG_PAYLOAD_MAX = K_CUBES * 30          # 120 words (constants.csl)


def p2p_cells(max_bodies_pe):
    """Leaf cells per P2P message (pe.csl P2P_CELLS)."""
    k = min(7, MSG_PAYLOAD_MAX // (1 + 4 * max_bodies_pe))
    assert k >= 1, "max_bodies_pe too large for one cell per message"
    return k


def pass_nmsgs(pi, max_bodies_pe):
    """Own messages of a sender in pass pi."""
    if PASSES[pi][0] == "P2P":
        return -(-H // p2p_cells(max_bodies_pe))
    return n_msgs(PASSES[pi][2])


def path(level):
    """Ordered list of (x, y) on the level's serpentine."""
    s = stride(level)
    rows = list(range(0, H, s))
    p = []
    for r, y in enumerate(rows):
        xs = range(W) if r % 2 == 0 else range(W - 1, -1, -1)
        p += [(x, y) for x in xs]
        if r + 1 < len(rows):
            xe = W - 1 if r % 2 == 0 else 0
            p += [(xe, yy) for yy in range(y + 1, rows[r + 1])]
    return p


def dir_between(a, b):
    """tx code for a single hop a -> b (must be neighbours)."""
    d = (b[0] - a[0], b[1] - a[1])
    for k, v in STEP.items():
        if v == d:
            return k
    raise ValueError(f"not neighbours: {a} {b}")


ALL = [(x, y) for y in range(H) for x in range(W)]


def m2l_partners(level):
    """receiver rep -> sender reps whose box parents are adjacent on the plane
    (the z test and dx=dy=0 are local to the receiver)."""
    s = stride(level)
    reps = [q for q in ALL if is_rep(*q, level)]
    out = {}
    for t in reps:
        tb = (t[0] // s, t[1] // s)
        out[t] = [q for q in reps if q != t and
                  abs(tb[0] // 2 - q[0] // s // 2) <= 1 and
                  abs(tb[1] // 2 - q[1] // s // 2) <= 1]
    return out


def _upward(child_level):
    """receiver = parent rep, senders = the other child reps of its box."""
    pl = child_level - 1
    out = {}
    for t in ALL:
        if is_rep(*t, pl):
            out[t] = [q for q in ALL if is_rep(*q, child_level) and q != t
                      and rep_of(*q, pl) == t]
    return out


def _downward(child_level):
    """receiver = child rep that is not the parent rep, sender = its parent rep."""
    pl = child_level - 1
    out = {}
    for t in ALL:
        if is_rep(*t, child_level) and not is_rep(*t, pl):
            out[t] = [rep_of(*t, pl)]
    return out


def _plane_neighbours():
    """receiver -> its (up to 8) plane-neighbour columns (P2P near field)."""
    return {t: [q for q in ALL if q != t and abs(q[0] - t[0]) <= 1
                and abs(q[1] - t[1]) <= 1] for t in ALL}


# name, path level, data level (own message buffer), sender predicate, needs()
PASSES = [
    ("M2M43", 4, 3, lambda x, y: is_rep(x, y, 4) and not is_rep(x, y, 3), lambda: _upward(4)),
    ("M2M32", 3, 2, lambda x, y: is_rep(x, y, 3) and not is_rep(x, y, 2), lambda: _upward(3)),
    ("M2L4",  4, 4, lambda x, y: is_rep(x, y, 4), lambda: m2l_partners(4)),
    ("M2L3",  3, 3, lambda x, y: is_rep(x, y, 3), lambda: m2l_partners(3)),
    ("M2L2",  2, 2, lambda x, y: is_rep(x, y, 2), lambda: m2l_partners(2)),
    ("L2L23", 3, 2, lambda x, y: is_rep(x, y, 2), lambda: _downward(3)),
    ("L2L34", 4, 3, lambda x, y: is_rep(x, y, 3), lambda: _downward(4)),
    ("P2P",   4, 4, lambda x, y: True, _plane_neighbours),
]
NPASS = len(PASSES)


def hmax(pi):
    name, pl, dl, snd, needs = PASSES[pi]
    idx = {q: i for i, q in enumerate(path(pl))}
    return max(abs(idx[t] - idx[q]) for t, qs in needs().items() for q in qs)


def pass_table(pi, max_bodies_pe):
    """dict (x, y) -> list of NF ints (idle PEs get NONE dirs and zero counts)."""
    name, pl, dl, snd, needs = PASSES[pi]
    p = path(pl)
    hm = hmax(pi)
    own = pass_nmsgs(pi, max_bodies_pe)
    L = len(p)
    tab = {q: [NONE, NONE, NONE, NONE, 0, 0, 0, 0, 0, hm] for q in ALL}
    for i, q in enumerate(p):
        row = tab[q]
        prev = p[i - 1] if i > 0 else None
        nxt = p[i + 1] if i + 1 < L else None
        if prev is not None:
            row[0] = dir_between(prev, q)       # FWD arrives travelling prev->q
            row[3] = dir_between(q, prev)       # BWD leaves towards prev
        if nxt is not None:
            row[1] = dir_between(q, nxt)        # FWD leaves towards nxt
            row[2] = dir_between(nxt, q)        # BWD arrives travelling nxt->q
        mine = own if snd(*q) else 0
        row[8] = mine
        rx = fw = 0
        for j in range(max(0, i - hm), i):      # FWD: senders upstream
            if snd(*p[j]):
                rx += own
                if nxt is not None and (i - j) < hm:
                    fw += own
        row[4] = rx
        row[5] = fw + (mine if nxt is not None else 0)
        rx = fw = 0
        for j in range(i + 1, min(L, i + hm + 1)):   # BWD: senders downstream
            if snd(*p[j]):
                rx += own
                if prev is not None and (j - i) < hm:
                    fw += own
        row[6] = rx
        row[7] = fw + (mine if prev is not None else 0)
    return tab


def device_table(max_bodies_pe):
    """u16 array shaped (H, W, NPASS*NF): row (py, px), pass-major."""
    out = np.zeros((H, W, NPASS * NF), dtype=np.uint16)
    for pi in range(NPASS):
        for (x, y), row in pass_table(pi, max_bodies_pe).items():
            out[y, x, pi * NF:(pi + 1) * NF] = row
    return out


def src_from_hop(pi, x, y, stream, hop):
    """Source PE of a message that arrived at (x,y) on `stream` after `hop` hops."""
    p = path(PASSES[pi][1])
    i = p.index((x, y))
    return p[i - hop] if stream == "fwd" else p[i + hop]


if __name__ == "__main__":
    import sys
    mb = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print(f"max_bodies_pe {mb}: P2P {p2p_cells(mb)} cells/msg")
    for pi, (name, pl, dl, snd, needs) in enumerate(PASSES):
        t = pass_table(pi, mb)
        busy = sum(1 for r in t.values() if r[4] + r[6] + r[8] > 0)
        nsend = sum(1 for q in ALL if snd(*q))
        print(f"pass {pi} {name:6s}: path L{pl} ({len(path(pl))} PEs), senders {nsend:3d} "
              f"x {pass_nmsgs(pi, mb)} msgs, HMAX {hmax(pi):3d}, busy PEs {busy:3d}, "
              f"rx/stream max {max(max(r[4], r[6]) for r in t.values()):3d}, "
              f"total arrivals {sum(r[4] + r[6] for r in t.values()):6d}")
    T = device_table(mb)
    print("device table", T.shape, T.dtype, "max", int(T.max()))
