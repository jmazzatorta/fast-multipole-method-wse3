#!/usr/bin/env python3
"""Message-passing tables for the z-split mapping (host side).

Each plane column (x, y) of C leaf cells along z is cut into NK = C / ZP chunks
of ZP cells; chunk k of column (x, y) is one PE, placed on the fabric by
pe_map.PeMap (block of NK PEs per column, snake order). ZP = C (NK = 1) is the
one-PE-per-column mapping of phase GEN and gives exactly its tables (the former
mp_tables.py, validated on hardware): regress() compares with their hashes.

Box ownership. A level-l box (bx, by, bz) has height e = 2**(H - l) cells; it
belongs to the plane column of its representative (bx << (H-l), by << (H-l))
and to the chunk of its lowest cell, k = (bz * e) // ZP:
  e <= ZP : a chunk holds ZP / e consecutive boxes of every level-l column;
  e >  ZP : only every (e / ZP)-th chunk holds one box (the z-rep).

Passes: same list and data levels as fmm_passes.PASSES. Who needs what is
defined box by box (all PEs of a column hold the same plane position):
  M2M c->c-1  the holder of a level-c child sends its partial of the parent
              (data level c-1, z = parent bz) to the parent's owner.
  M2L L       the holder of a source box sends it to the holder of every
              target box whose interaction list contains it (parents adjacent
              in 3D, boxes not adjacent).
  L2L p->p+1  the owner of a level-p parent sends it (z = parent bz) to the
              holder of each of its children.
  P2P         the holder of a leaf cell sends it to the holder of each of its
              26 neighbours.
Pairs with sender == receiver are local work, not messages. For each
(sender, receiver) the sent z indices are split into maximal runs of
consecutive boxes = ENTRIES (dest_fx, dest_fy, zbase, zcount), zbase global
along the column (as in the message header); the device splits an entry into
messages of at most K_CUBES boxes (P2P: p2p_cells). Entry lists are sorted by
(dest_fy, dest_fx, zbase) and rotated by (fx + 3 fy) mod n (as in GEN).

The plane and z parts are separable: per column pair the z relation is one of
a few 1D relations, tabulated per (sender chunk, receiver chunk). check() is
an independent box-by-box (brute force) derivation for small meshes.

    python3 zmp_tables.py [MB] [C] [P] [ZP] [--check] [--regress]
      (default 4 32 4 8; --check: brute force, small meshes; --regress: GEN)
"""
import sys
from itertools import product

import numpy as np

import fmm_passes as S
from pe_map import PeMap

HDR_F = 10              # mp_hdr words per pass (pe.csl TRANSPORT)
ENT_W = 3

ZP = 0
MAP = None
PES = []                # every PE, fabric (fx, fy), row-major


def configure(c, p, zp, lbw=None):
    """Mesh side c, order p, zp leaf cells per PE (power of two, zp | c);
    lbw: log2 of the block width (pe_map.block_shape; default square-ish)."""
    global ZP, MAP, PES
    S.configure(c, p)
    assert zp & (zp - 1) == 0 and 1 <= zp <= c
    ZP = zp
    MAP = PeMap(c, c, c, zp, lbw)
    PES = [(fx, fy) for fy in range(MAP.height) for fx in range(MAP.width)]
    _CACHE.clear()


def height(level):
    return 1 << (S.TREE_H - level)


def chunk_of(level, bz):
    """Chunk holding level-`level` box bz (chunk of its lowest cell)."""
    return (bz * height(level)) // ZP


def pe_of(col, k):
    """Fabric coordinates of chunk k of plane column col."""
    fx, fy, _ = MAP.cell_to_pe(col[0], col[1], k * ZP)
    return fx, fy


def owned(level, k):
    """Level boxes held by chunk k (sorted)."""
    nb = 1 << level
    return [bz for bz in range(nb) if chunk_of(level, bz) == k]


# ---------------------------------------------------------------------------
# 1D (z) relations: (sender chunk, receiver chunk) -> sorted z of sent boxes
# ---------------------------------------------------------------------------
def _z_m2l(level, near):
    """Source boxes of chunk ks needed by targets of chunk kt: z parents
    adjacent and, for plane-adjacent (near) column pairs, |dz| >= 2."""
    out = {}
    for ks, kt in product(range(MAP.nk), repeat=2):
        ts = owned(level, kt)
        zs = [sz for sz in owned(level, ks)
              if any(abs(tz // 2 - sz // 2) <= 1 and (not near or abs(tz - sz) >= 2)
                     for tz in ts)]
        if zs:
            out[(ks, kt)] = zs
    return out


def _z_up(child_level):
    """M2M: parent z (data level child_level-1) sent from the chunk holding a
    child to the chunk owning the parent."""
    out = {}
    for cz in range(1 << child_level):
        key = (chunk_of(child_level, cz), chunk_of(child_level - 1, cz // 2))
        out.setdefault(key, set()).add(cz // 2)
    return {k: sorted(v) for k, v in out.items()}


def _z_down(child_level):
    """L2L: parent z sent from the chunk owning the parent to the chunk
    holding a child."""
    out = {}
    for cz in range(1 << child_level):
        key = (chunk_of(child_level - 1, cz // 2), chunk_of(child_level, cz))
        out.setdefault(key, set()).add(cz // 2)
    return {k: sorted(v) for k, v in out.items()}


def _z_p2p():
    out = {}
    lv = S.TREE_H
    for ks, kt in product(range(MAP.nk), repeat=2):
        ts = owned(lv, kt)
        zs = [sz for sz in owned(lv, ks) if any(abs(tz - sz) <= 1 for tz in ts)]
        if zs:
            out[(ks, kt)] = zs
    return out


# ---------------------------------------------------------------------------
# Plane part: list of (sender column, receiver column, z relation)
# ---------------------------------------------------------------------------
def _m2l_partners(level):
    """receiver rep column -> sender rep columns with plane box parents
    adjacent."""
    st, nb = S.stride(level), 1 << level
    out = {}
    for ty, tx in product(range(nb), repeat=2):
        out[(tx * st, ty * st)] = [
            (sx * st, sy * st)
            for sy in range(max(0, ty - 3), min(nb, ty + 4))
            for sx in range(max(0, tx - 3), min(nb, tx + 4))
            if (sx, sy) != (tx, ty) and abs(tx // 2 - sx // 2) <= 1
            and abs(ty // 2 - sy // 2) <= 1]
    return out


def _col_pairs(pi):
    """Yields (sender column, receiver column, z relation) of pass pi."""
    name, lv = S.PASSES[pi][0], S.PASSES[pi][1]
    if name.startswith("M2M"):
        c = lv + 1
        rel = _z_up(c)
        st = S.stride(c)
        for qy, qx in product(range(0, S.W, st), repeat=2):
            yield (qx, qy), S.rep_of(qx, qy, lv), rel
    elif name.startswith("M2L"):
        st = S.stride(lv)
        far, near = _z_m2l(lv, False), _z_m2l(lv, True)
        for t, qs in _m2l_partners(lv).items():
            yield t, t, near                            # own column
            for q in qs:
                d = max(abs(q[0] - t[0]), abs(q[1] - t[1])) // st
                yield q, t, (far if d >= 2 else near)
    elif name.startswith("L2L"):
        rel = _z_down(lv + 1)
        st = S.stride(lv + 1)
        for ty, tx in product(range(0, S.W, st), repeat=2):
            yield S.rep_of(tx, ty, lv), (tx, ty), rel
    else:
        rel = _z_p2p()
        for ty, tx in product(range(S.W), repeat=2):
            for qy in range(max(0, ty - 1), min(S.W, ty + 2)):
                for qx in range(max(0, tx - 1), min(S.W, tx + 2)):
                    yield (qx, qy), (tx, ty), rel


def _iter_sent(pi):
    """Yields (sender PE, receiver PE, sorted z list), local pairs skipped."""
    for q, t, rel in _col_pairs(pi):
        for (ks, kt), zs in rel.items():
            a, b = pe_of(q, ks), pe_of(t, kt)
            if a != b:
                yield a, b, zs


_CACHE = {}


def sent(pi):
    """{(sender PE, receiver PE): sorted z list} of pass pi (no local pairs)."""
    if pi not in _CACHE:
        out = {}
        for a, b, zs in _iter_sent(pi):
            assert (a, b) not in out
            out[(a, b)] = zs
        _CACHE[pi] = out
    return _CACHE[pi]


def runs(zs):
    """Sorted z list -> [(zbase, zcount)] maximal runs."""
    out = []
    for z in zs:
        if out and out[-1][0] + out[-1][1] == z:
            out[-1][1] += 1
        else:
            out.append([z, 1])
    return [tuple(r) for r in out]


def pass_entries(pi):
    """sender PE -> rotated list of entries (dest_fx, dest_fy, zbase, zcount)."""
    per = {}
    for (a, b), zs in sent(pi).items():
        for zb, zn in runs(zs):
            per.setdefault(a, []).append((b[0], b[1], zb, zn))
    out = {}
    for a, ents in per.items():
        ents.sort(key=lambda e: (e[1], e[0], e[2]))
        r = (a[0] + 3 * a[1]) % len(ents)
        out[a] = ents[r:] + ents[:r]
    return out


def per_msg(pi, max_bodies_pe):
    return S.p2p_cells(max_bodies_pe) if S.PASSES[pi][0] == "P2P" else S.K_CUBES


def split(zbase, zcount, pi, max_bodies_pe):
    k = per_msg(pi, max_bodies_pe)
    return [(z, min(k, zbase + zcount - z)) for z in range(zbase, zbase + zcount, k)]


def pass_counts(pi, max_bodies_pe):
    """(n_rx, n_tx): PE -> messages received / sent in pass pi."""
    n_rx = {q: 0 for q in PES}
    n_tx = {q: 0 for q in PES}
    for a, lst in pass_entries(pi).items():
        for (dx, dy, zb, zn) in lst:
            m = len(split(zb, zn, pi, max_bodies_pe))
            n_tx[a] += m
            n_rx[(dx, dy)] += m
    return n_rx, n_tx


def expected_rx(pi, max_bodies_pe):
    """receiver PE -> set of (sender PE, message zbase) it must consume."""
    out = {q: set() for q in PES}
    for a, lst in pass_entries(pi).items():
        for (dx, dy, zb, zn) in lst:
            for (z, _) in split(zb, zn, pi, max_bodies_pe):
                out[(dx, dy)].add((a, z))
    return out


DIRS = "ewns"            # E, W, N, S as in xport.csl (0..3)


def out_of(src, dst):
    """First hop of a message under the all-static transport (pe.csl
    TRANSPORT, xport.csl; X first, then Y): 0 = E, 1 = W, 2 = N, 3 = S."""
    if dst[0] != src[0]:
        return 0 if dst[0] > src[0] else 1
    return 2 if dst[1] < src[1] else 3


def route_hops(src, dst):
    """Hops of src -> dst, X first then Y, as (receiver, direction index)."""
    (x, y), hops = src, []
    while x != dst[0]:
        x += 1 if dst[0] > x else -1
        hops.append(((x, y), 0 if dst[0] > src[0] else 1))
    while y != dst[1]:
        y += 1 if dst[1] > y else -1
        hops.append(((x, y), 2 if dst[1] < src[1] else 3))
    return hops


def lane_counts(pi, max_bodies_pe):
    """Per PE of pass pi (dicts PE -> count) under the all-static transport:
    p_e, p_w, p_n, p_s = messages arriving on the E / W / N / S port (data
    hops + credits: every hop arrives on the receiver's port of its direction,
    and the receiver returns a credit that arrives on the sender's opposite
    port); fwd = straight relays, turn = relays from E/W onto N/S (PEs strictly
    between source and destination)."""
    c = {k: {} for k in ("p_e", "p_w", "p_n", "p_s", "fwd", "turn")}

    def inc(k, q, n):
        c[k][q] = c[k].get(q, 0) + n

    for a, lst in pass_entries(pi).items():
        for (dx, dy, zb, zn) in lst:
            n = len(split(zb, zn, pi, max_bodies_pe))
            hops = route_hops(a, (dx, dy))
            prev = a
            for i, (q, di) in enumerate(hops):
                inc("p_" + DIRS[di], q, n)
                inc("p_" + DIRS[di ^ 1], prev, n)
                if i + 1 < len(hops):
                    inc("fwd" if hops[i + 1][1] == di else "turn", q, n)
                prev = q
    return c


def device_tables(max_bodies_pe):
    """(mp_hdr (Hf, Wf, NPASS*HDR_F) u16, mp_ent (Hf, Wf, 3*cap) u16, cap) on the
    fabric rectangle (Wf x Hf PEs). mp_hdr per pass: n_rx, arrivals expected
    on the E, W, N, S ports (data + credits), entry offset, n_e, n_w, n_n, n_s;
    the entries of a pass are grouped by first hop (E, W, N, S), each group in
    table order."""
    W, Hf = MAP.width, MAP.height
    rows = {q: [] for q in PES}
    hdr = np.zeros((Hf, W, S.NPASS * HDR_F), dtype=np.uint16)
    for pi in range(S.NPASS):
        ents = pass_entries(pi)
        n_rx, _ = pass_counts(pi, max_bodies_pe)
        lc = lane_counts(pi, max_bodies_pe)
        for q in PES:
            lst = ents.get(q, [])
            groups = [[e for e in lst if out_of(q, e[:2]) == o] for o in range(4)]
            words = ((n_rx[q],) + tuple(lc["p_" + d].get(q, 0) for d in DIRS)
                     + (len(rows[q]),) + tuple(len(g) for g in groups))
            assert max(words) < 65536
            hdr[q[1], q[0], pi * HDR_F:(pi + 1) * HDR_F] = words
            for g in groups:
                for (dx, dy, zb, zn) in g:
                    assert 0 <= zb < 256 and 0 < zn < 256
                    rows[q].append((dx, dy, zb | (zn << 8)))
    cap = max(1, max(len(r) for r in rows.values()))
    ent = np.zeros((Hf, W, ENT_W * cap), dtype=np.uint16)
    for (x, y), r in rows.items():
        for i, e in enumerate(r):
            ent[y, x, ENT_W * i:ENT_W * i + ENT_W] = e
    return hdr, ent, cap


def stats(max_bodies_pe):
    """Streaming (no table kept, usable at 128^3). Per pass: (messages, max
    rx/PE, max entries/PE, max tx/PE); and cap = max entries per PE summed
    over the passes."""
    shape = (MAP.height, MAP.width)
    tot = np.zeros(shape, dtype=np.int64)
    out = []
    for pi in range(S.NPASS):
        k = per_msg(pi, max_bodies_pe)
        ent = np.zeros(shape, dtype=np.int64)
        rx = np.zeros(shape, dtype=np.int64)
        tx = np.zeros(shape, dtype=np.int64)
        for a, b, zs in _iter_sent(pi):
            rr = runs(zs)
            m = sum(-(-zn // k) for _, zn in rr)
            ent[a[1], a[0]] += len(rr)
            tx[a[1], a[0]] += m
            rx[b[1], b[0]] += m
        tot += ent
        out.append((int(rx.sum()), int(rx.max()), int(ent.max()), int(tx.max())))
    return out, int(tot.max())


# ---------------------------------------------------------------------------
# Independent derivation (brute force, box by box) and regression
# ---------------------------------------------------------------------------
def _holder(level, bx, by, bz):
    s = height(level)                    # plane stride == box height in cells
    return pe_of((bx * s, by * s), chunk_of(level, bz))


def brute(pi):
    """{(sender PE, receiver PE): sorted z} from the box-level definitions."""
    name, lv = S.PASSES[pi][0], S.PASSES[pi][1]
    nb = 1 << lv
    out = {}

    def add(a, b, z):
        if a != b:
            out.setdefault((a, b), set()).add(z)

    if name.startswith("M2M"):
        c = lv + 1
        for cx, cy, cz in product(range(1 << c), repeat=3):
            add(_holder(c, cx, cy, cz), _holder(lv, cx // 2, cy // 2, cz // 2), cz // 2)
    elif name.startswith("L2L"):
        c = lv + 1
        for cx, cy, cz in product(range(1 << c), repeat=3):
            add(_holder(lv, cx // 2, cy // 2, cz // 2), _holder(c, cx, cy, cz), cz // 2)
    elif name.startswith("M2L"):
        for tx, ty, tz in product(range(nb), repeat=3):
            ht = _holder(lv, tx, ty, tz)
            for sx in range(max(0, tx - 3), min(nb, tx + 4)):
                if abs(tx // 2 - sx // 2) > 1:
                    continue
                for sy in range(max(0, ty - 3), min(nb, ty + 4)):
                    if abs(ty // 2 - sy // 2) > 1:
                        continue
                    for sz in range(max(0, tz - 3), min(nb, tz + 4)):
                        if abs(tz // 2 - sz // 2) > 1:
                            continue
                        if max(abs(tx - sx), abs(ty - sy), abs(tz - sz)) < 2:
                            continue
                        add(_holder(lv, sx, sy, sz), ht, sz)
    else:
        for tx, ty, tz in product(range(nb), repeat=3):
            ht = _holder(lv, tx, ty, tz)
            for sx, sy, sz in product(range(tx - 1, tx + 2), range(ty - 1, ty + 2),
                                      range(tz - 1, tz + 2)):
                if 0 <= sx < nb and 0 <= sy < nb and 0 <= sz < nb and (sx, sy, sz) != (tx, ty, tz):
                    add(_holder(lv, sx, sy, sz), ht, sz)
    return {k: sorted(v) for k, v in out.items()}


def check():
    """sent() == brute() on every pass; raises on a difference."""
    for pi in range(S.NPASS):
        a, b = sent(pi), brute(pi)
        assert a == b, f"{S.PASSES[pi][0]}: generator != brute force " \
                       f"({len(a)} vs {len(b)} pairs)"


# sha256 (first 16 hex) of mp_hdr + mp_ent bytes and cap of the GEN tables
# (mp_tables.py, commit 764c173, run on hardware), per (max_bodies_pe, C, P).
GEN_TABLES = {(8, 16, 4): (99, "478e95dfda1ca6e3"),
              (4, 32, 4): (137, "d624eb444eb39572"),
              (8, 16, 5): (99, "90b2354f520c845f")}


def _gen_layout(max_bodies_pe):
    """The tables in the GEN device layout (3 mp_hdr words per pass: n_rx,
    entries, offset; entries in table order), for regress() only: the content
    (who sends what to whom) is unchanged by the transport, the layout is not."""
    W, Hf = MAP.width, MAP.height
    rows = {q: [] for q in PES}
    hdr = np.zeros((Hf, W, S.NPASS * 3), dtype=np.uint16)
    for pi in range(S.NPASS):
        ents = pass_entries(pi)
        n_rx, _ = pass_counts(pi, max_bodies_pe)
        for q in PES:
            lst = ents.get(q, [])
            hdr[q[1], q[0], pi * 3:pi * 3 + 3] = (n_rx[q], len(lst), len(rows[q]))
            rows[q] += [(dx, dy, zb | (zn << 8)) for (dx, dy, zb, zn) in lst]
    cap = max(1, max(len(r) for r in rows.values()))
    ent = np.zeros((Hf, W, ENT_W * cap), dtype=np.uint16)
    for (x, y), r in rows.items():
        for i, e in enumerate(r):
            ent[y, x, ENT_W * i:ENT_W * i + ENT_W] = e
    return hdr, ent, cap


def regress():
    """With NK = 1 the table content equals that of GEN (every recorded case)."""
    import hashlib
    for (mb, c, p), (cap0, h0) in GEN_TABLES.items():
        configure(c, p, c)
        hdr, ent, cap = _gen_layout(mb)
        h = hashlib.sha256(hdr.tobytes() + ent.tobytes()).hexdigest()[:16]
        assert (cap, h) == (cap0, h0), f"NK = 1 tables differ from GEN at {(mb, c, p)}"


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    mb = int(args[0]) if len(args) > 0 else 4
    c = int(args[1]) if len(args) > 1 else 32
    p = int(args[2]) if len(args) > 2 else 4
    zp = int(args[3]) if len(args) > 3 else 8
    if "--regress" in sys.argv:
        regress()
        print(f"regress OK: NK = 1 tables == GEN at {sorted(GEN_TABLES)}")
    configure(c, p, zp)
    print(f"C={c}  P={p}  ZP={zp}  NK={MAP.nk}  block {MAP.bw}x{MAP.bh}  "
          f"rectangle {MAP.width}x{MAP.height} = {len(PES)} PE  max_bodies_pe={mb}  "
          f"NPASS={S.NPASS}  P2P {S.p2p_cells(mb)} cells/msg")
    if "--check" in sys.argv:
        check()
        print("check OK: generator == box-by-box brute force on every pass")
    st, cap = stats(mb)
    print(f"{'pass':7s} {'messages':>9s} {'max rx/PE':>9s} {'max ent/PE':>10s} {'max tx/PE':>9s}")
    for pi, (n, mrx, ment, mtx) in enumerate(st):
        print(f"{S.PASSES[pi][0]:7s} {n:9d} {mrx:9d} {ment:10d} {mtx:9d}")
    print(f"total messages {sum(r[0] for r in st)}; mp_ent cap {cap} entries = "
          f"{ENT_W * cap * 2} B per PE; mp_hdr {S.NPASS * HDR_F * 2} B per PE")
