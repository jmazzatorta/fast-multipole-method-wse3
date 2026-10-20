#!/usr/bin/env python3
"""Event-level simulation of the serpentine pass state machine of pe.csl.

Mirrors the SERPENTINE PASS functions of pe.csl (serp_prepare / serp_begin /
serp_on_rx / serp_on_tx / serp_maybe_arm / serp_emit_own / serp_check_done)
for each pass in serp_tables.PASSES (P2P counts for a given max_bodies_pe):
  - per stream (FWD, BWD): one rx buffer, counters only, STRICT ALTERNATION of
    send and drain (own messages first, a drain armed only while tx is idle);
  - a forward is sent straight from the rx buffer with hop + 1;
  - every PE is prepared (listening) before anyone begins, and the begins happen
    at random moments interleaved with fabric events, so a PE can receive and
    forward before its own launch.
Fabric = rendezvous (a send completes only when the receiver has an armed drain
on that link): the most restrictive model; real buffering only helps.
Task order is random (cooperative, run-to-completion).
Checks: no deadlock, exact counts, each PE consumes exactly the senders within
HMAX of it on the path, every required sender is delivered, the source decoded
from the hop count is right, and a drain and a send with the same queue number
are never active at once on a PE (shared microthread).

Usage: python3 sim_serp.py [SEEDS] [MAX_BODIES_PE]
"""
import random
import serp_tables as T

NONE = T.NONE


class Stream:
    def __init__(self, din, dout, n_rx, n_tx):
        self.din, self.dout, self.n_rx, self.n_tx = din, dout, n_rx, n_tx
        self.drains = 0
        self.emits = 0
        self.tx_busy = False


class PE:
    """Mirrors the SERPENTINE PASS functions of pe.csl (strict alternation)."""
    def __init__(s, sim, x, y, row, pi):
        s.sim, s.x, s.y, s.pi = sim, x, y, pi
        s.fwd = Stream(row[0], row[1], row[4], row[5])
        s.bwd = Stream(row[2], row[3], row[6], row[7])
        s.n_own = row[8]
        s.hmax = row[9]
        s.own = {id(s.fwd): 0, id(s.bwd): 0}
        s.rx_armed = [False] * 4      # pe.csl rx_armed[]: cleared by the rx TASK
        s.consumed = []
        s.done = False
        s.started = False
        s.errors = []

    def stream_of_rx(s, d):
        if d == s.fwd.din:
            return s.fwd
        if d == s.bwd.din:
            return s.bwd
        return None

    def ut_check(s, d_rx=None, d_tx=None):
        # A drain on rx code d and a send on tx code d use the same queue number
        # (iq d+2 / oq d+2), i.e. the same microthread: never both at once.
        for st in (s.fwd, s.bwd):
            if d_rx is not None and st.tx_busy and st.dout == d_rx:
                s.errors.append(f"ut conflict: drain {d_rx} while send busy")
        if d_tx is not None and s.rx_armed[d_tx]:
            s.errors.append(f"ut conflict: send {d_tx} while drain armed")

    def arm(s, d):
        s.ut_check(d_rx=d)
        s.rx_armed[d] = True
        s.sim.arm(s, d)

    def send(s, st, msg):
        s.ut_check(d_tx=st.dout)
        st.tx_busy = True
        s.sim.send(s, st.dout, msg)

    def own_left(s, st):
        if st.dout == NONE:
            return 0
        return s.n_own - s.own[id(st)]

    def emit_own(s, st):
        k = s.own[id(st)]
        s.own[id(st)] += 1
        s.send(st, {"src": (s.x, s.y), "k": k, "hop": 1})

    def maybe_arm(s, st):
        if st.din == NONE or st.drains >= st.n_rx:
            return
        if st.tx_busy or s.own_left(st) > 0:
            return
        if not s.rx_armed[st.din]:
            s.arm(st.din)

    def prepare(s):
        for st in (s.fwd, s.bwd):
            s.maybe_arm(st)

    def start(s):
        s.started = True
        for st in (s.fwd, s.bwd):
            if s.own_left(st) > 0:
                s.emit_own(st)
        s.check_done()

    def on_rx(s, d, msg):
        s.rx_armed[d] = False
        st = s.stream_of_rx(d)
        if st is None:
            s.errors.append(f"message on unused rx dir {d}")
            return
        st.drains += 1
        if st.drains > st.n_rx:
            s.errors.append("more drains than expected")
        hop = msg["hop"]
        fwd = st.dout != NONE and hop < s.hmax
        if fwd:
            if st.tx_busy:
                s.errors.append("forward while tx busy")
            s.send(st, dict(msg, hop=hop + 1))
        stream = "fwd" if st is s.fwd else "bwd"
        true_src = T.src_from_hop(s.pi, s.x, s.y, stream, hop)
        if true_src != msg["src"]:
            s.errors.append(f"hop decode mismatch {true_src} vs {msg['src']}")
        s.consumed.append((msg["src"], msg["k"], stream))
        if not fwd:
            s.maybe_arm(st)
        s.check_done()

    def on_tx(s, d):
        st = s.fwd if (d == s.fwd.dout and s.fwd.tx_busy) else s.bwd
        if not st.tx_busy or st.dout != d:
            s.errors.append("unexpected tx completion")
            return
        st.tx_busy = False
        st.emits += 1
        if s.started and s.own_left(st) > 0:
            s.emit_own(st)
        else:
            s.maybe_arm(st)
        s.check_done()

    def check_done(s):
        if s.done or not s.started:
            return
        for st in (s.fwd, s.bwd):
            if st.drains != st.n_rx or st.emits != st.n_tx or st.tx_busy:
                return
        s.done = True


class Sim:
    def __init__(s, pi, seed, mb):
        s.pi = pi
        s.rng = random.Random(seed)
        tab = T.pass_table(pi, mb)
        s.pes = {xy: PE(s, xy[0], xy[1], row, pi) for xy, row in tab.items()}
        s.armed = set()
        s.sends = {}
        s.ready = []

    def arm(s, pe, d):
        k = ((pe.x, pe.y), d)
        assert k not in s.armed
        s.armed.add(k)
        s.match()

    def send(s, pe, d, msg):
        k = ((pe.x, pe.y), d)
        assert k not in s.sends, "two sends in flight on one link"
        s.sends[k] = msg
        s.match()

    def match(s):
        for (xy, d), msg in list(s.sends.items()):
            dx, dy = T.STEP[d]
            rx = ((xy[0] + dx, xy[1] + dy), d)
            if rx in s.armed:
                del s.sends[(xy, d)]
                s.armed.discard(rx)
                s.ready.append(("tx", xy, d, None))
                s.ready.append(("rx", rx[0], d, msg))

    def run(s):
        # Every PE is PREPARED (listening) before anyone begins; the own
        # launches (begin) then happen at random moments interleaved with the
        # fabric events, so a PE may receive and forward before it begins.
        for pe in s.pes.values():
            pe.prepare()
        waiting = list(s.pes.values())
        s.rng.shuffle(waiting)
        while s.ready or waiting:
            if waiting and (not s.ready or s.rng.random() < 0.05):
                waiting.pop().start()
                continue
            kind, xy, d, msg = s.ready.pop(s.rng.randrange(len(s.ready)))
            pe = s.pes[xy]
            pe.on_tx(d) if kind == "tx" else pe.on_rx(d, msg)


def check(pi, seed, mb):
    sim = Sim(pi, seed, mb)
    sim.run()
    name, pl, dl, snd, needs = T.PASSES[pi]
    own = T.pass_nmsgs(pi, mb)
    p = T.path(pl)
    idx = {q: i for i, q in enumerate(p)}
    hm = T.hmax(pi)
    need = needs()
    bad = []
    for xy, pe in sim.pes.items():
        if pe.errors:
            bad.append((xy, pe.errors[:2]))
            continue
        if not pe.done:
            bad.append((xy, "not done"))
            continue
        got = {(src, k) for src, k, _ in pe.consumed}
        if len(got) != len(pe.consumed):
            bad.append((xy, "duplicate delivery"))
        if xy in idx:
            i = idx[xy]
            exp = {(q, k) for j, q in enumerate(p) if 0 < abs(j - i) <= hm
                   and snd(*q) for k in range(own)}
            if got != exp:
                bad.append((xy, "delivered set != window set"))
        req = {(q, k) for q in need.get(xy, []) for k in range(own)}
        if not req <= got:
            bad.append((xy, "missing required sender"))
    return bad, len(sim.sends)


if __name__ == "__main__":
    import sys
    seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    mb = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    allok = True
    for pi in range(T.NPASS):
        fails = 0
        for seed in range(seeds):
            bad, left = check(pi, seed, mb)
            if bad or left:
                fails += 1
                if fails == 1:
                    print("   first failure:", bad[:3], "leftover sends", left)
        allok &= fails == 0
        print(f"pass {pi} {T.PASSES[pi][0]:6s}: {seeds - fails}/{seeds} seeds OK", flush=True)
    print("ALL OK" if allok else "FAILURES")
