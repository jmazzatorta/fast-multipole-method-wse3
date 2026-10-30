#!/usr/bin/env python3
"""Plane-column -> PE block mapping for the z-split 2.5D FMM (reference decoder).

Every plane position (x, y) owns a column of NZ leaf cells along z. The column
is cut into NK = NZ / ZP chunks of ZP consecutive cells (ZP = 8 by default; a
power of two, ZP = NZ gives NK = 1, the one-PE-per-column mapping). The NK chunk
PEs of one (x, y) form a BW x BH block on the fabric (BW, BH powers of two,
BW * BH = NK). Block (x, y) starts at fabric (x * BW, y * BH): shifts only.

Inside a block the chunks follow a SNAKE order (row 0 left->right, row 1
right->left, ...), so chunks k and k+1 are always fabric neighbours and every
same-k PE of the neighbouring column sits exactly BW (or BH) PEs away.

All coordinates are relative to the program rectangle. Device arithmetic is
shifts, masks, adds and one conditional (no integer multiply).
"""
import sys

ZP = 8                       # default leaf cells per PE along z


def block_shape(nk, lbw=None):
    """(BW, BH) for NK chunks. Default: as square as possible, the longer side
    along y (the fabric is taller than wide: 762 x 1172). lbw = log2(BW) forces
    the width (lbw = 0: a 1 x NK column block, the narrowest rectangle)."""
    assert nk >= 1 and nk & (nk - 1) == 0, "NK must be a power of two"
    lk = nk.bit_length() - 1
    if lbw is None:
        lbw = lk // 2
    assert 0 <= lbw <= lk
    return 1 << lbw, 1 << (lk - lbw)


class PeMap:
    def __init__(self, nx, ny, nz, zp=ZP, lbw=None):
        assert zp >= 1 and zp & (zp - 1) == 0 and nz % zp == 0
        self.nx, self.ny, self.nz = nx, ny, nz
        self.zp = zp
        self.lzp = zp.bit_length() - 1
        self.nk = nz // zp
        self.bw, self.bh = block_shape(self.nk, lbw)
        self.lbw = self.bw.bit_length() - 1
        self.lbh = self.bh.bit_length() - 1
        self.width = nx << self.lbw
        self.height = ny << self.lbh

    # --- chunk index k <-> position (kx, ky) inside the block (snake) -------
    def k_to_kxy(self, k):
        ky = k >> self.lbw
        kx = k & (self.bw - 1)
        if ky & 1:
            kx = (self.bw - 1) - kx
        return kx, ky

    def kxy_to_k(self, kx, ky):
        if ky & 1:
            kx = (self.bw - 1) - kx
        return (ky << self.lbw) | kx

    # --- the decoder -------------------------------------------------------
    def cell_to_pe(self, x, y, z):
        """leaf cell (x, y, z) -> (fabric x, fabric y, local z in the chunk)."""
        k = z >> self.lzp
        kx, ky = self.k_to_kxy(k)
        return (x << self.lbw) | kx, (y << self.lbh) | ky, z & (self.zp - 1)

    def box_to_pe(self, x, y, zb, level, H):
        """box with z index zb at tree level `level` (leaf = H), plane owner
        column (x, y) -> PE holding it. A box taller than ZP spans several
        chunks; its owner is the chunk of its lowest cell (the z-rep)."""
        z0 = zb << (H - level)
        return self.cell_to_pe(x, y, z0)

    def pe_to_column(self, fx, fy):
        """inverse: fabric (fx, fy) -> (x, y, k, z_first)."""
        x, kx = fx >> self.lbw, fx & (self.bw - 1)
        y, ky = fy >> self.lbh, fy & (self.bh - 1)
        k = self.kxy_to_k(kx, ky)
        return x, y, k, k << self.lzp


def self_test(nx, ny, nz, zp=ZP):
    m = PeMap(nx, ny, nz, zp)
    seen = set()
    for y in range(ny):
        for x in range(nx):
            for z in range(nz):
                fx, fy, zl = m.cell_to_pe(x, y, z)
                assert 0 <= fx < m.width and 0 <= fy < m.height
                x2, y2, k, z0 = m.pe_to_column(fx, fy)
                assert (x2, y2, z0 + zl) == (x, y, z)
                seen.add((fx, fy))
    assert len(seen) == m.width * m.height == nx * ny * m.nk   # bijection
    for k in range(m.nk - 1):                                  # snake adjacency
        a, b = m.k_to_kxy(k), m.k_to_kxy(k + 1)
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1
    return m


if __name__ == "__main__":
    cases = [(16, 16, 16), (32, 32, 32), (64, 64, 64), (128, 128, 128),
             (256, 256, 256), (256, 256, 16), (512, 512, 16), (585, 585, 16)]
    if len(sys.argv) == 4:
        cases = [tuple(int(a) for a in sys.argv[1:])]
    FAB_W, FAB_H = 762, 1172       # --fabric-dims (memcpy margins not subtracted)
    print(f"{'domain':>14s} {'NK':>3s} {'block':>6s} {'rectangle':>11s} {'PEs':>9s}  fits fabric 762x1172")
    for nx, ny, nz in cases:
        if nx * ny * (nz // ZP) <= 4_000_000 and nx <= 128:
            m = self_test(nx, ny, nz)
        else:
            m = PeMap(nx, ny, nz)
        fits = m.width <= FAB_W and m.height <= FAB_H
        print(f"{nx:4d}x{ny:4d}x{nz:3d} {m.nk:3d} {m.bw:2d}x{m.bh:<3d} {m.width:5d}x{m.height:<5d} "
              f"{m.width * m.height:9d}  {'yes' if fits else 'NO'}")
    print("self-test OK (bijection, inverse, snake adjacency) on the cases up to 128")
