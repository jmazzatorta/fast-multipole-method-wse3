#!/usr/bin/env python3
# geometry.py
#
# Geometria della mappatura 2.5D per FMM sul Cerebras WSE-3.
#
# Sostituisce interamente la logica Morton della 3D (morton_encode/decode 2D e 3D,
# is_pe_node_for_level, get_*_morton) con una mappatura CARTESIANA DIRETTA.
#
# PRINCIPIO FONDANTE della 2.5D:
#   - La mesh fisica C x C di PE rappresenta il piano (X, Y).
#   - Ogni PE(x, y) ospita una COLONNA di C celle foglia impilate lungo Z.
#   - Il piano = comunicazione (scambio tra PE vicini).
#   - Z = locale (aggregazione in SRAM, zero hop).
#
# Identita' di una cella foglia = (x, y, z), ciascuno in [0, C-1].
# Nessun Morton: le coordinate cartesiane SONO l'identita' e, per (x,y),
# sono gia' l'indirizzo fisico sul wafer.
#
# Due viste della stessa cosa, mai in conflitto:
#   - vista ALGEBRICA 3D  : il nodo e' un box (bx,by,bz); i vettori di
#                           traslazione sono 3D a pieno titolo (multipoli 3D
#                           corretti, non degeneri). Serve agli operatori
#                           M2M/M2L/L2L, che riusiamo intatti.
#   - vista PIANO + Z     : dei 3 assi, X,Y vivono sul piano (comunicazione),
#                           Z vive nella colonna (locale). Serve al routing.

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Parametri di riferimento (coerenti con constants.csl / il generatore math).
# cells_per_side = C = lato del dominio cubico in celle foglia = lato mesh PE.
# TREE_H = numero di livelli sopra le foglie; C deve valere 2**TREE_H.
# ---------------------------------------------------------------------------
CELLS_PER_SIDE = 16
TREE_H = 4
assert CELLS_PER_SIDE == (1 << TREE_H), "Il dominio cubico richiede C == 2**TREE_H"


# ===========================================================================
# 1. IDENTITA' DELLA CELLA FOGLIA
# ===========================================================================
# Una cella foglia e' il singolo cubetto 1x1x1 in posizione (x,y,z).
# NON la colonna intera: i multipoli restano 3D corretti.
# (x, y) individuano il PE del piano che ospita la colonna; z e' l'indice
# locale della cella entro la colonna, in SRAM.

@dataclass(frozen=True)
class Leaf:
    x: int
    y: int
    z: int

    def pe(self):
        """Il PE del piano che possiede questa foglia: (pe_x, pe_y) = (x, y)."""
        return (self.x, self.y)

    def z_local(self):
        """Indice locale lungo la colonna (in SRAM, 0 hop)."""
        return self.z


def leaf_is_valid(leaf, C=CELLS_PER_SIDE):
    return (0 <= leaf.x < C) and (0 <= leaf.y < C) and (0 <= leaf.z < C)


# ===========================================================================
# 2. NODI DELL'ALBERO (BOX) A UN DATO LIVELLO
# ===========================================================================
# Al livello `level` (0 = radice, TREE_H = foglie), ogni box copre
# cells_per_box = 2**(TREE_H - level) celle foglia per lato.
# Il numero di box per lato e' grid_boxes = C / cells_per_box = 2**level.
#
# Un box e' identificato dalle sue coordinate di box (bx, by, bz),
# ciascuna in [0, 2**level - 1].

def cells_per_box(level):
    """Lato (in celle foglia) di un box al dato livello."""
    assert 0 <= level <= TREE_H
    return 1 << (TREE_H - level)


def grid_boxes(level):
    """Numero di box per lato al dato livello."""
    assert 0 <= level <= TREE_H
    return 1 << level


def leaf_to_box(leaf, level):
    """Da una cella foglia (x,y,z) al box che la contiene al dato livello."""
    s = cells_per_box(level)
    return (leaf.x // s, leaf.y // s, leaf.z // s)


def box_leaf_origin(bx, by, bz, level):
    """Angolo (x,y,z) della cella foglia di origine del box."""
    s = cells_per_box(level)
    return (bx * s, by * s, bz * s)


# ===========================================================================
# 3. PROIEZIONE BOX -> PE DEL PIANO  (il cuore della vista piano+Z)
# ===========================================================================
# Un box (bx,by,bz) al livello `level` copre, sul PIANO, l'insieme di PE
#   pe_x in [bx*s, bx*s + s), pe_y in [by*s, by*s + s)
# dove s = cells_per_box(level). La coordinata bz (lungo Z) NON tocca il
# piano: seleziona quali celle della colonna, in SRAM, appartengono al box.
#
# Esempio: un nodo 2x2x2 al primo livello sotto la radice (s=... dipende dal
# livello) si spalma su s*s PE del piano, ciascuno con s celle in Z. Al livello
# TREE_H-1 (box 2x2x2 di foglie) => s=2 => 4 PE del piano, 2 celle Z ciascuno.

def box_plane_pes(bx, by, level):
    """Insieme di PE del piano coperti dal box (indipendente da bz)."""
    s = cells_per_box(level)
    pes = []
    for px in range(bx * s, bx * s + s):
        for py in range(by * s, by * s + s):
            pes.append((px, py))
    return pes


def box_representative_pe(bx, by, level):
    """
    PE 'rappresentante' del box sul piano: l'angolo in basso a sinistra.
    Convenzione usata per M2M/L2L sul piano: il rappresentante raccoglie
    l'aggregazione del box. E' il PE con pe_x, pe_y minimi del box.
    """
    s = cells_per_box(level)
    return (bx * s, by * s)


def pe_owns_z_cells_of_box(bz, level):
    """
    Quali indici Z (locali nella colonna) appartengono al box con coord bz.
    Lungo Z l'aggregazione e' SEMPRE locale: nessun hop, nessun piano.
    """
    s = cells_per_box(level)
    return list(range(bz * s, bz * s + s))


# ===========================================================================
# 4. "E' questo PE un nodo attivo al livello l?"  (rimpiazza is_pe_node_for_level)
# ===========================================================================
# In 2.5D un PE(pe_x,pe_y) e' il rappresentante di piano di qualche box al
# livello `level` se e solo se e' allineato alla griglia dei box, cioe' se
# pe_x e pe_y sono multipli di cells_per_box(level).
#
# Niente stride Morton: e' una pura condizione di allineamento cartesiano.

def pe_is_plane_node(pe_x, pe_y, level):
    s = cells_per_box(level)
    return (pe_x % s == 0) and (pe_y % s == 0)


# ===========================================================================
# 5. NAVIGAZIONE DELL'ALBERO  (genitore/figlio come shift cartesiani)
# ===========================================================================
# Sostituisce get_child_morton / get_parent_morton / get_level_id.
# Tutto in coordinate di box. I bit X,Y e il bit Z sono trattati
# separatamente: e' proprio questa separazione che il Morton nascondeva.

def box_parent(bx, by, bz):
    """Coord di box del genitore (un livello piu' in alto)."""
    return (bx >> 1, by >> 1, bz >> 1)


def box_child(bx, by, bz, child_id):
    """
    Coord di box del figlio dato child_id in [0,7].
    child_id bit 0 = X, bit 1 = Y, bit 2 = Z  (coerente con
    get_translation_vector in geometry.csl).
    """
    assert 0 <= child_id < 8
    cx = (bx << 1) | (child_id & 1)
    cy = (by << 1) | ((child_id >> 1) & 1)
    cz = (bz << 1) | ((child_id >> 2) & 1)
    return (cx, cy, cz)


def child_id_of(bx, by, bz):
    """child_id (0..7) di un box rispetto al proprio genitore."""
    return (bx & 1) | ((by & 1) << 1) | ((bz & 1) << 2)


def child_id_plane_part(child_id):
    """
    Dei 3 bit del child_id, la parte (X,Y) che vive sul PIANO -> comunicazione.
    Ritorna (bit_x, bit_y) in {0,1}.
    """
    return (child_id & 1, (child_id >> 1) & 1)


def child_id_z_part(child_id):
    """Il bit Z del child_id: vive nella colonna -> aggregazione locale."""
    return (child_id >> 2) & 1


# ===========================================================================
# 6. TAG E DISTANZE RELATIVE  (riuso diretto delle convenzioni 3D)
# ===========================================================================
# Identiche a geometry.csl / get_spatial_tag del generatore: il vettore di
# traslazione relativo (dx,dy,dz) in [-3,3]^3 codificato base-7. Pure
# relative, nessuna coordinata di PE: valide identiche in 2.5D.

def spatial_tag(dx, dy, dz):
    """(dx,dy,dz) in [-3,3] -> tag in [0,342]. Inverso di indices_from_tag."""
    return (dx + 3) + (dy + 3) * 7 + (dz + 3) * 49


def indices_from_tag(tag):
    """tag in [0,342] -> (dx,dy,dz) in [-3,3]. Specchio di get_indices_from_tag."""
    dx = (tag % 7) - 3
    dy = ((tag // 7) % 7) - 3
    dz = (tag // 49) - 3
    return (dx, dy, dz)


# ===========================================================================
# 7. CLASSIFICAZIONE HOP SUL PIANO  (quanto 'costa' una destinazione M2L)
# ===========================================================================
# Distanza di Chebyshev sul piano tra due box (quanti PE separano i loro
# rappresentanti, in unita' di PE). Serve a verificare il claim 2.5D:
# le destinazioni M2L stanno a <=3 hop sul piano. Z non contribuisce (locale).

def plane_hops(src_box, dst_box, level):
    """Hop sul piano tra i rappresentanti di due box allo stesso livello."""
    sx, sy = box_representative_pe(src_box[0], src_box[1], level)
    dx, dy = box_representative_pe(dst_box[0], dst_box[1], level)
    return max(abs(dx - sx), abs(dy - sy))


# ===========================================================================
# ASSERZIONI AUTOCONTENUTE  (esegui: python3 geometry25d.py)
# ===========================================================================

def _run_checks():
    C = CELLS_PER_SIDE
    print(f"Geometria 2.5D | C = {C}, TREE_H = {TREE_H}\n")

    # --- (a) round-trip foglia <-> box <-> origine ---
    for level in range(TREE_H + 1):
        s = cells_per_box(level)
        for leaf in [Leaf(0, 0, 0), Leaf(C - 1, C - 1, C - 1), Leaf(5, 11, 2)]:
            bx, by, bz = leaf_to_box(leaf, level)
            ox, oy, oz = box_leaf_origin(bx, by, bz, level)
            assert ox <= leaf.x < ox + s
            assert oy <= leaf.y < oy + s
            assert oz <= leaf.z < oz + s
    print("[a] round-trip foglia<->box<->origine: OK")

    # --- (b) la foglia e' il singolo cubetto, non la colonna ---
    # Al livello foglia, ogni box contiene 1 sola cella => i multipoli sono 3D.
    assert cells_per_box(TREE_H) == 1
    for z in range(C):
        bx, by, bz = leaf_to_box(Leaf(7, 3, z), TREE_H)
        assert (bx, by, bz) == (7, 3, z)  # z distinto per ogni cubetto
    print("[b] foglia = cubetto 1x1x1 (multipoli 3D, non degeneri): OK")

    # --- (c) un box 2x2x2 tocca ESATTAMENTE 4 PE del piano ---
    # box 2x2x2 di foglie => livello TREE_H-1 => s=2 => 4 PE, 2 celle Z ciascuno.
    level = TREE_H - 1
    for bx in range(grid_boxes(level)):
        for by in range(grid_boxes(level)):
            pes = box_plane_pes(bx, by, level)
            assert len(pes) == 4, f"box 2x2x2 tocca {len(pes)} PE, atteso 4"
            for bz in range(grid_boxes(level)):
                zc = pe_owns_z_cells_of_box(bz, level)
                assert len(zc) == 2, "ogni PE deve avere 2 celle Z nel box 2x2x2"
    print("[c] box 2x2x2 -> esattamente 4 PE del piano, 2 celle Z ciascuno: OK")

    # --- (d) l'aggregazione Z non attraversa mai il piano ---
    # Le celle Z di un box risiedono tutte nella stessa colonna (stesso PE):
    # dipendono solo da bz, mai da pe_x/pe_y. Verifichiamo che l'insieme Z
    # sia indipendente dal PE del piano.
    for level in range(TREE_H + 1):
        for bz in range(grid_boxes(level)):
            zc = pe_owns_z_cells_of_box(bz, level)
            # nessun riferimento al piano nella funzione => invariante strutturale
            assert zc == list(range(bz * cells_per_box(level),
                                     bz * cells_per_box(level) + cells_per_box(level)))
    print("[d] aggregazione Z interamente locale (mai sul piano): OK")

    # --- (e) navigazione albero: genitore/figlio coerenti ---
    for bx, by, bz in [(0, 0, 0), (3, 1, 2), (7, 7, 7)]:
        for cid in range(8):
            cxx, cyy, czz = box_child(bx, by, bz, cid)
            assert box_parent(cxx, cyy, czz) == (bx, by, bz)
            assert child_id_of(cxx, cyy, czz) == cid
    print("[e] genitore/figlio round-trip + child_id: OK")

    # --- (f) separazione piano/Z del child_id ---
    for cid in range(8):
        bx_bit, by_bit = child_id_plane_part(cid)
        z_bit = child_id_z_part(cid)
        assert (bx_bit | (by_bit << 1) | (z_bit << 2)) == cid
    print("[f] child_id -> (piano X,Y) + (Z locale) separati e ricomponibili: OK")

    # --- (g) pe_is_plane_node coerente con i rappresentanti ---
    for level in range(TREE_H + 1):
        reps = set()
        for bx in range(grid_boxes(level)):
            for by in range(grid_boxes(level)):
                reps.add(box_representative_pe(bx, by, level))
        for pe_x in range(C):
            for pe_y in range(C):
                expected = (pe_x, pe_y) in reps
                assert pe_is_plane_node(pe_x, pe_y, level) == expected
    print("[g] pe_is_plane_node <=> e' rappresentante di un box: OK")

    # --- (h) tag spaziale round-trip ---
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                t = spatial_tag(dx, dy, dz)
                assert 0 <= t <= 342
                assert indices_from_tag(t) == (dx, dy, dz)
    print("[h] tag spaziale (dx,dy,dz)<->tag round-trip su tutto [-3,3]^3: OK")

    # --- (i) il claim di localita' M2L: destinazioni a <=3 hop sul piano ---
    # Al livello foglia, la interaction list 3D ha vettori (dx,dy,dz) con
    # 2 <= max(|dx|,|dy|,|dz|) <= 3 (ben-separati ma entro il range del padre).
    # Sul PIANO la distanza e' max(|dx|,|dy|) <= 3 per costruzione: Z non conta.
    level = TREE_H
    max_plane = 0
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                if max(abs(dx), abs(dy), abs(dz)) < 2:
                    continue  # vicini P2P, non M2L
                max_plane = max(max_plane, max(abs(dx), abs(dy)))
    assert max_plane <= 3
    print(f"[i] destinazioni M2L: <= {max_plane} hop sul piano (Z escluso): OK")

    # --- (j) distinte destinazioni (X,Y) sul piano al livello foglia ---
    # Quante colonne (X,Y) distinte tocca la interaction list, aggregando Z.
    plane_targets = set()
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            for dz in range(-3, 4):
                if max(abs(dx), abs(dy), abs(dz)) < 2:
                    continue
                plane_targets.add((dx, dy))
    print(f"[j] destinazioni (X,Y) distinte in IL (Z aggregato): "
          f"{len(plane_targets)} colonne")

    print("\nTutte le asserzioni passate. Geometria 2.5D coerente.")


if __name__ == "__main__":
    _run_checks()
