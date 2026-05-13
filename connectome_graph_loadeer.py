"""
connectome_graph_loader.py
==========================
Unified loader for all four BALBc connectome graphs.

Reads from CSV files and returns structured graph data
with canonical MAGMA vertex numbering.

Usage:
    from connectome_graph_loader import load_graph, load_all_graphs
    
    g = load_graph("Q_6")
    g = load_graph("Q_7P")
    g = load_graph("Q_7L")  
    g = load_graph("Q_8")
    
    graphs = load_all_graphs()

Graph data structure:
    g.name          : "Q_6" etc
    g.n_v           : number of vertices
    g.n_arr         : number of directed arrows
    g.vertices      : {int_idx: name}  canonical MAGMA numbering
    g.v_map         : {name: int_idx}
    g.arrows        : list of Arrow objects
    g.sym_pairs     : list of (name1, name2) symmetric pairs
    g.asym_arrows   : list of (src, tgt, weight) asymmetric arrows
    g.centroids     : {name: (x,y,z)} 3D centroid coordinates
    g.b1            : first Betti number (= len(sym_pairs) - n_v + 1)
    g.surface       : "cylinder" or "trinion"

Arrow object:
    a.idx           : arrow index (1-based, MAGMA compatible)
    a.src, a.tgt    : region names
    a.v_src, a.v_tgt: canonical vertex indices
    a.weight        : biological axonal weight
    a.is_sym        : True if has a reverse arrow
    a.rev_idx       : index of reverse arrow (0 if asymmetric)
"""

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ── Configuration ─────────────────────────────────────────────────────────────

CSV_FILES = {
    "Q_6":  ("regions_six.csv",           "region_edges_six.csv"),
    "Q_7L": ("regions_six_ANDLSX.csv",    "region_edges_six_ANDLSX.csv"),
    "Q_7P": ("regions_six_ANDPAL.csv",    "region_edges_six_ANDPAL.csv"),
    "Q_8":  ("regions_six_ANDPALLSX.csv", "region_edges_six_ANDPALLSX.csv"),
}

# Canonical vertex ordering (MAGMA convention)
BASE_ORDER = ["CA1sp", "BLA", "HY", "HPF", "sAMY", "LA"]

# Surface type from b1
SURFACE_MAP = {1: "cylinder", 2: "trinion"}

# Cyclic ordering at sAMY for ribbon graph (determines surface)
# Key insight from Phase 1: PAL must be between HPF and LA
SAMY_CYCLIC = {
    "Q_6":  ["BLA", "HY", "HPF", "LA"],
    "Q_7L": ["BLA", "HY", "HPF", "LA"],
    "Q_7P": ["BLA", "HY", "HPF", "PAL", "LA"],
    "Q_8":  ["BLA", "HY", "HPF", "PAL", "LA"],
}


@dataclass
class Arrow:
    idx:     int
    src:     str
    tgt:     str
    v_src:   int
    v_tgt:   int
    weight:  float
    is_sym:  bool
    rev_idx: int  # 0 if asymmetric

    def __repr__(self):
        rev = f"rev={self.rev_idx}" if self.is_sym else "ASYM"
        return (f"Arrow[{self.idx}] {self.src}→{self.tgt} "
                f"(v{self.v_src}→v{self.v_tgt}) w={self.weight:.4f} [{rev}]")


@dataclass
class ConnectomeGraph:
    name:       str
    n_v:        int
    n_arr:      int
    vertices:   Dict[int, str]
    v_map:      Dict[str, int]
    arrows:     List[Arrow]
    sym_pairs:  List[Tuple[str, str]]
    asym_arrows: List[Tuple[str, str, float]]
    centroids:  Dict[str, Tuple[float, float, float]]
    b1:         int
    surface:    str
    samy_cyclic: List[str]

    def arrow_by_idx(self, idx: int) -> Optional[Arrow]:
        return next((a for a in self.arrows if a.idx == idx), None)

    def arrows_from(self, region: str) -> List[Arrow]:
        return [a for a in self.arrows if a.src == region]

    def arrows_to(self, region: str) -> List[Arrow]:
        return [a for a in self.arrows if a.tgt == region]

    def sym_arrow_pair(self, r1: str, r2: str) -> Tuple[Optional[Arrow], Optional[Arrow]]:
        """Return (fwd arrow r1→r2, rev arrow r2→r1)."""
        fwd = next((a for a in self.arrows if a.src==r1 and a.tgt==r2), None)
        rev = next((a for a in self.arrows if a.src==r2 and a.tgt==r1), None)
        return fwd, rev

    def hashimoto_matrix(self):
        """B_Ihara: n_arr × n_arr nonbacktracking adjacency matrix."""
        import numpy as np
        B = np.zeros((self.n_arr, self.n_arr))
        for i, ai in enumerate(self.arrows):
            for j, aj in enumerate(self.arrows):
                # Composable and not backtracking
                if (ai.v_tgt == aj.v_src and
                    not (ai.is_sym and ai.rev_idx == aj.idx)):
                    B[i, j] = 1
        return B

    def weighted_hashimoto(self):
        """Weighted B_Ihara using biological edge weights."""
        import numpy as np
        B = np.zeros((self.n_arr, self.n_arr))
        for i, ai in enumerate(self.arrows):
            for j, aj in enumerate(self.arrows):
                if (ai.v_tgt == aj.v_src and
                    not (ai.is_sym and ai.rev_idx == aj.idx)):
                    B[i, j] = aj.weight  # weight of the continuation
        return B

    def spectral_radius(self):
        """ρ(B_Ihara) and ρ/√q."""
        import numpy as np
        B = self.hashimoto_matrix()
        eigs = np.linalg.eigvals(B)
        rho = max(abs(e) for e in eigs)
        # q_max = max nonbacktracking out-degree
        q_max = max(
            sum(1 for j, aj in enumerate(self.arrows)
                if ai.v_tgt == aj.v_src and
                not (ai.is_sym and ai.rev_idx == aj.idx))
            for ai in self.arrows
        )
        return rho, rho / (q_max ** 0.5) if q_max > 0 else rho

    def summary(self):
        rho, ratio = self.spectral_radius()
        print(f"{self.name}:")
        print(f"  Vertices ({self.n_v}): "
              f"{[self.vertices[i] for i in sorted(self.vertices)]}")
        print(f"  Arrows: {self.n_arr} "
              f"({len(self.sym_pairs)*2} sym + {len(self.asym_arrows)} asym)")
        print(f"  b₁ = {self.b1}  →  Surface: {self.surface}")
        print(f"  ρ(B_Ihara) = {rho:.4f},  ρ/√q = {ratio:.4f}")
        print(f"  Cyclic at sAMY: {self.samy_cyclic}")


def _canonical_idx(region: str, all_regions: List[str]) -> int:
    if region in BASE_ORDER:
        return BASE_ORDER.index(region) + 1
    extras = sorted(r for r in all_regions if r not in BASE_ORDER)
    return len(BASE_ORDER) + extras.index(region) + 1


def load_graph(name: str, data_dir: str = ".") -> ConnectomeGraph:
    """Load a single connectome graph from CSV files."""
    if name not in CSV_FILES:
        raise ValueError(f"Unknown graph: {name}. Choose from {list(CSV_FILES)}")

    rf, ef = CSV_FILES[name]

    # Load regions
    regions_by_idx = {}
    centroids = {}
    for d in [data_dir, "/mnt/user-data/uploads", "."]:
        rpath = os.path.join(d, rf)
        if os.path.exists(rpath):
            with open(rpath) as f:
                for row in csv.DictReader(f):
                    idx = int(row['index'])
                    regions_by_idx[idx] = row['region']
                    centroids[row['region']] = (
                        float(row['centroid_x']),
                        float(row['centroid_y']),
                        float(row['centroid_z'])
                    )
            break
    else:
        raise FileNotFoundError(f"Cannot find {rf}")

    all_regions = sorted(regions_by_idx.values())
    v_map = {r: _canonical_idx(r, all_regions) for r in all_regions}
    vertices = {_canonical_idx(r, all_regions): r for r in all_regions}

    # Load edges
    edges_raw = []
    for d in [data_dir, "/mnt/user-data/uploads", "."]:
        epath = os.path.join(d, ef)
        if os.path.exists(epath):
            with open(epath) as f:
                for row in csv.DictReader(f):
                    s = regions_by_idx[int(row['src_idx'])]
                    t = regions_by_idx[int(row['tgt_idx'])]
                    w = float(row['weight'])
                    if s != t and w > 0:
                        edges_raw.append((s, t, w))
            break
    else:
        raise FileNotFoundError(f"Cannot find {ef}")

    # Classify edges
    sym_dict = {}
    asym_list = []
    for s, t, w in edges_raw:
        has_rev = any(s2==t and t2==s for s2,t2,_ in edges_raw)
        if has_rev:
            key = tuple(sorted([s,t], key=lambda r: v_map[r]))
            if key not in sym_dict:
                fwd_w = next(w2 for s2,t2,w2 in edges_raw
                             if s2==key[0] and t2==key[1])
                rev_w = next(w2 for s2,t2,w2 in edges_raw
                             if s2==key[1] and t2==key[0])
                sym_dict[key] = (fwd_w, rev_w)
        else:
            if (s,t,w) not in asym_list:
                asym_list.append((s,t,w))

    # Build arrow list
    arrows = []
    idx = 1
    sym_pairs_list = []

    for key in sorted(sym_dict.keys(), key=lambda k: (v_map[k[0]], v_map[k[1]])):
        s, t = key
        fwd_w, rev_w = sym_dict[key]
        sym_pairs_list.append((s, t))

        arrows.append(Arrow(idx, s, t, v_map[s], v_map[t],
                            fwd_w, True, idx+1))
        idx += 1
        arrows.append(Arrow(idx, t, s, v_map[t], v_map[s],
                            rev_w, True, idx-1))
        idx += 1

    for s, t, w in sorted(asym_list, key=lambda x: (v_map[x[0]], v_map[x[1]])):
        arrows.append(Arrow(idx, s, t, v_map[s], v_map[t], w, False, 0))
        idx += 1

    n_v = len(all_regions)
    n_sym = len(sym_pairs_list)
    b1 = n_sym - n_v + 1
    surface = SURFACE_MAP.get(b1, f"genus-? (b1={b1})")

    return ConnectomeGraph(
        name=name,
        n_v=n_v,
        n_arr=len(arrows),
        vertices=vertices,
        v_map=v_map,
        arrows=arrows,
        sym_pairs=sym_pairs_list,
        asym_arrows=asym_list,
        centroids=centroids,
        b1=b1,
        surface=surface,
        samy_cyclic=SAMY_CYCLIC.get(name, []),
    )


def load_all_graphs(data_dir: str = ".") -> Dict[str, ConnectomeGraph]:
    """Load all four connectome graphs."""
    return {name: load_graph(name, data_dir) for name in CSV_FILES}


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading all connectome graphs...")
    print()
    graphs = load_all_graphs("/mnt/user-data/uploads")
    for name, g in graphs.items():
        g.summary()
        print()

    # Verify arrow counts match MAGMA programs
    expected = {"Q_6": 14, "Q_7L": 16, "Q_7P": 18, "Q_8": 20}
    print("Arrow count verification:")
    for name, g in graphs.items():
        ok = g.n_arr == expected[name]
        print(f"  {name}: {g.n_arr} arrows (expected {expected[name]}): "
              f"{'OK' if ok else 'MISMATCH'}")

    # Verify b1 values
    expected_b1 = {"Q_6": 1, "Q_7L": 1, "Q_7P": 2, "Q_8": 2}
    print("\nb₁ verification:")
    for name, g in graphs.items():
        ok = g.b1 == expected_b1[name]
        print(f"  {name}: b1={g.b1} (expected {expected_b1[name]}): "
              f"{'OK' if ok else 'MISMATCH'}")

    # Spectral radii
    print("\nSpectral radii:")
    for name, g in graphs.items():
        rho, ratio = g.spectral_radius()
        print(f"  {name}: ρ={rho:.4f}  ρ/√q={ratio:.4f}")
