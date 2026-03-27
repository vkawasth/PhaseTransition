"""
Complete Quiver Representation: From 3×3 Matrices to 6-Vertex, 28-Arrow Wavelet Quiver
=====================================================================================
This script prints:
1. 3×3 matrix with two 2×2 minors and overlap at each node
2. 4 transition arrows (HH, HT, TH, TT) between nodes
3. The full 6-vertex, 28-arrow wavelet parameter quiver
4. All mappings between levels
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import pandas as pd

# ============================================================================
# PART 1: 3×3 MATRIX WITH TWO 2×2 MINORS AND OVERLAP
# ============================================================================

@dataclass
class ThreeByThreeMatrix:
    """
    Represents a 3×3 matrix with:
    - Top-left 2×2 minor: Node 1 toric variety
    - Bottom-right 2×2 minor: Node 2 toric variety
    - Overlap element e: support modulator
    """
    a: float  # p_HH at Node 1
    b: float  # p_HT at Node 1
    c: float  # coupling term
    d: float  # p_TH at Node 1
    e: float  # OVERLAP (support modulator) - appears in both minors
    f: float  # coupling term
    g: float  # coupling term
    h: float  # p_TH at Node 2
    i: float  # p_TT at Node 2
    
    def __post_init__(self):
        # Ensure probabilities sum to 1 for each minor approximation
        self.normalize()
    
    def normalize(self):
        """Normalize to maintain probability conservation"""
        # Node 1 minor (top-left 2×2)
        total_1 = self.a + self.b + self.d + self.e
        if total_1 > 0:
            self.a /= total_1
            self.b /= total_1
            self.d /= total_1
            self.e /= total_1
        
        # Node 2 minor (bottom-right 2×2)
        total_2 = self.e + self.f + self.h + self.i
        if total_2 > 0:
            self.e /= total_2
            self.f /= total_2
            self.h /= total_2
            self.i /= total_2
    
    def top_left_minor(self) -> np.ndarray:
        """Return top-left 2×2 minor (Node 1 toric variety)"""
        return np.array([[self.a, self.b], [self.d, self.e]])
    
    def bottom_right_minor(self) -> np.ndarray:
        """Return bottom-right 2×2 minor (Node 2 toric variety)"""
        return np.array([[self.e, self.f], [self.h, self.i]])
    
    def toric_deviation_1(self) -> float:
        """ε₁ = det(top-left minor)"""
        return self.a * self.e - self.b * self.d
    
    def toric_deviation_2(self) -> float:
        """ε₂ = det(bottom-right minor)"""
        return self.e * self.i - self.f * self.h
    
    def print_matrix(self, label: str = ""):
        """Pretty print the 3×3 matrix"""
        print(f"\n{'='*60}")
        print(f"3×3 Matrix {label}")
        print(f"{'='*60}")
        
        # Create matrix display
        matrix = np.array([
            [self.a, self.b, self.c],
            [self.d, self.e, self.f],
            [self.g, self.h, self.i]
        ])
        
        print("\n     ┌─────────────────────────────────────────┐")
        for i, row in enumerate(matrix):
            if i == 0:
                print(f"     │  [{row[0]:6.3f}  {row[1]:6.3f}  {row[2]:6.3f}]  ← Node 1 Minor │")
            elif i == 1:
                print(f"     │  [{row[0]:6.3f}  {row[1]:6.3f}  {row[2]:6.3f}]  ← Overlap e   │")
            else:
                print(f"     │  [{row[0]:6.3f}  {row[1]:6.3f}  {row[2]:6.3f}]  ← Node 2 Minor │")
        print("     └─────────────────────────────────────────┘")
        
        print(f"\nTop-left 2×2 minor (Node 1):")
        print(f"    ┌──────────┐")
        print(f"    │ {self.a:6.3f}  {self.b:6.3f} │")
        print(f"    │ {self.d:6.3f}  {self.e:6.3f} │")
        print(f"    └──────────┘")
        print(f"    ε₁ = det = {self.toric_deviation_1():8.5f}")
        
        print(f"\nBottom-right 2×2 minor (Node 2):")
        print(f"    ┌──────────┐")
        print(f"    │ {self.e:6.3f}  {self.f:6.3f} │")
        print(f"    │ {self.h:6.3f}  {self.i:6.3f} │")
        print(f"    └──────────┘")
        print(f"    ε₂ = det = {self.toric_deviation_2():8.5f}")
        
        print(f"\nOverlap element e = {self.e:.5f} (support modulator)")
        print(f"    When e = 0: Nodes are independent")
        print(f"    When e ≠ 0: One toric is lifted relative to the other")


# ============================================================================
# PART 2: 4 TRANSITION ARROWS (HH, HT, TH, TT)
# ============================================================================

@dataclass
class TransitionArrows:
    """
    Represents the 4 transition arrows between nodes:
    - HH: both molecules in H state
    - HT: A=H, B=T
    - TH: A=T, B=H
    - TT: both molecules in T state
    """
    kappa_HH: float = 0.25
    kappa_HT: float = 0.25
    kappa_TH: float = 0.25
    kappa_TT: float = 0.25
    
    def __post_init__(self):
        self.normalize()
    
    def normalize(self):
        """Ensure probabilities sum to 1"""
        total = self.kappa_HH + self.kappa_HT + self.kappa_TH + self.kappa_TT
        if total > 0:
            self.kappa_HH /= total
            self.kappa_HT /= total
            self.kappa_TH /= total
            self.kappa_TT /= total
    
    def transition_matrix(self) -> np.ndarray:
        """Return 4×4 diagonal transition matrix"""
        return np.diag([self.kappa_HH, self.kappa_HT, self.kappa_TH, self.kappa_TT])
    
    def print_arrows(self, source: int, target: int):
        """Print the 4 arrows between nodes"""
        print(f"\n  4 Transition Arrows: Node {source} → Node {target}")
        print(f"  ┌─────────────────────────────────────────────────────────┐")
        print(f"  │  Arrow HH:  {self.kappa_HH:6.3f}   (κ_HH)  Both molecules in H state  │")
        print(f"  │  Arrow HT:  {self.kappa_HT:6.3f}   (κ_HT)  A=H, B=T                   │")
        print(f"  │  Arrow TH:  {self.kappa_TH:6.3f}   (κ_TH)  A=T, B=H                   │")
        print(f"  │  Arrow TT:  {self.kappa_TT:6.3f}   (κ_TT)  Both molecules in T state  │")
        print(f"  └─────────────────────────────────────────────────────────┘")
        
        # Show how these relate to molecular transitions
        print(f"\n  Molecular Interpretation:")
        print(f"    κ_HH = κ_A·κ_B   = {self.kappa_HH:.3f}")
        print(f"    κ_HT = κ_A·(1-κ_B) = {self.kappa_HT:.3f}")
        print(f"    κ_TH = (1-κ_A)·κ_B = {self.kappa_TH:.3f}")
        print(f"    κ_TT = (1-κ_A)·(1-κ_B) = {self.kappa_TT:.3f}")
        
        # Solve for κ_A, κ_B
        kappa_A = self.kappa_HH + self.kappa_HT
        kappa_B = self.kappa_HH + self.kappa_TH
        print(f"\n  Implied molecular transition rates:")
        print(f"    κ_A = {kappa_A:.3f} (molecule A transition probability)")
        print(f"    κ_B = {kappa_B:.3f} (molecule B transition probability)")


# ============================================================================
# PART 3: 6-VERTEX, 28-ARROW WAVELET PARAMETER QUIVER
# ============================================================================

@dataclass
class WaveletParameterQuiver:
    """
    6-vertex quiver for wavelet parameters (a, b, ω₁, ω₂, φ, κ)
    with 28 arrows representing all interactions.
    """
    
    # Vertex indices
    V_A = 0      # amplitude of molecule A
    V_B = 1      # amplitude of molecule B
    V_W1 = 2     # frequency of molecule A
    V_W2 = 3     # frequency of molecule B
    V_PHI = 4    # phase difference
    V_KAPPA = 5  # coupling strength
    
    vertex_names = ['a', 'b', 'ω₁', 'ω₂', 'φ', 'κ']
    
    # Transition rates (will be set by Renkin-Crone flow)
    rates: Dict[Tuple[int, int], float] = field(default_factory=dict)
    
    def __post_init__(self):
        self._initialize_rates()
    
    def _initialize_rates(self):
        """Initialize all 28 arrows with example rates"""
        
        # 1. Self-loops (6 arrows) - decay/drift
        self.rates[(self.V_A, self.V_A)] = -0.173    # decay
        self.rates[(self.V_B, self.V_B)] = -0.347    # decay
        self.rates[(self.V_W1, self.V_W1)] = -0.01   # frequency drift
        self.rates[(self.V_W2, self.V_W2)] = -0.01   # frequency drift
        self.rates[(self.V_PHI, self.V_PHI)] = -0.005 # phase drift
        self.rates[(self.V_KAPPA, self.V_KAPPA)] = -0.1 # coupling decay
        
        # 2. Amplitude → Frequency (2 arrows)
        self.rates[(self.V_A, self.V_W1)] = 0.1   # a → ω₁
        self.rates[(self.V_B, self.V_W2)] = 0.1   # b → ω₂
        
        # 3. Frequency → Amplitude (2 arrows)
        self.rates[(self.V_W1, self.V_A)] = 0.05  # ω₁ → a
        self.rates[(self.V_W2, self.V_B)] = 0.05  # ω₂ → b
        
        # 4. Cross-coupling A↔B (4 arrows)
        self.rates[(self.V_A, self.V_B)] = 0.2    # a → b
        self.rates[(self.V_B, self.V_A)] = 0.2    # b → a
        self.rates[(self.V_W1, self.V_W2)] = 0.03 # ω₁ → ω₂
        self.rates[(self.V_W2, self.V_W1)] = 0.03 # ω₂ → ω₁
        
        # 5. Phase coupling (6 arrows)
        self.rates[(self.V_PHI, self.V_A)] = 0.01   # φ → a
        self.rates[(self.V_PHI, self.V_B)] = 0.01   # φ → b
        self.rates[(self.V_A, self.V_PHI)] = 0.02   # a → φ
        self.rates[(self.V_B, self.V_PHI)] = 0.02   # b → φ
        self.rates[(self.V_W1, self.V_PHI)] = 0.015 # ω₁ → φ
        self.rates[(self.V_W2, self.V_PHI)] = 0.015 # ω₂ → φ
        
        # 6. Coupling strength interactions (8 arrows)
        self.rates[(self.V_KAPPA, self.V_A)] = 0.15   # κ → a
        self.rates[(self.V_KAPPA, self.V_B)] = 0.15   # κ → b
        self.rates[(self.V_KAPPA, self.V_W1)] = 0.1   # κ → ω₁
        self.rates[(self.V_KAPPA, self.V_W2)] = 0.1   # κ → ω₂
        self.rates[(self.V_A, self.V_KAPPA)] = 0.1    # a → κ
        self.rates[(self.V_B, self.V_KAPPA)] = 0.1    # b → κ
        self.rates[(self.V_W1, self.V_KAPPA)] = 0.08  # ω₁ → κ
        self.rates[(self.V_W2, self.V_KAPPA)] = 0.08  # ω₂ → κ
    
    def get_transition_matrix(self) -> np.ndarray:
        """Return 6×6 transition matrix M where d(state)/dt = M·state"""
        M = np.zeros((6, 6))
        for (src, tgt), rate in self.rates.items():
            M[tgt, src] = rate
        return M
    
    def print_quiver(self):
        """Print the 6-vertex, 28-arrow quiver structure"""
        print(f"\n{'='*80}")
        print("6-VERTEX, 28-ARROW WAVELET PARAMETER QUIVER")
        print(f"{'='*80}")
        
        print("\nVERTICES (6):")
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print("  │  v_a  = amplitude of molecule A (opiate)                    │")
        print("  │  v_b  = amplitude of molecule B (norcain)                  │")
        print("  │  v_ω₁ = frequency/oscillation rate of molecule A           │")
        print("  │  v_ω₂ = frequency/oscillation rate of molecule B           │")
        print("  │  v_φ  = phase difference between molecules                 │")
        print("  │  v_κ  = coupling strength (interaction between A and B)    │")
        print("  └─────────────────────────────────────────────────────────────┘")
        
        print("\nARROWS (28):")
        print("  ┌─────────────────────────────────────────────────────────────┐")
        print("  │ SELF-LOOPS (6):                                            │")
        for src in range(6):
            rate = self.rates.get((src, src), 0)
            print(f"  │   {self.vertex_names[src]} → {self.vertex_names[src]}: {rate:6.3f} (decay/drift)  │")
        
        print("  │                                                             │")
        print("  │ AMPLITUDE → FREQUENCY (2):                                  │")
        print(f"  │   a → ω₁: {self.rates[(self.V_A, self.V_W1)]:6.3f}                    │")
        print(f"  │   b → ω₂: {self.rates[(self.V_B, self.V_W2)]:6.3f}                    │")
        
        print("  │                                                             │")
        print("  │ FREQUENCY → AMPLITUDE (2):                                  │")
        print(f"  │   ω₁ → a: {self.rates[(self.V_W1, self.V_A)]:6.3f}                    │")
        print(f"  │   ω₂ → b: {self.rates[(self.V_W2, self.V_B)]:6.3f}                    │")
        
        print("  │                                                             │")
        print("  │ CROSS-COUPLING A↔B (4):                                     │")
        print(f"  │   a → b: {self.rates[(self.V_A, self.V_B)]:6.3f}                    │")
        print(f"  │   b → a: {self.rates[(self.V_B, self.V_A)]:6.3f}                    │")
        print(f"  │   ω₁ → ω₂: {self.rates[(self.V_W1, self.V_W2)]:6.3f}                  │")
        print(f"  │   ω₂ → ω₁: {self.rates[(self.V_W2, self.V_W1)]:6.3f}                  │")
        
        print("  │                                                             │")
        print("  │ PHASE COUPLING (6):                                         │")
        print(f"  │   φ → a: {self.rates[(self.V_PHI, self.V_A)]:6.3f}                    │")
        print(f"  │   φ → b: {self.rates[(self.V_PHI, self.V_B)]:6.3f}                    │")
        print(f"  │   a → φ: {self.rates[(self.V_A, self.V_PHI)]:6.3f}                    │")
        print(f"  │   b → φ: {self.rates[(self.V_B, self.V_PHI)]:6.3f}                    │")
        print(f"  │   ω₁ → φ: {self.rates[(self.V_W1, self.V_PHI)]:6.3f}                  │")
        print(f"  │   ω₂ → φ: {self.rates[(self.V_W2, self.V_PHI)]:6.3f}                  │")
        
        print("  │                                                             │")
        print("  │ COUPLING STRENGTH INTERACTIONS (8):                         │")
        print(f"  │   κ → a: {self.rates[(self.V_KAPPA, self.V_A)]:6.3f}                    │")
        print(f"  │   κ → b: {self.rates[(self.V_KAPPA, self.V_B)]:6.3f}                    │")
        print(f"  │   κ → ω₁: {self.rates[(self.V_KAPPA, self.V_W1)]:6.3f}                  │")
        print(f"  │   κ → ω₂: {self.rates[(self.V_KAPPA, self.V_W2)]:6.3f}                  │")
        print(f"  │   a → κ: {self.rates[(self.V_A, self.V_KAPPA)]:6.3f}                    │")
        print(f"  │   b → κ: {self.rates[(self.V_B, self.V_KAPPA)]:6.3f}                    │")
        print(f"  │   ω₁ → κ: {self.rates[(self.V_W1, self.V_KAPPA)]:6.3f}                  │")
        print(f"  │   ω₂ → κ: {self.rates[(self.V_W2, self.V_KAPPA)]:6.3f}                  │")
        print("  └─────────────────────────────────────────────────────────────┘")
        
        # Print total count
        print(f"\nTOTAL ARROWS: {len(self.rates)}")
        print("  (6 self-loops + 22 directed arrows = 28)")
    
    def evolve_state(self, state: np.ndarray, dt: float) -> np.ndarray:
        """Evolve state using quiver dynamics: d(state)/dt = M·state"""
        M = self.get_transition_matrix()
        return state + (M @ state) * dt
    
    def project_to_plucker(self, state: np.ndarray) -> np.ndarray:
        """Project 6-vertex quiver state to Gr(2,4) Plücker coordinates"""
        a, b, w1, w2, phi, kappa = state
        
        # Normalize to [0,1] for Plücker coordinates
        p12 = a / (a + b + 1e-8)
        p13 = w1 / (w1 + w2 + 1e-8)
        p14 = phi / np.pi
        p23 = kappa
        p24 = (a * b) / ((a + b)**2 + 1e-8)
        p34 = (w1 * w2) / ((w1 + w2)**2 + 1e-8)
        
        plucker = np.array([p12, p13, p14, p23, p24, p34])
        return plucker / np.linalg.norm(plucker)
    
    def reconstruct_signal(self, state: np.ndarray, t: float) -> float:
        """Reconstruct consciousness signal from quiver state"""
        a, b, w1, w2, phi, _ = state
        return a * np.cos(w1 * t + phi) + b * np.sin(w2 * t + phi)


# ============================================================================
# PART 4: COMPLETE 3-NODE, 7-EDGE GRAPH WITH ALL REPRESENTATIONS
# ============================================================================

class ThreeNodeSevenEdgeGraph:
    """
    3-node, 7-edge graph with:
    - 3×3 matrix at each node
    - 4 transition arrows per edge
    - 6-vertex, 28-arrow wavelet quiver for the entire system
    """
    
    def __init__(self):
        self.nodes = [0, 1, 2]
        
        # Edges: 7 directed edges
        self.edges = [
            (0, 1), (1, 0),  # bidirectional 0-1
            (0, 2), (2, 0),  # bidirectional 0-2
            (1, 2), (2, 1),  # bidirectional 1-2
            (1, 1)           # self-loop at node 1
        ]
        
        # Store 3×3 matrices for each node
        self.node_matrices: Dict[int, ThreeByThreeMatrix] = {}
        
        # Store transition arrows for each edge
        self.edge_arrows: Dict[Tuple[int, int], TransitionArrows] = {}
        
        # Store wavelet quiver for each node
        self.node_wavelet_quivers: Dict[int, WaveletParameterQuiver] = {}
        
        self._initialize_representations()
    
    def _initialize_representations(self):
        """Initialize all representations with example values"""
        
        # Initialize 3×3 matrices for each node
        # Node 0 (injection site)
        self.node_matrices[0] = ThreeByThreeMatrix(
            a=0.6, b=0.2, c=0.05,
            d=0.1, e=0.5, f=0.05,
            g=0.02, h=0.1, i=0.3
        )
        
        # Node 1 (hub with recirculation)
        self.node_matrices[1] = ThreeByThreeMatrix(
            a=0.5, b=0.25, c=0.05,
            d=0.15, e=0.45, f=0.05,
            g=0.03, h=0.12, i=0.35
        )
        
        # Node 2 (distal region)
        self.node_matrices[2] = ThreeByThreeMatrix(
            a=0.55, b=0.22, c=0.04,
            d=0.12, e=0.48, f=0.04,
            g=0.02, h=0.11, i=0.32
        )
        
        # Initialize transition arrows for each edge
        for edge in self.edges:
            src, tgt = edge
            
            # Different rates based on edge type
            if src == tgt:  # self-loop
                arrows = TransitionArrows(kappa_HH=0.4, kappa_HT=0.3, kappa_TH=0.2, kappa_TT=0.1)
            elif (src, tgt) in [(0,1), (1,0)]:  # thick edges (rapid flow)
                arrows = TransitionArrows(kappa_HH=0.35, kappa_HT=0.35, kappa_TH=0.15, kappa_TT=0.15)
            elif (src, tgt) in [(0,2), (2,0)]:  # thin edges (slow flow)
                arrows = TransitionArrows(kappa_HH=0.25, kappa_HT=0.25, kappa_TH=0.25, kappa_TT=0.25)
            else:  # medium edges
                arrows = TransitionArrows(kappa_HH=0.3, kappa_HT=0.3, kappa_TH=0.2, kappa_TT=0.2)
            
            self.edge_arrows[edge] = arrows
        
        # Initialize wavelet quivers for each node
        for node in self.nodes:
            self.node_wavelet_quivers[node] = WaveletParameterQuiver()
    
    def print_all_representations(self):
        """Print all representations for the 3-node, 7-edge graph"""
        
        print("\n" + "="*80)
        print("COMPLETE REPRESENTATION: 3-NODE, 7-EDGE GRAPH")
        print("="*80)
        
        # Print graph structure
        print("\nGRAPH STRUCTURE:")
        print(f"  Nodes: {self.nodes}")
        print(f"  Edges: {len(self.edges)} directed edges")
        for edge in self.edges:
            if edge[0] == edge[1]:
                print(f"    {edge[0]} → {edge[1]} (self-loop)")
            else:
                print(f"    {edge[0]} → {edge[1]}")
        
        # Print 3×3 matrices for each node
        print("\n" + "="*80)
        print("LEVEL 0: 3×3 MATRICES AT EACH NODE")
        print("="*80)
        
        for node in self.nodes:
            self.node_matrices[node].print_matrix(f"(Node {node})")
        
        # Print transition arrows for each edge
        print("\n" + "="*80)
        print("LEVEL 1: 4 TRANSITION ARROWS PER EDGE")
        print("="*80)
        
        for edge in self.edges:
            src, tgt = edge
            self.edge_arrows[edge].print_arrows(src, tgt)
        
        # Print wavelet quiver for one node (they are structurally identical)
        print("\n" + "="*80)
        print("LEVEL 2: 6-VERTEX, 28-ARROW WAVELET PARAMETER QUIVER")
        print("(Shown for Node 0 - identical structure at all nodes)")
        print("="*80)
        
        self.node_wavelet_quivers[0].print_quiver()
        
        # Show mapping from 4 arrows to 6 vertices
        print("\n" + "="*80)
        print("MAPPING: 4 ARROWS → 6 VERTICES (Wavelet Lift)")
        print("="*80)
        
        print("""
    The 4 transition arrows (HH, HT, TH, TT) lift to 6 wavelet parameters:
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │  HH Arrow (both H)        →  a (amplitude A), ω₁ (frequency A)         │
    │  HT Arrow (A=H, B=T)      →  b (amplitude B), φ (phase)                │
    │  TH Arrow (A=T, B=H)      →  ω₂ (frequency B), κ (coupling)            │
    │  TT Arrow (both T)        →  κ (coupling), φ (phase)                   │
    └─────────────────────────────────────────────────────────────────────────┘
    
    The 28 arrows in the wavelet quiver represent ALL interactions between these
    6 parameters, capturing the full dynamics of the two-molecule system.
        """)
        
        # Show how parameters reconstruct consciousness
        print("\n" + "="*80)
        print("RECONSTRUCTION: From 6 Vertices to Consciousness Signal")
        print("="*80)
        
        # Example state evolution
        state = np.array([0.5, 0.3, 1.0, 2.0, 0.5, 0.2])
        
        print(f"\nInitial quiver state: (a={state[0]:.2f}, b={state[1]:.2f}, ω₁={state[2]:.2f}, ω₂={state[3]:.2f}, φ={state[4]:.2f}, κ={state[5]:.2f})")
        
        # Evolve over time
        dt = 0.1
        t_values = np.arange(0, 2, dt)
        signals = []
        
        for t in t_values:
            signal = self.node_wavelet_quivers[0].reconstruct_signal(state, t)
            signals.append(signal)
            state = self.node_wavelet_quivers[0].evolve_state(state, dt)
        
        print(f"\nEvolved signal at t=0: {signals[0]:.4f}")
        print(f"Evolved signal at t=1: {signals[10]:.4f}")
        print(f"Evolved signal at t=2: {signals[-1]:.4f}")
        
        # Show projection to Gr(2,4)
        state = np.array([0.5, 0.3, 1.0, 2.0, 0.5, 0.2])
        plucker = self.node_wavelet_quivers[0].project_to_plucker(state)
        
        print(f"\nProjection to Gr(2,4) Plücker coordinates:")
        print(f"  [p₁₂, p₁₃, p₁₄, p₂₃, p₂₄, p₃₄] = [{plucker[0]:.4f}, {plucker[1]:.4f}, {plucker[2]:.4f}, {plucker[3]:.4f}, {plucker[4]:.4f}, {plucker[5]:.4f}]")
        
        # Verify Plücker relation
        plucker_rel = plucker[0]*plucker[5] - plucker[1]*plucker[4] + plucker[2]*plucker[3]
        print(f"\nPlücker relation (should be 0 on Klein quadric): {plucker_rel:.6f}")
    
    def print_quiver_statistics(self):
        """Print comprehensive statistics about the quiver representations"""
        
        print("\n" + "="*80)
        print("QUIVER STATISTICS")
        print("="*80)
        
        # Level 1: 4-arrow quiver per edge
        total_arrows_level1 = len(self.edges) * 4
        print(f"\nLEVEL 1 (Molecular Transport Quiver):")
        print(f"  Vertices: {len(self.nodes)} brain regions")
        print(f"  Edges: {len(self.edges)} directed edges")
        print(f"  Arrows per edge: 4 (HH, HT, TH, TT)")
        print(f"  Total arrows: {total_arrows_level1}")
        
        # Level 2: 6-vertex, 28-arrow quiver per node
        total_vertices_level2 = len(self.nodes) * 6
        total_arrows_level2 = len(self.nodes) * 28
        
        print(f"\nLEVEL 2 (Wavelet Parameter Quiver):")
        print(f"  Vertices per node: 6 (a, b, ω₁, ω₂, φ, κ)")
        print(f"  Total vertices: {total_vertices_level2}")
        print(f"  Arrows per node: 28")
        print(f"  Total arrows: {total_arrows_level2}")
        
        # Show hierarchy
        print(f"\nHIERARCHY SUMMARY:")
        print(f"  ┌─────────────────────────────────────────────────────────────┐")
        print(f"  │ 3×3 Matrix (2 minors + overlap) → 4 arrows per edge        │")
        print(f"  │        ↓                                                    │")
        print(f"  │ 4 arrows per edge → 6 vertices × 28 arrows per node        │")
        print(f"  │        ↓                                                    │")
        print(f"  │ 6 vertices × 28 arrows → Gr(2,4) Plücker coordinates       │")
        print(f"  │        ↓                                                    │")
        print(f"  │ Gr(2,4) → Consciousness Signal C(t)                        │")
        print(f"  └─────────────────────────────────────────────────────────────┘")
        
        print(f"\nMULTIPLICATION FACTOR:")
        print(f"  From 3×3 matrix to 4 arrows: ×{4}")
        print(f"  From 4 arrows to 6×28 quiver: ×{6*28/4:.0f}")
        print(f"  Total lift: {4 * (6*28/4):.0f}×")
        
        # Show the mapping algebraically
        print(f"\nMAPPING EQUATIONS:")
        print(f"""
    (3×3 matrix)                    (4 arrows)                    (6 vertices, 28 arrows)
    ┌───────────┐                  ┌───────────┐                  ┌───────────────────────┐
    │ a  b  c  │  4 arrows        │ HH: κ_HH  │  wavelet lift    │ a  = f_a(κ_HH, κ_HT)   │
    │ d  e  f  │ ────────►        │ HT: κ_HT  │ ─────────►       │ b  = f_b(κ_HT, κ_TH)   │
    │ g  h  i  │                  │ TH: κ_TH  │                  │ ω₁ = f_ω₁(κ_HH, κ_TH)  │
    └───────────┘                  │ TT: κ_TT  │                  │ ω₂ = f_ω₂(κ_HT, κ_TT)  │
                                   └───────────┘                  │ φ  = f_φ(κ_HH, κ_TT)   │
                                                                 │ κ  = f_κ(κ_HH, κ_TT)   │
                                                                 └───────────────────────┘
                                                                           │
                                                                           │ 28 arrows between
                                                                           │ these 6 vertices
                                                                           ▼
                                                                   ┌───────────────────┐
                                                                   │ 28 interactions:  │
                                                                   │ a↔b, a↔ω₁, ...    │
                                                                   └───────────────────┘
        """)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run the complete representation printing"""
    
    print("\n" + "="*80)
    print("QUIVER REPRESENTATION: FROM 3×3 MATRICES TO 6-VERTEX, 28-ARROW WAVELET QUIVER")
    print("For 3-Node, 7-Edge Graph")
    print("="*80)
    
    # Create graph
    graph = ThreeNodeSevenEdgeGraph()
    
    # Print all representations
    graph.print_all_representations()
    
    # Print statistics
    graph.print_quiver_statistics()
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)
    
    return graph


if __name__ == "__main__":
    graph = main()