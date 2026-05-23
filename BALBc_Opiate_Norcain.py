"""
COMPLETE 20‑PANEL DASHBOARD WITH GHOST SIGNAL & PRIME ZETA
==========================================================
- Aggressive flow ensures Node 1 & 2 become unconscious and are revived.
- Monodromy product verifies coherence (T₁ ∘ T₂ ∘ T₃ = I).
- Prime Zeta of Paths identifies prime paths (zeros on critical line).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import cwt, morlet, coherence, find_peaks
from scipy.signal import hilbert
import sklearn.cluster
from scipy.signal import hilbert
import warnings
import pandas as pd
from collections import deque
import subprocess
import json
import tempfile
import os
import ast
import re
import meshio
import pyvista as pv
import shutil
import time

BRAIN_FILE = "./nodes_edges_filtered_six.vtp"

NODES_FILE = "./node_regions_clean.csv"
EDGES_FILE = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"


warnings.filterwarnings('ignore')

JULIA_AINF_SCRIPT = "/Users/vaw1/Downloads/OGB/connectome/phaseTransition_phaseTransition_complex/curved_hh2_sparse_refactored_filteredA.jl"

# ── Two-phase A∞ computation strategy ────────────────────────────────────────
# Phase 1 (CURRENT):  flat A∞ with m0=0  (fast, collects baseline data)
#   - curved_hh2_sparse_refactored.jl runs WITHOUT filtration
#   - collect ainf_export_*.json for n=6, 7P, 7L, 8
#   - verify hinge chi_red=B_Ihara, Ramanujan, spectral rigidity
#
# Phase 2 (NEXT):     curved A∞ with m0≠0 + filtration
#   - FilteredAInfAlgebra(lambda=1.0, energy_cutoff=1e-8)
#   - m0_curvature encodes obstruction_deficit from blowup events
#   - prevents combinatorial blow-up in C4/C5/C6 enumeration
#   - bridges flat results to full MC equation b(mu)+1/2[mu,mu]=0
#
# Set AINF_PHASE=1 for current run (m0=0, no filtration)
# Set AINF_PHASE=2 after blowup_table.tsv is collected (m0≠0)
#AINF_PHASE = 2   # ← change to 2 after Phase 1 data is collected
AINF_PHASE = 1   # ← change to 2 after Phase 1 data is collected

# ── Graph type ────────────────────────────────────────────────────────────────
# Controls which connectome CSV files are loaded and which H1 cycles
# are used for Bridge B export in curved_hh2_sparse_refactored.jl
# Options: "Q_6" | "Q_7L" | "Q_7P" | "Q_8"
# Override with env var: CONNECTOME_GRAPH_TYPE=Q_7P python3 BALBc_Opiate_Norcain.py
GRAPH_TYPE = "Q_7P"   # ← change to Q_7P, Q_7L, or Q_8 for other graphs

# Filtration parameters for Phase 2
FILTRATION_LAMBDA      = 1.0     # exponential decay rate
FILTRATION_ENERGY_CUT  = 1e-8    # prune paths below this weight
FILTRATION_MAX_LEN     = 20      # max path length in C4/C5/C6
# curved_hh2_sparse_refactored_filteredA.jl reads filt_config.json to read these 
# values directly.
#M0_CURVATURE_SCALE     = 0.1     # scale factor: m0[v] = scale * obstruction_deficit[v]
#M0_CURVATURE_VALUE     = 1066176.0
M0_CURVATURE_SCALE     = 0.0     # scale factor: m0[v] = scale * obstruction_deficit[v]
M0_CURVATURE_VALUE     = 0.0

class MilnorSequestrator:
    """Isolates the 'Milnor Node' singular points during algebraic failure."""
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.singularities = []

    def isolate_node(self, t, consciousness, state_vector):
        # Detects the 'Snap' when consciousness drops below threshold
        if consciousness < self.threshold:
            self.singularities.append((t, state_vector))
            return True
        return False

class SiegelLock:
    """Maintains the Ghost Signal (Siegel Lock) for system coordination."""
    def __init__(self, anchor_value=0.6):
        self.anchor = anchor_value

    def apply_lock(self, qA, qB, C):
        # Maintains coordination even if the physical signal (C) vanishes
        coordination_strength = np.sqrt(qA**2 + qB**2)
        if C < 0.3:
            # System is in 'Ghost' mode; use the Siegel anchor
            return self.anchor 
        return coordination_strength

class FullGraphDynamics:
    """Full molecular dynamics with extreme flow to ensure recovery."""
    
    def __init__(self):
        # Load region graph
        # Graph type is controlled by GRAPH_TYPE constant (set at top of file)
        # or via environment variable CONNECTOME_GRAPH_TYPE
        import os as _os
        _gtype = _os.environ.get("CONNECTOME_GRAPH_TYPE", GRAPH_TYPE)
        _csv_map = {
            "Q_6":  ("regions_six.csv",           "region_edges_six.csv"),
            "Q_7L": ("regions_six_ANDLSX.csv",    "region_edges_six_ANDLSX.csv"),
            "Q_7P": ("regions_six_ANDPAL.csv",    "region_edges_six_ANDPAL.csv"),
            "Q_8":  ("regions_six_ANDPALLSX.csv", "region_edges_six_ANDPALLSX.csv"),
        }
        _regions_file, _edges_file = _csv_map.get(_gtype, _csv_map["Q_6"])
        print(f"  [Graph] Loading {_gtype}: {_regions_file}, {_edges_file}")
        regions_df = pd.read_csv(_regions_file)
        edges_df = pd.read_csv(_edges_file)
        self.graph_type = _gtype  # stored for Julia ainf export
        self.nodes = list(regions_df['index'])
        self.n_nodes = len(self.nodes)
        self.region_names = regions_df['region'].tolist()
        self.region_centroids = regions_df[['centroid_x','centroid_y','centroid_z']].values

        # Build edge list and edge_weights
        # Force integer conversion
        self.edges = [(int(row.src_idx), int(row.tgt_idx)) for _, row in edges_df.iterrows()]
        #self.edges = [(row.src_idx, row.tgt_idx) for _, row in edges_df.iterrows()]
        self.edge_weights = {(row.src_idx, row.tgt_idx): row.weight for _, row in edges_df.iterrows()}

        # Add self‑loops (optional, for recirculation)
        for i in range(self.n_nodes):
            if (i, i) not in self.edge_weights:
                self.edge_weights[(i, i)] = 1.0
                self.edges.append((i, i))




        # Our A infinity algebra based HH2, m2, m3, m4, m5, m6 extractor.
        self.julia_ainf_script = JULIA_AINF_SCRIPT
        self.toda_epsilon = []
        self.toda_lambda = []
        self.toda_nu = []
        self.prolate_ratio = []
        """
        self.nodes = [0, 1, 2, 3, 4, 5]
        self.edges = [
            # Bidirectional CA1sp ↔ HPF
            (0, 3), (3, 0),
            # Bidirectional BLA ↔ LA
            (1, 5), (5, 1),
            # Bidirectional BLA ↔ sAMY
            (1, 4), (4, 1),
            # Bidirectional CA1sp ↔ sAMY
            (0, 4), (4, 0),
            # HPF → BLA, HPF → sAMY
            (3, 1), (3, 4),
            # LA → sAMY, sAMY → LA
            (5, 4), (4, 5),
            # HY ↔ sAMY
            (2, 4), (4, 2),
            # Self‑loops for all nodes (local recirculation)
            (0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5)
        ]

        # Synthetic edge weights (directed; for self‑loops use 2.0 as in original)
        self.edge_weights = {
            # CA1sp ↔ HPF
            (0, 3): 15.0,   # strong projection
            (3, 0):  5.0,
            # BLA ↔ LA
            (1, 5):  8.0,
            (5, 1):  6.0,
            # BLA ↔ sAMY
            (1, 4): 12.0,
            (4, 1): 10.0,
            # CA1sp ↔ sAMY
            (0, 4):  9.0,
            (4, 0):  7.0,
            # HPF → BLA, HPF → sAMY
            (3, 1):  4.0,
            (3, 4):  6.0,
            # LA → sAMY, sAMY → LA
            (5, 4):  5.0,
            (4, 5):  5.0,
            # HY ↔ sAMY
            (2, 4):  3.0,
            (4, 2):  8.0,
            # Self‑loops
            (0, 0):  2.0,
            (1, 1):  2.0,
            (2, 2):  2.0,
            (3, 3):  2.0,
            (4, 4):  2.0,
            (5, 5):  2.0,
        }

        self.n_nodes = len(self.nodes)
        # Threshold, Milnor, Siegel etc. 
        #self.nodes = [0, 1, 2]
        #self.edges = [
        #    (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (1, 1)
        #]
        """
        # 2. DEFINE THRESHOLD HERE (Move this up!)
        self.threshold = 0.3  
        
        # 3. Now initialize Milnor and Siegel using the threshold
        self.milnor = MilnorSequestrator(threshold=self.threshold)
        self.siegel = SiegelLock(anchor_value=0.6)
        
        # 4. Initialize other tracking arrays
        self.ghost_signal = None
        
        # Molecule parameters
        self.half_life_A = 6.0
        self.half_life_B = 3.0   # Very short norcain half‑life (acts fast, decays quickly)
        self.delay_A = 0.1
        self.delay_B = 0.2
        
        # Pharmacodynamics – norcain extremely effective
        self.alpha = 3.0
        self.beta = 1.0
        self.EC50_A = 0.2
        self.EC50_B = 0.2
        self.hill_A = 2.0
        self.hill_B = 2.0
        
        self.threshold = 0.2
        self.dose_times = [2.5, 6.0, 9.5]
        self.dose_amount = 4.0   # Large dose
        self.sigma_scale = 1.0
        
        self.t = None
        self.C = None
        self.qA = None
        self.qB = None
        self.HH1 = None
        self.HH2 = None
        self.plucker = None
        self.reverse_trajectories = []
        self.m2 = {}   # will be built dynamically
        self.m3 = {}
        self.m4 = {}
        self.m5 = {}
        self.m6 = {}
        # Phase tracking
        self.ainf_phase          = AINF_PHASE   # 1=flat, 2=curved+filtered
        self.m0_curvature        = {}            # vertex->float, zero in Phase 1
        self.obstruction_deficit = {}            # vertex->float, from blowup events
        self.filtration_active   = False         # set True when Phase 2 starts
        # Phase 1 baseline data (collected during run, used to init Phase 2)
        self.phase1_blowup_deficits = []         # obstruction_deficit per blowup
        self.phase1_rho_ihara       = None       # spectral radius from Phase 1
        self.epsilon = 1.0
        self.lambda_val = 0.1
        self.nu = 0.0

        self.qA_free = None
        self.qA_trap = None
        self.qB_free = None
        self.qB_trap = None
        self.node_coords = self.region_centroids

        self.loopy_nodes = {0, 2}   # CA1 and BLA -- use {} set vs  list []

        # Trapping rates (free → trap) based on Renkin‑Crone: higher for CA1 (thinner? adjust)
        self.alpha_in_A = {0: 0.15, 2: 0.12}
        self.alpha_out_A = {0: 0.015, 2: 0.012}   # release 10× slower

        self.alpha_in_B = {0: 0.15, 2: 0.12}
        self.alpha_out_B = {0: 0.015, 2: 0.012}

        self.plucker = None
        self.spectral_gap = None
        self.annihilator = []
        self.support = []
        self.prime_ideals = []
        self.plucker_history = []      # list of (time, step_index, [q12, q13, q14, q23, q24, q34])
        self.recompute_step_indices = []   # step numbers where A∞ was recomputed
        self.prime_zeta_values = []
        self.prime_zeta_times = []
        self.plucker_zeta_times = []    # dense time stamps
        self.plucker_zeta_mags = []     # dense magnitudes
        self.json_export_step_indices = []   # one entry per JSON export, in order
        self.gr24_result = None
        self.gr24_frames = []          # per-step SchoperFrame objects
        self.gr24_blowup_log = []      # wall crossings detected by gr24_step
        self.schubert_history = []     # per-step Schubert cell data from schubert_cell_spectrum
        self.singularities = []        # singularity events
        self.prolate_theta = []        # prolate spheroid theta values
        self.prolate_theta_times = []  # times for prolate theta
        self.hh2_global_median = 1.0   # pendant nodes don't affect the spectrum but they do concentrate obstruction.

    def compute_single_zeta(self, t, plucker_vec):
        """Compute a simple zeta magnitude from the current Plücker vector."""
        # Use the magnitude of the Plücker vector as a proxy
        return np.linalg.norm(plucker_vec)
    
    def flow_rate(self, edge, t, molecule='B'):
        heartbeat = 1 + 0.3 * np.sin(2 * np.pi * t)
        weight = self.edge_weights.get(edge, 1.0)
        base_flow = 4.0 if molecule == 'A' else 6.0   # Norcain flows much faster
        return base_flow * weight * heartbeat
    
    def renkin_crone_factor(self, radius_ratio):
        if radius_ratio >= 1:
            return 0
        return (1 - radius_ratio)**2 * (1 - 2.104*radius_ratio + 2.09*radius_ratio**3 - 0.95*radius_ratio**5)
    
    def transition_rate(self, edge, t, molecule):
        flow = self.flow_rate(edge, t, molecule)
        radius_A, radius_B = 0.5e-9, 0.8e-9
        pore_radius = 1.0e-9
        radius = radius_A if molecule == 'A' else radius_B
        radius_ratio = radius / pore_radius
        renkin_factor = self.renkin_crone_factor(radius_ratio)
        diffusivity = 5e-10 if molecule == 'A' else 8e-10
        membrane_thickness = 1e-6
        permeability = (diffusivity / membrane_thickness) * renkin_factor
        return flow * permeability * 300
    
    def load_brain_mesh(self, vtp_file):
        if not os.path.exists(vtp_file):
            raise FileNotFoundError(f"Mesh file not found: {vtp_file}")
        mesh = pv.read(vtp_file)
        self.mesh_points = mesh.points
        self.mesh_cells = mesh.lines  # or mesh.cells if lines are stored differently
        # For unstructured grid, you might need to convert cells to meshio format.
        # Simpler: store the whole mesh object and use pyvista to write later.
        self.pv_mesh = mesh
        # Extract region_id point data
        if 'region_id' not in self.pv_mesh.point_data:
            raise ValueError("VTP file missing 'region_id' point data")
        self.mesh_region_ids = mesh.point_data['region_id'].astype(int)
    
    def load_brain_mesh_meshio(self, vtp_file):
        mesh = meshio.read(vtp_file)
        # Extract point coordinates and region_id array
        self.mesh_points = mesh.points
        self.mesh_cells = mesh.cells
        # Find the point data array that stores region_id
        region_id_array = mesh.point_data.get('region_id')
        if region_id_array is None:
            raise ValueError("VTP file missing 'region_id' point data")
        self.mesh_region_ids = region_id_array.astype(int)
        # Also store cell data if needed (e.g., edge_length)
        self.mesh_cell_data = mesh.cell_data

    def schubert_cell_spectrum(self, L, t):
        """
        Project the 3x3 Lax matrix onto Gr(2,4) Schubert cell structure
        and read off the local spectral hint from quantum cohomology of Gr(2,4).

        The two 2x2 corner minors of L carry the Schubert cell coordinates.
        The quantum cohomology eigenvalues of Gr(2,4) are 4th roots of unity
        scaled by sqrt(2), giving the ghost signal norm 2*sqrt(2).

        Returns dict with:
          - schubert_stratum: which cell (0-4) the current state occupies
          - qc_eigenvalue: nearest quantum cohomology eigenvalue
          - ihara_prediction: predicted Ihara spectral radius at next blowup
          - minor_top: top-left 2x2 minor (p12 coordinate)
          - minor_bot: bottom-right 2x2 minor (p34 coordinate)
          - ghost_proximity: distance to 2*sqrt(2) ghost signal
        """
        L = np.array(L, dtype=complex)

        # Extract the two 2x2 corner minors
        top = L[:2, :2]
        bot = L[1:,  1:]
        minor_top = np.linalg.det(top)  # p12 analog
        minor_bot = np.linalg.det(bot)  # p34 analog
        m22_pivot = L[1, 1]             # Schubert glue = b (sAMY)

        # Identify Schubert stratum:
        #   stratum 4 (open cell):  minor_top ≠ 0 AND minor_bot ≠ 0
        #   stratum 3:              minor_top ≠ 0 AND minor_bot = 0
        #   stratum 2 (boundary):   minor_top = 0 AND minor_bot ≠ 0
        #   stratum 1:              pivot ≠ 0, both minors near 0
        #   stratum 0 (basepoint):  all near 0
        tol = 1e-6
        if abs(minor_top) > tol and abs(minor_bot) > tol:
            stratum = 4
        elif abs(minor_top) > tol:
            stratum = 3
        elif abs(minor_bot) > tol:
            stratum = 2
        elif abs(m22_pivot) > tol:
            stratum = 1
        else:
            stratum = 0

        # Quantum cohomology eigenvalues of Gr(2,4): 4th roots of ±1, radius sqrt(2)
        # These are the eigenvalues of c1* acting on H*(Gr(2,4))
        qc_eigenvalues = np.array([
             np.sqrt(2) * np.exp(1j * k * np.pi / 2) for k in range(4)
        ])  # {sqrt(2), i*sqrt(2), -sqrt(2), -i*sqrt(2)}

        # Map current state to Plucker coords
        a    = np.real(L[0, 0])
        b    = np.real(L[1, 1])
        phi  = np.real(L[0, 1])
        kappa= np.real(L[0, 2])
        w12  = np.real(L[1, 2])

        # Effective Plucker phase angle from the two dominant minors
        phase_top = np.angle(complex(np.real(minor_top), np.imag(minor_top)))
        phase_bot = np.angle(complex(np.real(minor_bot), np.imag(minor_bot)))
        plucker_phase = 0.5 * (phase_top + phase_bot)  # average phase

        # Find nearest QC eigenvalue by phase distance
        phase_diffs = [abs(np.angle(qc) - plucker_phase) for qc in qc_eigenvalues]
        nearest_idx = int(np.argmin(phase_diffs))
        qc_nearest  = qc_eigenvalues[nearest_idx]

        # Ihara prediction: |qc_nearest| / sqrt(q) where q depends on stratum
        q_eff = {4: 5, 3: 4, 2: 3, 1: 2, 0: 1}.get(stratum, 5)
        ihara_pred = abs(qc_nearest) / np.sqrt(q_eff)

        # Ghost signal: norm of SO(4) monodromy should approach 2*sqrt(2)
        # at a wall crossing; measure current proximity
        from scipy.linalg import expm
        Omega = np.array([
            [ 0,   phi,  kappa, 0   ],
            [-phi,  0,   w12,   0   ],
            [-kappa,-w12, 0,    0   ],
            [ 0,    0,   0,     0   ]
        ])
        nf = np.linalg.norm(Omega, 'fro') / np.sqrt(2)
        ghost_target = 2 * np.sqrt(2)
        ghost_proximity = abs(nf - ghost_target)

        return {
            "schubert_stratum":   stratum,
            "qc_eigenvalue":      qc_nearest,
            "qc_phase_idx":       nearest_idx,
            "ihara_prediction":   ihara_pred,
            "minor_top":          minor_top,
            "minor_bot":          minor_bot,
            "m22_pivot":          m22_pivot,
            "plucker_phase":      plucker_phase,
            "q_eff":              q_eff,
            "ghost_proximity":    ghost_proximity,
            "ghost_target":       ghost_target,
            "at_wall_crossing":   ghost_proximity < 0.1,
        }
    
    def compute_toda_parameters(self, history_window, dt):
        """
        history_window: list of (t, state_matrix) where state_matrix is (n_regions, 3)
        Returns (epsilon, lambda, nu)
        """
        if len(history_window) < 10:
            return 1.0, 0.1, 0.0
        
        # Stack states into arrays: (n_times, n_regions, 3)
        times = np.array([h[0] for h in history_window])
        states = np.array([h[1] for h in history_window])  # shape (T, N, 3)
        T, N, _ = states.shape
        
        # Use consciousness (C) as the main signal (index 2)
        C_signals = states[:, :, 2]  # (T, N)
        
        # Compute pairwise correlations (over time) for epsilon
        corr_matrix = np.corrcoef(C_signals.T)  # (N, N)
        # Average off‑diagonal correlations
        triu_indices = np.triu_indices(N, k=1)
        epsilon = np.mean(corr_matrix[triu_indices])
        
        # Compute variance decay rate for lambda
        variances = np.var(C_signals, axis=0)  # (N,)
        # Fit exponential decay over time? Simpler: lambda = mean reversion rate
        # Use the average of negative derivative of log(variance) over the window
        # For simplicity, compute from the first and last quarter
        quarter = T // 4
        var_start = np.mean(variances[:quarter])
        var_end = np.mean(variances[-quarter:])
        dt_total = times[-1] - times[0]
        if var_start > 0 and dt_total > 0:
            lambda_val = -np.log(var_end / var_start) / dt_total
        else:
            lambda_val = 0.1
        lambda_val = np.clip(lambda_val, 0.01, 1.0)
        
        # Compute instantaneous frequency nu via Hilbert transform on mean signal
        mean_C = np.mean(C_signals, axis=1)  # (T,)
        analytic = hilbert(mean_C - np.mean(mean_C))
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2 * np.pi * dt)
        # Average over last half of the window
        nu = np.mean(inst_freq[-len(inst_freq)//2:]) if len(inst_freq) > 0 else 0.0
        # Clamp to plausible range (0–20 Hz)
        nu = np.clip(nu, 0.0, 20.0)
        
        return epsilon, lambda_val, nu
    
    def prolate_operator(self, states_window):
        """
        states_window: list of (t, state_matrix) over a time window (e.g., last 2 seconds)
        Returns:
            principal_eigenvalue_ratio, principal_eigenvector
        """
        if len(states_window) < 5:
            return 0.0, np.ones(self.n_nodes) / self.n_nodes
        
        # Stack consciousness signals (or full state) into a matrix (T, N)
        C_mat = np.array([h[1][:, 2] for h in states_window])  # (T, N)
        # Compute covariance matrix (N x N)
        cov = np.cov(C_mat.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Sort descending
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        ratio = eigvals[0] / (np.sum(eigvals) + 1e-12)
        return ratio, eigvecs[:, 0]
    
    def compute_hochschild_invariants(self, t_idx):
        if t_idx < 2:
            return 0, 0
        C_cur = self.C[:, t_idx]
        C_prev = self.C[:, t_idx - 1]
        qB_cur = self.qB[:, t_idx]
        dt_val = self.t[t_idx] - self.t[t_idx - 1]
        dC_dt = (C_cur - C_prev) / dt_val
        HH1_val = np.linalg.norm(dC_dt)
        if t_idx > 2:
            C_prev2 = self.C[:, t_idx - 2]
            d2C_dt2 = (C_cur - 2*C_prev + C_prev2) / (dt_val**2)
            HH2_val = np.linalg.norm(d2C_dt2)
        else:
            HH2_val = 0
        # Non‑commutativity (Gerstenhaber bracket analog)
        comm = 0
        for i in range(3):
            for j in range(i+1, 3):
                comm += abs(C_cur[i] * qB_cur[j] - C_cur[j] * qB_cur[i])
        HH2_val += 0.3 * comm
        return HH1_val, HH2_val
    
    # Implements Opiate, Norcain dynamics -- In loopy structure, there are thin curves, 
    #                                        where molecules get trapped, get absorbed
    #                                        in brain slowly and then trigger rebound 
    #                                        effect.
    
    def deterministic_rates_loopy(self, node, qA_free_cur, qA_trap_cur, qB_free_cur, qB_trap_cur,
                        C_cur, t, dt_step, i, lambda_A, lambda_B, incoming, outgoing,
                        loopy=False):
        """
        Compute deterministic changes for a single node over dt_step.
        For loopy nodes, uses free + trap compartments; for normal nodes, trap = 0.
        Returns (dqA_free, dqA_trap, dqB_free, dqB_trap, dC)
        """
        # For non‑loopy nodes, treat trap as zero and ignore trap dynamics
        if not loopy:
            qA_trap_cur = 0.0
            qB_trap_cur = 0.0

        # ---- Inflow from incoming edges (uses free concentration of neighbours) ----
        inflow_A_free = 0.0
        inflow_B_free = 0.0
        for (src, tgt) in incoming[node]:
            delay_A = self.delay_A if src != node else 0
            delay_B = self.delay_B if src != node else 0
            idx_A = max(0, i - int(delay_A / dt_step))
            idx_B = max(0, i - int(delay_B / dt_step))
            rate_A = self.transition_rate((src, node), t, 'A')
            rate_B = self.transition_rate((src, node), t, 'B')
            # Explicitly convert to Python int (safeguard against numpy types)
            src_int = int(src)
            idx_A_int = int(idx_A)
            idx_B_int = int(idx_B)
            inflow_A_free += rate_A * self.qA_free[src_int, idx_A_int] * dt_step
            inflow_B_free += rate_B * self.qB_free[src_int, idx_B_int] * dt_step

        # ---- Outflow from outgoing edges (uses free concentration of this node) ----
        outflow_A_free = 0.0
        outflow_B_free = 0.0
        for (src, tgt) in outgoing[node]:   # src == node
            rate_A = self.transition_rate((node, tgt), t, 'A')
            rate_B = self.transition_rate((node, tgt), t, 'B')
            outflow_A_free += rate_A * qA_free_cur * dt_step
            outflow_B_free += rate_B * qB_free_cur * dt_step

        # ---- Free compartment changes (diffusion + decay + trapping/release) ----
        dqA_free = inflow_A_free - outflow_A_free - qA_free_cur * lambda_A * dt_step
        dqB_free = inflow_B_free - outflow_B_free - qB_free_cur * lambda_B * dt_step

        if loopy:
            # Trapping and release parameters (could be region‑specific)
            alpha_in_A = self.alpha_in_A.get(node, 0.1)
            alpha_out_A = self.alpha_out_A.get(node, 0.01)
            alpha_in_B = self.alpha_in_B.get(node, 0.1)
            alpha_out_B = self.alpha_out_B.get(node, 0.01)

            dqA_free -= alpha_in_A * qA_free_cur * dt_step
            dqA_free += alpha_out_A * qA_trap_cur * dt_step

            dqB_free -= alpha_in_B * qB_free_cur * dt_step
            dqB_free += alpha_out_B * qB_trap_cur * dt_step

            dqA_trap = alpha_in_A * qA_free_cur * dt_step - alpha_out_A * qA_trap_cur * dt_step
            dqB_trap = alpha_in_B * qB_free_cur * dt_step - alpha_out_B * qB_trap_cur * dt_step
        else:
            dqA_trap = 0.0
            dqB_trap = 0.0

        # ---- Consciousness dynamics (uses free concentrations only) ----
        activation = self.alpha * (1 - C_cur) * (qB_free_cur**self.hill_B) / (self.EC50_B**self.hill_B + qB_free_cur**self.hill_B)
        dampening = self.beta * C_cur * (qA_free_cur**self.hill_A) / (self.EC50_A**self.hill_A + qA_free_cur**self.hill_A)
        oscillation = 0.1 * np.sin(2 * np.pi * 0.6 * t) * (1 - C_cur)
        dC = (activation - dampening + oscillation) * dt_step

        return dqA_free, dqA_trap, dqB_free, dqB_trap, dC

    def deterministic_rates(self, node, qA_cur, qB_cur, C_cur, t, dt_step, i,
                        lambda_A, lambda_B, incoming, outgoing):
        """
        Compute deterministic changes for a single node over dt_step.
        Returns (dqA, dqB, dC) – the *changes* (not per‑second rates).
        """
        # Inflow from incoming edges
        inflow_A = 0.0
        inflow_B = 0.0
        for (src, tgt) in incoming[node]:
            delay_A = self.delay_A if src != node else 0
            delay_B = self.delay_B if src != node else 0
            idx_A = max(0, i - int(delay_A / dt_step))
            idx_B = max(0, i - int(delay_B / dt_step))
            rate_A = self.transition_rate((src, node), t, 'A')
            rate_B = self.transition_rate((src, node), t, 'B')
            inflow_A += rate_A * self.qA[src, idx_A] * dt_step
            inflow_B += rate_B * self.qB[src, idx_B] * dt_step
        
        # Outflow from outgoing edges
        outflow_A = 0.0
        outflow_B = 0.0
        for (src, tgt) in outgoing[node]:   # src == node
            rate_A = self.transition_rate((node, tgt), t, 'A')
            rate_B = self.transition_rate((node, tgt), t, 'B')
            outflow_A += rate_A * qA_cur * dt_step
            outflow_B += rate_B * qB_cur * dt_step
        
        dqA = inflow_A - outflow_A - qA_cur * lambda_A * dt_step
        dqB = inflow_B - outflow_B - qB_cur * lambda_B * dt_step
        
        # Consciousness dynamics
        activation = self.alpha * (1 - C_cur) * (qB_cur**self.hill_B) / (self.EC50_B**self.hill_B + qB_cur**self.hill_B)
        dampening = self.beta * C_cur * (qA_cur**self.hill_A) / (self.EC50_A**self.hill_A + qA_cur**self.hill_A)
        oscillation = 0.1 * np.sin(2 * np.pi * 0.6 * t) * (1 - C_cur)
        dC = (activation - dampening + oscillation) * dt_step
        
        return dqA, dqB, dC
    
    def build_lax_matrix(self, state_wavelet, mean_C):
        """
        state_wavelet : [a, b, w1, w2, phi, kappa]
        mean_C : float (average consciousness)
        Returns 3x3 symmetric matrix.
        """
        a, b, w1, w2, phi, kappa = state_wavelet
        # Use a and b as diagonal elements, phi and kappa as off‑diagonals
        L = np.array([
            [a, phi, kappa],
            [phi, b, 0.5*(w1 + w2)],
            [kappa, 0.5*(w1 + w2), mean_C]
        ])
        # Ensure symmetry
        L = 0.5 * (L + L.T)
        return L
    
    # Call Julia for HH2
    def call_julia_ainf(self, current_weights, region_name=None, step_index=None, filt_config_path=None):
        """
        current_weights : dict mapping (u,v) -> float
        Returns (m3, m4, m5, m6, HH2_dim, prime_paths, gerstenhaber, cup_product, annihilator, support, prime_ideals)
        """
        # Build base weights dictionary
        weights_dict = {f"{int(u)}->{int(v)}": w for (u,v), w in current_weights.items()}
        
        if region_name is not None:
            region_idx = self.region_names.index(region_name)
            cx, cy, cz = self.region_centroids[region_idx]
            weights_dict["seed_region"] = region_name
            weights_dict["centroid_x"] = cx
            weights_dict["centroid_y"] = cy
            weights_dict["centroid_z"] = cz
            mode = "--full"
            extra_args = [region_name]
        else:
            mode = "--ainf-only"
            extra_args = []
        
        # Write weights to temporary JSON file
        weights_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump(weights_dict, weights_file)
        weights_file.close()
        
        output_file = tempfile.NamedTemporaryFile(suffix='.json', delete=False)
        output_file.close()

        # Build command
        # Phase 2: append filtration config path if provided
        if filt_config_path is not None and filt_config_path not in extra_args:
            extra_args = list(extra_args) + [filt_config_path]
        # Pass graph_type as extra arg for Bridge B H1 export
        gt = getattr(self, 'graph_type', GRAPH_TYPE)
        if gt != "Q_6" and gt not in extra_args:
            extra_args = list(extra_args) + [gt]
        cmd = ["julia", self.julia_ainf_script, mode, weights_file.name, output_file.name] + extra_args
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            print("Julia command:", cmd)
            print("Julia error (stderr):")
            print(e.stderr)
            print("Julia error (stdout):")
            print(e.stdout)
            raise

        # Read output JSON
        with open(output_file.name, 'r') as f:
            data = json.load(f)

        # Save a permanent copy (for debugging/plots)
        permanent_file = f"ainf_export_{region_name}_{time.time():.2f}.json" if region_name else f"ainf_export_{time.time():.2f}.json"
        shutil.copy(output_file.name, permanent_file)
        print(f"Saved A∞ data to {permanent_file}")

        # Record step index for alignment (must be done after successful read)
        if step_index is not None:
            self.json_export_step_indices.append(step_index)

        # Clean up temporary files
        os.unlink(weights_file.name)
        os.unlink(output_file.name)

        # Parse results
        m3 = self._parse_ainf_dict(data.get("m3", {}))
        m4 = self._parse_ainf_dict(data.get("m4", {}))
        m5 = self._parse_ainf_dict(data.get("m5", {}))
        m6 = self._parse_ainf_dict(data.get("m6", {}))
        HH2_dim = data.get("HH2_dim", 0)
        prime_paths = self._parse_prime_paths(data.get("prime_paths", []))
        gerstenhaber = data.get("gerstenhaber", [])
        cup_product = data.get("cup_product", [])
        
        annihilator = data.get("annihilator_infty", [])
        support = data.get("support_infty", [])
        prime_ideals = data.get("prime_higher_ideals", [])
        
        return m3, m4, m5, m6, HH2_dim, prime_paths, gerstenhaber, cup_product, annihilator, support, prime_ideals
    
    def _parse_ainf_key(self, key_str):
        """
        Convert a string like '(:f_CA1sp_HPF, :f_HPF_BLA, :e_BLA)' into a tuple
        of strings: ('f_CA1sp_HPF', 'f_HPF_BLA', 'e_BLA').
        """
        # Remove parentheses and split by comma
        key_str = key_str.strip('()')
        parts = key_str.split(',')
        result = []
        for p in parts:
            p = p.strip()
            # Remove leading ':' if present
            if p.startswith(':'):
                p = p[1:]
            result.append(p)
        return tuple(result)

    def _parse_ainf_dict(self, d):
        """Convert JSON dict with string keys to dict with tuple keys."""
        out = {}
        for key_str, val_dict in d.items():
            tup = self._parse_ainf_key(key_str)
            # val_dict maps target strings to floats
            parsed_val = {}
            for tgt_str, coeff in val_dict.items():
                tgt = self._parse_ainf_key(tgt_str)[0]  # single element tuple
                parsed_val[tgt] = coeff
            out[tup] = parsed_val
        return out

    def _parse_prime_paths(self, prime_paths_list):
        """Convert list of {'path': [...], 'weight': w} to internal list."""
        out = []
        for item in prime_paths_list:
            path = tuple(item['path'])
            weight = item['weight']
            out.append((path, weight))
        return out
    
    def toda_flow_step(self, L, dt, epsilon, lambda_val, nu, t):
        """
        L : 3x3 symmetric matrix
        dt : time step
        epsilon, lambda_val, nu : Toda parameters
        t : current time
        Returns updated L.
        """
        # Isospectral part: B = (L - L.T)/2 (skew‑symmetric)
        B = 0.5 * (L - L.T)
        dL_iso = epsilon * (B @ L - L @ B)
        # Damping: push off‑diagonals to zero
        dL_damp = -lambda_val * (L - np.diag(np.diag(L)))
        # Septal drive: periodic forcing on diagonal
        drive = nu * np.sin(2 * np.pi * 8.0 * t)   # 8 Hz theta
        dL_drive = drive * np.eye(3)
        dL = dL_iso + dL_damp + dL_drive
        L_new = L + dL * dt
        # Symmetrize
        L_new = 0.5 * (L_new + L_new.T)
        return L_new
    
    def toric_projection_to_Gr24(self, state, return_norm=False):
        """
        state : array of 6 floats [a, b, w1, w2, phi, kappa]
        Returns normalized Plücker coordinates (6‑vector) on Gr(2,4).
        """
        a, b, w1, w2, phi, kappa = state
        denom = w1 * w2
        if abs(denom) < 1e-12:
            return np.zeros(6)
        q12 = a * b * denom
        q13 = a * phi * denom
        q14 = a * kappa * denom
        q23 = b * phi * denom
        q24 = b * kappa * denom
        q34 = phi * kappa * denom
        q = np.array([q12, q13, q14, q23, q24, q34])
        norm = np.linalg.norm(q)
        print(f"Plucker Zeta raw norm = {norm}")
        if return_norm:
            return q, norm
        else:
            if norm > 0:
                q /= norm
            return q
    
    def compute_sheaf_Laplacian_eigenvalues(self, t, state):
        return self._sheafq.eigenvalues(t, state)


    def call_julia_ainf_filtered(self, current_weights, step_index=None):
        """
        Phase 2: curved A∞ with FilteredAInfAlgebra.

        Uses m0_curvature built from Phase 1 obstruction deficits.
        Passes filtration parameters to curved_hh2_sparse_refactored.jl
        via environment variables so the Julia script can activate
        FilteredAInfAlgebra(lambda=FILTRATION_LAMBDA, ...).

        Falls back to Phase 1 (flat) if filtration unavailable.
        """
        # Build m0_curvature from accumulated obstruction deficits
        m0_curv = {}
        for region, deficit in self.obstruction_deficit.items():
            m0_curv[region] = M0_CURVATURE_SCALE * float(deficit) ** 0.5
        self.m0_curvature = m0_curv

        # Pass filtration config to Julia via JSON sidecar
        import tempfile, json as _json
        filt_config = {
            "phase": 2,
            "lambda": FILTRATION_LAMBDA,
            "energy_cutoff": FILTRATION_ENERGY_CUT,
            "max_path_len": FILTRATION_MAX_LEN,
            "m0_curvature": m0_curv,
        }
        fd, filt_file = tempfile.mkstemp(suffix="_filt.json")
        import os
        os.close(fd)
        with open(filt_file, "w") as fp:
            _json.dump(filt_config, fp)

        print(f"    [Phase 2] m0_curvature: {m0_curv}")
        print(f"    [Phase 2] lambda={FILTRATION_LAMBDA} cutoff={FILTRATION_ENERGY_CUT}")

        # Call Julia with filtration config path as extra argument
        try:
            result = self.call_julia_ainf(current_weights, step_index=step_index,
                                          filt_config_path=filt_file)
            self.filtration_active = True
            return result
        except Exception as e:
            print(f"    [Phase 2] filtered call failed ({e}), falling back to Phase 1")
            return self.call_julia_ainf(current_weights, step_index=step_index)
        finally:
            try:
                os.remove(filt_file)
            except Exception:
                pass

    def recompute_ainf_from_julia(self, current_weights, output_json=None):
        """
        current_weights: dict mapping edge tuples (u,v) to float weight
        output_json: path to JSON file (if None, uses a temporary file)
        Returns a tuple (m3, m4, m5, m6) as dictionaries.
        """
        if output_json is None:
            fd, output_json = tempfile.mkstemp(suffix='.json')
            os.close(fd)
        
        # Write current_weights to a temporary JSON file for Julia
        weights_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
        json.dump({f"{u}->{v}": w for (u,v), w in current_weights.items()}, weights_file)
        weights_file.close()
        
        # Call Julia script
        julia_script = "path/to/your/export_ainf.jl"   # adjust
        cmd = ["julia", julia_script, weights_file.name, output_json]
        subprocess.run(cmd, check=True)
        
        # Load results
        with open(output_json, 'r') as f:
            data = json.load(f)
        
        # Convert string keys back to tuples of symbols (or strings)
        def str_to_tuple(s):
            # s is like "(:f_CA1sp_HPF, :f_HPF_BLA, :e_BLA)"
            # Remove parentheses and split by comma, then strip and remove ':'
            import re
            parts = re.findall(r':?([^,:]+)', s.strip('()'))
            return tuple(p.strip() for p in parts if p)
        
        m3 = {str_to_tuple(k): {str_to_tuple(tgt): v for tgt, v in d.items()}
            for k, d in data.get('m3', {}).items()}
        m4 = {str_to_tuple(k): {str_to_tuple(tgt): v for tgt, v in d.items()}
            for k, d in data.get('m4', {}).items()}
        m5 = {str_to_tuple(k): {str_to_tuple(tgt): v for tgt, v in d.items()}
            for k, d in data.get('m5', {}).items()}
        m6 = {str_to_tuple(k): {str_to_tuple(tgt): v for tgt, v in d.items()}
            for k, d in data.get('m6', {}).items()}
        
        # Clean up temporary files
        os.unlink(weights_file.name)
        if output_json != weights_file.name:  # only if we created it
            os.unlink(output_json)
        
        return m3, m4, m5, m6
    
    def write_vtu(self, step):
        """Write current simulation state as a VTK file using pyvista."""
        if not hasattr(self, 'pv_mesh'):
            return  # mesh not loaded

        # Work on a copy to avoid accumulating point data from previous steps
        mesh = self.pv_mesh.copy()

        n_points = mesh.n_points
        # Initialize arrays
        opiate = np.zeros(n_points, dtype=np.float32)
        norcain = np.zeros(n_points, dtype=np.float32)
        consciousness = np.zeros(n_points, dtype=np.float32)
        hh2 = np.full(n_points, self.HH2[step], dtype=np.float32)
        ghost = np.full(n_points, self.ghost_signal[step], dtype=np.float32)
        milnor = np.zeros(n_points, dtype=np.float32)

        # Region IDs from the mesh (must be integer array)
        region_ids = mesh.point_data['region_id'].astype(int)

        # Fill per‑region scalar values
        for reg_idx in range(self.n_nodes):
            mask = (region_ids == reg_idx)
            if not np.any(mask):
                continue
            opiate[mask] = self.qA[reg_idx, step]
            norcain[mask] = self.qB[reg_idx, step]
            consciousness[mask] = self.C[reg_idx, step]

        # Mark Milnor snap if it occurred at this time
        snap_times = [s[0] for s in self.milnor.singularities]
        if self.t[step] in snap_times:
            milnor[:] = 1.0

        # Assign point data
        mesh.point_data['opiate'] = opiate
        mesh.point_data['norcain'] = norcain
        mesh.point_data['consciousness'] = consciousness
        mesh.point_data['HH2'] = hh2
        mesh.point_data['ghost_signal'] = ghost
        mesh.point_data['milnor_snap'] = milnor

        # Save as VTK PolyData file (use .vtp extension)
        mesh.save(f"sim_step_{step:04d}.vtp")
    
    def write_vtu_meshio(self, step):
        if not hasattr(self, 'mesh_points'):
            return  # mesh not loaded
        # Build point data dictionaries for this time step
        point_data = {
            "opiate": np.zeros(len(self.mesh_points), dtype=np.float32),
            "norcain": np.zeros(len(self.mesh_points), dtype=np.float32),
            "consciousness": np.zeros(len(self.mesh_points), dtype=np.float32),
            "HH2": np.full(len(self.mesh_points), self.HH2[step], dtype=np.float32),
            "ghost_signal": np.full(len(self.mesh_points), self.ghost_signal[step], dtype=np.float32),
            "milnor_snap": np.zeros(len(self.mesh_points), dtype=np.float32),
        }
        # Map region index to value
        for reg_idx in range(self.n_nodes):
            mask = (self.mesh_region_ids == reg_idx)
            if not np.any(mask):
                continue
            point_data["opiate"][mask] = self.qA[reg_idx, step]
            point_data["norcain"][mask] = self.qB[reg_idx, step]
            point_data["consciousness"][mask] = self.C[reg_idx, step]
        # Mark Milnor snap if occurred at this time
        snap_times = [s[0] for s in self.milnor.singularities]
        if self.t[step] in snap_times:
            point_data["milnor_snap"][:] = 1.0
        # Write VTU file
        mesh_out = meshio.Mesh(
            points=self.mesh_points,
            cells=self.mesh_cells,
            point_data=point_data,
            cell_data=self.mesh_cell_data  # preserve original cell data (e.g., edge_length)
        )
        meshio.write(f"sim_step_{step:04d}.vtu", mesh_out)
    
    def write_vtu_centroid(self, step):
        if self.node_coords is None:
            return
        points = self.node_coords
        # Build connectivity from self.edges (already 0‑based indices)
        cells = [("line", [[u, v] for (u, v) in self.edges])]
        point_data = {
            "opiate": self.qA[:, step],
            "norcain": self.qB[:, step],
            "consciousness": self.C[:, step],
            "HH2": np.full(self.n_nodes, self.HH2[step]),
            "ghost_signal": np.full(self.n_nodes, self.ghost_signal[step]),
            "milnor_snap": np.zeros(self.n_nodes),
        }
        # Mark nodes where Milnor sequestration occurred at this time
        snap_times = [s[0] for s in self.milnor.singularities]
        if self.t[step] in snap_times:
            point_data["milnor_snap"] = np.ones(self.n_nodes)
        meshio.write_points_cells(f"sim_step_{step:04d}.vtu", points, cells, point_data=point_data)
    
    # ============================================================================
    # MODIFIED SIMULATION LOOP (inside FullGraphDynamics.simulate)
    # ============================================================================

    def simulate(self, t_span=(0, 25), dt=0.02, use_sde=True, use_toda_feedback=True,
                ainf_recompute_interval=50, history_duration=2.0):
        """
        Runs the full simulation with:
        - two‑compartment (free/trap) for loopy nodes,
        - SDE (Euler‑Maruyama) or deterministic,
        - Toda flow with parameters aggregated from history,
        - prolate eigenbasis lock,
        - periodic Julia calls for A∞ recomputation,
        - reverse Hironaka ancestor search and Rees blow‑up.
        """
        t = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t)
        self.t = t
        n_nodes = self.n_nodes
        self.C = np.zeros((n_nodes, n_steps))
        self.qA = np.zeros((self.n_nodes, n_steps))
        self.qB = np.zeros((self.n_nodes, n_steps)) 
        self.qA_free = np.zeros((n_nodes, n_steps))
        self.qA_trap = np.zeros((n_nodes, n_steps))
        self.qB_free = np.zeros((n_nodes, n_steps))
        self.qB_trap = np.zeros((n_nodes, n_steps))
        self.HH1 = np.zeros(n_steps)
        self.HH2 = np.zeros(n_steps)
        self.ghost_signal = np.zeros(n_steps)
        
        # Initial conditions: opiate only at node 0, free compartment
        self.qA_free[0, 0] = 4.5
        self.C[:, 0] = 1.0
        # Initially, all drug is free (no trapped)
        # Opiate (A): only free at node 0
        self.qA_free[0, 0] = 3.5
        self.qA_trap[0, 0] = 0.0
        # Norcain (B): none initially
        self.qB_free[0, 0] = 0.0
        self.qB_trap[0, 0] = 0.0
        
        lambda_A = np.log(2) / self.half_life_A
        lambda_B = np.log(2) / self.half_life_B
        
        # For storing Toda parameters over time
        self.toda_epsilon = np.zeros(n_steps)
        self.toda_lambda = np.zeros(n_steps)
        self.toda_nu = np.zeros(n_steps)
        self.prolate_ratio = np.zeros(n_steps)
        
        # Flag to indicate if we have locked the eigenbasis
        locked = False
        vtk_interval=10
        # Pre‑compute which edges are incident to each node (optimisation)
        outgoing = {node: [] for node in range(n_nodes)}
        incoming = {node: [] for node in range(n_nodes)}
        for (u,v) in self.edges:
            outgoing[u].append((u,v))
            incoming[v].append((u,v))
        
        # History buffer (deque of (time, state_matrix))
        from collections import deque
        history = deque(maxlen=int(history_duration / dt))

        # Lax matrix initialisation (3x3 identity)
        L = np.eye(3)

        # Create wavelet quiver and its spectral sheaf (once, before loop)
        self._quiver = DynamicWaveletQuiver(self)
        self._sheafq = QuiverSpectralSheaf(self._quiver)
        
        print("\n  Starting simulation with SDE mode =", use_sde)
        print("  A∞ recompute interval =", ainf_recompute_interval, "steps")

        # Gr(2,4) schober step function — import BEFORE the loop
        try:
            from gr24_schober_projection import gr24_step as _gr24_step_fn
            gr24_step = _gr24_step_fn
        except ImportError:
            gr24_step = lambda obj, i: None   # graceful fallback
        
        for i in range(n_steps - 1):
            dt_step = t[i+1] - t[i]
            # ------------------------------------------------------------
            # 1. Update each node (deterministic or SDE)
            # ------------------------------------------------------------
            for node in range(n_nodes):
                loopy = node in self.loopy_nodes

                # ---- Get current values ----
                if loopy:
                    qA_free_cur = self.qA_free[node, i]
                    qA_trap_cur = self.qA_trap[node, i]
                    qB_free_cur = self.qB_free[node, i]
                    qB_trap_cur = self.qB_trap[node, i]
                else:
                    # For non‑loopy nodes, treat trap as zero and use single array for free
                    qA_free_cur = self.qA[node, i]   # we store total in qA for simplicity
                    qA_trap_cur = 0.0
                    qB_free_cur = self.qB[node, i]
                    qB_trap_cur = 0.0
                C_cur = self.C[node, i]

                # ---- Compute deterministic changes ----
                dqA_free, dqA_trap, dqB_free, dqB_trap, dC = self.deterministic_rates_loopy(
                    node, qA_free_cur, qA_trap_cur, qB_free_cur, qB_trap_cur, C_cur,
                    t[i], dt_step, i, lambda_A, lambda_B, incoming, outgoing, loopy
                )

                # ---- Apply dose (only to free compartment of node 0, can be any node.) ----
                dose_extra = 0.0
                if node == 0:
                    for dose_time in self.dose_times:
                        if abs(t[i] - dose_time) < dt_step:
                            dose_extra += self.dose_amount

                # ---- SDE or deterministic update ----
                if use_sde:
                    # Add stochastic noise (multiplicative, scaled by sigma_scale)
                    sigma_free_A = 0.05 * qA_free_cur * self.sigma_scale
                    sigma_trap_A = 0.01 * qA_trap_cur * self.sigma_scale   # small noise in trap
                    sigma_free_B = 0.05 * qB_free_cur * self.sigma_scale
                    sigma_trap_B = 0.01 * qB_trap_cur * self.sigma_scale
                    sigma_C = 0.02 * (1 - C_cur) * self.sigma_scale
                    dW = np.random.normal(0, np.sqrt(dt_step), 5)   # 5 independent Wiener increments

                    qA_free_new = qA_free_cur + dqA_free + sigma_free_A * dW[0]
                    qA_trap_new = qA_trap_cur + dqA_trap + sigma_trap_A * dW[1]
                    qB_free_new = qB_free_cur + dqB_free + sigma_free_B * dW[2]
                    qB_trap_new = qB_trap_cur + dqB_trap + sigma_trap_B * dW[3]
                    C_new = C_cur + dC + sigma_C * dW[4]

                else:  # deterministic
                    qA_free_new = qA_free_cur + dqA_free
                    qA_trap_new = qA_trap_cur + dqA_trap
                    qB_free_new = qB_free_cur + dqB_free
                    qB_trap_new = qB_trap_cur + dqB_trap
                    C_new = C_cur + dC
                    
                # Add dose to free compartment of norcain
                qB_free_new += dose_extra

                # ---- Clipping ----
                qA_free_new = np.clip(qA_free_new, 0, 5)
                qA_trap_new = np.clip(qA_trap_new, 0, 5)
                qB_free_new = np.clip(qB_free_new, 0, 6)
                qB_trap_new = np.clip(qB_trap_new, 0, 6)
                C_new = np.clip(C_new, 0, 1)

                # ---- Store results ----
                if loopy:
                    self.qA_free[node, i+1] = qA_free_new
                    self.qA_trap[node, i+1] = qA_trap_new
                    self.qB_free[node, i+1] = qB_free_new
                    self.qB_trap[node, i+1] = qB_trap_new
                    # Total concentrations for logging / visualisation
                    self.qA[node, i+1] = qA_free_new + qA_trap_new
                    self.qB[node, i+1] = qB_free_new + qB_trap_new
                else:
                    # For non‑loopy nodes, 
                    self.qA_free[node, i+1] = qA_free_new
                    self.qA_trap[node, i+1] = 0.0
                    self.qB_free[node, i+1] = qB_free_new
                    self.qB_trap[node, i+1] = 0.0
                    self.qA[node, i+1] = qA_free_new
                    self.qB[node, i+1] = qB_free_new
                    

                self.C[node, i+1] = C_new

                # ---- Milnor sequestration (only node 0) ----
                if node == 0:
                    self.milnor.isolate_node(t[i+1], C_new, [qA_free_new, qB_free_new])

            # ------------------------------------------------------------
            # 2. Global ghost signal (Siegel lock) – unchanged
            # ------------------------------------------------------------
            avg_qA = np.mean(self.qA[:, i+1])
            avg_qB = np.mean(self.qB[:, i+1])
            avg_C = np.mean(self.C[:, i+1])
            self.ghost_signal[i+1] = self.siegel.apply_lock(avg_qA, avg_qB, avg_C)

            # --------------------------------------------------------------
            # 3. Wavelet quiver & Gr(2,4) sheaf (using free concentrations)
            # --------------------------------------------------------------
            # Build the wavelet state from free concentrations (for loopy nodes)
            # For simplicity, we use node 0 as representative; you may average.
            # (Adjust to your compute_wavelet_state implementation)
            qA_rep = self.qA_free[0, i+1] if 0 in self.loopy_nodes else self.qA[0, i+1]
            qB_rep = self.qB_free[0, i+1] if 0 in self.loopy_nodes else self.qB[0, i+1]
            C_rep = self.C[0, i+1]
            state_wavelet = np.array([
                C_rep * (1 - qA_rep),
                (1 - C_rep) * (1 - qB_rep),
                1.0 + qA_rep,
                2.0 + qB_rep,
                np.arctan2(qB_rep - qA_rep, qA_rep + qB_rep),
                qA_rep * qB_rep / ((qA_rep + qB_rep)**2 + 1e-8)
            ])
            self.plucker, plucker_norm = self.toric_projection_to_Gr24(state_wavelet, return_norm=True)  # you need this function
            # After self.plucker = self.toric_projection_to_Gr24(state_wavelet) collect plucker zetas
            zeta_mag = np.linalg.norm(self.plucker)
            self.plucker_zeta_times.append(t[i+1])
            self.plucker_zeta_mags.append(plucker_norm) # saving zeta_mag will store 1s

            sheaf_evals = self.compute_sheaf_Laplacian_eigenvalues(t[i+1], state_wavelet)  # optional
            self.spectral_gap = sheaf_evals[1] if len(sheaf_evals) > 1 else 0.0

            # Gr(2,4) schober projection (per-step)
            _frame = gr24_step(self, i)
            if _frame is not None:
                self.gr24_frames.append(_frame)
                if _frame.schubert.at_wall:
                    # This is a Schubert wall crossing = blowup event
                    self._log_blowup(i, _frame)
            
            # --------------------------------------------------------------
            # 4. Toda flow update (every step)
            # --------------------------------------------------------------
            # Build Lax matrix from wavelet state (3x3)
            L = self.build_lax_matrix(state_wavelet, np.mean(self.C[:, i+1]))
            L = self.toda_flow_step(L, dt_step, self.epsilon, self.lambda_val, self.nu, t[i+1])

            schubert_data = self.schubert_cell_spectrum(L, t[i+1])
            self.schubert_history.append({
                "t": t[i+1],
                "stratum": schubert_data["schubert_stratum"],
                "ihara_pred": schubert_data["ihara_prediction"],
                "ghost_proximity": schubert_data["ghost_proximity"],
                "qc_phase_idx": schubert_data["qc_phase_idx"],
                "at_wall": schubert_data["at_wall_crossing"],
            })

            # Log wall crossings — these are the Schubert cell boundary events
            if schubert_data["at_wall_crossing"]:
               print(f"  t={t[i+1]:.3f}: WALL CROSSING  stratum={schubert_data['schubert_stratum']}"
               f"  |minor_top|={abs(schubert_data['minor_top']):.4f}"
               f"  ihara_pred={schubert_data['ihara_prediction']:.4f}")


            target = [L[0,0], L[1,1], L[2,2]]

            for node in range(n_nodes):
                loopy = node in self.loopy_nodes
                if loopy:
                    self.qA_free[node, i+1] += 0.05 * (target[0] - self.qA_free[node, i+1])
                    self.qB_free[node, i+1] += 0.05 * (target[1] - self.qB_free[node, i+1])
                    self.C[node, i+1] += 0.05 * (target[2] - self.C[node, i+1])
                    # Update totals
                    self.qA[node, i+1] = self.qA_free[node, i+1] + self.qA_trap[node, i+1]
                    self.qB[node, i+1] = self.qB_free[node, i+1] + self.qB_trap[node, i+1]
                else:
                    self.qA[node, i+1] += 0.05 * (target[0] - self.qA[node, i+1])
                    self.qB[node, i+1] += 0.05 * (target[1] - self.qB[node, i+1])
                    self.C[node, i+1] += 0.05 * (target[2] - self.C[node, i+1])
                # Clip again
                if loopy:
                    self.qA_free[node, i+1] = np.clip(self.qA_free[node, i+1], 0, 5)
                    self.qB_free[node, i+1] = np.clip(self.qB_free[node, i+1], 0, 6)
                    self.C[node, i+1] = np.clip(self.C[node, i+1], 0, 1)
                else:
                    self.qA[node, i+1] = np.clip(self.qA[node, i+1], 0, 5)
                    self.qB[node, i+1] = np.clip(self.qB[node, i+1], 0, 6)
                    self.C[node, i+1] = np.clip(self.C[node, i+1], 0, 1)

            # ------------------------------------------------------------
            # 5. Compute HH¹ and HH² (same as before)
            # ------------------------------------------------------------
            self.HH1[i+1], self.HH2[i+1] = self.compute_hochschild_invariants(i+1)

            # ------------------------------------------------------------
            # 6. Update history buffer for Toda parameters and prolate operator
            # ------------------------------------------------------------
            state_matrix = np.vstack([self.qA[:, i+1], self.qB[:, i+1], self.C[:, i+1]]).T
            history.append((t[i+1], state_matrix))

            # ------------------------------------------------------------
            # 7. Every 0.5 seconds compute Toda parameters & prolate
            # ------------------------------------------------------------
            if i % int(0.5 / dt) == 0 and len(history) > 10:
                epsilon, lambda_val, nu = self.compute_toda_parameters(list(history), dt)
                prolate_ratio, prolate_vec = self.prolate_operator(list(history))
                self.toda_epsilon[i+1] = epsilon
                self.toda_lambda[i+1] = lambda_val
                self.toda_nu[i+1] = nu
                self.prolate_ratio[i+1] = prolate_ratio

                # ----------------------------------------------------------
                # Prolate angle: deviation of leading eigenvector from sAMY.
                # sAMY is index 3 in [CA1sp, HPF, BLA, sAMY, HY, LA].
                # theta_prolate should equal phi_equil - 1/2 = 1/120 = 0.00833
                # if the prolate and Bridgeland computations agree at equilibrium.
                # Stored in self.prolate_theta for post-simulation analysis.
                # ----------------------------------------------------------
                sAMY_idx = 5   # index of sAMY in region list, const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :sAMY] from region_six.csv file.
                sAMY_axis = np.zeros(self.n_nodes)
                if sAMY_idx < self.n_nodes:
                    sAMY_axis[sAMY_idx] = 1.0
                cos_angle = abs(np.dot(prolate_vec, sAMY_axis))
                cos_angle = np.clip(cos_angle, -1.0, 1.0)
                theta_prolate = np.arccos(cos_angle) / (2 * np.pi)  # in (0,1) units

                # Store for export
                if not hasattr(self, 'prolate_theta'):
                    self.prolate_theta = []
                    self.prolate_theta_times = []
                self.prolate_theta.append(float(theta_prolate))
                self.prolate_theta_times.append(float(t[i+1]))

                # Print comparison against finite-size prediction 1/2 + 1/120
                phi_target = 0.5 + 1/120   # = 0.50833 for 6-region system
                deviation  = abs(theta_prolate - (phi_target - 0.5))
                # Only print every 5 prolate updates to avoid log spam
                if (i // int(0.5 / dt)) % 5 == 0:
                    print(f"  [Prolate] t={t[i+1]:.1f}s  "
                          f"ratio={prolate_ratio:.4f}  "
                          f"theta={theta_prolate:.6f}  "
                          f"1/120={1/120:.6f}  "
                          f"diff={deviation:.6f}")

                if use_toda_feedback:
                    for (u, v) in self.edges:
                        u_int = int(u)
                        v_int = int(v)
                        base = self.edge_weights.get((u_int, v_int), 1.0)
                        mod = 1.0 + 0.5 * (prolate_vec[u_int] + prolate_vec[v_int])
                        self.edge_weights[(u_int, v_int)] = base * mod
                    if hasattr(self, 'sigma_scale'):
                        self.sigma_scale = 1.0 - 0.5 * prolate_ratio
                if prolate_ratio > 0.8 and not locked:
                    print(f"\n*** Prolate eigenbasis lock achieved at t={t[i+1]:.2f}s ***")
                    locked = True
                    self.qB_free[0, i+1] += 1.0
                    print("    -> Rees blow‑up: extra norcain injected.")
                    
                    # --------------------------------------------------------------
                    # Save Plucker for Associahedron Tube coupling
                    # --------------------------------------------------------------
                    q = self.plucker.copy() if hasattr(self, 'plucker') else None
                    if q is not None:
                        self.plucker_history.append((t[i+1], i+1, q.tolist()))
                        self.recompute_step_indices.append(i+1)
                        zeta_mag = self.compute_single_zeta(t[i+1], q)
                        # Append to prime_zeta lists (you may need to initialize them earlier)
                        self.prime_zeta_values.append(zeta_mag)
                        self.prime_zeta_times.append(t[i+1])

                    # --- Trigger Julia blow‑up diagram ---
                    min_region_idx = np.argmin(self.C[:, i+1])
                    region_name = self.region_names[min_region_idx]
                    print(f"    -> Generating blow‑up diagram for region {region_name}")
                    current_weights = {(u,v): self.edge_weights.get((u,v), 1.0) for (u,v) in self.edges}
                    try:
                        _r = self.call_julia_ainf(current_weights,
                            region_name=region_name, step_index=i+1)
                        print(f"    -> Blow-up diagram saved.")
                        # Phase 1: collect obstruction_deficit for m0 init
                        if _r is not None and len(_r) >= 4:
                            _deficit = sum(
                                abs(v) for mk in _r[:4]
                                for d in (mk.values() if isinstance(mk, dict) else [])
                                for v in (d.values() if isinstance(d, dict) else [d])
                                if isinstance(v, (int, float)))
                            self.phase1_blowup_deficits.append({
                                "step": i+1, "t": float(t[i+1]),
                                "region": region_name,
                                "obstruction_deficit": float(_deficit)
                            })
                            self.obstruction_deficit[region_name] = float(_deficit)
                    except Exception as e:
                        print(f"    -> Diagram failed: {e}")
            # ------------------------------------------------------------
            # 8. Dynamic A∞ recomputation (call Julia) – unchanged
            # ------------------------------------------------------------
            if i % ainf_recompute_interval == 0 and i > 0:
                # Build current dynamic edge weights from FREE concentrations (since trapped drug does not affect connectivity)
                current_weights = {}
                for (u, v) in self.edges:
                    u_int = int(u)
                    v_int = int(v)
                    base = self.edge_weights.get((u_int, v_int), 1.0)
                    # Use free concentrations for modulation
                    if u_int in self.loopy_nodes:
                        qA_free_u = self.qA_free[u_int, i]
                    else:
                        qA_free_u = self.qA[u_int, i]
                    if v_int in self.loopy_nodes:
                        qB_free_v = self.qB_free[v_int, i]
                    else:
                        qB_free_v = self.qB[v_int, i]
                    mod = 1.0 + 0.1 * (qA_free_u - qB_free_v)
                    current_weights[(u_int, v_int)] = base * np.clip(mod, 0.5, 2.0)
                # --------------------------------------------------------------
                # Save Plucker for Associahedron Tube coupling
                # --------------------------------------------------------------
                q = self.plucker.copy() if hasattr(self, 'plucker') else None
                if q is not None:
                    self.plucker_history.append((t[i+1], i+1, q.tolist()))
                    self.recompute_step_indices.append(i+1)
                    zeta_mag = self.compute_single_zeta(t[i+1], q)
                    # Append to prime_zeta lists (you may need to initialize them earlier)
                    self.prime_zeta_values.append(zeta_mag)
                    self.prime_zeta_times.append(t[i+1])

                print(f"  [A∞ Phase {self.ainf_phase}] Recomputing at t={t[i+1]:.2f}s (step {i})")
                if self.ainf_phase == 1:
                    # Phase 1: flat A∞, m0=0, no filtration (fast baseline)
                    (self.m3, self.m4, self.m5, self.m6, HH2_dim, prime_paths,
                        self.gerstenhaber, self.cup_product, self.annihilator,
                        self.support, self.prime_ideals) = self.call_julia_ainf(
                            current_weights, step_index=i+1)
                else:
                    # Phase 2: curved A∞ with filtration (m0 from blowup deficits)
                    (self.m3, self.m4, self.m5, self.m6, HH2_dim, prime_paths,
                        self.gerstenhaber, self.cup_product, self.annihilator,
                        self.support, self.prime_ideals) = self.call_julia_ainf_filtered(
                            current_weights, step_index=i+1)
                self.HH2_dim = HH2_dim
                self.prime_paths = prime_paths
                                
            
            # ------------------------------------------------------------
            # 9. Reverse Hironaka ancestor search if HH² spikes
            # ------------------------------------------------------------
            # Detect HH² ascent (simpler, more robust)
            if i > 50 and self.HH2[i+1] > np.max(self.HH2[max(0, i-200):i]) * 1.5:
                print(f"  [Reverse Hironaka] HH² increased by >50% at t={t[i+1]:.2f}")
            #if i > 10 and self.HH2[i+1] > 10.0 * np.median(self.HH2[max(0, i-100):i+1]):
                #print(f"  [Reverse Hironaka] High HH² spike at t={t[i+1]:.2f}, looking for good ancestor...")
                
                # --- NEW: determine the region with lowest consciousness at this step ---
                min_region_idx = np.argmin(self.C[:, i+1])        # 0..5
                region_name = self.region_names[min_region_idx]   # e.g., "sAMY"
                print(f"    Most affected region: {region_name}")
                
                # Build current edge weights (as you already do for A∞ recomputation)
                current_weights = {}
                for (u, v) in self.edges:
                    base = self.edge_weights.get((u, v), 1.0)
                    # you may modulate with free concentrations as in your existing loop
                    current_weights[(u, v)] = base
                
                # --------------------------------------------------------------
                # Save Plucker for Associahedron Tube coupling
                # --------------------------------------------------------------
                q = self.plucker.copy() if hasattr(self, 'plucker') else None
                if q is not None:
                    self.plucker_history.append((t[i+1], i+1, q.tolist()))
                    self.recompute_step_indices.append(i+1)
                    zeta_mag = self.compute_single_zeta(t[i+1], q)
                    # Append to prime_zeta lists (you may need to initialize them earlier)
                    self.prime_zeta_values.append(zeta_mag)
                    self.prime_zeta_times.append(t[i+1])

                # Call Julia in --full mode to generate the blow‑up diagram for that region
                try:
                    _ = self.call_julia_ainf(current_weights, region_name=region_name,  step_index=i+1)
                    print(f"    -> Blow‑up diagram saved for region {region_name}")
                except Exception as e:
                    print(f"    -> Failed to generate blow‑up diagram: {e}")
                
                # Then continue with the ancestor search or Rees blow‑up (inject norcain)
                best_ancestor = None
                _ancestor_threshold = self.hh2_global_median * 0.5
                min_ancestor = max(50, i // 20)
                for j in range(max(min_ancestor, i-500), i, 50):
                    if self.HH2[j] < _ancestor_threshold and j > 50:
                        best_ancestor = j
                        break
                if best_ancestor is not None:
                    # reset state to ancestor
                    print(f"    -> Found ancestor at t={t[best_ancestor]:.2f}. Resetting to that state.")
                    for node in range(n_nodes):
                        # Reset all compartments (free, trap, C) to ancestor values
                        if node in self.loopy_nodes:
                            self.qA_free[node, i+1] = self.qA_free[node, best_ancestor]
                            self.qA_trap[node, i+1] = self.qA_trap[node, best_ancestor]
                            self.qB_free[node, i+1] = self.qB_free[node, best_ancestor]
                            self.qB_trap[node, i+1] = self.qB_trap[node, best_ancestor]
                            self.qA[node, i+1] = self.qA_free[node, i+1] + self.qA_trap[node, i+1]
                            self.qB[node, i+1] = self.qB_free[node, i+1] + self.qB_trap[node, i+1]
                        else:
                            self.qA[node, i+1] = self.qA[node, best_ancestor]
                            self.qB[node, i+1] = self.qB[node, best_ancestor]
                        self.C[node, i+1] = self.C[node, best_ancestor]
                else:
                    # inject norcain (Rees blow‑up)
                    for node in range(n_nodes):
                        if node in self.loopy_nodes:
                            self.qB_free[node, i+1] += 1.0
                        else:
                            self.qB[node, i+1] += 1.0
                    print("    -> Rees blow‑up applied (norcain injected).")
                    
                    # --------------------------------------------------------------
                    # Save Plucker for Associahedron Tube coupling
                    # --------------------------------------------------------------
                    q = self.plucker.copy() if hasattr(self, 'plucker') else None
                    if q is not None:
                        self.plucker_history.append((t[i+1], i+1, q.tolist()))
                        self.recompute_step_indices.append(i+1)
                        zeta_mag = self.compute_single_zeta(t[i+1], q)
                        # Append to prime_zeta lists (you may need to initialize them earlier)
                        self.prime_zeta_values.append(zeta_mag)
                        self.prime_zeta_times.append(t[i+1])

                    # --- Trigger Julia blow‑up diagram ---
                    min_region_idx = np.argmin(self.C[:, i+1])
                    region_name = self.region_names[min_region_idx]
                    print(f"    -> Generating blow‑up diagram for region {region_name}")
                    current_weights = {(u,v): self.edge_weights.get((u,v), 1.0) for (u,v) in self.edges}
                    try:
                        _r = self.call_julia_ainf(current_weights,
                            region_name=region_name, step_index=i+1)
                        print(f"    -> Blow-up diagram saved.")
                        # Phase 1: collect obstruction_deficit for m0 init
                        if _r is not None and len(_r) >= 4:
                            _deficit = sum(
                                abs(v) for mk in _r[:4]
                                for d in (mk.values() if isinstance(mk, dict) else [])
                                for v in (d.values() if isinstance(d, dict) else [d])
                                if isinstance(v, (int, float)))
                            self.phase1_blowup_deficits.append({
                                "step": i+1, "t": float(t[i+1]),
                                "region": region_name,
                                "obstruction_deficit": float(_deficit)
                            })
                            self.obstruction_deficit[region_name] = float(_deficit)
                    except Exception as e:
                        print(f"    -> Diagram failed: {e}")
            
            
            # ------------------------------------------------------------        
            # After all updates for time step i+1 (e.g., after history.append)
            # ------------------------------------------------------------
            # Save diskspace
            #if i % vtk_interval == 0:   # vtk_interval = 10 or 20
            #    self.write_vtu(i+1)
            

        # After loop, compute Plücker trajectory and phase transitions
        self.plucker = self.compute_plucker_trajectory()
        self.detect_phase_transitions()

        # ── Phase 1 baseline export ──────────────────────────────────────────
        # Save obstruction deficits collected during blowup events.
        # These become the m0_curvature inputs for Phase 2.
        if self.phase1_blowup_deficits:
            import json as _json2
            baseline = {
                "ainf_phase": self.ainf_phase,
                "n_blowup_events": len(self.phase1_blowup_deficits),
                "blowup_deficits": self.phase1_blowup_deficits,
                "m0_curvature_preview": {
                    r: M0_CURVATURE_SCALE * float(d)**0.5
                    for r, d in self.obstruction_deficit.items()
                },
                "filtration_ready": self.ainf_phase == 1 and len(self.phase1_blowup_deficits) > 0,
                "next_step": "Set AINF_PHASE=2 and rerun to activate curved A∞ with filtration"
            }
            with open("phase1_baseline.json", "w") as fp:
                _json2.dump(baseline, fp, indent=2)
            print(f"  Phase 1 baseline saved: {len(self.phase1_blowup_deficits)} blowup deficits")
            print(f"  m0_curvature preview: {baseline['m0_curvature_preview']}")
            if self.ainf_phase == 2:
                print(f"  Phase 2 active: m0_curvature={self.m0_curvature}")
            else:
                print("-> To activate Phase 2: set AINF_PHASE=2 in BALBc_Opiate_Norcain.py")
        
        # ── Gr(2,4) schober projection ──────────────────────────────────────
        # Projects the full plucker_history onto Gr(2,4), identifying
        # Schubert strata, wall crossings, QC eigenvalues, and Ihara hints.
        # Requires plucker_history to be populated (done in loop above).
        # gr24_result is None-safe: if plucker_history is empty it skips.
        try:
            from gr24_schober_projection import attach_to_simulation
            if self.plucker_history:
                self.gr24_result = attach_to_simulation(self)
            else:
                self.gr24_result = None
        except ImportError:
            self.gr24_result = None   # module not found, continue silently 

        return self.t, self.C, self.qA, self.qB, self.HH1, self.HH2
    
    
    def compute_plucker_trajectory(self):
        n = len(self.t)
        # Plücker coordinates for Gr(2,4) are always 6: p12,p13,p14,p23,p24,p34
        # This is independent of the number of brain regions (n_nodes)
        plucker = np.zeros((n, 6))
        lambda_A = np.log(2) / self.half_life_A
        lambda_B = np.log(2) / self.half_life_B
        for i in range(n):
            C_mean = np.mean(self.C[:, i])
            qA_mean = np.mean(self.qA[:, i])
            qB_mean = np.mean(self.qB[:, i])
            p12 = C_mean / (1 + qA_mean / self.EC50_A)
            p13 = qB_mean / (1 + qB_mean / self.EC50_B)
            p14 = (1 - C_mean) * np.exp(-lambda_A * self.t[i])
            p23 = C_mean * np.exp(-lambda_B * self.t[i])
            p24 = qA_mean * np.exp(-lambda_A * self.t[i])
            p34 = qB_mean * np.exp(-lambda_B * self.t[i])
            plucker[i] = [p12, p13, p14, p23, p24, p34]
            norm = np.linalg.norm(plucker[i])
            if norm > 0:
                plucker[i] /= norm
        return plucker
    

    def _log_blowup(self, step_idx, frame):
        """Record a Schubert wall crossing detected by gr24_step."""
        if not hasattr(self, 'gr24_blowup_log'):
            self.gr24_blowup_log = []
        t_val = float(self.t[step_idx]) if hasattr(self, 't') and step_idx < len(self.t) else float(step_idx)
        self.gr24_blowup_log.append({
            "step": step_idx, "t": t_val,
            "stratum": frame.schubert.stratum,
            "wall_type": frame.schubert.wall_type,
            "klein_Q": float(frame.schubert.klein_Q),
            "ghost_proximity": float(frame.ghost_proximity),
            "qc_phase_idx": frame.qc_phase_idx,
            "ihara_prediction": float(frame.ihara_prediction),
        })

    def detect_phase_transitions(self):
        """Find times when consciousness drops below threshold, then trace reverse Hironaka path."""
        # Use Node 0 for detection (all nodes similar)
        C0 = self.C[0, :]
        below = C0 < self.threshold
        # Find falling edges (start of unconscious periods)
        falling = np.where(np.diff(below.astype(int)) == 1)[0]
        self.transition_times = self.t[falling]
        self.reverse_trajectories = []
        
        for idx in falling:
            # Trace backward in time until consciousness > 0.7
            traj = []
            cur = idx
            while cur >= 0 and len(traj) < 200:
                traj.append(self.plucker[cur])
                if self.C[0, cur] > 0.7:
                    break
                cur -= 1
            if len(traj) > 1:
                # Reverse to get path from singular to smooth
                self.reverse_trajectories.append(traj[::-1])

    def count_phase_transitions(self, threshold=None):
        """Count downward and upward transitions for each node."""
        if threshold is None:
            threshold = self.threshold
        counts = {}
        for node in range(self.n_nodes):
            C_node = self.C[node, :]
            below = C_node < threshold
            # Downward transitions: where below becomes True (falling edge)
            down = np.where(np.diff(below.astype(int)) == 1)[0]
            # Upward transitions: where below becomes False (rising edge)
            up = np.where(np.diff(below.astype(int)) == -1)[0]
            counts[node] = {'down': len(down), 'up': len(up)}
        return counts
    
    def compute_dehn_twist(self, dose_time):
        """Simplified Dehn twist matrix in Sp(4,Z)."""
        T = np.eye(4)
        T[0, 1] = 1
        return T
    
    def verify_coherence(self):
        """Compute product of Dehn twists for the three doses."""
        if len(self.dose_times) < 3:
            return False, 0
        T1 = self.compute_dehn_twist(self.dose_times[0])
        T2 = self.compute_dehn_twist(self.dose_times[1])
        T3 = self.compute_dehn_twist(self.dose_times[2])
        prod = T1 @ T2 @ T3
        error = np.linalg.norm(prod - np.eye(4))
        return error < 1e-6, error
    
    def compute_path_zeta(self, path):
        """Zeta function for a path on the Klein quadric."""
        if len(path) < 2:
            return 0+0j
        lengths = []
        phases = []
        for i in range(len(path)-1):
            p1, p2 = path[i], path[i+1]
            inner = np.abs(np.dot(p1, p2))
            inner = np.clip(inner, -0.999999, 0.999999)
            lengths.append(np.arccos(inner))
            z1 = p1[0] + 1j * p1[1]
            z2 = p2[0] + 1j * p2[1]
            phases.append(np.angle(z2) - np.angle(z1))
        L = np.sum(lengths)
        phi = np.sum(phases)
        s = 0.5 + 1j
        return np.exp(-s * L) * np.exp(1j * phi)
    
    def compute_prime_zeta(self):
        """Compute zeta values for all reverse trajectories."""
        prime_zeta = []
        for traj in self.reverse_trajectories:
            z = self.compute_path_zeta(traj)
            prime_zeta.append(z)
        return prime_zeta
    
# ============================================================================
# PART: WAVELET QUIVER (6 vertices, 28 arrows)
# ============================================================================

class DynamicWaveletQuiver:
    """
    6-vertex, 28-arrow quiver driven by molecular dynamics.
    Vertices: a (amp A), b (amp B), ω₁ (freq A), ω₂ (freq B), φ (phase), κ (coupling)
    """
    V_A, V_B, V_W1, V_W2, V_PHI, V_KAPPA = 0, 1, 2, 3, 4, 5
    vertex_names = ['a', 'b', 'ω₁', 'ω₂', 'φ', 'κ']

    def __init__(self, dynamics, dt=0.02):
        self.dynamics = dynamics
        self.dt = dt

    def _get_time_index(self, t):
        return np.argmin(np.abs(self.dynamics.t - t))

    def get_dynamic_rates(self, t, state):
        idx = self._get_time_index(t)
        if idx >= len(self.dynamics.t):
            idx = -1
        qA = self.dynamics.qA[0, idx] if idx >= 0 else 0
        qB = self.dynamics.qB[0, idx] if idx >= 0 else 0
        C = self.dynamics.C[0, idx] if idx >= 0 else 0.5

        rates = {}

        # Self-loops (6)
        lamA = np.log(2) / self.dynamics.half_life_A
        lamB = np.log(2) / self.dynamics.half_life_B
        rates[(self.V_A, self.V_A)] = -lamA * (1 - qA)
        rates[(self.V_B, self.V_B)] = -lamB * (1 - qB)
        rates[(self.V_W1, self.V_W1)] = -0.01
        rates[(self.V_W2, self.V_W2)] = -0.01
        rates[(self.V_PHI, self.V_PHI)] = -0.005
        rates[(self.V_KAPPA, self.V_KAPPA)] = -0.1 * (1 - qA * qB)

        # Amplitude → Frequency (2)
        rates[(self.V_A, self.V_W1)] = 0.1 * qA
        rates[(self.V_B, self.V_W2)] = 0.1 * qB

        # Frequency → Amplitude (2)
        rates[(self.V_W1, self.V_A)] = 0.05 * C
        rates[(self.V_W2, self.V_B)] = 0.05 * C

        # Cross-coupling A↔B (4)
        coupling = qA * qB
        rates[(self.V_A, self.V_B)] = 0.2 * coupling
        rates[(self.V_B, self.V_A)] = 0.2 * coupling
        rates[(self.V_W1, self.V_W2)] = 0.03 * (qA + qB) / 2
        rates[(self.V_W2, self.V_W1)] = 0.03 * (qA + qB) / 2

                # Phase coupling (6)
        rates[(self.V_PHI, self.V_A)] = 0.01 * qA
        rates[(self.V_PHI, self.V_B)] = 0.01 * qB
        rates[(self.V_A, self.V_PHI)] = 0.02 * C
        rates[(self.V_B, self.V_PHI)] = 0.02 * C
        rates[(self.V_W1, self.V_PHI)] = 0.015 * qA
        rates[(self.V_W2, self.V_PHI)] = 0.015 * qB

        # Coupling strength interactions (8)
        kappa = qA * qB / ((qA + qB)**2 + 1e-8)
        rates[(self.V_KAPPA, self.V_A)] = 0.15 * kappa
        rates[(self.V_KAPPA, self.V_B)] = 0.15 * kappa
        rates[(self.V_KAPPA, self.V_W1)] = 0.1 * kappa
        rates[(self.V_KAPPA, self.V_W2)] = 0.1 * kappa
        rates[(self.V_A, self.V_KAPPA)] = 0.1 * qA
        rates[(self.V_B, self.V_KAPPA)] = 0.1 * qB
        rates[(self.V_W1, self.V_KAPPA)] = 0.08 * C
        rates[(self.V_W2, self.V_KAPPA)] = 0.08 * C

        return rates

    def transition_matrix(self, t, state):
        M = np.zeros((6, 6))
        rates = self.get_dynamic_rates(t, state)
        for (src, tgt), rate in rates.items():
            M[tgt, src] = rate
        return M

    def evolve(self, state, t):
        M = self.transition_matrix(t, state)
        new_state = state + (M @ state) * self.dt
        # Clamp
        new_state[self.V_A] = np.clip(new_state[self.V_A], 0, 1)
        new_state[self.V_B] = np.clip(new_state[self.V_B], 0, 1)
        new_state[self.V_W1] = np.clip(new_state[self.V_W1], 0.5, 3)
        new_state[self.V_W2] = np.clip(new_state[self.V_W2], 0.5, 3)
        new_state[self.V_PHI] = np.clip(new_state[self.V_PHI], -np.pi, np.pi)
        new_state[self.V_KAPPA] = np.clip(new_state[self.V_KAPPA], 0, 1)
        return new_state

    def simulate_quiver(self, initial_state, t_span=None, t_array=None):
        if t_array is not None:
            t = t_array
        else:
            t = np.arange(t_span[0], t_span[1], self.dt)
        states = np.zeros((len(t), 6))
        current = initial_state.copy()
        for i, ti in enumerate(t):
            states[i] = current
            current = self.evolve(current, ti)
        return t, states



# Graph Spectral sheaf over physical graph
class SpectralSheaf:
    """
    Cellular sheaf on the original 3-node, 7-edge graph.
    Stalks = 4‑dim probability vectors.
    Restriction maps = 4×4 transition matrices.
    """
    def __init__(self, dynamics: FullGraphDynamics, quiver: DynamicWaveletQuiver):
        self.dynamics = dynamics
        self.quiver = quiver          # to get transition rates
        self.n_nodes = 3
        self.stalk_dim = 4
        self.total_dim = self.n_nodes * self.stalk_dim   # 12

    def build_restriction_matrix(self, edge, t):
        """Return 4×4 transition matrix for given edge at time t."""
        src, tgt = edge
        # Use existing transition_rate to get probabilities
        kA = self.dynamics.transition_rate(edge, t, 'A')
        kB = self.dynamics.transition_rate(edge, t, 'B')
        # Build 4×4 diagonal matrix? Actually it's not diagonal because molecules
        # can change state? In our model, the transition is diagonal in the joint state basis:
        # HH→HH, HT→HT, TH→TH, TT→TT. So it's diagonal.
        k_HH = kA * kB
        k_HT = kA * (1 - kB)
        k_TH = (1 - kA) * kB
        k_TT = (1 - kA) * (1 - kB)
        # Normalise?
        total = k_HH + k_HT + k_TH + k_TT
        if total > 0:
            k_HH /= total; k_HT /= total; k_TH /= total; k_TT /= total
        return np.diag([k_HH, k_HT, k_TH, k_TT])

    def laplacian(self, t):
        """
        Compute sheaf Laplacian (block matrix) at time t.
        L = sum over edges of (incidence matrix ⊗ restriction)^T (incidence ⊗ restriction)
        For simplicity, we build directly: L = D - A, where:
        - D: block diagonal of sum of restriction maps from each vertex.
        - A: block adjacency of restriction maps.
        """
        L = np.zeros((self.total_dim, self.total_dim))
        # For each vertex, sum of outgoing maps
        for v in range(self.n_nodes):
            # sum over edges incident to v (both directions)
            for edge in self.dynamics.edges:
                if edge[0] == v:   # outgoing
                    v = int(v)
                    R = self.build_restriction_matrix(edge, t)
                    # contribution to D at v
                    L[v*4:(v+1)*4, v*4:(v+1)*4] += R.T @ R
                elif edge[1] == v:  # incoming
                    R = self.build_restriction_matrix(edge, t)
                    # incoming edge contributes to D at v as well? Actually the sheaf Laplacian
                    # is defined as L = δ^* δ, where δ is the co-boundary operator.
                    # The standard formula: L_v = Σ_{e incident to v} φ_e^T φ_e (for a sheaf with inner products)
                    # So we add φ_e^T φ_e for each edge incident to v.
                    L[v*4:(v+1)*4, v*4:(v+1)*4] += R.T @ R
                    # Off‑diagonal blocks: for edge e = (u,v), we have -φ_e at block (u,v) and -φ_e^T at (v,u)
                    # Actually, the off-diagonal block for edge (u->v) is -φ_e, and for (v->u) is -φ_e^T.
                    # But our edges are directed; we must treat both directions.
        # We'll simplify: use undirected approach by adding both orientations.
        # Instead, we'll build using incidence matrix approach.
        return self._build_laplacian_direct(t)

    def _build_laplacian_direct(self, t):
        # Simpler: iterate over edges and add contributions
        L = np.zeros((self.total_dim, self.total_dim))
        for edge in self.dynamics.edges:
            u, v = edge
            u = int(u)
            v = int(v)
            R = self.build_restriction_matrix(edge, t)
            # Diagonal blocks: add R^T R to u and v
            L[u*4:(u+1)*4, u*4:(u+1)*4] += R.T @ R
            L[v*4:(v+1)*4, v*4:(v+1)*4] += R.T @ R
            # Off-diagonal blocks
            L[u*4:(u+1)*4, v*4:(v+1)*4] += -R
            L[v*4:(v+1)*4, u*4:(u+1)*4] += -R.T
        return L

    def eigenvalues(self, t):
        L = self.laplacian(t)
        # We're interested in the smallest eigenvalues (zero indicates missing structure)
        return np.linalg.eigvalsh(L)

    def detect_missing_edges(self, t, tol=1e-6):
        evals = self.eigenvalues(t)
        zero_evals = evals[evals < tol]
        if len(zero_evals) > 0:
            # There is at least one zero eigenvalue → missing structure
            # Compute eigenvectors to localise
            _, evecs = np.linalg.eigh(L)
            # The eigenvector corresponding to the smallest eigenvalue
            v = evecs[:, 0]
            # Reshape to (n_nodes, stalk_dim)
            v_reshaped = v.reshape(self.n_nodes, self.stalk_dim)
            # Find which vertex has largest norm
            node_contrib = np.linalg.norm(v_reshaped, axis=1)
            suspected_node = np.argmax(node_contrib)
            # Also check edges: the eigenvector entries on both endpoints of an edge
            # indicate if that edge is problematic
            return True, suspected_node, v_reshaped
        return False, None, None

# Adds HH3 over physical graph.

class GraphSpectralSheaf:
    """
    Cellular sheaf on the 3‑node, 7‑edge graph.
    Stalks = 4‑dim probability vectors.
    Restriction maps = 4×4 diagonal transition matrices.
    """
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.n_nodes = dynamics.n_nodes
        self.stalk_dim = 4
        self.total_dim = self.n_nodes * self.stalk_dim   # 24 6 nodes * 4

    def transition_matrix(self, edge, t):
        """Return the 4×4 diagonal transition matrix for the given edge at time t."""
        kA = self.dynamics.transition_rate(edge, t, 'A')
        kB = self.dynamics.transition_rate(edge, t, 'B')
        kHH = kA * kB
        kHT = kA * (1 - kB)
        kTH = (1 - kA) * kB
        kTT = (1 - kA) * (1 - kB)
        total = kHH + kHT + kTH + kTT
        if total > 0:
            kHH /= total
            kHT /= total
            kTH /= total
            kTT /= total
        return np.diag([kHH, kHT, kTH, kTT])

    def laplacian(self, t):
        """
        Compute the sheaf Laplacian (12×12) as L = D - A, where:
        D: block‑diagonal sum of R^T R for each incident edge,
        A: off‑diagonal blocks = -R for edge (u→v) and -R^T for (v→u).
        """
        L = np.zeros((self.total_dim, self.total_dim))
        for edge in self.dynamics.edges:
            u, v = edge
            R = self.transition_matrix(edge, t)
            u_int = int(u)
            v_int = int(v)
            # Add contributions to diagonal blocks
            L[u_int*4:(u_int+1)*4, u_int*4:(u_int+1)*4] += R.T @ R
            L[v_int*4:(v_int+1)*4, v_int*4:(v_int+1)*4] += R.T @ R
            # Off‑diagonal blocks
            L[u_int*4:(u_int+1)*4, v_int*4:(v_int+1)*4] += -R
            L[v_int*4:(v_int+1)*4, u_int*4:(u_int+1)*4] += -R.T
        # Add a tiny identity to ensure positive definiteness
        L += 1e-8 * np.eye(self.total_dim)
        return L

    def eigenvalues(self, t):
        """Return sorted eigenvalues of the sheaf Laplacian at time t."""
        L = self.laplacian(t)
        return np.linalg.eigvalsh(L)

    def spectral_clustering(self, t, n_clusters=2):
        """
        Cluster the 3 nodes (or the 4‑dim stalks) using eigenvectors of L.
        Returns labels for each vertex (3 labels).
        """
        L = self.laplacian(t)
        evals, evecs = np.linalg.eigh(L)
        # Use the smallest non‑zero eigenvectors (e.g., first n_clusters)
        # The trivial eigenvector (constant) is often zero; we skip it if it's small.
        # For simplicity, use the eigenvectors corresponding to the smallest eigenvalues.
        # Reshape to (n_nodes, stalk_dim) to get per‑vertex contributions.
        # Here we want to cluster vertices, so we take the norm of each stalk's eigenvector part.
        X = np.zeros((self.n_nodes, n_clusters))
        for i in range(self.n_nodes):
            for j in range(n_clusters):
                # The eigenvector entries for vertex i are in indices i*4 to i*4+3
                # Take the norm of that 4‑vector as a feature
                X[i, j] = np.linalg.norm(evecs[i*4:(i+1)*4, j+1])
        # Normalize rows
        X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)
        return labels

    def triple_interaction_matrix(self, t):
        """
        Approximate HH³ by (adjacency)^3, where adjacency is a 3×3 matrix
        with entries = norm of restriction maps (or the total flow).
        """
        # Build a 3×3 weighted adjacency where weight = total flow on edge
        A = np.zeros((self.n_nodes, self.n_nodes))
        for edge in self.dynamics.edges:
            u, v = edge
            u = int(u)
            v = int(v)
            R = self.transition_matrix(edge, t)
            # Use trace as a scalar weight (total probability transferred)
            weight = np.trace(R)   # sum of diagonal entries = total flow
            A[u, v] = weight
        # Symmetrize
        A_sym = (A + A.T) / 2
        T = A_sym @ A_sym @ A_sym
        return T

# Spectral sheaf over Quivers
# ============================================================================
# NEW: Quiver Spectral Sheaf Analysis
# ============================================================================

class QuiverSpectralSheaf:
    """
    Builds a sheaf on the 6-vertex quiver with 1‑dimensional stalks.
    The sheaf Laplacian is the symmetrized weighted Laplacian of the digraph.
    """
    def __init__(self, quiver):
        self.quiver = quiver
        self.n_vertices = 6
        self.vertex_names = quiver.vertex_names

    def adjacency_matrix(self, t, state):
        """6×6 weighted adjacency matrix A where A[i,j] = rate from j → i."""
        rates = self.quiver.get_dynamic_rates(t, state)
        A = np.zeros((self.n_vertices, self.n_vertices), dtype=np.float64)
        for (src, tgt), rate in rates.items():
            A[tgt, src] = rate
        return A

    def laplacian(self, t, state, regularization=1e-10):
        """Symmetric normalized Laplacian L = I - D^{-1/2} A D^{-1/2} with regularization."""
        A = self.adjacency_matrix(t, state)
        out_deg = A.sum(axis=0)
        in_deg = A.sum(axis=1)
        deg = (out_deg + in_deg) / 2.0
        # Add small epsilon to avoid division by zero
        deg = np.maximum(deg, 1e-12)
        D_inv_sqrt = np.diag(1.0 / np.sqrt(deg))
        A_sym = (A + A.T) / 2.0
        L = np.eye(self.n_vertices) - D_inv_sqrt @ A_sym @ D_inv_sqrt
        # Add regularization to ensure positive definiteness
        L += regularization * np.eye(self.n_vertices)
        return L

    def eigenvalues(self, t, state):
        """Return sorted eigenvalues of the sheaf Laplacian at time t."""
        L = self.laplacian(t, state)
        # Check for NaNs or Infs
        if not np.isfinite(L).all():
            return np.full(self.n_vertices, np.nan)
        try:
            evals = np.linalg.eigvalsh(L)
            return evals
        except np.linalg.LinAlgError:
            # Fallback: use eigh with regularized matrix
            L_reg = L + 1e-8 * np.eye(self.n_vertices)
            evals = np.linalg.eigvalsh(L_reg)
            return evals

    def spectral_clustering(self, t, state, n_clusters=2):
        """Cluster vertices using eigenvectors of the sheaf Laplacian."""
        L = self.laplacian(t, state)
        # Use eigh (more stable) and catch errors
        try:
            evals, evecs = np.linalg.eigh(L)
        except np.linalg.LinAlgError:
            L_reg = L + 1e-8 * np.eye(self.n_vertices)
            evals, evecs = np.linalg.eigh(L_reg)

        # Use the smallest n_clusters eigenvectors (skip the first if it's near zero)
        # We'll use eigenvectors 1..n_clusters (0-indexed)
        # For a connected graph, the first eigenvector is constant, so we skip it.
        X = evecs[:, 1:n_clusters+1]  # shape (6, n_clusters)
        # Normalize rows
        row_norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / (row_norms + 1e-12)
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
        labels = kmeans.fit_predict(X)
        return labels

    def triple_interaction_matrix(self, t, state):
        """Approximate HH³ by A^3 (three‑step paths)."""
        A = self.adjacency_matrix(t, state)
        A_sym = (A + A.T) / 2.0
        T = A_sym @ A_sym @ A_sym
        return T

# ----------------------------------------------------------------------
# Integration into main (after quiver simulation)
# ----------------------------------------------------------------------

def add_quiver_spectral_analysis(quiver, states, t_q):
    """Compute and plot spectral sheaf properties over time."""
    sheaf = QuiverSpectralSheaf(quiver)

    # Preallocate arrays
    n_t = len(t_q)
    eigvals_all = np.zeros((n_t, 6))
    second_eigenvalue = np.zeros(n_t)
    triple_heatmap = np.zeros((n_t, 6, 6))

    for i in range(n_t):
        state = states[i]
        t = t_q[i]
        L = sheaf.laplacian(t, state)
        evals = np.linalg.eigvalsh(L)
        eigvals_all[i] = evals
        second_eigenvalue[i] = evals[1]  # Fiedler value
        triple_heatmap[i] = sheaf.triple_interaction_matrix(t, state)

    # Plot 1: Evolution of eigenvalues (especially the second eigenvalue)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(t_q, second_eigenvalue, 'b-', linewidth=2)
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Second eigenvalue (spectral gap)')
    ax[0].set_title('Sheaf Laplacian Spectral Gap')
    ax[0].grid(True)

    # Plot the first few eigenvalues as a heatmap over time
    im = ax[1].imshow(eigvals_all.T, aspect='auto', cmap='viridis',
                      extent=[t_q[0], t_q[-1], 0, 5], origin='lower')
    ax[1].set_xlabel('Time (s)')
    ax[1].set_ylabel('Eigenvalue index')
    ax[1].set_title('Eigenvalues of Sheaf Laplacian')
    plt.colorbar(im, ax=ax[1], label='Eigenvalue')
    plt.tight_layout()
    plt.savefig('sheaf_eigenvalues.png', dpi=150)
    plt.show()
    print("✓ Saved: sheaf_eigenvalues.png")

    # Plot 2: Clustering over time (choose a few time points)
    sample_times = [0, 5, 10, 15]  # seconds
    fig, axes = plt.subplots(1, len(sample_times), figsize=(16, 4))
    for idx, t0 in enumerate(sample_times):
        i = np.argmin(np.abs(t_q - t0))
        labels = sheaf.spectral_clustering(t_q[i], states[i], n_clusters=2)
        # Visualize the assignment
        colors = ['red' if l == 0 else 'blue' for l in labels]
        ax = axes[idx]
        ax.bar(sheaf.vertex_names, colors, color=colors, edgecolor='black')
        ax.set_title(f'Clusters at t={t0:.1f}s')
        ax.set_ylabel('Cluster label')
    plt.tight_layout()
    plt.savefig('sheaf_clusters.png', dpi=150)
    plt.show()
    print("✓ Saved: sheaf_clusters.png")

    # Plot 3: Triple interaction heatmap averaged over time windows
    # Smooth over time with a moving window
    window = 50  # number of time points
    n_nodes_hm = triple_heatmap.shape[1]  # use actual size from data
    triple_avg = np.zeros((n_nodes_hm, n_nodes_hm))
    for i in range(0, n_t - window, window//2):
        triple_avg += np.mean(triple_heatmap[i:i+window], axis=0)
    triple_avg /= (2 * n_t / window)  # approximate average

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(triple_avg, cmap='hot', interpolation='nearest')
    ax.set_xticks(range(6))
    ax.set_yticks(range(6))
    ax.set_xticklabels(sheaf.vertex_names)
    ax.set_yticklabels(sheaf.vertex_names)
    ax.set_title('Average Triple Interaction Strength (A³)')
    plt.colorbar(im, ax=ax, label='Strength')
    plt.tight_layout()
    plt.savefig('triple_interaction.png', dpi=150)
    plt.show()
    print("✓ Saved: triple_interaction.png")

    return {
        'eigenvalues': eigvals_all,
        'second_eigenvalue': second_eigenvalue,
        'triple_heatmap': triple_heatmap
    }

# Tracks molecules as they move for Chern class jump
class ReverseHironakaMoleculeResolver:
    def __init__(self):
        # The 7 physical edges of the Opiate/Norcain graph
        self.edges = [(0,1), (1,0), (0,2), (2,0), (1,2), (2,1), (1,1)]
        # Lifting to 28 arrows for the double-cover resolution
        self.n_arrows = 28 

    def apply_hironaka_resolution(self, matrix_3x3, t):
        """
        Resolves the M22 Schubert overlap and transforms the 3x3 state
        into the harmonic 'a cos(w1t) + b sin(w2t)' form.
        """
        # Safety: Ensure we are working with a 3x3 matrix
        if matrix_3x3.ndim == 1:
            matrix_3x3 = matrix_3x3.reshape(3, 3)
            
        # 1. Identify the Schubert Glue (M22)
        # This is the overlap between the two 2x2 toric minors
        m22_glue = matrix_3x3[1, 1]
        
        # 2. Extract Frequencies (w1, w2) via Spectral Decomposition
        # These represent the 'Prime Paths' discovered in the 28-arrow quiver
        evals = np.linalg.eigvals(matrix_3x3)
        # Sort by magnitude to find the dominant molecular resonances
        w = np.sort(np.abs(np.imag(evals)))[::-1]
        w1, w2 = w[0], w[1]
        
        # 3. Harmonic Transformation (The Resolved Wavelet)
        # Coefficients a and b are derived from the Plücker coordinates
        a = np.real(matrix_3x3[0, 0]) # Molecule A density
        b = np.real(matrix_3x3[2, 2]) # Molecule B density
        
        # The Reverse Hironaka 'Smooth' Signal
        resolved_signal = a * np.cos(w1 * t) + b * np.sin(w2 * t)
        
        return {
            "m22_pivot": m22_glue,
            "resolved_wavelet": resolved_signal,
            "frequencies": (w1, w2),
            "chern_jump": "Resolved via 28-arrow lifting"
        }

# ============================================================================
# Visualization functions (20-panel dashboard + ghost signal + prime zeta)
# ============================================================================
def create_dashboard(dynamics):
    t, C, qA, qB, HH1, HH2 = dynamics.t, dynamics.C, dynamics.qA, dynamics.qB, dynamics.HH1, dynamics.HH2
    plucker = dynamics.plucker
    dose_times = dynamics.dose_times
    threshold = dynamics.threshold
    n_nodes = dynamics.n_nodes

    fig = plt.figure(figsize=(22, 32))
    # Use a colormap for nodes
    colors = plt.cm.tab10(np.linspace(0, 1, n_nodes))
    dt_val = t[1] - t[0]

    # 1. Consciousness across all nodes
    ax1 = plt.subplot(6, 4, 1)
    for node in range(n_nodes):
        ax1.plot(t, C[node, :], color=colors[node], lw=2, label=f'Node {node}')
    ax1.axhline(y=threshold, color='red', ls='--')
    for dt in dose_times:
        ax1.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax1.set_title('1. Consciousness Across All Nodes')
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # 2. Restoration with multiple doses
    ax2 = plt.subplot(6, 4, 2)
    for node in range(n_nodes):
        ax2.plot(t, C[node, :], color=colors[node], lw=2)
    ax2.fill_between(t, 0, threshold, alpha=0.3, color='red', label='Unconscious')
    ax2.fill_between(t, threshold, 1, alpha=0.2, color='green', label='Conscious')
    for i, dt in enumerate(dose_times):
        ax2.axvline(x=dt, color='green', ls=':', alpha=0.7)
        ax2.annotate(f'Dose {i+1}', xy=(dt, 0.85), xytext=(dt, 0.92),
                     arrowprops=dict(arrowstyle='->', color='green'), fontsize=7, ha='center')
    ax2.set_title('2. Restoration with Multiple Doses')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # 3. Opiate
    ax3 = plt.subplot(6, 4, 3)
    for node in range(n_nodes):
        ax3.plot(t, qA[node, :], color=colors[node], lw=1.5, ls='--')
    ax3.set_title('3. Opiate (A) Concentrations')
    ax3.legend([f'Node {i}' for i in range(n_nodes)], fontsize=7)
    ax3.grid(True, alpha=0.3)

    # 4. Norcain
    ax4 = plt.subplot(6, 4, 4)
    for node in range(n_nodes):
        ax4.plot(t, qB[node, :], color=colors[node], lw=2)
    for dt in dose_times:
        ax4.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax4.set_title('4. Norcain (B) Spreading')
    ax4.legend([f'Node {i}' for i in range(n_nodes)], fontsize=7)
    ax4.grid(True, alpha=0.3)

    # 5. HH¹
    ax5 = plt.subplot(6, 4, 5)
    ax5.plot(t, HH1, 'b-', lw=2)
    for dt in dose_times:
        ax5.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax5.set_title('5. HH¹ - Deformations')
    ax5.grid(True, alpha=0.3)

    # 6. HH²
    ax6 = plt.subplot(6, 4, 6)
    ax6.plot(t, HH2, 'r-', lw=2)
    for dt in dose_times:
        ax6.axvline(x=dt, color='green', ls=':', alpha=0.7)
    peaks, _ = find_peaks(HH2, height=0.5)
    ax6.scatter(t[peaks], HH2[peaks], color='red', s=50, zorder=5, label='Phase Transitions')
    ax6.set_title('6. HH² - Obstructions')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)

    # 7. Coherence over time (average pairwise correlation)
    ax7 = plt.subplot(6, 4, 7)
    window = 100
    t_centers = []
    mean_coh = []
    for i in range(window, len(t)-window, window//2):
        seg = C[:, i-window:i+window]  # shape (n_nodes, 2*window)
        corr_mat = np.corrcoef(seg)
        # average off-diagonal
        triu = np.triu_indices(n_nodes, k=1)
        mean_coh.append(np.mean(corr_mat[triu]))
        t_centers.append(t[i])
    ax7.plot(t_centers, mean_coh, 'b-', label='Mean coherence')
    ax7.axhline(y=0.85, color='gold', ls='--', alpha=0.7)
    for dt in dose_times:
        ax7.axvline(x=dt, color='green', ls=':', alpha=0.5)
    ax7.set_title('7. Mean Coherence Over Time')
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0,1)

    # 8. Consciousness with HH² overlay
    ax8 = plt.subplot(6, 4, 8)
    mean_C = np.mean(C, axis=0)
    ax8.plot(t, mean_C, 'k-', label='Mean C')
    ax8.fill_between(t, 0, threshold, alpha=0.3, color='red')
    ax8_twin = ax8.twinx()
    ax8_twin.plot(t, HH2, 'r-', alpha=0.7, label='HH²')
    ax8_twin.set_ylabel('HH²', color='red')
    ax8_twin.tick_params(axis='y', labelcolor='red')
    for dt in dose_times:
        ax8.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax8.set_title('8. Phase Transitions with HH²')
    ax8.legend(loc='upper left')
    ax8.grid(True, alpha=0.3)

    # 9. Plücker 3D (same as before)
    ax9 = fig.add_subplot(6, 4, 9, projection='3d')
    # After creating ax9, add a wireframe sphere (approximation of the quadric)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = 0.8 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0.8 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0.8 * np.outer(np.ones_like(u), np.cos(v))
    ax9.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)
    norm_time = plt.Normalize(vmin=t[0], vmax=t[-1])
    colors_time = plt.cm.viridis(norm_time(t))
    for i in range(len(plucker)-1):
        ax9.plot(plucker[i:i+2,0], plucker[i:i+2,1], plucker[i:i+2,2],
                color=colors_time[i], lw=1, alpha=0.7)
    ax9.scatter(plucker[0,0], plucker[0,1], plucker[0,2], c='green', s=50)
    ax9.scatter(plucker[-1,0], plucker[-1,1], plucker[-1,2], c='red', s=50)
    ax9.set_title('9. Plücker Trajectory')

    # 10. Plücker relation
    ax10 = plt.subplot(6, 4, 10)
    plucker_rel = plucker[:,0]*plucker[:,5] - plucker[:,1]*plucker[:,4] + plucker[:,2]*plucker[:,3]
    ax10.plot(t, plucker_rel, 'g-')
    ax10.axhline(y=0, color='black')
    ax10.set_title('10. Klein Quadric Verification')
    ax10.grid(True, alpha=0.3)

    # 11. Phase space Node 0
    ax11 = plt.subplot(6, 4, 11)
    dC0 = np.gradient(C[0,:], dt_val)
    ax11.plot(C[0,:], dC0, 'b-', alpha=0.7)
    ax11.scatter(C[0,0], dC0[0], c='green', s=50)
    ax11.scatter(C[0,-1], dC0[-1], c='red', s=50)
    ax11.set_title('11. Phase Space Node 0')
    ax11.grid(True, alpha=0.3)

    # 12. Phase space Node 1 with HH² color
    ax12 = plt.subplot(6, 4, 12)
    dC1 = np.gradient(C[1,:], dt_val)
    ax12.plot(C[1,:], dC1, 'g-', alpha=0.7)
    sc = ax12.scatter(C[1,::20], dC1[::20], c=HH2[::20], cmap='hot', s=30, alpha=0.7)
    ax12.scatter(C[1,0], dC1[0], c='green', s=50)
    ax12.scatter(C[1,-1], dC1[-1], c='red', s=50)
    ax12.set_title('12. Phase Space Node 1 (color=HH²)')
    plt.colorbar(sc, ax=ax12, label='HH²')
    ax12.grid(True, alpha=0.3)

    # 13. Wavelet scalogram
    ax13 = plt.subplot(6, 4, 13)
    signal = C[0,:]
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    scales = np.arange(2, 64)
    widths = scales / dt_val
    valid = widths >= 1
    valid_scales = scales[valid]
    valid_widths = widths[valid]
    try:
        import pywt
        coeffs, _ = pywt.cwt(signal_norm, valid_scales, 'morl', dt_val)
        im = ax13.imshow(np.abs(coeffs), aspect='auto', cmap='hot',
                        extent=[t[0], t[-1], valid_scales[0], valid_scales[-1]])
    except:
        from scipy.signal import cwt, morlet
        def wavelet_wrapper(width, t_points):
            M = len(t_points) if hasattr(t_points, '__len__') else int(width*6)
            sigma = max(width/6.0, 1.0)
            return morlet(M, w=5.0, s=sigma, complete=True).real
        coeffs = cwt(signal_norm, wavelet_wrapper, valid_widths)
        im = ax13.imshow(np.abs(coeffs), aspect='auto', cmap='hot',
                        extent=[t[0], t[-1], valid_scales[0], valid_scales[-1]])
    ax13.set_title('13. Wavelet Scalogram')
    plt.colorbar(im, ax=ax13, label='|Coeff|')
    
    # 14. Wavelet energy
    ax14 = plt.subplot(6, 4, 14)
    energy = np.sum(np.abs(coeffs)**2, axis=1)
    energy = energy / (np.max(energy)+1e-8)
    ax14.plot(valid_scales, energy, 'b-')
    ax14.fill_between(valid_scales, energy, alpha=0.3)
    ax14.set_title('14. Wavelet Energy')
    ax14.grid(True, alpha=0.3)

    # 15. Coherence 0-1 (still useful, but you can also show first two nodes)
    ax15 = plt.subplot(6, 4, 15)
    try:
        from scipy.signal import coherence
        f, coh = coherence(C[0,:], C[1,:], fs=1/dt_val)
        ax15.semilogy(f[1:], coh[1:], 'b-')
        ax15.set_ylim(0,1)
        ax15.set_title('15. Coherence Node 0-1')
    except:
        ax15.text(0.5,0.5,'failed',ha='center',va='center')
    ax15.grid(True, alpha=0.3)

    # 16. Coherence 1-2
    ax16 = plt.subplot(6, 4, 16)
    try:
        f, coh = coherence(C[1,:], C[2,:], fs=1/dt_val)
        ax16.semilogy(f[1:], coh[1:], 'g-')
        ax16.set_ylim(0,1)
        ax16.set_title('16. Coherence Node 1-2')
    except:
        ax16.text(0.5,0.5,'failed',ha='center',va='center')
    ax16.grid(True, alpha=0.3)

    # 17. Unconscious duration (per node)
    ax17 = plt.subplot(6, 4, 17)
    time_below = []
    for node in range(n_nodes):
        below = C[node,:] < threshold
        time_below.append(np.sum(below) * dt_val)
    ax17.bar([f'Node {i}' for i in range(n_nodes)], time_below, color=colors, alpha=0.7)
    ax17.set_title('17. Unconscious Duration')
    ax17.grid(True, alpha=0.3)

    # 18. Final consciousness
    ax18 = plt.subplot(6, 4, 18)
    final_C = C[:, -1]
    ax18.bar([f'Node {i}' for i in range(n_nodes)], final_C, color=colors, alpha=0.7)
    ax18.axhline(y=threshold, color='red', ls='--')
    ax18.set_title('18. Final Consciousness')
    ax18.set_ylim(0,1)
    ax18.grid(True, alpha=0.3)

    # 19. Siegel & Milnor (unchanged)
    ax_ghost = plt.subplot(6, 4, 19)
    if hasattr(dynamics, 'ghost_signal') and dynamics.ghost_signal is not None:
        ax_ghost.plot(t, dynamics.ghost_signal, color='gold', lw=3, label='Siegel Lock (Ghost)')
    if hasattr(dynamics, 'milnor'):
        snap_times = [s[0] for s in dynamics.milnor.singularities]
        if snap_times:
            ax_ghost.scatter(snap_times, [0.6]*len(snap_times), color='red', s=15, label='Milnor Snaps', zorder=5)
    ax_ghost.set_title('19. Siegel Lock & Milnor Sequestration')
    ax_ghost.legend(fontsize=7)
    ax_ghost.grid(True, alpha=0.3)

    # 20. Final coherence (average over all pairs)
    ax20 = plt.subplot(6, 4, 20)
    last_seg = C[:, -500:] if C.shape[1] >= 500 else C
    corr_mat = np.corrcoef(last_seg)
    triu = np.triu_indices(n_nodes, k=1)
    final_coh = np.mean(corr_mat[triu])
    ax20.bar(['Mean coherence'], [final_coh], color='blue', alpha=0.7)
    ax20.axhline(y=0.85, color='gold', ls='--')
    ax20.set_title('20. Final Mean Coherence')
    ax20.set_ylim(0,1)
    ax20.grid(True, alpha=0.3)

    # 21. HH² vs Consciousness (scatter)
    ax21 = plt.subplot(6, 4, 21)
    mean_C = np.mean(C, axis=0)
    ax21.scatter(mean_C, HH2, c=HH2, cmap='hot', alpha=0.5, s=20)
    trans_counts = dynamics.count_phase_transitions()
    text = "Phase transitions:\n"
    for node in range(n_nodes):
        text += f"Node {node}: {trans_counts[node]['down']}↓ {trans_counts[node]['up']}↑\n"
    ax21.text(0.05, 0.95, text, transform=ax21.transAxes, fontsize=8,
              verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax21.set_title('21. HH² vs Consciousness')
    ax21.grid(True, alpha=0.3)

    plt.suptitle(f'{n_nodes}-PANEL DASHBOARD: Dynamic Consciousness', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: comprehensive_dashboard.png")

def create_dashboard_3Nodes(dynamics):
    """Create the 21-panel dashboard with Siegel and Milnor integration."""
    t, C, qA, qB, HH1, HH2 = dynamics.t, dynamics.C, dynamics.qA, dynamics.qB, dynamics.HH1, dynamics.HH2
    plucker = dynamics.plucker
    dose_times = dynamics.dose_times
    threshold = dynamics.threshold
    
    # Expanded grid to 6x4 to accommodate 21+ panels
    fig = plt.figure(figsize=(22, 32))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    dt_val = t[1] - t[0]
    
    # 1. Consciousness across all nodes
    ax1 = plt.subplot(6, 4, 1)
    for node in range(3):
        ax1.plot(t, C[node, :], color=colors[node], lw=2, label=f'Node {node}')
    ax1.axhline(y=threshold, color='red', ls='--')
    for dt in dose_times:
        ax1.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax1.set_title('1. Consciousness Across All Nodes')
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # 2. Restoration with multiple doses
    ax2 = plt.subplot(6, 4, 2)
    for node in range(3):
        ax2.plot(t, C[node, :], color=colors[node], lw=2)
    ax2.fill_between(t, 0, threshold, alpha=0.3, color='red', label='Unconscious')
    ax2.fill_between(t, threshold, 1, alpha=0.2, color='green', label='Conscious')
    for i, dt in enumerate(dose_times):
        ax2.axvline(x=dt, color='green', ls=':', alpha=0.7)
        ax2.annotate(f'Dose {i+1}', xy=(dt, 0.85), xytext=(dt, 0.92),
                    arrowprops=dict(arrowstyle='->', color='green'), fontsize=7, ha='center')
    ax2.set_title('2. Restoration with Multiple Doses')
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # 3. Opiate
    ax3 = plt.subplot(6, 4, 3)
    for node in range(3):
        ax3.plot(t, qA[node, :], color=colors[node], lw=1.5, ls='--')
    ax3.set_title('3. Opiate (A) Concentrations')
    ax3.legend(['Node 0','Node 1','Node 2'], fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # 4. Norcain
    ax4 = plt.subplot(6, 4, 4)
    for node in range(3):
        ax4.plot(t, qB[node, :], color=colors[node], lw=2)
    for dt in dose_times:
        ax4.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax4.set_title('4. Norcain (B) Spreading')
    ax4.legend(['Node 0','Node 1','Node 2'], fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    # 5. HH¹
    ax5 = plt.subplot(6, 4, 5)
    ax5.plot(t, HH1, 'b-', lw=2)
    for dt in dose_times:
        ax5.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax5.set_title('5. HH¹ - Deformations')
    ax5.grid(True, alpha=0.3)
    
    # 6. HH²
    ax6 = plt.subplot(6, 4, 6)
    ax6.plot(t, HH2, 'r-', lw=2)
    for dt in dose_times:
        ax6.axvline(x=dt, color='green', ls=':', alpha=0.7)
    peaks, _ = find_peaks(HH2, height=0.5)
    ax6.scatter(t[peaks], HH2[peaks], color='red', s=50, zorder=5, label='Phase Transitions')
    ax6.set_title('6. HH² - Obstructions')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)
    
    # 7. Coherence over time
    ax7 = plt.subplot(6, 4, 7)
    window = 100
    coh01, coh12, coh02, t_centers = [], [], [], []
    for i in range(window, len(t)-window, window//2):
        c0 = C[0, i-window:i+window]
        c1 = C[1, i-window:i+window]
        c2 = C[2, i-window:i+window]
        coh01.append(np.abs(np.corrcoef(c0,c1)[0,1]))
        coh12.append(np.abs(np.corrcoef(c1,c2)[0,1]))
        coh02.append(np.abs(np.corrcoef(c0,c2)[0,1]))
        t_centers.append(t[i])
    ax7.plot(t_centers, coh01, 'b-', label='0-1')
    ax7.plot(t_centers, coh12, 'g-', label='1-2')
    ax7.plot(t_centers, coh02, 'r-', label='0-2')
    ax7.axhline(y=0.85, color='gold', ls='--', alpha=0.7)
    for dt in dose_times:
        ax7.axvline(x=dt, color='green', ls=':', alpha=0.5)
    ax7.set_title('7. Coherence Over Time')
    ax7.legend(fontsize=7)
    ax7.grid(True, alpha=0.3)
    ax7.set_ylim(0,1)
    
    # 8. Consciousness with HH² overlay
    ax8 = plt.subplot(6, 4, 8)
    ax8.plot(t, np.mean(C, axis=0), 'k-', label='Mean C')
    ax8.fill_between(t, 0, threshold, alpha=0.3, color='red')
    ax8_twin = ax8.twinx()
    ax8_twin.plot(t, HH2, 'r-', alpha=0.7, label='HH²')
    ax8_twin.set_ylabel('HH²', color='red')
    ax8_twin.tick_params(axis='y', labelcolor='red')
    for dt in dose_times:
        ax8.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax8.set_title('8. Phase Transitions with HH²')
    ax8.legend(loc='upper left')
    ax8.grid(True, alpha=0.3)
    
    # 9. Plücker 3D
    ax9 = fig.add_subplot(6, 4, 9, projection='3d')
    # After creating ax9, add a wireframe sphere (approximation of the quadric)
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = 0.8 * np.outer(np.cos(u), np.sin(v))
    y_sphere = 0.8 * np.outer(np.sin(u), np.sin(v))
    z_sphere = 0.8 * np.outer(np.ones_like(u), np.cos(v))
    ax9.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)
    norm_time = plt.Normalize(vmin=t[0], vmax=t[-1])
    colors_time = plt.cm.viridis(norm_time(t))
    for i in range(len(plucker)-1):
        ax9.plot(plucker[i:i+2,0], plucker[i:i+2,1], plucker[i:i+2,2],
                color=colors_time[i], lw=1, alpha=0.7)
    ax9.scatter(plucker[0,0], plucker[0,1], plucker[0,2], c='green', s=50)
    ax9.scatter(plucker[-1,0], plucker[-1,1], plucker[-1,2], c='red', s=50)
    ax9.set_title('9. Plücker Trajectory')
    
    # 10. Plücker relation
    ax10 = plt.subplot(6, 4, 10)
    plucker_rel = plucker[:,0]*plucker[:,5] - plucker[:,1]*plucker[:,4] + plucker[:,2]*plucker[:,3]
    ax10.plot(t, plucker_rel, 'g-')
    ax10.axhline(y=0, color='black')
    ax10.set_title('10. Klein Quadric Verification')
    ax10.grid(True, alpha=0.3)
    
    # 11. Phase space Node 0
    ax11 = plt.subplot(6, 4, 11)
    dC0 = np.gradient(C[0,:], dt_val)
    ax11.plot(C[0,:], dC0, 'b-', alpha=0.7)
    ax11.scatter(C[0,0], dC0[0], c='green', s=50)
    ax11.scatter(C[0,-1], dC0[-1], c='red', s=50)
    ax11.set_title('11. Phase Space Node 0')
    ax11.grid(True, alpha=0.3)
    
    # 12. Phase space Node 1 with HH² color
    ax12 = plt.subplot(6, 4, 12)
    dC1 = np.gradient(C[1,:], dt_val)
    ax12.plot(C[1,:], dC1, 'g-', alpha=0.7)
    sc = ax12.scatter(C[1,::20], dC1[::20], c=HH2[::20], cmap='hot', s=30, alpha=0.7)
    ax12.scatter(C[1,0], dC1[0], c='green', s=50)
    ax12.scatter(C[1,-1], dC1[-1], c='red', s=50)
    ax12.set_title('12. Phase Space Node 1 (color=HH²)')
    plt.colorbar(sc, ax=ax12, label='HH²')
    ax12.grid(True, alpha=0.3)
    
    # 13. Wavelet scalogram
    ax13 = plt.subplot(6, 4, 13)
    signal = C[0,:]
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    scales = np.arange(2, 64)
    widths = scales / dt_val
    valid = widths >= 1
    valid_scales = scales[valid]
    valid_widths = widths[valid]
    try:
        import pywt
        coeffs, _ = pywt.cwt(signal_norm, valid_scales, 'morl', dt_val)
        im = ax13.imshow(np.abs(coeffs), aspect='auto', cmap='hot',
                        extent=[t[0], t[-1], valid_scales[0], valid_scales[-1]])
    except:
        from scipy.signal import cwt, morlet
        def wavelet_wrapper(width, t_points):
            M = len(t_points) if hasattr(t_points, '__len__') else int(width*6)
            sigma = max(width/6.0, 1.0)
            return morlet(M, w=5.0, s=sigma, complete=True).real
        coeffs = cwt(signal_norm, wavelet_wrapper, valid_widths)
        im = ax13.imshow(np.abs(coeffs), aspect='auto', cmap='hot',
                        extent=[t[0], t[-1], valid_scales[0], valid_scales[-1]])
    ax13.set_title('13. Wavelet Scalogram')
    plt.colorbar(im, ax=ax13, label='|Coeff|')
    
    # 14. Wavelet energy
    ax14 = plt.subplot(6, 4, 14)
    energy = np.sum(np.abs(coeffs)**2, axis=1)
    energy = energy / (np.max(energy)+1e-8)
    ax14.plot(valid_scales, energy, 'b-')
    ax14.fill_between(valid_scales, energy, alpha=0.3)
    ax14.set_title('14. Wavelet Energy')
    ax14.grid(True, alpha=0.3)
    
    # 15. Coherence 0-1
    ax15 = plt.subplot(6, 4, 15)
    try:
        from scipy.signal import coherence
        f, coh = coherence(C[0,:], C[1,:], fs=1/dt_val)
        ax15.semilogy(f[1:], coh[1:], 'b-')
        ax15.set_ylim(0,1)
        ax15.set_title('15. Coherence Node 0-1')
    except:
        ax15.text(0.5,0.5,'failed',ha='center',va='center')
    ax15.grid(True, alpha=0.3)
    
    # 16. Coherence 1-2
    ax16 = plt.subplot(6, 4, 16)
    try:
        f, coh = coherence(C[1,:], C[2,:], fs=1/dt_val)
        ax16.semilogy(f[1:], coh[1:], 'g-')
        ax16.set_ylim(0,1)
        ax16.set_title('16. Coherence Node 1-2')
    except:
        ax16.text(0.5,0.5,'failed',ha='center',va='center')
    ax16.grid(True, alpha=0.3)
    
    # 17. Unconscious duration
    ax17 = plt.subplot(6, 4, 17)
    time_below = []
    for node in range(3):
        below = C[node,:] < threshold
        time_below.append(np.sum(below) * dt_val)
    ax17.bar(['Node 0','Node 1','Node 2'], time_below, color=colors, alpha=0.7)
    ax17.set_title('17. Unconscious Duration')
    ax17.grid(True, alpha=0.3)
    
    # 18. Final consciousness
    ax18 = plt.subplot(6, 4, 18)
    final_C = C[:, -1]
    node_labels = dynamics.region_names # [f'Node {i}' for i in range(dynamics.n_nodes)]
    ax18.bar(node_labels, final_C, color=plt.cm.viridis(np.linspace(0,1,dynamics.n_nodes)), alpha=0.7)
    ax18.axhline(y=threshold, color='red', ls='--')
    ax18.set_title('18. Final Consciousness')
    ax18.set_ylim(0,1)
    ax18.grid(True, alpha=0.3)
    
    # 19. Siegel & Milnor (FIXED: access via 'dynamics')
    ax_ghost = plt.subplot(6, 4, 19) 
    if hasattr(dynamics, 'ghost_signal') and dynamics.ghost_signal is not None:
        ax_ghost.plot(t, dynamics.ghost_signal, color='gold', lw=3, label='Siegel Lock (Ghost)')
    
    # Plot Milnor Nodes as red 'Snap' markers
    if hasattr(dynamics, 'milnor'):
        snap_times = [s[0] for s in dynamics.milnor.singularities]
        if snap_times:
            ax_ghost.scatter(snap_times, [0.6]*len(snap_times), color='red', s=15, label='Milnor Snaps', zorder=5)
    
    ax_ghost.set_title('19. Siegel Lock & Milnor Sequestration')
    ax_ghost.legend(fontsize=7)
    ax_ghost.grid(True, alpha=0.3)
    
    # 20. Final coherence
    ax20 = plt.subplot(6, 4, 20)
    final_coherence = [
        np.abs(np.corrcoef(C[0,-500:], C[1,-500:])[0,1]),
        np.abs(np.corrcoef(C[0,-500:], C[2,-500:])[0,1]),
        np.abs(np.corrcoef(C[1,-500:], C[2,-500:])[0,1])
    ]
    ax20.bar(['0-1','0-2','1-2'], final_coherence, color=['blue','red','green'], alpha=0.7)
    ax20.axhline(y=0.85, color='gold', ls='--')
    ax20.set_title('20. Final Coherence')
    ax20.set_ylim(0,1)
    ax20.grid(True, alpha=0.3)
    
    # 21. HH² vs Consciousness
    ax21 = plt.subplot(6, 4, 21)
    mean_C = np.mean(C, axis=0)
    ax21.scatter(mean_C, HH2, c=HH2, cmap='hot', alpha=0.5, s=20)
    
    trans_counts = dynamics.count_phase_transitions()
    text = "Phase transitions:\n"
    for node in range(3):
        text += f"Node {node}: {trans_counts[node]['down']}↓ {trans_counts[node]['up']}↑\n"
    ax21.text(0.05, 0.95, text, transform=ax21.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax21.set_title('21. HH² vs Consciousness')
    ax21.grid(True, alpha=0.3)
    
    plt.suptitle('21-PANEL DASHBOARD: Node 1 & 2 Show Dynamic Consciousness!', fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig('comprehensive_21panel_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: comprehensive_21panel_dashboard.png")


def plot_ghost_signal(dynamics):
    """Plot ghost signal: monodromy product and HH² persistence."""
    t = dynamics.t
    HH2 = dynamics.HH2
    dose_times = dynamics.dose_times
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: HH² with monodromy product annotation
    ax1.plot(t, HH2, 'r-', lw=2, label='HH²')
    for dt in dose_times:
        ax1.axvline(x=dt, color='green', ls=':', alpha=0.7)
    # Find times where HH² is low but monodromy persists (simplified)
    low_hh2 = np.where(HH2 < 0.2)[0]
    if len(low_hh2) > 0:
        ax1.scatter(t[low_hh2[::50]], HH2[low_hh2[::50]], color='blue', s=30, label='Ghost signal candidates')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('HH²')
    ax1.set_title('Ghost Signal: HH² near zero but monodromy persists')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right: Dehn twist product (coherence)
    is_coh, err = dynamics.verify_coherence()
    ax2.text(0.5, 0.5, f"Coherence condition: T₁∘T₂∘T₃ = I\nError = {err:.2e}\n{'✓ COHERENT' if is_coh else '✗ INCOHERENT'}",
             ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_title('Monodromy Product (Dehn Twist Factorization)')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('ghost_signal.png', dpi=150)
    plt.close()
    print("    ✓ Saved: ghost_signal.png")


def plot_prime_zeta(dynamics):
    """Plot prime zeta values on complex plane and identify zeros."""
    zeta_vals = dynamics.compute_prime_zeta()
    if len(zeta_vals) == 0:
        print("  No prime paths found.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    for z in zeta_vals:
        ax.scatter(z.real, z.imag, c='blue', s=60, alpha=0.7)
    # Mark those with |ζ| < 0.3 as prime (on critical line)
    prime_vals = [z for z in zeta_vals if abs(z) < 0.3]
    if prime_vals:
        ax.scatter([z.real for z in prime_vals], [z.imag for z in prime_vals],
                   c='red', s=100, marker='*', label='Prime paths (|ζ|<0.3)')
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_xlabel('Re(ζ)')
    ax.set_ylabel('Im(ζ)')
    ax.set_title('Prime Zeta of Paths (zeros on critical line)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig('prime_zeta.png', dpi=150)
    plt.close()
    print("    ✓ Saved: prime_zeta.png")

# Incremental 7 step REES Blow uo resolution
class ReesBlowUp:
    @staticmethod
    def resolve_singularity(model, t_idx):
        print(f"    [Rees Blow-up] Resolving singularity at t={model.t[t_idx]:.2f}")
        # Inject norcain (existing)
        model.qB[0, t_idx+1] += model.dose_amount * 1.5

        # ---- NEW: Generate blow‑up diagram for the region with lowest consciousness at this time ----
        min_region_idx = np.argmin(model.C[:, t_idx])
        region_name = model.region_names[min_region_idx]
        print(f"    -> Generating blow‑up diagram for region {region_name} (most affected)")

        # Build current edge weights
        current_weights = {}
        for (u, v) in model.edges:
            base = model.edge_weights.get((u, v), 1.0)
            current_weights[(u, v)] = base

        # --------------------------------------------------------------
        # Save Plücker coordinates for this blow‑up event
        # --------------------------------------------------------------
        if hasattr(model, 'plucker') and model.plucker is not None:
            q = model.plucker.tolist()  # model.plucker is a numpy array
            # Append to model's history lists
            if not hasattr(model, 'plucker_history'):
                model.plucker_history = []
                model.recompute_step_indices = []
            model.plucker_history.append((model.t[t_idx], t_idx, q))
            model.recompute_step_indices.append(t_idx)

            # --- Compute and record Plücker zeta magnitude at blow‑up ---
            # Use the L2 norm of the Plücker vector as a simple zeta proxy.
            # (You can replace with a more sophisticated measure if you have compute_single_zeta defined.)
            zeta_mag = np.linalg.norm(model.plucker)
            if not hasattr(model, 'prime_zeta_values'):
                model.prime_zeta_values = []
                model.prime_zeta_times = []
            model.prime_zeta_values.append(zeta_mag)
            model.prime_zeta_times.append(model.t[t_idx])

        # Call Julia with --full mode
        try:
            model.call_julia_ainf(current_weights, region_name=region_name,  step_index=t_idx)
            print(f"    -> Blow‑up diagram saved for region {region_name}")
        except Exception as e:
            print(f"    -> Failed to generate blow‑up diagram: {e}")

        return True
"""
class ReesBlowUp:
    # Handles the algebraic 'inflation' at a singularity.
    @staticmethod
    def resolve_singularity(model, t_idx):
        print(f"    [Rees Blow-up] Resolving singularity at t={model.t[t_idx]:.2f}")
        # Inject norcain (existing)
        model.qB[0, t_idx+1] += model.dose_amount * 1.5

        # ---- NEW: Generate blow‑up diagram for the region with lowest consciousness at this time ----
        # Find the region with minimal consciousness at this time step
        min_region_idx = np.argmin(model.C[:, t_idx])
        region_name = model.region_names[min_region_idx]
        print(f"    -> Generating blow‑up diagram for region {region_name} (most affected)")

        # Build current edge weights (use free concentrations as in the A∞ recomputation loop)
        current_weights = {}
        for (u, v) in model.edges:
            base = model.edge_weights.get((u, v), 1.0)
            # Optional: modulate with free concentrations (same as in simulate)
            # For simplicity, just use base weights
            current_weights[(u, v)] = base

        # --------------------------------------------------------------
        # Save Plucker for Associahedron Tube coupling
        # --------------------------------------------------------------
        q = self.plucker.copy() if hasattr(self, 'plucker') else None
        if q is not None:
            self.plucker_history.append((t[i+1], i+1, q.tolist()))
            self.recompute_step_indices.append(i+1)

        # Call Julia with --full mode
        try:
            model.call_julia_ainf(current_weights, region_name=region_name)
            print(f"    -> Blow‑up diagram saved for region {region_name}")
        except Exception as e:
            print(f"    -> Failed to generate blow‑up diagram: {e}")

        return True
"""
class SearchNavigator:
    """
    Implements a BFS-based search for the most unstable path (highest HH2 + Plucker residue).
    Includes culling (blow-down) and Rees blow-up (resolution).
    """
    def __init__(self, dynamics_model):
        self.model = dynamics_model
        self.visited = {} # DP table for culling: (t_idx, state_hash) -> instability
        self.restriction_horizon = 20 # 10-20 steps lookahead
        self.unstable_paths = []

    def get_state_hash(self, t_idx):
        # Normal Form equivalent for culling: round the Plucker coords
        coords = np.round(self.model.plucker[t_idx], 2)
        return hash(coords.tobytes())

    def get_instability(self, t_idx):
        # Cost Function: HH2 spike + Plucker residue
        p = self.model.plucker[t_idx]
        rel = abs(p[0]*p[5] - p[1]*p[4] + p[2]*p[3])
        return self.model.HH2[t_idx] + rel

    def find_most_unstable_path(self, start_t_idx):
        queue = deque([(start_t_idx, 0, [])])
        max_instability = -1
        best_path = []

        while queue:
            idx, depth, path = queue.popleft()
            if depth >= self.restriction_horizon or idx >= len(self.model.t) - 1:
                continue

            # Instability metric
            cost = self.get_instability(idx)
            current_path = path + [idx]

            # Track the peak instability
            if cost > max_instability:
                max_instability = cost
                best_path = current_path

            # Blow-down (Culling): If we've seen this state with lower cost, stop.
            state_hash = self.get_state_hash(idx)
            if state_hash in self.visited and self.visited[state_hash] >= cost:
                continue
            self.visited[state_hash] = cost

            # Move forward (BFS step)
            queue.append((idx + 1, depth + 1, current_path))

        return best_path, max_instability

def plot_unstable_paths(dynamics, navigator, unstable_path):
    """Visualization of the unstable path on the Plucker trajectory."""
    t = dynamics.t
    plucker = dynamics.plucker
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Base Trajectory
    ax.plot(plucker[:,0], plucker[:,1], plucker[:,2], color='gray', alpha=0.3, label='Full Trajectory')
    
    # Highlight Unstable Path
    p_unstable = plucker[unstable_path]
    ax.plot(p_unstable[:,0], p_unstable[:,1], p_unstable[:,2], color='red', lw=3, label='Most Unstable Path')
    
    # Mark the Resolution Point (where HH2 peaks)
    peak_idx = unstable_path[np.argmax(dynamics.HH2[unstable_path])]
    ax.scatter(plucker[peak_idx,0], plucker[peak_idx,1], plucker[peak_idx,2], 
               color='gold', s=200, marker='*', label='Rees Blow-up (Resolution Point)')

    ax.set_title('Unstable Path Exploration via SearchNavigator')
    ax.legend()
    plt.savefig('unstable_path_search.png', dpi=150)
    print("    ✓ Saved: unstable_path_search.png")

# reverse map wavelets to schubert cell torics (2) top left and bottom right one for Opiate and other for norcain.
class ReverseHironakaMoleculeResolver:
    """
    Resolves the 28-arrow quiver states into a dynamic harmonic trajectory.
    Maps 6-vertex Plücker coordinates to a 3x3 Schubert Configuration.
    """
    def __init__(self):
        pass

    def apply_hironaka_resolution_full(self, states_history, t_q):
        """
        Processes the full history of the 28-arrow quiver to reconstruct 
        the dynamic Hironaka trajectory a(t)cos(wt) + b(t)sin(wt).
        """
        resolved_signal = np.zeros(len(t_q))
        
        for i in range(len(t_q)):
            # 1. Get the 6-vertex state at this moment
            s = states_history[i]
            
            # 2. Map 6 vertices to 3x3 Schubert Matrix (The 'White Box' Pivot)
            matrix_3x3 = np.array([
                [s[0], s[1], 0.0],
                [s[1], s[2], s[3]],
                [0.0,  s[3], s[4]]
            ])
            
            # 3. Extract Instantaneous Frequencies from Eigenvalues
            evals = np.linalg.eigvals(matrix_3x3)
            # We look for the imaginary component for the oscillatory frequency
            freqs = np.sort(np.abs(np.imag(evals)))[::-1]
            w_instant = freqs[0] if len(freqs) > 0 else 1.0 # Default to 1.0 if real
            
            # 4. Extract Dynamic Amplitudes (Opiate and Norcain densities)
            a_t = s[0] # Molecule A
            b_t = s[4] # Molecule B
            
            # 5. Project onto the Trajectory (Reverse Hironaka Geodesic)
            resolved_signal[i] = a_t * np.cos(w_instant * t_q[i]) + b_t * np.sin(w_instant * t_q[i])
            
        return resolved_signal
# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*100)
    print(" " * 35 + "FINAL: GHOST SIGNAL & PRIME ZETA INCLUDED")
    print("="*100)
    
    dynamics = FullGraphDynamics()
    #nodes_df = pd.read_csv(NODES_FILE)   # adjust path
    #print("Columns in nodes_df:", nodes_df.columns.tolist())
    #coords = nodes_df[['pos_x', 'pos_y', 'pos_z']].values
    #dynamics.node_coords = coords
    dynamics.load_brain_mesh(BRAIN_FILE)
    t_span = (0, 800)  # Changed from (0, 25) or similar
    dt = 0.02  # Keep same timestep for resolution
    t, C, qA, qB, HH1, HH2 = dynamics.simulate(t_span=t_span, dt=dt)
    t_full, C_full, qA_full, qB_full, HH1_full, HH2_full = t, C, qA, qB, HH1, HH2

    n_nodes = dynamics.n_nodes

    # Hironaka Singularity Resolution -- PROTOCOL 7 Steps
    # Singularity index
    idx_254 = np.argmin(np.abs(t_full - 2.54))
    idx_604 = np.argmin(np.abs(t_full - 6.04))

    # Prepare Lookahead Data
    lookahead_t = t_full[idx_254:idx_604+1]
    n_steps = len(lookahead_t)

    # Scenario A: No Blow-up (Algebraic Dead-end)
    # Without the 'blow-up' resolution, the system remains in a 
    # state of decay and high obstruction.
    C_no = np.zeros((n_nodes, n_steps))
    HH2_no = np.zeros(n_steps)

    # Starting state at t=2.54
    C_no[:, 0] = C_full[:, idx_254]
    HH2_no[0] = HH2_full[idx_254]

    # Step through and simulate the "spectral dead-end"
    # In this scenario, we don't have the "exceptional" generators (recovery dose).
    dt = 0.02 # Assuming dt is consistent with the simulation
    for i in range(1, n_steps):
        # Natural decay without recovery (simulated as the 'unresolved' branch)
        # consciousness decays toward zero as the 'gluing' is broken.
        for node in range(n_nodes):
            C_no[node, i] = C_no[node, i-1] * 0.99
        # HH2 grows or stays high because the obstruction is not cleared.
        HH2_no[i] = HH2_no[i-1] * 1.01

    # Scenario B: Resolved (The Actual Full Simulation)
    C_res = C_full[:, idx_254:idx_604+1]
    HH2_res = HH2_full[idx_254:idx_604+1]

    # Plotting the Comparison
    plt.figure(figsize=(12, 6))

    # Subplot 1: Consciousness Recovery
    plt.subplot(1, 2, 1)
    plt.plot(lookahead_t, np.mean(C_res, axis=0), 'b-', label='With Blow-up (Resolution)')
    plt.plot(lookahead_t, np.mean(C_no, axis=0), 'r--', label='No Blow-up (Dead-end)')
    plt.axvline(2.54, color='k', linestyle=':', label='Singularity (t=2.54)')
    plt.title('Consciousness Locus Resolution')
    plt.xlabel('Time (s)')
    plt.ylabel('Mean Consciousness')
    plt.legend()

    # Subplot 2: HH2 Obstruction
    plt.subplot(1, 2, 2)
    plt.plot(lookahead_t, HH2_res, 'b-', label='With Blow-up (Resolved)')
    plt.plot(lookahead_t, HH2_no, 'r--', label='No Blow-up (Obstructed)')
    plt.axvline(2.54, color='k', linestyle=':', label='Singularity (t=2.54)')
    plt.yscale('log')
    plt.title('HH2 Singular Locus Comparison')
    plt.xlabel('Time (s)')
    plt.ylabel('Obstruction Intensity (HH2)')
    plt.legend()

    plt.tight_layout()
    plt.savefig('blowup_lookahead_test.png')

    # Output data for user
    res_df = pd.DataFrame({
        'time': lookahead_t,
        'C_resolved': np.mean(C_res, axis=0),
        'C_obstructed': np.mean(C_no, axis=0),
        'HH2_resolved': HH2_res,
        'HH2_obstructed': HH2_no
    })
    res_df.to_csv('lookahead_resolution_data.csv', index=False)

    print("Lookahead Analysis Complete.")
    print(f"At t=6.04s:")
    print(f"  Resolved Consciousness: {res_df['C_resolved'].iloc[-1]:.4f}")
    print(f"  Obstructed Consciousness: {res_df['C_obstructed'].iloc[-1]:.4f}")
    print(f"  Resolved HH2: {res_df['HH2_resolved'].iloc[-1]:.4f}")
    print(f"  Obstructed HH2: {res_df['HH2_obstructed'].iloc[-1]:.4f}")

    # Now do resolution using 7 step and find most interesting section
    nav = SearchNavigator(dynamics)

    # Find most unstable path around the first collapse (t ~ 3.0)
    start_idx = np.argmin(np.abs(dynamics.t - 3.0))
    best_path, cost = nav.find_most_unstable_path(start_idx)

    # Resolve the peak of that path
    peak_idx = best_path[np.argmax(dynamics.HH2[best_path])]
    ReesBlowUp.resolve_singularity(dynamics, peak_idx)

    # Save Plücker trajectory
    # After ReesBlowUp.resolve_singularity(dynamics, peak_idx)
    if dynamics.plucker_history:
        plucker_dict = {
            "times": [float(p[0]) for p in dynamics.plucker_history],
            "steps": [int(p[1]) for p in dynamics.plucker_history],
            "q12": [float(np.array(p[2]).flatten()[0]) for p in dynamics.plucker_history],
            "q13": [float(np.array(p[2]).flatten()[1]) for p in dynamics.plucker_history],
            "q14": [float(np.array(p[2]).flatten()[2]) for p in dynamics.plucker_history],
            "q23": [float(np.array(p[2]).flatten()[3]) for p in dynamics.plucker_history],
            "q24": [float(np.array(p[2]).flatten()[4]) for p in dynamics.plucker_history],
            "q34": [float(np.array(p[2]).flatten()[5]) for p in dynamics.plucker_history],
            "klein_constraint": [
                 abs(float(np.array(p[2]).flatten()[0] * np.array(p[2]).flatten()[5]
                     - np.array(p[2]).flatten()[1] * np.array(p[2]).flatten()[4]
                     + np.array(p[2]).flatten()[2] * np.array(p[2]).flatten()[3]))
                 for p in dynamics.plucker_history
             ],
        }
        with open("plucker_trajectory.json", "w") as f:
            json.dump(plucker_dict, f)
        print("Saved Plücker trajectory to plucker_trajectory.json")

    if hasattr(dynamics, 'prime_zeta_values') and dynamics.prime_zeta_values:
        prime_zeta_data = {
            "values": [{"real": float(z), "imag": 0.0} for z in dynamics.prime_zeta_values],
            "transition_times": [float(t) for t in dynamics.prime_zeta_times]
        }
        with open("prime_zeta.json", "w") as f:
            json.dump(prime_zeta_data, f)
        print("Saved prime_zeta.json from blow‑up events ({} entries).".format(len(dynamics.prime_zeta_values)))

    dense_zeta_data = {
        "times": [float(t) for t in dynamics.plucker_zeta_times],
        "magnitudes": [float(m) for m in dynamics.plucker_zeta_mags],
        "snapshot_indices": [int(idx) for idx in dynamics.json_export_step_indices]
    }
    with open("plucker_zeta_dense.json", "w") as f:
        json.dump(dense_zeta_data, f)
    print("Saved dense Plücker zeta time series to plucker_zeta_dense.json")

    # Save prolate angle time series
    # theta_prolate is the angle between the prolate eigenvector and sAMY axis.
    # At equilibrium this should equal 1/120 = 0.00833 (finite-size correction).
    # Comparing theta_prolate to phi_equil - 1/2 from Bridgeland analysis
    # tests whether the physical and algebraic computations agree.
    if hasattr(dynamics, 'prolate_theta') and dynamics.prolate_theta:
        prolate_data = {
            "theta": dynamics.prolate_theta,
            "times": dynamics.prolate_theta_times,
            "mean_theta": float(np.mean(dynamics.prolate_theta)),
            "std_theta":  float(np.std(dynamics.prolate_theta)),
            "finite_size_prediction": 1/120,
            "deviation_from_prediction": float(
                abs(np.mean(dynamics.prolate_theta) - 1/120)),
            "n_regions": dynamics.n_nodes,
            "phi_equil_prediction": 0.5 + 1/120,
        }
        with open("prolate_theta.json", "w") as f:
            json.dump(prolate_data, f)
        print(f"Saved prolate_theta.json  "
              f"mean={prolate_data['mean_theta']:.6f}  "
              f"prediction=1/120={1/120:.6f}  "
              f"diff={prolate_data['deviation_from_prediction']:.6f}")
    # Save result
    plot_unstable_paths(dynamics, nav, best_path)

    # Build the sheaf
    sheaf = GraphSpectralSheaf(dynamics)

    # Compute eigenvalues and triple interactions over time
    n_t = len(t)
    second_eigenvalue = np.zeros(n_t)
    triple_heatmap = np.zeros((n_t, n_nodes, n_nodes))

    for i, ti in enumerate(t):
        evals = sheaf.eigenvalues(ti)
        second_eigenvalue[i] = evals[1] if len(evals) > 1 else 0
        triple_heatmap[i] = sheaf.triple_interaction_matrix(ti)

    # Plot spectral gap (second eigenvalue)
    plt.figure(figsize=(8,4))
    plt.plot(t, second_eigenvalue, 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Second eigenvalue')
    plt.title('Sheaf Laplacian spectral gap')
    plt.grid(True)
    plt.savefig('sheaf_gap.png')
    plt.show()

    # Plot average triple interaction
    avg_triple = np.mean(triple_heatmap, axis=0)
    plt.figure()
    plt.imshow(avg_triple, cmap='hot')
    plt.colorbar()
    plt.xticks([0,1,2], ['Node 0', 'Node 1', 'Node 2'])
    plt.yticks([0,1,2], ['Node 0', 'Node 1', 'Node 2'])
    plt.title('Average triple interaction (HH³ proxy)')
    plt.savefig('triple_interaction.png')
    plt.show()

    # Spectral clustering at a few times
    times = [0, 5, 10, 15]
    print("\nSpectral clustering (graph sheaf):")
    for ti in times:
        idx = np.argmin(np.abs(t - ti))
        labels = sheaf.spectral_clustering(t[idx])
        print(f"t={ti:.1f}s, clusters: {labels}")

    # This is Quiver Dynamics
    # 1. Build initial quiver state from the first simulation point
    initial_state = np.array([
        C[0,0] * (1 - qA[0,0]),
        (1 - C[0,0]) * (1 - qB[0,0]),
        1.0 + qA[0,0],
        2.0 + qB[0,0],
        np.arctan2(qB[0,0] - qA[0,0], qA[0,0] + qB[0,0]),
        qA[0,0] * qB[0,0] / ((qA[0,0] + qB[0,0])**2 + 1e-8)
    ])

    # 2. Simulate the quiver
    quiver = DynamicWaveletQuiver(dynamics)   # 'dynamics' is your simulation object
    t_q, states = quiver.simulate_quiver(initial_state, t_array=t)

    """
    # Get Eigen Values
    n_t = len(t_q)
    eigvals_all = np.zeros((n_t, 6))   # 6 eigenvalues per time step

    for i in range(n_t):
        evals = sheaf.eigenvalues(t_q[i], states[i])
        if not np.isnan(evals).any():
            eigvals_all[i] = evals
        else:
            eigvals_all[i] = np.nan   # or skip

    # Repeat the simulation and store for APL+LSX
    # in eigen_all_pal

    # Then compute invariant summary
    # Remove NaN rows
    valid = ~np.isnan(eigvals_all).any(axis=1)
    lambda2 = eigvals_all[valid, 1]   # second eigenvalue (index 1)
    mean_lambda2 = np.mean(lambda2)
    std_lambda2 = np.std(lambda2)

    # Same for PAL+LSX model
    lambda2_pal = eigvals_all_pal[valid_pal, 1]
    mean_lambda2_pal = np.mean(lambda2_pal)
    #If the means differ significantly (t‑test), the models are not equivalent.

    # Full Kolmogorov-Smirnov Test
    from scipy.stats import ks_2samp

    all_evals = eigvals_all[valid].flatten()
    all_evals_pal = eigvals_all_pal[valid_pal].flatten()

    ks_stat, p_value = ks_2samp(all_evals, all_evals_pal)
    print(f"KS test: statistic = {ks_stat:.4f}, p = {p_value:.4e}")

    # If p<0.05p<0.05, the distributions are different → the sheaf Laplacian 
    # spectra are not identical → the A∞‑algebras are not derived equivalent.

    # Plots
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Time series of λ₂
    t_valid = t_q[valid]
    ax1.plot(t_valid, lambda2, 'b-', label='6‑region')
    ax1.plot(t_valid_pal, lambda2_pal, 'r-', label='+PAL+LSX')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Second eigenvalue λ₂')
    ax1.set_title('Spectral gap of the sheaf Laplacian')
    ax1.legend()
    ax1.grid(True)

    # Histograms of all eigenvalues
    ax2.hist(all_evals, bins=50, alpha=0.5, label='6‑region', density=True)
    ax2.hist(all_evals_pal, bins=50, alpha=0.5, label='+PAL+LSX', density=True)
    ax2.set_xlabel('Eigenvalue')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of all eigenvalues')
    ax2.legend()

    plt.tight_layout()
    plt.savefig('sheaf_spectrum_comparison.png', dpi=150)
    plt.show()

    """

    print("\n--- INITIATING DYNAMIC REVERSE HIRONAKA RESOLUTION ---")
    resolver = ReverseHironakaMoleculeResolver()
    
    # FIX: Call the correct method name 'apply_hironaka_resolution_full'
    # and pass the full 'states' history instead of a single molecule_matrix
    hironaka_wavelet = resolver.apply_hironaka_resolution_full(states, t_q)
    
    # Visualize the dynamic path
    plt.figure(figsize=(12, 5))
    plt.plot(t_q, hironaka_wavelet, color='cyan', linewidth=1.5, label='Resolved Trajectory')
    
    # Mark the shatter locus (where consciousness C is minimum)
    t_shatter_idx = np.argmin(C[1, :])
    plt.axvline(t_q[t_shatter_idx], color='red', linestyle='--', label='Shatter Locus')
    
    plt.title("Dynamic Reverse Hironaka Trajectory (Opiate/Norcain Revival)")
    plt.xlabel("Time (s)")
    plt.ylabel("a(t) cos(ωt) + b(t) sin(ωt)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('hironaka_revival_path_DYNAMIC.png', dpi=150)
    print("  -> Dynamic trajectory saved to hironaka_revival_path_DYNAMIC.png")

    # 3. Build the spectral sheaf on the quiver
    sheafq = QuiverSpectralSheaf(quiver)

    # 4. Compute eigenvalue series and triple interaction over time
    # Wavelet does not change with number of nodes.
    n_t = len(t_q)
    second_eigenvalue_q = np.zeros(n_t)
    triple_heatmap_q = np.zeros((n_t, 6, 6))

    for i in range(n_t):
        try:
            evals = sheafq.eigenvalues(t_q[i], states[i])
            if np.isnan(evals).any():
                continue
            second_eigenvalue_q[i] = evals[1] if len(evals) > 1 else 0
            triple_heatmap_q[i] = sheafq.triple_interaction_matrix(t_q[i], states[i])
        except Exception as e:
            print(f"Warning: failed at time {t_q[i]:.2f}: {e}")
            continue

    # Remove nan entries for plotting
    mask = ~np.isnan(second_eigenvalue_q)
    t_valid = t_q[mask]
    second_eigenvalue_valid_q = second_eigenvalue_q[mask]

    # Plot only valid points
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    if len(t_valid) > 0:
        plt.plot(t_valid, second_eigenvalue_valid_q, 'b-')
    plt.xlabel('Time (s)')
    plt.ylabel('Second eigenvalue (spectral gap)')
    plt.title('Quiver Sheaf Spectral Gap')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    avg_triple = np.mean(triple_heatmap_q, axis=0)
    plt.imshow(avg_triple, cmap='hot')
    plt.colorbar()
    plt.xticks(range(6), quiver.vertex_names, rotation=45)
    plt.yticks(range(6), quiver.vertex_names)
    plt.title('Average Triple Interaction (A³)')
    plt.tight_layout()
    plt.savefig('quiver_sheafq_analysis.png', dpi=150)
    plt.show()

    """
    # =========================================================================
    # NEW: REVERSE HIRONAKA RESOLUTION (Opiate/Norcain Revival)
    # =========================================================================
    print("\n--- INITIATING REVERSE HIRONAKA RESOLUTION ---")
    resolver = ReverseHironakaMoleculeResolver()
    
    # We pick the time index where Node 1 is most "unconscious" (lowest C)
    # to demonstrate the revival trajectory.
    t_shatter_idx = np.argmin(C[1, :]) 
    t_val = t[t_shatter_idx]
    
    state_6 = states[t_shatter_idx] # This is length 6
    
    # Map the 6 quiver vertices back to the 3x3 Schubert configuration
    # Vertices: 0, 1, 2, 3, 4, 5
    # Matrix Structure:
    # [ v0  v1  0  ]
    # [ v1  v2  v3 ]  <-- v2 is the M22 Schubert Pivot
    # [ 0   v3  v4 ]
    # (v5 can be used as a global scaling or feedback term)
    # We use the 3x3 matrix state at this specific shatter point
    # Mapping the 28-arrow quiver state back to the M22 Schubert pivot
    molecule_matrix = np.array([
        [state_6[0], state_6[1], 0.0],
        [state_6[1], state_6[2], state_6[3]],
        [0.0,        state_6[3], state_6[4]]
    ])
    
    # Resolve the singularity into the harmonic wavelet
    resolution = resolver.apply_hironaka_resolution(molecule_matrix, t_q)
    
    print(f"Shatter Locus Time: {t_val:.2f}s")
    print(f"Schubert Cell Pivot (M22) Intensity: {resolution['m22_pivot']:.4f}")
    print(f"Resolved Frequencies (Opiate w1, Norcain w2): {resolution['frequencies']}")
    print(f"Topological Status: {resolution['chern_jump']}")
    
    # Optional: Plot the resolved trajectory a*cos(w1t) + b*sin(w2t)
    plt.figure(figsize=(10, 4))
    plt.plot(t_q, resolution['resolved_wavelet'], label='Resolved Hironaka Trajectory', color='cyan')
    plt.title(f"Node 1 Revival Signal (Gr(2,4) Projection)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitue (a cos w1t + b sin w2t)")
    plt.legend()
    plt.savefig('hironaka_revival_path.png')
    # =========================================================================
    """

    # 6. Clustering at a few times
    print("\nSpectral clustering at selected times:")

    # 6. Clustering at a few times
    print("\nSpectral clustering at selected times:")
    for t0 in [0, 5, 10, 15]:
        idx = np.argmin(np.abs(t_q - t0))
        labels = sheafq.spectral_clustering(t_q[idx], states[idx], n_clusters=2)
        print(f"t={t0:.1f}s, clusters: {dict(zip(quiver.vertex_names, labels))}")


    
    print("\n  Consciousness summary:")
    for node in range(3):
        minC = np.min(C[node,:])
        maxC = np.max(C[node,:])
        print(f"    Node {node}: min={minC:.3f}, max={maxC:.3f}")
    
    # After simulation, before calling create_dashboard:
    trans_counts = dynamics.count_phase_transitions()
    print("\n  Phase transition counts (crossing threshold = 0.3):")
    for node in range(3):
        print(f"    Node {node}: {trans_counts[node]['down']} collapses, {trans_counts[node]['up']} recoveries")
    
    # Dashboard
    create_dashboard(dynamics)
    
    # Ghost signal & monodromy
    plot_ghost_signal(dynamics)
    
    # Prime zeta
    plot_prime_zeta(dynamics)
    # After simulation, save transition times to a JSON file
    # Save transition times
    transition_times = dynamics.transition_times
    if transition_times.size > 0:
        # Convert numpy types if needed
        transition_times = [float(t) if hasattr(t, 'item') else t for t in transition_times]
        with open("transition_times.json", "w") as f:
            json.dump(transition_times, f)
        print(f"Saved transition times: {transition_times}")

    # Save prime zeta values (call the method)
    if hasattr(dynamics, 'prime_zeta_values') and dynamics.prime_zeta_values:
        prime_zeta_data = {
            "values": [{"real": float(z), "imag": 0.0} for z in dynamics.prime_zeta_values],
            "transition_times": [float(t) for t in dynamics.prime_zeta_times]
        }
        with open("prime_zeta.json", "w") as f:
            json.dump(prime_zeta_data, f)
        print("Saved prime_zeta.json from blow‑up events ({} entries).".format(len(dynamics.prime_zeta_values)))
    else:
        # fallback to consciousness transitions
        prime_zeta_vals = dynamics.compute_prime_zeta()
        transition_times = dynamics.transition_times
        if transition_times.size > 0:
            transition_times_list = transition_times.tolist() if hasattr(transition_times, 'tolist') else list(transition_times)
            prime_zeta_data = {
                "values": [{"real": z.real, "imag": z.imag} for z in prime_zeta_vals],
                "transition_times": transition_times_list
            }
            with open("prime_zeta.json", "w") as f:
                json.dump(prime_zeta_data, f)
            print("Saved prime_zeta.json from consciousness transitions.")
        else:
            print("No prime zeta data to save (neither blow‑up nor consciousness transitions).")

    # Full Gr(2,4) projection result
    result = dynamics.gr24_result
    if result is not None:
        print(f"Wall crossings: {result.n_wall_crossings}")
        if len(result.frames) > 100:
            frame = result.frames[100]
            print(f"Stratum: {frame.schubert.stratum}")
            print(f"Klein Q: {frame.schubert.klein_Q:.6f}")
            print(f"Ihara hint: {frame.ihara_prediction:.4f}")
        try:
            from gr24_schober_projection import plot_gr24_projection
            plot_gr24_projection(result, save_path="gr24_projection.png")
        except Exception as _e:
            print(f"gr24 plot skipped: {_e}")
    else:
        print("gr24_result not available (module not found or plucker_history empty)")

    
    print("\n" + "="*100)
    print(" " * 40 + "ANALYSIS COMPLETE")
    print("="*100)
    return dynamics


if __name__ == "__main__":
    dynamics = main()
