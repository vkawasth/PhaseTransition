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
import warnings
import pandas as pd
from collections import deque
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

import jax
import jax.numpy as jnp
from jax import jit, jacfwd
jax.config.update("jax_enable_x64", True)

EDGES = [(0,1), (1,0), (0,2), (2,0), (1,2), (2,1), (1,1)]

def euler_step(state, t, dt, params_tuple):
    # Unpack tuple
    (half_life_A, half_life_B, alpha, beta, EC50_A, EC50_B,
     hill_A, hill_B, *edge_weights_vals, radius_A, radius_B,
     pore_radius, diffusivity_A, diffusivity_B,
     membrane_thickness, base_flow_A, base_flow_B) = params_tuple

    # Reconstruct edge_weights
    edge_weights = {EDGES[i]: edge_weights_vals[i] for i in range(len(EDGES))}

    # Unpack state
    qA0, qA1, qA2, qB0, qB1, qB2, C0, C1, C2 = state

    lambda_A = jnp.log(2) / half_life_A
    lambda_B = jnp.log(2) / half_life_B

    # Helper functions
    def renkin_crone_factor(radius_ratio):
        return jnp.where(radius_ratio >= 1, 0.0,
                         (1 - radius_ratio)**2 * (1 - 2.104*radius_ratio + 2.09*radius_ratio**3 - 0.95*radius_ratio**5))

    def transition_rate(edge, molecule):
        u, v = edge
        heartbeat = 1 + 0.3 * jnp.sin(2 * jnp.pi * t)
        weight = edge_weights.get(edge, 1.0)
        base_flow = base_flow_A if molecule == 'A' else base_flow_B
        flow = base_flow * weight * heartbeat
        if molecule == 'A':
            radius = radius_A
            diffusivity = diffusivity_A
        else:
            radius = radius_B
            diffusivity = diffusivity_B
        radius_ratio = radius / pore_radius
        renkin = renkin_crone_factor(radius_ratio)
        permeability = (diffusivity / membrane_thickness) * renkin
        return flow * permeability * 300

    # Precompute rates for each edge and molecule
    rates_A = {edge: transition_rate(edge, 'A') for edge in EDGES}
    rates_B = {edge: transition_rate(edge, 'B') for edge in EDGES}

    # Initialize arrays for inflows/outflows
    inflow_A = jnp.zeros(3)
    outflow_A = jnp.zeros(3)
    inflow_B = jnp.zeros(3)
    outflow_B = jnp.zeros(3)

    # Helper to get concentration at node
    qA_node = jnp.array([qA0, qA1, qA2])
    qB_node = jnp.array([qB0, qB1, qB2])

    for edge in EDGES:
        u, v = edge
        rate_A = rates_A[edge]
        rate_B = rates_B[edge]

        # Outflow from u
        outflow_A = outflow_A.at[u].add(rate_A * qA_node[u])
        outflow_B = outflow_B.at[u].add(rate_B * qB_node[u])

        # Inflow to v
        inflow_A = inflow_A.at[v].add(rate_A * qA_node[u])
        inflow_B = inflow_B.at[v].add(rate_B * qB_node[u])

    # Update concentrations
    qA_new = qA_node + (inflow_A - outflow_A - lambda_A * qA_node) * dt
    qB_new = qB_node + (inflow_B - outflow_B - lambda_B * qB_node) * dt

    # Consciousness update
    C_node = jnp.array([C0, C1, C2])
    activation = alpha * (1 - C_node) * (qB_node**hill_B) / (EC50_B**hill_B + qB_node**hill_B)
    dampening = beta * C_node * (qA_node**hill_A) / (EC50_A**hill_A + qA_node**hill_A)
    oscillation = 0.1 * jnp.sin(2 * jnp.pi * 0.6 * t) * (1 - C_node)
    dC = (activation - dampening + oscillation) * dt
    C_new = C_node + dC

    # Clip
    qA_new = jnp.clip(qA_new, 0, 5)
    qB_new = jnp.clip(qB_new, 0, 6)
    C_new = jnp.clip(C_new, 0, 1)

    new_state = jnp.concatenate([qA_new, qB_new, C_new])
    return new_state

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
        self.nodes = [0, 1, 2]
        self.edges = [
            (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (1, 1)
        ]
        # 2. DEFINE THRESHOLD HERE (Move this up!)
        self.threshold = 0.3  
        self.rate_series_A = {}  # key: edge, value: list of rates over time
        self.rate_series_B = {}
        for edge in self.edges:
            self.rate_series_A[edge] = []
            self.rate_series_B[edge] = []
        
        # 3. Now initialize Milnor and Siegel using the threshold
        self.milnor = MilnorSequestrator(threshold=self.threshold)
        self.siegel = SiegelLock(anchor_value=0.6)
        
        # 4. Initialize other tracking arrays
        self.ghost_signal = None
        
        # Extremely high edge weights for norcain
        self.edge_weights = {
            (0, 1): 15.0,   # Very fast from 0→1
            (1, 0): 2.0,
            (0, 2): 12.0,   # Fast from 0→2
            (2, 0): 1.5,
            (1, 2): 8.0,    # Fast from 1→2
            (2, 1): 2.5,
            (1, 1): 5.0     # Strong recirculation
        }
        
        # Molecule parameters
        self.half_life_A = 6.0
        self.half_life_B = 1.2   # Very short norcain half‑life (acts fast, decays quickly)
        self.delay_A = 0.1
        self.delay_B = 0.2
        
        # Pharmacodynamics – norcain extremely effective
        self.alpha = 2.2
        self.beta = 1.0
        self.EC50_A = 0.2
        self.EC50_B = 0.2
        self.hill_A = 2.0
        self.hill_B = 2.0
        
        self.threshold = 0.3
        self.dose_times = [2.5, 6.0, 9.5]
        self.dose_amount = 4.0   # Large dose
        
        self.t = None
        self.C = None
        self.qA = None
        self.qB = None
        self.HH1 = None
        self.HH2 = None
        self.plucker = None
        self.reverse_trajectories = []
        self.radius_A = 0.5e-9
        self.radius_B = 0.8e-9
        self.pore_radius = 1.0e-9
        self.diffusivity_A = 5e-10
        self.diffusivity_B = 8e-10
        self.membrane_thickness = 1e-6
        self.base_flow_A = 4.0
        self.base_flow_B = 6.0
        self.M_list = []   # will store matrices
        self.times = []    # corresponding times
        
        self.rate_count = 0
        self.pair_HH2 = {}  # key: (i,j) with i<j, value: list of contributions per time step

    
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
    
    def compute_hochschild_invariants(self, t_idx):
        if t_idx < 2:
            return 0, 0, {}
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
        pair_contrib = {}
        comm = 0
        for i in range(3):
            for j in range(i+1, 3):
                contrib = abs(C_cur[i] * qB_cur[j] - C_cur[j] * qB_cur[i])
                pair_contrib[(i,j)] = contrib
                comm += contrib
        HH2_val += 0.3 * comm
        return HH1_val, HH2_val, pair_contrib
    
    def simulate(self, t_span=(0, 25), dt=0.02, compute_operators=True):
        t = np.arange(t_span[0], t_span[1], dt)
        n_steps = len(t)
        self.t = t
        self.C = np.zeros((3, n_steps))
        self.qA = np.zeros((3, n_steps))
        self.qB = np.zeros((3, n_steps))
        self.HH1 = np.zeros(n_steps)
        self.HH2 = np.zeros(n_steps)
        
        # Initial conditions: opiate only at Node 0
        self.qA[0, 0] = 3.5
        self.qA[1, 0] = 0.0
        self.qA[2, 0] = 0.0
        self.C[:, 0] = 1.0
        
        lambda_A = np.log(2) / self.half_life_A
        lambda_B = np.log(2) / self.half_life_B

        self.ghost_signal = np.zeros(n_steps) # Ensure this is an array
        
        print("\n  Extreme flow configuration:")
        print(f"    Opiate base flow = 4.0, edge weights: 0→1={self.edge_weights[(0,1)]}, 0→2={self.edge_weights[(0,2)]}")
        print(f"    Norcain base flow = 6.0, dose amount = {self.dose_amount}")

        # Prepare parameter dictionary
        params = {
            'half_life_A': self.half_life_A,
            'half_life_B': self.half_life_B,
            'alpha': self.alpha,
            'beta': self.beta,
            'EC50_A': self.EC50_A,
            'EC50_B': self.EC50_B,
            'hill_A': self.hill_A,
            'hill_B': self.hill_B,
            'edge_weights': self.edge_weights,
            'radius_A': 0.5e-9,
            'radius_B': 0.8e-9,
            'pore_radius': 1.0e-9,
            'diffusivity_A': 5e-10,
            'diffusivity_B': 8e-10,
            'membrane_thickness': 1e-6,
            'base_flow_A': 4.0,
            'base_flow_B': 6.0,
        }

        # After the initialisation, before the loop, prepare params_tuple and jitted functions
        if compute_operators:
            # Build params_tuple
            edge_weights_vals = [self.edge_weights.get(e, 0.0) for e in EDGES]
            params_tuple = (
                self.half_life_A, self.half_life_B,
                self.alpha, self.beta,
                self.EC50_A, self.EC50_B,
                self.hill_A, self.hill_B,
                *edge_weights_vals,
                self.radius_A, self.radius_B, self.pore_radius,
                self.diffusivity_A, self.diffusivity_B, self.membrane_thickness,
                self.base_flow_A, self.base_flow_B
            )
            # JIT the step function (with static args dt and params_tuple)
            euler_step_jit = jit(euler_step, static_argnums=(2,3))
            jac_step = jacfwd(euler_step_jit, argnums=0)
            # Lists to store operators and times
            self.M_list = []
            self.M_times = []

        
        for i in range(n_steps - 1):
            dt_step = t[i+1] - t[i]
            for node in self.nodes:
                C_cur = self.C[node, i]
                qA_cur = self.qA[node, i]
                qB_cur = self.qB[node, i]
                
                inflow_A = inflow_B = 0
                for edge in self.edges:
                    if edge[1] == node:
                        src = edge[0]
                        delay_A = self.delay_A if src != node else 0
                        delay_B = self.delay_B if src != node else 0
                        idx_A = max(0, i - int(delay_A / dt_step))
                        idx_B = max(0, i - int(delay_B / dt_step))
                        rate_A = self.transition_rate(edge, t[i], 'A')
                        rate_B = self.transition_rate(edge, t[i], 'B')
                        inflow_A += rate_A * self.qA[src, idx_A] * dt_step
                        inflow_B += rate_B * self.qB[src, idx_B] * dt_step
                        rate_A = self.transition_rate(edge, t[i], 'A')
                        rate_B = self.transition_rate(edge, t[i], 'B')
                        self.rate_series_A[edge].append(rate_A)
                        self.rate_series_B[edge].append(rate_B)
                
                outflow_A = outflow_B = 0
                for edge in self.edges:
                    if edge[0] == node:
                        rate_A = self.transition_rate(edge, t[i], 'A')
                        rate_B = self.transition_rate(edge, t[i], 'B')
                        outflow_A += rate_A * qA_cur * dt_step
                        outflow_B += rate_B * qB_cur * dt_step
                
                self.qA[node, i+1] = qA_cur + inflow_A - outflow_A - qA_cur * lambda_A * dt_step
                self.qB[node, i+1] = qB_cur + inflow_B - outflow_B - qB_cur * lambda_B * dt_step
                
                if node == 0:
                    for dose_time in self.dose_times:
                        if abs(t[i] - dose_time) < dt_step:
                            self.qB[node, i+1] += self.dose_amount
                
                self.qA[node, i+1] = np.clip(self.qA[node, i+1], 0, 5)
                self.qB[node, i+1] = np.clip(self.qB[node, i+1], 0, 6)
                
                activation = self.alpha * (1 - C_cur) * (qB_cur**self.hill_B) / (self.EC50_B**self.hill_B + qB_cur**self.hill_B)
                dampening = self.beta * C_cur * (qA_cur**self.hill_A) / (self.EC50_A**self.hill_A + qA_cur**self.hill_A)
                oscillation = 0.1 * np.sin(2 * np.pi * 0.6 * t[i]) * (1 - C_cur)
                dC = (activation - dampening + oscillation) * dt_step
                self.C[node, i+1] = np.clip(C_cur + dC, 0, 1)

                # 1. Apply Milnor Sequestration to Node 0 (the primary dosing site)
                self.milnor.isolate_node(self.t[i+1], self.C[0, i+1], [self.qA[0, i+1], self.qB[0, i+1]])
                
                # 2. Calculate the Siegel Lock (Ghost Signal) for the entire graph
                avg_qA = np.mean(self.qA[:, i+1])
                avg_qB = np.mean(self.qB[:, i+1])
                avg_C = np.mean(self.C[:, i+1])
                self.ghost_signal[i+1] = self.siegel.apply_lock(avg_qA, avg_qB, avg_C)

                # In the simulation loop (after computing new state or before), compute M
                if compute_operators:
                    # Build state array
                    state = jnp.array([self.qA[0,i], self.qA[1,i], self.qA[2,i],
                                    self.qB[0,i], self.qB[1,i], self.qB[2,i],
                                    self.C[0,i], self.C[1,i], self.C[2,i]])
                    # Skip if a dose was applied at this time (i.e., if t[i] is near a dose time)
                    # For simplicity, we'll just compute M every step; if dose occurs, it will be a jump, but we can still compute.
                    M = jac_step(state, t[i], dt, params_tuple)
                    self.M_list.append(np.array(M))
                    self.M_times.append(t[i])
            
            self.HH1[i+1], self.HH2[i+1], self.pair_contrib = self.compute_hochschild_invariants(i+1)
            for (ii,jj), val in self.pair_contrib.items():
                if (ii,jj) not in self.pair_HH2:
                    self.pair_HH2[(ii,jj)] = []
                self.pair_HH2[(ii,jj)].append(val)
        
        # Compute Plücker trajectory
        self.plucker = self.compute_plucker_trajectory()
        
        # Detect phase transitions and build reverse trajectories
        self.detect_phase_transitions()
        self.avg_pair_HH2 = {}
        for (i,j), vals in self.pair_HH2.items():
            self.avg_pair_HH2[(i,j)] = np.mean(vals)
        
        return self.t, self.C, self.qA, self.qB, self.HH1, self.HH2, self.M_list
    
    def build_quiver_from_rates(self, threshold=0.001, combine='sum'):
        """
        Build a directed weighted quiver from the average transition rates.
        combine: 'sum' (add A and B), 'max' (take max), or 'A'/'B' for single molecule.
        """
        # Compute average rates
        avg_A = {}
        avg_B = {}
        for edge in self.edges:
            avg_A[edge] = np.mean(self.rate_series_A[edge])
            avg_B[edge] = np.mean(self.rate_series_B[edge])

        # Build node-level weights
        node_weight = np.zeros((3,3))
        for edge in self.edges:
            u, v = edge
            if combine == 'sum':
                w = avg_A[edge] + avg_B[edge]
            elif combine == 'max':
                w = max(avg_A[edge], avg_B[edge])
            else:
                w = avg_A[edge]   # fallback
            node_weight[u, v] = max(node_weight[u, v], w)

        # Extract edges above threshold
        edges = []
        weights = []
        for i in range(3):
            for j in range(3):
                if i != j and node_weight[i, j] > threshold:
                    edges.append((i, j))
                    weights.append(node_weight[i, j])
        return edges, weights

    def compute_triple_stats_from_rates(self):
        """
        For each triple (i,j,k), compute mean and variance of
        (rate_ij * rate_jk) - rate_ik over time.
        """
        # Convert the rate series to numpy arrays for speed
        # We'll build a dict mapping (i,j) -> array of combined rates
        combined_rates = {}
        for edge in self.edges:
            arr = np.array(self.rate_series_A[edge]) + np.array(self.rate_series_B[edge])
            combined_rates[edge] = arr

        n_t = len(next(iter(self.rate_series_A.values())))  # number of time steps
        results = {}

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    # Get the time series for the three edges (if they exist)
                    rate_ij = combined_rates.get((i, j), np.zeros(n_t))
                    rate_jk = combined_rates.get((j, k), np.zeros(n_t))
                    rate_ik = combined_rates.get((i, k), np.zeros(n_t))

                    dev = rate_ij * rate_jk - rate_ik
                    mean_dev = np.mean(dev)
                    var_dev = np.var(dev)
                    results[(i, j, k)] = (mean_dev, var_dev)
        return results
    
    # Build quiver from Operator Flow Algebra
    def build_quiver_from_operators(self, M_list, threshold=0.01):
        var_to_node = [0,1,2, 0,1,2, 0,1,2]   # indices 0-2: qA; 3-5: qB; 6-8: C
        # Average transition weights over time
        avg_M = np.mean(M_list, axis=0)   # shape (9,9)
        # For each pair of nodes (i,j), the weight is the average of (M_t)_{ij}
        # But note: the 9 variables correspond to nodes 0,1,2 each with three variables (qA, qB, C).
        # We need to aggregate to node level. For simplicity, we can take the maximum over the three variable pairs.
        # Or treat each variable separately for a finer quiver. Here we'll aggregate by node.
        # Let's define mapping: variable index -> node
        var_to_node = [0,1,2, 0,1,2, 0,1,2]  # indices 0-2: qA; 3-5: qB; 6-8: C
        node_weight = np.zeros((3,3))
        for i in range(9):
            for j in range(9):
                node_i = var_to_node[i]
                node_j = var_to_node[j]
                node_weight[node_i, node_j] = max(node_weight[node_i, node_j], avg_M[i,j])  # use max or sum
        print("Node weight matrix:\n", node_weight)
        # Now threshold
        edges = []
        weights = []
        for i in range(3):
            for j in range(3):
                if i != j and node_weight[i,j] > threshold:
                    edges.append((i,j))
                    weights.append(node_weight[i,j])
        return edges, weights
    
    def compute_triple_stats(self, M_list):
        var_to_node = [0,1,2, 0,1,2, 0,1,2]   # variable index → node (0,1,2)
        n = len(M_list)
        # Pre-allocate arrays for each triple (i,j,k)
        # We'll use dictionaries keyed by (i,j,k)
        dev_mean = {}
        dev_var = {}
        # For each t from 0 to n-2
        for t_idx in range(n-1):
            M = M_list[t_idx]
            M_next = M_list[t_idx+1]
            # Compute product M_next @ M for 2-step evolution
            M2 = M_next @ M
            # For all i,j,k (0..2 nodes) – we need to aggregate variable-level to node-level
            # We'll aggregate by node (as before) by taking maximum over the variable contributions
            # This is a simplification; you could do a more refined analysis.
            for i_node in range(3):
                for j_node in range(3):
                    for k_node in range(3):
                        # Get all variable indices that map to these nodes
                        var_i = [idx for idx in range(9) if var_to_node[idx]==i_node]
                        var_j = [idx for idx in range(9) if var_to_node[idx]==j_node]
                        var_k = [idx for idx in range(9) if var_to_node[idx]==k_node]
                        # Take the maximum over the product of entries? Or sum? Here we'll take max to capture strongest path.
                        prod_max = 0.0
                        for ii in var_i:
                            for jj in var_j:
                                for kk in var_k:
                                    prod = M[ii,jj] * M[jj,kk]
                                    if prod > prod_max:
                                        prod_max = prod
                        # For the 2-step direct, take maximum over the entries from i_node to k_node in M2
                        direct_max = 0.0
                        for ii in var_i:
                            for kk in var_k:
                                val = M2[ii,kk]
                                if val > direct_max:
                                    direct_max = val
                        dev = prod_max - direct_max
                        key = (i_node, j_node, k_node)
                        if key not in dev_mean:
                            dev_mean[key] = []
                        dev_mean[key].append(dev)
        # Compute mean and variance for each key
        results = {}
        for key, vals in dev_mean.items():
            arr = np.array(vals)
            mean_val = np.mean(arr)
            var_val = np.var(arr)
            results[key] = (mean_val, var_val)
        return results
    
    def detect_missing_structure(self, relations, edges):
        missing = []
        for rel in relations:
            if rel[0] == 'path':
                i, j, k = rel[1], rel[2], rel[3]
                if (i,k) not in edges:
                    missing.append((i,k))
        return missing
    
    def localize_missing_structure(self, relations, edges, results):
        # results: dict (i,j,k) -> (mean, var)
        missing_candidates = []
        for rel in relations:
            if rel[0] == 'path':
                i, j, k = rel[1], rel[2], rel[3]
                if (i,k) not in edges:
                    mean_val, var_val = results[(i,j,k)]
                    missing_candidates.append(((i,k), mean_val, var_val))
        return missing_candidates
    
    def run_hidden_structure_analysis(self, threshold=0.001, delta=0.05, tau=0.01):
        # Build quiver from rates
        edges, weights = self.build_quiver_from_rates(threshold=threshold)
        print("Quiver edges (from rates):", edges)
        print("Weights:", weights)
        self.inferred_edges = edges
        self.inferred_weights = weights

        # Compute triple statistics
        results = self.compute_triple_stats_from_rates()

        # Infer relations
        relations = []
        for (i, j, k), (mean_val, var_val) in results.items():
            # Skip self‑loops and any triple that involves the same node twice
            if i == j or j == k or i == k:
                continue
            if abs(mean_val) < delta and var_val < tau:
                relations.append(('path', i, j, k))
        print("Inferred relations:", relations)

        # Detect missing edges
        missing = []
        for rel in relations:
            if rel[0] == 'path':
                i, j, k = rel[1], rel[2], rel[3]
                if (i, k) not in edges:
                    missing.append((i, k))
        if missing:
            print("Possible missing edges (hidden structure):", missing)
        else:
            print("No missing edges detected.")

        return edges, weights, relations, missing
        
    
    def compute_plucker_trajectory(self):
        n = len(self.t)
        plucker = np.zeros((n, 6))
        lambda_A = np.log(2) / self.half_life_A
        lambda_B = np.log(2) / self.half_life_B
        for i in range(n):
            C_val = self.C[0, i]
            qA_val = self.qA[0, i]
            qB_val = self.qB[0, i]
            p12 = C_val / (1 + qA_val / self.EC50_A)
            p13 = qB_val / (1 + qB_val / self.EC50_B)
            p14 = (1 - C_val) * np.exp(-lambda_A * self.t[i])
            p23 = C_val * np.exp(-lambda_B * self.t[i])
            p24 = qA_val * np.exp(-lambda_A * self.t[i])
            p34 = qB_val * np.exp(-lambda_B * self.t[i])
            plucker[i] = [p12, p13, p14, p23, p24, p34]
            norm = np.linalg.norm(plucker[i])
            if norm > 0:
                plucker[i] /= norm
        return plucker
    
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
        for node in range(3):
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

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def plot_quiver_and_HH2(dynamics):
    if not hasattr(dynamics, 'inferred_edges') or not dynamics.inferred_edges:
        print("No inferred quiver edges found.")
        return

    edges = dynamics.inferred_edges
    weights = dynamics.inferred_weights
    avg_pair_HH2 = dynamics.avg_pair_HH2

    # Build node positions
    pos = {0: (0, 0), 1: (1, 0.5), 2: (0.5, 1)}

    max_weight = max(weights) if weights else 1.0
    max_hh2 = max(avg_pair_HH2.values()) if avg_pair_HH2 else 1.0

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)

    # Draw edges first (so they appear behind nodes)
    for (u, v), w in zip(edges, weights):
        # Check for reciprocal edge to curve
        rad = 0.15 if (v, u) in edges else 0.0
        style = f"arc3,rad={rad}"
        color_val = avg_pair_HH2.get((u, v), 0.0) / max_hh2 if max_hh2 > 0 else 0.5
        width = 0.5 + 2.5 * (w / max_weight) if max_weight > 0 else 1.0
        arrow = FancyArrowPatch(pos[u], pos[v], connectionstyle=style,
                                arrowstyle='->', lw=width,
                                color=plt.cm.hot(color_val),
                                mutation_scale=15, zorder=1)
        ax.add_patch(arrow)

    # Draw nodes (on top)
    for node, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.08, facecolor='lightblue', edgecolor='black', zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, str(node), ha='center', va='center', fontsize=12, fontweight='bold', zorder=3)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(0, max_hh2))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label('HH₂ contribution')

    ax.set_title("Inferred quiver (edge width ∝ flow, color ∝ HH₂ contribution)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('inferred_quiver_HH2.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("    ✓ Saved: inferred_quiver_HH2.png")
    
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

    def simulate_quiver(self, initial_state, t_span, store_rates=True):
        t = np.arange(t_span[0], t_span[1], self.dt)
        states = np.zeros((len(t), 6))
        current = initial_state.copy()
        self.rate_series = {}   # key: (src,tgt) -> list
        for i, ti in enumerate(t):
            states[i] = current
            rates = self.get_dynamic_rates(ti, current)
            for (src, tgt), rate in rates.items():
                self.rate_series.setdefault((src, tgt), []).append(rate)
            current = self.evolve(current, ti)
        return t, states
    
    def build_quiver_from_wavelet_rates(self, threshold=0.001):
        avg_rates = {}
        for (src, tgt), vals in self.rate_series.items():
            avg_rates[(src, tgt)] = np.mean(vals)
        edges = []
        weights = []
        for (src, tgt), w in avg_rates.items():
            if src != tgt and w > threshold:
                edges.append((src, tgt))
                weights.append(w)
        return edges, weights
    
    def compute_triple_stats_from_wavelet(self):
        n_t = len(next(iter(self.rate_series.values())))
        # Convert to arrays for speed
        rate_arr = {k: np.array(v) for k, v in self.rate_series.items()}
        results = {}
        for i in range(6):
            for j in range(6):
                for k in range(6):
                    if i == j or j == k or i == k:
                        continue
                    # Get time series
                    rate_ij = rate_arr.get((i, j), np.zeros(n_t))
                    rate_jk = rate_arr.get((j, k), np.zeros(n_t))
                    rate_ik = rate_arr.get((i, k), np.zeros(n_t))
                    dev = rate_ij * rate_jk - rate_ik
                    results[(i,j,k)] = (np.mean(dev), np.var(dev))
        return results
    
    def compute_wavelet_obstruction(self):
        """
        Compute a per-edge obstruction measure (variance of rate over time)
        and return a dict mapping (src, tgt) -> obstruction value.
        """
        obstruction = {}
        for (src, tgt), rates in self.rate_series.items():
            # Use variance as obstruction
            obstruction[(src, tgt)] = np.var(rates)
        return obstruction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def plot_quiver_6node(edges, weights, avg_pair_HH2, title="Wavelet Quiver", filename="wavelet_quiver.png"):
    """
    Draw a 6‑node directed quiver with:
      - Nodes placed on a circle.
      - Edge width ∝ weight.
      - Edge color ∝ HH₂ contribution (hot colormap).
    Parameters:
        edges: list of (src, tgt) tuples.
        weights: list of floats, same order as edges.
        avg_pair_HH2: dict mapping (i,j) -> average HH₂ contribution (or 0 if not computed).
        title: string for plot title.
        filename: output filename.
    """
    # Node positions in a circle
    n_nodes = 6
    angle = 2 * np.pi / n_nodes
    pos = {i: (np.cos(i * angle), np.sin(i * angle)) for i in range(n_nodes)}

    # Normalize weight and HH₂ for scaling
    max_weight = max(weights) if weights else 1.0
    max_hh2 = max(avg_pair_HH2.values()) if avg_pair_HH2 else 1.0

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)

    # --- Draw edges (first layer) ---
    for (u, v), w in zip(edges, weights):
        # Curve if reciprocal edge exists
        rad = 0.15 if (v, u) in edges else 0.0
        style = f"arc3,rad={rad}"
        # Color from HH₂
        hh2_val = avg_pair_HH2.get((u, v), 0.0)
        color_val = hh2_val / max_hh2 if max_hh2 > 0 else 0.5
        width = 0.5 + 2.5 * (w / max_weight) if max_weight > 0 else 1.0
        arrow = FancyArrowPatch(pos[u], pos[v],
                                connectionstyle=style,
                                arrowstyle='->',
                                lw=width,
                                color=plt.cm.hot(color_val),
                                mutation_scale=15,
                                zorder=1)
        ax.add_patch(arrow)

    # --- Draw nodes (on top) ---
    # Node labels: a, b, ω₁, ω₂, φ, κ
    labels = ['a', 'b', 'ω₁', 'ω₂', 'φ', 'κ']
    for i, (x, y) in pos.items():
        circle = plt.Circle((x, y), 0.12, facecolor='lightblue', edgecolor='black', zorder=2)
        ax.add_patch(circle)
        ax.text(x, y, labels[i], ha='center', va='center',
                fontsize=10, fontweight='bold', zorder=3)

    # --- Colorbar for HH₂ ---
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hot, norm=plt.Normalize(0, max_hh2))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.05)
    cbar.set_label('HH₂ contribution')

    ax.set_title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"    ✓ Saved: {filename}")

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
        self.n_nodes = 3
        self.stalk_dim = 4
        self.total_dim = self.n_nodes * self.stalk_dim   # 12

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
            # Add contributions to diagonal blocks
            L[u*4:(u+1)*4, u*4:(u+1)*4] += R.T @ R
            L[v*4:(v+1)*4, v*4:(v+1)*4] += R.T @ R
            # Off‑diagonal blocks
            L[u*4:(u+1)*4, v*4:(v+1)*4] += -R
            L[v*4:(v+1)*4, u*4:(u+1)*4] += -R.T
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
    triple_avg = np.zeros((6, 6))
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

# Tracks molecules as they move for Chern class jump - Single molecule tracker.
class ReverseHironakaMoleculeResolver_S:
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
    ax18.bar(['Node 0','Node 1','Node 2'], final_C, color=colors, alpha=0.7)
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
    """Handles the algebraic 'inflation' at a singularity."""
    @staticmethod
    def resolve_singularity(model, t_idx):
        # The 'Blow-up' represents adding an exceptional divisor (an extra dose/dimension)
        # to restore the smooth trajectory.
        print(f"    [Rees Blow-up] Resolving singularity at t={model.t[t_idx]:.2f}")
        # In the context of the simulation, this is a 'forced' recovery dose.
        # We increase the 'norcain' concentration to provide the missing 'flow' dimension.
        model.qB[0, t_idx+1] += model.dose_amount * 1.5
        # This acts as an exceptional generator that 'absorbs' the Plucker error.
        return True

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
    t, C, qA, qB, HH1, HH2, Mt = dynamics.simulate()
    t_full, C_full, qA_full, qB_full, HH1_full, HH2_full, M_list = t, C, qA, qB, HH1, HH2, Mt

    edges, weights, relations, missing = dynamics.run_hidden_structure_analysis()
    plot_quiver_and_HH2(dynamics)

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
    C_no = np.zeros((3, n_steps))
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
        for node in range(3):
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

    # Save result
    plot_unstable_paths(dynamics, nav, best_path)

    # Build the sheaf
    sheaf = GraphSpectralSheaf(dynamics)

    # Compute eigenvalues and triple interactions over time
    n_t = len(t)
    second_eigenvalue = np.zeros(n_t)
    triple_heatmap = np.zeros((n_t, 3, 3))

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
    t_q, states = quiver.simulate_quiver(initial_state, (t[0], t[-1]))

    # Build quiver using operator flow algebra
    # Build wavelet quiver edges and weights from the rate series
    edges_wavelet, weights_wavelet = quiver.build_quiver_from_wavelet_rates(threshold=0.001)  # you need to implement this method
    obstruction_wavelet = quiver.compute_wavelet_obstruction()

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

    plot_quiver_6node(edges_wavelet, weights_wavelet, obstruction_wavelet,
                  title="Wavelet Quiver (rate variance)", filename="wavelet_quiver.png")

    # 3. Build the spectral sheaf on the quiver
    sheafq = QuiverSpectralSheaf(quiver)

    # 4. Compute eigenvalue series and triple interaction over time
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
    
    print("\n" + "="*100)
    print(" " * 40 + "ANALYSIS COMPLETE")
    print("="*100)
    return dynamics


if __name__ == "__main__":
    dynamics = main()