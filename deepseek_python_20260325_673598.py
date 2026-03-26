"""
COMPLETE EXAMPLE: 3-Node, 7-Edge Graph with Wavelet-Spectral Reverse Hironaka
=============================================================================
Produces comprehensive charts showing:
- Consciousness dynamics with phase transitions
- Wavelet decomposition at multiple scales
- Plücker trajectories on the Klein quadric
- Reverse Hironaka trajectories
- Prime path identification via zeta functions
- Spectral flow and coherence analysis
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.signal import cwt, morlet, find_peaks
from scipy.linalg import expm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# ============================================================================
# PART 1: 3-NODE, 7-EDGE GRAPH DEFINITION
# ============================================================================

class ThreeNodeGraph:
    """
    3-node, 7-edge graph representing brain regions:
    - Node 0: Injection site (frontal cortex)
    - Node 1: Recirculation hub (thalamus)
    - Node 2: Distal region (occipital cortex)
    
    Edges:
    - Bidirectional between 0-1, 0-2, 1-2 (6 edges)
    - Self-loop at node 1 (recirculation)
    """
    
    def __init__(self):
        self.nodes = [0, 1, 2]
        self.edges = [
            (0, 1), (1, 0),  # Bidirectional 0-1
            (0, 2), (2, 0),  # Bidirectional 0-2
            (1, 2), (2, 1),  # Bidirectional 1-2
            (1, 1)           # Self-loop at node 1
        ]
        
        # Edge weights (thickness/capacity)
        self.edge_weights = {
            (0, 1): 1.2, (1, 0): 1.2,  # Thick edges - rapid flow
            (0, 2): 0.6, (2, 0): 0.6,  # Thin edges - slow flow
            (1, 2): 0.9, (2, 1): 0.9,  # Medium edges
            (1, 1): 0.4                 # Self-loop - recirculation
        }
        
        # Molecule parameters
        self.half_life_A = 4.0  # Opiate (seconds)
        self.half_life_B = 2.0  # Norcain (seconds)
        self.delay_A = 2.0      # Flow delay for A
        self.delay_B = 4.0      # Flow delay for B
        
    def flow_rate(self, edge, t, molecule='B'):
        """Heartbeat-modulated flow rate"""
        # Heartbeat rhythm (1 Hz = 60 bpm)
        heartbeat = 1 + 0.2 * np.sin(2 * np.pi * t)
        
        # Edge-specific scaling
        weight = self.edge_weights.get(edge, 1.0)
        
        # Base flow
        base_flow = 0.5 if molecule == 'A' else 0.8
        
        return base_flow * weight * heartbeat
    
    def transition_rate(self, edge, t, molecule):
        """Transition rate including Renkin-Crone permeability"""
        flow = self.flow_rate(edge, t, molecule)
        
        # Renkin-Crone factor (simplified)
        radius_ratio = 0.6 if molecule == 'A' else 0.8
        renkin_factor = (1 - radius_ratio)**2 * (1 - 2.104*radius_ratio + 2.09*radius_ratio**3)
        
        # Permeability
        diffusivity = 1e-10 if molecule == 'A' else 0.8e-10
        permeability = diffusivity / 1e-6 * renkin_factor
        
        return flow * permeability * 100  # Scale for simulation


# ============================================================================
# PART 2: CONSCIOUSNESS DYNAMICS SIMULATION
# ============================================================================

def simulate_consciousness_dynamics(graph: ThreeNodeGraph, t_span=(0, 20), dt=0.02):
    """
    Simulate consciousness dynamics for the 3-node graph with:
    - Molecular transport (A and B)
    - Pharmacodynamics (opiate dampening, norcain activation)
    - Sequential flow (A first, B later)
    """
    t = np.arange(t_span[0], t_span[1], dt)
    n_steps = len(t)
    
    # Initialize state arrays
    C = np.zeros((3, n_steps))  # Consciousness per node
    qA = np.zeros((3, n_steps))  # Opiate concentration
    qB = np.zeros((3, n_steps))  # Norcain concentration
    
    # Initial conditions
    # Node 0: initial opiate dose
    qA[0, 0] = 1.0
    # All nodes start fully conscious
    C[:, 0] = 1.0
    
    # Dosing schedule: norcain at t=2, 5, 8 seconds at node 0
    dose_times = [2.0, 5.0, 8.0]
    dose_amount = 0.8
    
    # Pharmacodynamic parameters
    alpha = 0.5   # Max activation rate (norcain)
    beta = 0.3    # Max dampening rate (opiate)
    EC50_A = 0.3  # Opiate half-max
    EC50_B = 0.2  # Norcain half-max
    hill_A = 2.0
    hill_B = 2.0
    
    # Simulation loop
    for i in range(n_steps - 1):
        dt_step = t[i+1] - t[i]
        
        for node in graph.nodes:
            # Current values
            C_cur = C[node, i]
            qA_cur = qA[node, i]
            qB_cur = qB[node, i]
            
            # Compute fluxes from incoming edges
            inflow_A = 0
            inflow_B = 0
            
            for edge in graph.edges:
                if edge[1] == node:  # Edge ends at this node
                    src = edge[0]
                    
                    # Delayed arrival
                    delay_A = graph.delay_A if node != src else 0
                    delay_B = graph.delay_B if node != src else 0
                    
                    # Find delayed index
                    idx_A = max(0, i - int(delay_A / dt_step))
                    idx_B = max(0, i - int(delay_B / dt_step))
                    
                    # Flow rates
                    rate_A = graph.transition_rate(edge, t[i], 'A')
                    rate_B = graph.transition_rate(edge, t[i], 'B')
                    
                    inflow_A += rate_A * qA[src, idx_A] * dt_step
                    inflow_B += rate_B * qB[src, idx_B] * dt_step
            
            # Outgoing fluxes
            outflow_A = 0
            outflow_B = 0
            
            for edge in graph.edges:
                if edge[0] == node:
                    rate_A = graph.transition_rate(edge, t[i], 'A')
                    rate_B = graph.transition_rate(edge, t[i], 'B')
                    outflow_A += rate_A * qA_cur * dt_step
                    outflow_B += rate_B * qB_cur * dt_step
            
            # Decay
            decay_A = graph.half_life_A * dt_step
            decay_B = graph.half_life_B * dt_step
            
            # Update concentrations
            qA[node, i+1] = qA_cur + inflow_A - outflow_A - qA_cur * dt_step / graph.half_life_A
            qB[node, i+1] = qB_cur + inflow_B - outflow_B - qB_cur * dt_step / graph.half_life_B
            
            # Add dose
            if node == 0:
                for dose_time in dose_times:
                    if abs(t[i] - dose_time) < dt_step:
                        qB[node, i+1] += dose_amount
            
            # Clamp concentrations
            qA[node, i+1] = np.clip(qA[node, i+1], 0, 2)
            qB[node, i+1] = np.clip(qB[node, i+1], 0, 2)
            
            # Pharmacodynamics: Consciousness dynamics
            activation = alpha * (1 - C_cur) * (qB_cur**hill_B) / (EC50_B**hill_B + qB_cur**hill_B)
            dampening = beta * C_cur * (qA_cur**hill_A) / (EC50_A**hill_A + qA_cur**hill_A)
            
            dC = (activation - dampening) * dt_step
            
            C[node, i+1] = np.clip(C_cur + dC, 0, 1)
    
    return t, C, qA, qB


# ============================================================================
# PART 3: TORIC VARIETY AND WAVELET MAPPING
# ============================================================================

def compute_toric_variety(C, qA, qB):
    """
    Compute toric deviation from 2×2 minor condition.
    Also compute Plücker coordinates for Gr(2,4).
    """
    n_nodes, n_times = C.shape
    
    epsilon = np.zeros_like(C)  # Toric deviation
    plucker = np.zeros((n_nodes, n_times, 6))  # Plücker coordinates
    
    for node in range(n_nodes):
        for i in range(n_times):
            # Probability distribution approximation from consciousness and concentrations
            # HH: both molecules in H state (consciousness high)
            p_HH = C[node, i] * np.exp(-qA[node, i] - qB[node, i])
            # HT: A in H, B in T
            p_HT = C[node, i] * (1 - np.exp(-qB[node, i]))
            # TH: A in T, B in H
            p_TH = (1 - C[node, i]) * np.exp(-qA[node, i])
            # TT: both in T
            p_TT = (1 - C[node, i]) * (1 - np.exp(-qA[node, i]))
            
            # Normalize
            total = p_HH + p_HT + p_TH + p_TT
            if total > 0:
                p_HH /= total
                p_HT /= total
                p_TH /= total
                p_TT /= total
            
            # Toric deviation: ε = p_HH·p_TT - p_HT·p_TH
            epsilon[node, i] = p_HH * p_TT - p_HT * p_TH
            
            # Plücker coordinates (mapping to Gr(2,4))
            plucker[node, i, 0] = C[node, i]  # p12
            plucker[node, i, 1] = qB[node, i] / (1 + qB[node, i])  # p13
            plucker[node, i, 2] = (1 - C[node, i]) * np.exp(-qA[node, i])  # p14
            plucker[node, i, 3] = C[node, i] * np.exp(-qB[node, i])  # p23
            plucker[node, i, 4] = qA[node, i] / (1 + qA[node, i])  # p24
            plucker[node, i, 5] = 1 - C[node, i]  # p34
            
            # Normalize Plücker coordinates
            norm = np.linalg.norm(plucker[node, i])
            if norm > 0:
                plucker[node, i] /= norm
    
    return epsilon, plucker


def wavelet_analysis(C, t, scales=None):
    """
    Continuous wavelet transform for consciousness signals.
    
    Parameters:
    -----------
    C : np.ndarray, shape (n_nodes, n_times)
        Consciousness signals
    t : np.ndarray, shape (n_times,)
        Time points
    scales : np.ndarray, optional
        Scales in seconds (physical time)
    
    Returns:
    --------
    wavelet_coeffs : np.ndarray, shape (n_nodes, n_scales, n_times)
        Wavelet coefficients
    energy : np.ndarray, shape (n_nodes, n_scales)
        Wavelet energy
    scales : np.ndarray
        Actual scales used
    """
    from scipy.signal import cwt
    import numpy as np
    
    if scales is None:
        scales = np.arange(2, 32)  # scales in samples
    
    n_nodes, n_times = C.shape
    wavelet_coeffs = np.zeros((n_nodes, len(scales), n_times), dtype=complex)
    
    # Compute sampling interval
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    
    # Convert physical scales to sample widths if needed
    # If scales are large (>10), assume they're physical seconds
    if np.max(scales) > 10:
        widths = scales / dt
    else:
        widths = scales
    
    # Filter valid widths
    valid_mask = widths >= 1
    valid_widths = widths[valid_mask]
    valid_scales = scales[valid_mask]
    
    if len(valid_widths) == 0:
        # Fallback: use simple integer widths
        valid_widths = np.arange(2, min(32, n_times//2))
        valid_scales = valid_widths
    
    # Create wavelet wrapper that adapts to cwt's calling convention
    def wavelet_wrapper(width, t_points):
        """Wrapper for cwt that returns a morlet wavelet."""
        from scipy.signal import morlet
        M = len(t_points)
        # width controls the spread: wavelet covers ~6*sigma samples
        sigma = width / 6.0
        return morlet(M, w=5.0, s=sigma, complete=True).real
    
    for node in range(n_nodes):
        signal = C[node, :]
        
        try:
            # Compute CWT using the wrapper
            coeffs = cwt(signal, wavelet_wrapper, valid_widths)
            wavelet_coeffs[node] = coeffs
        except Exception as e:
            print(f"Warning: CWT failed for node {node}: {e}")
            # Fallback to simple frequency analysis
            from scipy.fft import fft, fftfreq
            freqs = fftfreq(len(signal), dt)
            fft_vals = fft(signal)
            for i, w in enumerate(valid_widths):
                target_freq = 1.0 / (w * dt) if w > 0 else 0
                idx = np.argmin(np.abs(freqs - target_freq))
                coeffs = np.ones(len(signal)) * np.abs(fft_vals[idx])
                wavelet_coeffs[node, i] = coeffs
    
    # Compute wavelet energy (sum over time at each scale)
    energy = np.sum(np.abs(wavelet_coeffs)**2, axis=2)
    
    return wavelet_coeffs, energy, valid_scales


# ============================================================================
# PART 4: REVERSE HIRONAKA TRAJECTORY TRACKING
# ============================================================================

def track_reverse_hironaka_trajectories(plucker, C, t, threshold=0.3):
    """
    Track reverse Hironaka trajectories from singular points (C < threshold)
    back to smooth points along gradient of consciousness.
    """
    trajectories = []
    
    # Find singular points
    for node in range(C.shape[0]):
        singular_indices = np.where(C[node, :] < threshold)[0]
        
        for idx in singular_indices:
            if idx < 10:
                continue
            
            # Track backward in time
            trajectory = []
            current_idx = idx
            
            while current_idx > 0 and len(trajectory) < 100:
                # Add current point
                trajectory.append({
                    'time': t[current_idx],
                    'consciousness': C[node, current_idx],
                    'plucker': plucker[node, current_idx].copy(),
                    'node': node
                })
                
                # Move backward
                current_idx -= 1
                
                # Stop if we reach smooth region
                if C[node, current_idx] > 0.7:
                    trajectory.append({
                        'time': t[current_idx],
                        'consciousness': C[node, current_idx],
                        'plucker': plucker[node, current_idx].copy(),
                        'node': node
                    })
                    break
            
            if len(trajectory) > 3:
                trajectories.append({
                    'node': node,
                    'singular_time': t[idx],
                    'singular_consciousness': C[node, idx],
                    'trajectory': trajectory,
                    'length': len(trajectory),
                    'end_consciousness': trajectory[-1]['consciousness']
                })
    
    return trajectories


# ============================================================================
# PART 5: PRIME PATH ZETA FUNCTION
# ============================================================================

def compute_path_zeta(trajectory):
    """Compute zeta function value for a path on the Klein quadric"""
    if len(trajectory) < 2:
        return 0 + 0j
    
    # Compute path length in projective space
    lengths = []
    phases = []
    
    for i in range(len(trajectory) - 1):
        p1 = trajectory[i]['plucker']
        p2 = trajectory[i+1]['plucker']
        
        # Fubini-Study distance
        inner = np.abs(np.dot(p1, p2))
        dist = np.arccos(np.clip(inner, -1, 1))
        lengths.append(dist)
        
        # Geometric phase
        phase1 = np.angle(p1[0] + 1j * p1[1])
        phase2 = np.angle(p2[0] + 1j * p2[1])
        phases.append(phase2 - phase1)
    
    path_length = np.sum(lengths)
    total_phase = np.sum(phases)
    
    # Zeta function on critical line
    s = 0.5 + 1j
    zeta = np.exp(-s * path_length) * np.exp(1j * total_phase)
    
    return zeta


def identify_prime_paths(trajectories):
    """Identify prime paths using zeta function zeros"""
    prime_paths = []
    
    for i, traj in enumerate(trajectories):
        zeta_val = compute_path_zeta(traj['trajectory'])
        
        # Path is prime if zeta is near zero (high curvature)
        is_prime = abs(zeta_val) < 0.3
        
        if is_prime:
            prime_paths.append({
                'prime_index': len(prime_paths) + 1,
                'node': traj['node'],
                'singular_time': traj['singular_time'],
                'singular_consciousness': traj['singular_consciousness'],
                'zeta_value': zeta_val,
                'path_length': traj['length']
            })
    
    return prime_paths


# ============================================================================
# PART 6: MAIN EXECUTION WITH PLOTTING
# ============================================================================

def run_complete_analysis():
    """Run the complete 3-node, 7-edge graph analysis and produce charts"""
    
    print("="*80)
    print("3-NODE, 7-EDGE GRAPH ANALYSIS")
    print("Wavelet-Spectral Reverse Hironaka Framework")
    print("="*80)
    
    # Initialize graph
    graph = ThreeNodeGraph()
    print(f"\nGraph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print("Edges:", graph.edges)
    
    # Simulate consciousness dynamics
    print("\n[1] Simulating consciousness dynamics...")
    t, C, qA, qB = simulate_consciousness_dynamics(graph)
    print(f"    Time points: {len(t)}")
    print(f"    Consciousness range: [{C.min():.3f}, {C.max():.3f}]")
    
    # Compute toric variety and Plücker coordinates
    print("\n[2] Computing toric variety and Plücker coordinates...")
    epsilon, plucker = compute_toric_variety(C, qA, qB)
    print(f"    Toric deviation range: [{epsilon.min():.3f}, {epsilon.max():.3f}]")
    
    # Wavelet analysis
    print("\n[3] Performing wavelet analysis...")
    wavelet_coeffs, wavelet_energy, scales = wavelet_analysis(C, t)
    print(f"    Wavelet scales: {len(scales)}")
    print(f"    Max wavelet energy: {wavelet_energy.max():.3f}")
    
    # Track reverse Hironaka trajectories
    print("\n[4] Tracking reverse Hironaka trajectories...")
    trajectories = track_reverse_hironaka_trajectories(plucker, C, t)
    print(f"    Singular points found: {len(trajectories)}")
    for traj in trajectories:
        print(f"    Node {traj['node']}: t={traj['singular_time']:.2f}s, C={traj['singular_consciousness']:.3f}")
    
    # Identify prime paths
    print("\n[5] Identifying prime paths via zeta function...")
    prime_paths = identify_prime_paths(trajectories)
    print(f"    Prime paths identified: {len(prime_paths)}")
    for pp in prime_paths:
        print(f"    Prime {pp['prime_index']}: Node {pp['node']}, t={pp['singular_time']:.2f}s, |ζ|={abs(pp['zeta_value']):.4f}")
    
    # Create comprehensive visualization
    print("\n[6] Generating comprehensive charts...")
    create_comprehensive_charts(t, C, qA, qB, epsilon, plucker, 
                                wavelet_energy, scales, trajectories, prime_paths)
    
    # Final summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    return {
        't': t, 'C': C, 'qA': qA, 'qB': qB,
        'epsilon': epsilon, 'plucker': plucker,
        'wavelet_energy': wavelet_energy, 'scales': scales,
        'trajectories': trajectories, 'prime_paths': prime_paths
    }


def create_comprehensive_charts(t, C, qA, qB, epsilon, plucker, 
                                 wavelet_energy, scales, trajectories, prime_paths):
    """Create all visualization charts"""
    try:
        import pywt
        has_pywt = True
    except ImportError:
        has_pywt = False
        print("PyWavelets not installed. Install with: pip install PyWavelets")
    
    fig = plt.figure(figsize=(20, 24))
    
    # ========================================================================
    # Row 1: Consciousness Dynamics and Molecular Concentrations
    # ========================================================================
    
    # 1. Consciousness over time for all nodes
    ax1 = plt.subplot(5, 4, 1)
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    labels = ['Node 0 (Injection)', 'Node 1 (Hub)', 'Node 2 (Distal)']
    for node in range(3):
        ax1.plot(t, C[node, :], color=colors[node], linewidth=2, label=labels[node])
    ax1.axhline(y=0.3, color='red', linestyle='--', linewidth=1, label='Threshold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Consciousness')
    ax1.set_title('Consciousness Dynamics Across Nodes')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # 2. Opiate (A) concentrations
    ax2 = plt.subplot(5, 4, 2)
    for node in range(3):
        ax2.plot(t, qA[node, :], color=colors[node], linewidth=1.5)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Concentration')
    ax2.set_title('Opiate (A) Concentration')
    ax2.grid(True, alpha=0.3)
    
    # 3. Norcain (B) concentrations with dosing
    ax3 = plt.subplot(5, 4, 3)
    for node in range(3):
        ax3.plot(t, qB[node, :], color=colors[node], linewidth=1.5)
    ax3.axvline(x=2, color='green', linestyle=':', alpha=0.5, label='Doses')
    ax3.axvline(x=5, color='green', linestyle=':', alpha=0.5)
    ax3.axvline(x=8, color='green', linestyle=':', alpha=0.5)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Concentration')
    ax3.set_title('Norcain (B) Concentration')
    ax3.grid(True, alpha=0.3)
    
    # 4. Toric deviation (ε)
    ax4 = plt.subplot(5, 4, 4)
    for node in range(3):
        ax4.plot(t, epsilon[node, :], color=colors[node], linewidth=1.5)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('ε')
    ax4.set_title('Toric Deviation: p_HH·p_TT - p_HT·p_TH')
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # Row 2: Wavelet Analysis
    # ========================================================================
    
    # 5. Wavelet energy spectrum
    ax5 = plt.subplot(5, 4, 5)
    for node in range(3):
        ax5.plot(scales, wavelet_energy[node, :len(scales)], color=colors[node], 
                linewidth=1.5, label=labels[node])
    ax5.set_xlabel('Scale')
    ax5.set_ylabel('Energy')
    ax5.set_title('Wavelet Energy Spectrum')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # 6. Wavelet scalogram for Node 0
    ax6 = plt.subplot(5, 4, 6)
    from scipy.signal import cwt, morlet
    if has_pywt:
        signal = C[0, :]
        dt = t[1] - t[0]
        scales_full = np.arange(2, 64)
        
        # PyWavelets handles dt correctly
        coeffs, freqs = pywt.cwt(signal, scales_full, 'morl', dt)
        im = ax6.imshow(np.abs(coeffs), aspect='auto', cmap='hot',
                        extent=[t[0], t[-1], scales_full[0], scales_full[-1]])
    else:
        signal = C[0, :]
        dt = t[1] - t[0]
        scales_full = np.arange(2, 64)
        #coeffs = cwt(signal, morlet, scales_full, dt=dt)
        dt = t[1] - t[0]
        scales_full = np.arange(2, 64)
        widths = scales_full / dt
        valid_mask = widths >= 1
        valid_widths = widths[valid_mask]

        def wavelet_wrapper(width, t_points):
            from scipy.signal import morlet
            M = len(t_points)
            sigma = width / 6.0
            return morlet(M, w=5.0, s=sigma, complete=True).real

        coeffs = cwt(signal, wavelet_wrapper, valid_widths)
        im = ax6.imshow(np.abs(coeffs), aspect='auto', cmap='hot', 
                        extent=[t[0], t[-1], scales_full[0], scales_full[-1]])
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Scale')
    ax6.set_title('Wavelet Scalogram - Node 0')
    plt.colorbar(im, ax=ax6, label='|Coefficient|')
    
    # 7. Wavelet coherence between nodes
    ax7 = plt.subplot(5, 4, 7)
    # Compute wavelet coherence (simplified)
    from scipy.signal import coherence
    f, coh = coherence(C[0, :], C[1, :], fs=1/dt)
    ax7.semilogy(f[1:], coh[1:], 'b-', linewidth=1.5)
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Coherence')
    ax7.set_title('Wavelet Coherence: Node 0-1')
    ax7.set_ylim([0, 1])
    ax7.grid(True, alpha=0.3)
    
    # 8. Phase space (C vs dC/dt)
    ax8 = plt.subplot(5, 4, 8)
    dC = np.gradient(C[0, :], t)
    ax8.plot(C[0, :], dC, 'b-', linewidth=1, alpha=0.7)
    ax8.scatter(C[0, 0], dC[0], c='green', s=50, label='Start')
    ax8.scatter(C[0, -1], dC[-1], c='red', s=50, label='End')
    ax8.set_xlabel('Consciousness C')
    ax8.set_ylabel('dC/dt')
    ax8.set_title('Phase Space - Node 0')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # ========================================================================
    # Row 3: Plücker Trajectories on Klein Quadric (3D)
    # ========================================================================
    
    # 9. 3D Plücker trajectory for Node 0
    ax9 = fig.add_subplot(5, 4, 9, projection='3d')
    p0 = plucker[0, :, :]
    # Color by time
    colors_3d = plt.cm.viridis(np.linspace(0, 1, len(t)))
    for i in range(len(t)-1):
        ax9.plot(p0[i:i+2, 0], p0[i:i+2, 1], p0[i:i+2, 2], 
                color=colors_3d[i], linewidth=1, alpha=0.5)
    ax9.scatter(p0[0, 0], p0[0, 1], p0[0, 2], c='green', s=50, label='Start')
    ax9.scatter(p0[-1, 0], p0[-1, 1], p0[-1, 2], c='red', s=50, label='End')
    ax9.set_xlabel('p12')
    ax9.set_ylabel('p13')
    ax9.set_zlabel('p14')
    ax9.set_title('Plücker Trajectory - Node 0')
    ax9.legend()
    
    # 10. 3D Plücker trajectory for Node 1
    ax10 = fig.add_subplot(5, 4, 10, projection='3d')
    p1 = plucker[1, :, :]
    for i in range(len(t)-1):
        ax10.plot(p1[i:i+2, 0], p1[i:i+2, 1], p1[i:i+2, 2], 
                 color=colors_3d[i], linewidth=1, alpha=0.5)
    ax10.scatter(p1[0, 0], p1[0, 1], p1[0, 2], c='green', s=50)
    ax10.scatter(p1[-1, 0], p1[-1, 1], p1[-1, 2], c='red', s=50)
    ax10.set_xlabel('p12')
    ax10.set_ylabel('p13')
    ax10.set_zlabel('p14')
    ax10.set_title('Plücker Trajectory - Node 1')
    
    # 11. 3D Plücker trajectory for Node 2
    ax11 = fig.add_subplot(5, 4, 11, projection='3d')
    p2 = plucker[2, :, :]
    for i in range(len(t)-1):
        ax11.plot(p2[i:i+2, 0], p2[i:i+2, 1], p2[i:i+2, 2], 
                 color=colors_3d[i], linewidth=1, alpha=0.5)
    ax11.scatter(p2[0, 0], p2[0, 1], p2[0, 2], c='green', s=50)
    ax11.scatter(p2[-1, 0], p2[-1, 1], p2[-1, 2], c='red', s=50)
    ax11.set_xlabel('p12')
    ax11.set_ylabel('p13')
    ax11.set_zlabel('p14')
    ax11.set_title('Plücker Trajectory - Node 2')
    
    # 12. Plücker relation verification
    ax12 = plt.subplot(5, 4, 12)
    plucker_relation = plucker[:, :, 0] * plucker[:, :, 5] - \
                       plucker[:, :, 1] * plucker[:, :, 4] + \
                       plucker[:, :, 2] * plucker[:, :, 3]
    for node in range(3):
        ax12.plot(t, plucker_relation[node, :], color=colors[node], linewidth=1.5)
    ax12.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('Plücker Relation')
    ax12.set_title('Klein Quadric Verification (should be 0)')
    ax12.grid(True, alpha=0.3)
    
    # ========================================================================
    # Row 4: Reverse Hironaka Trajectories
    # ========================================================================
    
    # 13-15: Reverse Hironaka trajectories for each node
    for idx, node in enumerate(range(3)):
        ax = plt.subplot(5, 4, 13 + idx)
        
        # Plot full Plücker trajectory
        p_full = plucker[node, :, :2]  # Use first two coordinates for 2D view
        ax.plot(p_full[:, 0], p_full[:, 1], 'gray', linewidth=1, alpha=0.3, label='Full path')
        
        # Plot reverse trajectories
        node_trajectories = [t for t in trajectories if t['node'] == node]
        colors_traj = ['red', 'orange', 'darkred']
        for i, traj in enumerate(node_trajectories[:3]):
            traj_points = np.array([pt['plucker'][:2] for pt in traj['trajectory']])
            ax.plot(traj_points[:, 0], traj_points[:, 1], 
                   color=colors_traj[i % len(colors_traj)], 
                   linewidth=2, marker='o', markersize=3,
                   label=f"t={traj['singular_time']:.1f}s")
        
        ax.set_xlabel('p12')
        ax.set_ylabel('p13')
        ax.set_title(f'Reverse Hironaka Trajectories - Node {node}')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    
    # 16: Prime path summary
    ax16 = plt.subplot(5, 4, 16)
    if prime_paths:
        prime_indices = [p['prime_index'] for p in prime_paths]
        zeta_abs = [abs(p['zeta_value']) for p in prime_paths]
        ax16.bar(prime_indices, zeta_abs, color='red', alpha=0.7)
        ax16.axhline(y=0.3, color='black', linestyle='--', label='Prime threshold')
        ax16.set_xlabel('Prime Path Index')
        ax16.set_ylabel('|ζ(s)|')
        ax16.set_title('Prime Path Zeta Values')
        ax16.legend()
    else:
        ax16.text(0.5, 0.5, 'No prime paths detected\n(Zeta values > threshold)', 
                 ha='center', va='center', transform=ax16.transAxes)
        ax16.set_title('Prime Path Detection')
    
    # ========================================================================
    # Row 5: Spectral and Coherence Analysis
    # ========================================================================
    
    # 17: Spectral decomposition of Plücker trajectories
    ax17 = plt.subplot(5, 4, 17)
    for node in range(3):
        # Compute covariance matrix of Plücker coordinates
        P = plucker[node, :, :]
        cov = np.cov(P.T)
        eigvals = np.linalg.eigvalsh(cov)
        ax17.plot(eigvals[::-1], 'o-', color=colors[node], label=labels[node], markersize=4)
    ax17.set_xlabel('Eigenvalue Index')
    ax17.set_ylabel('Eigenvalue')
    ax17.set_title('Spectral Decomposition of Plücker Trajectories')
    ax17.legend(fontsize=7)
    ax17.grid(True, alpha=0.3)
    
    # 18: Spectral gap over time
    ax18 = plt.subplot(5, 4, 18)
    window = 20
    spectral_gaps = []
    for i in range(0, len(t) - window, window//2):
        P_window = plucker[0, i:i+window, :]
        cov = np.cov(P_window.T)
        eigvals = np.linalg.eigvalsh(cov)
        gap = eigvals[-1] - eigvals[-2] if len(eigvals) > 1 else 0
        spectral_gaps.append((t[i], gap))
    if spectral_gaps:
        gap_times, gap_vals = zip(*spectral_gaps)
        ax18.plot(gap_times, gap_vals, 'b-', linewidth=1.5)
    ax18.set_xlabel('Time (s)')
    ax18.set_ylabel('Spectral Gap')
    ax18.set_title('Spectral Gap Evolution - Node 0')
    ax18.grid(True, alpha=0.3)
    
    # 19: Toric vs Grassmannian deviation correlation
    ax19 = plt.subplot(5, 4, 19)
    grassmannian_dev = plucker[:, :, 0] * plucker[:, :, 5] - \
                       plucker[:, :, 1] * plucker[:, :, 4] + \
                       plucker[:, :, 2] * plucker[:, :, 3]
    for node in range(3):
        ax19.scatter(epsilon[node, ::10], grassmannian_dev[node, ::10], 
                    color=colors[node], alpha=0.5, s=10, label=labels[node])
    ax19.set_xlabel('Toric Deviation ε')
    ax19.set_ylabel('Grassmannian Deviation δ')
    ax19.set_title('Toric vs Grassmannian Deviation')
    ax19.legend(fontsize=7)
    ax19.grid(True, alpha=0.3)
    
    # 20: Phase transition summary
    ax20 = plt.subplot(5, 4, 20)
    # Detect phase transitions (C < 0.3)
    for node in range(3):
        below_threshold = C[node, :] < 0.3
        transitions = np.where(np.diff(below_threshold.astype(int)) != 0)[0]
        for trans in transitions:
            if trans < len(t):
                ax20.axvline(x=t[trans], color=colors[node], alpha=0.5, linewidth=1)
    ax20.plot(t, np.mean(C, axis=0), 'k-', linewidth=2, label='Mean Consciousness')
    ax20.axhline(y=0.3, color='red', linestyle='--', label='Threshold')
    ax20.set_xlabel('Time (s)')
    ax20.set_ylabel('Consciousness')
    ax20.set_title('Phase Transitions (vertical lines)')
    ax20.legend()
    ax20.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reverse_hironaka_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nChart saved: reverse_hironaka_analysis.png")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = run_complete_analysis()
    
    # Print final summary statistics
    print("\n" + "="*80)
    print("FINAL SUMMARY STATISTICS")
    print("="*80)
    
    print(f"\nConsciousness Statistics:")
    for node in range(3):
        C_node = results['C'][node, :]
        print(f"  Node {node}: min={C_node.min():.3f}, max={C_node.max():.3f}, "
              f"mean={C_node.mean():.3f}, std={C_node.std():.3f}")
    
    print(f"\nToric Deviation Statistics:")
    for node in range(3):
        eps_node = results['epsilon'][node, :]
        print(f"  Node {node}: min={eps_node.min():.3f}, max={eps_node.max():.3f}, "
              f"mean={np.abs(eps_node).mean():.3f}")
    
    print(f"\nReverse Hironaka Trajectories: {len(results['trajectories'])}")
    print(f"Prime Paths Identified: {len(results['prime_paths'])}")
    
    if results['prime_paths']:
        print("\nPrime Path Details:")
        for pp in results['prime_paths']:
            print(f"  Prime {pp['prime_index']}: Node {pp['node']}, "
                  f"Time {pp['singular_time']:.2f}s, "
                  f"|ζ|={abs(pp['zeta_value']):.4f}")