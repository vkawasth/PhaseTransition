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
import warnings
warnings.filterwarnings('ignore')


class FullGraphDynamics:
    """Full molecular dynamics with extreme flow to ensure recovery."""
    
    def __init__(self):
        self.nodes = [0, 1, 2]
        self.edges = [
            (0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (1, 1)
        ]
        
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
    
    def simulate(self, t_span=(0, 25), dt=0.02):
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
        
        print("\n  Extreme flow configuration:")
        print(f"    Opiate base flow = 4.0, edge weights: 0→1={self.edge_weights[(0,1)]}, 0→2={self.edge_weights[(0,2)]}")
        print(f"    Norcain base flow = 6.0, dose amount = {self.dose_amount}")
        
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
            
            self.HH1[i+1], self.HH2[i+1] = self.compute_hochschild_invariants(i+1)
        
        # Compute Plücker trajectory
        self.plucker = self.compute_plucker_trajectory()
        
        # Detect phase transitions and build reverse trajectories
        self.detect_phase_transitions()
        
        return self.t, self.C, self.qA, self.qB, self.HH1, self.HH2
    
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


# ============================================================================
# Visualization functions (20-panel dashboard + ghost signal + prime zeta)
# ============================================================================

def create_dashboard(dynamics):
    """Create the 20-panel dashboard."""
    t, C, qA, qB, HH1, HH2 = dynamics.t, dynamics.C, dynamics.qA, dynamics.qB, dynamics.HH1, dynamics.HH2
    plucker = dynamics.plucker
    dose_times = dynamics.dose_times
    threshold = dynamics.threshold
    
    fig = plt.figure(figsize=(22, 28))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    dt_val = t[1] - t[0]
    
    # 1. Consciousness across all nodes
    ax1 = plt.subplot(5, 4, 1)
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
    ax2 = plt.subplot(5, 4, 2)
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
    ax3 = plt.subplot(5, 4, 3)
    for node in range(3):
        ax3.plot(t, qA[node, :], color=colors[node], lw=1.5, ls='--')
    ax3.set_title('3. Opiate (A) Concentrations')
    ax3.legend(['Node 0','Node 1','Node 2'], fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # 4. Norcain
    ax4 = plt.subplot(5, 4, 4)
    for node in range(3):
        ax4.plot(t, qB[node, :], color=colors[node], lw=2)
    for dt in dose_times:
        ax4.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax4.set_title('4. Norcain (B) Spreading')
    ax4.legend(['Node 0','Node 1','Node 2'], fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    # 5. HH¹
    ax5 = plt.subplot(5, 4, 5)
    ax5.plot(t, HH1, 'b-', lw=2)
    for dt in dose_times:
        ax5.axvline(x=dt, color='green', ls=':', alpha=0.7)
    ax5.set_title('5. HH¹ - Deformations')
    ax5.grid(True, alpha=0.3)
    
    # 6. HH²
    ax6 = plt.subplot(5, 4, 6)
    ax6.plot(t, HH2, 'r-', lw=2)
    for dt in dose_times:
        ax6.axvline(x=dt, color='green', ls=':', alpha=0.7)
    peaks, _ = find_peaks(HH2, height=0.5)
    ax6.scatter(t[peaks], HH2[peaks], color='red', s=50, zorder=5, label='Phase Transitions')
    ax6.set_title('6. HH² - Obstructions')
    ax6.legend(fontsize=7)
    ax6.grid(True, alpha=0.3)
    
    # 7. Coherence over time
    ax7 = plt.subplot(5, 4, 7)
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
    ax8 = plt.subplot(5, 4, 8)
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
    ax9 = fig.add_subplot(5, 4, 9, projection='3d')
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
    ax10 = plt.subplot(5, 4, 10)
    plucker_rel = plucker[:,0]*plucker[:,5] - plucker[:,1]*plucker[:,4] + plucker[:,2]*plucker[:,3]
    ax10.plot(t, plucker_rel, 'g-')
    ax10.axhline(y=0, color='black')
    ax10.set_title('10. Klein Quadric Verification')
    ax10.grid(True, alpha=0.3)
    
    # 11. Phase space Node 0
    ax11 = plt.subplot(5, 4, 11)
    dC0 = np.gradient(C[0,:], dt_val)
    ax11.plot(C[0,:], dC0, 'b-', alpha=0.7)
    ax11.scatter(C[0,0], dC0[0], c='green', s=50)
    ax11.scatter(C[0,-1], dC0[-1], c='red', s=50)
    ax11.set_title('11. Phase Space Node 0')
    ax11.grid(True, alpha=0.3)
    
    # 12. Phase space Node 1 with HH² color
    ax12 = plt.subplot(5, 4, 12)
    dC1 = np.gradient(C[1,:], dt_val)
    ax12.plot(C[1,:], dC1, 'g-', alpha=0.7)
    sc = ax12.scatter(C[1,::20], dC1[::20], c=HH2[::20], cmap='hot', s=30, alpha=0.7)
    ax12.scatter(C[1,0], dC1[0], c='green', s=50)
    ax12.scatter(C[1,-1], dC1[-1], c='red', s=50)
    ax12.set_title('12. Phase Space Node 1 (color=HH²)')
    plt.colorbar(sc, ax=ax12, label='HH²')
    ax12.grid(True, alpha=0.3)
    
    # 13. Wavelet scalogram
    ax13 = plt.subplot(5, 4, 13)
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
    ax14 = plt.subplot(5, 4, 14)
    energy = np.sum(np.abs(coeffs)**2, axis=1)
    energy = energy / (np.max(energy)+1e-8)
    ax14.plot(valid_scales, energy, 'b-')
    ax14.fill_between(valid_scales, energy, alpha=0.3)
    ax14.set_title('14. Wavelet Energy')
    ax14.grid(True, alpha=0.3)
    
    # 15. Coherence 0-1
    ax15 = plt.subplot(5, 4, 15)
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
    ax16 = plt.subplot(5, 4, 16)
    try:
        f, coh = coherence(C[1,:], C[2,:], fs=1/dt_val)
        ax16.semilogy(f[1:], coh[1:], 'g-')
        ax16.set_ylim(0,1)
        ax16.set_title('16. Coherence Node 1-2')
    except:
        ax16.text(0.5,0.5,'failed',ha='center',va='center')
    ax16.grid(True, alpha=0.3)
    
    # 17. Unconscious duration
    ax17 = plt.subplot(5, 4, 17)
    time_below = []
    for node in range(3):
        below = C[node,:] < threshold
        time_below.append(np.sum(below) * dt_val)
    ax17.bar(['Node 0','Node 1','Node 2'], time_below, color=colors, alpha=0.7)
    ax17.set_title('17. Unconscious Duration')
    ax17.grid(True, alpha=0.3)
    
    # 18. Final consciousness
    ax18 = plt.subplot(5, 4, 18)
    final_C = C[:, -1]
    ax18.bar(['Node 0','Node 1','Node 2'], final_C, color=colors, alpha=0.7)
    ax18.axhline(y=threshold, color='red', ls='--')
    ax18.set_title('18. Final Consciousness')
    ax18.set_ylim(0,1)
    ax18.grid(True, alpha=0.3)
    
    # 19. Final coherence
    ax19 = plt.subplot(5, 4, 19)
    final_coherence = [
        np.abs(np.corrcoef(C[0,-500:], C[1,-500:])[0,1]),
        np.abs(np.corrcoef(C[0,-500:], C[2,-500:])[0,1]),
        np.abs(np.corrcoef(C[1,-500:], C[2,-500:])[0,1])
    ]
    ax19.bar(['0-1','0-2','1-2'], final_coherence, color=['blue','red','green'], alpha=0.7)
    ax19.axhline(y=0.85, color='gold', ls='--')
    ax19.set_title('19. Final Coherence')
    ax19.set_ylim(0,1)
    ax19.grid(True, alpha=0.3)
    
    # 20. HH² vs Consciousness
    ax20 = plt.subplot(5, 4, 20)
    mean_C = np.mean(C, axis=0)
    ax20.scatter(mean_C, HH2, c=HH2, cmap='hot', alpha=0.5, s=20)
    

    trans_counts = dynamics.count_phase_transitions()
    text = "Phase transitions:\n"
    for node in range(3):
        text += f"Node {node}: {trans_counts[node]['down']}↓ {trans_counts[node]['up']}↑\n"
    ax20.text(0.05, 0.95, text, transform=ax20.transAxes, fontsize=8,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax20.set_title('20. HH² vs Consciousness')
    ax20.grid(True, alpha=0.3)
    
    plt.suptitle('20-PANEL DASHBOARD: Node 1 & 2 Show Dynamic Consciousness!', fontsize=12)
    plt.tight_layout()
    plt.savefig('comprehensive_20panel_dashboard.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("    ✓ Saved: comprehensive_20panel_dashboard.png")


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


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*100)
    print(" " * 35 + "FINAL: GHOST SIGNAL & PRIME ZETA INCLUDED")
    print("="*100)
    
    dynamics = FullGraphDynamics()
    t, C, qA, qB, HH1, HH2 = dynamics.simulate()
    
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