"""
COMPREHENSIVE TRAJECTORY CHARTS FOR 3-NODE, 7-EDGE GRAPH
=========================================================
Generates detailed trajectory visualizations including:
- Plücker trajectories on the Klein quadric (3D)
- Reverse Hironaka trajectories from singular points
- Prime path identification with zeta function signatures
- Phase space trajectories
- Wavelet-scaled trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.animation as animation
from scipy.signal import cwt, morlet, find_peaks
import warnings
warnings.filterwarnings('ignore')

# Set up beautiful plotting style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150

# Create custom colormap for trajectories
colors_trajectory = plt.cm.viridis
colors_time = plt.cm.plasma

# ============================================================================
# GENERATE TRAJECTORY DATA
# ============================================================================

def generate_trajectory_data():
    """
    Generate realistic trajectory data for 3 nodes with phase transitions
    """
    t = np.linspace(0, 20, 500)
    
    # Node 0: Injection site - rapid oscillations with collapses at doses
    C0 = 0.5 + 0.3 * np.sin(2 * np.pi * t / 3.2) * np.exp(-t/12)
    C0 = C0 - 0.4 * np.exp(-(t-3.5)**2/0.8) - 0.35 * np.exp(-(t-6.8)**2/0.8) - 0.25 * np.exp(-(t-9.5)**2/0.8)
    C0 = np.clip(C0, 0.12, 0.98)
    
    # Node 1: Recirculation hub - smoother, sustained
    C1 = 0.6 + 0.2 * np.sin(2 * np.pi * t / 5.0) * np.exp(-t/18)
    C1 = C1 - 0.3 * np.exp(-(t-5.2)**2/1.0) - 0.2 * np.exp(-(t-11.0)**2/1.0)
    C1 = np.clip(C1, 0.18, 0.96)
    
    # Node 2: Distal region - delayed, attenuated
    C2 = 0.45 + 0.25 * np.sin(2 * np.pi * (t-1.5) / 4.0) * np.exp(-t/15)
    C2 = C2 - 0.3 * np.exp(-(t-7.8)**2/1.2) - 0.2 * np.exp(-(t-13.2)**2/1.2)
    C2 = np.clip(C2, 0.10, 0.94)
    
    C = np.array([C0, C1, C2])
    
    # Generate Plücker coordinates for each node
    plucker = np.zeros((3, len(t), 6))
    
    for node in range(3):
        for i, ti in enumerate(t):
            # Simulate molecular concentrations
            qA = 0.5 * np.exp(-ti/4) + 0.2 * np.random.randn() * 0.05
            qB = 0.3 * np.exp(-ti/2) + 0.3 * np.exp(-(ti-3)**2/0.5) + 0.3 * np.exp(-(ti-6)**2/0.5) + 0.2 * np.exp(-(ti-9)**2/0.5)
            qB = np.clip(qB, 0, 0.9)
            
            # Plücker coordinates
            plucker[node, i, 0] = C[node, i]  # p12
            plucker[node, i, 1] = qB / (1 + qB)  # p13
            plucker[node, i, 2] = (1 - C[node, i]) * np.exp(-qA)  # p14
            plucker[node, i, 3] = C[node, i] * np.exp(-qB)  # p23
            plucker[node, i, 4] = qA / (1 + qA)  # p24
            plucker[node, i, 5] = 1 - C[node, i]  # p34
            
            # Normalize
            norm = np.linalg.norm(plucker[node, i])
            if norm > 0:
                plucker[node, i] /= norm
    
    return t, C, plucker


def identify_singular_points(C, t, threshold=0.3):
    """Identify singular points where consciousness drops below threshold"""
    singular_points = []
    for node in range(C.shape[0]):
        below = C[node, :] < threshold
        # Find transitions into unconsciousness
        transitions = np.where(np.diff(below.astype(int)) == 1)[0]
        for idx in transitions:
            if idx < len(t) - 1:
                singular_points.append({
                    'node': node,
                    'time': t[idx],
                    'consciousness': C[node, idx],
                    'index': idx,
                    'type': 'collapse'
                })
    return singular_points


def compute_reverse_hironaka_trajectory(plucker, C, t, singular_point, n_steps=50):
    """
    Compute reverse Hironaka trajectory from singular point back along
    gradient of consciousness
    """
    node = singular_point['node']
    start_idx = singular_point['index']
    
    trajectory = []
    current_idx = start_idx
    
    # Move backward in time
    for step in range(n_steps):
        if current_idx < 0:
            break
        
        trajectory.append({
            'time': t[current_idx],
            'consciousness': C[node, current_idx],
            'plucker': plucker[node, current_idx].copy(),
            'step': step
        })
        
        current_idx -= 1
        
        # Stop if we reach smooth region
        if current_idx >= 0 and C[node, current_idx] > 0.7:
            trajectory.append({
                'time': t[current_idx],
                'consciousness': C[node, current_idx],
                'plucker': plucker[node, current_idx].copy(),
                'step': step + 1
            })
            break
    
    return trajectory[::-1]  # Reverse to go from singular to smooth


def compute_path_zeta(trajectory):
    """Compute zeta function for a path"""
    if len(trajectory) < 2:
        return 0 + 0j
    
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
    
    # Zeta on critical line
    s = 0.5 + 1j
    zeta = np.exp(-s * path_length) * np.exp(1j * total_phase)
    
    return zeta


# ============================================================================
# CHART 1: 3D PLÜCKER TRAJECTORIES ON KLEIN QUADRIC
# ============================================================================

def chart_1_plucker_3d_trajectories(plucker, t, C):
    """Create 3D visualization of Plücker trajectories on the Klein quadric"""
    fig = plt.figure(figsize=(18, 10))
    
    # Use p12, p13, p14 for 3D visualization (first three Plücker coordinates)
    coords = [(0, 1, 2), (3, 4, 5), (0, 3, 5)]  # Different projections
    
    for idx, (x_idx, y_idx, z_idx) in enumerate(coords):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        node_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        node_names = ['Node 0 (Injection)', 'Node 1 (Hub)', 'Node 2 (Distal)']
        
        for node in range(3):
            # Extract coordinates
            x = plucker[node, :, x_idx]
            y = plucker[node, :, y_idx]
            z = plucker[node, :, z_idx]
            
            # Color by time
            norm = Normalize(vmin=t[0], vmax=t[-1])
            cmap = plt.cm.viridis
            
            # Create line segments with color gradient
            points = np.array([x, y, z]).T.reshape(-1, 1, 3)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            lc = Line3DCollection(segments, cmap=cmap, norm=norm)
            lc.set_array(t[:-1])
            lc.set_linewidth(2)
            ax.add_collection(lc)
            
            # Mark start and end
            ax.scatter(x[0], y[0], z[0], c='green', s=80, marker='o', 
                      edgecolors='white', linewidth=2, zorder=5)
            ax.scatter(x[-1], y[-1], z[-1], c='red', s=80, marker='s', 
                      edgecolors='white', linewidth=2, zorder=5)
        
        # Add the Klein quadric surface (approximate)
        u = np.linspace(-1, 1, 20)
        v = np.linspace(-1, 1, 20)
        U, V = np.meshgrid(u, v)
        # Approximate quadric: p12*p34 - p13*p24 + p14*p23 = 0
        # For visualization, show a sphere-like shape
        R = 0.8
        X = R * np.cos(U) * np.cos(V)
        Y = R * np.sin(U) * np.cos(V)
        Z = R * np.sin(V)
        ax.plot_wireframe(X, Y, Z, alpha=0.1, color='gray', linewidth=0.5)
        
        ax.set_xlabel(f'p{["12","34","12"][idx]}')
        ax.set_ylabel(f'p{["13","45","34"][idx]}')
        ax.set_zlabel(f'p{["14","56","45"][idx]}')
        ax.set_title(f'Plücker Trajectory Projection {idx+1}')
        
        # Add colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(t)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Time (s)')
    
    plt.suptitle('3D Plücker Trajectories on the Klein Quadric', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_1_plucker_3d.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# ============================================================================
# CHART 2: REVERSE HIRONAKA TRAJECTORIES
# ============================================================================

def chart_2_reverse_hironaka_trajectories(plucker, C, t, singular_points):
    """Create visualization of reverse Hironaka trajectories from singular points"""
    fig = plt.figure(figsize=(18, 12))
    
    # Compute trajectories from singular points
    trajectories = []
    for sp in singular_points:
        traj = compute_reverse_hironaka_trajectory(plucker, C, t, sp, n_steps=80)
        trajectories.append({
            'singular_point': sp,
            'trajectory': traj,
            'zeta': compute_path_zeta(traj)
        })
    
    # Plot 1: 2D projection of all trajectories
    ax1 = plt.subplot(2, 3, 1)
    node_colors = {0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1'}
    
    for traj_data in trajectories:
        sp = traj_data['singular_point']
        traj = traj_data['trajectory']
        color = node_colors[sp['node']]
        
        # Extract p12, p13 for 2D projection
        p12 = [pt['plucker'][0] for pt in traj]
        p13 = [pt['plucker'][1] for pt in traj]
        
        ax1.plot(p12, p13, color=color, linewidth=2, alpha=0.7)
        ax1.scatter(p12[0], p13[0], c=color, s=100, marker='o', edgecolors='white', 
                   linewidth=2, zorder=5, label=f'Node {sp["node"]} at t={sp["time"]:.1f}s')
        ax1.scatter(p12[-1], p13[-1], c=color, s=80, marker='s', edgecolors='white', 
                   linewidth=2, zorder=5)
    
    ax1.set_xlabel('p12')
    ax1.set_ylabel('p13')
    ax1.set_title('Reverse Hironaka Trajectories\n(Singular → Smooth)')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Consciousness along trajectories
    ax2 = plt.subplot(2, 3, 2)
    for traj_data in trajectories:
        sp = traj_data['singular_point']
        traj = traj_data['trajectory']
        color = node_colors[sp['node']]
        
        times = [pt['time'] for pt in traj]
        consc = [pt['consciousness'] for pt in traj]
        
        ax2.plot(times, consc, color=color, linewidth=2, marker='o', markersize=3)
        ax2.scatter(sp['time'], sp['consciousness'], c='red', s=100, marker='x', zorder=5)
    
    ax2.axhline(y=0.3, color='red', linestyle='--', linewidth=1, label='Threshold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Consciousness')
    ax2.set_title('Consciousness Evolution Along Reverse Trajectories')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Zeta values for each trajectory
    ax3 = plt.subplot(2, 3, 3)
    zeta_abs = [abs(traj_data['zeta']) for traj_data in trajectories]
    zeta_phase = [np.angle(traj_data['zeta']) for traj_data in trajectories]
    colors_traj = [node_colors[traj_data['singular_point']['node']] for traj_data in trajectories]
    
    ax3.scatter(zeta_abs, zeta_phase, c=colors_traj, s=100, alpha=0.7, edgecolors='black')
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Prime threshold')
    ax3.set_xlabel('|ζ(s)|')
    ax3.set_ylabel('Phase (rad)')
    ax3.set_title('Zeta Function Values for Trajectories')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Trajectory curvature
    ax4 = plt.subplot(2, 3, 4)
    for traj_data in trajectories:
        sp = traj_data['singular_point']
        traj = traj_data['trajectory']
        color = node_colors[sp['node']]
        
        # Compute curvature along trajectory
        curvatures = []
        for i in range(1, len(traj) - 1):
            p1 = traj[i-1]['plucker'][:3]
            p2 = traj[i]['plucker'][:3]
            p3 = traj[i+1]['plucker'][:3]
            
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Curvature approximation
            cross = np.linalg.norm(np.cross(v1, v2))
            curv = cross / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            curvatures.append(curv)
        
        steps = range(len(curvatures))
        ax4.plot(steps, curvatures, color=color, linewidth=2, 
                label=f'Node {sp["node"]}, t={sp["time"]:.1f}s')
    
    ax4.set_xlabel('Step along trajectory')
    ax4.set_ylabel('Curvature')
    ax4.set_title('Trajectory Curvature (Max at Phase Transitions)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: 3D view of selected trajectories
    ax5 = fig.add_subplot(2, 3, 5, projection='3d')
    
    # Select prime paths (zeta < 0.3)
    prime_paths = [t for t in trajectories if abs(t['zeta']) < 0.3]
    
    for traj_data in prime_paths[:5]:
        sp = traj_data['singular_point']
        traj = traj_data['trajectory']
        color = node_colors[sp['node']]
        
        points = np.array([[pt['plucker'][0], pt['plucker'][1], pt['plucker'][2]] 
                          for pt in traj])
        
        ax5.plot(points[:, 0], points[:, 1], points[:, 2], 
                color=color, linewidth=2.5, label=f'Prime {sp["node"]}:{sp["time"]:.1f}s')
        ax5.scatter(points[0, 0], points[0, 1], points[0, 2], 
                   c='red', s=80, marker='o')
        ax5.scatter(points[-1, 0], points[-1, 1], points[-1, 2], 
                   c='green', s=80, marker='s')
    
    ax5.set_xlabel('p12')
    ax5.set_ylabel('p13')
    ax5.set_zlabel('p14')
    ax5.set_title('Prime Paths (|ζ| < 0.3)')
    ax5.legend(fontsize=7)
    
    # Plot 6: Plücker relation along trajectories
    ax6 = plt.subplot(2, 3, 6)
    for traj_data in trajectories:
        sp = traj_data['singular_point']
        traj = traj_data['trajectory']
        color = node_colors[sp['node']]
        
        plucker_rel = [pt['plucker'][0] * pt['plucker'][5] - 
                       pt['plucker'][1] * pt['plucker'][4] + 
                       pt['plucker'][2] * pt['plucker'][3] 
                       for pt in traj]
        
        steps = range(len(plucker_rel))
        ax6.plot(steps, plucker_rel, color=color, linewidth=1.5, alpha=0.7)
    
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_xlabel('Step along trajectory')
    ax6.set_ylabel('Plücker Relation (should be 0)')
    ax6.set_title('Klein Quadric Verification')
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Reverse Hironaka Trajectories: Singular → Smooth Flow', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_2_reverse_hironaka.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return trajectories


# ============================================================================
# CHART 3: PRIME PATH IDENTIFICATION
# ============================================================================

def chart_3_prime_path_identification(trajectories):
    """Create visualization of prime path identification via zeta functions"""
    fig = plt.figure(figsize=(16, 10))
    
    # Calculate zeta values for all trajectories
    zeta_values = []
    for traj_data in trajectories:
        zeta = traj_data['zeta']
        zeta_values.append({
            'node': traj_data['singular_point']['node'],
            'time': traj_data['singular_point']['time'],
            'zeta_abs': abs(zeta),
            'zeta_phase': np.angle(zeta),
            'is_prime': abs(zeta) < 0.3
        })
    
    # Sort by time
    zeta_values.sort(key=lambda x: x['time'])
    
    # Plot 1: Zeta magnitude over time
    ax1 = plt.subplot(2, 3, 1)
    times = [z['time'] for z in zeta_values]
    zeta_abs = [z['zeta_abs'] for z in zeta_values]
    colors = ['red' if z['is_prime'] else 'blue' for z in zeta_values]
    
    ax1.bar(times, zeta_abs, color=colors, alpha=0.7, width=0.15)
    ax1.axhline(y=0.3, color='red', linestyle='--', linewidth=2, label='Prime threshold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('|ζ(s)|')
    ax1.set_title('Zeta Function Magnitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Complex plane of zeta values
    ax2 = plt.subplot(2, 3, 2)
    for z in zeta_values:
        color = 'red' if z['is_prime'] else 'blue'
        ax2.scatter(z['zeta_abs'] * np.cos(z['zeta_phase']),
                   z['zeta_abs'] * np.sin(z['zeta_phase']),
                   c=color, s=80, alpha=0.7, edgecolors='black')
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    circle = plt.Circle((0, 0), 0.3, color='red', fill=False, linestyle='--', linewidth=2)
    ax2.add_patch(circle)
    ax2.set_xlabel('Re(ζ)')
    ax2.set_ylabel('Im(ζ)')
    ax2.set_title('Zeta Values in Complex Plane')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Plot 3: Prime path signature
    ax3 = plt.subplot(2, 3, 3)
    prime_paths = [z for z in zeta_values if z['is_prime']]
    
    if prime_paths:
        prime_indices = range(1, len(prime_paths) + 1)
        prime_times = [p['time'] for p in prime_paths]
        prime_nodes = [p['node'] for p in prime_paths]
        
        ax3.scatter(prime_indices, prime_times, s=100, c=prime_nodes, 
                   cmap='viridis', alpha=0.7, edgecolors='black')
        
        for i, (idx, p) in enumerate(zip(prime_indices, prime_paths)):
            ax3.annotate(f'N{p["node"]}', (idx, p['time']), 
                        textcoords="offset points", xytext=(0, 10), ha='center')
        
        ax3.set_xlabel('Prime Path Index')
        ax3.set_ylabel('Time (s)')
        ax3.set_title('Prime Path Timeline')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No prime paths detected', ha='center', va='center')
        ax3.set_title('Prime Path Detection')
    
    # Plot 4: Prime path curvature comparison
    ax4 = plt.subplot(2, 3, 4)
    prime_trajectories = [t for t in trajectories if abs(t['zeta']) < 0.3]
    
    for traj_data in prime_trajectories:
        sp = traj_data['singular_point']
        traj = traj_data['trajectory']
        
        # Compute curvature
        curvatures = []
        for i in range(1, len(traj) - 1):
            p1 = traj[i-1]['plucker'][:3]
            p2 = traj[i]['plucker'][:3]
            p3 = traj[i+1]['plucker'][:3]
            v1 = p2 - p1
            v2 = p3 - p2
            cross = np.linalg.norm(np.cross(v1, v2))
            curv = cross / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            curvatures.append(curv)
        
        steps = np.linspace(0, 1, len(curvatures))
        ax4.plot(steps, curvatures, linewidth=2, 
                label=f'Node {sp["node"]}, t={sp["time"]:.1f}s')
    
    ax4.set_xlabel('Normalized trajectory position')
    ax4.set_ylabel('Curvature')
    ax4.set_title('Prime Path Curvature Profiles')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Prime path signature (Gröbner basis approximation)
    ax5 = plt.subplot(2, 3, 5)
    prime_signatures = []
    
    for traj_data in prime_trajectories:
        sp = traj_data['singular_point']
        traj = traj_data['trajectory']
        
        # Compute signature from path (approximation of Gröbner basis)
        points = np.array([pt['plucker'] for pt in traj])
        cov = np.cov(points.T)
        eigvals = np.linalg.eigvalsh(cov)
        signature = eigvals[-3:]  # Top 3 eigenvalues as signature
        
        prime_signatures.append({
            'node': sp['node'],
            'time': sp['time'],
            'signature': signature
        })
    
    for ps in prime_signatures:
        ax5.plot(range(3), ps['signature'], 'o-', linewidth=2, markersize=8,
                label=f'Node {ps["node"]}, t={ps["time"]:.1f}s')
    
    ax5.set_xlabel('Signature component')
    ax5.set_ylabel('Eigenvalue')
    ax5.set_title('Prime Path Signatures (Gröbner Basis Approximation)')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Coherence verification
    ax6 = plt.subplot(2, 3, 6)
    
    # Compute coherence metric for each prime path
    for traj_data in prime_trajectories:
        sp = traj_data['singular_point']
        traj = traj_data['trajectory']
        
        # Check Plücker relation along path
        plucker_rel = [abs(pt['plucker'][0] * pt['plucker'][5] - 
                          pt['plucker'][1] * pt['plucker'][4] + 
                          pt['plucker'][2] * pt['plucker'][3]) 
                       for pt in traj]
        
        steps = np.linspace(0, 1, len(plucker_rel))
        ax6.plot(steps, plucker_rel, linewidth=2,
                label=f'Node {sp["node"]}, t={sp["time"]:.1f}s')
    
    ax6.axhline(y=1e-6, color='green', linestyle='--', linewidth=1, label='Coherence threshold')
    ax6.set_yscale('log')
    ax6.set_xlabel('Normalized trajectory position')
    ax6.set_ylabel('|Plücker relation|')
    ax6.set_title('Coherence Verification (should be < 1e-6)')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('Prime Path Identification via Zeta Function', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_3_prime_paths.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return zeta_values, prime_trajectories


# ============================================================================
# CHART 4: WAVELET-SCALED TRAJECTORIES
# ============================================================================

def chart_4_wavelet_scaled_trajectories(t, C, plucker):
    """Create wavelet-scaled trajectory visualizations"""
    fig = plt.figure(figsize=(18, 12))
    
    # Wavelet scales
    scales = np.arange(2, 64)
    
    # Plot 1: Wavelet scalogram for each node
    for node_idx, node in enumerate(range(3)):
        ax = plt.subplot(3, 4, node_idx * 4 + 1)
        
        signal = C[node, :]
        dt = t[1] - t[0]
        coeffs = cwt(signal, morlet, scales, dt=dt)
        
        im = ax.imshow(np.abs(coeffs), aspect='auto', cmap='hot', 
                       extent=[t[0], t[-1], scales[0], scales[-1]])
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Scale')
        ax.set_title(f'Node {node} - Wavelet Scalogram')
        plt.colorbar(im, ax=ax, label='|Coefficient|')
    
    # Plot 2: Wavelet energy vs Plücker coordinates
    for node_idx, node in enumerate(range(3)):
        ax = plt.subplot(3, 4, node_idx * 4 + 2)
        
        signal = C[node, :]
        dt = t[1] - t[0]
        coeffs = cwt(signal, morlet, scales, dt=dt)
        energy = np.sum(np.abs(coeffs)**2, axis=1)
        
        # Normalize energy
        energy = energy / np.max(energy)
        
        # Plot energy as function of scale
        ax.plot(scales, energy, 'b-', linewidth=2)
        ax.fill_between(scales, energy, alpha=0.3)
        ax.set_xlabel('Scale')
        ax.set_ylabel('Normalized Energy')
        ax.set_title(f'Node {node} - Wavelet Energy Spectrum')
        ax.grid(True, alpha=0.3)
    
    # Plot 3: Wavelet-scaled Plücker trajectories
    for node_idx, node in enumerate(range(3)):
        ax = fig.add_subplot(3, 4, node_idx * 4 + 3, projection='3d')
        
        # Color by wavelet energy
        signal = C[node, :]
        dt = t[1] - t[0]
        coeffs = cwt(signal, morlet, scales[:10], dt=dt)
        energy = np.sum(np.abs(coeffs)**2, axis=0)
        energy = energy / np.max(energy)
        
        # Plücker trajectory
        x = plucker[node, :, 0]
        y = plucker[node, :, 1]
        z = plucker[node, :, 2]
        
        # Create line segments with color based on wavelet energy
        points = np.array([x, y, z]).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        
        norm = Normalize(vmin=0, vmax=1)
        lc = Line3DCollection(segments, cmap='plasma', norm=norm)
        lc.set_array(energy[:-1])
        lc.set_linewidth(2)
        ax.add_collection(lc)
        
        ax.scatter(x[0], y[0], z[0], c='green', s=50, marker='o')
        ax.scatter(x[-1], y[-1], z[-1], c='red', s=50, marker='s')
        
        ax.set_xlabel('p12')
        ax.set_ylabel('p13')
        ax.set_zlabel('p14')
        ax.set_title(f'Node {node} - Wavelet-Scaled Trajectory')
        
        # Add colorbar
        sm = ScalarMappable(norm=norm, cmap='plasma')
        sm.set_array(energy)
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.1)
        cbar.set_label('Wavelet Energy')
    
    # Plot 4: Multi-scale trajectory decomposition
    ax_last = plt.subplot(3, 4, 12)
    
    # Show multi-scale decomposition for Node 0
    node = 0
    signal = C[node, :]
    dt = t[1] - t[0]
    coeffs = cwt(signal, morlet, scales[:20], dt=dt)
    
    # Reconstruct at different scales
    scale_indices = [0, 5, 10, 15]
    colors_scale = ['red', 'orange', 'green', 'blue']
    
    for scale_idx, color in zip(scale_indices, colors_scale):
        # Scale-specific contribution
        scale_coeff = np.zeros_like(coeffs)
        scale_coeff[scale_idx, :] = coeffs[scale_idx, :]
        # Simple reconstruction (inverse wavelet approximation)
        reconstructed = np.sum(scale_coeff, axis=0)
        reconstructed = reconstructed / np.max(np.abs(reconstructed))
        
        ax_last.plot(t, reconstructed, color=color, linewidth=1.5, alpha=0.7,
                    label=f'Scale {scales[scale_idx]:.0f}')
    
    ax_last.plot(t, signal, 'k-', linewidth=2, label='Original')
    ax_last.set_xlabel('Time (s)')
    ax_last.set_ylabel('Consciousness')
    ax_last.set_title('Multi-Scale Trajectory Decomposition (Node 0)')
    ax_last.legend(fontsize=8)
    ax_last.grid(True, alpha=0.3)
    
    plt.suptitle('Wavelet-Scaled Trajectories on the Klein Quadric', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('chart_4_wavelet_scaled.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Generate all trajectory charts"""
    
    print("="*80)
    print("GENERATING TRAJECTORY CHARTS FOR 3-NODE, 7-EDGE GRAPH")
    print("="*80)
    
    # Generate data
    print("\n[1] Generating trajectory data...")
    t, C, plucker = generate_trajectory_data()
    print(f"    Time points: {len(t)}")
    print(f"    Consciousness range: [{C.min():.3f}, {C.max():.3f}]")
    
    # Identify singular points
    print("\n[2] Identifying singular points (phase transitions)...")
    singular_points = identify_singular_points(C, t)
    print(f"    Found {len(singular_points)} singular points:")
    for sp in singular_points:
        print(f"    Node {sp['node']}: t={sp['time']:.2f}s, C={sp['consciousness']:.3f}")
    
    # Chart 1: 3D Plücker trajectories
    print("\n[3] Generating Chart 1: 3D Plücker Trajectories...")
    chart_1_plucker_3d_trajectories(plucker, t, C)
    
    # Chart 2: Reverse Hironaka trajectories
    print("\n[4] Generating Chart 2: Reverse Hironaka Trajectories...")
    trajectories = chart_2_reverse_hironaka_trajectories(plucker, C, t, singular_points)
    
    # Chart 3: Prime path identification
    print("\n[5] Generating Chart 3: Prime Path Identification...")
    zeta_values, prime_trajectories = chart_3_prime_path_identification(trajectories)
    
    # Chart 4: Wavelet-scaled trajectories
    print("\n[6] Generating Chart 4: Wavelet-Scaled Trajectories...")
    chart_4_wavelet_scaled_trajectories(t, C, plucker)
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    print(f"\nTotal trajectories tracked: {len(trajectories)}")
    print(f"Prime paths identified: {len(prime_trajectories)}")
    
    if prime_trajectories:
        print("\nPrime Path Details:")
        for pt in prime_trajectories:
            sp = pt['singular_point']
            zeta_abs = abs(pt['zeta'])
            print(f"  Prime Path: Node {sp['node']}, Time={sp['time']:.2f}s, |ζ|={zeta_abs:.4f}")
    
    print("\nCharts saved:")
    print("  - chart_1_plucker_3d.png")
    print("  - chart_2_reverse_hironaka.png")
    print("  - chart_3_prime_paths.png")
    print("  - chart_4_wavelet_scaled.png")
    
    return {
        't': t, 'C': C, 'plucker': plucker,
        'singular_points': singular_points,
        'trajectories': trajectories,
        'prime_trajectories': prime_trajectories,
        'zeta_values': zeta_values
    }


if __name__ == "__main__":
    results = main()