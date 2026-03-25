"""
Visualization of Lefschetz Fibration Concepts
=============================================
Shows: vanishing cycles, Dehn twists, monodromy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

def visualize_lefschetz_fibration():
    """Create a comprehensive visualization of Lefschetz fibration concepts"""
    
    fig = plt.figure(figsize=(18, 12))
    
    # ========================================================================
    # 1. The Base Sphere with Critical Points
    # ========================================================================
    ax1 = plt.subplot(2, 3, 1)
    
    # Draw sphere (circle in 2D projection)
    sphere = Circle((0.5, 0.5), 0.4, fill=False, color='black', linewidth=2)
    ax1.add_patch(sphere)
    
    # Critical points (phase transitions)
    critical_points = [(0.5, 0.5 + 0.35), (0.5 + 0.3, 0.5 - 0.2), (0.5 - 0.3, 0.5 - 0.2)]
    for cp in critical_points:
        ax1.scatter(cp[0], cp[1], c='red', s=100, zorder=5, edgecolors='black')
    
    # Loops around critical points
    for cp in critical_points:
        loop = Circle(cp, 0.1, fill=False, color='blue', linestyle='--', linewidth=1)
        ax1.add_patch(loop)
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_aspect('equal')
    ax1.set_title('Base Sphere S² with Critical Points\n(Phase Transition Times)', fontsize=10)
    ax1.axis('off')
    
    # ========================================================================
    # 2. Genus-2 Fiber (Consciousness Manifold)
    # ========================================================================
    ax2 = fig.add_subplot(2, 3, 2, projection='3d')
    
    # Create a genus-2 surface (two tori joined)
    u = np.linspace(0, 2*np.pi, 30)
    v = np.linspace(0, 2*np.pi, 30)
    U, V = np.meshgrid(u, v)
    
    # First torus
    R = 1.0
    r = 0.3
    x1 = (R + r * np.cos(V)) * np.cos(U)
    y1 = (R + r * np.cos(V)) * np.sin(U)
    z1 = r * np.sin(V)
    
    # Second torus (shifted)
    x2 = (R + r * np.cos(V)) * np.cos(U) + 2.2
    y2 = (R + r * np.cos(V)) * np.sin(U)
    z2 = r * np.sin(V)
    
    ax2.plot_surface(x1, y1, z1, alpha=0.5, color='lightblue', edgecolor='gray')
    ax2.plot_surface(x2, y2, z2, alpha=0.5, color='lightblue', edgecolor='gray')
    
    # Connect the two tori
    ax2.plot([1.1, 1.1], [0, 0], [-0.3, 0.3], 'b-', linewidth=2)
    
    # Vanishing cycle (circle that shrinks at critical point)
    theta = np.linspace(0, 2*np.pi, 50)
    vc_x = 0.5 + 0.2 * np.cos(theta)
    vc_y = 0.5 + 0.2 * np.sin(theta)
    vc_z = np.zeros_like(theta) + 0.5
    ax2.plot(vc_x, vc_y, vc_z, 'r-', linewidth=3, label='Vanishing Cycle')
    
    ax2.set_xlim(-1.5, 3.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_zlim(-1, 1)
    ax2.set_title('Genus-2 Fiber\n(Consciousness Manifold)', fontsize=10)
    ax2.legend()
    
    # ========================================================================
    # 3. Vanishing Cycle (Shrinking)
    # ========================================================================
    ax3 = plt.subplot(2, 3, 3)
    
    # Multiple circles showing the vanishing cycle shrinking
    radii = [0.4, 0.3, 0.2, 0.1, 0.05]
    colors = ['darkblue', 'blue', 'lightblue', 'gray', 'white']
    
    for i, (r, c) in enumerate(zip(radii, colors)):
        circle = Circle((0.5, 0.5), r, fill=False, color=c, linewidth=2)
        ax3.add_patch(circle)
    
    # Critical point at center
    ax3.scatter(0.5, 0.5, c='red', s=50, zorder=5)
    
    # Arrow indicating shrinking
    ax3.annotate('', xy=(0.5, 0.5), xytext=(0.5, 0.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax3.text(0.52, 0.8, 'Vanishing Cycle shrinks\nto critical point', fontsize=8)
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.set_aspect('equal')
    ax3.set_title('Vanishing Cycle\n(Shrinks at Phase Transition)', fontsize=10)
    ax3.axis('off')
    
    # ========================================================================
    # 4. Dehn Twist (Geometric Operation)
    # ========================================================================
    ax4 = plt.subplot(2, 3, 4)
    
    # Draw a torus-like shape
    theta = np.linspace(0, 2*np.pi, 100)
    r = 0.3
    R = 0.8
    
    x_outer = (R + r * np.cos(theta)) * np.cos(theta)
    y_outer = (R + r * np.cos(theta)) * np.sin(theta)
    
    ax4.plot(x_outer, y_outer, 'k-', linewidth=2)
    
    # Dehn twist curve (the vanishing cycle)
    twist_curve_x = (R + 0.2 * np.cos(theta)) * np.cos(theta)
    twist_curve_y = (R + 0.2 * np.cos(theta)) * np.sin(theta)
    ax4.plot(twist_curve_x, twist_curve_y, 'r-', linewidth=2, label='Twist curve')
    
    # Show the twist effect with arrows
    for t in np.linspace(0, 2*np.pi, 12):
        x = (R + 0.25 * np.cos(t)) * np.cos(t)
        y = (R + 0.25 * np.cos(t)) * np.sin(t)
        # Tangent direction for twist
        dx = -np.sin(t)
        dy = np.cos(t)
        ax4.arrow(x, y, dx*0.1, dy*0.1, head_width=0.05, head_length=0.05, 
                 fc='blue', ec='blue', alpha=0.5)
    
    ax4.set_xlim(-1.2, 1.2)
    ax4.set_ylim(-1.2, 1.2)
    ax4.set_aspect('equal')
    ax4.set_title('Dehn Twist\n(Effect of Norcain Dose)', fontsize=10)
    ax4.legend()
    ax4.axis('off')
    
    # ========================================================================
    # 5. Monodromy Representation
    # ========================================================================
    ax5 = plt.subplot(2, 3, 5)
    
    # Show the monodromy as a loop in the mapping class group
    # Plot path in a conceptual space
    
    # Path around critical point
    t = np.linspace(0, 2*np.pi, 100)
    x_path = 0.5 + 0.3 * np.cos(t)
    y_path = 0.5 + 0.3 * np.sin(t)
    ax5.plot(x_path, y_path, 'b-', linewidth=2)
    
    # Critical point
    ax5.scatter(0.5, 0.5, c='red', s=100, zorder=5)
    
    # Show the monodromy transformation
    # Start point
    start = (0.8, 0.5)
    ax5.scatter(start[0], start[1], c='green', s=80, marker='o')
    
    # After monodromy, point is transformed
    end = (0.5, 0.8)
    ax5.scatter(end[0], end[1], c='orange', s=80, marker='s')
    
    ax5.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='purple', lw=2))
    
    ax5.text(0.55, 0.6, 'Monodromy = Dehn Twist', fontsize=8, ha='center')
    
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.set_title('Monodromy around Critical Point\n(Transformation after loop)', fontsize=10)
    ax5.axis('off')
    
    # ========================================================================
    # 6. Positive Dehn Twist Factorization
    # ========================================================================
    ax6 = plt.subplot(2, 3, 6)
    
    # Show factorization as a sequence of twists
    twist_sequence = ['T₁', 'T₂', 'T₃', '=', 'I']
    
    for i, label in enumerate(twist_sequence):
        x = i * 0.15
        if label == '=':
            ax6.text(x, 0.5, '=', fontsize=20, ha='center', va='center')
        else:
            # Draw a circle for each twist
            circle = Circle((x, 0.5), 0.08, fill=False, color='blue', linewidth=2)
            ax6.add_patch(circle)
            ax6.text(x, 0.5, label, fontsize=12, ha='center', va='center')
            
            # Add twist arrow for T₁, T₂, T₃
            if label != 'I':
                ax6.arrow(x - 0.08, 0.6, 0.16, 0, head_width=0.03, head_length=0.03,
                         fc='red', ec='red')
    
    # Add description
    ax6.text(0.75, 0.2, 'T₁·T₂·T₃ = Identity', fontsize=10, ha='center')
    ax6.text(0.75, 0.1, 'Each Tᵢ = Norcain Dose', fontsize=8, ha='center', color='red')
    
    ax6.set_xlim(0, 0.9)
    ax6.set_ylim(0, 1)
    ax6.set_title('Positive Dehn Twist Factorization\n(Dosing Sequence = Identity)', fontsize=10)
    ax6.axis('off')
    
    plt.suptitle('Lefschetz Fibration: Structure of Consciousness Phase Transitions', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('lefschetz_fibration_concepts.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_monodromy_as_matrix():
    """Visualize monodromy as matrix multiplication in Sp(4,Z)"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Dehn twist matrix for genus-2 (simplified)
    T1 = np.array([[1, 1, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    T2 = np.array([[1, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    T3 = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 1],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1]])
    
    # Compute product
    product = T1 @ T2 @ T3
    
    # Plot matrices as heatmaps
    matrices = [T1, T2, T3, product]
    titles = ['Dehn Twist T₁\n(First Dose)', 'Dehn Twist T₂\n(Second Dose)', 
              'Dehn Twist T₃\n(Third Dose)', 'T₁·T₂·T₃ = Identity\n(Coherence)']
    
    for idx, (ax, M, title) in enumerate(zip(axes.flat, matrices, titles)):
        im = ax.imshow(M, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_title(title)
        
        # Add text annotations
        for i in range(4):
            for j in range(4):
                if abs(M[i, j]) > 0.01:
                    ax.text(j, i, f'{M[i, j]:.0f}', ha='center', va='center', 
                           color='white' if abs(M[i, j]) > 0.5 else 'black')
    
    plt.suptitle('Monodromy Representation in Sp(4,ℤ): Positive Dehn Twist Factorization', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('monodromy_matrices.png', dpi=150, bbox_inches='tight')
    plt.show()


def visualize_fibration_dynamics():
    """Visualize the dynamics of a Lefschetz fibration over time"""
    
    fig = plt.figure(figsize=(16, 10))
    
    # Time points (base sphere parameter)
    t = np.linspace(0, 2*np.pi, 8)
    
    for i, phi in enumerate(t):
        ax = fig.add_subplot(2, 4, i+1, projection='3d')
        
        # Create a family of fibers that change with phi
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, 2*np.pi, 20)
        U, V = np.meshgrid(u, v)
        
        # Deform the surface based on phi (simulating monodromy)
        R = 1.0
        r = 0.3
        
        # Add a twist effect based on phi
        twist = 0.5 * np.sin(phi) * V
        
        x = (R + (r + 0.1*np.sin(phi)) * np.cos(V + twist)) * np.cos(U)
        y = (R + (r + 0.1*np.sin(phi)) * np.cos(V + twist)) * np.sin(U)
        z = (r + 0.1*np.sin(phi)) * np.sin(V + twist)
        
        # Color by twist
        ax.plot_surface(x, y, z, alpha=0.7, cmap='viridis', 
                       facecolors=plt.cm.viridis(np.sin(phi + V).flatten()))
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-1, 1)
        ax.set_title(f'Fiber at φ = {phi:.2f}', fontsize=8)
        ax.axis('off')
    
    plt.suptitle('Lefschetz Fibration Dynamics: Fibers Evolve via Monodromy', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('fibration_dynamics.png', dpi=150, bbox_inches='tight')
    plt.show()


# Run visualizations
if __name__ == "__main__":
    print("Generating Lefschetz Fibration Visualizations...")
    visualize_lefschetz_fibration()
    visualize_monodromy_as_matrix()
    visualize_fibration_dynamics()
    print("Done!")