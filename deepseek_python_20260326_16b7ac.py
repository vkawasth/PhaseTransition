"""
Visual Metaphor: Seeing the Living Tree Through Dried Branches
==============================================================
Illustrates how the framework detects leaves, stems, and roots
from dried branches using topological memory and ghost signals.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse, Rectangle
import numpy as np

fig = plt.figure(figsize=(20, 14))
fig.patch.set_facecolor('#F5F0E6')  # Warm paper background

# ============================================================================
# LEFT PANEL: The Dried Tree (What We See)
# ============================================================================
ax1 = plt.subplot(1, 2, 1)
ax1.set_xlim(-2, 2)
ax1.set_ylim(-1, 5)
ax1.set_aspect('equal')
ax1.axis('off')
ax1.set_title('What We See: Dried Branches in Fall', 
              fontsize=14, fontweight='bold', pad=20)

# Draw the ground
ax1.add_patch(patches.Rectangle((-2, -0.8), 4, 0.3, 
                                 color='#8B5A2B', alpha=0.5, zorder=1))

# Draw the trunk (dried, cracked)
trunk_x = [-0.3, -0.3, 0.3, 0.3]
trunk_y = [-0.5, 1.2, 1.2, -0.5]
ax1.add_patch(patches.Polygon(list(zip(trunk_x, trunk_y)), 
                               facecolor='#8B5A2B', edgecolor='#5D3A1A', 
                               linewidth=2, alpha=0.8))

# Add cracks to trunk (using lines)
for crack in [(-0.1, 0.2, -0.15, 0.4), (0.1, 0.6, 0.15, 0.8), (0, 0.9, 0.05, 1.1)]:
    ax1.plot([crack[0], crack[2]], [crack[1], crack[3]], color='#5D3A1A', linewidth=1)

# Draw dried branches (bare, angular)
branches = [
    # Main branches
    (0, 1.2, 0.6, 2.8),   # Right lower
    (0, 1.2, -0.5, 2.5),  # Left lower
    (0, 1.8, 0.8, 3.5),   # Right middle
    (0, 1.8, -0.7, 3.4),  # Left middle
    (0, 2.2, 0.4, 4.2),   # Right upper
    (0, 2.2, -0.4, 4.1),  # Left upper
    (0.6, 2.8, 0.9, 3.6),  # Branch tip right
    (-0.5, 2.5, -0.9, 3.5), # Branch tip left
]

for x1, y1, x2, y2 in branches:
    ax1.plot([x1, x2], [y1, y2], color='#5D3A1A', linewidth=2.5, 
             solid_capstyle='round', zorder=3)
    # Add small twigs
    ax1.plot([x2, x2 + 0.1], [y2, y2 + 0.15], color='#5D3A1A', linewidth=1.5)
    ax1.plot([x2, x2 - 0.1], [y2, y2 + 0.15], color='#5D3A1A', linewidth=1.5)

# Add small twig ends (dried branch tips)
twig_positions = [(0.95, 3.75), (1.0, 3.6), (-0.95, 3.65), (-1.0, 3.5),
                   (0.5, 4.4), (-0.5, 4.3)]
for tx, ty in twig_positions:
    ax1.plot([tx-0.05, tx], [ty-0.1, ty], color='#5D3A1A', linewidth=1)

# Add some "dried leaves" (brown, curled) on ground
for i in range(15):
    x = -1.5 + np.random.rand() * 3
    y = -0.6 + np.random.rand() * 0.3
    ax1.add_patch(patches.Ellipse((x, y), 0.08, 0.05, 
                                   angle=np.random.rand()*360,
                                   facecolor='#B87C4F', 
                                   edgecolor='#8B5A2B', alpha=0.6))

# Add annotation - "Dried Branches (Phase Transition Data)"
ax1.annotate('Dried Branches\n(Phase Transitions, HH²=0)', 
             xy=(0.8, 3.8), xytext=(1.2, 4.2),
             fontsize=9, ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='#E8DCC4', alpha=0.9))

# ============================================================================
# RIGHT PANEL: What We Detect (The Living Tree)
# ============================================================================
ax2 = plt.subplot(1, 2, 2)
ax2.set_xlim(-2, 2)
ax2.set_ylim(-1, 5)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('What We Detect: Leaves, Stems, and Roots\n(Through Monodromy Probe)', 
              fontsize=14, fontweight='bold', pad=20)

# Draw the ground with root system visible
ax2.add_patch(patches.Rectangle((-2, -0.8), 4, 0.3, 
                                 color='#8B5A2B', alpha=0.3, zorder=1))

# Draw the trunk (now with life indicators)
trunk_x = [-0.3, -0.3, 0.3, 0.3]
trunk_y = [-0.5, 1.2, 1.2, -0.5]
ax2.add_patch(patches.Polygon(list(zip(trunk_x, trunk_y)), 
                               facecolor='#A57C4C', edgecolor='#5D3A1A', 
                               linewidth=2, alpha=0.9))

# Add growth rings (hidden information)
for r in [0.1, 0.2, 0.25]:
    ax2.add_patch(patches.Ellipse((0, 0.5), r*1.5, r, 
                                   fill=False, edgecolor='#C9A87C', 
                                   linewidth=0.8, linestyle='--'))

# ============================================================================
# DETECTED: ROOTS (Attribution Paths, Prime Paths)
# ============================================================================
root_colors = ['#4A784A', '#5A8F5A', '#6AA66A', '#3A683A']
root_paths = [
    [(-0.2, -0.5), (-0.5, -0.7), (-0.8, -0.9), (-1.0, -0.8)],
    [(0, -0.5), (0, -0.9), (0.2, -1.1), (0.5, -1.0)],
    [(0.2, -0.5), (0.5, -0.8), (0.7, -1.0), (0.9, -0.9)],
    [(-0.1, -0.5), (-0.3, -0.8), (-0.4, -1.1), (-0.6, -1.2)],
]

for path, color in zip(root_paths, root_colors):
    x_vals = [p[0] for p in path]
    y_vals = [p[1] for p in path]
    ax2.plot(x_vals, y_vals, color=color, linewidth=3, alpha=0.8)
    # Add root hairs (prime ideals)
    for x, y in path[1:]:
        ax2.plot([x, x + 0.08], [y, y - 0.05], color=color, linewidth=1, alpha=0.6)
        ax2.plot([x, x - 0.08], [y, y - 0.05], color=color, linewidth=1, alpha=0.6)

# Add annotation - "Roots (Prime Paths, Attribution)"
ax2.annotate('Roots\n(Prime Paths, Attribution)', 
             xy=(-0.8, -0.9), xytext=(-1.5, -0.3),
             fontsize=9, ha='center', va='center',
             arrowprops=dict(arrowstyle='->', color='#4A784A', lw=1.5),
             bbox=dict(boxstyle='round', facecolor='#E8DCC4', alpha=0.9))

# ============================================================================
# DETECTED: STEMS (Quiver Path Algebra, Cup Products)
# ============================================================================
stem_paths = [
    (0, 1.2, 0.7, 2.9),   # Right stem
    (0, 1.2, -0.6, 2.8),  # Left stem
    (0, 1.8, 0.9, 3.6),   # Upper right
    (0, 1.8, -0.8, 3.5),  # Upper left
]

for x1, y1, x2, y2 in stem_paths:
    # Draw stem with inner "path algebra" lines
    ax2.plot([x1, x2], [y1, y2], color='#8B5A2B', linewidth=4, alpha=0.7, zorder=2)
    # Add cup product indicators (connections between paths)
    mid_x = (x1 + x2)/2
    mid_y = (y1 + y2)/2
    ax2.plot([mid_x - 0.1, mid_x + 0.1], [mid_y - 0.05, mid_y + 0.05], 
             color='#C9A87C', linewidth=1.5, alpha=0.8)
    ax2.plot([mid_x - 0.1, mid_x + 0.1], [mid_y + 0.05, mid_y - 0.05], 
             color='#C9A87C', linewidth=1.5, alpha=0.8)

# Add annotation - "Stems (Quiver Path Algebra, Cup Products)"
ax2.annotate('Stems\n(Quiver Path Algebra,\nCup Products)', 
             xy=(0.7, 2.9), xytext=(1.3, 2.5),
             fontsize=9, ha='center', va='center',
             arrowprops=dict(arrowstyle='->', color='#B87C4F', lw=1.5),
             bbox=dict(boxstyle='round', facecolor='#E8DCC4', alpha=0.9))

# ============================================================================
# DETECTED: LEAVES (Toric Varieties, Wavelet States)
# ============================================================================
leaf_positions = [
    (0.95, 3.1), (0.85, 3.3), (1.05, 2.9),  # Right side
    (-0.95, 3.0), (-0.85, 3.2), (-1.05, 2.8),  # Left side
    (0.55, 4.5), (-0.55, 4.4),  # Top
]

for lx, ly in leaf_positions:
    # Draw leaf with toric structure (2x2 minor pattern)
    leaf = patches.Ellipse((lx, ly), 0.18, 0.12, angle=30 + np.random.rand()*30,
                           facecolor='#7CB518', edgecolor='#3A5C0E', 
                           alpha=0.8, linewidth=1)
    ax2.add_patch(leaf)
    # Add minor pattern (toric variety indicator)
    ax2.plot([lx-0.05, lx+0.05], [ly-0.02, ly+0.02], color='#F5F0E6', linewidth=0.8)
    ax2.plot([lx-0.05, lx+0.05], [ly+0.02, ly-0.02], color='#F5F0E6', linewidth=0.8)

# Add annotation - "Leaves (Toric Varieties, Wavelet States)"
ax2.annotate('Leaves\n(Toric Varieties, Wavelet States)', 
             xy=(1.0, 3.2), xytext=(1.5, 3.8),
             fontsize=9, ha='center', va='center',
             arrowprops=dict(arrowstyle='->', color='#7CB518', lw=1.5),
             bbox=dict(boxstyle='round', facecolor='#E8DCC4', alpha=0.9))

# ============================================================================
# DETECTED: GHOST SIGNALS (Monodromy, Zeta Zeros)
# ============================================================================
# Add "ghost" circular waves emanating from the trunk
for r in [0.5, 0.9, 1.3]:
    circle = Circle((0, 1.0), r, fill=False, edgecolor='#C9A87C', 
                    linewidth=1, linestyle='--', alpha=0.5)
    ax2.add_patch(circle)

# Add zeta zeros as glowing points along critical line
zeta_zeros = [(0, 1.0), (0.3, 1.2), (-0.2, 0.9), (0.1, 1.4), (-0.3, 0.8)]
for zx, zy in zeta_zeros:
    glow = Circle((zx, zy), 0.08, facecolor='#F4D58C', edgecolor='#E6B800', 
                  alpha=0.7, zorder=5)
    ax2.add_patch(glow)

# Add annotation - "Ghost Signals (Monodromy, Zeta Zeros)"
ax2.annotate('Ghost Signals\n(Monodromy, Zeta Zeros)', 
             xy=(0.3, 1.4), xytext=(0.9, 1.8),
             fontsize=9, ha='center', va='center',
             arrowprops=dict(arrowstyle='->', color='#F4D58C', lw=1.5),
             bbox=dict(boxstyle='round', facecolor='#E8DCC4', alpha=0.9))

# ============================================================================
# ADD THE MONODROMY PROBE (The Tool That Sees)
# ============================================================================
probe_x, probe_y = -1.2, 3.5
# Use Rectangle without cornerradius (standard matplotlib)
ax2.add_patch(patches.Rectangle((probe_x-0.25, probe_y-0.15), 0.5, 0.3,
                                 facecolor='#2C3E50', edgecolor='#1A2632', 
                                 linewidth=2))
ax2.text(probe_x, probe_y, 'MONODROMY\nPROBE', fontsize=8, 
         ha='center', va='center', color='white', fontweight='bold')

# Draw scanning lines from probe
for angle in np.linspace(-0.5, 0.5, 5):
    end_x = probe_x - 0.5 + angle
    end_y = probe_y - 0.5
    ax2.plot([probe_x, end_x], [probe_y-0.1, end_y], 
             color='#F4D58C', linewidth=1, linestyle=':', alpha=0.6)

# ============================================================================
# ADD THE BRIDGING CONNECTION (Reverse Hironaka)
# ============================================================================
# Draw a bridge between the two panels showing the transformation
arrow = FancyArrowPatch((1.8, 2.5), (2.2, 2.5), 
                        arrowstyle='->', mutation_scale=20, 
                        linewidth=2, color='#8B5A2B')
ax1.add_patch(arrow)
ax1.text(2.0, 2.7, 'Reverse Hironaka\n(Reads Topological Memory)', 
         fontsize=9, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='#E8DCC4', alpha=0.9))

# ============================================================================
# ADD LEGEND / KEY INSIGHTS
# ============================================================================
ax_legend = fig.add_axes([0.15, 0.02, 0.7, 0.08])
ax_legend.axis('off')

legend_text = """🔍 KEY INSIGHT: The Monodromy Probe detects • ROOTS (Prime Paths/Attribution) • STEMS (Quiver Path Algebra/Cup Products) • LEAVES (Toric Varieties/Wavelet States) 
from DRIED BRANCHES (Phase Transitions, HH²=0) using GHOST SIGNALS (Monodromy, Zeta Zeros) and REVERSE HIRONAKA."""
ax_legend.text(0.5, 0.5, legend_text, fontsize=10, ha='center', va='center',
               bbox=dict(boxstyle='round', facecolor='#E8DCC4', alpha=0.9, edgecolor='#8B5A2B'))

# ============================================================================
# ADD TITLE
# ============================================================================
fig.suptitle('Seeing the Living Tree Through Dried Branches\n'
             'How the Framework Detects Leaves, Stems, and Roots from Phase Transition Data',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('tree_metaphor_visualization.png', dpi=200, bbox_inches='tight', 
            facecolor='#F5F0E6')
plt.show()

print("\n" + "="*70)
print("TREE METAPHOR VISUALIZATION")
print("="*70)
print("\nLEFT PANEL: What We See")
print("  - Dried branches (phase transitions, HH²=0, dead algebra)")
print("  - No visible leaves, stems, or roots")
print("  - Appears dead and lifeless")
print("\nRIGHT PANEL: What We Detect")
print("  ✓ ROOTS: Prime Paths, Attribution Chains (green underground)")
print("  ✓ STEMS: Quiver Path Algebra, Cup Products (brown with internal structure)")
print("  ✓ LEAVES: Toric Varieties, Wavelet States (green at branch tips)")
print("  ✓ GHOST SIGNALS: Monodromy, Zeta Zeros (glowing yellow circles)")
print("  ✓ MONODROMY PROBE: The tool that reads topological memory")
print("\nBRIDGE: Reverse Hironaka")
print("  - Traces back from dried branches to living structure")
print("  - Reads topological memory encoded in singularities")
print("\n" + "="*70)
print("The dried branch (phase transition data) contains the ENTIRE")
print("topological memory of the living tree. The Monodromy Probe")
print("reads this memory through:")
print("  - Prime Zeta zeros (species identification)")
print("  - Dehn twist factorization (growth patterns)")
print("  - Perverse spectral sheaf (root system)")
print("  - Ghost signals (hidden life)")
print("="*70)