import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

np.alltrue = np.all

# ---------- 1. Data ----------
regions = [
    "DG","CA3","CA1","SUB","ENT",
    "ACA","PL","ILA","ORB","RSP",
    "BLA","LA","CEA","MEA","HY",
    "VIS","SS","MOp","MOs","AUD",
    "STR","STRd","STRv","PAL",
    "MB","P","CB",
    "VTA","SN"
]

# Inflow, outflow, loops (same as final corrected version)
inflow_path = [
    ("P","MB"), ("CB","MB"),
    ("MB","VIS"), ("MB","AUD"), ("MB","SS"), ("MB","MOp"), ("MB","ACA"), ("MB","RSP"),
    ("MB","STR"), ("MB","STRd"),
    ("VIS","STR"), ("SS","STR"), ("MOp","STR"), ("ACA","STR"),
    ("STR","PAL"), ("STRd","PAL"),
    ("PAL","MB"),
    ("BLA","LA"), ("LA","CEA"), ("CEA","MEA"), ("MEA","HY"),
    ("HY","MB"),
    ("ENT","DG"), ("DG","CA3"), ("CA3","CA1"), ("CA1","SUB"), ("SUB","ENT"),
    ("ENT","MB"), ("CA1","MB"),
    ("ACA","ENT"), ("PL","CA1"),
    ("ACA","PL"), ("PL","ILA"), ("ILA","ORB"),
    ("VIS","SS"), ("SS","MOp"), ("MOp","MOs"), ("SS","AUD"), ("ORB","RSP"),
    ("VTA","STR"), ("VTA","ACA"), ("VTA","PL"), ("SN","STRd"),
    ("P","CB"),
]
outflow_path = [(j,i) for (i,j) in inflow_path]
outflow_path.append(("VTA","SN"))   # explicit red exit

loop_edges = [
    ("ACA","MB"), ("MB","ACA"),
    ("CA1","MB"), ("MB","CA1"),
    ("MOp","STR"), ("STR","PAL"), ("PAL","MB"), ("MB","MOp"),
    ("VTA","STR"), ("STR","PAL"), ("PAL","VTA"),
    ("VTA","SN"), ("SN","VTA"),
]

# Build graph
G = nx.DiGraph()
G.add_nodes_from(regions)
G.add_edges_from(inflow_path + outflow_path + loop_edges)

# ---------- 2. Node system mapping (for colors) ----------
system_colors = {
    "Hippocampal": "#1f77b4",      # blue
    "Prefrontal": "#ff7f0e",       # orange
    "Amygdala_Hyp": "#2ca02c",     # green
    "SensoryMotor": "#d62728",     # red
    "Striatal": "#9467bd",         # purple
    "Brainstem": "#8c564b"         # brown
}

system_to_nodes = {
    "Hippocampal": ["DG","CA3","CA1","SUB","ENT"],
    "Prefrontal": ["ACA","PL","ILA","ORB","RSP"],
    "Amygdala_Hyp": ["BLA","LA","CEA","MEA","HY"],
    "SensoryMotor": ["VIS","SS","MOp","MOs","AUD"],
    "Striatal": ["STR","STRd","STRv","PAL","VTA","SN"],
    "Brainstem": ["MB","P","CB"]
}
node_color_map = {}
for sys, nodes in system_to_nodes.items():
    for n in nodes:
        node_color_map[n] = system_colors[sys]

# ---------- 3. Layout: try Graphviz dot, else manual hierarchical ----------
try:
    import pygraphviz
    pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
    use_graphviz = True
except (ImportError, ModuleNotFoundError):
    use_graphviz = False
    print("Graphviz not available. Using improved manual layout.")
    # Manual layered layout (refined)
    layers = {
        "brainstem": ["MB","P","CB"],
        "striatal": ["STR","STRd","STRv","PAL","VTA","SN"],
        "amygdala": ["BLA","LA","CEA","MEA","HY"],
        "prefrontal": ["ACA","PL","ILA","ORB","RSP"],
        "hippocampus": ["DG","CA3","CA1","SUB","ENT"],
        "cortex": ["VIS","SS","MOp","MOs","AUD"]
    }
    pos = {}
    y_step = 2.5
    x_center = 0
    for y_offset, (layer_name, nodes) in enumerate(layers.items()):
        y = -y_offset * y_step
        n_nodes = len(nodes)
        for i, node in enumerate(nodes):
            x = (i - (n_nodes-1)/2) * 2.0
            pos[node] = (x, y)

# ---------- 4. Drawing ----------
plt.figure(figsize=(20, 14))
ax = plt.gca()

# Draw nodes
nx.draw_networkx_nodes(G, pos, node_size=1200, node_color=[node_color_map[n] for n in G.nodes()],
                       edgecolors='black', linewidths=1.5, alpha=0.9)
nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold')

# Draw edges with curves
# We'll use connectionstyle for arcs to reduce overlap
# For simplicity, separate inflow, outflow, loops with different arc radii
def draw_curved_edges(edgelist, color, style='solid', rad=0.2, arrowsize=15, width=1.5):
    for u, v in edgelist:
        if u == v: continue
        # Use FancyArrowPatch for curved edges
        arrow = FancyArrowPatch(pos[u], pos[v],
                                connectionstyle=f"arc3,rad={rad}",
                                arrowstyle='-|>', mutation_scale=arrowsize,
                                color=color, linestyle=style, linewidth=width,
                                zorder=2)
        ax.add_patch(arrow)

# Draw inflow (blue, rad=0.1)
draw_curved_edges(inflow_path, 'blue', rad=0.1, arrowsize=18, width=1.5)
# Draw outflow (red, rad=-0.1 to curve opposite)
draw_curved_edges(outflow_path, 'red', rad=-0.1, arrowsize=18, width=1.5)
# Draw loops (green dashed, rad=0.2)
draw_curved_edges(loop_edges, 'green', style='dashed', rad=0.2, arrowsize=16, width=1.5)

# Edge labels – we need to place them near the middle of each curved edge
# Use a simple approach: compute midpoint of the two nodes and offset perpendicularly
def get_label_pos(u, v, rad):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    # Midpoint
    mx, my = (x1+x2)/2, (y1+y2)/2
    # Perpendicular offset based on rad sign
    dx, dy = x2-x1, y2-y1
    length = np.hypot(dx, dy)
    if length == 0:
        return mx, my
    # Unit perpendicular
    perp = (-dy/length, dx/length)
    offset = 0.3 * rad * length
    return mx + offset*perp[0], my + offset*perp[1]

# Create label dictionaries
in_labels = {edge: f"I{idx+1}" for idx,edge in enumerate(inflow_path)}
out_labels = {edge: f"O{idx+1}" for idx,edge in enumerate(outflow_path)}
loop_labels = {edge: f"L{idx+1}" for idx,edge in enumerate(loop_edges)}

# Draw inflow labels (blue)
for edge, label in in_labels.items():
    if edge[0]==edge[1]: continue
    xl, yl = get_label_pos(edge[0], edge[1], rad=0.1)
    ax.text(xl, yl, label, fontsize=7, color='blue', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# Draw outflow labels (red, rad negative)
for edge, label in out_labels.items():
    if edge[0]==edge[1]: continue
    xl, yl = get_label_pos(edge[0], edge[1], rad=-0.1)
    ax.text(xl, yl, label, fontsize=7, color='red', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# Draw loop labels (green, rad 0.2)
for edge, label in loop_labels.items():
    if edge[0]==edge[1]: continue
    xl, yl = get_label_pos(edge[0], edge[1], rad=0.2)
    ax.text(xl, yl, label, fontsize=7, color='green', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# Add system background rectangles (optional, improves readability)
# Group nodes by system and draw bounding boxes
system_bboxes = {}
for sys, nodes in system_to_nodes.items():
    xs = [pos[n][0] for n in nodes if n in pos]
    ys = [pos[n][1] for n in nodes if n in pos]
    if not xs: continue
    xmin, xmax = min(xs)-1.2, max(xs)+1.2
    ymin, ymax = min(ys)-0.8, max(ys)+0.8
    rect = Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                     facecolor=system_colors[sys], alpha=0.1, edgecolor=system_colors[sys],
                     linewidth=1, linestyle='--', zorder=0)
    ax.add_patch(rect)
    # Label system
    ax.text((xmin+xmax)/2, ymax+0.2, sys, fontsize=8, ha='center', va='bottom',
            color=system_colors[sys], fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(color='blue', label='Inflow (bottom-up)'),
    mpatches.Patch(color='red', label='Outflow (top-down)'),
    mpatches.Patch(color='green', linestyle='dashed', label='Recurrent loops'),
] + [mpatches.Patch(color=col, label=sys) for sys,col in system_colors.items()]

plt.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

plt.title("Mouse Brain Directional Flow Graph (BALB/c)", fontsize=16, fontweight='bold')
plt.axis('off')
plt.tight_layout()
plt.savefig("brain_flow_graph.pdf", dpi=300, bbox_inches='tight')
plt.savefig("brain_flow_graph.png", dpi=300, bbox_inches='tight')
plt.show()
