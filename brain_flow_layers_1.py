import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import FancyArrowPatch
import matplotlib.patches as mpatches

np.alltrue = np.all

# ---------- Data (same as before) ----------
regions = [
    "DG","CA3","CA1","SUB","ENT",
    "ACA","PL","ILA","ORB","RSP",
    "BLA","LA","CEA","MEA","HY",
    "VIS","SS","MOp","MOs","AUD",
    "STR","STRd","STRv","PAL",
    "MB","P","CB",
    "VTA","SN"
]

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
outflow_path.append(("VTA","SN"))

loop_edges = [
    ("ACA","MB"), ("MB","ACA"),
    ("CA1","MB"), ("MB","CA1"),
    ("MOp","STR"), ("STR","PAL"), ("PAL","MB"), ("MB","MOp"),
    ("VTA","STR"), ("STR","PAL"), ("PAL","VTA"),
    ("VTA","SN"), ("SN","VTA"),
]

G = nx.DiGraph()
G.add_nodes_from(regions)
G.add_edges_from(inflow_path + outflow_path + loop_edges)

# ---------- Node colors by system ----------
system_colors = {
    "Hippocampal": "#1f77b4",
    "Prefrontal": "#ff7f0e",
    "Amygdala_Hyp": "#2ca02c",
    "SensoryMotor": "#d62728",
    "Striatal": "#9467bd",
    "Brainstem": "#8c564b"
}
system_to_nodes = {
    "Hippocampal": ["DG","CA3","CA1","SUB","ENT"],
    "Prefrontal": ["ACA","PL","ILA","ORB","RSP"],
    "Amygdala_Hyp": ["BLA","LA","CEA","MEA","HY"],
    "SensoryMotor": ["VIS","SS","MOp","MOs","AUD"],
    "Striatal": ["STR","STRd","STRv","PAL","VTA","SN"],
    "Brainstem": ["MB","P","CB"]
}
node_color = {}
for sys, nodes in system_to_nodes.items():
    for n in nodes:
        node_color[n] = system_colors[sys]

# ---------- Layout (manual hierarchical, no Graphviz needed) ----------
layers = {
    "brainstem": ["MB","P","CB"],
    "striatal": ["STR","STRd","STRv","PAL","VTA","SN"],
    "amygdala": ["BLA","LA","CEA","MEA","HY"],
    "prefrontal": ["ACA","PL","ILA","ORB","RSP"],
    "hippocampus": ["DG","CA3","CA1","SUB","ENT"],
    "cortex": ["VIS","SS","MOp","MOs","AUD"]
}
base_pos = {}
y_step = 2.5
for y_offset, (layer_name, nodes) in enumerate(layers.items()):
    y = -y_offset * y_step
    n_nodes = len(nodes)
    for i, node in enumerate(nodes):
        x = (i - (n_nodes-1)/2) * 2.0
        base_pos[node] = (x, y)

# Create shifted positions for inflow (up) and outflow (down)
dy_in = 0.3
dy_out = -0.3
pos_in = {n: (x, y + dy_in) for n, (x, y) in base_pos.items()}
pos_out = {n: (x, y + dy_out) for n, (x, y) in base_pos.items()}
pos_loop = base_pos  # loops use original positions

# ---------- Helper to draw curved edges with given node positions ----------
def draw_curved_edges(ax, edgelist, pos_dict, color, style='solid', rad=0.2, arrowsize=15, width=1.5):
    for u, v in edgelist:
        if u == v:
            continue
        p1 = pos_dict[u]
        p2 = pos_dict[v]
        arrow = FancyArrowPatch(p1, p2,
                                connectionstyle=f"arc3,rad={rad}",
                                arrowstyle='-|>', mutation_scale=arrowsize,
                                color=color, linestyle=style, linewidth=width,
                                zorder=2)
        ax.add_patch(arrow)

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(20, 14))

# Draw nodes (using base positions for consistent placement)
nx.draw_networkx_nodes(G, base_pos, node_size=1200,
                       node_color=[node_color[n] for n in G.nodes()],
                       edgecolors='black', linewidths=1.5, alpha=0.9, ax=ax)
nx.draw_networkx_labels(G, base_pos, font_size=9, font_weight='bold', ax=ax)

# Draw edges with separate shifted positions
draw_curved_edges(ax, inflow_path, pos_in, 'blue', rad=0.1, arrowsize=18, width=1.5)
draw_curved_edges(ax, outflow_path, pos_out, 'red', rad=-0.1, arrowsize=18, width=1.5)
draw_curved_edges(ax, loop_edges, pos_loop, 'green', style='dashed', rad=0.2, arrowsize=16, width=1.5)

# Edge labels – position them at midpoints of the shifted edge paths
def get_label_pos(u, v, pos_dict, rad):
    x1, y1 = pos_dict[u]
    x2, y2 = pos_dict[v]
    mx, my = (x1+x2)/2, (y1+y2)/2
    dx, dy = x2-x1, y2-y1
    length = np.hypot(dx, dy)
    if length == 0:
        return mx, my
    perp = (-dy/length, dx/length)
    offset = 0.3 * rad * length
    return mx + offset*perp[0], my + offset*perp[1]

in_labels = {e: f"I{i+1}" for i, e in enumerate(inflow_path)}
out_labels = {e: f"O{i+1}" for i, e in enumerate(outflow_path)}
loop_labels = {e: f"L{i+1}" for i, e in enumerate(loop_edges)}

# Inflow labels (blue)
for edge, label in in_labels.items():
    if edge[0]==edge[1]: continue
    xl, yl = get_label_pos(edge[0], edge[1], pos_in, rad=0.1)
    ax.text(xl, yl, label, fontsize=7, color='blue', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# Outflow labels (red)
for edge, label in out_labels.items():
    if edge[0]==edge[1]: continue
    xl, yl = get_label_pos(edge[0], edge[1], pos_out, rad=-0.1)
    ax.text(xl, yl, label, fontsize=7, color='red', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# Loop labels (green)
for edge, label in loop_labels.items():
    if edge[0]==edge[1]: continue
    xl, yl = get_label_pos(edge[0], edge[1], pos_loop, rad=0.2)
    ax.text(xl, yl, label, fontsize=7, color='green', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none'))

# System background boxes
for sys, nodes in system_to_nodes.items():
    xs = [base_pos[n][0] for n in nodes if n in base_pos]
    ys = [base_pos[n][1] for n in nodes if n in base_pos]
    if not xs: continue
    xmin, xmax = min(xs)-1.2, max(xs)+1.2
    ymin, ymax = min(ys)-0.8, max(ys)+0.8
    rect = plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin,
                         facecolor=system_colors[sys], alpha=0.1, edgecolor=system_colors[sys],
                         linewidth=1, linestyle='--', zorder=0)
    ax.add_patch(rect)
    ax.text((xmin+xmax)/2, ymax+0.2, sys, fontsize=8, ha='center', va='bottom',
            color=system_colors[sys], fontweight='bold')

# Legend
legend_elements = [
    mpatches.Patch(color='blue', label='Inflow (bottom-up)'),
    mpatches.Patch(color='red', label='Outflow (top-down)'),
    mpatches.Patch(color='green', linestyle='dashed', label='Recurrent loops'),
] + [mpatches.Patch(color=col, label=sys) for sys, col in system_colors.items()]
ax.legend(handles=legend_elements, loc='upper right', fontsize=8, framealpha=0.9)

ax.set_title("Mouse Brain Directional Flow Graph (BALB/c)", fontsize=16, fontweight='bold')
ax.axis('off')
plt.tight_layout()
plt.savefig("brain_flow_separated.pdf", dpi=300, bbox_inches='tight')
plt.savefig("brain_flow_separated.png", dpi=300, bbox_inches='tight')
plt.show()
