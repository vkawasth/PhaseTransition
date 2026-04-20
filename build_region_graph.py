import pandas as pd
import numpy as np
from collections import defaultdict

# Paths (adjust as needed)
#NODES_CSV = "node_regions_cleanHippAmygHypthalamus_ALLWITH_PAL_LSX.csv"  # output from annotation script
NODES_CSV = "node_regions_cleanHippAmygHypthalamus.csv"  # output from annotation script
EDGES_CSV = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_edges.csv"
OUTPUT_CSV = "region_graph.csv"

# 1. Load filtered nodes (voxels belonging to target regions)
nodes_df = pd.read_csv(NODES_CSV, sep=';')
# Ensure 'regions' column is a list (it is stored as string like "['CA1sp']")
import ast
nodes_df['regions_list'] = nodes_df['regions'].apply(ast.literal_eval)

# 2. Create a mapping from node id to its region (first region if multiple? Usually each voxel belongs to exactly one region)
# We'll use the first region in the list (most are single)
node_to_region = {}
for idx, row in nodes_df.iterrows():
    reg_list = row['regions_list']
    if reg_list:
        node_to_region[row['id']] = reg_list[0]   # take first region

# 3. Load edges (full OGB edge file)
edges_df = pd.read_csv(EDGES_CSV, sep=';')
# Filter edges where both endpoints are in our node_to_region map
edges_df = edges_df[edges_df['node1id'].isin(node_to_region) & edges_df['node2id'].isin(node_to_region)]

# 4. Aggregate edge weights by region pair (directed)
region_weights = defaultdict(float)
for _, row in edges_df.iterrows():
    reg1 = node_to_region[row['node1id']]
    reg2 = node_to_region[row['node2id']]
    # Use edge length as weight (or volume, or curvature)
    weight = row['length']   # or row['volume']
    region_weights[(reg1, reg2)] += weight

# 5. Build region list (ordered alphabetically for consistency)
region_list = sorted(set(node_to_region.values()))
region_to_idx = {r: i for i, r in enumerate(region_list)}

# 6. Build edge list and weight dictionary for FullGraphDynamics
edges = []
edge_weights = {}
for (r1, r2), w in region_weights.items():
    i, j = region_to_idx[r1], region_to_idx[r2]
    edges.append((i, j))
    edge_weights[(i, j)] = w

# 7. Compute centroids of each region (mean coordinates of voxels in that region)
centroids = {}
for reg in region_list:
    voxels = nodes_df[nodes_df['regions_list'].apply(lambda x: reg in x)]
    coords = voxels[['pos_x', 'pos_y', 'pos_z']].mean().values
    centroids[reg] = coords

# 8. Save everything to a CSV for later use
region_graph_df = pd.DataFrame({
    'region': region_list,
    'index': [region_to_idx[r] for r in region_list],
    'centroid_x': [centroids[r][0] for r in region_list],
    'centroid_y': [centroids[r][1] for r in region_list],
    'centroid_z': [centroids[r][2] for r in region_list]
})
# Also save edges as separate DataFrame
edges_df_out = pd.DataFrame([(i, j, w) for (i, j), w in edge_weights.items()],
                            columns=['src_idx', 'tgt_idx', 'weight'])
region_graph_df.to_csv("regions.csv", index=False)
edges_df_out.to_csv("region_edges.csv", index=False)

print(f"Region graph saved: {len(region_list)} regions, {len(edges)} directed edges.")
print("Regions:", region_list)
