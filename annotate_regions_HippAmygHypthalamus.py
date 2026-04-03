import pandas as pd

NODE = "/Users/vaw1/Downloads/OGB/BALBc_no1_raw/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_nodes.csv"
ONEHOT_ENCODED_REGIONS="/Users/vaw1/Downloads/OGB/BALBc-no1_iso3um_stitched_segmentation_bulge_size_3.0_atlas_processed.csv" 

# 1. Load original node CSV
nodes = pd.read_csv(NODE, sep=";")

# 2. Load one-hot atlas map
onehot_map = pd.read_csv(ONEHOT_ENCODED_REGIONS, sep=";")

# 3. Ensure row counts match
assert len(nodes) == len(onehot_map), "Node file and one-hot map row count mismatch!"

# 4. Define target regions
target_regions = {"CA1sp", "HPF", "BLA", "sAMY", "LA", "HY"}

# 5. Build a boolean mask for rows that have any of the target regions
#    The one-hot columns are named e.g., "Region_Acronym_CA1"
mask = pd.Series(False, index=onehot_map.index)
for region in target_regions:
    col_name = f"Region_Acronym_{region}"
    if col_name in onehot_map.columns:
        mask |= (onehot_map[col_name] == 1)
    else:
        print(f"Warning: Column {col_name} not found in one-hot map.")

# Confirm missing MEA, CEA, CA3, DG
df = pd.read_csv(ONEHOT_ENCODED_REGIONS, sep=';')
# Look for columns containing 'DG', 'CA3', 'CEA', 'MEA'
matching_cols = [col for col in df.columns if 'DG' in col or 'CA3' in col or 'CEA' in col or 'MEA' in col]
print(matching_cols)


# 6. Filter both dataframes
filtered_nodes = nodes[mask]
filtered_onehot = onehot_map[mask]

# 7. Create 'regions' column for filtered rows (optional, but keeps functionality)
region_cols = filtered_onehot.columns.tolist()

def get_regions(row):
    return [col.replace("Region_Acronym_", "") for col in region_cols if row[col] == 1]

filtered_nodes["regions"] = filtered_onehot.apply(get_regions, axis=1)

# 8. Save final CSV (only nodes belonging to target regions)
filtered_nodes.to_csv("node_regions_cleanHippAmygHypthalamus_HPF_sAMY.csv", sep=";", index=False)

print(f"Done! Saved {len(filtered_nodes)} nodes (filtered to target regions).")


