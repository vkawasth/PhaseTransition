import pandas as pd
import sys
import os
import ast

def generate_consolidated_qpa(node_path, edge_path):
    # The regions you want to track in your GAP QPA model
    TARGET_REGIONS = ['CA1sp', 'BLA', 'HY', 'HPF', 'sAMY', 'LA']
    
    node_to_region = {}
    
    print("--- Phase 1: Mapping Nodes from Single File ---")
    if not os.path.exists(node_path):
        print(f"Error: Node file not found at {node_path}")
        return

    # Using chunksize to handle 3 million nodes efficiently
    node_chunks = pd.read_csv(node_path, sep=';', chunksize=500000)
    
    for chunk in node_chunks:
        # We look specifically at 'id' and 'regions'
        for _, row in chunk.iterrows():
            try:
                # Convert string "['bgr']" to actual list ['bgr']
                raw_regions = ast.literal_eval(row['regions'])
                
                # Check if any of our target regions are in this node's list
                for target in TARGET_REGIONS:
                    if target in raw_regions:
                        node_to_region[int(row['id'])] = target
                        break # Node assigned to the first matching target region
            except:
                continue
                
    print(f"  [+] Indexed {len(node_to_region)} nodes belonging to target regions.")

    if len(node_to_region) == 0:
        print("!!! WARNING: No nodes found matching target regions. Check spelling in CSV.")
        return

    print("\n--- Phase 2: Processing 5 Million Edges ---")
    if not os.path.exists(edge_path):
        print(f"Error: Edge file not found at {edge_path}")
        return

    consolidated_flows = {}
    edge_chunks = pd.read_csv(edge_path, sep=';', usecols=['node1id', 'node2id'], chunksize=1000000)
    
    total_edges = 0
    cross_region_count = 0
    
    for chunk in edge_chunks:
        # Map node IDs to the regions found in Phase 1
        r1 = chunk['node1id'].map(node_to_region)
        r2 = chunk['node2id'].map(node_to_region)
        
        # Identify edges where BOTH nodes are in target regions AND the regions are DIFFERENT
        mask = r1.notna() & r2.notna() & (r1 != r2)
        inter_region_df = pd.DataFrame({'src': r1[mask], 'tgt': r2[mask]})
        
        if not inter_region_df.empty:
            counts = inter_region_df.groupby(['src', 'tgt']).size()
            for (src, tgt), count in counts.items():
                consolidated_flows[(src, tgt)] = consolidated_flows.get((src, tgt), 0) + count
                cross_region_count += count
        
        total_edges += len(chunk)
        print(f"  Processed {total_edges} edges... (Found {cross_region_count} inter-regional connections so far)")

    print("\n--- Phase 3: Generating GAP QPA Output ---")
    arrow_defs = []
    for (src, tgt), weight in consolidated_flows.items():
        arrow_name = f"f_{src}_{tgt}"
        arrow_defs.append(f'["{src}", "{tgt}", "{arrow_name}"]')
        print(f"  Confirmed Flow: {src} -> {tgt} ({weight} voxel edges)")

    if not arrow_defs:
        print("!!! STILL NO ARROWS: This means that although nodes were found, no edges in your file connect two different target regions.")

    gap_code = [
        'LoadPackage("qpa");',
        'k := GF(101);',
        f'Q_vertices := {TARGET_REGIONS};',
        f'Q_arrows := [{", ".join(arrow_defs)}];',
        'Q := Quiver(Q_vertices, Q_arrows);',
        'A := PathAlgebra(k, Q);',
        'rels := [];',
        'I := Ideal(A, rels);',
        'B := A / I;',
        'Print("Success: Brain Algebra B defined with ", Dimension(B), " dimensions.\\n");'
    ]
    
    with open("brain_complex_quiver.g", "w") as f:
        f.write("\n".join(gap_code))
    print("\nFile saved as: brain_complex_quiver.g")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py nodes.csv edges.csv")
    else:
        generate_consolidated_qpa(sys.argv[1], sys.argv[2])
