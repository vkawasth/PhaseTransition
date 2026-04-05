import pandas as pd
import sys
import os
import ast
from itertools import permutations

def generate_consolidated_qpa(node_path, edge_path):
    TARGET_REGIONS = ['CA1sp', 'BLA', 'HY', 'HPF', 'sAMY', 'LA']
    
    node_to_region = {}
    
    print("--- Phase 1: Mapping Nodes ---")
    node_chunks = pd.read_csv(node_path, sep=';', chunksize=500000)
    
    for chunk in node_chunks:
        for _, row in chunk.iterrows():
            try:
                raw_regions = ast.literal_eval(row['regions'])
                for target in TARGET_REGIONS:
                    if target in raw_regions:
                        node_to_region[int(row['id'])] = target
                        break
            except:
                continue

    print(f"[+] Indexed {len(node_to_region)} nodes.")

    print("\n--- Phase 2: Processing Edges ---")
    consolidated_flows = {}

    edge_chunks = pd.read_csv(edge_path, sep=';', usecols=['node1id', 'node2id'], chunksize=1000000)

    for chunk in edge_chunks:
        r1 = chunk['node1id'].map(node_to_region)
        r2 = chunk['node2id'].map(node_to_region)

        mask = r1.notna() & r2.notna() & (r1 != r2)
        df = pd.DataFrame({'src': r1[mask], 'tgt': r2[mask]})

        counts = df.groupby(['src', 'tgt']).size()
        for (src, tgt), count in counts.items():
            consolidated_flows[(src, tgt)] = consolidated_flows.get((src, tgt), 0) + count

    print("\n--- Phase 3: Build Arrows ---")

    arrow_defs = []
    for (src, tgt), weight in consolidated_flows.items():
        arrow_name = f"f_{src}_{tgt}"
        arrow_defs.append(f'["{src}", "{tgt}", "{arrow_name}"]')
        print(f"Flow: {src} -> {tgt} ({weight})")

    # -----------------------------
    # NEW: Build HH² relations
    # -----------------------------
    print("\n--- Phase 4: Building HH² Relations ---")

    rels = []

    for (i, j, k) in permutations(TARGET_REGIONS, 3):
        if (i, j) in consolidated_flows and (j, k) in consolidated_flows:

            wij = consolidated_flows[(i, j)]
            wjk = consolidated_flows[(j, k)]
            wik = consolidated_flows.get((i, k), 0)

            delta = wij * wjk - wik

            # Threshold to avoid noise
            if abs(delta) > 10:
                rel = f"f_{i}_{j}*f_{j}_{k} - {delta % 101}*f_{i}_{k}"
                rels.append(rel)
                print(f"Relation: {rel}")

    # -----------------------------
    # NEW: Loop relations (2-cycles)
    # -----------------------------
    print("\n--- Phase 5: Loop Relations ---")

    for (i, j), wij in consolidated_flows.items():
        if (j, i) in consolidated_flows:
            wji = consolidated_flows[(j, i)]
            val = (wij * wji) % 101

            rel = f"f_{i}_{j}*f_{j}_{i} - {val}*e_{i}"
            rels.append(rel)

            print(f"Loop: {rel}")

    # -----------------------------
    # GAP OUTPUT
    # -----------------------------
    print("\n--- Phase 6: Writing GAP File ---")

    gap_code = [
        'LoadPackage("qpa");',
        'k := GF(101);',
        f'Q_vertices := {TARGET_REGIONS};',
        f'Q_arrows := [{", ".join(arrow_defs)}];',
        'Q := Quiver(Q_vertices, Q_arrows);',
        'A := PathAlgebra(k, Q);',
        '',
        '# Define idempotents',
        'for v in Q_vertices do',
        '  AssignGeneratorVariables(A);',
        'od;',
        '',
        '# Relations',
        f'rels := [{", ".join(rels)}];',
        '',
        'I := Ideal(A, rels);',
        'B := A / I;',
        '',
        'Print("Algebra constructed. Dimension: ", Dimension(B), "\\n");',
        '',
        '# Compute HH^2',
        'HH2 := HochschildCohomology(B,2);',
        'Print("HH^2 dimension: ", Dimension(HH2), "\\n");',
        '',
        '# Cartan matrix',
        'C := CartanMatrix(B);',
        'Print("Cartan matrix: ", C, "\\n");'
    ]

    with open("brain_complex_quiver_FIXED.g", "w") as f:
        f.write("\n".join(gap_code))

    print("\n✅ File saved: brain_complex_quiver_FIXED.g")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py nodes.csv edges.csv")
    else:
        generate_consolidated_qpa(sys.argv[1], sys.argv[2])
