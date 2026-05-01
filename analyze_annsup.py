import json
import glob
import os
import zipfile
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def analyze_json_files(zip_path, extract_to='./temp_analysis'):
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_to)
    
    json_files = glob.glob(os.path.join(extract_to, '*.json'))
    if not json_files:
        print("No JSON files found in the zip.")
        return
    
    json_files.sort()
    data_summary = []
    
    for f in json_files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        
        ann = data.get('annihilator_infty', [])
        supp = data.get('support_infty', [])
        prime_paths = data.get('prime_paths', [])   # list of {"path": [...], "weight": w}
        prime_ideals = data.get('prime_higher_ideals', [])  # list of dicts with "path", "weight", "closure", "total_support"
        
        set_ann = set(ann)
        set_supp = set(supp)
        overlap = set_ann.intersection(set_supp)
        
        # Prime paths statistics
        pp_weights = [p.get('weight', 0.0) for p in prime_paths]
        total_pp_weight = sum(pp_weights)
        max_pp_weight = max(pp_weights) if pp_weights else 0.0
        
        # Prime ideals statistics
        pi_total_support = sum(ideal.get('total_support', 0.0) for ideal in prime_ideals)
        pi_closure_sizes = [len(ideal.get('closure', [])) for ideal in prime_ideals]
        avg_closure_size = np.mean(pi_closure_sizes) if pi_closure_sizes else 0.0
        
        summary = {
            'file': os.path.basename(f),
            'ann_count': len(ann),
            'supp_count': len(supp),
            'overlap_count': len(overlap),
            'prime_paths_count': len(prime_paths),
            'prime_paths_total_weight': total_pp_weight,
            'prime_paths_max_weight': max_pp_weight,
            'prime_ideals_count': len(prime_ideals),
            'prime_ideals_total_support': pi_total_support,
            'prime_ideals_avg_closure_size': avg_closure_size,
            'sample_ann': ann[:5] if ann else [],
            'sample_supp': supp[:5] if supp else [],
            'sample_prime_paths': prime_paths[:3] if prime_paths else [],
        }
        data_summary.append(summary)
    
    # Print table
    print("\n=== Detailed Analysis of annihilator / support / prime paths ===\n")
    print(f"{'File':<30} | Ann | Supp | Overlap | PP_cnt | PP_total | PP_max | PI_cnt | PI_sup")
    print("-" * 100)
    for s in data_summary:
        print(f"{s['file'][:30]:<30} | {s['ann_count']:3d} | {s['supp_count']:3d} | {s['overlap_count']:3d} | "
              f"{s['prime_paths_count']:3d} | {s['prime_paths_total_weight']:8.2f} | {s['prime_paths_max_weight']:8.2f} | "
              f"{s['prime_ideals_count']:3d} | {s['prime_ideals_total_support']:8.2f}")
    
    # Plot trends
    if len(data_summary) > 1:
        indices = list(range(len(data_summary)))
        pp_counts = [s['prime_paths_count'] for s in data_summary]
        pp_totals = [s['prime_paths_total_weight'] for s in data_summary]
        pi_counts = [s['prime_ideals_count'] for s in data_summary]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes[0,0].plot(indices, pp_counts, 'o-', color='blue')
        axes[0,0].set_title('Prime paths count')
        axes[0,0].grid(True)
        
        axes[0,1].plot(indices, pp_totals, 's-', color='orange')
        axes[0,1].set_title('Prime paths total weight')
        axes[0,1].grid(True)
        
        axes[1,0].plot(indices, pi_counts, '^-', color='green')
        axes[1,0].set_title('Prime ideals count')
        axes[1,0].grid(True)
        
        axes[1,1].plot(indices, [s['prime_ideals_total_support'] for s in data_summary], 'd-', color='red')
        axes[1,1].set_title('Prime ideals total support')
        axes[1,1].grid(True)
        
        plt.tight_layout()
        plt.savefig('prime_paths_trends.png')
        print("\nPlot saved as prime_paths_trends.png")
    
    # Summary statistics
    files_with_pp = sum(1 for s in data_summary if s['prime_paths_count'] > 0)
    files_with_pi = sum(1 for s in data_summary if s['prime_ideals_count'] > 0)
    print(f"\nFiles with non‑zero prime paths: {files_with_pp}/{len(data_summary)}")
    print(f"Files with non‑zero prime ideals: {files_with_pi}/{len(data_summary)}")
    
    # Sample first file's prime paths (if any)
    if data_summary and data_summary[0]['prime_paths_count'] > 0:
        print(f"\nExample prime paths from {data_summary[0]['file']}:")
        for pp in data_summary[0]['sample_prime_paths']:
            print(f"  path: {pp.get('path', [])[:3]}... weight: {pp.get('weight', 0):.2f}")
    
    # Cleanup (optional)
    # import shutil; shutil.rmtree(extract_to)
    
    return data_summary

if __name__ == "__main__":
    zip_file_path = "Archive_full.zip"  # change this
    analyze_json_files(zip_file_path)
