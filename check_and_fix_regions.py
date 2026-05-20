"""
check_and_fix_regions.py
========================
Checks current region state of all Julia files and
applies the correct patches for the requested graph type.

Usage:
    python3 check_and_fix_regions.py          # just check
    python3 check_and_fix_regions.py Q_7P     # check + fix for Q_7P
    python3 check_and_fix_regions.py Q_7L     # check + fix for Q_7L
    python3 check_and_fix_regions.py Q_8      # check + fix for Q_8
    python3 check_and_fix_regions.py Q_6      # check + fix for Q_6 (base)
"""

import re, os, sys

BASE = os.path.dirname(os.path.abspath(__file__))

FILES = [
    "curved_hh2_sparse_refactored.jl",
    "IharaAssociahedronBridgeV2.jl",
    "OnlineAssociahedronNavigatorV3.jl",
    "ReesBlowupHybrid.jl",
    "SchobarNavigatorV2.jl",
    "SingularityTrackerV2.jl",
    "run_iharaSingV2_MonoTwist.jl",
]

# Expected REGIONS per graph type
EXPECTED = {
    "Q_6":  ["BLA","CA1sp","HPF","HY","LA","sAMY"],
    "Q_7L": ["BLA","CA1sp","HPF","HY","LA","LSX","sAMY"],
    "Q_7P": ["BLA","CA1sp","HPF","HY","LA","PAL","sAMY"],
    "Q_8":  ["BLA","CA1sp","HPF","HY","LA","LSX","PAL","sAMY"],
}

# ── Status checker ────────────────────────────────────────────────────────────

def get_regions_from_file(fpath):
    """Extract all REGIONS/nodes array definitions from a Julia file."""
    if not os.path.exists(fpath):
        return []
    with open(fpath) as f:
        content = f.read()
    defs = re.findall(
        r'(?:const\s+)?(?:REGIONS|nodes)\s*=\s*\[([^\]]+)\]', content)
    result = []
    for d in defs:
        regions = re.findall(r':(\w+)', d)
        if any(r in regions for r in ['CA1sp','BLA','sAMY']):
            result.append(sorted(regions))
    return result

def detect_graph_type(fpath):
    """Detect which graph type a file is currently set to."""
    defs = get_regions_from_file(fpath)
    if not defs:
        return "unknown"
    # Use the first significant definition
    for d in defs:
        s = set(d)
        if   s >= {'BLA','CA1sp','HPF','HY','LA','LSX','PAL','sAMY'}: return "Q_8"
        elif s >= {'BLA','CA1sp','HPF','HY','LA','PAL','sAMY'}:        return "Q_7P"
        elif s >= {'BLA','CA1sp','HPF','HY','LA','LSX','sAMY'}:        return "Q_7L"
        elif s >= {'BLA','CA1sp','HPF','HY','LA','sAMY'}:              return "Q_6"
    return "unknown"

print("Current region state of Julia files:")
print()
print(f"  {'File':<45} {'Current':>6}  Regions")
print(f"  {'-'*44} {'-------':>6}  -------")

current_states = {}
for fname in FILES:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        print(f"  {fname:<45} {'N/A':>6}  (not found)")
        continue
    gt = detect_graph_type(fpath)
    defs = get_regions_from_file(fpath)
    regions_str = str(defs[0]) if defs else "none"
    current_states[fname] = gt
    print(f"  {fname:<45} {gt:>6}  {regions_str}")

print()

# ── Fix if requested ──────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("To fix for a specific graph, run:")
    print("  python3 check_and_fix_regions.py Q_7P")
    sys.exit(0)

target = sys.argv[1]
if target not in EXPECTED:
    print(f"Unknown graph type: {target}")
    print(f"Choose from: {list(EXPECTED.keys())}")
    sys.exit(1)

print(f"Fixing all files to {target}...")
print(f"Target regions: {EXPECTED[target]}")
print()

# Build the target REGIONS strings
tgt = EXPECTED[target]
tgt_syms_spaced  = "[" + ", ".join(f":{r}" for r in tgt) + "]"
tgt_syms_compact = "[" + ",".join(f":{r}" for r in tgt) + "]"

# For each file, replace any REGIONS/nodes definition that doesn't match target
def fix_file(fpath, target_regions):
    if not os.path.exists(fpath):
        return 0
    with open(fpath) as f:
        content = f.read()

    original = content
    count = 0

    # Find and replace all REGIONS/nodes array definitions
    def replace_regions(match):
        nonlocal count
        full_match = match.group(0)
        inner = match.group(1)
        existing = re.findall(r':(\w+)', inner)
        existing_sig = set(e for e in existing
                          if e in ['CA1sp','BLA','HY','HPF','sAMY','LA',
                                   'PAL','LSX'])
        target_sig = set(target_regions)

        if existing_sig == target_sig:
            return full_match  # already correct

        # Preserve original formatting (spaces or compact)
        has_spaces = ', :' in full_match
        if has_spaces:
            new_inner = ", ".join(f":{r}" for r in target_regions)
        else:
            new_inner = ",".join(f":{r}" for r in target_regions)

        # Preserve const prefix and variable name
        prefix = re.match(r'(?:const\s+)?(?:REGIONS|nodes)\s*=\s*\[', full_match)
        if prefix:
            count += 1
            return full_match[:prefix.end()] + new_inner + "]"
        return full_match

    content = re.sub(
        r'(?:const\s+)?(?:REGIONS|nodes)\s*=\s*\[([^\]]+)\]',
        replace_regions,
        content
    )

    # Fix n_regions count
    for old_n, new_n in [("n_regions = 6", f"n_regions = {len(target_regions)}"),
                          ("n_regions = 7", f"n_regions = {len(target_regions)}"),
                          ("n_regions = 8", f"n_regions = {len(target_regions)}")]:
        if old_n in content and old_n != f"n_regions = {len(target_regions)}":
            content = content.replace(old_n, new_n)

    # Fix adjacency in OnlineAssociahedronNavigatorV3
    fname = os.path.basename(fpath)
    if fname == "OnlineAssociahedronNavigatorV3.jl":
        content = fix_adjacency(content, target_regions)

    # Fix region_to_idx in curved_hh2
    if fname == "curved_hh2_sparse_refactored.jl":
        content = fix_region_to_idx(content, target_regions)

    if content != original:
        with open(fpath, 'w') as f:
            f.write(content)
        return count
    return 0

def fix_adjacency(content, target_regions):
    """Fix the ADJACENCY dict in OnlineAssociahedronNavigatorV3."""
    # Base adjacency (Q_6 core, always present)
    base = {
        "CA1sp": [":HPF",":sAMY"],
        "BLA":   [":sAMY",":LA"],
        "HY":    [":sAMY"],
        "HPF":   [":CA1sp",":BLA",":sAMY"],
        "sAMY":  [":CA1sp",":BLA",":HY",":HPF",":LA"],
        "LA":    [":BLA",":sAMY"],
    }
    # Add extensions
    if "PAL" in target_regions:
        base["HY"].append(":PAL")
        base["sAMY"].append(":PAL")
        base["PAL"] = [":HY",":sAMY"]
    if "LSX" in target_regions:
        base["HPF"].append(":LSX")
        base["LSX"] = [":HPF"]

    # Build new adjacency block
    lines = []
    for region in ["CA1sp","BLA","HY","HPF","sAMY","LA"] + \
                  (["PAL"] if "PAL" in target_regions else []) + \
                  (["LSX"] if "LSX" in target_regions else []):
        if region in base:
            nbrs = ", ".join(base[region])
            lines.append(f"        :{region:<6} => [{nbrs}]")

    new_adj = ",\n".join(lines)

    # Replace the adjacency dict pattern
    content = re.sub(
        r'(:CA1sp\s*=>[^\}]+?:LA\s*=>\s*\[[^\]]*\](?:,\n[^\}]+?=>.*?\])*)',
        new_adj,
        content,
        flags=re.DOTALL
    )
    return content

def fix_region_to_idx(content, target_regions):
    """Fix region_to_idx mapping based on CSV index order."""
    # Index order from CSV files (alphabetical by region name = CSV index)
    idx_map = {r: i for i, r in enumerate(sorted(target_regions))}
    # But sAMY is last alphabetically — check CSV order
    # Q_6:  BLA=0,CA1sp=1,HPF=2,HY=3,LA=4,sAMY=5
    # Q_7P: BLA=0,CA1sp=1,HPF=2,HY=3,LA=4,PAL=5,sAMY=6
    # Q_7L: BLA=0,CA1sp=1,HPF=2,HY=3,LA=4,LSX=5,sAMY=6
    # Q_8:  BLA=0,CA1sp=1,HPF=2,HY=3,LA=4,LSX=5,PAL=6,sAMY=7
    csv_order = sorted(target_regions)  # alphabetical matches CSV
    idx_map = {r: i for i, r in enumerate(csv_order)}

    new_dict = 'region_to_idx = Dict(' + \
               ', '.join(f'"{r}"=>{i}' for r, i in sorted(idx_map.items())) + ')'

    content = re.sub(
        r'region_to_idx\s*=\s*Dict\([^)]+\)',
        new_dict,
        content
    )
    return content

# Apply fixes
total = 0
for fname in FILES:
    fpath = os.path.join(BASE, fname)
    n = fix_file(fpath, tgt)
    if n > 0:
        print(f"  Fixed {n} definition(s): {fname}")
    else:
        print(f"  Already OK: {fname}")
    total += n

print()
print(f"Total fixes: {total}")
print()

# Re-verify
print("Verification after fix:")
all_ok = True
for fname in FILES:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath): continue
    gt = detect_graph_type(fpath)
    ok = gt == target
    if not ok: all_ok = False
    print(f"  {fname:<45} {gt:>6}  {'✓' if ok else '✗ expected '+target}")

print()
print(f"All files match {target}: {all_ok}")
