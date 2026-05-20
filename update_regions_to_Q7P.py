"""
update_regions_to_Q7P.py
========================
Updates region definitions in all Julia files from Q_6 (6 regions)
to Q_7P (7 regions: adds PAL).

Target region order: [:BLA, :CA1sp, :HPF, :HY, :LA, :PAL, :sAMY]
(alphabetical, matching regions_six_ANDPAL.csv index order)

Files updated:
  curved_hh2_sparse_refactored.jl
  IharaAssociahedronBridgeV2.jl
  OnlineAssociahedronNavigatorV3.jl
  ReesBlowupHybrid.jl
  SchobarNavigatorV2.jl
  SingularityTrackerV2.jl
  run_iharaSingV2_MonoTwist.jl  (if present)

What is changed:
  - REGIONS const arrays
  - nodes const arrays
  - region_order arrays
  - n_regions fixed counts
  - region_to_idx dicts
  - ADJACENCY dict in OnlineAssociahedronNavigatorV3
  - yticks labels in SingularityTrackerV2

What is NOT changed:
  - Arrow relation strings (f_CA1sp_HPF etc) — these come from CSV
  - Biological weight constants — these come from CSV
  - Simulation logic
"""

import re
import os

FILES = [
    "curved_hh2_sparse_refactored.jl",
    "IharaAssociahedronBridgeV2.jl",
    "OnlineAssociahedronNavigatorV3.jl",
    "ReesBlowupHybrid.jl",
    "SchobarNavigatorV2.jl",
    "SingularityTrackerV2.jl",
    "run_iharaSingV2_MonoTwist.jl",
]

BASEDIR = os.path.dirname(os.path.abspath(__file__))

# ── New region definitions ────────────────────────────────────────────────────

# Alphabetical order matching regions_six_ANDPAL.csv
NEW_REGIONS_SORTED = ["BLA", "CA1sp", "HPF", "HY", "LA", "PAL", "sAMY"]

NEW_REGIONS_JULIA = "[:BLA, :CA1sp, :HPF, :HY, :LA, :PAL, :sAMY]"
NEW_REGIONS_COMPACT = "[:BLA,:CA1sp,:HPF,:HY,:LA,:PAL,:sAMY]"
NEW_N_REGIONS = 7

# Index mapping (0-based, matching CSV index column)
# From regions_six_ANDPAL.csv:
#   BLA=0, CA1sp=1, HPF=2, HY=3, LA=4, PAL=5, sAMY=6
NEW_REGION_TO_IDX = (
    '    "BLA"=>0, "CA1sp"=>1, "HPF"=>2, "sAMY"=>6, "HY"=>3, "LA"=>4, "PAL"=>5'
)
NEW_REGION_TO_IDX_COMPACT = (
    '"BLA"=>0,"CA1sp"=>1,"HPF"=>2,"sAMY"=>6,"HY"=>3,"LA"=>4,"PAL"=>5'
)

# Updated adjacency for OnlineAssociahedronNavigatorV3
# PAL connects to HY and sAMY (symmetric pairs)
NEW_ADJACENCY = """\
    :CA1sp => [:HPF, :sAMY],
        :BLA   => [:sAMY, :LA],
        :HY    => [:sAMY, :PAL],
        :HPF   => [:CA1sp, :BLA, :sAMY],
        :sAMY  => [:CA1sp, :BLA, :HY, :HPF, :LA, :PAL],
        :LA    => [:BLA, :sAMY],
        :PAL   => [:HY, :sAMY]"""

# Updated yticks for SingularityTrackerV2
NEW_YTICKS = '(0:7, ["Unk","CA1sp","BLA","HY","HPF","sAMY","LA","PAL"])'

# Updated node_symbols for curved_hh2
NEW_NODE_SYMBOLS = "[:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]"

# Updated nodes const for curved_hh2
NEW_NODES_CONST = "const nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]"

def patch_file(filepath, patches, label):
    """Apply a list of (old, new) string replacements to a file."""
    if not os.path.exists(filepath):
        print(f"  SKIP (not found): {filepath}")
        return 0

    with open(filepath) as f:
        content = f.read()

    n_applied = 0
    for old, new, desc in patches:
        if old in content:
            content = content.replace(old, new)
            n_applied += 1
            print(f"  [{label}] OK: {desc}")
        else:
            # Try regex for whitespace-flexible matches
            print(f"  [{label}] NOT FOUND: {desc}")

    if n_applied > 0:
        with open(filepath, 'w') as f:
            f.write(content)

    return n_applied


# ── Per-file patches ──────────────────────────────────────────────────────────

def patch_curved_hh2(filepath):
    patches = [
        # nodes const
        (
            "const nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA]",
            NEW_NODES_CONST,
            "nodes const"
        ),
        # region_to_idx dict
        (
            'region_to_idx = Dict("CA1sp"=>0, "HPF"=>1, "BLA"=>2, "sAMY"=>3, "HY"=>4, "LA"=>5)',
            f'region_to_idx = Dict("BLA"=>0, "CA1sp"=>1, "HPF"=>2, "HY"=>3, "LA"=>4, "PAL"=>5, "sAMY"=>6)',
            "region_to_idx dict"
        ),
        # node_symbols in function
        (
            "node_symbols = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA]",
            f"node_symbols = {NEW_NODE_SYMBOLS}",
            "node_symbols in function"
        ),
        # n_regions fixed count
        (
            "n_regions = 6   # fixed: CA1sp, HPF, BLA, sAMY, HY, LA",
            "n_regions = 7   # Q_7P: CA1sp, HPF, BLA, sAMY, HY, LA, PAL",
            "n_regions count"
        ),
        # region_order in ihara_radius
        (
            "region_order = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA]",
            "region_order = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]",
            "region_order in ihara_radius"
        ),
        # colour map — add PAL
        (
            '"CA1sp" => "red", "HPF" => "blue", "BLA" => "green",\n        "sAMY" => "purple", "HY" => "orange", "LA" => "cyan"',
            '"CA1sp" => "red", "HPF" => "blue", "BLA" => "green",\n        "sAMY" => "purple", "HY" => "orange", "LA" => "cyan", "PAL" => "magenta"',
            "colour map"
        ),
    ]
    return patch_file(filepath, patches, "curved_hh2")


def patch_ihara_bridge(filepath):
    patches = [
        (
            "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :sAMY]",
            "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :PAL, :sAMY]",
            "REGIONS const"
        ),
    ]
    return patch_file(filepath, patches, "IharaAssociahedronBridgeV2")


def patch_online_assoc(filepath):
    patches = [
        (
            "const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:sAMY]",
            "const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:PAL,:sAMY]",
            "REGIONS const"
        ),
        (
            """\
        :CA1sp => [:HPF,:sAMY],
        :BLA   => [:sAMY,:LA],
        :HY    => [:sAMY],
        :HPF   => [:CA1sp,:BLA,:sAMY],
        :sAMY  => [:CA1sp,:BLA,:HY,:HPF,:LA],
        :LA    => [:BLA,:sAMY]""",
            """\
        :CA1sp => [:HPF,:sAMY],
        :BLA   => [:sAMY,:LA],
        :HY    => [:sAMY,:PAL],
        :HPF   => [:CA1sp,:BLA,:sAMY],
        :sAMY  => [:CA1sp,:BLA,:HY,:HPF,:LA,:PAL],
        :LA    => [:BLA,:sAMY],
        :PAL   => [:HY,:sAMY]""",
            "ADJACENCY dict"
        ),
    ]
    return patch_file(filepath, patches, "OnlineAssociahedronNavigatorV3")


def patch_rees(filepath):
    patches = [
        (
            "const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA]",
            "const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA,:PAL]",
            "REGIONS const"
        ),
    ]
    return patch_file(filepath, patches, "ReesBlowupHybrid")


def patch_schober(filepath):
    patches = [
        (
            "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :sAMY] # matches regions_six.csv file",
            "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :PAL, :sAMY] # matches regions_six_ANDPAL.csv",
            "REGIONS const"
        ),
    ]
    return patch_file(filepath, patches, "SchobarNavigatorV2")


def patch_singularity(filepath):
    patches = [
        (
            "const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:sAMY]",
            "const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:PAL,:sAMY]",
            "REGIONS const"
        ),
        (
            "    :CA1sp => 1, :BLA => 2, :HY => 3, :HPF => 4,\n        :sAMY => 5, :LA => 6",
            "    :CA1sp => 1, :BLA => 2, :HY => 3, :HPF => 4,\n        :sAMY => 5, :LA => 6, :PAL => 7",
            "region index map"
        ),
        (
            'yticks = (0:6, ["Unk","CA1sp","BLA","HY","HPF","sAMY","LA"])',
            'yticks = (0:7, ["Unk","CA1sp","BLA","HY","HPF","sAMY","LA","PAL"])',
            "yticks labels"
        ),
    ]
    return patch_file(filepath, patches, "SingularityTrackerV2")


def patch_iharasing(filepath):
    patches = [
        (
            "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :sAMY]",
            "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :PAL, :sAMY]",
            "REGIONS const"
        ),
    ]
    return patch_file(filepath, patches, "run_iharaSingV2")


# ── Run all patches ───────────────────────────────────────────────────────────

print("Updating Julia files to Q_7P region definitions")
print(f"New regions: {NEW_REGIONS_SORTED}")
print()

total = 0
total += patch_curved_hh2(   os.path.join(BASEDIR, "curved_hh2_sparse_refactored.jl"))
total += patch_ihara_bridge(  os.path.join(BASEDIR, "IharaAssociahedronBridgeV2.jl"))
total += patch_online_assoc(  os.path.join(BASEDIR, "OnlineAssociahedronNavigatorV3.jl"))
total += patch_rees(          os.path.join(BASEDIR, "ReesBlowupHybrid.jl"))
total += patch_schober(       os.path.join(BASEDIR, "SchobarNavigatorV2.jl"))
total += patch_singularity(   os.path.join(BASEDIR, "SingularityTrackerV2.jl"))
total += patch_iharasing(     os.path.join(BASEDIR, "run_iharaSingV2_MonoTwist.jl"))

print()
print(f"Total patches applied: {total}")
print()

# ── Verify ───────────────────────────────────────────────────────────────────

print("Verification:")
for fname in FILES:
    fpath = os.path.join(BASEDIR, fname)
    if not os.path.exists(fpath):
        continue
    with open(fpath) as f:
        content = f.read()
    has_pal = ":PAL" in content
    has_old_6 = "[:BLA, :CA1sp, :HPF, :HY, :LA, :sAMY]" in content or \
                "[:BLA,:CA1sp,:HPF,:HY,:LA,:sAMY]" in content or \
                "[:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA]" in content or \
                "[:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA]" in content
    status = "OK" if has_pal and not has_old_6 else \
             "PAL MISSING" if not has_pal else \
             "OLD 6-REGION DEF REMAINS"
    print(f"  {fname:<45} {status}")

print()
print("Done. Run with different graph type by editing GRAPH_TYPE in BALBc_Opiate_Norcain.py")
print("or by running the equivalent LSX/PAL+LSX scripts.")
