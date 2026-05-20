"""
update_regions_to_Q8.py  —  Q_8 (PAL+LSX graph)
New regions: ['BLA', 'CA1sp', 'HPF', 'HY', 'LA', 'LSX', 'PAL', 'sAMY']
LSX→HPF (peripheral), PAL→{HY,sAMY} (connector).
"""
import re, os

BASE = os.path.dirname(os.path.abspath(__file__))

PATCHES = {
    "curved_hh2_sparse_refactored.jl": [
        ("const nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :LSX]",
         "const nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :LSX, :PAL]",
         "nodes const"),
        ('region_to_idx = Dict("BLA"=>0, "CA1sp"=>1, "HPF"=>2, "HY"=>3, "LA"=>4, "LSX"=>5, "sAMY"=>6)',
         'region_to_idx = Dict("BLA"=>0, "CA1sp"=>1, "HPF"=>2, "HY"=>3, "LA"=>4, "LSX"=>5, "PAL"=>6, "sAMY"=>7)',
         "region_to_idx dict"),
        ("node_symbols = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :LSX]",
         "node_symbols = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :LSX, :PAL]",
         "node_symbols"),
        ("n_regions = 7   # Q_7L: CA1sp, HPF, BLA, sAMY, HY, LA, LSX",
         "n_regions = 8   # Q_8: CA1sp, HPF, BLA, sAMY, HY, LA, LSX, PAL",
         "n_regions"),
        ("region_order = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :LSX]",
         "region_order = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :LSX, :PAL]",
         "region_order"),
        ('"LSX" => "grey"',
         '"LSX" => "grey", "PAL" => "magenta"',
         "colour map +PAL"),
    ],
    "IharaAssociahedronBridgeV2.jl": [
        ("const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :sAMY]",
         "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :PAL, :sAMY]",
         "REGIONS const"),
    ],
    "OnlineAssociahedronNavigatorV3.jl": [
        ("const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:LSX,:sAMY]",
         "const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:LSX,:PAL,:sAMY]",
         "REGIONS const"),
        ("        :HY    => [:sAMY],",
         "        :HY    => [:sAMY,:PAL],",
         "HY adjacency +PAL"),
        ("        :sAMY  => [:CA1sp,:BLA,:HY,:HPF,:LA],",
         "        :sAMY  => [:CA1sp,:BLA,:HY,:HPF,:LA,:PAL],",
         "sAMY adjacency +PAL"),
        ("        :LSX   => [:HPF]",
         "        :LSX   => [:HPF],\n        :PAL   => [:HY,:sAMY]",
         "PAL entry added"),
    ],
    "ReesBlowupHybrid.jl": [
        ("const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA,:LSX]",
         "const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA,:LSX,:PAL]",
         "REGIONS const"),
    ],
    "SchobarNavigatorV2.jl": [
        ("const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :sAMY] # matches regions_six_ANDLSX.csv",
         "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :PAL, :sAMY] # matches regions_six_ANDPALLSX.csv",
         "REGIONS const"),
    ],
    "SingularityTrackerV2.jl": [
        ("const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:LSX,:sAMY]",
         "const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:LSX,:PAL,:sAMY]",
         "REGIONS const"),
        (":sAMY => 5, :LA => 6, :LSX => 7",
         ":sAMY => 5, :LA => 6, :LSX => 7, :PAL => 8",
         "region index map"),
        ('(0:7, ["Unk","CA1sp","BLA","HY","HPF","sAMY","LA","LSX"])',
         '(0:8, ["Unk","CA1sp","BLA","HY","HPF","sAMY","LA","LSX","PAL"])',
         "yticks labels"),
    ],
    "run_iharaSingV2_MonoTwist.jl": [
        ("const REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :LSX]",
         "const REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :LSX, :PAL]",
         "REGIONS const (top)"),
        ("    REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :LSX]",
         "    REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :LSX, :PAL]",
         "REGIONS local var"),
    ],
}

total = 0
for fname, patches in PATCHES.items():
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath): print(f"  SKIP: {fname}"); continue
    with open(fpath) as f: content = f.read()
    for old, new, desc in patches:
        n = content.count(old)
        if n: content = content.replace(old, new); total += n; print(f"  OK ({n}x): [{fname}] {desc}")
        else: print(f"  NOT FOUND: [{fname}] {desc}")
    with open(fpath, 'w') as f: f.write(content)

print(f"\nTotal patches: {total}")
