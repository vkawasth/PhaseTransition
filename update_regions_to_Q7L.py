"""
update_regions_to_Q7L.py  —  Q_7L (LSX graph)
New regions: ['BLA', 'CA1sp', 'HPF', 'HY', 'LA', 'LSX', 'sAMY']
LSX connects only to HPF (peripheral leaf, degree 1).
"""
import re, os

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

PATCHES = {
    "curved_hh2_sparse_refactored.jl": [
        ("const nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]",
         "const nodes = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :LSX]",
         "nodes const"),
        ('region_to_idx = Dict("BLA"=>0, "CA1sp"=>1, "HPF"=>2, "HY"=>3, "LA"=>4, "PAL"=>5, "sAMY"=>6)',
         'region_to_idx = Dict("BLA"=>0, "CA1sp"=>1, "HPF"=>2, "HY"=>3, "LA"=>4, "LSX"=>5, "sAMY"=>6)',
         "region_to_idx dict"),
        ("node_symbols = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]",
         "node_symbols = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :LSX]",
         "node_symbols"),
        ("n_regions = 7   # Q_7P: CA1sp, HPF, BLA, sAMY, HY, LA, PAL",
         "n_regions = 7   # Q_7L: CA1sp, HPF, BLA, sAMY, HY, LA, LSX",
         "n_regions comment"),
        ("region_order = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :PAL]",
         "region_order = [:CA1sp, :HPF, :BLA, :sAMY, :HY, :LA, :LSX]",
         "region_order"),
        ('"PAL" => "magenta"', '"LSX" => "grey"', "colour map PAL→LSX"),
    ],
    "IharaAssociahedronBridgeV2.jl": [
        ("const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :PAL, :sAMY]",
         "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :sAMY]",
         "REGIONS const"),
    ],
    "OnlineAssociahedronNavigatorV3.jl": [
        ("const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:PAL,:sAMY]",
         "const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:LSX,:sAMY]",
         "REGIONS const"),
        # HPF gains LSX as neighbor; PAL entries removed
        ("        :HPF   => [:CA1sp,:BLA,:sAMY],",
         "        :HPF   => [:CA1sp,:BLA,:sAMY,:LSX],",
         "HPF adjacency +LSX"),
        ("        :HY    => [:sAMY,:PAL],",
         "        :HY    => [:sAMY],",
         "HY adjacency -PAL"),
        ("        :sAMY  => [:CA1sp,:BLA,:HY,:HPF,:LA,:PAL],",
         "        :sAMY  => [:CA1sp,:BLA,:HY,:HPF,:LA],",
         "sAMY adjacency -PAL"),
        ("        :PAL   => [:HY,:sAMY]",
         "        :LSX   => [:HPF]",
         "PAL→LSX entry"),
    ],
    "ReesBlowupHybrid.jl": [
        ("const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA,:PAL]",
         "const REGIONS = [:CA1sp,:BLA,:HY,:HPF,:sAMY,:LA,:LSX]",
         "REGIONS const"),
    ],
    "SchobarNavigatorV2.jl": [
        ("const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :PAL, :sAMY] # matches regions_six_ANDPAL.csv",
         "const REGIONS = [:BLA, :CA1sp, :HPF, :HY, :LA, :LSX, :sAMY] # matches regions_six_ANDLSX.csv",
         "REGIONS const"),
    ],
    "SingularityTrackerV2.jl": [
        ("const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:PAL,:sAMY]",
         "const REGIONS = [:BLA,:CA1sp,:HPF,:HY,:LA,:LSX,:sAMY]",
         "REGIONS const"),
        (":sAMY => 5, :LA => 6, :PAL => 7",
         ":sAMY => 5, :LA => 6, :LSX => 7",
         "region index map"),
        ('(0:7, ["Unk","CA1sp","BLA","HY","HPF","sAMY","LA","PAL"])',
         '(0:7, ["Unk","CA1sp","BLA","HY","HPF","sAMY","LA","LSX"])',
         "yticks labels"),
    ],
    "run_iharaSingV2_MonoTwist.jl": [
        ("const REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :PAL]",
         "const REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :LSX]",
         "REGIONS const (top)"),
        ("    REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :PAL]",
         "    REGIONS = [:CA1sp, :BLA, :HY, :HPF, :sAMY, :LA, :LSX]",
         "REGIONS local var"),
    ],
}

total = 0
for fname, patches in PATCHES.items():
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath):
        print(f"  SKIP: {fname}"); continue
    with open(fpath) as f: content = f.read()
    for old, new, desc in patches:
        n = content.count(old)
        if n: content = content.replace(old, new); total += n; print(f"  OK ({n}x): [{fname}] {desc}")
        else: print(f"  NOT FOUND: [{fname}] {desc}")
    with open(fpath, 'w') as f: f.write(content)

print(f"\nTotal patches: {total}")
print("\nVerification:")
for fname in FILES:
    fpath = os.path.join(BASE, fname)
    if not os.path.exists(fpath): continue
    with open(fpath) as f: content = f.read()
    defs = re.findall(r'(?:const\s+)?(?:REGIONS|nodes)\s*=\s*\[[^\]]+\]', content)
    miss = [d for d in defs if 'LSX' not in d and 'PAL' not in d and
            any(r in d for r in ['CA1sp','BLA','sAMY'])]
    print(f"  {fname:<45} {'OK' if not miss else 'WARN: '+str(miss[:1])}")
