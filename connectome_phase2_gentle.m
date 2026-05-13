/*
==============================================================
Phase 2: Gentle Algebra Check
MAGMA program: connectome_phase2_gentle.m

Checks whether A_bound (or its core subquiver) is a gentle algebra.

A gentle algebra kQ/I requires at each vertex v:
  (G1) At most 2 arrows ENTERING v participate in nonzero
       length-2 paths (i.e., are NOT in the start of a relation)
  (G2) At most 2 arrows LEAVING v participate in nonzero
       length-2 paths (i.e., are NOT the end of a relation)
  (G3) For each arrow a→b, at most one arrow b→c with
       a·c NOT in I  (at most one admissible continuation)
  (G4) For each arrow a→b, at most one arrow c→a with
       c·b NOT in I  (at most one admissible predecessor)

For A_bound = kQ/<f_ij·f_ji>, the relation ideal I consists
of all round-trip paths. So:
  - f_ij·f_ji ∈ I  for all symmetric pairs
  - all other length-2 paths are NOT in I

Gentleness condition for A_bound:
  At vertex v with symmetric neighbors u1,...,uk and
  (possibly) asymmetric incoming arrows:
  
  G1/G2: At most 2 arrows at v can participate in
         admissible (nonbacktracking) length-2 paths.
  
  For sAMY (degree 4+ symmetric): this likely FAILS
  for the full quiver but may hold for the CORE.

The program:
  1. Checks full A_bound for gentleness (expected to fail at sAMY)
  2. Identifies which vertices violate gentleness
  3. Checks the CORE subquiver (remove degree-1 vertices)
  4. Reports whether the core is gentle
==============================================================
*/

k := RationalField();

print "==============================================";
print "Phase 2: Gentle Algebra Check";
print "==============================================";
print "";

// ── Quiver data (Q_6 base, 14 arrows) ─────────────────────────────────────

n_v := 6;
vertex_names := ["CA1sp","BLA","HY","HPF","sAMY","LA"];

arrows := [
    [1, 1, 4],   // f14: CA1sp->HPF
    [2, 4, 1],   // f41: HPF->CA1sp
    [3, 2, 5],   // f25: BLA->sAMY
    [4, 5, 2],   // f52: sAMY->BLA
    [5, 4, 5],   // f45: HPF->sAMY
    [6, 5, 4],   // f54: sAMY->HPF
    [7, 3, 5],   // f35: HY->sAMY
    [8, 5, 3],   // f53: sAMY->HY
    [9, 5, 6],   // f56: sAMY->LA
    [10, 6, 5],  // f65: LA->sAMY
    [11, 2, 6],  // f26: BLA->LA
    [12, 6, 2],  // f62: LA->BLA
    [13, 1, 5],  // f15: CA1sp->sAMY  ASYMMETRIC
    [14, 4, 2]   // f42: HPF->BLA     ASYMMETRIC
];
n_arr := #arrows;

// Reversal map
rev := [0 : i in [1..n_arr]];
for i in [1..n_arr] do
    for j in [1..n_arr] do
        if i ne j and arrows[i][2] eq arrows[j][3]
                  and arrows[i][3] eq arrows[j][2] then
            rev[i] := j;
        end if;
    end for;
end for;

// Relation ideal: round-trip paths f_ij*f_ji
// A length-2 path (i,j) is in I iff j = rev[i]
function in_ideal(i, j, rev_map)
    return rev_map[i] ne 0 and rev_map[i] eq j;
end function;

// Admissible length-2 paths: composable and NOT in ideal
function is_admissible_pair(i, j, arrows, rev_map)
    // composable
    if arrows[i][3] ne arrows[j][2] then return false; end if;
    // not a round-trip relation
    if in_ideal(i, j, rev_map) then return false; end if;
    return true;
end function;

// ── Gentleness check ────────────────────────────────────────────────────────

print "--- Gentleness conditions for A_bound (Q_6) ---";
print "";
print "For each vertex v, check:";
print "  G1: <= 2 incoming arrows have admissible continuations";
print "  G2: <= 2 outgoing arrows have admissible predecessors";
print "  G3: Each arrow a->v has at most 1 admissible continuation v->w";
print "  G4: Each arrow v->b has at most 1 admissible predecessor u->v";
print "";

all_gentle := true;
gentle_violations := [];

for v in [1..n_v] do
    in_arrows  := [i : i in [1..n_arr] | arrows[i][3] eq v];
    out_arrows := [i : i in [1..n_arr] | arrows[i][2] eq v];
    
    // G1: count incoming arrows that have at least one admissible continuation
    in_with_continuation := [];
    for a in in_arrows do
        conts := [b : b in out_arrows | is_admissible_pair(a, b, arrows, rev)];
        if #conts gt 0 then
            Append(~in_with_continuation, a);
        end if;
    end for;
    g1_ok := #in_with_continuation le 2;
    
    // G2: count outgoing arrows that have at least one admissible predecessor
    out_with_predecessor := [];
    for b in out_arrows do
        preds := [a : a in in_arrows | is_admissible_pair(a, b, arrows, rev)];
        if #preds gt 0 then
            Append(~out_with_predecessor, b);
        end if;
    end for;
    g2_ok := #out_with_predecessor le 2;
    
    // G3: for each incoming arrow a, count admissible continuations
    g3_ok := true;
    for a in in_arrows do
        conts := [b : b in out_arrows | is_admissible_pair(a, b, arrows, rev)];
        if #conts gt 1 then
            g3_ok := false;
        end if;
    end for;
    
    // G4: for each outgoing arrow b, count admissible predecessors
    g4_ok := true;
    for b in out_arrows do
        preds := [a : a in in_arrows | is_admissible_pair(a, b, arrows, rev)];
        if #preds gt 1 then
            g4_ok := false;
        end if;
    end for;
    
    v_gentle := g1_ok and g2_ok and g3_ok and g4_ok;
    if not v_gentle then
        all_gentle := false;
        Append(~gentle_violations, v);
    end if;
    
    printf "  v=%o (%o): in=%o out=%o\n",
           v, vertex_names[v], #in_arrows, #out_arrows;
    printf "    in_with_cont=%o, G1(%o<=2): %o\n",
           #in_with_continuation, #in_with_continuation,
           g1_ok select "OK" else "FAIL";
    printf "    out_with_pred=%o, G2(%o<=2): %o\n",
           #out_with_predecessor, #out_with_predecessor,
           g2_ok select "OK" else "FAIL";
    printf "    G3 (each in-arrow <= 1 cont): %o\n",
           g3_ok select "OK" else "FAIL";
    printf "    G4 (each out-arrow <= 1 pred): %o\n",
           g4_ok select "OK" else "FAIL";
    printf "    Gentle at %o: %o\n\n",
           vertex_names[v], v_gentle select "YES" else "NO";
end for;

printf "Full A_bound (Q_6) is gentle: %o\n",
       all_gentle select "YES" else "NO";
if #gentle_violations gt 0 then
    printf "Violations at: %o\n",
           [vertex_names[v] : v in gentle_violations];
end if;

// ── Core subquiver check ────────────────────────────────────────────────────

print "";
print "==============================================";
print "--- Core subquiver (remove degree-1 vertices) ---";
print "==============================================";
print "";

// Degree-1 vertices: CA1sp (connects only to HPF via sym pair)
// and HY (connects only to sAMY via sym pair)
// These are the peripheral leaf nodes

// Compute symmetric degree of each vertex
sym_deg := [0 : v in [1..n_v]];
for i in [1..n_arr] do
    if rev[i] gt 0 then  // symmetric arrow
        sym_deg[arrows[i][2]] +:= 1;
    end if;
end for;

printf "Symmetric degrees:\n";
for v in [1..n_v] do
    printf "  %o (%o): sym_deg=%o\n", v, vertex_names[v], sym_deg[v];
end for;

// Core = vertices with sym_deg >= 2
core_vertices := [v : v in [1..n_v] | sym_deg[v] ge 2];
printf "\nCore vertices (sym_deg >= 2): %o\n",
       [vertex_names[v] : v in core_vertices];

leaf_vertices := [v : v in [1..n_v] | sym_deg[v] eq 1];
printf "Leaf vertices (sym_deg = 1, peripheral): %o\n",
       [vertex_names[v] : v in leaf_vertices];

// Core arrows: both endpoints in core
core_arrows := [i : i in [1..n_arr] |
    arrows[i][2] in core_vertices and
    arrows[i][3] in core_vertices];

printf "\nCore arrows (%o):\n", #core_arrows;
for i in core_arrows do
    r := rev[i];
    rev_str := r gt 0 select Sprintf("rev=arr%o", r) else "ASYMMETRIC";
    printf "  arr%o: %o->%o  [%o]\n",
           i, vertex_names[arrows[i][2]],
           vertex_names[arrows[i][3]], rev_str;
end for;

// Check gentleness of core
print "";
print "Gentleness check for CORE subquiver:";
print "";

core_gentle := true;
core_violations := [];

for v in core_vertices do
    in_core  := [i : i in core_arrows | arrows[i][3] eq v];
    out_core := [i : i in core_arrows | arrows[i][2] eq v];
    
    in_with_cont := [];
    for a in in_core do
        conts := [b : b in out_core | is_admissible_pair(a, b, arrows, rev)];
        if #conts gt 0 then Append(~in_with_cont, a); end if;
    end for;
    
    out_with_pred := [];
    for b in out_core do
        preds := [a : a in in_core | is_admissible_pair(a, b, arrows, rev)];
        if #preds gt 0 then Append(~out_with_pred, b); end if;
    end for;
    
    g1 := #in_with_cont le 2;
    g2 := #out_with_pred le 2;
    
    g3 := true;
    for a in in_core do
        conts := [b : b in out_core | is_admissible_pair(a, b, arrows, rev)];
        if #conts gt 1 then g3 := false; end if;
    end for;
    
    g4 := true;
    for b in out_core do
        preds := [a : a in in_core | is_admissible_pair(a, b, arrows, rev)];
        if #preds gt 1 then g4 := false; end if;
    end for;
    
    v_gentle := g1 and g2 and g3 and g4;
    if not v_gentle then
        core_gentle := false;
        Append(~core_violations, v);
    end if;
    
    printf "  %o (%o): in=%o out=%o\n",
           v, vertex_names[v], #in_core, #out_core;
    printf "    G1(%o<=2):%o  G2(%o<=2):%o  G3:%o  G4:%o  => %o\n\n",
           #in_with_cont, g1 select "OK" else "FAIL",
           #out_with_pred, g2 select "OK" else "FAIL",
           g3 select "OK" else "FAIL",
           g4 select "OK" else "FAIL",
           v_gentle select "GENTLE" else "NOT GENTLE";
end for;

printf "Core subquiver is gentle: %o\n",
       core_gentle select "YES" else "NO";
if #core_violations gt 0 then
    printf "Core violations at: %o\n",
           [vertex_names[v] : v in core_violations];
end if;

// ── Summary ─────────────────────────────────────────────────────────────────

print "";
print "==============================================";
print "PHASE 2 SUMMARY";
print "==============================================";
print "";
printf "Full A_bound gentle:  %o\n",
       all_gentle select "YES" else "NO";
printf "Core subquiver gentle: %o\n",
       core_gentle select "YES" else "NO";
print "";
if core_gentle then
    print "CONSEQUENCE:";
    print "  By HKK (Haiden-Katzarkov-Kontsevich 2017):";
    print "  D^b(A_core) ≃ W(Σ_core, Λ_red)";
    print "  where Σ_core is the surface from Phase 1.";
    print "";
    print "  The core spectrum determines ρ(B_Ihara)";
    print "  (peripheral vertices don't affect the biconnected core).";
    print "";
    print "  Phase 2 COMPLETE for core subquiver.";
    print "  Proceed to Phase 3 (K_0 identification).";
else
    print "Core is NOT gentle.";
    print "Identify which relations need adjustment.";
    print "Check: is the core of CORE (remove next shell) gentle?";
end if;

print "";
print "==============================================";
print "DONE -- connectome_phase2_gentle.m";
print "==============================================";
