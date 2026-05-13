/*
Phase 2 v2: Gentleness analysis with three sub-checks:
  (A) Core with asymmetric arrows included (already failed)
  (B) Core with asymmetric arrows EXCLUDED (symmetric core only)
  (C) Core-of-core: remove BLA (sym_deg=2 but highest violation)
      keep only {HPF, sAMY, LA} = the hub triangle
  (D) Check if any subquiver is gentle and identify it precisely

Also reports: for each failing vertex, exactly which pairs
(a, b) violate G3/G4, so we can see the minimal obstruction.
*/

k := RationalField();

print "==============================================";
print "Phase 2 v2: Gentleness — sub-case analysis";
print "==============================================";
print "";

vertex_names := ["CA1sp","BLA","HY","HPF","sAMY","LA"];

arrows := [
    [1, 1, 4],   // f14
    [2, 4, 1],   // f41
    [3, 2, 5],   // f25 BLA->sAMY
    [4, 5, 2],   // f52 sAMY->BLA
    [5, 4, 5],   // f45 HPF->sAMY
    [6, 5, 4],   // f54 sAMY->HPF
    [7, 3, 5],   // f35 HY->sAMY
    [8, 5, 3],   // f53 sAMY->HY
    [9, 5, 6],   // f56 sAMY->LA
    [10, 6, 5],  // f65 LA->sAMY
    [11, 2, 6],  // f26 BLA->LA
    [12, 6, 2],  // f62 LA->BLA
    [13, 1, 5],  // f15 CA1sp->sAMY ASYM
    [14, 4, 2]   // f42 HPF->BLA    ASYM
];
n_arr := #arrows;

rev := [0 : i in [1..n_arr]];
for i in [1..n_arr] do
    for j in [1..n_arr] do
        if i ne j and arrows[i][2] eq arrows[j][3]
                  and arrows[i][3] eq arrows[j][2] then
            rev[i] := j;
        end if;
    end for;
end for;

function in_ideal(i, j, rev_map)
    return rev_map[i] ne 0 and rev_map[i] eq j;
end function;

function is_adm(i, j, arrows, rev_map)
    if arrows[i][3] ne arrows[j][2] then return false; end if;
    if in_ideal(i, j, rev_map) then return false; end if;
    return true;
end function;

procedure check_gentle(name, verts, arr_set, arrows, rev, vertex_names)
    printf "--- %o ---\n", name;
    printf "Vertices: %o\n", [vertex_names[v] : v in verts];
    printf "Arrows (%o):\n", #arr_set;
    for i in arr_set do
        asym := rev[i] eq 0 select " [ASYM]" else "";
        printf "  arr%o: %o->%o%o\n",
               i, vertex_names[arrows[i][2]],
               vertex_names[arrows[i][3]], asym;
    end for;
    print "";
    
    all_gentle := true;
    
    for v in verts do
        in_v  := [i : i in arr_set | arrows[i][3] eq v];
        out_v := [i : i in arr_set | arrows[i][2] eq v];
        
        in_cont := [a : a in in_v |
            #[b : b in out_v | is_adm(a,b,arrows,rev)] gt 0];
        out_pred := [b : b in out_v |
            #[a : a in in_v | is_adm(a,b,arrows,rev)] gt 0];
        
        g1 := #in_cont le 2;
        g2 := #out_pred le 2;
        
        g3 := true;
        g3_viols := [];
        for a in in_v do
            conts := [b : b in out_v | is_adm(a,b,arrows,rev)];
            if #conts gt 1 then
                g3 := false;
                Append(~g3_viols, <a, conts>);
            end if;
        end for;
        
        g4 := true;
        g4_viols := [];
        for b in out_v do
            preds := [a : a in in_v | is_adm(a,b,arrows,rev)];
            if #preds gt 1 then
                g4 := false;
                Append(~g4_viols, <b, preds>);
            end if;
        end for;
        
        v_gentle := g1 and g2 and g3 and g4;
        if not v_gentle then all_gentle := false; end if;
        
        printf "  %o: in=%o out=%o  G1:%o G2:%o G3:%o G4:%o => %o\n",
               vertex_names[v], #in_v, #out_v,
               g1 select "OK" else "FAIL",
               g2 select "OK" else "FAIL",
               g3 select "OK" else "FAIL",
               g4 select "OK" else "FAIL",
               v_gentle select "GENTLE" else "NOT GENTLE";
        
        if #g3_viols gt 0 then
            print "    G3 violations (arrow a has >1 continuations):";
            for viol in g3_viols do
                a := viol[1]; conts := viol[2];
                cont_names := [Sprintf("arr%o(%o->%o)",
                    c, vertex_names[arrows[c][2]],
                    vertex_names[arrows[c][3]]) : c in conts];
                printf "      arr%o(%o->%o) -> {%o}\n",
                       a, vertex_names[arrows[a][2]],
                       vertex_names[arrows[a][3]],
                       Join(cont_names, ", ");
            end for;
        end if;
        if #g4_viols gt 0 then
            print "    G4 violations (arrow b has >1 predecessors):";
            for viol in g4_viols do
                b := viol[1]; preds := viol[2];
                pred_names := [Sprintf("arr%o(%o->%o)",
                    p, vertex_names[arrows[p][2]],
                    vertex_names[arrows[p][3]]) : p in preds];
                printf "      arr%o(%o->%o) <- {%o}\n",
                       b, vertex_names[arrows[b][2]],
                       vertex_names[arrows[b][3]],
                       Join(pred_names, ", ");
            end for;
        end if;
    end for;
    
    printf "\n%o is gentle: %o\n\n",
           name, all_gentle select "YES" else "NO";
end procedure;

// Case A: Core {BLA,HPF,sAMY,LA} with all arrows including asymmetric
verts_A := [2,4,5,6];
arr_A := [i : i in [1..n_arr] |
    arrows[i][2] in verts_A and arrows[i][3] in verts_A];
check_gentle("A: Core {BLA,HPF,sAMY,LA} with asym f42",
    verts_A, arr_A, arrows, rev, vertex_names);

// Case B: Core {BLA,HPF,sAMY,LA} SYMMETRIC arrows only
arr_B := [i : i in arr_A | rev[i] ne 0];
check_gentle("B: Core {BLA,HPF,sAMY,LA} symmetric only (no f42)",
    verts_A, arr_B, arrows, rev, vertex_names);

// Case C: Hub triangle {HPF,sAMY,LA} only
verts_C := [4,5,6];
arr_C := [i : i in [1..n_arr] |
    arrows[i][2] in verts_C and arrows[i][3] in verts_C];
check_gentle("C: Hub triangle {HPF,sAMY,LA}",
    verts_C, arr_C, arrows, rev, vertex_names);

// Case D: Hub pair {sAMY,LA} 
verts_D := [5,6];
arr_D := [i : i in [1..n_arr] |
    arrows[i][2] in verts_D and arrows[i][3] in verts_D];
check_gentle("D: Pair {sAMY,LA}",
    verts_D, arr_D, arrows, rev, vertex_names);

// Case E: {BLA,sAMY,LA} — remove HPF
verts_E := [2,5,6];
arr_E := [i : i in [1..n_arr] |
    arrows[i][2] in verts_E and arrows[i][3] in verts_E
    and rev[i] ne 0];  // symmetric only
check_gentle("E: {BLA,sAMY,LA} symmetric only",
    verts_E, arr_E, arrows, rev, vertex_names);

// Case F: Single symmetric pair as sanity check
verts_F := [5,6];
arr_F := [9,10];  // sAMY<->LA only
check_gentle("F: Single pair {sAMY,LA} (sanity check)",
    verts_F, arr_F, arrows, rev, vertex_names);

// ── What would make sAMY gentle? ────────────────────────────────────────────

print "==============================================";
print "What would make sAMY gentle?";
print "==============================================";
print "";
print "sAMY has 4 symmetric neighbors in Q_6: BLA, HY, HPF, LA";
print "Gentle requires: at most 2 in each direction have admissible paths.";
print "";
print "The gentle condition on sAMY fails because:";
print "  After removing round-trips (f_ij*f_ji=0),";
print "  each incoming arrow f_X->sAMY has 3 admissible";
print "  continuations (the 3 other outgoing directions).";
print "";
print "Gentle is equivalent to: at each vertex, the";
print "  'nonbacktracking fan' has width at most 2.";
print "  sAMY has width 3+ -> violates gentle.";
print "";
print "Options to restore gentleness:";
print "  1. Reduce sAMY connectivity to degree 2 (too destructive)";
print "  2. Use a DIFFERENT algebraic model (not path algebra)";
print "  3. Use the SURFACE model directly (bypasses gentleness)";
print "  4. Use microlocal sheaves on the ribbon graph surface";
print "     (GPS framework: sheaves, not gentle algebras)";
print "";
print "CONCLUSION:";
print "  HKK (gentle algebra -> Fukaya category) does NOT apply";
print "  directly to A_bound or its core.";
print "  The correct framework is GPS microlocal sheaves:";
print "    Sh_{Lambda_red}(Sigma) ≃ W(Sigma, Lambda_red)";
print "  which does NOT require gentleness.";
print "  The surface from Phase 1 is still the correct object.";
print "  Phase 2 proceeds via GPS, not HKK.";
print "";
print "==============================================";
print "DONE -- connectome_phase2_gentle_v2.m";
print "==============================================";
