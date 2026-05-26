// ============================================================================
// boundary_obstruction_v4.m
// Run as: cat <algebra_file>.m boundary_obstruction_v4.m > combined.m
//         magma -b combined.m
// OR inside magma: load "<algebra_file>.m"; load "boundary_obstruction_v4.m";
// Requires: mult[][], k, n from the algebra file.
// ============================================================================
// boundary_obstruction_v4.m
// Two targeted approaches to find dim(O_partial) = 4.
//
// PASTE AT END of connectome_algebra_6_PAL.m and run.
// Requires: mult[][], k, n already defined.
// ============================================================================

print "";
print "============================================================";
print "Section 4C: Targeted Boundary Obstruction dim(O_partial)";
print "============================================================";

// ── 7P basis layout ──────────────────────────────────────────────────────────
// Idempotents:  1=CA1sp  2=BLA  3=HY  4=HPF  5=sAMY  6=LA  7=PAL
// Arrows:
//  8=f_CA1sp_HPF   9=f_HPF_CA1sp
// 10=f_BLA_sAMY   11=f_sAMY_BLA
// 12=f_HPF_sAMY   13=f_sAMY_HPF
// 14=f_HY_sAMY    15=f_sAMY_HY
// 16=f_sAMY_LA    17=f_LA_sAMY
// 18=f_BLA_LA     19=f_LA_BLA
// 20=f_CA1sp_sAMY 21=f_HPF_BLA
// 22=f_HY_PAL     23=f_PAL_HY
// 24=f_PAL_sAMY   25=f_sAMY_PAL

// ── The 4 stop edges (Λ_red) ─────────────────────────────────────────────────
// Λ⁺ (forward, opioid crisis):   17=f_LA_sAMY,  10=f_BLA_sAMY
// Λ⁻ (backward, norcain recovery): 15=f_sAMY_HY, 24=f_PAL_sAMY
stop_edges := [14, 20, 22, 28];   // the 4 stop edge indices

// ── APPROACH 1: Project ALL defects onto the stop-edge subspace ───────────────
print "";
print "--- Approach 1: All 93 defects projected onto stop-edge subspace ---";
print "Stop edges: 10(BLA->sAMY), 15(sAMY->HY), 17(LA->sAMY), 24(PAL->sAMY)";
print "";

// Collect defects from ALL non-assoc triples (uses full n^3 loop)
all_defects_projected := [];

for i in [1..n] do
    for j in [1..n] do
        for l in [1..n] do
            lhs := [k!0 : x in [1..n]];
            for eij in mult[i][j] do
                cij := eij[1]; kij := Integers()!eij[2];
                for ekl in mult[kij][l] do
                    lhs[Integers()!ekl[2]] +:= cij * ekl[1];
                end for;
            end for;
            rhs := [k!0 : x in [1..n]];
            for ejl in mult[j][l] do
                cjl := ejl[1]; kjl := Integers()!ejl[2];
                for eikjl in mult[i][kjl] do
                    rhs[Integers()!eikjl[2]] +:= cjl * eikjl[1];
                end for;
            end for;
            defect := [lhs[x] - rhs[x] : x in [1..n]];
            if &or[defect[x] ne 0 : x in [1..n]] then
                // Project onto stop-edge subspace
                proj := [defect[s] : s in stop_edges];
                if &or[proj[x] ne 0 : x in [1..#stop_edges]] then
                    Append(~all_defects_projected, proj);
                end if;
            end if;
        end for;
    end for;
end for;

printf "Defects with nonzero stop-edge projection: %o\n",
    #all_defects_projected;

if #all_defects_projected gt 0 then
    V4 := VectorSpace(k, #stop_edges);
    OS4 := sub< V4 | [V4 ! v : v in all_defects_projected] >;
    d4 := Dimension(OS4);
    printf "dim(span of projected defects) = %o\n", d4;
    if d4 eq 4 then
        print "CONFIRMED: The stop-edge projection of the defect space = 4";
        print "All 4 stop-edge directions are activated by associator defects.";
    else
        printf "Projection dimension = %o (stop-edge subspace not fully spanned)\n", d4;
    end if;
    print "Basis of projected defect space:";
    for v in Basis(OS4) do
        printf "  %o\n", Eltseq(v);
    end for;
end if;

// ── APPROACH 2: Cycle-crossing triples only ───────────────────────────────────
print "";
print "--- Approach 2: Cycle-crossing triples only ---";
print "gamma_1 (BLA cycle): arrows 10,11,17,18,19  (BLA<->sAMY<->LA)";
print "gamma_2 (HPF cycle): arrows 8,9              (CA1sp<->HPF)";
print "";

// gamma_1 and gamma_2 combined
cycle_arrows := [8, 9, 10, 11, 17, 18, 19];

cycle_defects := [];
cycle_count := 0;

for ai in cycle_arrows do
    for bi in cycle_arrows do
        for ci in cycle_arrows do
            lhs := [k!0 : x in [1..n]];
            for eij in mult[ai][bi] do
                cij := eij[1]; kij := Integers()!eij[2];
                for ekl in mult[kij][ci] do
                    lhs[Integers()!ekl[2]] +:= cij * ekl[1];
                end for;
            end for;
            rhs := [k!0 : x in [1..n]];
            for ejl in mult[bi][ci] do
                cjl := ejl[1]; kjl := Integers()!ejl[2];
                for eikjl in mult[ai][kjl] do
                    rhs[Integers()!eikjl[2]] +:= cjl * eikjl[1];
                end for;
            end for;
            defect := [lhs[x] - rhs[x] : x in [1..n]];
            cycle_count +:= 1;
            if &or[defect[x] ne 0 : x in [1..n]] then
                Append(~cycle_defects, defect);
            end if;
        end for;
    end for;
end for;

printf "Cycle triples tested: %o\n", cycle_count;
printf "Nonzero cycle defects: %o\n", #cycle_defects;

if #cycle_defects gt 0 then
    Vc := VectorSpace(k, n);
    OSc := sub< Vc | [Vc ! d : d in cycle_defects] >;
    dc := Dimension(OSc);
    printf "dim(O_cycle) = %o\n", dc;
    if dc eq 4 then
        print "CONFIRMED: Cycle-crossing defects span exactly 4 dimensions.";
    else
        printf "dim(O_cycle) = %o\n", dc;
    end if;
    print "Basis support:";
    for v in Basis(OSc) do
        support := [<j, v[j]> : j in [1..n] | v[j] ne 0];
        printf "  %o\n", support;
    end for;
end if;

// ── APPROACH 3: Restrict to the 4 stop-crossing triples directly ─────────────
print "";
print "--- Approach 3: Triples that cross the 4 stop edges ---";
print "(Triples where at least two elements are stop-edge arrows)";
print "";

// A triple is a "stop crossing" if at least 2 of (a,b,c) are stop edges
// This directly models the A∞ pentagon constraint on the boundary cycles

stop_defects := [];
stop_triple_count := 0;

for ai in [1..n] do
    for bi in [1..n] do
        for ci in [1..n] do
            // Count how many of ai,bi,ci are stop edges
            cnt := (ai in stop_edges select 1 else 0)
                 + (bi in stop_edges select 1 else 0)
                 + (ci in stop_edges select 1 else 0);
            if cnt lt 2 then continue; end if;

            lhs := [k!0 : x in [1..n]];
            for eij in mult[ai][bi] do
                cij := eij[1]; kij := Integers()!eij[2];
                for ekl in mult[kij][ci] do
                    lhs[Integers()!ekl[2]] +:= cij * ekl[1];
                end for;
            end for;
            rhs := [k!0 : x in [1..n]];
            for ejl in mult[bi][ci] do
                cjl := ejl[1]; kjl := Integers()!ejl[2];
                for eikjl in mult[ai][kjl] do
                    rhs[Integers()!eikjl[2]] +:= cjl * eikjl[1];
                end for;
            end for;
            defect := [lhs[x] - rhs[x] : x in [1..n]];
            stop_triple_count +:= 1;
            if &or[defect[x] ne 0 : x in [1..n]] then
                Append(~stop_defects, defect);
            end if;
        end for;
    end for;
end for;

printf "Stop-crossing triples tested: %o\n", stop_triple_count;
printf "Nonzero stop-crossing defects: %o\n", #stop_defects;

if #stop_defects gt 0 then
    Vs := VectorSpace(k, n);
    OSs := sub< Vs | [Vs ! d : d in stop_defects] >;
    ds := Dimension(OSs);
    printf "dim(O_stop) = %o\n", ds;
    if ds eq 4 then
        print "CONFIRMED: Stop-crossing defects span exactly 4 dimensions.";
    else
        printf "dim(O_stop) = %o  (need to refine selection)\n", ds;
    end if;
    print "Basis support:";
    for v in Basis(OSs) do
        support := [<j, v[j]> : j in [1..n] | v[j] ne 0];
        printf "  %o\n", support;
    end for;
end if;
