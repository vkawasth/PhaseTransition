// ============================================================================
// boundary_obstruction_v3.m
// CORRECT approach: compute boundary obstruction space from mult[][] table,
// not from matrix algebra B.
//
// PASTE THIS AT THE END of connectome_algebra_6_PAL.m and run.
// It uses mult[][] which is already built by that file's Section 1.
//
// The mult[][] table is non-associative (93 failing triples for n=7P).
// We restrict to boundary triples and compute rank of defect vectors.
// ============================================================================

print "";
print "======================================================";
print "Section 4B: Boundary Obstruction Space  dim(O_partial)";
print "======================================================";
print "";

// ── Basis layout for connectome_algebra_6_PAL.m (n=25) ────────────────────
// Idempotents: 1=CA1sp  2=BLA  3=HY  4=HPF  5=sAMY  6=LA  7=PAL
// Arrows:
//  8=f_CA1sp_HPF   9=f_HPF_CA1sp  10=f_BLA_sAMY  11=f_sAMY_BLA
// 12=f_HPF_sAMY  13=f_sAMY_HPF  14=f_HY_sAMY   15=f_sAMY_HY
// 16=f_sAMY_LA   17=f_LA_sAMY   18=f_BLA_LA    19=f_LA_BLA
// 20=f_CA1sp_sAMY 21=f_HPF_BLA
// 22=f_HY_PAL    23=f_PAL_HY    24=f_PAL_sAMY  25=f_sAMY_PAL

// Boundary nodes = nodes on the trinion ∂Σ_Q or stop edges:
// BLA(2), HY(3), sAMY(5), LA(6), PAL(7)
// "Boundary basis element" = idempotent or arrow whose src OR tgt is boundary

// Arrows with source or target in {2,3,5,6,7}:
//  10: BLA->sAMY   [2,5] ✓   11: sAMY->BLA  [5,2] ✓
//  12: HPF->sAMY   [4,5] ✓   13: sAMY->HPF  [5,4] ✓
//  14: HY->sAMY    [3,5] ✓   15: sAMY->HY   [5,3] ✓
//  16: sAMY->LA    [5,6] ✓   17: LA->sAMY   [6,5] ✓
//  18: BLA->LA     [2,6] ✓   19: LA->BLA    [6,2] ✓
//  20: CA1sp->sAMY [1,5] ✓   21: HPF->BLA   [4,2] ✓
//  22: HY->PAL     [3,7] ✓   23: PAL->HY    [7,3] ✓
//  24: PAL->sAMY   [7,5] ✓   25: sAMY->PAL  [5,7] ✓
//  (8: CA1sp->HPF [1,4] - neither boundary, excluded)
//  (9: HPF->CA1sp [4,1] - neither boundary, excluded)

boundary_idx := [2, 3, 5, 6, 7,          // vertex idempotents
                 10, 11, 12, 13, 14, 15,  // sAMY-hub arrows
                 16, 17, 18, 19,           // LA-hub arrows
                 20, 21,                   // CA1sp->sAMY, HPF->BLA
                 22, 23, 24, 25];          // PAL arrows

printf "Boundary basis indices (%o elements):\n  %o\n\n",
    #boundary_idx, boundary_idx;

// ── Compute alpha(a,b,c) for boundary triples from mult table ───────────────
print "Computing alpha(a,b,c) = (ab)c - a(bc) using mult[][] ...";

defect_vecs := [];
total := 0;
nonzero := 0;

for ai in boundary_idx do
    for bi in boundary_idx do
        for ci in boundary_idx do
            total +:= 1;

            // (ai*bi)*ci
            lhs := [k!0 : x in [1..n]];
            for eij in mult[ai][bi] do
                cij := eij[1];
                kij := Integers()!eij[2];
                for ekl in mult[kij][ci] do
                    lhs[Integers()!ekl[2]] +:= cij * ekl[1];
                end for;
            end for;

            // ai*(bi*ci)
            rhs := [k!0 : x in [1..n]];
            for ejl in mult[bi][ci] do
                cjl := ejl[1];
                kjl := Integers()!ejl[2];
                for eikjl in mult[ai][kjl] do
                    rhs[Integers()!eikjl[2]] +:= cjl * eikjl[1];
                end for;
            end for;

            defect := [lhs[x] - rhs[x] : x in [1..n]];
            if &or[defect[x] ne 0 : x in [1..n]] then
                nonzero +:= 1;
                Append(~defect_vecs, defect);
            end if;
        end for;
    end for;
end for;

printf "Boundary triples tested:    %o\n", total;
printf "Nonzero defect triples:     %o\n", nonzero;

// ── Rank computation ────────────────────────────────────────────────────────
if #defect_vecs eq 0 then
    print "dim(O_partial) = 0  (no defects on boundary)";
else
    V   := VectorSpace(k, n);
    OS  := sub< V | [V ! d : d in defect_vecs] >;
    dim := Dimension(OS);

    printf "\n=== RESULT: dim(O_partial) = %o ===\n\n", dim;

    if dim eq 4 then
        print "CONFIRMED: dim(O_partial) = 4";
        print "The boundary associator defect space has exactly 4 dimensions.";
        print "Gr(2,4) embedding theorem is justified.";
    else
        printf "dim(O_partial) = %o  (expected 4)\n", dim;
        print "Check boundary node selection or basis indexing.";
    end if;

    // Show the basis
    print "";
    print "Basis vectors of O_partial (support only):";
    bas := Basis(OS);
    for idx in [1..#bas] do
        v := bas[idx];
        support := [<j, v[j]> : j in [1..n] | v[j] ne 0];
        printf "  v[%o]: %o\n", idx, support;
    end for;
end if;
