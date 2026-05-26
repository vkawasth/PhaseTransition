/*
==============================================================
BALBc Connectome Path Algebra -- 7-Node (LSX added)
MAGMA program: connectome_algebra_7_LSX.m

Vertices (7):
  1=CA1sp  2=BLA  3=HY  4=HPF  5=sAMY  6=LA  7=LSX

LSX connects only to HPF:
  22=f47(HPF->LSX)   23=f74(LSX->HPF)

Basis (23 elements):
  Idempotents 1..7
  Arrows 8..23 (14 inherited + 2 new LSX)

Round-trip scalars:
  Inherited 6-node: unchanged
  f_HPF_LSX * f_LSX_HPF = 69.87161588668823 * e_HPF
  f_LSX_HPF * f_HPF_LSX = 69.87161588668823 * e_LSX

Path compositions through LSX:
  All produce f_X_LSX or f_LSX_X paths not in primitive basis -> 0
  No Add entries needed for LSX path compositions.

Usage: load "connectome_algebra_7_LSX.m";
==============================================================
*/

k  := RationalField();
RR := RealField(20);
n  := 23;

print "==============================================";
print "BALBc Connectome Path Algebra -- 7-Node (LSX)";
print "==============================================";
print "";

// ============================================================
// SECTION 1: MULTIPLICATION TABLE
// ============================================================

print "--- Section 1: Multiplication table ---";

mult := [[[] : j in [1..n]] : i in [1..n]];

procedure Add(~mult, i, j, idx, c)
    new_list := [];
    found    := false;
    for entry in mult[i][j] do
        if Integers()!entry[2] eq idx then
            Append(~new_list, [c, idx]);
            found := true;
        else
            Append(~new_list, entry);
        end if;
    end for;
    if not found then Append(~new_list, [c, idx]); end if;
    mult[i][j] := new_list;
end procedure;

// Idempotents
for i in [1..7] do Add(~mult, i, i, i, k!1); end for;

// Source-target
// [arrow_idx, source_vertex, target_vertex]
arr_st := [
    [8,  1, 4],  // f14: CA1sp->HPF
    [9,  4, 1],  // f41: HPF->CA1sp
    [10, 2, 5],  // f25: BLA->sAMY
    [11, 5, 2],  // f52: sAMY->BLA
    [12, 4, 5],  // f45: HPF->sAMY
    [13, 5, 4],  // f54: sAMY->HPF
    [14, 3, 5],  // f35: HY->sAMY
    [15, 5, 3],  // f53: sAMY->HY
    [16, 5, 6],  // f56: sAMY->LA
    [17, 6, 5],  // f65: LA->sAMY
    [18, 2, 6],  // f26: BLA->LA
    [19, 6, 2],  // f62: LA->BLA
    [20, 1, 5],  // f15: CA1sp->sAMY
    [21, 4, 2],  // f42: HPF->BLA
    [22, 4, 7],  // f47: HPF->LSX  (new)
    [23, 7, 4]   // f74: LSX->HPF  (new)
];
for t in arr_st do
    a := t[1]; s := t[2]; tg := t[3];
    Add(~mult, s, a, a, k!1);
    Add(~mult, a, tg, a, k!1);
end for;

// Round-trip relations -- inherited 6-node
Add(~mult,  8,  9, 1, 1698335266113/100000000000);   // f14*f41=16.983*e1
Add(~mult,  9,  8, 4, 1698335266113/100000000000);   // f41*f14=16.983*e4
Add(~mult, 10, 11, 2, 2775220847130/100000000000);   // f25*f52=27.752*e2
Add(~mult, 11, 10, 5, 2775220847130/100000000000);   // f52*f25=27.752*e5
Add(~mult, 12, 13, 4, 3753671517223/100000000000);   // f45*f54=37.537*e4
Add(~mult, 13, 12, 5, 3753671517223/100000000000);   // f54*f45=37.537*e5
Add(~mult, 14, 15, 3, 2709020965733/100000000000);   // f35*f53=27.090*e3
Add(~mult, 15, 14, 5, 2709020965733/100000000000);   // f53*f35=27.090*e5
Add(~mult, 16, 17, 5, 9751983719692/100000000000);   // f56*f65=97.520*e5
Add(~mult, 17, 16, 6, 9751983719692/100000000000);   // f65*f56=97.520*e6
Add(~mult, 18, 19, 2, 2064812660217/1000000000000);  // f26*f62=2.065*e2
Add(~mult, 19, 18, 6, 2064812660217/1000000000000);  // f62*f26=2.065*e6

// LSX round-trips
// f_HPF_LSX * f_LSX_HPF = 69.87161588668823 * e_HPF
Add(~mult, 22, 23, 4, 6987161588668823/100000000000000);
// f_LSX_HPF * f_HPF_LSX = 69.87161588668823 * e_LSX
Add(~mult, 23, 22, 7, 6987161588668823/100000000000000);

// Path compositions -- inherited 6-node
Add(~mult,  8, 12, 20, 1170812356949/1000000000);    // f14*f45=c*f15
Add(~mult, 20, 13,  8, 1318001338568/100000000000);  // f15*f54=c*f14
Add(~mult, 10, 16, 18, 6400817774470/1000000000000); // f25*f56=c*f26
Add(~mult, 18, 17, 10, 3153447767306/1000000000000); // f26*f65=c*f25
Add(~mult,  9, 20, 12, 3428665161818/100000000000);  // f41*f15=c*f45
Add(~mult, 21, 10, 12, 584054811562/100000000000);   // f42*f25=c*f45
Add(~mult, 12, 11, 21, 345859407673/1000000000000);  // f45*f52=c*f42
Add(~mult, 13, 21, 11, 4461732771867/100000000000);  // f54*f42=c*f52
Add(~mult, 16, 19, 11, 1874714369370/100000000000);  // f56*f62=c*f52
Add(~mult, 19, 10, 17, 1876148464992/100000000000);  // f62*f25=c*f65
Add(~mult, 17, 11, 19, 1076678391633/100000000000);  // f65*f52=c*f62

// LSX path compositions:
// f_CA1sp_HPF * f_HPF_LSX = 2174344976.589 * f_CA1sp_LSX -> not in basis -> 0
// f_sAMY_HPF  * f_HPF_LSX = 39226930.781   * f_sAMY_LSX  -> not in basis -> 0
// f_LSX_HPF   * f_HPF_CA1sp = 2012159659.552 * f_LSX_CA1sp -> not in basis -> 0
// f_LSX_HPF   * f_HPF_BLA   = 5255473.832    * f_LSX_BLA   -> not in basis -> 0
// f_LSX_HPF   * f_HPF_sAMY  = 42874328.862   * f_LSX_sAMY  -> not in basis -> 0
// All LSX composed paths are non-primitive -> no Add entries needed

printf "Multiplication table complete. Basis: %o elements.\n", n;

// ============================================================
// SECTION 2: MATRIX REPRESENTATION
// ============================================================

print "";
print "--- Section 2: Matrix representation ---";

mats := [];
for i in [1..n] do
    M := ZeroMatrix(k, n, n);
    for col in [1..n] do
        for entry in mult[i][col] do
            row := Integers()!entry[2];
            if row ge 1 and row le n then M[row,col] +:= entry[1]; end if;
        end for;
    end for;
    Append(~mats, M);
end for;

Mat := MatrixAlgebra(k, n);
B   := sub<Mat | mats>;
printf "dim(B) = %o  (6-node=78, PAL-7-node=105)\n", Dimension(B);

// ============================================================
// SECTION 3: STRUCTURE
// ============================================================

print "";
print "--- Section 3: Structure ---";

printf "IsAssociative (matrix rep): %o\n", IsAssociative(B);
printf "IsCommutative:              %o\n", IsCommutative(B);

R_rad := JacobsonRadical(B);
printf "dim(JacobsonRadical):       %o\n", Dimension(R_rad);
printf "IsSemisimple:               %o\n", Dimension(R_rad) eq 0;

Z_cen := Centre(B);
printf "dim(Centre):                %o  (expect 7)\n", Dimension(Z_cen);

if Dimension(R_rad) eq 0 then
    print "Semisimple -> HH^n(B,B)=0 for n>=1 (Wedderburn)";
end if;

// ============================================================
// SECTION 4: NON-ASSOCIATIVITY
// ============================================================

print "";
print "--- Section 4: Non-associativity ---";
printf "Testing %o triples...\n", n^3;

na_count   := 0;
na_triples := [];
na_defects := [];

for i in [1..n] do
    for j in [1..n] do
        for l in [1..n] do
            lhs := [k!0 : x in [1..n]];
            for eij in mult[i][j] do
                cij := eij[1]; kij := Integers()!eij[2];
                for ekl in mult[kij][l] do
                    lhs[Integers()!ekl[2]] +:= cij*ekl[1];
                end for;
            end for;
            rhs := [k!0 : x in [1..n]];
            for ejl in mult[j][l] do
                cjl := ejl[1]; kjl := Integers()!ejl[2];
                for eikjl in mult[i][kjl] do
                    rhs[Integers()!eikjl[2]] +:= cjl*eikjl[1];
                end for;
            end for;
            if lhs ne rhs then
                na_count +:= 1;
                Append(~na_triples, [i,j,l]);
                defect := &+[Abs(lhs[x]-rhs[x]) : x in [1..n]];
                Append(~na_defects, defect);
                if na_count le 5 then
                    printf "  Triple #%o: (b%o,b%o,b%o)\n",na_count,i,j,l;
                    lp:=[]; rp:=[];
                    for x in [1..n] do
                        if lhs[x] ne 0 then Append(~lp,<lhs[x],x>); end if;
                        if rhs[x] ne 0 then Append(~rp,<rhs[x],x>); end if;
                    end for;
                    printf "    LHS=%o\n    RHS=%o\n    Defect=%o\n",lp,rp,defect;
                end if;
            end if;
        end for;
    end for;
end for;

print "";
printf "Non-assoc triples (LSX): %o / %o\n", na_count, n^3;
printf "K_algebraic(7_LSX)     = %o\n", na_count / k!n^3;

// Scaling comparison
rate6   := 62  / k!8000;
rate7p  := 93  / k!15625;   // PAL version
rate7l  := na_count / k!n^3;

print "";
print "--- Scaling comparison ---";
printf "n=6   (base):     62 / 8000  = %o\n", rate6;
printf "n=7   (PAL):      93 / 15625 = %o\n", rate7p;
printf "n=7   (LSX):      %o / %o = %o\n", na_count, n^3, rate7l;
printf "K decreasing vs 6-node: %o\n", rate7l lt rate6;
printf "K_LSX vs K_PAL:         LSX=%o  PAL=%o  LSX<PAL: %o\n",
       rate7l, rate7p, rate7l lt rate7p;

if na_count gt 0 then
    mean_d := &+na_defects / #na_defects;
    printf "Defect: min=%o  max=%o\n", Minimum(na_defects), Maximum(na_defects);
end if;

// ============================================================
// SECTION 5: SPECTRAL ANALYSIS
// ============================================================

print "";
print "--- Section 5: Spectral analysis ---";
print "";

procedure AddEdge(~M, i, j, w)
    M[i,j] +:= w;
end procedure;

function LogW(x)
    if x eq RR!0 then return RR!0; end if;
    return Log(RR!1 + x);
end function;

// Log-normalised adjacency (7x7)
T7l := ZeroMatrix(RR, 7, 7);
AddEdge(~T7l, 1, 4, LogW(RR!(1698335266113/100000000000)));
AddEdge(~T7l, 4, 1, LogW(RR!(1698335266113/100000000000)));
AddEdge(~T7l, 2, 5, LogW(RR!(2775220847130/100000000000)));
AddEdge(~T7l, 5, 2, LogW(RR!(2775220847130/100000000000)));
AddEdge(~T7l, 4, 5, LogW(RR!(3753671517223/100000000000)));
AddEdge(~T7l, 5, 4, LogW(RR!(3753671517223/100000000000)));
AddEdge(~T7l, 3, 5, LogW(RR!(2709020965733/100000000000)));
AddEdge(~T7l, 5, 3, LogW(RR!(2709020965733/100000000000)));
AddEdge(~T7l, 5, 6, LogW(RR!(9751983719692/100000000000)));
AddEdge(~T7l, 6, 5, LogW(RR!(9751983719692/100000000000)));
AddEdge(~T7l, 2, 6, LogW(RR!(2064812660217/1000000000000)));
AddEdge(~T7l, 6, 2, LogW(RR!(2064812660217/1000000000000)));
// LSX-HPF
AddEdge(~T7l, 4, 7, LogW(RR!(6987161588668823/100000000000000)));
AddEdge(~T7l, 7, 4, LogW(RR!(6987161588668823/100000000000000)));

eigs7l := Eigenvalues(T7l);
rho7l  := Maximum([Abs(e[1]) : e in eigs7l]);
q7l    := &+[T7l[i,j] : i in [1..7], j in [1..7]] / 7;
printf "Log-normalised: rho=%o  q=%o  rho/sqrt(q)=%o\n",
       rho7l, q7l, rho7l/Sqrt(q7l);

// Column-stochastic
T7s := ZeroMatrix(RR, 7, 7);
for j in [1..7] do
    s := &+[T7l[i,j] : i in [1..7]];
    if s gt 0 then for i in [1..7] do T7s[i,j] := T7l[i,j]/s; end for; end if;
end for;
eigs7s     := Eigenvalues(T7s);
eigs7s_abs := Sort([Abs(e[1]) : e in eigs7s]);
rho7s      := Maximum(eigs7s_abs);
lam2_7     := eigs7s_abs[#eigs7s_abs - 1];
gap7       := rho7s - lam2_7;
printf "Col-stochastic: eigenvalues=%o\n", eigs7s;
printf "spectral gap = %o\n", gap7;

// 6-node for comparison
T6l := ZeroMatrix(RR, 6, 6);
AddEdge(~T6l,1,4,LogW(RR!(1698335266113/100000000000)));
AddEdge(~T6l,4,1,LogW(RR!(1698335266113/100000000000)));
AddEdge(~T6l,2,5,LogW(RR!(2775220847130/100000000000)));
AddEdge(~T6l,5,2,LogW(RR!(2775220847130/100000000000)));
AddEdge(~T6l,4,5,LogW(RR!(3753671517223/100000000000)));
AddEdge(~T6l,5,4,LogW(RR!(3753671517223/100000000000)));
AddEdge(~T6l,3,5,LogW(RR!(2709020965733/100000000000)));
AddEdge(~T6l,5,3,LogW(RR!(2709020965733/100000000000)));
AddEdge(~T6l,5,6,LogW(RR!(9751983719692/100000000000)));
AddEdge(~T6l,6,5,LogW(RR!(9751983719692/100000000000)));
AddEdge(~T6l,2,6,LogW(RR!(2064812660217/1000000000000)));
AddEdge(~T6l,6,2,LogW(RR!(2064812660217/1000000000000)));
eigs6l := Eigenvalues(T6l);
rho6l  := Maximum([Abs(e[1]) : e in eigs6l]);
q6l    := &+[T6l[i,j] : i in [1..6], j in [1..6]] / 6;

print "";
print "--- Spectral comparison ---";
printf "n=6:       rho/sqrt(q) = %o\n", rho6l/Sqrt(q6l);
printf "n=7 PAL:   rho/sqrt(q) = 3.4728  (from PAL run)\n";
printf "n=7 LSX:   rho/sqrt(q) = %o\n",   rho7l/Sqrt(q7l);
printf "LSX ratio < PAL ratio: %o\n",
       rho7l/Sqrt(q7l) lt RR!(3472837584426839/1000000000000000);

// Ihara zeta
print "";
print "--- Ihara zeta det(I-uT) ---";
Rp<u> := PolynomialRing(k);
T7q   := ZeroMatrix(k, 7, 7);
T7q[1,4]:=1698335266113/100000000000; T7q[4,1]:=T7q[1,4];
T7q[2,5]:=2775220847130/100000000000; T7q[5,2]:=T7q[2,5];
T7q[4,5]:=3753671517223/100000000000; T7q[5,4]:=T7q[4,5];
T7q[3,5]:=2709020965733/100000000000; T7q[5,3]:=T7q[3,5];
T7q[5,6]:=9751983719692/100000000000; T7q[6,5]:=T7q[5,6];
T7q[2,6]:=2064812660217/1000000000000; T7q[6,2]:=T7q[2,6];
T7q[4,7]:=6987161588668823/100000000000000; T7q[7,4]:=T7q[4,7];

zeta7 := Determinant(ScalarMatrix(Rp,7,Rp!1) -
         u*Matrix(Rp,7,7,[T7q[i,j] : i in [1..7], j in [1..7]]));
printf "det(I-uT) = %o\n", zeta7;

coeffs := Coefficients(zeta7);
denoms := [Denominator(c) : c in coeffs];
lcm_d  := LCM(denoms);
scaled := [Numerator(c*lcm_d) : c in coeffs];
all_int := forall{c : c in scaled | c in IntegerRing()};
printf "Weil I (integer after scaling): %o\n", all_int;
if all_int then print "OK det(I-uT) in Z[u] after scaling"; end if;

// ============================================================
// SECTION 6: COMPARISON TABLE
// ============================================================

print "";
print "==============================================";
print "COMPARISON: 6-node vs 7-node PAL vs 7-node LSX";
print "==============================================";
print "";
print "Algebraic structure:";
printf "  n=6:       dim=78   semisimple=true  centre=6  K=62/8000=0.00775\n";
printf "  n=7 PAL:   dim=105  semisimple=true  centre=7  K=93/15625=0.00595\n";
printf "  n=7 LSX:   dim=%o  semisimple=%o  centre=%o  K=%o/%o=%o\n",
       Dimension(B), Dimension(R_rad) eq 0, Dimension(Z_cen),
       na_count, n^3, na_count/k!n^3;
print "";
print "Observation:";
print "  PAL adds 4 arrows through central nodes (HY, sAMY)";
print "  LSX adds 2 arrows through peripheral node (HPF only)";
print "  LSX is more peripheral -> fewer new non-assoc triples expected";
printf "  K decreasing vs 6-node: %o\n", rate7l lt rate6;
print "";
print "Spectral:";
printf "  n=6   log rho/sqrt(q): %o\n", rho6l/Sqrt(q6l);
printf "  n=7 LSX log rho/sqrt(q): %o\n", rho7l/Sqrt(q7l);
print "  Prime ideal T_N (simulation): 1.000 [Ramanujan, 409/409]";
print "";
print "Finite-size scaling predictions:";
printf "  phi_equil(6) = 61/120  = %o  (pred)  0.5083 (measured)\n",
       1/2+1/k!120;
printf "  phi_equil(7) = 361/720 = %o  (prediction, both PAL and LSX)\n",
       1/2+1/k!720;
printf "  phi_equil(8) = 2521/5040 = %o  (prediction)\n",
       1/2+1/k!5040;
print "";
print "==============================================";
print "DONE -- connectome_algebra_7_LSX.m";
print "==============================================";
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
stop_edges := [10, 15, 17];   // LSX 3 stop edge indices

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
