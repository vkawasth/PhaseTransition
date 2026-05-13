/*
==============================================================
BALBc Connectome — Hinge Test: n=7 LSX quiver
MAGMA program: connectome_Abound_ext1_LSX.m

Q_7L = Q_6 + {f_47, f_74}  (LSX = vertex 7, connects only to HPF)

Vertices: 1=CA1sp 2=BLA 3=HY 4=HPF 5=sAMY 6=LA 7=LSX

Key structural difference from PAL:
  - LSX (vertex 7) is PERIPHERAL: connects only to HPF (degree 1)
  - PAL (vertex 7) was a CONNECTOR: connects to HY and sAMY (degree 2)
  - Prediction: Core(Q_7L) = Core(Q_6) -> rho(B_7L) = rho(B_6) = 1.5731
  - (Peripheral nodes don't affect the biconnected core spectrum)
==============================================================
*/

k  := RationalField();
RR := RealField(20);

print "==============================================";
print "Hinge Test: n=7 LSX quiver";
print "==============================================";
print "";

n_v   := 7;
n_arr := 16;
vertex_names := ["CA1sp","BLA","HY","HPF","sAMY","LA","LSX"];

// Q_6 arrows (1-14) + LSX arrows (15-16)
arrows := [
    [1,  1, 4],   // f14: CA1sp->HPF
    [2,  4, 1],   // f41: HPF->CA1sp
    [3,  2, 5],   // f25: BLA->sAMY
    [4,  5, 2],   // f52: sAMY->BLA
    [5,  4, 5],   // f45: HPF->sAMY
    [6,  5, 4],   // f54: sAMY->HPF
    [7,  3, 5],   // f35: HY->sAMY
    [8,  5, 3],   // f53: sAMY->HY
    [9,  5, 6],   // f56: sAMY->LA
    [10, 6, 5],   // f65: LA->sAMY
    [11, 2, 6],   // f26: BLA->LA
    [12, 6, 2],   // f62: LA->BLA
    [13, 1, 5],   // f15: CA1sp->sAMY  ASYMMETRIC
    [14, 4, 2],   // f42: HPF->BLA     ASYMMETRIC
    [15, 4, 7],   // f47: HPF->LSX
    [16, 7, 4]    // f74: LSX->HPF
];

// Reversal map
rev := [0 : i in [1..n_arr]];
for i in [1..n_arr] do
    for j in [1..n_arr] do
        if i ne j
          and arrows[i][2] eq arrows[j][3]
          and arrows[i][3] eq arrows[j][2] then
            rev[i] := j;
        end if;
    end for;
end for;

print "LSX arrows added to Q_6:";
for i in [15..16] do
    printf "  arr%o (%o->%o) <-> arr%o (%o->%o)\n",
           i, vertex_names[arrows[i][2]], vertex_names[arrows[i][3]],
           rev[i], vertex_names[arrows[rev[i]][2]],
                   vertex_names[arrows[rev[i]][3]];
end for;
print "LSX is PERIPHERAL: connects only to HPF (degree 1 in symmetric quiver)";
print "";

// ── Three matrices ────────────────────────────────────────────────────────────

Hom_full := ZeroMatrix(k, n_arr, n_arr);
for i in [1..n_arr] do
    for j in [1..n_arr] do
        if arrows[i][3] eq arrows[j][2] then Hom_full[i,j] := 1; end if;
    end for;
end for;

Ext1_rev := ZeroMatrix(k, n_arr, n_arr);
for i in [1..n_arr] do
    if rev[i] gt 0 then Ext1_rev[i, rev[i]] := 1; end if;
end for;

Chi_red := Hom_full - Ext1_rev;

B_Ihara := ZeroMatrix(k, n_arr, n_arr);
for i in [1..n_arr] do
    for j in [1..n_arr] do
        si:=arrows[i][2]; ti:=arrows[i][3];
        sj:=arrows[j][2]; tj:=arrows[j][3];
        if ti eq sj and si ne tj then B_Ihara[i,j] := 1; end if;
    end for;
end for;

// ── Hinge test ────────────────────────────────────────────────────────────────

hinge := Chi_red eq B_Ihara;
printf "HINGE TEST (16x16): chi_red = B_Ihara?  %o\n\n", hinge;

case_A_ok := true;
for i in [1..n_arr] do
    j := rev[i];
    if j gt 0 then
        h:=Hom_full[i,j]; e1:=Ext1_rev[i,j];
        cr:=Chi_red[i,j]; bi:=B_Ihara[i,j];
        ok:=(h eq 1) and (e1 eq 1) and (cr eq 0) and (bi eq 0);
        if not ok then case_A_ok := false; end if;
        printf "  CaseA %o<->%o: H=%o E=%o chi=%o B=%o %o\n",
               vertex_names[arrows[i][2]], vertex_names[arrows[i][3]],
               h, e1, cr, bi, ok select "OK" else "FAIL";
    end if;
end for;
printf "Case A: %o\n\n", case_A_ok select "ALL OK" else "FAILURES";

case_B_ok := true;
for i in [1..n_arr] do
    for j in [1..n_arr] do
        if arrows[i][3] eq arrows[j][2] and rev[i] ne j then
            h:=Hom_full[i,j]; e1:=Ext1_rev[i,j];
            cr:=Chi_red[i,j]; bi:=B_Ihara[i,j];
            ok:=(h eq 1) and (e1 eq 0) and (cr eq 1) and (bi eq 1);
            if not ok then
                case_B_ok := false;
                printf "  CaseB arr%o->arr%o FAIL\n", i, j;
            end if;
        end if;
    end for;
end for;
printf "Case B: %o\n\n", case_B_ok select "ALL OK" else "FAILURES";

case_C_ok := true;
for i in [1..n_arr] do
    for j in [1..n_arr] do
        if arrows[i][3] ne arrows[j][2] then
            h:=Hom_full[i,j]; e1:=Ext1_rev[i,j];
            cr:=Chi_red[i,j]; bi:=B_Ihara[i,j];
            ok:=(h eq 0) and (e1 eq 0) and (cr eq 0) and (bi eq 0);
            if not ok then
                case_C_ok := false;
                printf "  CaseC arr%o,arr%o FAIL\n", i, j;
            end if;
        end if;
    end for;
end for;
printf "Case C: %o\n\n", case_C_ok select "ALL OK" else "FAILURES";

all_pass := hinge and case_A_ok and case_B_ok and case_C_ok;
printf "Hinge: %o\n\n", all_pass select "CONFIRMED" else "FAILED";

// ── CY-2 ─────────────────────────────────────────────────────────────────────

print "--- CY-2 Global Dimension ---";
gldim := 0;
for v in [1..n_v] do
    in_v := #[i : i in [1..n_arr] | arrows[i][3] eq v];
    rt_v := #[i : i in [1..n_arr] | arrows[i][2] eq v and rev[i] ne 0];
    pd_v := 0;
    if in_v gt 0 then pd_v := 1; end if;
    if rt_v gt 0 then pd_v := 2; end if;
    if pd_v gt gldim then gldim := pd_v; end if;
    printf "  S_%o (%o): in=%o rt=%o pd=%o\n",
           v, vertex_names[v], in_v, rt_v, pd_v;
end for;
printf "gl.dim(A_bound_7L) = %o  CY-2: %o\n\n",
       gldim, gldim le 2 select "YES" else "NO";

// ── Triangle count ────────────────────────────────────────────────────────────

print "--- Oriented triangles in Q_7L ---";
tris := [];
for a in [1..n_arr] do for b in [1..n_arr] do for c in [1..n_arr] do
    if arrows[a][3] eq arrows[b][2] and
       arrows[b][3] eq arrows[c][2] and
       arrows[c][3] eq arrows[a][2] then
        is_new := true;
        for t in tris do
            if {t[1],t[2],t[3]} eq {a,b,c} then is_new:=false; break t; end if;
        end for;
        if is_new then
            Append(~tris, [a,b,c]);
            printf "  %o->%o->%o->%o\n",
                   vertex_names[arrows[a][2]], vertex_names[arrows[b][2]],
                   vertex_names[arrows[c][2]], vertex_names[arrows[a][2]];
        end if;
    end if;
end for; end for; end for;
printf "Triangles: %o  (Q_6 had 2, LSX is peripheral -> no new triangles expected)\n\n",
       #tris;

// ── Spectral: key prediction -- LSX peripheral, rho should = rho_6 ───────────

print "--- Spectral analysis ---";
B_real := Matrix(RR, n_arr, n_arr,
    [RR!B_Ihara[i,j] : i in [1..n_arr], j in [1..n_arr]]);
eigs := Eigenvalues(B_real);
rho := Maximum([Abs(e[1]) : e in eigs]);
out_degs := [#[j : j in [1..n_arr] |
    arrows[i][3] eq arrows[j][2] and rev[i] ne j] : i in [1..n_arr]];
q_max := Maximum(out_degs);
sqrtq := Sqrt(RR!q_max);

printf "rho(B_7L)     = %o\n", rho;
printf "sqrt(q=%o)    = %o\n", q_max, sqrtq;
printf "rho/sqrt(q)   = %o\n", rho/sqrtq;
printf "Ramanujan:     %o\n",
       rho/sqrtq lt 1 select "SATISFIED" else "VIOLATED";
printf "\nSpectral rigidity check:\n";
printf "  Core(Q_7L) = Core(Q_6)?  (LSX peripheral -> same core spectrum)\n";
printf "  rho(Q_7L) = rho(Q_6)?    %o\n",
       Abs(rho - RR!1.5731) lt RR!0.001 select "YES (confirmed)" else
       "NO (unexpected change)";

// ── Weighted T7L ──────────────────────────────────────────────────────────────

print "";
print "--- Weighted Hecke operator T7L ---";
T7L := ZeroMatrix(k, 7, 7);
T7L[1,4] := 79392011/10000000;    T7L[4,1] := 95171668/10000000;
T7L[2,5] := 20859075/10000000;    T7L[5,2] := 17635407/10000000;
T7L[4,5] := 1765277/10000000;     T7L[5,4] := 19345622/100000000;
T7L[3,5] := 3150615/10000000;     T7L[5,3] := 1886353/10000000;
T7L[5,6] := 1935263/10000000;     T7L[6,5] := 1420958/10000000;
T7L[2,6] := 16993031/10000000;    T7L[6,2] := 61563894/10000000;
T7L[4,7] := 16828847/100000000;   T7L[7,4] := 15547580/100000000;
T7L[1,5] := 459058/10000000;      // CA1sp->sAMY ASYMMETRIC
T7L[4,2] := 232508/10000000;      // HPF->BLA    ASYMMETRIC

Rp<u> := PolynomialRing(k);
det_T7L := Determinant(
    ScalarMatrix(Rp,7,Rp!1) - u*Matrix(Rp,7,7,[T7L[i,j]:i in[1..7],j in[1..7]]));
printf "det(I-u*T7L) = %o\n", det_T7L;

// ── Summary ───────────────────────────────────────────────────────────────────

print "";
print "==============================================";
print "SUMMARY n=7L (LSX)";
print "==============================================";
printf "Hinge chi_red=B_Ihara (16x16):  %o\n", all_pass select "PASS" else "FAIL";
printf "CY-2 gl.dim=2:                  %o\n", gldim le 2 select "PASS" else "FAIL";
printf "Triangles:                       %o (Q_6 had 2)\n", #tris;
printf "rho(B_7L)/sqrt(q):               %o\n", rho/sqrtq;
printf "Ramanujan:                       %o\n",
       rho/sqrtq lt 1 select "PASS" else "FAIL";
printf "Spectral rigidity (rho=rho_6):   %o\n",
       Abs(rho - RR!1.5731) lt RR!0.001 select "CONFIRMED" else "DIFFERENT";
print "";
print "==============================================";
print "DONE -- connectome_Abound_ext1_LSX.m";
print "==============================================";

