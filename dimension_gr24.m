// ============================================================================
// dimension_gr24.m  —  MAGMA-version-safe
// Compute dim_k(O_partial) = boundary associator defect space dimension
// for A = BALBc connectome matrix algebra (Q_{7P})
//
// Does NOT use Quiver() — works directly with the matrix algebra.
// Run with:  magma dimension_gr24.m
// ============================================================================

// ── 1. Load the matrix algebra ───────────────────────────────────────────────
print "Loading connectome algebra Q_{7P}...";
load "connectome_algebra_6_PAL.m";
A := B;

k    := BaseRing(A);
n    := Degree(A);
bas  := Basis(A);
N    := #bas;
M    := MatrixRing(k, n);

printf "Loaded: %ox%o matrix algebra over %o, dim = %o\n", n, n, k, N;

// ── 2. Boundary node indices ─────────────────────────────────────────────────
// Q_{7P} node order assumed: 1=CA1sp, 2=BLA, 3=HPF, 4=sAMY, 5=HY, 6=LA, 7=PAL
// Boundary = nodes on the trinion boundary cycles or stop edges
boundary_nodes := {2, 4, 5, 6, 7};   // BLA, sAMY, HY, LA, PAL
names := ["CA1sp","BLA","HPF","sAMY","HY","LA","PAL"];

// ── 3. Select boundary-supported basis elements ──────────────────────────────
boundary_bas := [];
for idx in [1..N] do
    m := bas[idx];
    hit := false;
    for i in [1..n] do
        for j in [1..n] do
            if m[i][j] ne 0 and (i in boundary_nodes or j in boundary_nodes) then
                hit := true;
            end if;
        end for;
    end for;
    if hit then Append(~boundary_bas, m); end if;
end for;
printf "Boundary-supported basis elements: %o / %o\n", #boundary_bas, N;

// ── 4. Compute associator defects on boundary triples ────────────────────────
printf "Computing alpha(a,b,c) = (ab)c - a(bc) on boundary triples...\n";
defect_vecs := [];
total := 0; nonzero := 0;
for ai in [1..#boundary_bas] do
    for bi in [1..#boundary_bas] do
        for ci in [1..#boundary_bas] do
            a := boundary_bas[ai];
            b := boundary_bas[bi];
            c := boundary_bas[ci];
            defect := (a*b)*c - a*(b*c);
            total +:= 1;
            if defect ne Zero(M) then
                nonzero +:= 1;
                Append(~defect_vecs, Eltseq(defect));
            end if;
        end for;
    end for;
end for;
printf "Total triples: %o | Nonzero defects: %o\n", total, nonzero;

// ── 5. Rank of the defect space ──────────────────────────────────────────────
if #defect_vecs eq 0 then
    printf "dim(O_partial) = 0  (algebra is associative on boundary)\n";
else
    V  := RSpace(k, n^2);
    OS := sub< V | [ V ! v : v in defect_vecs ] >;
    d  := Dimension(OS);
    printf "\n=== RESULT: dim(O_partial) = %o ===\n", d;
    if d eq 4 then
        printf "CONFIRMED: dim = 4.  Gr(2,4) embedding justified.\n";
    else
        printf "dim = %o (expected 4). Check boundary node selection.\n", d;
    end if;
end if;
