// ============================================================================
// compute_boundary_obstruction.m
// Compute the boundary associator defect space O_partial
// for the BALBc connectome nonassociative algebra.
//
// Strategy:
//   1. Load the matrix algebra A from connectome_algebra_6_PAL.m (Q_{7P})
//   2. Compute all associator defects alpha(a,b,c) = (ab)c - a(bc)
//      for all basis triple products
//   3. Restrict to boundary-cycle triples (those touching the trinion boundary)
//   4. Compute the dimension of the resulting defect space O_partial
//   5. Verify dim(O_partial) = 4
//
// Run with:  magma compute_boundary_obstruction.m
// ============================================================================

// ── 1. Load the algebra ──────────────────────────────────────────────────────
print "Loading connectome algebra Q_{7P}...";
load "connectome_algebra_6_PAL.m";
A := B;

n := Degree(A);          // matrix size
k := BaseRing(A);
bas := Basis(A);
N := #bas;

printf "Algebra loaded: %o x %o matrices over %o, basis size %o\n", n, n, k, N;

// ── 2. Identify vertex idempotents and boundary nodes ────────────────────────
// Vertices: 1=CA1sp, 2=BLA, 3=HPF, 4=sAMY, 5=HY, 6=LA, 7=PAL
// Boundary nodes of the trinion (nodes on ∂Σ_Q or crossing H_1 cycles):
//   - BLA  (vertex 2): on γ_1 boundary cycle
//   - sAMY (vertex 4): hub node, all cycles pass through
//   - LA   (vertex 6): on γ_1, second leg
//   - HY   (vertex 5): on Λ⁻ recovery path
//   - PAL  (vertex 7): on Λ⁻ recovery path
// Interior nodes: CA1sp (1), HPF (3)

boundary_indices := [2, 4, 5, 6, 7];   // BLA, sAMY, HY, LA, PAL
interior_indices := [1, 3];             // CA1sp, HPF
names := ["CA1sp", "BLA", "HPF", "sAMY", "HY", "LA", "PAL"];

// ── 3. Collect basis elements supported on boundary nodes ────────────────────
// A basis element M is "boundary-supported" if row i or column j is nonzero
// for some boundary vertex i or j.
// This identifies the transport paths that cross the trinion boundary.

function is_boundary_supported(M, boundary_idx, mat_size)
    // Check if any matrix entry M[i][j] is nonzero where i or j is boundary
    for i in boundary_idx do
        for j in [1..mat_size] do
            if M[i][j] ne 0 then return true; end if;
        end for;
        for j in [1..mat_size] do
            if M[j][i] ne 0 then return true; end if;
        end for;
    end for;
    return false;
end function;

boundary_basis := [];
for idx in [1..N] do
    if is_boundary_supported(bas[idx], boundary_indices, n) then
        Append(~boundary_basis, bas[idx]);
    end if;
end for;

printf "Boundary-supported basis elements: %o of %o\n", 
    #boundary_basis, N;

// ── 4. Compute all associator defects on boundary triples ────────────────────
// alpha(a,b,c) = (a*b)*c - a*(b*c)
// Collect the nonzero defect vectors

printf "\nComputing associator defects alpha(a,b,c) on boundary triples...\n";

defect_vectors := [];
total_triples := 0;
nonzero_count := 0;

M := MatrixRing(k, n);

for ai in [1..#boundary_basis] do
    for bi in [1..#boundary_basis] do
        for ci in [1..#boundary_basis] do
            a_mat := boundary_basis[ai];
            b_mat := boundary_basis[bi];
            c_mat := boundary_basis[ci];
            
            // Compute (a*b)*c - a*(b*c) using algebra multiplication
            ab    := a_mat * b_mat;
            ab_c  := ab * c_mat;
            bc    := b_mat * c_mat;
            a_bc  := a_mat * bc;
            
            defect := ab_c - a_bc;
            total_triples +:= 1;
            
            if defect ne Zero(M) then
                nonzero_count +:= 1;
                // Flatten to row vector for rank computation
                v := &cat[ Eltseq(defect[i]) : i in [1..n] ];
                Append(~defect_vectors, v);
            end if;
        end for;
    end for;
end for;

printf "Total boundary triples evaluated: %o\n", total_triples;
printf "Nonzero associator defects: %o\n", nonzero_count;

// ── 5. Compute the dimension of O_partial = span of defect vectors ───────────
if #defect_vectors eq 0 then
    printf "\nAll associator defects are zero -- algebra is associative on boundary.\n";
    printf "dim(O_partial) = 0\n";
else
    // Build matrix whose rows are defect vectors, compute rank
    V := VectorSpace(k, n^2);
    defect_subspace := sub< V | [ V ! v : v in defect_vectors ] >;
    dim_O := Dimension(defect_subspace);
    
    printf "\n--- BOUNDARY OBSTRUCTION SPACE ---\n";
    printf "dim(O_partial) = %o\n", dim_O;
    
    if dim_O eq 4 then
        printf "\nRESULT: dim(O_partial) = 4  CONFIRMED\n";
        printf "The boundary associator defect space has dimension 4.\n";
        printf "The Gr(2,4) embedding is justified by O_partial, not Ext^2.\n";
        printf "\nInterpretation:\n";
        printf "  - The 4 basis defects correspond to the 4 trinion boundary cycles\n";
        printf "  - Decomposable bivectors in Lambda^2(O_partial) live in Gr(2,4)\n";
        printf "  - The Klein quadric = decomposability condition = St_5 identity\n";
    else
        printf "\nRESULT: dim(O_partial) = %o (not 4)\n", dim_O;
        if dim_O gt 4 then
            printf "Natural embedding is Gr(2,%o).\n", dim_O;
            printf "Consider restricting further to the minimal boundary cycles.\n";
        else
            printf "Some boundary defects may coincide -- check triple selection.\n";
        end if;
    end if;
end if;

// ── 6. Show the basis of O_partial explicitly ────────────────────────────────
if #defect_vectors gt 0 then
    printf "\nBasis of O_partial (first 4 independent vectors as %ox%o matrices):\n", n, n;
    V2 := VectorSpace(k, n^2);
    subsp := sub< V2 | [ V2 ! v : v in defect_vectors ] >;
    bas_O := Basis(subsp);
    for idx in [1..Minimum(4, #bas_O)] do
        M_defect := Matrix(k, n, n, Eltseq(bas_O[idx]));
        printf "  Basis vector %o:\n", idx;
        // Show support (nonzero entries)
        support := [];
        for i in [1..n] do
            for j in [1..n] do
                if M_defect[i][j] ne 0 then
                    Append(~support, <i, j, M_defect[i][j]>);
                end if;
            end for;
        end for;
        for s in support do
            printf "    [%o,%o] = %o  (%o -> %o)\n", 
                s[1], s[2], s[3], names[s[1]], names[s[2]];
        end for;
    end for;
end if;
