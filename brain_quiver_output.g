LoadPackage("qpa");
k := GF(101);
Q_vertices := ["CA1sp", "BLA", "HY"];
Q_arrows := [];
Q := Quiver(Q_vertices, Q_arrows);
A := PathAlgebra(k, Q);
rels := []; # Define HH2 obstructions here
I := Ideal(A, rels);
B := A / I;
Print("Success: Brain Algebra B defined with ", Dimension(B), " dimensions.\n");