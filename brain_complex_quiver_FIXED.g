LoadPackage("qpa");
k := GF(101);
Q_vertices := ['CA1sp', 'BLA', 'HY', 'HPF', 'sAMY', 'LA'];
Q_arrows := [["CA1sp", "HPF", "f_CA1sp_HPF"], ["HPF", "CA1sp", "f_HPF_CA1sp"], ["BLA", "LA", "f_BLA_LA"], ["BLA", "sAMY", "f_BLA_sAMY"], ["CA1sp", "sAMY", "f_CA1sp_sAMY"], ["HPF", "BLA", "f_HPF_BLA"], ["HPF", "sAMY", "f_HPF_sAMY"], ["LA", "BLA", "f_LA_BLA"], ["LA", "sAMY", "f_LA_sAMY"], ["sAMY", "BLA", "f_sAMY_BLA"], ["sAMY", "HPF", "f_sAMY_HPF"], ["sAMY", "LA", "f_sAMY_LA"], ["HY", "sAMY", "f_HY_sAMY"], ["sAMY", "HY", "f_sAMY_HY"]];
Q := Quiver(Q_vertices, Q_arrows);
A := PathAlgebra(k, Q);

# Define idempotents
for v in Q_vertices do
  AssignGeneratorVariables(A);
od;

# Relations
rels := [f_CA1sp_HPF*f_HPF_BLA - 37*f_CA1sp_BLA, f_CA1sp_HPF*f_HPF_sAMY - 12*f_CA1sp_sAMY, f_CA1sp_sAMY*f_sAMY_BLA - 69*f_CA1sp_BLA, f_CA1sp_sAMY*f_sAMY_HY - 16*f_CA1sp_HY, f_CA1sp_sAMY*f_sAMY_HPF - 78*f_CA1sp_HPF, f_BLA_sAMY*f_sAMY_HY - 68*f_BLA_HY, f_BLA_sAMY*f_sAMY_HPF - 9*f_BLA_HPF, f_BLA_sAMY*f_sAMY_LA - 73*f_BLA_LA, f_BLA_LA*f_LA_sAMY - 14*f_BLA_sAMY, f_HY_sAMY*f_sAMY_BLA - 72*f_HY_BLA, f_HY_sAMY*f_sAMY_HPF - 19*f_HY_HPF, f_HY_sAMY*f_sAMY_LA - 60*f_HY_LA, f_HPF_CA1sp*f_CA1sp_sAMY - 97*f_HPF_sAMY, f_HPF_BLA*f_BLA_sAMY - 66*f_HPF_sAMY, f_HPF_BLA*f_BLA_LA - 83*f_HPF_LA, f_HPF_sAMY*f_sAMY_BLA - 56*f_HPF_BLA, f_HPF_sAMY*f_sAMY_HY - 22*f_HPF_HY, f_HPF_sAMY*f_sAMY_LA - 98*f_HPF_LA, f_sAMY_BLA*f_BLA_LA - 64*f_sAMY_LA, f_sAMY_HPF*f_HPF_CA1sp - 39*f_sAMY_CA1sp, f_sAMY_HPF*f_HPF_BLA - 46*f_sAMY_BLA, f_sAMY_LA*f_LA_BLA - 27*f_sAMY_BLA, f_LA_BLA*f_BLA_sAMY - 9*f_LA_sAMY, f_LA_sAMY*f_sAMY_BLA - 54*f_LA_BLA, f_LA_sAMY*f_sAMY_HY - 96*f_LA_HY, f_LA_sAMY*f_sAMY_HPF - 84*f_LA_HPF, f_CA1sp_HPF*f_HPF_CA1sp - 67*e_CA1sp, f_HPF_CA1sp*f_CA1sp_HPF - 67*e_HPF, f_BLA_LA*f_LA_BLA - 85*e_BLA, f_BLA_sAMY*f_sAMY_BLA - 66*e_BLA, f_HPF_sAMY*f_sAMY_HPF - 95*e_HPF, f_LA_BLA*f_BLA_LA - 85*e_LA, f_LA_sAMY*f_sAMY_LA - 42*e_LA, f_sAMY_BLA*f_BLA_sAMY - 66*e_sAMY, f_sAMY_HPF*f_HPF_sAMY - 95*e_sAMY, f_sAMY_LA*f_LA_sAMY - 42*e_sAMY, f_HY_sAMY*f_sAMY_HY - 65*e_HY, f_sAMY_HY*f_HY_sAMY - 65*e_sAMY];

I := Ideal(A, rels);
B := A / I;

Print("Algebra constructed. Dimension: ", Dimension(B), "\n");

# Compute HH^2
HH2 := HochschildCohomology(B,2);
Print("HH^2 dimension: ", Dimension(HH2), "\n");

# Cartan matrix
C := CartanMatrix(B);
Print("Cartan matrix: ", C, "\n");