load "connectome_algebra_6_PAL_LSX.m";
> // Check which arrows are the stop edges
> // by looking at what each candidate index multiplies into
> for idx in [11,16,18,25] do
>     printf "mult[%o] (first nonzero entry): %o\n", idx, mult[idx];
> end for;
