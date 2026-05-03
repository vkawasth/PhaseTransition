newPackage("PathAlgebras",
     Headline => "data types for path algebras",
     Version => "0.3",
     Date => "Apr 18 2023",
     Authors => {
	  {Name => "Frank Moore",
	   HomePage => "http://www.math.wfu.edu/Faculty/Moore.html",
	   Email => "moorewf@wfu.edu"},
	  {Name => "Yunmeng Wu",
	   HomePage => "https://ymwu.sites.northeastern.edu",
	   Email => "wu.yunm@northeastern.edu"}},
     Keywords => {"Noncommutative Algebra", "Path Algebra"},
     AuxiliaryFiles => true,
     DebuggingMode => true,
     CacheExampleOutput => true
     )
--added "paPath" for example purpose
export { "PAPath","paPath", -- not exporting paPath constructor
         "PAGraph", "paGraph",
	 "PAElement",
	 "PathAlgebra",
	 "PathAlgebraQuotient",
	 "PAMatrix","paMatrix",
	 "PAMap", "paMap",
         "weight",
         "paBasis",
	 "PAIdeal", "paIdeal",
	 "PAModMon","paModMon",
	 "PAVector","paVector",
	 "PAModule",
	 "allComponent",
	 "allSubModMon",
	 "areComposable",
	 "buchAlgorithm",
	 "buchPAModule",	 
         "composePath",
	 "divAlgorithm",
	 "endVertex",	 
	 "findOverlaps",
	 "findRecurrence",
	 "findSubModMon",
	 "freePAModule",
	 "getSyzygies",
	 "getSyzygiesMatrix",
	 "getMinSyzygies",
	 "getMinSyzMatrix",
	 "getUnsubVar",
	 "hasDuplicate",
	 "interreduce",
	 "isPrefix",	 
	 "isSubModMon",
	 "isSubpath",
	 "isSubpathOnly",
	 "isSubword",
	 "isVertex",
	 "isUniform",	 
	 "leadEdgeList",
	 "leadModMon",
	 "leadPair",	
	 "leadPath",
	 "leftLexCompare",
	 "leftOverlaps",
	 "leftWeightLexCompare",
	 "leftWeightReverseLexCompare",	 	 
	 "makeMonic",
	 "numEdges",
	 "numVertices",
	 "overlaps",	 	 
	 "paBasisLength",
	 "paBasisDegree",
	 "paHilbertSeries",
	 "pathTerms",	
	 "putInPathAlgebra", 
         "rightLexCompare",
	 "rightOverlaps",
	 "rightWeightLexCompare",
	 "rightWeightReverseLexCompare",
	 "reduceOverlap",
	 "removeZeroes",	 
	 "removeZeropath",
	 "selectComponent",
	 "startVertex",
	 "stdBasisVector",
	 "toRationalFunction",
	 "toUniform"	 
	  }
      
protect ChainLimit

-- ** update this **
-- Interface:
--- Graph object
   --- List of Vertex objects  
   --- List of Edge objects
--- Path object
   --- Either, a single vertex, or
   ---         a list of composable edges
--- PathAlgebraElement object
   --- Linear combination of paths, stored as a hash table
   --- keep track of the tip with respect to an admissible order.
--- PathOrder object
   --- implement (left) weight-lex, weight-reverse-lex (with length-lex a special case of weight-lex)
--- define: 1) origin : Edges -> Vertices
---         2) terminus : Edges -> Vertices
---         3) inclusions: Vertices -> Paths
---                           Edges -> Paths
---                           Paths -> PathAlgebraElements
---         4) concatenation: Paths x Paths -> Paths or zero
---         5) define the path algebra ring using these operations

-- the prefix PA here is just to make sure these symbols are not
-- already defined elsewhere
PAPath               = new Type of HashTable
PAGraph              = new Type of HashTable
PAElement            = new Type of HashTable
PAIdeal              = new Type of HashTable
PAMatrix             = new Type of HashTable
PAMap                = new Type of HashTable
PAModMon             = new Type of HashTable
PAVector             = new Type of HashTable
PathAlgebra          = new Type of Ring
PathAlgebraQuotient  = new Type of Ring
PAModule             = new Type of MutableHashTable

-- Possible term orders:
-- Q: make this a type as well?
leftLexCompare = method()
leftLexCompare(PAPath,PAPath) := (p,q) -> (
    -- this is easy since list comparison of integers is automatically lexicographic
    if length p < length q then return symbol <;
    if length p > length q then return symbol >;
    if length p == length q and length p == 0 then return p#"vertex" ? q#"vertex"
    else return q#"edgeList" ? p#"edgeList";
);

rightLexCompare = method()
rightLexCompare(PAPath,PAPath):= (p,q) -> (
    if length p < length q then return symbol <;
    if length p > length q then return symbol >;
    if length p == length q and length p == 0 then return p#"vertex" ? q#"vertex"
    else return (reverse q#"edgeList") ? (reverse p#"edgeList");
)

leftWeightLexCompare = method()
leftWeightLexCompare(List,PAPath,PAPath) := (w,p,q) -> (
    if length p == length q and length p == 0 then return p#"vertex" ? q#"vertex";
    comp := p.weight ? q.weight;
    if comp != (symbol ==) then return comp;
    return leftLexCompare(p,q);
);

rightWeightLexCompare = method()
rightWeightLexCompare(List,PAPath,PAPath) := (w,p,q) -> (
    if length p == length q and length p == 0 then return p#"vertex" ? q#"vertex";
    comp := p.weight ? q.weight;
    if comp != (symbol ==) then return comp;
    return rightLexCompare(p,q);
);

leftWeightReverseLexCompare = method()
leftWeightReverseLexCompare(List,PAPath,PAPath) := (w,p,q) -> (
    if length p == length q and length p == 0 then return symbol ==;
    comp := p.weight ? q.weight;
    if comp != (symbol ==) then return comp;
    return rightLexCompare(q,p);
);

rightWeightReverseLexCompare = method()
rightWeightReverseLexCompare(List,PAPath,PAPath) := (w,p,q) -> (
    if length p == length q and length p == 0 then return symbol ==;
    comp := p.weight ? q.weight;
    if comp != (symbol ==) then return comp;
    return leftLexCompare(q,p);
);

PAPath == PAPath := (p,q) -> (
   if p#"graph" =!= q#"graph" then return false;
   if p.weight != q.weight then return false;
   if p.weight == 0 then return p#"vertex" == q#"vertex"
   else return p#"edgeList" == q#"edgeList";
)

--edge list prefix check
isPrefix = method()
isPrefix (List,List) := (p,q) -> (
    if #p > #q then return false;
    if p == take(q,#p) then return true
    else return false
)
isPrefix(PAPath,PAPath) := (p,q) -> isPrefix(p#"edgeList",q#"edgeList")

--edgelist suffix check
isSuffix = method()
isSuffix (List,List) := (p,q) -> (
    if #p > #q then return false;
    if p == take(q,-(#p)) then return true
    else return false
)
isSuffix(PAPath,PAPath) := (p,q) -> isSuffix(p#"edgeList",q#"edgeList")

--check is p is a subword of q and returns the position of first occurrence
--this allows proper subword
isSubword = method()
isSubword (List,List) := (p,q) -> (
   -- return value is (boolean,ZZ) where the boolean is
   -- true if p is a subword of q and false otherwise,
   -- and the integer is the first occurrence (from the left) of p in q
   -- returns -1 if not subword
   findmatch:= false;
   firstoccur:= -1;
   for j from 0 to (length q - length p ) do(
         if findmatch == false then (
	    if p == take(q,{j,j+ length p - 1}) then (
		findmatch = true;
		firstoccur = j;
	    );
	 );
   );   
   return (findmatch,firstoccur);
);


--check if p is a subpath of q or q is a subpath of p. returns </> respectively.
isSubpath = method ()
isSubpath (PAPath,PAPath) := (p,q) -> isSubword(p#"edgeList",q#"edgeList")
isSubpath (PAElement,PAElement) := (p,q) ->(
   lelp := leadEdgeList(p);
   lelq := leadEdgeList(q);
   u := {};
   v := {};
   A := p.ring;
   --uedges := 1_(A.CoefficientRing);
   --possible change:
   uedges := leadCoefficient q / (leadCoefficient p);
   vedges := 1_(A.CoefficientRing);

   if (first isSubword(lelp,lelq) == true) then(
       if last isSubword(lelp,lelq) > 0 then(
	    u = take(lelq,{0,last isSubword(lelp,lelq)-1});
	    uedges = (leadCoefficient q / (leadCoefficient p))*putInPathAlgebra(A,u);
	    );
    
       if last isSubword(lelp,lelq) < (length lelq - length lelp) then (
	    v = take(lelq,{(last isSubword(lelp,lelq)+ length lelp),(length lelq -1)}); 
	    vedges = putInPathAlgebra(A,v);
	    );

       if uedges*p*vedges == q then return (true, symbol <, uedges, vedges);
       --error "err";
    )

    else if (first isSubword(lelq,lelp) == true) then(
       lelk := lelq;
       lelq = lelp;
       lelp = lelk;
       if last isSubword(lelp,lelq) > 0 then(
	    u = take(lelq,{0,last isSubword(lelp,lelq)-1});
	    uedges = (leadCoefficient p / (leadCoefficient q))*putInPathAlgebra(A,u);
	    );
    
       if last isSubword(lelp,lelq) < (length lelq - length lelp) then (
	    v = take(lelq,{(last isSubword(lelp,lelq))+ length lelp,length lelq -1}); 
	    vedges = putInPathAlgebra(A,v);
	    );
       if uedges*q*vedges == p then return (true, symbol >, uedges, vedges);
    )
    else return (false,"not Subpath",-1,-1);
);

--this only checks if p is a subpath of q. 
isSubPathLeft = method()
isSubPathLeft (PAElement,PAElement) := (p,q) -> (
    lelp := leadEdgeList(p);
    lelq := leadEdgeList(q);
    u := {};
    v := {};
    A := p.ring;
    uedges := leadCoefficient q / (leadCoefficient p);
    vedges := 1_(A.CoefficientRing);
    if (first isSubword(lelp,lelq) == true) then(
	if last isSubword(lelp,lelq) > 0 then(
	    u = take(lelq,{0,last isSubword(lelp,lelq)-1});
	    uedges = (leadCoefficient q / (leadCoefficient p))*putInPathAlgebra(A,u);
	);
    
        if last isSubword(lelp,lelq) < (length lelq - length lelp) then (
	    v = take(lelq,{(last isSubword(lelp,lelq)+ length lelp),(length lelq -1)}); 
	    vedges = putInPathAlgebra(A,v);
	);
    );
    if uedges*p*vedges == q then return (true, symbol <, uedges, vedges);
 
    return (false,"not Subpath",-1,-1)
)

--check if p is a subpath of q and returns boolean 
isSubpathOnly = method()
isSubpathOnly (PAElement,PAElement) := (p,q) -> (  
   lelp := leadEdgeList(p);
   lelq := leadEdgeList(q);
   first isSubword(lelp,lelq)
)

-- check if subword first for findoverlaps?
--   if first isSubword(p,q) == true or first isSubword(q,p) == true then return (false);

-- Edward's notation is not consistent
-- I'll stick with the following definition:
-- if LT(f)*c=b*LT(g) and LT(f) does not divide b and LT(g) does not divide c
-- then o(f,g,b,c)=fc-bg


--notice that this does not include proper containment
findOverlaps = method()
findOverlaps (List,List) := (p,q) -> (
   -- return value is a list of positions in p where
   -- a (proper nontrivial) suffix of p is a (proper nontrivial) prefix of q.
   olPosition := 0;   
   L := min(#p,#q);
   olIndex := new MutableList from (L:0);
   for i from 1 to L-1 do(
        if take(p,-i) == take(q,i) then (olIndex#i=i);	    
   ); 
   olPosition = positions(olIndex,i->i>0);
   return olPosition;
);

-- prefix of q could be proper?
moduleFindOverlaps = method()
moduleFindOverlaps (List,List) := (p,q) -> (
   -- return value is a list of positions in p where
   -- a (proper nontrivial) suffix of p is a (proper nontrivial) prefix of q.
   olPosition := 0;   
   L := min(#p,#q);
   olIndex := new MutableList from (L:0);
   for i from 1 to L do(
        if take(p,-i) == take(q,i) then (olIndex#i=i);	    
   ); 
   olPosition = positions(olIndex,i->i>0);
   return olPosition;
);


--leftoverlaps means a suffix of p is a prefix of q
leftOverlaps = method ()
leftOverlaps (PAElement, PAElement) := (p,q) -> (
    A := p.ring;
    overlapRelations := new MutableHashTable;
    lelp := leadEdgeList(p);
    lelq := leadEdgeList(q);
    L    := findOverlaps(lelp,lelq);

    for i from 0 to length L - 1 do (
	bel := take(lelp,{0,length lelp - L#i-1});
	cel := take(lelq,{L#i,length lelq -1});
	b := putInPathAlgebra(A,bel);
	c := putInPathAlgebra(A,cel);

	if ((not first isSubword(lelp,bel)) and  (not first isSubword(lelq,cel))) then
	overlapRelations#(p,q,b,c) = (1_(A.CoefficientRing)/leadCoefficient(p))*p*c - (1_(A.CoefficientRing)/leadCoefficient(q))*b*q;
    ); 

    return overlapRelations;
);


rightOverlaps = method ()
rightOverlaps (PAElement,PAElement) := (p,q) -> (
    A := p.ring;
    overlapRelations := new MutableHashTable;
    lelp := leadEdgeList(p);
    lelq := leadEdgeList(q);
    R    := findOverlaps(lelq,lelp);
        
    for i from 0 to length R - 1 do (
	bel := take(lelq,{0,length lelq - R#i-1});
	cel := take(lelp,{R#i,length lelp -1});
	b := putInPathAlgebra(A,bel);
	c := putInPathAlgebra(A,cel);
	if ((not first isSubword(lelq,bel)) and  (not first isSubword(lelp,cel))) then
	overlapRelations#(q,p,b,c) = (1_(A.CoefficientRing)/leadCoefficient(q))*q*c - (1_(A.CoefficientRing)/leadCoefficient(p))*b*p;
    );
    return overlapRelations; 

);

overlaps = method()
overlaps (PAElement,PAElement) := (p,q) -> (
    lol := leftOverlaps(p,q);
    rol := rightOverlaps(p,q);
    return merge(lol,rol,first);   
);

divAlgorithm = method ()
divAlgorithm (List,PAElement) := (D,y) -> (
-- D should be a list of PAPaths
-- n is number of divisors
     A := y.ring;
     n := #D;
     z := y;
     r := 0_A;
     i := 0;
     divoccur := false;
     findmatch := false;
     m := new MutableList from (n:0);
     L := new MutableHashTable;
     R := new MutableHashTable;
     if z == 0_A then return (r,L,R,m);
     lelz := leadEdgeList z;
     
     while (z != 0_A) do(
       lelz = leadEdgeList z;
       divoccur = false;
       i = 1;
       while (i <= n and not divoccur) do (
         x := leadEdgeList(D#(i-1));
	 lengthx := length x;
	 
	 if first isSubword(x,lelz) == true then (
	     m#(i-1) = m#(i-1)+1;
	     u := {};
	     v := {};
             uedges := 1_(A.CoefficientRing);
	     vedges := 1_(A.CoefficientRing);
	     if last isSubword(x,lelz) == 0 then uedges = (leadCoefficient z / leadCoefficient D#(i-1)); 
	     if last isSubword(x,lelz) > 0 then(
	     	 u = take(lelz,{0,last isSubword(x,lelz)-1});
		 uedges = (leadCoefficient z / (leadCoefficient D#(i-1)))*putInPathAlgebra(A,u);
	     );
	     if last isSubword(x,lelz) < (length lelz - lengthx) then (
		 v = take(lelz,{last isSubword(x,lelz)+ lengthx,length lelz -1}); 
		 vedges = putInPathAlgebra(A,v);
	     );
	     L#(i,m#(i-1)) = u;
	     R#(i,m#(i-1)) = v;
	     divoccur = true;
	     z = z-uedges*(D#(i-1))*vedges;
	     if z == 0_A then return (r,L,R,m);
	     lelz = leadEdgeList z;
	 );
     	 i = i + 1;
	      
         );
         if not divoccur then (
	   r = r + leadTerm z;
	   z = z - leadTerm z;
         );
     );
	
     return (r,L,R,m);
);

  -- input: L = {f_1,..f_c} a list of nonzero elements
  --                        sorted by term order on the lead terms
  -- returns: L' = {r_1,r_2,...,r_c}
  -- where r_1 = f_1,
  --       r_i = divAlg({r_1,...,r_(i-1)}, f_i)

interreduce = method()
interreduce List := L -> (
    if L == {} then return L;
    G := sort (L / makeMonic);
    H := {first G};
    
    for i from 1 to #G - 1 do (
	X := first divAlgorithm(H,G#i);
	if X != 0_((first G).ring) then H = append(H,X);           
    );

    H
);


buchAlgorithm = method (Options => {DegreeLimit => 8})
buchAlgorithm (List) := opts -> (L) -> (
    
    A := ring first L;
    if any(L, f -> ring f =!= A) then error "Expected elements from a common path algebra.";
    G := interreduce (L / toUniform // flatten);
    count := #G;
    overlapsTodo := {};
   
    for i from 0 to count -1 do(
	overlapsTodo = join(overlapsTodo,
 	    J := flatten for j from i to count-1 list (
		K := select(values overlaps(G#i,G#j), f -> f != 0_A and weight f <= opts#DegreeLimit)
            	--for p in K do (
	    --	scan(G, i -> if i==p then K = delete(p,K));
            --);
	    --if K != {} then G=join(G,K);
    	    ) 
    	)		
    );
   
    overlapsTodo = interreduce(overlapsTodo);   
    -- make sure we sort overlapsTodo by weight/term order
   
    
    while overlapsTodo != {} do(
        -- 'remove' the first element of overlapsToDo
    	olap := first overlapsTodo;
	overlapsTodo = drop(overlapsTodo,1);
	rem := first divAlgorithm(G,olap);
	if rem != 0_A then (
	    rem = makeMonic rem;
	    
	    -- add rem to G
	    -- add new overlaps to overlapsTodo
	    -- optional: interreduce G
	    G = append(G,rem);
	    newOverlaps := flatten for i from 0 to #G -1 list (
	    	K := select(values overlaps(G#i,last G), f -> f != 0_A and weight f <= opts#DegreeLimit)
	    );
	    overlapsTodo = join(overlapsTodo,newOverlaps);
	    if overlapsTodo != {} then overlapsTodo = interreduce(overlapsTodo);	   
	    G = interreduce(G);
        );
    
        if (rem == 0_A) then continue;
    );

    G    
);

buchAlgorithm PAIdeal := opts -> (I) -> (
   -- check the cache first
   toCompute := {};
   if (I.cache#?"GB" and I.cache#"GB Degree" >= opts#DegreeLimit) then return I.cache#"GB"
   --else if (I.cache#?"GB" and I.cache#"GB Degree" < opts#DegreeLimit) then toCompute = I.cache#"GB"
   else toCompute = I.generators;
     
   gbI := buchAlgorithm(toCompute,opts);
   I.cache#"GB" = gbI;
   I.cache#"GB Degree" = opts#DegreeLimit;
   gbI
)

PAElement % List := (f,I) -> first divAlgorithm(I,f)
PAElement % PAIdeal := (f,I) -> first divAlgorithm(I.generators,f)

paGraph = method(Options => {Weights => {}, Degrees => {}})
paGraph (List,List,Matrix) := opts -> (verts,edges,adj) -> (
  -- 'constructor' for PAGraph objects
  -- edges are indexed lexicographically using the ordered pair (source, target) of each edge
  -- together with an arbitrary choice for edges with the same source and target.
  if ring adj =!= ZZ then error "Expected a ring over ZZ.";
  if any(flatten entries adj, e -> e < 0) then error "Expected a non-negative integer matrix.";
  if numrows adj != numcols adj then error "Expected a square matrix.";
  if numrows adj != #verts then error "Expected compatible vertex list";
  entriesAdj := flatten entries adj;
  numEdges := sum entriesAdj;
  if numEdges != #edges then error "Expected compatible edge list.";
  -- building the hash table which allows for easy source/target information about an edge
  edgeIndex := 0;
  edgeHash' := hashTable flatten for i from 0 to #entriesAdj - 1 list (
     sourceVert := i // (numcols adj);
     targetVert := i % (numcols adj);
     newEdgeHash := apply(entriesAdj#i, j -> (edgeIndex + j) => (sourceVert,targetVert));
     edgeIndex = edgeIndex + entriesAdj#i;
     newEdgeHash
  );

  myWeights := if opts#Weights == {} then toList(#edges:1) else opts#Weights;
  if #myWeights != #edges then error "Expected weight list to match number of edges.";
  if not all(myWeights, c -> ring c === ZZ and c > 0) then error "Expected a list of positive integers for the weight.";
  
  myDegrees := if opts#Degrees == {} then toList(#edges:1) else opts#Degrees;
  if #myDegrees != #edges then error "Expected edge degree list to match number of edges.";
  if not all(myDegrees, c -> ring c === ZZ and c > 0) then error "Expected a list of positive integers for the weight.";
    
  retVal := new PAGraph from {"adjacencyMatrix" => adj,
                              "vertexLabels" => verts,
			      "edgeLabels" => edges,
			      "edgeHash" => edgeHash',
			      "weights" => myWeights,
			      symbol degrees => myDegrees,
--			      symbol compareTerms => (p,q) -> leftWeightLexCompare(myWeights,p,q)};
			      "compareTerms" => (p,q) -> leftWeightReverseLexCompare(myWeights,p,q)};
  retVal  
)

getWeight = (w,p) -> sum apply(p, i -> w#i)
getDegree = (w,p) -> sum apply(p, i -> w#i)

net PAGraph := G -> net G#"adjacencyMatrix"

paPath = method()
paPath (PAGraph, List) := (G, edgeList') -> (
  if edgeList' == {} then error "Expected a nonempty list.";
  new PAPath from { "vertex" => null,
                    "edgeList" => edgeList',
		    symbol degree => getDegree(G.degrees, edgeList'),
		    symbol weight => getWeight(G#"weights",edgeList'),
                          "graph" => G }
)

paPath (PAGraph, ZZ) := (G, p) -> (
  if p > #(G#"vertexLabels") or p < 0 then error "Expected an integer in 0..(#G.vertexLabels)";
  new PAPath from { "vertex" => p,
                    "edgeList" => {},
		    symbol degree => 0,
		    symbol weight => 0,
                          "graph" => G }
)

length PAPath := p -> #(p#"edgeList")
removeZeroes := h -> select(h, f -> f != 0)

removeZeropath = method ()
removeZeropath (List) := L->(
    select (L, f -> f != 0_((L#0).ring))        
)

--PAPath == PAPath := (p,q) -> (
--   if length p != length q then false
--   else if length p > 0 then p.edgeList == q.edgeList
--   else p.vertex == q.vertex
--)

-- only implementing left-lex order for now

PAPath ? PAPath := (p,q) -> p#"graph"#"compareTerms"(p,q)

endVertex = method()
endVertex PAPath := p -> (
   if p#"edgeList" == {} then return p#"vertex";
   last (p#"graph")#"edgeHash"#(last p#"edgeList")
)

startVertex = method()
startVertex PAPath := p -> (
    if p#"edgeList" == {} then return p#"vertex";
    first (p#"graph")#"edgeHash"#(first p#"edgeList")
)

areComposable = method()
areComposable (PAGraph, PAPath, PAPath) := (G,p,q) -> (
   -- returns true or false, depending on whether p and q
   -- are composable
   if length p > 0 and length q > 0 then 
     endVertex p == startVertex q
   else if length p > 0 then
     endVertex p == q#"vertex"
   else if length q > 0 then 
     p#"vertex" == startVertex q
   else
     p#"vertex" == q#"vertex")

-- this function should only be called in a call to combine
-- for a hash table for multiplication (in a ring or a module)
-- note the continue below!
composePath = method()
composePath (PAPath, PAPath) := (p,q) -> (
   -- this function assumes p and q are composable paths
   -- and produces the composition of p and q.
   G := p#"graph";
   if not areComposable(G,p,q) then continue else (
      if length p > 0 and length q > 0 then (
  	  new PAPath from { "vertex" => null,
	                    "edgeList" => p#"edgeList" | q#"edgeList",
			    symbol degree => p.degree + q.degree,
			    symbol weight => p.weight + q.weight,
                            "graph" => G }
      )
      else if length p > 0 then p
      else q
   )
)

pathTerms = method()
pathTerms PAElement := f -> sort pairs f.terms

leadPath = method()
leadPath PAElement := f -> (
  if f.cache#?"leadPath" then return f.cache#"leadPath";
  if #f.terms == 0 then return f;

  ltpath := first max pairs f.terms;
  f.cache#"leadPath" = ltpath;
  ltpath
)

leadTerm PAElement := f -> (
  if f.cache#?"leadTerm" then return f.cache#"leadTerm";
  if #f.terms == 0 then return f;
  lt := max pairs f.terms;
  ltElt := putInPathAlgebra(f.ring,first lt,last lt);
  f.cache#"leadTerm" = ltElt;
  ltElt
)

-- error if f=0
leadEdgeList = method()
leadEdgeList PAElement := f -> (
    if f.cache#?"leadEdgeList" then return f.cache#"leadEdgeList";
    if #f.terms == 0 then return {};
    
    lpath := leadPath f;
    lel := lpath#"edgeList";
    f.cache#"leadEdgeList" = lel;
    return lel;
    
)

PAElement ? PAElement := (f,g) -> (
    if f == 0 and g == 0 then return symbol ==;
    if f == 0 and g != 0 then return symbol <;
    if g == 0 and f != 0 then return symbol >;
    
    retval := (leadPath f) ? (leadPath g);
    if retval =!= symbol == then return retval;
    (reverse pathTerms f) ? (reverse pathTerms g)
)

weight = method()
weight PAElement := f -> (
  (leadPath f).weight
)

leadCoefficient PAElement := f -> (
  ltf := leadTerm f;
  ltCoeff := first (values (ltf.terms)); 
  ltCoeff    

)

leadMonomial PAElement := f -> (
  ltf := leadTerm f;
  lm := first (keys (ltf.terms)); 
  putInPathAlgebra(f.ring,lm)
)

PAElement == PAElement := (f,g) -> (
  if (f#"graph" =!= g#"graph") then return false;
  fTerms := pathTerms f;
  gTerms := pathTerms g;
  if #fTerms != #gTerms then return false;
  return fTerms == gTerms;
)

PAElement == ZZ := (f,n) -> (
  if n == 0 and #f.terms == 0 then true
  else f == 1_(f.ring)*n
)

ZZ == PAElement := (n,f) -> (f == n)

startVertex PAElement := f -> startVertex leadPath f
startVertex List := L -> for p in L list startVertex p

endVertex PAElement := f -> endVertex leadPath f
endVertex List := L -> for p in L list endVertex p

vertices = method()
vertices PAElement := f -> (startVertex f, endVertex f)


isVertex = method()
isVertex PAPath := f -> f.weight == 0
isVertex PAElement := f -> if f == 0 then return false else isVertex leadPath f
isVertex PAVector := f -> isVertex (leadModMon f).path

terms PAElement := f -> apply(pairs f.terms, p -> putInPathAlgebra(f.ring,first p,last p))
length PAElement := f -> length leadPath f

toUniform = method()
toUniform PAElement := f -> (
   uniformHash := new MutableHashTable from {};
   for p in terms f do (
      s := startVertex p;
      e := endVertex p;
      if not uniformHash#?(s,e) then uniformHash#(s,e) = p else uniformHash#(s,e) = uniformHash#(s,e) + p;
   );
   sort values uniformHash
)

isUniform = method()
isUniform PAElement := f -> (
   if f == 0 then return true;
   s := startVertex f;
   e := endVertex f;
   all(terms f, t -> startVertex t == s and endVertex t == e)
)

 --   (leadPath f).edgeList
ring PAElement := f -> f.ring

new PathAlgebra from List := (PathAlgebra, inits) -> new PathAlgebra of PAElement from new HashTable from inits

putInPathAlgebra = method()
putInPathAlgebra (PathAlgebraQuotient, PAPath) :=
putInPathAlgebra (PathAlgebra, PAPath) := (A,p) -> (
    if p#"graph" === A#"graph" then
       new A from hashTable {("graph", p#"graph"),
                             (symbol ring, A),
                             (symbol terms, hashTable { p => 1_(A.CoefficientRing) }),
			     (symbol cache, new CacheTable from {})}   
    else error "Expected the same underlying graph."
)
putInPathAlgebra (PathAlgebraQuotient, PAPath, ZZ) :=
putInPathAlgebra (PathAlgebraQuotient, PAPath, QQ) :=
putInPathAlgebra (PathAlgebraQuotient, PAPath, RingElement) :=
putInPathAlgebra (PathAlgebra, PAPath, ZZ) := 
putInPathAlgebra (PathAlgebra, PAPath, QQ) := 
putInPathAlgebra (PathAlgebra, PAPath, RingElement) := (A,p,c) -> (
    if p#"graph" === A#"graph" and ring c === A.CoefficientRing then
       new A from hashTable {( "graph", p#"graph"),
                             (symbol ring, A),
                             (symbol terms, hashTable { p => c }),
			     (symbol cache, new CacheTable from {})}   
    else error "Expected the same underlying graph or coefficient ring."
)

-- helper function for combining hash table
addVals := (c,d) -> (
   e := c+d;
   if e == 0 then continue else e
);

Ring PAGraph := (R, G) -> (
   A := new PathAlgebra from {("graph") => G,
           (symbol generators) => {},
       	   (symbol degreesRing) => degreesRing 1,
           "edgeGens" => {},
	   "vertexGens" => {},
	   (symbol CoefficientRing) => R,
	   (symbol cache) => new CacheTable from {},
	   (symbol baseRings) => {ZZ},
	   (symbol degreeLength) => 1,
	   "unit" => null};

   vertexLabels := G#"vertexLabels" / baseName;
   vertexGens := apply(#vertexLabels, i -> putInPathAlgebra(A,paPath(G,i)));
   A#"vertexGens" = apply(#vertexLabels, i -> (vertexLabels#i) <- vertexGens#i);

   edgeLabels := G#"edgeLabels" / baseName;
   edgeGens := apply(#edgeLabels, i -> putInPathAlgebra(A,paPath(G,{i})));
   A#"edgeGens" = apply(#edgeLabels, i -> (edgeLabels#i) <- edgeGens#i);

   A.generators = A#"vertexGens" | A#"edgeGens";
   
   A + A := (f,g) -> (
      newHash := merge(f.terms,g.terms,addVals);
      new A from hashTable {("graph", f#"graph"),
	                    (symbol ring, A),
                            (symbol terms, newHash),
			    (symbol cache, new CacheTable from {})}   
   );
   A#"unit" = sum A#"vertexGens";

   ZZ * A :=
   QQ * A :=
   R * A := (r,f) -> new A from hashTable {("graph", f#"graph"),
                                           (symbol ring, A),
                                           (symbol terms, removeZeroes applyValues(f.terms, c -> r*c)),
					   (symbol cache, new CacheTable from {})};
   A * ZZ :=
   A * QQ := 
   A * R := (f,r) -> r*f;

   promote (ZZ,A) := 
   promote (QQ,A) := 
   promote (R,A)  := (r,A) -> r*A#"unit";
   
   ZZ + A := 
   QQ + A := 
   R + A  := (r,f) -> r*A#"unit" + f;

   A + ZZ :=
   A + QQ :=
   A + R  := (f,r) -> r + f;
         
   A - A := (f,g) -> f + (-1)*g;

   A * A := (f,g) -> (
      newHash := combine(f.terms,g.terms,composePath,times,addVals);
      new A from hashTable {("graph", f#"graph"),
	                    (symbol ring, A),
	                    (symbol terms, newHash),
			    (symbol cache, new CacheTable from {})}
   );

   A + R := (f,r) -> f + r*A#"unit";
   R + A := (r,f) -> f + r;
   
   A ^ ZZ := (f,n) -> product toList (n:f);
   
   - A := f -> new A from hashTable {( "graph", f#"graph"),
	                             (symbol ring, A),
	                             (symbol terms, applyValues(f.terms, c -> -c)),
			             (symbol cache, new CacheTable from {})};

   A
)

ideal PathAlgebra := A -> paIdeal {0_A}
    
numVertices = method()
numVertices PAGraph := G -> #(G#"vertexLabels")
numVertices PathAlgebraQuotient := 
numVertices PathAlgebra := A -> numVertices A#"graph"

numEdges = method()
numEdges PAGraph := G -> #(G#"edgeLabels")
numEdges PathAlgebraQuotient := 
numEdges PathAlgebra := A -> numEdges A#"graph"

hasDuplicate = method()
hasDuplicate List := L -> (
  numbersFound := new MutableHashTable;
  for n in L do (
    if not numbersFound#?n then numbersFound#n = n
    else return true;
  );
  return false;
)

getUnsubVar = method()
getUnsubVar Symbol := identity
getUnsubVar IndexedVariable := first

net PAPath := p -> (
   edgeList := p#"edgeList";
   
   -- single vertex case
   if edgeList === {} then return net p#"graph"#"vertexLabels"#(p#"vertex");

   -- path case
   myNet := net "";
   tempVar := first edgeList;
   curDegree := 0;
   hasDupe := hasDuplicate edgeList;
   for e in edgeList do (
      if e === tempVar then curDegree = curDegree + 1
      else (
        if hasDupe then (
          nosubscriptNet := net getUnsubVar p#"graph"#"edgeLabels"#tempVar;
          emptySpace := horizontalJoin (width nosubscriptNet : " ");
	  powerNet := emptySpace | (if curDegree == 1 then net "" else (net curDegree));
	  myNet = myNet | stack( powerNet, net p#"graph"#"edgeLabels"#tempVar);
	)
        else
	  myNet = myNet | net p#"graph"#"edgeLabels"#tempVar;
	tempVar = e;
        curDegree = 1;
      );
   );
   if hasDupe then (
     nosubscriptNet := net getUnsubVar p#"graph"#"edgeLabels"#tempVar;
     emptySpace := horizontalJoin (width nosubscriptNet : " ");
     powerNet := emptySpace | (if curDegree == 1 then net "" else (net curDegree));
     myNet = myNet | stack( powerNet, net p#"graph"#"edgeLabels"#tempVar);
     myNet^1
   )
   else (
     myNet = myNet | net p#"graph"#"edgeLabels"#tempVar;
     myNet
   )
)

isNumeric = method()
isNumeric QQ := p -> return true;
isNumeric ZZ := p -> return true;
isNumeric RingElement := p -> (
   possField := ZZ/(char ultimate(coefficientRing, ring p));
   if class p === possField then return true;
)

net PAElement := f -> (
   fPairs := pairs f.terms;

   if #fPairs == 0 then return net 0;

   myNet := net "";
   firstTime := true;
   printedNeg := false;
   for p in reverse sort fPairs do (
      coeffNet := if last p == 1 then net ""
                  else if last p == -1 then (
	             printedNeg = true;
	             net "- "
                  )
	          else if isNumeric last p then (
	             tempCoeff := value toString last p;
             	     isNeg := (abs tempCoeff != tempCoeff);
	     	     tempCoeff = abs tempCoeff;
	     	     printedNeg = isNeg;
	     	     (if isNeg then net "- " else net "") | (net tempCoeff)
	  	  )
                  else (
	     	      printParen := #(terms last p) > 1;
	     	      net (if printParen then "(" else "") | net last p | net (if printParen then ")" else "")
	  	  );
      myNet = myNet | (if firstTime then net "" else if printedNeg then " " else net " + ") | coeffNet | net first p;
      firstTime = false;
      printedNeg = false;
   );

   myNet
)

makeMonic = method()
makeMonic(PAElement) := f -> (
   A := ring f;
   (1_(A.CoefficientRing)/leadCoefficient(f))*f
)
makeMonic(List) := L -> apply(L,m -> makeMonic m)
-----------------
-- PAIdeal code
-----------------
paIdeal = method()
paIdeal List := L -> (
  if L == {} then error "Expected a nonempty list.";
  R := ring first L;
  if class R =!= PathAlgebra then error "Expected elements in a path algebra.";
  if not all(L, f -> ring f === R) then error "Expected elements in a common path algebra.";
  retVal := new PAIdeal from {symbol ring => R,
                              symbol generators => L,
			      symbol cache => new CacheTable from {}};
  retVal
)

PAIdeal + PAIdeal := (I,J) -> (
  if I.ring =!= J.ring then error "Expected ideals in the same ring.";
  retVal := new PAIdeal from {symbol ring => I.ring,
                              symbol generators => I.generators | J.generators,
			      symbol cache => new CacheTable from {}};
  retVal
)
--paBasis of a PathAlgebra

paBasis = method(Options => {Strategy => "Degree"})
paBasis(ZZ, PathAlgebra) := opts -> (n,R) -> (
    if opts#Strategy == "Degree" then paBasisDegree(n,R)
    else if opts#Strategy == "Length" then paBasisLength(n,R)
    else error "Expected Strategy to be either Degree or Length."
)

paBasis(ZZ, PathAlgebra, List) := opts -> (n,R,M) -> (
    if opts#Strategy == "Degree" then paBasisDegree(n,R,M)
    else if opts#Strategy == "Length" then paBasisLength(n,R,M)
    else error "Expected Strategy to be either Degree or Length."
)

paBasisLength = method()
paBasisLength(ZZ,PathAlgebra) := (n,R) -> paBasisLength(n,R,{})
paBasisLength(ZZ,PathAlgebra,List) := (n,R,M) -> (
   -- M a list of PAElements whose lead terms the basis avoids
   -- this gives a basis of the quotient R/M
   -- usually, this is used when M is the ideal of lead terms
   -- of some other ideal I
   if R.cache#?("LengthBasis",n,M) then return R.cache#("LengthBasis",n,M);
   G := R#"edgeGens";
   g := #G;
   local myBasis;
   if n == 0 then myBasis = R#"vertexGens"
   else if n == 1 then myBasis = select(G, f -> all(M, m -> not isSubpathOnly(m,f)))
   else (
      prevBasis := paBasisLength(n-1,R);
      mynewBasis := flatten for p in G list apply(prevBasis,i->p*i);
      myBasis = select(mynewBasis, f -> f != 0 and all(M, m -> not isSubpathOnly(m,f)));
   );
   R.cache#("LengthBasis",n,M) = myBasis;
   return myBasis
)


-- paBasis with degrees

paBasisDegree = method()
paBasisDegree(ZZ,PathAlgebra) := (n,R) -> paBasisDegree(n,R,{})

paBasisDegree(ZZ,PathAlgebra,List) := (n,R,M) -> (
   if R.cache#?("DegreeBasis",n,M) then return R.cache#("DegreeBasis",n,M);
   G := R#"edgeGens";
   g := #G;
   local myBasis;
   if n == 0 then myBasis = R#"vertexGens"
   else if n == 1 then myBasis = select(G, f -> (length f) == 1 and all(M, m -> not isSubpathOnly(m,f)))
   else (
      tempList := flatten for g in G list if (length g) < n then apply(paBasisDegree(n-(length g),R,M), m -> g*m)
                                          else if (length g) == n then {g}
				          else {};
      myBasis = select(tempList, f -> f != 0 and all(M, m -> not isSubpathOnly(m,f)));      
   );
   R.cache#("DegreeBasis",n,M) = myBasis;
   return myBasis
)

paHilbertSeries = method(Options => options paBasis)
paHilbertSeries (ZZ,PathAlgebra,List) := opts -> (n, R, M) -> (
   if not (opts#Strategy == "Degree" or opts#Strategy == "Length") then error "Expected Degree or Length Strategy.";
   d := #R#"vertexGens";
   G := R#"graph";
   A := degreesRing 1;
   F := A^d;
   sum for i from 0 to n list (
      if i == 0 then id_F
      else (
         -- we should allow paBasis to accept a hashtable
         -- with previous bases computed that can be used
         -- for future basis computations
         basisHere := paBasis(i,R,M,opts);
	 (A_0)^i*(transpose map(F,F,apply(pairs tally apply(basisHere, f -> (startVertex leadPath f, endVertex leadPath f)), p -> p#0 => p#1)))
      )
   )
)


-- this code is lifted from AssociativeAlgebras, see example below
findRecurrence = method()
findRecurrence List := as -> (
   revas := reverse as;
   deg := 1;
   extra := 2;  -- used to build an overdetermined system to help rule out false recurrences.
   recurrence := null;
   while (2*(deg+extra) < #revas) do (
      eqns := matrix apply(deg+extra, i -> revas_(toList((1+i)..(deg+i))));
      out := transpose matrix{revas_(toList(0..(deg+extra-1)))};
      soln := out // eqns;
      if eqns*soln != out then
         deg = deg + 1
      else (
	 recurrence = soln;
	 deg = deg + 1;
	 break;
      )
   );
   recurrence
)

toRationalFunction = method()
toRationalFunction List := as -> (
   S := getSymbol "S";
   soln := findRecurrence(as);
   revas := reverse as;
   if (soln === null) then return null;
   deg := numgens target soln;
   A := ZZ(monoid[S]);
   fixLow := g -> g*(leadCoefficient last terms g)^(-1);
   B := frac A;
   C := degreesRing 1;
   phi := map(C,A,gens C);
   Q := 1 - first flatten entries ((matrix {apply(deg, i -> (B_0)^(i+1))})*soln);
   totalDeg := #as - 1;
   numDeg := totalDeg - deg;
   f := sum apply(numDeg + 1, i -> revas#(totalDeg - i)*(B_0)^i);
   guess1 := f + sum apply(deg, i -> (B_0)^(totalDeg - i)*revas#i/Q);
   numGuess := sum select(terms numerator guess1, t -> sum degree t <= numDeg);
   guess2elt := numGuess / Q;
   num := phi fixLow numerator guess2elt;
   den := phi fixLow denominator guess2elt;
   guess2expr := expression num / expression factor den;
   (num, den, guess2expr)
)

------------------------------
-- PathAlgebraQuotient Code --
------------------------------

new PathAlgebraQuotient from List := (PathAlgebraQuotient, inits) -> new PathAlgebraQuotient of PAElement from new HashTable from inits
PathAlgebra / PAIdeal := (A, I) -> (
   Igb := buchAlgorithm I;
   B := new PathAlgebraQuotient from {(symbol generators) => {},
                                      "vertexGens" => {},
				      "edgeGens" => {},
       			     	      (symbol CoefficientRing) => A.CoefficientRing,
  			     	      (symbol ambient) => A,
                             	      ("graph") => A#"graph",
				      (symbol cache) => new CacheTable from {},
          		     	      (symbol baseRings) => {ZZ},    -- this will be for quotients of quotients
				       "unit" => null,
				      (symbol ideal) => I};
   
   
   G := B#"graph";
   vertexLabels := G#"vertexLabels" / baseName;
   vertexGens := apply(#vertexLabels, i -> putInPathAlgebra(B,paPath(G,i)));
   B#"vertexGens" = apply(#vertexLabels, i -> (vertexLabels#i) <- vertexGens#i);

   edgeLabels := G#"edgeLabels" / baseName;
   edgeGens := apply(#edgeLabels, i -> putInPathAlgebra(B,paPath(G,{i})));
   B#"edgeGens" = apply(#edgeLabels, i -> (edgeLabels#i) <- edgeGens#i);

   B.generators = B#"vertexGens" | B#"edgeGens";

   R := A.CoefficientRing;
   
   -- function to reduce the result and place it in the quotient ring.
   push := f -> (
      temp := f % Igb;
      new B from {("graph", B#"graph"),
	          (symbol ring) => B,
                  (symbol cache) => new CacheTable from {},
                  (symbol terms) => temp.terms}
   );
   
   --- all these promotes will need to be written between this ring and all base rings.
   promote (A,B) := (f,B) -> new B from {("graph", B#"graph"),
                                         (symbol ring) => B,
                                         (symbol cache) => new CacheTable from {},
                                         (symbol terms) => f.terms};
                                     
   promote (B,A) := (f,A) -> new A from {("graph", B#"graph"),
                                         (symbol ring) => A,
                                         (symbol cache) => new CacheTable from {},
                                         (symbol terms) => f.terms};

   lift B := opts -> f -> promote(f,A);

   B * B := (f,g) -> push((lift f)*(lift g));
   B ^ ZZ := (f,n) -> product toList (n:f);
   B + B := (f,g) -> push((lift f)+(lift g));
   R * B := (r,f) -> push(r*(lift f));
   B * R := (f,r) -> r*f;
   A * B := (f,g) -> push(f*(lift g));
   B * A := (f,g) -> push((lift f)*g);
   QQ * B := (r,f) -> push(r*(lift f));
   B * QQ := (f,r) -> r*f;
   ZZ * B := (r,f) -> push(r*(lift f));
   B * ZZ := (f,r) -> r*f;
   B - B := (f,g) -> f + (-1)*g;
   - B := f -> (-1)*f;
   B + ZZ := (f,r) -> push((lift f) + r);
   ZZ + B := (r,f) -> f + r;
   B + QQ := (f,r) -> push((lift f) + r);
   QQ + B := (r,f) -> f + r;

   B == B := (f,g) -> (lift(f - g) % Igb) == 0;
   B == ZZ := (f,n) -> (lift f) == n;
   ZZ == B := (n,f) -> f == n;
   B == QQ := (f,n) -> (lift f) == n;
   QQ == B := (n,f) -> f == n;
   B == R := (f,n) -> (lift f) == n;
   R == B := (n,f) -> f == n;
   
   B#"unit" = sum B#"vertexGens";   

   promote (ZZ,B) := 
   promote (QQ,B) := 
   promote (R,B)  := (r,B) -> r*B#"unit";

   B
)

use PathAlgebraQuotient :=
use PathAlgebra := A -> (
   scan(A#"graph"#"vertexLabels", A#"vertexGens", (sym,val) -> sym <- val);
   scan(A#"graph"#"edgeLabels", A#"edgeGens", (sym,val) -> sym <- val);
   A
)

ideal PathAlgebraQuotient := B -> B.ideal

ambient PathAlgebraQuotient := B -> B.ambient

paMatrix = method()
paMatrix (List,HashTable) := (L,compHash) -> (
   if #L == 0 then error "Expected a nonempty list.";
   cols := #(L#0);
   if cols == 0 then error "Expected nonempty rows.";
   if not all(L, r -> #r == cols) then error "Expected a rectangular array.";
   if not all(L, r -> all(r, e -> member(PAElement, ancestors class e))) then
      error "Expected an array of PAElements.";
   A := (L#0#0).ring;
   -- really, should promote the elements to a common PathAlgebra(Quotient?)
   if not all(L, r -> all(r, e -> e.ring === A)) then
      error "Expected an array of elements over a common ring.";   
   if #compHash != #L then error "Expected component hash to be the same size as number of rows.";
   result := new PAMatrix from { (symbol entries) => L,
                                 (symbol numcols) => #(L#0),
				 (symbol numrows) => #L,
				 (symbol module) => freePAModule(A,#L,compHash),
				 (symbol ring) => A};
   result
)

paMatrix List := L -> (
   if #L == 0 then error "Expected a nonempty list.";
   cols := #(L#0);
   if cols == 0 then error "Expected nonempty rows.";
   if not all(L, r -> #r == cols) then error "Expected a rectangular array.";
   if not all(L, r -> all(r, e -> member(PAElement, ancestors class e))) then
      error "Expected an array of PAElements.";
   A := (L#0#0).ring;
   paMatrix(L,hashTable apply(#L, i -> (i,1_A)))
)

net PAMatrix := M -> net expression M
expression PAMatrix := M -> MatrixExpression applyTable(M.entries, e -> net expression e)
entries PAMatrix := M -> M.entries

PAMatrix + PAMatrix := (M,N) -> (
   if M.numcols != N.numcols or M.numrows != N.numrows then
      error "Expected matrices of the same shape.";
   if M.ring =!= N.ring then
      error "Expected matrices over the same ring.";
   result := new PAMatrix from { (symbol entries) => table(M.numrows,M.numcols, (i,j) -> M.entries#i#j + N.entries#i#j),
                                 (symbol numcols) => M.numcols,
				 (symbol numrows) => N.numrows,
				 (symbol ring) => M.ring};
   result   
)

transpose PAMatrix := M -> (
   result := new PAMatrix from { (symbol entries) => transpose M.entries,
                                 (symbol numcols) => M.numrows,
				 (symbol numrows) => M.numcols,
				 (symbol ring) => M.ring};
   result   
   )

PAMatrix * PAMatrix := (M,N) -> (
   if M.numcols != N.numrows then
      error "Expected matrices of compatible shapes.";
   if M.ring =!= N.ring then
      error "Expected matrices over the same ring.";
   result := new PAMatrix from { (symbol entries) => table(M.numrows,N.numcols, (i,j) -> sum apply(M.numcols, k -> (M.entries#i#k)*(N.entries#k#j))),
                                 (symbol numcols) => N.numcols,
				 (symbol numrows) => M.numrows,
				 (symbol ring) => M.ring};
   result   
)

PAMatrix _ Sequence := (M,p) -> if length p == 2 then M.entries#(p#0)#(p#1) else error "Expected a sequence of length two.";

ZZ * PAMatrix :=
QQ * PAMatrix := (r,M) -> (
   return result := new PAMatrix from { (symbol entries) => table(M.numrows,M.numcols, (i,j) -> r*M.entries#i#j),
                                 (symbol numcols) => M.numcols,
				 (symbol numrows) => M.numrows,
				 (symbol ring) => M.ring} 
)

PAMatrix * ZZ  :=
PAMatrix * QQ := (M,r) -> r*M

PAMatrix ^ ZZ := (M,n) -> product toList (n:M);

- PAMatrix := M -> (
   return result := new PAMatrix from { (symbol entries) => table(M.numrows,M.numcols, (i,j) -> -(M.entries#i#j)),
                                 (symbol numcols) => M.numcols,
				 (symbol numrows) => M.numrows,
				 (symbol ring) => M.ring}
)   

PAMatrix - PAMatrix := (M,N) -> M + (-N)

numrows PAMatrix :=  M -> M.numrows
numcols PAMatrix :=  M -> M.numcols


----- PAMap object ------
paMap = method()
paMap (PathAlgebra, PathAlgebraQuotient, List, List) := 
paMap (PathAlgebraQuotient, PathAlgebra, List, List) := 
paMap (PathAlgebraQuotient, PathAlgebraQuotient, List, List) := 
paMap (PathAlgebra, PathAlgebra, List, List) := (B,A,vertexMaps,edgeMaps) -> (
   if #vertexMaps != numVertices A then error "Wrong number of vertices received for map.";
   if #edgeMaps != numEdges A then error "Wrong number of PAElements received for map.";
   if not all(vertexMaps | edgeMaps, v -> ring v === B) then error "Expected lists of PAElements in target.";
   if not all(vertexMaps, v -> isVertex v or v == 0) then error "Expected a list of vertices or zero.";
   if not all(edgeMaps, e -> isUniform e) then error "Expected a list of uniform elements.";
   
   new PAMap from hashTable {(symbol source) => A,
       	                     (symbol target) => B,
			     "vertexMap" => vertexMaps,
			     "edgeMap" => edgeMaps} 
)

PAMap PAElement := (phi, f) -> (
   if class f =!= phi.source then error "Expected input in the domain of f.";
   --f = paMapNoneKernel(phi,f);
   oldList := pairs f.terms;
   -- these two finds the edge number of the input path and then find the new edge number of the output path
   newList := for p in oldList list (
                  if (p#0#"vertex") =!= null then (p#1)*(phi#"vertexMap"#(p#0#"vertex"))
		  else (p#1)*(product for i in (p#0)#"edgeList" list phi#"edgeMap"#i)
   	      );
   --newPaths := for p in newList list putInPathAlgebra(phi.target,paPath(phi.target.graph,p));
   --if newPaths == {} then return 0_(phi.target)
   --else return sum newPaths
   result := sum newList;
   if result == 0 then 0_(phi.target) else result
)

------------------------------
 -- PathAlgebraModule Code --
------------------------------

paModMon = method()
paModMon (PAModule,ZZ,PAPath) := (M,n,p)-> (
   if n < 0 then error "Expected a non-negative Mod.";
   result := new PAModMon from {"component" => n,
                                "compDegree" => 0,
				(symbol module) => M,
				(symbol path) => p};
   result
)

PAModMon * PAPath := (f,g) -> new PAModMon from  {"component" => f#"component",
                                                  "compDegree" => f#"compDegree",
						  (symbol module) => f.module,
				                  (symbol path) => composePath(f.path,g)};

PAModMon * PAElement := (f,g) -> f * (leadPath g)

new PAModule from List := (PAModule, inits) -> new PAModule of PAVector from new HashTable from inits

freePAModule = method()
freePAModule (PathAlgebraQuotient,ZZ,HashTable) := 
freePAModule (PathAlgebra,ZZ,HashTable) := (A,r,compHash) -> (
   if r <= 0 then error "Expected a positive integer.";
   M := new PAModule from {(symbol ring) => A,
		           (symbol numgens) => r,
			   "componentHash" => compHash,
			   "compareTerms" => (p,q) -> schreyerCompare(compHash,p,q),
		           (symbol cache) => new CacheTable from {}};
		       
   M + M := (u,v) -> (
      newHash := merge(u.terms,v.terms,(i,j)->(s := i+j; if s == 0 then continue else s));
      new M from {(symbol terms) => newHash,
	          (symbol cache) => new CacheTable from {}}
   );

   - M := f -> new M from {(symbol terms) => removeZeroes applyValues(f.terms, c -> -c),
                           (symbol cache) => new CacheTable from {}};
   
   M - M := (m,n) -> m + (-n);

   M * A := (f,g) -> (
      newHash := combine(f.terms,g.terms,(u,v) -> paModMon(M,u#"component",composePath(u.path,v)),times,addVals);
      new M from {(symbol terms) => newHash,
	          (symbol cache) => new CacheTable from {}}
   );
   
   ZZ * M :=
   QQ * M := (r,f) -> new M from {(symbol terms) => removeZeroes applyValues(f.terms, c -> r*c),
                                  (symbol cache) => new CacheTable from {}};

   M * ZZ :=
   M * QQ := (f,r) -> r*f;
  
   M
)
freePAModule(PathAlgebra,ZZ) := (A,r) -> freePAModule(A,r,hashTable apply(r, i -> (i,1_A)))

PathAlgebra ^ ZZ := (A,r) -> freePAModule(A,r)

net PAModule := M -> (
    result := net M.ring | "^" | net M.numgens;
    result
)

degrees PAModule := M -> toList(M.numgens, {0})

rank PAModule := M -> M.numgens

PAModule _ ZZ := (M,n) -> (
   if n < 0 or n >= M.numgens then error "Subscript out of bounds.";
   stdBasisVector(M,n)
)

paVector = method()
paVector (PAModule,HashTable) := (M, termHash) -> new M from {(symbol terms) => termHash,
							      (symbol cache) => new CacheTable from {}}

paVector (PAModule,List,List) := (M,coeffs,modmons) -> (
   if #coeffs != #modmons then error "Expected two lists of the same size";
   if #coeffs == 0 then error "Expected a nonempty list";
   if any(modmons, m -> m#"component" < 0 or m#"component" >= M.numgens) then error "Expected indices in correct range.";
   R := ring first coeffs;
   if any(coeffs, c -> class c =!= R) then error "Expected a list of elements from the same Ring.";
   paVector (M, hashTable apply(#modmons, i -> (modmons#i,coeffs#i)))
)

-- create paVector List
-- input is List of PAElements
-- this will create the paFreeModule that contains the paVector with the default

paVector(List) := L -> (
  
    num := #L;
    A := class first select(L,m -> m != 0);
    M := A^num;
    return sum apply(num, i -> (M_i)*(L#i))
)


ZZ _ PAModule := (n,M) -> if n == 0 then new M from {(symbol terms) => hashTable {},
						     (symbol cache) => new CacheTable from {}} else error "Expected zero."
net PAVector := M -> (
    mons := keys M.terms;
    if #mons == 0 then return net 0;
    myNet := net "";
    firstTime := true;
    for p in reverse sort pairs M.terms do (
       myNet = myNet | (if firstTime then net "" else net " + ") | (net p#1) | (net ((p#0)#"component",(p#0).path));
       firstTime = false;
    );
    myNet
)


-- output should look like "paVector <entries of M as a list>"
-- so that the above paVector List function can read it in.
-- always want "value toString XX == XX"

toString PAElement := f -> (
   fPairs := pairs f.terms;

   if #fPairs == 0 then return toString 0;

   myNet := "";
   firstTime := true;
   for p in reverse sort fPairs do (
      coeffNet := if last p == 1 then "" else (toString last p | "*");
      myNet = myNet | (if firstTime then "" else " + ") | coeffNet | toString first p;
      firstTime = false;
   );

   myNet
)

toString PAPath := f -> (
    edgeList := f#"edgeList";
    if edgeList == {} then return toString f#"graph"#"vertexLabels"#(f#"vertex");
    myNet := "";
    tempVar := first edgeList;
    curDegree := 0;
 
    for v in edgeList do(
        if v == tempVar then curDegree = curDegree + 1
	else (
	    myNet = myNet | (toString (f#"graph"#"edgeLabels")#tempVar) | if curDegree == 1 then "*" else "^" | curDegree | "*";
            tempVar = v;
	    curDegree = 1;
	);
    );
    myNet | (toString (f#"graph"#"edgeLabels")#(last edgeList)) | if curDegree == 1 then "" else "^" | curDegree
   
) 
      
--all terms
toString PAVector := f -> (

    myNet := "paVector ";
    z := 0_(class f);
    comp := rank class f;
    compList := for i from 0 to comp - 1 list selectComponent(f,i);
    result := for q in compList list(
	if q == z then "0" 
	else(
	    myNet1 := "";
    	    fpairs := pairs q.terms; 
    	    firstTime := true;
    	    for p in fpairs do (
	    	coeff := last p;
	    	myNet1 = myNet1 | (if firstTime then "" else " + ") |(if coeff == 1 then "" else (toString coeff | "*")) | (toString (first p).path);
            	firstTime = false;	    	    
    	    );
            myNet1
        )
    );
    return myNet | toString result
)

entries PAVector := f -> (
    A := (class f).ring;
    comp := rank class f;
    compList := for i from 0 to comp - 1 list (
       temp := selectComponent(f,i);
       putInPathAlgebra(A,temp)
    )   
)

toMatrix = method()
toMatrix PAVector := v -> makeMatrix({v},rank class v)

PAModMon ? PAModMon := (p,q) -> (
    p.module#"compareTerms"(p,q)
    -- position over term (POT) ordering
    --comp := p.component ? q.component;
    --if comp =!= symbol == then comp else p.path ? q.path
)
--assuming PAVector only contain one PAModMon
isPrefix (PAVector,PAVector) := (p,q) -> (
    if leadComponent p != leadComponent q then return false;
    return isPrefix(leadModMon p,leadModMon q)
)    

isPrefix (PAModMon,PAModMon) := (p,q) ->(
    if p#"component" != q#"component" then return false;
    if p.path#"edgeList" == {} then return (startVertex p.path) == (startVertex q.path);
    return isPrefix(p.path#"edgeList",q.path#"edgeList")
)

--in this case, we only consider if f is a prefix of g
isSubModMon = method()
isSubModMon(PAModMon,PAPath) := (f,g) -> isPrefix(f.path#"edgeList",g#"edgeList")

isSubModMon(PAModMon,PAModMon) := (f,g) -> (
    if f#"component" != g#"component" then return false;
    
    return isPrefix(f,g)
)
--only check the leadTerm of a PAElement
isSubModMon(PAModMon,PAElement) := (f,g) -> isSubModMon(f,leadPath g) 

isSubModMon(PAElement,PAModMon) := (f,g) -> isPrefix((leadPath f)#"edgeList",g.path#"edgeList")

isSubword(PAElement,PAModMon) := (f,g) -> first isSubword((leadPath f)#"edgeList",g.path#"edgeList")
isSuffix(PAElement,PAModMon) := (f,g) -> isSuffix((leadPath f)#"edgeList",g.path#"edgeList")
isSuffix(PAElement,PAElement) := (f,g) -> isSuffix(leadEdgeList f,leadEdgeList g)

length PAModMon := f-> length f.path

-- asssuming f is a submodmon of g
-- return a paPath
findOverlaps(PAModMon,PAModMon) := (f,g) -> (
    if length f == length g then return paPath(f.path#"graph",0)
    else return paPath(f.path#"graph",drop(g.path#"edgeList",length f))
)

findOverlaps(PAModMon,PAElement) := (f,g) -> findOverlaps(f,leadPath g)
findOverlaps(PAModMon,PAPath) := (f,g) -> (
    if length f == length g then return paPath(f.path#"graph",0)
    else return paPath(f.path#"graph",drop(g#"edgeList",length f))
)

findOverlaps(PAElement,PAModMon) := (f,g) -> (
    fpath := leadPath f;
    if length fpath == length g then return paPath(fpath#"graph",0)
    else return paPath(fpath#"graph",drop(g.path#"edgeList", - (length f)))
)

findOverlaps(PAPath,PAModMon) := (f,g) -> (
    if length f == length g then return paPath(f#"graph",0)
    else return paPath(f#"graph",drop(g.path#"edgeList",length f))
)
--findOverlaps(PAModMon,PAElement) := (f,g) -> putInPathAlgebra(g.ring,findOverlaps(f,leadPath g))

-- the following returns a PAElement
findOverlapsPAE = method()
findOverlapsPAE(PAModMon,PAModMon) := (f,g) -> (
    A := f.module.ring;
    if length f == length g then return 1_A
    else return putInPathAlgebra(A,drop(g.path#"edgeList",length f))
)

findOverlapsPAE(PAModMon,PAElement) := (f,g) -> findOverlapsPAE(f,leadPath g)
findOverlapsPAE(PAModMon,PAPath) := (f,g) -> (
    A := f.module.ring;
    if length f == length g then return 1_A
    else return putInPathAlgebra(A,drop(g#"edgeList",length f))
)

findOverlapsPAE(PAElement,PAModMon) := (f,g) -> (
    A := f.ring;
    fpath := leadPath f;
    if length fpath == length g then return 1_A
    else return putInPathAlgebra(A,drop(g.path#"edgeList", - (length f)))
)

findOverlapsPAE(PAPath,PAModMon) := (f,g) -> (
    A := g.module.ring;
    if length f == length g then return 1_A
    else return putInPathAlgebra(A,drop(g.path#"edgeList",length f))
)



--get the list of component of each paModMon of f 
allComponent = method()
allComponent(PAVector) := f -> for p in keys f.terms list p#"component"

maxComponent = method()
maxComponent(PAVector) := f -> max allComponent f

-- the component of leadTerm
leadComponent PAVector := f -> first allComponent leadTerm f
 
--input PAVector and component info, get all modmons with that component
selectComponent = method()
selectComponent(PAVector,ZZ) := (f,n) -> (
    modterms := select(pairs f.terms,p->p#0#"component" == n);
    result := new (class f) from {
                                  (symbol terms) => hashTable modterms,
				  (symbol cache) => new CacheTable from {}
			        };
    result
) 

leadModMon = method()
leadModMon(PAVector) := f -> (
    if f.cache#?"leadModMon" then return f.cache#"leadModMon";
    result := last sort keys f.terms;
    f.cache#"leadModMon" = result;
    result
)

vectorLeadPAElement = method()
vectorLeadPAElement PAVector := f -> (
   lmm := leadModMon f;
   putInPathAlgebra((class f).ring, lmm.path)
)

weight PAVector := f -> (
  (leadModMon f).path.weight
)


leadPair = method()
leadPair(PAVector) := f -> (
  if #f.terms == 0 then return f;
  lt := last pathTerms f;
  lt
)

leadTerm PAVector := f -> (
  if #f.terms == 0 then return f;
  paVector(class f,{last leadPair f},{first leadPair f})
)

leadCoefficient PAVector := f -> last leadPair(f);

terms PAVector := f -> apply(reverse pathTerms f, p -> paVector(class f,{last p},{first p}));

pathTerms PAVector := f -> sort pairs f.terms

PAModMon == PAModMon := (p,q) -> (
   if p#"component" != q#"component" then return false;
   if p.path != q.path then return false;
   true
)

PAVector == PAVector := (f,g) -> (
  if (class f =!= class g) then return false;
  fTerms := pathTerms f;
  gTerms := pathTerms g;
  if #fTerms != #gTerms then return false;
  return fTerms == gTerms;
)

PAVector == ZZ := (f,n) -> (
  if n == 0 then return #(f.terms) == 0;
  error "Cannot compare a nonzero integer and a PAVector.";
)

PAVector ? PAVector := (f,g) -> (class f)#"compareTerms"(f,g)

stdBasisVector = method()
stdBasisVector (PAModule,ZZ) := (M,n) -> (
    A := M.ring;
    G := A#"graph";
    R := A.CoefficientRing;
    m := numVertices G;
    paVector(M,toList(m:1_R),apply(m, i -> paModMon(M,n,paPath(G,i))))
)

--First list: module generators
--Second list: defining equations of the algebra
findSubModMon = method()
findSubModMon(PAVector,List,List) := (f,A,B) -> (
    for i from 0 to #B - 1 do (
	if isSubword(B#i,leadModMon f) then return (1,i,B#i,findOverlapsPAE(B#i,leadModMon f));
    );
    for j from 0 to #A - 1 do (
	if isSubModMon(leadModMon A#j, leadModMon f) then return (2,j,A#j,findOverlapsPAE(leadModMon A#j, leadModMon f));
    );
    return (0,0,0,"Not subModMon")
)

findSubModMon(Sequence,List,List) := (g,A,B) -> (
    f := first g;
    for i from 0 to #B - 1 do (
	if isSuffix(B#i,leadModMon f) then return (1,i,B#i,findOverlapsPAE(B#i,leadModMon f));
    );
    for j from 0 to #A - 1 do (
	if isSubModMon(leadModMon first A#j, leadModMon f) then return (2,j,first A#j,findOverlapsPAE(leadModMon first A#j, leadModMon f));
    );
    return (0,0,0,"Not subModMon")
)

-- list all the submodmon and overlaps 
-- created for the beginning setup of next function
allSubModMon = method()
allSubModMon(PAVector,List) := (f,B) -> (
    L:= for q in B list(
	if isSubModMon(leadModMon f, q) then (f,q,findOverlaps(leadModMon f, q))
	else {}
    );
    L = delete({},L);
    -- sorts the list by path degree of the overlap polynomial
    return L
)


makeMonic PAVector := m -> ((leadCoefficient m)^(-1))*m

changeLabels = method()
changeLabels (List,List) := (L, perm) -> apply(L, l -> changeLabels(l,perm))
changeLabels (PAVector, List) := (m, perm) -> makeMonic paVector(class m, applyKeys(m.terms, k -> paModMon(class m,perm_(k#"component"),k.path)))


putInPathAlgebra(PathAlgebra,PAVector):= (A,f) -> (
    if keys f.terms == {} then return 0_A
    else sum for p in pairs f.terms list (last p)*putInPathAlgebra(A,(first p).path)
)
putInPathAlgebra(PathAlgebra,List) := (A,L) ->  putInPathAlgebra(A,paPath(A#"graph",L))

putAllInPathAlgebra = method()
putAllInPathAlgebra(PathAlgebra,List) := (A,L)-> for p in L list putInPathAlgebra(A,p)



sumPAVector = method()
sumPAVector(List) := L ->(
    sumf := L#0;
    L = drop(L,1);
    for p in L do sumf = sumf + p;
    return sumf
)   

sumPAElement = method()
sumPAElement(List) := L ->(
    sumf := L#0;
    L = drop(L,1);
    for p in L do sumf = sumf + p;
    return sumf
)   


changeOfPosition = method()
changeOfPosition(List) := M ->(
    sortM := sort M;
    perm := apply(M, m -> position(sortM, n -> n == m));
    invPerm := apply(sortM, m -> position(M, n -> n == m));
    return (perm,invPerm)
)


nonZeroElts = method()
nonZeroElts List := L -> select(L, l -> l != 0)

--this tracks the leadterm when we go back to the full PAElement
moduleTrack = method()
moduleTrack(PAMatrix,HashTable) := (M,N) ->(
    if numRows M != # N then error "Expected same number of rows and number of keys";
    -- for each column, we get back to the original PAElement, which will be set to component 0,1,...etc for the next szy step.
    L := for i from 0 to (numColumns M)-1 list (i,last sort nonZeroElts for j from 0 to (numRows M)-1 list (N#j)*(M.entries#j#i));
    return new HashTable from L
 ) 

schreyerLeadTerm = method ()
schreyerLeadTerm (PAVector,HashTable) := (f,N) -> (
    --N is a moduleTrack hashtable
    A := (class leadTerm f).ring;
    vecterms := terms f;
    fullTerms := for p in pathTerms f list (p#1)*(N#((p#0)#"component"))*putInPathAlgebra(A,(p#0).path);
    ltposition := maxPosition(first changeOfPosition(fullTerms));
    return vecterms#ltposition   
)
    
schreyerCompare = method()
schreyerCompare(HashTable,PAModMon,PAModMon) := (compHash,p,q) -> (
  -- compHash has at least one thing in it, we get the ring from there.
  A := ring compHash#0;
  schP := compHash#(p#"component") * putInPathAlgebra(A,p.path);
  schQ := compHash#(q#"component") * putInPathAlgebra(A,q.path);
  result := schP ? schQ;
  if result =!= symbol == then return result;
  p#"component"? q#"component"
)

schreyerCompare(HashTable,PAVector,PAVector) := (compHash,p,q) -> (
  (reverse sort pairs p.terms) ? (reverse sort pairs q.terms)
)  


schreyerTrack = method()
schreyerTrack (PAVector,HashTable) := (f,N) -> (
    A := (class leadTerm f).ring;
    vecpair := first terms f;
    mmpair := first pathTerms vecpair;
    return (last mmpair)*(N#((first mmpair)#"component"))*putInPathAlgebra(A,(first mmpair).path)
)
    
schreyerTerms = method()
schreyerTerms (PAVector,HashTable) := (f,N) -> (
    vecpairs := terms f;
    schreyerpairs := for p in vecpairs list schreyerTrack(p,N);
    return sort schreyerpairs
)  

getCompHash = method()
getCompHash(PAMatrix,HashTable) := (M,compHash) ->(
    A := M.ring;
    getelement := for i from 0 to (M.numColumns -1) list (
	              for j from 0 to (M.numRows - 1) list (
		          if M.entries#j#i == 0_A then continue
		          else (compHash#j)*(M.entries#j#i)
		      )
	          ); 
    getlist := for p in getelement list sumPAElement(p);
    newCompHash := for i from 0 to #getlist - 1 list (i,getlist#i);
      
    return new HashTable from newCompHash
)

--this give a list of vector list of each column
getVecListFromMatrix = method()
getVecListFromMatrix(PAMatrix,PAModule) := (M,F) -> (
    A := M.ring;
    getlist := for i from 0 to (M.numColumns -1) list (
	           for j from 0 to (M.numRows - 1) list (
		       if M.entries#j#i == 0_A then continue
		       else (F_j)*(M.entries#j#i)
		   )
	       );
    return getlist	   
)   

getVecFromMatrix = method()
getVecFromMatrix(PAMatrix,PAModule) := (M,F) -> for p in getVecListFromMatrix(M,F) list sumPAVector(p)


overlapsToHash = method()
overlapsToHash (HashTable,ZZ) := (ovl,n) ->(
    if keys ovl == {} then return {};
    ovlpairs := pairs ovl;
    --first ovl #3 is what module element is multiplied by 
    ovlToDo := for p in ovlpairs list (last p,new MutableHashTable from {(n,(first p)#3)});
    return ovlToDo;
    
)

overlapsToList = method()
overlapsToList (HashTable,ZZ) := (ovl,n) ->(
    if keys ovl == {} then return {};
    ovlpairs := pairs ovl;
    A := (class last ovlpairs#0).ring;
    --first ovl #3 is what module element is multiplied by 
    ovlToDo := for p in ovlpairs list (last p,new MutableList from join(toList(n-1:0_A),{(first p)#3}));
    return ovlToDo;
    
)


insertIntoSortedList = method()
insertIntoSortedList (List,List) := (L,M) -> (
   M = sort M;
   Lindex := 0;
   Mindex := 0;
   local toTake;
   soFar := while (Lindex < #L and Mindex < #M) list (
      if L#Lindex <= M#Mindex then (
         toTake = L#Lindex;
	 Lindex = Lindex + 1;
      )
      else (
         toTake = M#Mindex;
	 Mindex = Mindex + 1;
      );
      toTake
   );
   soFar | take(M,-(#M - Mindex)) | take(L,-(#L - Lindex))
)


-- this function gets the overlap relation between PAVector and PAElement
-- and only check if a sufix of PAVector is a prefix of PAElement
--note: for this overlap, we allow p to be a subword of q
overlaps(PAVector,PAElement,ZZ):= (p,q,n) ->( 
    F := class p;
    A := F.ring;
    overlapRelations := {};
    lmmp := leadModMon p;
    lelp := lmmp.path#"edgeList";
    lelq := leadEdgeList(q);
    -- remove any overlaps that are too long
    L    := select(moduleFindOverlaps(lelp,lelq), l -> #lelq + l <= n);
    
    for i from 0 to length L - 1 do (
	b:= 1_(A.CoefficientRing);
	c:= 1_(A.CoefficientRing);
	if ((length lelp) - L#i) != 0 then (bel := take(lelp,length lelp - L#i);
	                      	            b = putInPathAlgebra(A,bel);
					   );
        if ((length lelq) - L#i) != 0 then (cel := take(lelq,-(length lelq - L#i));
   	                                    c = putInPathAlgebra(A,cel);
                                           );
	overlapRelations = overlapRelations | {(p,q,b,c)};
    ); 

    return overlapRelations;
);

overlaps(Sequence,PAElement,ZZ):= (f,q,n) ->(
    p := first f;
    track := last f;
    F := class p;
    A := F.ring;
    overlapRelations := {};
    lmmp := leadModMon p;
    lelp := lmmp.path#"edgeList";
    lelq := leadEdgeList(q);
    -- remove any overlaps that are too long
    L    := select(moduleFindOverlaps(lelp,lelq), l -> #lelq + l <= n);
    
    for i from 0 to length L - 1 do (
	b:= 1_(A.CoefficientRing);
	c:= 1_(A.CoefficientRing);
	if ((length lelp) - L#i) != 0 then (bel := take(lelp,length lelp - L#i);
	                      	            b = putInPathAlgebra(A,bel);
					   );
        if ((length lelq) - L#i) != 0 then (cel := take(lelq,-(length lelq - L#i));
   	                                    c = putInPathAlgebra(A,cel);
                                           );
	result := (p*c-(F_(lmmp#"component"))*b*q, apply(track, m -> m*c),f,q,b,c);
	overlapRelations = overlapRelations | {result};
    ); 

    return overlapRelations
);


-- this function gives all of the overlap relations between a PAVector and a list of PAElements
overlapsList = method()
overlapsList(PAVector,List,ZZ):= 
overlapsList(Sequence,List,ZZ):= (f,L,n) -> (    
    olapslist := for q in L list overlaps(f,q,n);
    
    --olapslist = olapslist | (vertexOverlap f);
    return flatten olapslist    
)

vertexOverlap = method()
vertexOverlap(PAVector,List) := (p,track) -> (
    vertexlist := notEndVertex p;
    olapslist := {};
    if vertexlist != {} then (
    	--error "err";
	F := class p;
    	A := F.ring;
    	lmmp := leadModMon p;
    	lelp := lmmp.path#"edgeList";
	if lelp == {} then lelp = lmmp.path#"vertex";
	for j in vertexlist do (
	    b := putInPathAlgebra(A,lelp);
	    olapslist = olapslist | {(0_F,apply(track, m -> m*j),(p,track),j,b,j)};
    	)
    );
    return olapslist
) 

vertexSyz = method()
vertexSyz(List,PAModule) := (L,F) -> (
    syzlist := {};
    for i from 0 to #L - 1 do (
	p := first L#i;
	track := last L#i;	
	vertexlist := notEndVertex p;
	for q in vertexlist do (if p*q == 0 then syzlist = syzlist |{(F_i*q,apply(track, n -> n*q))});     	
    );
       
    return syzlist
)
    

trackModInit = method()
trackModInit (List) := M ->(
    -- this method also uniformizes the input
    A := (class M#0).ring;
    flatten for i from 1 to #M list (
	f := M#(i-1);
	unif := toUniform f;
	for t in unif list (t,toList(i-1:0_A)|{endVertexLabel vectorLeadPAElement t}|toList(#M-i:0_A))
	--for t in unif list (t,toList(i-1:0_A)|{1_A}|toList(#M-i:0_A))
    )
)

trackModInit2 = method()
trackModInit2 (List) := M ->(
    A := (class M#0).ring;
    flatten for i from 1 to #M list  (M#(i-1),toList(i-1:0_A)|{1_A}|toList(#M-i:0_A))  
)

makeMonicTrack = method()
makeMonicTrack (List) := L-> for p in L list (((leadCoefficient p#0)^(-1))*(p#0), apply(p#1, m-> ((leadCoefficient p#0)^(-1))*m))
makeMonicTrack (Sequence) := p ->  (((leadCoefficient p#0)^(-1))*(p#0), apply(p#1, m-> ((leadCoefficient p#0)^(-1))*m))
makeMonicTrack (PAVector,List) := (p,q) ->  (((leadCoefficient p)^(-1))*p, apply(q, m-> ((leadCoefficient p)^(-1))*m))

changeOfPositionTrack = method()
changeOfPositionTrack (List) := M ->(
    sortM := sort M;
    perm := apply(M, m -> position(sortM, n -> first n == first m));
    invPerm := apply(sortM, m -> position(M, n -> first n == first m));
    return (perm,invPerm)
)


interredVecTrack = method()
interredVecTrack List := L ->(
    if L == {} then return L;
    G := makeMonicTrack L;
    G = sort G;
    H := {first G};
    inputsyz := {};
    
    for i from 1 to #G - 1 do (
	X := reduceOverlap(G#i,H,{},{});
	-- if first X == 0, then the second entry of X is a syzygy
	-- and should be added to the list of syzygies
	--error "err";
	if first X != 0_(class first first G) then H = append(H,X)
	else inputsyz = inputsyz | {last X}     
    );

    return (H,inputsyz)     
)

reduceOverlap = method(Options => {Module => 0})
reduceOverlap(Sequence,List,List,List):= opts ->(f,F,G,L) -> (
    -- f is one element of overlaps
    -- F is the list of module generators
    -- G is the list of defining equations of the algebra
    p := first f;
    track := f#1;
    F0 := class p;
    A := F0.ring;
    tempf := p;
    
    if opts#Module =!= 0 then (
	F1 := opts#Module;
    	inputIndex := position(F, f' -> f' == f#2);
    	mySyz := F1_(L#inputIndex) * (f#5);
    );
    rem := 0_F0;
    --error "err";
    while tempf != 0 do (
	ov:= findSubModMon((tempf,track),F,G);
	-- ov is of the form (code,index,relation,overlap) where:
	-- code: 0,1,2 where 0 means nothing found, 1 means algebra relation and 2 means module relation
        -- index: index of the algebra/module relation
	-- relation: corresponding relation used
	-- overlap: necessary multiple to cancel the lead terms

	if (first ov == 0) then (
	    -- properly handle when no submodmon is found
	    rem = rem + leadTerm  tempf;
	    tempf = tempf - leadTerm tempf;
	    continue;
	);
	--error "err";
	reducer := 0;
	-- if a ring relation was used, need to put the relation in a component
	if ov#0 == 1 then reducer = (F0_((leadModMon tempf)#"component"))*(leadCoefficient tempf)*(ov#3)*(ov#2);
        if ov#0 == 2 then (--error "err";
	    	           reducer = (leadCoefficient tempf)*(ov#2)*(ov#3);
	                   track = flatten for i from 0 to #track -1 list (track#i - (leadCoefficient tempf)*((last F#(ov#1))#i)*(ov#3));
                           if opts#Module =!= 0 then mySyz = mySyz - (leadCoefficient tempf)*(F1_(ov#1))*(ov#3);
	                  );
	tempf = tempf - reducer;
	
    );
    if opts#Module =!= 0 then return (rem,track,mySyz)
    else return (rem,track)
)

getOverlaps = method(Options => {DegreeLimit => 8})
getOverlaps(List,List) := opts -> (M,I) -> (
    F := class first first M;
    overlaps := flatten for f in M list overlapsList(f,I,opts#DegreeLimit);
    overlaps = delete({},overlaps);
    
    --overlaps = sort select(overlaps, f -> first f != (0_F));

    return sort overlaps
    
 )

-- M is the list of module generators
-- I is the list of defining equations of the algebra
buchPAModule = method(Options => {DegreeLimit => 8})
buchPAModule(List,List) := opts -> (M,I) -> (
    --M = sort M;
    --M = for p in M list (makeMonic p);

    M = trackModInit(M);
    M = first interredVecTrack M;
    F1 := class first first M;
    A := F1.ring;
    
    overlapsToDo := getOverlaps(M,I);
    overlapsToDo = select(overlapsToDo, f -> first f != (0_F1));
    --overlapsToDo := flatten for p in M list overlapsList(p,I,opts#DegreeLimit);
    --overlapsToDo = sort select(overlapsToDo, f -> first f != (0_F1));

    while overlapsToDo != {} do (
	-- error "err";
	olap := first overlapsToDo;       	
	overlapsToDo = drop(overlapsToDo,1);
        if length (leadModMon first olap).path > opts#DegreeLimit then (
	    --error "Err";
	    continue;
	);
         
	olapred := elapsedTime reduceOverlap(olap,M,I,{});
        --error "err";
	if first olapred != 0_F1 then (
	                         M = M|{makeMonicTrack olapred};
	    	                 M = first interredVecTrack M;
	    	    	    	 newOverlaps := (overlapsList(olapred,I,opts#DegreeLimit));
				 --error "err";
				 newOverlaps = select(newOverlaps, f -> first f != (0_F1));
				 overlapsToDo = insertIntoSortedList(overlapsToDo,newOverlaps);
				 );
       	     				
    );
    return M
    
)

getTrackMatrix = method()
getTrackMatrix (List) := L ->(
    allLists := for p in L list p#1;
    
    return transpose paMatrix(allLists)
)

getSyzygies = method(Options => {DegreeLimit => 8})
getSyzygies(List,List) := opts -> (M,I) -> (
    -- M is the list of module generators
    -- I is the list of defining equations of the algebra
    if class first M === Sequence then  M = apply(M, m -> first m);
    M1 := interredVecTrack(trackModInit2 M);
    inputSyz := last M1;
    M1 = first M1;
    
    M2 := buchPAModule(M,I,DegreeLimit => opts#DegreeLimit);
    F := class first first M2;
    A := F.ring;

    overlaps := getOverlaps(M2,I);
    M = trackModInit2 M;
    newCompList := sort unique apply(M | M2, m -> first m);
    Mposition := apply(M2,m -> position(newCompList, f' -> f' == first m));
    compHash := hashTable apply(#newCompList, i -> (i,last sort schreyerTerms(newCompList#i, F#"componentHash")));
    
    F1 := freePAModule(A,#newCompList,compHash);

    syzygies := for p in overlaps list (
      result := reduceOverlap(p,M2,I,Mposition,Module => F1);
      (last result,result#1)
      --elapsedTime last reduceOverlap(p,M,I, Module => F1)

    );
    
    initialSyz := for p in inputSyz list (
	g := (F1_(position(newCompList, f -> f == first M#0)))*(p#0);
	for i from 1 to #p - 1 do(g = g + (F1_(position(newCompList, f -> f == first M#i)))*(p#i));
	(g,p)
    );
	 --error "err" ;
    return select(syzygies|vertexSyz(M1,F1)|initialSyz, f -> last f != toList(#M1:0))
)


getSyzMatrix = method(Options => {DegreeLimit => 8})
getSyzMatrix(PAMatrix,List) := opts -> (M,I) -> (
    A := M.ring;
    -- build a function that builds the component hash for a matrix
    --F0 := freePAModule(A,M.numRows);
    F0 := M.module;
    getlist := getVecListFromMatrix(M,F0);
    veclist := for p in getlist list sumPAVector(p);
    compHash := hashTable apply(#veclist, i -> (i,last sort schreyerTerms(veclist#i, (class veclist#i)#"componentHash")));
    getsyz := getSyzygies(veclist,I,opts);
    getsyz = getVecFromTrack(getsyz);
    if getsyz == {} then return paMatrix{{0_A}};
    numcol := #getsyz;

    makeMatrix(getsyz,compHash)   
)

makeMatrix = method()
makeMatrix(List) := L -> (
    F := class first L;
    A := F.ring;
    numcol := #L;
    numrow := max for p in L list maxComponent(p);
    result := for i from 0 to numrow list (
	      	  for j from 0 to numcol -1 list selectComponent(L#j,i)
	      );
    finalList := for p in result list putInPathAlgebra(A,p);   
    return paMatrix(finalList)   
)  
 
makeMatrix(List,ZZ) := (L,numrow) -> (
    F := class first L;
    A := F.ring;
    numcol := #L;
    result := for i from 0 to numrow - 1 list (
	      	  for j from 0 to numcol -1 list selectComponent(L#j,i)
	      );
    finalList := for p in result list putInPathAlgebra(A,p);   
    return paMatrix(finalList)    
)

makeMatrix(List,HashTable) := (L,compHash) -> (
    F := class first L;
    A := F.ring;
    numcol := #L;
    numrow := max for p in L list maxComponent(p);
    result := for i from 0 to numrow list (
	      	  for j from 0 to numcol -1 list selectComponent(L#j,i)
	      );
    finalList := for p in result list putInPathAlgebra(A,p);   
    return paMatrix(finalList,compHash)
)
makeMatrix(List,HashTable,ZZ) := (L,compHash,numrow) -> (
    F := class first L;
    A := F.ring;
    numcol := #L;
    result := for i from 0 to numrow - 1 list (
	      	  for j from 0 to numcol -1 list selectComponent(L#j,i)
	      );
    finalList := for p in result list putInPathAlgebra(A,p);   
    return paMatrix(finalList,compHash)    
)   

PAMatrix | PAMatrix := (A,B) ->(
    ARowEntries := A.entries;
    BRowEntries := B.entries;
    ANumRows := #ARowEntries;
    BNumRows := #BRowEntries;
    if ANumRows != BNumRows then error "Expected same number of rows.";
    result := apply(ANumRows, i -> ARowEntries#i | BRowEntries#i);
    return paMatrix(result)
)
	  
PAMatrix || PAMatrix := (A,B) -> (
    ARowEntries := A.entries;
    BRowEntries := B.entries;
    ANumCols := #(first ARowEntries);
    BNumCols := #(first BRowEntries);   
    if ANumCols != BNumCols then error "Expected same number of columns.";
    return paMatrix(ARowEntries | BRowEntries)
)

PAMap PAMatrix := (phi,M) -> paMatrix applyTable(M.entries,f -> phi f)

getVecFromTrack = method()
getVecFromTrack(List) := L -> L / first

getMinSyzygies = method(Options => {DegreeLimit => 8})
getMinSyzygies(List,List) := opts -> (M,I) -> (
    F := class M#0;
    edgeList := F.ring#"edgeGens";
    Nsyz := getSyzygies(M,I);
    --error "err1";
    NsyzM := select(flatten for p in edgeList list apply(Nsyz, m-> (first m)*p), m -> m != 0);
    --error "err2";
    NsyzMgb := buchPAModule(NsyzM,I,DegreeLimit => opts#DegreeLimit);
    --error "err3";
    return (select(apply(Nsyz, m -> first reduceOverlap(m,NsyzMgb,I,{})), f -> f != 0))

)

getMinSyzMatrix = method(Options => {DegreeLimit => 8})
getMinSyzMatrix(PAMatrix,List) := opts -> (M,I) -> (
    A := M.ring;

    F0 := M.module;
    getlist := getVecListFromMatrix(M,F0);
    veclist := for p in getlist list sumPAVector(p);
    compHash := hashTable apply(#veclist, i -> (i,last sort schreyerTerms(veclist#i, (class veclist#i)#"componentHash")));
    getsyz := getMinSyzygies(veclist,I,opts);
 
    if getsyz == {} then return paMatrix{{0_A}};
    numcol := #getsyz;

    makeMatrix(getsyz,compHash)
)

PAMap Sequence := (phi,l) -> apply(l,m-> phi m)

ambient PAMap := phi -> paMap(phi.target, ambient phi.source, phi#"vertexMap", phi#"edgeMap")

isWellDefined PAMap := f -> (
   -- here we check that the image of the vertices defining an edge define the image of the edge,
   -- as well as checking that the defining ideal goes to zero.
   A := f.source;
   B := f.target;
   result := all(A#"edgeGens", x -> f(verticesLabels x) == verticesLabels(f(x)));
   if not result then return result;
   if instance(A,PathAlgebra) then return true;
   phi := ambient f;
   all(A.ideal.generators, F -> phi F == 0_B)
)

startVertexLabel = method()
startVertexLabel PAElement := f -> (class f)#"vertexGens"#(startVertex leadPath f)
startVertexLabel PAVector := f -> (class f).ring#"vertexGens"#(startVertex (leadModMon f).path)

endVertexLabel = method()
endVertexLabel PAElement := f -> (class f)#"vertexGens"#(endVertex leadPath f)
endVertexLabel PAVector := f -> (class f).ring#"vertexGens"#(endVertex (leadModMon f).path)

verticesLabels = method()
verticesLabels PAPath :=
verticesLabels PAElement := 
verticesLabels PAVector := f -> (startVertexLabel f, endVertexLabel f)


-- for this one, we only check if the endvertex are the same.
toUniform PAVector := f -> (
    uniformHash := new MutableHashTable from {};
    for p in terms f do (
	endvertex := endVertex (first keys p.terms).path;
	if not uniformHash#?endvertex then uniformHash#endvertex = p else uniformHash#endvertex = uniformHash#endvertex + p;
	
    );
    sort values uniformHash

)

-- both edgesOrigin and edgesTerminus assume uniform

originTerminus = method()
originTerminus(PAElement) := f ->(
    key := keys f.terms;
    ver := first key;
    if #(key) > 1 then error "Input must be one vertex.";
    if ver#"edgeList" != {} then error "Input must be a vertex.";
    vernum := ver#"vertex";
    return (select(f.ring#"edgeGens", p -> startVertex p == vernum), select(f.ring#"edgeGens", p -> endVertex p == vernum))
)

edgesOrigin = method()
edgesOrigin(PAElement) := f -> first originTerminus f

edgesTerminus = method()
edgesTerminus(PAElement) := f -> last originTerminus f
  

endVertex PAVector := f -> endVertex (leadModMon f).path

-- assuming input is uniform
notEndVertex = method()
notEndVertex(PAVector) := f -> (
    fvertex := endVertex f;
    return drop((class f).ring#"vertexGens",{fvertex,fvertex})
)    

notEndVertex(PAElement) := f -> (
    fvertex := endVertex f;
    return drop((class f)#"vertexGens",{fvertex,fvertex})
)    


-- this overlap is used in getGammas (not the usual overlap relation)
gammaOverlap = method()
gammaOverlap(PAElement,PAElement):= (p,q) ->(
    A    := p.ring;
    lelp := leadEdgeList(p);
    lelq := leadEdgeList(q);
    R    := findOverlaps(lelp,lelq);
    result:= for i from 0 to length R - 1 list (bel  := take(lelp,{0,length p - R#i - 1}));
    overlaps := for l in result list putInPathAlgebra(A,l)*(leadTerm q);
    return overlaps
)

--- GTN === GammaTreeNode
GTN = new Type of HashTable
gtn = method()
gtn (GTN, PAElement) := (L, w) -> (
    new GTN from { (symbol left) => L,
	                  "word" => L#"word" * w,
		   "gamma" => L#"gamma" + 1,
		   (symbol right) => w }
)
gtn PAElement := v -> (
    if not isVertex v then error "Expected a vertex.";
    new GTN from { (symbol left) => null,
	                  "word" => v,
		   "gamma" => 0,
		   (symbol right) => null }
)

--need to change networker here: v*g=g does not make sense in 2 vertex case.
isVertex GTN := g -> g.left === null
netWorker = g -> if isVertex g then net "" else netWorker(g.left) | (if g#"gamma" != 1 then net "." else net "") | (net g.right)
net GTN := g -> if isVertex g then (net g#"word") else netWorker(g)

GTN ? GTN := (g,h) -> g#"word" ? h#"word"

GTN * PAElement := (L,g) -> (
    new GTN from { (symbol left) => L.left,
	                  "word" => (L#"word") * g,
		   "gamma" => L#"gamma",
		   (symbol right) => (L.right) * g }
)


-- easily get m-1 and m-2 chain from a gtn
-- have chains point to one another.  For example, the chain g = w.z.y.x in Gamma_4 should have g.left === w.z.y in Gamma_3
-- getGammas ({w,z,y,x}, I)


-- if we have a m-chain and we want m-1 and m-2, then it should be just .left and .left.left
-- should check it when we create m-chain from m-1 chain.
getm1chain = method()
getm1chain(GTN) := f -> f.left
getm2chain = method()
getm2chain(GTN) := f -> f.left.left

checkTip = method()
checkTip(PAElement,List):=(f,L) ->(
    L = apply(L,m-> leadTerm m);
    result := false;
    for p in L do (if isSuffix(p,f) then result = true);
    return result
) 

gammaOverlap(GTN,PAElement):= (p,q) -> (
    A    := q.ring;
    lelp := leadEdgeList(p#"word");
    lelq := leadEdgeList(q);
    R    := findOverlaps(lelp,lelq);
    result := for i from 0 to length R - 1 list (bel  := take(lelq,-(length q - R#i)));
    overlaps := for l in result list gtn(p,putInPathAlgebra(A,l));
    return overlaps   

)

gammaOverlap(GTN,GTN):= (p,q) -> gammaOverlap(p,q#"word")

--overlap between a gtn and the list of defining equations
gammaOverlap(GTN,List):= (p,L) -> (
    if L == {} then error "need a non-empty list.";
    A := (first L).ring;
    lelp := leadEdgeList(p#"word");
    ovlp := {};
    for q in L do (
	lelq := leadEdgeList(q);
	R    := findOverlaps(lelp,lelq);
    	result := for i from 0 to length R - 1 list (bel  := take(lelq,-(length q - R#i)));
	result = apply(result,l->putInPathAlgebra(A,l));
	for l in result do (if checkTip((p.right)*l,L) then ovlp = ovlp | {gtn(p,l)});
    );
    return ovlp
)


-- this computes all of the Gamma sets up to Gamma_n
-- I is the list of defining equations of the algebra
-- n is the chain limit
getGammas = method(Options => {DegreeLimit => 8,ChainLimit => 20})
getGammas(List,List) := opts -> (G,I) -> (
    -- n considered >= 3
    A := (first I).ring;
    I = apply(I,m->leadTerm m);
    --G0 := apply(A.vertexGens,v -> gtn v);
    G0 := {};
    G1 := {};
    for i from 0 to #G - 1 do (
	g := gtn(A#"vertexGens"#(startVertex G#i));
	h := gtn(g,G#i);
	G0 = G0|{g};
	G1 = G1|{h};
	
    ); 
    G0 = unique G0;
    G1 = unique G1;       
    -- build both of these at the same time to make sure they point to the right place.
    --G0 := unique apply(G, m -> gtn(A.vertexGens#(startVertex m)));
    --G1 := flatten apply(G,m -> gtn(G0#(startVertex m),m));  
  
    G2 := flatten for i from 0 to #G1-1 list(
	         flatten for j from 0 to #I -1 list (if isPrefix(G1#i,I#j) then  gtn(G1#i,putInPathAlgebra(A,drop(leadEdgeList I#j,length(G1#i#"word")))) )
    );
    G2 = delete(null,G2);
    G2 = select(G2, m -> length m <= opts#DegreeLimit);
    result := {G0,G1,G2};
    n := opts#ChainLimit;

    for k from 3 to n do (
	G := flatten for i from 0 to #result#(k-1) - 1 list (
	                      if (length result#(k-1)#i) < opts#DegreeLimit then  gammaOverlap(result#(k-1)#i,I)	
			   );
	G = sort delete(null,G);
	if G == {} then return result;
	i := 0;
        while i < (#G - 1) do (
	    j := i + 1;
	    while j < #G do (
		if isPrefix(G#i,G#j) then
		   G = drop(G,{j,j})
		else
		   j = j + 1;
	    );
	    i = i + 1;
	);
        G = select(G, m -> length m <= opts#DegreeLimit);
	G = delete(null,G);
	if G == {} then return result;
        result = result | {G};
    );
    return apply(result, m -> sort m)
)

-- this version restricts degree
   -*
getGammas1 = method(Options =>{ChainLimit => 5})
getGammas1(List,List,ZZ) := opts -> (G,I,n) -> (
    CL := opts#ChainLimit;
    A := (first I).ring;
    I = apply(I,m->leadTerm m);
    termsByDegree := new MutableHashTable from {};
    G0 := {};
    G1 := {};
    for i from 0 to #G - 1 do (
	g := gtn(A.vertexGens#(startVertex G#i));
	h := gtn(g,G#i);
	G0 = G0|{g};
	G1 = G1|{h};	
    ); 
    G0 = unique G0;
    G1 = select(unique G1, m -> length m < opts#ChainLimit);
    G2 := flatten for i from 0 to #G1-1 list(
	         flatten for j from 0 to #I -1 list (
		     if isPrefix(G1#i,I#j) then 
		         newel := gtn(G1#i,putInPathAlgebra(A,paPath(A.graph,drop(leadEdgeList I#j,length(G1#i.word)))));
    	    	     	 if newel === null then newel
    	    	     	 else if (length newel) <= opts#ChainLimit then newel
		     
		 )
    );
    G2 = unique delete(null,G2);
    result := {G0,G1,G2};
        error "err";
	
    termsByChain := new MutableHashTable from {0 => G0, 1 => G1, 2 => G2};
    for i from 0 to 2 do (
	apply(termsByChain#i, m ->  (if termsByDegree#?(length m,i) then termsByDegree#(length m,i) =  termsByDegree#(length m,i)|{m}
	     	                     else termsByDegree#(length m,i) = {m}))
    );
 
    for k from 3 to n do (
	for l from 3 to CL do (
	    G := flatten for i from 0 to #termsByChain#(l-1) - 1 list gammaOverlap(result#(l-1)#i,I);
	    

   

    return (termsByDegree,termsByChain)

)  
*-	

isPrefix(PAElement,PAElement) := (f,g) -> isPrefix(leadEdgeList f,leadEdgeList g)
isPrefix(GTN,PAElement) := (f,g) -> isPrefix(f#"word",g)
isPrefix(GTN,GTN) := (f,g) ->  f.left === g.left and isPrefix(f#"word",g#"word")


factorGTN = method()
factorGTN(GTN) := f ->(f.left,f.right)
factorGTN(GTN,PAElement) := (f,g)->(f.left,(f.right)*g)

GTNVec = new Type of HashTable
gtnVec = method()
-- gtns is a list of GTNs
-- coeffs is a list of PAElements that have a single term (coefficient + PAPath)
gtnVec (List, List) := (gtns, coeffs) -> (
   if #gtns != #coeffs then error "Expected lists of the same length";
   new GTNVec from {
       (symbol terms) => new HashTable from apply(#gtns, i -> (gtns#i, coeffs#i)),
       (symbol cache) => new CacheTable from {}
   }
)

GTNVec + GTNVec := (gs, hs) -> new GTNVec from HashTable { (symbol terms) => merge(gs.terms,hs.terms,sum) }

--leadTerm GTNVec := gs -> (
--   if gs.cache#?"leadTerm" then return gs.cache#"leadTerm";
--   myTerms := apply(terms gs, p -> (p_0).word * 
--)

--for eta2 we need L = Gamma2
eta = method()
eta(GTN,List,List) := (f,L,I) -> (
    findSubGTN := false;
    reducedGTN := ringReduce(f,I);
    rFactor := reducedGTN.right;
    
    lel := leadEdgeList rFactor;
    for p in L do (if f.left === p.left then (
	               if (take(lel,length p.right) == leadEdgeList p.right) then 
	                   return gtn(p,removeEdge(rFactor,length p.right))
		  )
    )
)

-- this function remove the first n edge of a PAElement
removeEdge = method()
removeEdge(PAElement,ZZ) := (f,n) -> (
    A := f.ring;
    lel := leadEdgeList f;
    return putInPathAlgebra(A,drop(lel,n))
    
)


GAMMA = new Type of HashTable

getGammaHash = method(Options => options getGammas)
getGammaHash(List,List) := opts -> (G,I) -> (
    gamma := getGammas(G,I,opts);
    --error "err";
    new GAMMA from {
       (symbol terms) => new HashTable from apply(#gamma, i -> (i,gamma#i)),
       "delResults" => new CacheTable from {},
       --(symbol delResults) => new HashTable from {}
       (symbol ideal) => makeMonic I
    }
)	

--(degree,chain)
getDegreeGammaHash = method()
getDegreeGammaHash(GAMMA) := GAM -> (
    tempHash := partition (m -> (m_0, m_1), sort flatten apply(pairs GAM.terms, p -> apply(last p, q -> (length q, first p, q))));
    degreeHash := applyValues(tempHash, p -> apply(p, last));
    new GAMMA from {
       (symbol terms) => GAM.terms,
       "delResults" => GAM#"delResults",
       (symbol ideal) => GAM.ideal,
       "degreeTerms" => degreeHash,
       (symbol degree) => first max keys tempHash
    }
)   
    
--for del2 we need L = Gamma1
-- actually we need all Gamma less than n

--the gtn input should be an element in gamma

del = method(Options => {DegreeLimit => 8})
del(GTN,GAMMA) :=  opts -> (f,G) -> (
    I := G.ideal;
    A := (first I).ring;
    if G#"delResults"#?f then return (true,G#"delResults"#f);
    n    := f#"gamma";
    toDo := {};
    results := {};
    if not member(f,G.terms#n) then error "f should be an element in GAMMA.";
    if n == 1 then (if not G#"delResults"#?f then G#"delResults"#f = {(f.left,f.right)};
	            return (true,G#"delResults"#f)
		    );
    results = results | {(f.left,f.right)} ;
    toDo = apply(last del(f.left,G),m ->(first m,(last m)*(f.right)));
    --error "err3";
    toDo = apply(toDo,m-> ringReduce(m,I));
    toDo = select(toDo, m -> last m != 0_A);
    --error "err";
    newResults :={};
    
    --toDo = ringInterReduce(toDo,I);

    while toDo != {} do (
	--error "err6";
	toDo = sortAccordingTo(sepTensorPairs(toDo), f -> ((f#0)#"word")* leadMonomial (f#1));
	if toDo == {} then break;
	--if n == 5 then error "err2";
	next := last toDo;
	reduced := false;
	local delResultPAE;
	local nextPAE;
	local gtncoeff;
	local newToDo;
	--if n == 5 then error "err2";    	
	for p in G.terms#(n-1) do (
	    if not reduced then (
	    if not G#"delResults"#?p then del(p,G);
	    delRes := G#"delResults"#p;
	    subGTN := isSubGTN(last G#"delResults"#p,next);
	    --if n == 2 then error "err5";
	    if subGTN then (
		delResultPAE = last last G#"delResults"#p;
		nextPAE = last next;		
		gtncoeff = last prefixOverlap(delResultPAE,nextPAE);
		newToDo = apply(G#"delResults"#p,m->(first m,(-1)*(last m)*gtncoeff));
		--if n == 3 then error "err1";		
	        newResults = newResults|{(p,gtncoeff)};
		toDo = toDo | newToDo;
		reduced = true;
	      );
	    );
	);
    	--if n == 3 then error "look";
	if (not reduced) then error "uhoh";    
    	if (reduced == false and (length first next + length last next) > opts#DegreeLimit) then return (false, "Reached Degree/Length Limit");
	toDo = ringInterReduce(toDo,I);
	toDo = select(toDo, m -> last m != 0_A);
        --if n == 5 then error "err7";		    
    );
    results = results | apply(newResults, m-> (first m,(-1)*(last m)));
    --results = sort ringInterReduce(results,I);
    results = sortAccordingTo(ringInterReduce(results,I), f -> (f#0#"word") * leadMonomial (f#1));
    G#"delResults"#f = results;
    return (true,results)
)
-- inputs are of the form (gtn,PAElement)
isSubGTN = method()
isSubGTN(Sequence,Sequence) := (f,g) -> (first f === first g) and isPrefix(last f, last g)
isSubGTN(List,Sequence) := (L,g) -> isSubGTN(last L, g)
--    for i from 0 to #L-1 do (if isSubGTN(L#i,g) then return (true,i));
--    return (false,-1)
--)


-- find the preimage of a GTN over del
-- todo: this one only works for 1-1. Need to do the general case.
delPreImage = method()
delPreImage(GTN,GAMMA) := (f,G) -> (
    delPairs := pairs G#"delResults";
    for p in delPairs do (if any(last p,m -> first m === f) then return first p);
    error "Not in the image";
)

-- done
-- I is the list of defining equations of the algebra
ringReduce = method()
ringReduce(GTN,List) := (f,I) -> (
    reduced := false;
    rFactor := f.right;
    while reduced == false do (
	reduced = true;	
        for q in I do  (   checksubpath := isSubPathLeft(leadTerm q,rFactor);
	                   if (first checksubpath == true and checksubpath#1 == symbol <) then (
		  	   rFactor = rFactor-(checksubpath#2)*q*(checksubpath#3);
			   reduced = false;
			   );
    	);
    ); 
    return gtn(f.left,rFactor)
)
 
ringReduce(PAElement,List) := (f,I) -> f % I
-*
    reduced := false;
    A := f.ring;
    if f == 0_A then return f;
    while reduced == false do (
    reduced = true;	
        for q in I do ( ltf := leadTerm f;
	                checksubpath := isSubPathLeft(leadTerm q,ltf);
	                if (first checksubpath == true and checksubpath#1 == symbol <)  then (
		        --monicf := makeMonic leadTerm f;
		        redResult := isSubpath(leadTerm q,ltf);	
		        f =ltf-(redResult#2)*q*(redResult#3);
			if f == 0_A then return f;
			reduced = false;
			);
    	);
    ); 
    return f
)*-

ringReduce(Sequence,List) := (s,I) -> (first s,ringReduce(last s,I))

-- I is the list of defining equations of the algebra
ringInterReduce = method()
ringInterReduce(List,List) := (L,I) -> (
    if L == {} then return {};
    A := (last first L).ring;
    L = reverse sort L;
    final := {};
    while L != {} do(
    	f := first L;
    	L = drop(L,1);
	--error "err1";
	for p in L do if first p === first f then (f = (first f, last f + last p);
	                                           L = delete(p,L);
	              				   );
	final = final | {f};
	--error "err2";
    );
    --error "err1";
    result := apply(final,m-> ringReduce(m,I));
    --error "err2";
    result = select(result, m-> last m != 0_A);
    --error "err3";
    return result
)

prefixOverlap = method()
prefixOverlap(PAElement,PAElement) := (f,g) -> (
    A := f.ring;
    if not isPrefix(f,g) then return (false,-1);
    if length f == length g then return (true,((leadCoefficient g)*(leadCoefficient f)^(-1)*A#"vertexGens"#(endVertex(f))));
    --if isPrefix(f,g) and (length f == length g) then return (true,g);
    return (true,(leadCoefficient g)*(leadCoefficient f)^(-1)*putInPathAlgebra(A,drop(leadEdgeList g,length(f))))
)


allDelta = method(Options => {DegreeLimit => 5})
allDelta(GAMMA,ZZ) :=  opts -> (G,m) -> (
    maxTermNum := max keys G.terms;
    m = min(m,maxTermNum);
    n := 1;
    while n <= m do (
    	for p in (G.terms)#n do del(p,G,opts);
	n = n + 1;
    );
    return G
)
allDelta(GAMMA) := G -> allDelta(G,max keys G.terms)


allDeltaMatrices = method(Options => {LengthLimit => infinity})
allDeltaMatrices GAMMA := opts -> G -> (
   A := (first G.ideal).ring;
   n := max keys G.terms;
   apply(toList(1..n), i -> transpose paMatrix apply(apply(select(G.terms#i, t -> length t <= opts#LengthLimit), t -> last del(t,G)), vec -> (
       mutList := new MutableList from toList ((#(G.terms#(i-1))):0_A);
       scan(vec, c -> (pos := position(G.terms#(i-1), m -> m === c_0); mutList#pos = c_1));
       toList mutList)))
)
allDeltaMatrices (GAMMA,ZZ) := opts -> (G,n) -> (
   A := (first G.ideal).ring;
   apply(toList(1..n), i -> transpose paMatrix apply(apply(select(G.terms#i, t -> length t <= opts#LengthLimit), t -> last del(t,G)), vec -> (
       mutList := new MutableList from toList ((#(G.terms#(i-1))):0_A);
       scan(vec, c -> (pos := position(G.terms#(i-1), m -> m === c_0); mutList#pos = c_1));
       toList mutList)))
)

sepTensorPairs = method()
sepTensorPairs List := L -> (
    newL := {};
    for p in L do (tensorPair := apply(terms last p, m -> (first p,m));
    	    	   newL = newL |tensorPair;	
    );
    return newL
)

length GTN := f -> (
    if f#"gamma" == 0 then return 0;
    return (length f.left + length f.right)
)

-- sequence provides GAM.terms#?#? first entry is chain number
delResultsCheck = method()
delResultsCheck (GAMMA,Sequence) := (GAM,f) -> (
    L := last del(GAM.terms#(first f)#(last f),GAM);
    result :=sort flatten apply(L, m->(apply(last del(first m,GAM), n -> (first n,(last n)*(last m)))));
    return (result, ringInterReduce(result,GAM.ideal))
)

-- do we need a multidegree? Currently it used total degree
bettiTally = method()
bettiTally(GAMMA) := G -> (
    if not G#?"degreeTerms" then G = getDegreeGammaHash(G);
    new BettiTally from  apply(pairs G#"degreeTerms",m-> (last first m,{first first m},first first  m) => #(last m))
    -- sum apply(numVertices A, i -> bettiTally(G,i))
)   

bettiTally(GAMMA, ZZ) := (G,endVert) -> (
    if not G#?"degreeTerms" then G = getDegreeGammaHash(G);
    -- change to only include those chains that end at endVert
    new BettiTally from  apply(pairs G#"degreeTerms",m-> (last first m,{first first m},first first  m) => #(selectEndVertex(last m,endVert)))
)

endVertex GTN := f -> endVertex f#"word"

selectEndVertex = method()
selectEndVertex(List,ZZ) := (L,n) -> select(L,m-> (endVertex m) == n)


findReducePositions = method()
findReducePositions(GAMMA,ZZ) := (G,endVert) -> findReducePositions  bettiTally(G,endVert)
findReducePositions(BettiTally) := B -> (
    degPairs := sort select(pairs B, m -> (last m != 0));
    simpdegpairs := apply(degPairs, m-> (first first m,last first m));
    redPositions := for p in degPairs list (
	    	    	chain := first first p;
			deg := last first p;
			if any(simpdegpairs, m-> m == (chain - 1,deg)) then p
		    );
    redPositions = sort delete(null,redPositions);
    return redPositions
)

gtnChainList = method()
gtnChainList (GTN) := g -> (
    results := {g};
    for n from 2 to g#"gamma" do (
	results = results | {(last results).left};
    );
    return results
)	

-- f is a function taking elements of L to another type
-- that we know how to sort already
sortAccordingTo = method()
sortAccordingTo (List, Function) := (L, f) -> (
   (sort apply(L, l -> (f l, l))) / last
)

getVertexInfo = method()
getVertexInfo (List) := L -> apply(L,m->(m,startVertex m,endVertex m))
    
edgesEndsAt = method()
edgesEndsAt (PathAlgebra,ZZ) := (A,n) -> edgesEndsAt(A,(A#"vertexGens")#n)
edgesEndsAt (PathAlgebra,PAElement) := (A,v) -> select(apply(A#"edgeGens", m-> m*v),n -> n!=0_A)

--v is the vertex we wants to end at
--n is the degree of the path
pathsEndsAt = method()
pathsEndsAt (PathAlgebra,PAElement,ZZ) := (A,v,n) -> (
    results := edgesEndsAt(A,v);
    for m from 2 to n do(
	results = flatten apply(results, m-> apply(A#"edgeGens",n->n*m));
	results = select(results,n->n!=0_A);    
    );
    return results
)
pathsEndsAt (PathAlgebra,List,ZZ) := (A,L,n) -> (
    for m from 1 to n do(
	L = flatten apply(L, m-> apply(A#"edgeGens",n->n*m));
	L = select(L,n->n!=0_A);
    );
    return L
)
    
pathsEndsAt (PathAlgebra,ZZ,ZZ) := (A,m,n) -> pathsEndsAt(A,(A#"vertexGens")#m,n)

findGAMMABases = method()
findGAMMABases (GTN,GAMMA) := (g,GAM) -> (
    gtnLength := length g;
    n := g#"gamma";
    A := g#"word".ring;
    --gtn in n-1 chain that have length leq g
    gtnOptions := apply(select(GAM.terms#(n-1), m -> length m <= gtnLength), m->(m,length m));
    lengthList := sort unique apply(gtnOptions,m-> last m);
    
    pathPairs := {(gtnLength - first lengthList,pathsEndsAt(A,endVertex g,gtnLength -(first lengthList)))};
    if #lengthList > 1 then for i from 1 to #lengthList-1 do (
	pathPairs = pathPairs |{(lengthList#i,pathsEndsAt(A,last last pathPairs,lengthList#i - first last pathPairs))};
    );
    pathPairs = pathPairs | {(0,A#"vertexGens")};
    allPaths := new HashTable from pathPairs;
    results := apply(GAM.terms#(n-1),m->(m, select(allPaths#(gtnLength -length m),l->(m*l)#"word" != 0_A)));
    return results
)

--n chain with m length with endvertex v
getReduceTerms = method()
getReduceTerms(GAMMA,ZZ,ZZ,ZZ) := (G,v,n,m) -> select(G.terms#n, l -> length l == m and endVertex(l#"word") == v)
    
getReduceMatrix = method()
getReduceMatrix(GAMMA,List) := (G,L) -> (
    A := (first L)#"word".ring;
    n := (first L)#"gamma";
    return transpose matrix apply(L, l -> (
    --return transpose paMatrix apply(L, l -> (
    	    mutList := new MutableList from toList ((#(G.terms#(n-1))):0_(A.CoefficientRing));
	    --mutList := new MutableList from toList ((#(G.terms#(n-1))):0_(A));
    	    scan(last del(l,G), c -> (pos := position(G.terms#(n-1), m -> m === c_0); if (length c_1 == 0) then mutList#pos = leadCoefficient(c_1)));
	    --scan(last del(l,G), c -> (pos := position(G.terms#(n-1), m -> m === c_0); mutList#pos = c_1));
    	    toList mutList
	    )
	)
  
)

getReducedBT = method()
getReducedBT(GAMMA,ZZ) := (G,v) -> (
    BT := bettiTally(G,v);
    reduceList := findReducePositions(BT);
    newList := {};
    for p in reduceList do(
	r := rank getReduceMatrix(G,getReduceTerms(G,v,first first p,last first p));
        BT = BT + new BettiTally from {(first p, -r),((first first p -1,(first p)#1,last first p), -r)};
    );
    return BT
)
    

--WYM
-*
findReducePositions = method()
findReducePositions(BettiTally,ZZ) := (B,n) -> (
    degPairs := sort pairs B;
    simpdegpairs := apply(degPairs, m-> (first first m,last first m));
    ndegPairs := select(degPairs, m->(first first m == n));
    
    
    redPositions := for p in ndegPairs list (
	    	    	chain := first first p;
			deg := last first p;
			if any(simpdegpairs, m-> m == (chain - 1,deg)) then p
		    );
    redPositions = sort delete(null,redPositions);
    return redPositions
)

findReducePositions(BettiTally) := B -> (
    maxDeg := max(apply(keys B, m -> first m));
    return flatten apply(maxDeg,m-> findReducePositions(B,m))
)
*-
--load "/Users/y.w./WFU/Spring2022/M2/PathAlgebrasDoc.m2"
--load "/Users/y.w./github/Pathalgebras/PathAlgebrasDoc.m2"
load "/Users/vaw1/Downloads/m2/Pathalgebras/PathAlgebrasDoc.m2"
--load "PathAlgebrasCheck.m2"  
end



restart
uninstallPackage "PathAlgebras"
restart
debug needsPackage "PathAlgebras"
--check PathAlgebras
installPackage "PathAlgebras"
viewHelp "PathAlgebras"
restart

--xxx

-- things to do with PAIdeals:
-- PathAlgebraQuotient := PathAlgebra / PAIdeal
--   (including operations on elements in a PathAlgebraQuotient)
-- paBasis of a PathAlgebra or PathAlgebraQuotient
--  paBasis(n,A) = compute paBasis(n-1,A), and left multiply by edges to get a list
--                 of monomials of length n, then remove the ones that are divisible by a lead term
--                 in the GB of the defining ideal of A
-- paHilbertSeries (will return an nxn matrix where n = #vertices)

-- Future TODO: 
-- 1.Matrix printing

restart
debug needsPackage "PathAlgebras"
M = matrix {{3}}
G = paGraph({v},{a,b,c},M,Weights=>{1,1,1})
kk = ZZ/32003
R = kk G
I = paIdeal {2*a*b + 3*b*a + 5*c^2,
             2*b*c + 3*c*b + 5*a^2,
             2*c*a + 3*a*c + 5*b^2}
S = R/I


I = I / makeMonic // sort
interreduce I
gbI = {I#0,I#1,I#2}
I = paIdeal I.generators
gbI = sort buchAlgorithm(I,DegreeLimit=>7)
I = paIdeal I.generators
elapsedTime gbI = sort buchAlgorithm(I,DegreeLimit=>8)

paBasis(4,R)
-- leadTerm of an ideal
ltI = apply(gbI, f -> leadTerm f)
-- for every n, one should have
n = 8
#paBasis(n,R,gbI) == binomial(n+2,2)


restart
needsPackage "AssociativeAlgebras"
S = ZZ/32003<|x,y,z|>
J = ideal {2*x*y + 3*y*x + 5*z^2,
     2*y*z + 3*z*y + 5*x^2,
     2*z*x + 3*x*z + 5*y^2}
phi = map(S,R,gens S)
gbJ = sort flatten entries NCGB(J, 7)
netList gbJ
R = S/J
A = ZZ[T]
sum apply(10, i -> (#(flatten entries ncBasis(i,R))*T^i))S_0*(phi ncBasis(2,R))
x^2
ncBasis(2,R)
varMap = n -> S_n;
paMap = f -> sum apply(pairs f.terms, p -> (((p#0).edgeList) / varMap // product) * p#1)
gbJ == gbI / paMap


restart
debug needsPackage "PathAlgebras"
M = matrix {{1,0},{1,1}}
G = paGraph({v,w},{e,f,g},M,Weights=>{2,1,1},Degrees=>{2,1,1})
R = QQ
A = R G

I=paIdeal{e*f,f*g}
A/I


newadjMatrix = matrix{{0,0,0},{3,1,0},{3,1,1}}
coker newadjMatrix

d = 7
paHF = paHilbertSeries(d,A,{})
paHilbertSeries(6,A,{e*f})

-- would be nice to use this code in paHilbertSeries
-- and return it if a recurrence is found


restart
debug needsPackage "PathAlgebras"
M = matrix {{1,0},{1,1}}
--G = paGraph({v,w},{e,f,g},M,Weights=>{2,1,1})
G = paGraph({v,w},{e,f,g},M,Weights=>{2,1,1},Degrees => {2,1,1})
R = QQ
A = R G
d = 20
paHF = paHilbertSeries(d,A,{})
 MatrixExpression table(#A.vertexGens,#A.vertexGens,
                            p -> if paHF_p == 0 then expression paHF_p 
                                 else last toRationalFunction flatten entries sub(last coefficients(paHF_p,Monomials => matrix{apply(d+1,i -> (ring paHF)_0^i)}),ZZ))

toRationalFunction {1,0,1,0,1,0,1,0,1,0,1,0,1}
toRationalFunction {1,1,1,1,1,1,1,1,1,1,1,1,1}
findRecurrence {1,0,1,0,1,0,1,0,1,0,1,0,1}
findRecurrence {0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8}


B = frac(ZZ[T])
M = sub(A.graph.adjacencyMatrix,B)
P = (id_(B^2) - M*T)
P^(-1)
HS = sum apply(10, i -> M^i*T^i)


-- to create the ring elements corresponding to u and v you will need
-- something like putInPathAlgebra(A,paPath(A.graph,"desired edge list"))
putInPathAlgebra(A,paPath(A.graph,{0,1}),3_R)

k = (f+g)^5
#(h.terms)
M^10

apply(A.generators, x -> (v+w)*x)

 -- internally, a path should be a list of integers corresponding
-- to the indices of the edges used.
-- example: M = matrix {{1,0},{1,1}}
-- edge list: {e,f,g} e : v_1->v_1, f : v_1 -> v_2, g : v_2 -> v_2
-- for example the list {0,0,1,2,2,2} <--> the path eefggg
vPath = paPath(G, 0)
ePath = paPath(G, {0})
ePath' = paPath(G, {0})
fPath = paPath(G, {1})
gPath = paPath(G, {2})
v = paElement(G, hashTable { vPath => 1 })
e = paElement(G, hashTable { ePath => 1 })
e' = paElement(G, hashTable { ePath' => 1 })
f = paElement(G, hashTable { fPath => 1 })
g = paElement(G, hashTable { gPath => 1 })

--    3 
-- e e
--  2 3
gamma = pathAlgebra(QQ,graphM)


restart
debug needsPackage "PathAlgebras"
M = matrix {{3}}
G = paGraph({v},{a,b,c},M,Weights=>{1,1,1})
kk = ZZ/32003
R = kk G
I = paIdeal {2*a*b + 3*b*a + 5*c^2,
             2*b*c + 3*c*b + 5*a^2,
             2*c*a + 3*a*c + 5*b^2}
	 
I= I.generators
ov1 = a*I#0 - I#2*c
overlaps(I#0,I#2)
red1 = makeMonic first divAlgorithm(gbI, ov1)
gbI = gbI | {red1}


ov2 = a*I#1 - I#2*b
overlaps((I.generators)#1,(I.generators)#2)
red2 = makeMonic first divAlgorithm(gbI, ov2);
gbI = gbI | {red2};
-- new overlaps --

ov3 = b*b*I#0 - gbI#3*c
overlaps(gbI#0,gbI#3)
red3 = makeMonic first divAlgorithm(gbI,ov3);
gbI = gbI | {red3};


ov4 = b*b*I#2 - gbI#3*a
overlaps(gbI#2,gbI#3)
red4 = makeMonic first divAlgorithm(gbI,ov4);
gbI = gbI | {red4}

ov5 = a*gbI#4 - I#1*b*c
overlaps(gbI#4,gbI#1)
red5 = first divAlgorithm(gbI,ov5)
-- 0 here no new overlaps


ov6 = b*b*gbI#3 - gbI#5*a
overlaps(gbI#3,gbI#5)
red6 = makeMonic first divAlgorithm(gbI,ov6)
gbI = gbI | {red6}

ov7 = b*b*b*gbI#3 - gbI#5*b*a
red7 = makeMonic first divAlgorithm(gbI,ov7)
gbI = gbI | {red7}

ov8 = b*b*gbI#4 - gbI#5*c
overlaps(gbI#4,gbI#5)
red8 = makeMonic first divAlgorithm(gbI,ov8)
gbI = gbI | {red8}

ov9 = b*b*b*gbI#4 - gbI#5*b*c
red9 = makeMonic first divAlgorithm(gbI,ov9)
gbI = gbI | {red9}

composePath:
        M = matrix {{1,0},{1,1}}
	G = paGraph({v,w},{e,f,g},M,Weights=>{2,1,1},Degrees=>{2,1,1})
	L=paPath(G,{0,1,2})
	P=paPath(G,{2,2,2})
	composePath(L,J)
	
   
--- Build S from the paper (top of page 49)


restart
debug needsPackage "PathAlgebras"
adj = matrix {{1,0},{1,1}}
G = paGraph({v,w},{e,f,g},adj)
R = QQ
A = R G
--I=paIdeal{e*f,f*g}
--B = A/I
--  a_1e^2 = 0, a_1e*f - a_2e^2 = 0
-- a_(1,1)e^2, a_(1,1)e*f + a_(2,1)g^2 - a_(1,2)e^2
M = paMatrix {{e^2,e*f + g^2},{0_A,-e^2}} -- fix this output!!

-- first subscript indicates start vertex of path, second indicates component number
numgensM = numrows M
numvertsG = numVertices G
newAdj = (matrix {toList((numvertsG + 1):0)}) || ((transpose matrix {toList(numvertsG : numgensM)}) | adj)
MG = paGraph({x} | G.vertexLabels, toList(a_(1,1)..a_(numvertsG,numgensM)) | G#"edgeLabels", newAdj)
MA = R MG
phi = paMap(MA,A,drop(MA.vertexGens,1),drop(MA#"edgeGens",numvertsG*numgensM))
entsM = transpose entries M
IM = paIdeal apply(numcols M, j -> sum apply(numgensM, i -> sum apply(toUniform M_(i,j), u -> a_(startVertex u + 1,i + 1)*(phi u))))
buchAlgorithm IM


restart
debug needsPackage "PathAlgebras"

adj = matrix {{1,0},{1,1}}
G = paGraph({v,w},{e,f,g},adj)
R = QQ
A = R G
L=e*f*g+e*f
M = paMatrix {{e^2,e*f},{0_A,-g^2}} 
numgensM = numrows M
numvertsG = numVertices G
newAdj = (matrix {toList((numvertsG + 1):0)}) || ((transpose matrix {toList(numvertsG : numgensM)}) | adj)
MG = paGraph({x} | G.vertexLabels, toList(a_(1,1)..a_(numvertsG,numgensM)) | G.edgeLabels, newAdj)
MA = R MG
phi = paMap(MA,A,{v,w},{e,f,g})
h = A.vertexGens#0 + (A.edgeGens#0)*(A.edgeGens#1) - (A.edgeGens#2)^2
phi h
psi = paMap(A,MA,{0_A} | A.vertexGens,{0_A,0_A,0_A,0_A} | A.edgeGens)
psi f

paMapKernel(phi,L) 
paMapNoneKernel(phi,L)

positions(phi.edgeMap, i-> i == 0_(phi.target))


restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
I = paIdeal {x*y - y*x, x*z - z*x, y*z - z*y}
M = paMatrix {{x,y,z}} -- f_1 : F_1 --> F_0
M = paMatrix {{-y,-z,0_A},{x,0_A,-z},{0_A,x,y}} -- f_2 : F_2 --> F_1
M = paMatrix {{z},{-y},{x}} -- f_3 : F_3 --> F_2

-- first subscript indicates start vertex of path, second indicates component number
numgensM = numrows M
numvertsG = numVertices G
newAdj = (matrix {toList((numvertsG + 1):0)}) || ((transpose matrix {toList(numvertsG : numgensM)}) | adj)
MG = paGraph({w} | G.vertexLabels, toList(a_(1,1)..a_(numvertsG,numgensM)) | G.edgeLabels, newAdj)
MA = R MG
phi = paMap(MA,A,drop(MA.vertexGens,1),drop(MA.edgeGens,numvertsG*numgensM))
entsM = transpose entries M
IM = paIdeal apply(numcols M, j -> sum apply(numgensM, i -> sum apply(toUniform M_(i,j), u -> a_(startVertex u + 1,i + 1)*(phi u))))
IM = IM + paIdeal apply(I.generators, f -> phi f)
buchAlgorithm IM

-- an overlap looks like
-- (a_(1,1)*x)*y - a_(1,1)*(x*y - y*x) == a_(1,1)*y*x
-- now reduce this overlap:
-- a_(1,1)*y*x = (a_(1,1)*y)*x

-- a_(1,1)(x*y - y*x) = (a_(1,1)*y)*x - (a_(1,1)*x)*y
-- f_2(\hat(e_1)) = -d_1'*y + d_2'*x

f1 = IM.generators#0
f2 = IM.generators#1
f3 = IM.generators#2

f1*z + a_(1,1)*(IM.generators#5) - f2*y - a_(1,2)*(IM.generators#4) + f3*x + a_(1,3)*(IM.generators#3)

f1*z - f2*y + f3*x

f_2(\hat{e_1}) = d_1'*z - d_2'*y + d_3'*x
(d_i' |-> f's)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - 2*y*x, x*z - 3*z*x, y*z - 5*z*y}
e0 = F0_0
f1 = e0*x
f2 = e0*y
f3 = e0*z
M = {f1,f2,f3}
F1 = A^#(M)
e11 = F1_0 -- maps to f1
e12 = F1_1 -- maps to f2
e13 = F1_2 -- maps to f3
----
-- this is what you do for the overlap between e0*x and xy - yx
isSubModMon (leadModMon f1,leadPath I#0)
findOverlaps(leadModMon f1,leadPath I#0)
syz11 = e11*y
ourComp = (leadModMon f1).component
tempf = f1*y - (F0_ourComp)*(I#0)
isSubModMon (leadModMon tempf,leadPath I#0)
isSubModMon (leadModMon tempf,leadPath I#1)
isSubModMon (leadModMon tempf,leadPath I#2)
isSubModMon (leadModMon f1, leadModMon tempf)
isSubModMon (leadModMon f2, leadModMon tempf)
findOverlaps(leadModMon f2, leadModMon tempf)
syz11 = syz11 - (leadTermCoeff tempf)*e12*x
tempf = tempf - (leadTermCoeff tempf)*f2*x
-- next overlap is between e0*x and xz - zx
modElt = f1
ringElt = I#1
isSubModMon (leadModMon modElt,leadPath ringElt)
findOverlaps(leadModMon modElt,leadPath ringElt)
syz12 = e11*z
ourComp = (leadModMon modElt).component
tempf = modElt*z - (F0_ourComp)*(ringElt)
-- reduce the result
isSubModMon (leadModMon tempf,leadPath I#0)
isSubModMon (leadModMon tempf,leadPath I#1)
isSubModMon (leadModMon tempf,leadPath I#2)
isSubModMon (leadModMon f1, leadModMon tempf)
isSubModMon (leadModMon f2, leadModMon tempf)
isSubModMon (leadModMon f3, leadModMon tempf)
ov = findOverlaps(leadModMon f3, leadModMon tempf)
ourComp = (leadModMon tempf).component
syz12 = syz12 - (leadTermCoeff tempf)*(F1_ourComp)*x
tempf = tempf - (leadTermCoeff tempf)*f3*x
-- next overlap is between e0*y and yz - zy
modElt = f2
ringElt = I#2
isSubModMon (leadModMon modElt,leadPath ringElt)
findOverlaps(leadModMon modElt,leadPath ringElt)

syz13 = e12*z
ourComp = (leadModMon modElt).component
tempf = modElt*z - (F0_ourComp)*(ringElt)
-- reduce the result
isSubModMon (leadModMon tempf,leadPath I#0)
isSubModMon (leadModMon tempf,leadPath I#1)
isSubModMon (leadModMon tempf,leadPath I#2)
isSubModMon (leadModMon f1, leadModMon tempf)
isSubModMon (leadModMon f2, leadModMon tempf)
isSubModMon (leadModMon f3, leadModMon tempf)
ov = findOverlaps(leadModMon f3, leadModMon tempf)
ourComp = (leadModMon tempf).component
syz13 = syz13 - (leadTermCoeff tempf)*(F1_ourComp)*y
tempf = tempf - (leadTermCoeff tempf)*f3*y
-- now done with the process of finding (a GB of) the kernel of the map
-- from A^3 --> A^1 given by the matrix (x  y  z)
-- the kernel has GB given by the syzygies found above:
kerGens = {syz11,syz12,syz13}
F2 = A^#(kerGens)



--the above reduces to the following:
restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - 2*y*x, x*z - 3*z*x, y*z - 5*z*y}
I = {x*y - y*x, x*z - z*x, y*z - z*y}
e0 = F0_0
f1 = e0*x
f2 = e0*y
f3 = e0*z
M = {f1,f2,f3}
F1 = A^#(M)
e11 = F1_0 -- maps to f1
e12 = F1_1 -- maps to f2
e13 = F1_2 -- maps to f3
buchPAModule(M,I)

Goal:

-- F0_0 and F1_0 are the same
M2 = getSyzygies(sort M,I)
-- answer:
M2 = {-2(1, x) + 1(0, y), -3(0, x) + 1(0, z), -5(0, y) + 1(1, z)}
--my answer: {1(1, z) + -5(0, y), 1(2, y) + -2(1, x), -3(0, x) + 1(2, z)}

--looks like we won't have M3 since these can't cancel because of coeffs

M3 = getSyzygies(M2,I)
-- answer:
M3 = { one element }
M4 = getSyzygies(M3,I)
-- answer: 
M4 = {}

----
-- this is what you do for the overlap between e0*x and xy - yx
ao=allSubModMon(f1,I)
ov=ao#0
ov=ao#1
reduceOverlap(ov,M,I)
ao2=allSubModMon(f2,I)
ov=ao2#0
reduceOverlap(ov,M,I)

syz11 = e11*(ov#2)
ourComp = (leadModMon f1).component
tempf = f1*(ov#2) - (F0_ourComp)*(ov#1)
findSubModMon(tempf,M,I)
--we do need a putInPathAlgebra here
syz11 = syz11 - (leadTermCoeff tempf)*e12*x
tempf = tempf - (leadTermCoeff tempf)*f2*x

-- next overlap is between e0*x and xz - zx
ov1=ao#1
syz12 = e11*(ov1#2)
ourComp = (leadModMon ov1#0).component
tempf = (ov1#0)*(ov1#2) - (F0_ourComp)*(ov1#1)
-- reduce the result
findSubModMon(tempf,M,I)

ourComp = (leadModMon tempf).component
syz12 = syz12 - (leadTermCoeff tempf)*(F1_ourComp)*x
tempf = tempf - (leadTermCoeff tempf)*f3*x

-- next overlap is between e0*y and yz - zy
ao1=allSubModMon(f2,I)
ov2=ao1#0
syz13 = e12*(ov2#2)
ourComp = (leadModMon ov2#0).component
tempf = (ov2#0)*(ov2#2) - (F0_ourComp)*(ov2#1)
-- reduce the result
findSubModMon(tempf,M,I)

ourComp = (leadModMon tempf).component
syz13 = syz13 - (leadTermCoeff tempf)*(F1_ourComp)*y
tempf = tempf - (leadTermCoeff tempf)*f3*y
-- now done with the process of finding (a GB of) the kernel of the map
-- from A^3 --> A^1 given by the matrix (x  y  z)
-- the kernel has GB given by the syzygies found above:
kerGens = {syz11,syz12,syz13}
F2 = A^#(kerGens)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{1,1},{0,1}}
G = paGraph({v,w},{e,f,g},adj)
R = QQ
A = R G
M = A^5
M_2
17*M_2*e

e0 = stdBasisVector(A,0)
l = 17*e0*e*f
L=paModMon(0,paPath(A.graph,{0,1}))
K=paModMon(1,paPath(A.graph,{0,1}))
J=paModMon(0,paPath(A.graph,{0,1,2}))
H=paModMon(0,paPath(A.graph,{0,0,1,2}))
L ? K
L ? J
isSubModMon(L,H)
findOverlaps(L,J)
{L,K,J}

a=paVector(M,{17_R},{paModMon(0,paPath(A.graph,{0,1}))})
a==l
b=paVector(M,{17_R},{paModMon(1,paPath(A.graph,{0,1}))})
c=paVector(M,{1_R},{paModMon(0,paPath(A.graph,{0,1,2}))})
d=a+b+c
s=a+c+b

allComponent(d)

selectComponent(d,0)
selectComponent(d,1)
selectComponent(d,17)

selectLeadCom(d)

weight d
listModMon d
leadModMon d
leadTermCoeff d

pathTerms d
s==d
leadTerm d

a + a
a*e
a*f
a*g

3*a

--M * PAPath := (f,g) -> new M from {(symbol terms) => (first keys f.terms) * g };

restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - y*x, x*z - z*x, x*w - w*x, y*z - z*y, y*w - w*y, z*w - w*z}
--I = {x*y-y*x-z^2, x^2-y*z+z*y, x*z+y^2-z*x, y*z*y-z^2*x, y^2*x+y*z^2-z*y*z+z^2*y, y^4-y^2*z*x-y*z^3+z*y*z^2-z^2*y*z, y^3*z-z*y*z*x+z^2*y*x, y*z^3*x+z^2*y^3-z^3*y*x-z^5, y*z^2*y*z-y*z^3*y+z^3*y^2-z^4*x, y^2*z^2*x-z*y*z^3, y^2*z^2*y^2+y*z^2*y^3+z*y*z^4+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y^2-y*z^4*x-z^3*y^3+z^4*y*x+z^6, y^2*z^2*y*x+y^2*z^4-z*y*z^3*y, y*z^3*y*x+y*z^5+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y*z-y*z^4*y-z^2*y^2*z^2-z^2*y*z^2*y-z^3*y^2*z-z^4*y^2+z^5*x}
e0 = F0_0
f1 = e0*x
f2 = e0*y
f3 = e0*z
--f4 = e0*w
M = {f1,f2,f3}
sortM = sort M
perm = apply(M, m -> position(sortM, n -> n == m))
invPerm = apply(#perm, i -> position(perm, j -> j == i))
(0 1 2 3)
perm = {1,2,3,0}
(0 3 2 1)

M2 = getSyzygies(sortM,I,DegreeLimit => 6)
M3 = getSyzygies(M2,I)
M4 = getSyzygies(M3,I)
M5 = getSyzygies(M4,I)

N = paMatrix{{x,y,z}}
N2 = getSyzygiesMatrix(N,I,DegreeLimit => 6)
N3 = getSyzygiesMatrix(N2,I,DegreeLimit => 6)
N4 = getSyzygiesMatrix(N3,I,DegreeLimit => 6)

M = paMatrix {{x,y,z}}
M2 = getSyzygies(M,I,DegreeLimit => 6)

mM2 = M2 * (all the paths in the path algebra)
find a GB of that module (have to think about this)


m = take(M2,-3)
B = A/(paIdeal I)

-- once the matrices are built, multiplying them
-- should give the zero matrix over the quotient A/I

      +-----------------------------+
o15 = |-1(0, x) + 1(1, y) + 1(2, z) |
      +-----------------------------+
      |-1(0, z) + -1(1, x) + 1(2, y)|
      +-----------------------------+
      |1(0, y) + -1(1, z) + 1(2, x) |
      +-----------------------------+

restart
needsPackage "AssociativeAlgebras"
R = QQ<|x,y,z|>
I = ideal {x*y - y*x - z^2, y*z - z*y - x^2, z*x - x*z - y^2}
Igb = NCGB(I,10)
B = R/I
M = matrix {{x,y,z}}
M2 = rightKernel(M, DegreeLimit => 10)



(0, y^3) -- lead term of module elt
y^2 + l.o.t.


(0 component)*y*(gb elt with y^2 lead term)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - y*x, x*z - z*x, y*z - z*y}
--I = {x*y - 2*y*x, x*z - 3*z*x, y*z - 5*z*y}
--I = {x*y-y*x-z^2, x^2-y*z+z*y, x*z+y^2-z*x, y*z*y-z^2*x, y^2*x+y*z^2-z*y*z+z^2*y, y^4-y^2*z*x-y*z^3+z*y*z^2-z^2*y*z, y^3*z-z*y*z*x+z^2*y*x, y*z^3*x+z^2*y^3-z^3*y*x-z^5, y*z^2*y*z-y*z^3*y+z^3*y^2-z^4*x, y^2*z^2*x-z*y*z^3, y^2*z^2*y^2+y*z^2*y^3+z*y*z^4+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y^2-y*z^4*x-z^3*y^3+z^4*y*x+z^6, y^2*z^2*y*x+y^2*z^4-z*y*z^3*y, y*z^3*y*x+y*z^5+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y*z-y*z^4*y-z^2*y^2*z^2-z^2*y*z^2*y-z^3*y^2*z-z^4*y^2+z^5*x}
N = paMatrix{{x,y,z}}
f1 = F0_0*x
f2 = F0_0*y
f3 = F0_0*z
N=getSyzygies({f1,f2,f3},I)
getMinSyzygies({f1,f2,f3},I)
N2 = getSyzMatrix(N,I)
N2 = getMinSyzMatrix(N,I)
N3 = getSyzygiesMatrix(N2,I)
N4 = getSyzygiesMatrix(N3,I)


e0 = F0_0
f1 = e0*x
f2 = e0*y
f3 = e0*z
M = {f1,f2,f3}
M2 = getSyzygies(M,I)
M3 = getSyzygies(M2,I)
M4 = getSyzygies(M3,I)

f4 = e0*x*y
asmm=allSubModMon(f4,I)
F1=A^1
reduceOverlapOnly(asmm#0,{f4},I,F1)

buchPAModule({f4},I)
changeOfPosition(M)
M1= {f1,f3,f2}
changeOfPosition(M1)

F3 = A^3
stdBasisVector(F3,2)


K= new HashTable from {(0,1_A)}
K2 = moduleTrack(N,K)
moduleTrack(N2,K2)

getVecFromMatrix(N2)
K2=getCompHash(N,K)
getCompHash(N2,K2)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
M = A^3

K= new HashTable from {(0,y),(1,x),(2,z)}
F = freePAModule(A,3,K)
f1 = F_0*y
f2 = 2*F_1*z
f3 = F_2*y

f4=f1+f2+f3
schreyerLeadTerm(f4,K)

f5 = F_0*y

f6 = F_1*y
f7 = F_2*x
f8 = f5+f6+f7
schreyerLeadTerm(f8,K)
f7 ? f6
f8 ? f4
f3' = F_2*y + F_2*x
f3 ? f3'
schreyerTerms(f4,K)
schreyerTerms(f8,K)


restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - y*x, x*z - z*x, y*z - z*y}
F1 = A^3
f1 = -F1_0*y + F1_1*x
f2 = -F1_0*z + F1_2*x
f3 = -F1_1*z + F1_2*y
buchPAModule({f1,f2,f3},I)
--- term order is term over position


f0 = F0_0*x*y
buchPAModule({f0},I)

--allSubModMon(f1,I)

--reduceOverlapOnly(oo#0,{f1,f2,f3},I,F1)

f4 = f1*y - F1_1*(x*y - y*x)
 -- lm is (1,yx) [f1 with g1]
f5 = f2*y - F1_2*(x*y - y*x)
f5red = f5 - f3*x  -- lm is (1,zx) [f2 with g1]
f6 = f2*z - F1_2*(x*z - z*x) -- lm is (2,zx) [f3 with g2]

f7 = f4*y - F1_1*y*(x*y - y*x) -- lm is (1,yyx)  [f4 with g1]
f8 = -(f4*z - F1_1*y*(x*z - z*x)) -- lm is (1,yyz) [f4 with g2]
f9 = f5red*y - F1_1*z*(x*y - y*x) -- lm is (1,zyx) [f5 with g1]
f10 = -(f5red*z - F1_1*z*(x*z - z*x)) -- lm is (0,zyz) [f5 with g2]
f11 = f6*y - F1_2*z*(x*y - y*x) -- lm is (2,zyx) [f6 with g1]
f12 = f6*z - F1_2*z*(x*z - z*x) -- lm is (2,zzx) [f6 with g2]

--also the following
f13 = f3*z-F1_2*(I#2) -- lm is (2,zy) [f3 with g3]
f14 = f2*z-F1_2*(I#1) -- lm is (2,zx) [f2 with g2] same as f6

-- +8 more overlaps in the next step (2 from each lead term ending in x) (module with ring elts)

-- ensure input is monic before doing any calculation

overlaps(f1,I#0)


restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - y*x, x*z - z*x, y*z - z*y}
F1 = freePAModule(A,3,hashTable {(0,x),(1,y),(2,z)})
f1 = F1_0*y - F1_1*x
f2 = F1_0*z - F1_2*x
f3 = F1_1*z - F1_2*y
buchPAModule({f1,f2,f3},I)

-
restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - y*x, x*z - z*x, y*z - z*y}
F1 = A^3
f1 = -F1_0*y + F1_1*x
f2 = -F1_0*z + F1_2*x
f3 = -F1_1*z + F1_2*y
buchPAModule({f1,f2,f3},I,DegreeLimit => 5)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y-y*x-z^2, x^2-y*z+z*y, x*z+y^2-z*x, y*z*y-z^2*x, y^2*x+y*z^2-z*y*z+z^2*y, y^4-y^2*z*x-y*z^3+z*y*z^2-z^2*y*z, y^3*z-z*y*z*x+z^2*y*x, y*z^3*x+z^2*y^3-z^3*y*x-z^5, y*z^2*y*z-y*z^3*y+z^3*y^2-z^4*x, y^2*z^2*x-z*y*z^3, y^2*z^2*y^2+y*z^2*y^3+z*y*z^4+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y^2-y*z^4*x-z^3*y^3+z^4*y*x+z^6, y^2*z^2*y*x+y^2*z^4-z*y*z^3*y, y*z^3*y*x+y*z^5+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y*z-y*z^4*y-z^2*y^2*z^2-z^2*y*z^2*y-z^3*y^2*z-z^4*y^2+z^5*x}
f1 = F0_0*x
f2 = F0_0*y
f3 = F0_0*z
elapsedTime getMinSyzygies({f1,f2,f3},I,DegreeLimit => 4)
N = paMatrix{{x,y,z}}
elapsedTime getSyzMatrix(N,I,DegreeLimit => 4)

Nsyz = getSyzygies({f1,f2,f3},I)
NsyzM = flatten apply( Nsyz, m -> {(first m)*x,(first m)*y,(first m)*z})
NsyzMgb = buchPAModule(NsyzM,I,DegreeLimit => 4)
apply(Nsyz, m -> moduleReduce(m,I,NsyzMgb))
N = paMatrix {{x,y,z}}
NsyzMat = getSyzygiesMatrix(N,I)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y-y*x, x*z-z*x, y*z-z*y, z*x^2-z*y*x, z*y*x^2-z*y^2*x, z*y^2*x^2-z*y^3*x, z*y^3*x^2-z*y^4*x}
f1 = F0_0*x
f2 = F0_0*y
f3 = F0_0*z
buchPAModule({f1,f2,f3},I,DegreeLimit => 6)

N1 = getMinSyzygies({f1,f2,f3},I,DegreeLimit => 6)
M1 = getSyzygies({f1,f2,f3},I,DegreeLimit => 6)
F1 = class first N1
peek F1
buchPAModule(N1,I,DegreeLimit => 6)

N2 = getSyzygies(N1,I,DegreeLimit => 6)
return; continue
overlaps#3

N1 = getSyzygies({f1,f2,f3},I)
moduleReduce(N1#0,I,M1)

N = paMatrix {{x,y,z}}
N1 = elapsedTime getMinSyzMatrix(N,I,DegreeLimit => 6)
N1 = getSyzMatrix(N,I,DegreeLimit=>6)
N2 = elapsedTime getMinSyzMatrix(N1,I,DegreeLimit => 6)
N3 = elapsedTime getMinSyzMatrix(N2,I,DegreeLimit => 6)
N4 = elapsedTime getMinSyzMatrix(N3,I,DegreeLimit => 6)

restart
needsPackage "AssociativeAlgebras"
R = QQ<|x,y,z|>
I = ideal {x*y-y*x, x*z-z*x, y*z-z*y, z*x^2-z*y*x}
Igb = NCGB(I,6)


restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - y*x, x*z - z*x, y*z - z*y}
f1 = F0_0*x
f2 = F0_0*y
f3 = F0_0*z
K = getSyzygies({f1,f2,f3},I)
L=first K
J = K#1
moduleReduce(L,I,K)
moduleReduce(J,I,K)



restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - y*x, x*z - z*x, y*z - z*y}
F1 = A^3
f1 = -F1_0*y + F1_1*x
f2 = -F1_0*z + F1_2*x
f3 = -F1_1*z + F1_2*y
trackModInit({f1,f2,f3})
L = elapsedTime buchPAModule({f1,f2,f3},I,DegreeLimit => 5)
getSyzygies({f1,f2,f3},I)
getMinSyzygies({f1,f2,f3},I)

f4 = F1_1*y*x - F1_0*y^2

apply(3, i -> overlaps(f4,I#i,5))

o = (f4,x*y - y*x, y, y)
f4*y - F1_1*y*(x*y - y*x)
n = reduceOverlap(o,{f1,f2,f3,f4},I,F1)
n = 1(3,y) -> 1(0,y)*y = 1(0,y^2)

getTrackMatrix L


N1 = paMatrix {{x,y,z}}
N2 = transpose paMatrix {{x,y,z}}
N1*N2


restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y - y*x, x*z - z*x, y*z - z*y}
F1 = A^3
f1 = -F1_0*y + F1_1*x
f2 = -F1_0*z + F1_2*x
f3 = -F1_1*z + F1_2*y
f1 = F1_0*z
f2 = F1_0*y
f3 = F1_0*x
getSyzygies({f1,f2,f3},I,DegreeLimit => 2)  -- ???
L = elapsedTime buchPAModule({f1,f2,f3},I,DegreeLimit => 5)
K = elapsedTime buchPAModule({f1,f2,f3},I,DegreeLimit => 3)
J = elapsedTime buchPAModule({f1,f2,f3},I,DegreeLimit => 4)
N = getTrackMatrix L
L1 = getVecFromTrack L
M = makeMatrix L1

L#11

getSyzygies({f1,f2,f3},I)

I = paIdeal I
buchAlgorithm I
B = A/I
phi = paMap(B,A,{v},{x,y,z})
isWellDefined phi
use A
psi = paMap(A,B,{v},{x,y,z})
isWellDefined psi
phi(A.edgeGens#0)

fmat = makeMatrix {f1,f2,f3}
phi (fmat*N)
makeMatrix L1
phi (fmat*N) - phi makeMatrix L1

M1 = transpose matrix {{1,2,3}}
M1 | M1 --stacks horizontally
M1 || M1 -- stacks vertically
-- look at 9th column of fmat*N vs makeMatrix L1
M2 = matrix {{M1,M1}}

A=makeMatrix({f1,f2,f3})
B=A
A|B
A||B



restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
F0 = A^1
I = {x*y-y*x-z^2, x^2-y*z+z*y, x*z+y^2-z*x, y*z*y-z^2*x, y^2*x+y*z^2-z*y*z+z^2*y, y^4-y^2*z*x-y*z^3+z*y*z^2-z^2*y*z, y^3*z-z*y*z*x+z^2*y*x, y*z^3*x+z^2*y^3-z^3*y*x-z^5, y*z^2*y*z-y*z^3*y+z^3*y^2-z^4*x, y^2*z^2*x-z*y*z^3, y^2*z^2*y^2+y*z^2*y^3+z*y*z^4+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y^2-y*z^4*x-z^3*y^3+z^4*y*x+z^6, y^2*z^2*y*x+y^2*z^4-z*y*z^3*y, y*z^3*y*x+y*z^5+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y*z-y*z^4*y-z^2*y^2*z^2-z^2*y*z^2*y-z^3*y^2*z-z^4*y^2+z^5*x}
f1 = F0_0*x
f2 = F0_0*y
f3 = F0_0*z
S = elapsedTime getMinSyzygies({f1,f2,f3},I,DegreeLimit => 5)
J = elapsedTime buchPAModule(S,I,DegreeLimit => 5)
T = getTrackMatrix J
I = paIdeal I
buchAlgorithm I
B = A/I
phi = paMap(B,A,{v},{x,y,z})
phi ((makeMatrix S)*T) -
phi makeMatrix getVecFromTrack J

-----

restart
debug needsPackage "PathAlgebras"
adj = matrix {{1,0},{1,1}}
G = paGraph({v,w},{e,f,g},adj)
R = QQ
A = R G
N = paMatrix {{e,f,g}}
I = {f*g - g^2}
M = A^1
f1 = M_0*e
f2 = M_0*f
f3 = M_0*g
Msyz = getSyzygies({f1+f2,f3},I)
Msyz = getSyzygies({f1,f2,f3},I) -- wrong order
T = trackModInit{f1,f2,f3}
interredVecTrack T
Msyz = getSyzygies({f3,f2,f1},I) -- sorted coming in, so ok
-- the result (2,w) represents e with w since the input will be sorted and in the compHash,
-- 2 represents (0,e). previously we use tracklist to track how and where the syzgies come from
-- but now we get all zeroes in tracklist.
-- for example we have  (1(0, g), {0, 0, w}) then g overlaps with v and wv = 0 so we get all zeroes

Msyz = getSyzygies({(M_0)*v},I)
Msyz = getSyzygies({M_0},I)  -- should stop with {}
trackModInit {M_0}
trackModInit {(M_0)*w,(M_0)*v}
Msyz = getSyzygies({(M_0)*w,(M_0)*v},I)  -- should stop with {(0,v), (1,w)}
Nsyz = getSyzMatrix(N,I)

L=f1+f2+f3
toUniform L

trackModInit {f1,f2,f3}
trackModInit {f1+f2,f3}
trackModInit {M_0*v}
trackModInit {M_0}

restart
debug needsPackage "PathAlgebras"
adj = matrix {{1}}
G = paGraph({v},{x},adj)
R = QQ
A = R G
F0 = A^1
I = {x^4}
f1 = F0_0*x
getSyzygies({f1},I)
    

restart
debug needsPackage "PathAlgebras"
adj = matrix {{0,0,0,0,0,0},
              {0,0,0,0,0,0},
	      {1,1,0,0,0,0},
	      {0,0,1,0,0,0},
	      {0,0,1,0,0,0},
	      {0,0,0,1,1,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_6),adj)
R = QQ
A = R G
I = paIdeal {a_1*a_3,a_2*a_4,a_3*a_5-a_4*a_6}
I = buchAlgorithm I
M = A^1

Msyz1 = getSyzygies({M_0*a_3,M_0*a_4},I)
Msyz2 = getMinSyzygies({M_0*a_3,M_0*a_4},I)
Msyz3 = getSyzygies({M_0*a_4,M_0*a_3},I)
Msyz4 = getMinSyzygies({M_0*a_4,M_0*a_3},I)

peek (leadModMon first first Msyz3).module
peek (leadModMon first last Msyz4).module

N0 = paMatrix {{a_1}}
N0 = paMatrix {{a_3,a_4}}
N0 = paMatrix {{a_4,a_3}}

Msyz = getSyzygies( {M_0*a_3, M_0*a_4}, I); first Msyz
Msyz = getSyzygies( {M_0*a_4, M_0*a_3}, I); first Msyz
peek (leadModMon first last Msyz).module#"componentHash"

N1 = getSyzMatrix (N0, I)    -- ok
N1 = getMinSyzMatrix (N0, I) -- missing one term
N2 = getMinSyzMatrix (N1, I)
N3 = getMinSyzMatrix (N2, I)


restart
R = QQ[x]
I = ideal (x,2*x)
Igb = gb(I, ChangeMatrix => true)
getChangeMatrix Igb


restart
debug needsPackage "PathAlgebras"
adj = matrix {{1,1},{0,1}}
G = paGraph({v,w},{e,f,g},adj)
R = QQ
A = R G
N = paMatrix {{e,f,g}}
I = {f*g - g^2}
M = A^1

e*f*g+e*f
M_0*oo
toString oo


Msyz1 = getSyzygies({M_0},I)  -- should stop with {}
Msyz2 = getSyzygies({M_0*v,M_0*w},I)  -- should have 2 syzygies
Msyz3 = getSyzygies({M_0*g},I) -- should have 1 syzygy
Msyz4 = getSyzygies({M_0*(e+f)},I) -- should have 1 syzygy   ???  should stop with {}?
Msyz5 = getSyzygies({M_0*e,M_0*f},I) -- should have 1 syzygy ???
Msyz6 = getMinSyzygies({M_0*e,M_0*f},I)
Msyz7 = getSyzygies({M_0*e,M_0*e*f},I)
Msyz8 = getSyzygies({M_0*e,2*M_0*e},I)
peek (leadModMon first last Msyz1).module
peek (leadModMon first last Msyz2).module
peek (leadModMon first last Msyz5).module
 
getSyzMatrix( paMatrix {{1_A}}, I)
getSyzMatrix( paMatrix {{v,w}}, I)
getSyzMatrix( paMatrix {{e+f}}, I)
getSyzMatrix( paMatrix {{e,f}}, I)

assert(Msyz2 / first / toMatrix / entries / flatten == {{w, 0}, {0, v}})

M = A^5
L = M_0*(e+f)+M_1*(e*f)+M_4*g
L = toString L
value L 

restart
R = QQ[x]
foo = value "vector {x^3,x^2,x,1,1}"



restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y}

n0 = gtn v
n1 = gtn(n0,w)
n2 = gtn(n1,z)
n3 = gtn(n2,y)
n4 = gtn(n3,x)

gammaOverlap(w*z-z*w,z*y-y*z)
getGammas(I,3)
getGammas(I,4)
getGammas(I,5)
getGammas(I,6)

M = {{w, z}}


restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y}
M = A^1
getSyzygies({M_0*w, M_0*z}, I)

G_0 = {v}
G_1 = {w,z}
G_2 = {wz,wy,wx,zy,zx}
G_3 = {wzy,wzx,wyx,zyx}
G_4 = {wzyx}
-- ensure that g.left.right * g.right is a tip in I (this is just saying that g overlaps with a tip in I
-- we also need g.left.left to be an m-2 chain.

restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y,z*w}
getGammas({w,x,y,z},I,3)
G_0 = {v}
G_1 = {w}


restart
debug needsPackage "PathAlgebras"
adj = matrix {{1,0},{1,1}}
G = paGraph({v,w},{e,f,g},adj)
R = QQ
A = R G
I= {e*f,f*g}
getGammas({e,f,g},I,3)
l=gtn v
k=gtn(l,g) 


restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y}
getGammas({w,z,y,x},I,3)
L = getGammas({w,z,y,x},I,4)
getGammas({w,z,y,x},I,5)

wor = first last L

l=gtn v
k=gtn(l,x)
g=gtn(k,y)

getm1chain(g)
getm2chain(g)

gammaOverlap(g,y*z)
isPrefix(k,x*y)


factorGTN(first last L)
factorGTN(oo)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y,w^2 - x*y}
getGammas({w,z,y,x},I,7)
oo / length
oo / sort

I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y,w^2 - x*y}
gammas = getGammas({w,y,x},I,7)
gammas / length
netList gammas
oo / sort

I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y,w^2 - y*z}
gammas = getGammas({w,z,y},I,7)
gammas / length
netList gammas
oo / sort

I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y,z^2 - x*y}
gammas = getGammas({z,y,x},I,7)
gammas / length
netList gammas
oo / sort

I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y,z^2 - x*y}
gammas = getGammas({x*z,x*y},I,7)
gammas / length
netList gammas
oo / sort

restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y}
--B = A/paIdeal I
G = getGammas({w,z},I,6)

GAM = getGammaHash({w,z},I,6)
word1 = GAM.terms#2#0
GAM.delResults#word1 = "stuff"
GAM.delResults#word1


M = A^1
getSyzygies({M_0*w, M_0*z}, I, DegreeLimit => 3)
getMinSyzygies({M_0*w, M_0*z}, I,DegreeLimit => 3) -- problem?
--huge sets

removeEdge(x*y*z,2)

Gamma1 = G#1
Gamma2 = G#2
gtn1 = gtn(Gamma1#0,x*y)
gtn1*z
eta(gtn1,Gamma2,I)
gtnVec({gtn1},{1})

del(w.y.x) = (w.y)x - eta(del(w.y)x)                       -- differentiate the left half
           = (w.y)x - eta((w)yx - (y)wx)  
	   = (w.y)x - eta((w)xy - (y)xw)                   -- reduce yx and wx to xy and xw, respectively
	   = (w.y)x - (w.x)y - eta((x)wy - (y)xw)          -- move x to the left in (w)x and replace with (x).w
	   = (w.y)x - (w.x)y - eta((x)yw - (y)xw)          -- reduce wy to yw
           = (w.y)x - (w.x)y + (y.x)w - eta((x)yw - (x)yw) -- inside eta is zero now
	   = (w.y)x - (w.x)y + (y.x)w

del(w.z.y.x) = (w.z.y)x - eta(del((w.z.y)x))
 = (w.z.y)x - eta((w.z)yx - (w.y)zx + (z.y)wx)
 = (w.z.y)x - eta((w.z)xy - (w.y)xz + (z.y)xw)  --- del((w.z.x)y) = (w.z)xy - (w.x)zy + (z.x)wy
 = (w.z.y)x - (w.z.x)y - eta((w.x)zy - (z.x)wy - (w.y)xz + (z.y)xw)


--YW
restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y}
--G = getGammas({w,z,y,x},I,6)
GAM = getGammaHash({w,z,y,x},I,6)
allDelta(GAM,4)

f1=GAM.terms#1#0
del(f1,GAM)
f2=GAM.terms#2#0
del(f2,GAM)
f7=GAM.terms#2#2
del(f7,GAM)
f8=GAM.terms#2#1
del(f8,GAM)
f9=GAM.terms#2#3
del(f9,GAM)

f10=GAM.terms#2#5
del(f10,GAM)
f11=GAM.terms#2#4
del(f11,GAM)
f3=GAM.terms#3#0
del(f3,GAM)

f4=GAM.terms#3#1
del(f4,GAM)
f5=GAM.terms#3#2
del(f5,GAM)
f6=GAM.terms#3#3
del(f6,GAM)
f12=GAM.terms#4#0
del(f12,GAM)

peek (GAM.delResults)

delReduce(toDo#0,I)

ringReduce(f*y,I)
L = first GAM.terms#0
delCheck(L,GAM)

L = {(GAM.terms#1#2,-z*w),(GAM.terms#1#2,z*w)}
ringInterReduce(L,I)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y}
G = getGammas({w,z},I,6)
GAM = getGammaHash({w,z},I,6)

f7=GAM.terms#2#3
del(f7,GAM,I)
--problem here
ringReduce(x*y-y*x,I)


--- example list
restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y}
G = getGammas({w,z,y,x},I,6)
GAM = getGammaHash({w,z,y,x},I,6)

-- create a function that takes dels of all terms in GAM
del(GAM.terms#3#0,GAM,I)

peek (GAM.delResults)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = ZZ/32003
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y,x^2 + y^2 + z^2 + w^2}
G = getGammas({w,z,y,x},I,6)
GAM = getGammaHash({w,z,y,x},I,6)

-- create a function that takes dels of all terms in GAM
apply(8, i -> del(GAM.terms#6#i,GAM,I))
peek (GAM.delResults)

restart
debug needsPackage "PathAlgebras"
adj = matrix {{2}}
G = paGraph({v},{y,x},adj)
R = ZZ/32003
A = R G
I = {y*x-x*y, x^2, y^2}
G = getGammas({y,x},I,6)
GAM = getGammaHash({y,x},I,6)

-- create a function that takes dels of all terms in GAM
-- make sure I is monic before running the algorithm

netList apply(7, i -> del(GAM.terms#6#i,GAM,I))
peek (GAM.delResults)

-- an example to investigate
restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{z,y,x},adj)
R = ZZ/32003
A = R G
I = {z*y-y*z,z*x-x*z,y*x-x*y,x^2 + y^2}
GAM = getGammaHash({z,y,x},I,10)
allDelta(GAM,10)
allDeltaMatrices GAM

restart
R = QQ[z,y,x]
I = ideal (x^2 + y^2)
S = R/I
kRes = res(coker vars S, LengthLimit => 10)
kRes.dd_6
kRes.dd_7

restart
debug needsPackage "PathAlgebras"
M = matrix {{1,0},{1,1}}
--G = paGraph({v,w},{e,f,g},M,Weights=>{2,1,1})
G = paGraph({v,w},{e,f,g},M)
R = QQ
A = R G
I = {e^2,g^2}
--I = {e^3, e^2*f - e*f}
G = getGammas({e,f},I,6)
GAM = getGammaHash({e,f},I,6)
allDelta(GAM,6)
allDeltaMatrices GAM
G = getGammas({g},I,6)
GAM = getGammaHash({g},I,6)
allDelta(GAM,6)
allDeltaMatrices GAM -- TODO


--Examples:

restart
debug needsPackage "PathAlgebras"
adj = matrix {{4}}
G = paGraph({v},{w,z,y,x},adj)
R = QQ
A = R G
I = {w*z-z*w,w*y-y*w,w*x-x*w,z*y-y*z,z*x-x*z,y*x-x*y}
GAM = getGammaHash({w,z,y,x},I,10)
allDelta(GAM,10,DegreeLimit => 3) -- TODO: if n is beyond the max then don't call the del function --Done
allDelta(GAM,2)
del(GAM.terms#3#0,GAM)
allDeltaMatrices GAM


restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
--I = {x*y-y*x-z^2, x^2-y*z+z*y, x*z+y^2-z*x, y*z*y-z^2*x, y^2*x+y*z^2-z*y*z+z^2*y, y^4-y^2*z*x-y*z^3+z*y*z^2-z^2*y*z, y^3*z-z*y*z*x+z^2*y*x, y*z^3*x+z^2*y^3-z^3*y*x-z^5, y*z^2*y*z-y*z^3*y+z^3*y^2-z^4*x, y^2*z^2*x-z*y*z^3, y^2*z^2*y^2+y*z^2*y^3+z*y*z^4+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y^2-y*z^4*x-z^3*y^3+z^4*y*x+z^6, y^2*z^2*y*x+y^2*z^4-z*y*z^3*y, y*z^3*y*x+y*z^5+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2, y*z^3*y*z-y*z^4*y-z^2*y^2*z^2-z^2*y*z^2*y-z^3*y^2*z-z^4*y^2+z^5*x}
I = {x*y-y*x-z^2, x^2-y*z+z*y, x*z+y^2-z*x, y*z*y-z^2*x, y^2*x+y*z^2-z*y*z+z^2*y, y^4-y^2*z*x-y*z^3+z*y*z^2-z^2*y*z,
       y^3*z-z*y*z*x+z^2*y*x, y*z^3*x+z^2*y^3-z^3*y*x-z^5, y*z^2*y*z-y*z^3*y+z^3*y^2-z^4*x,
       y^2*z^2*x-z*y*z^3, y^2*z^2*y^2+y*z^2*y^3+z*y*z^4+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2,
       y*z^3*y^2-y*z^4*x-z^3*y^3+z^4*y*x+z^6, y^2*z^2*y*x+y^2*z^4-z*y*z^3*y, 
       y*z^3*y*x+y*z^5+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2,
       y*z^3*y*z-y*z^4*y-z^2*y^2*z^2-z^2*y*z^2*y-z^3*y^2*z-z^4*y^2+z^5*x,
       y*z^4*y^2-y*z^5*x+z^2*y^2*z^2*y+z^2*y*z^2*y^2+z^3*y*z^2*x+z^4*y^3-z^5*y*x-z^7,
       y*z^4*y*z+z^2*y^2*z^3+z^2*y*z^3*y-z^3*y*z^2*y-z^5*y^2+z^6*x,
       y^2*z^4*y+y*z^2*y^2*z^2+y*z^5*x-z*y*z^4*x-z^3*y*z^2*x-z^4*y^3+z^5*y*x+z^7, 
       y*z^6*x+z^2*y^2*z^3*y+z^2*y*z^4*x-z^3*y*z^2*y^2,
       y^2*z^5*x-z*y*z^4*y*x-z*y*z^6-z^4*y^2*z*x-z^4*y*z^3+z^5*y*z^2, 
       y^2*z^5*y^2+y*z^2*y^2*z^3*y-z*y*z^5*y*x+z^3*y*z^2*y^3+z^5*y^2*z*x+z^5*y*z^3-z^6*y*z^2,
       y*z^6*y^2-y*z^7*x+z^3*y*z^2*y^2*z-z^3*y*z^4*x-z^4*y^2*z^2*y-z^4*y*z^2*y^2-z^5*y*z^2*x-2*z^6*y^3+2*z^7*y*x+2*z^9,
       y*z^6*y*x+y*z^8+z^2*y^2*z^4*x+z^2*y*z^4*y*x+z^2*y*z^6-z^3*y*z^2*y^3+z^5*y^2*z*x+z^5*y*z^3-z^6*y*z^2, 
       y*z^6*y*z-y*z^7*y-z^2*y^2*z^5-z^2*y*z^2*y^2*z*x-z^2*y*z^5*y+z^3*y*z^4*y-z^4*y^2*z^3-z^4*y*z^3*y+z^5*y^2*z^2+z^5*y*z^2*y+z^7*y^2-z^8*x,
       y*z^5*y^3+z^2*y^2*z^4*x+z^2*y*z^4*y*x+z^2*y*z^6-2*z^3*y*z^2*y^3-z^3*y*z^5+z^6*y*z^2,
       y^2*z^5*y*x+y^2*z^7-z*y*z^5*y*z-z^4*y^2*z^3-z^4*y*z^3*y+z^5*y*z^2*y+z^7*y^2-z^8*x,
       y^2*z^5*y*z-y^2*z^6*y+z*y*z^5*y^2+2*z^3*y^2*z^3*y+z^3*y*z^2*y^2*z+z^3*y*z^4*x+z^4*y^2*z^2*y-z^4*y*z^2*y^2+z^5*y*z^2*x+z^6*y^3-z^7*y*x-z^9, 
       y^2*z^7*y+y*z^2*y^2*z^5+y*z^8*x-z*y*z^7*x-2*z^4*y^2*z^3*y+z^4*y*z^2*y^2*z-3*z^4*y*z^4*x-2*z^5*y^2*z^2*y-3*z^6*y*z^2*x-3*z^7*y^3+3*z^8*y*x+3*z^10,
       y*z^7*y*z+z^2*y^2*z^6+z^2*y*z^6*y-z^3*y*z^2*y^2*z*x-z^3*y*z^5*y-z^6*y^2*z^2-2*z^6*y*z^2*y-z^7*y^2*z-2*z^8*y^2+2*z^9*x,
       y*z^5*y^2*z^2+y*z^5*y*z^2*y-2*z^3*y*z^2*y^2*z^2+z^4*y^2*z^3*y+z^4*y*z^2*y^2*z+z^5*y^2*z^2*y+z^6*y*z^2*x+z^7*y^3-z^8*y*x-z^10,
       y*z^5*y^2*z*x+y*z^5*y*z^3-2*z^3*y*z^2*y^2*z*x-2*z^3*y*z^5*y-z^4*y^2*z^4-z^4*y*z^4*y-z^5*y^2*z^3-z^6*y^2*z^2, 
       y*z^7*y^2-y*z^8*x+z^2*y^2*z^5*y+z^2*y*z^2*y^2*z^3-z^3*y*z^5*x+z^5*y*z^2*y^2+z^6*y*z^2*x+z^7*y^3-z^8*y*x-z^10}
apply(I, leadTerm)
--gam = getGammas1({z,y,x},I,3)
GAM = getGammaHash({z,y,x},I,3)
GAM = getGammaHash({z,y,x},I,4)
GAM = getGammaHash({z,y,x},I,5)
GAM1 = getDegreeGammaHash(GAM)

tempHash = partition (m -> (m_0, m_1), sort flatten apply(pairs GAM.terms, p -> apply(last p, q -> (length q, first p, q))))
applyValues(tempHash, p -> apply(p, last))

del(GAM.terms#2#7,GAM)
del(GAM.terms#3#0,GAM)
del(GAM.terms#3#1,GAM)
del(GAM.terms#3#2,GAM)
del(GAM.terms#3#3,GAM)
del(GAM.terms#3#4,GAM)
del(GAM.terms#3#5,GAM)
del(GAM.terms#3#6,GAM)
del(GAM.terms#3#7,GAM)
del(GAM.terms#3#8,GAM)
del(GAM.terms#3#9,GAM)
del(GAM.terms#3#10,GAM)
del(GAM.terms#3#11,GAM)
del(GAM.terms#3#12,GAM)
del(GAM.terms#3#13,GAM) -- infinite loop?
del(GAM.terms#3#13,GAM,DegreeLimit => 4)
del(GAM.terms#3#14,GAM)
del(GAM.terms#3#15,GAM)
del(GAM.terms#3#16,GAM) -- infinite loop?
del(GAM.terms#3#17,GAM) -- infinite loop?
del(GAM.terms#3#18,GAM) -- infinite loop?
del(GAM.terms#3#19,GAM)
del(GAM.terms#3#20,GAM)
del(GAM.terms#3#21,GAM)
del(GAM.terms#3#22,GAM)
del(GAM.terms#3#23,GAM)
del(GAM.terms#3#24,GAM)
del(GAM.terms#3#25,GAM)
del(GAM.terms#3#26,GAM)
del(GAM.terms#3#27,GAM)
del(GAM.terms#3#28,GAM)
del(GAM.terms#3#29,GAM)
del(GAM.terms#3#30,GAM)
del(GAM.terms#3#31,GAM)
del(GAM.terms#3#32,GAM)
del(GAM.terms#3#33,GAM)
del(GAM.terms#3#34,GAM)
del(GAM.terms#3#35,GAM)
del(GAM.terms#3#36,GAM)
del(GAM.terms#3#37,GAM)
del(GAM.terms#3#38,GAM)
del(GAM.terms#3#39,GAM)

del(GAM.terms#3#40,GAM) -- Degree limit



del(GAM.terms#4#0,GAM)
del(GAM.terms#4#1,GAM)
del(GAM.terms#4#2,GAM)
del(GAM.terms#4#3,GAM)
del(GAM.terms#4#4,GAM)
del(GAM.terms#4#5,GAM)
del(GAM.terms#4#6,GAM)
del(GAM.terms#4#7,GAM)
elapsedTime allDelta(GAM,2);
elapsedTime allDelta(GAM,3);
allDelta(GAM,3,DegreeLimit => 8)
allDeltaMatrices GAM
allDeltaMatrices (GAM,2, LengthLimit => 5)
allDeltaMatrices (GAM,3, LengthLimit => 8)
GAM
GAM.terms#2#7
del(GAM.terms#2#12,GAM)
del(GAM.terms#2#13,GAM)
del(GAM.terms#2#14,GAM)
del(GAM.terms#2#0,GAM)
del(GAM.terms#2#1,GAM)
del(GAM.terms#2#2,GAM)
del(GAM.terms#2#3,GAM)
del(GAM.terms#2#4,GAM)
sepTensorPairs(oo)
-x*y-x*y


delResultsCheck(GAM,(3,0))
delResultsCheck(GAM,(3,1))
delResultsCheck(GAM,(3,2))
for n from 0 to 10 list delResultsCheck(GAM,(3,n))

restart
needsPackage "AssociativeAlgebras"
A = QQ<|x,y,z|>
I = ideal {x*y-y*x-z^2, x^2-y*z+z*y, x*z+y^2-z*x}
B = A/I
M1 = matrix {{x,y,z}}
M2 = rightKernel(M1, DegreeLimit => 10)
M3 = rightKernel(M2, DegreeLimit => 10)
M4 = rightKernel(M3, DegreeLimit => 10)


restart
debug needsPackage "PathAlgebras"
adj = matrix {{3}}
G = paGraph({v},{x,y,z},adj)
R = QQ
A = R G
I = {x*y-y*x-z^2, x^2-y*z+z*y, x*z+y^2-z*x, y*z*y-z^2*x, y^2*x+y*z^2-z*y*z+z^2*y, y^4-y^2*z*x-y*z^3+z*y*z^2-z^2*y*z,
       y^3*z-z*y*z*x+z^2*y*x, y*z^3*x+z^2*y^3-z^3*y*x-z^5, y*z^2*y*z-y*z^3*y+z^3*y^2-z^4*x,
       y^2*z^2*x-z*y*z^3, y^2*z^2*y^2+y*z^2*y^3+z*y*z^4+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2,
       y*z^3*y^2-y*z^4*x-z^3*y^3+z^4*y*x+z^6, y^2*z^2*y*x+y^2*z^4-z*y*z^3*y, 
       y*z^3*y*x+y*z^5+z^2*y^2*z*x+z^2*y*z^3-z^3*y*z^2,
       y*z^3*y*z-y*z^4*y-z^2*y^2*z^2-z^2*y*z^2*y-z^3*y^2*z-z^4*y^2+z^5*x,
       y*z^4*y^2-y*z^5*x+z^2*y^2*z^2*y+z^2*y*z^2*y^2+z^3*y*z^2*x+z^4*y^3-z^5*y*x-z^7,
       y*z^4*y*z+z^2*y^2*z^3+z^2*y*z^3*y-z^3*y*z^2*y-z^5*y^2+z^6*x,
       y^2*z^4*y+y*z^2*y^2*z^2+y*z^5*x-z*y*z^4*x-z^3*y*z^2*x-z^4*y^3+z^5*y*x+z^7}
GAM = getGammaHash({z,y,x},I,DegreeLimit => 7)
GAM1 = getDegreeGammaHash(GAM)
bettiTally GAM

getGammas1({z,y,x},I,5,ChainLimit => 5)
getGammas({z,y,x},I)
-- if a degree limit is supplied:

-- only consider chains of <= to that degree
-- only create 2 chains from GB elements <= to that degree

-- also include a 'chain' limit (I think you already have this)

-- degree 7 limit

matrix apply(toList(0..7), r -> apply(toList(0..7), c -> if GAM1.degreeTerms#?(r+c,c) then #(GAM1.degreeTerms#(r+c,c)) else 0))

BettiTally 

    0   1   2   3   4   5   6
  +--------------------------
0 | 1   3   3   1   -   -   -
1 | -   -   -   -   -   -
2 | -   -   -   -   -
3 | -   -   -   -
4 | -   -   -
5 | -   -
6 | -


-- BettiTally for path algebra example:
restart
debug needsPackage "PathAlgebras"
adj = matrix {{0,0,0,0,0,0},
              {0,0,0,0,0,0},
	      {1,1,0,0,0,0},
	      {0,0,1,0,0,0},
	      {0,0,1,0,0,0},
	      {0,0,0,1,1,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_6),adj)
R = QQ
A = R G
I = {a_1*a_3,a_2*a_4,a_3*a_5-a_4*a_6,a_1*a_4*a_6}

GAMS1 = getGammaHash({a_1},I,DegreeLimit => 7)
bettiTally GAMS1
bettiTally(GAMS1,5)
allDeltaMatrices GAMS1

del(GAMS1.terms#1#0, GAMS1)
del(GAMS1.terms#2#0, GAMS1)
del(GAMS1.terms#3#0, GAMS1)

GAMS2 = getGammaHash({a_2},I,DegreeLimit => 7)
bettiTally GAMS2
GAMS3 = getGammaHash({a_3,a_4},I,DegreeLimit => 7)
bettiTally GAMS3
GAMS4 = getGammaHash({a_5},I,DegreeLimit => 7)
bettiTally GAMS4
GAMS5 = getGammaHash({a_6},I,DegreeLimit => 7)
bettiTally GAMS5
--GAMS6 = getGammaHash({v_6},I,DegreeLimit => 7) -- what to put in the presentation matrix for S_6 = P_6 ???
--bettiTally GAMS6

B1 = new BettiTally from { (0,{0},0) => {0,1}, (0,{1},1) => {1,0} }
B2 = new BettiTally from { (0,{0},0) => {1,0} }

GAMS1Deg = getDegreeGammaHash(GAMS1)

tal = tally apply(GAMS1Deg.degreeTerms#(3,2), g -> endVertex(g#"word"))
tal#4

endVertexCount = (gtns,A) -> (
    tal := tally apply(gtns, g -> endVertex(g#"word"));
    apply(numVertices A, i -> if not tal#?i then 0 else tal#i)
)
debug Core
Betti = new BettiTally from apply(pairs GAMS1Deg.degreeTerms, m -> (last first m,{first first m},first first  m) => endVertexCount(last m,A));
rawBettiTally Betti
code rawBettiTally

restart
debug needsPackage "PathAlgebras"
adj = matrix {{0,0,0,0,1,1},
              {1,0,0,0,0,0},
	      {1,0,0,0,0,0},
	      {0,1,1,0,0,0},
	      {0,0,0,1,0,0},
	      {0,0,0,1,0,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_8),adj,Weights => {1,2,1,1,1,1,1,1})
R = QQ
A = R G
--I = { a_1*a_3 - a_2*a_4, a_4*a_6*a_8*a_2, a_4*a_5*a_7*a_1, a_3*a_6*a_8*a_2, a_3*a_5*a_7*a_1, a_4*a_5*a_7*a_2*a_4, a_3*a_5*a_7*a_2*a_4 }
I = { -a_1*a_3 + a_2*a_4, a_4*a_6*a_8*a_2, a_4*a_5*a_7*a_1, a_3*a_6*a_8*a_2, a_3*a_5*a_7*a_1, a_4*a_6*a_8*a_1*a_3, a_3*a_6*a_8*a_1*a_3 }
--I = { a_2*a_4, a_4*a_6*a_8*a_2, a_4*a_5*a_7*a_1, a_3*a_6*a_8*a_2, a_3*a_5*a_7*a_1, a_4*a_6*a_8*a_1*a_3, a_3*a_6*a_8*a_1*a_3 }
B = A/(paIdeal I)

GAMS1 = getGammaHash({a_1,a_2},I,DegreeLimit => 15)  -- error on allDeltaMatrices if we put 15 here...
GAMS1 = getDegreeGammaHash(GAMS1)
B = bettiTally GAMS1
apply(6, i -> bettiTally(GAMS1,i))
bettiTally GAMS1
bettiTally GAMS1

findReducePositions(GAMS1,3)
findReducePositions B

del(GAMS1.terms#5#0,GAMS1)
gtn = GAMS1.terms#5#0

leftGtn = GAMS1.terms#4#0
del(GAMS1.terms#4#0,GAMS1)
a_3*a_6*a_8*a_2

del(GAMS1.terms#4)

del(GAMS1.terms#5#0,GAMS1)

del(GAMS1.terms#4#0,GAMS1)
del(GAMS1.terms#4#1,GAMS1)

allDeltaMatrices GAMS1

bettiTally GAMS1

del(GAMS1.terms#6#0,GAMS1)
apply(GAMS1.terms#5, m-> del(m,GAMS1))



restart
debug needsPackage "PathAlgebras"
adj = matrix {{0,0,0,0,1,1},
              {1,0,0,0,0,0},
	      {1,0,0,0,0,0},
	      {0,1,1,0,0,0},
	      {0,0,0,1,0,0},
	      {0,0,0,1,0,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_8),adj,Weights => {1,2,1,1,1,1,1,1})
R = QQ
A = R G
I = { -a_1*a_3 + a_2*a_4, a_4*a_6*a_8*a_2, a_4*a_5*a_7*a_1, a_3*a_6*a_8*a_2, a_3*a_5*a_7*a_1, a_4*a_6*a_8*a_1*a_3, a_3*a_6*a_8*a_1*a_3 }
GAMS1 = getGammaHash({a_1,a_2},I,DegreeLimit => 15)
--GAMS1 = getDegreeGammaHash(GAMS1)
apply(6, i -> bettiTally(GAMS1,i))
allDeltaMatrices GAMS1

B = A/(paIdeal I)

findGAMMABases(GAMS1.terms#3#0,GAMS1)
findGAMMABases(GAMS1.terms#4#0,GAMS1)

L = getReduceTerms(GAMS1,3,4,6)
getReduceMatrix(GAMS1,L)

M1 = getReduceTerms(GAMS1,3,8,14)
del1 = getReduceMatrix(GAMS1,M1)

M2 = getReduceTerms(GAMS1,3,7,14)
del2 = getReduceMatrix(GAMS1,M2)
rank del2

M3 = getReduceTerms(GAMS1,3,6,14)
del3 = getReduceMatrix(GAMS1,M3)
rank del3




L = getReduceTerms(GAMS1,1,5,9)
getReduceMatrix(GAMS1,L)


apply(6, i -> getReducedBT(GAMS1,i))

 getReducedBT(GAMS1,3)



restart
debug needsPackage "PathAlgebras"
adj = matrix {{0,0,0,0,1,1},
              {1,0,0,0,0,0},
	      {1,0,0,0,0,0},
	      {0,1,1,0,0,0},
	      {0,0,0,1,0,0},
	      {0,0,0,1,0,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_8),adj,Weights => {1,2,1,1,1,1,1,1})
R = QQ
A = R G
I = { -a_1*a_3 + a_2*a_4, a_4*a_6*a_8*a_2, a_4*a_5*a_7*a_1, a_3*a_6*a_8*a_2, a_3*a_5*a_7*a_1, a_4*a_6*a_8*a_1*a_3, a_3*a_6*a_8*a_1*a_3 }
GAMS1 = getGammaHash({a_1,a_2},I,DegreeLimit => 15)
apply(6, i -> getReducedBT(GAMS1,i))
GAMS2 = getGammaHash({a_3},I,DegreeLimit => 15)
apply(6, i -> getReducedBT(GAMS2,i))
GAMS3 = getGammaHash({a_4},I,DegreeLimit => 15)
apply(6, i -> getReducedBT(GAMS3,i))

restart
debug needsPackage "PathAlgebras"
adj = transpose matrix {{0,1,0,0,0,0},
              {1,0,0,0,0,0},
	      {0,1,0,0,0,0},
	      {0,1,0,0,0,0},
	      {0,1,0,0,0,0},
	      {0,0,1,1,1,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_8),adj)
R = QQ
A = R G
I = { a_1*a_2, a_2*a_1*a_3*a_6 - a_4*a_7, a_3*a_6 + a_4*a_7 + a_5*a_8 }
GAMS1 = getGammaHash({a_1,a_2},I,DegreeLimit => 15)
apply(6, i -> getReducedBT(GAMS1,i))
GAMS2 = getGammaHash({a_3},I,DegreeLimit => 15)
apply(6, i -> getReducedBT(GAMS2,i))
GAMS3 = getGammaHash({a_4},I,DegreeLimit => 15)
apply(6, i -> getReducedBT(GAMS3,i))



restart
debug needsPackage "PathAlgebras"
adj = matrix {{0,0,0,0,1,1},
              {1,0,0,0,0,0},
	      {1,0,0,0,0,0},
	      {0,1,1,0,0,0},
	      {0,0,0,1,0,0},
	      {0,0,0,1,0,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_8),adj,Weights => {1,2,1,1,1,1,1,1})
R = QQ
A = R G
I = { -a_1*a_3 + a_2*a_4, a_4*a_6*a_8*a_2, a_4*a_5*a_7*a_1, a_3*a_6*a_8*a_2, a_3*a_5*a_7*a_1, a_4*a_6*a_8*a_1*a_3, a_3*a_6*a_8*a_1*a_3 }
GAMS1 = getGammaHash({a_1,a_2,a_3,a_4,a_5,a_6,a_7,a_8},I,DegreeLimit => 15)



restart
debug needsPackage "PathAlgebras"
adj = matrix {{0,0,0,0,1,1},
              {1,0,0,0,0,0},
	      {1,0,0,0,0,0},
	      {0,1,1,0,0,0},
	      {0,0,0,1,0,0},
	      {0,0,0,1,0,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_8),adj,Weights => {1,2,1,1,1,1,1,1})
R = QQ
A = R G
I = { -a_1*a_3 + a_2*a_4, a_4*a_6*a_8*a_2, a_4*a_5*a_7*a_1, a_3*a_6*a_8*a_2, a_3*a_5*a_7*a_1, a_4*a_6*a_8*a_1*a_3, a_3*a_6*a_8*a_1*a_3 }
GAMS1 = getGammaHash({a_3},I,DegreeLimit => 15)
apply(6, i -> getReducedBT(GAMS1,i))




--S_1 projective resolution
restart
debug needsPackage "PathAlgebras"
adj = transpose matrix {{0,0,0,0,1,1},
              {1,0,0,0,0,0},
	      {1,0,0,0,0,0},
	      {0,1,1,0,0,0},
	      {0,0,0,1,0,0},
	      {0,0,0,1,0,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_8),adj,Weights => {1,2,1,1,1,1,1,1})
R = QQ
A = R G
I = { -a_1*a_3 + a_2*a_4, a_4*a_6*a_8*a_2, a_4*a_5*a_7*a_1, a_3*a_6*a_8*a_2, a_3*a_5*a_7*a_1, a_4*a_6*a_8*a_1*a_3, a_3*a_6*a_8*a_1*a_3 }
GAMS1 = getGammaHash({a_3},I,DegreeLimit => 15)
RBT = apply(6, i -> getReducedBT(GAMS1,i))
netList RBT

restart
debug needsPackage "PathAlgebras"
      	M = matrix {{3}}
	G = paGraph({v},{a,b,c},M,Weights => {1,1,1})
	R = QQ
	A = R G
	N = A^2
	0_N
	putInPathAlgebra(A,{1,2,1})
paPath(G,0)

        adj = matrix {{1,1},{0,1}}
	G = paGraph({v,w},{e,f,g},adj)
	R = QQ
	A = R G
	M = A^2
	E = 5*e*f+6*e*f*g*g
	D = 4*g*g*g
        x=M_0*E+M_1*D


restart
debug needsPackage "PathAlgebras"
M = matrix {{3}}
G = paGraph({v},{a,b,c},M,Weights=>{1,1,1})
kk = ZZ/32003
R = kk G
I = paIdeal {2*a*b + 3*b*a + 5*c^2,
             2*b*c + 3*c*b + 5*a^2,
             2*c*a + 3*a*c + 5*b^2}
	 
K=I.generators
ov1=overlaps(K#0,K#2)
 divAlgorithm(K, first values ov1)
ov2=overlaps(K#1,K#2)
ov3=overlaps(K#1,K#0)

restart
debug needsPackage "PathAlgebras"
	 adj = matrix {{1,1},{0,1}}
	 G = paGraph({v,w},{e,f,g},adj)
	 R = QQ
	 A = R G
         L = e*f
	 H = paModMon(A^1,0,paPath(G,{0,0,1,2}))
	 K = e*e
	 isSubModMon(L,H)
	 isSubModMon(K,H)
	 
restart
debug needsPackage "PathAlgebras"
	 adj = matrix {{4}}
	 G = paGraph({v},{x_1,x_2,x_3,x_4},adj)
	 R = QQ
	 A = R G
	 L = x_1*x_2*x_3-1/2*x_1+1/2*x_2-1/2*x_3
	 K = x_1*x_2*x_4-1/2*x_1+1/2*x_2-1/2*x_4
	 L*K
	 --I = paIdeal{w*x+x*w-1_A,w*y+y*w-1_A,w*z+z*w-1_A,x*y+y*x-1_A,x*z+z*x-1_A,y*z+z*y-1_A}
	 I ={x_1*x_2+x_2*x_1-1_A,x_1*x_3+x_3*x_1-1_A,x_1*x_4+x_4*x_1-1_A,x_2*x_3+x_3*x_2-1_A,x_2*x_4+x_4*x_2-1_A,x_3*x_4+x_4*x_3-1_A}
	 divAlgorithm(I,L*K)
	 M = x_1*x_2-(1/2)_A
	 N = x_1*x_2*x_3*x_4-1/2*x_1*x_2+1/2*x_1*x_3-1/2*x_1*x_4-1/2*x_2*x_3+1/2*x_2*x_4-1/2*x_3*x_4+1/4
	 M*N
	 divAlgorithm(I,M*N)
	 divAlgorithm(I,L*K-M*N)
	 
restart
debug needsPackage "PathAlgebras" 
	 adj = matrix {{2}}
	 G = paGraph({v},{y,z},adj)
	 kk = ZZ/3
	 R = kk G
	 I = {y^3,y*z-z*y,z^3}
	 B = R/(paIdeal I)
	 GAMS1 = getGammaHash({y,z},I,DegreeLimit => 15)  -- error on allDeltaMatrices if we put 15 here...
GAMS1 = getDegreeGammaHash(GAMS1)
B = bettiTally GAMS1
apply(6, i -> bettiTally(GAMS1,i))
bettiTally GAMS1
bettiTally GAMS1

restart
debug needsPackage "PathAlgebras"
adj = transpose matrix {{0,0,0,0,1,1},
              {1,0,0,0,0,0},
	      {1,0,0,0,0,0},
	      {0,1,1,0,0,0},
	      {0,0,0,1,0,0},
	      {0,0,0,1,0,0}}
G = paGraph(toList (v_1..v_6),toList (a_1..a_8),adj,Weights => {1,2,1,1,1,1,1,1})
R = QQ
A = R G
I = { -a_1*a_3 + a_2*a_4, a_4*a_6*a_8*a_2, a_4*a_5*a_7*a_1, a_3*a_6*a_8*a_2, a_3*a_5*a_7*a_1, a_4*a_6*a_8*a_1*a_3, a_3*a_6*a_8*a_1*a_3 }
GAMS1 = getGammaHash({a_1,a_2},I,DegreeLimit => 15)
getReducedBT(GAMS1,0)
apply(6, i -> getReducedBT(GAMS1,i))
	 
restart
debug needsPackage "PathAlgebras" 
	 adj = matrix {{1}}
	 G = paGraph({v},{y},adj)
	 kk = ZZ/3
	 R = kk G
	 I = {y^3}
GAMS1 = getGammaHash({y},I,DegreeLimit => 15)
getReducedBT(GAMS1,0)
GAMS2 = getGammaHash({z},I,DegreeLimit => 15)
getReducedBT(GAMS2,0)


restart
debug needsPackage "PathAlgebras" 
	 adj = matrix {{3}}
	 G = paGraph({v},{x,y,z},adj)
	 kk = ZZ/3
	 R = kk G
	 I = {y^3,y*z-z*y,z^3,x*y-y*x,x*z-z*x,x^3}
GAMS1 = getGammaHash({x,y,z},I,DegreeLimit => 20)
getReducedBT(GAMS1,0)

restart
debug needsPackage "PathAlgebras" 
	 adj = matrix {{1,1,0},{0,1,1},{1,0,1}}
	 G = paGraph({v_0,v_1,v_2},{z_0,y_0,z_1,y_1,y_2,z_2},adj)
	 kk = ZZ/3
	 R = kk G
	 I = {y_0*y_1*y_2,y_2*y_0*y_1,y_1*y_2*y_0,z_0^3,z_1^3,z_2^3,z_0*y_0-y_0*z_1,z_1*y_1-y_1*z_2,z_2*y_2-y_2*z_0}
GAMS1 = getGammaHash({z_0,y_0,z_1,y_1,y_2,z_2},I,DegreeLimit => 15)
getReducedBT(GAMS1,2)

apply(3, i -> bettiTally(GAMS1,i))

apply(3, i -> startVertex y_(i))
apply(3, i -> endVertex y_(i))
apply(3, i -> startVertex z_(i))
apply(3, i -> endVertex z_(i))



restart
uninstallPackage "PathAlgebras"
restart
debug needsPackage "PathAlgebras"
--check PathAlgebras
installPackage "PathAlgebras"
viewHelp "PathAlgebras"
restart

restart
debug needsPackage "PathAlgebras"
	 adj = matrix {{1,1},{0,1}}
	 G = paGraph({v,w},{e,f,g},adj)
	 R = QQ
	 A = R G
         L = e*f
