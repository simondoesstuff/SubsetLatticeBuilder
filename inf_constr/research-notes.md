## Explore this:

10/8/2023  
(Bottom-up trickle)
I use a dummy node's edge list to point to the leaf nodes. If I add the dummy node before I sort the layers by size,
but skip the largest layer versus sort the layers, then add the node, similarly avoiding iteration of the dummy node...
I am seeing a difference in edges added.  
Also, the content of the dummy node shouldn't matter. However,
it is drastically changing the behavior of the program. Likely implementation specific issues.

10/9/2023  
The edge counter in the digraph seems to be consistently slightly higher
than the actual amount as seen by exporting the graph. This implies there are not
duplicate edges, but that the counter itself is off.  
I don't think the dummy node's contents any longer has an affect, but the amount of edges (exported) varies with each
run slightly (regardless of parameters).