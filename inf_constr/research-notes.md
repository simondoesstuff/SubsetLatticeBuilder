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

## Next Steps:

10/10/2023  
1. Determine what's responsible for race-condition dependent nodes/edges
   - Remove the no decrease constraint. Likely reundant anyway...
2. Compare graph neighbors about query node with KNN scan over starting nodes.
   - This probably entails removing inferred nodes and collapsing the edges they inhabitted.
3. Simply "clumps" of attributes
   - This might make debugging easier as nodes with become easier to read. It should also
  reduce the memory footprint and improve speed.
4. Try rebuilding graph of inferred (& observed) nodes on 100% similarity afterward.

### For future,
Is there a viable algorithm to verify the solutions?  
Tricky because while detecting wrong edges could be viable via random checks (depending on the condition, like $A \subset B$),
detecting the lack of a right edge requires knowing what they are. Is verifying the solution and solving the solution
one in the same?
