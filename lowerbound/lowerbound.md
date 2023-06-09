# Space & Time Complexity

I believe I have achieved the lowerbound runtime and space complexity.

**Time**: $O(n^2)$  
**Space**: $O(n^2)$  

## Undrstanding My Diagrams

Throughout my explanation I will make use of several graph diagrams that organize nodes by layer (amount of labels). For example given the graph,

```
2 -> 1 2
2 -> 2 3
1 2 -> 1 2 3
2 3 -> 1 2 3
1 2 3 -> 1 2 3 4
1 2 3 4 -> 1 2 3 4 5
```

I would draw the graph diagram as:

<img src="./diagramExample1.png" width="40%"/>

but, the direction of the edges can be implied from the layer because nodes with less labels always point to nodes with more labels. And, often the structure of the graph is more important than the arbitrary node contents. So it can be simplified further...

<img src="./diagramExample2.png" width="20%"/>

## Lowerbound Runtime

**Worst case graph**

A worst case set of nodes is one that maximizes the amount of edges in the solution. It would not be disjoint because a connected graph has more edges. Note that it is valid for every layer to draw an edge to every node in the next layer. The worst case graph would include this. The amount of these edges between those two layers in this case would be $|layer_1| \cdot |layer_2|$. This expression is maximized with larger layers and so the worst case graph maximizes edges by maximizing the size of the layers and minimzing the amount of layers.

<img src="./worstCase.png" width="40%"/>

The amount of edges are given by $\lfloor n/2 \rfloor \lceil n/2 \rceil \approx n^2/4$.  
$O(n^2)$ amount of edges.

In this scenario, all edges can be present in the solution, or not. That is, removing any edge, or even every edge, yields a valid solution. In order not to confuse permutations of the solution, the most efficient algorithm would need to confirm if every edge exists in the solution necessarily.

In other words, the fastest algorithm that produces every correct edge will run in proportion to the size of the solution it produces. $O(n^2)$ means such an algorithm is running exponentially in the quantity of nodes, but linear in the quantity of edges.

## Lowerbound space

It may be possible to achieve a space complexity lower than mine. I am not definitive on a space lowerbound. It may be possible to exceed my algorithm if it could keep all nodes in a file and produce the solution without ever having the entire solution or input nodes in memory at once.

However, it is necesarily the case that any function that produces the entire solution in memory would have an $O(n^2)$ space complexity because the solution is $O(n^2)$ in nature.
