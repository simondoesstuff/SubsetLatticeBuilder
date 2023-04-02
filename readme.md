# Subset Lattice Builder

Simon Walker - April, 2023.

Constructs a subset lattice from a finite amount of "allowed" nodes.

## High Performance

On a dataset with [79,867 nodes](data/79867.txt), it finishes in **12.914s**.

```sh
Layer-43 done.
    - Progress: 100.00%
    - Current duration: 12.914s
    - ETA: 0.0003233896564758254s
Done. Exporting solution...
```

See more details about this test here: [vm benchmark](vm_benchmark.md).

## Usage

Build from source  
`RUSTFLAGS="-C target-cpu=native" cargo build --release`

Run the executable in target/release  
`subsetlatticebuilder "inputFile.txt" "outputFile.txt"`

### File Formats

**Input File**  
Each line represents a node. Each label within the node is space-delineated.
```
2
1 2
2 3
1 2 3
1 2 3 4
1 2 3 4 5
```

**Output File**  
Each line represents an edge in the form `parent -> child`.
```
2 -> 1 2
2 -> 2 3
1 2 -> 1 2 3
2 3 -> 1 2 3
1 2 3 -> 1 2 3 4
1 2 3 4 -> 1 2 3 4 5
```

## Algorithm

Everything's in Rust. It's using two-  
**dependencies**
1. bit-set
    - A set implementation based on a bit array from (bit-vec). A bit vector mainly makes storage very efficient because each value occupies one bit and lookups can be done fast with bitwise operations. Insert operations are not necessarily faster than the standard HashSet though.
2. rayon
    - Convenient library for multi-threading. Auotmatically balances jobs between threads and **scales the pool to the amount of available cores**.

### **Traversal**

The algorithm performs many insert operations, one per node.  
**Insert ( graph, new\_node )**  
1. Traverse the graph either from top-down or bottom-up. The search is agnostic to the type of traversal. But, let's assume a top-down DFS.
2. Only continue to search along edges that satisfy $parent \subset new\_node \subset child$.
3. Search until we find a parent that satisfies $parent \subset new\_node$ and there is no child of the parent such that $new\_node \subset child$. Append the edge $parent \to new\_node$.

**DFS vs BFS**  
In my experimentation, the DFS is *much, much* more performant for my data. I suspect this is because the DFS more efficiently finds the goal edges which are generally deep within a wide graph whereas BFS wastes time meandering at the surface.

### **Parallelization**
Think of the top-down traversal (as I use in my implementation) as the process of finding a suitable parent for the $new\_node$. Therefore, if the suitable parent does not exist in the graph during the traversal, the algorithm will draw unnecessary (and wrong) edges.

So, any given node *depends* on nodes with less labels in a top-down search, but *NOT* on nodes with a greater than or equal amount of labels. This allows me to parallelize by-

**Breaking up the nodes by layer**  
where each *layer* represents a group of nodes with equal amounts of labels.

1. Divide nodes into layers.
2. Iterate layers ascending.
3. Perform searches for all nodes in the current layer in parallel
    - This is okay because nodes of equal amounts of labels do not depend on each other.
4. Apply all found edges to the graph in sync (on the main thread).
    - This could also be easily parallelized too if the edges are implemented with an adjacency matrix. I use an adjacency list for the memory savings and minimal performance cost.

### **Binary representation** for edges and labels

Notice that all labels and edges are binary attributes. I organize nodes by index (sequential IDs). The labels within a node are also sequential - for my dataset of 79,867 nodes, every label is $< ~~~\sim 720$. This enabled me to represent all edges in an adjacency list, that is, a list of BitSets with each index corresponding to a node. Each index also corresponds to an index in another list of BitSets reprsenting node labels.

## Time & Space Complexity

I believe I have achieved the lowerbound runtime and space complexity. See my explanation, [here](lowerbound/lowerbound.md).

**Time**: $O(n^2)$  
**Space**: $O(n^2)$  
