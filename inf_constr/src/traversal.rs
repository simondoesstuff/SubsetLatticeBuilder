use crate::digraph::{DiGraph, NodeCoord};
use bit_set::{BitSet};
use std::collections::{HashMap, HashSet, VecDeque};

pub enum EdgeOp {
    Add_RealReal(NodeCoord, NodeCoord),
    Add_RealInferred(NodeCoord, u16),
    Add_InferredReal(u16, NodeCoord),
    Del(NodeCoord, NodeCoord)
}

fn is_proper_subset(a: &BitSet, b: &BitSet) -> bool {
    a.len() < b.len() && a.is_subset(b)
}

fn is_proper_superset(a: &BitSet, b: &BitSet) -> bool {
    a.len() < b.len() && a.is_superset(b)
}

fn intersect(a: &BitSet, b: &BitSet) -> BitSet {
    // todo note allocating and freeing vec here is slow
    let intersection = a.intersection(b);
    let capacity = {
        let hint = intersection.size_hint();
        if hint.1.is_some() {
            hint.1.unwrap()
        } else {
            hint.0
        }
    };

    let mut vec = Vec::<u32>::with_capacity(capacity);
    for n in intersection {
        vec.push(n as u32);
    }

    let max = vec.iter().max().unwrap_or(&0);
    let mut bitset = BitSet::with_capacity(*max as usize + 1);

    for n in vec {
        bitset.insert(n as usize);
    }

    return bitset;
}

/// Finds forks by traversing the graph from the bottom-up. While scanning,
/// it will find edges where the new node is a subset of one endpoint and a
/// superset of the other. In this case, it creates a new node representing
/// the relevant intersection and stores the fork. Forks will only be created
/// if the magnitude of the intersection is >= similarity_min.
pub fn inferred_analysis(graph: &DiGraph, new_id: NodeCoord, similarity_threshold: u32) -> (Vec<EdgeOp>, Vec<BitSet>) {
    let new_node = graph.node_content(&new_id);

    let mut ops = Vec::<EdgeOp>::default();
    let mut new_nodes = HashMap::<BitSet, u16>::default();
    let mut visited = HashSet::<NodeCoord>::default();

    let origin_node_coord = NodeCoord(0, graph.data[0].len() - 1); // dummy node stores leaf nodes
    let leafs: Vec<(NodeCoord, BitSet)> = graph.out(&origin_node_coord)
        .iter().map(|n| (n.clone(), intersect(graph.node_content(n), new_node)))
        .filter(|(_, int)| int.len() >= similarity_threshold as usize)
        .collect();
    let mut stack = VecDeque::from(leafs);

    // there are no viable leaf nodes to start from
    if stack.len() == 0 {
        ops.push(EdgeOp::Add_RealReal(origin_node_coord.clone(), new_id.clone()));
        return (ops, Vec::default());
    }

    // given an "ideal" fork location, the intersections of the supersets
    // following the fork don't alter the intersection at all, but
    // following past the fork to the subsets, will reduce the intersection.
    // ie: essentially following the intersection along a path.

    while stack.len() > 0 {
        let (n1_id, n1_int) = stack.pop_back().unwrap();
        let n1_int_len = n1_int.len();

        visited.insert(n1_id.clone());
        let candidates = graph.out(&n1_id); // out edges

        let mut dead_end = true; // assume dead end until proven otherwise

        for n2_id in candidates {
            if visited.contains(&n2_id) {
                // this is not considered a dead end because
                // one of the previous searches has already
                // gone 'through' this node
                dead_end = false;
                continue;
            }

            let n2 = graph.node_content(&n2_id);
            let n2_int = intersect(n2, new_node);
            let n2_int_len = n2_int.len();

            if n2_int_len >= n1_int_len {
                if n2_int_len >= similarity_threshold as usize {
                    // this path is viable
                    stack.push_back((n2_id.clone(), n2_int));
                    dead_end = false;
                }
            }
        }

        // in the case of dead end, no new viable paths were found
        // we have two options:
        //   1) N1 --> New
        //          That is, the new node is a (in bottom-up context) leaf node.
        //          Only operation is to add the edge.
        //   2) New, N1 --> x
        //          Then, for a given N2 in N1.neighbors:
        //              2a) x ---> N2
        //              2b) x -/-> N2
        //
        //          This requires adding at least one edge and node
        //          and moving (removing and adding) an edge.
        if dead_end {
            if n1_int == *new_node {
                // case 1, new node is a leaf node
                ops.push(EdgeOp::Add_RealReal(n1_id.clone(), new_id.clone()));
            } else {
                // case 2, there is an inferred node
                let inf_id = {
                    let next = new_nodes.len() as u16;
                    new_nodes
                        .entry(n1_int.clone())
                        .or_insert(next)
                };

                ops.push(EdgeOp::Add_RealInferred(new_id.clone(), *inf_id));

                // for each N2 in N1.neighbors: x --?--> N2
                for n2_id in candidates {
                    let n2 = graph.node_content(&n2_id);

                    if is_proper_subset(n2, &n1_int) {
                        // x ---> N2
                        ops.push(EdgeOp::Del(n1_id.clone(), n2_id.clone()));
                        ops.push(EdgeOp::Add_InferredReal(*inf_id, n2_id.clone()));
                    }
                }
            }
        }
    }

    // the new_nodes map (node -> int) needs to be converted to a vec (int -> node)

    let mut new_nodes_vec = {
        // this vector should never be resizing
        let mut vec = vec![BitSet::default(); new_nodes.len()];
        // unfolding the hashmap into vector
        let l = new_nodes.len();
        for (node, i) in new_nodes {
            vec.insert(i as usize, node);
        }
        vec
    };

    return (ops, new_nodes_vec);
}

/// Finds parents of a new node in a graph using a depth-first search.
/// It assumes that the the ideal parent node is within the search.
/// That is, it relies on the entirety of the data that's made of smaller
/// nodes than the new node.
pub fn find_children_dfs(graph: &DiGraph, current_node_id: NodeCoord) -> Vec<NodeCoord> {
    let mut new_edges: Vec<NodeCoord> = Vec::default();
    let origin_node = NodeCoord(0, graph.len() - 1);
    let mut stack: VecDeque<NodeCoord> = VecDeque::from([origin_node]);
    let mut visited: HashSet<NodeCoord> = HashSet::default();

    while stack.len() > 0 {
        let pop = stack.pop_back().unwrap();
        let candidates = graph.out(&pop);
        visited.insert(pop.clone());

        let mut deadend = true;
        for candidate_id in candidates {
            if visited.contains(&candidate_id) {
                deadend = false;
                continue;
            }

            let candidate = graph.node_content(candidate_id);
            let current = graph.node_content(&current_node_id);

            if is_proper_subset(current, candidate) {
                stack.push_back((*candidate_id).clone());
                deadend = false;
            }
        }

        if deadend {
            new_edges.push(pop);
        }
    }

    return new_edges;
}
