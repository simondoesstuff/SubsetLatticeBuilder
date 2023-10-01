use crate::digraph::{DiGraph, Node, NodeCoord};
use bit_set::{BitSet, Intersection};
use std::collections::{HashMap, HashSet, VecDeque};

fn is_proper_subset(a: &BitSet, b: &BitSet) -> bool {
    a.len() < b.len() && a.is_subset(b)
}

fn is_proper_superset(a: &BitSet, b: &BitSet) -> bool {
    a.len() < b.len() && a.is_superset(b)
}

// note that u16 may not be sufficient for large graphs
/// form: (N1, x, N2), representing the fork along edge N1 --- x --> N2
/// Note that x is the intersection of N1 and N2. And,
/// the edges are bottom-up, that is N1 is a superset of N2.
/// If the fork denotes a leaf node connection and not a fork,
/// the form is (N1, none) implying N1 ---> New Node
pub type Fork = (NodeCoord, Option<(u16, NodeCoord)>);

fn intersect(a: &BitSet, b: &BitSet) -> BitSet {
    let intersection = a.intersection(b);
    let mut vec = Vec::<u32>::with_capacity(intersection.size_hint().0);

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
pub fn find_forks(graph: &DiGraph, new_id: NodeCoord, similarity_threshold: u32) -> (Vec<Fork>, Vec<BitSet>) {
    let mut forks = Vec::<Fork>::default();
    let mut new_nodes = HashMap::<BitSet, u16>::default();
    let mut visited = HashSet::<NodeCoord>::default();

    // form (N1, x intersection with N1, N2)
    // ie: N1 -> N2, storing edges (N1 -> x -> N2) and potential node x
    let origin_node: (NodeCoord, BitSet, NodeCoord) = (
        NodeCoord(0, 0),
        BitSet::new(),
        NodeCoord(0, graph.len() - 1),  // super node with every attribute
    );
    let mut stack = VecDeque::from([origin_node]);

    let new_node = graph.node_content(&new_id);

    // given an "ideal" fork location, the intersections of the supersets
    // following the fork don't alter the intersection at all, but
    // following past the fork to the subsets, will reduce the intersection.
    // ie: essentially following the intersection along a path.

    while stack.len() > 0 {
        // note nodes are not visited at pop time
        let pop = stack.pop_back().unwrap(); // considered as N1 -> x -> N2
        let candidates = graph.out(&pop.2); // out edges

        let mut dead_end = true; // assume dead end until proven otherwise

        for n3_id in candidates {
            // think
            // (N1 -- x -> N2) -- y -> N3
            // x = N2 & New, y = N3 & New

            if visited.contains(n3_id) {
                // this is not considered a dead end because
                // one of the previous searches has already
                // gone 'through' this node
                dead_end = false;
                continue;
            }

            // note that the node is NOT visited at this point either.
            // only if the node contains further edges that are worth
            // forking. That is, the search goes down that path. If this
            // search ends here and forks the previous edge, the
            // node is not considered visited.

            let n3 = graph.node_content(&n3_id);
            let n3_intersect = intersect(n3, new_node);
            let n3_intersect_len = n3_intersect.len();

            if n3_intersect_len >= pop.1.len() {
                if n3_intersect_len >= similarity_threshold as usize {
                    // this path is viable
                    stack.push_back((pop.2.clone(), n3_intersect, n3_id.clone()));
                    // since we pushed the node (continuing down the path)
                    // the next node is now considered visited
                    visited.insert(n3_id.clone());
                    dead_end = false;
                }
            }
        }

        // in the case of dead end, no new viable paths were found
        // we have two options:
        //   1) fork the previous edge, or
        //   2) make the new node -> previous node
        //          That is, the new node is a (in top-down context) leaf node.
        if dead_end {
            // note that pop is in form (N1,x,N2):  N1 --- x --> N2

            let n2 = graph.node_content(&pop.2);
            if is_proper_subset(&pop.1, n2) {
                // case 2, new node is a leaf node
                forks.push((pop.2, None));
            } else {
                // case 1, fork the previous edge
                let intersect_id = {
                    let next = new_nodes.len() as u16;
                    new_nodes
                        .entry(pop.1)
                        .or_insert(next)
                };

                forks.push((pop.0, Some((*intersect_id, pop.2))));
            }
        }
    }

    // the new_nodes map (node -> int) needs to be converted to a vec (int -> node)

    let mut new_nodes_vec = {
        // this vector should never be resizing
        let mut vec = Vec::<BitSet>::with_capacity(new_nodes.len());
        // unfolding the hashmap into vector
        for (node, i) in new_nodes {
            vec.insert(i as usize, node);
        }
        vec
    };

    return (forks, new_nodes_vec);
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
