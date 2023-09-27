use crate::digraph::{DiGraph, Node, NodeCoord};
use bit_set::BitSet;
use std::collections::{HashSet, VecDeque};

fn is_proper_subset(a: &BitSet, b: &BitSet) -> bool {
    a.len() < b.len() && a.is_subset(b)
}

fn is_proper_superset(a: &BitSet, b: &BitSet) -> bool {
    a.len() < b.len() && a.is_superset(b)
}

/// Finds the parents of the new node using a DFS.
/// In the case that the successors are no longer
/// supersets of the node, their parent edges are
/// forked, by inserting the intersection (inferred node).
/// Inferred nodes and new edges are returned.
pub fn scan() {}

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
