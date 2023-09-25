use crate::digraph::{DiGraph, Node, NodeCoord};
use bit_set::BitSet;
use std::collections::{HashSet, VecDeque};

fn is_proper_subset(a: &BitSet, b: &BitSet) -> bool {
    a.is_subset(b) && a.len() < b.len()
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
pub fn find_parents_dfs(graph: &DiGraph, new_node: NodeCoord) -> HashSet<NodeCoord> {
    // todo can this be a vector?
    let mut new_edges: HashSet<NodeCoord> = HashSet::default();
    let origin_node = NodeCoord(0, graph.len() - 1);
    let mut stack: VecDeque<NodeCoord> = VecDeque::from([origin_node]);
    let mut visited: HashSet<NodeCoord> = HashSet::default();

    while stack.len() > 0 {
        let parent_id = stack.pop_back().unwrap();
        let childs = graph.out(&parent_id);
        visited.insert(parent_id.clone());

        let mut deadend = true;
        for child_id in childs {
            if visited.contains(&child_id) {
                deadend = false;
                continue;
            }

            let child_content = graph.node_content(child_id);
            let new_node_content = graph.node_content(&new_node);

            if is_proper_subset(child_content, new_node_content) {
                stack.push_back((*child_id).clone());
                deadend = false;
            }
        } // release lock

        if deadend {
            new_edges.insert(parent_id);
        }
    }

    return new_edges;
}
