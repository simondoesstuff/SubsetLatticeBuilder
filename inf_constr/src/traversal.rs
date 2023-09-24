use crate::digraph::{DiGraph, Node};
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
pub fn find_parents_dfs(graph: &DiGraph, new_node: &usize) {
    // todo should be a hashset
    let mut new_edges = BitSet::new();
    // todo not sure why this is starting at the end
    let mut stack: VecDeque<usize> = VecDeque::from([graph.len() - 1]);
    let mut visited: HashSet<usize> = HashSet::default();

    while stack.len() > 0 {
        let parent_id = stack.pop_back().unwrap();
        visited.insert(parent_id);

        let childs = graph.out(parent_id);

        let mut deadend = true;
        for child_id in &childs {
            if visited.contains(&child_id) {
                deadend = false;
                continue;
            }

            let nodes = graph.nodes.read();
            let child_content = &nodes[child_id].contents;
            let new_node_content = &nodes[*new_node].contents;

            if is_proper_subset(child_content, new_node_content) {
                stack.push_back(child_id);
                deadend = false;
            }
        } // release lock

        if deadend {
            new_edges.insert(parent_id);
        }
    }

    // write new edges

    let edges = graph.edges.read();

    for parent_id in &new_edges {
        edges[parent_id].write().insert(*new_node);
    } // release lock
}
