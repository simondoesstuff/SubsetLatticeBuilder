use std::collections::{HashSet, VecDeque};
use bit_set::BitSet;
use crate::fixed_dag::FixedDAG;


fn is_proper_subset(a: &BitSet, b: &BitSet) -> bool {
    a.is_subset(b) && a.len() < b.len()
}


/// Finds parents of a new node in a graph using a depth-first search.
/// It assumes that the the ideal parent node is within the search.
/// That is, it relies on the entirety of the data that's made of smaller
/// nodes than the new node.
pub fn find_parents_dfs(graph: &FixedDAG, new_node: &usize) -> BitSet {
    let mut new_edges = BitSet::new();
    let mut stack: VecDeque<usize> = VecDeque::from([graph.nodes.len() - 1]);
    let mut visited: HashSet<usize> = HashSet::default();

    while stack.len() > 0 {
        let parent_id = stack.pop_back().unwrap();
        visited.insert(parent_id);

        let childs = &graph.edges[parent_id];

        let mut deadend = true;
        for child_id in childs {
            let child = &graph.nodes[child_id];

            if visited.contains(&child_id) {
                deadend = false;
                continue;
            }

            if is_proper_subset(child, &graph.nodes[*new_node]) {
                stack.push_back(child_id);
                deadend = false;
            }
        }

        if deadend {
            new_edges.insert(parent_id);
        }
    }

    return new_edges;
}