use std::collections::{HashSet, VecDeque};
use bit_set::BitSet;
use crate::fixed_dag::FixedDAG;


fn is_proper_subset(a: &BitSet, b: &BitSet) -> bool {
    a.is_subset(b) && a.len() < b.len()
}


// pub fn find_parents_bfs(graph: &FixedDAG, new_node: &usize) -> BitSet {
//     let mut new_edges = BitSet::new();
//
//     let mut frontier = {
//         let new_node_contents = &graph.nodes[*new_node];
//         let mut frontier = BitSet::new();
//
//         for root in graph.roots.iter() {
//             let root_contents = &graph.nodes[*root];
//             if is_proper_subset(root_contents, new_node_contents) {
//                 frontier.insert(*root);
//             }
//         }
//
//         frontier
//     };
//
//     // BFS
//     while frontier.len() > 0 {
//         // frontier.pop
//         let parent_id = frontier.iter().take(1).next().unwrap();
//         frontier.remove(parent_id);
//
//         let childs = {
//             let edges = &graph.edges[parent_id];
//             edges.iter().collect::<Vec<usize>>()
//         };
//
//         let mut deadend = true;
//         for child_id in childs {
//             let child = &graph.nodes[child_id];
//
//             if is_proper_subset(child, &graph.nodes[*new_node]) {
//                 frontier.insert(child_id);
//                 deadend = false;
//             }
//         }
//
//         if deadend {
//             new_edges.insert(parent_id);
//         }
//     }
//
//     return new_edges;
// }


pub fn find_parents_dfs(graph: &FixedDAG, new_node: &usize) -> BitSet {
    let mut new_edges = BitSet::new();

    let mut stack = {
        let new_node_contents = &graph.nodes[*new_node];
        let mut stack: VecDeque<usize> = VecDeque::new();

        for root in graph.roots.iter() {
            let root_contents = &graph.nodes[*root];
            if is_proper_subset(root_contents, new_node_contents) {
                stack.push_back(*root);
            }
        }

        stack
    };

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