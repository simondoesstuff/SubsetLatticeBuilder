use std::collections::HashMap;

use bit_set::BitSet;


pub fn nodes_by_size(node_contents: &[BitSet]) -> HashMap<usize, Vec<usize>> {
    let mut by_size = HashMap::new();

    for (i, node) in node_contents.iter().enumerate() {
        let size = node.len();
        by_size
            .entry(size)
            .or_insert(vec![])
            .push(i);
    }

    return by_size;
}