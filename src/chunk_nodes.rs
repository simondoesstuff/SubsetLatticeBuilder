use std::collections::{HashMap, HashSet};
use bit_set::BitSet;
use crate::fixed_dag::FixedDAG;
use crate::id_generator::IdGenerator;


fn nodes_by_size(node_contents: &[BitSet]) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
    let mut by_size = HashMap::new();

    for (i, node) in node_contents.iter().enumerate() {
        let size = node.len();
        by_size
            .entry(size)
            .or_insert(vec![])
            .push(i);
    }

    let layer_keys = {
        let mut keys = layers.keys().map(|k| *k).collect::<Vec<usize>>();
        keys.sort_by(|a, b| a.cmp(b));
        keys
    };

    return (by_size, layer_keys);
}


pub fn chunk_nodes<'a>(graph: &FixedDAG, nodes: &'a[usize]) -> Vec<&'a [usize]> {
    let (layers, layer_keys) = nodes_by_size(&graph.nodes);

    if layer_keys.len() <= 2 {
        return vec![nodes];
    }

    // let mut chunks: HashSet<
    let mut pivot = Some(layer_keys[layer_keys.len() / 2][0]);

    let mut remaining: Vec<&usize> = {
        nodes
            .iter()
            .filter(|n| **n != pivot.unwrap())
            .collect()
    };

    while let Some(pivot) = pivot {
        let mut new_chunk: &Vec<&usize>;

        for chunk in chunks {
            // need to find a chunk that contains the pivot
            // and have it merge into the new_chunk.
            // but, not sure how the concurrent modification will work.
        }
    }

    return chunks;
}