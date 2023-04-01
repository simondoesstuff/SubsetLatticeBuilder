mod traversal;
mod fixed_dag;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, LineWriter, Write};
use bit_set::BitSet;
use rayon::prelude::*;
use crate::fixed_dag::FixedDAG;
use crate::traversal::find_parents_dfs;


fn parse_input(path: &str) -> Vec<BitSet> {
    let file = File::open(path).expect("file not found");
    let reader = BufReader::new(file);

    return reader.lines().map(|line| {
        let line = line.unwrap(); // ignore errors
        let labels = line.split(" ");
        let as_num = labels.map(|label| label.parse::<usize>());
        let mut bs = BitSet::new();

        for n in as_num {
            bs.insert(n.unwrap());
        }

        return bs;
    }).collect::<Vec<BitSet>>();
}


fn nodes_by_size(node_contents: &[BitSet]) -> HashMap<usize, Vec<usize>> {
    let mut by_size = HashMap::new();

    for (i, node) in node_contents.iter().enumerate() {
        let size = node.len();
        by_size
            .entry(size)
            .or_insert(vec![])
            .push(i);
    }

    by_size
}


fn export_graph(path: &str, graph: &FixedDAG) {
    let file = File::create(path).unwrap();
    let mut writer = LineWriter::new(file);

    for (parent_id, edges) in graph.edges.iter().enumerate() {
        for child_id in edges {
            let parent = &graph.nodes[parent_id];
            let child = &graph.nodes[child_id];
            let parent_str = parent.iter().map(|n| n.to_string()).collect::<Vec<String>>().join(" ");
            let child_str = child.iter().map(|n| n.to_string()).collect::<Vec<String>>().join(" ");
            let line = format!("{} -> {}\n", parent_str, child_str);
            writer.write_all(line.as_bytes()).unwrap();
        }
    }

    writer.flush().unwrap();
}


fn solve(in_path: &str, out_path: &str) {
    // temporary variable
    let node_contents = &parse_input(in_path)[..]; // nodes are organized by index

    // separate nodes into layers (by size)
    let layers: HashMap<usize, Vec<usize>> = nodes_by_size(node_contents);

    // sort layer keys by size so we iterate higher layers first
    let layer_keys = {
        let mut keys = layers.keys().map(|k| *k).collect::<Vec<usize>>();
        keys.sort_by(|a, b| a.cmp(b));
        keys
    };

    let mut graph = FixedDAG {
        nodes: node_contents,
        edges: vec![BitSet::new(); node_contents.len()],
        roots: layers[&layer_keys[0]].as_slice(),
    };

    // timing
    let t_1 = std::time::Instant::now();
    let mut n_1_sqrt: usize = 0;
    let n_2 = node_contents.len() as f64 * node_contents.len() as f64;

    // start the algorithm
    for layer_key in layer_keys.iter().skip(1) {
        // entire layer is handled at once
        let layer = &layers[layer_key];

        let new_edges: Vec<(&usize, BitSet)> = layer
            .par_iter()
            .map(|new_node| {
                (new_node, find_parents_dfs(&graph, new_node))
            })
            .collect(); // collect -- join the threads

        // apply changes in sync
        for (child, edges) in new_edges {
            for parent in &edges {
                graph.edges[parent].insert(*child);
            }
        }

        // timing
        n_1_sqrt += layer.len();
        let t_2: f64 = (std::time::Instant::now() - t_1).as_millis() as f64 / 1000 as f64;
        let n_1 = n_1_sqrt as f64 * n_1_sqrt as f64;
        let progress = n_1 / n_2;
        let remaining_time = t_2 * n_2 / n_1 - t_2;
        println!("Layer-{} done.\n\t- Progress: {:.2}%", layer_key, progress * 100.0);
        println!("\t- Current duration: {:?}s", t_2);
        println!("\t- ETA: {:?}s", remaining_time);
    }

    println!("Done. Exporting solution...");
    export_graph(out_path, &graph);
}


fn main() {
    let args = std::env::args().collect::<Vec<String>>();

    let in_path = if let Some(path) = args.get(1) { path } else {
        eprintln!("No input file specified");
        std::process::exit(1);
    };

    let out_path = if let Some(path) = args.get(2) { path } else {
        eprintln!("No output file specified");
        std::process::exit(1);
    };

    solve(in_path, out_path);
}
