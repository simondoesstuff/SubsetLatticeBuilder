use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufRead, BufReader, LineWriter, Write};

use bit_set::BitSet;
use rayon::prelude::*;

use crate::by_size::nodes_by_size;
use crate::digraph::{DiGraph, NodeCoord};
use crate::traversal::{EdgeOp, inferred_analysis};

mod by_size;
mod digraph;
mod traversal;

fn parse_input(path: &str) -> Vec<BitSet> {
    let file = File::open(path).expect("file not found");
    let reader = BufReader::new(file);

    let node_set = reader
        .lines()
        .map(|line| {
            let line = line.unwrap(); // ignore errors
            let labels = line.split(" ");
            let as_num = labels.map(|label| label.parse::<usize>());
            let mut node = BitSet::new();

            for n in as_num {
                node.insert(n.expect("Failed to parse input. Expected integer values only."));
            }

            return node;
        }).collect::<HashSet<BitSet>>();

    return node_set.into_iter().collect::<Vec<BitSet>>();
}

fn export_graph(path: &str, graph: &DiGraph) {
    let file = File::create(path).unwrap();
    let mut writer = LineWriter::new(file);

    for (x0, row) in graph.data.iter().enumerate() {
        for (x1, node) in row.iter().enumerate() {
            let entry = &graph.data[x0][x1];
            let parent = &entry.0;

            for child_id in &entry.1 {
                let child = &graph.get_node(&child_id);
                let parent_str = parent
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");
                let child_str = child
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<String>>()
                    .join(" ");
                // let line = format!("{} -> {}\n", parent_str, child_str);
                // reversed
                let line = format!("{} -> {}\n", child_str, parent_str);
                writer.write_all(line.as_bytes()).unwrap();
            }
        }
    }

    writer.flush().unwrap();
}

fn inf_constr_alg(in_path: &str, out_path: &str, similarity_coefficient: f32) {
    let mut node_contents = parse_input(in_path); // nodes are organized by index

    // separate nodes into layers (by size)
    let layers: HashMap<usize, Vec<usize>> = nodes_by_size(&node_contents);

    // sort layer keys by size so we iterate higher layers first
    let layer_keys = {
        let mut keys = layers.keys().map(|k| *k).collect::<Vec<usize>>();
        // note the .reverse()
        keys.sort_by(|a, b| a.cmp(b).reverse());
        keys
    };

    let super_node = BitSet::new();
    node_contents.push(super_node);

    let mut graph = DiGraph::new(Some(node_contents));

    // timing
    let t_1 = std::time::Instant::now();
    let mut n_1_sqrt: usize = 0;
    let n_2 = graph.len_nodes() as f64 * graph.len_nodes() as f64;

    // start the algorithm
    for layer_key in layer_keys.iter() {
        // entire layer is handled at once
        let layer = &layers[layer_key];

        let results: Vec<(Vec<EdgeOp>, Vec<BitSet>)> = layer
            .par_iter()
            .map(|new_node| {
                let new_node_id = NodeCoord(0, *new_node);
                let new_node_content = graph.node_content(&new_node_id);
                // note, setting the threshold to length means only direct parent -> child edges are
                // considered, no inferred nodes
                let len: f32 = new_node_content.len() as f32;
                // the ceil is important because a node len 1 can't have a similarity threshold of 0
                // because it would permit inference of null nodes
                let similarity_threshold: u32 = (similarity_coefficient * len).ceil() as u32;
                let (results, inf_nodes) = inferred_analysis(&graph, new_node_id.clone(), similarity_threshold);
                (results, inf_nodes)
            })
            .collect(); // collect -- join the threads

        graph.push_layer();

        let mut inf_nodes = HashMap::<BitSet, NodeCoord>::default();
        let mut seen_ops = HashSet::<EdgeOp>::default();

        for (ops, inferred_nodes) in results {
            // Forks are (usually) in form:     N1 --- x -> N2
            // x represents the intersection. In the fork, it is an id, 0-indexed,
            // and it corresponds to a node in the inferred_nodes vector. These
            // nodes are first added to the graph and the ids are adjusted
            // to represent the coordinates in the graph.

            // Removing Duplicate Operations
            //     Particularly tricky because duplicates can occur in inferred nodes
            //      and the edge operations and the edge operations depend on relative
            //      ids corresponding to inferred nodes.
            //
            //     Approach:   1. HashMap of inferred nodes to their coordinates
            //                    as they are added to the graph
            //                 2. Map edge operations in terms of the coordinates
            //                 3. HashSet of edge operations to remove duplicates

            let mut inf_coords = Vec::<NodeCoord>::with_capacity(inferred_nodes.len());

            for inf in inferred_nodes.into_iter() {
                let coord = inf_nodes.entry(inf.clone()).or_insert_with(|| {
                    graph.add_node(inf)
                });
                inf_coords.push(coord.clone());
            }

            for op1 in ops {
                let op2 = match op1 {
                    EdgeOp::Add_RealReal(_, _) => {
                        op1
                    },
                    EdgeOp::Add_RealInferred(from, to) => {
                        let to = &inf_coords[to as usize];
                        EdgeOp::Add_RealReal(from, to.clone())
                    },
                    EdgeOp::Add_InferredReal(from, to) => {
                        let from = &inf_coords[from as usize];
                        EdgeOp::Add_RealReal(from.clone(), to)
                    },
                    EdgeOp::Del(_, _) => {
                        op1
                    }
                };

                // op2 is the mapping of op1 to a real-real operation
                // adding or deleting, all real-real operations should be unique
                // because adding or deleting the same edge twice is redundant
                // and separate threads with conflicting adding/deleting operations
                // should not be allowed

                if !seen_ops.contains(&op2) {
                    // todo suspicious that some duplicate ops are getting through
                    // because (x,y) != (y,x)

                    seen_ops.insert(op2.clone());
                    match op2 {
                        EdgeOp::Add_RealReal(from, to) => {
                            graph.edge(&from, to);
                        },
                        EdgeOp::Del(from, to) => {
                            graph.remove_edge(&from, &to);
                        },
                        _ => unreachable!()
                    }
                }
            }
        }

        // nodes - 1 accounts for dummy node
        println!("Graph size: {:?} nodes, {:?} edges", graph.len_nodes() - 1, graph.len_edges());

        // timing
        n_1_sqrt += layer.len();
        let t_2: f64 = (std::time::Instant::now() - t_1).as_millis() as f64 / 1000f64;
        let n_1 = n_1_sqrt as f64 * n_1_sqrt as f64;
        let progress = n_1 / n_2;
        let remaining_time = t_2 * n_2 / n_1 - t_2;
        println!(
            "Layer-{} done.\n\t- Progress: {:.2}%",
            layer_key,
            progress * 100.0
        );
        println!("\t- Current duration: {:?}s", t_2);
        println!("\t- ETA: {:?}s", remaining_time);
    }

    // remove null node, which was only used to track roots
    graph.data[0].pop();

    println!("Done. Exporting solution...");
    export_graph(out_path, &graph);
}

fn main() {
    let args = std::env::args().collect::<Vec<String>>();

    // todo remove hard coded paths
    let args = vec![
        args[0].clone(),
        "../data/dirty/1109.txt".to_string(),
        "../data/tmp/1109.soln".to_string(),
        ".9".to_string(),
    ];

    let in_path = if let Some(path) = args.get(1) {
        path
    } else {
        eprintln!("No input file specified");
        std::process::exit(1);
    };

    let out_path = if let Some(path) = args.get(2) {
        path
    } else {
        eprintln!("No output file specified");
        std::process::exit(1);
    };

    let similarity = if let Some(similarity) = args.get(3) {
        similarity.parse::<f32>().expect("Failed to parse similarity coefficient.")
    } else {
        eprintln!("No similarity coefficient specified. Defaulting to 1");
        1.0
    };

    println!("Running with similarity coefficient: {}", similarity);

    inf_constr_alg(in_path, out_path, similarity);
}
