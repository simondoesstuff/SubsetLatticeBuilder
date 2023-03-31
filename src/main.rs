use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, LineWriter, Write};
use bit_set::BitSet;


struct FixedDAG<'a> {
    nodes: &'a [BitSet],
    edges: Vec<BitSet>,
    roots: &'a [usize],
}


fn is_proper_subset(a: &BitSet, b: &BitSet) -> bool {
    a.is_subset(b) && a.len() < b.len()
}


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


fn find_parents(graph: &FixedDAG, new_node: &usize) -> BitSet {
    let mut new_edges = BitSet::new();

    let mut frontier = {
        let new_node_contents = &graph.nodes[*new_node];
        let mut frontier = BitSet::new();

        for root in graph.roots.iter() {
            let root_contents = &graph.nodes[*root];
            if is_proper_subset(root_contents, new_node_contents) {
                frontier.insert(*root);
            }
        }

        frontier
    };

    // BFS
    while frontier.len() > 0 {
        // frontier.pop()
        let parent_id = frontier.iter().take(1).next().unwrap();
        frontier.remove(parent_id);

        let childs = {
            let edges = &graph.edges[parent_id];
            edges.iter().collect::<Vec<usize>>()
        };

        let mut deadend = true;
        for child_id in childs {
            let child = &graph.nodes[child_id];

            if is_proper_subset(child, &graph.nodes[*new_node]) {
                frontier.insert(child_id);
                deadend = false;
            }
        }

        if deadend {
            new_edges.insert(parent_id);
        }
    }

    return new_edges;
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


fn main() {
    let path: &str = "data/1109.txt";

    // temporary variable
    let node_contents = &parse_input(path)[..]; // nodes are organized by index

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

    let t_1 = std::time::Instant::now();
    let mut n_1_sqrt: usize = 0;
    let n_2 = node_contents.len() as f64 * node_contents.len() as f64;

    // start the algorithm
    for layer_key in layer_keys.iter().skip(1) {
        // entire layer is handled at once
        let layer = &layers[layer_key];
        n_1_sqrt += layer.len();

        for new_node in layer {
            let new_edges = find_parents(&graph, &new_node);
            for edge in &new_edges {
                graph.edges[edge].insert(*new_node);
            }
        }

        let t_2: f64 = (std::time::Instant::now() - t_1).as_millis() as f64 / 1000 as f64;
        let n_1 = n_1_sqrt as f64 * n_1_sqrt as f64;
        let progress = n_1 / n_2;
        println!("Layer-{} done.\n\t- Progress: {:.2}%", layer_key, progress * 100.0);
        println!("\t- Current duration: {:?}s", t_2);
    }

    println!("Done. Exporting solution...");
    export_graph("data/1109.soln", &graph);
}
