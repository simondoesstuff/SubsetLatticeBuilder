use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use bit_set::BitSet;


struct FixedDAG<'a> {
    nodes: &'a [BitSet],
    edges: Vec<BitSet>,
    roots: &'a [usize],
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


fn extend_edges(graph: &FixedDAG, new_node: &usize) -> Vec<usize> {
    let mut new_edges: Vec<usize> = vec![];
    let mut frontier = graph.roots.to_vec();

    // BFS
    while frontier.len() > 0 {
        let parent_id = frontier.pop().unwrap();
        let parent = &graph.nodes[parent_id];

        let childs = {
            let edges = &graph.edges[parent_id];
            edges.iter().collect::<Vec<usize>>()
        };

        let mut deadend = true;
        for child_id in childs {
            let child = &graph.nodes[child_id];

            if graph.nodes[*new_node].is_subset(child) {
                frontier.push(child_id);
                deadend = false;
            }
        }

        if deadend {
            new_edges.push(parent_id);
        }
    }

    return new_edges;
}


fn main() {
    let path: &str = "data/6.txt";

    // temporary variables
    let node_contents = &parse_input(path)[..]; // nodes are organized by index
    let mut edges = vec![BitSet::new(); node_contents.len()]; // corresponds to node_contents

    // separate nodes into layers (by size)
    let layers: HashMap<usize, Vec<usize>> = nodes_by_size(node_contents);

    // sort layers by size
    let layer_keys = {
        let mut keys = layers.keys().map(|k| *k).collect::<Vec<usize>>();
        keys.sort_by(|a, b| a.cmp(b));
        keys
    };

    let mut graph = FixedDAG {
        nodes: node_contents,
        edges,
        roots: layers[&layer_keys[0]].as_slice()
    };

    println!("layer_keys: {:?}", layer_keys);

    println!("node index: {:?}", graph.nodes);

    println!("roots: {:?}", graph.roots);

    // print
    for (size, nodes) in layers.iter() {
        println!("{}: {:?}", size, nodes);
    }

    let new_edges = extend_edges(&graph, &0);
    println!("new_edges: {:?}", new_edges);
}
