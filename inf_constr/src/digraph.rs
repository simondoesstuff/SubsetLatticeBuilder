use std::fmt::{Debug, Formatter};
use bit_set::BitSet;


/// Pass ID, Node ID
#[derive(Eq, Hash, PartialEq)] // todo learn this
pub struct NodeCoord(pub u16, pub usize);

impl Clone for NodeCoord {
    fn clone(&self) -> Self {
        NodeCoord(self.0, self.1)
    }
}

impl Debug for NodeCoord {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.0, self.1)
    }
}

pub type EdgeList = Vec<NodeCoord>;
pub type Node = BitSet;

pub struct DiGraph {
    pub data: Vec<Vec<(Node, EdgeList)>>,
    len_node: usize,
    len_edge: usize,
}

impl DiGraph {
    pub fn new(default_nodes: Option<Vec<BitSet>>) -> Self {
        let nodes = default_nodes.clone().unwrap_or_default();

        DiGraph {
            len_node: nodes.len(),
            len_edge: 0,
            data: vec![
                {
                    nodes
                        .into_iter()
                        .map(|node| (node, Vec::new()))
                        .collect::<Vec<(Node, EdgeList)>>()
                }
            ],
        }
    }

    pub fn get_node(&self, coord: &NodeCoord) -> &Node {
        &self.data[coord.0 as usize][coord.1].0
    }

    pub fn node_content(&self, coord: &NodeCoord) -> &BitSet {
        &self.get_node(coord)
    }

    pub fn splice_edge(&mut self, from: &NodeCoord, middle: NodeCoord, to: NodeCoord) {
        // take edge from --> to
        // and replace with from --> middle --> to
        self.remove_edge(from, &to);
        self.edge(from, middle.clone());
        self.edge(&middle, to.clone());
    }

    pub fn edge(&mut self, from: &NodeCoord, to: NodeCoord) {
        self.len_edge += 1;
        self.data[from.0 as usize][from.1].1.push(to);
    }

    pub fn remove_edge(&mut self, from: &NodeCoord, to: &NodeCoord) {
        self.len_edge -= 1;
        self.data[from.0 as usize][from.1].1.retain(|n| *n != *to);
    }

    /// Out edges from a node.
    pub fn out(&self, from: &NodeCoord) -> &EdgeList {
        &self.data[from.0 as usize][from.1].1
    }

    /// Adds a new layer to the graph. New nodes are added to the top layer.
    pub fn push_layer(&mut self) {
        // shouldn't ever have an issue, graph should be initialized with at least one layer
        // this should significantly reduce resizing
        let prev_capacity = self.data.last().unwrap().capacity();
        self.data.push(Vec::with_capacity(prev_capacity));
    }

    pub fn next_open_coord(&self) -> NodeCoord {
        let top_layer = self.data.len() - 1;
        let top_node = self.data[top_layer].len();
        NodeCoord(top_layer as u16, top_node)
    }

    /// Adds a set of nodes to the latest layer of the graph.
    /// Returns the coordinates that were assigned.
    pub fn bulk_add(&mut self, new_nodes: Vec<Node>) -> Vec<NodeCoord> {
        self.len_node += new_nodes.len();
        let mut coords = Vec::with_capacity(new_nodes.len());
        let top = self.next_open_coord();

        for (i, node) in new_nodes.into_iter().enumerate() {
            let coord = NodeCoord(top.0, top.1 + i);
            self.data[top.0 as usize].push((node, Vec::new()));
            coords.push(coord);
        }

        coords
    }

    pub fn add_node(&mut self, new_node: Node) -> NodeCoord {
        self.len_node += 1;
        let coord = self.next_open_coord();
        self.data[coord.0 as usize].push((new_node, Vec::new()));
        coord
    }

    pub fn len_nodes(&self) -> usize {
        self.len_node
    }

    pub fn len_edges(&self) -> usize {
        self.len_edge
    }
}

impl Default for DiGraph {
    fn default() -> Self {
        DiGraph::new(None)
    }
}