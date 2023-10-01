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

// todo Node no longer needs 'potential' field
pub struct Node {
    pub contents: BitSet,
    pub potential: BitSet,
}

impl Node {
    pub fn new(contents: Option<BitSet>, potential: Option<BitSet>) -> Self {
        Node {
            contents: contents.unwrap_or(BitSet::new()),
            potential: potential.unwrap_or(BitSet::new()),
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Node::new(None, None)
    }
}

pub struct DiGraph {
    pub data: Vec<Vec<(Node, EdgeList)>>,
    len: usize,
}

impl DiGraph {
    pub fn new(default_nodes: Option<Vec<BitSet>>) -> Self {
        let nodes = default_nodes.clone().unwrap_or_default();

        DiGraph {
            len: nodes.len(),
            data: vec![
                {
                    nodes
                        .into_iter()
                        .map(|node| (Node::new(Some(node), None), Vec::new()))
                        .collect::<Vec<(Node, EdgeList)>>()
                }
            ],
        }
    }

    pub fn get_node(&self, coord: &NodeCoord) -> &Node {
        &self.data[coord.0 as usize][coord.1].0
    }

    pub fn node_content(&self, coord: &NodeCoord) -> &BitSet {
        &self.get_node(coord).contents
    }

    pub fn edge(&mut self, from: &NodeCoord, to: NodeCoord) {
        self.data[from.0 as usize][from.1].1.push(to);
    }

    /// Out edges from a node.
    pub fn out(&self, from: &NodeCoord) -> &EdgeList {
        &self.data[from.0 as usize][from.1].1
    }

    pub fn add_nodes(&self, new_nodes: Vec<Node>) {
        // todo important for len()
        panic!("Not implemented")
    }

    /// Amount of nodes in the graph.
    pub fn len(&self) -> usize {
        self.len
    }
}

impl Default for DiGraph {
    fn default() -> Self {
        DiGraph::new(None)
    }
}