use bit_set::BitSet;
use parking_lot::RwLock;


type EdgeList = RwLock<BitSet>;

pub struct Node {
    pub contents: BitSet,
    pub potential: BitSet,
}

impl Node {
    pub fn new() -> Self {
        Node {
            contents: BitSet::new(),
            potential: BitSet::new(),
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Node::new()
    }
}

// exclusively relies on interior mutability
pub struct DiGraph {
    pub nodes: RwLock<Vec<Node>>,
    pub edges: RwLock<Vec<EdgeList>>,
}

impl DiGraph {
    pub fn new() -> Self {
        return DiGraph {
            nodes: RwLock::new(Vec::new()),
            edges: RwLock::new(Vec::new()),
        };
    }

    pub fn edge(&self, from: usize, to: usize) {
        self.edges.read()[from].write().insert(to);
    }

    /// Out edges from a node.
    pub fn out(&self, from: usize) -> BitSet {
        self.edges.read()[from].read().clone()
    }

    /// Expensive operation because it requires a lock on the entire graph.
    pub fn add_node(&self, new_node: Node) {
        let mut nodes = self.nodes.write();
        let mut edges = self.edges.write();
        nodes.push(new_node);
        edges.push(RwLock::new(BitSet::new()));
    }

    /// Expensive operation because it requires a lock on the entire graph.
    pub fn add_nodes(&self, new_nodes: Vec<Node>) {
        let mut nodes = self.nodes.write();
        let mut edges = self.edges.write();
        for node in new_nodes {
            nodes.push(node);
            edges.push(RwLock::new(BitSet::new()));
        }
    }
}

impl Default for DiGraph {
    fn default() -> Self {
        DiGraph::new()
    }
}