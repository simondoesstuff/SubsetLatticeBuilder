use bit_set::BitSet;

pub struct FixedDAG {
    pub nodes: Vec<BitSet>,
    pub edges: Vec<BitSet>
}

impl FixedDAG {
    pub fn apply_parent_edges(&mut self, node: &usize, parent_edges: &BitSet) {
        for parent_id in parent_edges {
            self.edges[parent_id].insert(*node);
        }
    }
}