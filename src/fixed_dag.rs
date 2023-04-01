use bit_set::BitSet;

pub struct FixedDAG<'a> {
    pub nodes: &'a [BitSet],
    pub edges: Vec<BitSet>,
    pub roots: &'a [usize],
}