use bit_set::BitSet;
use std::collections::{HashSet, LinkedList, HashMap};

fn export_edge(parent: &BitSet, child: &BitSet) {
    println!("{:?} -> {:?}", parent, child);
}

fn first_pivot(node_contents: &[BitSet], node_filter: &Vec<usize>) -> Option<usize> {
    // todo adjust algorithm -- picking a perfectly centered pivot is unnecessary
    if node_filter.len() <= 1 {
        return None;
    }

    let mut by_size = HashMap::new();
    
    for i in node_filter {
        let size = node_contents[*i].len();
        by_size
            .entry(size)
            .or_insert(vec![])
            .push(*i);
    }
    
    // sort keys
    let mut keys = by_size.keys().map(|k| *k).collect::<Vec<usize>>();
    keys.sort_by(|a, b| a.cmp(b));
    
    if keys.len() <= 2 {
        return None;
    }
    
    return Some(by_size[&keys[keys.len() / 2]][0]);
}

fn base_case(node_contents: &[BitSet], node_filter: &Vec<usize>) {
    // todo implement
    
    // for now, only print nodes
    for i in node_filter {
        print!("{:?}, ", node_contents[*i]);
    }
    
    println!();
    
    
    // let mut as_nodes = node_filter
    //     .iter()
    //     .map(|i| node_contents[*i].clone())
    //     .collect::<Vec<BitSet>>();

    // as_nodes.sort_by(|a, b| a.len().cmp(&b.len()));

    // for i in 0 .. as_nodes.len() - 1 {
    //     let parent = &as_nodes[i];
    //     let child = &as_nodes[i + 1];
    //     export_edge(parent, child);
    // }
}

/// Repeatedly select pivots and scan nodes to find nodes in their lineage.
/// Overlapping lineage is merged into a single chunk.
/// Chunks are recursively broken down into smaller chunks.
/// Edges are exported directly to the files as they are found recursively.
pub fn divide_chunks(node_contents: &[BitSet]) {
    // this stack is in place of recursion
    let mut stack: LinkedList<Vec<usize>> = LinkedList::new();

    // the first chunk is the entire data set
    stack.push_back(
        node_contents
            .iter()
            .enumerate()
            .map(|(i, _)| i)
            .collect::<Vec<usize>>(),
    );

    while let Some(mut remaining) = stack.pop_back() {
        // handle Base Case
        let mut pivot = match first_pivot(node_contents, &remaining) {
            Some(pivot) => pivot,
            None => {
                base_case(node_contents, &remaining);
                continue;
            }
        };
        
        // new pivots are selected from the remaining nodes

        let mut sups: Vec<Vec<usize>> = Vec::new();
        let mut subs: Vec<Vec<usize>> = Vec::new();

        loop {
            let pivot_contents = &node_contents[pivot];

            // ---- handling superset nodes ----

            let mut merge = Vec::new();

            for i in (0 .. sups.len()).rev() {
                for node in &sups[i] {
                    let contents: &BitSet = &node_contents[*node];
                    // if delete {
                    if contents.is_superset(pivot_contents) {
                        // remove current element (iterating backwards)
                        let mut sup_pop = sups.swap_remove(i);
                        merge.append(&mut sup_pop);
                        break;
                    }
                }
            }

            if merge.len() > 0 {
                merge.push(pivot);
                sups.push(merge);
            }

            // ---- handling subset nodes ----
            
            let mut merge = Vec::new();
            
            for i in (0..subs.len()).rev() {
                for node in &subs[i] {
                    let contents: &BitSet = &node_contents[*node];
                    // if delete {
                    if contents.is_subset(pivot_contents) {
                        // remove current element (iterating backwards)
                        let mut sub_pop = subs.swap_remove(i);
                        merge.append(&mut sub_pop);
                        break;
                    }
                }
            }
            
            if merge.len() > 0 {
                merge.push(pivot);
                subs.push(merge);
            }

            // ---- creating new chunks ----

            let mut new_sup = Vec::new();
            let mut new_sub = Vec::new();

            for i in (0 .. remaining.len()).rev() {
                let node = remaining[i];
                let contents: &BitSet = &node_contents[node];

                if pivot == node {
                    new_sub.push(node);
                    new_sup.push(node);
                    remaining.swap_remove(i);
                } else if pivot_contents.is_superset(contents) {
                    new_sub.push(node);
                    remaining.swap_remove(i);
                } else if pivot_contents.is_subset(contents) {
                    new_sup.push(node);
                    remaining.swap_remove(i);
                }
            }

            if new_sup.len() > 0 {
                sups.push(new_sup);
            }
            
            if new_sub.len() > 0 {
                subs.push(new_sub);
            }

            // do while loop
            match remaining.pop() {
                Some(node) => pivot = node,
                None => break,
            }
        }

        // ---- done dividing data, now export edges ----
        stack.extend(sups.into_iter().chain(subs.into_iter()));
    }
}
