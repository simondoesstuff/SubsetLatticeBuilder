use bit_set::BitSet;
use std::collections::{HashSet, LinkedList};

fn export_edge(parent: &BitSet, child: &BitSet) {
    println!("{:?} -> {:?}", parent, child);
}

fn first_pivot(node_contents: &[BitSet], node_filter: &LinkedList<usize>) -> Option<usize> {
    if node_contents.len() <= 1 {
        return None;
    }

    let mut by_size = HashSet::new();

    for i in node_filter {
        let node = &node_contents[*i];
        let size = node.len();

        if by_size.contains(&size) {
            return Some(*i);
        } else {
            by_size.insert(size);
        }
    }

    return None;
}

fn lineage_line(node_contents: &[BitSet], node_filter: &LinkedList<usize>) {
    let mut as_nodes = node_filter
        .iter()
        .map(|i| node_contents[*i].clone())
        .collect::<Vec<BitSet>>();

    as_nodes.sort_by(|a, b| a.len().cmp(&b.len()));

    for i in 0..as_nodes.len() - 1 {
        let parent = &as_nodes[i];
        let child = &as_nodes[i + 1];
        export_edge(parent, child);
    }
}

/// Repeatedly select pivots and scan nodes to find nodes in their lineage.
/// Overlapping lineage is merged into a single chunk.
/// Chunks are recursively broken down into smaller chunks.
/// Edges are exported directly to the files as they are found recursively.
fn divide_chunks(node_contents: &[BitSet]) {
    // this stack is in place of recursion
    let mut stack = LinkedList::new();

    // the first chunk is the entire data set
    stack.push_back(
        node_contents
            .iter()
            .enumerate()
            .map(|(i, _)| i)
            .collect::<LinkedList<usize>>(),
    );

    while let Some(data) = stack.pop_back() {
        // handle Base Case
        let mut pivot = match first_pivot(node_contents, &data) {
            Some(pivot) => pivot,
            None => return lineage_line(node_contents, &data),
        };

        // new pivots are selected from the remaining nodes
        let mut remaining: LinkedList<usize> = LinkedList::new();
        remaining.extend(
            node_contents
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != pivot)
                .map(|(i, _)| i),
        );

        // todo consider making these vectors
        let mut sups: LinkedList<LinkedList<usize>> = LinkedList::new();
        let mut subs: LinkedList<LinkedList<usize>> = LinkedList::new();

        loop {
            let pivot_contents = &node_contents[pivot];

            // ---- handling superset nodes ----

            let mut merge = LinkedList::new();
            merge.push_back(pivot);

            let mut new_sups: LinkedList<LinkedList<usize>> = LinkedList::new();
            
            'outer: for sup in sups.into_iter() {
                for node in sup {
                    let contents: &BitSet = &node_contents[node];
                    if contents.is_subset(pivot_contents) {
                        merge.extend(sup);
                        continue 'outer;
                    }
                }
                
                new_sups.push_back(sup);
            }
            
            new_sups.push_back(merge);
            sups = new_sups;

            // ---- handling subset nodes ----

            let mut merge = LinkedList::new();
            merge.push_back(pivot);

            subs = subs
                .into_iter()
                .filter(|sub: &LinkedList<usize>| {
                    for node in sub {
                        if pivot_contents.is_subset(&node_contents[*node]) {
                            merge.extend(sub.clone());
                            return false;
                        }
                    }

                    return true;
                })
                .collect();

            subs.push_back(merge);

            // ---- creating new chunks ----

            let mut new_sub = LinkedList::new();
            let mut new_sup = LinkedList::new();

            for _ in 0..remaining.len() {
                let node = remaining.pop_front().unwrap();
                let node_contents = &node_contents[node];

                if node_contents.is_subset(pivot_contents) {
                    new_sub.push_back(node);
                } else if pivot_contents.is_subset(node_contents) {
                    new_sup.push_back(node);
                } else {
                    remaining.push_back(node);
                }
            }

            subs.push_back(new_sub);
            sups.push_back(new_sup);

            // do while loop
            if remaining.len() == 0 {
                break;
            } else {
                pivot = remaining.pop_front().unwrap();
            }

            // ---- done dividing data, now export edges ----

            stack.extend(sups.into_iter().chain(subs.into_iter()));
        }
    }
}
