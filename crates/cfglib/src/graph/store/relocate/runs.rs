//! The replacement runs along one direction: a dense per-node index into the
//! run storage.

extern crate alloc;

use alloc::vec::Vec;
use core::fmt;

use super::super::adjacency::NONE;
use super::super::raw_index;

/// Both vectors start empty, so a direction that never relocates costs
/// nothing. A node is in the index only once it has a run, and its run keeps
/// its place in the storage for as long as the overlay lives.
#[derive(Clone, Default)]
pub(super) struct Runs {
    /// Each node's position in `runs`, or [`NONE`]. Shorter than the node
    /// count until a later node is relocated; a node past the end has no run.
    pub(super) index: Vec<u32>,
    /// The runs themselves, in the order their nodes were first relocated.
    pub(super) runs: Vec<Vec<u32>>,
}

impl Runs {
    /// `node`'s run, when it has one.
    pub(super) fn get(&self, node: usize) -> Option<&[u32]> {
        match *self.index.get(node)? {
            NONE => None,
            slot => Some(&self.runs[slot as usize]),
        }
    }

    /// `node`'s run, for editing, when it has one.
    pub(super) fn get_mut(&mut self, node: usize) -> Option<&mut Vec<u32>> {
        match *self.index.get(node)? {
            NONE => None,
            slot => Some(&mut self.runs[slot as usize]),
        }
    }

    /// Start `node`'s run as `run`.
    ///
    /// `node` has no run yet and is below `node_bound`, the store's node
    /// count. The index grows to cover all of them at once, so nodes added in
    /// bulk before the next relocation do not each regrow it.
    pub(super) fn insert(&mut self, node: usize, node_bound: usize, run: Vec<u32>) {
        if node >= self.index.len() {
            self.index.resize(node_bound, NONE);
        }
        assert!(self.index[node] == NONE, "node {node} already has a run");
        self.index[node] = raw_index(self.runs.len());
        self.runs.push(run);
    }

    /// Whether no node has a run.
    pub(super) fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    /// Every node that has a run with the run, in ascending node order.
    pub(super) fn iter(&self) -> impl Iterator<Item = (u32, &[u32])> {
        self.index
            .iter()
            .enumerate()
            .filter(|&(_, &slot)| slot != NONE)
            .map(|(node, &slot)| (raw_index(node), self.runs[slot as usize].as_slice()))
    }
}

/// Runs are equal when the same nodes have the same runs, wherever the index
/// and the storage happen to put them.
impl PartialEq for Runs {
    fn eq(&self, other: &Self) -> bool {
        self.iter().eq(other.iter())
    }
}

impl Eq for Runs {}

impl fmt::Debug for Runs {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_map().entries(self.iter()).finish()
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;
    use alloc::vec::Vec;

    use super::Runs;

    #[test]
    fn a_direction_that_never_relocates_allocates_nothing() {
        let runs = Runs::default();
        assert_eq!(runs.index.capacity(), 0);
        assert_eq!(runs.runs.capacity(), 0);
        assert!(runs.is_empty());
        assert_eq!(runs.get(7), None);
    }

    #[test]
    fn the_first_run_sizes_the_index_for_every_node() {
        let mut runs = Runs::default();
        runs.insert(5, 9, vec![1, 2]);
        assert_eq!(runs.index.len(), 9);

        runs.insert(8, 9, Vec::new());
        assert_eq!(
            runs.index.len(),
            9,
            "a node already covered regrows nothing"
        );
        assert_eq!(runs.get(5), Some(&[1, 2][..]));
        assert_eq!(runs.get(8), Some(&[][..]), "an emptied run is still a run");
        assert_eq!(runs.get(6), None);
        assert_eq!(runs.get(9), None, "a node past the index has no run");
    }

    #[test]
    fn a_run_is_edited_in_place_and_only_by_the_node_that_has_it() {
        let mut runs = Runs::default();
        runs.insert(2, 4, vec![1]);
        runs.insert(3, 4, vec![7]);

        runs.get_mut(2).expect("node 2 has a run").push(2);

        assert_eq!(runs.get(2), Some(&[1, 2][..]));
        assert_eq!(runs.get(3), Some(&[7][..]));
        assert!(runs.get_mut(0).is_none());
        assert!(runs.get_mut(4).is_none(), "past the index");
    }

    #[test]
    #[should_panic(expected = "node 2 already has a run")]
    fn a_node_cannot_start_a_second_run() {
        let mut runs = Runs::default();
        runs.insert(2, 4, vec![1]);
        runs.insert(2, 4, vec![9]);
    }

    #[test]
    fn nodes_past_the_index_grow_it_to_the_new_node_count() {
        let mut runs = Runs::default();
        runs.insert(1, 3, vec![4]);
        runs.insert(6, 10, vec![5]);

        assert_eq!(runs.index.len(), 10);
        assert_eq!(runs.get(1), Some(&[4][..]));
        assert_eq!(runs.get(6), Some(&[5][..]));
    }

    #[test]
    fn runs_list_by_ascending_node_whatever_the_order_they_began() {
        let mut runs = Runs::default();
        for node in [6_u32, 1, 3] {
            runs.insert(node as usize, 8, vec![node * 10]);
        }

        assert_eq!(
            runs.iter().collect::<Vec<_>>(),
            [(1, &[10][..]), (3, &[30][..]), (6, &[60][..])]
        );
    }

    #[test]
    fn runs_with_the_same_nodes_and_contents_are_equal() {
        let build = |nodes: [u32; 2], bound: usize| {
            let mut runs = Runs::default();
            for node in nodes {
                runs.insert(node as usize, bound, vec![node]);
            }
            runs
        };

        assert_eq!(build([2, 5], 6), build([5, 2], 9));
        assert_ne!(build([2, 5], 6), build([2, 4], 6));
    }
}
