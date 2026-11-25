//! Block orders for linearization: the hot/cold partition of a base order.

use cfglib::{ColdClass, ColdnessClasses, ColdnessGraph, SplitOrder, hot_cold_order};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Diamonds in the chain. Every fourth false arm aborts, seeded `Rare`.
const DIAMONDS: usize = 1_024;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let chain = DiamondChain::new(DIAMONDS);
    let classes = ColdnessClasses::compute(&chain);
    let base: Vec<usize> = (0..chain.node_bound()).collect();
    let rare = DIAMONDS.div_ceil(4);
    benchmark_case!(
        suite,
        "api_hot_cold_order",
        covers[hot_cold_order],
        || hot_cold_order(&base, &classes, 0),
        |split: &SplitOrder<usize>| {
            assert_eq!(split.order.len(), base.len());
            assert_eq!(split.order[0], 0);
            assert_eq!(split.cold_start, base.len() - rare);
            assert_eq!(split.never_start, base.len());
        }
    );
}

/// A chain of if/else diamonds over numbered nodes: diamond `i` branches
/// from node `3i` to the arms `3i + 1` and `3i + 2`, which join at
/// `3i + 3`. The false arm of every fourth diamond aborts instead.
struct DiamondChain {
    records: Vec<Vec<usize>>,
    incoming: Vec<Vec<usize>>,
}

impl DiamondChain {
    fn new(diamonds: usize) -> Self {
        let count = 3 * diamonds + 1;
        let mut records = vec![Vec::new(); count];
        for diamond in 0..diamonds {
            let (branch, join) = (3 * diamond, 3 * diamond + 3);
            records[branch] = vec![branch + 1, branch + 2];
            records[branch + 1] = vec![join];
            if !Self::aborts(branch + 2) {
                records[branch + 2] = vec![join];
            }
        }
        let mut incoming = vec![Vec::new(); count];
        for (source, targets) in records.iter().enumerate() {
            for &target in targets {
                incoming[target].push(source);
            }
        }
        Self { records, incoming }
    }

    /// Whether `node` is the false arm of every fourth diamond.
    fn aborts(node: usize) -> bool {
        node % 12 == 2
    }
}

impl ColdnessGraph for DiamondChain {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.records.len()
    }

    fn entry(&self) -> usize {
        0
    }

    fn reachable(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.records.len()
    }

    fn records(&self, node: usize) -> impl Iterator<Item = (usize, ColdClass)> + '_ {
        self.records[node]
            .iter()
            .map(|&target| (target, ColdClass::Normal))
    }

    fn incoming(&self, node: usize) -> impl Iterator<Item = (usize, ColdClass)> + '_ {
        self.incoming[node]
            .iter()
            .map(|&source| (source, ColdClass::Normal))
    }

    fn seed(&self, node: usize) -> ColdClass {
        if Self::aborts(node) {
            ColdClass::Rare
        } else {
            ColdClass::Normal
        }
    }

    fn ends(&self, _node: usize) -> bool {
        false
    }
}
