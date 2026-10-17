//! Profile annotations — branch probabilities on edges and the block
//! frequencies they imply, for profile-guided analysis and optimization.
//!
//! An edge weight is a branch probability (see
//! [`Edge::weight`](crate::Edge::weight)): how likely control is to take the
//! edge, relative to the other edges leaving its source block.
//! [`CfgProfile::from_edge_weights`] propagates those probabilities from the
//! entry into an execution frequency for every block.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::cfg_view::CfgView;
use crate::edge::EdgeId;
use crate::graph::traverse::{TraversalDirection, reverse_postorder};

/// The most sweeps over the blocks that frequency propagation makes. A graph
/// without cycles settles in one. A loop settles geometrically, by its
/// probability of staying in the loop per sweep, so only a loop that almost
/// never exits reaches this bound.
const MAX_SWEEPS: usize = 1_000;

/// The relative change of a block frequency under which a sweep has settled.
const SETTLED: f64 = 1e-12;

/// Profile data for a CFG.
#[derive(Debug, Clone)]
pub struct CfgProfile {
    /// Expected executions of each block per execution of the entry: the
    /// entry is `1.0` plus whatever loops back into it, a block that a loop
    /// of probability `p` of staying in runs `1 / (1 - p)` times as often as
    /// the block that enters the loop.
    ///
    /// A block that no weighted path from the entry reaches has no entry.
    pub block_weights: BTreeMap<BlockId, f64>,
    /// The branch probability of each weighted edge.
    pub edge_weights: BTreeMap<EdgeId, f64>,
}

impl CfgProfile {
    /// Build a profile from the edge weights that the view reports, and
    /// propagate them from the entry into block frequencies.
    ///
    /// The entry's frequency is `1.0` and every other block's is the sum,
    /// over its weighted incoming edges, of the source block's frequency
    /// times the edge's probability. Unweighted edges carry no flow. A graph
    /// without cycles has exactly one such solution; a loop is solved by
    /// repeated sweeps in reverse postorder until the frequencies settle.
    /// A loop that can never exit (probabilities that stay in it sum to one
    /// or more) has no finite frequency, and its blocks report the value the
    /// bounded sweeps reach.
    ///
    /// A [`Cfg`] reports the weights set on its edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, CfgProfile, EdgeKind, set_uniform_edge_weights};
    ///
    /// let mut cfg = Cfg::<u32>::new();
    /// let entry = cfg.entry();
    /// let [left, right, merge] = std::array::from_fn(|_| cfg.new_block());
    /// cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
    /// cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
    /// cfg.add_edge(left, merge, EdgeKind::Fallthrough);
    /// cfg.add_edge(right, merge, EdgeKind::Fallthrough);
    ///
    /// set_uniform_edge_weights(&mut cfg);
    /// let profile = CfgProfile::from_edge_weights(&cfg);
    /// assert_eq!(profile.block_weights[&left], 0.5);
    /// // The two halves of the flow rejoin: the merge runs as often as the entry.
    /// assert_eq!(profile.block_weights[&merge], 1.0);
    /// ```
    #[must_use]
    pub fn from_edge_weights<G: CfgView>(cfg: &G) -> Self {
        let edge_weights: BTreeMap<EdgeId, f64> = cfg
            .edge_ids()
            .filter_map(|edge| Some((edge, cfg.edge_weight(edge)?)))
            .collect();
        let block_weights = block_frequencies(cfg, &edge_weights);
        Self {
            block_weights,
            edge_weights,
        }
    }

    /// The hottest block (highest frequency). Of equal blocks the lowest
    /// identity wins.
    #[must_use]
    pub fn hottest_block(&self) -> Option<(BlockId, f64)> {
        self.block_weights
            .iter()
            .rev()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&block, &weight)| (block, weight))
    }

    /// The coldest block (lowest frequency). Of equal blocks the lowest
    /// identity wins.
    #[must_use]
    pub fn coldest_block(&self) -> Option<(BlockId, f64)> {
        self.block_weights
            .iter()
            .min_by(|a, b| a.1.total_cmp(b.1))
            .map(|(&block, &weight)| (block, weight))
    }

    /// Hot blocks: those whose frequency is at least `threshold`.
    #[must_use]
    pub fn hot_blocks(&self, threshold: f64) -> Vec<BlockId> {
        self.block_weights
            .iter()
            .filter(|&(_, w)| *w >= threshold)
            .map(|(&block, _)| block)
            .collect()
    }
}

/// The frequency of every block that a weighted path from the entry reaches.
fn block_frequencies<G: CfgView>(
    cfg: &G,
    edge_weights: &BTreeMap<EdgeId, f64>,
) -> BTreeMap<BlockId, f64> {
    let root = cfg.root();
    let order = reverse_postorder(cfg, root, TraversalDirection::Outgoing);
    let mut position_of = vec![None; cfg.node_bound()];
    for (position, block) in order.iter().enumerate() {
        position_of[block.index()] = Some(position);
    }

    // The weighted flow into each block, as (source position, probability).
    let mut has_cycle = false;
    let inflow: Vec<Vec<(usize, f64)>> = order
        .iter()
        .enumerate()
        .map(|(position, &block)| {
            let incoming: Vec<_> = cfg
                .incoming(block)
                .filter_map(|edge| {
                    let source = position_of[cfg.edge(edge).source().index()]?;
                    Some((source, *edge_weights.get(&edge)?))
                })
                .collect();
            // Reverse postorder puts every source before its target except
            // along a retreating edge, which closes a cycle.
            has_cycle |= incoming.iter().any(|&(source, _)| source >= position);
            incoming
        })
        .collect();

    let mut frequency = vec![0.0_f64; order.len()];
    for _ in 0..MAX_SWEEPS {
        let mut change = 0.0_f64;
        for (position, &block) in order.iter().enumerate() {
            let entry_flow = if block == root { 1.0 } else { 0.0 };
            let value = inflow[position]
                .iter()
                .fold(entry_flow, |sum, &(source, probability)| {
                    sum + frequency[source] * probability
                });
            change = change.max((value - frequency[position]).abs() / value.abs().max(1.0));
            frequency[position] = value;
        }
        if !has_cycle || change <= SETTLED {
            break;
        }
    }

    order
        .into_iter()
        .zip(frequency)
        .filter(|&(_, frequency)| frequency > 0.0)
        .collect()
}

/// Set uniform edge weights on `cfg`: every edge leaving a block gets the
/// probability `1 / n` for the block's `n` outgoing edges.
pub fn set_uniform_edge_weights<I, E>(cfg: &mut Cfg<I, E>) {
    let block_ids: Vec<BlockId> = cfg.block_ids().collect();
    let mut succs = Vec::new();
    for bid in block_ids {
        succs.clear();
        succs.extend(cfg.outgoing(bid));
        if succs.is_empty() {
            continue;
        }
        let w = 1.0 / crate::usize_to_f64(succs.len());
        for &eid in &succs {
            cfg.edge_mut(eid).set_weight(Some(w));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn uniform_weights() {
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("br"));
        cfg.block_mut(a).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        set_uniform_edge_weights(&mut cfg);
        let profile = CfgProfile::from_edge_weights(&cfg);
        assert_eq!(profile.edge_weights.len(), 2);
        for &w in profile.edge_weights.values() {
            assert!((w - 0.5).abs() < 1e-9);
        }
    }

    #[test]
    fn uniform_weights_follow_each_blocks_own_out_degree() {
        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.entry();
        let [a, b, c, d] = core::array::from_fn(|_| cfg.new_block());
        cfg.add_edge(entry, a, EdgeKind::ConditionalTrue);
        cfg.add_edge(entry, b, EdgeKind::ConditionalFalse);
        cfg.add_edge(entry, c, EdgeKind::IndirectJump);
        cfg.add_edge(a, d, EdgeKind::Fallthrough);
        cfg.add_edge(b, d, EdgeKind::Fallthrough);
        set_uniform_edge_weights(&mut cfg);

        let mut weights: Vec<_> = cfg
            .edges()
            .map(|edge| (edge.source(), edge.target(), edge.weight()))
            .collect();
        weights.sort_unstable_by_key(|&(source, target, _)| (source, target));
        assert_eq!(weights.len(), 5);
        for (source, _, weight) in weights {
            let expected = if source == entry { 1.0 / 3.0 } else { 1.0 };
            let weight = weight.expect("every edge of a block with successors is weighted");
            assert!((weight - expected).abs() < 1e-12, "{source}: {weight}");
        }
    }

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() <= 1e-9 * expected.max(1.0),
            "{actual} is not {expected}"
        );
    }

    /// A CFG of `count` blocks (the first is the entry) with a weighted
    /// `Fallthrough` edge for each `(source, target, probability)`.
    fn weighted(count: usize, edges: &[(usize, usize, f64)]) -> (Cfg<u32>, Vec<BlockId>) {
        let mut cfg = Cfg::<u32>::new();
        let mut blocks = vec![cfg.entry()];
        blocks.extend((1..count).map(|_| cfg.new_block()));
        for &(source, target, probability) in edges {
            cfg.add_weighted_edge(
                blocks[source],
                blocks[target],
                EdgeKind::Fallthrough,
                probability,
            );
        }
        (cfg, blocks)
    }

    #[test]
    fn a_merge_runs_as_often_as_the_branch_that_feeds_it() {
        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.entry();
        let [left, right, merge] = core::array::from_fn(|_| cfg.new_block());
        cfg.add_edge(entry, left, EdgeKind::ConditionalTrue);
        cfg.add_edge(entry, right, EdgeKind::ConditionalFalse);
        cfg.add_edge(left, merge, EdgeKind::Fallthrough);
        cfg.add_edge(right, merge, EdgeKind::Fallthrough);
        set_uniform_edge_weights(&mut cfg);

        let profile = CfgProfile::from_edge_weights(&cfg);
        let frequencies: Vec<_> = profile.block_weights.values().copied().collect();
        assert_eq!(frequencies, [1.0, 0.5, 0.5, 1.0]);
        assert_eq!(
            profile.hottest_block(),
            Some((entry, 1.0)),
            "entry and merge tie, and the lowest identity wins"
        );
        assert_eq!(profile.coldest_block(), Some((left, 0.5)));
    }

    #[test]
    fn a_loop_runs_its_body_the_expected_trip_count_times() {
        // entry -> header; header stays in the loop with probability 0.9.
        let (cfg, blocks) = weighted(4, &[(0, 1, 1.0), (1, 2, 0.9), (1, 3, 0.1), (2, 1, 1.0)]);
        let [entry, header, body, exit] = blocks[..] else {
            unreachable!("four blocks were created")
        };

        let profile = CfgProfile::from_edge_weights(&cfg);
        assert_close(profile.block_weights[&entry], 1.0);
        assert_close(profile.block_weights[&header], 10.0);
        assert_close(profile.block_weights[&body], 9.0);
        assert_close(profile.block_weights[&exit], 1.0);
        assert_eq!(
            profile.hottest_block().map(|(block, _)| block),
            Some(header)
        );
    }

    #[test]
    fn a_self_loop_on_the_entry_adds_to_the_entry_frequency() {
        let (cfg, blocks) = weighted(2, &[(0, 0, 0.5), (0, 1, 0.5)]);
        let profile = CfgProfile::from_edge_weights(&cfg);
        assert_close(profile.block_weights[&blocks[0]], 2.0);
        assert_close(profile.block_weights[&blocks[1]], 1.0);
    }

    #[test]
    fn flow_from_unreachable_blocks_and_unweighted_edges_is_ignored() {
        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.entry();
        let [reached, dead, unprofiled] = core::array::from_fn(|_| cfg.new_block());
        cfg.add_weighted_edge(entry, reached, EdgeKind::Fallthrough, 1.0);
        cfg.add_weighted_edge(dead, reached, EdgeKind::Fallthrough, 1.0);
        cfg.add_edge(reached, unprofiled, EdgeKind::Fallthrough);

        let profile = CfgProfile::from_edge_weights(&cfg);
        assert_eq!(
            profile.block_weights.into_iter().collect::<Vec<_>>(),
            [(entry, 1.0), (reached, 1.0)]
        );
    }

    #[test]
    fn profiles_come_from_graphs_with_any_edge_payload() {
        use crate::exception::{ExceptionDisposition, ExceptionFlow, ExceptionPhase};

        let mut cfg = Cfg::<(), ExceptionFlow<u8>>::with_edge_payload();
        let entry = cfg.entry();
        let [handler, normal] = core::array::from_fn(|_| cfg.new_block());
        cfg.add_edge_with_payload(
            entry,
            handler,
            EdgeKind::ExceptionHandler,
            ExceptionFlow::exceptional(
                ExceptionPhase::Unwind,
                Some(ExceptionDisposition::ExecuteHandler),
                7,
            ),
        );
        cfg.add_edge_with_payload(
            entry,
            normal,
            EdgeKind::Fallthrough,
            ExceptionFlow::normal(8),
        );
        set_uniform_edge_weights(&mut cfg);

        let profile = CfgProfile::from_edge_weights(&cfg);
        assert_eq!(profile.block_weights[&handler], 0.5);
        assert_eq!(profile.block_weights[&normal], 0.5);
        assert!(
            cfg.edges().all(|edge| edge.weight() == Some(0.5)),
            "every edge of the branching block is weighted"
        );
    }

    #[test]
    fn hottest_and_coldest_blocks_break_ties_by_lowest_identity() {
        let mut profile = CfgProfile {
            block_weights: BTreeMap::new(),
            edge_weights: BTreeMap::new(),
        };
        assert_eq!(profile.hottest_block(), None);
        for (raw, weight) in [(0, 10.0), (1, 100.0), (2, 100.0), (3, 10.0)] {
            profile.block_weights.insert(BlockId::from_raw(raw), weight);
        }
        assert_eq!(profile.hottest_block(), Some((BlockId::from_raw(1), 100.0)));
        assert_eq!(profile.coldest_block(), Some((BlockId::from_raw(0), 10.0)));
    }

    #[test]
    fn hot_blocks_keep_every_block_at_or_over_the_threshold() {
        let mut profile = CfgProfile {
            block_weights: BTreeMap::new(),
            edge_weights: BTreeMap::new(),
        };
        for (raw, weight) in [(0, 1.0), (1, 50.0), (2, 100.0)] {
            profile.block_weights.insert(BlockId::from_raw(raw), weight);
        }
        assert_eq!(
            profile.hot_blocks(50.0),
            [BlockId::from_raw(1), BlockId::from_raw(2)]
        );
    }
}
