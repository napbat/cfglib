//! Structural comparison of two CFGs (bindiff-style).
//!
//! Matches the blocks of two CFGs by how they are wired and compares each
//! matched pair: block counts, edge patterns, and instruction counts.
//! Produces a [`CfgDiff`] summary describing the structural differences.
//!
//! This is intentionally a *structural* comparison — it compares the
//! shape of the graph, not the semantics of individual instructions.
//! Instruction-level comparison is left to the consumer since it
//! depends on the concrete instruction type.

extern crate alloc;
use alloc::collections::{BTreeMap, VecDeque};
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;

/// Per-block structural fingerprint used for matching.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockFingerprint {
    /// Number of instructions in the block.
    pub instruction_count: usize,
    /// Number of successor edges.
    pub out_degree: usize,
    /// Number of predecessor edges.
    pub in_degree: usize,
    /// Sorted edge kind discriminants of outgoing edges.
    pub out_edge_discriminants: Vec<u8>,
}

impl BlockFingerprint {
    /// What two blocks must share to be matched: how they are wired, not how
    /// long they are. A block that only gained or lost instructions is the
    /// same block, changed.
    fn shape(&self) -> (usize, usize, &[u8]) {
        (
            self.out_degree,
            self.in_degree,
            &self.out_edge_discriminants,
        )
    }
}

/// A matched pair of blocks between two CFGs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMatch {
    /// Block in the left (old) CFG.
    pub left: BlockId,
    /// Block in the right (new) CFG.
    pub right: BlockId,
    /// Whether instruction counts differ.
    pub instruction_count_changed: bool,
    /// Whether the outgoing edges differ: the left block's edges must reach
    /// the right counterparts of the same blocks, kind for kind, as the right
    /// block's edges do. An edge into a block that has no counterpart differs.
    pub edges_changed: bool,
}

/// Summary of structural differences between two CFGs.
///
/// Every live block takes part, reachable from the entry or not: the counts
/// below, the matches, and the unmatched lists all describe the same set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgDiff {
    /// Blocks matched between left and right CFGs, in ascending order of the
    /// left block.
    pub matched: Vec<BlockMatch>,
    /// Blocks only in the left (old) CFG — removed, in ascending order.
    pub left_only: Vec<BlockId>,
    /// Blocks only in the right (new) CFG — added, in ascending order.
    pub right_only: Vec<BlockId>,
    /// Number of blocks in left CFG.
    pub left_block_count: usize,
    /// Number of blocks in right CFG.
    pub right_block_count: usize,
    /// Number of edges in left CFG.
    pub left_edge_count: usize,
    /// Number of edges in right CFG.
    pub right_edge_count: usize,
}

impl CfgDiff {
    /// True when the two CFGs are structurally identical: every block has a
    /// counterpart with the same instruction count, and every block's edges
    /// reach the counterparts of the same blocks, kind for kind.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.left_only.is_empty()
            && self.right_only.is_empty()
            && self
                .matched
                .iter()
                .all(|pair| !pair.instruction_count_changed && !pair.edges_changed)
    }

    /// Fraction of blocks successfully matched (0.0–1.0).
    #[must_use]
    pub fn match_ratio(&self) -> f64 {
        let total = self.left_block_count.max(self.right_block_count);
        if total == 0 {
            return 1.0;
        }
        crate::usize_to_f64(self.matched.len()) / crate::usize_to_f64(total)
    }
}

/// What the matcher knows about one block.
struct BlockFacts {
    fingerprint: BlockFingerprint,
    /// The outgoing `(kind discriminant, target)` pairs, grouped by kind with
    /// each group in adjacency order.
    out_edges: Vec<(u8, BlockId)>,
}

/// One CFG, as the matcher sees it.
struct Side {
    entry: BlockId,
    blocks: BTreeMap<BlockId, BlockFacts>,
}

impl Side {
    fn new<I>(cfg: &Cfg<I>) -> Self {
        let blocks = cfg
            .block_ids()
            .map(|block| {
                let mut out_edges: Vec<(u8, BlockId)> = cfg
                    .outgoing(block)
                    .map(|edge| {
                        let edge = cfg.edge(edge);
                        // The discriminant is declaration order, which is
                        // what `EdgeKind::ALL` lists.
                        (edge.kind() as u8, edge.target())
                    })
                    .collect();
                // A stable sort keeps adjacency order within one kind.
                out_edges.sort_by_key(|&(kind, _)| kind);
                let fingerprint = BlockFingerprint {
                    instruction_count: cfg.block(block).instructions().len(),
                    out_degree: out_edges.len(),
                    in_degree: cfg.incoming(block).count(),
                    out_edge_discriminants: out_edges.iter().map(|&(kind, _)| kind).collect(),
                };
                (
                    block,
                    BlockFacts {
                        fingerprint,
                        out_edges,
                    },
                )
            })
            .collect();
        Self {
            entry: cfg.entry(),
            blocks,
        }
    }

    fn facts(&self, block: BlockId) -> &BlockFacts {
        &self.blocks[&block]
    }
}

/// A partial one-to-one pairing of the blocks of two CFGs.
#[derive(Default)]
struct Matching {
    right_of: BTreeMap<BlockId, BlockId>,
    left_of: BTreeMap<BlockId, BlockId>,
}

impl Matching {
    fn pair(&mut self, left: BlockId, right: BlockId) {
        self.right_of.insert(left, right);
        self.left_of.insert(right, left);
    }

    fn both_free(&self, left: BlockId, right: BlockId) -> bool {
        !self.right_of.contains_key(&left) && !self.left_of.contains_key(&right)
    }
}

/// Whether `left`'s edges reach the counterparts of the blocks `right`'s
/// edges reach, kind for kind.
fn edges_preserved(left: &BlockFacts, right: &BlockFacts, matching: &Matching) -> bool {
    let mut mapped = Vec::with_capacity(left.out_edges.len());
    for &(kind, target) in &left.out_edges {
        let Some(&counterpart) = matching.right_of.get(&target) else {
            return false;
        };
        mapped.push((kind, counterpart));
    }
    mapped.sort_unstable();
    let mut expected = right.out_edges.clone();
    expected.sort_unstable();
    mapped == expected
}

/// Extends `matching` along the edges of the pairs in `pending`.
///
/// The edges two matched blocks have of one kind are paired in adjacency
/// order, and the targets of a pair join the matching when they are free and
/// wired alike. Each new pair is queued, so the walk follows the matching out
/// from where it started.
fn follow_edges(
    left: &Side,
    right: &Side,
    matching: &mut Matching,
    pending: &mut VecDeque<(BlockId, BlockId)>,
) {
    while let Some((left_block, right_block)) = pending.pop_front() {
        let left_edges = &left.facts(left_block).out_edges;
        let right_edges = &right.facts(right_block).out_edges;
        let mut right_groups = right_edges.chunk_by(|a, b| a.0 == b.0).peekable();
        for left_group in left_edges.chunk_by(|a, b| a.0 == b.0) {
            let kind = left_group[0].0;
            while right_groups.next_if(|group| group[0].0 < kind).is_some() {}
            let Some(right_group) = right_groups.next_if(|group| group[0].0 == kind) else {
                continue;
            };
            for (&(_, left_target), &(_, right_target)) in left_group.iter().zip(right_group) {
                if matching.both_free(left_target, right_target)
                    && left.facts(left_target).fingerprint.shape()
                        == right.facts(right_target).fingerprint.shape()
                {
                    matching.pair(left_target, right_target);
                    pending.push_back((left_target, right_target));
                }
            }
        }
    }
}

impl CfgDiff {
    /// Compare two CFGs structurally.
    ///
    /// Every live block takes part, whether the entry reaches it or not. The
    /// matching is built in three steps:
    /// 1. The entry blocks are paired, whatever their wiring.
    /// 2. The matching is extended along edges: the targets of the edges two
    ///    matched blocks have of one kind are paired when they have the same
    ///    number of predecessors and successors and the same edge kinds.
    /// 3. Each left block still unmatched takes the first free right block
    ///    with the same wiring (preferring one with the same instruction
    ///    count), and the matching is extended along edges from that pair.
    ///
    /// A matched pair whose instruction counts or edges differ is reported as
    /// changed rather than as an unmatched pair of blocks, and
    /// [`is_identical`](Self::is_identical) is false for it. Edges of one kind
    /// are paired in adjacency order, so a block with several edges of a kind
    /// whose targets are only distinguished by their position can be reported
    /// as changed when the other CFG lists them in another order.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, CfgDiff, EdgeKind};
    ///
    /// // `entry -> body -> exit`, numbered in two different orders.
    /// let mut old = Cfg::<u32>::new();
    /// let (body, exit) = (old.new_block(), old.new_block());
    /// old.add_edge(old.entry(), body, EdgeKind::Fallthrough);
    /// old.add_edge(body, exit, EdgeKind::Fallthrough);
    ///
    /// let mut new = Cfg::<u32>::new();
    /// let (exit, body) = (new.new_block(), new.new_block());
    /// new.add_edge(new.entry(), body, EdgeKind::Fallthrough);
    /// new.add_edge(body, exit, EdgeKind::Fallthrough);
    ///
    /// let diff = CfgDiff::compute(&old, &new);
    /// assert!(diff.is_identical());
    /// assert!((diff.match_ratio() - 1.0).abs() < f64::EPSILON);
    ///
    /// // A block that gains a successor is no longer the same shape.
    /// let extra = new.new_block();
    /// new.add_edge(exit, extra, EdgeKind::Fallthrough);
    /// let diff = CfgDiff::compute(&old, &new);
    /// assert!(!diff.is_identical());
    /// assert_eq!(diff.right_only.len(), 1);
    /// ```
    #[must_use]
    pub fn compute<I, I2>(left: &Cfg<I>, right: &Cfg<I2>) -> Self {
        let (left_side, right_side) = (Side::new(left), Side::new(right));
        let mut matching = Matching::default();
        let mut pending = VecDeque::new();

        matching.pair(left_side.entry, right_side.entry);
        pending.push_back((left_side.entry, right_side.entry));
        follow_edges(&left_side, &right_side, &mut matching, &mut pending);

        let mut free_by_shape: BTreeMap<(usize, usize, &[u8]), Vec<BlockId>> = BTreeMap::new();
        for (&block, facts) in &right_side.blocks {
            if !matching.left_of.contains_key(&block) {
                free_by_shape
                    .entry(facts.fingerprint.shape())
                    .or_default()
                    .push(block);
            }
        }
        for (&block, facts) in &left_side.blocks {
            if matching.right_of.contains_key(&block) {
                continue;
            }
            let Some(candidates) = free_by_shape.get_mut(&facts.fingerprint.shape()) else {
                continue;
            };
            // The walks above may have claimed some since the index was built.
            candidates.retain(|candidate| !matching.left_of.contains_key(candidate));
            let same_length = candidates.iter().position(|&candidate| {
                right_side.facts(candidate).fingerprint.instruction_count
                    == facts.fingerprint.instruction_count
            });
            let Some(position) = same_length.or_else(|| (!candidates.is_empty()).then_some(0))
            else {
                continue;
            };
            let counterpart = candidates.remove(position);
            matching.pair(block, counterpart);
            pending.push_back((block, counterpart));
            follow_edges(&left_side, &right_side, &mut matching, &mut pending);
        }

        let matched = left_side
            .blocks
            .iter()
            .filter_map(|(&block, facts)| {
                let counterpart = *matching.right_of.get(&block)?;
                let other = right_side.facts(counterpart);
                Some(BlockMatch {
                    left: block,
                    right: counterpart,
                    instruction_count_changed: facts.fingerprint.instruction_count
                        != other.fingerprint.instruction_count,
                    edges_changed: !edges_preserved(facts, other, &matching),
                })
            })
            .collect();

        CfgDiff {
            matched,
            left_only: left_side
                .blocks
                .keys()
                .filter(|block| !matching.right_of.contains_key(block))
                .copied()
                .collect(),
            right_only: right_side
                .blocks
                .keys()
                .filter(|block| !matching.left_of.contains_key(block))
                .copied()
                .collect(),
            left_block_count: left.block_count(),
            right_block_count: right.block_count(),
            left_edge_count: left.edge_count(),
            right_edge_count: right.edge_count(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeKind;
    use crate::test_util::ff;

    #[test]
    fn identical_cfgs_produce_identical_diff() {
        let mut a = Cfg::new();
        let b1 = a.new_block();
        a.block_mut(a.entry()).instructions_mut().push(ff("e"));
        a.block_mut(b1).instructions_mut().push(ff("b"));
        a.add_edge(a.entry(), b1, EdgeKind::Fallthrough);

        let mut b = Cfg::new();
        let b2 = b.new_block();
        b.block_mut(b.entry()).instructions_mut().push(ff("e"));
        b.block_mut(b2).instructions_mut().push(ff("b"));
        b.add_edge(b.entry(), b2, EdgeKind::Fallthrough);

        let diff = CfgDiff::compute(&a, &b);
        assert!(diff.is_identical());
        assert!((diff.match_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn added_block_detected() {
        let mut a = Cfg::new();
        a.block_mut(a.entry()).instructions_mut().push(ff("e"));

        let mut b = Cfg::new();
        let extra = b.new_block();
        b.block_mut(b.entry()).instructions_mut().push(ff("e"));
        b.block_mut(extra).instructions_mut().push(ff("x"));
        b.add_edge(b.entry(), extra, EdgeKind::Fallthrough);

        let diff = CfgDiff::compute(&a, &b);
        assert!(!diff.is_identical());
        assert!(!diff.right_only.is_empty(), "new block should be unmatched");
    }

    #[test]
    fn removed_block_detected() {
        let mut a = Cfg::new();
        let extra = a.new_block();
        a.block_mut(a.entry()).instructions_mut().push(ff("e"));
        a.block_mut(extra).instructions_mut().push(ff("x"));
        a.add_edge(a.entry(), extra, EdgeKind::Fallthrough);

        let mut b = Cfg::new();
        b.block_mut(b.entry()).instructions_mut().push(ff("e"));

        let diff = CfgDiff::compute(&a, &b);
        assert!(!diff.is_identical());
        assert!(
            !diff.left_only.is_empty(),
            "removed block should be unmatched"
        );
    }

    #[test]
    fn match_ratio_is_correct() {
        let mut a = Cfg::new();
        a.block_mut(a.entry()).instructions_mut().push(ff("e"));
        let b1 = a.new_block();
        a.block_mut(b1).instructions_mut().push(ff("b"));
        a.add_edge(a.entry(), b1, EdgeKind::Fallthrough);

        // b has entry + 2 extra blocks (different topology)
        let mut b = Cfg::new();
        b.block_mut(b.entry()).instructions_mut().push(ff("e"));
        let c1 = b.new_block();
        let c2 = b.new_block();
        b.block_mut(c1).instructions_mut().push(ff("c1"));
        b.block_mut(c2).instructions_mut().push(ff("c2"));
        b.add_edge(b.entry(), c1, EdgeKind::ConditionalTrue);
        b.add_edge(b.entry(), c2, EdgeKind::ConditionalFalse);

        let diff = CfgDiff::compute(&a, &b);
        assert!(diff.match_ratio() > 0.0);
        assert!(diff.match_ratio() <= 1.0);
    }

    #[test]
    fn a_block_lists_its_edges_by_kind_in_adjacency_order() {
        let mut cfg = Cfg::<u32>::new();
        let (first, second, third) = (cfg.new_block(), cfg.new_block(), cfg.new_block());
        cfg.add_edge(cfg.entry(), first, EdgeKind::Back);
        cfg.add_edge(cfg.entry(), second, EdgeKind::Fallthrough);
        cfg.add_edge(cfg.entry(), third, EdgeKind::Back);

        let side = Side::new(&cfg);
        let facts = side.facts(cfg.entry());
        assert_eq!(facts.fingerprint.out_edge_discriminants, [0, 4, 4]);
        assert_eq!(facts.out_edges, [(0, second), (4, first), (4, third)]);
    }
}
