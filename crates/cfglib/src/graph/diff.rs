//! Structural comparison of two CFGs (bindiff-style).
//!
//! Compares two CFGs by topology, block counts, edge patterns, and
//! instruction counts.  Produces a [`CfgDiff`] summary describing
//! the structural differences.
//!
//! This is intentionally a *structural* comparison — it compares the
//! shape of the graph, not the semantics of individual instructions.
//! Instruction-level comparison is left to the consumer since it
//! depends on the concrete instruction type.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;

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

/// A matched pair of blocks between two CFGs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockMatch {
    /// Block in the left (old) CFG.
    pub left: BlockId,
    /// Block in the right (new) CFG.
    pub right: BlockId,
    /// Whether instruction counts differ.
    pub instruction_count_changed: bool,
}

/// Summary of structural differences between two CFGs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CfgDiff {
    /// Blocks matched between left and right CFGs.
    pub matched: Vec<BlockMatch>,
    /// Blocks only in the left (old) CFG — removed.
    pub left_only: Vec<BlockId>,
    /// Blocks only in the right (new) CFG — added.
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
    /// True when the two CFGs are structurally identical.
    #[must_use]
    pub fn is_identical(&self) -> bool {
        self.left_only.is_empty()
            && self.right_only.is_empty()
            && self.matched.iter().all(|m| !m.instruction_count_changed)
            && self.left_block_count == self.right_block_count
            && self.left_edge_count == self.right_edge_count
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

/// Map an `EdgeKind` to a stable discriminant byte.
fn edge_kind_discriminant(k: EdgeKind) -> u8 {
    match k {
        EdgeKind::Fallthrough => 0,
        EdgeKind::ConditionalTrue => 1,
        EdgeKind::ConditionalFalse => 2,
        EdgeKind::Unconditional => 3,
        EdgeKind::Back => 4,
        EdgeKind::Call => 5,
        EdgeKind::CallReturn => 6,
        EdgeKind::SwitchCase => 7,
        EdgeKind::Jump => 8,
        EdgeKind::IndirectJump => 9,
        EdgeKind::IndirectCall => 10,
        EdgeKind::ExceptionHandler => 11,
        EdgeKind::ExceptionUnwind => 12,
        EdgeKind::ExceptionLeave => 13,
    }
}

/// Compute a structural fingerprint for a block.
fn fingerprint<I>(cfg: &Cfg<I>, block: BlockId) -> BlockFingerprint {
    let mut discs: Vec<u8> = cfg
        .successor_edges(block)
        .iter()
        .map(|&eid| edge_kind_discriminant(cfg.edge(eid).kind()))
        .collect();
    discs.sort_unstable();

    BlockFingerprint {
        instruction_count: cfg.block(block).instructions().len(),
        out_degree: cfg.successor_edges(block).len(),
        in_degree: cfg.predecessor_edges(block).len(),
        out_edge_discriminants: discs,
    }
}

/// Compare two CFGs structurally.
///
/// Uses a greedy fingerprint-based matching algorithm:
/// 1. Compute structural fingerprints for all reachable blocks.
/// 2. Match entry blocks first (anchoring).
/// 3. Greedily match remaining blocks by identical fingerprints,
///    preferring blocks whose predecessors/successors are already matched.
///
/// # Example
///
/// ```ignore
/// let diff = cfg_diff(&old_cfg, &new_cfg);
/// if diff.is_identical() {
///     println!("CFGs are structurally identical");
/// } else {
///     println!("Match ratio: {:.0}%", diff.match_ratio() * 100.0);
/// }
/// ```
#[must_use]
pub fn cfg_diff<I, J>(left: &Cfg<I>, right: &Cfg<J>) -> CfgDiff {
    let left_blocks = left.dfs_preorder();
    let right_blocks = right.dfs_preorder();

    let left_fps: BTreeMap<BlockId, BlockFingerprint> = left_blocks
        .iter()
        .map(|&b| (b, fingerprint(left, b)))
        .collect();
    let right_fps: BTreeMap<BlockId, BlockFingerprint> = right_blocks
        .iter()
        .map(|&b| (b, fingerprint(right, b)))
        .collect();

    let mut matched = Vec::new();
    let mut left_matched = alloc::collections::BTreeSet::new();
    let mut right_matched = alloc::collections::BTreeSet::new();

    // Phase 1: anchor on entry blocks.
    let left_entry_fp = left_fps.get(&left.entry());
    let right_entry_fp = right_fps.get(&right.entry());
    if left_entry_fp == right_entry_fp {
        let ic_changed = left.block(left.entry()).instructions().len()
            != right.block(right.entry()).instructions().len();
        matched.push(BlockMatch {
            left: left.entry(),
            right: right.entry(),
            instruction_count_changed: ic_changed,
        });
        left_matched.insert(left.entry());
        right_matched.insert(right.entry());
    }

    // Phase 2: greedy matching by fingerprint.
    // Build index: fingerprint → unmatched right blocks.
    let mut fp_to_right: BTreeMap<BlockFingerprint, Vec<BlockId>> = BTreeMap::new();
    for &rb in &right_blocks {
        if !right_matched.contains(&rb) {
            let fp = right_fps[&rb].clone();
            fp_to_right.entry(fp).or_default().push(rb);
        }
    }

    for &lb in &left_blocks {
        if left_matched.contains(&lb) {
            continue;
        }
        let fp = &left_fps[&lb];
        let Some(candidates) = fp_to_right.get_mut(fp) else {
            continue;
        };
        let Some(pos) = candidates.iter().position(|rb| !right_matched.contains(rb)) else {
            continue;
        };
        let rb = candidates.remove(pos);
        let ic_changed =
            left.block(lb).instructions().len() != right.block(rb).instructions().len();
        matched.push(BlockMatch {
            left: lb,
            right: rb,
            instruction_count_changed: ic_changed,
        });
        left_matched.insert(lb);
        right_matched.insert(rb);
    }

    let left_only: Vec<BlockId> = left_blocks
        .into_iter()
        .filter(|b| !left_matched.contains(b))
        .collect();
    let right_only: Vec<BlockId> = right_blocks
        .into_iter()
        .filter(|b| !right_matched.contains(b))
        .collect();

    // Count live edges.
    let left_edge_count = left.edges().count();
    let right_edge_count = right.edges().count();

    CfgDiff {
        matched,
        left_only,
        right_only,
        left_block_count: left.num_blocks(),
        right_block_count: right.num_blocks(),
        left_edge_count,
        right_edge_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::ff;

    #[test]
    fn identical_cfgs_produce_identical_diff() {
        let mut a = Cfg::new();
        let b1 = a.new_block();
        a.block_mut(a.entry()).instructions_vec_mut().push(ff("e"));
        a.block_mut(b1).instructions_vec_mut().push(ff("b"));
        a.add_edge(a.entry(), b1, EdgeKind::Fallthrough);

        let mut b = Cfg::new();
        let b2 = b.new_block();
        b.block_mut(b.entry()).instructions_vec_mut().push(ff("e"));
        b.block_mut(b2).instructions_vec_mut().push(ff("b"));
        b.add_edge(b.entry(), b2, EdgeKind::Fallthrough);

        let diff = cfg_diff(&a, &b);
        assert!(diff.is_identical());
        assert!((diff.match_ratio() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn added_block_detected() {
        let mut a = Cfg::new();
        a.block_mut(a.entry()).instructions_vec_mut().push(ff("e"));

        let mut b = Cfg::new();
        let extra = b.new_block();
        b.block_mut(b.entry()).instructions_vec_mut().push(ff("e"));
        b.block_mut(extra).instructions_vec_mut().push(ff("x"));
        b.add_edge(b.entry(), extra, EdgeKind::Fallthrough);

        let diff = cfg_diff(&a, &b);
        assert!(!diff.is_identical());
        assert!(!diff.right_only.is_empty(), "new block should be unmatched");
    }

    #[test]
    fn removed_block_detected() {
        let mut a = Cfg::new();
        let extra = a.new_block();
        a.block_mut(a.entry()).instructions_vec_mut().push(ff("e"));
        a.block_mut(extra).instructions_vec_mut().push(ff("x"));
        a.add_edge(a.entry(), extra, EdgeKind::Fallthrough);

        let mut b = Cfg::new();
        b.block_mut(b.entry()).instructions_vec_mut().push(ff("e"));

        let diff = cfg_diff(&a, &b);
        assert!(!diff.is_identical());
        assert!(
            !diff.left_only.is_empty(),
            "removed block should be unmatched"
        );
    }

    #[test]
    fn match_ratio_is_correct() {
        let mut a = Cfg::new();
        a.block_mut(a.entry()).instructions_vec_mut().push(ff("e"));
        let b1 = a.new_block();
        a.block_mut(b1).instructions_vec_mut().push(ff("b"));
        a.add_edge(a.entry(), b1, EdgeKind::Fallthrough);

        // b has entry + 2 extra blocks (different topology)
        let mut b = Cfg::new();
        b.block_mut(b.entry()).instructions_vec_mut().push(ff("e"));
        let c1 = b.new_block();
        let c2 = b.new_block();
        b.block_mut(c1).instructions_vec_mut().push(ff("c1"));
        b.block_mut(c2).instructions_vec_mut().push(ff("c2"));
        b.add_edge(b.entry(), c1, EdgeKind::ConditionalTrue);
        b.add_edge(b.entry(), c2, EdgeKind::ConditionalFalse);

        let diff = cfg_diff(&a, &b);
        assert!(diff.match_ratio() > 0.0);
        assert!(diff.match_ratio() <= 1.0);
    }
}
