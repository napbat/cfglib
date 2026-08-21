//! Tail call detection.
//!
//! Identifies blocks that end with a call immediately followed by a return
//! (heuristic, via [`FlowControl`]), or whose call instructions are already
//! marked as tail calls (explicit, via [`CallInfo`]). These are candidates
//! for tail call optimization.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::flow::{CallInfo, FlowControl, FlowEffect};

/// A detected tail call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailCall {
    /// The block containing the tail call.
    pub block: BlockId,
    /// Index of the call instruction within the block (if identifiable).
    pub inst_idx: Option<usize>,
    /// Whether this was explicitly marked via [`CallInfo::is_tail_call`].
    pub explicit: bool,
}

/// Detect potential tail calls heuristically.
///
/// A block is a candidate when its only successor is an exit block (return)
/// and its last instruction is a call. For instructions that carry explicit
/// tail-call markers, use [`detect_explicit_tail_calls`] instead (or in
/// addition).
#[must_use]
pub fn detect_tail_calls<I: FlowControl>(cfg: &Cfg<I>) -> Vec<TailCall> {
    let mut results = Vec::new();
    let exit_blocks: alloc::collections::BTreeSet<BlockId> = cfg.exit_blocks().collect();

    for block in cfg.blocks() {
        let bid = block.id();
        let succs: Vec<BlockId> = cfg.successors(bid).collect();
        if succs.len() != 1 || !exit_blocks.contains(&succs[0]) {
            continue;
        }
        let Some(last) = block.instructions().last() else {
            continue;
        };
        if matches!(
            last.flow_effect(),
            FlowEffect::Call | FlowEffect::ConditionalCall
        ) {
            let idx = block.instructions().len().saturating_sub(1);
            results.push(TailCall {
                block: bid,
                inst_idx: Some(idx),
                explicit: false,
            });
        }
    }

    results
}

/// Detect call instructions explicitly marked as tail calls.
///
/// Scans every instruction for [`CallInfo::is_tail_call`]. Instructions with
/// no callee are still reported when marked — an indirect tail call has no
/// resolved target but is a tail call nonetheless.
#[must_use]
pub fn detect_explicit_tail_calls<I: CallInfo>(cfg: &Cfg<I>) -> Vec<TailCall> {
    let mut results = Vec::new();
    for block in cfg.blocks() {
        for (idx, instruction) in block.instructions().iter().enumerate() {
            if instruction.is_tail_call() {
                results.push(TailCall {
                    block: block.id(),
                    inst_idx: Some(idx),
                    explicit: true,
                });
            }
        }
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeKind;
    use crate::test_util::{df_call, ff};

    #[test]
    fn explicit_tail_call_detected() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(df_call("call", "foo", true));

        let tails = detect_explicit_tail_calls(&cfg);
        assert_eq!(tails.len(), 1);
        assert!(tails[0].explicit);
        assert_eq!(tails[0].inst_idx, Some(0));
    }

    #[test]
    fn no_tail_calls_in_simple_cfg() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);

        let tails = detect_tail_calls(&cfg);
        assert_eq!(tails.len(), 0);
    }
}
