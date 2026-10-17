//! Tail call detection.
//!
//! Identifies blocks that end with a call immediately followed by a return
//! (heuristic, via [`FlowControl`]), or whose call instructions are already
//! marked as tail calls (explicit, via [`CallInfo`]). These are candidates
//! for tail call optimization.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::flow::{CallInfo, FlowControl, FlowEffect};

/// A detected tail call site.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailCall {
    /// The block containing the tail call.
    pub block: BlockId,
    /// Index of the call instruction within the block.
    pub inst_idx: usize,
    /// Whether this was explicitly marked via [`CallInfo::is_tail_call`].
    pub explicit: bool,
}

/// Detect potential tail calls heuristically.
///
/// A block is a candidate when its last instruction is a call
/// ([`Call`](FlowEffect::Call), [`IndirectCall`](FlowEffect::IndirectCall),
/// or [`ConditionalCall`](FlowEffect::ConditionalCall)) and its only
/// successor is an exit block (one with no successors, where the function
/// returns) reached by an ordinary edge. A call that can also unwind into a
/// handler has a second successor and is not a candidate, nor is one whose
/// only successor is an exceptional edge. For instructions that carry
/// explicit tail-call markers, use [`detect_explicit_tail_calls`] instead
/// (or in addition).
#[must_use]
pub fn detect_tail_calls<G>(cfg: &G) -> Vec<TailCall>
where
    G: CfgView,
    G::Inst: FlowControl,
{
    let mut results = Vec::new();
    for bid in cfg.node_ids() {
        let mut outgoing = cfg.outgoing(bid);
        let (Some(edge), None) = (outgoing.next(), outgoing.next()) else {
            continue;
        };
        let instructions = cfg.instructions(bid);
        if !cfg.edge_kind(edge).is_exceptional()
            && cfg.outgoing(cfg.edge(edge).target()).next().is_none()
            && let Some(last) = instructions.last()
            && matches!(
                last.flow_effect(),
                FlowEffect::Call | FlowEffect::IndirectCall | FlowEffect::ConditionalCall
            )
        {
            results.push(TailCall {
                block: bid,
                inst_idx: instructions.len() - 1,
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
pub fn detect_explicit_tail_calls<G>(cfg: &G) -> Vec<TailCall>
where
    G: CfgView,
    G::Inst: CallInfo,
{
    let mut results = Vec::new();
    for block_id in cfg.node_ids() {
        for (idx, instruction) in cfg.instructions(block_id).iter().enumerate() {
            if instruction.is_tail_call() {
                results.push(TailCall {
                    block: block_id,
                    inst_idx: idx,
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
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::flow::FlowEffect;
    use crate::test_util::{DfInst, df_call, df_ff as ff, df_with_effect};

    #[test]
    fn explicit_tail_call_detected() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(df_call("call", "foo", true));

        let tails = detect_explicit_tail_calls(&cfg);
        assert_eq!(
            tails,
            [TailCall {
                block: cfg.entry(),
                inst_idx: 0,
                explicit: true,
            }]
        );
    }

    #[test]
    fn explicit_tail_calls_name_every_marked_instruction_with_or_without_a_callee() {
        let mut cfg = Cfg::new();
        let other = cfg.new_block();
        let indirect = DfInst {
            callee: None,
            ..df_call("call", "unused", true)
        };
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            df_call("call", "plain", false),
            df_call("call", "tail", true),
            ff("after"),
        ]);
        cfg.block_mut(other).instructions_mut().push(indirect);

        let tails = detect_explicit_tail_calls(&cfg);
        assert_eq!(
            tails
                .iter()
                .map(|tail| (tail.block, tail.inst_idx, tail.explicit))
                .collect::<Vec<_>>(),
            [(cfg.entry(), 1, true), (other, 0, true)]
        );
    }

    #[test]
    fn a_call_that_ends_its_block_and_flows_into_the_exit_is_a_tail_call() {
        let mut cfg = Cfg::new();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .extend([ff("setup"), df_call("call", "callee", false)]);
        cfg.block_mut(exit).instructions_mut().push(ff("ret"));
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

        assert_eq!(
            detect_tail_calls(&cfg),
            [TailCall {
                block: cfg.entry(),
                inst_idx: 1,
                explicit: false,
            }]
        );
    }

    #[test]
    fn no_tail_calls_in_simple_cfg() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);

        let tails = detect_tail_calls(&cfg);
        assert_eq!(tails, [] as [crate::analysis::tail_call::TailCall; 0]);
    }

    #[test]
    fn a_call_that_work_follows_is_not_a_tail_call() {
        // The call is not the block's last instruction.
        let mut not_last = Cfg::new();
        let exit = not_last.new_block();
        not_last
            .block_mut(not_last.entry())
            .instructions_mut()
            .extend([df_call("call", "callee", false), ff("cleanup")]);
        not_last.add_edge(not_last.entry(), exit, EdgeKind::Fallthrough);
        assert!(detect_tail_calls(&not_last).is_empty());

        // The call's successor is not the exit: the function goes on.
        let mut continues = Cfg::new();
        let next = continues.new_block();
        let exit = continues.new_block();
        continues
            .block_mut(continues.entry())
            .instructions_mut()
            .push(df_call("call", "callee", false));
        continues
            .block_mut(next)
            .instructions_mut()
            .push(ff("more"));
        continues.add_edge(continues.entry(), next, EdgeKind::Fallthrough);
        continues.add_edge(next, exit, EdgeKind::Fallthrough);
        assert!(detect_tail_calls(&continues).is_empty());
    }

    #[test]
    fn a_call_guarded_by_a_handler_is_not_a_tail_call() {
        // The call returns normally to the exit but can also unwind into a
        // handler, which has to run: two successors.
        let mut cfg = Cfg::new();
        let exit = cfg.new_block();
        let handler = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(df_call("call", "callee", false));
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);
        cfg.add_edge(cfg.entry(), handler, EdgeKind::ExceptionUnwind);

        assert!(detect_tail_calls(&cfg).is_empty());
    }

    #[test]
    fn every_kind_of_call_in_tail_position_is_a_candidate() {
        for effect in [
            FlowEffect::Call,
            FlowEffect::IndirectCall,
            FlowEffect::ConditionalCall,
        ] {
            let mut cfg = Cfg::new();
            let exit = cfg.new_block();
            cfg.block_mut(cfg.entry())
                .instructions_mut()
                .push(df_with_effect(ff("call"), effect));
            cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

            assert_eq!(
                detect_tail_calls(&cfg)
                    .iter()
                    .map(|tail| tail.block)
                    .collect::<Vec<_>>(),
                [cfg.entry()],
                "{effect:?}"
            );
        }
    }

    #[test]
    fn an_unwind_edge_is_not_a_return_path() {
        // The call's only successor is a handler with no successors of its
        // own: control leaves by unwinding, not by returning.
        let mut cfg = Cfg::new();
        let handler = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(df_call("call", "callee", false));
        cfg.add_edge(cfg.entry(), handler, EdgeKind::ExceptionUnwind);

        assert!(detect_tail_calls(&cfg).is_empty());
    }
}
