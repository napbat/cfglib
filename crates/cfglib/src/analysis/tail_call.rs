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
/// A call is any of [`Call`](FlowEffect::Call),
/// [`IndirectCall`](FlowEffect::IndirectCall), or
/// [`ConditionalCall`](FlowEffect::ConditionalCall). A block holds a
/// candidate in one of two shapes:
///
/// - **Call, then return, in one block.** The block has no successor and
///   ends in a [`Return`](FlowEffect::Return) instruction directly after a
///   call; this is what [`CfgBuilder`](crate::CfgBuilder) produces for
///   `call; ret`. The report names the call.
/// - **Call ending its block.** The call is the block's last instruction and
///   its only successor is an exit block (one with no successors) reached by
///   an ordinary edge that returns at once: the exit block is empty, an
///   implicit return, or its first instruction past any
///   [`Label`](FlowEffect::Label) or [`Declaration`](FlowEffect::Declaration)
///   is a [`Return`](FlowEffect::Return).
///
/// An exit block that does other work first, such as using the call's
/// result, or that raises or terminates instead of returning, is not a
/// return. A block with an exceptional edge is never a candidate, since the
/// call can unwind into a handler that has to run. For instructions that
/// carry explicit tail-call markers, use [`detect_explicit_tail_calls`]
/// instead (or in addition).
#[must_use]
pub fn detect_tail_calls<G>(cfg: &G) -> Vec<TailCall>
where
    G: CfgView,
    G::Inst: FlowControl,
{
    let is_call = |instruction: &G::Inst| {
        matches!(
            instruction.flow_effect(),
            FlowEffect::Call | FlowEffect::IndirectCall | FlowEffect::ConditionalCall
        )
    };
    let mut results = Vec::new();
    for bid in cfg.node_ids() {
        let instructions = cfg.instructions(bid);
        let mut outgoing = cfg.outgoing(bid);
        let call_idx = match (outgoing.next(), outgoing.next()) {
            (None, _) => match instructions {
                [.., call, last] if is_call(call) && last.flow_effect() == FlowEffect::Return => {
                    Some(instructions.len() - 2)
                }
                _ => None,
            },
            (Some(edge), None) => (!cfg.edge_kind(edge).is_exceptional()
                && returns_at_once(cfg, cfg.edge(edge).target())
                && instructions.last().is_some_and(is_call))
            .then(|| instructions.len() - 1),
            _ => None,
        };
        if let Some(inst_idx) = call_idx {
            results.push(TailCall {
                block: bid,
                inst_idx,
                explicit: false,
            });
        }
    }

    results
}

/// Whether `block` has no successors and returns at once: it is empty, or
/// its first instruction that does something is a return.
fn returns_at_once<G>(cfg: &G, block: BlockId) -> bool
where
    G: CfgView,
    G::Inst: FlowControl,
{
    cfg.outgoing(block).next().is_none()
        && cfg
            .instructions(block)
            .iter()
            .map(FlowControl::flow_effect)
            .find(|effect| !matches!(effect, FlowEffect::Label | FlowEffect::Declaration))
            .is_none_or(|effect| effect == FlowEffect::Return)
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
        cfg.block_mut(exit)
            .instructions_mut()
            .push(df_with_effect(ff("ret"), FlowEffect::Return));
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
        assert_eq!(detect_tail_calls(&not_last), []);

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
        assert_eq!(detect_tail_calls(&continues), []);
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

        assert_eq!(detect_tail_calls(&cfg), []);
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

        assert_eq!(detect_tail_calls(&cfg), []);
    }

    #[test]
    fn a_call_followed_by_a_return_in_one_built_block_is_a_tail_call() {
        // `CfgBuilder` keeps the return in the block that holds the call, so
        // this block has no successor at all: the canonical `call; ret`.
        let build = |instructions: alloc::vec::Vec<DfInst>| {
            crate::builder::CfgBuilder::build(instructions).expect("a flat stream always builds")
        };
        let ret = || effect("ret", FlowEffect::Return);

        let cfg = build(alloc::vec![ff("setup"), df_call("call", "f", false), ret()]);
        assert_eq!(
            detect_tail_calls(&cfg),
            [TailCall {
                block: cfg.entry(),
                inst_idx: 1,
                explicit: false,
            }]
        );

        let indirect = effect("icall", FlowEffect::IndirectCall);
        let cfg = build(alloc::vec![indirect, ret()]);
        assert_eq!(detect_tail_calls(&cfg).len(), 1, "an indirect call too");

        let cfg = build(alloc::vec![df_call("call", "f", false), ff("use"), ret()]);
        assert_eq!(detect_tail_calls(&cfg), [], "work between call and return");

        let cfg = build(alloc::vec![df_call("call", "f", false), ff("use")]);
        assert_eq!(detect_tail_calls(&cfg), [], "no return follows the call");

        let cfg = build(alloc::vec![ff("setup"), ret()]);
        assert_eq!(detect_tail_calls(&cfg), [], "a return with no call");
    }

    #[test]
    fn a_call_and_return_block_that_can_unwind_is_not_a_tail_call() {
        let mut cfg = Cfg::new();
        let handler = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            df_call("call", "f", false),
            effect("ret", FlowEffect::Return),
        ]);
        cfg.add_edge(cfg.entry(), handler, EdgeKind::ExceptionUnwind);

        assert_eq!(detect_tail_calls(&cfg), []);
    }

    fn effect(name: &'static str, effect: FlowEffect) -> DfInst {
        df_with_effect(ff(name), effect)
    }

    /// A CFG whose entry ends in a call that falls into one exit block
    /// holding `exit`.
    fn call_into(exit: impl IntoIterator<Item = DfInst>) -> Cfg<DfInst> {
        let mut cfg = Cfg::new();
        let exit_block = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(df_call("call", "callee", false));
        cfg.block_mut(exit_block).instructions_mut().extend(exit);
        cfg.add_edge(cfg.entry(), exit_block, EdgeKind::Fallthrough);
        cfg
    }

    #[test]
    fn an_exit_block_that_returns_at_once_makes_the_call_a_tail_call() {
        let ret = || effect("ret", FlowEffect::Return);
        let cases = [
            ("an empty exit is an implicit return", alloc::vec![]),
            ("an explicit return", alloc::vec![ret()]),
            (
                "labels and declarations do no work",
                alloc::vec![
                    effect("label", FlowEffect::Label),
                    effect("dcl", FlowEffect::Declaration),
                    ret(),
                ],
            ),
        ];
        for (name, exit) in cases {
            assert_eq!(detect_tail_calls(&call_into(exit)).len(), 1, "{name}");
        }
    }

    #[test]
    fn an_exit_block_that_does_work_or_does_not_return_is_not_a_tail_call() {
        let ret = || effect("ret", FlowEffect::Return);
        let cases = [
            (
                "the call's result is used before the return",
                alloc::vec![ff("add"), ret()],
            ),
            (
                "the exit raises instead of returning",
                alloc::vec![effect("raise", FlowEffect::MayThrow)],
            ),
            (
                "the exit terminates the invocation",
                alloc::vec![effect("abort", FlowEffect::Terminate)],
            ),
            (
                "only a conditional return",
                alloc::vec![effect("retc", FlowEffect::ConditionalReturn)],
            ),
        ];
        for (name, exit) in cases {
            assert!(detect_tail_calls(&call_into(exit)).is_empty(), "{name}");
        }
    }
}
