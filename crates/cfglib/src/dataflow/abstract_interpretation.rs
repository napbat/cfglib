//! Abstract interpretation framework.
//!
//! Generalises the fixpoint engine to work over arbitrary lattices,
//! enabling interval analysis, sign analysis, taint tracking, etc.
//!
//! Internally delegates to the [edge-sensitive solver](super::edge_fixpoint),
//! so an unwind that leaves its block before a throwing instruction carries
//! the abstract state before that instruction.

extern crate alloc;
use alloc::collections::BTreeMap;

use crate::block::BlockId;
use crate::cfg::{Cfg, CfgEdge};
use crate::dataflow::edge_fixpoint::{self, EdgeProblem};
use crate::dataflow::fixpoint::Direction;
use crate::dataflow::{InstrInfo, departs_before_throws};

/// A lattice element for abstract interpretation.
pub trait Lattice: Clone + PartialEq {
    /// Bottom element (least precise / most conservative).
    fn bottom() -> Self;
    /// Top element (most precise / most optimistic).
    fn top() -> Self;
    /// Meet (greatest lower bound) of two elements.
    #[must_use]
    fn meet(&self, other: &Self) -> Self;
    /// Returns `true` if `self ⊑ other` in the lattice ordering.
    fn leq(&self, other: &Self) -> bool;
}

/// An abstract domain defines how instructions transform lattice values.
pub trait AbstractDomain<I>: Lattice {
    /// Transfer function: transform abstract state after executing
    /// instruction `inst`.
    fn transfer(state: &Self, inst: &I) -> Self;

    /// Initial abstract value for the entry block.
    fn entry_value() -> Self;
}

/// Per-block abstract states computed by abstract interpretation.
#[derive(Debug, Clone)]
pub struct AbstractFacts<D> {
    /// Abstract state at each block entry.
    block_in: BTreeMap<BlockId, D>,
    /// Abstract state at each block exit.
    block_out: BTreeMap<BlockId, D>,
}

impl<D> AbstractFacts<D> {
    /// The abstract state entering `block`, if the block was reached.
    #[must_use]
    pub fn fact_in(&self, block: BlockId) -> Option<&D> {
        self.block_in.get(&block)
    }

    /// The abstract state after `block` completes normally, if the block
    /// was reached.
    #[must_use]
    pub fn fact_out(&self, block: BlockId) -> Option<&D> {
        self.block_out.get(&block)
    }
}

/// Bridge that adapts an [`AbstractDomain`] into an [`EdgeProblem`] so we
/// can reuse the generic edge-sensitive solver.
struct AbstractProblem<D> {
    _marker: core::marker::PhantomData<D>,
}

impl<D> AbstractProblem<D> {
    /// Replays `block` from `input`, handing the state before each throwing
    /// instruction to `before_unwind`.
    fn replay<I: InstrInfo, E>(
        cfg: &Cfg<I, E>,
        block: BlockId,
        input: &D,
        mut before_unwind: impl FnMut(&D),
    ) -> D
    where
        D: AbstractDomain<I>,
    {
        let mut state = input.clone();
        for inst in cfg.block(block).instructions() {
            if inst.may_unwind() {
                before_unwind(&state);
            }
            state = D::transfer(&state, inst);
        }
        state
    }
}

impl<I: InstrInfo, E, D: AbstractDomain<I>> EdgeProblem<Cfg<I, E>> for AbstractProblem<D> {
    type Fact = D;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _cfg: &Cfg<I, E>) -> D {
        D::bottom()
    }

    /// A reached block that nothing branches to starts from the entry value.
    fn boundary(&self, cfg: &Cfg<I, E>, block: BlockId) -> Option<D> {
        cfg.incoming(block).next().is_none().then(D::entry_value)
    }

    fn meet(&self, left: &D, right: &D) -> D {
        left.meet(right)
    }

    fn transfer_node(&self, cfg: &Cfg<I, E>, block: BlockId, input: &D) -> D {
        Self::replay(cfg, block, input, |_| {})
    }

    fn transfer_edge(&self, cfg: &Cfg<I, E>, edge: CfgEdge<'_, E>, input: &D, output: &D) -> D {
        if !departs_before_throws(cfg, edge) {
            return output.clone();
        }
        let mut departing: Option<D> = None;
        Self::replay(cfg, edge.source(), input, |state| {
            departing = Some(match departing.take() {
                None => state.clone(),
                Some(earlier) => earlier.meet(state),
            });
        });
        departing.expect("an edge departing before throws leaves a block that may unwind")
    }
}

/// Run abstract interpretation over a CFG.
///
/// Forward analysis on the edge-sensitive solver. The abstract domain `D`
/// determines both the lattice and the per-instruction transfer function.
/// Blocks reached from the entry are solved; the others keep the bottom
/// state. An [`ExceptionUnwind`](crate::EdgeKind::ExceptionUnwind) edge
/// carries the meet of the states before each instruction of its source
/// that [`may_unwind`](InstrInfo::may_unwind), so a throwing instruction's
/// own effect and every later one in its block never reach the handler.
///
/// # Panics
///
/// Panics only if the unbounded fixpoint solve reports a step-limit error,
/// which the unbounded configuration cannot produce.
#[must_use]
pub fn abstract_interpret<I, E, D>(cfg: &Cfg<I, E>) -> AbstractFacts<D>
where
    I: InstrInfo,
    D: AbstractDomain<I>,
{
    let problem = AbstractProblem::<D> {
        _marker: core::marker::PhantomData,
    };
    let reachable = cfg.depth_first_preorder();
    let result = edge_fixpoint::solve_edge_problem_from(cfg, &problem, &reachable)
        .expect("an unbounded solve cannot exceed a step limit");

    // Convert Vec-indexed results to BTreeMap-keyed results.
    let mut block_in = BTreeMap::new();
    let mut block_out = BTreeMap::new();
    for block_id in cfg.block_ids() {
        block_in.insert(block_id, result.fact_in(block_id).clone());
        block_out.insert(block_id, result.fact_out(block_id).clone());
    }

    AbstractFacts {
        block_in,
        block_out,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::flow::FlowEffect;
    use crate::test_util::{DfInst, df_const, df_ff, df_with_effect};

    /// A trivial sign lattice: Bottom < Neg/Zero/Pos < Top.
    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Sign {
        Bottom,
        Neg,
        Zero,
        Pos,
        Top,
    }

    impl Lattice for Sign {
        fn bottom() -> Self {
            Sign::Bottom
        }
        fn top() -> Self {
            Sign::Top
        }
        fn meet(&self, other: &Self) -> Self {
            match (self, other) {
                (Sign::Top, x) | (x, Sign::Top) => x.clone(),
                (a, b) if a == b => a.clone(),
                _ => Sign::Bottom,
            }
        }
        fn leq(&self, other: &Self) -> bool {
            matches!(
                (self, other),
                (Sign::Bottom, _)
                    | (_, Sign::Top)
                    | (Sign::Neg, Sign::Neg)
                    | (Sign::Zero, Sign::Zero)
                    | (Sign::Pos, Sign::Pos)
            )
        }
    }

    /// The sign of the last constant written to variable 0; any other write
    /// to it is unknown.
    impl AbstractDomain<DfInst> for Sign {
        fn transfer(state: &Self, inst: &DfInst) -> Self {
            if !inst.defs.contains(&0) {
                return state.clone();
            }
            match inst.constant {
                Some(value) if value < 0 => Sign::Neg,
                Some(0) => Sign::Zero,
                Some(_) => Sign::Pos,
                None => Sign::Bottom,
            }
        }
        fn entry_value() -> Self {
            Sign::Zero
        }
    }

    #[test]
    fn sign_lattice_basics() {
        assert_eq!(Sign::Top.meet(&Sign::Neg), Sign::Neg);
        assert_eq!(Sign::Neg.meet(&Sign::Pos), Sign::Bottom);
        assert!(Sign::Bottom.leq(&Sign::Top));
    }

    #[test]
    fn abstract_interpret_linear() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(df_ff("a"));
        cfg.block_mut(b).instructions_mut().push(df_ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let result: AbstractFacts<Sign> = abstract_interpret(&cfg);
        // Entry block out should be present and equal to entry value
        // (identity transfer).
        assert_eq!(result.fact_out(cfg.entry()), Some(&Sign::Zero));
    }

    fn throwing(instruction: DfInst) -> DfInst {
        df_with_effect(instruction, FlowEffect::MayThrow)
    }

    /// `entry: x = 3`, then `protected` holding `instructions`, which falls
    /// through to `normal` and unwinds to `handler`. Returns
    /// `(facts, normal, handler)`.
    fn protected_region(
        instructions: impl IntoIterator<Item = DfInst>,
    ) -> (AbstractFacts<Sign>, BlockId, BlockId) {
        let mut cfg = Cfg::new();
        let protected = cfg.new_block();
        let normal = cfg.new_block();
        let handler = cfg.new_block();
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_edge(protected, normal, EdgeKind::Fallthrough);
        cfg.add_edge(protected, handler, EdgeKind::ExceptionUnwind);
        cfg.block_mut(cfg.entry()).push(df_const("prior", 0, 3));
        cfg.block_mut(protected)
            .instructions_mut()
            .extend(instructions);
        (abstract_interpret(&cfg), normal, handler)
    }

    #[test]
    fn a_handler_sees_the_state_before_a_faulting_write() {
        let (facts, normal, handler) = protected_region([throwing(df_const("load", 0, -2))]);

        assert_eq!(facts.fact_in(handler), Some(&Sign::Pos));
        assert_eq!(facts.fact_in(normal), Some(&Sign::Neg));
    }

    #[test]
    fn a_handler_meets_the_state_each_throw_observes() {
        let (facts, normal, handler) = protected_region([
            throwing(df_ff("first call")),
            df_const("negative", 0, -2),
            throwing(df_ff("second call")),
            df_const("zero", 0, 0),
        ]);

        assert_eq!(
            facts.fact_in(handler),
            Some(&Sign::Bottom),
            "the first call observes 3 and the second observes -2"
        );
        assert_eq!(facts.fact_in(normal), Some(&Sign::Zero));
    }
}
