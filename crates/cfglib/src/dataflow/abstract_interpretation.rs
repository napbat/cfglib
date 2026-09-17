//! Abstract interpretation framework.
//!
//! Generalises the fixpoint engine to work over arbitrary lattices,
//! enabling interval analysis, sign analysis, taint tracking, etc.
//!
//! Internally delegates to [`fixpoint::solve_problem`]
//! so there is a single worklist implementation in the crate.

extern crate alloc;
use alloc::collections::BTreeMap;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::fixpoint::{self, Direction, Problem};

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

    /// The abstract state leaving `block`, if the block was reached.
    #[must_use]
    pub fn fact_out(&self, block: BlockId) -> Option<&D> {
        self.block_out.get(&block)
    }
}

/// Bridge that adapts an [`AbstractDomain`] into a [`Problem`] so we
/// can reuse the generic fixpoint solver.
struct AbstractProblem<D> {
    _marker: core::marker::PhantomData<D>,
}

impl<I, E, D: AbstractDomain<I>> Problem<I, E> for AbstractProblem<D> {
    type Fact = D;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self) -> D {
        D::bottom()
    }

    fn entry_fact(&self) -> D {
        D::entry_value()
    }

    fn meet(&self, a: &D, b: &D) -> D {
        a.meet(b)
    }

    fn transfer(&self, cfg: &Cfg<I, E>, block: BlockId, input: &D) -> D {
        let mut state = input.clone();
        for inst in cfg.block(block).instructions() {
            state = D::transfer(&state, inst);
        }
        state
    }
}

/// Run abstract interpretation over a CFG.
///
/// Forward analysis delegating to the generic fixpoint solver.
/// The abstract domain `D` determines both the lattice and the
/// per-instruction transfer function.
///
/// # Panics
///
/// Panics only if the unbounded fixpoint solve reports a step-limit error,
/// which the unbounded configuration cannot produce.
#[must_use]
pub fn abstract_interpret<I, E, D: AbstractDomain<I>>(cfg: &Cfg<I, E>) -> AbstractFacts<D> {
    let problem = AbstractProblem::<D> {
        _marker: core::marker::PhantomData,
    };
    let result = fixpoint::solve_problem(cfg, &problem)
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
    use crate::test_util::ff;

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

    impl AbstractDomain<crate::test_util::MockInst> for Sign {
        fn transfer(state: &Self, _inst: &crate::test_util::MockInst) -> Self {
            state.clone() // identity transfer for testing
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
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let result: AbstractFacts<Sign> = abstract_interpret(&cfg);
        // Entry block out should be present and equal to entry value
        // (identity transfer).
        assert_eq!(result.fact_out(cfg.entry()), Some(&Sign::Zero));
    }
}
