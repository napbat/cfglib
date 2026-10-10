//! Constant propagation analysis.
//!
//! A **forward** dataflow analysis that tracks which IR variables hold
//! known constant values. Uses a simple three-level lattice per
//! variable: `Top` (unknown/uninitialized) → `Const(C)` → `Bottom`
//! (overdefined / multiple conflicting values).
//!
//! The constant domain `C` is consumer-typed via
//! [`ConstantFolder::Const`] — a machine word for a binary adapter, a
//! literal enum (int/float/string/bool) for a source language.
//!
//! An unwind that leaves its block before a throwing instruction carries
//! the constants known before that instruction, so the solve runs on the
//! [edge-sensitive solver](super::edge_fixpoint).

extern crate alloc;
use alloc::collections::BTreeMap;

use super::edge_fixpoint::{self, EdgeProblem};
use super::fixpoint::{Direction, Facts};
use super::{InstrInfo, departs_before_throws};
use crate::block::BlockId;
use crate::cfg::{Cfg, CfgEdge};

/// The lattice value for a single variable, over constant domain `C`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ConstValue<C> {
    /// Not yet analyzed (top of lattice).
    Top,
    /// Known constant.
    Const(C),
    /// Overdefined — seen different values on different paths.
    Bottom,
}

impl<C: Clone + Eq> ConstValue<C> {
    /// Meet (join) two lattice values.
    ///
    /// - Top ⊓ x = x
    /// - x ⊓ Top = x
    /// - Const(a) ⊓ Const(a) = Const(a)
    /// - Const(a) ⊓ Const(b) = Bottom  (a ≠ b)
    /// - Bottom ⊓ x = Bottom
    ///
    /// This is [`meet_with`](Self::meet_with) under the generic equality
    /// rule. A caller that has a [`ConstantFolder`] adapter meets through
    /// [`ConstantFolder::meet_constants`] instead, so a domain that holds
    /// partial knowledge keeps the part both values agree on.
    #[must_use]
    pub fn meet(self, other: Self) -> Self {
        self.meet_with(other, |a, b| (a == b).then(|| a.clone()))
    }

    /// Meet two lattice values with a domain-defined constant meet.
    ///
    /// `meet_constants` answers the greatest constant below both of its
    /// arguments, or `None` when no constant is below both. `Top` and
    /// `Bottom` behave as in [`meet`](Self::meet); only two constants
    /// consult the hook.
    #[must_use]
    pub fn meet_with(self, other: Self, meet_constants: impl FnOnce(&C, &C) -> Option<C>) -> Self {
        match (self, other) {
            (ConstValue::Top, x) | (x, ConstValue::Top) => x,
            (ConstValue::Const(a), ConstValue::Const(b)) => {
                meet_constants(&a, &b).map_or(ConstValue::Bottom, ConstValue::Const)
            }
            _ => ConstValue::Bottom,
        }
    }

    /// Whether this is a known constant.
    #[must_use]
    pub fn is_const(&self) -> bool {
        matches!(self, ConstValue::Const(_))
    }

    /// Borrow the constant value, if any.
    #[must_use]
    pub fn as_const(&self) -> Option<&C> {
        match self {
            ConstValue::Const(v) => Some(v),
            _ => None,
        }
    }
}

/// Trait for instructions that can produce constant values.
///
/// This is the IR-specific bridge: the consumer implements this
/// to tell the analysis what constant, if any, an instruction
/// produces given known-constant inputs.
pub trait ConstantFolder: InstrInfo {
    /// Consumer constant domain: a machine word, float bits, or a
    /// source-literal enum. `Eq` rather than `Ord` so float-bits wrappers
    /// qualify.
    type Const: Clone + Eq;

    /// If this instruction produces a constant for a defined variable
    /// given the current known constants, return `Some((loc, value))`.
    ///
    /// `known` maps variables to their known constant values (only
    /// entries with `Const(v)` are present).
    ///
    /// Return `None` to leave the default behavior (mark all defs as
    /// Bottom).
    fn fold_constant(
        &self,
        known: &BTreeMap<Self::Variable, Self::Const>,
    ) -> Option<(Self::Variable, Self::Const)>;

    /// Decides the conditional transfer this instruction ends its block
    /// with.
    ///
    /// Returns `Some(true)` when the known constants prove that the
    /// [`ConditionalTrue`](crate::EdgeKind::ConditionalTrue) edge is taken,
    /// and `Some(false)` when they prove that the
    /// [`ConditionalFalse`](crate::EdgeKind::ConditionalFalse) edge is
    /// taken. Returns `None` when the instruction is not a conditional
    /// terminator, or when the condition is not decided.
    ///
    /// `known` is built exactly as for
    /// [`fold_constant`](Self::fold_constant): only variables whose lattice
    /// value is `Const(v)` are present.
    ///
    /// The default answers `None`, which leaves every outgoing edge of the
    /// block executable. Only
    /// [`SccpAnalysis`](crate::SccpAnalysis) reads this hook; the
    /// unconditional [`constant_propagation`] solve ignores it.
    fn fold_branch(&self, known: &BTreeMap<Self::Variable, Self::Const>) -> Option<bool> {
        let _ = known;
        None
    }

    /// Meets two constants of this domain.
    ///
    /// `Some(c)` is the greatest constant below both `a` and `b`. `None` is
    /// the lattice bottom: no constant of the domain is below both.
    ///
    /// A domain of exact values answers `Some` only for equal arguments,
    /// which is the default. A domain that carries partial knowledge, such
    /// as known bits, answers with the part on which `a` and `b` agree.
    ///
    /// The result must be below both arguments, so meeting it again with
    /// either argument must return the result itself. A hook that breaks
    /// that contract could raise a lattice value and make the solve
    /// diverge; [`SccpAnalysis`](crate::SccpAnalysis) checks it with a
    /// debug assertion.
    fn meet_constants(a: &Self::Const, b: &Self::Const) -> Option<Self::Const> {
        (a == b).then(|| a.clone())
    }
}

/// The constant propagation problem.
///
/// A normal edge carries the constants known at its source's end. An edge
/// that leaves before throwing instructions carries the meet of the
/// constants known before each of them, so a throwing instruction's own
/// definitions and every later one in its block never reach the handler.
pub struct ConstPropProblem;

/// The flow fact: a map from variable to lattice value.
pub type ConstFact<V, C> = BTreeMap<V, ConstValue<C>>;

impl ConstPropProblem {
    /// Meets two facts variable by variable through the domain's meet.
    fn meet_facts<I: ConstantFolder>(
        left: &ConstFact<I::Variable, I::Const>,
        right: &ConstFact<I::Variable, I::Const>,
    ) -> ConstFact<I::Variable, I::Const> {
        let mut result = left.clone();
        for (variable, value) in right {
            let entry = result.entry(variable.clone()).or_insert(ConstValue::Top);
            *entry = entry.clone().meet_with(value.clone(), I::meet_constants);
        }
        result
    }

    /// Replays `block` over the constants known at its entry, handing the
    /// state before each throwing instruction to `before_unwind`.
    fn replay<I: ConstantFolder, E>(
        cfg: &Cfg<I, E>,
        block: BlockId,
        input: &ConstFact<I::Variable, I::Const>,
        mut before_unwind: impl FnMut(&ConstFact<I::Variable, I::Const>),
    ) -> ConstFact<I::Variable, I::Const> {
        let mut state = input.clone();
        let mut known: BTreeMap<I::Variable, I::Const> = state
            .iter()
            .filter_map(|(variable, value)| {
                value
                    .as_const()
                    .map(|constant| (variable.clone(), constant.clone()))
            })
            .collect();

        for inst in cfg.block(block).instructions() {
            if inst.may_unwind() {
                before_unwind(&state);
            }
            // Try constant folding. The folder answers for ONE def, but a
            // multi-def instruction redefined its co-defined variables too:
            // bottom every def first so no stale constant survives, then
            // record the folded one.
            let folded = inst.fold_constant(&known);
            for variable in inst.defs() {
                state.insert(variable.clone(), ConstValue::Bottom);
                known.remove(variable);
            }
            if let Some((loc, val)) = folded {
                state.insert(loc.clone(), ConstValue::Const(val.clone()));
                known.insert(loc, val);
            }
        }

        state
    }
}

impl<I: ConstantFolder, E> EdgeProblem<Cfg<I, E>> for ConstPropProblem {
    type Fact = ConstFact<I::Variable, I::Const>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _cfg: &Cfg<I, E>) -> Self::Fact {
        BTreeMap::new()
    }

    fn meet(&self, left: &Self::Fact, right: &Self::Fact) -> Self::Fact {
        Self::meet_facts::<I>(left, right)
    }

    fn transfer_node(&self, cfg: &Cfg<I, E>, block: BlockId, input: &Self::Fact) -> Self::Fact {
        Self::replay(cfg, block, input, |_| {})
    }

    fn transfer_edge(
        &self,
        cfg: &Cfg<I, E>,
        edge: CfgEdge<'_, E>,
        input: &Self::Fact,
        output: &Self::Fact,
    ) -> Self::Fact {
        if !departs_before_throws(cfg, edge) {
            return output.clone();
        }
        let mut departing: Option<Self::Fact> = None;
        Self::replay(cfg, edge.source(), input, |state| {
            departing = Some(match departing.take() {
                None => state.clone(),
                Some(earlier) => Self::meet_facts::<I>(&earlier, state),
            });
        });
        departing.expect("an edge departing before throws leaves a block that may unwind")
    }
}

/// Run constant propagation on the CFG.
///
/// Blocks unreachable from the entry keep empty facts and contribute
/// nothing to the blocks they branch to. A block's entry fact meets every
/// incoming edge, and an unwind edge carries only the constants known
/// before its source's throwing instructions.
///
/// # Panics
///
/// Panics only if the unbounded fixpoint solve reports a step-limit error,
/// which the unbounded configuration cannot produce.
#[must_use]
pub fn constant_propagation<I: ConstantFolder, E>(
    cfg: &Cfg<I, E>,
) -> Facts<ConstFact<I::Variable, I::Const>> {
    let reachable = cfg.depth_first_preorder();
    edge_fixpoint::solve_edge_problem_from(cfg, &ConstPropProblem, &reachable)
        .expect("an unbounded solve cannot exceed a step limit")
        .into_block_facts()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::flow::FlowEffect;
    use crate::test_util::{
        DfInst, UdInst, df_const, df_def, df_ff, df_use, df_with_effect, ud_inst,
    };

    #[derive(Debug, Clone, Copy)]
    enum Fold {
        Literal(i64),
        Copy(u16),
        Opaque,
    }

    /// A known-map-sensitive folder over the shared uses/defs mock.
    type KnownSensitiveInst = UdInst<Fold>;

    impl ConstantFolder for KnownSensitiveInst {
        type Const = i64;

        fn fold_constant(
            &self,
            known: &BTreeMap<Self::Variable, Self::Const>,
        ) -> Option<(Self::Variable, Self::Const)> {
            let destination = *self.defs.first()?;
            match self.payload {
                Fold::Literal(value) => Some((destination, value)),
                Fold::Copy(source) => known
                    .get(&source)
                    .copied()
                    .map(|value| (destination, value)),
                Fold::Opaque => None,
            }
        }
    }

    #[test]
    fn meet_top_with_const() {
        assert_eq!(
            ConstValue::Top.meet(ConstValue::Const(42)),
            ConstValue::Const(42)
        );
        assert_eq!(
            ConstValue::Const(42).meet(ConstValue::Top),
            ConstValue::Const(42)
        );
    }

    #[test]
    fn meet_same_const() {
        assert_eq!(
            ConstValue::Const(7).meet(ConstValue::Const(7)),
            ConstValue::Const(7)
        );
    }

    #[test]
    fn meet_different_consts_is_bottom() {
        assert_eq!(
            ConstValue::Const(1).meet(ConstValue::Const(2)),
            ConstValue::Bottom
        );
    }

    #[test]
    fn meet_bottom_absorbs() {
        assert_eq!(
            ConstValue::Bottom.meet(ConstValue::Const(5)),
            ConstValue::Bottom
        );
        assert_eq!(
            ConstValue::Const(5).meet(ConstValue::Bottom),
            ConstValue::Bottom
        );
        assert_eq!(
            ConstValue::<i64>::Bottom.meet(ConstValue::Top),
            ConstValue::Bottom
        );
    }

    #[test]
    fn is_const_and_as_const() {
        assert!(ConstValue::Const(1).is_const());
        assert_eq!(ConstValue::Const(1).as_const(), Some(&1));
        assert!(!ConstValue::<i64>::Top.is_const());
        assert_eq!(ConstValue::<i64>::Top.as_const(), None);
        assert!(!ConstValue::<i64>::Bottom.is_const());
        assert_eq!(ConstValue::<i64>::Bottom.as_const(), None);
    }

    #[test]
    fn constant_propagation_tracks_const_def() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();
        // Entry: const 42 → loc0, then use loc0
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .extend([df_const("load_42", 0, 42)]);
        cfg.block_mut(exit)
            .instructions_mut()
            .push(df_use("use0", 0));
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

        let result = constant_propagation(&cfg);
        let fact_out = result.fact_out(cfg.entry());
        assert_eq!(fact_out.get(&0), Some(&ConstValue::Const(42)));
    }

    #[test]
    fn folding_one_def_still_bottoms_its_co_defined_variables() {
        // var 2 holds Const(5); a multi-def instruction {1, 2} folds only
        // var 1. Var 2 was still redefined — its stale constant must not
        // survive (a dxbc udiv writes quotient AND remainder).
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            df_const("load5", 2, 5),
            DfInst {
                defs: alloc::vec![1, 2],
                constant: Some(9),
                ..crate::test_util::df_ff("multi_def")
            },
        ]);

        let result = constant_propagation(&cfg);
        let out = result.fact_out(cfg.entry());
        assert_eq!(out.get(&1), Some(&ConstValue::Const(9)));
        assert_eq!(
            out.get(&2),
            Some(&ConstValue::Bottom),
            "co-defined variable must not keep its pre-redefinition constant"
        );
    }

    #[test]
    fn constant_propagation_non_const_def_is_bottom() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        // Entry: def loc0 (non-constant)
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(df_def("generic_def", 0));

        let result = constant_propagation(&cfg);
        let fact_out = result.fact_out(cfg.entry());
        assert_eq!(fact_out.get(&0), Some(&ConstValue::Bottom));
    }

    #[test]
    fn known_constants_are_updated_and_killed_within_one_block() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            ud_inst(&[], &[0], Fold::Literal(7)),
            ud_inst(&[0], &[1], Fold::Copy(0)),
            ud_inst(&[], &[0], Fold::Opaque),
            ud_inst(&[0], &[2], Fold::Copy(0)),
            ud_inst(&[1], &[3], Fold::Copy(1)),
        ]);

        let result = constant_propagation(&cfg);
        let out = result.fact_out(cfg.entry());
        assert_eq!(out.get(&0), Some(&ConstValue::Bottom));
        assert_eq!(out.get(&1), Some(&ConstValue::Const(7)));
        assert_eq!(out.get(&2), Some(&ConstValue::Bottom));
        assert_eq!(out.get(&3), Some(&ConstValue::Const(7)));
    }

    fn throwing(instruction: DfInst) -> DfInst {
        df_with_effect(instruction, FlowEffect::MayThrow)
    }

    /// `entry: x = 1`, then `protected` holding `instructions`, which falls
    /// through to `normal` and unwinds to `handler`. Returns
    /// `(cfg, normal, handler)`.
    fn protected_region(
        instructions: impl IntoIterator<Item = DfInst>,
    ) -> (Cfg<DfInst>, BlockId, BlockId) {
        let mut cfg = Cfg::new();
        let protected = cfg.new_block();
        let normal = cfg.new_block();
        let handler = cfg.new_block();
        cfg.add_edge(cfg.entry(), protected, EdgeKind::Fallthrough);
        cfg.add_edge(protected, normal, EdgeKind::Fallthrough);
        cfg.add_edge(protected, handler, EdgeKind::ExceptionUnwind);
        cfg.block_mut(cfg.entry()).push(df_const("prior", 0, 1));
        cfg.block_mut(protected)
            .instructions_mut()
            .extend(instructions);
        (cfg, normal, handler)
    }

    #[test]
    fn a_handler_sees_the_constant_before_a_faulting_load() {
        let (cfg, normal, handler) = protected_region([throwing(df_const("load", 0, 2))]);

        let result = constant_propagation(&cfg);
        assert_eq!(result.fact_in(handler).get(&0), Some(&ConstValue::Const(1)));
        assert_eq!(result.fact_in(normal).get(&0), Some(&ConstValue::Const(2)));
    }

    #[test]
    fn a_handler_meets_the_constants_each_throw_observes() {
        let (cfg, normal, handler) = protected_region([
            throwing(df_ff("first call")),
            df_const("second value", 0, 2),
            throwing(df_ff("second call")),
            df_const("final value", 0, 3),
        ]);

        let result = constant_propagation(&cfg);
        assert_eq!(
            result.fact_in(handler).get(&0),
            Some(&ConstValue::Bottom),
            "the first call observes 1 and the second observes 2"
        );
        assert_eq!(result.fact_in(normal).get(&0), Some(&ConstValue::Const(3)));
    }
}
