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

extern crate alloc;
use alloc::collections::BTreeMap;

use super::InstrInfo;
use super::fixpoint::{self, Direction, Facts, Problem};
use crate::block::BlockId;
use crate::cfg::Cfg;

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
pub struct ConstPropProblem;

/// The flow fact: a map from variable to lattice value.
pub type ConstFact<V, C> = BTreeMap<V, ConstValue<C>>;

impl<I: ConstantFolder, E> Problem<I, E> for ConstPropProblem {
    type Fact = ConstFact<I::Variable, I::Const>;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self) -> Self::Fact {
        BTreeMap::new()
    }

    fn entry_fact(&self) -> Self::Fact {
        BTreeMap::new()
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        let mut result = a.clone();
        for (variable, value) in b {
            let entry = result.entry(variable.clone()).or_insert(ConstValue::Top);
            *entry = entry.clone().meet_with(value.clone(), I::meet_constants);
        }
        result
    }

    fn transfer(&self, cfg: &Cfg<I, E>, block: BlockId, input: &Self::Fact) -> Self::Fact {
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

/// Run constant propagation on the CFG.
///
/// # Panics
///
/// Panics only if the unbounded fixpoint solve reports a step-limit error,
/// which the unbounded configuration cannot produce.
#[must_use]
pub fn constant_propagation<I: ConstantFolder, E>(
    cfg: &Cfg<I, E>,
) -> Facts<ConstFact<I::Variable, I::Const>> {
    fixpoint::solve_problem(cfg, &ConstPropProblem)
        .expect("an unbounded solve cannot exceed a step limit")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::{DfInst, UdInst, df_const, df_def, df_use, ud_inst};

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
}
