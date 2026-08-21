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
use super::fixpoint::{self, Direction, FixpointResult, Problem};
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
    #[must_use]
    pub fn meet(self, other: Self) -> Self {
        match (self, other) {
            (ConstValue::Top, x) | (x, ConstValue::Top) => x,
            (ConstValue::Const(a), ConstValue::Const(b)) if a == b => ConstValue::Const(a),
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
}

/// The constant propagation problem.
pub struct ConstPropProblem;

/// The flow fact: a map from variable to lattice value.
pub type ConstFact<V, C> = BTreeMap<V, ConstValue<C>>;

impl<I: ConstantFolder> Problem<I> for ConstPropProblem {
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
            *entry = entry.clone().meet(value.clone());
        }
        result
    }

    fn transfer(&self, cfg: &Cfg<I>, block: BlockId, input: &Self::Fact) -> Self::Fact {
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
#[must_use]
pub fn constant_propagation<I: ConstantFolder>(
    cfg: &Cfg<I>,
) -> FixpointResult<ConstFact<I::Variable, I::Const>> {
    fixpoint::solve(cfg, &ConstPropProblem)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::{DfInst, df_const, df_def, df_use};

    #[derive(Clone, Copy)]
    enum Fold {
        Literal(i64),
        Copy(u16),
        Opaque,
    }

    struct KnownSensitiveInst {
        uses: alloc::vec::Vec<u16>,
        defs: alloc::vec::Vec<u16>,
        fold: Fold,
    }

    impl InstrInfo for KnownSensitiveInst {
        type Variable = u16;

        fn uses(&self) -> &[Self::Variable] {
            &self.uses
        }

        fn defs(&self) -> &[Self::Variable] {
            &self.defs
        }
    }

    impl ConstantFolder for KnownSensitiveInst {
        type Const = i64;

        fn fold_constant(
            &self,
            known: &BTreeMap<Self::Variable, Self::Const>,
        ) -> Option<(Self::Variable, Self::Const)> {
            let destination = *self.defs.first()?;
            match self.fold {
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
            .instructions_vec_mut()
            .extend([df_const("load_42", 0, 42)]);
        cfg.block_mut(exit)
            .instructions_vec_mut()
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
        cfg.block_mut(cfg.entry()).instructions_vec_mut().extend([
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
            .instructions_vec_mut()
            .push(df_def("generic_def", 0));

        let result = constant_propagation(&cfg);
        let fact_out = result.fact_out(cfg.entry());
        assert_eq!(fact_out.get(&0), Some(&ConstValue::Bottom));
    }

    #[test]
    fn known_constants_are_updated_and_killed_within_one_block() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_vec_mut().extend([
            KnownSensitiveInst {
                uses: alloc::vec![],
                defs: alloc::vec![0],
                fold: Fold::Literal(7),
            },
            KnownSensitiveInst {
                uses: alloc::vec![0],
                defs: alloc::vec![1],
                fold: Fold::Copy(0),
            },
            KnownSensitiveInst {
                uses: alloc::vec![],
                defs: alloc::vec![0],
                fold: Fold::Opaque,
            },
            KnownSensitiveInst {
                uses: alloc::vec![0],
                defs: alloc::vec![2],
                fold: Fold::Copy(0),
            },
            KnownSensitiveInst {
                uses: alloc::vec![1],
                defs: alloc::vec![3],
                fold: Fold::Copy(1),
            },
        ]);

        let result = constant_propagation(&cfg);
        let out = result.fact_out(cfg.entry());
        assert_eq!(out.get(&0), Some(&ConstValue::Bottom));
        assert_eq!(out.get(&1), Some(&ConstValue::Const(7)));
        assert_eq!(out.get(&2), Some(&ConstValue::Bottom));
        assert_eq!(out.get(&3), Some(&ConstValue::Const(7)));
    }
}
