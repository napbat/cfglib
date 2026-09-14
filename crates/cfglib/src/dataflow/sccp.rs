//! Sparse Conditional Constant Propagation (SCCP).
//!
//! SCCP operates on the generic renamed values in [`SsaForm`] while asking
//! the source instruction adapter to fold native instructions.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::constant_propagation::{ConstValue, ConstantFolder};
use crate::dataflow::ssa::{SsaForm, SsaInstruction, SsaValue};
use crate::dataflow::{InstrInfo, VariableId};
use crate::edge::EdgeKind;

/// Result of SCCP analysis.
#[derive(Debug, Clone)]
pub struct SccpAnalysis<V, C> {
    /// Lattice value computed for each renamed SSA value.
    pub values: BTreeMap<SsaValue<V>, ConstValue<C>>,
    /// CFG edges proven executable.
    pub executable_edges: BTreeSet<(BlockId, BlockId)>,
    /// CFG blocks proven reachable.
    pub reachable_blocks: BTreeSet<BlockId>,
}

/// The current lattice value of an SSA value. An absent version-0 value is a
/// live-in — an input the analysis cannot know — so it is `Bottom`, never the
/// optimistic `Top` (which would let a phi fold a runtime-varying input into
/// one arm's constant). Absent positive versions are merely not-yet-computed.
fn lattice_of<V: VariableId, C: Clone + Eq>(
    values: &BTreeMap<SsaValue<V>, ConstValue<C>>,
    value: &SsaValue<V>,
) -> ConstValue<C> {
    values.get(value).cloned().unwrap_or(if value.version == 0 {
        ConstValue::Bottom
    } else {
        ConstValue::Top
    })
}

/// Whether `lower` is at or below `upper` in the lattice the adapter's
/// [`ConstantFolder::meet_constants`] defines.
///
/// Two constants compare through the hook itself: `lower` is below `upper`
/// exactly when meeting the two returns `lower` again.
fn is_below<I: ConstantFolder>(lower: &ConstValue<I::Const>, upper: &ConstValue<I::Const>) -> bool {
    match (lower, upper) {
        (ConstValue::Bottom, _) | (_, ConstValue::Top) => true,
        (ConstValue::Const(lower), ConstValue::Const(upper)) => {
            I::meet_constants(lower, upper).as_ref() == Some(lower)
        }
        _ => false,
    }
}

fn update_value<I: ConstantFolder>(
    values: &mut BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    worklist: &mut Vec<SsaValue<I::Variable>>,
    value: &SsaValue<I::Variable>,
    candidate: ConstValue<I::Const>,
) {
    let previous = lattice_of(values, value);
    let next = previous.clone().meet_with(candidate, I::meet_constants);
    debug_assert!(
        is_below::<I>(&next, &previous),
        "a constant meet must stay below the value it refines, or the solve can rise and diverge"
    );
    if next != previous {
        values.insert(value.clone(), next);
        worklist.push(value.clone());
    }
}

/// The constants known at one instruction's inputs: the entries of its uses
/// that already hold `Const`. Both folding hooks read the same map, so the
/// branch predicate sees exactly what the value folder saw.
fn known_constants<V: VariableId, C: Clone + Eq>(
    annotation: &SsaInstruction<V>,
    values: &BTreeMap<SsaValue<V>, ConstValue<C>>,
) -> BTreeMap<V, C> {
    annotation
        .uses
        .iter()
        .filter_map(|value| {
            values
                .get(value)
                .and_then(ConstValue::as_const)
                .map(|constant| (value.variable.clone(), constant.clone()))
        })
        .collect()
}

fn evaluate_block<I: ConstantFolder, E>(
    cfg: &Cfg<I, E>,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    values: &mut BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    worklist: &mut Vec<SsaValue<I::Variable>>,
) {
    for (instruction, annotation) in cfg
        .block(block)
        .instructions()
        .iter()
        .zip(&ssa.block(block).instructions)
    {
        let known = known_constants(annotation, values);

        if let Some((variable, constant)) = instruction.fold_constant(&known) {
            // The folder answers for ONE def; every co-defined variable of a
            // multi-def instruction was still redefined and must bottom.
            for definition in &annotation.defs {
                if definition.variable == variable {
                    update_value::<I>(
                        values,
                        worklist,
                        definition,
                        ConstValue::Const(constant.clone()),
                    );
                } else {
                    update_value::<I>(values, worklist, definition, ConstValue::Bottom);
                }
            }
        } else {
            for definition in &annotation.defs {
                update_value::<I>(values, worklist, definition, ConstValue::Bottom);
            }
        }
    }
}

/// The branch outcome the last instruction of `block` proves, or `None` when
/// the block does not end in a decided conditional transfer.
fn decided_branch<I: ConstantFolder, E>(
    cfg: &Cfg<I, E>,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    values: &BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
) -> Option<bool> {
    let (instruction, annotation) = cfg
        .block(block)
        .instructions()
        .iter()
        .zip(&ssa.block(block).instructions)
        .next_back()?;
    instruction.fold_branch(&known_constants(annotation, values))
}

/// Queue every outgoing edge of `block` the terminator leaves executable.
///
/// A decided predicate proves exactly one of the two conditional arms, so
/// the opposite arm is withheld. An outgoing edge of any other kind stays
/// executable: the predicate says nothing about an exceptional unwind, a
/// back edge, or any other transfer a frontend attached to the same block.
///
/// Edges are only ever added. A predicate that later lowers to `Bottom`
/// re-derives as undecided and then activates the arm it had withheld, so
/// the executable set stays monotone. Re-derivation happens on every drain
/// scan, so an edge already in `executable` is skipped rather than queued
/// again for a pop that would do nothing.
fn activate_successors<I: ConstantFolder, E>(
    cfg: &Cfg<I, E>,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    values: &BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    executable: &BTreeSet<(BlockId, BlockId)>,
    worklist: &mut Vec<(BlockId, BlockId)>,
) {
    let withheld = match decided_branch(cfg, ssa, block, values) {
        Some(true) => Some(EdgeKind::ConditionalFalse),
        Some(false) => Some(EdgeKind::ConditionalTrue),
        None => None,
    };
    for &id in cfg.successor_edges(block) {
        let edge = cfg.edge(id);
        if Some(edge.kind()) != withheld && !executable.contains(&(block, edge.target())) {
            worklist.push((block, edge.target()));
        }
    }
}

/// Re-meet every phi of `block` over its currently executable incoming
/// edges. Called when an edge activates AND from the value-worklist drain:
/// a phi whose operand lowers after all its edges are already executable
/// must be re-evaluated, or it keeps a stale over-optimistic constant.
fn evaluate_phis<I: ConstantFolder>(
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    executable_edges: &BTreeSet<(BlockId, BlockId)>,
    values: &mut BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    worklist: &mut Vec<SsaValue<I::Variable>>,
) {
    for phi in &ssa.block(block).phis {
        let mut candidate = ConstValue::Top;
        for (predecessor, operand) in &phi.operands {
            if executable_edges.contains(&(*predecessor, block)) {
                candidate = candidate.meet_with(lattice_of(values, operand), I::meet_constants);
            }
        }
        update_value::<I>(values, worklist, &phi.result, candidate);
    }
}

impl<V: VariableId, C: Clone + Eq> SccpAnalysis<V, C> {
    /// Run sparse conditional constant propagation over a renamed SSA form.
    ///
    /// `ssa` must have been computed from `cfg`. The control-flow adapter
    /// exposes branch predicates through [`ConstantFolder::fold_branch`]: a
    /// block whose last instruction decides its condition activates only the
    /// matching [`EdgeKind::ConditionalTrue`] or
    /// [`EdgeKind::ConditionalFalse`] successor edge. An outgoing edge of any
    /// other kind stays executable, because the predicate says nothing about
    /// it. A block with no decided predicate activates every successor.
    ///
    /// Edges are only ever added, so the result stays monotone: a condition
    /// that lowers to `Bottom` on a later pass activates the arm that was
    /// withheld, and `executable_edges` then names both arms.
    #[must_use]
    pub fn compute<I, E>(cfg: &Cfg<I, E>, ssa: &SsaForm<V>) -> Self
    where
        I: ConstantFolder<Const = C> + InstrInfo<Variable = V>,
    {
        let mut values = BTreeMap::new();
        let mut executable_edges = BTreeSet::new();
        let mut reachable_blocks = BTreeSet::new();
        let mut cfg_worklist = Vec::new();
        let mut ssa_worklist = Vec::new();

        reachable_blocks.insert(cfg.entry());
        evaluate_block(cfg, ssa, cfg.entry(), &mut values, &mut ssa_worklist);
        activate_successors(
            cfg,
            ssa,
            cfg.entry(),
            &values,
            &executable_edges,
            &mut cfg_worklist,
        );

        while !cfg_worklist.is_empty() || !ssa_worklist.is_empty() {
            while let Some((source, target)) = cfg_worklist.pop() {
                if !executable_edges.insert((source, target)) {
                    continue;
                }

                let newly_reachable = reachable_blocks.insert(target);
                evaluate_phis::<I>(
                    ssa,
                    target,
                    &executable_edges,
                    &mut values,
                    &mut ssa_worklist,
                );

                if newly_reachable {
                    evaluate_block(cfg, ssa, target, &mut values, &mut ssa_worklist);
                    activate_successors(
                        cfg,
                        ssa,
                        target,
                        &values,
                        &executable_edges,
                        &mut cfg_worklist,
                    );
                }
            }

            // Re-evaluate once per batch of lowered values. The value identity
            // is not yet used to target a consumer, so K entries must not
            // trigger K identical whole-program scans. Changes found during a
            // scan form the next batch, which still handles phis whose operands
            // lower after all incoming edges activated. Each scan also
            // re-derives branch edges, so a predicate that lowers here
            // activates the arm its earlier constant had withheld.
            while !ssa_worklist.is_empty() {
                ssa_worklist.clear();
                for &block in &reachable_blocks {
                    evaluate_phis::<I>(
                        ssa,
                        block,
                        &executable_edges,
                        &mut values,
                        &mut ssa_worklist,
                    );
                    evaluate_block(cfg, ssa, block, &mut values, &mut ssa_worklist);
                    activate_successors(
                        cfg,
                        ssa,
                        block,
                        &values,
                        &executable_edges,
                        &mut cfg_worklist,
                    );
                }
            }
        }

        SccpAnalysis {
            values,
            executable_edges,
            reachable_blocks,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dataflow::constant_propagation::constant_propagation;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::{DfInst, df_const, df_def, df_pred, df_use};

    /// A test constant domain whose meet is not equality: two exact values
    /// of the same sign keep that sign instead of dropping to bottom.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Signed {
        /// One exact value.
        Exact(i64),
        /// Only the sign of a value, shared by both meet arguments.
        Sign(bool),
    }

    impl Signed {
        /// The sign the value carries, which the meet keeps.
        const fn sign(self) -> bool {
            match self {
                Self::Exact(value) => value < 0,
                Self::Sign(negative) => negative,
            }
        }
    }

    /// [`DfInst`] read under the [`Signed`] domain. The wrapper keeps the
    /// shared mock's uses, defs, and constant loads while replacing the
    /// constant domain, so the solver meets through a real hook.
    #[derive(Debug, Clone)]
    struct SignedInst(DfInst);

    impl InstrInfo for SignedInst {
        type Variable = u16;

        fn uses(&self) -> &[u16] {
            self.0.uses()
        }

        fn defs(&self) -> &[u16] {
            self.0.defs()
        }
    }

    impl ConstantFolder for SignedInst {
        type Const = Signed;

        fn fold_constant(&self, _known: &BTreeMap<u16, Signed>) -> Option<(u16, Signed)> {
            let value = self.0.constant?;
            Some((*self.0.defs().first()?, Signed::Exact(value)))
        }

        fn meet_constants(a: &Signed, b: &Signed) -> Option<Signed> {
            if a == b {
                return Some(*a);
            }
            (a.sign() == b.sign()).then(|| Signed::Sign(a.sign()))
        }
    }

    /// Two arms that load different positive constants into variable 0 and
    /// join. The join is the only place a meet of two constants happens.
    fn signed_merge() -> (Cfg<SignedInst>, BlockId) {
        let mut cfg = Cfg::<SignedInst>::new();
        let arm_a = cfg.new_block();
        let arm_b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .push(SignedInst(df_def("branch", 9)));
        cfg.block_mut(arm_a).push(SignedInst(df_const("x5", 0, 5)));
        cfg.block_mut(arm_b).push(SignedInst(df_const("x7", 0, 7)));
        cfg.block_mut(merge).push(SignedInst(df_use("use_x", 0)));
        cfg.add_edge(cfg.entry(), arm_a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), arm_b, EdgeKind::ConditionalFalse);
        cfg.add_edge(arm_a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(arm_b, merge, EdgeKind::Fallthrough);
        (cfg, merge)
    }

    #[test]
    fn a_phi_of_two_constants_keeps_the_part_the_domain_meet_agrees_on() {
        let (cfg, merge) = signed_merge();
        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        let result = SccpAnalysis::compute(&cfg, &ssa);

        let phi = &ssa.block(merge).phis[0];
        assert_eq!(
            result.values.get(&phi.result),
            Some(&ConstValue::Const(Signed::Sign(false))),
            "the hook keeps the shared sign where equality would reach bottom"
        );
    }

    #[test]
    fn the_dense_solver_meets_through_the_same_hook() {
        let (cfg, merge) = signed_merge();
        let result = constant_propagation(&cfg);

        assert_eq!(
            result.fact_in(merge).get(&0),
            Some(&ConstValue::Const(Signed::Sign(false))),
            "the dense meet must agree with the sparse one"
        );
    }

    #[test]
    fn the_monotonicity_guard_rejects_a_constant_above_the_previous_value() {
        // A partial constant carries less information, so it sits below the
        // exact value it agrees with.
        let exact = ConstValue::Const(Signed::Exact(5));
        let sign = ConstValue::Const(Signed::Sign(false));

        assert!(is_below::<SignedInst>(&sign, &exact));
        assert!(
            !is_below::<SignedInst>(&exact, &sign),
            "a hook that answered the exact value here would raise the lattice value"
        );
        assert!(is_below::<SignedInst>(&ConstValue::Bottom, &exact));
        assert!(!is_below::<SignedInst>(&ConstValue::Top, &exact));
    }

    fn analyze(cfg: &Cfg<DfInst>) -> SccpAnalysis<u16, i64> {
        let dom = DominatorTree::compute(cfg);
        let ssa = SsaForm::compute(cfg, &dom);
        SccpAnalysis::compute(cfg, &ssa)
    }

    #[test]
    fn entry_is_reachable() {
        let cfg = Cfg::<DfInst>::new();
        assert!(analyze(&cfg).reachable_blocks.contains(&cfg.entry()));
    }

    #[test]
    fn linear_cfg_is_reachable() {
        let mut cfg = Cfg::<DfInst>::new();
        let next = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("def", 0));
        cfg.block_mut(next).push(df_use("use", 0));
        cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);
        assert!(analyze(&cfg).reachable_blocks.contains(&next));
    }

    #[test]
    fn constants_are_keyed_by_ssa_value() {
        let mut cfg = Cfg::<DfInst>::new();
        cfg.block_mut(cfg.entry()).push(df_const("constant", 0, 42));
        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        let definition = ssa.block(cfg.entry()).instructions[0].defs[0].clone();
        let result = SccpAnalysis::compute(&cfg, &ssa);
        assert_eq!(result.values[&definition], ConstValue::Const(42));
    }

    #[test]
    fn unreachable_block_is_excluded() {
        let mut cfg = Cfg::<DfInst>::new();
        let reachable = cfg.new_block();
        let unreachable = cfg.new_block();
        cfg.add_edge(cfg.entry(), reachable, EdgeKind::Fallthrough);
        let result = analyze(&cfg);
        assert!(result.reachable_blocks.contains(&reachable));
        assert!(!result.reachable_blocks.contains(&unreachable));
    }

    /// Builds a two-armed branch whose entry block loads `condition` into
    /// variable 0 and then branches on it. Returns the taken and not-taken
    /// arms.
    fn constant_branch(cfg: &mut Cfg<DfInst>, condition: i64) -> (BlockId, BlockId) {
        let taken = cfg.new_block();
        let not_taken = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .push(df_const("condition", 0, condition));
        cfg.block_mut(cfg.entry()).push(df_pred("branch", 0, true));
        cfg.add_edge(cfg.entry(), taken, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), not_taken, EdgeKind::ConditionalFalse);
        (taken, not_taken)
    }

    #[test]
    fn a_constant_true_branch_activates_only_the_true_edge() {
        let mut cfg = Cfg::<DfInst>::new();
        let (taken, not_taken) = constant_branch(&mut cfg, 1);
        let result = analyze(&cfg);

        assert!(result.executable_edges.contains(&(cfg.entry(), taken)));
        assert!(!result.executable_edges.contains(&(cfg.entry(), not_taken)));
        assert!(result.reachable_blocks.contains(&taken));
        assert!(
            !result.reachable_blocks.contains(&not_taken),
            "the proven predicate leaves the false arm unreachable"
        );
    }

    #[test]
    fn a_constant_false_branch_activates_only_the_false_edge() {
        let mut cfg = Cfg::<DfInst>::new();
        let (taken, not_taken) = constant_branch(&mut cfg, 0);
        let result = analyze(&cfg);

        assert!(result.executable_edges.contains(&(cfg.entry(), not_taken)));
        assert!(!result.executable_edges.contains(&(cfg.entry(), taken)));
        assert!(result.reachable_blocks.contains(&not_taken));
        assert!(
            !result.reachable_blocks.contains(&taken),
            "the proven predicate leaves the true arm unreachable"
        );
    }

    #[test]
    fn a_decided_branch_keeps_its_non_conditional_edges() {
        // The predicate proves the true arm, but it says nothing about an
        // unwind out of the same block, so that edge stays executable.
        let mut cfg = Cfg::<DfInst>::new();
        let (taken, not_taken) = constant_branch(&mut cfg, 1);
        let handler = cfg.new_block();
        cfg.add_edge(cfg.entry(), handler, EdgeKind::ExceptionUnwind);
        let result = analyze(&cfg);

        assert!(result.executable_edges.contains(&(cfg.entry(), handler)));
        assert!(result.reachable_blocks.contains(&handler));
        assert!(!result.reachable_blocks.contains(&not_taken));
        assert!(result.reachable_blocks.contains(&taken));
    }

    #[test]
    fn a_branch_on_a_live_in_activates_both_edges() {
        // Variable 0 is never defined, so it is an unknowable live-in and
        // the predicate stays undecided.
        let mut cfg = Cfg::<DfInst>::new();
        let taken = cfg.new_block();
        let not_taken = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_pred("branch", 0, true));
        cfg.add_edge(cfg.entry(), taken, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), not_taken, EdgeKind::ConditionalFalse);
        let result = analyze(&cfg);

        assert!(result.executable_edges.contains(&(cfg.entry(), taken)));
        assert!(result.executable_edges.contains(&(cfg.entry(), not_taken)));
    }

    #[test]
    fn a_branch_whose_phi_lowers_late_activates_both_edges() {
        // The header branches on a phi that is Const(1) until the back edge
        // activates. Withholding the false arm must not be permanent: once
        // the loop body lowers the phi, re-evaluation activates the exit.
        let mut cfg = Cfg::<DfInst>::new();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_const("x1", 0, 1));
        cfg.block_mut(header).push(df_pred("branch", 0, true));
        cfg.block_mut(body).push(DfInst {
            defs: alloc::vec![0],
            uses: alloc::vec![0, 1],
            ..crate::test_util::df_ff("x_varies")
        });
        cfg.block_mut(exit).push(df_use("after", 0));
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Back);
        let result = analyze(&cfg);

        assert!(result.executable_edges.contains(&(header, body)));
        assert!(
            result.executable_edges.contains(&(header, exit)),
            "a predicate that lowers to Bottom must activate the withheld arm"
        );
        assert!(result.reachable_blocks.contains(&exit));
    }

    #[test]
    fn live_in_phi_operand_is_not_folded_to_one_arms_constant() {
        // entry branches: arm A defines x = 5; arm B leaves the live-in
        // untouched. The merge phi meets Const(5) with the UNKNOWN live-in
        // (version 0) — which must be Bottom, never an optimistic Top that
        // would fold a runtime-varying input into 5.
        let mut cfg = Cfg::<DfInst>::new();
        let arm_a = cfg.new_block();
        let arm_b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("branch", 9));
        cfg.block_mut(arm_a).push(df_const("x5", 0, 5));
        cfg.block_mut(arm_b).push(df_use("noop", 9));
        cfg.block_mut(merge).push(df_use("use_x", 0));
        cfg.add_edge(cfg.entry(), arm_a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), arm_b, EdgeKind::ConditionalFalse);
        cfg.add_edge(arm_a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(arm_b, merge, EdgeKind::Fallthrough);

        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        let result = SccpAnalysis::compute(&cfg, &ssa);
        let phi = &ssa.block(merge).phis[0];
        assert_eq!(
            result.values.get(&phi.result),
            Some(&ConstValue::Bottom),
            "live-in arm makes the phi unknowable"
        );
    }

    #[test]
    fn phi_re_evaluates_when_an_operand_lowers_late() {
        // A loop phi over x: initial arm gives Const(1); the loop body
        // redefines x non-constantly. The body's lowering happens AFTER
        // the back edge is already executable, so only a drain-loop phi
        // re-evaluation can lower the phi from its stale Const(1).
        let mut cfg = Cfg::<DfInst>::new();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_const("x1", 0, 1));
        cfg.block_mut(header).push(df_use("test_x", 0));
        cfg.block_mut(body).push(DfInst {
            defs: alloc::vec![0],
            uses: alloc::vec![0, 1],
            ..crate::test_util::df_ff("x_varies")
        });
        cfg.block_mut(exit).push(df_use("after", 0));
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Back);

        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        let result = SccpAnalysis::compute(&cfg, &ssa);
        let phi = &ssa.block(header).phis[0];
        assert_eq!(
            result.values.get(&phi.result),
            Some(&ConstValue::Bottom),
            "loop-carried redefinition must lower the header phi"
        );
    }
}
