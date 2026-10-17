//! Sparse Conditional Constant Propagation (SCCP).
//!
//! SCCP operates on the generic renamed values in [`SsaForm`] while asking
//! the source instruction adapter to fold native instructions.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::constant_propagation::{ConstValue, ConstantFolder, fold_definitions};
use crate::dataflow::ssa::{SsaForm, SsaInstruction, SsaValue};
use crate::edge::{EdgeId, EdgeKind, KindedEdge};
use crate::{InstrInfo, ProgramPoint, VariableId, departs_before_throws};

/// Result of SCCP analysis.
#[derive(Debug, Clone)]
pub struct SccpAnalysis<V, C> {
    /// Lattice value computed for each renamed SSA value.
    pub values: BTreeMap<SsaValue<V>, ConstValue<C>>,
    /// CFG edges proven executable, by edge identity.
    ///
    /// Edges that join the same pair of blocks are judged apart. A branch arm
    /// the predicate withheld stays out of the set while a parallel edge, or
    /// an unwind into the same block, is in it. A phi operand arrives along
    /// the executable edge it departs on;
    /// [`operand_is_executable`](Self::operand_is_executable) asks exactly
    /// that.
    pub executable_edges: BTreeSet<EdgeId>,
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

/// Meet `candidate` into the value of `value`, recording in `changed` that
/// the value moved down the lattice.
fn update_value<I: ConstantFolder>(
    values: &mut BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    changed: &mut bool,
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
        *changed = true;
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
        .uses()
        .iter()
        .filter_map(|value| {
            values
                .get(value)
                .and_then(ConstValue::as_const)
                .map(|constant| (value.variable.clone(), constant.clone()))
        })
        .collect()
}

fn evaluate_block<G, I>(
    cfg: &G,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    values: &mut BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    changed: &mut bool,
) where
    G: CfgView<Inst = I>,
    I: ConstantFolder,
{
    for (instruction, annotation) in cfg
        .instructions(block)
        .iter()
        .zip(&ssa.block(block).instructions)
    {
        let known = known_constants(annotation, values);
        // The renamer gives each source definition one SSA definition, in
        // the same order, so the two sequences zip.
        for ((_, value), definition) in fold_definitions(instruction, &known).zip(annotation.defs())
        {
            update_value::<I>(values, changed, definition, value);
        }
    }
}

/// The branch outcome the last instruction of `block` proves, or `None` when
/// the block does not end in a decided conditional transfer.
fn decided_branch<G, I>(
    cfg: &G,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    values: &BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
) -> Option<bool>
where
    G: CfgView<Inst = I>,
    I: ConstantFolder,
{
    let (instruction, annotation) = cfg
        .instructions(block)
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
fn activate_successors<G, I>(
    cfg: &G,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    values: &BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    executable: &BTreeSet<EdgeId>,
    worklist: &mut Vec<EdgeId>,
) where
    G: CfgView<Inst = I>,
    I: ConstantFolder,
{
    let withheld = match decided_branch(cfg, ssa, block, values) {
        Some(true) => Some(EdgeKind::ConditionalFalse),
        Some(false) => Some(EdgeKind::ConditionalTrue),
        None => None,
    };
    for id in cfg.outgoing(block) {
        let edge = cfg.edge(id);
        if Some(edge.data().kind()) != withheld && !executable.contains(&id) {
            worklist.push(id);
        }
    }
}

/// What the phi operands of one edge have in common: the block they leave
/// and whether they leave before that block's end.
///
/// An edge that [departs before throws](departs_before_throws) delivers the
/// state before each throwing instruction, so each of its operands is named
/// by an instruction index inside the block. Every other edge delivers one
/// operand, named by the block's instruction count. An operand and its edge
/// therefore agree on this pair, and two edges between the same blocks agree
/// on it only when they depart the same way.
type Departure = (BlockId, bool);

/// The [`Departure`] shared by the operands that `edge` delivers.
fn edge_departure<G>(cfg: &G, edge: EdgeId) -> Departure
where
    G: CfgView,
    G::Inst: InstrInfo,
{
    let edge = cfg.edge(edge);
    (edge.source(), departs_before_throws(cfg, edge))
}

/// The [`Departure`] of the edge that delivers the operand named by `point`.
fn operand_departure<G: CfgView>(cfg: &G, point: ProgramPoint) -> Departure {
    (
        point.block,
        point.inst_idx < cfg.instructions(point.block).len(),
    )
}

/// Re-meet every phi of `block` over the operands that arrive along its
/// currently executable incoming edges. Called when an edge activates AND
/// from the rescan after a value lowers: a phi whose operand lowers after
/// all its edges are already executable must be re-evaluated, or it keeps a
/// stale over-optimistic constant.
fn evaluate_phis<G, I>(
    cfg: &G,
    ssa: &SsaForm<I::Variable>,
    block: BlockId,
    executable: &BTreeSet<EdgeId>,
    values: &mut BTreeMap<SsaValue<I::Variable>, ConstValue<I::Const>>,
    changed: &mut bool,
) where
    G: CfgView<Inst = I>,
    I: ConstantFolder,
{
    let phis = &ssa.block(block).phis;
    if phis.is_empty() {
        return;
    }
    let arriving: BTreeSet<Departure> = cfg
        .incoming(block)
        .filter(|edge| executable.contains(edge))
        .map(|edge| edge_departure(cfg, edge))
        .collect();
    for phi in phis {
        let mut candidate = ConstValue::Top;
        for (departure, operand) in phi.operands() {
            if arriving.contains(&operand_departure(cfg, *departure)) {
                candidate = candidate.meet_with(lattice_of(values, operand), I::meet_constants);
            }
        }
        update_value::<I>(values, changed, &phi.result, candidate);
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
    ///
    /// Every live-in is unknown here. Use
    /// [`compute_seeded`](Self::compute_seeded) when the caller knows what
    /// one of them holds.
    #[must_use]
    pub fn compute<G>(cfg: &G, ssa: &SsaForm<V>) -> Self
    where
        G: CfgView,
        G::Inst: ConstantFolder<Const = C> + InstrInfo<Variable = V>,
    {
        Self::compute_seeded(cfg, ssa, &BTreeMap::new())
    }

    /// Run sparse conditional constant propagation with known live-ins.
    ///
    /// `live_ins` names the constant that one variable already holds when
    /// the function starts. The solve seeds each entry as the version-zero
    /// value of that variable, which is what [`SsaForm`] gives a read that
    /// no definition reaches. A live-in has no definition, so nothing in the
    /// solve can lower it: the seed stays for the whole analysis.
    ///
    /// A caller supplies a live-in it knows from outside the function. An
    /// argument the calling convention names is one such value. A hidden
    /// register an observation captured is another. A variable that
    /// `live_ins` does not name stays `Bottom`, as it is in
    /// [`compute`](Self::compute).
    ///
    /// `ssa` must have been computed from `cfg`. Every other rule of
    /// [`compute`](Self::compute) applies without change.
    #[must_use]
    pub fn compute_seeded<G>(cfg: &G, ssa: &SsaForm<V>, live_ins: &BTreeMap<V, C>) -> Self
    where
        G: CfgView,
        G::Inst: ConstantFolder<Const = C> + InstrInfo<Variable = V>,
    {
        let mut values = live_ins
            .iter()
            .map(|(variable, constant)| {
                (
                    SsaValue::live_in(variable.clone()),
                    ConstValue::Const(constant.clone()),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let mut executable_edges = BTreeSet::new();
        let mut reachable_blocks = BTreeSet::new();
        let mut cfg_worklist = Vec::new();
        let mut changed = false;

        reachable_blocks.insert(cfg.root());
        evaluate_block(cfg, ssa, cfg.root(), &mut values, &mut changed);
        activate_successors(
            cfg,
            ssa,
            cfg.root(),
            &values,
            &executable_edges,
            &mut cfg_worklist,
        );

        while !cfg_worklist.is_empty() || changed {
            while let Some(edge) = cfg_worklist.pop() {
                if !executable_edges.insert(edge) {
                    continue;
                }

                let target = cfg.edge(edge).target();
                let newly_reachable = reachable_blocks.insert(target);
                evaluate_phis(
                    cfg,
                    ssa,
                    target,
                    &executable_edges,
                    &mut values,
                    &mut changed,
                );

                if newly_reachable {
                    evaluate_block(cfg, ssa, target, &mut values, &mut changed);
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

            // Rescan every reachable block once per batch of lowered values.
            // The scan finds its dependents by re-evaluating each block, not
            // from a per-value users index, so it runs once per batch rather
            // than once per lowered value. Changes found during a scan form
            // the next batch, which still handles phis whose operands lower
            // after all incoming edges activated. Each scan also re-derives
            // branch edges, so a predicate that lowers here activates the
            // arm its earlier constant had withheld.
            while changed {
                changed = false;
                for &block in &reachable_blocks {
                    evaluate_phis(
                        cfg,
                        ssa,
                        block,
                        &executable_edges,
                        &mut values,
                        &mut changed,
                    );
                    evaluate_block(cfg, ssa, block, &mut values, &mut changed);
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

    /// Whether the phi operand that departs `departure` reaches `block` along
    /// an executable edge.
    ///
    /// `departure` is the [`ProgramPoint`] that
    /// [`SsaPhi::operands`](crate::SsaPhi::operands) names for the operand.
    /// Two edges can join the same blocks and still depart differently: an
    /// unwind carries the state before a throwing instruction, a branch arm
    /// the state at the end of its source. An operand arrives only along an
    /// executable edge of its own kind of departure, so a withheld arm does
    /// not feed a phi that an executable unwind into the same block reaches.
    /// The solve applies this test to every phi.
    #[must_use]
    pub fn operand_is_executable<G>(&self, cfg: &G, departure: ProgramPoint, block: BlockId) -> bool
    where
        G: CfgView,
        G::Inst: InstrInfo,
    {
        let wanted = operand_departure(cfg, departure);
        cfg.incoming(block).any(|edge| {
            self.executable_edges.contains(&edge) && edge_departure(cfg, edge) == wanted
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::cfg::Cfg;
    use crate::dataflow::constant_propagation::ConstantPropagation;
    use crate::flow::FlowEffect;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::{DfInst, df_const, df_def, df_ff, df_pred, df_use, df_with_effect};

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
        let result = ConstantPropagation::compute(&cfg);

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

    /// Whether the only edge from `source` to `target` is executable.
    fn is_executable(
        cfg: &Cfg<DfInst>,
        result: &SccpAnalysis<u16, i64>,
        source: BlockId,
        target: BlockId,
    ) -> bool {
        let mut edges = cfg
            .outgoing(source)
            .filter(|&edge| cfg.edge(edge).target() == target);
        let edge = edges.next().expect("the fixture connects the blocks");
        assert!(edges.next().is_none(), "the fixture connects them once");
        result.executable_edges.contains(&edge)
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
        let definition = ssa.block(cfg.entry()).instructions[0].defs()[0].clone();
        let result = SccpAnalysis::compute(&cfg, &ssa);
        assert_eq!(result.values[&definition], ConstValue::Const(42));
    }

    #[test]
    fn a_folded_multi_definition_bottoms_the_co_defined_values() {
        // The folder answers for the first definition only. The second one
        // was redefined too, so it cannot keep a constant.
        let mut cfg = Cfg::<DfInst>::new();
        cfg.block_mut(cfg.entry()).push(DfInst {
            defs: alloc::vec![0, 1],
            constant: Some(3),
            ..df_ff("pair")
        });
        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        let defs = ssa.block(cfg.entry()).instructions[0].defs();
        let result = SccpAnalysis::compute(&cfg, &ssa);

        assert_eq!(result.values[&defs[0]], ConstValue::Const(3));
        assert_eq!(result.values[&defs[1]], ConstValue::Bottom);
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

        assert!(is_executable(&cfg, &result, cfg.entry(), taken));
        assert!(!is_executable(&cfg, &result, cfg.entry(), not_taken));
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

        assert!(is_executable(&cfg, &result, cfg.entry(), not_taken));
        assert!(!is_executable(&cfg, &result, cfg.entry(), taken));
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

        assert!(is_executable(&cfg, &result, cfg.entry(), handler));
        assert!(result.reachable_blocks.contains(&handler));
        assert!(!result.reachable_blocks.contains(&not_taken));
        assert!(result.reachable_blocks.contains(&taken));
    }

    /// A block whose conditional arm and unwind both reach `target`.
    struct UnwindBesideArm {
        cfg: Cfg<DfInst>,
        target: BlockId,
        /// The `ConditionalTrue` edge into `target`.
        arm: EdgeId,
        /// The `ExceptionUnwind` edge into `target`.
        unwind: EdgeId,
    }

    /// `entry` holds `x = 5`, `condition` (variable 0), a throwing `x = 7`,
    /// and a branch on variable 0 that goes to `target` when true. An unwind
    /// out of `entry` reaches `target` as well, so the phi of `x` at `target`
    /// merges the 7 that the arm carries with the 5 that the unwind carries
    /// from before the throwing store.
    fn unwind_beside_an_arm(condition: DfInst) -> UnwindBesideArm {
        let mut cfg = Cfg::<DfInst>::new();
        let target = cfg.new_block();
        let other = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            df_const("x5", 1, 5),
            condition,
            df_with_effect(df_const("x7", 1, 7), FlowEffect::MayThrow),
            df_pred("branch", 0, true),
        ]);
        cfg.block_mut(target).push(df_use("use_x", 1));
        let arm = cfg.add_edge(cfg.entry(), target, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), other, EdgeKind::ConditionalFalse);
        let unwind = cfg.add_edge(cfg.entry(), target, EdgeKind::ExceptionUnwind);
        UnwindBesideArm {
            cfg,
            target,
            arm,
            unwind,
        }
    }

    #[test]
    fn a_withheld_arm_does_not_feed_a_phi_the_unwind_reaches() {
        // The predicate proves the false arm, so the true arm never runs.
        // The unwind into the same block carries the value before the
        // throwing store (5); the store's own 7 only travels along the
        // withheld arm and must not reach the phi.
        let fixture = unwind_beside_an_arm(df_const("condition", 0, 0));
        let cfg = &fixture.cfg;
        let dom = DominatorTree::compute(cfg);
        let ssa = SsaForm::compute(cfg, &dom);
        let result = SccpAnalysis::compute(cfg, &ssa);

        let phi = &ssa.block(fixture.target).phis[0];
        assert_eq!(phi.operands().len(), 2, "one operand per departure");
        assert_eq!(
            result.values.get(&phi.result),
            Some(&ConstValue::Const(5)),
            "only the unwind edge into the block is executable"
        );
    }

    #[test]
    fn a_withheld_arm_and_an_unwind_into_one_block_are_judged_apart() {
        let fixture = unwind_beside_an_arm(df_const("condition", 0, 0));
        let cfg = &fixture.cfg;
        let result = analyze(cfg);

        assert!(!result.executable_edges.contains(&fixture.arm));
        assert!(result.executable_edges.contains(&fixture.unwind));

        // The arm departs at the end of `entry`, the unwind before its
        // throwing store at index 2.
        let at_end = ProgramPoint {
            block: cfg.entry(),
            inst_idx: cfg.instructions(cfg.entry()).len(),
        };
        let before_store = ProgramPoint {
            block: cfg.entry(),
            inst_idx: 2,
        };
        assert!(!result.operand_is_executable(cfg, at_end, fixture.target));
        assert!(result.operand_is_executable(cfg, before_store, fixture.target));
    }

    #[test]
    fn an_undecided_arm_and_an_unwind_into_one_block_both_feed_the_phi() {
        // Nothing proves the condition, so both edges are executable and the
        // phi merges the 7 with the 5.
        let fixture = unwind_beside_an_arm(df_def("condition", 0));
        let cfg = &fixture.cfg;
        let dom = DominatorTree::compute(cfg);
        let ssa = SsaForm::compute(cfg, &dom);
        let result = SccpAnalysis::compute(cfg, &ssa);

        assert!(result.executable_edges.contains(&fixture.arm));
        assert!(result.executable_edges.contains(&fixture.unwind));
        let phi = &ssa.block(fixture.target).phis[0];
        assert_eq!(result.values.get(&phi.result), Some(&ConstValue::Bottom));
        for &(departure, _) in phi.operands() {
            assert!(result.operand_is_executable(cfg, departure, fixture.target));
        }
    }

    #[test]
    fn a_decided_branch_withholds_one_of_two_arms_into_the_same_block() {
        // Both arms reach `target`, but only the proven one is executable.
        let mut cfg = Cfg::<DfInst>::new();
        let target = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_const("condition", 0, 1));
        cfg.block_mut(cfg.entry()).push(df_pred("branch", 0, true));
        let when_true = cfg.add_edge(cfg.entry(), target, EdgeKind::ConditionalTrue);
        let when_false = cfg.add_edge(cfg.entry(), target, EdgeKind::ConditionalFalse);
        let result = analyze(&cfg);

        assert!(result.executable_edges.contains(&when_true));
        assert!(!result.executable_edges.contains(&when_false));
        assert!(result.reachable_blocks.contains(&target));
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

        assert!(is_executable(&cfg, &result, cfg.entry(), taken));
        assert!(is_executable(&cfg, &result, cfg.entry(), not_taken));
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
            ..df_ff("x_varies")
        });
        cfg.block_mut(exit).push(df_use("after", 0));
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Back);
        let result = analyze(&cfg);

        assert!(is_executable(&cfg, &result, header, body));
        assert!(
            is_executable(&cfg, &result, header, exit),
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
    fn a_seeded_live_in_decides_a_branch_on_it() {
        // Variable 0 is never defined, so the predicate is undecided
        // without a seed. The seed names what the caller knows enters the
        // function, and the true arm alone stays executable.
        let mut cfg = Cfg::<DfInst>::new();
        let taken = cfg.new_block();
        let not_taken = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_pred("branch", 0, true));
        cfg.add_edge(cfg.entry(), taken, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), not_taken, EdgeKind::ConditionalFalse);

        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        let seeds = BTreeMap::from([(0_u16, 1_i64)]);
        let result = SccpAnalysis::compute_seeded(&cfg, &ssa, &seeds);

        assert_eq!(
            result.values.get(&SsaValue::live_in(0)),
            Some(&ConstValue::Const(1))
        );
        assert!(is_executable(&cfg, &result, cfg.entry(), taken));
        assert!(!is_executable(&cfg, &result, cfg.entry(), not_taken));
        assert!(!result.reachable_blocks.contains(&not_taken));
    }

    #[test]
    fn a_phi_over_a_seeded_live_in_folds() {
        // Arm A defines x = 5 and arm B leaves the live-in untouched. The
        // seed states that the live-in is also 5, so the merge phi folds
        // where the unseeded solve reaches bottom.
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
        let phi = &ssa.block(merge).phis[0];
        let seeds = BTreeMap::from([(0_u16, 5_i64)]);
        let seeded = SccpAnalysis::compute_seeded(&cfg, &ssa, &seeds);

        assert_eq!(
            seeded.values.get(&phi.result),
            Some(&ConstValue::Const(5)),
            "both incoming values are the same constant"
        );
    }

    #[test]
    fn an_unseeded_live_in_still_bottoms() {
        // Variable 0 carries a seed and variable 1 does not. Seeding one
        // live-in must not make another one optimistic.
        let mut cfg = Cfg::<DfInst>::new();
        cfg.block_mut(cfg.entry()).push(df_use("read", 0));
        cfg.block_mut(cfg.entry()).push(df_pred("branch", 1, true));
        let taken = cfg.new_block();
        let not_taken = cfg.new_block();
        cfg.add_edge(cfg.entry(), taken, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), not_taken, EdgeKind::ConditionalFalse);

        let dom = DominatorTree::compute(&cfg);
        let ssa = SsaForm::compute(&cfg, &dom);
        let seeds = BTreeMap::from([(0_u16, 7_i64)]);
        let result = SccpAnalysis::compute_seeded(&cfg, &ssa, &seeds);

        assert_eq!(result.values.get(&SsaValue::live_in(1)), None);
        assert_eq!(
            lattice_of(&result.values, &SsaValue::live_in(1)),
            ConstValue::<i64>::Bottom
        );
        assert!(is_executable(&cfg, &result, cfg.entry(), taken));
        assert!(is_executable(&cfg, &result, cfg.entry(), not_taken));
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
            ..df_ff("x_varies")
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
