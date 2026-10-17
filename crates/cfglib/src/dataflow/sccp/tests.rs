//! Sparse conditional constant propagation: lattice, edges, and phi operands.

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

/// The entry doubles as a loop header. It reads variable 1 (the loop's
/// carried value) and branches on variable 0, and its latch redefines
/// variable 1 as `latch_value`. Returns `(cfg, latch)`.
fn entry_loop(latch_value: i64, condition: Option<i64>) -> (Cfg<DfInst>, BlockId) {
    let mut cfg = Cfg::<DfInst>::new();
    let latch = cfg.new_block();
    let exit = cfg.new_block();
    let entry = cfg.entry();
    if let Some(condition) = condition {
        cfg.block_mut(entry)
            .push(df_const("condition", 0, condition));
    }
    cfg.block_mut(entry).push(df_use("read carried", 1));
    cfg.block_mut(entry).push(df_pred("branch", 0, true));
    cfg.block_mut(latch)
        .push(df_const("redefine", 1, latch_value));
    cfg.add_edge(entry, latch, EdgeKind::ConditionalTrue);
    cfg.add_edge(entry, exit, EdgeKind::ConditionalFalse);
    cfg.add_edge(latch, entry, EdgeKind::Back);
    (cfg, latch)
}

fn entry_phi_value(
    cfg: &Cfg<DfInst>,
    seeds: &BTreeMap<u16, i64>,
) -> (ConstValue<i64>, SccpAnalysis<u16, i64>) {
    let ssa = SsaForm::compute(cfg, &DominatorTree::compute(cfg));
    let result = SccpAnalysis::compute_seeded(cfg, &ssa, seeds);
    let phi = ssa
        .block(cfg.entry())
        .phis
        .iter()
        .find(|phi| phi.result.variable == 1)
        .expect("the entry merges the carried variable");
    assert_eq!(phi.start_value(), Some(SsaValue::live_in(1)));
    (result.values[&phi.result].clone(), result)
}

#[test]
fn an_entry_phi_meets_the_unknown_live_in_with_the_back_edge_constant() {
    // The first iteration reads whatever the caller passed in, so the loop
    // constant 7 must not be the phi's value.
    let (cfg, _) = entry_loop(7, None);
    let (value, result) = entry_phi_value(&cfg, &BTreeMap::new());

    assert_eq!(value, ConstValue::Bottom);
    assert!(
        result.reachable_blocks.contains(&BlockId::from_index(1)),
        "an undecided branch runs the latch"
    );
}

#[test]
fn an_entry_phi_over_a_seeded_start_value_folds_only_when_the_back_edge_agrees() {
    let seeds = BTreeMap::from([(1_u16, 7_i64)]);
    let (agreeing, _) = entry_loop(7, None);
    assert_eq!(entry_phi_value(&agreeing, &seeds).0, ConstValue::Const(7));

    let (disagreeing, _) = entry_loop(5, None);
    assert_eq!(entry_phi_value(&disagreeing, &seeds).0, ConstValue::Bottom);
}

#[test]
fn an_entry_phi_takes_the_seed_when_no_back_edge_ever_runs() {
    // The branch is decided false, so the latch and the back edge never run.
    // The start value is the phi's only input and needs no edge to arrive.
    let seeds = BTreeMap::from([(1_u16, 5_i64)]);
    let (cfg, latch) = entry_loop(7, Some(0));
    let (value, result) = entry_phi_value(&cfg, &seeds);

    assert_eq!(value, ConstValue::Const(5));
    assert!(
        !result.reachable_blocks.contains(&latch),
        "the decided branch leaves the latch, and so the back edge, unrun"
    );
}

#[test]
fn an_unseeded_entry_phi_is_bottom_even_when_no_back_edge_runs() {
    let (cfg, _) = entry_loop(7, Some(0));
    assert_eq!(
        entry_phi_value(&cfg, &BTreeMap::new()).0,
        ConstValue::Bottom
    );
}
