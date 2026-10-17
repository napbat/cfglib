//! The two transformations a consumer writes itself on top of cfglib's
//! analyses: folding a program with SCCP's verdicts, and lowering SSA to a
//! program over renamed values with the phi copies cfglib computes.

use std::collections::BTreeMap;

use cfglib::{
    BlockId, Cfg, ConstValue, EdgeKind, FlowEffect, SccpAnalysis, SemanticValidator, SsaForm,
    SsaValue, copies_by_predecessor, eliminate_phis,
};

use crate::interp::Inputs;
use crate::toy::{Inst, Op, Var};

/// Rewrite `cfg` with what SCCP proved: a single-definition instruction
/// whose value is constant becomes that constant, a branch with one
/// executable arm becomes a plain fallthrough, and a block no executable
/// edge reaches is removed.
pub fn fold_with_sccp(
    cfg: &Cfg<Inst<Var>>,
    ssa: &SsaForm<Var>,
    sccp: &SccpAnalysis<Var, i64>,
) -> Cfg<Inst<Var>> {
    let mut folded = cfg.clone();
    for block in cfg.block_ids() {
        if !sccp.reachable_blocks.contains(&block) {
            continue;
        }
        for (index, annotation) in ssa.block(block).instructions.iter().enumerate() {
            let [definition] = annotation.defs() else {
                continue;
            };
            let slot = &mut folded.block_mut(block).instructions_mut()[index];
            if matches!(slot.op, Op::Const(_)) {
                continue;
            }
            if let Some(ConstValue::Const(value)) = sccp.values.get(definition) {
                *slot = Inst::konst(definition.variable.0, *value).with_flow(slot.flow);
            }
        }
        fold_decided_branch(&mut folded, block, sccp);
    }
    for block in cfg.block_ids() {
        if block != cfg.entry() && !sccp.reachable_blocks.contains(&block) {
            folded.remove_block(block);
        }
    }
    folded
}

/// Replace the branch ending `block` by a fallthrough when exactly one of
/// its two arms is executable.
fn fold_decided_branch(cfg: &mut Cfg<Inst<Var>>, block: BlockId, sccp: &SccpAnalysis<Var, i64>) {
    let arms: Vec<_> = cfg
        .outgoing(block)
        .map(|edge| (edge, cfg.edge(edge).kind(), cfg.edge(edge).target()))
        .collect();
    let arm = |kind| {
        arms.iter()
            .find(|(_, arm_kind, _)| *arm_kind == kind)
            .copied()
    };
    let (Some(taken), Some(untaken)) = (
        arm(EdgeKind::ConditionalTrue),
        arm(EdgeKind::ConditionalFalse),
    ) else {
        return;
    };
    if taken.2 == untaken.2 {
        return;
    }
    let executable = |(edge, _, _): (_, _, BlockId)| sccp.executable_edges.contains(&edge);
    let (survivor, withheld) = match (executable(taken), executable(untaken)) {
        (true, false) => (taken, untaken),
        (false, true) => (untaken, taken),
        _ => return,
    };
    cfg.remove_edge(withheld.0);
    cfg.edge_mut(survivor.0).set_kind(EdgeKind::Fallthrough);
    let last = cfg
        .block_mut(block)
        .instructions_mut()
        .last_mut()
        .expect("a block with two conditional arms ends in its branch");
    assert_eq!(last.op, Op::Branch);
    *last = Inst::marker(FlowEffect::Fallthrough);
}

/// The program over SSA values that `cfg` and its form describe: every
/// instruction reads and writes renamed values, the phis are gone, and each
/// group of phi copies is one parallel copy at its departure point.
///
/// # Panics
///
/// Panics when a copy group would have to run after a branch: the caller
/// must split critical edges first, so that no predecessor of a phi block
/// chooses between successors.
pub fn destruct(cfg: &Cfg<Inst<Var>>, ssa: &SsaForm<Var>) -> Cfg<Inst<SsaValue<Var>>> {
    let mut lowered: Cfg<Inst<SsaValue<Var>>> = Cfg::new();
    let mut ids = vec![lowered.entry()];
    ids.extend((1..cfg.block_bound()).map(|_| lowered.new_block()));
    for (index, id) in ids.iter().enumerate() {
        if !cfg.block_ids().any(|block| block.index() == index) {
            lowered.remove_block(*id);
        }
    }
    for edge in cfg.edges() {
        lowered.add_edge(
            ids[edge.source().index()],
            ids[edge.target().index()],
            edge.kind(),
        );
    }
    for block in cfg.block_ids() {
        let renamed = cfg
            .block(block)
            .instructions()
            .iter()
            .zip(&ssa.block(block).instructions)
            .map(|(inst, annotation)| {
                inst.renamed(annotation.defs().to_vec(), annotation.uses().to_vec())
            });
        lowered
            .block_mut(ids[block.index()])
            .instructions_mut()
            .extend(renamed);
    }

    let copies = eliminate_phis(ssa);
    // Later departure points first, so an insertion never shifts one still
    // to come.
    for edge_copies in copies_by_predecessor(&copies).iter().rev() {
        let (point, group) = (edge_copies.from, &edge_copies.copies);
        let instructions = lowered
            .block_mut(ids[point.block.index()])
            .instructions_mut();
        assert!(
            point.inst_idx < instructions.len()
                || instructions.last().is_none_or(|last| last.op != Op::Branch),
            "copies at the end of {:?} would run after its branch",
            point.block
        );
        let destinations = group.iter().map(|copy| copy.destination.clone()).collect();
        let sources = group.iter().map(|copy| copy.source.clone()).collect();
        instructions.insert(point.inst_idx, Inst::parallel_copy(destinations, sources));
    }
    lowered
}

/// The inputs of a lowered program: each source variable's live-in value,
/// and zero for every other variable the program mentions, since a phi may
/// copy a live-in nothing ever reads.
pub fn lowered_inputs(cfg: &Cfg<Inst<Var>>, inputs: &Inputs) -> BTreeMap<SsaValue<Var>, i64> {
    let mut lowered = BTreeMap::new();
    for block in cfg.blocks() {
        for inst in block.instructions() {
            for variable in inst.defs.iter().chain(&inst.uses) {
                lowered.insert(
                    SsaValue::live_in(*variable),
                    inputs.get(variable).copied().unwrap_or(0),
                );
            }
        }
    }
    lowered
}

/// Consumer semantics for the toy language, checked by `verify_with`: a
/// block ending in a branch has exactly one true and one false successor, a
/// returning block has no normal successor, and any other block has at most
/// one and no conditional edge.
pub struct ToyShape;

impl<V> SemanticValidator<Inst<V>, ()> for ToyShape {
    type Error = String;

    fn validate_block(&self, cfg: &Cfg<Inst<V>>, block: BlockId, errors: &mut Vec<String>) {
        let mut kinds: Vec<EdgeKind> = cfg
            .outgoing(block)
            .map(|edge| cfg.edge(edge).kind())
            .filter(|kind| *kind != EdgeKind::ExceptionUnwind)
            .collect();
        kinds.sort_by_key(|kind| kind.name());
        let expected: &[EdgeKind] = match cfg.block(block).instructions().last().map(|inst| inst.op)
        {
            Some(Op::Branch) => &[EdgeKind::ConditionalFalse, EdgeKind::ConditionalTrue],
            Some(Op::Return) => &[],
            _ if kinds.len() <= 1
                && !kinds.iter().any(|kind| {
                    matches!(kind, EdgeKind::ConditionalTrue | EdgeKind::ConditionalFalse)
                }) =>
            {
                return;
            }
            _ => &[],
        };
        if kinds != expected {
            errors.push(format!(
                "{block:?} has normal edges {kinds:?}, expected {expected:?}"
            ));
        }
    }
}
