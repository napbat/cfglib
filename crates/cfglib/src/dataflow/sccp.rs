//! Sparse Conditional Constant Propagation (SCCP).
//!
//! SCCP operates on the generic renamed values in [`SsaForm`] while asking
//! the source instruction adapter to fold native instructions.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::VariableId;
use crate::dataflow::constprop::{ConstValue, ConstantFolder};
use crate::dataflow::ssa::{SsaForm, SsaValue};

/// Result of SCCP analysis.
#[derive(Debug, Clone)]
pub struct SccpResult<V, C> {
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

fn update_value<V: VariableId, C: Clone + Eq>(
    values: &mut BTreeMap<SsaValue<V>, ConstValue<C>>,
    worklist: &mut Vec<SsaValue<V>>,
    value: &SsaValue<V>,
    candidate: ConstValue<C>,
) {
    let previous = lattice_of(values, value);
    let next = previous.clone().meet(candidate);
    if next != previous {
        values.insert(value.clone(), next);
        worklist.push(value.clone());
    }
}

fn evaluate_block<I: ConstantFolder>(
    cfg: &Cfg<I>,
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
        let known: BTreeMap<I::Variable, I::Const> = annotation
            .uses
            .iter()
            .filter_map(|value| {
                values
                    .get(value)
                    .and_then(ConstValue::as_const)
                    .map(|constant| (value.variable.clone(), constant.clone()))
            })
            .collect();

        if let Some((variable, constant)) = instruction.fold_constant(&known) {
            // The folder answers for ONE def; every co-defined variable of a
            // multi-def instruction was still redefined and must bottom.
            for definition in &annotation.defs {
                if definition.variable == variable {
                    update_value(
                        values,
                        worklist,
                        definition,
                        ConstValue::Const(constant.clone()),
                    );
                } else {
                    update_value(values, worklist, definition, ConstValue::Bottom);
                }
            }
        } else {
            for definition in &annotation.defs {
                update_value(values, worklist, definition, ConstValue::Bottom);
            }
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
                candidate = candidate.meet(lattice_of(values, operand));
            }
        }
        update_value(values, worklist, &phi.result, candidate);
    }
}

/// Run sparse conditional constant propagation over a renamed SSA form.
///
/// `ssa` must have been built from `cfg`. The current control-flow adapter
/// exposes reachability but not branch predicates, so SCCP conservatively marks
/// every successor of a reachable block executable.
#[must_use]
pub fn sccp<I: ConstantFolder>(
    cfg: &Cfg<I>,
    ssa: &SsaForm<I::Variable>,
) -> SccpResult<I::Variable, I::Const> {
    let mut values = BTreeMap::new();
    let mut executable_edges = BTreeSet::new();
    let mut reachable_blocks = BTreeSet::new();
    let mut cfg_worklist = Vec::new();
    let mut ssa_worklist = Vec::new();

    reachable_blocks.insert(cfg.entry());
    cfg_worklist.extend(
        cfg.successors(cfg.entry())
            .map(|target| (cfg.entry(), target)),
    );
    evaluate_block(cfg, ssa, cfg.entry(), &mut values, &mut ssa_worklist);

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
                cfg_worklist.extend(cfg.successors(target).map(|next| (target, next)));
            }
        }

        // Re-evaluate once per batch of lowered values. The value identity is
        // not yet used to target a consumer, so popping K entries and doing K
        // identical whole-program scans only multiplies work. Changes found
        // during a scan form the next batch, which still handles phis whose
        // operands lower after all incoming edges activated.
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
            }
        }
    }

    SccpResult {
        values,
        executable_edges,
        reachable_blocks,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataflow::ssa::build_ssa;
    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::{DfInst, df_const, df_def, df_use};

    fn analyse(cfg: &Cfg<DfInst>) -> SccpResult<u16, i64> {
        let dom = DominatorTree::compute(cfg);
        let ssa = build_ssa(cfg, &dom);
        sccp(cfg, &ssa)
    }

    #[test]
    fn entry_is_reachable() {
        let cfg = Cfg::<DfInst>::new();
        assert!(analyse(&cfg).reachable_blocks.contains(&cfg.entry()));
    }

    #[test]
    fn linear_cfg_is_reachable() {
        let mut cfg = Cfg::<DfInst>::new();
        let next = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("def", 0));
        cfg.block_mut(next).push(df_use("use", 0));
        cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);
        assert!(analyse(&cfg).reachable_blocks.contains(&next));
    }

    #[test]
    fn constants_are_keyed_by_ssa_value() {
        let mut cfg = Cfg::<DfInst>::new();
        cfg.block_mut(cfg.entry()).push(df_const("constant", 0, 42));
        let dom = DominatorTree::compute(&cfg);
        let ssa = build_ssa(&cfg, &dom);
        let definition = ssa.block(cfg.entry()).instructions[0].defs[0].clone();
        let result = sccp(&cfg, &ssa);
        assert_eq!(result.values[&definition], ConstValue::Const(42));
    }

    #[test]
    fn unreachable_block_is_excluded() {
        let mut cfg = Cfg::<DfInst>::new();
        let reachable = cfg.new_block();
        let unreachable = cfg.new_block();
        cfg.add_edge(cfg.entry(), reachable, EdgeKind::Fallthrough);
        let result = analyse(&cfg);
        assert!(result.reachable_blocks.contains(&reachable));
        assert!(!result.reachable_blocks.contains(&unreachable));
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
        let ssa = build_ssa(&cfg, &dom);
        let result = sccp(&cfg, &ssa);
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
        let ssa = build_ssa(&cfg, &dom);
        let result = sccp(&cfg, &ssa);
        let phi = &ssa.block(header).phis[0];
        assert_eq!(
            result.values.get(&phi.result),
            Some(&ConstValue::Bottom),
            "loop-carried redefinition must lower the header phi"
        );
    }
}
