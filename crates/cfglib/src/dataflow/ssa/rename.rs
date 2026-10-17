//! Dominator-tree renaming: the walk that turns placed phis and source
//! instructions into versioned values.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::unwind::unwind_edges;
use crate::graph::dominator::DominatorTree;
use crate::{InstrInfo, ProgramPoint, VariableId};

use super::scratch::SsaScratch;
use super::{SsaBlock, SsaInstruction, SsaPhi, SsaValue, SsaVersion};

pub(super) fn current_value<V: VariableId>(
    variable: &V,
    stacks: &BTreeMap<V, Vec<SsaValue<V>>>,
) -> SsaValue<V> {
    stacks
        .get(variable)
        .and_then(|stack| stack.last())
        .cloned()
        .unwrap_or_else(|| SsaValue::live_in(variable.clone()))
}

pub(super) fn fresh_value<V: VariableId>(
    variable: &V,
    max_versions: &mut BTreeMap<V, SsaVersion>,
) -> SsaValue<V> {
    let version = max_versions.entry(variable.clone()).or_default();
    *version += 1;
    SsaValue::new(variable.clone(), *version)
}

/// Rename `block`: its phis, its instructions, and the operands it supplies to
/// its successors' phis.
///
/// Returns how many variables were pushed onto the shared stack, which is
/// what leaving the block's scope pops. Every definition a block pushes is
/// appended to one stack, so unwinding is popping a count and the walk
/// allocates no vector per block.
fn rename_block<G, V>(
    scratch: &mut SsaScratch<V>,
    cfg: &G,
    block: BlockId,
    max_versions: &mut BTreeMap<V, SsaVersion>,
) -> usize
where
    G: CfgView,
    G::Inst: InstrInfo<Variable = V>,
    V: VariableId,
{
    let SsaScratch {
        phis,
        instructions,
        stacks,
        value_pool,
        pushed,
        ..
    } = scratch;
    let pushed_before = pushed.len();

    for phi in &mut phis.by_block[block.index()] {
        let result = fresh_value(&phi.variable, max_versions);
        phi.result = Some(result.clone());
        push_value(stacks, value_pool, &phi.variable, result);
        pushed.push(phi.variable.clone());
    }

    let block_instructions = &mut instructions[block.index()];
    // The vector moves into the finished block, so it is sized to the block
    // rather than to the allocator's minimum growth step.
    block_instructions.reserve_exact(cfg.instructions(block).len());
    for (inst_idx, instruction) in cfg.instructions(block).iter().enumerate() {
        if instruction.may_unwind() {
            for edge in unwind_edges(cfg, block) {
                phis.set_operands(
                    edge.target(),
                    ProgramPoint { block, inst_idx },
                    |variable| current_value(variable, stacks),
                );
            }
        }
        let uses = instruction
            .uses()
            .iter()
            .map(|variable| current_value(variable, stacks))
            .collect();
        let mut defs = SmallVec::new();
        for variable in instruction.defs() {
            let value = fresh_value(variable, max_versions);
            push_value(stacks, value_pool, variable, value.clone());
            pushed.push(variable.clone());
            defs.push(value);
        }
        block_instructions.push(SsaInstruction {
            point: ProgramPoint { block, inst_idx },
            uses,
            defs,
        });
    }

    for successor in cfg.successors(block) {
        phis.set_operands(
            successor,
            ProgramPoint {
                block,
                inst_idx: cfg.instructions(block).len(),
            },
            |variable| current_value(variable, stacks),
        );
    }
    pushed.len() - pushed_before
}

/// Push `value` onto `variable`'s renaming stack, taking a recycled vector
/// when the variable is new to this procedure.
fn push_value<V: VariableId>(
    stacks: &mut BTreeMap<V, Vec<SsaValue<V>>>,
    pool: &mut Vec<Vec<SsaValue<V>>>,
    variable: &V,
    value: SsaValue<V>,
) {
    use alloc::collections::btree_map::Entry;
    let stack = match stacks.entry(variable.clone()) {
        Entry::Vacant(slot) => slot.insert(pool.pop().unwrap_or_default()),
        Entry::Occupied(slot) => slot.into_mut(),
    };
    stack.push(value);
}

/// Rename every block of the dominator forest, each exactly once.
///
/// `dom` has to be a forest that holds every block of `cfg`: the tree of the
/// entry, plus the trees `complete_dominator_forest` roots at the code the
/// entry misses. A block in no tree has no span to walk, so it would be
/// skipped without a word, and its instructions would reach the form
/// un-renamed; the forest is what rules that out, and it is asserted here.
pub(super) fn rename<G, V>(
    scratch: &mut SsaScratch<V>,
    cfg: &G,
    dom: &DominatorTree,
    max_versions: &mut BTreeMap<V, SsaVersion>,
) where
    G: CfgView,
    G::Inst: InstrInfo<Variable = V>,
    V: VariableId,
{
    debug_assert!(
        cfg.node_ids().all(|block| dom.is_reachable(block)),
        "the dominator forest leaves a block out of every tree"
    );
    // The entry's tree first, then every other tree of the forest in
    // ascending root order: the order fixes which block gets which version.
    let roots = core::iter::once(cfg.root()).chain(
        cfg.node_ids()
            .filter(|&block| block != cfg.root() && dom.idom(block).is_none()),
    );
    let mut open = core::mem::take(&mut scratch.open);
    dom.scoped_walk(
        roots,
        scratch,
        &mut open,
        |scratch, block| rename_block(scratch, cfg, block, max_versions),
        |scratch, _, pushed| {
            for _ in 0..pushed {
                let variable = scratch
                    .pushed
                    .pop()
                    .expect("an unwind pops only what its block pushed");
                if let Some(stack) = scratch.stacks.get_mut(&variable) {
                    stack.pop();
                }
            }
        },
    );
    scratch.open = open;
}

/// Turn the renamed drafts into the finished blocks, in block-identity order.
///
/// A phi whose block was never renamed has no result yet, and gets one here,
/// which is what keeps a version assigned to every phi in the form.
pub(super) fn finish_blocks<V: VariableId>(
    scratch: &mut SsaScratch<V>,
    block_bound: usize,
    max_versions: &mut BTreeMap<V, SsaVersion>,
) -> Vec<SsaBlock<V>> {
    (0..block_bound)
        .map(|index| {
            let block = BlockId::from_index(index);
            let phis = scratch.phis.by_block[index]
                .drain(..)
                .map(|draft| {
                    let result = draft
                        .result
                        .unwrap_or_else(|| fresh_value(&draft.variable, max_versions));
                    let operands = scratch.phis.predecessors[draft.operands.clone()]
                        .iter()
                        .zip(&scratch.phis.operands[draft.operands])
                        .map(|(&predecessor, value)| {
                            let value = value
                                .clone()
                                .unwrap_or_else(|| SsaValue::live_in(draft.variable.clone()));
                            (predecessor, value)
                        })
                        .collect();
                    SsaPhi { result, operands }
                })
                .collect();
            SsaBlock {
                block,
                phis,
                instructions: core::mem::take(&mut scratch.instructions[index]),
            }
        })
        .collect()
}
