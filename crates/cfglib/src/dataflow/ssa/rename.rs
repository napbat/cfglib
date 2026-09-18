//! Dominator-tree renaming: the walk that turns placed phis and source
//! instructions into versioned values.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::{InstrInfo, ProgramPoint, VariableId};
use crate::graph::dominator::{DominatorChildOrder, DominatorTree};

use super::scratch::SsaScratch;
use super::{SsaBlock, SsaInstruction, SsaPhi, SsaValue, SsaVersion};

/// One step of the iterative dominator-tree walk.
///
/// The exit step carries a *count* rather than the variables themselves: every
/// definition pushed by a block is appended to one shared stack, so unwinding
/// is popping that many, and the walk allocates no vector per block.
#[derive(Debug)]
pub(super) enum RenameEvent {
    /// Rename this block, then unwind it.
    Enter(BlockId),
    /// Pop this many variables from the shared pushed-variable stack.
    Exit(usize),
}

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
/// Returns how many variables were pushed onto the shared stack, which is what
/// the matching [`RenameEvent::Exit`] unwinds.
fn rename_block<I: InstrInfo, E>(
    scratch: &mut SsaScratch<I::Variable>,
    cfg: &Cfg<I, E>,
    block: BlockId,
    max_versions: &mut BTreeMap<I::Variable, SsaVersion>,
) -> usize {
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
    block_instructions.reserve(cfg.block(block).instructions().len());
    for (inst_idx, instruction) in cfg.block(block).instructions().iter().enumerate() {
        let uses = instruction
            .uses()
            .iter()
            .map(|variable| current_value(variable, stacks))
            .collect();
        let mut defs = Vec::with_capacity(instruction.defs().len());
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
        phis.set_operands(successor, block, |variable| current_value(variable, stacks));
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

/// Rename every block reachable from a root of the dominator forest.
pub(super) fn rename<I: InstrInfo, E>(
    scratch: &mut SsaScratch<I::Variable>,
    cfg: &Cfg<I, E>,
    dom: &DominatorTree,
    max_versions: &mut BTreeMap<I::Variable, SsaVersion>,
) {
    // The event stack consumes siblings in reverse, so descending links
    // preserve `DominatorTree::children`'s ascending DFS visitation.
    dom.child_links_in(&mut scratch.children, DominatorChildOrder::Descending);
    scratch.roots.clear();
    scratch.roots.push(cfg.entry());
    scratch.roots.extend(
        cfg.block_ids()
            .filter(|&block| block != cfg.entry() && dom.idom(block).is_none()),
    );

    for index in 0..scratch.roots.len() {
        scratch.events.clear();
        scratch
            .events
            .push(RenameEvent::Enter(scratch.roots[index]));
        while let Some(event) = scratch.events.pop() {
            match event {
                RenameEvent::Enter(block) if !scratch.renamed.is_marked(block.index()) => {
                    scratch.renamed.mark(block.index());
                    let pushed = rename_block(scratch, cfg, block, max_versions);
                    scratch.events.push(RenameEvent::Exit(pushed));
                    let mut child = scratch.children.first_child(block);
                    while let Some(next) = child {
                        scratch.events.push(RenameEvent::Enter(next));
                        child = scratch.children.next_sibling(next);
                    }
                }
                RenameEvent::Enter(_) => {}
                RenameEvent::Exit(count) => {
                    for _ in 0..count {
                        let variable = scratch
                            .pushed
                            .pop()
                            .expect("an unwind pops only what its block pushed");
                        if let Some(stack) = scratch.stacks.get_mut(&variable) {
                            stack.pop();
                        }
                    }
                }
            }
        }
    }
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
