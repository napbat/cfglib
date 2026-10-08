//! Value numbering — local (LVN) and global (GVN).
//!
//! Identifies redundant computations by assigning the same "value number"
//! to expressions that compute identical results.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use smallvec::SmallVec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::InstrInfo;
use crate::graph::dominator::{DominatorChildOrder, DominatorTree};

/// A value number — opaque identifier for a computed value.
pub type ValueNumber = u32;

/// An expression key used for hash-consing, over a consumer operator
/// identity `Op`.
///
/// Uses `SmallVec` to avoid heap allocation for expressions with ≤ 4
/// operands (the vast majority of real instructions).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct ExprKey<Op> {
    /// The operator applied (raw opcode, mnemonic enum, interned symbol).
    operation: Op,
    /// Value numbers of the operands.
    operands: SmallVec<[ValueNumber; 4]>,
}

/// Result of value numbering for one block.
#[derive(Debug, Clone)]
pub struct BlockValueNumbers {
    /// Value number assigned to each instruction's def (if any).
    /// Indexed by instruction index within the block.
    pub inst_vn: Vec<Option<ValueNumber>>,
    /// Instructions that are redundant (their value was already computed).
    pub redundant: Vec<usize>,
}

/// Result of value numbering for the whole CFG.
#[derive(Debug, Clone)]
pub struct ValueNumbering {
    /// Per-block results.
    pub blocks: BTreeMap<BlockId, BlockValueNumbers>,
    /// Total value numbers assigned.
    pub value_count: u32,
}

/// Trait for instructions to provide an operation identity for value
/// numbering.
pub trait ValueNumberInfo: InstrInfo {
    /// Operation identity for hash-consing. Two pure instructions with equal
    /// operation and operand value numbers compute the same value. `Ord`
    /// because expression keys live in a `BTreeMap`.
    type Operator: Clone + Ord;

    /// The operation this instruction performs.
    fn operator(&self) -> Self::Operator;

    /// Whether this instruction is pure (no side effects).
    /// Only pure instructions can be value-numbered.
    fn is_pure(&self) -> bool;
}

impl BlockValueNumbers {
    /// Run local value numbering on a single block.
    ///
    /// Returns the block's numbering together with the next unassigned value
    /// number, so successive blocks can thread a shared counter.
    #[must_use]
    pub fn compute<I: ValueNumberInfo>(
        cfg: &Cfg<I>,
        block: BlockId,
        start_vn: ValueNumber,
    ) -> (Self, ValueNumber) {
        let mut next_vn = start_vn;
        let mut variable_values: BTreeMap<I::Variable, ValueNumber> = BTreeMap::new();
        let mut expr_to_vn: BTreeMap<ExprKey<I::Operator>, ValueNumber> = BTreeMap::new();
        let insts = cfg.block(block).instructions();
        let mut inst_vn = Vec::with_capacity(insts.len());
        let mut redundant = Vec::new();

        for (idx, inst) in insts.iter().enumerate() {
            if !inst.is_pure() || inst.defs().is_empty() {
                // A skipped instruction still REDEFINES its defs: give each a
                // fresh value number so later expressions over them are not
                // falsely matched against pre-redefinition keys.
                for variable in inst.defs() {
                    let vn = next_vn;
                    next_vn += 1;
                    variable_values.insert(variable.clone(), vn);
                }
                inst_vn.push(None);
                continue;
            }

            // Build expression key from operand value numbers.
            let operands: SmallVec<[ValueNumber; 4]> = inst
                .uses()
                .iter()
                .map(|variable| {
                    *variable_values.entry(variable.clone()).or_insert_with(|| {
                        let vn = next_vn;
                        next_vn += 1;
                        vn
                    })
                })
                .collect();

            let key = ExprKey {
                operation: inst.operator(),
                operands,
            };

            if let Some(&existing_vn) = expr_to_vn.get(&key) {
                // Redundant — same expression already computed.
                inst_vn.push(Some(existing_vn));
                redundant.push(idx);
                for variable in inst.defs() {
                    variable_values.insert(variable.clone(), existing_vn);
                }
            } else {
                let vn = next_vn;
                next_vn += 1;
                expr_to_vn.insert(key, vn);
                inst_vn.push(Some(vn));
                for variable in inst.defs() {
                    variable_values.insert(variable.clone(), vn);
                }
            }
        }

        (BlockValueNumbers { inst_vn, redundant }, next_vn)
    }
}

impl ValueNumbering {
    /// Run global value numbering over the dominator tree.
    ///
    /// Performs a single DFS walk over the dominator tree, maintaining
    /// scoped `loc → VN` and `expr → VN` tables that are pushed on
    /// entry and popped on exit. This avoids cloning maps for every
    /// block and runs in O(n · α) time per instruction (where α is the
    /// `BTreeMap` operation cost).
    #[must_use]
    pub fn compute<I: ValueNumberInfo>(cfg: &Cfg<I>, dom: &DominatorTree) -> Self {
        let mut blocks = BTreeMap::new();
        let mut variable_values: BTreeMap<I::Variable, ValueNumber> = BTreeMap::new();
        let mut expr_to_vn: BTreeMap<ExprKey<I::Operator>, ValueNumber> = BTreeMap::new();
        let mut next_vn: ValueNumber = 0;
        let children = dom.child_links(DominatorChildOrder::Ascending);
        let mut events = vec![GvnEvent::Enter(cfg.entry())];

        while let Some(event) = events.pop() {
            match event {
                GvnEvent::Enter(block) => {
                    let scope = number_block(
                        cfg,
                        block,
                        &mut variable_values,
                        &mut expr_to_vn,
                        &mut next_vn,
                        &mut blocks,
                    );
                    events.push(GvnEvent::Exit(scope));
                    if let Some(child) = children.first_child(block) {
                        events.push(GvnEvent::Sibling(child));
                    }
                }
                GvnEvent::Sibling(block) => {
                    if let Some(sibling) = children.next_sibling(block) {
                        events.push(GvnEvent::Sibling(sibling));
                    }
                    events.push(GvnEvent::Enter(block));
                }
                GvnEvent::Exit(scope) => {
                    exit_scope(scope, &mut variable_values, &mut expr_to_vn);
                }
            }
        }

        ValueNumbering {
            blocks,
            value_count: next_vn,
        }
    }
}

struct GvnScope<V, Op> {
    saved_variables: BTreeMap<V, Option<ValueNumber>>,
    expressions: Vec<ExprKey<Op>>,
}

enum GvnEvent<V, Op> {
    Enter(BlockId),
    Sibling(BlockId),
    Exit(GvnScope<V, Op>),
}

/// Number one block and return the mutations to undo when its scope exits.
fn number_block<I: ValueNumberInfo>(
    cfg: &Cfg<I>,
    bid: BlockId,
    variable_values: &mut BTreeMap<I::Variable, ValueNumber>,
    expr_to_vn: &mut BTreeMap<ExprKey<I::Operator>, ValueNumber>,
    next_vn: &mut ValueNumber,
    blocks: &mut BTreeMap<BlockId, BlockValueNumbers>,
) -> GvnScope<I::Variable, I::Operator> {
    // Snapshot the current scope so we can restore on exit.
    let mut saved_variables: BTreeMap<I::Variable, Option<ValueNumber>> = BTreeMap::new();
    let mut expr_added: Vec<ExprKey<I::Operator>> = Vec::new();

    // Process instructions in this block.
    let insts = cfg.block(bid).instructions();
    let mut inst_vn = Vec::with_capacity(insts.len());
    let mut redundant = Vec::new();

    for (idx, inst) in insts.iter().enumerate() {
        if !inst.is_pure() || inst.defs().is_empty() {
            // A skipped instruction still REDEFINES its defs: give each a
            // fresh value number (scoped, restored on exit) so later
            // expressions over them are not falsely matched against
            // pre-redefinition keys.
            for variable in inst.defs() {
                saved_variables
                    .entry(variable.clone())
                    .or_insert_with(|| variable_values.get(variable).copied());
                let vn = *next_vn;
                *next_vn += 1;
                variable_values.insert(variable.clone(), vn);
            }
            inst_vn.push(None);
            continue;
        }

        let operands: SmallVec<[ValueNumber; 4]> = inst
            .uses()
            .iter()
            .map(|variable| {
                if let Some(&vn) = variable_values.get(variable) {
                    vn
                } else {
                    let vn = *next_vn;
                    *next_vn += 1;
                    saved_variables.insert(variable.clone(), None);
                    variable_values.insert(variable.clone(), vn);
                    vn
                }
            })
            .collect();

        let key = ExprKey {
            operation: inst.operator(),
            operands,
        };

        if let Some(&existing_vn) = expr_to_vn.get(&key) {
            inst_vn.push(Some(existing_vn));
            redundant.push(idx);
            for variable in inst.defs() {
                saved_variables
                    .entry(variable.clone())
                    .or_insert_with(|| variable_values.get(variable).copied());
                variable_values.insert(variable.clone(), existing_vn);
            }
        } else {
            let vn = *next_vn;
            *next_vn += 1;
            expr_added.push(key.clone());
            expr_to_vn.insert(key, vn);
            inst_vn.push(Some(vn));
            for variable in inst.defs() {
                saved_variables
                    .entry(variable.clone())
                    .or_insert_with(|| variable_values.get(variable).copied());
                variable_values.insert(variable.clone(), vn);
            }
        }
    }

    blocks.insert(bid, BlockValueNumbers { inst_vn, redundant });
    GvnScope {
        saved_variables,
        expressions: expr_added,
    }
}

fn exit_scope<V: Ord, Op: Ord>(
    scope: GvnScope<V, Op>,
    variable_values: &mut BTreeMap<V, ValueNumber>,
    expr_to_vn: &mut BTreeMap<ExprKey<Op>, ValueNumber>,
) {
    for key in scope.expressions {
        expr_to_vn.remove(&key);
    }
    for (variable, previous) in scope.saved_variables {
        if let Some(value_number) = previous {
            variable_values.insert(variable, value_number);
        } else {
            variable_values.remove(&variable);
        }
    }
}

impl ValueNumbering {
    /// Count total redundant instructions across all blocks.
    #[must_use]
    pub fn redundant_count(&self) -> usize {
        self.blocks.values().map(|b| b.redundant.len()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::{VnInst, vn_impure, vn_inst};

    #[test]
    fn impure_redefinition_invalidates_value_numbers() {
        // t = add(a, b); z = add2(t, c); t = load (impure, skipped);
        // y = add2(t, c) — y must NOT match z's key: t was redefined.
        let mut cfg: Cfg<VnInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            vn_inst(1, &[0, 1], &[10]),
            vn_inst(2, &[10, 2], &[11]),
            vn_impure(99, &[], &[10]),
            vn_inst(2, &[10, 2], &[12]),
        ]);

        let (numbers, _) = BlockValueNumbers::compute(&cfg, cfg.entry(), 0);
        assert!(
            numbers.redundant.is_empty(),
            "y reads the RELOADED t and is not redundant: {numbers:?}"
        );
    }

    #[test]
    fn lvn_detects_redundant() {
        // t0 = add(a, b), t1 = add(a, b) → t1 is redundant
        let mut cfg: Cfg<VnInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            vn_inst(1, &[0, 1], &[2]), // t2 = op1(loc0, loc1)
            vn_inst(1, &[0, 1], &[3]), // t3 = op1(loc0, loc1) → redundant
        ]);
        let (bvn, _) = BlockValueNumbers::compute(&cfg, cfg.entry(), 0);
        assert_eq!(bvn.redundant.len(), 1);
        assert_eq!(bvn.redundant[0], 1);
    }

    #[test]
    fn lvn_different_ops_not_redundant() {
        let mut cfg: Cfg<VnInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            vn_inst(1, &[0, 1], &[2]),
            vn_inst(2, &[0, 1], &[3]), // different opcode
        ]);
        let (bvn, _) = BlockValueNumbers::compute(&cfg, cfg.entry(), 0);
        assert_eq!(bvn.redundant, [] as [usize; 0]);
    }

    #[test]
    fn gvn_detects_cross_block_redundancy() {
        // Block 0: t2 = op1(loc0, loc1)
        // Block 1: t3 = op1(loc0, loc1)  ← redundant (same expr, dominator has it)
        let mut cfg: Cfg<VnInst> = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(vn_inst(1, &[0, 1], &[2]));
        cfg.block_mut(b)
            .instructions_mut()
            .push(vn_inst(1, &[0, 1], &[3]));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let dom = DominatorTree::compute(&cfg);
        let vn = ValueNumbering::compute(&cfg, &dom);
        // The instruction in block b should be marked redundant.
        let b_vn = &vn.blocks[&b];
        assert_eq!(
            b_vn.redundant.len(),
            1,
            "cross-block redundancy not detected"
        );
        assert_eq!(b_vn.redundant[0], 0);
        // Both instructions should share the same value number.
        let entry_vn = vn.blocks[&cfg.entry()].inst_vn[0].unwrap();
        let b_inst_vn = b_vn.inst_vn[0].unwrap();
        assert_eq!(entry_vn, b_inst_vn);
    }

    #[test]
    fn gvn_no_cross_block_without_dominance() {
        // Diamond: entry → A, entry → B. Same expr in A and B.
        // Neither dominates the other, so no redundancy.
        let mut cfg: Cfg<VnInst> = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        cfg.block_mut(a)
            .instructions_mut()
            .push(vn_inst(1, &[0, 1], &[2]));
        cfg.block_mut(b)
            .instructions_mut()
            .push(vn_inst(1, &[0, 1], &[3]));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        let dom = DominatorTree::compute(&cfg);
        let vn = ValueNumbering::compute(&cfg, &dom);
        assert_eq!(vn.blocks[&a].redundant, [] as [usize; 0]);
        assert_eq!(vn.blocks[&b].redundant, [] as [usize; 0]);
    }

    #[test]
    fn gvn_handles_a_deep_dominator_tree_iteratively() {
        const BLOCK_COUNT: usize = 4_096;

        let mut cfg: Cfg<VnInst> = Cfg::new();
        let mut block = cfg.entry();
        for index in 0..BLOCK_COUNT {
            let operation = u32::try_from(index).expect("test block index fits in u32");
            cfg.block_mut(block)
                .instructions_mut()
                .push(vn_inst(operation, &[], &[0]));
            if index + 1 < BLOCK_COUNT {
                let next = cfg.new_block();
                cfg.add_edge(block, next, EdgeKind::Fallthrough);
                block = next;
            }
        }

        let dominators = DominatorTree::compute(&cfg);
        let numbering = ValueNumbering::compute(&cfg, &dominators);

        assert_eq!(numbering.blocks.len(), BLOCK_COUNT);
        assert_eq!(
            numbering.value_count,
            ValueNumber::try_from(BLOCK_COUNT).expect("test block count fits in a value number")
        );
    }
}
