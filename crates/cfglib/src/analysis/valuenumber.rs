//! Value numbering — local (LVN) and global (GVN).
//!
//! Identifies redundant computations by assigning the same "value number"
//! to expressions that compute identical results.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use smallvec::SmallVec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::InstrInfo;
use crate::graph::dominator::{DominatorChildLinks, DominatorChildOrder, DominatorTree};

/// A value number — opaque identifier for a computed value.
pub type ValueNumber = u32;

/// An expression key used for hash-consing, over a consumer operation
/// identity `Op`.
///
/// Uses `SmallVec` to avoid heap allocation for expressions with ≤ 4
/// operands (the vast majority of real instructions).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExprKey<Op> {
    /// The operation performed (raw opcode, mnemonic enum, interned symbol).
    pub operation: Op,
    /// Value numbers of the operands.
    pub operands: SmallVec<[ValueNumber; 4]>,
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
    pub num_values: u32,
}

/// Trait for instructions to provide an operation identity for value
/// numbering.
pub trait ValueNumberInfo: InstrInfo {
    /// Operation identity for hash-consing. Two pure instructions with equal
    /// operation and operand value numbers compute the same value. `Ord`
    /// because expression keys live in a `BTreeMap`.
    type Operation: Clone + Ord;

    /// The operation this instruction performs.
    fn operation(&self) -> Self::Operation;

    /// Whether this instruction is pure (no side effects).
    /// Only pure instructions can be value-numbered.
    fn is_pure(&self) -> bool;
}

/// Run local value numbering on a single block.
#[must_use]
pub fn local_value_numbering<I: ValueNumberInfo>(
    cfg: &Cfg<I>,
    block: BlockId,
    start_vn: ValueNumber,
) -> (BlockValueNumbers, ValueNumber) {
    let mut next_vn = start_vn;
    let mut variable_values: BTreeMap<I::Variable, ValueNumber> = BTreeMap::new();
    let mut expr_to_vn: BTreeMap<ExprKey<I::Operation>, ValueNumber> = BTreeMap::new();
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
            operation: inst.operation(),
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

/// Run global value numbering over the dominator tree.
///
/// Performs a single DFS walk over the dominator tree, maintaining
/// scoped `loc → VN` and `expr → VN` tables that are pushed on
/// entry and popped on exit. This avoids cloning maps for every
/// block and runs in O(n · α) time per instruction (where α is the
/// `BTreeMap` operation cost).
#[must_use]
pub fn global_value_numbering<I: ValueNumberInfo>(
    cfg: &Cfg<I>,
    dom: &DominatorTree,
) -> ValueNumbering {
    let mut blocks = BTreeMap::new();
    let mut variable_values: BTreeMap<I::Variable, ValueNumber> = BTreeMap::new();
    let mut expr_to_vn: BTreeMap<ExprKey<I::Operation>, ValueNumber> = BTreeMap::new();
    let mut next_vn: ValueNumber = 0;
    let children = dom.child_links(DominatorChildOrder::Ascending);

    gvn_dfs(
        cfg,
        &children,
        cfg.entry(),
        &mut variable_values,
        &mut expr_to_vn,
        &mut next_vn,
        &mut blocks,
    );

    ValueNumbering {
        blocks,
        num_values: next_vn,
    }
}

/// Recursive DFS over the dominator tree with push/pop scoping.
fn gvn_dfs<I: ValueNumberInfo>(
    cfg: &Cfg<I>,
    children: &DominatorChildLinks<BlockId>,
    bid: BlockId,
    variable_values: &mut BTreeMap<I::Variable, ValueNumber>,
    expr_to_vn: &mut BTreeMap<ExprKey<I::Operation>, ValueNumber>,
    next_vn: &mut ValueNumber,
    blocks: &mut BTreeMap<BlockId, BlockValueNumbers>,
) {
    // Snapshot the current scope so we can restore on exit.
    let mut saved_variables: BTreeMap<I::Variable, Option<ValueNumber>> = BTreeMap::new();
    let mut expr_added: Vec<ExprKey<I::Operation>> = Vec::new();

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
            operation: inst.operation(),
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

    // Recurse into dominator-tree children.
    let mut child = children.first_child(bid);
    while let Some(next) = child {
        gvn_dfs(
            cfg,
            children,
            next,
            variable_values,
            expr_to_vn,
            next_vn,
            blocks,
        );
        child = children.next_sibling(next);
    }

    // Pop scope: undo all insertions/overwrites from this block.
    for key in expr_added {
        expr_to_vn.remove(&key);
    }
    for (variable, previous) in saved_variables {
        if let Some(value_number) = previous {
            variable_values.insert(variable, value_number);
        } else {
            variable_values.remove(&variable);
        }
    }
}

/// Count total redundant instructions across all blocks.
#[must_use]
pub fn count_redundant(vn: &ValueNumbering) -> usize {
    vn.blocks.values().map(|b| b.redundant.len()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::flow::{FlowControl, FlowEffect};

    #[derive(Debug, Clone)]
    struct VnInst {
        op: u32,
        uses: Vec<u16>,
        defs: Vec<u16>,
        pure_: bool,
    }

    impl FlowControl for VnInst {
        fn flow_effect(&self) -> FlowEffect {
            FlowEffect::Fallthrough
        }
    }

    impl InstrInfo for VnInst {
        type Variable = u16;

        fn uses(&self) -> &[u16] {
            &self.uses
        }
        fn defs(&self) -> &[u16] {
            &self.defs
        }
    }

    impl ValueNumberInfo for VnInst {
        type Operation = u32;

        fn operation(&self) -> u32 {
            self.op
        }
        fn is_pure(&self) -> bool {
            self.pure_
        }
    }

    fn vn_inst(op: u32, uses: &[u16], defs: &[u16]) -> VnInst {
        VnInst {
            op,
            uses: uses.to_vec(),
            defs: defs.to_vec(),
            pure_: true,
        }
    }

    #[test]
    fn impure_redefinition_invalidates_value_numbers() {
        // t = add(a, b); z = add2(t, c); t = load (impure, skipped);
        // y = add2(t, c) — y must NOT match z's key: t was redefined.
        let mut cfg: Cfg<VnInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_vec_mut().extend([
            vn_inst(1, &[0, 1], &[10]),
            vn_inst(2, &[10, 2], &[11]),
            VnInst {
                op: 99,
                uses: alloc::vec![],
                defs: alloc::vec![10],
                pure_: false,
            },
            vn_inst(2, &[10, 2], &[12]),
        ]);

        let (numbers, _) = local_value_numbering(&cfg, cfg.entry(), 0);
        assert!(
            numbers.redundant.is_empty(),
            "y reads the RELOADED t and is not redundant: {numbers:?}"
        );
    }

    #[test]
    fn lvn_detects_redundant() {
        // t0 = add(a, b), t1 = add(a, b) → t1 is redundant
        let mut cfg: Cfg<VnInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_vec_mut().extend([
            vn_inst(1, &[0, 1], &[2]), // t2 = op1(loc0, loc1)
            vn_inst(1, &[0, 1], &[3]), // t3 = op1(loc0, loc1) → redundant
        ]);
        let (bvn, _) = local_value_numbering(&cfg, cfg.entry(), 0);
        assert_eq!(bvn.redundant.len(), 1);
        assert_eq!(bvn.redundant[0], 1);
    }

    #[test]
    fn lvn_different_ops_not_redundant() {
        let mut cfg: Cfg<VnInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_vec_mut().extend([
            vn_inst(1, &[0, 1], &[2]),
            vn_inst(2, &[0, 1], &[3]), // different opcode
        ]);
        let (bvn, _) = local_value_numbering(&cfg, cfg.entry(), 0);
        assert_eq!(bvn.redundant.len(), 0);
    }

    #[test]
    fn gvn_detects_cross_block_redundancy() {
        // Block 0: t2 = op1(loc0, loc1)
        // Block 1: t3 = op1(loc0, loc1)  ← redundant (same expr, dominator has it)
        let mut cfg: Cfg<VnInst> = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(vn_inst(1, &[0, 1], &[2]));
        cfg.block_mut(b)
            .instructions_vec_mut()
            .push(vn_inst(1, &[0, 1], &[3]));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let dom = DominatorTree::compute(&cfg);
        let vn = global_value_numbering(&cfg, &dom);
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
            .instructions_vec_mut()
            .push(vn_inst(1, &[0, 1], &[2]));
        cfg.block_mut(b)
            .instructions_vec_mut()
            .push(vn_inst(1, &[0, 1], &[3]));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        let dom = DominatorTree::compute(&cfg);
        let vn = global_value_numbering(&cfg, &dom);
        assert_eq!(vn.blocks[&a].redundant.len(), 0);
        assert_eq!(vn.blocks[&b].redundant.len(), 0);
    }
}
