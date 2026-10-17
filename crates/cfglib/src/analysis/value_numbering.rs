//! Value numbering — local (LVN) and global (GVN).
//!
//! Identifies redundant computations by assigning the same "value number"
//! to expressions that compute identical results.
//!
//! Both flavors apply one rule to each instruction. A pure instruction that
//! defines something is keyed by its operator and the value numbers of its
//! operands; it takes the value number of an equal key already in scope, or
//! a fresh one. Every other instruction gives each of its definitions a
//! fresh value number. LVN numbers one block from empty tables, and GVN walks
//! the dominator tree and scopes the tables to it.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec::Vec;

use smallvec::SmallVec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::unwind::{first_unwind_point, unwind_reach};
use crate::graph::dominator::DominatorTree;
use crate::{InstrInfo, VariableId};

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

/// The tables the numbering rule reads and writes.
struct NumberingTables<V, Op> {
    variable_values: BTreeMap<V, ValueNumber>,
    expr_to_vn: BTreeMap<ExprKey<Op>, ValueNumber>,
    next_vn: ValueNumber,
}

/// How [`NumberingTables::number_instruction`] numbered one instruction.
struct Numbered<Op> {
    /// The value number of its definitions, for a value-numbered instruction.
    value: Option<ValueNumber>,
    /// Whether an equal expression was already in scope.
    redundant: bool,
    /// The expression it made available, when it is the first of its kind in
    /// scope.
    introduced: Option<ExprKey<Op>>,
}

impl<V: VariableId, Op: Clone + Ord> NumberingTables<V, Op> {
    const fn new(next_vn: ValueNumber) -> Self {
        Self {
            variable_values: BTreeMap::new(),
            expr_to_vn: BTreeMap::new(),
            next_vn,
        }
    }

    fn fresh(&mut self) -> ValueNumber {
        let value = self.next_vn;
        self.next_vn += 1;
        value
    }

    /// Bind `variable` to `value`, telling `saving` what it held before.
    fn bind(
        &mut self,
        variable: &V,
        value: ValueNumber,
        saving: &mut impl FnMut(&V, Option<ValueNumber>),
    ) {
        let previous = self.variable_values.insert(variable.clone(), value);
        saving(variable, previous);
    }

    /// Give `variable` a value number no expression has used.
    fn redefine(&mut self, variable: &V, saving: &mut impl FnMut(&V, Option<ValueNumber>)) {
        let value = self.fresh();
        self.bind(variable, value, saving);
    }

    /// The value number `variable` currently holds. A variable never seen
    /// holds an unknown value, which gets a number of its own.
    fn value_of(
        &mut self,
        variable: &V,
        saving: &mut impl FnMut(&V, Option<ValueNumber>),
    ) -> ValueNumber {
        if let Some(&value) = self.variable_values.get(variable) {
            return value;
        }
        let value = self.fresh();
        self.bind(variable, value, saving);
        value
    }

    /// The numbering rule. `saving` hears every variable binding the
    /// instruction changes, with the previous value number, so a scoped
    /// caller can undo it.
    fn number_instruction<I>(
        &mut self,
        inst: &I,
        saving: &mut impl FnMut(&V, Option<ValueNumber>),
    ) -> Numbered<Op>
    where
        I: ValueNumberInfo<Variable = V, Operator = Op>,
    {
        if !inst.is_pure() || inst.defs().is_empty() {
            // A skipped instruction still REDEFINES its defs: give each a
            // fresh value number so later expressions over them are not
            // falsely matched against pre-redefinition keys.
            for variable in inst.defs() {
                self.redefine(variable, saving);
            }
            return Numbered {
                value: None,
                redundant: false,
                introduced: None,
            };
        }

        let operands: SmallVec<[ValueNumber; 4]> = inst
            .uses()
            .iter()
            .map(|variable| self.value_of(variable, saving))
            .collect();
        let key = ExprKey {
            operation: inst.operator(),
            operands,
        };

        let (value, redundant, introduced) = if let Some(&existing) = self.expr_to_vn.get(&key) {
            (existing, true, None)
        } else {
            let value = self.fresh();
            self.expr_to_vn.insert(key.clone(), value);
            (value, false, Some(key))
        };
        for variable in inst.defs() {
            self.bind(variable, value, saving);
        }
        Numbered {
            value: Some(value),
            redundant,
            introduced,
        }
    }
}

impl BlockValueNumbers {
    /// Run local value numbering on a single block.
    ///
    /// Returns the block's numbering together with the next unassigned value
    /// number, so successive blocks can thread a shared counter.
    #[must_use]
    pub fn compute<G>(cfg: &G, block: BlockId, start_vn: ValueNumber) -> (Self, ValueNumber)
    where
        G: CfgView,
        G::Inst: ValueNumberInfo,
    {
        let mut tables = NumberingTables::new(start_vn);
        let insts = cfg.instructions(block);
        let mut inst_vn = Vec::with_capacity(insts.len());
        let mut redundant = Vec::new();
        for (idx, inst) in insts.iter().enumerate() {
            let numbered = tables.number_instruction(inst, &mut |_, _| {});
            inst_vn.push(numbered.value);
            if numbered.redundant {
                redundant.push(idx);
            }
        }
        (BlockValueNumbers { inst_vn, redundant }, tables.next_vn)
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
    ///
    /// A block starts from the tables its immediate dominator ended with,
    /// which hold only what every path into the block agrees on. A variable
    /// written by a block that lies between the two, such as a branch arm or
    /// a loop body that closes back on the block, may hold another value on
    /// some path in, so it receives a fresh value number in the block's
    /// scope: no expression over its earlier value is reused there.
    ///
    /// An [`ExceptionUnwind`](crate::EdgeKind::ExceptionUnwind) edge leaves
    /// its block before each instruction that
    /// [`may_unwind`](crate::InstrInfo::may_unwind). A dominator child that
    /// such an unwind reaches without passing through its parent again
    /// therefore sees neither the expressions its parent first computes at
    /// or after its first throwing instruction nor the values the parent
    /// writes there: those variables receive fresh value numbers in the
    /// child's scope. A block without such an unwind adds no work.
    #[must_use]
    pub fn compute<G>(cfg: &G, dom: &DominatorTree) -> Self
    where
        G: CfgView,
        G::Inst: ValueNumberInfo,
    {
        let mut state = (
            BTreeMap::new(),
            NumberingTables::new(0),
            BTreeMap::<BlockId, UnwindSuffix<_, _>>::new(),
        );
        let mut open = Vec::new();
        dom.scoped_walk(
            [cfg.root()],
            &mut state,
            &mut open,
            |(blocks, tables, suffixes), block| {
                let mut after_unwind = None;
                if let Some(parent) = dom.idom(block)
                    && let Some(suffix) = suffixes.get_mut(&parent)
                {
                    let reach = suffix
                        .reach
                        .get_or_insert_with(|| unwind_reach(cfg, parent));
                    if reach[block.index()] {
                        after_unwind = Some(&*suffix);
                    }
                }
                let entry = BlockEntry {
                    after_unwind,
                    redefined: side_path_definitions(cfg, dom, block),
                };
                let NumberedBlock { scope, suffix } =
                    number_block(cfg, block, &entry, tables, blocks);
                if let Some(suffix) = suffix {
                    suffixes.insert(block, suffix);
                }
                scope
            },
            |(_, tables, suffixes), _, scope| {
                suffixes.remove(&scope.block);
                exit_scope(scope, tables);
            },
        );
        let (blocks, tables, _) = state;

        ValueNumbering {
            blocks,
            value_count: tables.next_vn,
        }
    }
}

/// What a block's numbering must forget before its instructions run.
struct BlockEntry<'a, V, Op> {
    /// The parent's suffix, when an unwind out of the parent reaches the
    /// block.
    after_unwind: Option<&'a UnwindSuffix<V, Op>>,
    /// Variables that a side path into the block can rewrite.
    redefined: Vec<V>,
}

/// What a block does from its first unwind point on, which a dominator
/// child entered after an unwind out of the block must not see.
struct UnwindSuffix<V, Op> {
    /// Variables written at or after the first throwing instruction.
    variables: Vec<V>,
    /// Expressions first made available at or after it.
    expressions: Vec<ExprKey<Op>>,
    /// Per block, whether an unwind out of the block reaches it without
    /// re-entering the block; computed when a child first asks.
    reach: Option<Vec<bool>>,
}

struct GvnScope<V, Op> {
    block: BlockId,
    saved_variables: BTreeMap<V, Option<ValueNumber>>,
    expressions: Vec<ExprKey<Op>>,
    /// Parent expressions hidden from this scope, restored on exit.
    hidden: Vec<(ExprKey<Op>, ValueNumber)>,
}

struct NumberedBlock<V, Op> {
    scope: GvnScope<V, Op>,
    suffix: Option<UnwindSuffix<V, Op>>,
}

/// The variables written by blocks that lie on a path from `block`'s
/// immediate dominator to `block` without passing through that dominator
/// again.
///
/// Those blocks are dominated by the immediate dominator but do not dominate
/// `block`, so the dominator walk never sees their writes: a branch arm that
/// rejoins at `block`, or, when `block` heads a loop, the loop body.
fn side_path_definitions<G>(
    cfg: &G,
    dom: &DominatorTree,
    block: BlockId,
) -> Vec<<G::Inst as InstrInfo>::Variable>
where
    G: CfgView,
    G::Inst: InstrInfo,
{
    let Some(parent) = dom.idom(block) else {
        return Vec::new();
    };
    // Walk backward from `block`, never through `parent`. Allocation-free
    // when `parent` is the only predecessor, the common case.
    let mut between = BTreeSet::new();
    let mut pending = Vec::new();
    let mut node = block;
    loop {
        for predecessor in cfg.predecessors(node) {
            if predecessor != parent && dom.is_reachable(predecessor) && between.insert(predecessor)
            {
                pending.push(predecessor);
            }
        }
        let Some(next) = pending.pop() else {
            break;
        };
        node = next;
    }

    let mut variables: Vec<_> = between
        .iter()
        .flat_map(|&node| cfg.instructions(node))
        .flat_map(|inst| inst.defs().iter().cloned())
        .collect();
    variables.sort_unstable();
    variables.dedup();
    variables
}

/// Number one block and return the mutations to undo when its scope exits,
/// with what the block does from its first unwind point on when an unwind
/// can leave it.
///
/// `entry` states what the block must forget first: the parent's suffix when
/// an unwind out of the parent reaches this block, and the variables a side
/// path into it can rewrite.
fn number_block<G, I>(
    cfg: &G,
    bid: BlockId,
    entry: &BlockEntry<'_, I::Variable, I::Operator>,
    tables: &mut NumberingTables<I::Variable, I::Operator>,
    blocks: &mut BTreeMap<BlockId, BlockValueNumbers>,
) -> NumberedBlock<I::Variable, I::Operator>
where
    G: CfgView<Inst = I>,
    I: ValueNumberInfo,
{
    // Snapshot the current scope so we can restore on exit.
    let mut saved_variables: BTreeMap<I::Variable, Option<ValueNumber>> = BTreeMap::new();
    let mut saving = |variable: &I::Variable, previous: Option<ValueNumber>| {
        saved_variables.entry(variable.clone()).or_insert(previous);
    };
    let mut expr_added: Vec<ExprKey<I::Operator>> = Vec::new();
    let mut hidden = Vec::new();
    if let Some(suffix) = entry.after_unwind {
        for variable in &suffix.variables {
            tables.redefine(variable, &mut saving);
        }
        for key in &suffix.expressions {
            if let Some(vn) = tables.expr_to_vn.remove(key) {
                hidden.push((key.clone(), vn));
            }
        }
    }
    for variable in &entry.redefined {
        tables.redefine(variable, &mut saving);
    }

    let first_unwind = first_unwind_point(cfg, bid);
    let mut suffix = first_unwind.map(|_| UnwindSuffix {
        variables: Vec::new(),
        expressions: Vec::new(),
        reach: None,
    });

    // Process instructions in this block.
    let insts = cfg.instructions(bid);
    let mut inst_vn = Vec::with_capacity(insts.len());
    let mut redundant = Vec::new();

    for (idx, inst) in insts.iter().enumerate() {
        // From the first throwing instruction on, an unwind can leave
        // before this instruction's writes and expressions.
        let unwinding = first_unwind.is_some_and(|first| idx >= first);
        if unwinding && let Some(unwound) = &mut suffix {
            unwound.variables.extend(inst.defs().iter().cloned());
        }
        let numbered = tables.number_instruction(inst, &mut saving);
        inst_vn.push(numbered.value);
        if numbered.redundant {
            redundant.push(idx);
        }
        if let Some(key) = numbered.introduced {
            if unwinding && let Some(unwound) = &mut suffix {
                unwound.expressions.push(key.clone());
            }
            expr_added.push(key);
        }
    }

    if let Some(suffix) = &mut suffix {
        suffix.variables.sort();
        suffix.variables.dedup();
    }
    blocks.insert(bid, BlockValueNumbers { inst_vn, redundant });
    let scope = GvnScope {
        block: bid,
        saved_variables,
        expressions: expr_added,
        hidden,
    };
    NumberedBlock { scope, suffix }
}

fn exit_scope<V: Ord, Op: Ord>(scope: GvnScope<V, Op>, tables: &mut NumberingTables<V, Op>) {
    for key in scope.expressions {
        tables.expr_to_vn.remove(&key);
    }
    for (variable, previous) in scope.saved_variables {
        if let Some(value_number) = previous {
            tables.variable_values.insert(variable, value_number);
        } else {
            tables.variable_values.remove(&variable);
        }
    }
    for (key, value_number) in scope.hidden {
        tables.expr_to_vn.insert(key, value_number);
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
    use crate::test_util::{VnInst, protected_region, vn_impure, vn_inst, vn_throwing};

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
    fn local_and_global_numbering_agree_on_a_single_block() {
        let mut cfg: Cfg<VnInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            vn_inst(1, &[0, 1], &[2]),
            vn_inst(2, &[2, 3], &[4]),
            vn_impure(99, &[], &[2]),
            vn_inst(2, &[2, 3], &[5]),
            vn_inst(1, &[0, 1], &[6]),
            vn_inst(1, &[0, 1], &[0]),
            vn_inst(1, &[0, 1], &[7]),
        ]);
        let dominators = DominatorTree::compute(&cfg);

        let (local, next) = BlockValueNumbers::compute(&cfg, cfg.entry(), 0);
        let global = ValueNumbering::compute(&cfg, &dominators);

        let global_block = &global.blocks[&cfg.entry()];
        assert_eq!(local.inst_vn, global_block.inst_vn);
        assert_eq!(local.redundant, global_block.redundant);
        assert_eq!(next, global.value_count);
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

    /// A protected region (see [`protected_region`]) with `protected`'s
    /// instructions, and the same `successor_instructions` in `normal` and
    /// `handler`, numbered. Returns `(numbering, protected, normal, handler)`.
    fn number_protected_region(
        protected_instructions: impl IntoIterator<Item = VnInst>,
        successor_instructions: &[VnInst],
    ) -> (ValueNumbering, BlockId, BlockId, BlockId) {
        let (mut cfg, [protected, normal, handler]) = protected_region(protected_instructions);
        for block in [normal, handler] {
            cfg.block_mut(block)
                .instructions_mut()
                .extend(successor_instructions.iter().cloned());
        }
        let dominators = DominatorTree::compute(&cfg);
        (
            ValueNumbering::compute(&cfg, &dominators),
            protected,
            normal,
            handler,
        )
    }

    #[test]
    fn a_handler_reuses_only_expressions_computed_before_the_throw() {
        let (numbering, _, normal, handler) = number_protected_region(
            [
                vn_inst(1, &[0, 1], &[10]),
                vn_throwing(99, &[], &[]),
                vn_inst(2, &[0, 1], &[11]),
            ],
            &[vn_inst(1, &[0, 1], &[12]), vn_inst(2, &[0, 1], &[13])],
        );

        assert_eq!(numbering.blocks[&normal].redundant, [0, 1]);
        assert_eq!(
            numbering.blocks[&handler].redundant,
            [0],
            "the second expression is computed after the call can unwind"
        );
    }

    #[test]
    fn a_handler_does_not_reuse_values_a_later_throw_observes_differently() {
        // protected: x = op1(a); call; x = op2(b); y = op4(x); call
        // The first call unwinds with x = op1(a) and the second with
        // x = op2(b), so op4(x) in the handler matches neither.
        let (numbering, protected, normal, handler) = number_protected_region(
            [
                vn_inst(1, &[0], &[5]),
                vn_throwing(99, &[], &[]),
                vn_inst(2, &[1], &[5]),
                vn_inst(4, &[5], &[6]),
                vn_throwing(99, &[], &[]),
            ],
            &[vn_inst(4, &[5], &[7])],
        );

        let computed = numbering.blocks[&protected].inst_vn[3];
        assert_eq!(numbering.blocks[&normal].redundant, [0]);
        assert_eq!(numbering.blocks[&normal].inst_vn[0], computed);
        assert_eq!(numbering.blocks[&handler].redundant, [] as [usize; 0]);
        assert_ne!(numbering.blocks[&handler].inst_vn[0], computed);
    }

    /// `entry` computes `r10 = op1(r0, r2)` and branches either through
    /// `arm` or straight to `join`, which recomputes `op1(r0, r2)`.
    fn number_around_an_arm(
        arm_instructions: impl IntoIterator<Item = VnInst>,
    ) -> (ValueNumbering, BlockId) {
        let mut cfg: Cfg<VnInst> = Cfg::new();
        let arm = cfg.new_block();
        let join = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .extend([vn_inst(1, &[0, 2], &[10]), vn_inst(2, &[0, 1], &[11])]);
        cfg.block_mut(arm)
            .instructions_mut()
            .extend(arm_instructions);
        cfg.block_mut(join)
            .instructions_mut()
            .push(vn_inst(1, &[0, 2], &[12]));
        cfg.add_edge(cfg.entry(), arm, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), join, EdgeKind::ConditionalFalse);
        cfg.add_edge(arm, join, EdgeKind::Fallthrough);
        let dominators = DominatorTree::compute(&cfg);
        (ValueNumbering::compute(&cfg, &dominators), join)
    }

    #[test]
    fn an_expression_is_not_reused_where_a_side_path_rewrites_its_operand() {
        for rewrite in [vn_inst(3, &[], &[2]), vn_impure(9, &[], &[2])] {
            let (numbering, join) = number_around_an_arm([rewrite]);
            assert_eq!(
                numbering.blocks[&join].redundant,
                [] as [usize; 0],
                "the arm overwrites r2 on one path into the join"
            );
        }
    }

    #[test]
    fn an_expression_is_reused_across_a_join_no_path_invalidates() {
        let (numbering, join) = number_around_an_arm([vn_inst(3, &[], &[5])]);
        assert_eq!(numbering.blocks[&join].redundant, [0]);
    }

    #[test]
    fn a_loop_header_does_not_reuse_an_expression_the_loop_invalidates() {
        // entry: t = op1(x, y); header: u = op1(x, y); body: x = op9()
        // closes the loop; exit: v = op1(x, y).
        let mut cfg: Cfg<VnInst> = Cfg::new();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(vn_inst(1, &[0, 1], &[10]));
        cfg.block_mut(header)
            .instructions_mut()
            .push(vn_inst(1, &[0, 1], &[11]));
        cfg.block_mut(body)
            .instructions_mut()
            .push(vn_impure(9, &[], &[0]));
        cfg.block_mut(exit)
            .instructions_mut()
            .push(vn_inst(1, &[0, 1], &[12]));
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Back);
        let dominators = DominatorTree::compute(&cfg);

        let numbering = ValueNumbering::compute(&cfg, &dominators);

        assert_eq!(
            numbering.blocks[&header].redundant,
            [] as [usize; 0],
            "x changes from one iteration to the next"
        );
        assert_eq!(
            numbering.blocks[&exit].redundant,
            [0],
            "the exit follows the header's own computation directly"
        );
    }
}
