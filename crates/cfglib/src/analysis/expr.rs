//! Expression tree / DAG recovery.
//!
//! Reconstructs high-level expression trees from flat instruction
//! sequences. Given flat instructions like:
//!
//! ```text
//! mul t0, b, c
//! add t1, a, t0
//! ```
//!
//! This module recovers the expression tree `add(a, mul(b, c))`.
//!
//! The consumer implements [`ExprInstr`] to describe how each
//! instruction maps to an expression operator, and the framework
//! builds expression DAGs per block using def-use chains.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::{InstrInfo, VariableId};

/// A node in an expression tree over variables `V`, operators `Op`, and
/// constants `C` — all consumer-typed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprNode<V, Op, C> {
    /// A leaf source-IR variable.
    Leaf(V),
    /// An operation with a consumer operator identity and operand
    /// sub-expressions.
    Op {
        /// The operator (a mnemonic enum, an opcode, an interned symbol).
        operator: Op,
        /// The operands, each an expression sub-tree.
        operands: Vec<ExprNode<V, Op, C>>,
    },
    /// A constant value.
    Const(C),
    /// An opaque instruction that couldn't be decomposed further.
    /// Contains the instruction index within the block.
    Opaque {
        /// Block containing the instruction.
        block: BlockId,
        /// Index of the instruction.
        inst_idx: usize,
    },
}

impl<V, Op, C> ExprNode<V, Op, C> {
    /// Whether this is a leaf node.
    #[must_use]
    pub fn is_leaf(&self) -> bool {
        matches!(self, ExprNode::Leaf(_))
    }

    /// Whether this is a compound operation.
    #[must_use]
    pub fn is_op(&self) -> bool {
        matches!(self, ExprNode::Op { .. })
    }

    /// Depth of the expression tree.
    #[must_use]
    pub fn depth(&self) -> usize {
        match self {
            ExprNode::Leaf(_) | ExprNode::Const(_) | ExprNode::Opaque { .. } => 1,
            ExprNode::Op { operands, .. } => {
                1 + operands.iter().map(ExprNode::depth).max().unwrap_or(0)
            }
        }
    }

    /// Count total nodes in the expression tree.
    #[must_use]
    pub fn node_count(&self) -> usize {
        match self {
            ExprNode::Leaf(_) | ExprNode::Const(_) | ExprNode::Opaque { .. } => 1,
            ExprNode::Op { operands, .. } => {
                1 + operands.iter().map(ExprNode::node_count).sum::<usize>()
            }
        }
    }
}

/// Trait for instructions that can be decomposed into expression operators.
///
/// The consumer implements this to tell the framework what operator
/// each instruction represents and what its operands are.
pub trait ExprInstr: InstrInfo {
    /// Consumer operator identity (mnemonic enum, opcode, interned symbol).
    type Operator: Clone + Eq;

    /// Consumer constant domain (may coincide with
    /// [`ConstantFolder::Const`](crate::ConstantFolder::Const)).
    type Const: Clone + Eq;

    /// If this instruction can be represented as an expression,
    /// return the operator and the operand variables, in operand order.
    ///
    /// The slice is the expression's operand list: the recovered tree has
    /// exactly these operands, in this order. Each operand must also be one
    /// of [`uses`](InstrInfo::uses). The instruction may read more than it
    /// takes as operands (a flags register, an implicit input); those reads
    /// stay out of the tree, and a block-local definition feeding one stays
    /// a root of its block.
    ///
    /// Return `None` for instructions that can't be decomposed
    /// (side-effecting, control flow, etc.).
    fn as_expr(&self) -> Option<(Self::Operator, &[Self::Variable])>;

    /// If this instruction loads a constant, return the value.
    fn as_const(&self) -> Option<Self::Const> {
        None
    }
}

/// Expression trees recovered for a single block.
#[derive(Debug, Clone)]
pub struct BlockExprTrees<V, Op, C> {
    /// The block these trees belong to.
    pub block: BlockId,
    /// Expression tree for each root definition: a def NOT inlined into a
    /// later expression of this block, ordered by variable. A variable that
    /// the block defines more than once has one root per definition that
    /// nothing absorbed, in definition order. Inlining is decided by
    /// **intra-block** use count alone (this is a per-block recovery with
    /// no cross-block liveness), so a def used once here and again in
    /// another block is inlined and absent from `roots` — consumers
    /// needing every cross-block-live def materialized should consult
    /// [`Liveness`](crate::Liveness) or
    /// [`DefUseChains`](crate::DefUseChains) alongside.
    pub roots: Vec<(V, ExprNode<V, Op, C>)>,
}

/// The [`BlockExprTrees`] that instructions of type `I` recover to.
type TreesOf<I> =
    BlockExprTrees<<I as InstrInfo>::Variable, <I as ExprInstr>::Operator, <I as ExprInstr>::Const>;

/// Recover expression trees for a single block.
///
/// Walks the block's instructions and builds expression trees by
/// inlining single-use temporaries into their use sites. A variable is a
/// single-use temporary when the block reads it once across every
/// instruction's [`uses`](InstrInfo::uses), operand or not.
///
/// A definition is inlined only while it still means what it did where it
/// was made. One that the block overwrites before anything reads it, or
/// whose operands the block redefines before the reader, stays a root, so a
/// block that assigns a variable twice has one root per assignment.
#[must_use]
pub fn recover_block_expressions<G>(cfg: &G, block: BlockId) -> TreesOf<G::Inst>
where
    G: CfgView,
    G::Inst: ExprInstr,
{
    let insts = cfg.instructions(block);

    let mut use_count = BTreeMap::new();
    for inst in insts {
        for variable in inst.uses() {
            *use_count.entry(variable.clone()).or_insert(0_usize) += 1;
        }
    }

    let mut recovery = Recovery::new();
    for (inst_idx, inst) in insts.iter().enumerate() {
        let defs = inst.defs();
        let definition = if let Some(constant) = inst.as_const() {
            Some(Pending {
                expression: ExprNode::Const(constant),
                reads: Vec::new(),
            })
        } else if let Some((operator, operand_variables)) = inst.as_expr() {
            let mut reads = Vec::new();
            let operands = operand_variables
                .iter()
                .map(|variable| {
                    let single_use = use_count.get(variable) == Some(&1);
                    recovery.operand(variable, single_use, &mut reads)
                })
                .collect();
            Some(Pending {
                expression: ExprNode::Op { operator, operands },
                reads,
            })
        } else {
            None
        };

        if let Some(definition) = definition {
            recovery.redefine_all(defs);
            if let Some(destination) = defs.first() {
                recovery.pending.insert(destination.clone(), definition);
            }
        } else {
            // An opaque instruction defines every output as itself.
            let reads = recovery.reads_of(inst.uses());
            recovery.redefine_all(defs);
            for destination in defs {
                recovery.pending.insert(
                    destination.clone(),
                    Pending {
                        expression: ExprNode::Opaque { block, inst_idx },
                        reads: reads.clone(),
                    },
                );
            }
        }
    }

    BlockExprTrees {
        block,
        roots: recovery.into_roots(),
    }
}

/// A block-local definition that no later expression has absorbed yet.
struct Pending<V, Op, C> {
    expression: ExprNode<V, Op, C>,
    /// Each variable the expression reads, with how many times the block had
    /// defined it when the read happened.
    reads: Vec<(V, usize)>,
}

/// The state of recovering one block.
struct Recovery<V, Op, C> {
    /// The latest definition of each variable, until something absorbs it.
    pending: BTreeMap<V, Pending<V, Op, C>>,
    /// Definitions that a later definition of the same variable overwrote
    /// before anything absorbed them, in the order they were overwritten.
    overwritten: Vec<(V, ExprNode<V, Op, C>)>,
    /// How many times the block has defined each variable so far.
    definition_count: BTreeMap<V, usize>,
}

impl<V: VariableId, Op, C> Recovery<V, Op, C> {
    fn new() -> Self {
        Self {
            pending: BTreeMap::new(),
            overwritten: Vec::new(),
            definition_count: BTreeMap::new(),
        }
    }

    fn generation(&self, variable: &V) -> usize {
        self.definition_count.get(variable).copied().unwrap_or(0)
    }

    fn reads_of(&self, variables: &[V]) -> Vec<(V, usize)> {
        variables
            .iter()
            .map(|variable| (variable.clone(), self.generation(variable)))
            .collect()
    }

    /// The operand for a read of `variable`, recorded in `reads`: the
    /// pending definition when this is the variable's only read and nothing
    /// that definition read has been redefined since, else a leaf.
    fn operand(
        &mut self,
        variable: &V,
        single_use: bool,
        reads: &mut Vec<(V, usize)>,
    ) -> ExprNode<V, Op, C> {
        let absorbable = single_use
            && self.pending.get(variable).is_some_and(|definition| {
                definition
                    .reads
                    .iter()
                    .all(|(read, generation)| self.generation(read) == *generation)
            });
        if absorbable && let Some(definition) = self.pending.remove(variable) {
            reads.extend(definition.reads);
            return definition.expression;
        }
        reads.push((variable.clone(), self.generation(variable)));
        ExprNode::Leaf(variable.clone())
    }

    /// Record that one instruction defined every variable of `defs`: what
    /// each held before can no longer be absorbed.
    fn redefine_all(&mut self, defs: &[V]) {
        for destination in defs {
            if let Some(overwritten) = self.pending.remove(destination) {
                self.overwritten
                    .push((destination.clone(), overwritten.expression));
            }
            *self
                .definition_count
                .entry(destination.clone())
                .or_insert(0) += 1;
        }
    }

    /// Every definition nothing absorbed, ordered by variable and, for one
    /// variable, by definition.
    fn into_roots(self) -> Vec<(V, ExprNode<V, Op, C>)> {
        let mut roots = self.overwritten;
        roots.extend(
            self.pending
                .into_iter()
                .map(|(variable, definition)| (variable, definition.expression)),
        );
        roots.sort_by(|left, right| left.0.cmp(&right.0));
        roots
    }
}

/// Recover expression trees for all blocks in the CFG.
#[must_use]
pub fn recover_expressions<G>(cfg: &G) -> Vec<TreesOf<G::Inst>>
where
    G: CfgView,
    G::Inst: ExprInstr,
{
    cfg.node_ids()
        .map(|block| recover_block_expressions(cfg, block))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::test_util::{DfInst, df_const, df_op};

    #[test]
    fn recover_simple_expression_tree() {
        // mul t0, r1, r2    (t0 = r1 * r2)
        // add t1, r0, t0    (t1 = r0 + t0 = r0 + r1 * r2)
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            df_op("mul", "mul", 10, &[1, 2]),  // t0(loc10) = r1 * r2
            df_op("add", "add", 11, &[0, 10]), // t1(loc11) = r0 + t0
        ]);

        let trees = recover_block_expressions(&cfg, cfg.entry());
        assert_eq!(trees.roots.len(), 1, "only t1 should remain as root");
        let (loc, ref expr) = trees.roots[0];
        assert_eq!(loc, 11);

        // Should be Op("add", [Leaf(r0), Op("mul", [Leaf(r1), Leaf(r2)])])
        match expr {
            ExprNode::Op { operator, operands } => {
                assert_eq!(*operator, "add");
                assert_eq!(operands.len(), 2);
                assert_eq!(operands[0], ExprNode::Leaf(0));
                match &operands[1] {
                    ExprNode::Op {
                        operator: inner_op,
                        operands: inner_ops,
                    } => {
                        assert_eq!(*inner_op, "mul");
                        assert_eq!(inner_ops.len(), 2);
                        assert_eq!(inner_ops[0], ExprNode::Leaf(1));
                        assert_eq!(inner_ops[1], ExprNode::Leaf(2));
                    }
                    _ => panic!("expected nested Op for mul"),
                }
            }
            _ => panic!("expected Op at root"),
        }
    }

    #[test]
    fn constant_folding_in_tree() {
        // const t0 = 42; add t1, r0, t0 → add(Leaf(r0), Const(42))
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .extend([df_const("ldc", 10, 42), df_op("add", "add", 11, &[0, 10])]);

        let trees = recover_block_expressions(&cfg, cfg.entry());
        assert_eq!(trees.roots.len(), 1);
        let (_, ref expr) = trees.roots[0];
        match expr {
            ExprNode::Op { operands, .. } => {
                assert_eq!(operands[1], ExprNode::Const(42));
            }
            _ => panic!("expected Op"),
        }
    }

    #[test]
    fn multi_use_not_inlined() {
        // mul t0, r1, r2; add t1, r0, t0; sub t2, t0, r3
        // t0 has 2 uses → should NOT be inlined, stays as Leaf.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            df_op("mul", "mul", 10, &[1, 2]),
            df_op("add", "add", 11, &[0, 10]),
            df_op("sub", "sub", 12, &[10, 3]),
        ]);

        let trees = recover_block_expressions(&cfg, cfg.entry());
        // t0 is read twice, so it stays a root and both readers keep it as a
        // leaf instead of inlining it.
        assert_eq!(
            trees.roots,
            [
                (10, op("mul", [leaf(1), leaf(2)])),
                (11, op("add", [leaf(0), leaf(10)])),
                (12, op("sub", [leaf(10), leaf(3)])),
            ]
        );
    }

    /// An operation whose reads can exceed its expression operands.
    struct Operation {
        operator: &'static str,
        operands: Vec<u16>,
        reads: Vec<u16>,
        defs: Vec<u16>,
    }

    /// `dest = operator(operands)`, reading exactly `reads`.
    fn operation(dest: u16, operator: &'static str, operands: &[u16], reads: &[u16]) -> Operation {
        Operation {
            operator,
            operands: operands.to_vec(),
            reads: reads.to_vec(),
            defs: alloc::vec![dest],
        }
    }

    impl InstrInfo for Operation {
        type Variable = u16;

        fn uses(&self) -> &[u16] {
            &self.reads
        }

        fn defs(&self) -> &[u16] {
            &self.defs
        }
    }

    impl ExprInstr for Operation {
        type Operator = &'static str;
        type Const = i64;

        fn as_expr(&self) -> Option<(&'static str, &[u16])> {
            Some((self.operator, &self.operands))
        }
    }

    fn leaf(variable: u16) -> ExprNode<u16, &'static str, i64> {
        ExprNode::Leaf(variable)
    }

    fn op<const N: usize>(
        operator: &'static str,
        operands: [ExprNode<u16, &'static str, i64>; N],
    ) -> ExprNode<u16, &'static str, i64> {
        ExprNode::Op {
            operator,
            operands: operands.into(),
        }
    }

    fn recover(
        operations: impl IntoIterator<Item = Operation>,
    ) -> Vec<(u16, ExprNode<u16, &'static str, i64>)> {
        let mut cfg: Cfg<Operation> = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .extend(operations);
        recover_block_expressions(&cfg, cfg.entry()).roots
    }

    #[test]
    fn a_read_that_is_not_an_operand_stays_out_of_the_tree() {
        // flags = cmp(r1, r2); t = add(r1), which also reads the flags.
        let roots = recover([
            operation(9, "cmp", &[1, 2], &[1, 2]),
            operation(10, "add", &[1], &[1, 9]),
        ]);
        assert_eq!(
            roots,
            [
                (9, op("cmp", [leaf(1), leaf(2)])),
                (10, op("add", [leaf(1)])),
            ],
            "the flags definition is read once but is not an operand, so it stays a root"
        );
    }

    #[test]
    fn operands_keep_the_order_the_instruction_reports() {
        // The reads are listed in a different order than the operands.
        let roots = recover([operation(10, "sub", &[1, 2], &[2, 1])]);
        assert_eq!(roots, [(10, op("sub", [leaf(1), leaf(2)]))]);
    }

    #[test]
    fn a_temporary_read_by_a_non_operand_is_not_a_single_use() {
        // t0 feeds `add` as an operand and `probe` as an implicit read: two
        // readers, so it must stay materialized.
        let roots = recover([
            operation(10, "mul", &[1, 2], &[1, 2]),
            operation(11, "add", &[3, 10], &[3, 10]),
            operation(12, "probe", &[4], &[4, 10]),
        ]);
        assert_eq!(
            roots,
            [
                (10, op("mul", [leaf(1), leaf(2)])),
                (11, op("add", [leaf(3), leaf(10)])),
                (12, op("probe", [leaf(4)])),
            ]
        );
    }

    #[test]
    fn a_redefined_variable_keeps_its_earlier_definition_as_a_root() {
        // x = add(r1, r2); x = mul(x, r3); use(x): the first x is read by the
        // second, which also redefines it, and again by `use`.
        let roots = recover([
            operation(10, "add", &[1, 2], &[1, 2]),
            operation(10, "mul", &[10, 3], &[10, 3]),
            operation(11, "use", &[10], &[10]),
        ]);
        assert_eq!(
            roots,
            [
                (10, op("add", [leaf(1), leaf(2)])),
                (10, op("mul", [leaf(10), leaf(3)])),
                (11, op("use", [leaf(10)])),
            ],
            "both definitions of x are roots, in definition order"
        );
    }

    #[test]
    fn a_temporary_is_not_inlined_past_a_redefinition_of_its_operand() {
        // t = add(r1, r2); r1 = neg(r4); u = mul(t, r3): reading r1 inside
        // `u` would see the new r1, not the one `t` was computed from.
        let roots = recover([
            operation(10, "add", &[1, 2], &[1, 2]),
            operation(1, "neg", &[4], &[4]),
            operation(11, "mul", &[10, 3], &[10, 3]),
        ]);
        assert_eq!(
            roots,
            [
                (1, op("neg", [leaf(4)])),
                (10, op("add", [leaf(1), leaf(2)])),
                (11, op("mul", [leaf(10), leaf(3)])),
            ]
        );
    }

    #[test]
    fn a_temporary_is_inlined_when_its_operands_are_redefined_before_it() {
        // r1 = neg(r4); t = add(r1, r2); u = mul(t, r3): every read happens
        // after the definition it names, so the whole chain folds into `u`.
        let roots = recover([
            operation(1, "neg", &[4], &[4]),
            operation(10, "add", &[1, 2], &[1, 2]),
            operation(11, "mul", &[10, 3], &[10, 3]),
        ]);
        assert_eq!(
            roots,
            [(
                11,
                op("mul", [op("add", [op("neg", [leaf(4)]), leaf(2)]), leaf(3)])
            )]
        );
    }
}
