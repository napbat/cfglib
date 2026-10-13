//! A toy medium-level dialect over `Vec`-backed storage.
//!
//! The dialect has integer constants, a few pure operators, one load and one
//! store as ordered operations, and three terminators. A switch edge carries
//! its case value, and `None` is the default case.

extern crate alloc;

use alloc::vec::Vec;

use crate::graph::view::{DenseId, GraphView, RootedView};
use crate::ir::mlil::{
    AnalysisDialect, Dialect, Effects, FunctionView, Lattice, Successor, ValueDef,
};

macro_rules! dense_id {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub(super) struct $name(pub(super) u32);

        impl DenseId for $name {
            fn from_index(index: usize) -> Self {
                Self(u32::try_from(index).unwrap())
            }

            fn index(self) -> usize {
                usize::try_from(self.0).unwrap()
            }
        }
    };
}

dense_id!(
    /// A block of a toy function.
    Block
);
dense_id!(
    /// An instruction of a toy function.
    Inst
);
dense_id!(
    /// A value of a toy function.
    Value
);

/// The marker of the toy dialect.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct Toy;

/// An operation of the toy dialect.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(super) enum Op {
    /// A constant result.
    Const(i64),
    /// The sum of two operands.
    Add,
    /// One when the first operand is below the second, and zero otherwise.
    Lt,
    /// A copy of the only operand.
    Copy,
    /// An ordered read of memory that gives one result.
    Load,
    /// An ordered write of memory.
    Store,
    /// A pure operation without results.
    Nop,
    /// A terminator with one successor.
    Jump,
    /// A terminator that picks the successor whose edge case equals the
    /// operand, or the default successor.
    Switch,
    /// A terminator without successors.
    Return,
}

impl Dialect for Toy {
    type Op = Op;
    type Type = ();
    type Edge = Option<i64>;
}

impl AnalysisDialect for Toy {
    type Constant = i64;

    fn effects(op: &Op) -> Effects {
        match op {
            Op::Load | Op::Store => Effects {
                ordered: true,
                ..Effects::default()
            },
            Op::Jump | Op::Switch | Op::Return => Effects {
                terminator: true,
                ..Effects::default()
            },
            Op::Const(_) | Op::Add | Op::Lt | Op::Copy | Op::Nop => Effects::default(),
        }
    }

    fn fold(op: &Op, args: &[Lattice<i64>], results: &mut [Lattice<i64>]) {
        match op {
            Op::Const(value) => results[0] = Lattice::Constant(*value),
            Op::Add => results[0] = binary(args, i64::wrapping_add),
            Op::Lt => results[0] = binary(args, |left, right| i64::from(left < right)),
            Op::Copy => results[0] = args[0].clone(),
            Op::Load | Op::Store | Op::Nop | Op::Jump | Op::Switch | Op::Return => {}
        }
    }

    fn fold_branch<'e>(
        op: &Op,
        args: &[Lattice<i64>],
        edges: impl ExactSizeIterator<Item = &'e Option<i64>>,
    ) -> Option<usize> {
        let (Op::Switch, [Lattice::Constant(scrutinee)]) = (op, args) else {
            return None;
        };
        let cases: Vec<Option<i64>> = edges.copied().collect();
        cases
            .iter()
            .position(|case| *case == Some(*scrutinee))
            .or_else(|| cases.iter().position(Option::is_none))
    }

    fn copy_of(op: &Op) -> Option<usize> {
        (*op == Op::Copy).then_some(0)
    }
}

fn binary(args: &[Lattice<i64>], operation: impl Fn(i64, i64) -> i64) -> Lattice<i64> {
    match args {
        [Lattice::Constant(left), Lattice::Constant(right)] => {
            Lattice::Constant(operation(*left, *right))
        }
        _ => Lattice::Varying,
    }
}

/// One block of storage.
#[derive(Debug, Default)]
struct BlockData {
    params: Vec<Value>,
    insts: Vec<Inst>,
}

/// One successor record of storage.
#[derive(Debug)]
struct SuccessorData {
    target: Block,
    args: Vec<Value>,
    edge: Option<i64>,
}

/// One instruction of storage.
#[derive(Debug)]
struct InstData {
    op: Op,
    block: Block,
    operands: Vec<Value>,
    results: Vec<Value>,
    successors: Vec<SuccessorData>,
}

/// A toy function. The first block is the entry block.
#[derive(Debug, Default)]
pub(super) struct Function {
    blocks: Vec<BlockData>,
    insts: Vec<InstData>,
    defs: Vec<ValueDef<Inst, Block>>,
}

impl Function {
    /// Creates a function without blocks.
    pub(super) fn new() -> Self {
        Self::default()
    }

    /// Adds a block.
    pub(super) fn block(&mut self) -> Block {
        self.blocks.push(BlockData::default());
        Block::from_index(self.blocks.len() - 1)
    }

    /// Adds a parameter to `block`.
    pub(super) fn param(&mut self, block: Block) -> Value {
        let index = self.blocks[block.index()].params.len();
        let value = self.define(ValueDef::Param {
            block,
            index: u32::try_from(index).unwrap(),
        });
        self.blocks[block.index()].params.push(value);
        value
    }

    /// Appends an instruction with `result_count` results to `block`.
    pub(super) fn emit(
        &mut self,
        block: Block,
        op: Op,
        operands: &[Value],
        result_count: u32,
    ) -> (Inst, Vec<Value>) {
        let inst = Inst::from_index(self.insts.len());
        let results = (0..result_count)
            .map(|index| self.define(ValueDef::Result { inst, index }))
            .collect::<Vec<_>>();
        self.insts.push(InstData {
            op,
            block,
            operands: operands.to_vec(),
            results: results.clone(),
            successors: Vec::new(),
        });
        self.blocks[block.index()].insts.push(inst);
        (inst, results)
    }

    /// Appends an instruction with one result to `block` and returns the
    /// result.
    pub(super) fn value(&mut self, block: Block, op: Op, operands: &[Value]) -> Value {
        self.emit(block, op, operands, 1).1[0]
    }

    /// Appends the terminator of `block`. A successor is a target, the
    /// arguments for the parameters of the target, and the edge case.
    pub(super) fn terminate(
        &mut self,
        block: Block,
        op: Op,
        operands: &[Value],
        successors: &[(Block, &[Value], Option<i64>)],
    ) -> Inst {
        let (inst, _) = self.emit(block, op, operands, 0);
        self.insts[inst.index()].successors = successors
            .iter()
            .map(|&(target, args, edge)| SuccessorData {
                target,
                args: args.to_vec(),
                edge,
            })
            .collect();
        inst
    }

    fn define(&mut self, def: ValueDef<Inst, Block>) -> Value {
        self.defs.push(def);
        Value::from_index(self.defs.len() - 1)
    }

    fn records(&self, block: Block) -> &[SuccessorData] {
        match self.blocks[block.index()].insts.last() {
            Some(inst) => &self.insts[inst.index()].successors,
            None => &[],
        }
    }
}

impl GraphView for Function {
    type NodeId = Block;

    fn node_bound(&self) -> usize {
        self.blocks.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = Block> + '_ {
        (0..self.blocks.len()).map(Block::from_index)
    }

    fn successors(&self, node: Block) -> impl Iterator<Item = Block> + '_ {
        self.records(node).iter().map(|record| record.target)
    }

    fn predecessors(&self, node: Block) -> impl Iterator<Item = Block> + '_ {
        self.node_ids().flat_map(move |source| {
            self.records(source)
                .iter()
                .filter(move |record| record.target == node)
                .map(move |_| source)
        })
    }
}

impl RootedView for Function {
    fn root(&self) -> Block {
        Block(0)
    }
}

impl FunctionView<Toy> for Function {
    type Block = Block;
    type Inst = Inst;
    type Value = Value;

    fn inst_bound(&self) -> usize {
        self.insts.len()
    }

    fn value_bound(&self) -> usize {
        self.defs.len()
    }

    fn params(&self, block: Block) -> &[Value] {
        &self.blocks[block.index()].params
    }

    fn insts(&self, block: Block) -> &[Inst] {
        &self.blocks[block.index()].insts
    }

    fn op(&self, inst: Inst) -> &Op {
        &self.insts[inst.index()].op
    }

    fn operands(&self, inst: Inst) -> &[Value] {
        &self.insts[inst.index()].operands
    }

    fn results(&self, inst: Inst) -> &[Value] {
        &self.insts[inst.index()].results
    }

    fn successors(
        &self,
        block: Block,
    ) -> impl ExactSizeIterator<Item = Successor<'_, Block, Value, Option<i64>>> + '_ {
        self.records(block).iter().map(|record| Successor {
            target: record.target,
            args: &record.args,
            edge: &record.edge,
        })
    }

    fn block_of(&self, inst: Inst) -> Block {
        self.insts[inst.index()].block
    }

    fn def(&self, value: Value) -> ValueDef<Inst, Block> {
        self.defs[value.index()]
    }

    fn ty(&self, _value: Value) -> &() {
        &()
    }
}
