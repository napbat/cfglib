//! A view that forwards to a valid function and breaks one rule of the graph
//! adjacency on purpose, the way a buggy adapter over consumer storage would.

use cfglib::ir::mlil::{self, Successor, ValueDef};
use cfglib::{DenseId, GraphView, RootedView};

use crate::glint::{Arm, Block, Function, Glint, Inst, Op, Ty, Value};

/// The one rule that a skewed view breaks.
#[derive(Clone, Copy, Debug)]
pub enum Fault {
    /// The block lists one predecessor fewer than its records give.
    DropPredecessor(Block),
    /// The block lists its first successor twice.
    ExtraSuccessor(Block),
    /// The root is not below the node bound.
    RootOutOfBounds,
    /// Nothing is broken: the control case.
    None,
}

pub struct Skewed<'f> {
    pub inner: &'f Function,
    pub fault: Fault,
}

impl GraphView for Skewed<'_> {
    type NodeId = Block;

    fn node_bound(&self) -> usize {
        GraphView::node_bound(self.inner)
    }

    fn node_ids(&self) -> impl Iterator<Item = Block> + '_ {
        GraphView::node_ids(self.inner)
    }

    fn successors(&self, node: Block) -> impl Iterator<Item = Block> + '_ {
        let extra = match self.fault {
            Fault::ExtraSuccessor(target) if target == node => {
                GraphView::successors(self.inner, node).next()
            }
            _ => None,
        };
        GraphView::successors(self.inner, node).chain(extra)
    }

    fn predecessors(&self, node: Block) -> impl Iterator<Item = Block> + '_ {
        let skip = matches!(self.fault, Fault::DropPredecessor(target) if target == node);
        GraphView::predecessors(self.inner, node).skip(usize::from(skip))
    }
}

impl RootedView for Skewed<'_> {
    fn root(&self) -> Block {
        match self.fault {
            Fault::RootOutOfBounds => Block::from_index(GraphView::node_bound(self.inner)),
            _ => RootedView::root(self.inner),
        }
    }
}

impl mlil::FunctionView<Glint> for Skewed<'_> {
    type Block = Block;
    type Inst = Inst;
    type Value = Value;

    fn inst_bound(&self) -> usize {
        mlil::FunctionView::inst_bound(self.inner)
    }

    fn value_bound(&self) -> usize {
        mlil::FunctionView::value_bound(self.inner)
    }

    fn params(&self, block: Block) -> &[Value] {
        mlil::FunctionView::params(self.inner, block)
    }

    fn insts(&self, block: Block) -> &[Inst] {
        mlil::FunctionView::insts(self.inner, block)
    }

    fn op(&self, inst: Inst) -> &Op {
        mlil::FunctionView::op(self.inner, inst)
    }

    fn operands(&self, inst: Inst) -> &[Value] {
        mlil::FunctionView::operands(self.inner, inst)
    }

    fn results(&self, inst: Inst) -> &[Value] {
        mlil::FunctionView::results(self.inner, inst)
    }

    fn successors(
        &self,
        block: Block,
    ) -> impl ExactSizeIterator<Item = Successor<'_, Block, Value, Arm>> + '_ {
        mlil::FunctionView::<Glint>::successors(self.inner, block)
    }

    fn block_of(&self, inst: Inst) -> Block {
        mlil::FunctionView::block_of(self.inner, inst)
    }

    fn def(&self, value: Value) -> ValueDef<Inst, Block> {
        mlil::FunctionView::def(self.inner, value)
    }

    fn ty(&self, value: Value) -> &Ty {
        mlil::FunctionView::ty(self.inner, value)
    }
}
