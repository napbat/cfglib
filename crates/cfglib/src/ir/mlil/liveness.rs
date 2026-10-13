//! Value liveness.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::csr::Csr;
use super::uses::{Use, UseIndex};
use crate::graph::view::DenseId;
use crate::ir::mlil::{Dialect, FunctionView, ValueDef};

/// Which values are live at the start and at the end of each block.
///
/// The rules follow the form of the function:
///
/// - A block parameter is defined at the start of its block, so it is not
///   live there.
/// - An argument of a successor record is a use on the edge, at the end of
///   the block that holds the record. The argument is live at the end of that
///   block, and it is not live at the start of the target block.
/// - A value is live at the start of a block when a path from the block
///   reaches a use of the value without passing through the definition.
///
/// Both sets list their values in ascending order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValueLiveness<V> {
    /// The values live at the start of each block.
    live_in: Csr<V>,
    /// The values live at the end of each block.
    live_out: Csr<V>,
}

impl<V: DenseId> ValueLiveness<V> {
    /// Computes the liveness of every value that `function` uses.
    ///
    /// The algorithm starts at each use of a value and walks backward through
    /// the predecessors up to the definition. Its cost follows the live
    /// ranges and not the product of blocks and values, so a large function
    /// with short ranges stays cheap. `uses` is the use index of `function`.
    ///
    /// # Panics
    ///
    /// Panics when `uses` was not built from `function`.
    #[must_use]
    pub fn compute<D, F>(function: &F, uses: &UseIndex<F::Inst, F::Block>) -> Self
    where
        D: Dialect,
        F: FunctionView<D, Value = V>,
    {
        let bound = function.node_bound();
        let mut walk = Walk {
            in_marks: vec![usize::MAX; bound],
            out_marks: vec![usize::MAX; bound],
            live_in: Vec::new(),
            live_out: Vec::new(),
            stack: Vec::new(),
            value: 0,
            def_block: 0,
        };
        for index in 0..function.value_bound() {
            let value = V::from_index(index);
            let value_uses = uses.uses(value);
            if value_uses.is_empty() {
                continue;
            }
            walk.value = index;
            walk.def_block = match function.def(value) {
                ValueDef::Result { inst, .. } => function.block_of(inst),
                ValueDef::Param { block, .. } => block,
            }
            .index();
            for &value_use in value_uses {
                let use_block = match value_use {
                    Use::Operand { inst, .. } => function.block_of(inst).index(),
                    Use::Edge { block, .. } => {
                        walk.live_at_end(block.index());
                        block.index()
                    }
                };
                walk.live_at_start(use_block);
            }
            while let Some(block) = walk.stack.pop() {
                for predecessor in function.predecessors(F::Block::from_index(block)) {
                    walk.live_at_end(predecessor.index());
                    walk.live_at_start(predecessor.index());
                }
            }
        }

        let group = |pairs: &[(usize, V)]| {
            Csr::build(bound, |emit| {
                for &(block, value) in pairs {
                    emit(block, value);
                }
            })
        };
        Self {
            live_in: group(&walk.live_in),
            live_out: group(&walk.live_out),
        }
    }

    /// Returns the values live at the start of `block`, in ascending order.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn live_in<B: DenseId>(&self, block: B) -> &[V] {
        self.live_in.row(block.index())
    }

    /// Returns the values live at the end of `block`, in ascending order.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn live_out<B: DenseId>(&self, block: B) -> &[V] {
        self.live_out.row(block.index())
    }

    /// Returns whether `value` is live at the start of `block`.
    #[must_use]
    pub fn is_live_in<B: DenseId>(&self, block: B, value: V) -> bool {
        self.live_in(block).binary_search(&value).is_ok()
    }

    /// Returns whether `value` is live at the end of `block`.
    #[must_use]
    pub fn is_live_out<B: DenseId>(&self, block: B, value: V) -> bool {
        self.live_out(block).binary_search(&value).is_ok()
    }
}

/// The state of the walks that mark the live range of one value at a time.
struct Walk<V> {
    /// For each block, the value whose walk last marked it live at the start.
    in_marks: Vec<usize>,
    /// For each block, the value whose walk last marked it live at the end.
    out_marks: Vec<usize>,
    /// The `(block, value)` pairs live at the start, by ascending value.
    live_in: Vec<(usize, V)>,
    /// The `(block, value)` pairs live at the end, by ascending value.
    live_out: Vec<(usize, V)>,
    /// The blocks whose predecessors the walk still has to visit.
    stack: Vec<usize>,
    /// The index of the value in hand.
    value: usize,
    /// The index of the block that defines the value in hand.
    def_block: usize,
}

impl<V: DenseId> Walk<V> {
    /// Marks the value live at the end of `block`.
    fn live_at_end(&mut self, block: usize) {
        if self.out_marks[block] != self.value {
            self.out_marks[block] = self.value;
            self.live_out.push((block, V::from_index(self.value)));
        }
    }

    /// Marks the value live at the start of `block` and queues the block,
    /// unless the block defines the value or the walk already marked it.
    fn live_at_start(&mut self, block: usize) {
        if block != self.def_block && self.in_marks[block] != self.value {
            self.in_marks[block] = self.value;
            self.live_in.push((block, V::from_index(self.value)));
            self.stack.push(block);
        }
    }
}
