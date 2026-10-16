//! The pending expression trees of one instruction list.
//!
//! A builder is write-only, so the lift keeps the expressions that it has
//! planned but not yet used in its own arena. A statement consumes a tree
//! when it is built, and only then does the lift call the builder, children
//! first. Everything that the lift edits after the fact (the negation of a
//! loop condition) or reads back (the variables of an operand, the type of a
//! stored value) happens here.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::dataflow::ProgramPoint;

use super::super::dialect::Dialect;

/// The position of one node in a [`Scratch`] arena.
pub(super) type Slot = usize;

/// What one pending expression is.
pub(super) enum PendingKind<H: Dialect, V> {
    /// A read of a flat variable.
    Variable(V),
    /// A literal.
    Constant(H::Constant),
    /// An operation over earlier nodes of the arena.
    Operation { op: H::Op, operands: Vec<Slot> },
}

/// One pending expression.
pub(super) struct Pending<H: Dialect, V> {
    pub(super) kind: PendingKind<H, V>,
    pub(super) ty: H::Type,
    /// The flat instructions whose value this expression is. An inlined copy
    /// shares the node of the definition that it copies, so a node can stand
    /// for several instructions.
    pub(super) points: Vec<ProgramPoint>,
}

/// The pending expressions of one instruction list. An operation's operands
/// always precede it in the arena.
pub(super) struct Scratch<H: Dialect, V> {
    nodes: Vec<Pending<H, V>>,
}

impl<H: Dialect, V: Copy + Ord> Scratch<H, V> {
    pub(super) const fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    fn push(&mut self, kind: PendingKind<H, V>, ty: H::Type) -> Slot {
        self.nodes.push(Pending {
            kind,
            ty,
            points: Vec::new(),
        });
        self.nodes.len() - 1
    }

    pub(super) fn variable(&mut self, variable: V, ty: H::Type) -> Slot {
        self.push(PendingKind::Variable(variable), ty)
    }

    pub(super) fn constant(&mut self, constant: H::Constant, ty: H::Type) -> Slot {
        self.push(PendingKind::Constant(constant), ty)
    }

    pub(super) fn operation(&mut self, op: H::Op, operands: Vec<Slot>, ty: H::Type) -> Slot {
        debug_assert!(
            operands.iter().all(|&operand| operand < self.nodes.len()),
            "an operation follows its operands"
        );
        self.push(PendingKind::Operation { op, operands }, ty)
    }

    pub(super) fn node(&self, slot: Slot) -> &Pending<H, V> {
        &self.nodes[slot]
    }

    /// Records that `slot` is the value of the instruction at `point`.
    pub(super) fn attach(&mut self, slot: Slot, point: ProgramPoint) {
        self.nodes[slot].points.push(point);
    }

    /// Replaces the operation of an operation node by `rewrite` of it, keeping
    /// the operands and the instructions that the node stands for. Reports
    /// whether the node was rewritten: `rewrite` can decline, and only an
    /// operation node has an operation.
    pub(super) fn rewrite_operation(
        &mut self,
        slot: Slot,
        rewrite: impl FnOnce(&H::Op) -> Option<H::Op>,
    ) -> bool {
        let PendingKind::Operation { op, .. } = &mut self.nodes[slot].kind else {
            return false;
        };
        let Some(rewritten) = rewrite(op) else {
            return false;
        };
        *op = rewritten;
        true
    }

    /// Adds every flat variable that the tree under `root` reads to `reads`.
    pub(super) fn collect_reads(&self, root: Slot, reads: &mut BTreeSet<V>) {
        let mut work = vec![root];
        while let Some(slot) = work.pop() {
            match &self.nodes[slot].kind {
                PendingKind::Variable(variable) => {
                    reads.insert(*variable);
                }
                PendingKind::Constant(_) => {}
                PendingKind::Operation { operands, .. } => work.extend(operands),
            }
        }
    }
}
