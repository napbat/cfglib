//! Exact goto-label placement over a structured tree.
//!
//! Labels are applied after the whole walk so only blocks an emitted goto
//! actually targets are wrapped — proactive labeling would force consumers
//! into unstructured fallbacks for transfers that resolved structurally.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::super::node::{ControlBody, ControlNode, ControlNodeId};
use super::Structurer;
use crate::block::BlockId;

/// The block whose emission this node anchors, when it has one.
const fn anchor(node: &ControlNode) -> Option<BlockId> {
    match node {
        ControlNode::Block(span) | ControlNode::Return(span) => Some(span.block()),
        ControlNode::IfThenElse { condition, .. } | ControlNode::Switch { condition, .. } => {
            Some(condition.block())
        }
        ControlNode::Loop { header, .. } => Some(*header),
        _ => None,
    }
}

impl<I, E> Structurer<'_, I, E> {
    /// Wrap the emission of every block in `pending` with its label, removing
    /// each target as it is anchored. Targets that remain afterwards have no
    /// structured anchor — dangling gotos the caller reports.
    ///
    /// A wrapped node is replaced in its parent's body by a
    /// [`ControlNode::Label`] whose body holds just that node, so no node
    /// moves and every other handle stays valid.
    pub(super) fn apply_labels(&mut self, body: ControlBody, pending: &mut BTreeSet<BlockId>) {
        for position in 0..body.len() {
            if pending.is_empty() {
                return;
            }
            let id = self.tables.body(body)[position];
            let node = *self.tables.node(id);
            let wrap = anchor(&node).filter(|block| pending.remove(block));
            let children: Vec<ControlBody> = self.tables.child_bodies(node).collect();
            for child in children {
                self.apply_labels(child, pending);
            }
            if let Some(block) = wrap {
                let wrapped = self.label_node(block, id);
                self.tables.body_mut(body)[position] = wrapped;
            }
        }
    }

    /// A new label node for `block` whose body is the single node `id`.
    fn label_node(&mut self, block: BlockId, id: ControlNodeId) -> ControlNodeId {
        let body = self.tables.seal_ids(&[id]);
        self.tables.push_node(ControlNode::Label { block, body })
    }
}
