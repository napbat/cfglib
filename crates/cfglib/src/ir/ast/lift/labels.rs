//! Exact goto-label placement over a lifted tree.
//!
//! Labels are applied after the whole walk so only blocks an emitted goto
//! actually targets are wrapped — proactive labeling would force consumers
//! into unstructured fallbacks for transfers that resolved structurally.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use super::super::node::{AstNode, CatchHandler, SwitchCase};
use super::block_label_name;
use crate::cfg::Cfg;

/// The block whose emission this node anchors, when it has one.
fn anchor<I>(node: &AstNode<I>) -> Option<u32> {
    match node {
        AstNode::Block { id, .. } | AstNode::Return { id, .. } => Some(id.raw()),
        AstNode::IfThenElse { condition, .. } | AstNode::Switch { condition, .. } => {
            Some(condition.raw())
        }
        AstNode::Loop { header, .. } => Some(header.raw()),
        _ => None,
    }
}

/// Wrap the emission of every block in `pending` with its label, removing
/// each target as it is anchored. Targets that remain afterwards have no
/// structured anchor — dangling gotos the caller reports.
pub(super) fn apply_labels<I, E, O>(
    cfg: &Cfg<I, E>,
    nodes: Vec<AstNode<O>>,
    pending: &mut BTreeSet<u32>,
) -> Vec<AstNode<O>> {
    nodes
        .into_iter()
        .map(|node| apply_to_node(cfg, node, pending))
        .collect()
}

fn apply_to_node<I, E, O>(
    cfg: &Cfg<I, E>,
    node: AstNode<O>,
    pending: &mut BTreeSet<u32>,
) -> AstNode<O> {
    let wrap = anchor(&node).filter(|block| pending.remove(block));
    let node = apply_to_children(cfg, node, pending);
    match wrap {
        Some(block) => AstNode::Label {
            name: block_label_name(cfg, crate::BlockId::from_raw(block)),
            body: vec![node],
        },
        None => node,
    }
}

fn apply_to_children<I, E, O>(
    cfg: &Cfg<I, E>,
    node: AstNode<O>,
    pending: &mut BTreeSet<u32>,
) -> AstNode<O> {
    match node {
        AstNode::Sequence { body } => AstNode::Sequence {
            body: apply_labels(cfg, body, pending),
        },
        AstNode::Label { name, body } => AstNode::Label {
            name,
            body: apply_labels(cfg, body, pending),
        },
        AstNode::Guarded {
            predicate,
            when_true,
            body,
        } => AstNode::Guarded {
            predicate,
            when_true,
            body: apply_labels(cfg, body, pending),
        },
        AstNode::Loop { header, kind, body } => AstNode::Loop {
            header,
            kind,
            body: apply_labels(cfg, body, pending),
        },
        AstNode::IfThenElse {
            condition,
            condition_instructions,
            then_body,
            else_body,
        } => AstNode::IfThenElse {
            condition,
            condition_instructions,
            then_body: apply_labels(cfg, then_body, pending),
            else_body: apply_labels(cfg, else_body, pending),
        },
        AstNode::Switch {
            condition,
            condition_instructions,
            cases,
            default_body,
            default_edge,
        } => AstNode::Switch {
            condition,
            condition_instructions,
            cases: cases
                .into_iter()
                .map(|case| SwitchCase {
                    id: case.id,
                    edges: case.edges,
                    body: apply_labels(cfg, case.body, pending),
                })
                .collect(),
            default_body: apply_labels(cfg, default_body, pending),
            default_edge,
        },
        AstNode::TryCatch {
            try_body,
            handlers,
            finally_body,
        } => AstNode::TryCatch {
            try_body: apply_labels(cfg, try_body, pending),
            handlers: handlers
                .into_iter()
                .map(|handler| CatchHandler {
                    handler: handler.handler,
                    entry: handler.entry,
                    kind: handler.kind,
                    body: apply_labels(cfg, handler.body, pending),
                })
                .collect(),
            finally_body: apply_labels(cfg, finally_body, pending),
        },
        leaf @ (AstNode::Block { .. }
        | AstNode::Return { .. }
        | AstNode::Break { .. }
        | AstNode::Continue { .. }
        | AstNode::Goto { .. }) => leaf,
    }
}
