//! Reading a tree the way a lifter does: by matching its nodes.

use cfglib::{BlockId, ControlBody, ControlNode, ControlTree};

/// The statements of the function body: the root's children when the root is a
/// sequence, and the root itself otherwise.
pub fn top_level(tree: &ControlTree) -> Vec<ControlNode> {
    match *tree.node(tree.root()) {
        ControlNode::Sequence(body) => body_nodes(tree, body),
        other => vec![other],
    }
}

/// The nodes of one body, in execution order.
pub fn body_nodes(tree: &ControlTree, body: ControlBody) -> Vec<ControlNode> {
    tree.body(body).iter().map(|&id| *tree.node(id)).collect()
}

/// The block each statement of a body of plain leaves stands for.
///
/// # Panics
///
/// Panics when the body holds anything but a `Block` or `Return` leaf.
pub fn leaf_blocks(tree: &ControlTree, body: ControlBody) -> Vec<BlockId> {
    body_nodes(tree, body)
        .into_iter()
        .map(|node| match node {
            ControlNode::Block(span) | ControlNode::Return(span) => span.block(),
            other => panic!("expected a leaf, found {other:?}"),
        })
        .collect()
}

/// A short name for a node's kind, so two trees can be compared by shape.
pub fn kind_name(node: &ControlNode) -> &'static str {
    match node {
        ControlNode::Block(_) => "block",
        ControlNode::Sequence(_) => "sequence",
        ControlNode::IfThenElse { .. } => "if",
        ControlNode::Loop { .. } => "loop",
        ControlNode::Switch { .. } => "switch",
        ControlNode::Break { .. } => "break",
        ControlNode::Continue { .. } => "continue",
        ControlNode::Return(_) => "return",
        ControlNode::Label { .. } => "label",
        ControlNode::Goto { .. } => "goto",
        ControlNode::TryCatch { .. } => "try",
        ControlNode::Guarded { .. } => "guarded",
    }
}

/// The kinds of every node, in preorder.
pub fn shape(tree: &ControlTree) -> Vec<&'static str> {
    tree.preorder().map(|id| kind_name(tree.node(id))).collect()
}
