//! Traversal utilities over control trees.
//!
//! None of them recurses, so the depth of a tree is bounded by memory, not by
//! the call stack.

extern crate alloc;
use alloc::vec::Vec;
use core::slice;

use super::node::{BlockSpan, ControlNode, ControlNodeId, LoopKind};
use super::tree::ControlTree;
use crate::ProgramPoint;
use crate::dataflow::cfg_view::CfgView;

/// What [`ControlVisitor::enter`] asks the walk to do with a node's
/// children.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlDescent {
    /// Visit the children of the node.
    Into,
    /// Skip the children of the node.
    Skip,
}

/// A visitor of a control tree, driven by [`ControlTree::walk`].
///
/// The walk calls [`enter`](Self::enter) when it reaches a node and
/// [`leave`](Self::leave) after the node's children, in execution order. The
/// two calls pair up for every node, including one whose children are skipped,
/// so a visitor that keeps a stack pushes in `enter` and pops in `leave`.
pub trait ControlVisitor {
    /// Called when the walk reaches `node`, before its children.
    fn enter(&mut self, tree: &ControlTree, node: ControlNodeId) -> ControlDescent {
        let _ = (tree, node);
        ControlDescent::Into
    }

    /// Called after the children of `node`, or right after
    /// [`enter`](Self::enter) when its children were skipped.
    fn leave(&mut self, tree: &ControlTree, node: ControlNodeId) {
        let _ = (tree, node);
    }
}

/// The nodes of a tree in preorder: a node, then each of its children's
/// subtrees in execution order.
///
/// Returned by [`ControlTree::preorder`].
#[derive(Debug, Clone)]
pub struct Preorder<'tree> {
    tree: &'tree ControlTree,
    /// The siblings still to visit, innermost body last.
    stack: Vec<slice::Iter<'tree, ControlNodeId>>,
}

impl Iterator for Preorder<'_> {
    type Item = ControlNodeId;

    fn next(&mut self) -> Option<ControlNodeId> {
        loop {
            let siblings = self.stack.last_mut()?;
            let Some(&id) = siblings.next() else {
                self.stack.pop();
                continue;
            };
            // Pushed last-to-first so the first child body is on top.
            for body in self.tree.child_bodies(id).rev() {
                self.stack.push(self.tree.body(body).iter());
            }
            return Some(id);
        }
    }
}

enum Step {
    Enter(ControlNodeId),
    Leave(ControlNodeId),
}

impl ControlTree {
    /// Visit every node in preorder: a node, then each of its children's
    /// subtrees in execution order.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{Cfg, ControlNode, ControlTree};
    ///
    /// let mut cfg = Cfg::<&str>::new();
    /// cfg.block_mut(cfg.entry()).push("only");
    /// let tree = ControlTree::compute(&cfg);
    /// let leaves = tree
    ///     .preorder()
    ///     .filter(|&id| matches!(tree.node(id), ControlNode::Return(_)))
    ///     .count();
    /// assert_eq!(leaves, 1);
    /// ```
    #[must_use]
    pub fn preorder(&self) -> Preorder<'_> {
        Preorder {
            tree: self,
            stack: alloc::vec![slice::from_ref(&self.root).iter()],
        }
    }

    /// Drive `visitor` over every node, with an `enter` and a `leave` call
    /// around each node's children.
    pub fn walk(&self, visitor: &mut impl ControlVisitor) {
        let mut steps = alloc::vec![Step::Enter(self.root())];
        while let Some(step) = steps.pop() {
            match step {
                Step::Enter(id) => {
                    let descent = visitor.enter(self, id);
                    steps.push(Step::Leave(id));
                    if descent == ControlDescent::Into {
                        // Pushed last-to-first so the first child runs first.
                        for body in self.child_bodies(id).rev() {
                            steps.extend(self.body(body).iter().rev().map(|&id| Step::Enter(id)));
                        }
                    }
                }
                Step::Leave(id) => visitor.leave(self, id),
            }
        }
    }

    /// Visit every instruction the tree stands for, in program-structure
    /// order, with its position and the instruction read from `graph`.
    ///
    /// A pre-tested loop's condition comes before its body and a post-tested
    /// loop's condition after it. A branch or dispatch block's instructions
    /// come before its arms. `graph` is the graph the tree was computed from.
    ///
    /// # Panics
    ///
    /// Panics when `graph` is not the graph the tree was computed from, or
    /// has changed since.
    pub fn for_each_instruction<'graph, G: CfgView>(
        &self,
        graph: &'graph G,
        visit: impl FnMut(ProgramPoint, &'graph G::Inst),
    ) {
        struct Instructions<'graph, G, F> {
            graph: &'graph G,
            visit: F,
        }

        impl<'graph, G: CfgView, F: FnMut(ProgramPoint, &'graph G::Inst)> Instructions<'graph, G, F> {
            fn span(&mut self, span: BlockSpan) {
                let instructions = span.instructions(self.graph);
                for (point, instruction) in span.points().zip(instructions) {
                    (self.visit)(point, instruction);
                }
            }
        }

        impl<'graph, G: CfgView, F: FnMut(ProgramPoint, &'graph G::Inst)> ControlVisitor
            for Instructions<'graph, G, F>
        {
            fn enter(&mut self, tree: &ControlTree, node: ControlNodeId) -> ControlDescent {
                match *tree.node(node) {
                    ControlNode::Block(span) | ControlNode::Return(span) => self.span(span),
                    ControlNode::IfThenElse { condition, .. }
                    | ControlNode::Switch { condition, .. } => self.span(condition),
                    ControlNode::Loop {
                        kind: LoopKind::While { condition, .. },
                        ..
                    } => {
                        for &span in tree.spans(condition) {
                            self.span(span);
                        }
                    }
                    _ => {}
                }
                ControlDescent::Into
            }

            fn leave(&mut self, tree: &ControlTree, node: ControlNodeId) {
                if let ControlNode::Loop {
                    kind: LoopKind::DoWhile { latch, .. },
                    ..
                } = *tree.node(node)
                {
                    self.span(latch);
                }
            }
        }

        self.walk(&mut Instructions { graph, visit });
    }
}
