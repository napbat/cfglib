//! The flat arena that stores a control tree.

extern crate alloc;
use alloc::vec::Vec;

use super::node::{
    BlockSpan, CatchHandler, ControlBody, ControlNode, ControlNodeId, ControlRun, SwitchCase,
};
use crate::edge::EdgeId;

/// One table of runs: elements are appended in runs and never moved.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Pool<T> {
    items: Vec<T>,
}

impl<T> Pool<T> {
    const fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Append `items` as one run.
    ///
    /// # Panics
    ///
    /// Panics when the table would exceed `u32::MAX` elements.
    fn push_run(&mut self, items: impl ExactSizeIterator<Item = T>) -> ControlRun<T> {
        let start = u32::try_from(self.items.len()).expect("control tree table exceeds u32::MAX");
        let len = u32::try_from(items.len()).expect("control tree run exceeds u32::MAX");
        self.items.extend(items);
        assert!(
            u32::try_from(self.items.len()).is_ok(),
            "control tree table exceeds u32::MAX"
        );
        ControlRun::new(start, len)
    }

    fn get(&self, run: ControlRun<T>) -> &[T] {
        &self.items[run.range()]
    }

    fn get_mut(&mut self, run: ControlRun<T>) -> &mut [T] {
        &mut self.items[run.range()]
    }
}

/// The tables of a tree, without a root: what the structurer fills in and
/// the tree keeps.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Tables {
    nodes: Vec<ControlNode>,
    bodies: Pool<ControlNodeId>,
    spans: Pool<BlockSpan>,
    cases: Pool<SwitchCase>,
    edges: Pool<EdgeId>,
    handlers: Pool<CatchHandler>,
}

impl Tables {
    pub(super) const fn new() -> Self {
        Self {
            nodes: Vec::new(),
            bodies: Pool::new(),
            spans: Pool::new(),
            cases: Pool::new(),
            edges: Pool::new(),
            handlers: Pool::new(),
        }
    }

    pub(super) fn finish(self, root: ControlNodeId) -> ControlTree {
        ControlTree { tables: self, root }
    }

    pub(super) fn node(&self, id: ControlNodeId) -> &ControlNode {
        &self.nodes[id.index()]
    }

    pub(super) fn push_node(&mut self, node: ControlNode) -> ControlNodeId {
        let id = ControlNodeId::from_index(self.nodes.len());
        self.nodes.push(node);
        id
    }

    /// Store `nodes` as one body, giving each node its handle.
    pub(super) fn seal(&mut self, nodes: Vec<ControlNode>) -> ControlBody {
        let ids: Vec<ControlNodeId> = nodes.into_iter().map(|node| self.push_node(node)).collect();
        self.seal_ids(&ids)
    }

    /// Store existing nodes as one body.
    pub(super) fn seal_ids(&mut self, ids: &[ControlNodeId]) -> ControlBody {
        self.bodies.push_run(ids.iter().copied())
    }

    pub(super) fn push_spans(
        &mut self,
        spans: impl ExactSizeIterator<Item = BlockSpan>,
    ) -> ControlRun<BlockSpan> {
        self.spans.push_run(spans)
    }

    pub(super) fn push_cases(&mut self, cases: Vec<SwitchCase>) -> ControlRun<SwitchCase> {
        self.cases.push_run(cases.into_iter())
    }

    pub(super) fn push_edges(&mut self, edges: Vec<EdgeId>) -> ControlRun<EdgeId> {
        self.edges.push_run(edges.into_iter())
    }

    pub(super) fn push_handlers(
        &mut self,
        handlers: Vec<CatchHandler>,
    ) -> ControlRun<CatchHandler> {
        self.handlers.push_run(handlers.into_iter())
    }

    pub(super) fn body(&self, body: ControlBody) -> &[ControlNodeId] {
        self.bodies.get(body)
    }

    /// The children of a body, writable so a pass can wrap one in place.
    pub(super) fn body_mut(&mut self, body: ControlBody) -> &mut [ControlNodeId] {
        self.bodies.get_mut(body)
    }

    pub(super) fn spans(&self, run: ControlRun<BlockSpan>) -> &[BlockSpan] {
        self.spans.get(run)
    }

    pub(super) fn cases(&self, run: ControlRun<SwitchCase>) -> &[SwitchCase] {
        self.cases.get(run)
    }

    pub(super) fn edges(&self, run: ControlRun<EdgeId>) -> &[EdgeId] {
        self.edges.get(run)
    }

    pub(super) fn handlers(&self, run: ControlRun<CatchHandler>) -> &[CatchHandler] {
        self.handlers.get(run)
    }

    pub(super) fn child_bodies(&self, node: ControlNode) -> ChildBodies<'_> {
        ChildBodies {
            tables: self,
            node,
            front: 0,
            back: node_body_count(self, node),
        }
    }
}

/// The number of child bodies of `node`.
fn node_body_count(tables: &Tables, node: ControlNode) -> usize {
    match node {
        ControlNode::Block(_)
        | ControlNode::Break { .. }
        | ControlNode::Continue { .. }
        | ControlNode::Return(_)
        | ControlNode::Goto { .. } => 0,
        ControlNode::Sequence(_)
        | ControlNode::Loop { .. }
        | ControlNode::Label { .. }
        | ControlNode::Guarded { .. } => 1,
        ControlNode::IfThenElse { .. } => 2,
        ControlNode::Switch { cases, .. } => cases.len() + 1,
        ControlNode::TryCatch { handlers, .. } => tables.handlers(handlers).len() + 2,
    }
}

/// A control tree: the structured form of a [`Cfg`](crate::Cfg).
///
/// The tree is a fact about the CFG it was computed from, like
/// [`LoopNestingTree`](crate::LoopNestingTree) or
/// [`DominatorTree`](crate::DominatorTree). It holds no instruction: leaves and
/// references name [`BlockId`](crate::BlockId)s, [`EdgeId`]s,
/// [`RegionId`](crate::RegionId)s, and, for a guard, a
/// [`ProgramPoint`](crate::ProgramPoint). The consumer maps a leaf back to its
/// own instructions through the CFG, so the tree needs no copy of the
/// instruction type and never outlives or borrows the CFG.
///
/// # Storage
///
/// The nodes live in one flat arena addressed by [`ControlNodeId`]. A node
/// refers to its children through [`ControlBody`] runs, and to the other
/// things it stores — switch cases, case edges, handlers, condition spans —
/// through typed [`ControlRun`]s. Nodes are allocated children first, so
/// every child has a lower handle than its parent and the [`root`](Self::root)
/// has the highest.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, ControlNode, ControlTree, EdgeKind};
///
/// let mut cfg = Cfg::<&str>::new();
/// let then_block = cfg.new_block();
/// let merge = cfg.new_block();
/// cfg.block_mut(cfg.entry()).push("test");
/// cfg.block_mut(then_block).push("then");
/// cfg.block_mut(merge).push("merge");
/// cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
/// cfg.add_edge(cfg.entry(), merge, EdgeKind::ConditionalFalse);
/// cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
///
/// let tree = ControlTree::compute(&cfg);
/// let ControlNode::Sequence(body) = *tree.node(tree.root()) else {
///     panic!("an if followed by its merge is a sequence");
/// };
/// let statements = tree.body(body);
/// assert!(matches!(tree.node(statements[0]), ControlNode::IfThenElse { .. }));
/// assert!(matches!(tree.node(statements[1]), ControlNode::Return(_) | ControlNode::Block(_)));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlTree {
    tables: Tables,
    pub(super) root: ControlNodeId,
}

impl ControlTree {
    /// The root node: the whole function body. A function with one statement
    /// has that statement as its root; any other body is a
    /// [`ControlNode::Sequence`].
    #[must_use]
    pub const fn root(&self) -> ControlNodeId {
        self.root
    }

    /// The node `id` names.
    ///
    /// # Panics
    ///
    /// Panics when `id` is not a node of this tree.
    #[must_use]
    pub fn node(&self, id: ControlNodeId) -> &ControlNode {
        self.tables.node(id)
    }

    /// The number of nodes in the tree. Every node is reachable from the
    /// [`root`](Self::root).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.tables.nodes.len()
    }

    /// The handle of every node, in allocation order: children before their
    /// parent.
    pub fn node_ids(&self) -> impl ExactSizeIterator<Item = ControlNodeId> + use<> {
        (0..self.tables.nodes.len()).map(ControlNodeId::from_index)
    }

    /// The statements of a body, in execution order.
    #[must_use]
    pub fn body(&self, body: ControlBody) -> &[ControlNodeId] {
        self.tables.body(body)
    }

    /// The case arms of a switch, in dispatch order.
    #[must_use]
    pub fn cases(&self, cases: ControlRun<SwitchCase>) -> &[SwitchCase] {
        self.tables.cases(cases)
    }

    /// The handler arms of a try region, in declaration order.
    #[must_use]
    pub fn handlers(&self, handlers: ControlRun<CatchHandler>) -> &[CatchHandler] {
        self.tables.handlers(handlers)
    }

    /// The edges of a run: a switch case's dispatch edges.
    #[must_use]
    pub fn edges(&self, edges: ControlRun<EdgeId>) -> &[EdgeId] {
        self.tables.edges(edges)
    }

    /// The spans of a run: a pre-tested loop's condition chain.
    #[must_use]
    pub fn spans(&self, spans: ControlRun<BlockSpan>) -> &[BlockSpan] {
        self.tables.spans(spans)
    }

    /// Whether the tree has no statement: the root is an empty sequence.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        matches!(
            *self.node(self.root),
            ControlNode::Sequence(body) if body.is_empty()
        )
    }

    /// The child bodies of `node`, in execution order.
    ///
    /// Leaf nodes yield nothing. `Switch` yields each case arm and then the
    /// default; `TryCatch` yields the protected body, each handler body, and
    /// then finally. A consumer walking or folding the tree uses this instead
    /// of matching every variant, so a new node shape is picked up
    /// automatically.
    ///
    /// # Panics
    ///
    /// Panics when `id` is not a node of this tree.
    #[must_use]
    pub fn child_bodies(&self, id: ControlNodeId) -> ChildBodies<'_> {
        self.tables.child_bodies(*self.node(id))
    }
}

/// The child bodies of one node, in execution order.
///
/// Returned by [`ControlTree::child_bodies`].
#[derive(Debug, Clone)]
pub struct ChildBodies<'tree> {
    tables: &'tree Tables,
    node: ControlNode,
    front: usize,
    back: usize,
}

impl ChildBodies<'_> {
    fn body_at(&self, index: usize) -> ControlBody {
        match self.node {
            ControlNode::Sequence(body)
            | ControlNode::Loop { body, .. }
            | ControlNode::Label { body, .. }
            | ControlNode::Guarded { body, .. } => body,
            ControlNode::IfThenElse {
                then_body,
                else_body,
                ..
            } => [then_body, else_body][index],
            ControlNode::Switch {
                cases,
                default_body,
                ..
            } => self
                .tables
                .cases(cases)
                .get(index)
                .map_or(default_body, |case| case.body),
            ControlNode::TryCatch {
                try_body,
                handlers,
                finally_body,
            } => {
                let handlers = self.tables.handlers(handlers);
                match index {
                    0 => try_body,
                    _ => handlers
                        .get(index - 1)
                        .map_or(finally_body, |handler| handler.body),
                }
            }
            ControlNode::Block(_)
            | ControlNode::Break { .. }
            | ControlNode::Continue { .. }
            | ControlNode::Return(_)
            | ControlNode::Goto { .. } => unreachable!("a leaf node has no child body"),
        }
    }
}

impl Iterator for ChildBodies<'_> {
    type Item = ControlBody;

    fn next(&mut self) -> Option<ControlBody> {
        if self.front == self.back {
            return None;
        }
        let body = self.body_at(self.front);
        self.front += 1;
        Some(body)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back - self.front;
        (remaining, Some(remaining))
    }
}

impl DoubleEndedIterator for ChildBodies<'_> {
    fn next_back(&mut self) -> Option<ControlBody> {
        if self.front == self.back {
            return None;
        }
        self.back -= 1;
        Some(self.body_at(self.back))
    }
}

impl ExactSizeIterator for ChildBodies<'_> {}
