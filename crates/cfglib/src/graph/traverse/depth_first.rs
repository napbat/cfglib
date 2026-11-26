//! The depth-first walk core every depth-first walk of the crate runs.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::Steps;
use crate::graph::epoch::EpochMarks;
use crate::graph::search::Visit;
use crate::graph::view::{DenseId, GraphView};

/// Where a node stands in a depth-first walk when a step reaches it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mark {
    /// Not entered: a step to it is a tree edge.
    Unseen,
    /// Entered and not yet finished, so on the current path: a step to it is
    /// a back edge.
    Open,
    /// Finished: a step to it is a forward or cross edge.
    Done,
}

/// The marks a [`depth_first_core`] walk keeps, by dense node index.
///
/// The policy of a walk lives here rather than in the core: marks that
/// remember a finished node visit each node once, marks that forget it on
/// [`close`](Self::close) visit a node once per path, and a table of [`Mark`]s
/// tells back edges from forward and cross edges.
pub(crate) trait DepthFirstMarks {
    /// The mark of the node at `index`.
    fn mark(&self, index: usize) -> Mark;
    /// The walk entered the node at `index`.
    fn open(&mut self, index: usize);
    /// The walk finished the node at `index`.
    fn close(&mut self, index: usize);
}

/// Visited flags: a node is either unseen or seen, and seen for good.
impl DepthFirstMarks for [bool] {
    #[inline]
    fn mark(&self, index: usize) -> Mark {
        if self[index] {
            Mark::Done
        } else {
            Mark::Unseen
        }
    }

    #[inline]
    fn open(&mut self, index: usize) {
        self[index] = true;
    }

    #[inline]
    fn close(&mut self, _index: usize) {}
}

/// Epoch marks held for the whole walk, like visited flags.
impl DepthFirstMarks for EpochMarks {
    #[inline]
    fn mark(&self, index: usize) -> Mark {
        if self.is_marked(index) {
            Mark::Done
        } else {
            Mark::Unseen
        }
    }

    #[inline]
    fn open(&mut self, index: usize) {
        EpochMarks::mark(self, index);
    }

    #[inline]
    fn close(&mut self, _index: usize) {}
}

/// The three marks themselves, for a walk that classifies every edge.
impl DepthFirstMarks for [Mark] {
    #[inline]
    fn mark(&self, index: usize) -> Mark {
        self[index]
    }

    #[inline]
    fn open(&mut self, index: usize) {
        self[index] = Mark::Open;
    }

    #[inline]
    fn close(&mut self, index: usize) {
        self[index] = Mark::Done;
    }
}

/// Epoch marks held only while a node is on the current path, so a walk
/// enters a node once per path that reaches it and refuses only a node that
/// would close a cycle.
pub(crate) struct PathMarks<'a>(pub(crate) &'a mut EpochMarks);

impl DepthFirstMarks for PathMarks<'_> {
    #[inline]
    fn mark(&self, index: usize) -> Mark {
        if self.0.is_marked(index) {
            Mark::Open
        } else {
            Mark::Unseen
        }
    }

    #[inline]
    fn open(&mut self, index: usize) {
        self.0.mark(index);
    }

    #[inline]
    fn close(&mut self, index: usize) {
        self.0.unmark(index);
    }
}

/// One node on the depth-first stack.
///
/// A frame's steps live in the walk's single arena rather than in a `Vec` of
/// their own: frames are pushed and popped strictly last in, first out, so
/// the frame on top always owns the arena's tail and a pop truncates the
/// arena back to where its steps began. One allocation for the walk, instead
/// of one per expanded node. A frame's depth is its position on the stack.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Frame {
    node: usize,
    /// Where this frame's steps start in the arena; the pop truncates to it.
    start: usize,
    /// The next step to examine, as an arena index. The steps are exhausted
    /// when it reaches the arena's length, since the top frame's run ends
    /// there.
    cursor: usize,
}

/// The buffers of a [`depth_first_core`] walk: the frame stack and the arena
/// of their steps, both dense indices, so one instance serves every graph.
#[derive(Debug, Clone, Default)]
pub(crate) struct DepthFirstStack {
    arena: Vec<usize>,
    frames: Vec<Frame>,
}

/// The classification of a step a [`depth_first_core`] walk examined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StepKind {
    /// The step entered its target: an edge of the depth-first tree.
    Tree,
    /// The step reached a node on the current path, closing a cycle.
    Back,
    /// The step reached a finished node: a forward or a cross edge.
    ForwardOrCross,
}

/// The callbacks of a [`depth_first_core`] walk. Each can end the walk with
/// [`ControlFlow::Break`].
pub(crate) trait DepthFirstVisitor<B> {
    /// The walk entered `node` at `depth` (0 for a seed). [`Visit::Skip`]
    /// leaves its steps unexamined and finishes it at once.
    fn discover(&mut self, node: usize, depth: usize) -> ControlFlow<B, Visit>;

    /// The walk examined a step from one node to another, which it
    /// classified as `kind`. A tree step runs before the discovery of its
    /// target.
    #[inline]
    fn step(&mut self, _from: usize, _step: usize, _to: usize, _kind: StepKind) -> ControlFlow<B> {
        ControlFlow::Continue(())
    }

    /// Every step out of a node has been examined and its subtree is
    /// complete.
    #[inline]
    fn finish(&mut self, _node: usize) -> ControlFlow<B> {
        ControlFlow::Continue(())
    }
}

/// A closure is a visitor that only discovers.
impl<B, F: FnMut(usize, usize) -> ControlFlow<B, Visit>> DepthFirstVisitor<B> for F {
    #[inline]
    fn discover(&mut self, node: usize, depth: usize) -> ControlFlow<B, Visit> {
        self(node, depth)
    }
}

/// Walk `graph` depth-first from each of `seeds` in turn, and return the
/// value a callback broke with.
///
/// A seed whose mark is not [`Mark::Unseen`] when its turn comes is passed
/// over, so seeds share the marks of the walks before them: under marks that
/// remember finished nodes the walk visits each node once overall, as a
/// depth-first forest. Steps are examined in adjacency order, so the walk
/// visits in the order a recursive depth-first search would.
///
/// The stack is emptied on entry, so a walk never inherits a frame, including
/// from one that broke off.
#[inline]
pub(crate) fn depth_first_core<G, S, M, B>(
    steps: S,
    graph: &G,
    seeds: impl IntoIterator<Item = usize>,
    marks: &mut M,
    stack: &mut DepthFirstStack,
    visitor: &mut impl DepthFirstVisitor<B>,
) -> Option<B>
where
    G: GraphView,
    S: Steps<G>,
    M: DepthFirstMarks + ?Sized,
{
    let DepthFirstStack { arena, frames } = stack;
    arena.clear();
    frames.clear();

    macro_rules! emit {
        ($callback:expr) => {
            if let ControlFlow::Break(value) = $callback {
                return Some(value);
            }
        };
    }

    // Enter a node at a depth: mark it, discover it, and either push its
    // frame with its steps on the arena's tail or finish it at once.
    macro_rules! enter {
        ($node:expr, $depth:expr) => {{
            let node = $node;
            marks.open(node);
            match visitor.discover(node, $depth) {
                ControlFlow::Break(value) => return Some(value),
                ControlFlow::Continue(Visit::Skip) => {
                    marks.close(node);
                    emit!(visitor.finish(node));
                }
                ControlFlow::Continue(Visit::Descend) => {
                    let start = arena.len();
                    arena.extend(steps.steps(graph, G::NodeId::from_index(node)));
                    frames.push(Frame {
                        node,
                        start,
                        cursor: start,
                    });
                }
            }
        }};
    }

    for seed in seeds {
        if marks.mark(seed) != Mark::Unseen {
            continue;
        }
        enter!(seed, 0);
        // Read frames through `last_mut` and copy the fields out; no frame is
        // ever cloned, and the depth of a child is the stack's length.
        while let Some(frame) = frames.last_mut() {
            if frame.cursor < arena.len() {
                let out = arena[frame.cursor];
                frame.cursor += 1;
                let from = frame.node;
                let to = steps.target(graph, out);
                match marks.mark(to) {
                    Mark::Unseen => {
                        emit!(visitor.step(from, out, to, StepKind::Tree));
                        let depth = frames.len();
                        enter!(to, depth);
                    }
                    Mark::Open => emit!(visitor.step(from, out, to, StepKind::Back)),
                    Mark::Done => emit!(visitor.step(from, out, to, StepKind::ForwardOrCross)),
                }
                continue;
            }
            let Some(finished) = frames.pop() else { break };
            arena.truncate(finished.start);
            marks.close(finished.node);
            emit!(visitor.finish(finished.node));
        }
    }
    None
}
