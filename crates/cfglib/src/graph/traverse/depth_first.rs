//! The depth-first walk core every depth-first walk of the crate runs.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use super::Steps;
use crate::graph::epoch::EpochMarks;
use crate::graph::search::Visit;
use crate::graph::view::GraphView;

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

/// One open node on the depth-first stack.
///
/// A frame's pending steps live in the walk's single arena rather than in a
/// `Vec` of their own: frames are pushed and popped strictly last in, first
/// out, so the frame on top always owns the arena's tail. Each run is stored
/// reversed and consumed by popping, so the arena holds only steps not yet
/// examined and a frame needs no cursor: it is exhausted when the arena is
/// back at its start. One allocation for the walk, instead of one per
/// expanded node, and two words per open node on a path that can run the
/// length of the graph.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Frame<N> {
    node: N,
    /// Where this frame's pending steps start in the arena.
    start: usize,
}

/// The buffers of a [`depth_first_core`] walk: the frame stack and the arena
/// of their steps, holding the walk's [`Steps`] representation (`N` for a
/// node, `S` for a step). A scratch that outlives one graph holds the dense
/// indices of [`Indexed`](super::Indexed), so one instance serves every graph.
#[derive(Debug, Clone)]
pub(crate) struct DepthFirstStack<N, S> {
    arena: Vec<S>,
    frames: Vec<Frame<N>>,
}

impl<N, S> Default for DepthFirstStack<N, S> {
    fn default() -> Self {
        Self {
            arena: Vec::new(),
            frames: Vec::new(),
        }
    }
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

/// The callbacks of a [`depth_first_core`] walk over nodes `N` and steps `S`.
/// Each can end the walk with [`ControlFlow::Break`].
pub(crate) trait DepthFirstVisitor<N, S, B> {
    /// The walk entered `node` at `depth` (0 for a seed). [`Visit::Skip`]
    /// leaves its steps unexamined and finishes it at once.
    fn discover(&mut self, node: N, depth: usize) -> ControlFlow<B, Visit>;

    /// The walk examined a step from one node to another, which it
    /// classified as `kind`. A tree step runs before the discovery of its
    /// target.
    #[inline]
    fn step(&mut self, _from: N, _step: S, _to: N, _kind: StepKind) -> ControlFlow<B> {
        ControlFlow::Continue(())
    }

    /// Every step out of a node has been examined and its subtree is
    /// complete.
    #[inline]
    fn finish(&mut self, _node: N) -> ControlFlow<B> {
        ControlFlow::Continue(())
    }
}

/// A closure is a visitor that only discovers.
impl<N, S, B, F: FnMut(N, usize) -> ControlFlow<B, Visit>> DepthFirstVisitor<N, S, B> for F {
    #[inline]
    fn discover(&mut self, node: N, depth: usize) -> ControlFlow<B, Visit> {
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
    seeds: impl IntoIterator<Item = S::Node>,
    marks: &mut M,
    stack: &mut DepthFirstStack<S::Node, S::Step>,
    visitor: &mut impl DepthFirstVisitor<S::Node, S::Step, B>,
) -> Option<B>
where
    G: GraphView,
    S: Steps<G>,
    M: DepthFirstMarks + ?Sized,
{
    let DepthFirstStack { arena, frames } = stack;
    arena.clear();
    frames.clear();
    // Sized for the deepest walk the graph allows up front: growing the
    // stack by doubling copied it a dozen times on a 4,096-node chain and
    // cost a fifth of the walk. The pages stay untouched until a walk that
    // deep uses them, and a reused scratch reserves only once.
    frames.reserve(graph.node_bound());
    arena.reserve(graph.node_bound());

    macro_rules! emit {
        ($callback:expr) => {
            if let ControlFlow::Break(value) = $callback {
                return Some(value);
            }
        };
    }

    // Enter a node at a depth: mark it and discover it, and either hand back
    // its frame, with its steps reversed on the arena's tail, or finish it at
    // once.
    macro_rules! enter {
        ($node:expr, $depth:expr) => {{
            let node = $node;
            marks.open(S::index(node));
            match visitor.discover(node, $depth) {
                ControlFlow::Break(value) => return Some(value),
                ControlFlow::Continue(Visit::Skip) => {
                    marks.close(S::index(node));
                    emit!(visitor.finish(node));
                    None
                }
                ControlFlow::Continue(Visit::Descend) => {
                    let start = arena.len();
                    arena.extend(steps.steps(graph, node));
                    arena[start..].reverse();
                    Some(Frame { node, start })
                }
            }
        }};
    }

    for seed in seeds {
        if marks.mark(S::index(seed)) != Mark::Unseen {
            continue;
        }
        let Some(mut top) = enter!(seed, 0) else {
            continue;
        };
        // The open frame lives in locals and only its ancestors on the stack.
        // The depth of a child is the number of frames above the seed, plus
        // one.
        loop {
            if arena.len() > top.start {
                let Some(out) = arena.pop() else { break };
                let to = steps.target(graph, out);
                match marks.mark(S::index(to)) {
                    Mark::Unseen => {
                        emit!(visitor.step(top.node, out, to, StepKind::Tree));
                        if let Some(child) = enter!(to, frames.len() + 1) {
                            frames.push(top);
                            top = child;
                        }
                    }
                    Mark::Open => emit!(visitor.step(top.node, out, to, StepKind::Back)),
                    Mark::Done => {
                        emit!(visitor.step(top.node, out, to, StepKind::ForwardOrCross));
                    }
                }
                continue;
            }
            marks.close(S::index(top.node));
            emit!(visitor.finish(top.node));
            let Some(parent) = frames.pop() else { break };
            top = parent;
        }
    }
    None
}
