//! The iteration order of a solve and the bounding at loop heads that the
//! node and edge solver cores share.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::Range;

use super::bits::DenseBits;
use super::fixpoint::IterationOrder;
use super::worklist::{Ranked, Worklist};
use crate::graph::view::Direction;
use crate::graph::view::{DenseId, GraphView};
use crate::graph::weak_topological_order::{WeakTopologicalOrder, depth_first_forest};

/// Which operator a loop head applies to the fact flowing into it.
#[derive(Debug, Clone, Copy)]
pub(super) enum Bound {
    /// The problem's `widen`, while the solve ascends.
    Widen,
    /// The problem's `narrow`, during a descending pass.
    Narrow,
}

/// What a visit changed: whether bounding moved the fact entering a head, and
/// whether the node's output moved.
#[derive(Debug, Clone, Copy)]
pub(super) struct Changed {
    pub(super) bounded: bool,
    pub(super) output: bool,
}

/// How a solver core runs, as a type, so that the core that never bounds or
/// narrows compiles without any of it: [`Plain`] for an index or
/// reverse-postorder solve, [`Ordered`] for a solve in a weak topological
/// order.
pub(super) trait Mode: Copy {
    /// Whether the solve widens and narrows at heads and requeues only past
    /// the component it narrows.
    const ORDERED: bool;
}

/// The mode of a solve that never bounds a fact.
#[derive(Debug, Clone, Copy)]
pub(super) struct Plain;

impl Mode for Plain {
    const ORDERED: bool = false;
}

/// The mode of a solve in a weak topological order.
#[derive(Debug, Clone, Copy)]
pub(super) struct Ordered;

impl Mode for Ordered {
    const ORDERED: bool = true;
}

/// The outermost element of the weak topological order that a solve is in,
/// so that it can narrow each component, and let go of the facts of each
/// element, as soon as it has left it.
///
/// Once the worklist pops a node past the end of an outermost element,
/// nothing in that element is ever queued again: an edge into it either
/// runs forward from an earlier node, which is not queued either, or closes a
/// loop from inside it. The element is then final, and only nodes past it
/// read what narrowing changes.
pub(super) struct Leaving<'o, N> {
    order: &'o WeakTopologicalOrder<N>,
    /// Whether a loose node at the top level counts as an element, or only a
    /// component does.
    every: bool,
    current: Option<Range<usize>>,
}

impl<'o, N: DenseId> Leaving<'o, N> {
    pub(super) const fn new(order: &'o WeakTopologicalOrder<N>, every: bool) -> Self {
        Self {
            order,
            every,
            current: None,
        }
    }

    /// Notes the node the worklist popped, `None` once it ran dry, and
    /// returns the positions of the outermost element that the solve has
    /// just left, if any.
    pub(super) fn leave(&mut self, popped: Option<usize>) -> Option<Range<usize>> {
        let position = popped.map(|index| {
            let rank = self.order.rank(N::from_index(index));
            rank.expect("the order covers every node a solve queues") as usize
        });
        if let Some(current) = &self.current
            && position.is_none_or(|position| position >= current.end)
        {
            return self.current.take();
        }
        if self.current.is_none()
            && let Some(position) = position
        {
            let element = self.order.outermost(position);
            self.current = (self.every || self.order.heads_at(element.start)).then_some(element);
        }
        None
    }
}

/// An outermost component to narrow: where it lies in the order, and which
/// nodes the ascending solve visited.
pub(super) struct Sweep<'a, N> {
    pub(super) order: &'a WeakTopologicalOrder<N>,
    pub(super) component: Range<usize>,
    pub(super) visited: &'a DenseBits,
}

impl<N: DenseId> Sweep<'_, N> {
    /// The component's visited nodes, in order; a seeded solve never
    /// narrows a node it did not reach.
    pub(super) fn nodes(&self) -> impl Iterator<Item = N> + '_ {
        self.order.order()[self.component.clone()]
            .iter()
            .copied()
            .filter(|node| self.visited.get(node.index()))
    }
}

/// The queue a solve runs on, in the iteration order that its config selects.
pub(super) enum Schedule<N> {
    /// Index order: the worklist as the entry point built it.
    Index(Worklist),
    /// Reverse postorder: a ranked queue, with nothing to bound.
    Ranked(Ranked),
    /// A weak topological order: a ranked queue, and the order the solve
    /// widens and narrows by.
    Weak(Ranked, WeakTopologicalOrder<N>),
}

/// Puts `worklist` in the iteration order that `order` selects.
///
/// The order walks along `direction`, first from `entry` when it is queued,
/// then from every queued node with no upstream neighbor, then from the
/// lowest-index queued node it has not reached, until every queued node is
/// ordered. It covers exactly what a solve can queue: the queued nodes and
/// everything they reach. [`IterationOrder::Index`] keeps the worklist as it
/// is.
pub(super) fn schedule<G: GraphView>(
    graph: &G,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    direction: Direction,
    order: IterationOrder,
) -> Schedule<G::NodeId> {
    if order == IterationOrder::Index {
        return Schedule::Index(worklist);
    }
    let entry = entry.filter(|node| worklist.contains(node.index()));
    let queued: Vec<G::NodeId> = worklist.queued().map(G::NodeId::from_index).collect();
    let is_source = |node: &G::NodeId| match direction {
        Direction::Forward => graph.predecessors(*node).next().is_none(),
        Direction::Backward => graph.successors(*node).next().is_none(),
    };
    let roots = entry
        .into_iter()
        .chain(queued.iter().copied().filter(is_source))
        .chain(queued.iter().copied());
    let bound = graph.node_bound();
    if order == IterationOrder::ReversePostorder {
        let forest = depth_first_forest(graph, roots, direction);
        let ranked = Ranked::new(&worklist, bound, forest.reverse_postorder::<G::NodeId>());
        return Schedule::Ranked(ranked);
    }
    let order = WeakTopologicalOrder::from_roots(graph, roots, direction);
    let ranked = Ranked::new(&worklist, bound, order.order().iter().copied());
    Schedule::Weak(ranked, order)
}
