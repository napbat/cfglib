//! The iteration order of a solve, the bounding at loop heads, and the
//! worklist loops that the node and edge solver cores share.

extern crate alloc;
use alloc::vec::Vec;
use core::ops::Range;

use super::bits::DenseBits;
use super::solve::{IterationOrder, SolveConfig, TrySolveError, count_step};
use super::worklist::{Queue, Ranked, Worklist};
use crate::graph::traverse::postorder_in;
use crate::graph::view::Direction;
use crate::graph::view::{DenseId, GraphView};
use crate::graph::weak_topological_order::WeakTopologicalOrder;

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

/// One solve in progress, as the shared worklist loops drive it: the node
/// solver and the edge solver each implement it over their own facts.
pub(super) trait Visit<N> {
    /// The problem's error.
    type Error;

    /// Whether the solve calls [`finish`](Self::finish) as it leaves each
    /// outermost element of its weak topological order.
    const FINISHES: bool;

    /// The node bound of the graph being solved.
    fn node_bound(&self) -> usize;

    /// Recomputes `node` and queues what its changed facts flow to. In
    /// [`Ordered`] mode, a `bound` widens or narrows a fact flowing in that
    /// differs from the one the node holds, and a node before position
    /// `floor` of the iteration order is not queued.
    fn visit<M: Mode, Q: Queue>(
        &mut self,
        mode: M,
        queue: &mut Q,
        floor: usize,
        node: N,
        bound: Option<Bound>,
    ) -> Result<Changed, Self::Error>;

    /// Every node before position `end` of `order` holds its final facts,
    /// and no visit reads the facts of what is final unless a node at or
    /// after `end` reads it.
    fn finish(&mut self, order: &WeakTopologicalOrder<N>, end: usize);
}

/// Runs `worklist` to a fixpoint in the order `config` selects, on the
/// solve that `start` makes once that order is known, and returns the solve
/// with the number of steps it took.
pub(super) fn drive<G, V>(
    graph: &G,
    worklist: Worklist,
    entry: Option<G::NodeId>,
    direction: Direction,
    config: SolveConfig,
    start: impl FnOnce(Option<&WeakTopologicalOrder<G::NodeId>>) -> V,
) -> Result<(V, usize), TrySolveError<V::Error>>
where
    G: GraphView,
    V: Visit<G::NodeId>,
{
    match schedule(graph, worklist, entry, direction, config.order) {
        Schedule::Index(mut worklist) => {
            let mut solve = start(None);
            let steps = run_plain(&mut solve, &mut worklist, config)?;
            Ok((solve, steps))
        }
        Schedule::Ranked(mut ranked) => {
            let mut solve = start(None);
            let steps = run_plain(&mut solve, &mut ranked, config)?;
            Ok((solve, steps))
        }
        Schedule::Weak(mut ranked, order) => {
            let mut solve = start(Some(&order));
            let steps = run_ordered(&mut solve, &mut ranked, &order, config)?;
            Ok((solve, steps))
        }
    }
}

/// The worklist loop of a solve that never bounds a fact: the index or
/// reverse-postorder order. A solve whose store finishes elements runs in a
/// weak topological order, so none comes here.
fn run_plain<N: DenseId, V: Visit<N>>(
    solve: &mut V,
    queue: &mut impl Queue,
    config: SolveConfig,
) -> Result<usize, TrySolveError<V::Error>> {
    let mut steps = 0;
    while let Some(node_index) = queue.pop() {
        count_step(config.max_steps, &mut steps, node_index)?;
        solve
            .visit(Plain, queue, 0, N::from_index(node_index), None)
            .map_err(TrySolveError::Problem)?;
    }
    Ok(steps)
}

/// The worklist loop of a solve in a weak topological order: it widens at
/// heads, narrows each outermost component once it has left it, and
/// finishes each outermost element as it leaves it.
///
/// Out of line, so that the default solve's entry point holds only the plain
/// loop.
#[inline(never)]
fn run_ordered<N: DenseId, V: Visit<N>>(
    solve: &mut V,
    queue: &mut Ranked,
    order: &WeakTopologicalOrder<N>,
    config: SolveConfig,
) -> Result<usize, TrySolveError<V::Error>> {
    let narrowing = config.narrowing_passes > 0;
    let mut leaving = (V::FINISHES || narrowing).then(|| Leaving::new(order, V::FINISHES));
    let mut visited = narrowing.then(|| DenseBits::new(solve.node_bound()));
    let mut steps = 0;
    loop {
        let popped = queue.pop();
        if let Some(element) = leaving.as_mut().and_then(|leaving| leaving.leave(popped)) {
            if let Some(node_index) = popped {
                queue.push(node_index);
            }
            if let Some(visited) = &visited
                && order.heads_at(element.start)
            {
                let sweep = Sweep {
                    order,
                    component: element.clone(),
                    visited,
                };
                narrow(solve, &sweep, queue, config, &mut steps)?;
            }
            solve.finish(order, element.end);
            continue;
        }
        let Some(node_index) = popped else {
            break;
        };
        count_step(config.max_steps, &mut steps, node_index)?;
        let node = N::from_index(node_index);
        let bound = order.is_head(node).then_some(Bound::Widen);
        solve
            .visit(Ordered, queue, 0, node, bound)
            .map_err(TrySolveError::Problem)?;
        if let Some(visited) = &mut visited {
            visited.insert(node_index);
        }
    }
    if V::FINISHES {
        solve.finish(order, order.order().len());
    }
    Ok(steps)
}

/// Runs the descending passes over a component the solve has left: each node
/// of it that the solve visited, once per pass in order, narrowing at heads,
/// until a pass changes nothing or the configured passes are spent. A changed
/// output re-queues only the nodes past the component, so nothing in it
/// widens again.
fn narrow<N: DenseId, V: Visit<N>>(
    solve: &mut V,
    sweep: &Sweep<'_, N>,
    queue: &mut Ranked,
    config: SolveConfig,
    steps: &mut usize,
) -> Result<(), TrySolveError<V::Error>> {
    let floor = sweep.component.end;
    for _ in 0..config.narrowing_passes {
        let mut changed = false;
        for node in sweep.nodes() {
            count_step(config.max_steps, steps, node.index())?;
            let bound = sweep.order.is_head(node).then_some(Bound::Narrow);
            let visit = solve
                .visit(Ordered, queue, floor, node, bound)
                .map_err(TrySolveError::Problem)?;
            changed |= visit.bounded || visit.output;
        }
        if !changed {
            break;
        }
    }
    Ok(())
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
struct Leaving<'o, N> {
    order: &'o WeakTopologicalOrder<N>,
    /// Whether a loose node at the top level counts as an element, or only a
    /// component does.
    every: bool,
    current: Option<Range<usize>>,
}

impl<'o, N: DenseId> Leaving<'o, N> {
    const fn new(order: &'o WeakTopologicalOrder<N>, every: bool) -> Self {
        Self {
            order,
            every,
            current: None,
        }
    }

    /// Notes the node the worklist popped, `None` once it ran dry, and
    /// returns the positions of the outermost element that the solve has
    /// just left, if any.
    fn leave(&mut self, popped: Option<usize>) -> Option<Range<usize>> {
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
struct Sweep<'a, N> {
    order: &'a WeakTopologicalOrder<N>,
    component: Range<usize>,
    visited: &'a DenseBits,
}

impl<N: DenseId> Sweep<'_, N> {
    /// The component's visited nodes, in order; a seeded solve never
    /// narrows a node it did not reach.
    fn nodes(&self) -> impl Iterator<Item = N> + '_ {
        self.order.order()[self.component.clone()]
            .iter()
            .copied()
            .filter(|node| self.visited.get(node.index()))
    }
}

/// The queue a solve runs on, in the iteration order that its config selects.
enum Schedule<N> {
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
fn schedule<G: GraphView>(
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
        let postorder = postorder_in(graph, roots, direction, &mut Vec::new());
        let ranked = Ranked::new(&worklist, bound, postorder.into_iter().rev());
        return Schedule::Ranked(ranked);
    }
    let order = WeakTopologicalOrder::from_roots(graph, roots, direction);
    let ranked = Ranked::new(&worklist, bound, order.order().iter().copied());
    Schedule::Weak(ranked, order)
}
