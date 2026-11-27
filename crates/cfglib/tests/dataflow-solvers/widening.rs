//! Widening and narrowing at the heads of a weak topological order, with an
//! interval domain over `i = 0; while i < n { i += 1 }`: an infinite-height
//! lattice that the index order cannot finish and the weak topological order
//! finishes in a few steps, through every solver and problem trait.

use std::convert::Infallible;

use cfglib::{
    BlockId, Cfg, DenseId, Direction, EdgeKind, EdgeProblem, Graph, GraphView, IterationOrder,
    NodeId, NodeProblem, SolveConfig, SolveError, TryEdgeProblem, TryNodeProblem, TrySolveError,
    meet_options, solve_edge_problem, solve_node_problem,
};

use crate::graphs::{live, rooted, seeded, step_limit};

/// The values `i` may hold; `i64::MIN` and `i64::MAX` stand for the
/// unbounded ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Interval {
    low: i64,
    high: i64,
}

/// An interval, or `None` where no execution arrives.
type Range = Option<Interval>;

/// The values from `low` to `high`; none at all when `low > high`, which is
/// where a guard rules out every value that arrives.
fn range(low: i64, high: i64) -> Range {
    (low <= high).then_some(Interval { low, high })
}

/// The loop bound `n`: far more iterations than a step limit allows.
const N: i64 = 1_000_000;

/// The nodes of the loop in both graphs.
const INIT: usize = 0;
const HEAD: usize = 1;
const BODY: usize = 2;
const EXIT: usize = 3;

/// `i = 0` at `INIT`, the test `i < n` at `HEAD`, `i += 1` in `BODY` under
/// `i < n`, and `EXIT` under `i >= n`. Backward, the number of increments
/// left before the exit.
struct Counter {
    bound: i64,
    direction: Direction,
}

impl Counter {
    /// The forward count up to `N`.
    const fn forward() -> Self {
        Self::up_to(N)
    }

    const fn up_to(bound: i64) -> Self {
        Self {
            bound,
            direction: Direction::Forward,
        }
    }

    const fn backward() -> Self {
        Self {
            bound: N,
            direction: Direction::Backward,
        }
    }

    fn direction(&self) -> Direction {
        self.direction
    }

    fn join(left: &Range, right: &Range) -> Range {
        meet_options(left, right, |left, right| Interval {
            low: left.low.min(right.low),
            high: left.high.max(right.high),
        })
    }

    fn increment(value: i64) -> i64 {
        if value == i64::MAX { value } else { value + 1 }
    }

    /// Forward: the interval leaving `node`. Backward: how many increments
    /// lie ahead of `node` on the way to the exit.
    fn transfer(&self, node: usize, input: &Range) -> Range {
        let interval = (*input)?;
        if self.direction == Direction::Backward {
            return Some(match node {
                BODY => Interval {
                    low: Self::increment(interval.low),
                    high: Self::increment(interval.high),
                },
                _ => interval,
            });
        }
        match node {
            INIT => range(0, 0),
            BODY => range(interval.low, interval.high.min(self.bound - 1)).map(|kept| Interval {
                low: Self::increment(kept.low),
                high: Self::increment(kept.high),
            }),
            EXIT => range(interval.low.max(self.bound), interval.high),
            _ => Some(interval),
        }
    }

    /// The fact entering the analysis: `i` unknown before `INIT`, nothing
    /// left to do at `EXIT` backward.
    fn boundary(&self) -> Range {
        match self.direction {
            Direction::Forward => range(i64::MIN, i64::MAX),
            Direction::Backward => range(0, 0),
        }
    }

    /// Jumps every bound that grew to its unbounded end.
    fn widen(previous: &Range, next: &Range) -> Range {
        match (previous, next) {
            (Some(previous), Some(next)) => Some(Interval {
                low: if next.low < previous.low {
                    i64::MIN
                } else {
                    previous.low
                },
                high: if next.high > previous.high {
                    i64::MAX
                } else {
                    previous.high
                },
            }),
            _ => *next,
        }
    }

    /// Replaces only the unbounded ends with what flows in now.
    fn narrow(previous: &Range, next: &Range) -> Range {
        match (previous, next) {
            (Some(previous), Some(next)) => Some(Interval {
                low: if previous.low == i64::MIN {
                    next.low
                } else {
                    previous.low
                },
                high: if previous.high == i64::MAX {
                    next.high
                } else {
                    previous.high
                },
            }),
            _ => *next,
        }
    }
}

impl<G: GraphView> NodeProblem<G> for Counter {
    type Fact = Range;

    fn direction(&self) -> Direction {
        self.direction()
    }

    fn bottom(&self, _graph: &G) -> Range {
        None
    }

    fn boundary(&self, _graph: &G) -> Range {
        self.boundary()
    }

    fn meet(&self, left: &Range, right: &Range) -> Range {
        Self::join(left, right)
    }

    fn transfer(&self, _graph: &G, node: G::NodeId, input: &Range) -> Range {
        self.transfer(node.index(), input)
    }

    fn widen(&self, previous: &Range, next: &Range) -> Range {
        Self::widen(previous, next)
    }

    fn narrow(&self, previous: &Range, next: &Range) -> Range {
        Self::narrow(previous, next)
    }
}

impl EdgeProblem<Cfg<()>> for Counter {
    type Fact = Range;

    fn direction(&self) -> Direction {
        self.direction()
    }

    fn bottom(&self, _cfg: &Cfg<()>) -> Range {
        None
    }

    fn boundary(&self, _cfg: &Cfg<()>, block: BlockId) -> Option<Range> {
        (block.index() == INIT).then(|| self.boundary())
    }

    fn meet(&self, left: &Range, right: &Range) -> Range {
        Self::join(left, right)
    }

    fn transfer_node(&self, _cfg: &Cfg<()>, block: BlockId, input: &Range) -> Range {
        self.transfer(block.index(), input)
    }

    fn widen(&self, previous: &Range, next: &Range) -> Range {
        Self::widen(previous, next)
    }

    fn narrow(&self, previous: &Range, next: &Range) -> Range {
        Self::narrow(previous, next)
    }
}

/// [`Counter`] through the fallible traits, whose `widen` and `narrow` also
/// see the head.
struct Fallible(Counter);

/// Asserts that only the loop head is asked to widen or narrow.
fn assert_head<V: DenseId>(node: V) {
    assert_eq!(node.index(), HEAD, "only the loop head widens or narrows");
}

impl TryNodeProblem<Graph<(), ()>> for Fallible {
    type Fact = Range;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        self.0.direction()
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> Range {
        None
    }

    fn boundary(&self, _graph: &Graph<(), ()>) -> Result<Range, Infallible> {
        Ok(self.0.boundary())
    }

    fn meet(
        &self,
        _graph: &Graph<(), ()>,
        _node: NodeId,
        left: &Range,
        right: &Range,
    ) -> Result<Range, Infallible> {
        Ok(Counter::join(left, right))
    }

    fn transfer(
        &self,
        _graph: &Graph<(), ()>,
        node: NodeId,
        input: &Range,
    ) -> Result<Range, Infallible> {
        Ok(self.0.transfer(node.index(), input))
    }

    fn widen(
        &self,
        _graph: &Graph<(), ()>,
        node: NodeId,
        previous: &Range,
        next: &Range,
    ) -> Result<Range, Infallible> {
        assert_head(node);
        Ok(Counter::widen(previous, next))
    }

    fn narrow(
        &self,
        _graph: &Graph<(), ()>,
        node: NodeId,
        previous: &Range,
        next: &Range,
    ) -> Result<Range, Infallible> {
        assert_head(node);
        Ok(Counter::narrow(previous, next))
    }
}

impl TryNodeProblem<Cfg<()>> for Fallible {
    type Fact = Range;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        self.0.direction()
    }

    fn bottom(&self, _cfg: &Cfg<()>) -> Range {
        None
    }

    fn boundary(&self, _cfg: &Cfg<()>) -> Result<Range, Infallible> {
        Ok(self.0.boundary())
    }

    fn meet(
        &self,
        _cfg: &Cfg<()>,
        _block: BlockId,
        left: &Range,
        right: &Range,
    ) -> Result<Range, Infallible> {
        Ok(Counter::join(left, right))
    }

    fn transfer(&self, _cfg: &Cfg<()>, block: BlockId, input: &Range) -> Result<Range, Infallible> {
        Ok(self.0.transfer(block.index(), input))
    }

    fn widen(
        &self,
        _cfg: &Cfg<()>,
        block: BlockId,
        previous: &Range,
        next: &Range,
    ) -> Result<Range, Infallible> {
        assert_head(block);
        Ok(Counter::widen(previous, next))
    }

    fn narrow(
        &self,
        _cfg: &Cfg<()>,
        block: BlockId,
        previous: &Range,
        next: &Range,
    ) -> Result<Range, Infallible> {
        assert_head(block);
        Ok(Counter::narrow(previous, next))
    }
}

impl TryEdgeProblem<Cfg<()>> for Fallible {
    type Fact = Range;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        self.0.direction()
    }

    fn bottom(&self, _cfg: &Cfg<()>) -> Range {
        None
    }

    fn boundary(&self, _cfg: &Cfg<()>, block: BlockId) -> Result<Option<Range>, Infallible> {
        Ok((block.index() == INIT).then(|| self.0.boundary()))
    }

    fn meet(
        &self,
        _cfg: &Cfg<()>,
        _block: BlockId,
        left: &Range,
        right: &Range,
    ) -> Result<Range, Infallible> {
        Ok(Counter::join(left, right))
    }

    fn transfer_node(
        &self,
        _cfg: &Cfg<()>,
        block: BlockId,
        input: &Range,
    ) -> Result<Range, Infallible> {
        Ok(self.0.transfer(block.index(), input))
    }

    fn widen(
        &self,
        _cfg: &Cfg<()>,
        block: BlockId,
        previous: &Range,
        next: &Range,
    ) -> Result<Range, Infallible> {
        assert_head(block);
        Ok(Counter::widen(previous, next))
    }

    fn narrow(
        &self,
        _cfg: &Cfg<()>,
        block: BlockId,
        previous: &Range,
        next: &Range,
    ) -> Result<Range, Infallible> {
        assert_head(block);
        Ok(Counter::narrow(previous, next))
    }
}

const EDGES: [(usize, usize); 4] = [(INIT, HEAD), (HEAD, BODY), (BODY, HEAD), (HEAD, EXIT)];

fn loop_graph() -> Graph<(), ()> {
    let mut graph = Graph::new();
    let nodes: Vec<NodeId> = (0..4).map(|_| graph.add_node(())).collect();
    for (source, target) in EDGES {
        graph.add_edge(nodes[source], nodes[target], ());
    }
    graph
}

fn loop_cfg() -> Cfg<()> {
    let mut cfg = Cfg::new();
    let blocks = [
        cfg.entry(),
        cfg.new_block(),
        cfg.new_block(),
        cfg.new_block(),
    ];
    for (source, target) in EDGES {
        cfg.add_edge(blocks[source], blocks[target], EdgeKind::Fallthrough);
    }
    cfg
}

fn node(index: usize) -> NodeId {
    NodeId::from_index(index)
}

fn block(index: usize) -> BlockId {
    BlockId::from_index(index)
}

const LIMIT: usize = 1_000;

/// The weak topological order, bounded by `LIMIT`, without narrowing.
fn widening() -> SolveConfig {
    SolveConfig {
        order: IterationOrder::WeakTopological,
        ..step_limit(LIMIT)
    }
}

/// [`widening`] with `passes` narrowing passes.
fn narrowing(passes: usize) -> SolveConfig {
    SolveConfig {
        narrowing_passes: passes,
        ..widening()
    }
}

#[test]
fn the_index_order_runs_out_of_steps_on_an_infinite_height_lattice() {
    let graph = loop_graph();
    for order in [IterationOrder::Index, IterationOrder::ReversePostorder] {
        let config = SolveConfig {
            order,
            ..step_limit(LIMIT)
        };
        let error = solve_node_problem(&graph, &Counter::forward(), live(config)).unwrap_err();
        assert!(matches!(
            SolveError::from(error),
            SolveError::StepLimitExceeded { limit: LIMIT, .. }
        ));
    }
}

#[test]
fn widening_at_the_loop_head_converges_to_the_unbounded_interval() {
    let graph = loop_graph();
    let facts = solve_node_problem(&graph, &Counter::forward(), live(widening())).unwrap();
    let steps = facts.stats().step_count();
    assert!(steps < 20, "converged in {steps} steps");
    assert_eq!(*facts.fact_out(node(INIT)), range(0, 0));
    assert_eq!(*facts.fact_in(node(HEAD)), range(0, i64::MAX));
    assert_eq!(*facts.fact_out(node(HEAD)), range(0, i64::MAX));
    assert_eq!(*facts.fact_out(node(BODY)), range(1, N));
    assert_eq!(*facts.fact_out(node(EXIT)), range(N, i64::MAX));
}

#[test]
fn one_narrowing_pass_recovers_the_bound_that_the_guard_implies() {
    let graph = loop_graph();
    let widened = solve_node_problem(&graph, &Counter::forward(), live(widening())).unwrap();
    let facts = solve_node_problem(&graph, &Counter::forward(), live(narrowing(1))).unwrap();
    assert_eq!(*facts.fact_in(node(HEAD)), range(0, N));
    assert_eq!(*facts.fact_out(node(BODY)), range(1, N));
    assert_eq!(*facts.fact_out(node(EXIT)), range(N, N));
    // The pass sweeps the loop's two nodes as soon as the solve leaves the
    // loop, so `EXIT` still runs once, on the narrowed fact.
    assert_eq!(facts.stats().step_count(), widened.stats().step_count() + 2);

    // A second pass finds nothing left to narrow: one more sweep of the loop,
    // and the passes stop.
    let again = solve_node_problem(&graph, &Counter::forward(), live(narrowing(5))).unwrap();
    assert_eq!(*again.fact_in(node(HEAD)), range(0, N));
    assert_eq!(again.stats().step_count(), facts.stats().step_count() + 2);
}

#[test]
fn narrowing_passes_run_only_in_the_weak_topological_order() {
    // Counting to 5 terminates without widening, at the least fixpoint, where
    // a descending pass would change nothing; the other orders skip them.
    let graph = loop_graph();
    let counter = Counter::up_to(5);
    for order in [IterationOrder::Index, IterationOrder::ReversePostorder] {
        let plain = SolveConfig {
            order,
            ..SolveConfig::default()
        };
        let exact = solve_node_problem(&graph, &counter, live(plain)).unwrap();
        assert_eq!(*exact.fact_in(node(HEAD)), range(0, 5));
        let passes = SolveConfig {
            narrowing_passes: 3,
            ..plain
        };
        let same = solve_node_problem(&graph, &counter, live(passes)).unwrap();
        assert_eq!(
            same.stats().step_count(),
            exact.stats().step_count(),
            "{order:?}"
        );
        assert_eq!(*same.fact_in(node(HEAD)), range(0, 5));
    }
    let widened = solve_node_problem(&graph, &counter, live(widening())).unwrap();
    assert_eq!(*widened.fact_in(node(HEAD)), range(0, i64::MAX));
    let narrowed = solve_node_problem(&graph, &counter, live(narrowing(1))).unwrap();
    assert_eq!(*narrowed.fact_in(node(HEAD)), range(0, 5));
}

#[test]
fn the_step_limit_also_bounds_the_narrowing_passes() {
    let graph = loop_graph();
    let needed = solve_node_problem(&graph, &Counter::forward(), live(narrowing(1)))
        .unwrap()
        .stats()
        .step_count();
    // The last three steps are the sweep of `HEAD` and `BODY`, then `EXIT`:
    // a limit two short stops the sweep before `BODY`.
    let config = SolveConfig {
        max_steps: Some(needed - 2),
        ..narrowing(1)
    };
    let error = solve_node_problem(&graph, &Counter::forward(), live(config)).unwrap_err();
    let SolveError::StepLimitExceeded {
        steps,
        pending_node,
        ..
    } = SolveError::from(error);
    assert_eq!(steps, needed - 2);
    assert_eq!(pending_node, BODY);
}

#[test]
fn a_seeded_solve_narrows_only_what_it_visited() {
    let graph = loop_graph();
    let body = [node(BODY)];
    let facts = solve_node_problem(&graph, &Counter::forward(), seeded(&body, narrowing(1)));
    let facts = facts.unwrap();
    // Only `BODY` is visited: once ascending, and once by the sweep of its
    // loop, which skips the head that the seeded solve never reached. Nothing
    // flows into `BODY`, so it transfers bottom and queues nothing.
    assert_eq!(facts.stats().step_count(), 2);
    for index in [INIT, HEAD, BODY, EXIT] {
        assert_eq!(*facts.fact_out(node(index)), None, "node {index}");
    }
}

#[test]
fn a_backward_solve_widens_the_fact_flowing_into_the_head_which_is_its_out_fact() {
    let graph = loop_graph();
    let backward = Counter::backward();
    assert!(solve_node_problem(&graph, &backward, live(step_limit(LIMIT))).is_err());
    let facts = solve_node_problem(&graph, &backward, live(widening())).unwrap();
    assert_eq!(*facts.fact_out(node(HEAD)), range(0, i64::MAX));
    assert_eq!(*facts.fact_in(node(INIT)), range(0, i64::MAX));
    assert_eq!(*facts.fact_in(node(EXIT)), range(0, 0));
}

#[test]
fn the_node_and_edge_solvers_widen_and_narrow_the_same_way_over_a_cfg() {
    let cfg = loop_cfg();
    let counter = Counter::forward();
    let fallible = Fallible(Counter::forward());
    let from_entry = |config| rooted(&cfg, Direction::Forward, config);
    let bounded = step_limit(LIMIT);
    assert!(solve_node_problem(&cfg, &counter, from_entry(bounded)).is_err());
    assert!(solve_edge_problem(&cfg, &counter, live(bounded)).is_err());
    let expected = [
        (0, range(0, i64::MAX), range(N, i64::MAX)),
        (1, range(0, N), range(N, N)),
    ];
    for (passes, head, exit) in expected {
        let config = narrowing(passes);
        let blocks = solve_node_problem(&cfg, &counter, from_entry(config)).unwrap();
        let edges = solve_edge_problem(&cfg, &counter, live(config)).unwrap();
        let try_blocks = solve_node_problem(&cfg, &fallible, from_entry(config)).unwrap();
        let try_edges = solve_edge_problem(&cfg, &fallible, live(config)).unwrap();
        let solves = [
            (
                "node",
                blocks.fact_in(block(HEAD)),
                blocks.fact_out(block(EXIT)),
            ),
            (
                "edge",
                edges.fact_in(block(HEAD)),
                edges.fact_out(block(EXIT)),
            ),
            (
                "fallible node",
                try_blocks.fact_in(block(HEAD)),
                try_blocks.fact_out(block(EXIT)),
            ),
            (
                "fallible edge",
                try_edges.fact_in(block(HEAD)),
                try_edges.fact_out(block(EXIT)),
            ),
        ];
        for (what, at_head, at_exit) in solves {
            assert_eq!(*at_head, head, "{what} solve with {passes} passes");
            assert_eq!(*at_exit, exit, "{what} solve with {passes} passes");
        }
    }
}

#[test]
fn the_fallible_node_solver_widens_and_narrows_only_at_the_head() {
    let graph = loop_graph();
    let fallible = Fallible(Counter::forward());
    let facts = solve_node_problem(&graph, &fallible, live(narrowing(1))).unwrap();
    assert_eq!(*facts.fact_in(node(HEAD)), range(0, N));
    assert!(matches!(
        solve_node_problem(&graph, &fallible, live(step_limit(LIMIT))),
        Err(TrySolveError::Solver(_))
    ));
}
