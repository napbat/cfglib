//! Peak memory of a streamed solve: a fact type that counts its own live
//! instances shows that a chain of sequential loops is solved with a constant
//! number of facts alive, where the solve that returns them holds two per
//! node.

use std::cell::Cell;
use std::convert::Infallible;
use std::rc::Rc;

use cfglib::{
    Direction, Graph, NodeId, SolveConfig, TryNodeProblem, stream_node_problem,
    try_solve_node_problem,
};

/// How many facts exist right now, and the most that ever did.
#[derive(Default)]
struct Census {
    live: Cell<usize>,
    peak: Cell<usize>,
}

impl Census {
    fn born(&self) {
        self.live.set(self.live.get() + 1);
        self.peak.set(self.peak.get().max(self.live.get()));
    }
}

/// A taint bit that reports its birth and death to a census.
struct Counted {
    tainted: bool,
    census: Rc<Census>,
}

impl Counted {
    fn new(tainted: bool, census: &Rc<Census>) -> Self {
        census.born();
        Self {
            tainted,
            census: Rc::clone(census),
        }
    }
}

impl Clone for Counted {
    fn clone(&self) -> Self {
        Self::new(self.tainted, &self.census)
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        self.census.live.set(self.census.live.get() - 1);
    }
}

impl PartialEq for Counted {
    fn eq(&self, other: &Self) -> bool {
        self.tainted == other.tainted
    }
}

/// Taint from the first loop's head onward.
struct Taint {
    source: NodeId,
    census: Rc<Census>,
}

impl TryNodeProblem<Graph<(), ()>> for Taint {
    type Fact = Counted;
    type Error = Infallible;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _graph: &Graph<(), ()>) -> Counted {
        Counted::new(false, &self.census)
    }

    fn boundary(&self, _graph: &Graph<(), ()>) -> Result<Counted, Infallible> {
        Ok(Counted::new(false, &self.census))
    }

    fn meet(
        &self,
        _: &Graph<(), ()>,
        _: NodeId,
        a: &Counted,
        b: &Counted,
    ) -> Result<Counted, Infallible> {
        Ok(Counted::new(a.tainted || b.tainted, &self.census))
    }

    fn transfer(
        &self,
        _: &Graph<(), ()>,
        node: NodeId,
        input: &Counted,
    ) -> Result<Counted, Infallible> {
        Ok(Counted::new(
            input.tainted || node == self.source,
            &self.census,
        ))
    }
}

/// `loops` loops one after another: each head runs its latch, which loops
/// back, and goes on to the next head. Returns the graph and its first head.
fn chain_of_loops(loops: usize) -> (Graph<(), ()>, NodeId) {
    let mut graph = Graph::new();
    let mut heads = Vec::with_capacity(loops);
    for _ in 0..loops {
        let head = graph.add_node(());
        let latch = graph.add_node(());
        graph.add_edge(head, latch, ());
        graph.add_edge(latch, head, ());
        if let Some(&previous) = heads.last() {
            graph.add_edge(previous, head, ());
        }
        heads.push(head);
    }
    (graph, heads[0])
}

/// The stream's own count of stored facts, and the census's count of every
/// fact alive, temporaries and the problem's boundary included.
fn streamed_peaks(loops: usize) -> (usize, usize) {
    let (graph, source) = chain_of_loops(loops);
    let census = Rc::new(Census::default());
    let problem = Taint {
        source,
        census: Rc::clone(&census),
    };
    let mut visited = 0;
    let stats = stream_node_problem(&graph, &problem, SolveConfig::new(), |_, input, output| {
        assert!(input.tainted && output.tainted);
        visited += 1;
    })
    .unwrap();
    assert_eq!(visited, 2 * loops);
    assert_eq!(census.live.get(), 0, "the stream freed every fact");
    (stats.peak_live_facts(), census.peak.get())
}

#[test]
fn a_chain_of_a_thousand_loops_streams_in_constant_memory() {
    let (stored, alive) = streamed_peaks(1_000);
    assert!(stored <= 8, "{stored} facts stored at once");
    assert!(alive <= 16, "{alive} facts alive at once");
    // The bound does not grow with the chain.
    assert_eq!(streamed_peaks(10), (stored, alive));
}

#[test]
fn the_solve_that_returns_every_fact_holds_two_per_node() {
    let loops = 1_000;
    let (graph, source) = chain_of_loops(loops);
    let census = Rc::new(Census::default());
    let problem = Taint {
        source,
        census: Rc::clone(&census),
    };
    let facts = try_solve_node_problem(&graph, &problem).unwrap();
    assert!(census.live.get() >= 2 * 2 * loops);
    drop(facts);
    assert_eq!(census.live.get(), 0);
}
