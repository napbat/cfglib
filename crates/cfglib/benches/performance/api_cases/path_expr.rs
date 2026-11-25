//! `path_expressions` and `interpret` over a structured CFG of 10^5 blocks.
//!
//! The CFG is a sequence of plain blocks, if/else diamonds and while loops
//! nested up to eight deep, so it is reducible and the elimination never
//! falls back to Gaussian elimination. `interpret` runs the gen/kill algebra
//! of a 64-bit forward problem, checked against the node fixpoint.

use cfglib::{
    DenseBits, Direction, EdgeId, GenKill, GenKillAlgebra, Graph, NodeId, NodeProblem,
    PathExpressions, Rooted, interpret, path_expressions, solve_node_problem,
};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;

/// Blocks of the structured CFG.
const BLOCKS: usize = 100_000;
/// Bits of the gen/kill problem.
const UNIVERSE: usize = 64;

type Plain = Graph<(), ()>;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let graph = structured_cfg(BLOCKS);
    let root = NodeId::from_index(0);
    benchmark_case!(
        suite,
        "api_path_expressions",
        covers[path_expressions],
        || path_expressions(&Rooted::new(&graph, root)),
        |expressions: &PathExpressions<NodeId, EdgeId>| {
            assert!(expressions.arena().node_count() <= 8 * graph.edge_count());
            assert!(graph.node_ids().all(|node| expressions.to(node).is_some()));
        },
    );

    let expressions = path_expressions(&Rooted::new(&graph, root));
    let transfers = Transfers(
        (0..graph.node_bound())
            .map(|node| {
                let mut generate = DenseBits::new(UNIVERSE);
                let mut kill = DenseBits::new(UNIVERSE);
                generate.insert(node % UNIVERSE);
                kill.insert((node * 7 + 3) % UNIVERSE);
                kill.insert((node * 13 + 5) % UNIVERSE);
                GenKill::new(generate, kill)
            })
            .collect(),
    );
    let algebra = GenKillAlgebra::new(UNIVERSE, |edge: EdgeId| {
        transfers.0[graph.edge(edge).target().index()].clone()
    });
    benchmark_case!(
        suite,
        "api_interpret_gen_kill",
        covers[interpret],
        || interpret(&expressions, &algebra),
        |values: &Vec<Option<Option<GenKill>>>| {
            let fixpoint = solve_node_problem(&graph, &transfers).expect("an unbounded solve");
            let empty = DenseBits::new(UNIVERSE);
            for node in graph.node_ids() {
                let paths = values[node.index()]
                    .clone()
                    .flatten()
                    .expect("the entry reaches every block");
                let out = transfers.0[0].then(&paths).apply(&empty);
                assert_eq!(&out, fixpoint.fact_out(node));
            }
        },
    );
}

/// A forward union-meet gen/kill problem with one transfer per block.
struct Transfers(Vec<GenKill>);

impl NodeProblem<Plain> for Transfers {
    type Fact = DenseBits;

    fn direction(&self) -> Direction {
        Direction::Forward
    }

    fn bottom(&self, _: &Plain) -> DenseBits {
        DenseBits::new(UNIVERSE)
    }

    fn boundary(&self, _: &Plain) -> DenseBits {
        DenseBits::new(UNIVERSE)
    }

    fn meet(&self, a: &DenseBits, b: &DenseBits) -> DenseBits {
        let mut met = a.clone();
        met.union_with(b);
        met
    }

    fn transfer(&self, _: &Plain, node: NodeId, input: &DenseBits) -> DenseBits {
        self.0[node.index()].apply(input)
    }
}

/// A structured CFG of about `blocks` blocks, the same on every run.
fn structured_cfg(blocks: usize) -> Plain {
    let mut state = 0x9E37_79B9_7F4A_7C15_u64;
    let mut roll = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % 10
    };
    let mut graph = Graph::with_capacity(blocks + 16, blocks * 2);
    let mut current = graph.add_node(());
    let mut open = Vec::new();
    while graph.node_count() < blocks {
        match roll() {
            0..=2 => {
                let [then, other, join] = [(); 3].map(|()| graph.add_node(()));
                graph.add_edge(current, then, ());
                graph.add_edge(current, other, ());
                graph.add_edge(then, join, ());
                graph.add_edge(other, join, ());
                current = join;
            }
            3 | 4 if open.len() < 8 => {
                let header = graph.add_node(());
                graph.add_edge(current, header, ());
                open.push(header);
                current = header;
            }
            5 | 6 if !open.is_empty() => {
                let header = open.pop().expect("a loop is open");
                graph.add_edge(current, header, ());
                let exit = graph.add_node(());
                graph.add_edge(header, exit, ());
                current = exit;
            }
            _ => {
                let next = graph.add_node(());
                graph.add_edge(current, next, ());
                current = next;
            }
        }
    }
    while let Some(header) = open.pop() {
        graph.add_edge(current, header, ());
        let exit = graph.add_node(());
        graph.add_edge(header, exit, ());
        current = exit;
    }
    graph
}
