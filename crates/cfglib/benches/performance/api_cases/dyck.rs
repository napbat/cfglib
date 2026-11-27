//! `bidirected_dyck_classes` over random bidirected graphs of three sizes.
//!
//! Each graph has three edges per node with uniform random endpoints: most
//! open one of three field labels and one in eight is a plain copy edge, the
//! shape of a field-sensitive alias graph. The three sizes, ten times apart,
//! show how the build scales; a near-linear build keeps the time per edge
//! flat across them.

use std::collections::BTreeMap;

use cfglib::{BidirectedDyck, Graph, NodeId, bidirected_dyck_classes};

use super::BenchmarkSuite;
use crate::harness::benchmark_case;
use crate::rng::Rng;

/// Edges per node.
const FAN: usize = 3;
/// Distinct field labels.
const LABELS: u64 = 3;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    benchmark_case!(
        suite,
        "api_bidirected_dyck_classes_1e4",
        covers[bidirected_dyck_classes],
        classes_of_random(10_000),
        |classes: &BidirectedDyck| check_classes(10_000, classes),
    );
    benchmark_case!(
        suite,
        "api_bidirected_dyck_classes_1e5",
        classes_of_random(100_000),
        |classes: &BidirectedDyck| check_classes(100_000, classes),
    );
    benchmark_case!(
        suite,
        "api_bidirected_dyck_classes_1e6",
        classes_of_random(1_000_000),
        |classes: &BidirectedDyck| check_classes(1_000_000, classes),
    );
}

/// The measured operation: the classes of the random graph of `nodes` nodes.
///
/// A million-node graph takes a while to build, so the graph is built on the
/// first run, which is the untimed oracle run, and a filter that skips the
/// case does not pay for it.
fn classes_of_random(nodes: usize) -> impl FnMut() -> BidirectedDyck {
    let mut graph = None;
    move || {
        let graph = graph.get_or_insert_with(|| random_bidirected(nodes));
        bidirected_dyck_classes(graph, |edge| *edge.data())
    }
}

/// A random graph with `FAN` edges per node, the same for every run.
fn random_bidirected(nodes: usize) -> Graph<(), Option<u32>> {
    let mut graph = Graph::with_capacity(nodes, nodes * FAN);
    for _ in 0..nodes {
        graph.add_node(());
    }
    let mut rng = Rng::xorshift(0x2545_F491_4F6C_DD1D);
    for _ in 0..nodes * FAN {
        let from = NodeId::from_index(rng.below(nodes));
        let to = NodeId::from_index(rng.below(nodes));
        let roll = rng.next_u64();
        let label =
            (!roll.is_multiple_of(8)).then(|| u32::try_from(roll % LABELS).expect("a small label"));
        graph.add_edge(from, to, label);
    }
    graph
}

/// The closure's local laws on the very graph that was solved: both ends of
/// a plain edge share a class, every source of one label into one class
/// shares a class, and each class is named by a member no larger than it.
fn check_classes(nodes: usize, classes: &BidirectedDyck) {
    let mut classes = classes.clone();
    let graph = random_bidirected(nodes);
    assert_eq!(classes.node_bound(), nodes);
    let mut witness = BTreeMap::new();
    for edge in graph.edges() {
        let (from, to) = (edge.source().index(), edge.target().index());
        match *edge.payload() {
            None => assert!(classes.same_class(from, to)),
            Some(label) => {
                let class = classes.class_of(to);
                let source = *witness.entry((class, label)).or_insert(from);
                assert!(classes.same_class(source, from));
            }
        }
    }
    for node in 0..nodes {
        assert!(classes.class_of(node) <= node);
    }
}
