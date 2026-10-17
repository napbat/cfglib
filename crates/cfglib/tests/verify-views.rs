//! Integration tests: validating consumer-owned views whose adapters break
//! the dense-identity contract.
//!
//! The validators exist to catch adapter bugs, so a broken adapter must come
//! back as a report and never abort the validator.

use cfglib::{
    EdgeRef, EdgeView, GraphView, RootedView, VerifyReport, verify_edge_view, verify_view,
};

/// A two-node view over explicit tables, rooted at node 0. Each field can be
/// made to break one clause of the view contract.
#[derive(Default)]
struct Adapter {
    nodes: Vec<usize>,
    successors: Vec<Vec<usize>>,
    predecessors: Vec<Vec<usize>>,
}

impl Adapter {
    fn chain() -> Self {
        Self {
            nodes: vec![0, 1],
            successors: vec![vec![1], vec![]],
            predecessors: vec![vec![], vec![0]],
        }
    }
}

impl GraphView for Adapter {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        2
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.nodes.iter().copied()
    }

    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.successors.get(node).into_iter().flatten().copied()
    }

    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.predecessors.get(node).into_iter().flatten().copied()
    }
}

impl RootedView for Adapter {
    fn root(&self) -> usize {
        0
    }
}

/// One edge slot per successor entry, so an edge-aware validator can run over
/// the same tables.
impl EdgeView for Adapter {
    type EdgeId = usize;
    type EdgeData = ();

    fn edge_bound(&self) -> usize {
        1
    }

    fn edge_ids(&self) -> impl Iterator<Item = usize> + '_ {
        0..1
    }

    fn outgoing(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        (node == 0).then_some(0).into_iter()
    }

    fn incoming(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        (node == 1).then_some(0).into_iter()
    }

    fn edge(&self, edge: usize) -> EdgeRef<'_, usize, usize, ()> {
        EdgeRef::new(edge, 0, 1, &())
    }
}

fn assert_reported_once(report: &VerifyReport) {
    assert!(!report.is_ok());
    assert_eq!(report.error_count(), 1, "{:?}", report.errors);
}

#[test]
fn a_sound_adapter_verifies() {
    let adapter = Adapter::chain();
    assert!(verify_view(&adapter).is_ok());
    assert!(verify_edge_view(&adapter).is_ok());
}

#[test]
fn a_successor_beyond_the_node_bound_is_reported_not_indexed() {
    let mut adapter = Adapter::chain();
    adapter.successors[1].push(5);

    assert_reported_once(&verify_view(&adapter));
    assert_reported_once(&verify_edge_view(&adapter));
}

#[test]
fn a_predecessor_beyond_the_node_bound_is_reported_not_indexed() {
    let mut adapter = Adapter::chain();
    adapter.predecessors[0].push(5);

    assert_reported_once(&verify_view(&adapter));
    assert_reported_once(&verify_edge_view(&adapter));
}

#[test]
fn a_node_beyond_the_node_bound_is_reported_not_indexed() {
    let mut adapter = Adapter::chain();
    adapter.nodes.push(7);

    assert_reported_once(&verify_view(&adapter));
    assert_reported_once(&verify_edge_view(&adapter));
}
