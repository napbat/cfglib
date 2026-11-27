//! Tests for the structural verification of graph views.

use super::*;
use alloc::vec;

#[test]
fn a_verify_error_is_an_error_whose_display_is_its_message() {
    let error = VerifyError {
        message: "edge index 3 appears more than once".into(),
    };
    let error: &dyn core::error::Error = &error;
    assert_eq!(
        alloc::format!("{error}"),
        "edge index 3 appears more than once"
    );
    assert!(error.source().is_none());
}

#[test]
fn a_report_counts_structural_and_semantic_errors_together() {
    let structural = VerifyError {
        message: "reachable node 2 has no predecessors".into(),
    };
    let clean = VerifyReport::<u8> {
        errors: vec![],
        semantic_errors: vec![],
    };
    assert!(clean.is_ok());
    assert_eq!(clean.error_count(), 0);

    let semantic_only = VerifyReport {
        errors: vec![],
        semantic_errors: vec![7_u8],
    };
    assert!(!semantic_only.is_ok());
    assert_eq!(semantic_only.error_count(), 1);

    let both = VerifyReport {
        errors: vec![structural],
        semantic_errors: vec![7_u8, 9],
    };
    assert!(!both.is_ok());
    assert_eq!(both.error_count(), 3);
}

#[test]
fn verify_view_accepts_consistent_consumer_view() {
    let mut graph = crate::graph::store::Graph::new();
    let root = graph.add_node(());
    let child = graph.add_node(());
    graph.add_edge(root, child, ());
    graph.add_edge(root, child, ());

    let rooted = crate::graph::view::Rooted::new(&graph, root);
    assert!(verify_view(&rooted).is_ok());
}

#[test]
fn verify_view_reports_one_sided_adjacency() {
    struct Broken;
    impl crate::graph::view::GraphView for Broken {
        type NodeId = usize;

        fn node_bound(&self) -> usize {
            2
        }

        fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
            0..2
        }

        fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
            (node == 0).then_some(1).into_iter()
        }

        fn predecessors(&self, _node: usize) -> impl Iterator<Item = usize> + '_ {
            core::iter::empty()
        }
    }
    impl crate::graph::view::RootedView for Broken {
        fn root(&self) -> usize {
            0
        }
    }

    let result = verify_view(&Broken);
    assert!(!result.is_ok());
    assert!(
        result
            .errors
            .iter()
            .any(|error| error.message.contains("adjacency mismatch"))
    );
    assert!(
        result
            .errors
            .iter()
            .any(|error| error.message.contains("no predecessors"))
    );
}

/// Explicit adjacency tables, so a test can corrupt each side on its own.
struct Table {
    out_edges: Vec<Vec<usize>>,
    in_edges: Vec<Vec<usize>>,
    endpoints: Vec<(usize, usize)>,
    live: Vec<usize>,
}

impl crate::graph::view::GraphView for Table {
    type NodeId = usize;

    fn node_bound(&self) -> usize {
        self.out_edges.len()
    }

    fn node_ids(&self) -> impl Iterator<Item = usize> + '_ {
        0..self.out_edges.len()
    }

    fn successors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        let endpoints = &self.endpoints;
        self.out_edges[node]
            .iter()
            .filter_map(|&edge| endpoints.get(edge))
            .map(|&(_, target)| target)
    }

    fn predecessors(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        let endpoints = &self.endpoints;
        self.in_edges[node]
            .iter()
            .filter_map(|&edge| endpoints.get(edge))
            .map(|&(source, _)| source)
    }
}

impl RootedView for Table {
    fn root(&self) -> usize {
        0
    }
}

impl EdgeView for Table {
    type EdgeId = usize;
    type EdgeData = ();

    fn edge_bound(&self) -> usize {
        self.endpoints.len()
    }

    fn edge_ids(&self) -> impl Iterator<Item = usize> + '_ {
        self.live.iter().copied()
    }

    fn outgoing(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.out_edges[node].iter().copied()
    }

    fn incoming(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        self.in_edges[node].iter().copied()
    }

    fn edge(&self, edge: usize) -> crate::graph::edge_view::EdgeRef<'_, usize, usize, ()> {
        let (source, target) = self.endpoints[edge];
        crate::graph::edge_view::EdgeRef::new(edge, source, target, &())
    }
}

fn messages(report: &VerifyReport) -> Vec<&str> {
    report
        .errors
        .iter()
        .map(|error| error.message.as_str())
        .collect()
}

#[test]
fn verify_view_reports_multiplicity_mismatches_in_pair_order() {
    // Edge slots 5 and 6 are listed on the reverse side only.
    let table = Table {
        out_edges: vec![vec![0, 1, 2], vec![3], vec![4], vec![]],
        in_edges: vec![vec![5], vec![0], vec![2], vec![3, 4, 4, 6]],
        endpoints: vec![(0, 1), (0, 1), (0, 2), (1, 3), (2, 3), (3, 0), (0, 3)],
        live: (0..7).collect(),
    };

    let report = verify_view(&table);
    assert_eq!(
        messages(&report),
        [
            "adjacency mismatch: 2 successor edge(s) 0->1 but 1 predecessor entry(ies)",
            "adjacency mismatch: 1 successor edge(s) 2->3 but 2 predecessor entry(ies)",
            "adjacency mismatch: 1 predecessor entry(ies) 0->3 with no successor edge",
            "adjacency mismatch: 1 predecessor entry(ies) 3->0 with no successor edge",
        ]
    );
}

#[test]
fn verify_edge_view_reports_every_edge_level_defect_in_order() {
    // Live list: edge 1 twice, edge 8 beyond the bound, edge 3 pointing at
    // node 9, and edge 4 removed. Adjacency: stray, repeated, misoriented
    // and dangling entries.
    let table = Table {
        out_edges: vec![vec![0, 5, 4], vec![1, 1, 2], vec![2]],
        in_edges: vec![vec![2], vec![0, 7], vec![1, 5, 4, 0]],
        endpoints: vec![(0, 1), (1, 2), (2, 0), (0, 9), (1, 2), (0, 2)],
        live: vec![0, 1, 1, 2, 3, 5, 8],
    };

    let report = verify_edge_view(&table);
    assert_eq!(
        messages(&report),
        [
            "adjacency mismatch: 1 successor edge(s) 1->0 but 0 predecessor entry(ies)",
            "edge index 1 appears more than once",
            "edge index 3 has endpoint outside node_bound 3",
            "edge index 8 out of bounds (edge_bound=6)",
            "node 0 has non-live outgoing edge index 4",
            "node 1 lists edge index 2, whose source is 2",
            "node 1 has non-live incoming edge index 7",
            "node 2 has non-live incoming edge index 4",
            "node 2 lists edge index 0, whose target is 1",
            "edge index 0 appears 1 time(s) in outgoing and 2 time(s) in incoming adjacency",
            "edge index 1 appears 2 time(s) in outgoing and 1 time(s) in incoming adjacency",
            "edge index 2 appears 2 time(s) in outgoing and 1 time(s) in incoming adjacency",
            "edge index 3 appears 0 time(s) in outgoing and 0 time(s) in incoming adjacency",
        ]
    );
}
