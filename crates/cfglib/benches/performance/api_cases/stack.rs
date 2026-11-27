//! Stack-graph resolution: direct search, the linear walk, per-file partial
//! paths, and stitching of those partial paths.
//!
//! The decoy chain is one file whose references each climb a chain of scopes
//! with a dead end at every hop, the shape the linear walk follows without a
//! frontier. The joined files are a client and a library that meet at the
//! root: every reference reaches every definition there, and the symbol stack
//! rejects all but the one of its own symbol, so direct search, the partial
//! paths and stitching all pay one rejected extension per definition.

use cfglib::{
    PathSearchConfig, StackFileId, StackGraph, StackNodeId, StackNodeKind,
    StackPartialPathDatabase, StackResolutionIndex,
};

use super::BenchmarkSuite;
use crate::fixtures::fixture_u32;
use crate::harness::benchmark_case;

/// References and scope hops per reference of the decoy chain.
const DECOY_REFERENCES: usize = 256;
const DECOY_HOPS: usize = 12;
/// Symbols the client file of the joined files references and the library
/// defines.
const JOINED_SYMBOLS: usize = 256;

type Stack = StackGraph<(), u32, (), ()>;

/// Adds a node of `kind` to `file`.
fn node(graph: &mut Stack, file: StackFileId, kind: StackNodeKind<u32>) -> StackNodeId {
    graph.add_node(file, kind, ()).expect("a valid file node")
}

const fn reference(symbol: u32) -> StackNodeKind<u32> {
    StackNodeKind::PushSymbol {
        symbol,
        is_reference: true,
    }
}

const fn definition(symbol: u32, is_definition: bool) -> StackNodeKind<u32> {
    StackNodeKind::PopSymbol {
        symbol,
        is_definition,
    }
}

const SCOPE: StackNodeKind<u32> = StackNodeKind::Scope { exported: false };

/// One file of `DECOY_REFERENCES` references. Each pushes its symbol, hops
/// through `DECOY_HOPS` scopes, and pops it at its definition; every scope also
/// has an edge to a pop of another symbol, which every path rejects. The decoy
/// is listed first on even hops and last on odd ones.
fn decoy_chain() -> (Stack, Vec<(StackNodeId, StackNodeId)>) {
    let mut graph = Stack::new();
    let file = graph.add_file(());
    let mut bindings = Vec::with_capacity(DECOY_REFERENCES);
    for index in 0..DECOY_REFERENCES {
        let symbol = fixture_u32(index);
        let use_site = node(&mut graph, file, reference(symbol));
        let declaration = node(&mut graph, file, definition(symbol, true));
        let decoy = node(&mut graph, file, definition(u32::MAX, false));
        let scopes: Vec<StackNodeId> = (0..DECOY_HOPS)
            .map(|_| node(&mut graph, file, SCOPE))
            .collect();
        let mut add = |source, target| {
            graph
                .add_edge(source, target, 0, ())
                .expect("file-local edge");
        };
        add(use_site, scopes[0]);
        for (hop, &scope) in scopes.iter().enumerate() {
            let onward = scopes.get(hop + 1).copied().unwrap_or(declaration);
            if hop % 2 == 0 {
                add(scope, decoy);
                add(scope, onward);
            } else {
                add(scope, onward);
                add(scope, decoy);
            }
        }
        bindings.push((use_site, declaration));
    }
    (graph, bindings)
}

/// A client file whose references each push a symbol and climb one scope to
/// the root, and a library file whose definitions of those symbols the root
/// reaches.
fn joined_files() -> (Stack, Vec<(StackNodeId, StackNodeId)>) {
    let mut graph = Stack::new();
    let client = graph.add_file(());
    let library = graph.add_file(());
    let root = graph.root_node();
    let mut bindings = Vec::with_capacity(JOINED_SYMBOLS);
    for index in 0..JOINED_SYMBOLS {
        let symbol = fixture_u32(index);
        let use_site = node(&mut graph, client, reference(symbol));
        let scope = node(&mut graph, client, SCOPE);
        let declaration = node(&mut graph, library, definition(symbol, true));
        for (source, target) in [(use_site, scope), (scope, root), (root, declaration)] {
            graph
                .add_edge(source, target, 0, ())
                .expect("a file-local edge or one that touches the root");
        }
        bindings.push((use_site, declaration));
    }
    (graph, bindings)
}

/// Checks that each reference resolves to its own definition alone.
fn assert_bound(index: &StackResolutionIndex<u32>, bindings: &[(StackNodeId, StackNodeId)]) {
    for &(reference, definition) in bindings {
        assert_eq!(index.references_to(definition), [reference]);
    }
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let config = PathSearchConfig::default();
    let (chain, chain_bindings) = decoy_chain();
    benchmark_case!(
        suite,
        "stack_resolution_index_linear",
        || StackResolutionIndex::try_compute_linear(&chain, config),
        |index| assert_bound(
            index.as_ref().expect("every query is linear"),
            &chain_bindings
        )
    );

    let (joined, bindings) = joined_files();
    benchmark_case!(
        suite,
        "stack_resolution_index_direct",
        || StackResolutionIndex::compute(&joined, config),
        |index| assert_bound(index, &bindings)
    );
    benchmark_case!(
        suite,
        "stack_partial_paths_joined_files",
        || StackPartialPathDatabase::compute(&joined, config),
        |database| assert_eq!(database.path_count(), 2 * JOINED_SYMBOLS)
    );
    let database = StackPartialPathDatabase::compute(&joined, config);
    benchmark_case!(
        suite,
        "stack_resolution_index_stitched",
        || StackResolutionIndex::try_compute_stitched(&joined, &database, config),
        |index| assert_bound(
            index.as_ref().expect("the database matches the graph"),
            &bindings
        )
    );
}
