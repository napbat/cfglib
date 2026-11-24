//! Scope-graph indexes: every reference resolved on its own, against one
//! environment cache shared by all of them, or again after an edit.
//!
//! The lexical chain is the shape of a source file: references in nested
//! blocks climb the same parents and imports, which `compute` walks once per
//! reference and `compute_cached` once per scope. An edit of one leaf scope
//! there is read by the references of one block, which `update` resolves
//! again. The stacked inheritance diamonds reach one member along `2^d`
//! equally specific paths, every one of which both resolvers keep, so their
//! cost grows with that output.

use std::cmp::Ordering;

use cfglib::{
    EdgeRef, ScopeDatumId, ScopeEdgeId, ScopeEnvironmentQuery, ScopeGraph, ScopeGraphChanges,
    ScopeGraphPathLabel, ScopeGraphQuery, ScopeId, ScopeIndexUpdate, ScopePath, ScopeQuery,
    ScopeResolutionConfig, ScopeResolutionIndex,
};

use super::BenchmarkSuite;
use crate::fixtures::fixture_u32;
use crate::harness::benchmark_case;

/// Nested levels of the lexical chain.
const LEVELS: u32 = 100;
/// References spread over the levels of the lexical chain.
const REFERENCES: u32 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Label {
    Parent,
    Import,
}

/// Scopes without payload, and data and references named by a number.
type Names = ScopeGraph<(), Label, (), u32, u32>;

/// Lexical lookup through any number of parents and then at most one import.
/// A declaration shadows an import, which shadows a parent.
struct Lexical;

fn step(imported: bool, label: Label) -> Option<bool> {
    match (imported, label) {
        (false, Label::Parent) => Some(false),
        (false, Label::Import) => Some(true),
        (true, _) => None,
    }
}

fn compare(
    left: ScopeGraphPathLabel<'_, Label>,
    right: ScopeGraphPathLabel<'_, Label>,
) -> Ordering {
    let rank = |label| match label {
        ScopeGraphPathLabel::Declaration => 0,
        ScopeGraphPathLabel::Edge(Label::Import) => 1,
        ScopeGraphPathLabel::Edge(Label::Parent) => 2,
    };
    rank(left).cmp(&rank(right))
}

impl ScopeGraphQuery<(), Label, (), u32, u32> for Lexical {
    type State = bool;

    fn initial_state(&self, _graph: &Names, _input: &ScopeQuery<'_, u32>) -> bool {
        false
    }

    fn step(
        &self,
        _graph: &Names,
        _input: &ScopeQuery<'_, u32>,
        _path: &ScopePath,
        imported: &bool,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<bool> {
        step(*imported, *edge.data())
    }

    fn accepts(
        &self,
        graph: &Names,
        input: &ScopeQuery<'_, u32>,
        _path: &ScopePath,
        _imported: &bool,
        datum: ScopeDatumId,
    ) -> bool {
        graph.datum(datum).data() == input.data()
    }

    fn compare_labels(
        &self,
        _graph: &Names,
        _input: &ScopeQuery<'_, u32>,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        Some(compare(left, right))
    }
}

impl ScopeEnvironmentQuery<(), Label, (), u32, u32> for Lexical {
    type State = bool;
    type Key = u32;

    fn initial_state(&self, _graph: &Names) -> bool {
        false
    }

    fn step(
        &self,
        _graph: &Names,
        _scope: ScopeId,
        imported: &bool,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<bool> {
        step(*imported, *edge.data())
    }

    fn accepts(&self, _graph: &Names, _scope: ScopeId, _imported: &bool, _: ScopeDatumId) -> bool {
        true
    }

    fn key(&self, graph: &Names, datum: ScopeDatumId) -> u32 {
        *graph.datum(datum).data()
    }

    fn query_key(&self, _graph: &Names, query: &u32) -> u32 {
        *query
    }

    fn compare_labels(
        &self,
        _graph: &Names,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        Some(compare(left, right))
    }
}

/// `LEVELS` nested blocks, each with a parent edge to the enclosing one and
/// an import of its own module. Block `i` declares `2 i` and its module
/// declares `2 i + 1`. `REFERENCES` references, spread over the blocks, each
/// name something declared at or around its block. Also returns the
/// innermost module, a leaf scope that only the innermost block reads.
fn lexical_chain() -> (Names, ScopeId) {
    let mut graph = Names::new();
    let mut blocks = Vec::new();
    let mut module = None;
    for level in 0..LEVELS {
        let block = graph.add_scope(());
        let imported = graph.add_scope(());
        if let Some(&outer) = blocks.last() {
            graph.add_edge(block, outer, Label::Parent);
        }
        graph.add_edge(block, imported, Label::Import);
        graph.add_datum(block, (), 2 * level);
        graph.add_datum(imported, (), 2 * level + 1);
        blocks.push(block);
        module = Some(imported);
    }
    for reference in 0..REFERENCES {
        let level = reference % LEVELS;
        let name = fixture_u32(reference as usize) % (2 * (level + 1));
        graph.add_reference(blocks[level as usize], name);
    }
    (graph, module.expect("the chain has a level"))
}

/// `diamonds` stacked inheritance diamonds: class `3 k` derives `3 k + 1`
/// and `3 k + 2`, which both derive class `3 k + 3`. The last class declares
/// the member `0`, and every class has a reference to it.
fn stacked_diamonds(diamonds: u32) -> Names {
    let mut graph = Names::new();
    let classes: Vec<ScopeId> = (0..=3 * diamonds).map(|_| graph.add_scope(())).collect();
    for level in 0..diamonds as usize {
        let [top, left, right, next] = [0, 1, 2, 3].map(|offset| classes[3 * level + offset]);
        graph.add_edge(top, left, Label::Parent);
        graph.add_edge(top, right, Label::Parent);
        graph.add_edge(left, next, Label::Parent);
        graph.add_edge(right, next, Label::Parent);
    }
    if let Some(&bottom) = classes.last() {
        graph.add_datum(bottom, (), 0);
    }
    for &class in &classes {
        graph.add_reference(class, 0);
    }
    graph
}

/// Checks that every reference of the chain resolves to the one datum of its
/// name, as both indexes must.
fn assert_chain_resolved(graph: &Names, index: &ScopeResolutionIndex) {
    for reference in graph.reference_ids() {
        let resolution = index.resolution(reference);
        let [candidate] = resolution.candidates() else {
            panic!("{reference} has one visible declaration");
        };
        assert_eq!(
            graph.datum(candidate.datum()).data(),
            graph.reference(reference).data()
        );
    }
}

/// Checks that the most derived class sees the member along all `2^d`
/// paths, as both indexes must.
fn assert_diamond_routes(graph: &Names, index: &ScopeResolutionIndex, diamonds: u32) {
    let top = graph
        .reference_ids()
        .next()
        .expect("every class has a reference");
    assert_eq!(index.resolution(top).candidates().len(), 1 << diamonds);
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let config = ScopeResolutionConfig::new();
    let (chain, leaf) = lexical_chain();
    benchmark_case!(
        suite,
        "scope_index_lexical_chain",
        || ScopeResolutionIndex::compute(&chain, &Lexical, config),
        |index| assert_chain_resolved(&chain, index)
    );
    benchmark_case!(
        suite,
        "scope_index_cached_lexical_chain",
        || ScopeResolutionIndex::compute_cached(&chain, &Lexical, config),
        |index| assert_chain_resolved(&chain, index)
    );
    // One new datum in the innermost module: only the references of the
    // innermost block read it. Every run repeats the same update, which
    // resolves them again and changes nothing after the first.
    let mut index = ScopeResolutionIndex::compute(&chain, &Lexical, config);
    let mut edited = chain.clone();
    edited.add_datum(leaf, (), 2 * LEVELS);
    let changes = ScopeGraphChanges {
        scopes: vec![leaf],
        references: Vec::new(),
    };
    benchmark_case!(
        suite,
        "scope_index_update_leaf_datum",
        || index.update(&edited, &Lexical, config, &changes),
        |update: &ScopeIndexUpdate| assert_eq!(
            update.resolved_count,
            (REFERENCES / LEVELS) as usize
        )
    );

    // Sixteen diamonds would give the top class 2^16 visible paths, which
    // both resolvers filter pairwise for shadowing: out of reach for either.
    for (diamonds, general, cached) in [
        (8, "scope_index_diamonds_8", "scope_index_cached_diamonds_8"),
        (
            10,
            "scope_index_diamonds_10",
            "scope_index_cached_diamonds_10",
        ),
    ] {
        let graph = stacked_diamonds(diamonds);
        // The case names are variables here, which `benchmark_case!` takes
        // only as literals.
        suite.case(
            general,
            &[],
            || ScopeResolutionIndex::compute(&graph, &Lexical, config),
            |index| assert_diamond_routes(&graph, index, diamonds),
        );
        suite.case(
            cached,
            &[],
            || ScopeResolutionIndex::compute_cached(&graph, &Lexical, config),
            |index| assert_diamond_routes(&graph, index, diamonds),
        );
    }
}
