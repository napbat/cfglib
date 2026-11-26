extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Ordering;

use super::super::test_support::{Label, name, parent_or_import, random_graph, rank};
use super::{ScopeGraphChanges, ScopeIndexConfig, ScopeIndexUpdate, ScopeResolutionIndex};
use crate::PathSearchConfig;
use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::scope::{
    ScopeDatumId, ScopeEdgeId, ScopeEnvironmentCache, ScopeEnvironmentQuery, ScopeGraph,
    ScopeGraphPathLabel, ScopeGraphQuery, ScopeId, ScopePath, ScopeQuery, ScopeReferenceId,
};
use crate::graph::view::GraphView;
use crate::test_util::rng::Rng;

/// Scopes that may be opaque, and data and references named by a small
/// number.
type Graph = ScopeGraph<bool, Label, (), u8, u8>;

/// Parents, then imports (`P* I*`). An opaque scope hides its data from
/// every path that does not start in it. The hooks read only the scope at
/// the end of the path, as an incremental update requires.
struct Lookup;

impl ScopeGraphQuery<bool, Label, (), u8, u8> for Lookup {
    /// Whether the path has taken an import.
    type State = bool;

    fn initial_state(&self, _graph: &Graph, _input: &ScopeQuery<'_, u8>) -> bool {
        false
    }

    fn step(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, u8>,
        _path: &ScopePath,
        imported: &bool,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<bool> {
        match (imported, edge.data()) {
            (false, Label::Parent) => Some(false),
            (_, Label::Import) => Some(true),
            (true, Label::Parent) | (_, Label::Hidden | Label::A | Label::B) => None,
        }
    }

    fn accepts(
        &self,
        graph: &Graph,
        input: &ScopeQuery<'_, u8>,
        path: &ScopePath,
        _imported: &bool,
        datum: ScopeDatumId,
    ) -> bool {
        let opaque = *graph.scope(path.end()).payload();
        graph.datum(datum).data() == input.data() && (path.is_empty() || !opaque)
    }

    fn compare_labels(
        &self,
        _graph: &Graph,
        _input: &ScopeQuery<'_, u8>,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        Some(rank(left).cmp(&rank(right)))
    }
}

/// The edits of the property test, each with the changes it reports.
fn edit(graph: &mut Graph, random: &mut Rng) -> ScopeGraphChanges {
    let mut changes = ScopeGraphChanges::default();
    let scopes: Vec<ScopeId> = graph.node_ids().collect();
    let edges: Vec<ScopeEdgeId> = graph.edge_ids().collect();
    let scope = scopes[random.below(scopes.len())];
    match random.below(9) {
        0 => {
            let target = scopes[random.below(scopes.len())];
            graph.add_edge(scope, target, parent_or_import(random));
            changes.scopes.push(scope);
        }
        1 if !edges.is_empty() => {
            let edge = edges[random.below(edges.len())];
            changes.scopes.push(graph.edge(edge).source());
            graph.remove_edge(edge);
        }
        2 => {
            graph.add_datum(scope, (), name(random));
            changes.scopes.push(scope);
        }
        3 if graph.datum_count() > 0 => {
            let datum = ScopeDatumId::from_index(random.below(graph.datum_count()));
            *graph.datum_mut(datum).data_mut() = name(random);
            changes.scopes.push(graph.datum(datum).scope());
        }
        4 if !edges.is_empty() => {
            let edge = edges[random.below(edges.len())];
            *graph.edge_mut(edge) = parent_or_import(random);
            changes.scopes.push(graph.edge(edge).source());
        }
        5 => {
            let reference = graph.add_reference(scope, name(random));
            // A new reference is resolved whether or not it is reported.
            if random.one_in(2) {
                changes.references.push(reference);
            }
        }
        6 if graph.reference_count() > 0 => {
            let reference = ScopeReferenceId::from_index(random.below(graph.reference_count()));
            *graph.reference_mut(reference).data_mut() = name(random);
            changes.references.push(reference);
        }
        7 => {
            let opaque = graph.scope_mut(scope).payload_mut();
            *opaque = !*opaque;
            changes.scopes.push(scope);
        }
        _ => {
            // A new scope that no edge reaches yet needs no report.
            graph.add_scope(random.one_in(3));
        }
    }
    changes
}

/// The references whose resolution differs between `before` and `after`,
/// counting the references `before` did not have.
fn differing(before: &ScopeResolutionIndex, after: &ScopeResolutionIndex) -> Vec<ScopeReferenceId> {
    (0..after.resolutions().len())
        .map(ScopeReferenceId::from_index)
        .filter(|reference| {
            before.resolutions().get(reference.index()) != Some(after.resolution(*reference))
        })
        .collect()
}

#[test]
fn updates_match_a_fresh_index_after_random_edits() {
    let mut random = Rng::new(0x73_5EED);
    for _ in 0..300 {
        // Up to seven scopes, a quarter of them opaque, with edges anywhere.
        let mut graph = random_graph(&mut random, 7, None, |random| random.one_in(4));
        let config = match random.below(3) {
            0 => PathSearchConfig::default(),
            1 => PathSearchConfig {
                max_path_count: Some(1 + random.below(4)),
                ..PathSearchConfig::default()
            },
            _ => PathSearchConfig {
                max_path_length: Some(random.below(3)),
                ..PathSearchConfig::default()
            },
        };
        let plain = ScopeIndexConfig {
            search: config,
            track_reads: false,
        };
        let tracking = ScopeIndexConfig {
            track_reads: true,
            ..plain
        };
        // A plain index records nothing, so its first update resolves every
        // reference and records them; a tracked one is incremental at once.
        let config = if random.one_in(4) { plain } else { tracking };
        let mut index = ScopeResolutionIndex::compute(&graph, &Lookup, config);
        for _ in 0..12 {
            let changes = edit(&mut graph, &mut random);
            let before = index.clone();
            let update = index.update(&graph, &Lookup, config, &changes);
            let fresh = ScopeResolutionIndex::compute(&graph, &Lookup, plain);
            assert_eq!(index, fresh, "{graph:?} {changes:?}");
            let tracked = ScopeResolutionIndex::compute(&graph, &Lookup, tracking);
            assert_eq!(index.reads, tracked.reads, "{graph:?} {changes:?}");
            assert_eq!(update.changed, differing(&before, &fresh));
            assert!(update.resolved_count >= update.changed.len());
            assert!(update.resolved_count <= graph.reference_count());
        }
    }
}

/// A chain of blocks, each with a parent edge to the one before and an
/// import of its own module, and one reference per block to the name of
/// its module.
fn blocks(count: usize) -> (Graph, Vec<ScopeId>, Vec<ScopeId>) {
    let mut graph = Graph::new();
    let mut blocks: Vec<ScopeId> = Vec::new();
    let mut modules = Vec::new();
    for level in 0..count {
        let block = graph.add_scope(false);
        let module = graph.add_scope(false);
        if let Some(&outer) = blocks.last() {
            graph.add_edge(block, outer, Label::Parent);
        }
        graph.add_edge(block, module, Label::Import);
        let name = u8::try_from(level).expect("a short chain");
        graph.add_reference(block, name);
        blocks.push(block);
        modules.push(module);
    }
    (graph, blocks, modules)
}

#[test]
fn a_datum_in_a_leaf_scope_resolves_only_its_readers_again() {
    let (mut graph, _, modules) = blocks(6);
    let config = ScopeIndexConfig {
        track_reads: true,
        ..ScopeIndexConfig::default()
    };
    let mut index = ScopeResolutionIndex::compute(&graph, &Lookup, config);
    // Only the innermost block imports the innermost module.
    let datum = graph.add_datum(modules[5], (), 5);
    let changes = ScopeGraphChanges {
        scopes: vec![modules[5]],
        references: Vec::new(),
    };
    let update = index.update(&graph, &Lookup, config, &changes);
    let innermost = ScopeReferenceId::from_index(5);
    assert_eq!(
        update,
        ScopeIndexUpdate {
            resolved_count: 1,
            changed: vec![innermost],
        }
    );
    assert_eq!(index.references_to(datum), [innermost]);
    assert_eq!(
        index,
        ScopeResolutionIndex::compute(&graph, &Lookup, config)
    );

    // The outermost module is read by every block below it.
    graph.add_datum(modules[0], (), 0);
    let changes = ScopeGraphChanges {
        scopes: vec![modules[0]],
        references: Vec::new(),
    };
    let update = index.update(&graph, &Lookup, config, &changes);
    assert_eq!(update.resolved_count, 6);
    assert_eq!(update.changed, [ScopeReferenceId::from_index(0)]);

    // Nothing reported, nothing resolved.
    let update = index.update(&graph, &Lookup, config, &ScopeGraphChanges::default());
    assert_eq!(update, ScopeIndexUpdate::default());
}

#[test]
fn a_plain_index_records_nothing_until_its_first_update() {
    let (mut graph, blocks, _) = blocks(4);
    let config = ScopeIndexConfig::default();
    let mut index = ScopeResolutionIndex::compute(&graph, &Lookup, config);
    assert!(index.reads.is_none());
    let tracking = ScopeIndexConfig {
        track_reads: true,
        ..config
    };
    let tracked = ScopeResolutionIndex::compute(&graph, &Lookup, tracking);
    assert!(tracked.reads.is_some());
    assert_eq!(index, tracked);

    // The first update has no record to go by: it resolves every reference.
    graph.add_datum(blocks[3], (), 3);
    let changes = ScopeGraphChanges {
        scopes: vec![blocks[3]],
        references: Vec::new(),
    };
    let update = index.update(&graph, &Lookup, config, &changes);
    assert_eq!(update.resolved_count, 4);
    assert_eq!(update.changed, [ScopeReferenceId::from_index(3)]);
    assert_eq!(
        index,
        ScopeResolutionIndex::compute(&graph, &Lookup, config)
    );

    // From then on it updates incrementally.
    graph.add_datum(blocks[3], (), 0);
    let update = index.update(&graph, &Lookup, config, &changes);
    assert_eq!(update.resolved_count, 1);
    assert_eq!(
        index,
        ScopeResolutionIndex::compute(&graph, &Lookup, config)
    );
}

/// The same policy, path-free, for the cached index.
impl ScopeEnvironmentQuery<bool, Label, (), u8, u8> for Lookup {
    type State = bool;
    type Key = u8;

    fn initial_state(&self, _graph: &Graph) -> bool {
        false
    }

    fn step(
        &self,
        _graph: &Graph,
        _scope: ScopeId,
        imported: &bool,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, Label>,
    ) -> Option<bool> {
        match (imported, edge.data()) {
            (false, Label::Parent) => Some(false),
            (_, Label::Import) => Some(true),
            (true, Label::Parent) | (_, Label::Hidden | Label::A | Label::B) => None,
        }
    }

    fn accepts(&self, _graph: &Graph, _scope: ScopeId, _imported: &bool, _: ScopeDatumId) -> bool {
        true
    }

    fn key(&self, graph: &Graph, datum: ScopeDatumId) -> u8 {
        *graph.datum(datum).data()
    }

    fn query_key(&self, _graph: &Graph, query: &u8) -> u8 {
        *query
    }

    fn compare_labels(
        &self,
        _graph: &Graph,
        left: ScopeGraphPathLabel<'_, Label>,
        right: ScopeGraphPathLabel<'_, Label>,
    ) -> Option<Ordering> {
        Some(rank(left).cmp(&rank(right)))
    }
}

#[test]
fn a_cached_index_resolves_every_reference_on_its_first_update() {
    let (mut graph, blocks, modules) = blocks(4);
    for (level, &module) in modules.iter().enumerate() {
        graph.add_datum(module, (), u8::try_from(level).expect("a short chain"));
    }
    let config = ScopeIndexConfig::default();
    let mut index = ScopeEnvironmentCache::new(&graph).index(&graph, &Lookup, config.search);
    let cached: Vec<_> = index
        .resolutions()
        .iter()
        .map(|resolution| resolution.candidates().to_vec())
        .collect();
    let update = index.update(&graph, &Lookup, config, &ScopeGraphChanges::default());
    assert_eq!(update.resolved_count, 4);
    // The candidates agree; only the statistics of the two forms differ.
    for (resolution, candidates) in index.resolutions().iter().zip(&cached) {
        assert_eq!(resolution.candidates(), candidates.as_slice());
    }
    assert_eq!(
        index,
        ScopeResolutionIndex::compute(&graph, &Lookup, config)
    );

    // From then on the update is incremental.
    graph.add_datum(blocks[3], (), 0);
    let changes = ScopeGraphChanges {
        scopes: vec![blocks[3]],
        references: Vec::new(),
    };
    let update = index.update(&graph, &Lookup, config, &changes);
    assert_eq!(update.resolved_count, 1);
    assert_eq!(
        index,
        ScopeResolutionIndex::compute(&graph, &Lookup, config)
    );
}
