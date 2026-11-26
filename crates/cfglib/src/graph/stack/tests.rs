extern crate alloc;

#[cfg(feature = "serde")]
use alloc::string::String;
use alloc::vec;

use crate::graph::edge_view::EdgeView;
use crate::{GraphView, PathSearchConfig, SearchOrder};

use super::{
    StackEdgeId, StackFileId, StackGraph, StackGraphError, StackLinearResolutionError, StackNodeId,
    StackNodeKind, StackPartialPathDatabase, StackPartialPathSet, StackPath, StackPathError,
    StackPathStep, StackResolution, StackResolutionIndex,
};

type Graph = StackGraph<&'static str, &'static str, &'static str, ()>;
type Kind = StackNodeKind<&'static str>;

/// Adds a node that the tests build valid.
fn node(graph: &mut Graph, file: StackFileId, kind: Kind, payload: &'static str) -> StackNodeId {
    graph
        .add_node(file, kind, payload)
        .expect("a test node obeys the rules")
}

const fn scope(exported: bool) -> Kind {
    StackNodeKind::Scope { exported }
}

const fn push(symbol: &'static str, is_reference: bool) -> Kind {
    StackNodeKind::PushSymbol {
        symbol,
        is_reference,
    }
}

const fn pop(symbol: &'static str, is_definition: bool) -> Kind {
    StackNodeKind::PopSymbol {
        symbol,
        is_definition,
    }
}

const fn push_scoped(symbol: &'static str, scope: StackNodeId, is_reference: bool) -> Kind {
    StackNodeKind::PushScopedSymbol {
        symbol,
        scope,
        is_reference,
    }
}

const fn pop_scoped(symbol: &'static str, is_definition: bool) -> Kind {
    StackNodeKind::PopScopedSymbol {
        symbol,
        is_definition,
    }
}

fn add_edge(
    graph: &mut Graph,
    source: StackNodeId,
    target: StackNodeId,
    precedence: i32,
) -> StackEdgeId {
    graph
        .add_edge(source, target, precedence, ())
        .expect("test edge is file-local or touches root")
}

#[test]
fn storage_enforces_file_incremental_boundaries() {
    let mut graph = Graph::new();
    let first = graph.add_file("first.rs");
    let second = graph.add_file("second.rs");
    let exported = node(&mut graph, first, scope(true), "exported");
    let internal = node(&mut graph, first, scope(false), "internal");
    let other = node(&mut graph, second, scope(true), "other");

    assert!(matches!(
        graph.node(graph.root_node()).kind(),
        StackNodeKind::Root
    ));
    assert!(matches!(
        graph.node(graph.jump_to_scope_node()).kind(),
        StackNodeKind::JumpToScope
    ));
    assert_eq!(graph.file_nodes(first), &[exported, internal]);
    assert_eq!(graph.graph().node_count(), 5);
    assert_eq!(graph.node_ids().count(), 5);

    assert_eq!(
        graph.add_node(first, push_scoped("x", internal, false), "bad"),
        Err(StackGraphError::ScopeNotExported(internal))
    );
    assert_eq!(
        graph.add_node(second, push_scoped("x", exported, false), "bad"),
        Err(StackGraphError::ScopeFromOtherFile {
            file: second,
            scope: exported,
        })
    );
    for singleton in [StackNodeKind::Root, StackNodeKind::JumpToScope] {
        assert_eq!(
            graph.add_node(first, singleton, "bad"),
            Err(StackGraphError::SingletonKind)
        );
    }
    assert_eq!(graph.file_nodes(first), &[exported, internal]);
    assert_eq!(
        graph.add_edge(exported, other, 0, ()),
        Err(StackGraphError::CrossFileEdge {
            source: exported,
            target: other,
        })
    );
    assert_eq!(
        graph.add_edge(graph.jump_to_scope_node(), exported, 0, ()),
        Err(StackGraphError::EdgeFromJumpToScope)
    );
}

#[test]
fn direct_resolution_applies_symbol_guards_and_builds_reverse_bindings() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use x");
    let scope = node(&mut graph, file, scope(false), "scope");
    let wrong = node(&mut graph, file, pop("y", true), "def y");
    let definition = node(&mut graph, file, pop("x", true), "def x");
    let first_edge = add_edge(&mut graph, reference, scope, 0);
    add_edge(&mut graph, scope, wrong, 0);
    add_edge(&mut graph, scope, definition, 0);

    assert_eq!(graph.edge(first_edge).source(), reference);
    assert_eq!(
        graph.successors(reference).collect::<alloc::vec::Vec<_>>(),
        vec![scope]
    );
    let resolution = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    assert_eq!(resolution.target_count(), 1);
    assert_eq!(resolution.paths()[0].end(), definition);
    assert_eq!(resolution.stats().rejected_extension_count, 1);
    assert!(!resolution.is_ambiguous());

    let index = StackResolutionIndex::compute(&graph, PathSearchConfig::default());
    assert_eq!(index.references_to(definition), &[reference]);
    assert_eq!(index.references_to(wrong), []);
    assert_eq!(index.resolution(reference), Some(&resolution));
    assert_eq!(index.resolution(scope), None);

    let linear = StackResolutionIndex::try_compute_linear(&graph, PathSearchConfig::default())
        .expect("the only well-formed path is linear");
    assert_eq!(
        linear.references_to(definition),
        index.references_to(definition)
    );
    assert_eq!(linear.references_to(wrong), index.references_to(wrong));
}

#[test]
fn linear_resolution_matches_direct_search_and_rejects_branching() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use x");
    let scope = node(&mut graph, file, scope(false), "scope");
    let definition = node(&mut graph, file, pop("x", true), "def x");
    add_edge(&mut graph, reference, scope, 0);
    add_edge(&mut graph, scope, definition, 0);

    let direct = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    let linear =
        StackResolution::try_compute_linear(&graph, reference, PathSearchConfig::default())
            .expect("the only well-formed path is linear");
    assert_eq!(linear, direct);

    let competing = node(&mut graph, file, pop("x", true), "competing def x");
    add_edge(&mut graph, scope, competing, 0);
    assert_eq!(
        StackResolution::try_compute_linear(&graph, reference, PathSearchConfig::default()),
        Err(StackLinearResolutionError::MultipleEdges { node: scope })
    );
}

#[test]
#[should_panic(expected = "is not a live source reference")]
fn a_seed_that_is_not_a_reference_panics() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let scope = node(&mut graph, file, scope(false), "scope");
    let _ = StackResolution::compute(&graph, scope, PathSearchConfig::default());
}

#[test]
#[should_panic(expected = "is out of range")]
fn a_builder_given_an_unknown_file_panics() {
    let mut graph = Graph::new();
    let _ = graph.add_node(StackFileId::from_index(0), scope(false), "scope");
}

#[test]
#[should_panic(expected = "is out of range or retired")]
fn an_edge_to_a_retired_node_panics() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let retired = node(&mut graph, file, scope(false), "retired");
    graph.clear_file(file);
    let _ = graph.add_edge(graph.root_node(), retired, 0, ());
}

/// `use x -> a -> b -> c -> def x`, with an edge from every scope to a pop of
/// another symbol that no path can take, listed before or after the onward
/// edge.
fn decoyed_chain(decoy_first: bool) -> (Graph, super::StackNodeId, super::StackNodeId) {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use x");
    let decoy = node(&mut graph, file, pop("y", false), "decoy");
    let scopes: alloc::vec::Vec<_> = ["a", "b", "c"]
        .into_iter()
        .map(|name| node(&mut graph, file, scope(false), name))
        .collect();
    let definition = node(&mut graph, file, pop("x", true), "def x");
    add_edge(&mut graph, reference, scopes[0], 0);
    for (index, &scope) in scopes.iter().enumerate() {
        let onward = scopes.get(index + 1).copied().unwrap_or(definition);
        if decoy_first {
            add_edge(&mut graph, scope, decoy, 0);
        }
        add_edge(&mut graph, scope, onward, 0);
        if !decoy_first {
            add_edge(&mut graph, scope, decoy, 0);
        }
    }
    (graph, reference, definition)
}

#[test]
fn linear_resolution_agrees_with_direct_search_whichever_edge_comes_last() {
    let configs = [
        PathSearchConfig::default(),
        PathSearchConfig {
            max_path_length: Some(2),
            ..PathSearchConfig::default()
        },
        PathSearchConfig {
            max_path_count: Some(3),
            ..PathSearchConfig::default()
        },
        PathSearchConfig {
            max_complete_path_count: Some(0),
            ..PathSearchConfig::default()
        },
    ];
    for decoy_first in [true, false] {
        let (graph, reference, definition) = decoyed_chain(decoy_first);
        for config in configs {
            let direct = StackResolution::compute(&graph, reference, config);
            let linear = StackResolution::try_compute_linear(&graph, reference, config)
                .expect("the decoy is rejected, so the chain is linear");
            assert_eq!(linear, direct, "decoy first: {decoy_first}, {config:?}");
        }
        let full = StackResolution::try_compute_linear(&graph, reference, configs[0])
            .expect("the chain is linear");
        assert_eq!(full.paths().len(), 1);
        assert_eq!(full.paths()[0].end(), definition);
        assert_eq!(full.stats().rejected_extension_count, 3);
    }
}

#[test]
fn linear_resolution_cuts_a_cycle_it_walks_into() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use x");
    let a = node(&mut graph, file, scope(false), "a");
    let b = node(&mut graph, file, scope(false), "b");
    add_edge(&mut graph, reference, a, 0);
    add_edge(&mut graph, a, b, 0);
    add_edge(&mut graph, b, a, 0);

    let direct = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    let linear =
        StackResolution::try_compute_linear(&graph, reference, PathSearchConfig::default())
            .expect("the cycle has one way on at every step");
    assert_eq!(linear, direct);
    assert_eq!(linear.stats().cycle_cut_count, 1);
    assert!(!linear.is_resolved());
}

#[test]
fn greater_edge_precedence_shadows_a_competing_definition() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use");
    let scope = node(&mut graph, file, scope(false), "scope");
    let preferred = node(&mut graph, file, pop("x", true), "preferred");
    let fallback = node(&mut graph, file, pop("x", true), "fallback");
    add_edge(&mut graph, reference, scope, 0);
    let preferred_edge = add_edge(&mut graph, scope, preferred, 2);
    add_edge(&mut graph, scope, fallback, 1);

    let resolution = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    assert_eq!(resolution.stats().search.complete_path_count, 2);
    assert_eq!(resolution.paths().len(), 1);
    assert_eq!(resolution.paths()[0].end(), preferred);
    assert_eq!(graph.edge(preferred_edge).data().precedence(), 2);

    *graph.edge_mut(preferred_edge).precedence_mut() = 1;
    let ambiguous = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    assert_eq!(ambiguous.target_count(), 2);
    assert!(ambiguous.is_ambiguous());
}

#[test]
fn scoped_symbols_pause_a_lookup_and_jump_to_an_exported_scope() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let member_reference = node(&mut graph, file, push("member", true), "member use");
    let member_scope = node(&mut graph, file, scope(true), "receiver members");
    let push_receiver = node(
        &mut graph,
        file,
        push_scoped("receiver", member_scope, false),
        "pause",
    );
    let pop_receiver = node(&mut graph, file, pop_scoped("receiver", false), "resume");
    let member_definition = node(&mut graph, file, pop("member", true), "member def");
    add_edge(&mut graph, member_reference, push_receiver, 0);
    add_edge(&mut graph, push_receiver, pop_receiver, 0);
    let jump_to_scope = graph.jump_to_scope_node();
    add_edge(&mut graph, pop_receiver, jump_to_scope, 0);
    add_edge(&mut graph, member_scope, member_definition, 0);

    let resolution =
        StackResolution::compute(&graph, member_reference, PathSearchConfig::default());
    assert_eq!(resolution.paths().len(), 1);
    let path = &resolution.paths()[0];
    assert_eq!(path.end(), member_definition);
    assert_eq!(path.symbol_stack(), []);
    assert_eq!(path.scope_stack(), []);
    assert!(path.steps().iter().any(|step| matches!(
        step,
        StackPathStep::Jump { target, .. } if *target == member_scope
    )));
}

#[test]
fn path_extension_errors_are_transactional() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use");
    let wrong = node(&mut graph, file, pop("y", true), "wrong");
    let edge = add_edge(&mut graph, reference, wrong, 0);
    let mut path = StackPath::from_reference(&graph, reference);
    let before = path.clone();

    assert_eq!(
        path.append(&graph, edge),
        Err(StackPathError::IncorrectPoppedSymbol(wrong))
    );
    assert_eq!(path, before);
    assert_eq!(
        path.append(&graph, super::StackEdgeId::from_raw(u32::MAX)),
        Err(StackPathError::UnknownEdge(super::StackEdgeId::from_raw(
            u32::MAX
        )))
    );
}

#[test]
fn a_jump_without_a_scope_leaves_the_path_unchanged() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use");
    let jump_to_scope = graph.jump_to_scope_node();
    let edge = add_edge(&mut graph, reference, jump_to_scope, 0);
    let mut path = StackPath::from_reference(&graph, reference);
    let before = path.clone();

    assert_eq!(
        path.append(&graph, edge),
        Err(StackPathError::EmptyScopeStack(jump_to_scope))
    );
    assert_eq!(path, before);
}

#[test]
fn a_summary_replays_onto_the_path_it_is_given() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use");
    let scope = node(&mut graph, file, scope(false), "scope");
    let definition = node(&mut graph, file, pop("x", true), "def");
    let wrong = node(&mut graph, file, pop("y", true), "wrong");
    let first = add_edge(&mut graph, reference, scope, 0);
    let second = add_edge(&mut graph, scope, definition, 0);
    add_edge(&mut graph, scope, wrong, 0);
    let set = StackPartialPathSet::compute(&graph, file, PathSearchConfig::default());
    let path = StackPath::from_reference(&graph, reference);

    let summary = |end| {
        set.paths()
            .iter()
            .find(|summary| summary.start() == reference && summary.end() == end)
            .expect("extraction records a summary at every definition")
    };
    let replayed = summary(definition)
        .append_to(&graph, path.clone())
        .expect("the summary spells `x`");
    assert_eq!(replayed.end(), definition);
    assert!(replayed.contains_edge(first) && replayed.contains_edge(second));
    assert!(replayed.is_complete(&graph));
    assert_eq!(
        summary(wrong).append_to(&graph, path),
        Err(StackPathError::IncorrectPoppedSymbol(wrong))
    );
}

#[test]
fn root_edges_join_independently_constructed_file_partitions() {
    let mut graph = Graph::new();
    let consumer = graph.add_file("consumer.rs");
    let provider = graph.add_file("provider.rs");
    let reference = node(&mut graph, consumer, push("package", true), "import");
    let definition = node(&mut graph, provider, pop("package", true), "module");
    let root = graph.root_node();
    add_edge(&mut graph, reference, root, 0);
    add_edge(&mut graph, root, definition, 0);

    let resolution = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    assert_eq!(resolution.target_count(), 1);
    assert_eq!(resolution.paths()[0].end(), definition);
}

#[test]
fn search_bounds_and_cycle_guards_are_reported() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use");
    let scope = node(&mut graph, file, scope(false), "scope");
    let definition = node(&mut graph, file, pop("x", true), "def");
    add_edge(&mut graph, reference, scope, 0);
    add_edge(&mut graph, scope, reference, 0);
    add_edge(&mut graph, scope, definition, 0);

    let cycle_guarded = StackResolution::compute(
        &graph,
        reference,
        PathSearchConfig {
            order: SearchOrder::DepthFirst,
            ..PathSearchConfig::default()
        },
    );
    assert!(cycle_guarded.is_resolved());
    assert!(cycle_guarded.stats().cycle_cut_count > 0);

    let length_limited = StackResolution::compute(
        &graph,
        reference,
        PathSearchConfig {
            max_path_length: Some(1),
            ..PathSearchConfig::default()
        },
    );
    assert!(!length_limited.is_resolved());
    assert!(length_limited.stats().search.path_length_limited);

    let count_limited = StackResolution::compute(
        &graph,
        reference,
        PathSearchConfig {
            max_path_count: Some(1),
            ..PathSearchConfig::default()
        },
    );
    assert!(!count_limited.is_resolved());
    assert!(count_limited.stats().search.path_count_limited);

    let complete_limited = StackResolution::compute(
        &graph,
        reference,
        PathSearchConfig {
            max_complete_path_count: Some(0),
            ..PathSearchConfig::default()
        },
    );
    assert!(!complete_limited.is_resolved());
    assert!(complete_limited.stats().search.complete_path_count_limited);
}

#[test]
fn stitched_resolution_matches_direct_precedence_and_ambiguity() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use");
    let scope = node(&mut graph, file, scope(false), "scope");
    let preferred = node(&mut graph, file, pop("x", true), "preferred");
    let fallback = node(&mut graph, file, pop("x", true), "fallback");
    add_edge(&mut graph, reference, scope, 0);
    let preferred_edge = add_edge(&mut graph, scope, preferred, 2);
    add_edge(&mut graph, scope, fallback, 1);
    let database = StackPartialPathDatabase::compute(&graph, PathSearchConfig::default());

    let direct = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    let stitched = StackResolution::try_compute_stitched(
        &graph,
        &database,
        reference,
        PathSearchConfig::default(),
    )
    .expect("the database matches the graph");
    assert_eq!(stitched.paths(), direct.paths());

    *graph.edge_mut(preferred_edge).precedence_mut() = 1;
    let direct = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    let stitched = StackResolution::try_compute_stitched(
        &graph,
        &database,
        reference,
        PathSearchConfig::default(),
    )
    .expect("the database matches the graph");
    assert_eq!(stitched.paths(), direct.paths());
    assert!(stitched.is_ambiguous());
}

#[test]
fn partial_extraction_continues_through_incomplete_definition_nodes() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use x");
    let push_y = node(&mut graph, file, push("y", false), "push y");
    let intermediate_definition = node(&mut graph, file, pop("y", true), "def y");
    let final_definition = node(&mut graph, file, pop("x", true), "def x");
    add_edge(&mut graph, reference, push_y, 0);
    add_edge(&mut graph, push_y, intermediate_definition, 0);
    add_edge(&mut graph, intermediate_definition, final_definition, 0);

    let direct = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    let database = StackPartialPathDatabase::compute(&graph, PathSearchConfig::default());
    let stitched = StackResolution::try_compute_stitched(
        &graph,
        &database,
        reference,
        PathSearchConfig::default(),
    )
    .expect("the database matches the graph");

    assert_eq!(direct.paths(), stitched.paths());
    assert_eq!(stitched.paths().len(), 1);
    assert_eq!(stitched.paths()[0].end(), final_definition);
}

#[test]
fn stitched_resolution_crosses_files_and_follows_scoped_jumps() {
    let mut graph = Graph::new();
    let consumer = graph.add_file("consumer.rs");
    let provider = graph.add_file("provider.rs");
    let reference = node(&mut graph, consumer, push("member", true), "member use");
    let member_scope = node(&mut graph, provider, scope(true), "members");
    let push_receiver = graph
        .add_node(
            consumer,
            StackNodeKind::PushScopedSymbol {
                symbol: "receiver",
                scope: member_scope,
                is_reference: false,
            },
            "pause",
        )
        .expect_err("cross-file scoped symbols require a file-local exported scope");
    assert_eq!(
        push_receiver,
        StackGraphError::ScopeFromOtherFile {
            file: consumer,
            scope: member_scope,
        }
    );

    let consumer_scope = node(&mut graph, consumer, scope(true), "imported receiver");
    let push_receiver = node(
        &mut graph,
        consumer,
        push_scoped("receiver", consumer_scope, false),
        "pause",
    );
    let pop_receiver = node(
        &mut graph,
        consumer,
        pop_scoped("receiver", false),
        "resume",
    );
    let definition = node(&mut graph, provider, pop("member", true), "member def");
    add_edge(&mut graph, reference, push_receiver, 0);
    add_edge(&mut graph, push_receiver, pop_receiver, 0);
    let jump = graph.jump_to_scope_node();
    add_edge(&mut graph, pop_receiver, jump, 0);
    let root = graph.root_node();
    add_edge(&mut graph, consumer_scope, root, 0);
    add_edge(&mut graph, root, member_scope, 0);
    add_edge(&mut graph, member_scope, definition, 0);

    let database = StackPartialPathDatabase::compute(&graph, PathSearchConfig::default());
    let direct = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    let stitched = StackResolution::try_compute_stitched(
        &graph,
        &database,
        reference,
        PathSearchConfig::default(),
    )
    .expect("the database matches the graph");
    assert_eq!(stitched.paths(), direct.paths());
    assert_eq!(stitched.paths()[0].end(), definition);
    assert!(stitched.paths()[0].steps().iter().any(|step| matches!(
        step,
        StackPathStep::Jump { target, .. } if *target == consumer_scope
    )));
}

#[test]
fn replacing_one_files_partials_preserves_other_identities() {
    let mut graph = Graph::new();
    let first = graph.add_file("first.rs");
    let second = graph.add_file("second.rs");
    let first_reference = node(&mut graph, first, push("x", true), "first use");
    let first_definition = node(&mut graph, first, pop("x", true), "first def");
    let second_reference = node(&mut graph, second, push("y", true), "second use");
    let second_definition = node(&mut graph, second, pop("y", true), "second def");
    add_edge(&mut graph, first_reference, first_definition, 0);
    add_edge(&mut graph, second_reference, second_definition, 0);
    let mut database = StackPartialPathDatabase::compute(&graph, PathSearchConfig::default());
    let first_ids = database
        .file_paths(first)
        .map(|(id, _)| id)
        .collect::<alloc::vec::Vec<_>>();
    let second_ids = database
        .file_paths(second)
        .map(|(id, _)| id)
        .collect::<alloc::vec::Vec<_>>();
    let old_slot_count = database.path_slot_count();

    database.replace_file(&graph, first, PathSearchConfig::default());

    assert!(first_ids.iter().all(|&id| database.path(id).is_none()));
    assert!(second_ids.iter().all(|&id| database.path(id).is_some()));
    assert!(database.path_slot_count() > old_slot_count);
    let index =
        StackResolutionIndex::try_compute_stitched(&graph, &database, PathSearchConfig::default())
            .expect("the database matches the graph");
    assert_eq!(index.references_to(first_definition), &[first_reference]);
    assert_eq!(index.references_to(second_definition), &[second_reference]);
}

#[test]
fn changed_file_partitions_rebuild_without_disturbing_other_files() {
    let mut graph = Graph::new();
    let changed = graph.add_file("changed.rs");
    let stable = graph.add_file("stable.rs");
    let old_reference = node(&mut graph, changed, push("old", true), "old use");
    let old_definition = node(&mut graph, changed, pop("old", true), "old def");
    let old_edge = add_edge(&mut graph, old_reference, old_definition, 0);
    let stable_reference = node(&mut graph, stable, push("stable", true), "stable use");
    let stable_definition = node(&mut graph, stable, pop("stable", true), "stable def");
    add_edge(&mut graph, stable_reference, stable_definition, 0);
    let mut database = StackPartialPathDatabase::compute(&graph, PathSearchConfig::default());
    let old_partial_ids = database
        .file_paths(changed)
        .map(|(id, _)| id)
        .collect::<alloc::vec::Vec<_>>();
    let stable_partial_ids = database
        .file_paths(stable)
        .map(|(id, _)| id)
        .collect::<alloc::vec::Vec<_>>();
    let old_slot_count = graph.node_bound();

    assert_eq!(
        graph.clear_file(changed),
        vec![old_reference, old_definition]
    );
    assert!(!graph.graph().contains_node(old_reference));
    assert!(!graph.graph().contains_node(old_definition));
    assert!(!graph.graph().contains_edge(old_edge));
    let new_reference = node(&mut graph, changed, push("new", true), "new use");
    let new_definition = node(&mut graph, changed, pop("new", true), "new def");
    add_edge(&mut graph, new_reference, new_definition, 0);
    database.replace_file(&graph, changed, PathSearchConfig::default());

    assert!(graph.node_bound() > old_slot_count);
    assert_eq!(graph.graph().node_count(), 6);
    assert!(
        old_partial_ids
            .iter()
            .all(|&id| database.path(id).is_none())
    );
    assert!(
        stable_partial_ids
            .iter()
            .all(|&id| database.path(id).is_some())
    );
    let index =
        StackResolutionIndex::try_compute_stitched(&graph, &database, PathSearchConfig::default())
            .expect("the database matches the graph");
    assert_eq!(index.resolution(old_reference), None);
    assert_eq!(index.references_to(old_definition), []);
    assert_eq!(index.references_to(new_definition), &[new_reference]);
    assert_eq!(index.references_to(stable_definition), &[stable_reference]);
}

#[test]
fn stale_partial_path_edges_report_a_structured_error() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use");
    let definition = node(&mut graph, file, pop("x", true), "def");
    let edge = add_edge(&mut graph, reference, definition, 0);
    let database = StackPartialPathDatabase::compute(&graph, PathSearchConfig::default());
    assert!(graph.remove_edge(edge));

    assert_eq!(
        StackResolution::try_compute_stitched(
            &graph,
            &database,
            reference,
            PathSearchConfig::default(),
        ),
        Err(StackPathError::UnknownEdge(edge))
    );
    assert_eq!(
        StackResolutionIndex::try_compute_stitched(&graph, &database, PathSearchConfig::default()),
        Err(StackPathError::UnknownEdge(edge))
    );
}

#[test]
fn partial_path_extraction_reports_bounds() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = node(&mut graph, file, push("x", true), "use");
    let scope = node(&mut graph, file, scope(false), "scope");
    let definition = node(&mut graph, file, pop("x", true), "def");
    add_edge(&mut graph, reference, scope, 0);
    add_edge(&mut graph, scope, definition, 0);

    let length_limited = StackPartialPathSet::compute(
        &graph,
        file,
        PathSearchConfig {
            max_path_length: Some(1),
            ..PathSearchConfig::default()
        },
    );
    assert!(length_limited.stats().search.path_length_limited);
    assert!(length_limited.stats().search.is_truncated());

    let count_limited = StackPartialPathSet::compute(
        &graph,
        file,
        PathSearchConfig {
            max_path_count: Some(0),
            ..PathSearchConfig::default()
        },
    );
    assert!(count_limited.stats().search.path_count_limited);
    assert_eq!(count_limited.paths(), []);
}

#[cfg(feature = "serde")]
#[test]
fn graph_and_partial_database_round_trip_together() {
    let mut graph = StackGraph::<String, String, String, ()>::new();
    let file = graph.add_file("main.rs".into());
    let reference = graph
        .add_node(
            file,
            StackNodeKind::PushSymbol {
                symbol: "x".into(),
                is_reference: true,
            },
            "use".into(),
        )
        .expect("a file node");
    let definition = graph
        .add_node(
            file,
            StackNodeKind::PopSymbol {
                symbol: "x".into(),
                is_definition: true,
            },
            "def".into(),
        )
        .expect("a file node");
    graph
        .add_edge(reference, definition, 0, ())
        .expect("file-local edge");
    let database = StackPartialPathDatabase::compute(&graph, PathSearchConfig::default());

    let graph_json = serde_json::to_string(&graph).expect("serializable graph");
    let database_json = serde_json::to_string(&database).expect("serializable database");
    let decoded_graph: StackGraph<String, String, String, ()> =
        serde_json::from_str(&graph_json).expect("valid graph encoding");
    let decoded_database: StackPartialPathDatabase =
        serde_json::from_str(&database_json).expect("valid database encoding");

    assert_eq!(decoded_graph, graph);
    assert_eq!(decoded_database, database);
    let resolution = StackResolution::try_compute_stitched(
        &decoded_graph,
        &decoded_database,
        reference,
        PathSearchConfig::default(),
    )
    .expect("the database matches the graph");
    assert_eq!(resolution.paths()[0].end(), definition);
}
