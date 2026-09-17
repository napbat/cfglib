extern crate alloc;

#[cfg(feature = "serde")]
use alloc::string::String;
use alloc::vec;

use crate::{GraphView, SearchOrder};

use super::{
    StackGraph, StackGraphError, StackLinearResolutionError, StackNodeKind, StackPartialPathConfig,
    StackPartialPathDatabase, StackPartialPathSet, StackPath, StackPathError, StackPathStep,
    StackResolution, StackResolutionIndex, StackReverseIndex, StackSearchConfig,
};

type Graph = StackGraph<&'static str, &'static str, &'static str, ()>;

fn add_edge(
    graph: &mut Graph,
    source: super::StackNodeId,
    target: super::StackNodeId,
    precedence: i32,
) -> super::StackEdgeId {
    graph
        .add_edge(source, target, precedence, ())
        .expect("test edge is file-local or touches root")
}

#[test]
fn storage_enforces_file_incremental_boundaries() {
    let mut graph = Graph::new();
    let first = graph.add_file("first.rs");
    let second = graph.add_file("second.rs");
    let exported = graph
        .add_scope_node(first, true, "exported")
        .expect("known file");
    let internal = graph
        .add_scope_node(first, false, "internal")
        .expect("known file");
    let other = graph
        .add_scope_node(second, true, "other")
        .expect("known file");

    assert!(matches!(
        graph.node(graph.root_node()).kind(),
        StackNodeKind::Root
    ));
    assert!(matches!(
        graph.node(graph.jump_to_scope_node()).kind(),
        StackNodeKind::JumpToScope
    ));
    assert_eq!(graph.file_nodes(first), &[exported, internal]);
    assert_eq!(graph.node_count(), 5);
    assert_eq!(graph.node_ids().count(), 5);

    assert_eq!(
        graph.add_push_scoped_symbol_node(first, "x", internal, false, "bad"),
        Err(StackGraphError::ScopeNotExported(internal))
    );
    assert_eq!(
        graph.add_push_scoped_symbol_node(second, "x", exported, false, "bad"),
        Err(StackGraphError::ScopeFromOtherFile {
            file: second,
            scope: exported,
        })
    );
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
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use x")
        .expect("known file");
    let scope = graph
        .add_scope_node(file, false, "scope")
        .expect("known file");
    let wrong = graph
        .add_pop_symbol_node(file, "y", true, "def y")
        .expect("known file");
    let definition = graph
        .add_pop_symbol_node(file, "x", true, "def x")
        .expect("known file");
    let first_edge = add_edge(&mut graph, reference, scope, 0);
    add_edge(&mut graph, scope, wrong, 0);
    add_edge(&mut graph, scope, definition, 0);

    assert_eq!(graph.edge(first_edge).source(), reference);
    assert_eq!(
        graph.successors(reference).collect::<alloc::vec::Vec<_>>(),
        vec![scope]
    );
    let resolution = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    assert_eq!(resolution.target_count(), 1);
    assert_eq!(resolution.paths()[0].end(), definition);
    assert_eq!(resolution.stats().rejected_extension_count, 1);
    assert!(!resolution.is_ambiguous());

    let index = StackResolutionIndex::compute(&graph, StackSearchConfig::new());
    assert_eq!(index.references_to(definition), &[reference]);
    assert!(index.references_to(wrong).is_empty());
    assert_eq!(index.resolution(reference), Some(&resolution));
    assert_eq!(index.resolution(scope), None);

    let reverse = StackReverseIndex::try_compute_linear(&graph, StackSearchConfig::new())
        .expect("the only well-formed path is linear");
    assert_eq!(
        reverse.references_to(definition),
        index.references_to(definition)
    );
    assert_eq!(reverse.references_to(wrong), index.references_to(wrong));
}

#[test]
fn linear_resolution_matches_direct_search_and_rejects_branching() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use x")
        .expect("known file");
    let scope = graph
        .add_scope_node(file, false, "scope")
        .expect("known file");
    let definition = graph
        .add_pop_symbol_node(file, "x", true, "def x")
        .expect("known file");
    add_edge(&mut graph, reference, scope, 0);
    add_edge(&mut graph, scope, definition, 0);

    let direct = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    let linear = StackResolution::try_compute_linear(&graph, reference, StackSearchConfig::new())
        .expect("the only well-formed path is linear");
    assert_eq!(linear, direct);

    let competing = graph
        .add_pop_symbol_node(file, "x", true, "competing def x")
        .expect("known file");
    add_edge(&mut graph, scope, competing, 0);
    assert_eq!(
        StackResolution::try_compute_linear(&graph, reference, StackSearchConfig::new()),
        Err(StackLinearResolutionError::MultipleEdges { node: scope })
    );
    assert_eq!(
        StackResolution::try_compute_linear(&graph, scope, StackSearchConfig::new()),
        Err(StackLinearResolutionError::Path(
            StackPathError::NotReference(scope)
        ))
    );
}

#[test]
fn greater_edge_precedence_shadows_a_competing_definition() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use")
        .expect("known file");
    let scope = graph
        .add_scope_node(file, false, "scope")
        .expect("known file");
    let preferred = graph
        .add_pop_symbol_node(file, "x", true, "preferred")
        .expect("known file");
    let fallback = graph
        .add_pop_symbol_node(file, "x", true, "fallback")
        .expect("known file");
    add_edge(&mut graph, reference, scope, 0);
    let preferred_edge = add_edge(&mut graph, scope, preferred, 2);
    add_edge(&mut graph, scope, fallback, 1);

    let resolution = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    assert_eq!(resolution.stats().complete_path_count, 2);
    assert_eq!(resolution.paths().len(), 1);
    assert_eq!(resolution.paths()[0].end(), preferred);
    assert_eq!(graph.edge(preferred_edge).data().precedence(), 2);

    graph.set_edge_precedence(preferred_edge, 1);
    let ambiguous = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    assert_eq!(ambiguous.target_count(), 2);
    assert!(ambiguous.is_ambiguous());
}

#[test]
fn scoped_symbols_pause_a_lookup_and_jump_to_an_exported_scope() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let member_reference = graph
        .add_push_symbol_node(file, "member", true, "member use")
        .expect("known file");
    let member_scope = graph
        .add_scope_node(file, true, "receiver members")
        .expect("known file");
    let push_receiver = graph
        .add_push_scoped_symbol_node(file, "receiver", member_scope, false, "pause")
        .expect("exported local scope");
    let pop_receiver = graph
        .add_pop_scoped_symbol_node(file, "receiver", false, "resume")
        .expect("known file");
    let member_definition = graph
        .add_pop_symbol_node(file, "member", true, "member def")
        .expect("known file");
    add_edge(&mut graph, member_reference, push_receiver, 0);
    add_edge(&mut graph, push_receiver, pop_receiver, 0);
    let jump_to_scope = graph.jump_to_scope_node();
    add_edge(&mut graph, pop_receiver, jump_to_scope, 0);
    add_edge(&mut graph, member_scope, member_definition, 0);

    let resolution = StackResolution::compute(&graph, member_reference, StackSearchConfig::new());
    assert_eq!(resolution.paths().len(), 1);
    let path = &resolution.paths()[0];
    assert_eq!(path.end(), member_definition);
    assert!(path.symbol_stack().is_empty());
    assert!(path.scope_stack().is_empty());
    assert!(path.steps().iter().any(|step| matches!(
        step,
        StackPathStep::Jump { target, .. } if *target == member_scope
    )));
}

#[test]
fn path_extension_errors_are_transactional() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use")
        .expect("known file");
    let wrong = graph
        .add_pop_symbol_node(file, "y", true, "wrong")
        .expect("known file");
    let edge = add_edge(&mut graph, reference, wrong, 0);
    let mut path = StackPath::from_reference(&graph, reference).expect("reference path");
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
fn root_edges_join_independently_constructed_file_partitions() {
    let mut graph = Graph::new();
    let consumer = graph.add_file("consumer.rs");
    let provider = graph.add_file("provider.rs");
    let reference = graph
        .add_push_symbol_node(consumer, "package", true, "import")
        .expect("known file");
    let definition = graph
        .add_pop_symbol_node(provider, "package", true, "module")
        .expect("known file");
    let root = graph.root_node();
    add_edge(&mut graph, reference, root, 0);
    add_edge(&mut graph, root, definition, 0);

    let resolution = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    assert_eq!(resolution.target_count(), 1);
    assert_eq!(resolution.paths()[0].end(), definition);
}

#[test]
fn search_bounds_and_cycle_guards_are_reported() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use")
        .expect("known file");
    let scope = graph
        .add_scope_node(file, false, "scope")
        .expect("known file");
    let definition = graph
        .add_pop_symbol_node(file, "x", true, "def")
        .expect("known file");
    add_edge(&mut graph, reference, scope, 0);
    add_edge(&mut graph, scope, reference, 0);
    add_edge(&mut graph, scope, definition, 0);

    let cycle_guarded = StackResolution::compute(
        &graph,
        reference,
        StackSearchConfig::new().with_order(SearchOrder::DepthFirst),
    );
    assert!(cycle_guarded.is_resolved());
    assert!(cycle_guarded.stats().cycle_cut_count > 0);

    let length_limited = StackResolution::compute(
        &graph,
        reference,
        StackSearchConfig::new().with_max_path_length(1),
    );
    assert!(!length_limited.is_resolved());
    assert!(length_limited.stats().path_length_limited);

    let count_limited = StackResolution::compute(
        &graph,
        reference,
        StackSearchConfig::new().with_max_path_count(1),
    );
    assert!(!count_limited.is_resolved());
    assert!(count_limited.stats().path_count_limited);

    let complete_limited = StackResolution::compute(
        &graph,
        reference,
        StackSearchConfig::new().with_max_complete_path_count(0),
    );
    assert!(!complete_limited.is_resolved());
    assert!(complete_limited.stats().complete_path_count_limited);
}

#[test]
fn stitched_resolution_matches_direct_precedence_and_ambiguity() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use")
        .expect("known file");
    let scope = graph
        .add_scope_node(file, false, "scope")
        .expect("known file");
    let preferred = graph
        .add_pop_symbol_node(file, "x", true, "preferred")
        .expect("known file");
    let fallback = graph
        .add_pop_symbol_node(file, "x", true, "fallback")
        .expect("known file");
    add_edge(&mut graph, reference, scope, 0);
    let preferred_edge = add_edge(&mut graph, scope, preferred, 2);
    add_edge(&mut graph, scope, fallback, 1);
    let database = StackPartialPathDatabase::compute(&graph, StackPartialPathConfig::new());

    let direct = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    let stitched = StackResolution::compute_from_partials(
        &graph,
        &database,
        reference,
        StackSearchConfig::new(),
    );
    assert_eq!(stitched.paths(), direct.paths());

    graph.set_edge_precedence(preferred_edge, 1);
    let direct = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    let stitched = StackResolution::compute_from_partials(
        &graph,
        &database,
        reference,
        StackSearchConfig::new(),
    );
    assert_eq!(stitched.paths(), direct.paths());
    assert!(stitched.is_ambiguous());
}

#[test]
fn partial_extraction_continues_through_incomplete_definition_nodes() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use x")
        .expect("known file");
    let push_y = graph
        .add_push_symbol_node(file, "y", false, "push y")
        .expect("known file");
    let intermediate_definition = graph
        .add_pop_symbol_node(file, "y", true, "def y")
        .expect("known file");
    let final_definition = graph
        .add_pop_symbol_node(file, "x", true, "def x")
        .expect("known file");
    add_edge(&mut graph, reference, push_y, 0);
    add_edge(&mut graph, push_y, intermediate_definition, 0);
    add_edge(&mut graph, intermediate_definition, final_definition, 0);

    let direct = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    let database = StackPartialPathDatabase::compute(&graph, StackPartialPathConfig::new());
    let stitched = StackResolution::compute_from_partials(
        &graph,
        &database,
        reference,
        StackSearchConfig::new(),
    );

    assert_eq!(direct.paths(), stitched.paths());
    assert_eq!(stitched.paths().len(), 1);
    assert_eq!(stitched.paths()[0].end(), final_definition);
}

#[test]
fn stitched_resolution_crosses_files_and_follows_scoped_jumps() {
    let mut graph = Graph::new();
    let consumer = graph.add_file("consumer.rs");
    let provider = graph.add_file("provider.rs");
    let reference = graph
        .add_push_symbol_node(consumer, "member", true, "member use")
        .expect("known file");
    let member_scope = graph
        .add_scope_node(provider, true, "members")
        .expect("known file");
    let push_receiver = graph
        .add_push_scoped_symbol_node(consumer, "receiver", member_scope, false, "pause")
        .expect_err("cross-file scoped symbols require a file-local exported scope");
    assert_eq!(
        push_receiver,
        StackGraphError::ScopeFromOtherFile {
            file: consumer,
            scope: member_scope,
        }
    );

    let consumer_scope = graph
        .add_scope_node(consumer, true, "imported receiver")
        .expect("known file");
    let push_receiver = graph
        .add_push_scoped_symbol_node(consumer, "receiver", consumer_scope, false, "pause")
        .expect("file-local exported scope");
    let pop_receiver = graph
        .add_pop_scoped_symbol_node(consumer, "receiver", false, "resume")
        .expect("known file");
    let definition = graph
        .add_pop_symbol_node(provider, "member", true, "member def")
        .expect("known file");
    add_edge(&mut graph, reference, push_receiver, 0);
    add_edge(&mut graph, push_receiver, pop_receiver, 0);
    let jump = graph.jump_to_scope_node();
    add_edge(&mut graph, pop_receiver, jump, 0);
    let root = graph.root_node();
    add_edge(&mut graph, consumer_scope, root, 0);
    add_edge(&mut graph, root, member_scope, 0);
    add_edge(&mut graph, member_scope, definition, 0);

    let database = StackPartialPathDatabase::compute(&graph, StackPartialPathConfig::new());
    let direct = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    let stitched = StackResolution::compute_from_partials(
        &graph,
        &database,
        reference,
        StackSearchConfig::new(),
    );
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
    let first_reference = graph
        .add_push_symbol_node(first, "x", true, "first use")
        .expect("known file");
    let first_definition = graph
        .add_pop_symbol_node(first, "x", true, "first def")
        .expect("known file");
    let second_reference = graph
        .add_push_symbol_node(second, "y", true, "second use")
        .expect("known file");
    let second_definition = graph
        .add_pop_symbol_node(second, "y", true, "second def")
        .expect("known file");
    add_edge(&mut graph, first_reference, first_definition, 0);
    add_edge(&mut graph, second_reference, second_definition, 0);
    let mut database = StackPartialPathDatabase::compute(&graph, StackPartialPathConfig::new());
    let first_ids = database
        .file_paths(first)
        .map(|(id, _)| id)
        .collect::<alloc::vec::Vec<_>>();
    let second_ids = database
        .file_paths(second)
        .map(|(id, _)| id)
        .collect::<alloc::vec::Vec<_>>();
    let old_slot_count = database.path_slot_count();

    database
        .replace_file(&graph, first, StackPartialPathConfig::new())
        .expect("known file");

    assert!(first_ids.iter().all(|&id| database.path(id).is_none()));
    assert!(second_ids.iter().all(|&id| database.path(id).is_some()));
    assert!(database.path_slot_count() > old_slot_count);
    let index =
        StackResolutionIndex::compute_from_partials(&graph, &database, StackSearchConfig::new());
    assert_eq!(index.references_to(first_definition), &[first_reference]);
    assert_eq!(index.references_to(second_definition), &[second_reference]);
}

#[test]
fn changed_file_partitions_rebuild_without_disturbing_other_files() {
    let mut graph = Graph::new();
    let changed = graph.add_file("changed.rs");
    let stable = graph.add_file("stable.rs");
    let old_reference = graph
        .add_push_symbol_node(changed, "old", true, "old use")
        .expect("known file");
    let old_definition = graph
        .add_pop_symbol_node(changed, "old", true, "old def")
        .expect("known file");
    let old_edge = add_edge(&mut graph, old_reference, old_definition, 0);
    let stable_reference = graph
        .add_push_symbol_node(stable, "stable", true, "stable use")
        .expect("known file");
    let stable_definition = graph
        .add_pop_symbol_node(stable, "stable", true, "stable def")
        .expect("known file");
    add_edge(&mut graph, stable_reference, stable_definition, 0);
    let mut database = StackPartialPathDatabase::compute(&graph, StackPartialPathConfig::new());
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
        graph.clear_file(changed).expect("known file"),
        vec![old_reference, old_definition]
    );
    assert!(!graph.contains_node(old_reference));
    assert!(!graph.contains_node(old_definition));
    assert!(!graph.contains_edge(old_edge));
    let new_reference = graph
        .add_push_symbol_node(changed, "new", true, "new use")
        .expect("known file");
    let new_definition = graph
        .add_pop_symbol_node(changed, "new", true, "new def")
        .expect("known file");
    add_edge(&mut graph, new_reference, new_definition, 0);
    database
        .replace_file(&graph, changed, StackPartialPathConfig::new())
        .expect("known file");

    assert!(graph.node_bound() > old_slot_count);
    assert_eq!(graph.node_count(), 6);
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
        StackResolutionIndex::compute_from_partials(&graph, &database, StackSearchConfig::new());
    assert_eq!(index.resolution(old_reference), None);
    assert!(index.references_to(old_definition).is_empty());
    assert_eq!(index.references_to(new_definition), &[new_reference]);
    assert_eq!(index.references_to(stable_definition), &[stable_reference]);
}

#[test]
fn stale_partial_path_edges_report_a_structured_error() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use")
        .expect("known file");
    let definition = graph
        .add_pop_symbol_node(file, "x", true, "def")
        .expect("known file");
    let edge = add_edge(&mut graph, reference, definition, 0);
    let database = StackPartialPathDatabase::compute(&graph, StackPartialPathConfig::new());
    assert!(graph.remove_edge(edge));

    assert_eq!(
        StackResolution::try_compute_from_partials(
            &graph,
            &database,
            reference,
            StackSearchConfig::new(),
        ),
        Err(StackPathError::UnknownEdge(edge))
    );
}

#[test]
fn partial_path_extraction_reports_bounds() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use")
        .expect("known file");
    let scope = graph
        .add_scope_node(file, false, "scope")
        .expect("known file");
    let definition = graph
        .add_pop_symbol_node(file, "x", true, "def")
        .expect("known file");
    add_edge(&mut graph, reference, scope, 0);
    add_edge(&mut graph, scope, definition, 0);

    let length_limited = StackPartialPathSet::compute(
        &graph,
        file,
        StackPartialPathConfig::new().with_max_path_length(1),
    )
    .expect("known file");
    assert!(length_limited.stats().path_length_limited);
    assert!(length_limited.stats().is_truncated());

    let count_limited = StackPartialPathSet::compute(
        &graph,
        file,
        StackPartialPathConfig::new().with_max_path_count(0),
    )
    .expect("known file");
    assert!(count_limited.stats().path_count_limited);
    assert!(count_limited.paths().is_empty());
}

#[cfg(feature = "serde")]
#[test]
fn graph_and_partial_database_round_trip_together() {
    let mut graph = StackGraph::<String, String, String, ()>::new();
    let file = graph.add_file("main.rs".into());
    let reference = graph
        .add_push_symbol_node(file, "x".into(), true, "use".into())
        .expect("known file");
    let definition = graph
        .add_pop_symbol_node(file, "x".into(), true, "def".into())
        .expect("known file");
    graph
        .add_edge(reference, definition, 0, ())
        .expect("file-local edge");
    let database = StackPartialPathDatabase::compute(&graph, StackPartialPathConfig::new());

    let graph_json = serde_json::to_string(&graph).expect("serializable graph");
    let database_json = serde_json::to_string(&database).expect("serializable database");
    let decoded_graph: StackGraph<String, String, String, ()> =
        serde_json::from_str(&graph_json).expect("valid graph encoding");
    let decoded_database: StackPartialPathDatabase =
        serde_json::from_str(&database_json).expect("valid database encoding");

    assert_eq!(decoded_graph, graph);
    assert_eq!(decoded_database, database);
    let resolution = StackResolution::compute_from_partials(
        &decoded_graph,
        &decoded_database,
        reference,
        StackSearchConfig::new(),
    );
    assert_eq!(resolution.paths()[0].end(), definition);
}
