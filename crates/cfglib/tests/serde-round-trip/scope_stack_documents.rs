//! Scope graphs, stack graphs, and per-file partial-path databases: what an
//! incremental indexer persists so a later process resolves names without
//! rebuilding the graph.

use core::cmp::Ordering;

use cfglib::{
    EdgeRef, Scope, ScopeDatum, ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeGraphPathLabel,
    ScopeGraphQuery, ScopeId, ScopePath, ScopeQuery, ScopeReference, ScopeResolutionConfig,
    ScopeResolutionIndex, StackEdge, StackFileId, StackGraph, StackNode, StackNodeId,
    StackNodeKind, StackPartialPath, StackPartialPathConfig, StackPartialPathDatabase,
    StackPartialPathId, StackPartialPathSet, StackPartialPathStats, StackResolution,
    StackResolutionIndex, StackSearchConfig,
};

use crate::json::round_trip;

type Names = ScopeGraph<String, String, String, String, String>;

/// Name lookup through `parent` and `import` edges, matching data of the
/// `value` relation by name, with declarations shadowing imports and imports
/// shadowing enclosing scopes.
struct NameLookup;

fn rank(label: &ScopeGraphPathLabel<'_, String>) -> u8 {
    match label {
        ScopeGraphPathLabel::Declaration => 0,
        ScopeGraphPathLabel::Edge(name) if name.as_str() == "import" => 1,
        ScopeGraphPathLabel::Edge(_) => 2,
    }
}

impl ScopeGraphQuery<String, String, String, String, String> for NameLookup {
    type State = ();

    fn initial_state(&self, _graph: &Names, _input: &ScopeQuery<'_, String>) -> Self::State {}

    fn step(
        &self,
        _graph: &Names,
        _input: &ScopeQuery<'_, String>,
        _path: &ScopePath,
        _state: &Self::State,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, String>,
    ) -> Option<Self::State> {
        matches!(edge.data().as_str(), "parent" | "import").then_some(())
    }

    fn accepts(
        &self,
        graph: &Names,
        input: &ScopeQuery<'_, String>,
        _path: &ScopePath,
        _state: &Self::State,
        datum: ScopeDatumId,
    ) -> bool {
        let candidate = graph.datum(datum);
        candidate.relation() == "value" && candidate.data() == input.data()
    }

    fn compare_labels(
        &self,
        _graph: &Names,
        _input: &ScopeQuery<'_, String>,
        left: ScopeGraphPathLabel<'_, String>,
        right: ScopeGraphPathLabel<'_, String>,
    ) -> Option<Ordering> {
        Some(rank(&left).cmp(&rank(&right)))
    }
}

/// A module with an import scope, a body that sees both, one reference that
/// resolves locally, one through the parent, one through the import, one to
/// nothing, and an edge that was added and removed.
fn names() -> Names {
    let mut graph = Names::new();
    let module = graph.add_scope("module".to_owned());
    let imports = graph.add_scope("imports".to_owned());
    let body = graph.add_scope("body".to_owned());
    graph.add_edge(body, module, "parent".to_owned());
    graph.add_edge(body, imports, "import".to_owned());
    let withdrawn = graph.add_edge(imports, module, "parent".to_owned());
    graph.remove_edge(withdrawn);

    graph.add_datum(module, "value".to_owned(), "outer".to_owned());
    graph.add_datum(module, "type".to_owned(), "Outer".to_owned());
    graph.add_datum(imports, "value".to_owned(), "package".to_owned());
    graph.add_datum(body, "value".to_owned(), "item".to_owned());
    for name in ["item", "outer", "package", "Outer", "missing"] {
        graph.add_reference(body, name.to_owned());
    }
    graph
}

/// What every reference of `graph` resolves to, and which references point at
/// each datum: the analysis a consumer would run on a loaded graph.
fn resolutions(graph: &Names) -> (Vec<Vec<ScopeDatumId>>, Vec<Vec<cfglib::ScopeReferenceId>>) {
    let index = ScopeResolutionIndex::compute(graph, &NameLookup, ScopeResolutionConfig::new());
    let targets = graph
        .reference_ids()
        .map(|reference| {
            index
                .resolution(reference)
                .candidates()
                .iter()
                .map(cfglib::ScopeResolutionCandidate::datum)
                .collect()
        })
        .collect();
    let users = graph
        .datum_ids()
        .map(|datum| index.references_to(datum).to_vec())
        .collect();
    (targets, users)
}

#[test]
fn a_scope_graph_decodes_equal_and_resolves_every_reference_like_the_original() {
    let original = names();
    let decoded: Names = round_trip(&original);
    assert_eq!(decoded, original);

    let (targets, users) = resolutions(&original);
    assert_eq!(resolutions(&decoded), (targets.clone(), users));
    let resolved: Vec<_> = targets.iter().map(Vec::len).collect();
    assert_eq!(
        resolved,
        [1, 1, 1, 0, 0],
        "the fixture resolves what it should: {targets:?}"
    );
}

#[test]
fn a_scope_graph_keeps_its_removed_edge_slot_and_adjacency() {
    let original = names();
    let decoded: Names = round_trip(&original);

    assert_eq!(decoded.scope_count(), original.scope_count());
    assert_eq!(decoded.edge_count(), original.edge_count());
    assert_eq!(decoded.edge_bound(), original.edge_bound());
    assert!(
        decoded.edge_bound() > decoded.edge_count(),
        "the removed edge keeps its slot"
    );
    for scope in original.scope_ids() {
        assert_eq!(
            decoded.scope(scope).payload(),
            original.scope(scope).payload()
        );
        assert_eq!(
            decoded.outgoing(scope).collect::<Vec<_>>(),
            original.outgoing(scope).collect::<Vec<_>>()
        );
        assert_eq!(
            decoded.incoming(scope).collect::<Vec<_>>(),
            original.incoming(scope).collect::<Vec<_>>()
        );
        assert_eq!(decoded.scope_data(scope), original.scope_data(scope));
        assert_eq!(
            decoded.scope_references(scope),
            original.scope_references(scope)
        );
    }
}

#[test]
fn a_decoded_scope_graph_keeps_growing_and_resolving_like_the_original() {
    let mut original = names();
    let mut decoded: Names = round_trip(&original);
    for graph in [&mut original, &mut decoded] {
        let body = graph.scope_ids().last().expect("the fixture has scopes");
        graph.add_datum(body, "value".to_owned(), "missing".to_owned());
        graph.add_reference(body, "outer".to_owned());
    }
    assert_eq!(decoded, original);
    assert_eq!(resolutions(&decoded), resolutions(&original));
    let (targets, _) = resolutions(&decoded);
    assert_eq!(
        targets[3],
        [],
        "Outer is a type, the lookup asks for values"
    );
    assert_eq!(targets[4].len(), 1, "missing now resolves to the new datum");
}

type Files = StackGraph<String, String, String, String>;

struct Indexed {
    graph: Files,
    reference: StackNodeId,
    preferred: StackNodeId,
    fallback: StackNodeId,
    provider: StackFileId,
}

/// Two files joined through the root: a reference in one, two competing
/// exports in the other (one preferred by precedence), an exported scope, and
/// an edge that was added and removed.
fn indexed() -> Indexed {
    let mut graph = Files::new();
    let consumer = graph.add_file("consumer.rs".to_owned());
    let provider = graph.add_file("provider.rs".to_owned());
    let reference = graph
        .add_push_symbol_node(
            consumer,
            "package".to_owned(),
            true,
            "import package".to_owned(),
        )
        .expect("a known file");
    let preferred = graph
        .add_pop_symbol_node(
            provider,
            "package".to_owned(),
            true,
            "public package".to_owned(),
        )
        .expect("a known file");
    let fallback = graph
        .add_pop_symbol_node(
            provider,
            "package".to_owned(),
            true,
            "fallback package".to_owned(),
        )
        .expect("a known file");
    let scope = graph
        .add_scope_node(provider, true, "exports".to_owned())
        .expect("a known file");
    let root = graph.root_node();

    graph
        .add_edge(reference, root, 0, "leave consumer".to_owned())
        .expect("root joins files");
    graph
        .add_edge(root, scope, 0, "enter provider".to_owned())
        .expect("root joins files");
    graph
        .add_edge(scope, preferred, 2, "preferred export".to_owned())
        .expect("same file");
    graph
        .add_edge(scope, fallback, 1, "fallback export".to_owned())
        .expect("same file");
    let withdrawn = graph
        .add_edge(root, fallback, 9, "withdrawn shortcut".to_owned())
        .expect("root joins files");
    assert!(graph.remove_edge(withdrawn));
    Indexed {
        graph,
        reference,
        preferred,
        fallback,
        provider,
    }
}

#[test]
fn a_stack_graph_decodes_equal_and_resolves_like_the_original() {
    let indexed = indexed();
    let decoded: Files = round_trip(&indexed.graph);
    assert_eq!(decoded, indexed.graph);

    assert_eq!(decoded.node_count(), indexed.graph.node_count());
    assert_eq!(decoded.edge_count(), indexed.graph.edge_count());
    assert!(
        decoded.edge_bound() > decoded.edge_count(),
        "the removed edge keeps its slot"
    );
    assert_eq!(decoded.root_node(), indexed.graph.root_node());
    assert_eq!(
        decoded.jump_to_scope_node(),
        indexed.graph.jump_to_scope_node()
    );

    let from_original =
        StackResolution::compute(&indexed.graph, indexed.reference, StackSearchConfig::new());
    let from_decoded =
        StackResolution::compute(&decoded, indexed.reference, StackSearchConfig::new());
    assert_eq!(from_decoded.paths(), from_original.paths());
    assert_eq!(
        from_decoded.paths()[0].end(),
        indexed.preferred,
        "precedence picks the preferred export"
    );
    assert_eq!(from_decoded.target_count(), 1);
}

#[test]
fn partial_paths_persisted_beside_the_graph_resolve_without_extraction() {
    let indexed = indexed();
    let database = StackPartialPathDatabase::compute(&indexed.graph, StackPartialPathConfig::new());
    let decoded_graph: Files = round_trip(&indexed.graph);
    let decoded_database: StackPartialPathDatabase = round_trip(&database);
    assert_eq!(decoded_database, database);

    let direct =
        StackResolution::compute(&indexed.graph, indexed.reference, StackSearchConfig::new());
    let stitched = StackResolution::compute_from_partials(
        &decoded_graph,
        &decoded_database,
        indexed.reference,
        StackSearchConfig::new(),
    );
    assert_eq!(
        stitched.paths(),
        direct.paths(),
        "stitching the loaded summaries is direct resolution"
    );

    let reverse = StackResolutionIndex::compute_from_partials(
        &decoded_graph,
        &decoded_database,
        StackSearchConfig::new(),
    );
    assert_eq!(
        reverse.references_to(indexed.preferred),
        [indexed.reference]
    );
    assert_eq!(reverse.references_to(indexed.fallback), []);
}

#[test]
fn a_decoded_database_replaces_one_file_exactly_like_the_original() {
    let indexed = indexed();
    let mut original_graph = indexed.graph.clone();
    let mut decoded_graph: Files = round_trip(&indexed.graph);
    let config = StackPartialPathConfig::new();
    let mut original_database = StackPartialPathDatabase::compute(&original_graph, config);
    let mut decoded_database: StackPartialPathDatabase = round_trip(&original_database);

    // The provider is re-indexed: its old nodes are cleared and one new
    // export, now the only one, replaces them.
    let mut replacements = Vec::new();
    for (graph, database) in [
        (&mut original_graph, &mut original_database),
        (&mut decoded_graph, &mut decoded_database),
    ] {
        graph.clear_file(indexed.provider).expect("a known file");
        let export = graph
            .add_pop_symbol_node(
                indexed.provider,
                "package".to_owned(),
                true,
                "rewritten".to_owned(),
            )
            .expect("a known file");
        graph
            .add_edge(graph.root_node(), export, 0, "export".to_owned())
            .expect("root joins files");
        database
            .replace_file(graph, indexed.provider, config)
            .expect("a known file");
        replacements.push(export);
    }
    assert_eq!(
        replacements[0], replacements[1],
        "the new node gets the same identity"
    );
    assert_eq!(decoded_database, original_database);
    assert_eq!(decoded_graph, original_graph);

    let resolution = StackResolution::compute_from_partials(
        &decoded_graph,
        &decoded_database,
        indexed.reference,
        StackSearchConfig::new(),
    );
    let direct =
        StackResolution::compute(&original_graph, indexed.reference, StackSearchConfig::new());
    assert_eq!(resolution.paths(), direct.paths());
    assert_eq!(resolution.paths()[0].end(), replacements[0]);
}

#[test]
fn the_pieces_of_a_scope_graph_round_trip_on_their_own() {
    let graph = names();
    for scope in graph.scope_ids() {
        let decoded: Scope<String> = round_trip(graph.scope(scope));
        assert_eq!(&decoded, graph.scope(scope));
        assert_eq!(round_trip(&scope), scope);
    }
    for datum in graph.datum_ids() {
        let decoded: ScopeDatum<String, String> = round_trip(graph.datum(datum));
        assert_eq!(&decoded, graph.datum(datum));
        assert_eq!(round_trip(&datum), datum);
    }
    for reference in graph.reference_ids() {
        let decoded: ScopeReference<String> = round_trip(graph.reference(reference));
        assert_eq!(&decoded, graph.reference(reference));
        assert_eq!(round_trip(&reference), reference);
    }
    for edge in graph.edge_ids() {
        assert_eq!(round_trip(&edge), edge);
    }
}

#[test]
fn the_pieces_of_a_stack_graph_round_trip_on_their_own() {
    let indexed = indexed();
    let graph = &indexed.graph;
    for node in graph.node_ids() {
        let decoded: StackNode<String, String> = round_trip(graph.node(node));
        assert_eq!(&decoded, graph.node(node));
        let kind: StackNodeKind<String> = round_trip(graph.node(node).kind());
        assert_eq!(&kind, graph.node(node).kind());
        assert_eq!(round_trip(&node), node);
    }
    for edge in graph.edge_ids() {
        let decoded: StackEdge<String> = round_trip(graph.edge(edge).data());
        assert_eq!(&decoded, graph.edge(edge).data());
        assert_eq!(decoded.precedence(), graph.edge(edge).data().precedence());
    }
    for file in graph.file_ids() {
        assert_eq!(round_trip(&file), file);
    }
}

#[test]
fn extracted_partial_paths_and_their_statistics_round_trip_on_their_own() {
    let indexed = indexed();
    let config = StackPartialPathConfig::new();
    let set = StackPartialPathSet::compute(&indexed.graph, indexed.provider, config)
        .expect("a known file");
    assert!(!set.paths().is_empty(), "the provider has partial paths");
    let decoded: StackPartialPathSet = round_trip(&set);
    assert_eq!(decoded, set);
    let stats: StackPartialPathStats = round_trip(&set.stats());
    assert_eq!(stats, set.stats());

    let database = StackPartialPathDatabase::compute(&indexed.graph, config);
    for (id, path) in database.file_paths(indexed.provider) {
        let decoded: StackPartialPath = round_trip(path);
        assert_eq!(&decoded, path);
        assert_eq!(round_trip::<StackPartialPathId>(&id), id);
    }
}
