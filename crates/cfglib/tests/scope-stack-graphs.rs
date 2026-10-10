//! Integration tests: scope stack graphs.

use core::cmp::Ordering;

use cfglib::EdgeRef;
use cfglib::graph::scope::{
    ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeGraphPathLabel, ScopeGraphQuery, ScopeId,
    ScopePath, ScopeQuery, ScopeResolution, ScopeResolutionConfig, ScopeResolutionIndex,
};
use cfglib::graph::stack::{
    StackGraph, StackPartialPathConfig, StackPartialPathDatabase, StackResolution,
    StackResolutionIndex, StackSearchConfig,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ScopeEdge {
    Parent,
    Import,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Namespace {
    Value,
    Type,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ScopeMetadata {
    ordered: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Definition {
    name: String,
    end: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NameMatch {
    Exact,
    Prefix,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Lookup {
    name: String,
    namespace: Namespace,
    position: u32,
    name_match: NameMatch,
}

type SymbolGraph = ScopeGraph<ScopeMetadata, ScopeEdge, Namespace, Definition, Lookup>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LookupState {
    Lexical,
    Imported,
}

struct SymtreeStyleLookup;

impl ScopeGraphQuery<ScopeMetadata, ScopeEdge, Namespace, Definition, Lookup>
    for SymtreeStyleLookup
{
    type State = LookupState;

    fn initial_state(&self, _graph: &SymbolGraph, _input: &ScopeQuery<'_, Lookup>) -> Self::State {
        LookupState::Lexical
    }

    fn step(
        &self,
        _graph: &SymbolGraph,
        _input: &ScopeQuery<'_, Lookup>,
        _path: &ScopePath,
        state: &Self::State,
        edge: EdgeRef<'_, ScopeId, ScopeEdgeId, ScopeEdge>,
    ) -> Option<Self::State> {
        match (state, edge.data()) {
            (LookupState::Lexical, ScopeEdge::Parent) => Some(LookupState::Lexical),
            (LookupState::Lexical, ScopeEdge::Import) => Some(LookupState::Imported),
            (LookupState::Imported, ScopeEdge::Parent | ScopeEdge::Import) => None,
        }
    }

    fn accepts(
        &self,
        graph: &SymbolGraph,
        input: &ScopeQuery<'_, Lookup>,
        path: &ScopePath,
        _state: &Self::State,
        datum: ScopeDatumId,
    ) -> bool {
        let request = input.data();
        let candidate = graph.datum(datum);
        if candidate.relation() != &request.namespace || !name_matches(request, candidate.data()) {
            return false;
        }
        if !graph.scope(path.end()).payload().ordered {
            return true;
        }

        graph
            .scope_data(path.end())
            .iter()
            .copied()
            .filter(|&other| {
                let other = graph.datum(other);
                other.relation() == &request.namespace
                    && other.data().name == candidate.data().name
                    && other.data().end <= request.position
            })
            .max_by_key(|&other| (graph.datum(other).data().end, other))
            == Some(datum)
    }

    fn compare_labels(
        &self,
        _graph: &SymbolGraph,
        _input: &ScopeQuery<'_, Lookup>,
        left: ScopeGraphPathLabel<'_, ScopeEdge>,
        right: ScopeGraphPathLabel<'_, ScopeEdge>,
    ) -> Option<Ordering> {
        fn rank(label: ScopeGraphPathLabel<'_, ScopeEdge>) -> u8 {
            match label {
                ScopeGraphPathLabel::Declaration => 0,
                ScopeGraphPathLabel::Edge(ScopeEdge::Import) => 1,
                ScopeGraphPathLabel::Edge(ScopeEdge::Parent) => 2,
            }
        }
        Some(rank(left).cmp(&rank(right)))
    }

    fn data_leq(
        &self,
        graph: &SymbolGraph,
        _input: &ScopeQuery<'_, Lookup>,
        left: ScopeDatumId,
        right: ScopeDatumId,
    ) -> bool {
        graph.datum(left).data().name == graph.datum(right).data().name
    }
}

fn name_matches(request: &Lookup, definition: &Definition) -> bool {
    match request.name_match {
        NameMatch::Exact => definition.name == request.name,
        NameMatch::Prefix => definition.name.starts_with(&request.name),
    }
}

fn exact(name: &str, namespace: Namespace, position: u32) -> Lookup {
    Lookup {
        name: name.into(),
        namespace,
        position,
        name_match: NameMatch::Exact,
    }
}

#[test]
fn symtree_shaped_scope_queries_cover_definition_completion_and_rename() {
    let mut graph = SymbolGraph::new();
    let module = graph.add_scope(ScopeMetadata { ordered: false });
    let imported = graph.add_scope(ScopeMetadata { ordered: false });
    let body = graph.add_scope(ScopeMetadata { ordered: true });
    graph.add_edge(body, module, ScopeEdge::Parent);
    graph.add_edge(body, imported, ScopeEdge::Import);

    let outer = graph.add_datum(
        module,
        Namespace::Value,
        Definition {
            name: "outer".into(),
            end: 1,
        },
    );
    graph.add_datum(
        module,
        Namespace::Type,
        Definition {
            name: "Outer".into(),
            end: 2,
        },
    );
    let package = graph.add_datum(
        imported,
        Namespace::Value,
        Definition {
            name: "package".into(),
            end: 1,
        },
    );
    let early_item = graph.add_datum(
        body,
        Namespace::Value,
        Definition {
            name: "item".into(),
            end: 5,
        },
    );
    let late_item = graph.add_datum(
        body,
        Namespace::Value,
        Definition {
            name: "item".into(),
            end: 15,
        },
    );
    let early_use = graph.add_reference(body, exact("item", Namespace::Value, 10));
    let late_use = graph.add_reference(body, exact("item", Namespace::Value, 20));
    let import_use = graph.add_reference(body, exact("package", Namespace::Value, 20));

    let index =
        ScopeResolutionIndex::compute(&graph, &SymtreeStyleLookup, ScopeResolutionConfig::new());
    assert_eq!(
        index.resolution(early_use).candidates()[0].datum(),
        early_item
    );
    assert_eq!(
        index.resolution(late_use).candidates()[0].datum(),
        late_item
    );
    assert_eq!(
        index.resolution(import_use).candidates()[0].datum(),
        package
    );
    assert_eq!(index.references_to(early_item), &[early_use]);
    assert_eq!(index.references_to(late_item), &[late_use]);

    let reference_count = graph.reference_count();
    let completion = Lookup {
        name: String::new(),
        namespace: Namespace::Value,
        position: 10,
        name_match: NameMatch::Prefix,
    };
    let completion = ScopeResolution::compute_from(
        &graph,
        body,
        &completion,
        &SymtreeStyleLookup,
        ScopeResolutionConfig::new(),
    );
    let targets: Vec<_> = completion
        .candidates()
        .iter()
        .map(cfglib::ScopeResolutionCandidate::datum)
        .collect();
    assert!(targets.contains(&early_item));
    assert!(targets.contains(&outer));
    assert!(targets.contains(&package));
    assert!(!targets.contains(&late_item));
    assert_eq!(graph.reference_count(), reference_count);
    assert_eq!(completion.reference(), None);
}

#[test]
fn non_copy_stack_symbols_resolve_directly_and_from_file_summaries() {
    let mut graph = StackGraph::<String, String, String, &'static str>::new();
    let consumer = graph.add_file("consumer.rs".into());
    let provider = graph.add_file("provider.rs".into());
    let reference = graph
        .add_push_symbol_node(consumer, "package".into(), true, "import package".into())
        .expect("known file");
    let preferred = graph
        .add_pop_symbol_node(provider, "package".into(), true, "public package".into())
        .expect("known file");
    let fallback = graph
        .add_pop_symbol_node(provider, "package".into(), true, "fallback package".into())
        .expect("known file");
    let root = graph.root_node();
    graph
        .add_edge(reference, root, 0, "leave consumer")
        .expect("root joins files");
    graph
        .add_edge(root, preferred, 2, "preferred export")
        .expect("root joins files");
    graph
        .add_edge(root, fallback, 1, "fallback export")
        .expect("root joins files");

    let direct = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    let partials = StackPartialPathDatabase::compute(&graph, StackPartialPathConfig::new());
    let stitched = StackResolution::compute_from_partials(
        &graph,
        &partials,
        reference,
        StackSearchConfig::new(),
    );
    assert_eq!(direct.paths(), stitched.paths());
    assert_eq!(stitched.target_count(), 1);
    assert_eq!(stitched.paths()[0].end(), preferred);

    let reverse =
        StackResolutionIndex::compute_from_partials(&graph, &partials, StackSearchConfig::new());
    assert_eq!(reverse.references_to(preferred), &[reference]);
    assert_eq!(reverse.references_to(fallback), []);
}
