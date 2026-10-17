//! Integration tests: frontier order of the bounded stack-graph path searches.

use cfglib::{
    SearchOrder, StackGraph, StackNodeId, StackPartialPathConfig, StackPartialPathDatabase,
    StackPartialPathSet, StackPath, StackResolution, StackSearchConfig,
};

type Graph = StackGraph<&'static str, &'static str, &'static str, ()>;

/// `use` reaches two definitions of `x`: through a short branch added second
/// and a long branch added first, so the frontier order decides which complete
/// path is discovered first.
struct ShortAndLong {
    graph: Graph,
    reference: StackNodeId,
    short_definition: StackNodeId,
    long_definition: StackNodeId,
}

fn short_and_long() -> ShortAndLong {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use")
        .expect("known file");
    let long_entry = graph
        .add_scope_node(file, false, "long entry")
        .expect("known file");
    let long_exit = graph
        .add_scope_node(file, false, "long exit")
        .expect("known file");
    let short_entry = graph
        .add_scope_node(file, false, "short entry")
        .expect("known file");
    let long_definition = graph
        .add_pop_symbol_node(file, "x", true, "long def")
        .expect("known file");
    let short_definition = graph
        .add_pop_symbol_node(file, "x", true, "short def")
        .expect("known file");
    for (source, target) in [
        (reference, long_entry),
        (reference, short_entry),
        (long_entry, long_exit),
        (long_exit, long_definition),
        (short_entry, short_definition),
    ] {
        graph
            .add_edge(source, target, 0, ())
            .expect("edge stays inside one file");
    }
    ShortAndLong {
        graph,
        reference,
        short_definition,
        long_definition,
    }
}

fn ends(resolution: &StackResolution<&'static str>) -> Vec<StackNodeId> {
    resolution.paths().iter().map(StackPath::end).collect()
}

#[test]
fn direct_search_discovers_complete_paths_by_frontier_order() {
    let fixture = short_and_long();
    let search = |order| {
        StackResolution::compute(
            &fixture.graph,
            fixture.reference,
            StackSearchConfig::new().with_order(order),
        )
    };

    assert_eq!(
        ends(&search(SearchOrder::BreadthFirst)),
        [fixture.short_definition, fixture.long_definition],
        "breadth-first reaches the shorter branch first"
    );
    assert_eq!(
        ends(&search(SearchOrder::DepthFirst)),
        [fixture.long_definition, fixture.short_definition],
        "depth-first follows the first outgoing edge to its end"
    );
}

#[test]
fn stitched_search_finds_the_same_paths_under_either_order() {
    let fixture = short_and_long();
    let database = StackPartialPathDatabase::compute(&fixture.graph, StackPartialPathConfig::new());
    for order in [SearchOrder::BreadthFirst, SearchOrder::DepthFirst] {
        let config = StackSearchConfig::new().with_order(order);
        let direct = StackResolution::compute(&fixture.graph, fixture.reference, config);
        let stitched = StackResolution::compute_from_partials(
            &fixture.graph,
            &database,
            fixture.reference,
            config,
        );
        let mut direct_ends = ends(&direct);
        let mut stitched_ends = ends(&stitched);
        direct_ends.sort();
        stitched_ends.sort();
        assert_eq!(stitched_ends, direct_ends, "{order:?}");
    }
}

#[test]
fn partial_extraction_expands_starting_points_in_declaration_order() {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let mut starts_and_ends = Vec::new();
    for name in ["a", "b", "c"] {
        let reference = graph
            .add_push_symbol_node(file, name, true, "use")
            .expect("known file");
        let definition = graph
            .add_pop_symbol_node(file, name, true, "def")
            .expect("known file");
        graph
            .add_edge(reference, definition, 0, ())
            .expect("edge stays inside one file");
        starts_and_ends.push((reference, definition));
    }

    for order in [SearchOrder::BreadthFirst, SearchOrder::DepthFirst] {
        let set = StackPartialPathSet::compute(
            &graph,
            file,
            StackPartialPathConfig::new().with_order(order),
        )
        .expect("known file");
        let summaries: Vec<_> = set
            .paths()
            .iter()
            .map(|path| (path.start(), path.end()))
            .collect();
        assert_eq!(summaries, starts_and_ends, "{order:?}");
    }
}
