//! Integration tests: the linear and general stack-graph searches bound and
//! report a search identically.

use cfglib::{SearchOrder, StackGraph, StackResolution, StackSearchConfig};

type Graph = StackGraph<&'static str, &'static str, &'static str, ()>;

/// A single chain `use x -> def x -> push x -> def x` whose two definitions
/// are two complete paths of different length, with no branching anywhere.
fn chain_with_two_complete_paths() -> (Graph, cfglib::StackNodeId) {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let reference = graph
        .add_push_symbol_node(file, "x", true, "use")
        .expect("known file");
    let first = graph
        .add_pop_symbol_node(file, "x", true, "first def")
        .expect("known file");
    let again = graph
        .add_push_symbol_node(file, "x", false, "again")
        .expect("known file");
    let second = graph
        .add_pop_symbol_node(file, "x", true, "second def")
        .expect("known file");
    for (source, target) in [(reference, first), (first, again), (again, second)] {
        graph
            .add_edge(source, target, 0, ())
            .expect("edge stays inside one file");
    }
    (graph, reference)
}

#[test]
fn linear_search_reports_exactly_what_the_general_search_does_under_every_bound() {
    let (graph, reference) = chain_with_two_complete_paths();
    let unbounded = StackResolution::compute(&graph, reference, StackSearchConfig::new());
    assert_eq!(unbounded.paths().len(), 2, "the fixture has two answers");

    let limits = [None, Some(0), Some(1), Some(2), Some(3), Some(4)];
    let mut truncations = [false; 3];
    for order in [SearchOrder::BreadthFirst, SearchOrder::DepthFirst] {
        for max_path_length in limits {
            for max_path_count in limits {
                for max_complete_path_count in limits {
                    let mut config = StackSearchConfig::new().with_order(order);
                    config.max_path_length = max_path_length;
                    config.max_path_count = max_path_count;
                    config.max_complete_path_count = max_complete_path_count;

                    let general = StackResolution::compute(&graph, reference, config);
                    let linear = StackResolution::try_compute_linear(&graph, reference, config)
                        .expect("the chain never branches");
                    assert_eq!(linear, general, "{config:?}");
                    let stats = general.stats();
                    truncations[0] |= stats.path_length_limited;
                    truncations[1] |= stats.path_count_limited;
                    truncations[2] |= stats.complete_path_count_limited;
                }
            }
        }
    }
    assert_eq!(
        truncations, [true; 3],
        "the sweep hits the length, count, and complete-path bounds"
    );
}
