//! Integration tests: the linear and general stack-graph searches bound and
//! report a search identically.

use cfglib::{PathSearchConfig, SearchOrder, StackGraph, StackNodeKind, StackResolution};

type Graph = StackGraph<&'static str, &'static str, &'static str, ()>;

/// A single chain `use x -> def x -> push x -> def x` whose two definitions
/// are two complete paths of different length, with no branching anywhere.
fn chain_with_two_complete_paths() -> (Graph, cfglib::StackNodeId) {
    let mut graph = Graph::new();
    let file = graph.add_file("main.rs");
    let mut add = |kind, payload| graph.add_node(file, kind, payload).expect("a file node");
    let reference = add(
        StackNodeKind::PushSymbol {
            symbol: "x",
            is_reference: true,
        },
        "use",
    );
    let first = add(
        StackNodeKind::PopSymbol {
            symbol: "x",
            is_definition: true,
        },
        "first def",
    );
    let again = add(
        StackNodeKind::PushSymbol {
            symbol: "x",
            is_reference: false,
        },
        "again",
    );
    let second = add(
        StackNodeKind::PopSymbol {
            symbol: "x",
            is_definition: true,
        },
        "second def",
    );
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
    let unbounded = StackResolution::compute(&graph, reference, PathSearchConfig::default());
    assert_eq!(unbounded.paths().len(), 2, "the fixture has two answers");

    let limits = [None, Some(0), Some(1), Some(2), Some(3), Some(4)];
    let mut truncations = [false; 3];
    for order in [SearchOrder::BreadthFirst, SearchOrder::DepthFirst] {
        for max_path_length in limits {
            for max_path_count in limits {
                for max_complete_path_count in limits {
                    let config = PathSearchConfig {
                        order,
                        max_path_length,
                        max_path_count,
                        max_complete_path_count,
                    };
                    let general = StackResolution::compute(&graph, reference, config);
                    let linear = StackResolution::try_compute_linear(&graph, reference, config)
                        .expect("the chain never branches");
                    assert_eq!(linear, general, "{config:?}");
                    let stats = general.stats().search;
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
