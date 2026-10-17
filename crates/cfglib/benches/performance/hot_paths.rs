//! The original hot-path registry: the operations the optimization work
//! measured, each beside its semantic oracle.
//!
//! `api_cases` attaches alias families to these cases by name with
//! `benchmark_coverage!`, so this registry runs first. Add a case to the
//! module that owns its responsibility.

mod construction;
mod dataflow;
mod dominance;
mod edits;
mod structure;
mod traversal;

use crate::fixtures::{branchy_cfg, branchy_graph};
use crate::harness::BenchmarkSuite;

/// The node count of the branchy shapes most cases read, and the scale of the
/// chain-shaped fixtures.
const NODE_COUNT: usize = 4_096;

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    let cfg = branchy_cfg(NODE_COUNT);
    let graph = branchy_graph(NODE_COUNT);
    println!(
        "hot-path fixtures: {NODE_COUNT} nodes, {} CFG edges, {} graph edges",
        cfg.edge_count(),
        graph.edge_count()
    );
    construction::register(suite);
    traversal::register(suite, &cfg, &graph);
    dominance::register(suite, &cfg);
    structure::register(suite, &cfg, &graph);
    dataflow::register(suite, &cfg, &graph);
    edits::register(suite);
}
