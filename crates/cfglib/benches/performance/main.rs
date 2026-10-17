//! Benchmarks: performance.

mod analysis_oracles;
mod api_cases;
mod fixtures;
mod harness;
mod hot_paths;
mod rtl_placement;
mod structural_oracles;

use harness::{BenchmarkSuite, benchmark_target};

fn main() {
    // Cargo passes `--bench` to custom benchmark binaries. Treat only a
    // positional argument as the optional name filter.
    let filter = std::env::args()
        .skip(1)
        .find(|argument| !argument.starts_with('-'))
        .unwrap_or_default();
    let mut suite = BenchmarkSuite::new(&filter, benchmark_target());
    hot_paths::register(&mut suite);
    api_cases::register(&mut suite);
    suite.finish(api_cases::PUBLIC_API_FUNCTIONS);
}
