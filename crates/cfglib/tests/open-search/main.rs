//! Consumer scenario: the walks semctx and symtree-xref run over code-intelligence graphs.
//!
//! Dense walks (`search`, `search_in`, `depth_first_events`) run over a `Graph` or any
//! `GraphView`. Open walks (`open_search`,
//! `open_breadth_first_events`, `open_depth_first_events`, `open_breadth_first_paths`,
//! `open_fold`, `follow`, `follow_path`) run over a node space that has no graph: the
//! successor relation is a closure, and the space is generated lazily over `u64` keys or grid
//! coordinates.
//!
//! The oracles are brute-force walks written from the documented disciplines, recursive where the
//! walk is depth-first, plus closed-form counts: the number of monotone routes through a grid is a
//! binomial coefficient, and the Collatz trajectory of 27 has 111 steps and a peak of 9232.

#[path = "../common/rng.rs"]
pub mod rng;

mod chases_and_views;
mod dense_events;
mod dense_scratch;
mod dense_search;
mod open_events;
mod open_fold;
mod open_routes;
mod open_searches;
mod reference;
mod spaces;
