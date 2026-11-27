//! Consumer scenario: the small graph algorithms symtree-xref, symtree-grammar, symtree-member and
//! reify reach for beside the big analyses.
//!
//! `HornClauses` (grammar nullability, "all callers dead"), `min_label_relaxation` (shortest-path
//! shaped analyses over a `Graph` with edge payloads), `common_ancestors` and
//! `nearest_common_ancestor` (class linearization, shared callers), `Reversed`, `reachable` and
//! `reverse_postorder` over views that store only forward adjacency, `call_graph` with
//! `propagate_summaries` over a multi-function program, `ReachIndex` for repeated pairwise
//! reachability, and `DisjointSet`.
//!
//! Every answer is checked against a brute-force oracle written in the test: naive repeated-scan
//! forward chaining, Bellman-Ford style relaxation, explicit ancestor sets with hop distances, a
//! recursive depth-first walk, reachability by enumeration, breadth-first distances, and a plain
//! relabeling partition.

#[path = "../common/rng.rs"]
pub mod rng;

mod ancestors_and_views;
mod call_programs;
mod digraph;
mod disjoint_sets;
mod horn_and_relaxation;
mod reach_index;
