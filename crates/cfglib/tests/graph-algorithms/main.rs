//! Consumer scenario: the small graph algorithms symtree-xref, symtree-grammar, symtree-member and
//! reify reach for beside the big analyses.
//!
//! `HornClauses` (grammar nullability, "all callers dead"), `min_label_relaxation` (shortest-path
//! shaped analyses over a `Graph` with edge payloads), `common_ancestors` and
//! `nearest_common_ancestor` (class linearization, shared callers), `Reversed`, `reachable` and
//! `reverse_postorder` over views that store only forward adjacency, `call_graph` with
//! `propagate_summaries` over a multi-function program, and `DisjointSet`.
//!
//! Every answer is checked against a brute-force oracle written in the test: naive repeated-scan
//! forward chaining, Bellman-Ford style relaxation, explicit ancestor sets with hop distances, a
//! recursive depth-first walk, reachability by enumeration, and a plain relabeling partition.

mod ancestors_and_views;
mod call_programs;
mod digraph;
mod disjoint_sets;
mod horn_and_relaxation;

/// A linear congruential generator, so every scenario is reproducible without a dependency.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 33
    }

    fn below(&mut self, bound: usize) -> usize {
        let bound = u64::try_from(bound).expect("the bound fits in u64");
        usize::try_from(self.next_u64() % bound).expect("a value below the bound fits in usize")
    }
}
