//! Consumer scenario: the walks semctx and symtree-xref run over code-intelligence graphs.
//!
//! Dense walks (`search`, `search_with_scratch`, `search_with_marks`, `breadth_first_events`,
//! `depth_first_events`) run over a `Graph` or any `GraphView`. Open walks (`open_search`,
//! `open_breadth_first_events`, `open_depth_first_events`, `open_breadth_first_paths`,
//! `open_fold_post_order`, `follow`, `follow_path`) run over a node space that has no graph: the
//! successor relation is a closure, and the space is generated lazily over `u64` keys or grid
//! coordinates.
//!
//! The oracles are brute-force walks written from the documented disciplines, recursive where the
//! walk is depth-first, plus closed-form counts: the number of monotone routes through a grid is a
//! binomial coefficient, and the Collatz trajectory of 27 has 111 steps and a peak of 9232.

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
