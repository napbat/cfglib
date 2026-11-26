//! The [`search`](super::search::search) disciplines over an *open* node
//! space — one discovered lazily, with no dense identities.
//!
//! [`GraphView`](super::view::GraphView) requires the graph to
//! exist: dense ids in `0..node_count`, adjacency on demand. Plenty of real
//! walks have no such graph. An import chase probes a live resolver and mints
//! the next node from the answer; a re-export chase walks a `(file, name)`
//! space whose edges are barrel files it has not read yet; an emission walk
//! interleaves children by source offset as it computes them. Materializing
//! those spaces to walk them is exactly backwards — the walk is what decides
//! which nodes exist.
//!
//! [`open_search`] takes the successor relation as a closure and everything
//! else from the same [`SearchConfig`](crate::SearchConfig) a dense
//! [`search`](crate::search) takes, so an open walk gets the same first-match,
//! pruning, depth-bounding, and backtracking discipline as a dense one. There
//! is no direction, because the closure defines the edges.
//!
//! [`open_breadth_first_events`] exposes the breadth-first tree as nodes are
//! discovered and reports routes refused by its global marks.
//!
//! [`open_depth_first_events`] walks the same open space for a *fold*. A search
//! can only stop; a fold computes a node's answer from its subtrees' answers,
//! which needs the unwind — so the walk reports
//! [`Discover`](OpenDfsEvent::Discover) and [`Finish`](OpenDfsEvent::Finish)
//! pairs (and the re-entries its mark policy refuses) instead of nodes. C++
//! member lookup is the shape: a class answers with its own declaration if it
//! has one, else with the answer of exactly one yielding base subobject, and
//! two yielding subobjects are an ambiguity — so a diamond's shared base must
//! be folded once *per route*, which is what [`crate::VisitedPolicy::Path`] means
//! here.
//!
//! [`open_breadth_first_paths`] is the once-per-route walk in breadth-first
//! form: no global marks at all, each frontier entry owning its route, so
//! routes arrive shortest-first at simple-path-enumeration cost. It is the
//! breadth-first counterpart of [`open_depth_first_events`] under
//! [`crate::VisitedPolicy::Path`].
//!
//! [`open_fold_post_order`] packages that fold shape completely: an
//! [`OpenFold`] answers per node (or opens an accumulator), absorbs each
//! child's answer with early exit, rewrites on the unwind, and controls the
//! cycle guard itself — which nodes carry a mark key at all, and whether a
//! node's subtree marks persist ([`MarkScope::Shared`]) or stay per-route
//! ([`MarkScope::Isolated`]), so both per-path ambiguity detection and
//! prune-on-revisit evaluators fold without hand-written frame stacks.
//!
//! [`open_fold_post_order_memoized`] is the third choice for a node reached
//! again: a node with an [`OpenFold::memo_key`] is folded once, and every
//! later route absorbs its finished answer. A diamond's base is folded once
//! but still reaches the fold along both routes, so the ambiguity check of a
//! member lookup costs one fold per class instead of one per route, as long
//! as the answer of a node does not depend on the route to it.
//!
//! [`open_fold_fixpoint`] is the driver for a cyclic space whose answers
//! must travel around a cycle, such as the names a module exports through
//! `export *` cycles. The other drivers refuse a node that is still open on
//! the route, so on a cycle they return an answer that depends on where the
//! walk entered it and can miss what comes back around. The fixpoint driver
//! keeps one answer per mark, finds the cycles of marked nodes as it walks,
//! and re-folds each cycle until its answers are stable. It needs monotone
//! hooks (or a round limit) and a mark on every node of a cycle.
//!
//! [`follow`] and [`follow_path`] are the degenerate case that shows up just
//! as often: an out-degree ≤ 1 chase (import alias to import alias, symlink to
//! symlink, type alias to type alias) which needs a hop bound and a cycle
//! guard and nothing else.

mod events;
mod fold;
mod follow;
mod marks;
mod paths;
mod search;

pub use events::{OpenBfsEvent, OpenDfsEvent, open_breadth_first_events, open_depth_first_events};
pub use fold::{
    FoldEnter, MarkScope, OpenFixpointConfig, OpenFixpointError, OpenFold, OpenFoldConfig,
    open_fold_fixpoint, open_fold_post_order, open_fold_post_order_memoized,
};
pub use follow::{follow, follow_path};
pub use paths::{OpenPathsEvent, open_breadth_first_paths};
pub use search::open_search;

#[cfg(test)]
mod tests;
