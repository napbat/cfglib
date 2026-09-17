//! The [`search`](super::search::search) disciplines over an *open* node
//! space — one discovered lazily, with no dense identities.
//!
//! [`GraphView`](super::view::GraphView) requires the graph to
//! exist: dense ids in `0..node_count`, adjacency on demand. Plenty of real
//! walks have no such graph. An import chase probes a live resolver and mints
//! the next node from the answer; a re-export chase walks a `(file, name)`
//! space whose edges are barrel files it has not read yet; an emission walk
//! interleaves children by source offset as it computes them. Materialising
//! those spaces to walk them is exactly backwards — the walk is what decides
//! which nodes exist.
//!
//! [`open_search`] takes the successor relation as a closure and everything
//! else from [`OpenSearchConfig`], so an open walk gets the same first-match,
//! pruning, depth-bounding, and backtracking discipline as a dense one. The
//! configuration is deliberately *reduced*: there is no `direction`, because
//! the closure defines the edges and a direction field could only lie.
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
//! [`follow`] and [`follow_path`] are the degenerate case that shows up just
//! as often: an out-degree ≤ 1 chase (import alias to import alias, symlink to
//! symlink, type alias to type alias) which needs a hop bound and a cycle
//! guard and nothing else.

mod events;
mod fold;
mod follow;
mod paths;
mod search;

pub use events::{
    OpenBfsConfig, OpenBfsEvent, OpenDfsConfig, OpenDfsEvent, open_breadth_first_events,
    open_depth_first_events,
};
pub use fold::{FoldEnter, MarkScope, OpenFold, OpenFoldConfig, open_fold_post_order};
pub use follow::{follow, follow_path};
pub use paths::{OpenPathsConfig, OpenPathsEvent, open_breadth_first_paths};
pub use search::{OpenSearchConfig, open_search};

/// Whether a node at `depth` may expand under a `max_depth` bound.
fn may_expand(max_depth: Option<usize>, depth: usize) -> bool {
    max_depth.is_none_or(|limit| depth < limit)
}

#[cfg(test)]
mod tests;
