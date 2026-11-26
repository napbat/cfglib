//! Query-time stitching of cached, file-local stack-graph path summaries.

extern crate alloc;

use alloc::vec::Vec;

use crate::graph::search::{PathFrontier, PathSearchConfig};

use super::partial::{StackPartialPathDatabase, StackPartialPathId};
use super::path::{StackPath, StackPathError};
use super::search::{SearchState, StackResolution, StackResolutionIndex};
use super::storage::{StackGraph, StackNodeId};

impl<S: Clone + Eq> StackResolution<S> {
    /// Resolve one reference by replaying compatible cached summaries.
    ///
    /// Summary replay checks the same symbol/scope stack transitions as direct
    /// search. A summary or stored edge may occur at most once in a stitched
    /// path. The resulting complete paths use the same precedence shadowing as
    /// [`compute`](Self::compute). The length bound counts the stored edges
    /// of the stitched path, and a prefix is one stitched path.
    ///
    /// # Errors
    ///
    /// Returns [`StackPathError::UnknownEdge`] when `database` holds a
    /// summary whose stored edge `graph` no longer has: the database is stale
    /// for that graph.
    ///
    /// # Panics
    ///
    /// Panics when `reference` is out of range, retired, or not a source
    /// reference node.
    pub fn try_compute_stitched<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        database: &StackPartialPathDatabase,
        reference: StackNodeId,
        config: PathSearchConfig,
    ) -> Result<Self, StackPathError> {
        let initial = StackPath::from_reference(graph, reference);
        let mut frontier = PathFrontier::new(config.order);
        frontier.push(StitchWork {
            path: initial,
            partials: Vec::new(),
        });
        let mut state = SearchState::new(&config);

        while let Some(work) = frontier.pop() {
            if !state.budget.admit_expansion() {
                break;
            }
            for (partial_id, partial) in database.paths_from(work.path.end()) {
                if work.partials.contains(&partial_id)
                    || partial
                        .edges()
                        .iter()
                        .any(|&edge| work.path.contains_edge(edge))
                {
                    state.cycle_cut_count += 1;
                    continue;
                }
                if !state.budget.admit_length(work.path.len() + partial.len()) {
                    continue;
                }
                let next = match partial.append_to(graph, work.path.clone()) {
                    Ok(next) => next,
                    Err(error @ StackPathError::UnknownEdge(_)) => return Err(error),
                    Err(_) => {
                        state.rejected_extension_count += 1;
                        continue;
                    }
                };
                if next.is_complete(graph) {
                    state.record_complete(&next);
                }
                let mut partials = work.partials.clone();
                partials.push(partial_id);
                frontier.push(StitchWork {
                    path: next,
                    partials,
                });
            }
        }

        Ok(state.finish(graph, reference))
    }
}

impl<S: Clone + Eq> StackResolutionIndex<S> {
    /// Resolve every source reference in node order by stitching cached
    /// per-file summaries, and build the reverse bindings.
    ///
    /// # Errors
    ///
    /// Returns [`StackPathError::UnknownEdge`] when `database` is stale for
    /// `graph`, as [`StackResolution::try_compute_stitched`] does.
    pub fn try_compute_stitched<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        database: &StackPartialPathDatabase,
        config: PathSearchConfig,
    ) -> Result<Self, StackPathError> {
        let resolutions = graph
            .reference_nodes()
            .map(|reference| {
                StackResolution::try_compute_stitched(graph, database, reference, config)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_resolutions(graph, resolutions))
    }
}

struct StitchWork<S> {
    path: StackPath<S>,
    partials: Vec<StackPartialPathId>,
}
