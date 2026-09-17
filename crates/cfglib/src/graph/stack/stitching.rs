//! Query-time stitching of cached, file-local stack-graph path summaries.

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec::Vec;

use crate::graph::search::SearchOrder;

use super::partial::{StackPartialPathDatabase, StackPartialPathId};
use super::path::{StackPath, StackPathError};
use super::search::{StackResolution, StackResolutionIndex, StackSearchConfig, StackSearchStats};
use super::storage::{StackGraph, StackNodeId};

impl<S: Clone + Eq> StackResolution<S> {
    /// Resolve one reference by stitching precomputed per-file summaries.
    ///
    /// Use [`try_compute_from_partials`](Self::try_compute_from_partials) when
    /// the seed is not already known to be a source reference.
    ///
    /// # Panics
    ///
    /// Panics when `reference` is not a source reference node or `database`
    /// contains a summary whose stored edge is no longer live in `graph`.
    #[must_use]
    pub fn compute_from_partials<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        database: &StackPartialPathDatabase,
        reference: StackNodeId,
        config: StackSearchConfig,
    ) -> Self {
        Self::try_compute_from_partials(graph, database, reference, config)
            .expect("stitched stack resolution requires a source reference node")
    }

    /// Resolve one reference by replaying compatible cached summaries.
    ///
    /// Summary replay checks the same symbol/scope stack transitions as direct
    /// search. A summary or stored edge may occur at most once in a stitched
    /// path. The resulting complete paths use the same precedence shadowing as
    /// [`compute`](Self::compute).
    ///
    /// # Errors
    ///
    /// Returns [`StackPathError::UnknownNode`] when `reference` is outside the
    /// graph, or [`StackPathError::NotReference`] when it is not a source
    /// reference node. Stale summaries return [`StackPathError::UnknownEdge`].
    pub fn try_compute_from_partials<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        database: &StackPartialPathDatabase,
        reference: StackNodeId,
        config: StackSearchConfig,
    ) -> Result<Self, StackPathError> {
        let initial = StackPath::from_reference(graph, reference)?;
        let mut frontier = VecDeque::from([StitchWork {
            path: initial,
            partials: Vec::new(),
        }]);
        let mut complete = Vec::new();
        let mut stats = StackSearchStats::default();

        while let Some(work) = pop_work(&mut frontier, config.order) {
            if config
                .max_path_count
                .is_some_and(|limit| stats.expanded_path_count >= limit)
            {
                stats.path_count_limited = true;
                break;
            }
            stats.expanded_path_count += 1;
            let mut extensions = Vec::new();
            for (partial_id, partial) in database.paths_from(work.path.end()) {
                if work.partials.contains(&partial_id)
                    || partial
                        .edges()
                        .iter()
                        .any(|&edge| work.path.contains_edge(edge))
                {
                    stats.cycle_cut_count += 1;
                    continue;
                }
                if config
                    .max_path_length
                    .is_some_and(|limit| work.path.len() + partial.len() > limit)
                {
                    stats.path_length_limited = true;
                    continue;
                }
                let mut next = work.path.clone();
                if let Err(error) = partial.append_to(graph, &mut next) {
                    if matches!(
                        error,
                        StackPathError::UnknownNode(_) | StackPathError::UnknownEdge(_)
                    ) {
                        return Err(error);
                    }
                    stats.rejected_extension_count += 1;
                    continue;
                }
                if next.is_complete(graph) {
                    stats.complete_path_count += 1;
                    if config
                        .max_complete_path_count
                        .is_none_or(|limit| complete.len() < limit)
                    {
                        complete.push(next.clone());
                    } else {
                        stats.complete_path_count_limited = true;
                    }
                }
                let mut partials = work.partials.clone();
                partials.push(partial_id);
                extensions.push(StitchWork {
                    path: next,
                    partials,
                });
            }
            push_work(&mut frontier, extensions, config.order);
        }

        Ok(Self::from_complete(graph, reference, &complete, stats))
    }
}

impl<S: Clone + Eq> StackResolutionIndex<S> {
    /// Resolve every source reference from cached per-file summaries.
    #[must_use]
    pub fn compute_from_partials<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        database: &StackPartialPathDatabase,
        config: StackSearchConfig,
    ) -> Self {
        let resolutions = graph
            .node_ids()
            .filter(|&node| graph.node(node).kind().is_reference())
            .map(|reference| {
                StackResolution::compute_from_partials(graph, database, reference, config)
            })
            .collect();
        Self::from_resolutions(graph, resolutions)
    }
}

struct StitchWork<S> {
    path: StackPath<S>,
    partials: Vec<StackPartialPathId>,
}

fn pop_work<S>(
    frontier: &mut VecDeque<StitchWork<S>>,
    order: SearchOrder,
) -> Option<StitchWork<S>> {
    match order {
        SearchOrder::BreadthFirst => frontier.pop_front(),
        SearchOrder::DepthFirst => frontier.pop_back(),
    }
}

fn push_work<S>(
    frontier: &mut VecDeque<StitchWork<S>>,
    work: Vec<StitchWork<S>>,
    order: SearchOrder,
) {
    match order {
        SearchOrder::BreadthFirst => frontier.extend(work),
        SearchOrder::DepthFirst => frontier.extend(work.into_iter().rev()),
    }
}
