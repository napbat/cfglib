//! Direct and linear stack-graph path finding, and the binding index.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::graph::edge_view::EdgeView;
use crate::graph::search::{
    PathBudget, PathFrontier, PathSearchConfig, PathSearchStats, retain_unshadowed,
    sources_by_target,
};

use super::path::StackPath;
use super::storage::{StackGraph, StackNodeId};

/// Work, rejection, and truncation information for one stack-graph query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StackSearchStats {
    /// The prefixes expanded, the complete paths found, and the bounds that
    /// refused work.
    pub search: PathSearchStats,
    /// Extensions rejected by symbol or scope stack semantics.
    pub rejected_extension_count: usize,
    /// Extensions skipped because the stored edge already occurs on the path.
    pub cycle_cut_count: usize,
}

/// A query that the linear walk resolves admitted several viable outgoing
/// edges at one prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackLinearResolutionError {
    /// More than one outgoing edge extended the same path prefix successfully.
    MultipleEdges {
        /// Node whose viable outgoing edges caused the branch.
        node: StackNodeId,
    },
}

impl core::fmt::Display for StackLinearResolutionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MultipleEdges { node } => {
                write!(
                    formatter,
                    "stack path at {node} admits multiple outgoing edges"
                )
            }
        }
    }
}

impl core::error::Error for StackLinearResolutionError {}

/// Visible complete paths for one stack-graph reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackResolution<S> {
    reference: StackNodeId,
    paths: SmallVec<[StackPath<S>; 1]>,
    stats: StackSearchStats,
}

impl<S: Clone + Eq> StackResolution<S> {
    /// Resolve one reference by direct search.
    ///
    /// Search follows only well-formed stack transitions. A stored edge may
    /// occur at most once on a path, matching the standard direct stack-graph
    /// cycle guard. Complete paths shadowed by greater edge precedence are
    /// removed after search.
    ///
    /// # Panics
    ///
    /// Panics when `reference` is out of range, retired, or not a source
    /// reference node.
    #[must_use]
    pub fn compute<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        reference: StackNodeId,
        config: PathSearchConfig,
    ) -> Self {
        let initial = StackPath::from_reference(graph, reference);
        let mut frontier = PathFrontier::new(config.order);
        frontier.push(initial);
        let mut state = SearchState::new(&config);

        while let Some(path) = frontier.pop() {
            if !state.budget.admit_expansion() {
                break;
            }
            for edge in graph.outgoing(path.end()) {
                if path.contains_edge(edge) {
                    state.cycle_cut_count += 1;
                    continue;
                }
                if !state.budget.admit_length(path.edge_count() + 1) {
                    continue;
                }
                let mut next = path.clone();
                if next.append(graph, edge).is_err() {
                    state.rejected_extension_count += 1;
                    continue;
                }
                if next.is_complete(graph) {
                    state.record_complete(&next);
                }
                frontier.push(next);
            }
        }

        state.finish(graph, reference)
    }

    /// Resolve a reference whose well-formed path has at most one viable
    /// outgoing edge at every prefix.
    ///
    /// This applies the same stack transitions, cycle guard, bounds, complete
    /// path retention, and precedence filtering as [`compute`](Self::compute),
    /// while updating a single path in place. The walk follows one path, so
    /// it does not read `config.order`.
    ///
    /// # Errors
    ///
    /// Returns [`StackLinearResolutionError::MultipleEdges`] when a prefix
    /// branches into several well-formed paths.
    ///
    /// # Panics
    ///
    /// Panics when `reference` is out of range, retired, or not a source
    /// reference node.
    pub fn try_compute_linear<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        reference: StackNodeId,
        config: PathSearchConfig,
    ) -> Result<Self, StackLinearResolutionError> {
        let mut path = StackPath::from_reference(graph, reference);
        let mut state = SearchState::new(&config);

        while state.budget.admit_expansion() {
            let mut next = None;
            let end = path.end();
            let mut edges = graph.outgoing(end).peekable();
            // The prefix is copied only for an extension that a later edge
            // will extend again; the last one consumes it.
            let mut held = Some(path);
            while let Some(edge) = edges.next() {
                let Some(current) = &held else {
                    unreachable!("the prefix is held until the last outgoing edge consumes it")
                };
                if current.contains_edge(edge) {
                    state.cycle_cut_count += 1;
                    continue;
                }
                if !state.budget.admit_length(current.edge_count() + 1) {
                    continue;
                }
                let mut candidate = if edges.peek().is_some() {
                    current.clone()
                } else {
                    let Some(prefix) = held.take() else {
                        unreachable!("the prefix is held until the last outgoing edge consumes it")
                    };
                    prefix
                };
                if candidate.append(graph, edge).is_err() {
                    state.rejected_extension_count += 1;
                    continue;
                }
                if next.is_some() {
                    return Err(StackLinearResolutionError::MultipleEdges { node: end });
                }
                if candidate.is_complete(graph) {
                    state.record_complete(&candidate);
                }
                next = Some(candidate);
            }
            let Some(candidate) = next else {
                break;
            };
            path = candidate;
        }

        Ok(state.finish(graph, reference))
    }

    /// The source reference node.
    #[must_use]
    pub const fn reference(&self) -> StackNodeId {
        self.reference
    }

    /// Visible complete paths, in stable discovery order.
    #[must_use]
    pub fn paths(&self) -> &[StackPath<S>] {
        &self.paths
    }

    /// Search work and truncation information.
    #[must_use]
    pub const fn stats(&self) -> StackSearchStats {
        self.stats
    }

    /// Whether at least one definition is visible.
    #[must_use]
    pub fn is_resolved(&self) -> bool {
        !self.paths.is_empty()
    }

    /// Number of distinct definition targets.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.paths
            .iter()
            .map(StackPath::end)
            .collect::<BTreeSet<_>>()
            .len()
    }

    /// Whether more than one distinct definition remains visible.
    #[must_use]
    pub fn is_ambiguous(&self) -> bool {
        self.target_count() > 1
    }
}

/// Resolutions for every source reference, plus the reverse bindings from
/// each definition to the references that resolve to it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackResolutionIndex<S> {
    resolutions: Vec<StackResolution<S>>,
    resolution_by_node: Vec<Option<usize>>,
    references_by_definition: Vec<Vec<StackNodeId>>,
}

impl<S: Clone + Eq> StackResolutionIndex<S> {
    /// Resolve every source reference in node order by direct search and
    /// build the reverse bindings.
    #[must_use]
    pub fn compute<F, N, E>(graph: &StackGraph<F, S, N, E>, config: PathSearchConfig) -> Self {
        let resolutions = graph
            .reference_nodes()
            .map(|reference| StackResolution::compute(graph, reference, config))
            .collect();
        Self::from_resolutions(graph, resolutions)
    }

    /// Resolve every source reference under the linear-path contract.
    ///
    /// # Errors
    ///
    /// Returns the first [`StackLinearResolutionError`] in source-node order.
    pub fn try_compute_linear<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        config: PathSearchConfig,
    ) -> Result<Self, StackLinearResolutionError> {
        let resolutions = graph
            .reference_nodes()
            .map(|reference| StackResolution::try_compute_linear(graph, reference, config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_resolutions(graph, resolutions))
    }

    /// The index over `resolutions`, one per source reference of `graph` in
    /// node order.
    pub(super) fn from_resolutions<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        resolutions: Vec<StackResolution<S>>,
    ) -> Self {
        let bound = graph.graph().node_bound();
        let mut resolution_by_node = vec![None; bound];
        for (index, resolution) in resolutions.iter().enumerate() {
            resolution_by_node[resolution.reference().index()] = Some(index);
        }
        let references_by_definition = sources_by_target(
            bound,
            resolutions.iter().map(|resolution| {
                let paths = resolution.paths().iter();
                (resolution.reference(), paths.map(|path| path.end().index()))
            }),
        );
        Self {
            resolutions,
            resolution_by_node,
            references_by_definition,
        }
    }

    /// Borrow one source reference's result, or `None` for a non-reference.
    #[must_use]
    pub fn resolution(&self, reference: StackNodeId) -> Option<&StackResolution<S>> {
        self.resolution_by_node
            .get(reference.index())
            .and_then(|index| index.map(|index| &self.resolutions[index]))
    }

    /// References resolving to `definition`, in source node order.
    ///
    /// # Panics
    ///
    /// Panics when `definition` was not present in the resolved graph.
    #[must_use]
    pub fn references_to(&self, definition: StackNodeId) -> &[StackNodeId] {
        &self.references_by_definition[definition.index()]
    }

    /// Results in source reference node order.
    #[must_use]
    pub fn resolutions(&self) -> &[StackResolution<S>] {
        &self.resolutions
    }
}

/// Work accounting and complete-path retention shared by the direct, linear,
/// and stitched resolutions, so that all three bound and report a search
/// identically.
pub(super) struct SearchState<S> {
    pub(super) budget: PathBudget,
    pub(super) cycle_cut_count: usize,
    pub(super) rejected_extension_count: usize,
    complete: SmallVec<[StackPath<S>; 1]>,
}

impl<S: Clone + Eq> SearchState<S> {
    pub(super) fn new(config: &PathSearchConfig) -> Self {
        Self {
            budget: PathBudget::new(config),
            cycle_cut_count: 0,
            rejected_extension_count: 0,
            complete: SmallVec::new(),
        }
    }

    /// Count a complete path just found and retain a copy unless the
    /// retention bound refuses it.
    pub(super) fn record_complete(&mut self, path: &StackPath<S>) {
        if self.budget.admit_complete(self.complete.len()) {
            self.complete.push(path.clone());
        }
    }

    /// Keep the distinct retained paths that no other one shadows by
    /// precedence, and report the search.
    pub(super) fn finish<F, N, E>(
        self,
        graph: &StackGraph<F, S, N, E>,
        reference: StackNodeId,
    ) -> StackResolution<S> {
        let stats = StackSearchStats {
            search: self.budget.stats(),
            rejected_extension_count: self.rejected_extension_count,
            cycle_cut_count: self.cycle_cut_count,
        };
        let mut paths = self.complete;
        if paths.len() > 1 {
            let mut unique: SmallVec<[StackPath<S>; 1]> = SmallVec::new();
            for path in paths {
                if !unique.contains(&path) {
                    unique.push(path);
                }
            }
            paths = unique;
        }
        retain_unshadowed(&mut paths, |other, path| other.shadows(graph, path));
        StackResolution {
            reference,
            paths,
            stats,
        }
    }
}
