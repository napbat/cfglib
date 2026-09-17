//! Direct stack-graph path finding and forward/reverse binding indexes.

extern crate alloc;

use alloc::collections::{BTreeSet, VecDeque};
use alloc::vec;
use alloc::vec::Vec;
use smallvec::SmallVec;

use crate::graph::search::SearchOrder;

use super::path::{StackPath, StackPathError};
use super::storage::{StackGraph, StackNodeId};

/// Deterministic bounds and frontier order for direct stack-graph search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackSearchConfig {
    /// Whether the next path prefix comes from the front or back of the frontier.
    pub order: SearchOrder,
    /// Maximum stored edges in a path. `None` is unbounded.
    pub max_path_length: Option<usize>,
    /// Maximum path prefixes to expand. `None` is unbounded.
    pub max_path_count: Option<usize>,
    /// Maximum complete paths retained before shadow filtering. `None` is unbounded.
    pub max_complete_path_count: Option<usize>,
}

impl StackSearchConfig {
    /// Construct an unbounded breadth-first search configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            order: SearchOrder::BreadthFirst,
            max_path_length: None,
            max_path_count: None,
            max_complete_path_count: None,
        }
    }

    /// Return this configuration with a different frontier order.
    #[must_use]
    pub const fn with_order(mut self, order: SearchOrder) -> Self {
        self.order = order;
        self
    }

    /// Return this configuration with a stored-edge length bound.
    #[must_use]
    pub const fn with_max_path_length(mut self, max_path_length: usize) -> Self {
        self.max_path_length = Some(max_path_length);
        self
    }

    /// Return this configuration with an expanded-prefix bound.
    #[must_use]
    pub const fn with_max_path_count(mut self, max_path_count: usize) -> Self {
        self.max_path_count = Some(max_path_count);
        self
    }

    /// Return this configuration with a retained-complete-path bound.
    #[must_use]
    pub const fn with_max_complete_path_count(mut self, max_complete_path_count: usize) -> Self {
        self.max_complete_path_count = Some(max_complete_path_count);
        self
    }
}

impl Default for StackSearchConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Work, rejection, and truncation information for one stack-graph query.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StackSearchStats {
    /// Number of path prefixes expanded.
    pub expanded_path_count: usize,
    /// Number of complete paths found before shadow filtering and retention bounds.
    pub complete_path_count: usize,
    /// Extensions rejected by symbol or scope stack semantics.
    pub rejected_extension_count: usize,
    /// Extensions skipped because the stored edge already occurs on the path.
    pub cycle_cut_count: usize,
    /// Whether an unused outgoing edge hit the path-length bound.
    pub path_length_limited: bool,
    /// Whether queued prefixes remained when the path-count bound was reached.
    pub path_count_limited: bool,
    /// Whether a complete path was omitted by the complete-path bound.
    pub complete_path_count_limited: bool,
}

impl StackSearchStats {
    /// Whether configured resource bounds may have omitted a binding.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.path_length_limited || self.path_count_limited || self.complete_path_count_limited
    }
}

/// A stack query expected to be linear either encountered invalid path state or
/// admitted several viable outgoing edges.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackLinearResolutionError {
    /// A node or edge could not participate in a well-formed stack path.
    Path(StackPathError),
    /// More than one outgoing edge extended the same path prefix successfully.
    MultipleEdges {
        /// Node whose viable outgoing edges caused the branch.
        node: StackNodeId,
    },
}

impl core::fmt::Display for StackLinearResolutionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Path(error) => error.fmt(formatter),
            Self::MultipleEdges { node } => {
                write!(
                    formatter,
                    "stack path at {node} admits multiple outgoing edges"
                )
            }
        }
    }
}

impl core::error::Error for StackLinearResolutionError {
    fn source(&self) -> Option<&(dyn core::error::Error + 'static)> {
        match self {
            Self::Path(error) => Some(error),
            Self::MultipleEdges { .. } => None,
        }
    }
}

impl From<StackPathError> for StackLinearResolutionError {
    fn from(error: StackPathError) -> Self {
        Self::Path(error)
    }
}

/// Visible complete paths for one stack-graph reference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackResolution<S> {
    reference: StackNodeId,
    paths: SmallVec<[StackPath<S>; 1]>,
    stats: StackSearchStats,
}

impl<S: Clone + Eq> StackResolution<S> {
    /// Resolve one reference, panicking if the node is not a source reference.
    ///
    /// Use [`try_compute`](Self::try_compute) when the node identity is not
    /// already known to be a reference.
    ///
    /// # Panics
    ///
    /// Panics when `reference` is not a source reference node.
    #[must_use]
    pub fn compute<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        reference: StackNodeId,
        config: StackSearchConfig,
    ) -> Self {
        Self::try_compute(graph, reference, config)
            .expect("stack resolution requires a source reference node")
    }

    /// Resolve one reference with a structured invalid-seed error.
    ///
    /// Search follows only well-formed stack transitions. A stored edge may
    /// occur at most once on a path, matching the standard direct stack-graph
    /// cycle guard. Complete paths shadowed by greater edge precedence are
    /// removed after search.
    ///
    /// # Errors
    ///
    /// Returns [`StackPathError::UnknownNode`] when `reference` is outside the
    /// graph, or [`StackPathError::NotReference`] when it is not a source
    /// reference node.
    pub fn try_compute<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        reference: StackNodeId,
        config: StackSearchConfig,
    ) -> Result<Self, StackPathError> {
        let initial = StackPath::from_reference(graph, reference)?;
        let mut frontier = VecDeque::from([initial]);
        let mut complete = Vec::new();
        let mut stats = StackSearchStats::default();

        while let Some(path) = pop_frontier(&mut frontier, config.order) {
            if config
                .max_path_count
                .is_some_and(|limit| stats.expanded_path_count >= limit)
            {
                stats.path_count_limited = true;
                break;
            }
            stats.expanded_path_count += 1;

            let at_length_limit = config
                .max_path_length
                .is_some_and(|limit| path.len() >= limit);
            let mut extensions = Vec::new();
            for edge in graph.outgoing(path.end()) {
                if path.contains_edge(edge) {
                    stats.cycle_cut_count += 1;
                    continue;
                }
                if at_length_limit {
                    stats.path_length_limited = true;
                    continue;
                }
                let mut next = path.clone();
                if next.append(graph, edge).is_err() {
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
                extensions.push(next);
            }
            push_extensions(&mut frontier, extensions, config.order);
        }

        Ok(Self::from_complete(graph, reference, &complete, stats))
    }

    /// Resolve a reference whose well-formed path has at most one viable
    /// outgoing edge at every prefix.
    ///
    /// This applies the same stack transitions, cycle guard, bounds, complete
    /// path retention, and precedence filtering as [`try_compute`](Self::try_compute),
    /// while updating a single path in place.
    ///
    /// # Errors
    ///
    /// Returns [`StackLinearResolutionError::Path`] for an invalid reference or
    /// path identity, and [`StackLinearResolutionError::MultipleEdges`] when a
    /// prefix branches into several well-formed paths.
    pub fn try_compute_linear<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        reference: StackNodeId,
        config: StackSearchConfig,
    ) -> Result<Self, StackLinearResolutionError> {
        let mut path = StackPath::from_reference(graph, reference)?;
        let mut complete: SmallVec<[StackPath<S>; 1]> = SmallVec::new();
        let mut stats = StackSearchStats::default();

        loop {
            if config
                .max_path_count
                .is_some_and(|limit| stats.expanded_path_count >= limit)
            {
                stats.path_count_limited = true;
                break;
            }
            stats.expanded_path_count += 1;
            let at_length_limit = config
                .max_path_length
                .is_some_and(|limit| path.len() >= limit);
            let mut next = None;
            for edge in graph.outgoing(path.end()) {
                if path.contains_edge(edge) {
                    stats.cycle_cut_count += 1;
                    continue;
                }
                if at_length_limit {
                    stats.path_length_limited = true;
                    continue;
                }
                let Ok(candidate) = path.clone().append_owned(graph, edge) else {
                    stats.rejected_extension_count += 1;
                    continue;
                };
                if next.is_some() {
                    return Err(StackLinearResolutionError::MultipleEdges { node: path.end() });
                }
                if candidate.is_complete(graph) {
                    stats.complete_path_count += 1;
                    if config
                        .max_complete_path_count
                        .is_none_or(|limit| complete.len() < limit)
                    {
                        complete.push(candidate.clone());
                    } else {
                        stats.complete_path_count_limited = true;
                    }
                }
                next = Some(candidate);
            }
            let Some(candidate) = next else {
                break;
            };
            path = candidate;
        }

        if complete.len() <= 1 {
            return Ok(Self {
                reference,
                paths: complete,
                stats,
            });
        }
        Ok(Self::from_complete(graph, reference, &complete, stats))
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

    pub(super) fn from_complete<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        reference: StackNodeId,
        complete: &[StackPath<S>],
        stats: StackSearchStats,
    ) -> Self {
        let mut unique: SmallVec<[StackPath<S>; 1]> = SmallVec::new();
        for path in complete {
            if !unique.contains(path) {
                unique.push(path.clone());
            }
        }
        let paths: SmallVec<[StackPath<S>; 1]> = unique
            .iter()
            .enumerate()
            .filter(|&(index, path)| {
                !unique
                    .iter()
                    .enumerate()
                    .any(|(other_index, other)| index != other_index && other.shadows(graph, path))
            })
            .map(|(_, path)| path.clone())
            .collect();
        Self {
            reference,
            paths,
            stats,
        }
    }
}

/// Direct resolutions for every source reference, plus a reverse index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackResolutionIndex<S> {
    resolutions: Vec<StackResolution<S>>,
    resolution_by_node: Vec<Option<usize>>,
    references_by_definition: Vec<Vec<StackNodeId>>,
}

impl<S: Clone + Eq> StackResolutionIndex<S> {
    /// Resolve every source reference in node order and build reverse bindings.
    #[must_use]
    pub fn compute<F, N, E>(graph: &StackGraph<F, S, N, E>, config: StackSearchConfig) -> Self {
        let resolutions = graph
            .node_ids()
            .filter(|&node| graph.node(node).kind().is_reference())
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
        config: StackSearchConfig,
    ) -> Result<Self, StackLinearResolutionError> {
        let resolutions = graph
            .node_ids()
            .filter(|&node| graph.node(node).kind().is_reference())
            .map(|reference| StackResolution::try_compute_linear(graph, reference, config))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self::from_resolutions(graph, resolutions))
    }

    pub(super) fn from_resolutions<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        resolutions: Vec<StackResolution<S>>,
    ) -> Self {
        let mut resolution_by_node = vec![None; graph.node_bound()];
        let mut references_by_definition = vec![Vec::new(); graph.node_bound()];
        for (index, resolution) in resolutions.iter().enumerate() {
            let reference = resolution.reference();
            resolution_by_node[reference.index()] = Some(index);
            add_reverse_references(&mut references_by_definition, resolution);
        }
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

/// Definition-to-reference bindings without retained forward resolutions.
///
/// This is the memory-efficient index for consumers that only need find-
/// references queries. Each resolution is incorporated and discarded during
/// construction; use [`StackResolutionIndex`] when forward paths are also
/// required.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackReverseIndex {
    references_by_definition: Vec<Vec<StackNodeId>>,
}

impl StackReverseIndex {
    /// Resolve every source reference and build only reverse bindings.
    #[must_use]
    pub fn compute<F, S: Clone + Eq, N, E>(
        graph: &StackGraph<F, S, N, E>,
        config: StackSearchConfig,
    ) -> Self {
        let mut index = Self::empty(graph);
        for reference in graph
            .node_ids()
            .filter(|&node| graph.node(node).kind().is_reference())
        {
            let resolution = StackResolution::compute(graph, reference, config);
            index.add_resolution(&resolution);
        }
        index
    }

    /// Resolve every source reference under the linear-path contract and build
    /// only reverse bindings.
    ///
    /// # Errors
    ///
    /// Returns the first [`StackLinearResolutionError`] in source-node order.
    pub fn try_compute_linear<F, S: Clone + Eq, N, E>(
        graph: &StackGraph<F, S, N, E>,
        config: StackSearchConfig,
    ) -> Result<Self, StackLinearResolutionError> {
        let mut index = Self::empty(graph);
        for reference in graph
            .node_ids()
            .filter(|&node| graph.node(node).kind().is_reference())
        {
            let resolution = StackResolution::try_compute_linear(graph, reference, config)?;
            index.add_resolution(&resolution);
        }
        Ok(index)
    }

    fn empty<F, S, N, E>(graph: &StackGraph<F, S, N, E>) -> Self {
        Self {
            references_by_definition: vec![Vec::new(); graph.node_bound()],
        }
    }

    fn add_resolution<S: Clone + Eq>(&mut self, resolution: &StackResolution<S>) {
        add_reverse_references(&mut self.references_by_definition, resolution);
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
}

fn add_reverse_references<S: Clone + Eq>(
    references_by_definition: &mut [Vec<StackNodeId>],
    resolution: &StackResolution<S>,
) {
    let reference = resolution.reference();
    match resolution.paths() {
        [] => {}
        [path] => references_by_definition[path.end().index()].push(reference),
        paths => {
            let mut seen = BTreeSet::new();
            for path in paths {
                let definition = path.end();
                if seen.insert(definition) {
                    references_by_definition[definition.index()].push(reference);
                }
            }
        }
    }
}

fn pop_frontier<S>(
    frontier: &mut VecDeque<StackPath<S>>,
    order: SearchOrder,
) -> Option<StackPath<S>> {
    match order {
        SearchOrder::BreadthFirst => frontier.pop_front(),
        SearchOrder::DepthFirst => frontier.pop_back(),
    }
}

fn push_extensions<S>(
    frontier: &mut VecDeque<StackPath<S>>,
    extensions: Vec<StackPath<S>>,
    order: SearchOrder,
) {
    match order {
        SearchOrder::BreadthFirst => frontier.extend(extensions),
        SearchOrder::DepthFirst => frontier.extend(extensions.into_iter().rev()),
    }
}
