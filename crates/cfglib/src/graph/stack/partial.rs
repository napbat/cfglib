//! Per-file structural path summaries for incremental stitching.

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::search::SearchOrder;
use crate::identity::define_dense_id;

use super::path::{StackPath, StackPathError};
use super::storage::{StackEdgeId, StackFileId, StackGraph, StackGraphError, StackNodeId};

define_dense_id! {
    /// Stable identity of a path summary in a [`StackPartialPathDatabase`].
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct StackPartialPathId(u32);
    display = "pp";
    /// Construct an identity from a dense arena index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    from_index = "partial-path index exceeds u32::MAX";
}

/// A file-local structural route between stack-graph stitching endpoints.
///
/// A summary deliberately stores the original edge identities instead of
/// specializing its symbol/scope preconditions. At query time
/// [`append_to`](Self::append_to) replays those edges against a concrete path,
/// which enforces the exact stack semantics and keeps summaries independent of
/// the consumer's symbol representation.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StackPartialPath {
    file: StackFileId,
    start: StackNodeId,
    end: StackNodeId,
    edges: Vec<StackEdgeId>,
}

impl StackPartialPath {
    /// Replay this summary onto `path` transactionally.
    ///
    /// A summary ending at the jump-to-scope singleton may leave the concrete
    /// path at a different exported scope because the target comes from that
    /// path's scope stack.
    ///
    /// # Errors
    ///
    /// Returns a [`StackPathError`] when the concrete path does not end at this
    /// summary's start or an edge violates the current symbol/scope stacks.
    pub fn append_to<F, S: Clone + Eq, N, E>(
        &self,
        graph: &StackGraph<F, S, N, E>,
        path: &mut StackPath<S>,
    ) -> Result<(), StackPathError> {
        if path.end() != self.start {
            return Err(StackPathError::IncorrectSource {
                expected: path.end(),
                actual: self.start,
            });
        }
        let mut next = path.clone();
        for &edge in &self.edges {
            next.append(graph, edge)?;
        }
        *path = next;
        Ok(())
    }

    /// Create a concrete path when this summary starts at `reference`.
    ///
    /// # Errors
    ///
    /// Returns a [`StackPathError`] when the start node is not a source
    /// reference or replay rejects an edge.
    pub fn promote<F, S: Clone + Eq, N, E>(
        &self,
        graph: &StackGraph<F, S, N, E>,
    ) -> Result<StackPath<S>, StackPathError> {
        let mut path = StackPath::from_reference(graph, self.start)?;
        self.append_to(graph, &mut path)?;
        Ok(path)
    }

    /// File partition this summary was extracted from.
    #[must_use]
    pub const fn file(&self) -> StackFileId {
        self.file
    }

    /// Structural start endpoint.
    #[must_use]
    pub const fn start(&self) -> StackNodeId {
        self.start
    }

    /// Structural end endpoint before a possible virtual jump is resolved.
    #[must_use]
    pub const fn end(&self) -> StackNodeId {
        self.end
    }

    /// Stored edge identities in traversal order.
    #[must_use]
    pub fn edges(&self) -> &[StackEdgeId] {
        &self.edges
    }

    /// Number of stored edges in this summary.
    #[must_use]
    pub fn len(&self) -> usize {
        self.edges.len()
    }

    /// Whether this summary contains no stored edges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Deterministic bounds and frontier order for per-file path extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StackPartialPathConfig {
    /// Structural frontier order.
    pub order: SearchOrder,
    /// Maximum edges in one summary. `None` is unbounded.
    pub max_path_length: Option<usize>,
    /// Maximum structural prefixes expanded per file. `None` is unbounded.
    pub max_path_count: Option<usize>,
}

impl StackPartialPathConfig {
    /// Construct an unbounded breadth-first extraction configuration.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            order: SearchOrder::BreadthFirst,
            max_path_length: None,
            max_path_count: None,
        }
    }

    /// Return this configuration with a different frontier order.
    #[must_use]
    pub const fn with_order(mut self, order: SearchOrder) -> Self {
        self.order = order;
        self
    }

    /// Return this configuration with a structural path-length bound.
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
}

impl Default for StackPartialPathConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Work and truncation information for one file's summary extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StackPartialPathStats {
    /// Number of structural path prefixes expanded.
    pub expanded_path_count: usize,
    /// Number of endpoint summaries produced.
    pub partial_path_count: usize,
    /// Extensions skipped because the edge already occurred in the route.
    pub cycle_cut_count: usize,
    /// Whether an otherwise usable edge hit the path-length bound.
    pub path_length_limited: bool,
    /// Whether queued prefixes remained when the path-count bound was reached.
    pub path_count_limited: bool,
}

impl StackPartialPathStats {
    /// Whether configured bounds may have omitted a summary.
    #[must_use]
    pub const fn is_truncated(&self) -> bool {
        self.path_length_limited || self.path_count_limited
    }
}

/// All structural path summaries extracted independently for one file.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StackPartialPathSet {
    file: StackFileId,
    paths: Vec<StackPartialPath>,
    stats: StackPartialPathStats,
}

impl StackPartialPathSet {
    /// Extract all productive routes between stitching endpoints in `file`.
    ///
    /// Starts are the root singleton, source references in the file, and its
    /// exported scopes. A summary is recorded at the root, an exported scope,
    /// a source definition, or the jump singleton. Exploration also continues
    /// past those endpoints because a definition may be only an intermediate
    /// pop while symbols remain. Only edges belonging to this file are
    /// traversed, so the result can be cached by file content.
    ///
    /// # Errors
    ///
    /// Returns [`StackGraphError::UnknownFile`] when `file` is outside `graph`.
    pub fn compute<F, S, N, E>(
        graph: &StackGraph<F, S, N, E>,
        file: StackFileId,
        config: StackPartialPathConfig,
    ) -> Result<Self, StackGraphError> {
        if file.index() >= graph.file_count() {
            return Err(StackGraphError::UnknownFile(file));
        }
        let mut start_nodes = Vec::with_capacity(graph.file_nodes(file).len() + 1);
        start_nodes.push(graph.root_node());
        start_nodes.extend(graph.file_nodes(file).iter().copied().filter(|&node| {
            let kind = graph.node(node).kind();
            kind.is_reference() || kind.is_exported_scope()
        }));

        let mut frontier = VecDeque::new();
        for start in start_nodes {
            frontier.push_back(Route {
                start,
                current: start,
                edges: Vec::new(),
            });
        }
        let mut paths = Vec::new();
        let mut stats = StackPartialPathStats::default();

        while let Some(route) = pop_route(&mut frontier, config.order) {
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
                .is_some_and(|limit| route.edges.len() >= limit);
            let mut extensions = Vec::new();
            for edge in graph
                .outgoing(route.current)
                .filter(|&edge| edge_belongs_to_file(graph, edge, file))
            {
                if route.edges.contains(&edge) {
                    stats.cycle_cut_count += 1;
                    continue;
                }
                if at_length_limit {
                    stats.path_length_limited = true;
                    continue;
                }
                let target = graph.edge(edge).target();
                let mut edges = route.edges.clone();
                edges.push(edge);
                if is_endpoint(graph, target) {
                    paths.push(StackPartialPath {
                        file,
                        start: route.start,
                        end: target,
                        edges: edges.clone(),
                    });
                }
                extensions.push(Route {
                    start: route.start,
                    current: target,
                    edges,
                });
            }
            push_routes(&mut frontier, extensions, config.order);
        }

        stats.partial_path_count = paths.len();
        Ok(Self { file, paths, stats })
    }

    /// File these summaries belong to.
    #[must_use]
    pub const fn file(&self) -> StackFileId {
        self.file
    }

    /// Extracted summaries in stable discovery order.
    #[must_use]
    pub fn paths(&self) -> &[StackPartialPath] {
        &self.paths
    }

    /// Extraction work and truncation information.
    #[must_use]
    pub const fn stats(&self) -> StackPartialPathStats {
        self.stats
    }
}

/// Incrementally replaceable database of per-file path summaries.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StackPartialPathDatabase {
    paths: Vec<Option<StackPartialPath>>,
    paths_by_start: Vec<Vec<StackPartialPathId>>,
    paths_by_file: Vec<Vec<StackPartialPathId>>,
}

impl StackPartialPathDatabase {
    /// Create an empty database sized for `graph`.
    #[must_use]
    pub fn new<F, S, N, E>(graph: &StackGraph<F, S, N, E>) -> Self {
        Self {
            paths: Vec::new(),
            paths_by_start: vec![Vec::new(); graph.node_bound()],
            paths_by_file: vec![Vec::new(); graph.file_count()],
        }
    }

    /// Extract and install summaries for every file in identity order.
    ///
    /// # Panics
    ///
    /// Panics only if an identity produced by `graph.file_ids()` is rejected
    /// by the same graph, which would violate [`StackGraph`]'s invariants.
    #[must_use]
    pub fn compute<F, S, N, E>(
        graph: &StackGraph<F, S, N, E>,
        config: StackPartialPathConfig,
    ) -> Self {
        let mut database = Self::new(graph);
        for file in graph.file_ids() {
            let set = StackPartialPathSet::compute(graph, file, config)
                .expect("a graph-provided file identity remains valid");
            database.install(graph, set);
        }
        database
    }

    /// Re-extract one file and atomically replace its installed summaries.
    ///
    /// Existing partial-path identities become tombstones; every other
    /// identity remains stable.
    ///
    /// # Errors
    ///
    /// Returns [`StackGraphError::UnknownFile`] when `file` is outside `graph`.
    pub fn replace_file<F, S, N, E>(
        &mut self,
        graph: &StackGraph<F, S, N, E>,
        file: StackFileId,
        config: StackPartialPathConfig,
    ) -> Result<StackPartialPathStats, StackGraphError> {
        let set = StackPartialPathSet::compute(graph, file, config)?;
        self.remove_file(file);
        let stats = set.stats();
        self.install(graph, set);
        Ok(stats)
    }

    /// Borrow one live summary, or `None` for an unknown identity or tombstone.
    #[must_use]
    pub fn path(&self, path: StackPartialPathId) -> Option<&StackPartialPath> {
        self.paths.get(path.index()).and_then(Option::as_ref)
    }

    /// Live summaries starting at `node`, in installation order.
    pub fn paths_from(
        &self,
        node: StackNodeId,
    ) -> impl Iterator<Item = (StackPartialPathId, &StackPartialPath)> {
        self.paths_by_start
            .get(node.index())
            .into_iter()
            .flatten()
            .filter_map(|&id| self.path(id).map(|path| (id, path)))
    }

    /// Live summaries belonging to `file`, in installation order.
    pub fn file_paths(
        &self,
        file: StackFileId,
    ) -> impl Iterator<Item = (StackPartialPathId, &StackPartialPath)> {
        self.paths_by_file
            .get(file.index())
            .into_iter()
            .flatten()
            .filter_map(|&id| self.path(id).map(|path| (id, path)))
    }

    /// Number of live summaries.
    #[must_use]
    pub fn path_count(&self) -> usize {
        self.paths.iter().filter(|path| path.is_some()).count()
    }

    /// Number of summary slots, including replaced-file tombstones.
    #[must_use]
    pub fn path_slot_count(&self) -> usize {
        self.paths.len()
    }

    fn remove_file(&mut self, file: StackFileId) {
        let Some(ids) = self.paths_by_file.get_mut(file.index()) else {
            return;
        };
        let removed = core::mem::take(ids);
        for id in removed {
            let Some(path) = self.paths[id.index()].take() else {
                continue;
            };
            self.paths_by_start[path.start().index()].retain(|&candidate| candidate != id);
        }
    }

    fn install<F, S, N, E>(&mut self, graph: &StackGraph<F, S, N, E>, set: StackPartialPathSet) {
        self.paths_by_start
            .resize_with(graph.node_bound(), Vec::new);
        self.paths_by_file.resize_with(graph.file_count(), Vec::new);
        for path in set.paths {
            let id = StackPartialPathId::from_index(self.paths.len());
            self.paths_by_start[path.start().index()].push(id);
            self.paths_by_file[path.file().index()].push(id);
            self.paths.push(Some(path));
        }
    }
}

struct Route {
    start: StackNodeId,
    current: StackNodeId,
    edges: Vec<StackEdgeId>,
}

fn edge_belongs_to_file<F, S, N, E>(
    graph: &StackGraph<F, S, N, E>,
    edge: StackEdgeId,
    file: StackFileId,
) -> bool {
    let edge = graph.edge(edge);
    graph.node(edge.source()).file() == Some(file) || graph.node(edge.target()).file() == Some(file)
}

fn is_endpoint<F, S, N, E>(graph: &StackGraph<F, S, N, E>, node: StackNodeId) -> bool {
    let kind = graph.node(node).kind();
    kind.is_root() || kind.is_exported_scope() || kind.is_definition() || kind.is_jump_to_scope()
}

fn pop_route(frontier: &mut VecDeque<Route>, order: SearchOrder) -> Option<Route> {
    match order {
        SearchOrder::BreadthFirst => frontier.pop_front(),
        SearchOrder::DepthFirst => frontier.pop_back(),
    }
}

fn push_routes(frontier: &mut VecDeque<Route>, routes: Vec<Route>, order: SearchOrder) {
    match order {
        SearchOrder::BreadthFirst => frontier.extend(routes),
        SearchOrder::DepthFirst => frontier.extend(routes.into_iter().rev()),
    }
}
