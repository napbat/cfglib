//! Per-file structural path summaries for incremental stitching.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::graph::edge_view::EdgeView;
use crate::graph::search::{PathBudget, PathFrontier, PathSearchConfig, PathSearchStats};
use crate::identity::define_dense_id;

use super::path::{StackPath, StackPathError};
use super::storage::{StackEdgeId, StackFileId, StackGraph, StackNodeId};

#[cfg(feature = "serde")]
mod stored;

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
///
/// A summary holds at least the edge that reached its end and takes no edge
/// twice, as extraction guarantees; a deserialized summary is checked for
/// both. Its identities name the graph it was extracted from, which a
/// document cannot check.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "stored::Path"))]
pub struct StackPartialPath {
    file: StackFileId,
    start: StackNodeId,
    end: StackNodeId,
    edges: Vec<StackEdgeId>,
}

impl StackPartialPath {
    /// Replay this summary onto `path` and return the extended path.
    ///
    /// A summary ending at the jump-to-scope singleton may leave the concrete
    /// path at a different exported scope because the target comes from that
    /// path's scope stack. The path is consumed: a caller that needs it after
    /// a rejected replay passes a clone.
    ///
    /// # Errors
    ///
    /// Returns a [`StackPathError`] when the concrete path does not end at this
    /// summary's start or an edge violates the current symbol/scope stacks.
    pub fn append_to<F, S: Clone + Eq, N, E>(
        &self,
        graph: &StackGraph<F, S, N, E>,
        mut path: StackPath<S>,
    ) -> Result<StackPath<S>, StackPathError> {
        if path.end() != self.start {
            return Err(StackPathError::IncorrectSource {
                expected: path.end(),
                actual: self.start,
            });
        }
        for &edge in &self.edges {
            path.append(graph, edge)?;
        }
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
    ///
    /// Always `false`: a summary is recorded after the edge that reached its
    /// end, and a deserialized one without an edge is rejected.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.edges.is_empty()
    }
}

/// Work and truncation information for one file's summary extraction.
///
/// `search.complete_path_count` is the number of summaries extracted.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StackPartialPathStats {
    /// The prefixes expanded, the summaries recorded, and the bounds that
    /// refused work.
    pub search: PathSearchStats,
    /// Extensions skipped because the edge already occurred in the route.
    pub cycle_cut_count: usize,
}

/// All structural path summaries extracted independently for one file.
///
/// Every summary belongs to the set's file, and the statistics count exactly
/// the summaries the set holds; a deserialized set is checked for both.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "stored::Set"))]
pub struct StackPartialPathSet {
    file: StackFileId,
    paths: Vec<StackPartialPath>,
    stats: StackPartialPathStats,
}

impl StackPartialPathSet {
    /// Extract all productive routes between stitching endpoints in `file`.
    ///
    /// Starts are the root singleton, source references in the file, and its
    /// exported scopes, expanded in that order in `config.order`. A summary
    /// is recorded at the root, an exported scope, a source definition, or
    /// the jump singleton. Exploration also continues past those endpoints
    /// because a definition may be only an intermediate pop while symbols
    /// remain. Only edges belonging to this file are traversed, so the result
    /// can be cached by file content. Every summary is kept, so extraction
    /// does not read `config.max_complete_path_count`.
    ///
    /// # Panics
    ///
    /// Panics when `file` does not belong to `graph`.
    #[must_use]
    pub fn compute<F, S, N, E>(
        graph: &StackGraph<F, S, N, E>,
        file: StackFileId,
        config: PathSearchConfig,
    ) -> Self {
        assert!(
            file.index() < graph.file_count(),
            "stack-graph file {file} is out of range"
        );
        let mut start_nodes = Vec::with_capacity(graph.file_nodes(file).len() + 1);
        start_nodes.push(graph.root_node());
        start_nodes.extend(graph.file_nodes(file).iter().copied().filter(|&node| {
            let kind = graph.node(node).kind();
            kind.is_reference() || kind.is_exported_scope()
        }));

        let mut frontier = PathFrontier::new(config.order);
        for start in start_nodes {
            frontier.push(Route {
                start,
                current: start,
                edges: Vec::new(),
            });
        }
        let mut paths = Vec::new();
        let mut budget = PathBudget::new(&config);
        let mut cycle_cut_count = 0;

        while let Some(route) = frontier.pop() {
            if !budget.admit_expansion() {
                break;
            }
            for edge in graph
                .outgoing(route.current)
                .filter(|&edge| edge_belongs_to_file(graph, edge, file))
            {
                if route.edges.contains(&edge) {
                    cycle_cut_count += 1;
                    continue;
                }
                if !budget.admit_length(route.edges.len() + 1) {
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
                frontier.push(Route {
                    start: route.start,
                    current: target,
                    edges,
                });
            }
        }

        let stats = StackPartialPathStats {
            search: PathSearchStats {
                complete_path_count: paths.len(),
                ..budget.stats()
            },
            cycle_cut_count,
        };
        Self { file, paths, stats }
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
///
/// The per-start and per-file tables list every live summary exactly once,
/// under its own start and file, in installation order, and no tombstone. A
/// deserialized database is checked for that and for each summary (see
/// [`StackPartialPath`]). What a database cannot check alone is that its node,
/// edge, and file identities fit the graph it is used with: a summary whose
/// edge that graph no longer holds is reported by stitching, as documented
/// there, so a database is only meaningful beside the graph it was built for.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "stored::Database"))]
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
            paths_by_start: vec![Vec::new(); graph.graph().node_bound()],
            paths_by_file: vec![Vec::new(); graph.file_count()],
        }
    }

    /// Extract and install summaries for every file in identity order.
    #[must_use]
    pub fn compute<F, S, N, E>(graph: &StackGraph<F, S, N, E>, config: PathSearchConfig) -> Self {
        let mut database = Self::new(graph);
        for file in graph.file_ids() {
            database.install(graph, StackPartialPathSet::compute(graph, file, config));
        }
        database
    }

    /// Re-extract one file and atomically replace its installed summaries.
    ///
    /// Existing partial-path identities become tombstones; every other
    /// identity remains stable.
    ///
    /// # Panics
    ///
    /// Panics when `file` does not belong to `graph`.
    pub fn replace_file<F, S, N, E>(
        &mut self,
        graph: &StackGraph<F, S, N, E>,
        file: StackFileId,
        config: PathSearchConfig,
    ) -> StackPartialPathStats {
        let set = StackPartialPathSet::compute(graph, file, config);
        self.remove_file(file);
        let stats = set.stats();
        self.install(graph, set);
        stats
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
    #[cfg(test)]
    pub(super) fn path_slot_count(&self) -> usize {
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
            .resize_with(graph.graph().node_bound(), Vec::new);
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
