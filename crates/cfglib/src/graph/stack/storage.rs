//! Owned stack-graph storage and construction invariants.

extern crate alloc;

use alloc::vec::Vec;
use core::ops::Index;

use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::store::{Graph, Id, IdTag};
use crate::graph::view::GraphView;

/// Tag marking an identity that addresses a stack-graph file partition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StackFileTag;

impl IdTag for StackFileTag {
    const PREFIX: &'static str = "f";
}

/// Tag marking an identity that addresses a stack-graph node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StackNodeTag;

impl IdTag for StackNodeTag {
    const PREFIX: &'static str = "sn";
}

/// Tag marking an identity that addresses a stack-graph edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct StackEdgeTag;

impl IdTag for StackEdgeTag {
    const PREFIX: &'static str = "sg-e";
}

/// Dense identity of a file partition in a [`StackGraph`].
///
/// A file is not a graph entity — it indexes the graph's file table — but it
/// is dense and stable, so it shares the crate's one identity definition.
pub type StackFileId = Id<StackFileTag>;

/// Dense identity of a node in a [`StackGraph`].
pub type StackNodeId = Id<StackNodeTag>;

/// Stable identity of an edge in a [`StackGraph`].
pub type StackEdgeId = Id<StackEdgeTag>;

/// The store a [`StackGraph`] is built on.
type StackStore<S, N, E> = Graph<StackNode<S, N>, StackEdge<E>, StackNodeTag, StackEdgeTag>;

/// The semantic kind of one stack-graph node.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum StackNodeKind<S> {
    /// The singleton root through which file partitions connect.
    Root,
    /// A scope; exported scopes may appear on the scope stack.
    Scope {
        /// Whether other nodes may refer to this scope on the scope stack.
        exported: bool,
    },
    /// Push an unscoped symbol onto the symbol stack.
    PushSymbol {
        /// Consumer-defined symbol.
        symbol: S,
        /// Whether this node represents a source reference.
        is_reference: bool,
    },
    /// Pop a matching unscoped symbol from the symbol stack.
    PopSymbol {
        /// Consumer-defined symbol.
        symbol: S,
        /// Whether this node represents a source definition.
        is_definition: bool,
    },
    /// Push a symbol with an attached exported scope list.
    PushScopedSymbol {
        /// Consumer-defined symbol.
        symbol: S,
        /// Exported scope prepended to the current scope stack.
        scope: StackNodeId,
        /// Whether this node represents a source reference.
        is_reference: bool,
    },
    /// Pop a matching scoped symbol and restore its attached scope list.
    PopScopedSymbol {
        /// Consumer-defined symbol.
        symbol: S,
        /// Whether this node represents a source definition.
        is_definition: bool,
    },
    /// Clear the scope stack.
    DropScopes,
    /// Pop and jump to the exported scope atop the scope stack.
    JumpToScope,
}

impl<S> StackNodeKind<S> {
    /// Borrow this node's symbol, if it carries one.
    #[must_use]
    pub const fn symbol(&self) -> Option<&S> {
        match self {
            Self::PushSymbol { symbol, .. }
            | Self::PopSymbol { symbol, .. }
            | Self::PushScopedSymbol { symbol, .. }
            | Self::PopScopedSymbol { symbol, .. } => Some(symbol),
            Self::Root | Self::Scope { .. } | Self::DropScopes | Self::JumpToScope => None,
        }
    }

    /// Whether this is a source reference node.
    #[must_use]
    pub const fn is_reference(&self) -> bool {
        matches!(
            self,
            Self::PushSymbol {
                is_reference: true,
                ..
            } | Self::PushScopedSymbol {
                is_reference: true,
                ..
            }
        )
    }

    /// Whether this is a source definition node.
    #[must_use]
    pub const fn is_definition(&self) -> bool {
        matches!(
            self,
            Self::PopSymbol {
                is_definition: true,
                ..
            } | Self::PopScopedSymbol {
                is_definition: true,
                ..
            }
        )
    }

    /// Whether this is an exported scope node.
    #[must_use]
    pub const fn is_exported_scope(&self) -> bool {
        matches!(self, Self::Scope { exported: true })
    }

    /// Whether this is the singleton root node.
    #[must_use]
    pub const fn is_root(&self) -> bool {
        matches!(self, Self::Root)
    }

    /// Whether this is the singleton jump-to-scope node.
    #[must_use]
    pub const fn is_jump_to_scope(&self) -> bool {
        matches!(self, Self::JumpToScope)
    }
}

/// One stack-graph node with optional file and consumer metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StackNode<S, N> {
    file: Option<StackFileId>,
    kind: StackNodeKind<S>,
    payload: Option<N>,
}

impl<S, N> StackNode<S, N> {
    /// The file partition containing this node, or `None` for a singleton.
    #[must_use]
    pub const fn file(&self) -> Option<StackFileId> {
        self.file
    }

    /// Borrow the node's semantic kind.
    #[must_use]
    pub const fn kind(&self) -> &StackNodeKind<S> {
        &self.kind
    }

    /// Borrow consumer metadata, absent only for singleton nodes.
    #[must_use]
    pub const fn payload(&self) -> Option<&N> {
        self.payload.as_ref()
    }

    /// Mutably borrow consumer metadata, absent only for singleton nodes.
    pub const fn payload_mut(&mut self) -> Option<&mut N> {
        self.payload.as_mut()
    }
}

/// Consumer data and precedence carried by one stack-graph edge.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StackEdge<E> {
    precedence: i32,
    payload: E,
}

impl<E> StackEdge<E> {
    /// Edge precedence used for path shadowing.
    #[must_use]
    pub const fn precedence(&self) -> i32 {
        self.precedence
    }

    /// Borrow consumer-defined edge data.
    #[must_use]
    pub const fn payload(&self) -> &E {
        &self.payload
    }

    /// Mutably borrow consumer-defined edge data.
    pub const fn payload_mut(&mut self) -> &mut E {
        &mut self.payload
    }
}

/// A rejected stack-graph construction operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackGraphError {
    /// A file identity is outside this graph.
    UnknownFile(StackFileId),
    /// A node identity is outside this graph or belongs to a retired file generation.
    UnknownNode(StackNodeId),
    /// A scoped-symbol node names something other than an exported scope.
    ScopeNotExported(StackNodeId),
    /// A scoped-symbol node refers to an exported scope in another file.
    ScopeFromOtherFile {
        /// File receiving the scoped-symbol node.
        file: StackFileId,
        /// Exported scope owned by another file.
        scope: StackNodeId,
    },
    /// Explicit edges may not leave the jump-to-scope singleton.
    EdgeFromJumpToScope,
    /// A stored edge would cross directly between two file partitions.
    CrossFileEdge {
        /// Edge source.
        source: StackNodeId,
        /// Edge target.
        target: StackNodeId,
    },
}

impl core::fmt::Display for StackGraphError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownFile(file) => write!(formatter, "unknown stack-graph file {file}"),
            Self::UnknownNode(node) => write!(formatter, "unknown stack-graph node {node}"),
            Self::ScopeNotExported(scope) => {
                write!(
                    formatter,
                    "stack-graph node {scope} is not an exported scope"
                )
            }
            Self::ScopeFromOtherFile { file, scope } => write!(
                formatter,
                "stack-graph scope {scope} does not belong to file {file}"
            ),
            Self::EdgeFromJumpToScope => {
                formatter.write_str("explicit edges may not leave the jump-to-scope node")
            }
            Self::CrossFileEdge { source, target } => write!(
                formatter,
                "stack-graph edge {source} -> {target} crosses file partitions"
            ),
        }
    }
}

impl core::error::Error for StackGraphError {}

/// Owned, file-partitioned stack-graph storage.
///
/// Every ordinary node belongs to exactly one file. Stored edges stay within a
/// file or touch the root singleton; cross-file resolution therefore occurs by
/// joining paths at the root rather than by baking project knowledge into a
/// file's subgraph.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StackGraph<F = (), S = (), N = (), E = ()> {
    graph: StackStore<S, N, E>,
    files: Vec<F>,
    file_nodes: Vec<Vec<StackNodeId>>,
    root: StackNodeId,
    jump_to_scope: StackNodeId,
}

impl<F, S, N, E> StackGraph<F, S, N, E> {
    /// Create an empty graph containing the root and jump-to-scope singletons.
    #[must_use]
    pub fn new() -> Self {
        let mut graph = StackStore::tagged();
        let root = graph.add_node(StackNode {
            file: None,
            kind: StackNodeKind::Root,
            payload: None,
        });
        let jump_to_scope = graph.add_node(StackNode {
            file: None,
            kind: StackNodeKind::JumpToScope,
            payload: None,
        });
        Self {
            graph,
            files: Vec::new(),
            file_nodes: Vec::new(),
            root,
            jump_to_scope,
        }
    }

    /// Add a file partition and return its stable identity.
    pub fn add_file(&mut self, payload: F) -> StackFileId {
        let file = StackFileId::from_index(self.files.len());
        self.files.push(payload);
        self.file_nodes.push(Vec::new());
        file
    }

    /// Add an internal or exported scope node.
    ///
    /// # Errors
    ///
    /// Returns [`StackGraphError::UnknownFile`] for an invalid file identity.
    pub fn add_scope_node(
        &mut self,
        file: StackFileId,
        exported: bool,
        payload: N,
    ) -> Result<StackNodeId, StackGraphError> {
        self.add_local_node(file, StackNodeKind::Scope { exported }, payload)
    }

    /// Add an unscoped symbol-push node.
    ///
    /// # Errors
    ///
    /// Returns [`StackGraphError::UnknownFile`] for an invalid file identity.
    pub fn add_push_symbol_node(
        &mut self,
        file: StackFileId,
        symbol: S,
        is_reference: bool,
        payload: N,
    ) -> Result<StackNodeId, StackGraphError> {
        self.add_local_node(
            file,
            StackNodeKind::PushSymbol {
                symbol,
                is_reference,
            },
            payload,
        )
    }

    /// Add an unscoped symbol-pop node.
    ///
    /// # Errors
    ///
    /// Returns [`StackGraphError::UnknownFile`] for an invalid file identity.
    pub fn add_pop_symbol_node(
        &mut self,
        file: StackFileId,
        symbol: S,
        is_definition: bool,
        payload: N,
    ) -> Result<StackNodeId, StackGraphError> {
        self.add_local_node(
            file,
            StackNodeKind::PopSymbol {
                symbol,
                is_definition,
            },
            payload,
        )
    }

    /// Add a scoped symbol-push node.
    ///
    /// # Errors
    ///
    /// Returns an error when `file` is unknown, `scope` is unknown or not
    /// exported, or the scope belongs to a different file partition.
    pub fn add_push_scoped_symbol_node(
        &mut self,
        file: StackFileId,
        symbol: S,
        scope: StackNodeId,
        is_reference: bool,
        payload: N,
    ) -> Result<StackNodeId, StackGraphError> {
        self.require_file(file)?;
        self.require_exported_scope(scope)?;
        if self.node(scope).file() != Some(file) {
            return Err(StackGraphError::ScopeFromOtherFile { file, scope });
        }
        self.add_local_node(
            file,
            StackNodeKind::PushScopedSymbol {
                symbol,
                scope,
                is_reference,
            },
            payload,
        )
    }

    /// Add a scoped symbol-pop node.
    ///
    /// # Errors
    ///
    /// Returns [`StackGraphError::UnknownFile`] for an invalid file identity.
    pub fn add_pop_scoped_symbol_node(
        &mut self,
        file: StackFileId,
        symbol: S,
        is_definition: bool,
        payload: N,
    ) -> Result<StackNodeId, StackGraphError> {
        self.add_local_node(
            file,
            StackNodeKind::PopScopedSymbol {
                symbol,
                is_definition,
            },
            payload,
        )
    }

    /// Add a node that clears the scope stack.
    ///
    /// # Errors
    ///
    /// Returns [`StackGraphError::UnknownFile`] for an invalid file identity.
    pub fn add_drop_scopes_node(
        &mut self,
        file: StackFileId,
        payload: N,
    ) -> Result<StackNodeId, StackGraphError> {
        self.add_local_node(file, StackNodeKind::DropScopes, payload)
    }

    fn add_local_node(
        &mut self,
        file: StackFileId,
        kind: StackNodeKind<S>,
        payload: N,
    ) -> Result<StackNodeId, StackGraphError> {
        self.require_file(file)?;
        let node = self.graph.add_node(StackNode {
            file: Some(file),
            kind,
            payload: Some(payload),
        });
        self.file_nodes[file.index()].push(node);
        Ok(node)
    }

    /// Add an edge with precedence and consumer data.
    ///
    /// Direct edges between different file partitions are rejected. Either
    /// endpoint may instead be the root singleton, which is how independently
    /// constructed file subgraphs participate in a combined query.
    ///
    /// # Errors
    ///
    /// Returns an error for unknown endpoints, an edge leaving the
    /// jump-to-scope singleton, or a direct edge between different files.
    pub fn add_edge(
        &mut self,
        source: StackNodeId,
        target: StackNodeId,
        precedence: i32,
        payload: E,
    ) -> Result<StackEdgeId, StackGraphError> {
        let source_node = self.require_node(source)?;
        let target_node = self.require_node(target)?;
        if source_node.kind().is_jump_to_scope() {
            return Err(StackGraphError::EdgeFromJumpToScope);
        }
        if let (Some(source_file), Some(target_file)) = (source_node.file(), target_node.file())
            && source_file != target_file
        {
            return Err(StackGraphError::CrossFileEdge { source, target });
        }
        Ok(self.graph.add_edge(
            source,
            target,
            StackEdge {
                precedence,
                payload,
            },
        ))
    }

    /// Remove an edge while preserving every other identity.
    ///
    /// Returns whether the edge had been live.
    pub fn remove_edge(&mut self, edge: StackEdgeId) -> bool {
        self.graph.remove_edge(edge)
    }

    /// Remove every node and incident edge belonging to `file`.
    ///
    /// The file identity and payload remain available for rebuilding. Removed
    /// node and edge slots are never reused, and their payloads stay readable
    /// through [`node`](Self::node), so identities in diagnostics and caches
    /// never silently refer to newly allocated nodes. After adding the
    /// replacement nodes and edges, update a
    /// [`StackPartialPathDatabase`](super::StackPartialPathDatabase) with
    /// [`replace_file`](super::StackPartialPathDatabase::replace_file).
    ///
    /// # Errors
    ///
    /// Returns [`StackGraphError::UnknownFile`] for an invalid file identity.
    pub fn clear_file(&mut self, file: StackFileId) -> Result<Vec<StackNodeId>, StackGraphError> {
        self.require_file(file)?;
        let retired = core::mem::take(&mut self.file_nodes[file.index()]);
        for &node in &retired {
            self.graph.remove_node(node);
        }
        Ok(retired)
    }

    /// The singleton root node.
    #[must_use]
    pub const fn root_node(&self) -> StackNodeId {
        self.root
    }

    /// The singleton jump-to-scope node.
    #[must_use]
    pub const fn jump_to_scope_node(&self) -> StackNodeId {
        self.jump_to_scope
    }

    /// Borrow a file payload.
    ///
    /// # Panics
    ///
    /// Panics when `file` does not belong to this graph.
    #[must_use]
    pub fn file(&self, file: StackFileId) -> &F {
        &self.files[file.index()]
    }

    /// Mutably borrow a file payload.
    ///
    /// # Panics
    ///
    /// Panics when `file` does not belong to this graph.
    pub fn file_mut(&mut self, file: StackFileId) -> &mut F {
        &mut self.files[file.index()]
    }

    /// Live nodes in the current generation of `file`, in insertion order.
    ///
    /// # Panics
    ///
    /// Panics when `file` does not belong to this graph.
    #[must_use]
    pub fn file_nodes(&self, file: StackFileId) -> &[StackNodeId] {
        &self.file_nodes[file.index()]
    }

    /// Borrow one node.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this graph.
    #[must_use]
    pub fn node(&self, node: StackNodeId) -> &StackNode<S, N> {
        self.graph.node(node)
    }

    /// Mutably borrow one node.
    ///
    /// # Panics
    ///
    /// Panics when `node` does not belong to this graph.
    pub fn node_mut(&mut self, node: StackNodeId) -> &mut StackNode<S, N> {
        self.graph.node_mut(node)
    }

    /// Borrow one live edge.
    ///
    /// # Panics
    ///
    /// Panics when `edge` is out of range or was removed.
    #[must_use]
    pub fn edge(&self, edge: StackEdgeId) -> EdgeRef<'_, StackNodeId, StackEdgeId, StackEdge<E>> {
        assert!(
            self.graph.contains_edge(edge),
            "stack-graph edge {edge} has been removed"
        );
        let value = self.graph.edge(edge);
        EdgeRef::new(edge, value.source(), value.target(), value.payload())
    }

    /// Mutably borrow one live edge's data and precedence.
    ///
    /// # Panics
    ///
    /// Panics when `edge` is out of range or was removed.
    pub fn edge_mut(&mut self, edge: StackEdgeId) -> &mut StackEdge<E> {
        assert!(
            self.graph.contains_edge(edge),
            "stack-graph edge {edge} has been removed"
        );
        self.graph.edge_mut(edge).payload_mut()
    }

    /// Set the precedence used when comparing paths through `edge`.
    ///
    /// # Panics
    ///
    /// Panics when `edge` is out of range or was removed.
    pub fn set_edge_precedence(&mut self, edge: StackEdgeId, precedence: i32) {
        self.edge_mut(edge).precedence = precedence;
    }

    /// Iterate over every file identity in insertion order.
    pub fn file_ids(&self) -> impl ExactSizeIterator<Item = StackFileId> + '_ {
        (0..self.files.len()).map(StackFileId::from_index)
    }

    /// Iterate over the singleton nodes and current file-generation nodes.
    pub fn node_ids(&self) -> impl Iterator<Item = StackNodeId> + '_ {
        self.graph.node_ids()
    }

    /// Iterate over every live edge identity in insertion order.
    pub fn edge_ids(&self) -> impl Iterator<Item = StackEdgeId> + '_ {
        self.graph.edge_ids()
    }

    /// Return outgoing edge identities in insertion order.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn outgoing(&self, node: StackNodeId) -> impl Iterator<Item = StackEdgeId> + '_ {
        self.graph.outgoing(node)
    }

    /// Return incoming edge identities in insertion order.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn incoming(&self, node: StackNodeId) -> impl Iterator<Item = StackEdgeId> + '_ {
        self.graph.incoming(node)
    }

    /// Return the number of file partitions.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    /// Return the number of live nodes, including the two singletons.
    #[must_use]
    pub const fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// An exclusive upper bound on every live [`StackNodeId`] index.
    ///
    /// The right size for a node-indexed side table; it exceeds
    /// [`node_count`](Self::node_count) once a file has been cleared.
    #[must_use]
    pub fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    /// Return whether `node` belongs to the current graph generation.
    #[must_use]
    pub fn contains_node(&self, node: StackNodeId) -> bool {
        self.graph.contains_node(node)
    }

    /// Return the number of live edges.
    #[must_use]
    pub const fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// An exclusive upper bound on every live [`StackEdgeId`] index.
    #[must_use]
    pub fn edge_bound(&self) -> usize {
        self.graph.edge_bound()
    }

    /// Return whether `edge` identifies a live edge in this graph.
    #[must_use]
    pub fn contains_edge(&self, edge: StackEdgeId) -> bool {
        self.graph.contains_edge(edge)
    }

    fn require_file(&self, file: StackFileId) -> Result<(), StackGraphError> {
        if file.index() < self.files.len() {
            Ok(())
        } else {
            Err(StackGraphError::UnknownFile(file))
        }
    }

    fn require_node(&self, node: StackNodeId) -> Result<&StackNode<S, N>, StackGraphError> {
        if self.contains_node(node) {
            Ok(self.node(node))
        } else {
            Err(StackGraphError::UnknownNode(node))
        }
    }

    fn require_exported_scope(&self, scope: StackNodeId) -> Result<(), StackGraphError> {
        if self.require_node(scope)?.kind().is_exported_scope() {
            Ok(())
        } else {
            Err(StackGraphError::ScopeNotExported(scope))
        }
    }
}

impl<F, S, N, E> Default for StackGraph<F, S, N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, S, N, E> GraphView for StackGraph<F, S, N, E> {
    type NodeId = StackNodeId;

    fn node_bound(&self) -> usize {
        StackGraph::node_bound(self)
    }

    fn node_ids(&self) -> impl Iterator<Item = StackNodeId> + '_ {
        StackGraph::node_ids(self)
    }

    fn successors(&self, node: StackNodeId) -> impl Iterator<Item = StackNodeId> + '_ {
        self.graph.successors(node)
    }

    fn predecessors(&self, node: StackNodeId) -> impl Iterator<Item = StackNodeId> + '_ {
        self.graph.predecessors(node)
    }
}

impl<F, S, N, E> EdgeView for StackGraph<F, S, N, E> {
    type EdgeId = StackEdgeId;
    type EdgeData = StackEdge<E>;

    fn edge_bound(&self) -> usize {
        StackGraph::edge_bound(self)
    }

    fn edge_ids(&self) -> impl Iterator<Item = StackEdgeId> + '_ {
        StackGraph::edge_ids(self)
    }

    fn outgoing(&self, node: StackNodeId) -> impl Iterator<Item = StackEdgeId> + '_ {
        StackGraph::outgoing(self, node)
    }

    fn incoming(&self, node: StackNodeId) -> impl Iterator<Item = StackEdgeId> + '_ {
        StackGraph::incoming(self, node)
    }

    fn edge(&self, edge: StackEdgeId) -> EdgeRef<'_, StackNodeId, StackEdgeId, StackEdge<E>> {
        StackGraph::edge(self, edge)
    }
}

impl<F, S, N, E> Index<StackNodeId> for StackGraph<F, S, N, E> {
    type Output = StackNode<S, N>;

    fn index(&self, node: StackNodeId) -> &Self::Output {
        self.node(node)
    }
}
