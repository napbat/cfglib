//! Owned stack-graph storage and construction invariants.

extern crate alloc;

use alloc::vec::Vec;
use core::ops::Index;

use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::store::{Graph, Id, IdTag};
use crate::graph::view::GraphView;
use crate::identity::define_dense_id;

#[cfg(feature = "serde")]
mod stored;

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

define_dense_id! {
    /// Dense identity of a file partition in a [`StackGraph`]: an index into
    /// the graph's file table, which the store does not mint.
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct StackFileId(u32);
    display = "f";
    /// Construct a file identity from a dense zero-based index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    from_index = "stack-graph file index exceeds u32::MAX";
}

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

    /// Mutably borrow the edge precedence used for path shadowing.
    pub const fn precedence_mut(&mut self) -> &mut i32 {
        &mut self.precedence
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

/// A construction operation that would break a rule of the stack-graph
/// structure.
///
/// An identity that is not part of the graph is a caller bug that the
/// builders report by panicking, as the scope-graph builders do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackGraphError {
    /// A file node of a singleton kind: the root and the jump-to-scope node
    /// exist once, outside every file.
    SingletonKind,
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
            Self::SingletonKind => {
                formatter.write_str("the root and the jump-to-scope node belong to no file")
            }
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
///
/// A deserialized graph is checked before it is returned, against every rule
/// the builders enforce: the two singletons, one file and one payload for each
/// other node (retired nodes included), the per-file node lists, the exported
/// scope of each scoped push, and the edges. A document that breaks one is
/// rejected, because queries index the node and file tables by those
/// identities.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(
        bound(
            serialize = "F: serde::Serialize, S: serde::Serialize, N: serde::Serialize, \
                         E: serde::Serialize",
            deserialize = "F: serde::Deserialize<'de>, S: serde::Deserialize<'de>, \
                           N: serde::Deserialize<'de>, E: serde::Deserialize<'de>"
        ),
        try_from = "stored::Stored<F, S, N, E>"
    )
)]
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

    /// Add a node of `kind` to `file` and return its stable identity.
    ///
    /// A [`PushScopedSymbol`](StackNodeKind::PushScopedSymbol) must name a
    /// live exported scope of the same file.
    ///
    /// # Errors
    ///
    /// Returns [`StackGraphError::SingletonKind`] for a root or jump-to-scope
    /// kind, and [`StackGraphError::ScopeNotExported`] or
    /// [`StackGraphError::ScopeFromOtherFile`] for a scoped push whose scope
    /// is not an exported scope of `file`.
    ///
    /// # Panics
    ///
    /// Panics when `file` does not belong to this graph, or when a scoped
    /// push names a node that is out of range or retired.
    pub fn add_node(
        &mut self,
        file: StackFileId,
        kind: StackNodeKind<S>,
        payload: N,
    ) -> Result<StackNodeId, StackGraphError> {
        self.require_file(file);
        match &kind {
            StackNodeKind::Root | StackNodeKind::JumpToScope => {
                return Err(StackGraphError::SingletonKind);
            }
            StackNodeKind::PushScopedSymbol { scope, .. } => {
                require_attachable_scope(file, *scope, self.require_node(*scope))?;
            }
            StackNodeKind::Scope { .. }
            | StackNodeKind::PushSymbol { .. }
            | StackNodeKind::PopSymbol { .. }
            | StackNodeKind::PopScopedSymbol { .. }
            | StackNodeKind::DropScopes => {}
        }
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
    /// Returns [`StackGraphError::EdgeFromJumpToScope`] for an edge leaving
    /// the jump-to-scope singleton, and [`StackGraphError::CrossFileEdge`]
    /// for a direct edge between different files.
    ///
    /// # Panics
    ///
    /// Panics when an endpoint is out of range or retired.
    pub fn add_edge(
        &mut self,
        source: StackNodeId,
        target: StackNodeId,
        precedence: i32,
        payload: E,
    ) -> Result<StackEdgeId, StackGraphError> {
        require_joinable(
            source,
            self.require_node(source),
            target,
            self.require_node(target),
        )?;
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

    /// Remove every node and incident edge belonging to `file`, and return
    /// the retired nodes.
    ///
    /// The file identity and payload remain available for rebuilding. Removed
    /// node and edge slots are never reused, and their payloads stay readable
    /// through [`node`](Self::node), so identities in diagnostics and caches
    /// never silently refer to newly allocated nodes. After adding the
    /// replacement nodes and edges, update a
    /// [`StackPartialPathDatabase`](super::StackPartialPathDatabase) with
    /// [`replace_file`](super::StackPartialPathDatabase::replace_file).
    ///
    /// # Panics
    ///
    /// Panics when `file` does not belong to this graph.
    pub fn clear_file(&mut self, file: StackFileId) -> Vec<StackNodeId> {
        self.require_file(file);
        let retired = core::mem::take(&mut self.file_nodes[file.index()]);
        for &node in &retired {
            self.graph.remove_node(node);
        }
        retired
    }

    /// Borrow the store of the nodes and edges.
    ///
    /// The stack graph is a [`GraphView`] and an [`EdgeView`] through the
    /// store; its counts, bounds, and liveness tests are the store's. The
    /// bound exceeds the node count once a file has been cleared, and a
    /// retired node stays readable.
    #[must_use]
    pub const fn graph(&self) -> &Graph<StackNode<S, N>, StackEdge<E>, StackNodeTag, StackEdgeTag> {
        &self.graph
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

    /// Borrow one node, live or retired.
    ///
    /// # Panics
    ///
    /// Panics when `node` is out of range.
    #[must_use]
    pub fn node(&self, node: StackNodeId) -> &StackNode<S, N> {
        self.graph.node(node)
    }

    /// Mutably borrow one node.
    ///
    /// # Panics
    ///
    /// Panics when `node` is out of range.
    pub fn node_mut(&mut self, node: StackNodeId) -> &mut StackNode<S, N> {
        self.graph.node_mut(node)
    }

    /// Mutably borrow one live edge's data and precedence.
    ///
    /// # Panics
    ///
    /// Panics when `edge` is out of range or was removed.
    pub fn edge_mut(&mut self, edge: StackEdgeId) -> &mut StackEdge<E> {
        assert!(self.graph.contains_edge(edge), "edge has been removed");
        self.graph.edge_mut(edge).payload_mut()
    }

    /// Iterate over every file identity in insertion order.
    pub fn file_ids(&self) -> impl ExactSizeIterator<Item = StackFileId> + '_ {
        (0..self.files.len()).map(StackFileId::from_index)
    }

    /// Iterate over every source reference node in node order: the seeds of
    /// a whole-graph resolution.
    pub(super) fn reference_nodes(&self) -> impl Iterator<Item = StackNodeId> + '_ {
        self.graph
            .node_ids()
            .filter(|&node| self.node(node).kind().is_reference())
    }

    /// Return the number of file partitions.
    #[must_use]
    pub fn file_count(&self) -> usize {
        self.files.len()
    }

    #[track_caller]
    fn require_file(&self, file: StackFileId) {
        assert!(
            file.index() < self.files.len(),
            "stack-graph file {file} is out of range"
        );
    }

    #[track_caller]
    fn require_node(&self, node: StackNodeId) -> &StackNode<S, N> {
        assert!(
            self.graph.contains_node(node),
            "stack-graph node {node} is out of range or retired"
        );
        self.node(node)
    }
}

/// The rule a scoped push obeys: the scope `node` it names is an exported
/// scope of the push's own `file`.
///
/// Whether `node` is still live is the caller's question; a retired push still
/// names the scope it was built with.
fn require_attachable_scope<S, N>(
    file: StackFileId,
    scope: StackNodeId,
    node: &StackNode<S, N>,
) -> Result<(), StackGraphError> {
    if !node.kind().is_exported_scope() {
        return Err(StackGraphError::ScopeNotExported(scope));
    }
    if node.file() != Some(file) {
        return Err(StackGraphError::ScopeFromOtherFile { file, scope });
    }
    Ok(())
}

/// The rule an edge obeys: it does not leave the jump-to-scope singleton and
/// does not join two different files directly.
fn require_joinable<S, N>(
    source: StackNodeId,
    source_node: &StackNode<S, N>,
    target: StackNodeId,
    target_node: &StackNode<S, N>,
) -> Result<(), StackGraphError> {
    if source_node.kind().is_jump_to_scope() {
        return Err(StackGraphError::EdgeFromJumpToScope);
    }
    if let (Some(source_file), Some(target_file)) = (source_node.file(), target_node.file())
        && source_file != target_file
    {
        return Err(StackGraphError::CrossFileEdge { source, target });
    }
    Ok(())
}

impl<F, S, N, E> Default for StackGraph<F, S, N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F, S, N, E> GraphView for StackGraph<F, S, N, E> {
    type NodeId = StackNodeId;

    fn node_bound(&self) -> usize {
        self.graph.node_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = StackNodeId> + '_ {
        self.graph.node_ids()
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
        self.graph.edge_bound()
    }

    fn edge_ids(&self) -> impl Iterator<Item = StackEdgeId> + '_ {
        self.graph.edge_ids()
    }

    fn outgoing(&self, node: StackNodeId) -> impl Iterator<Item = StackEdgeId> + '_ {
        self.graph.outgoing(node)
    }

    fn incoming(&self, node: StackNodeId) -> impl Iterator<Item = StackEdgeId> + '_ {
        self.graph.incoming(node)
    }

    /// # Panics
    ///
    /// Panics when `edge` is out of range or was removed.
    fn edge(&self, edge: StackEdgeId) -> EdgeRef<'_, StackNodeId, StackEdgeId, StackEdge<E>> {
        EdgeView::edge(&self.graph, edge)
    }
}

impl<F, S, N, E> Index<StackNodeId> for StackGraph<F, S, N, E> {
    type Output = StackNode<S, N>;

    fn index(&self, node: StackNodeId) -> &Self::Output {
        self.node(node)
    }
}
