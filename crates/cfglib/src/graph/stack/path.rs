//! Concrete stack-graph path state and node-transition semantics.

use smallvec::SmallVec;

use super::storage::{StackEdgeId, StackGraph, StackNodeId, StackNodeKind};

/// A symbol on a stack-graph path's symbol stack.
///
/// `scopes == None` is an unscoped symbol. `Some(scopes)` is a scoped symbol
/// whose list is nonempty; the final element is the top of that attached scope
/// stack.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackScopedSymbol<S> {
    symbol: S,
    scopes: Option<SmallVec<[StackNodeId; 4]>>,
}

impl<S> StackScopedSymbol<S> {
    /// Borrow the consumer-defined symbol.
    #[must_use]
    pub const fn symbol(&self) -> &S {
        &self.symbol
    }

    /// Attached exported scopes, from bottom to top, if this is scoped.
    #[must_use]
    pub fn scopes(&self) -> Option<&[StackNodeId]> {
        self.scopes.as_deref()
    }

    /// Whether this symbol has an attached scope list.
    #[must_use]
    pub const fn is_scoped(&self) -> bool {
        self.scopes.is_some()
    }
}

/// One stored or virtual transition in a [`StackPath`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum StackPathStep {
    /// A stored graph edge.
    Edge(StackEdgeId),
    /// The virtual edge produced by resolving the jump-to-scope singleton.
    Jump {
        /// Singleton jump node.
        source: StackNodeId,
        /// Exported scope popped from the scope stack.
        target: StackNodeId,
    },
}

/// A rejected extension of a concrete stack-graph path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackPathError {
    /// The requested node identity is outside the graph or has been retired.
    UnknownNode(StackNodeId),
    /// The requested edge identity is outside the graph or was removed.
    UnknownEdge(StackEdgeId),
    /// An initial path was requested for a node that is not a source reference.
    NotReference(StackNodeId),
    /// The appended edge does not leave the path's current end node.
    IncorrectSource {
        /// Current path end.
        expected: StackNodeId,
        /// Edge source.
        actual: StackNodeId,
    },
    /// A pop node encountered an empty symbol stack.
    EmptySymbolStack(StackNodeId),
    /// A pop node's symbol did not match the stack top.
    IncorrectPoppedSymbol(StackNodeId),
    /// An unscoped pop encountered a symbol with attached scopes.
    UnexpectedAttachedScopes(StackNodeId),
    /// A scoped pop encountered an unscoped symbol.
    MissingAttachedScopes(StackNodeId),
    /// A jump node encountered an empty scope stack.
    EmptyScopeStack(StackNodeId),
    /// A scoped push refers to a node that is no longer an exported scope.
    UnknownAttachedScope(StackNodeId),
}

impl core::fmt::Display for StackPathError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownNode(node) => write!(formatter, "unknown stack-graph node {node}"),
            Self::UnknownEdge(edge) => write!(formatter, "unknown stack-graph edge {edge}"),
            Self::NotReference(node) => {
                write!(formatter, "stack-graph node {node} is not a reference")
            }
            Self::IncorrectSource { expected, actual } => write!(
                formatter,
                "stack-graph path ends at {expected}, but the edge starts at {actual}"
            ),
            Self::EmptySymbolStack(node) => {
                write!(
                    formatter,
                    "stack-graph pop node {node} found an empty symbol stack"
                )
            }
            Self::IncorrectPoppedSymbol(node) => write!(
                formatter,
                "stack-graph pop node {node} does not match the symbol stack"
            ),
            Self::UnexpectedAttachedScopes(node) => write!(
                formatter,
                "stack-graph unscoped pop node {node} found attached scopes"
            ),
            Self::MissingAttachedScopes(node) => write!(
                formatter,
                "stack-graph scoped pop node {node} found no attached scopes"
            ),
            Self::EmptyScopeStack(node) => write!(
                formatter,
                "stack-graph jump node {node} found an empty scope stack"
            ),
            Self::UnknownAttachedScope(scope) => write!(
                formatter,
                "stack-graph scoped symbol refers to non-exported scope {scope}"
            ),
        }
    }
}

impl core::error::Error for StackPathError {}

/// A concrete, well-formed path through a stack graph.
///
/// The path records both stacks after processing its end node. Symbol and
/// scope stacks store their tops at the end of their slices. A complete path
/// starts at a reference, ends at a definition, and leaves both stacks empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackPath<S> {
    start: StackNodeId,
    end: StackNodeId,
    symbol_stack: SmallVec<[StackScopedSymbol<S>; 4]>,
    scope_stack: SmallVec<[StackNodeId; 4]>,
    steps: SmallVec<[StackPathStep; 8]>,
    edge_count: usize,
}

impl<S: Clone + Eq> StackPath<S> {
    /// Create the initial path for a source reference node.
    ///
    /// The reference node itself seeds the symbol stack, including the
    /// attached scope for a scoped-symbol reference.
    ///
    /// # Errors
    ///
    /// Returns [`StackPathError::UnknownNode`] when `reference` is outside the
    /// graph, or [`StackPathError::NotReference`] when it is not a source
    /// reference node.
    pub fn from_reference<F, N, E>(
        graph: &StackGraph<F, S, N, E>,
        reference: StackNodeId,
    ) -> Result<Self, StackPathError> {
        if !graph.contains_node(reference) {
            return Err(StackPathError::UnknownNode(reference));
        }
        if !graph.node(reference).kind().is_reference() {
            return Err(StackPathError::NotReference(reference));
        }
        let mut path = Self {
            start: reference,
            end: reference,
            symbol_stack: SmallVec::new(),
            scope_stack: SmallVec::new(),
            steps: SmallVec::new(),
            edge_count: 0,
        };
        path.apply_node(graph, reference)?;
        Ok(path)
    }

    /// Append one stored edge and apply its target node's stack semantics.
    ///
    /// Entering the jump-to-scope singleton immediately adds a virtual jump to
    /// the exported scope popped from the scope stack. On error, this path is
    /// left unchanged.
    ///
    /// # Errors
    ///
    /// Returns a structured [`StackPathError`] when the edge is disconnected
    /// from this path or its target node cannot operate on the current stacks.
    pub fn append<F, N, E>(
        &mut self,
        graph: &StackGraph<F, S, N, E>,
        edge: StackEdgeId,
    ) -> Result<(), StackPathError> {
        let next = self.clone().append_owned(graph, edge)?;
        *self = next;
        Ok(())
    }

    pub(super) fn append_owned<F, N, E>(
        mut self,
        graph: &StackGraph<F, S, N, E>,
        edge: StackEdgeId,
    ) -> Result<Self, StackPathError> {
        if !graph.contains_edge(edge) {
            return Err(StackPathError::UnknownEdge(edge));
        }
        let stored = graph.edge(edge);
        if stored.source() != self.end {
            return Err(StackPathError::IncorrectSource {
                expected: self.end,
                actual: stored.source(),
            });
        }

        let target = stored.target();
        self.apply_node(graph, target)?;
        self.end = target;
        self.steps.push(StackPathStep::Edge(edge));
        self.edge_count += 1;
        if graph.node(self.end).kind().is_jump_to_scope() {
            let scope = self
                .scope_stack
                .pop()
                .ok_or(StackPathError::EmptyScopeStack(self.end))?;
            if !graph.contains_node(scope) || !graph.node(scope).kind().is_exported_scope() {
                return Err(StackPathError::UnknownAttachedScope(scope));
            }
            let source = self.end;
            self.end = scope;
            self.steps.push(StackPathStep::Jump {
                source,
                target: scope,
            });
        }
        Ok(self)
    }

    fn apply_node<F, N, E>(
        &mut self,
        graph: &StackGraph<F, S, N, E>,
        node: StackNodeId,
    ) -> Result<(), StackPathError> {
        match graph.node(node).kind() {
            StackNodeKind::PushSymbol { symbol, .. } => {
                self.symbol_stack.push(StackScopedSymbol {
                    symbol: symbol.clone(),
                    scopes: None,
                });
            }
            StackNodeKind::PushScopedSymbol { symbol, scope, .. } => {
                if !graph.contains_node(*scope) || !graph.node(*scope).kind().is_exported_scope() {
                    return Err(StackPathError::UnknownAttachedScope(*scope));
                }
                let mut scopes = self.scope_stack.clone();
                scopes.push(*scope);
                self.symbol_stack.push(StackScopedSymbol {
                    symbol: symbol.clone(),
                    scopes: Some(scopes),
                });
            }
            StackNodeKind::PopSymbol { symbol, .. } => {
                let top = self
                    .symbol_stack
                    .last()
                    .ok_or(StackPathError::EmptySymbolStack(node))?;
                if top.symbol() != symbol {
                    return Err(StackPathError::IncorrectPoppedSymbol(node));
                }
                if top.is_scoped() {
                    return Err(StackPathError::UnexpectedAttachedScopes(node));
                }
                let _ = self.symbol_stack.pop();
            }
            StackNodeKind::PopScopedSymbol { symbol, .. } => {
                let top = self
                    .symbol_stack
                    .last()
                    .ok_or(StackPathError::EmptySymbolStack(node))?;
                if top.symbol() != symbol {
                    return Err(StackPathError::IncorrectPoppedSymbol(node));
                }
                if !top.is_scoped() {
                    return Err(StackPathError::MissingAttachedScopes(node));
                }
                let top = self
                    .symbol_stack
                    .pop()
                    .expect("the checked symbol-stack top remains present");
                self.scope_stack = top
                    .scopes
                    .expect("a checked scoped symbol retains its attached scopes");
            }
            StackNodeKind::DropScopes => self.scope_stack.clear(),
            StackNodeKind::Root | StackNodeKind::Scope { .. } | StackNodeKind::JumpToScope => {}
        }
        Ok(())
    }

    /// Whether this path is a complete reference-to-definition binding.
    #[must_use]
    pub fn is_complete<F, N, E>(&self, graph: &StackGraph<F, S, N, E>) -> bool {
        graph.node(self.start).kind().is_reference()
            && graph.node(self.end).kind().is_definition()
            && self.symbol_stack.is_empty()
            && self.scope_stack.is_empty()
    }

    /// Whether this path shadows `other` through edge precedence.
    ///
    /// Corresponding steps must leave the same source nodes. At least one
    /// corresponding edge in `self` must then have strictly greater precedence.
    /// The relation is intentionally not commutative.
    #[must_use]
    pub fn shadows<F, N, E>(&self, graph: &StackGraph<F, S, N, E>, other: &Self) -> bool {
        for (&left, &right) in self.steps.iter().zip(&other.steps) {
            let (left_source, left_precedence) = step_key(graph, left);
            let (right_source, right_precedence) = step_key(graph, right);
            if left_source != right_source {
                return false;
            }
            if left_precedence > right_precedence {
                return true;
            }
        }
        false
    }

    /// Whether this path already contains stored `edge`.
    #[must_use]
    pub fn contains_edge(&self, edge: StackEdgeId) -> bool {
        self.steps.contains(&StackPathStep::Edge(edge))
    }

    /// The source reference node.
    #[must_use]
    pub const fn start(&self) -> StackNodeId {
        self.start
    }

    /// The current end node after resolving any jump.
    #[must_use]
    pub const fn end(&self) -> StackNodeId {
        self.end
    }

    /// The definition node when this path is complete.
    #[must_use]
    pub fn definition<F, N, E>(&self, graph: &StackGraph<F, S, N, E>) -> Option<StackNodeId> {
        self.is_complete(graph).then_some(self.end)
    }

    /// The symbol stack from bottom to top.
    #[must_use]
    pub fn symbol_stack(&self) -> &[StackScopedSymbol<S>] {
        &self.symbol_stack
    }

    /// The scope stack from bottom to top.
    #[must_use]
    pub fn scope_stack(&self) -> &[StackNodeId] {
        &self.scope_stack
    }

    /// Stored and virtual transitions in traversal order.
    #[must_use]
    pub fn steps(&self) -> &[StackPathStep] {
        &self.steps
    }

    /// Number of stored graph edges in the path.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.edge_count
    }

    /// Whether this path contains no stored graph edges.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.edge_count == 0
    }
}

fn step_key<F, S, N, E>(graph: &StackGraph<F, S, N, E>, step: StackPathStep) -> (StackNodeId, i32) {
    match step {
        StackPathStep::Edge(edge) => {
            let edge = graph.edge(edge);
            (edge.source(), edge.data().precedence())
        }
        StackPathStep::Jump { source, .. } => (source, 0),
    }
}
