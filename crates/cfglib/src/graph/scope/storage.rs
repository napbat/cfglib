//! Owned scope-graph storage and dense identities.

extern crate alloc;

use alloc::vec::Vec;
use core::ops::Index;

use crate::graph::edge_view::{EdgeRef, EdgeView};
use crate::graph::store::{Graph, Id, IdTag};
use crate::graph::view::GraphView;
use crate::identity::define_dense_id;

/// Tag marking an identity that addresses a scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeTag;

impl IdTag for ScopeTag {
    const PREFIX: &'static str = "s";
}

/// Tag marking an identity that addresses a labeled scope-graph edge.
///
/// A scope edge is an edge, but it is not a control-flow edge: keeping its
/// own tag is what stops a [`ScopeEdgeId`] and a CFG [`EdgeId`](crate::EdgeId)
/// from being interchanged, and it costs nothing at run time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ScopeEdgeTag;

impl IdTag for ScopeEdgeTag {
    const PREFIX: &'static str = "se";
}

/// Dense identity of a scope in a [`ScopeGraph`].
pub type ScopeId = Id<ScopeTag>;

/// Stable identity of a labeled edge in a [`ScopeGraph`].
pub type ScopeEdgeId = Id<ScopeEdgeTag>;

/// The store a [`ScopeGraph`] is built on.
type ScopeStore<S, L> = Graph<Scope<S>, L, ScopeTag, ScopeEdgeTag>;

define_dense_id! {
    /// Stable identity of relation-tagged data in a [`ScopeGraph`].
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ScopeDatumId(u32);
    display = "d";
    /// Construct a datum identity from a dense zero-based index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    from_index = "scope datum index exceeds u32::MAX";
}

define_dense_id! {
    /// Stable identity of a reference in a [`ScopeGraph`].
    #[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
    pub struct ScopeReferenceId(u32);
    display = "r";
    /// Construct a reference identity from a dense zero-based index.
    ///
    /// # Panics
    ///
    /// Panics when `index` exceeds `u32::MAX`.
    from_index = "scope reference index exceeds u32::MAX";
}

/// One scope carrying consumer-defined metadata.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Scope<S> {
    payload: S,
}

impl<S> Scope<S> {
    /// Borrow the consumer-defined scope payload.
    #[must_use]
    pub const fn payload(&self) -> &S {
        &self.payload
    }

    /// Mutably borrow the consumer-defined scope payload.
    pub const fn payload_mut(&mut self) -> &mut S {
        &mut self.payload
    }

    /// Consume the scope and return its payload.
    #[must_use]
    pub fn into_payload(self) -> S {
        self.payload
    }
}

/// One relation-tagged datum owned by a scope.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScopeDatum<R, D> {
    scope: ScopeId,
    relation: R,
    data: D,
}

impl<R, D> ScopeDatum<R, D> {
    /// The scope that owns this datum.
    #[must_use]
    pub const fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Borrow the consumer-defined relation tag.
    #[must_use]
    pub const fn relation(&self) -> &R {
        &self.relation
    }

    /// Borrow the consumer-defined data.
    #[must_use]
    pub const fn data(&self) -> &D {
        &self.data
    }

    /// Mutably borrow the consumer-defined data.
    pub const fn data_mut(&mut self) -> &mut D {
        &mut self.data
    }

    /// Consume the datum and return its relation and data.
    #[must_use]
    pub fn into_parts(self) -> (R, D) {
        (self.relation, self.data)
    }
}

/// One reference whose lookup starts in a scope.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScopeReference<Q> {
    scope: ScopeId,
    data: Q,
}

impl<Q> ScopeReference<Q> {
    /// The scope where lookup starts.
    #[must_use]
    pub const fn scope(&self) -> ScopeId {
        self.scope
    }

    /// Borrow the consumer-defined reference data.
    #[must_use]
    pub const fn data(&self) -> &Q {
        &self.data
    }

    /// Mutably borrow the consumer-defined reference data.
    pub const fn data_mut(&mut self) -> &mut Q {
        &mut self.data
    }

    /// Consume the reference and return its data.
    #[must_use]
    pub fn into_data(self) -> Q {
        self.data
    }
}

/// Owned, language-parametric scope-graph storage.
///
/// Edges are directed from the scope where a query is running to a scope whose
/// data becomes reachable. Their labels remain entirely consumer-defined.
/// Data carry a separate relation tag so one graph can hold value, type,
/// member, label, macro, or other namespaces without parallel graph stores.
#[derive(Debug, Clone, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ScopeGraph<S = (), L = (), R = (), D = (), Q = ()> {
    graph: ScopeStore<S, L>,
    data: Vec<ScopeDatum<R, D>>,
    references: Vec<ScopeReference<Q>>,
    scope_data: Vec<Vec<ScopeDatumId>>,
    scope_references: Vec<Vec<ScopeReferenceId>>,
}

impl<S, L, R, D, Q> ScopeGraph<S, L, R, D, Q> {
    /// Create an empty scope graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: ScopeStore::tagged(),
            data: Vec::new(),
            references: Vec::new(),
            scope_data: Vec::new(),
            scope_references: Vec::new(),
        }
    }

    /// Create an empty scope graph with the requested capacities.
    #[must_use]
    pub fn with_capacity(scopes: usize, edges: usize, data: usize, references: usize) -> Self {
        Self {
            graph: ScopeStore::tagged_with_capacity(scopes, edges),
            data: Vec::with_capacity(data),
            references: Vec::with_capacity(references),
            scope_data: Vec::with_capacity(scopes),
            scope_references: Vec::with_capacity(scopes),
        }
    }

    /// Add a scope and return its stable identity.
    pub fn add_scope(&mut self, payload: S) -> ScopeId {
        let id = self.graph.add_node(Scope { payload });
        self.scope_data.push(Vec::new());
        self.scope_references.push(Vec::new());
        id
    }

    /// Add a labeled reachability edge and return its stable identity.
    ///
    /// Parallel and cyclic edges are valid. Resolution paths themselves are
    /// cycle-free, so cyclic import or parent relations remain total.
    ///
    /// # Panics
    ///
    /// Panics when either endpoint does not belong to this graph.
    pub fn add_edge(&mut self, source: ScopeId, target: ScopeId, label: L) -> ScopeEdgeId {
        self.graph.add_edge(source, target, label)
    }

    /// Add relation-tagged data to `scope` and return its stable identity.
    ///
    /// # Panics
    ///
    /// Panics when `scope` does not belong to this graph.
    pub fn add_datum(&mut self, scope: ScopeId, relation: R, data: D) -> ScopeDatumId {
        assert!(scope.index() < self.scope_bound(), "scope is out of range");
        let id = ScopeDatumId::from_index(self.data.len());
        self.data.push(ScopeDatum {
            scope,
            relation,
            data,
        });
        self.scope_data[scope.index()].push(id);
        id
    }

    /// Add a reference whose lookup begins in `scope`.
    ///
    /// # Panics
    ///
    /// Panics when `scope` does not belong to this graph.
    pub fn add_reference(&mut self, scope: ScopeId, data: Q) -> ScopeReferenceId {
        assert!(scope.index() < self.scope_bound(), "scope is out of range");
        let id = ScopeReferenceId::from_index(self.references.len());
        self.references.push(ScopeReference { scope, data });
        self.scope_references[scope.index()].push(id);
        id
    }

    /// Borrow a scope.
    ///
    /// # Panics
    ///
    /// Panics when `scope` does not belong to this graph.
    #[must_use]
    pub fn scope(&self, scope: ScopeId) -> &Scope<S> {
        self.graph.node(scope)
    }

    /// Mutably borrow a scope.
    ///
    /// # Panics
    ///
    /// Panics when `scope` does not belong to this graph.
    pub fn scope_mut(&mut self, scope: ScopeId) -> &mut Scope<S> {
        self.graph.node_mut(scope)
    }

    /// Borrow one live edge.
    ///
    /// # Panics
    ///
    /// Panics when `edge` is out of range or was removed.
    #[must_use]
    pub fn edge(&self, edge: ScopeEdgeId) -> EdgeRef<'_, ScopeId, ScopeEdgeId, L> {
        assert!(
            self.graph.contains_edge(edge),
            "scope edge {edge} has been removed"
        );
        let value = self.graph.edge(edge);
        EdgeRef::new(edge, value.source(), value.target(), value.payload())
    }

    /// Mutably borrow a live edge's label.
    ///
    /// # Panics
    ///
    /// Panics when `edge` is out of range or was removed.
    pub fn edge_label_mut(&mut self, edge: ScopeEdgeId) -> &mut L {
        assert!(
            self.graph.contains_edge(edge),
            "scope edge {edge} has been removed"
        );
        self.graph.edge_mut(edge).payload_mut()
    }

    /// Remove an edge while preserving every other identity.
    ///
    /// Returns whether the edge had been live.
    pub fn remove_edge(&mut self, edge: ScopeEdgeId) -> bool {
        self.graph.remove_edge(edge)
    }

    /// Borrow relation-tagged data.
    ///
    /// # Panics
    ///
    /// Panics when `datum` does not belong to this graph.
    #[must_use]
    pub fn datum(&self, datum: ScopeDatumId) -> &ScopeDatum<R, D> {
        &self.data[datum.index()]
    }

    /// Mutably borrow relation-tagged data.
    ///
    /// # Panics
    ///
    /// Panics when `datum` does not belong to this graph.
    pub fn datum_mut(&mut self, datum: ScopeDatumId) -> &mut ScopeDatum<R, D> {
        &mut self.data[datum.index()]
    }

    /// Borrow a reference.
    ///
    /// # Panics
    ///
    /// Panics when `reference` does not belong to this graph.
    #[must_use]
    pub fn reference(&self, reference: ScopeReferenceId) -> &ScopeReference<Q> {
        &self.references[reference.index()]
    }

    /// Mutably borrow a reference.
    ///
    /// # Panics
    ///
    /// Panics when `reference` does not belong to this graph.
    pub fn reference_mut(&mut self, reference: ScopeReferenceId) -> &mut ScopeReference<Q> {
        &mut self.references[reference.index()]
    }

    /// Data identities owned by `scope`, in insertion order.
    ///
    /// # Panics
    ///
    /// Panics when `scope` does not belong to this graph.
    #[must_use]
    pub fn scope_data(&self, scope: ScopeId) -> &[ScopeDatumId] {
        &self.scope_data[scope.index()]
    }

    /// Reference identities owned by `scope`, in insertion order.
    ///
    /// # Panics
    ///
    /// Panics when `scope` does not belong to this graph.
    #[must_use]
    pub fn scope_references(&self, scope: ScopeId) -> &[ScopeReferenceId] {
        &self.scope_references[scope.index()]
    }

    /// Iterate over every live scope identity in allocation order.
    pub fn scope_ids(&self) -> impl Iterator<Item = ScopeId> + '_ {
        self.graph.node_ids()
    }

    /// Iterate over every live edge identity in insertion order.
    pub fn edge_ids(&self) -> impl Iterator<Item = ScopeEdgeId> + '_ {
        self.graph.edge_ids()
    }

    /// Iterate over every datum identity in insertion order.
    pub fn datum_ids(&self) -> impl ExactSizeIterator<Item = ScopeDatumId> + '_ {
        (0..self.data.len()).map(ScopeDatumId::from_index)
    }

    /// Iterate over every reference identity in insertion order.
    pub fn reference_ids(&self) -> impl ExactSizeIterator<Item = ScopeReferenceId> + '_ {
        (0..self.references.len()).map(ScopeReferenceId::from_index)
    }

    /// Return outgoing edge identities in insertion order.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn outgoing(&self, scope: ScopeId) -> impl Iterator<Item = ScopeEdgeId> + '_ {
        self.graph.outgoing(scope)
    }

    /// Return incoming edge identities in insertion order.
    #[must_use = "iterators are lazy and do nothing unless consumed"]
    pub fn incoming(&self, scope: ScopeId) -> impl Iterator<Item = ScopeEdgeId> + '_ {
        self.graph.incoming(scope)
    }

    /// Return the number of scopes.
    #[must_use]
    pub fn scope_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Return the number of live scope edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// An exclusive upper bound on every live [`ScopeEdgeId`] index.
    #[must_use]
    pub fn edge_bound(&self) -> usize {
        self.graph.edge_bound()
    }

    /// An exclusive upper bound on every live [`ScopeId`] index.
    #[must_use]
    pub fn scope_bound(&self) -> usize {
        self.graph.node_bound()
    }

    /// Return the number of relation-tagged data entries.
    #[must_use]
    pub fn datum_count(&self) -> usize {
        self.data.len()
    }

    /// Return the number of references.
    #[must_use]
    pub fn reference_count(&self) -> usize {
        self.references.len()
    }

    /// Return whether the graph contains no scopes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }
}

impl<S, L, R, D, Q> Default for ScopeGraph<S, L, R, D, Q> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S, L, R, D, Q> GraphView for ScopeGraph<S, L, R, D, Q> {
    type NodeId = ScopeId;

    fn node_bound(&self) -> usize {
        self.scope_bound()
    }

    fn node_ids(&self) -> impl Iterator<Item = ScopeId> + '_ {
        ScopeGraph::scope_ids(self)
    }

    fn successors(&self, node: ScopeId) -> impl Iterator<Item = ScopeId> + '_ {
        self.graph.successors(node)
    }

    fn predecessors(&self, node: ScopeId) -> impl Iterator<Item = ScopeId> + '_ {
        self.graph.predecessors(node)
    }
}

impl<S, L, R, D, Q> EdgeView for ScopeGraph<S, L, R, D, Q> {
    type EdgeId = ScopeEdgeId;
    type EdgeData = L;

    fn edge_bound(&self) -> usize {
        ScopeGraph::edge_bound(self)
    }

    fn edge_ids(&self) -> impl Iterator<Item = ScopeEdgeId> + '_ {
        ScopeGraph::edge_ids(self)
    }

    fn outgoing(&self, node: ScopeId) -> impl Iterator<Item = ScopeEdgeId> + '_ {
        ScopeGraph::outgoing(self, node)
    }

    fn incoming(&self, node: ScopeId) -> impl Iterator<Item = ScopeEdgeId> + '_ {
        ScopeGraph::incoming(self, node)
    }

    fn edge(&self, edge: ScopeEdgeId) -> EdgeRef<'_, ScopeId, ScopeEdgeId, L> {
        ScopeGraph::edge(self, edge)
    }
}

impl<S, L, R, D, Q> Index<ScopeId> for ScopeGraph<S, L, R, D, Q> {
    type Output = Scope<S>;

    fn index(&self, scope: ScopeId) -> &Self::Output {
        self.scope(scope)
    }
}

impl<S, L, R, D, Q> Index<ScopeDatumId> for ScopeGraph<S, L, R, D, Q> {
    type Output = ScopeDatum<R, D>;

    fn index(&self, datum: ScopeDatumId) -> &Self::Output {
        self.datum(datum)
    }
}

impl<S, L, R, D, Q> Index<ScopeReferenceId> for ScopeGraph<S, L, R, D, Q> {
    type Output = ScopeReference<Q>;

    fn index(&self, reference: ScopeReferenceId) -> &Self::Output {
        self.reference(reference)
    }
}
