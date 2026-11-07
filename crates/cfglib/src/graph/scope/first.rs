//! Path-free resolution of the first datum along one upward path.

use super::cursor::{ScopeCursor, ScopeCursorQuery};
use super::query::{ScopeQuery, ScopeResolutionConfig, ScopeResolutionStats};
use super::storage::{ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeId, ScopeReferenceId};

#[cfg(test)]
mod tests;

/// The first datum that a linear query accepts on an upward path.
///
/// This resolver follows the one path that
/// [`ScopeResolution::try_compute_linear`](super::ScopeResolution::try_compute_linear)
/// follows, but it keeps only a [`ScopeCursor`] and the query state. A query
/// that climbs `D` scopes costs `O(D)` hook calls, does no cycle check, and
/// allocates nothing beyond what the query hooks allocate.
///
/// # Precondition
///
/// Every edge that the query admits must point upward: the edge leads to a
/// scope with a smaller index than its source, as a lexical parent does when
/// scopes are added in preorder. No path can then return to a scope it left,
/// so the walk needs no record of the scopes it visited, and it takes at most
/// one step for each scope in the graph. An admitted edge that does not point
/// upward is an error ([`ScopeFirstResolutionError::EdgeNotUpward`]). An edge
/// that the query rejects can point anywhere.
///
/// # Relation to linear resolution
///
/// The resolver stops at the first accepted datum and does not consult
/// [`should_extend`](ScopeCursorQuery::should_extend) there, and it applies no
/// shadowing order. For a query that stops extending once it accepts a datum,
/// the result and the statistics equal those of
/// [`try_compute_linear`](super::ScopeResolution::try_compute_linear), whose
/// one visible candidate is the datum found here. A query that extends past a
/// candidate can reach more data, and a more specific later path could shadow
/// the first datum; this resolver does not look for either.
///
/// # Examples
///
/// ```
/// use core::cmp::Ordering;
///
/// use cfglib::{
///     EdgeRef, ScopeCursor, ScopeCursorQuery, ScopeDatumId, ScopeEdgeId, ScopeFirstResolution,
///     ScopeGraph, ScopeGraphPathLabel, ScopeId, ScopeQuery, ScopeResolutionConfig,
/// };
///
/// type Names = ScopeGraph<(), (), (), &'static str, &'static str>;
///
/// /// Lexical lookup of a name through parent edges.
/// struct Lexical;
///
/// impl ScopeCursorQuery<(), (), (), &'static str, &'static str> for Lexical {
///     type State = ();
///
///     fn initial_state(&self, _: &Names, _: &ScopeQuery<'_, &'static str>) {}
///
///     fn step(
///         &self,
///         _: &Names,
///         _: &ScopeQuery<'_, &'static str>,
///         _: ScopeCursor,
///         (): &(),
///         _: EdgeRef<'_, ScopeId, ScopeEdgeId, ()>,
///     ) -> Option<()> {
///         Some(())
///     }
///
///     fn accepts(
///         &self,
///         graph: &Names,
///         input: &ScopeQuery<'_, &'static str>,
///         _: ScopeCursor,
///         (): &(),
///         datum: ScopeDatumId,
///     ) -> bool {
///         graph.datum(datum).data() == input.data()
///     }
///
///     fn should_extend(
///         &self,
///         _: &Names,
///         _: &ScopeQuery<'_, &'static str>,
///         _: ScopeCursor,
///         (): &(),
///         accepted_candidate_count: usize,
///     ) -> bool {
///         accepted_candidate_count == 0
///     }
///
///     fn compare_labels(
///         &self,
///         _: &Names,
///         _: &ScopeQuery<'_, &'static str>,
///         left: ScopeGraphPathLabel<'_, ()>,
///         right: ScopeGraphPathLabel<'_, ()>,
///     ) -> Option<Ordering> {
///         let is_edge = |label| matches!(label, ScopeGraphPathLabel::Edge(()));
///         Some(is_edge(left).cmp(&is_edge(right)))
///     }
/// }
///
/// // Scopes are added outside-in, so every parent edge points upward.
/// let mut graph = Names::new();
/// let module = graph.add_scope(());
/// let function = graph.add_scope(());
/// let block = graph.add_scope(());
/// graph.add_edge(function, module, ());
/// graph.add_edge(block, function, ());
/// let global = graph.add_datum(module, (), "x");
/// graph.add_datum(function, (), "y");
///
/// let resolution = ScopeFirstResolution::try_compute_from(
///     &graph,
///     block,
///     &"x",
///     &Lexical,
///     ScopeResolutionConfig::new(),
/// )?;
/// assert_eq!(resolution.candidate(), Some((global, module)));
/// assert_eq!(resolution.stats().explored_path_count, 3);
/// # Ok::<(), cfglib::ScopeFirstResolutionError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeFirstResolution {
    candidate: Option<(ScopeDatumId, ScopeId)>,
    stats: ScopeResolutionStats,
}

impl ScopeFirstResolution {
    /// Resolve one stored reference to the first datum that `query` accepts.
    ///
    /// # Errors
    ///
    /// Returns [`ScopeFirstResolutionError`] if a prefix accepts several data,
    /// admits several outgoing edges, or admits an edge that does not point
    /// upward.
    ///
    /// # Panics
    ///
    /// Panics when `reference` does not belong to `graph`.
    pub fn try_compute<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        reference: ScopeReferenceId,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Result<Self, ScopeFirstResolutionError>
    where
        P: ScopeCursorQuery<S, L, R, D, Q>,
    {
        Self::try_compute_query(
            graph,
            &ScopeQuery::from_reference(graph, reference),
            query,
            config,
        )
    }

    /// Resolve one ephemeral query beginning in `scope` to the first datum that
    /// `query` accepts.
    ///
    /// This is the non-mutating counterpart of [`try_compute`](Self::try_compute).
    ///
    /// # Errors
    ///
    /// Returns [`ScopeFirstResolutionError`] if a prefix accepts several data,
    /// admits several outgoing edges, or admits an edge that does not point
    /// upward.
    ///
    /// # Panics
    ///
    /// Panics when `scope` does not belong to `graph`.
    pub fn try_compute_from<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        scope: ScopeId,
        data: &Q,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Result<Self, ScopeFirstResolutionError>
    where
        P: ScopeCursorQuery<S, L, R, D, Q>,
    {
        let _ = graph.scope(scope);
        Self::try_compute_query(graph, &ScopeQuery::new(scope, data), query, config)
    }

    fn try_compute_query<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: &ScopeQuery<'_, Q>,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Result<Self, ScopeFirstResolutionError>
    where
        P: ScopeCursorQuery<S, L, R, D, Q>,
    {
        let mut cursor = ScopeCursor::new(input.scope(), input.scope(), 0);
        let mut state = query.initial_state(graph, input);
        let mut budget = config.budget();
        let mut candidate = None;

        while budget.admit_expansion() {
            let mut accepted = None;
            let mut multiple_candidates = false;
            query.visit_accepted_data(graph, input, cursor, &state, &mut |datum| {
                if accepted.replace(datum).is_some() {
                    multiple_candidates = true;
                }
            });
            if multiple_candidates {
                return Err(ScopeFirstResolutionError::MultipleCandidates {
                    scope: cursor.end(),
                });
            }
            if let Some(datum) = accepted {
                candidate = Some((datum, cursor.end()));
                break;
            }
            if !query.should_extend(graph, input, cursor, &state, 0) {
                break;
            }

            let mut next = None;
            for edge_id in graph.outgoing(cursor.end()) {
                let edge = graph.edge(edge_id);
                let Some(next_state) = query.step(graph, input, cursor, &state, edge) else {
                    continue;
                };
                // The upward order is what makes the walk cycle-free without
                // a record of the visited scopes.
                if edge.target().index() >= edge.source().index() {
                    return Err(ScopeFirstResolutionError::EdgeNotUpward { edge: edge_id });
                }
                if !budget.admit_length(cursor.len() + 1) {
                    continue;
                }
                if next.is_some() {
                    return Err(ScopeFirstResolutionError::MultipleEdges {
                        scope: cursor.end(),
                    });
                }
                next = Some((edge.target(), next_state));
            }
            let Some((target, next_state)) = next else {
                break;
            };
            cursor = cursor.advanced(target);
            state = next_state;
        }

        Ok(Self {
            candidate,
            stats: ScopeResolutionStats::from_budget(&budget, usize::from(candidate.is_some())),
        })
    }

    /// The first accepted datum and the scope that owns it, or `None` when the
    /// path ends, or a bound stops it, before any scope accepts a datum.
    #[must_use]
    pub const fn candidate(&self) -> Option<(ScopeDatumId, ScopeId)> {
        self.candidate
    }

    /// Whether the query accepted a datum.
    #[must_use]
    pub const fn is_resolved(&self) -> bool {
        self.candidate.is_some()
    }

    /// Query work and truncation information.
    ///
    /// `explored_path_count` is the number of scopes the walk tested, and
    /// `reachable_candidate_count` is one for a resolved query and zero
    /// otherwise.
    #[must_use]
    pub const fn stats(&self) -> ScopeResolutionStats {
        self.stats
    }
}

/// A query violated the contract of [`ScopeFirstResolution`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeFirstResolutionError {
    /// One path prefix accepted more than one datum.
    MultipleCandidates {
        /// Scope where the competing candidates were accepted.
        scope: ScopeId,
    },
    /// One path prefix admitted more than one outgoing edge.
    MultipleEdges {
        /// Scope where the traversal branched.
        scope: ScopeId,
    },
    /// The query admitted an edge whose target index is not smaller than the
    /// index of its source.
    EdgeNotUpward {
        /// The admitted edge.
        edge: ScopeEdgeId,
    },
}

impl core::fmt::Display for ScopeFirstResolutionError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MultipleCandidates { scope } => {
                write!(formatter, "scope {scope:?} accepted multiple candidates")
            }
            Self::MultipleEdges { scope } => {
                write!(
                    formatter,
                    "scope {scope:?} admitted multiple outgoing edges"
                )
            }
            Self::EdgeNotUpward { edge } => {
                write!(
                    formatter,
                    "admitted scope edge {edge:?} does not point upward"
                )
            }
        }
    }
}

impl core::error::Error for ScopeFirstResolutionError {}
