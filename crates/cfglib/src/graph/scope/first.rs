//! Resolution of the first datum along one upward path, without a stored
//! path.

use core::fmt;

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
/// that climbs `D` scopes costs `O(D)` hook calls and no cycle check. It
/// allocates nothing more than the query hooks allocate.
///
/// # Precondition
///
/// Every edge that the query admits must point upward: the edge goes to a
/// scope with a smaller index than its source. A lexical parent edge does
/// this when the scopes are added in preorder. Then no path can come back to
/// a scope that it left, so the walk needs no record of the scopes that it
/// visited, and it takes at most one step for each scope of the graph. An
/// admitted edge that does not point upward is an error
/// ([`ScopeFirstResolutionError::EdgeNotUpward`]). An edge that the query
/// rejects can point in any direction.
///
/// # Relation to linear resolution
///
/// The resolver stops at the first accepted datum. It does not call
/// [`should_extend`](ScopeCursorQuery::should_extend) there, and it applies
/// no shadowing order. For a query that stops the extension when it accepts
/// a datum, the result and the statistics are equal to those of
/// [`try_compute_linear`](super::ScopeResolution::try_compute_linear): its
/// one visible candidate is the datum found here. A query that extends past
/// a candidate can reach more data, and a more specific later path can
/// shadow the first datum. This resolver looks for neither.
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
/// // The scopes are added from the outside in, so every parent edge points
/// // upward.
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
    /// Resolves one stored reference to the first datum that `query`
    /// accepts.
    ///
    /// # Errors
    ///
    /// Returns [`ScopeFirstResolutionError`] when a prefix accepts more than
    /// one datum, admits more than one outgoing edge, or admits an edge that
    /// does not point upward.
    ///
    /// # Panics
    ///
    /// Panics when `reference` is not a reference of `graph`.
    pub fn try_compute<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        reference: ScopeReferenceId,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Result<Self, ScopeFirstResolutionError>
    where
        P: ScopeCursorQuery<S, L, R, D, Q> + ?Sized,
    {
        Self::try_compute_query(
            graph,
            &ScopeQuery::from_reference(graph, reference),
            query,
            config,
        )
    }

    /// Resolves one ephemeral query that starts in `scope` to the first
    /// datum that `query` accepts.
    ///
    /// This is the counterpart of [`try_compute`](Self::try_compute) for a
    /// query that the graph does not store.
    ///
    /// # Errors
    ///
    /// Returns [`ScopeFirstResolutionError`] when a prefix accepts more than
    /// one datum, admits more than one outgoing edge, or admits an edge that
    /// does not point upward.
    ///
    /// # Panics
    ///
    /// Panics when `scope` is not a scope of `graph`.
    pub fn try_compute_from<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        scope: ScopeId,
        data: &Q,
        query: &P,
        config: ScopeResolutionConfig,
    ) -> Result<Self, ScopeFirstResolutionError>
    where
        P: ScopeCursorQuery<S, L, R, D, Q> + ?Sized,
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
        P: ScopeCursorQuery<S, L, R, D, Q> + ?Sized,
    {
        let mut cursor = ScopeCursor::new(input.scope(), input.scope(), 0);
        let mut state = query.initial_state(graph, input);
        let mut budget = Budget::new(config);
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
                // The upward order makes the walk free of cycles without a
                // record of the visited scopes.
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

        budget.stats.reachable_candidate_count = usize::from(candidate.is_some());
        Ok(Self {
            candidate,
            stats: budget.stats,
        })
    }

    /// Returns the first accepted datum and the scope that owns it. Returns
    /// `None` when the path ends, or a bound stops it, before a scope accepts
    /// a datum.
    #[must_use]
    pub const fn candidate(&self) -> Option<(ScopeDatumId, ScopeId)> {
        self.candidate
    }

    /// Returns whether the query accepted a datum.
    #[must_use]
    pub const fn is_resolved(&self) -> bool {
        self.candidate.is_some()
    }

    /// Returns the work of the query and the bounds that stopped it.
    ///
    /// `explored_path_count` is the number of scopes that the walk tested.
    /// `reachable_candidate_count` is one for a resolved query and zero
    /// otherwise.
    #[must_use]
    pub const fn stats(&self) -> ScopeResolutionStats {
        self.stats
    }
}

/// The bounds of one walk and the record of the bound that refused work.
///
/// A truncation flag is set only when a bound refuses work, not when the
/// work only reaches the bound. These are the rules of
/// [`ScopeResolution`](super::ScopeResolution), so the statistics of the
/// two resolvers are comparable.
struct Budget {
    max_path_length: Option<usize>,
    max_path_count: Option<usize>,
    stats: ScopeResolutionStats,
}

impl Budget {
    fn new(config: ScopeResolutionConfig) -> Self {
        Self {
            max_path_length: config.max_path_length,
            max_path_count: config.max_path_count,
            stats: ScopeResolutionStats::default(),
        }
    }

    /// Counts one more tested prefix. Returns `false` and records the
    /// truncation when the count bound has no space for it.
    fn admit_expansion(&mut self) -> bool {
        if self
            .max_path_count
            .is_some_and(|limit| self.stats.explored_path_count >= limit)
        {
            self.stats.path_count_limited = true;
            return false;
        }
        self.stats.explored_path_count += 1;
        true
    }

    /// Returns whether a path of `extended_len` edges fits the length bound.
    /// Otherwise records the truncation and returns `false`.
    fn admit_length(&mut self, extended_len: usize) -> bool {
        if self
            .max_path_length
            .is_some_and(|limit| extended_len > limit)
        {
            self.stats.path_length_limited = true;
            return false;
        }
        true
    }
}

/// A query broke the contract of [`ScopeFirstResolution`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScopeFirstResolutionError {
    /// One path prefix accepted more than one datum.
    MultipleCandidates {
        /// The scope where the query accepted the competing candidates.
        scope: ScopeId,
    },
    /// One path prefix admitted more than one outgoing edge.
    MultipleEdges {
        /// The scope where the walk branched.
        scope: ScopeId,
    },
    /// The query admitted an edge whose target index is not smaller than the
    /// index of its source.
    EdgeNotUpward {
        /// The admitted edge.
        edge: ScopeEdgeId,
    },
}

impl fmt::Display for ScopeFirstResolutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
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
