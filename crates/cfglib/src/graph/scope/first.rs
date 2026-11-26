//! Resolution of the first datum along one upward path, without a stored
//! path.

use core::fmt;

use crate::graph::edge_view::EdgeView;
use crate::graph::search::{PathBudget, PathSearchConfig, PathSearchStats};
use crate::graph::view::assert_in_bound;

use super::cursor::{ScopeCursor, ScopeCursorQuery};
use super::query::ScopeQuery;
use super::storage::{ScopeDatumId, ScopeEdgeId, ScopeGraph, ScopeId};

#[cfg(test)]
mod tests;

/// The first datum that a linear query accepts on an upward path: the scope
/// resolver for a policy that admits one edge at each step.
///
/// The resolver keeps only a [`ScopeCursor`] and the query state, not the
/// path. A query that climbs `D` scopes costs `O(D)` hook calls and no cycle
/// check. It allocates nothing more than the query hooks allocate.
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
/// # Relation to the general resolver
///
/// The resolver stops at the first accepted datum. It does not call
/// [`should_extend`](ScopeCursorQuery::should_extend) there, and it applies
/// no shadowing order. Under [`ScopeCursorPolicy`](super::ScopeCursorPolicy),
/// [`ScopeResolution::compute`](super::ScopeResolution::compute) runs the
/// same policy breadth-first: on a walk that this resolver completes, the
/// first candidate there is the datum found here, and for a query that stops
/// the extension when it accepts a datum, the statistics are equal too. A
/// query that extends past a candidate can reach more data. This resolver
/// looks for none.
///
/// # Examples
///
/// ```
/// use cfglib::{
///     EdgeRef, PathSearchConfig, ScopeCursor, ScopeCursorQuery, ScopeDatumId, ScopeEdgeId,
///     ScopeFirstResolution, ScopeGraph, ScopeId, ScopeQuery,
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
/// let resolution = ScopeFirstResolution::try_compute(
///     &graph,
///     ScopeQuery::new(block, &"x"),
///     &Lexical,
///     PathSearchConfig::default(),
/// )?;
/// assert_eq!(resolution.candidate(), Some((global, module)));
/// assert_eq!(resolution.stats().expanded_path_count, 3);
/// # Ok::<(), cfglib::ScopeFirstResolutionError>(())
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScopeFirstResolution {
    candidate: Option<(ScopeDatumId, ScopeId)>,
    stats: PathSearchStats,
}

impl ScopeFirstResolution {
    /// Resolves `input`, a stored reference or an ephemeral query, to the
    /// first datum that `query` accepts.
    ///
    /// The walk follows one path, so it reads neither `config.order` nor
    /// `config.max_complete_path_count`.
    ///
    /// # Errors
    ///
    /// Returns [`ScopeFirstResolutionError`] when a prefix accepts more than
    /// one datum, admits more than one outgoing edge, or admits an edge that
    /// does not point upward.
    ///
    /// # Panics
    ///
    /// Panics when `input` starts in a scope that does not belong to `graph`.
    pub fn try_compute<S, L, R, D, Q, P>(
        graph: &ScopeGraph<S, L, R, D, Q>,
        input: ScopeQuery<'_, Q>,
        query: &P,
        config: PathSearchConfig,
    ) -> Result<Self, ScopeFirstResolutionError>
    where
        P: ScopeCursorQuery<S, L, R, D, Q> + ?Sized,
    {
        assert_in_bound(graph, input.scope(), "query scope");
        let input = &input;
        let mut cursor = ScopeCursor::new(input.scope(), input.scope(), 0);
        let mut state = query.initial_state(graph, input);
        let mut budget = PathBudget::new(&config);
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

        Ok(Self {
            candidate,
            stats: PathSearchStats {
                complete_path_count: usize::from(candidate.is_some()),
                ..budget.stats()
            },
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
    /// `expanded_path_count` is the number of scopes that the walk tested.
    /// `complete_path_count` is one for a resolved query and zero otherwise.
    #[must_use]
    pub const fn stats(&self) -> PathSearchStats {
        self.stats
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
