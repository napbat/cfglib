//! The environments of a [`ScopeEnvironmentCache`](super::ScopeEnvironmentCache)
//! as an open fold: a scope's environment folds its own accepted data and
//! the environments its admitted edges reach.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use core::ops::ControlFlow;
use smallvec::SmallVec;

use crate::graph::edge_view::EdgeView;
use crate::graph::open::{FoldEnter, MarkScope, OpenFold};
use crate::graph::search::{PathBudget, retain_unshadowed};

use super::super::query::label_word_shadows;
use super::super::storage::{ScopeEdgeId, ScopeGraph, ScopeId};
use super::{Entry, Environment, Link, ScopeEnvironmentQuery};

/// One reach of a scope: the scope, the path-language state there, the
/// edges left under the length bound, and the edge the parent reached it
/// through.
#[derive(Debug, Clone)]
pub(super) struct Reach<St> {
    scope: ScopeId,
    state: St,
    remaining: Option<usize>,
    via: Option<ScopeEdgeId>,
    /// Whether the length bound refuses `via`: the parent had no edge left.
    beyond: bool,
}

impl<St> Reach<St> {
    /// The reach of the scope where a query starts.
    pub(super) const fn start(scope: ScopeId, state: St, remaining: Option<usize>) -> Self {
        Self {
            scope,
            state,
            remaining,
            via: None,
            beyond: false,
        }
    }
}

/// What an environment is stored under, or what guards a route.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum EnvironmentMark<St> {
    /// The environment of a scope on no cycle, by what it depends on: the
    /// scope, the state there, and the edges left under the length bound.
    Environment(ScopeId, St, Option<usize>),
    /// A scope on a cycle in a state. A route never comes back to it, and
    /// its environment depends on the route, so it is never stored.
    OnCycle(ScopeId, St),
}

/// The candidates of one scope being folded, by key, each with its source:
/// `0` for the scope's own data and `n` for the `n`-th reached edge.
pub(super) struct Candidates<K> {
    scope: ScopeId,
    by_key: BTreeMap<K, SmallVec<[(usize, Entry); 1]>>,
    length_limited: bool,
    sources: usize,
}

/// The fold that computes the environments one query needs, over the link
/// table of its cache.
///
/// The length bound lives in the reaches, not in the walk's depth bound,
/// for three reasons: a refused edge sets the environment's
/// `length_limited`, which the walk's silent bound could not report; a
/// route back to a scope on a cycle is refused before the bound is charged,
/// as the general resolver does; and environments keyed by the edges left
/// serve every bound from one cache.
pub(super) struct EnvironmentFold<'a, S, L, R, D, Q, P> {
    pub(super) graph: &'a ScopeGraph<S, L, R, D, Q>,
    pub(super) query: &'a P,
    pub(super) on_cycle: &'a [bool],
    pub(super) links: &'a mut Vec<Link>,
    /// Counts the environments the query computes against
    /// `max_path_count`.
    pub(super) budget: PathBudget,
}

impl<S, L, R, D, Q, P> EnvironmentFold<'_, S, L, R, D, Q, P>
where
    P: ScopeEnvironmentQuery<S, L, R, D, Q>,
{
    fn limited(&self) -> bool {
        self.budget.stats().path_count_limited
    }

    /// Whether the path from link `left` is lexicographically more specific
    /// than the one from link `right`; both start in the same scope.
    fn shadows(&self, left: usize, right: usize) -> bool {
        label_word_shadows(
            edge_labels(self.graph, self.links, left),
            edge_labels(self.graph, self.links, right),
            |left, right| self.query.compare_labels(self.graph, left, right),
        )
    }
}

/// The labels of the edges of the stored path that starts at `link`.
fn edge_labels<'a, S, L, R, D, Q>(
    graph: &'a ScopeGraph<S, L, R, D, Q>,
    links: &'a [Link],
    mut link: usize,
) -> impl Iterator<Item = &'a L> {
    core::iter::from_fn(move || {
        let (edge, next) = links[link].hop?;
        link = next;
        Some(graph.edge(edge).data())
    })
}

impl<S, L, R, D, Q, P> OpenFold for EnvironmentFold<'_, S, L, R, D, Q, P>
where
    P: ScopeEnvironmentQuery<S, L, R, D, Q>,
{
    type Node = Reach<P::State>;
    type Mark = EnvironmentMark<P::State>;
    type Value = Environment<P::Key>;
    type Accumulator = Candidates<P::Key>;

    /// The reaches through the edges the query admits. With no edge left,
    /// they are reaches beyond the bound, which a route back to a scope on a
    /// cycle still refuses first.
    fn successors(&mut self, reach: &Self::Node, out: &mut Vec<Self::Node>) {
        let beyond = reach.remaining == Some(0);
        let remaining = reach.remaining.map(|remaining| remaining.saturating_sub(1));
        for edge_id in self.graph.outgoing(reach.scope) {
            let edge = self.graph.edge(edge_id);
            let Some(state) = self.query.step(self.graph, reach.scope, &reach.state, edge) else {
                continue;
            };
            out.push(Reach {
                scope: edge.target(),
                state,
                remaining,
                via: Some(edge_id),
                beyond,
            });
        }
    }

    fn mark(&mut self, reach: &Self::Node) -> Option<Self::Mark> {
        if self.on_cycle[reach.scope.index()] {
            Some(EnvironmentMark::OnCycle(reach.scope, reach.state.clone()))
        } else if reach.beyond {
            // A scope on no cycle needs no guard, and the reach no answer.
            None
        } else {
            Some(EnvironmentMark::Environment(
                reach.scope,
                reach.state.clone(),
                reach.remaining,
            ))
        }
    }

    /// An environment that the count bound cut short is not the scope's
    /// environment, and one on a cycle depends on the route.
    fn memoize(&mut self, reach: &Self::Node) -> bool {
        !reach.beyond && !self.on_cycle[reach.scope.index()] && !self.limited()
    }

    fn enter(&mut self, reach: &Self::Node) -> FoldEnter<Self::Value, Self::Accumulator> {
        if reach.beyond {
            // An edge the length bound refuses: an empty environment that
            // reports the cut. A reach that the walk refused first, a route
            // back to an open scope on a cycle, reports nothing.
            return FoldEnter::Leaf(Some(Environment {
                entries: BTreeMap::new(),
                length_limited: true,
            }));
        }
        if !self.budget.admit_expansion() {
            return FoldEnter::Leaf(None);
        }
        // The path that ends here, shared by every datum of the scope.
        let mut end = None;
        let mut by_key: BTreeMap<P::Key, SmallVec<[(usize, Entry); 1]>> = BTreeMap::new();
        for &datum in self.graph.scope_data(reach.scope) {
            if self
                .query
                .accepts(self.graph, reach.scope, &reach.state, datum)
            {
                let path = *end.get_or_insert_with(|| {
                    self.links.push(Link {
                        scope: reach.scope,
                        hop: None,
                    });
                    self.links.len() - 1
                });
                by_key
                    .entry(self.query.key(self.graph, datum))
                    .or_default()
                    .push((0, Entry { datum, path }));
            }
        }
        FoldEnter::Fold {
            accumulator: Candidates {
                scope: reach.scope,
                by_key,
                length_limited: false,
                sources: 0,
            },
            marks: MarkScope::Isolated,
        }
    }

    /// Adds the entries of the environment reached through `via` to the
    /// candidates, with paths that start with that edge. Once the count bound
    /// refused an environment, the scope stops after the child it absorbs.
    fn absorb(
        &mut self,
        candidates: &mut Self::Accumulator,
        reach: &Self::Node,
        environment: Option<Self::Value>,
    ) -> ControlFlow<()> {
        candidates.sources += 1;
        if let (Some(environment), Some(edge)) = (environment, reach.via) {
            candidates.length_limited |= environment.length_limited;
            for (key, entries) in environment.entries {
                let by_key = candidates.by_key.entry(key).or_default();
                for entry in entries {
                    let path = self.links.len();
                    self.links.push(Link {
                        scope: candidates.scope,
                        hop: Some((edge, entry.path)),
                    });
                    by_key.push((
                        candidates.sources,
                        Entry {
                            datum: entry.datum,
                            path,
                        },
                    ));
                }
            }
        }
        if self.limited() {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    }

    /// Keeps, within each key, the candidates that no other candidate
    /// shadows.
    ///
    /// Two candidates from one source never shadow each other: two data of
    /// the scope both end at the declaration marker, and the candidates
    /// through one edge are the visible ones of one environment, which no
    /// other one of them shadows. Only pairs from different sources are
    /// compared.
    fn finish(
        &mut self,
        _reach: &Self::Node,
        candidates: Self::Accumulator,
    ) -> Option<Self::Value> {
        let entries = candidates
            .by_key
            .into_iter()
            .map(|(key, mut by_key)| {
                retain_unshadowed(
                    &mut by_key,
                    |&(other_source, other), &(source, candidate)| {
                        other_source != source && self.shadows(other.path, candidate.path)
                    },
                );
                let visible = by_key.into_iter().map(|(_, candidate)| candidate).collect();
                (key, visible)
            })
            .collect();
        Some(Environment {
            entries,
            length_limited: candidates.length_limited,
        })
    }
}
