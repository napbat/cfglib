//! Breadth-first simple-route enumeration over an open node space.
//!
//! This is the breadth-first form of once-per-route walking. The depth-first
//! form is [`open_depth_first_events`](super::open_depth_first_events) under
//! [`VisitedPolicy::Path`](crate::VisitedPolicy::Path), which a stack gets
//! almost for free: one path is active at a time, and its marks release on
//! the unwind. A breadth-first frontier holds many partial routes at once and
//! never unwinds, so here every frontier entry owns its route outright — the
//! same semantics, bought with the state a frontier actually has.

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::vec;
use alloc::vec::Vec;
use core::ops::ControlFlow;

use crate::graph::search::Visit;

use super::may_expand;

/// Configuration for an [`open_breadth_first_paths`] walk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OpenPathsConfig {
    /// Maximum route length to extend, in hops from a seed. `None` is
    /// unbounded; `Some(0)` emits the seed routes alone.
    pub max_depth: Option<usize>,
}

impl OpenPathsConfig {
    /// An unbounded route enumeration.
    ///
    /// Unbounded enumeration of an unbounded space does not terminate; rely
    /// on a finite space, a depth bound, or a breaking event callback.
    #[must_use]
    pub const fn new() -> Self {
        Self { max_depth: None }
    }

    /// Return this configuration bounded to `max_depth` hops from a seed.
    #[must_use]
    pub const fn with_max_depth(mut self, max_depth: usize) -> Self {
        self.max_depth = Some(max_depth);
        self
    }
}

/// An event of an [`open_breadth_first_paths`] walk.
///
/// Routes are reported by reference so the walk retains ownership of its
/// frontier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OpenPathsEvent<'a, N> {
    /// A simple route from a seed, in nondecreasing length order.
    ///
    /// This is the only event whose [`Visit`] verdict is read:
    /// [`Visit::Descend`] extends the route through its last node's
    /// successors, [`Visit::Skip`] seals it.
    Route(&'a [N]),
    /// The walk declined to extend the route with the node: it is already on
    /// that route, so the extension would close a cycle.
    ///
    /// This is the per-route walk's only refusal — a node shared between
    /// *different* routes is not refused, it is each route's own
    /// [`Route`](OpenPathsEvent::Route) discovery. The returned [`Visit`]
    /// verdict is ignored.
    Refused(&'a [N], &'a N),
}

/// Enumerate the simple routes of a lazily discovered node space
/// breadth-first from `seeds`, shortest routes first, and return the value
/// the callback broke with (or `None` when the walk completed).
///
/// [`open_breadth_first_events`](super::open_breadth_first_events) marks
/// nodes globally, so a node shared by several routes is discovered once and
/// every later route is refused. This walk keeps **no global marks at all**:
/// each frontier entry owns its route, a node is refused only when it is
/// already on that entry's own route, and a shared node is reported once per
/// distinct route. The first [`Route`](OpenPathsEvent::Route) satisfying a
/// consumer predicate is therefore a *shortest* satisfying route — the
/// guarantee the depth-first form cannot make.
///
/// `successors` receives a node and a scratch buffer that is **cleared before
/// every call**. It pushes successors in their desired discovery order. Nodes
/// are compared by `PartialEq` along one route only, so no ordering is
/// required.
///
/// # Cost
///
/// Simple-route enumeration is exponential in the worst case — a route tree,
/// not a node set. Memory is proportional to the live frontier's total route
/// length and there are no `O(nodes)` marks to bound it. Bound the walk with
/// [`OpenPathsConfig::with_max_depth`], prune with [`Visit::Skip`], or stop
/// early with [`ControlFlow::Break`].
///
/// # Semantics
///
/// - Seeds each start a route of length one, emitted at depth zero in the
///   order supplied. Duplicate seeds start independent routes.
/// - Routes are extended level by level, so events arrive in nondecreasing
///   route length. Successors extend a route in the order the closure pushed
///   them.
/// - [`Visit::Skip`] seals a route without extending it.
/// - `max_depth` bounds extension, not discovery: a route at the bound is
///   still emitted, and never extended.
/// - An extension whose node is already on that route produces
///   [`Refused`](OpenPathsEvent::Refused) instead of a route.
/// - [`ControlFlow::Break`] stops immediately and is returned as
///   `Some(value)`.
///
/// # Examples
///
/// The property the global-marks walk cannot offer — the first route to reach
/// a goal is a shortest one, even when a longer route was seeded first:
///
/// ```
/// use core::ops::ControlFlow;
///
/// use cfglib::{OpenPathsConfig, OpenPathsEvent, Visit, open_breadth_first_paths};
///
/// // "app" imports "util" directly and through two re-export hops.
/// let imports = |module: &&'static str, out: &mut Vec<&'static str>| match *module {
///     "app" => out.extend(["barrel", "util"]),
///     "barrel" => out.push("inner"),
///     "inner" => out.push("util"),
///     _ => {}
/// };
///
/// let shortest = open_breadth_first_paths(
///     ["app"],
///     OpenPathsConfig::new(),
///     imports,
///     |event| match event {
///         OpenPathsEvent::Route(route) if *route.last().unwrap() == "util" => {
///             ControlFlow::Break(route.to_vec())
///         }
///         _ => ControlFlow::Continue(Visit::Descend),
///     },
/// );
///
/// assert_eq!(shortest, Some(vec!["app", "util"]));
/// ```
#[must_use]
pub fn open_breadth_first_paths<N: Clone + PartialEq, B>(
    seeds: impl IntoIterator<Item = N>,
    config: OpenPathsConfig,
    mut successors: impl FnMut(&N, &mut Vec<N>),
    mut on_event: impl FnMut(OpenPathsEvent<'_, N>) -> ControlFlow<B, Visit>,
) -> Option<B> {
    let mut frontier: VecDeque<Vec<N>> = VecDeque::new();
    let mut scratch: Vec<N> = Vec::new();
    // Routes the frontier has finished with, kept with their capacity: every
    // route a walk extends is copied once per descending successor, and a
    // recycled route turns each copy into a `clone_from` instead of an
    // allocation.
    let mut spare: Vec<Vec<N>> = Vec::new();

    for seed in seeds {
        let route = vec![seed];
        match on_event(OpenPathsEvent::Route(&route)) {
            ControlFlow::Break(value) => return Some(value),
            ControlFlow::Continue(Visit::Descend) if may_expand(config.max_depth, 0) => {
                frontier.push_back(route);
            }
            ControlFlow::Continue(Visit::Descend | Visit::Skip) => {}
        }
    }

    'routes: while let Some(mut route) = frontier.pop_front() {
        scratch.clear();
        if let Some(current) = route.last() {
            successors(current, &mut scratch);
        }
        let mut extensions = scratch.drain(..).peekable();
        while let Some(next) = extensions.next() {
            if route.contains(&next) {
                if let ControlFlow::Break(value) = on_event(OpenPathsEvent::Refused(&route, &next))
                {
                    return Some(value);
                }
                continue;
            }

            route.push(next);
            let verdict = match on_event(OpenPathsEvent::Route(&route)) {
                ControlFlow::Break(value) => return Some(value),
                ControlFlow::Continue(verdict) => verdict,
            };
            if matches!(verdict, Visit::Descend) && may_expand(config.max_depth, route.len() - 1) {
                if extensions.peek().is_none() {
                    // No successor is left to extend this route, so the
                    // frontier takes the route itself instead of a copy.
                    frontier.push_back(route);
                    continue 'routes;
                }
                let mut copy = spare.pop().unwrap_or_default();
                copy.clone_from(&route);
                frontier.push_back(copy);
            }
            route.pop();
        }
        spare.push(route);
    }

    None
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::*;

    /// The diamond: app → {left, right} → merge.
    fn diamond(node: &&'static str, out: &mut Vec<&'static str>) {
        match *node {
            "app" => out.extend(["left", "right"]),
            "left" | "right" => out.push("merge"),
            _ => {}
        }
    }

    /// Collect every emitted route, running to completion.
    fn all_routes(
        seeds: impl IntoIterator<Item = &'static str>,
        config: OpenPathsConfig,
        successors: impl FnMut(&&'static str, &mut Vec<&'static str>),
    ) -> Vec<Vec<&'static str>> {
        let mut routes = Vec::new();
        let outcome = open_breadth_first_paths::<_, ()>(seeds, config, successors, |event| {
            if let OpenPathsEvent::Route(route) = event {
                routes.push(route.to_vec());
            }
            ControlFlow::Continue(Visit::Descend)
        });
        assert_eq!(outcome, None);
        routes
    }

    #[test]
    fn a_shared_node_is_reached_once_per_route() {
        let routes = all_routes(["app"], OpenPathsConfig::new(), diamond);
        let merged: Vec<_> = routes
            .iter()
            .filter(|route| *route.last().unwrap() == "merge")
            .collect();
        assert_eq!(merged.len(), 2, "one route per side of the diamond");
        assert!(merged.contains(&&alloc::vec!["app", "left", "merge"]));
        assert!(merged.contains(&&alloc::vec!["app", "right", "merge"]));
    }

    #[test]
    fn routes_arrive_in_nondecreasing_length_order() {
        let routes = all_routes(["app"], OpenPathsConfig::new(), diamond);
        let lengths: Vec<_> = routes.iter().map(Vec::len).collect();
        let mut sorted = lengths.clone();
        sorted.sort_unstable();
        assert_eq!(lengths, sorted, "breadth-first means shortest-first");
    }

    #[test]
    fn the_first_route_to_a_goal_is_a_shortest_one() {
        // A long chain is seeded before the direct edge exists on it.
        let successors = |node: &&'static str, out: &mut Vec<&'static str>| match *node {
            "app" => out.extend(["a", "goal"]),
            "a" => out.push("b"),
            "b" => out.push("goal"),
            _ => {}
        };
        let first =
            open_breadth_first_paths(["app"], OpenPathsConfig::new(), successors, |event| {
                match event {
                    OpenPathsEvent::Route(route) if *route.last().unwrap() == "goal" => {
                        ControlFlow::Break(route.to_vec())
                    }
                    _ => ControlFlow::Continue(Visit::Descend),
                }
            });
        assert_eq!(first, Some(alloc::vec!["app", "goal"]));
    }

    #[test]
    fn a_cycle_is_refused_and_the_walk_terminates() {
        let successors = |node: &&'static str, out: &mut Vec<&'static str>| match *node {
            "a" => out.push("b"),
            "b" => out.push("a"),
            _ => {}
        };
        let mut refusals = Vec::new();
        let outcome =
            open_breadth_first_paths::<_, ()>(["a"], OpenPathsConfig::new(), successors, |event| {
                if let OpenPathsEvent::Refused(route, node) = event {
                    refusals.push((route.to_vec(), *node));
                }
                ControlFlow::Continue(Visit::Descend)
            });
        assert_eq!(outcome, None);
        assert_eq!(refusals, alloc::vec![(alloc::vec!["a", "b"], "a")]);
    }

    #[test]
    fn skip_seals_a_route_without_extending_it() {
        let routes = all_routes(["app"], OpenPathsConfig::new(), diamond)
            .into_iter()
            .collect::<Vec<_>>();
        // Baseline: both sides reach merge. Now seal routes ending in left.
        assert_eq!(routes.iter().filter(|r| r.ends_with(&["merge"])).count(), 2);

        let mut sealed_routes = Vec::new();
        let outcome =
            open_breadth_first_paths::<_, ()>(["app"], OpenPathsConfig::new(), diamond, |event| {
                if let OpenPathsEvent::Route(route) = event {
                    sealed_routes.push(route.to_vec());
                    if *route.last().unwrap() == "left" {
                        return ControlFlow::Continue(Visit::Skip);
                    }
                }
                ControlFlow::Continue(Visit::Descend)
            });
        assert_eq!(outcome, None);
        let merged: Vec<_> = sealed_routes
            .iter()
            .filter(|route| *route.last().unwrap() == "merge")
            .collect();
        assert_eq!(merged, alloc::vec![&alloc::vec!["app", "right", "merge"]]);
    }

    #[test]
    fn max_depth_bounds_extension_not_discovery() {
        let routes = all_routes(["app"], OpenPathsConfig::new().with_max_depth(1), diamond);
        // Depth-1 routes are still discovered; nothing is extended past them.
        assert!(routes.contains(&alloc::vec!["app", "left"]));
        assert!(routes.contains(&alloc::vec!["app", "right"]));
        assert!(routes.iter().all(|route| route.len() <= 2));
    }

    #[test]
    fn duplicate_seeds_start_independent_routes() {
        let routes = all_routes(
            ["app", "app"],
            OpenPathsConfig::new().with_max_depth(0),
            diamond,
        );
        assert_eq!(routes, alloc::vec![alloc::vec!["app"], alloc::vec!["app"]]);
    }

    /// Every node leads to every other, in an order that puts the refused
    /// successors (those already on the route) before, between, and after the
    /// accepted ones as routes grow.
    fn complete_from(node: u32, out: &mut Vec<u32>) {
        out.extend((0..5).map(|offset| (node + 3 * offset) % 5));
    }

    /// Routes level by level, the plain way: no frontier, no recycled or moved
    /// route, a new vector per extension.
    fn levels(seed: u32, max_depth: usize, seal: impl Fn(&[u32]) -> bool) -> Vec<Vec<u32>> {
        let mut all = alloc::vec![alloc::vec![seed]];
        let mut level = alloc::vec![alloc::vec![seed]];
        for _ in 0..max_depth {
            let mut next_level = Vec::new();
            for route in &level {
                if seal(route) {
                    continue;
                }
                let mut successors = Vec::new();
                complete_from(*route.last().unwrap(), &mut successors);
                for node in successors.into_iter().filter(|node| !route.contains(node)) {
                    let mut extended = route.clone();
                    extended.push(node);
                    next_level.push(extended);
                }
            }
            all.extend(next_level.iter().cloned());
            level = next_level;
        }
        all
    }

    #[test]
    fn extended_routes_match_a_level_by_level_enumeration() {
        // Sealing routes that end in 4 makes some nodes' last successor sealed
        // and others' extended, so the route moved into the frontier and the
        // copies made for earlier successors both get exercised.
        let seal = |route: &[u32]| route.last() == Some(&4);
        for max_depth in [0, 1, 2, 4, 5] {
            let mut walked = Vec::new();
            let outcome = open_breadth_first_paths::<_, ()>(
                [0_u32],
                OpenPathsConfig::new().with_max_depth(max_depth),
                |node: &u32, out: &mut Vec<u32>| complete_from(*node, out),
                |event| {
                    let OpenPathsEvent::Route(route) = event else {
                        return ControlFlow::Continue(Visit::Descend);
                    };
                    walked.push(route.to_vec());
                    ControlFlow::Continue(if seal(route) {
                        Visit::Skip
                    } else {
                        Visit::Descend
                    })
                },
            );
            assert_eq!(outcome, None);
            assert_eq!(walked, levels(0, max_depth, seal), "max depth {max_depth}");
        }
    }
}
