//! Brute-force references for the search disciplines, written from the documentation.

use std::collections::VecDeque;

use cfglib::{Graph, NodeId};

use crate::Lcg;

/// A directed graph as plain successor lists in adjacency order.
#[derive(Clone)]
pub(crate) struct Table {
    pub(crate) successors: Vec<Vec<usize>>,
}

impl Table {
    /// Random edges, so parallel edges and self loops occur.
    pub(crate) fn random(rng: &mut Lcg, nodes: usize, edges: usize) -> Self {
        let mut successors = vec![Vec::new(); nodes];
        for _ in 0..edges {
            let source = rng.below(nodes);
            successors[source].push(rng.below(nodes));
        }
        Self { successors }
    }

    pub(crate) fn node_count(&self) -> usize {
        self.successors.len()
    }

    /// The same graph as a store whose adjacency order is this table's.
    pub(crate) fn graph(&self) -> Graph<(), ()> {
        let mut graph = Graph::new();
        let nodes: Vec<NodeId> = (0..self.node_count()).map(|_| graph.add_node(())).collect();
        for (source, targets) in self.successors.iter().enumerate() {
            for &target in targets {
                graph.add_edge(nodes[source], nodes[target], ());
            }
        }
        graph
    }

    /// The predecessor lists, in the order a store records the edges that arrive at a node: by
    /// source, then by position in the source's list.
    pub(crate) fn reversed(&self) -> Self {
        let mut successors = vec![Vec::new(); self.node_count()];
        for (source, targets) in self.successors.iter().enumerate() {
            for &target in targets {
                successors[target].push(source);
            }
        }
        Self { successors }
    }
}

/// Every `(node, depth)` a walk visits, in visiting order.
pub(crate) type Visits = Vec<(usize, usize)>;

/// One discipline of `search`: an optional depth bound and a pruning predicate.
pub(crate) struct Walk<'a> {
    table: &'a Table,
    limit: Option<usize>,
    prune: &'a dyn Fn(usize) -> bool,
    visits: Visits,
}

impl<'a> Walk<'a> {
    pub(crate) fn new(
        table: &'a Table,
        limit: Option<usize>,
        prune: &'a dyn Fn(usize) -> bool,
    ) -> Self {
        Self {
            table,
            limit,
            prune,
            visits: Vec::new(),
        }
    }

    /// A node is expanded unless the visitor pruned it or the walk is at its depth bound.
    fn expands(&self, node: usize, depth: usize) -> bool {
        !(self.prune)(node) && self.limit.is_none_or(|limit| depth < limit)
    }

    /// Recursive depth-first with global marks: the first successor is explored first.
    pub(crate) fn depth_first(mut self, seeds: &[usize]) -> Visits {
        let mut seen = vec![false; self.table.node_count()];
        for &seed in seeds {
            self.enter_marked(seed, 0, &mut seen);
        }
        self.visits
    }

    fn enter_marked(&mut self, node: usize, depth: usize, seen: &mut [bool]) {
        if seen[node] {
            return;
        }
        seen[node] = true;
        self.visits.push((node, depth));
        if self.expands(node, depth) {
            let table = self.table;
            for &next in &table.successors[node] {
                self.enter_marked(next, depth + 1, seen);
            }
        }
    }

    /// Recursive depth-first with per-path marks: one visit per distinct simple route.
    pub(crate) fn depth_first_routes(mut self, seeds: &[usize]) -> Visits {
        for &seed in seeds {
            let mut on_route = vec![false; self.table.node_count()];
            self.enter_route(seed, 0, &mut on_route);
        }
        self.visits
    }

    fn enter_route(&mut self, node: usize, depth: usize, on_route: &mut [bool]) {
        if on_route[node] {
            return;
        }
        on_route[node] = true;
        self.visits.push((node, depth));
        if self.expands(node, depth) {
            let table = self.table;
            for &next in &table.successors[node] {
                self.enter_route(next, depth + 1, on_route);
            }
        }
        on_route[node] = false;
    }

    /// Breadth-first with global marks set when a node is queued.
    pub(crate) fn breadth_first(mut self, seeds: &[usize]) -> Visits {
        let mut seen = vec![false; self.table.node_count()];
        let mut queue = VecDeque::new();
        for &seed in seeds {
            if !seen[seed] {
                seen[seed] = true;
                queue.push_back((seed, 0));
            }
        }
        while let Some((node, depth)) = queue.pop_front() {
            self.visits.push((node, depth));
            if !self.expands(node, depth) {
                continue;
            }
            for &next in &self.table.successors[node] {
                if !seen[next] {
                    seen[next] = true;
                    queue.push_back((next, depth + 1));
                }
            }
        }
        self.visits
    }
}

/// The nodes a walk visited, ignoring order and depth.
pub(crate) fn nodes_of(visits: &Visits) -> Vec<usize> {
    visits.iter().map(|&(node, _)| node).collect()
}

/// Never prune.
pub(crate) fn keep_all(_: usize) -> bool {
    false
}

/// Prune every fifth node.
pub(crate) fn prune_multiples_of_five(node: usize) -> bool {
    node.is_multiple_of(5)
}

impl Table {
    /// The table after the given nodes are removed: their lists are emptied and every edge into
    /// them disappears, while every other list keeps its order.
    pub(crate) fn without(&self, removed: &[usize]) -> Self {
        let successors = self
            .successors
            .iter()
            .enumerate()
            .map(|(node, targets)| {
                if removed.contains(&node) {
                    return Vec::new();
                }
                targets
                    .iter()
                    .copied()
                    .filter(|target| !removed.contains(target))
                    .collect()
            })
            .collect();
        Self { successors }
    }
}

/// Every simple route from `seed` with at most `limit` hops, by recursion: a route, then each of
/// its extensions in turn. Parallel edges give repeated routes.
pub(crate) fn simple_routes(table: &Table, seed: usize, limit: Option<usize>) -> Vec<Vec<usize>> {
    let mut routes = Vec::new();
    extend_route(table, &mut vec![seed], limit, &mut routes);
    routes
}

fn extend_route(
    table: &Table,
    route: &mut Vec<usize>,
    limit: Option<usize>,
    out: &mut Vec<Vec<usize>>,
) {
    out.push(route.clone());
    if limit.is_some_and(|limit| route.len() > limit) {
        return;
    }
    let last = *route.last().expect("a route is never empty");
    for &next in &table.successors[last] {
        if !route.contains(&next) {
            route.push(next);
            extend_route(table, route, limit, out);
            route.pop();
        }
    }
}

/// The successor closure of an open walk over a table.
pub(crate) fn table_successors(table: &Table) -> impl FnMut(&usize, &mut Vec<usize>) + '_ {
    move |node: &usize, out: &mut Vec<usize>| out.extend_from_slice(&table.successors[*node])
}

/// A chain `0 -> 1 -> ... -> length - 1`: as deep as a walk over `length` nodes can go.
pub(crate) fn chain(length: usize) -> Table {
    let successors = (0..length)
        .map(|node| {
            if node + 1 < length {
                vec![node + 1]
            } else {
                Vec::new()
            }
        })
        .collect();
    Table { successors }
}
