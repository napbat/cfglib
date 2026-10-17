//! Brute-force reference answers over the successor records of a function.
//!
//! The graph is read from `FunctionView::successors` alone, never from the
//! graph view that the algorithms under test consume. Dominators come from
//! "delete a block and see what the entry still reaches", components from
//! mutual reachability, post-dominators from deleting a block and asking
//! whether an exit is still reachable, and control dependence from its
//! definition on arms.

use std::collections::BTreeSet;

use cfglib::ir::mlil;
use cfglib::{DenseId, GraphView, RootedView};

/// The control-flow graph of a function, one successor list per slot.
pub struct Cfg {
    pub successors: Vec<Vec<usize>>,
    pub live: Vec<bool>,
    pub root: usize,
}

pub fn extract<D: mlil::Dialect, F: mlil::FunctionView<D>>(function: &F) -> Cfg {
    let bound = GraphView::node_bound(function);
    let mut cfg = Cfg {
        successors: vec![Vec::new(); bound],
        live: vec![false; bound],
        root: RootedView::root(function).index(),
    };
    for block in GraphView::node_ids(function) {
        cfg.live[block.index()] = true;
        cfg.successors[block.index()] = mlil::FunctionView::<D>::successors(function, block)
            .map(|successor| successor.target.index())
            .collect();
    }
    cfg
}

impl Cfg {
    pub fn live_nodes(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.live.len()).filter(|&node| self.live[node])
    }

    /// The nodes reachable from `start` when `skip` is deleted.
    fn reach_from(&self, start: usize, skip: Option<usize>) -> Vec<bool> {
        let mut seen = vec![false; self.live.len()];
        if skip == Some(start) {
            return seen;
        }
        let mut stack = vec![start];
        seen[start] = true;
        while let Some(node) = stack.pop() {
            for &next in &self.successors[node] {
                if Some(next) != skip && !seen[next] {
                    seen[next] = true;
                    stack.push(next);
                }
            }
        }
        seen
    }

    pub fn reachable(&self) -> Vec<bool> {
        self.reach_from(self.root, None)
    }

    /// The dominators of every reachable node, itself included: `d`
    /// dominates `n` when deleting `d` leaves `n` unreachable.
    pub fn dominators(&self) -> Vec<BTreeSet<usize>> {
        let reachable = self.reachable();
        (0..self.live.len())
            .map(|node| {
                if !reachable[node] {
                    return BTreeSet::new();
                }
                self.live_nodes()
                    .filter(|&candidate| {
                        candidate == node || !self.reach_from(self.root, Some(candidate))[node]
                    })
                    .collect()
            })
            .collect()
    }

    /// The strict dominator of `node` that every other strict dominator
    /// dominates.
    pub fn immediate_dominator(dominators: &[BTreeSet<usize>], node: usize) -> Option<usize> {
        dominators[node]
            .iter()
            .copied()
            .filter(|&candidate| candidate != node)
            .find(|&candidate| {
                dominators[node]
                    .iter()
                    .filter(|&&other| other != node && other != candidate)
                    .all(|other| dominators[candidate].contains(other))
            })
    }

    /// The groups of mutually reachable live nodes.
    pub fn components(&self) -> Vec<Vec<usize>> {
        let reach: Vec<Vec<bool>> = (0..self.live.len())
            .map(|node| {
                if self.live[node] {
                    self.reach_from(node, None)
                } else {
                    vec![false; self.live.len()]
                }
            })
            .collect();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for node in self.live_nodes() {
            match groups
                .iter_mut()
                .find(|group| reach[group[0]][node] && reach[node][group[0]])
            {
                Some(group) => group.push(node),
                None => groups.push(vec![node]),
            }
        }
        groups
    }

    /// Whether every cycle that the entry reaches is a natural loop: the edges
    /// whose head dominates their tail are the back edges, and the graph
    /// without them has no cycle.
    pub fn is_reducible(&self) -> bool {
        let dominators = self.dominators();
        let reachable = self.reachable();
        let forward: Vec<Vec<usize>> = (0..self.live.len())
            .map(|node| {
                if !reachable[node] {
                    return Vec::new();
                }
                self.successors[node]
                    .iter()
                    .copied()
                    .filter(|head| !dominators[node].contains(head))
                    .collect()
            })
            .collect();
        let without_back_edges = Self {
            successors: forward,
            live: self.live.clone(),
            root: self.root,
        };
        without_back_edges.live_nodes().all(|node| {
            // A node on a cycle reaches itself through at least one edge.
            !without_back_edges.successors[node]
                .iter()
                .any(|&next| without_back_edges.reach_from(next, None)[node])
        })
    }

    /// Whether some path leads from `from` to an exit.
    pub fn can_reach_exit(&self, from: usize) -> bool {
        let seen = self.reach_from(from, None);
        self.exits().iter().any(|&exit| seen[exit])
    }

    /// The dominance frontier of every reachable node: `y` is in the frontier
    /// of `x` when `x` dominates a predecessor of `y` and does not strictly
    /// dominate `y`.
    pub fn frontiers(&self) -> Vec<BTreeSet<usize>> {
        let dominators = self.dominators();
        let reachable = self.reachable();
        (0..self.live.len())
            .map(|x| {
                if !reachable[x] {
                    return BTreeSet::new();
                }
                self.live_nodes()
                    .filter(|&y| {
                        let dominates_a_predecessor = self
                            .live_nodes()
                            .filter(|&p| self.successors[p].contains(&y))
                            .any(|p| dominators[p].contains(&x));
                        let strictly_dominates = x != y && dominators[y].contains(&x);
                        dominates_a_predecessor && !strictly_dominates
                    })
                    .collect()
            })
            .collect()
    }

    /// The reachable nodes without successors.
    pub fn exits(&self) -> Vec<usize> {
        let reachable = self.reachable();
        self.live_nodes()
            .filter(|&node| reachable[node] && self.successors[node].is_empty())
            .collect()
    }

    /// The post-dominators of every reachable node, itself included, when
    /// every reachable node can reach an exit. `b` post-dominates `a` when
    /// deleting `b` leaves `a` with no exit to reach. Otherwise `None`.
    pub fn post_dominators(&self) -> Option<Vec<BTreeSet<usize>>> {
        let reachable = self.reachable();
        let exits = self.exits();
        let reaches_exit = |from: usize, skip: Option<usize>| {
            let seen = self.reach_from(from, skip);
            exits.iter().any(|&exit| seen[exit])
        };
        if !self
            .live_nodes()
            .all(|node| !reachable[node] || reaches_exit(node, None))
        {
            return None;
        }
        Some(
            (0..self.live.len())
                .map(|node| {
                    if !reachable[node] {
                        return BTreeSet::new();
                    }
                    self.live_nodes()
                        .filter(|&candidate| {
                            reachable[candidate]
                                && (candidate == node || !reaches_exit(node, Some(candidate)))
                        })
                        .collect()
                })
                .collect(),
        )
    }

    /// The pairs `(a, b)` where `b` depends on an arm `a -> t`: `b`
    /// post-dominates `t` and does not strictly post-dominate `a`.
    pub fn control_dependence(&self, post: &[BTreeSet<usize>]) -> BTreeSet<(usize, usize)> {
        let reachable = self.reachable();
        let mut pairs = BTreeSet::new();
        for a in self.live_nodes().filter(|&node| reachable[node]) {
            for &t in &self.successors[a] {
                for b in self.live_nodes().filter(|&node| reachable[node]) {
                    let post_dominates_arm = post[t].contains(&b);
                    let strictly_post_dominates_a = b != a && post[a].contains(&b);
                    if post_dominates_arm && !strictly_post_dominates_a {
                        pairs.insert((a, b));
                    }
                }
            }
        }
        pairs
    }
}
