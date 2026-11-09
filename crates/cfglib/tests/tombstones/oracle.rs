//! Reference algorithms over the plain topology.
//!
//! Each one is the textbook definition written the slow way (remove a node
//! and see what is still reachable, mutual reachability, backward path
//! search), so it shares no code and no idea of "slot" with the library.

use std::collections::{BTreeMap, BTreeSet};

use crate::topology::Topology;

/// The live adjacency of a topology, in edge insertion order.
pub struct Reference<'t> {
    topology: &'t Topology,
    successors: Vec<Vec<usize>>,
    predecessors: Vec<Vec<usize>>,
}

/// The post-dominator of a block: another block, or the virtual exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PostReference {
    Block(usize),
    Exit,
}

/// One natural loop: its blocks and the blocks that jump back to its header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopReference {
    pub body: BTreeSet<usize>,
    pub latches: BTreeSet<usize>,
}

/// The counts a metrics report states, computed by counting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetricsReference {
    pub node_count: usize,
    pub edge_count: usize,
    pub cyclomatic_complexity: usize,
    pub reachable_node_count: usize,
    pub unreachable_node_count: usize,
    pub exit_count: usize,
    pub instruction_count: usize,
}

impl<'t> Reference<'t> {
    pub fn new(topology: &'t Topology) -> Self {
        let mut successors = vec![Vec::new(); topology.slots];
        let mut predecessors = vec![Vec::new(); topology.slots];
        for (index, &(source, target)) in topology.edges.iter().enumerate() {
            if topology.is_live_edge(index) {
                successors[source].push(target);
                predecessors[target].push(source);
            }
        }
        Self {
            topology,
            successors,
            predecessors,
        }
    }

    pub fn live(&self) -> impl Iterator<Item = usize> + '_ {
        (0..self.topology.slots).filter(|&slot| self.topology.is_live(slot))
    }

    pub fn live_count(&self) -> usize {
        self.live().count()
    }

    pub fn edge_count(&self) -> usize {
        self.successors.iter().map(Vec::len).sum()
    }

    pub fn successors(&self, slot: usize) -> &[usize] {
        &self.successors[slot]
    }

    pub fn predecessors(&self, slot: usize) -> &[usize] {
        &self.predecessors[slot]
    }

    /// Nodes reachable from `from` without passing through `avoid`.
    pub fn reach(&self, from: usize, avoid: Option<usize>) -> BTreeSet<usize> {
        let mut seen = BTreeSet::new();
        if Some(from) == avoid {
            return seen;
        }
        let mut stack = vec![from];
        seen.insert(from);
        while let Some(node) = stack.pop() {
            for &next in &self.successors[node] {
                if Some(next) != avoid && seen.insert(next) {
                    stack.push(next);
                }
            }
        }
        seen
    }

    pub fn reachable(&self) -> BTreeSet<usize> {
        self.reach(0, None)
    }

    /// Live blocks with no successors.
    pub fn sinks(&self) -> BTreeSet<usize> {
        self.live()
            .filter(|&slot| self.successors[slot].is_empty())
            .collect()
    }

    /// Reachable blocks with no successors.
    pub fn exits(&self) -> BTreeSet<usize> {
        self.sinks()
            .intersection(&self.reachable())
            .copied()
            .collect()
    }

    /// A block dominates another when every path from the entry to it passes
    /// through the first: removing it leaves the second unreachable.
    pub fn dominates(&self, dominator: usize, node: usize) -> bool {
        if dominator == node {
            return true;
        }
        let reachable = self.reachable();
        reachable.contains(&dominator)
            && reachable.contains(&node)
            && !self.reach(0, Some(dominator)).contains(&node)
    }

    pub fn strict_dominators(&self, node: usize) -> BTreeSet<usize> {
        self.live()
            .filter(|&candidate| candidate != node && self.dominates(candidate, node))
            .collect()
    }

    /// The strict dominator every other strict dominator dominates.
    pub fn immediate_dominator(&self, node: usize) -> Option<usize> {
        let strict = self.strict_dominators(node);
        strict
            .iter()
            .copied()
            .find(|&candidate| strict.iter().all(|&other| self.dominates(other, candidate)))
    }

    /// Whether every path from `block` to an exit passes through
    /// `post_dominator`; exact only when `block` reaches an exit.
    pub fn post_dominates(&self, post_dominator: usize, block: usize) -> bool {
        if post_dominator == block {
            return true;
        }
        let reachable = self.reachable();
        if !reachable.contains(&post_dominator) || !reachable.contains(&block) {
            return false;
        }
        let exits = self.exits();
        let escapes = self.reach(block, Some(post_dominator));
        escapes.is_disjoint(&exits)
    }

    pub fn strict_post_dominators(&self, block: usize) -> BTreeSet<usize> {
        self.live()
            .filter(|&candidate| candidate != block && self.post_dominates(candidate, block))
            .collect()
    }

    pub fn immediate_post_dominator(&self, block: usize) -> PostReference {
        let strict = self.strict_post_dominators(block);
        strict
            .iter()
            .copied()
            .find(|&candidate| {
                strict
                    .iter()
                    .all(|&other| self.post_dominates(other, candidate))
            })
            .map_or(PostReference::Exit, PostReference::Block)
    }

    /// The blocks control-dependent on the arms leaving `controller`: those
    /// that post-dominate an arm's target without strictly post-dominating
    /// the controller.
    pub fn control_dependents(&self, controller: usize) -> BTreeSet<usize> {
        let mut dependents = BTreeSet::new();
        for &target in &self.successors[controller] {
            for block in self.live() {
                let strictly_after_controller =
                    block != controller && self.post_dominates(block, controller);
                if self.post_dominates(block, target) && !strictly_after_controller {
                    dependents.insert(block);
                }
            }
        }
        dependents
    }

    /// Strongly connected components by mutual reachability.
    pub fn components(&self) -> BTreeSet<BTreeSet<usize>> {
        let reach: BTreeMap<usize, BTreeSet<usize>> = self
            .live()
            .map(|node| (node, self.reach(node, None)))
            .collect();
        self.live()
            .map(|node| {
                reach[&node]
                    .iter()
                    .copied()
                    .filter(|other| reach[other].contains(&node))
                    .collect()
            })
            .collect()
    }

    pub fn has_self_loop(&self) -> bool {
        self.live()
            .any(|node| self.successors[node].contains(&node))
    }

    /// The edges between distinct components.
    pub fn condensation_edges(&self) -> BTreeSet<(BTreeSet<usize>, BTreeSet<usize>)> {
        let components = self.components();
        let component_of = |node: usize| {
            components
                .iter()
                .find(|component| component.contains(&node))
                .expect("every live node is in a component")
        };
        let mut edges = BTreeSet::new();
        for source in self.live() {
            for &target in &self.successors[source] {
                let (from, to) = (component_of(source), component_of(target));
                if from != to {
                    edges.insert((from.clone(), to.clone()));
                }
            }
        }
        edges
    }

    /// The edges whose target dominates their source, as `(tail, header)`.
    pub fn back_edges(&self) -> BTreeSet<(usize, usize)> {
        let reachable = self.reachable();
        reachable
            .iter()
            .flat_map(|&tail| self.successors[tail].iter().map(move |&head| (tail, head)))
            .filter(|&(tail, head)| self.dominates(head, tail))
            .collect()
    }

    /// Natural loops by header: the header plus every reachable block that
    /// can reach a latch without passing through the header.
    pub fn loops(&self) -> BTreeMap<usize, LoopReference> {
        let reachable = self.reachable();
        let mut loops: BTreeMap<usize, LoopReference> = BTreeMap::new();
        for (tail, header) in self.back_edges() {
            let entry = loops.entry(header).or_insert_with(|| LoopReference {
                body: BTreeSet::from([header]),
                latches: BTreeSet::new(),
            });
            entry.latches.insert(tail);
            let mut stack = vec![tail];
            while let Some(node) = stack.pop() {
                if node != header && reachable.contains(&node) && entry.body.insert(node) {
                    stack.extend(self.predecessors[node].iter().copied());
                }
            }
        }
        loops
    }

    /// A flow graph is reducible exactly when deleting its back edges leaves
    /// the reachable part acyclic.
    pub fn is_reducible(&self) -> bool {
        let reachable = self.reachable();
        let back = &self.back_edges();
        let mut indegree: BTreeMap<usize, usize> = reachable.iter().map(|&n| (n, 0)).collect();
        let forward = |node: usize| {
            self.successors[node]
                .iter()
                .copied()
                .filter(move |&next| !back.contains(&(node, next)))
                .collect::<Vec<_>>()
        };
        for &node in &reachable {
            for next in forward(node) {
                *indegree
                    .get_mut(&next)
                    .expect("a successor of a reachable node") += 1;
            }
        }
        let mut ready: Vec<usize> = indegree
            .iter()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(&node, _)| node)
            .collect();
        let mut removed = 0;
        while let Some(node) = ready.pop() {
            removed += 1;
            for next in forward(node) {
                let degree = indegree
                    .get_mut(&next)
                    .expect("a successor of a reachable node");
                *degree -= 1;
                if *degree == 0 {
                    ready.push(next);
                }
            }
        }
        removed == reachable.len()
    }

    pub fn metrics(&self) -> MetricsReference {
        let reachable = self.reachable();
        let reachable_edges: usize = reachable.iter().map(|&n| self.successors[n].len()).sum();
        MetricsReference {
            node_count: self.live_count(),
            edge_count: self.edge_count(),
            cyclomatic_complexity: reachable_edges + 2 - reachable.len(),
            reachable_node_count: reachable.len(),
            unreachable_node_count: self.live_count() - reachable.len(),
            exit_count: self.sinks().len(),
            instruction_count: self.live().map(|n| self.topology.blocks[n].len()).sum(),
        }
    }
}
