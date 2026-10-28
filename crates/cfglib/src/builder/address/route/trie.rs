//! Hash-consed tries of the paths of unwind routes.
//!
//! A node stands for the state of the runtime after it entered one funclet:
//! the ways that go on from there. Nodes with the same funclet and the same
//! ways are one node, so two contexts that continue in the same way share one
//! instance of the funclet, and contexts that continue differently do not.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use super::{AddressSpace, UnwindEnd, UnwindRoute, UnwindStep};

/// The position of a node in [`Tries::nodes`].
pub(super) type NodeId = usize;

/// One state of the runtime.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct Node<A> {
    /// The funclet that the runtime entered, or `None` for the root of a
    /// route: the protected instruction.
    pub(super) funclet: Option<A>,
    /// The next funclets: the calls before the entry, and the node after it.
    /// Sorted.
    pub(super) edges: Vec<(Vec<u32>, NodeId)>,
    /// The ends: the calls before the end, and the end. Sorted.
    pub(super) ends: Vec<(Vec<u32>, UnwindEnd<A>)>,
}

/// All nodes of all routes.
#[derive(Debug)]
pub(super) struct Tries<A> {
    /// The nodes in creation order. A node follows its children.
    pub(super) nodes: Vec<Node<A>>,
    /// The root node of each route, in route order.
    pub(super) roots: Vec<NodeId>,
    index: BTreeMap<Node<A>, NodeId>,
}

type Remaining<'p, A> = (&'p [UnwindStep<A>], UnwindEnd<A>);

impl<A: AddressSpace> Tries<A> {
    pub(super) fn build(routes: &[UnwindRoute<A>]) -> Self {
        let mut tries = Self {
            nodes: Vec::new(),
            roots: Vec::with_capacity(routes.len()),
            index: BTreeMap::new(),
        };
        for route in routes {
            let paths: Vec<Remaining<'_, A>> = route
                .paths
                .iter()
                .map(|path| (path.steps.as_slice(), path.end))
                .collect();
            let root = tries.intern(None, &paths);
            tries.roots.push(root);
        }
        tries
    }

    /// The first route to reach each node, by node.
    pub(super) fn first_routes(&self) -> Vec<usize> {
        let mut first = vec![usize::MAX; self.nodes.len()];
        for (route, &root) in self.roots.iter().enumerate() {
            let mut pending = Vec::from([root]);
            while let Some(node) = pending.pop() {
                if first[node] != usize::MAX {
                    continue;
                }
                first[node] = route;
                pending.extend(self.nodes[node].edges.iter().map(|&(_, child)| child));
            }
        }
        first
    }

    /// Builds the node for the runtime state that has these remaining paths.
    fn intern(&mut self, funclet: Option<A>, paths: &[Remaining<'_, A>]) -> NodeId {
        let mut ends = BTreeSet::new();
        let mut groups: BTreeMap<(Vec<u32>, A), Vec<Remaining<'_, A>>> = BTreeMap::new();
        for &(steps, end) in paths {
            let next = steps
                .iter()
                .enumerate()
                .find_map(|(position, step)| match step {
                    UnwindStep::Funclet(address) => Some((position, *address)),
                    UnwindStep::Call(_) => None,
                });
            match next {
                None => {
                    ends.insert((calls(steps), end));
                }
                Some((position, address)) => groups
                    .entry((calls(&steps[..position]), address))
                    .or_default()
                    .push((&steps[position + 1..], end)),
            }
        }
        let mut edges: Vec<(Vec<u32>, NodeId)> = groups
            .into_iter()
            .map(|((calls, address), rest)| (calls, self.intern(Some(address), &rest)))
            .collect();
        edges.sort_unstable();
        let node = Node {
            funclet,
            edges,
            ends: ends.into_iter().collect(),
        };
        if let Some(&existing) = self.index.get(&node) {
            return existing;
        }
        let id = self.nodes.len();
        self.nodes.push(node.clone());
        self.index.insert(node, id);
        id
    }
}

/// The tags of the calls in `steps`, in order.
fn calls<A>(steps: &[UnwindStep<A>]) -> Vec<u32> {
    steps
        .iter()
        .filter_map(|step| match step {
            UnwindStep::Call(tag) => Some(*tag),
            UnwindStep::Funclet(_) => None,
        })
        .collect()
}
