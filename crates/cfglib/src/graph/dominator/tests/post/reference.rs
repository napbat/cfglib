//! What a graph's post-dominators are by definition, found by brute force.
//!
//! The cut is derived here without the production code: strongly connected
//! components are read off a reachability matrix, and the rest of the rule is
//! applied literally. A block `a` then post-dominates `b` exactly when the
//! virtual exit is unreachable from `b` once `a` is removed from the augmented
//! graph, so a disagreement convicts the implementation.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec;
use alloc::vec::Vec;

use crate::graph::dominator::tests::support::{Digraph, node};
use crate::graph::dominator::{FrontierScratch, PostDominatorTree, PostNode};
use crate::graph::view::GraphView;

/// The post-dominator facts of one graph rooted at node 0.
pub(super) struct PostReference {
    reachable: Vec<bool>,
    predecessors: Vec<Vec<usize>>,
    exits: Vec<usize>,
    reaches_exit: Vec<bool>,
    reaches_trapped: Vec<bool>,
    cut_roots: Vec<usize>,
    /// `strict[a][b]`: `a` and `b` are distinct reachable blocks and every path
    /// from `b` to the virtual exit passes `a`, on the graph with the cut.
    strict: Vec<Vec<bool>>,
    /// The same relation on the graph that joins only the real exits to the
    /// virtual exit.
    plain: Vec<Vec<bool>>,
    level: Vec<usize>,
    preorder: Vec<usize>,
    /// `frontiers[b]`: the frontier of `b` on the reverse graph, in preorder.
    frontiers: Vec<Vec<usize>>,
}

impl PostReference {
    /// The facts of `graph`.
    pub(super) fn new(graph: &Digraph) -> Self {
        let count = graph.node_bound();
        let successors: Vec<Vec<usize>> = (0..count)
            .map(|from| graph.successors(node(from)).map(|to| to as usize).collect())
            .collect();
        let predecessors: Vec<Vec<usize>> = (0..count)
            .map(|to| {
                graph
                    .predecessors(node(to))
                    .map(|from| from as usize)
                    .collect()
            })
            .collect();
        let reach = closure(&successors);
        let reachable = reach[0].clone();
        let exits: Vec<usize> = (0..count)
            .filter(|&block| reachable[block] && successors[block].is_empty())
            .collect();
        let reaches_exit: Vec<bool> = (0..count)
            .map(|block| reachable[block] && exits.iter().any(|&exit| reach[block][exit]))
            .collect();
        let trapped: Vec<bool> = (0..count)
            .map(|block| reachable[block] && !reaches_exit[block])
            .collect();
        let reaches_trapped: Vec<bool> = (0..count)
            .map(|block| (0..count).any(|other| trapped[other] && reach[block][other]))
            .collect();

        let mut cut_roots = Vec::new();
        for start in (0..count).filter(|&block| trapped[block]) {
            let component: Vec<usize> = (0..count)
                .filter(|&other| trapped[other] && reach[start][other] && reach[other][start])
                .collect();
            let is_sink = component.iter().all(|&member| {
                successors[member]
                    .iter()
                    .all(|next| component.contains(next))
            });
            if start != component[0] || !is_sink {
                continue;
            }
            let entries: Vec<usize> = component
                .iter()
                .copied()
                .filter(|&member| {
                    member == 0
                        || predecessors[member]
                            .iter()
                            .any(|before| reachable[*before] && !component.contains(before))
                })
                .collect();
            cut_roots.extend(
                component
                    .iter()
                    .copied()
                    .filter(|&member| successors[member].iter().any(|next| entries.contains(next))),
            );
        }
        cut_roots.sort_unstable();

        let with_cut: Vec<usize> = exits.iter().chain(&cut_roots).copied().collect();
        let augmented = augment(&successors, &reachable, &with_cut);
        assert!(
            (0..count)
                .all(|block| !reachable[block] || reaches_virtual(&augmented, block, usize::MAX)),
            "the cut left a reachable block without a path to the virtual exit"
        );
        let strict = strict_table(&augmented, &reachable);
        let plain = strict_table(&augment(&successors, &reachable, &exits), &reachable);
        let level = (0..count)
            .map(|block| (0..count).filter(|&other| strict[other][block]).count())
            .collect();
        let mut reference = Self {
            reachable,
            predecessors,
            exits,
            reaches_exit,
            reaches_trapped,
            cut_roots,
            strict,
            plain,
            level,
            preorder: Vec::new(),
            frontiers: Vec::new(),
        };
        reference.preorder = reference.compute_preorder();
        reference.frontiers = (0..count)
            .map(|block| reference.compute_frontier(block))
            .collect();
        reference
    }

    fn node_count(&self) -> usize {
        self.reachable.len()
    }

    /// Whether the entry reaches `block`.
    pub(super) fn contains(&self, block: usize) -> bool {
        self.reachable[block]
    }

    /// Whether some path leads from `block` to an exit.
    pub(super) fn reaches_exit(&self, block: usize) -> bool {
        self.reaches_exit[block]
    }

    /// Whether `block` reaches a block that cannot reach an exit.
    pub(super) fn reaches_trapped(&self, block: usize) -> bool {
        self.reaches_trapped[block]
    }

    pub(super) fn exits(&self) -> &[usize] {
        &self.exits
    }

    pub(super) fn cut_roots(&self) -> &[usize] {
        &self.cut_roots
    }

    /// Cfglib's answer to "does `post_dominator` post-dominate `block`":
    /// every block post-dominates itself, and a pair with an unreachable member
    /// does not post-dominate otherwise.
    pub(super) fn post_dominates(&self, post_dominator: usize, block: usize) -> bool {
        post_dominator == block || self.strict[post_dominator][block]
    }

    /// The same question on the graph without the cut, which only the blocks
    /// that reach an exit may ask.
    pub(super) fn plain_post_dominates(&self, post_dominator: usize, block: usize) -> bool {
        post_dominator == block || self.plain[post_dominator][block]
    }

    /// The immediate post-dominator: `None` outside the tree.
    pub(super) fn ipdom(&self, block: usize) -> Option<PostNode<usize>> {
        if !self.reachable[block] {
            return None;
        }
        let strict: Vec<usize> = (0..self.node_count())
            .filter(|&other| self.strict[other][block])
            .collect();
        let nearest = strict.iter().copied().find(|&candidate| {
            strict
                .iter()
                .all(|&other| other == candidate || self.strict[other][candidate])
        });
        Some(nearest.map_or(PostNode::Exit, PostNode::Node))
    }

    /// The deepest node that post-dominates every reachable block of `blocks`:
    /// `None` when none is reachable.
    pub(super) fn nearest_common(&self, blocks: &[usize]) -> Option<PostNode<usize>> {
        let contained: Vec<usize> = blocks
            .iter()
            .copied()
            .filter(|&block| self.reachable[block])
            .collect();
        if contained.is_empty() {
            return None;
        }
        let deepest = (0..self.node_count())
            .filter(|&candidate| {
                self.reachable[candidate]
                    && contained
                        .iter()
                        .all(|&block| self.post_dominates(candidate, block))
            })
            .max_by_key(|&candidate| self.level[candidate]);
        Some(deepest.map_or(PostNode::Exit, PostNode::Node))
    }

    /// The nodes whose immediate post-dominator is `parent`, in ascending
    /// index.
    pub(super) fn children(&self, parent: PostNode<usize>) -> Vec<usize> {
        (0..self.node_count())
            .filter(|&block| self.ipdom(block) == Some(parent))
            .collect()
    }

    /// The blocks `block` post-dominates, in ascending index, for a reachable
    /// block.
    pub(super) fn subtree(&self, block: usize) -> Vec<usize> {
        (0..self.node_count())
            .filter(|&other| {
                self.reachable[block] && self.reachable[other] && self.post_dominates(block, other)
            })
            .collect()
    }

    /// The forest's preorder, with roots and children in ascending index.
    pub(super) fn preorder(&self) -> &[usize] {
        &self.preorder
    }

    fn compute_preorder(&self) -> Vec<usize> {
        let mut order = Vec::new();
        let mut stack: Vec<usize> = self.children(PostNode::Exit).into_iter().rev().collect();
        while let Some(current) = stack.pop() {
            order.push(current);
            stack.extend(self.children(PostNode::Node(current)).into_iter().rev());
        }
        order
    }

    /// The dominance frontier of `block` on the reverse graph, in preorder.
    pub(super) fn frontier(&self, block: usize) -> &[usize] {
        &self.frontiers[block]
    }

    fn compute_frontier(&self, block: usize) -> Vec<usize> {
        let sources: BTreeSet<usize> = self
            .subtree(block)
            .into_iter()
            .flat_map(|member| self.predecessors[member].iter().copied())
            .filter(|&source| {
                self.reachable[source] && !(source != block && self.post_dominates(block, source))
            })
            .collect();
        self.in_preorder(&sources)
    }

    /// The least set that holds the frontier of every reachable block of
    /// `blocks` and of every member, keeping only members `live` accepts. A
    /// block `live` rejects is not a member and is not expanded.
    pub(super) fn iterated_frontier(
        &self,
        blocks: &[usize],
        live: Option<&dyn Fn(usize) -> bool>,
    ) -> Vec<usize> {
        let mut members = BTreeSet::new();
        let mut pending = blocks.to_vec();
        while let Some(current) = pending.pop() {
            for &source in self.frontier(current) {
                if live.is_none_or(|accepts| accepts(source)) && members.insert(source) {
                    pending.push(source);
                }
            }
        }
        self.in_preorder(&members)
    }

    fn in_preorder(&self, members: &BTreeSet<usize>) -> Vec<usize> {
        self.preorder
            .iter()
            .copied()
            .filter(|block| members.contains(block))
            .collect()
    }
}

/// `reach[from][to]`: a path of any length, including zero, joins them.
fn closure(successors: &[Vec<usize>]) -> Vec<Vec<bool>> {
    (0..successors.len())
        .map(|from| {
            let mut seen = vec![false; successors.len()];
            seen[from] = true;
            let mut stack = vec![from];
            while let Some(current) = stack.pop() {
                for &next in &successors[current] {
                    if !seen[next] {
                        seen[next] = true;
                        stack.push(next);
                    }
                }
            }
            seen
        })
        .collect()
}

/// The edges among the reachable blocks, plus an edge from each of `joined` to
/// a virtual exit at index `successors.len()`.
fn augment(successors: &[Vec<usize>], reachable: &[bool], joined: &[usize]) -> Vec<Vec<usize>> {
    let mut adjacency: Vec<Vec<usize>> = (0..successors.len())
        .map(|block| {
            if reachable[block] {
                successors[block].clone()
            } else {
                Vec::new()
            }
        })
        .collect();
    adjacency.push(Vec::new());
    for &block in joined {
        adjacency[block].push(successors.len());
    }
    adjacency
}

/// Whether the virtual exit, the last node, is reachable from `from` along
/// paths that avoid `removed`.
fn reaches_virtual(adjacency: &[Vec<usize>], from: usize, removed: usize) -> bool {
    let virtual_exit = adjacency.len() - 1;
    let mut seen = vec![false; adjacency.len()];
    seen[from] = true;
    let mut stack = vec![from];
    while let Some(current) = stack.pop() {
        if current == virtual_exit {
            return true;
        }
        for &next in &adjacency[current] {
            if next != removed && !seen[next] {
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    false
}

/// `table[a][b]`: `a` and `b` are distinct reachable blocks and removing `a`
/// cuts `b` off from the virtual exit.
fn strict_table(adjacency: &[Vec<usize>], reachable: &[bool]) -> Vec<Vec<bool>> {
    (0..reachable.len())
        .map(|post_dominator| {
            (0..reachable.len())
                .map(|block| {
                    post_dominator != block
                        && reachable[post_dominator]
                        && reachable[block]
                        && !reaches_virtual(adjacency, block, post_dominator)
                })
                .collect()
        })
        .collect()
}

/// `parent` as the tree names it.
pub(super) fn post_node(parent: PostNode<usize>) -> PostNode<u32> {
    match parent {
        PostNode::Node(block) => PostNode::Node(node(block)),
        PostNode::Exit => PostNode::Exit,
    }
}

/// `members` as the tree's identities.
pub(super) fn blocks(members: &[usize]) -> Vec<u32> {
    members.iter().copied().map(node).collect()
}

/// Check every query of `post` against `reference`, for a graph described by
/// `context`.
pub(super) fn assert_matches(
    post: &PostDominatorTree<u32>,
    reference: &PostReference,
    context: &dyn core::fmt::Debug,
) {
    let count = reference.node_count();
    assert_eq!(
        post.exits(),
        blocks(reference.exits()),
        "exits of {context:?}"
    );
    assert_eq!(
        post.cut_roots(),
        blocks(reference.cut_roots()),
        "cut roots of {context:?}"
    );
    assert_eq!(
        post.preorder(),
        blocks(reference.preorder()),
        "preorder of {context:?}"
    );
    assert_eq!(
        post.children(PostNode::Exit).collect::<Vec<_>>(),
        blocks(&reference.children(PostNode::Exit)),
        "roots of {context:?}"
    );

    for block in 0..count {
        let id = node(block);
        assert_eq!(
            post.contains(id),
            reference.contains(block),
            "{block} in {context:?}"
        );
        assert_eq!(
            post.reaches_exit(id),
            reference.reaches_exit(block),
            "reaches_exit({block}) in {context:?}"
        );
        assert_eq!(
            post.ipdom(id),
            reference.ipdom(block).map(post_node),
            "ipdom({block}) in {context:?}"
        );
        let mut subtree = post.subtree(id).to_vec();
        assert_eq!(
            subtree.first().copied(),
            reference.contains(block).then_some(id)
        );
        subtree.sort_unstable();
        assert_eq!(
            subtree,
            blocks(&reference.subtree(block)),
            "subtree of {block} in {context:?}"
        );
        assert_eq!(
            post.children(PostNode::Node(id)).collect::<Vec<_>>(),
            blocks(&reference.children(PostNode::Node(block))),
            "children of {block} in {context:?}"
        );

        for other in 0..count {
            let other_id = node(other);
            let expected = reference.post_dominates(block, other);
            assert_eq!(
                post.post_dominates(id, other_id),
                expected,
                "{block} post-dominates {other} in {context:?}"
            );
            assert_eq!(
                post.strictly_post_dominates(id, other_id),
                block != other && expected,
                "{block} strictly post-dominates {other} in {context:?}"
            );
            let common = (reference.contains(block) && reference.contains(other))
                .then(|| reference.nearest_common(&[block, other]).map(post_node))
                .flatten();
            assert_eq!(
                post.nearest_common_post_dominator(id, other_id),
                common,
                "nearest common post-dominator of {block} and {other} in {context:?}"
            );
        }
    }
}

/// Check what the cut may and may not change on a block that reaches an exit.
///
/// Its facts are never more than the facts of the plain graph that joins only
/// the real exits, because the cut adds paths and a path can only remove a
/// post-dominator. When no path from the block leads into an exit-free block,
/// the cut adds no path that starts at it, and the facts are exactly the plain
/// ones.
pub(super) fn assert_contract(
    post: &PostDominatorTree<u32>,
    reference: &PostReference,
    context: &dyn core::fmt::Debug,
) {
    for block in (0..reference.node_count()).filter(|&block| reference.reaches_exit(block)) {
        for other in 0..reference.node_count() {
            let fact = post.post_dominates(node(other), node(block));
            let plain = reference.plain_post_dominates(other, block);
            assert!(
                plain || !fact,
                "{other} post-dominates {block} only through the cut in {context:?}"
            );
            if !reference.reaches_trapped(block) {
                assert_eq!(
                    fact, plain,
                    "the cut changed {other} post-dominating {block} in {context:?}"
                );
            }
        }
    }
}

/// Every set of blocks of a graph with `count` blocks, each in ascending
/// index.
pub(super) fn all_subsets(count: usize) -> Vec<Vec<usize>> {
    (0..1_u32 << count)
        .map(|subset| (0..count).filter(|&bit| (subset >> bit) & 1 == 1).collect())
        .collect()
}

/// Check the frontier of every block, and the iterated frontier of each of
/// `sets` pruned three ways, against the reference.
///
/// The frontier on the reverse graph is where a block's control dependences
/// come from. The pruning keeps every block, only even ones, or only odd ones,
/// and every set is named twice to show that a repeated definition changes
/// nothing.
pub(super) fn assert_frontiers(
    post: &PostDominatorTree<u32>,
    graph: &Digraph,
    reference: &PostReference,
    scratch: &mut FrontierScratch,
    sets: &[Vec<usize>],
    context: &dyn core::fmt::Debug,
) {
    let mut out = Vec::new();
    for block in 0..reference.node_count() {
        post.frontier(graph, node(block), &mut out);
        assert_eq!(
            out,
            blocks(reference.frontier(block)),
            "frontier of {block} in {context:?}"
        );
    }

    for members in sets {
        let twice: Vec<u32> = blocks(members).repeat(2);
        for parity in [None, Some(0_u32), Some(1)] {
            let accepts_id = parity.map(|parity| move |block: u32| block % 2 == parity);
            let accepts_index = parity.map(|parity| move |block: usize| node(block) % 2 == parity);
            post.iterated_frontier(
                graph,
                scratch,
                twice.iter().copied(),
                accepts_id
                    .as_ref()
                    .map(|accepts| accepts as &dyn Fn(u32) -> bool),
                &mut out,
            );
            let expected = reference.iterated_frontier(
                members,
                accepts_index
                    .as_ref()
                    .map(|accepts| accepts as &dyn Fn(usize) -> bool),
            );
            assert_eq!(
                out,
                blocks(&expected),
                "iterated frontier of {members:?} pruned to {parity:?} in {context:?}"
            );
        }
    }
}
