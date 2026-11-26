//! n:m correspondences between the blocks of two CFGs: [`RegionDiff`].

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec;
use alloc::vec::Vec;

use super::{BlockFacts, BlockFingerprint, Side, edges_preserved, match_by_wiring};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;

/// The kinds of edge that can join two blocks into one chain: layout and
/// jumps, which a split or a merge of blocks adds or removes.
const LINK_KINDS: [u8; 3] = [
    EdgeKind::Fallthrough as u8,
    EdgeKind::Unconditional as u8,
    EdgeKind::Jump as u8,
];

/// A group of blocks of one CFG that corresponds to a group of blocks of the
/// other.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RegionMatch {
    /// Blocks of the left (old) CFG, in chain order.
    pub left: Vec<BlockId>,
    /// Blocks of the right (new) CFG, in chain order.
    pub right: Vec<BlockId>,
    /// Whether the two sides hold different numbers of instructions.
    pub instruction_count_changed: bool,
    /// Whether the edges that leave the region differ. The last region of a
    /// chain is left by the chain's out-edges, which must reach the
    /// counterparts of the same chains, kind for kind. Any other region is
    /// left by the one edge to the next region, whose kind must agree.
    pub edges_changed: bool,
    /// The member-level pairs, in chain order.
    pairs: Vec<(BlockId, BlockId)>,
}

/// n:m correspondences between the blocks of two CFGs: blocks split in two,
/// and blocks merged into one.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct RegionDiff {
    /// The matched regions, ordered by the first block of the left chain
    /// they lie on, and in chain order along it.
    pub regions: Vec<RegionMatch>,
    /// Blocks only in the left (old) CFG — removed, in ascending order.
    pub left_only: Vec<BlockId>,
    /// Blocks only in the right (new) CFG — added, in ascending order.
    pub right_only: Vec<BlockId>,
}

impl RegionDiff {
    /// Compare two CFGs, matching split and merged blocks as groups.
    ///
    /// Each CFG is cut into maximal *chains*: paths `v1 .. vk` in which
    /// every link `vi -> vi+1` is the only out-edge of `vi` and the only
    /// in-edge of `vi+1`, and is a fallthrough, unconditional or jump edge.
    /// A block with an exceptional out-edge therefore ends its chain, and the
    /// entry always starts one. A ring of links that nothing else enters is
    /// cut before its lowest block. Every live block lies on exactly one
    /// chain, and the chains are read off the graph without changing it.
    ///
    /// The chains are matched like [`CfgDiff::compute`](super::CfgDiff::compute)
    /// matches blocks. A chain's wiring is the in-edges of its first block
    /// and the out-edges of its last, and its size is the sum of its
    /// instruction counts. Each matched pair of chains is then split into
    /// regions by aligning the members by cumulative instruction count. A
    /// left member corresponds to every right member whose range of
    /// instructions overlaps its own, members without instructions align in
    /// order, and a region is a group that no correspondence crosses. Two
    /// chains of equal members give 1:1 regions. A block split in two gives
    /// one region of one block against two. An instruction count changed
    /// inside a chain moves the ranges after it, so the region it lies in
    /// reaches on to the next member boundary that the two sides share.
    ///
    /// The members of an unmatched chain are unmatched blocks. Blocks split
    /// or merged across a branch (if-conversion, tail duplication) are not
    /// regions: a chain never passes a block with two successors or two
    /// predecessors.
    ///
    /// The comparison takes O((n + m) log(n + m)) time for `n` blocks and `m`
    /// edges.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{BlockId, Cfg, EdgeKind, RegionDiff};
    ///
    /// // A straight line of blocks after the entry, one per instruction list.
    /// let build = |blocks: &[&[u32]]| {
    ///     let mut cfg = Cfg::<u32>::new();
    ///     let mut ids = vec![cfg.entry()];
    ///     for &instructions in blocks {
    ///         let block = cfg.new_block();
    ///         cfg.block_mut(block).instructions_mut().extend_from_slice(instructions);
    ///         cfg.add_edge(ids[ids.len() - 1], block, EdgeKind::Fallthrough);
    ///         ids.push(block);
    ///     }
    ///     (cfg, ids)
    /// };
    /// // `entry -> a -> b -> exit`, and a copy in which `b` was split in two.
    /// let (old, ids) = build(&[&[1], &[2, 3], &[4]]);
    /// let b = ids[2];
    /// let (new, ids) = build(&[&[1], &[2], &[3], &[4]]);
    /// let (b1, b2) = (ids[2], ids[3]);
    ///
    /// let diff = RegionDiff::compute(&old, &new);
    /// assert!(diff.left_only.is_empty() && diff.right_only.is_empty());
    /// let split = diff.regions.iter().position(|region| region.right.len() == 2);
    /// let split = split.expect("one region holds the two halves");
    /// assert_eq!(diff.regions[split].left, [b]);
    /// assert_eq!(diff.regions[split].right, [b1, b2]);
    /// let pairs: Vec<(BlockId, BlockId)> = diff.member_pairs(split).collect();
    /// assert_eq!(pairs, [(b, b1), (b, b2)]);
    /// // The other regions pair one block with one block.
    /// assert_eq!(diff.regions.len(), 4);
    /// ```
    #[must_use]
    pub fn compute<I, E, I2, E2>(left: &Cfg<I, E>, right: &Cfg<I2, E2>) -> Self {
        let blocks = (Side::new(left), Side::new(right));
        let left_chains = Chains::new(&blocks.0, left.block_bound());
        let right_chains = Chains::new(&blocks.1, right.block_bound());
        let (left_side, right_side) = (
            left_chains.contracted(&blocks.0),
            right_chains.contracted(&blocks.1),
        );
        let matching = match_by_wiring(&left_side, &right_side);

        let mut diff = Self {
            regions: Vec::new(),
            left_only: Vec::new(),
            right_only: Vec::new(),
        };
        for (&head, facts) in &left_side.blocks {
            let members = left_chains.members(head);
            let Some(&counterpart) = matching.right_of.get(&head) else {
                diff.left_only.extend_from_slice(members);
                continue;
            };
            let edges_changed = !edges_preserved(facts, right_side.facts(counterpart), &matching);
            diff.align(
                (&blocks.0, &blocks.1),
                (members, right_chains.members(counterpart)),
                edges_changed,
            );
        }
        for &head in right_side.blocks.keys() {
            if !matching.left_of.contains_key(&head) {
                diff.right_only
                    .extend_from_slice(right_chains.members(head));
            }
        }
        diff.left_only.sort_unstable();
        diff.right_only.sort_unstable();
        diff
    }

    /// The member-level pairs of `regions[region]`, in chain order: each
    /// left block with every right block whose range of instructions
    /// overlaps its own.
    ///
    /// # Panics
    ///
    /// Panics if `region` is not an index of [`regions`](Self::regions).
    pub fn member_pairs(&self, region: usize) -> impl Iterator<Item = (BlockId, BlockId)> + '_ {
        self.regions[region].pairs.iter().copied()
    }

    /// Adds the regions of one matched pair of chains: a two-pointer merge
    /// of the members by the instruction count each side has reached.
    ///
    /// Each step pairs the current members and moves past the one that ends
    /// first, or past both when they end together, which closes a region.
    /// The side that runs out of members first stays on its last one. So
    /// members without instructions, which end where they start, align in
    /// order.
    fn align(
        &mut self,
        (left_blocks, right_blocks): (&Side, &Side),
        (left, right): (&[BlockId], &[BlockId]),
        chain_edges_changed: bool,
    ) {
        let count = |side: &Side, block| side.facts(block).fingerprint.instruction_count;
        let total = |side: &Side, members: &[BlockId]| -> usize {
            members.iter().map(|&block| count(side, block)).sum()
        };
        let (mut i, mut j) = (0, 0);
        let (mut left_end, mut right_end) =
            (count(left_blocks, left[0]), count(right_blocks, right[0]));
        let (mut left_start, mut right_start) = (0, 0);
        let mut pairs = Vec::new();
        loop {
            pairs.push((left[i], right[j]));
            let (left_last, right_last) = (i + 1 == left.len(), j + 1 == right.len());
            let edges_changed = if left_last && right_last {
                chain_edges_changed
            } else if right_last || (!left_last && left_end < right_end) {
                i += 1;
                left_end += count(left_blocks, left[i]);
                continue;
            } else if left_last || right_end < left_end {
                j += 1;
                right_end += count(right_blocks, right[j]);
                continue;
            } else {
                // Both end here; each is left by its one link onward.
                left_blocks.facts(left[i]).out_edges[0].0
                    != right_blocks.facts(right[j]).out_edges[0].0
            };
            let (ours, theirs) = (&left[left_start..=i], &right[right_start..=j]);
            self.regions.push(RegionMatch {
                left: ours.to_vec(),
                right: theirs.to_vec(),
                instruction_count_changed: total(left_blocks, ours) != total(right_blocks, theirs),
                edges_changed,
                pairs: core::mem::take(&mut pairs),
            });
            if left_last && right_last {
                return;
            }
            (i, j) = (i + 1, j + 1);
            left_end += count(left_blocks, left[i]);
            right_end += count(right_blocks, right[j]);
            (left_start, right_start) = (i, j);
        }
    }
}

/// The maximal chains of one CFG.
struct Chains {
    /// Every live block, chain after chain, each chain in order.
    members: Vec<BlockId>,
    /// Each chain by its first block: where it lies in `members`.
    spans: BTreeMap<BlockId, (usize, usize)>,
}

impl Chains {
    /// The chains of the CFG that `blocks` describes; `bound` exceeds every
    /// block index.
    fn new(blocks: &Side, bound: usize) -> Self {
        // The block each block links to, and whether a link enters it.
        let mut next = vec![None; bound];
        let mut linked = vec![false; bound];
        for (&block, facts) in &blocks.blocks {
            if let &[(kind, target)] = facts.out_edges.as_slice()
                && LINK_KINDS.contains(&kind)
                && target != block
                && target != blocks.entry
                && blocks.facts(target).fingerprint.in_degree == 1
            {
                next[block.index()] = Some(target);
                linked[target.index()] = true;
            }
        }
        let mut seen = vec![false; bound];
        let mut members = Vec::with_capacity(blocks.blocks.len());
        let mut spans = BTreeMap::new();
        // Chains start at the blocks no link enters. What is left lies on
        // rings of links, each cut before its lowest block.
        let heads = blocks.blocks.keys().filter(|block| !linked[block.index()]);
        for &head in heads.chain(blocks.blocks.keys()) {
            if seen[head.index()] {
                continue;
            }
            let start = members.len();
            let mut block = head;
            loop {
                seen[block.index()] = true;
                members.push(block);
                match next[block.index()] {
                    Some(following) if !seen[following.index()] => block = following,
                    _ => break,
                }
            }
            spans.insert(head, (start, members.len()));
        }
        Self { members, spans }
    }

    /// The members of the chain that starts at `head`, in order.
    fn members(&self, head: BlockId) -> &[BlockId] {
        let (start, end) = self.spans[&head];
        &self.members[start..end]
    }

    /// The CFG with every chain contracted to its first block: the chain's
    /// size is the sum of its instruction counts, its in-edges are its first
    /// block's and its out-edges its last block's. Every edge that is not a
    /// link enters the first block of a chain, so the contracted edges join
    /// chains.
    fn contracted(&self, blocks: &Side) -> Side {
        let contracted = self
            .spans
            .keys()
            .map(|&head| {
                let members = self.members(head);
                let last = blocks.facts(members[members.len() - 1]);
                let fingerprint = BlockFingerprint {
                    instruction_count: members
                        .iter()
                        .map(|&member| blocks.facts(member).fingerprint.instruction_count)
                        .sum(),
                    out_degree: last.out_edges.len(),
                    in_degree: blocks.facts(head).fingerprint.in_degree,
                    out_edge_discriminants: last.fingerprint.out_edge_discriminants.clone(),
                };
                let out_edges = last.out_edges.clone();
                (
                    head,
                    BlockFacts {
                        fingerprint,
                        out_edges,
                    },
                )
            })
            .collect();
        Side {
            entry: blocks.entry,
            blocks: contracted,
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeSet;
    use alloc::vec::Vec;

    use super::*;
    use crate::edge::EdgeId;
    use crate::graph::diff::CfgDiff;
    use crate::test_util::rng::Rng;

    /// `entry -> b1 -> b2 -> ...` by fallthrough, block `bi` holding
    /// `counts[i - 1]` instructions, and the blocks in order.
    fn line(counts: &[usize]) -> (Cfg<u32>, Vec<BlockId>) {
        let mut cfg = Cfg::new();
        let mut blocks = vec![cfg.entry()];
        for &count in counts {
            let block = cfg.new_block();
            cfg.block_mut(block)
                .instructions_mut()
                .extend((0..count).map(|op| u32::try_from(op).expect("a small count")));
            cfg.add_edge(blocks[blocks.len() - 1], block, EdgeKind::Fallthrough);
            blocks.push(block);
        }
        (cfg, blocks)
    }

    fn regions(diff: &RegionDiff) -> Vec<(Vec<BlockId>, Vec<BlockId>)> {
        diff.regions
            .iter()
            .map(|region| (region.left.clone(), region.right.clone()))
            .collect()
    }

    #[test]
    fn a_split_block_is_one_region_of_one_block_against_two() {
        // `entry -> a -> b -> exit`, with `b` split into `b1 -> b2`.
        let (left, l) = line(&[1, 2, 1]);
        let (right, r) = line(&[1, 1, 1, 1]);

        let structural = CfgDiff::compute(&left, &right);
        assert_eq!(structural.right_only, [r[3]], "b1 stands in for b");

        let diff = RegionDiff::compute(&left, &right);
        assert!(diff.left_only.is_empty() && diff.right_only.is_empty());
        assert_eq!(
            regions(&diff),
            [
                (vec![l[0]], vec![r[0]]),
                (vec![l[1]], vec![r[1]]),
                (vec![l[2]], vec![r[2], r[3]]),
                (vec![l[3]], vec![r[4]]),
            ]
        );
        assert_eq!(
            diff.member_pairs(2).collect::<Vec<_>>(),
            [(l[2], r[2]), (l[2], r[3])]
        );
        assert!(
            diff.regions
                .iter()
                .all(|region| !region.instruction_count_changed && !region.edges_changed)
        );
    }

    #[test]
    fn merged_blocks_are_one_region_of_two_blocks_against_one() {
        let (left, l) = line(&[1, 1, 1, 1]);
        let (right, r) = line(&[1, 2, 1]);
        let diff = RegionDiff::compute(&left, &right);
        assert!(diff.left_only.is_empty() && diff.right_only.is_empty());
        assert_eq!(
            regions(&diff),
            [
                (vec![l[0]], vec![r[0]]),
                (vec![l[1]], vec![r[1]]),
                (vec![l[2], l[3]], vec![r[2]]),
                (vec![l[4]], vec![r[3]]),
            ]
        );
        assert_eq!(
            diff.member_pairs(2).collect::<Vec<_>>(),
            [(l[2], r[2]), (l[3], r[2])]
        );
    }

    #[test]
    fn blocks_without_instructions_align_in_order() {
        let (left, l) = line(&[0, 0]);
        let (right, r) = line(&[0, 0, 0]);
        let diff = RegionDiff::compute(&left, &right);
        assert_eq!(
            regions(&diff),
            [
                (vec![l[0]], vec![r[0]]),
                (vec![l[1]], vec![r[1]]),
                (vec![l[2]], vec![r[2], r[3]]),
            ]
        );
    }

    #[test]
    fn a_count_changed_inside_a_chain_widens_its_region_to_the_next_shared_boundary() {
        let (left, l) = line(&[2, 2, 2]);
        let (right, r) = line(&[3, 1, 2]);
        let diff = RegionDiff::compute(&left, &right);
        assert_eq!(
            regions(&diff),
            [
                (vec![l[0]], vec![r[0]]),
                (vec![l[1], l[2]], vec![r[1], r[2]]),
                (vec![l[3]], vec![r[3]]),
            ]
        );
        assert!(!diff.regions[1].instruction_count_changed);

        let (right, _) = line(&[2, 2, 3]);
        let diff = RegionDiff::compute(&left, &right);
        assert_eq!(diff.regions.len(), 4);
        assert!(diff.regions[3].instruction_count_changed);
    }

    #[test]
    fn changed_links_and_exits_mark_the_regions_they_leave() {
        // A link of another kind inside the chain.
        let (left, ours) = line(&[1, 1]);
        let (mut right, theirs) = line(&[1, 1]);
        let link = right.outgoing(theirs[1]).next().expect("a links to b");
        right.edge_mut(link).set_kind(EdgeKind::Jump);
        let diff = RegionDiff::compute(&left, &right);
        assert_eq!(diff.regions.len(), 3, "a jump still links: {diff:?}");
        let changed: Vec<bool> = diff
            .regions
            .iter()
            .map(|region| region.edges_changed)
            .collect();
        assert_eq!(changed, [false, true, false]);
        assert_eq!(diff.regions[1].left, [ours[1]]);

        // The chain `entry -> b` branches to `c` and `d`; on the right both
        // branches reach `c`, so the chain's exits no longer match.
        let branch = |both_to_c: bool| {
            let mut cfg = Cfg::<u32>::new();
            let [b, c, d] = [(); 3].map(|()| cfg.new_block());
            cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
            cfg.add_edge(b, c, EdgeKind::ConditionalTrue);
            let other = if both_to_c { c } else { d };
            cfg.add_edge(b, other, EdgeKind::ConditionalFalse);
            cfg
        };
        let diff = RegionDiff::compute(&branch(false), &branch(true));
        let changed: Vec<bool> = diff
            .regions
            .iter()
            .map(|region| region.edges_changed)
            .collect();
        assert_eq!(changed, [false, true], "{diff:?}");
    }

    #[test]
    fn an_unmatched_chain_leaves_all_its_blocks_unmatched() {
        let (left, _) = line(&[1]);
        let (mut right, _) = line(&[1]);
        let (p, q) = (right.new_block(), right.new_block());
        right.add_edge(p, q, EdgeKind::Fallthrough);
        right.block_mut(q).push(1);
        let diff = RegionDiff::compute(&left, &right);
        assert_eq!(diff.right_only, [p, q]);
        assert_eq!(diff.left_only, Vec::<BlockId>::new());
    }

    fn chains_of(cfg: &Cfg<u32>) -> Vec<Vec<BlockId>> {
        let side = Side::new(cfg);
        let chains = Chains::new(&side, cfg.block_bound());
        chains
            .spans
            .keys()
            .map(|&head| chains.members(head).to_vec())
            .collect()
    }

    #[test]
    fn chains_stop_at_branches_exceptional_edges_and_the_entry() {
        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.entry();
        let [a, b, c, d, handler, ring, other, looped] = [(); 8].map(|()| cfg.new_block());
        cfg.add_edge(entry, a, EdgeKind::Fallthrough);
        cfg.add_edge(a, b, EdgeKind::Jump);
        cfg.add_edge(b, c, EdgeKind::Fallthrough);
        cfg.add_edge(b, handler, EdgeKind::ExceptionUnwind);
        cfg.add_edge(c, d, EdgeKind::Unconditional);
        cfg.add_edge(d, entry, EdgeKind::Jump);
        cfg.add_edge(ring, other, EdgeKind::Fallthrough);
        cfg.add_edge(other, ring, EdgeKind::Fallthrough);
        cfg.add_edge(looped, looped, EdgeKind::Fallthrough);
        assert_eq!(
            chains_of(&cfg),
            [
                vec![entry, a, b],
                vec![c, d],
                vec![handler],
                vec![ring, other],
                vec![looped],
            ]
        );
    }

    const KINDS: [EdgeKind; 10] = [
        EdgeKind::Fallthrough,
        EdgeKind::Fallthrough,
        EdgeKind::Fallthrough,
        EdgeKind::Jump,
        EdgeKind::Unconditional,
        EdgeKind::ConditionalTrue,
        EdgeKind::ConditionalFalse,
        EdgeKind::SwitchCase,
        EdgeKind::Back,
        EdgeKind::ExceptionUnwind,
    ];

    /// A CFG whose entry reaches every block: each block is entered from an
    /// earlier one, a few more edges run anywhere, and every block holds one
    /// to four instructions. `renumber` shuffles the order in which the
    /// blocks after the entry are made; the CFG is otherwise a function of
    /// `rng` alone.
    fn random_cfg(rng: &mut Rng, renumber: Option<&mut Rng>) -> Cfg<u32> {
        let count = 1 + rng.below(24);
        let mut slots: Vec<usize> = (0..count).collect();
        if let Some(renumber) = renumber {
            renumber.shuffle(&mut slots[1..]);
        }
        let mut cfg = Cfg::new();
        for _ in 1..count {
            cfg.new_block();
        }
        let id = |role: usize| BlockId::from_index(slots[role]);
        for role in 0..count {
            let length = 1 + rng.below(4);
            cfg.block_mut(id(role))
                .instructions_mut()
                .extend((0..length).map(|op| u32::try_from(op).expect("a small count")));
        }
        for role in 1..count {
            let kind = KINDS[rng.below(KINDS.len())];
            cfg.add_edge(id(rng.below(role)), id(role), kind);
        }
        for _ in 0..rng.below(count) {
            let kind = KINDS[rng.below(KINDS.len())];
            cfg.add_edge(id(rng.below(count)), id(rng.below(count)), kind);
        }
        cfg
    }

    #[test]
    fn every_chain_is_maximal_and_joined_only_by_links() {
        let mut rng = Rng::new(45);
        for _ in 0..300 {
            let cfg = random_cfg(&mut rng, None);
            let link = |source: BlockId| {
                let mut edges = cfg.outgoing(source);
                let (Some(edge), None) = (edges.next(), edges.next()) else {
                    return None;
                };
                let target = cfg.edge(edge).target();
                let kind = cfg[edge].kind();
                (matches!(
                    kind,
                    EdgeKind::Fallthrough | EdgeKind::Unconditional | EdgeKind::Jump
                ) && target != source
                    && target != cfg.entry()
                    && cfg.incoming(target).count() == 1)
                    .then_some(target)
            };
            let chains = chains_of(&cfg);
            let mut seen = BTreeSet::new();
            for chain in &chains {
                for pair in chain.windows(2) {
                    assert_eq!(link(pair[0]), Some(pair[1]), "{chain:?}");
                }
                assert!(chain.iter().all(|&block| seen.insert(block)));
                assert!(!chain[1..].contains(&cfg.entry()));
                // Nothing links on from the last block, and nothing links
                // into the first: the entry reaches every block, so no ring.
                assert_eq!(link(chain[chain.len() - 1]), None, "{chain:?}");
                assert!(cfg.block_ids().all(|block| link(block) != Some(chain[0])));
            }
            assert_eq!(seen.len(), cfg.block_count());
        }
    }

    #[test]
    fn identical_and_renumbered_copies_pair_one_to_one_like_cfg_diff() {
        let mut rng = Rng::new(46);
        // Chains of several blocks, whose members the alignment must pair
        // one to one rather than as a group.
        let mut long_chains = 0;
        for _ in 0..200 {
            let seed = rng.below(1 << 20) as u64;
            let original = random_cfg(&mut Rng::new(seed), None);
            let renumbered = random_cfg(&mut Rng::new(seed), Some(&mut rng));
            long_chains += chains_of(&original)
                .iter()
                .filter(|chain| chain.len() > 1)
                .count();
            for copy in [&original, &renumbered] {
                let diff = RegionDiff::compute(&original, copy);
                assert!(diff.left_only.is_empty() && diff.right_only.is_empty());
                assert!(
                    diff.regions
                        .iter()
                        .all(|region| region.left.len() == 1 && region.right.len() == 1),
                    "{diff:?}"
                );
                let pairs: BTreeSet<(BlockId, BlockId)> = (0..diff.regions.len())
                    .flat_map(|region| diff.member_pairs(region))
                    .collect();
                let matched: BTreeSet<(BlockId, BlockId)> = CfgDiff::compute(&original, copy)
                    .matched
                    .iter()
                    .map(|pair| (pair.left, pair.right))
                    .collect();
                assert_eq!(pairs, matched);
            }
        }
        assert!(long_chains > 100, "{long_chains} chains of several blocks");
    }

    /// Merges `next` into `block`, which links to it: `next`'s
    /// instructions follow `block`'s and its out-edges leave `block`.
    fn merge(cfg: &mut Cfg<u32>, block: BlockId, next: BlockId) {
        let link = cfg.outgoing(block).next().expect("block links to next");
        cfg.remove_edge(link);
        let moved = cfg.block(next).instructions().to_vec();
        cfg.block_mut(block).instructions_mut().extend(moved);
        let leaving: Vec<EdgeId> = cfg.outgoing(next).collect();
        for edge in leaving {
            cfg.redirect_edge_source(edge, block);
        }
        cfg.remove_block(next);
    }

    #[test]
    fn random_splits_and_merges_are_recovered_as_regions() {
        let mut rng = Rng::new(47);
        let (mut split_count, mut merge_count) = (0, 0);
        for _ in 0..300 {
            let left = random_cfg(&mut rng, None);
            let mut right = left.clone();
            let mut touched = BTreeSet::new();
            let mut merges = Vec::new();
            for chain in chains_of(&left) {
                for pair in chain.windows(2) {
                    if !touched.contains(&pair[0]) && rng.one_in(2) {
                        touched.extend([pair[0], pair[1]]);
                        merges.push((pair[0], pair[1]));
                    }
                }
            }
            let mut splits = Vec::new();
            for block in left.block_ids() {
                let length = left.block(block).instructions().len();
                if !touched.contains(&block) && length > 1 && rng.one_in(3) {
                    splits.push((block, 1 + rng.below(length - 1)));
                }
            }
            for &(block, next) in &merges {
                merge(&mut right, block, next);
            }
            let splits: Vec<(BlockId, BlockId)> = splits
                .into_iter()
                .map(|(block, at)| (block, right.split_block(block, at, EdgeKind::Fallthrough)))
                .collect();

            let diff = RegionDiff::compute(&left, &right);
            assert!(
                diff.left_only.is_empty() && diff.right_only.is_empty(),
                "{diff:?}"
            );
            let lefts: Vec<BlockId> = diff
                .regions
                .iter()
                .flat_map(|region| region.left.iter().copied())
                .collect();
            let rights: Vec<BlockId> = diff
                .regions
                .iter()
                .flat_map(|region| region.right.iter().copied())
                .collect();
            assert_eq!(
                lefts.iter().copied().collect::<BTreeSet<_>>().len(),
                lefts.len()
            );
            assert_eq!(
                rights.iter().copied().collect::<BTreeSet<_>>().len(),
                rights.len()
            );
            assert_eq!(lefts.len(), left.block_count());
            assert_eq!(rights.len(), right.block_count());
            let region_of = |block: BlockId| {
                diff.regions
                    .iter()
                    .find(|region| region.left.contains(&block))
                    .expect("every left block is in a region")
            };
            (split_count, merge_count) = (split_count + splits.len(), merge_count + merges.len());
            for (block, half) in splits {
                let region = region_of(block);
                assert!(
                    region.right.contains(&block) && region.right.contains(&half),
                    "{region:?}"
                );
            }
            for (block, next) in merges {
                let region = region_of(block);
                assert!(
                    region.left.contains(&next) && region.right.contains(&block),
                    "{region:?}"
                );
            }
        }
        assert!(
            split_count > 100 && merge_count > 100,
            "{split_count} splits, {merge_count} merges"
        );
    }
}
