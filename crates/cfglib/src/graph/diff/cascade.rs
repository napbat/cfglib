//! The content-hash cascade of [`CascadeDiff`].

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet, VecDeque};
use alloc::vec;
use alloc::vec::Vec;

use super::{BlockMatch, CfgDiff, Matching, Side, follow_edges};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::graph::fingerprint::{Mixer, wl_colors};

/// Consumer-supplied content hashes of one block.
///
/// Reading instructions and operands is the consumer's, so cfglib never
/// looks inside a block: the consumer hashes each one, and
/// [`CascadeDiff`] combines the hashes with the wiring. Blocks with equal
/// hashes are taken to hold the same content, so a hash that cannot tell
/// two blocks apart makes them interchangeable to the cascade.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockHashes {
    /// Changes when any instruction changes: opcodes and operands, in
    /// order.
    pub strict: u64,
    /// Survives operand and register changes: for example, a hash of the
    /// sorted set of opcodes.
    pub loose: u64,
}

/// The evidence for one pair of a [`CascadeDiff`]. The variants are ordered
/// from the strongest evidence to the weakest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum MatchLevel {
    /// The two entry blocks, which are paired whatever their content.
    Entry,
    /// Equal `strict` hashes and equal surroundings: the multisets of
    /// `(edge kind, loose hash)` over the successors, and over the
    /// predecessors, are equal too.
    Full,
    /// Equal `strict` hashes.
    Strict,
    /// Equal `loose` hashes.
    Loose,
    /// No equal hash: the walk paired the blocks along the edges of a
    /// matched pair.
    Walk,
    /// No equal hash: the blocks have the same shape (degrees and out-edge
    /// kinds) and were the nearest in position.
    Shape,
}

/// A [`CfgDiff`] matched through content hashes the consumer supplies,
/// with the evidence for each pair.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CascadeDiff {
    /// The structural differences, reported like [`CfgDiff::compute`]
    /// reports them.
    pub diff: CfgDiff,
    /// The level of each pair of `diff.matched`, in the same order.
    pub levels: Vec<MatchLevel>,
}

impl CascadeDiff {
    /// Compare two CFGs through the content hashes of their blocks.
    ///
    /// `left_hashes` and `right_hashes` are called once per live block,
    /// with the block and its instructions. The matching is built as a
    /// cascade:
    /// 1. The entry blocks are paired, whatever their content.
    /// 2. Then, for each key in turn (`full`, `strict`, `loose`, and last
    ///    the shape that [`CfgDiff::compute`] matches by), the blocks still
    ///    free are bucketed by the key. Within a bucket, each left block, in
    ///    position order, takes the free right block whose relative position
    ///    is the nearest to its own, ties to the lower id. After each key,
    ///    the walk of `CfgDiff::compute` extends the matching along the edges
    ///    of every pair. The walk also pairs two targets of different shapes
    ///    when their `strict` hashes are equal.
    ///
    /// The `full` key mixes a block's `strict` hash with the multisets of
    /// `(edge kind, loose hash)` over its successors and over its
    /// predecessors: one round of [`wl_colors`](crate::wl_colors) seeded
    /// with the loose hashes. A block's position is its index in reverse
    /// postorder from the entry, which takes each block's out-edges in kind
    /// order. The blocks the entry does not reach follow in ascending
    /// order. The relative position divides the position by the number of
    /// live blocks, so the two CFGs compare on one scale.
    ///
    /// Each key pairs blocks before the walk runs from them, so content
    /// decides before wiring does. Two unchanged blocks of one shape but
    /// different content pair with their own counterparts, not crosswise,
    /// and a block that gained or lost an edge still pairs by its `strict`
    /// hash.
    ///
    /// Each pair's [`MatchLevel`] is the first key of the cascade that the
    /// two blocks share. When they share none, it says whether the walk or
    /// the shape fallback paired them. A pair the walk formed between two
    /// blocks of equal content therefore reports that content match.
    ///
    /// [`diff`](Self::diff) reports the changes like `CfgDiff::compute`
    /// does. The comparison takes O((n + m) log n) time for `n` blocks and
    /// `m` edges, besides the hash calls.
    ///
    /// # Examples
    ///
    /// ```
    /// use cfglib::{BlockHashes, BlockId, CascadeDiff, Cfg, CfgDiff, EdgeKind, MatchLevel, Mixer};
    ///
    /// // `entry -> body`, and a copy in which a jump from new code reaches
    /// // `body` too, which changes its shape.
    /// let build = || {
    ///     let mut cfg = Cfg::<u32>::new();
    ///     let body = cfg.new_block();
    ///     cfg.block_mut(body).push(1);
    ///     cfg.add_edge(cfg.entry(), body, EdgeKind::Fallthrough);
    ///     (cfg, body)
    /// };
    /// let (old, body) = build();
    /// let (mut new, _) = build();
    /// let jump = new.new_block();
    /// new.block_mut(jump).push(2);
    /// new.add_edge(jump, body, EdgeKind::Jump);
    ///
    /// // By wiring alone, `body` has no counterpart.
    /// assert!(CfgDiff::compute(&old, &new).left_only.contains(&body));
    ///
    /// // The consumer hashes its own instructions, with the crate's mixer.
    /// let hash = |_: BlockId, instructions: &[u32]| {
    ///     let strict = instructions
    ///         .iter()
    ///         .fold(Mixer::new(), |mixer, &op| mixer.mix(op.into()))
    ///         .finish();
    ///     BlockHashes { strict, loose: strict }
    /// };
    /// let cascade = CascadeDiff::compute(&old, &new, hash, hash);
    /// assert_eq!(cascade.match_level(old.entry()), Some(MatchLevel::Entry));
    /// assert_eq!(cascade.match_level(body), Some(MatchLevel::Strict));
    /// assert_eq!(cascade.diff.right_only, [jump]);
    /// ```
    #[must_use]
    pub fn compute<I, E, I2, E2>(
        left: &Cfg<I, E>,
        right: &Cfg<I2, E2>,
        left_hashes: impl FnMut(BlockId, &[I]) -> BlockHashes,
        right_hashes: impl FnMut(BlockId, &[I2]) -> BlockHashes,
    ) -> Self {
        let (diff, (left_keys, right_keys, by_shape)) =
            CfgDiff::compute_by(left, right, |left_side, right_side| {
                let left_keys = Keys::new(left, left_side, left_hashes);
                let right_keys = Keys::new(right, right_side, right_hashes);
                let (matching, by_shape) =
                    match_by_cascade((left_side, right_side), (&left_keys, &right_keys));
                (matching, (left_keys, right_keys, by_shape))
            });
        let keys = (&left_keys, &right_keys);
        let levels = match_levels(&diff.matched, left.entry(), keys, &by_shape);
        Self { diff, levels }
    }

    /// The level of the pair that contains the left block `left`, or `None`
    /// when `left` is unmatched.
    #[must_use]
    pub fn match_level(&self, left: BlockId) -> Option<MatchLevel> {
        let index = self
            .diff
            .matched
            .binary_search_by_key(&left, |pair| pair.left)
            .ok()?;
        Some(self.levels[index])
    }
}

/// The cascade's matching, and the left blocks the shape level paired.
fn match_by_cascade(
    (left_side, right_side): (&Side, &Side),
    keys: (&Keys, &Keys),
) -> (Matching, BTreeSet<BlockId>) {
    let (left_keys, right_keys) = keys;
    let mut matching = Matching::default();
    let mut pending = VecDeque::new();
    matching.pair(left_side.entry, right_side.entry);
    pending.push_back((left_side.entry, right_side.entry));

    let same_content = |left_block: BlockId, right_block: BlockId| {
        left_keys.strict(left_block) == right_keys.strict(right_block)
    };
    let walk = |matching: &mut Matching, pending: &mut VecDeque<_>| {
        follow_edges(
            left_side,
            right_side,
            matching,
            pending,
            Some(&same_content),
        );
    };
    pair_nearest(keys, &mut matching, &mut pending, |keys, block| {
        keys.full[block.index()]
    });
    walk(&mut matching, &mut pending);
    pair_nearest(keys, &mut matching, &mut pending, Keys::strict);
    walk(&mut matching, &mut pending);
    pair_nearest(keys, &mut matching, &mut pending, |keys, block| {
        keys.hashes[block.index()].loose
    });
    walk(&mut matching, &mut pending);
    pair_nearest_with(
        keys,
        &mut matching,
        &mut pending,
        |block| left_side.facts(block).shape(),
        |block| right_side.facts(block).shape(),
    );
    let by_shape: BTreeSet<BlockId> = pending.iter().map(|&(block, _)| block).collect();
    walk(&mut matching, &mut pending);
    (matching, by_shape)
}

/// The level of each pair of `matched`: the entry pair's, else the first key
/// of the cascade the two blocks share, else the step that paired them.
fn match_levels(
    matched: &[BlockMatch],
    entry: BlockId,
    (left_keys, right_keys): (&Keys, &Keys),
    by_shape: &BTreeSet<BlockId>,
) -> Vec<MatchLevel> {
    matched
        .iter()
        .map(|pair| {
            let (ours, theirs) = (pair.left.index(), pair.right.index());
            let (our_hashes, their_hashes) = (left_keys.hashes[ours], right_keys.hashes[theirs]);
            if pair.left == entry {
                MatchLevel::Entry
            } else if left_keys.full[ours] == right_keys.full[theirs] {
                MatchLevel::Full
            } else if our_hashes.strict == their_hashes.strict {
                MatchLevel::Strict
            } else if our_hashes.loose == their_hashes.loose {
                MatchLevel::Loose
            } else if by_shape.contains(&pair.left) {
                MatchLevel::Shape
            } else {
                MatchLevel::Walk
            }
        })
        .collect()
}

/// What the cascade knows about the blocks of one CFG, by block index.
struct Keys {
    hashes: Vec<BlockHashes>,
    /// The `strict` hash mixed with one Weisfeiler–Leman round over the
    /// loose hashes.
    full: Vec<u64>,
    /// The live blocks in position order.
    order: Vec<BlockId>,
    /// Each live block's index in `order`.
    position: Vec<usize>,
}

impl Keys {
    fn new<I, E>(
        cfg: &Cfg<I, E>,
        side: &Side,
        mut hash: impl FnMut(BlockId, &[I]) -> BlockHashes,
    ) -> Self {
        let bound = cfg.block_bound();
        let mut hashes = vec![
            BlockHashes {
                strict: 0,
                loose: 0
            };
            bound
        ];
        for &block in side.blocks.keys() {
            hashes[block.index()] = hash(block, cfg.block(block).instructions());
        }
        let surroundings = wl_colors(
            cfg,
            1,
            |block| hashes[block.index()].loose,
            |edge| u64::from(cfg[edge].kind() as u8),
        );
        let full = hashes
            .iter()
            .zip(&surroundings)
            .map(|(hashes, &surrounding)| Mixer::new().mix(hashes.strict).mix(surrounding).finish())
            .collect();
        let order = position_order(side, bound);
        let mut position = vec![0; bound];
        for (index, &block) in order.iter().enumerate() {
            position[block.index()] = index;
        }
        Self {
            hashes,
            full,
            order,
            position,
        }
    }

    fn strict(&self, block: BlockId) -> u64 {
        self.hashes[block.index()].strict
    }
}

/// The live blocks of `side`: reverse postorder from the entry, taking each
/// block's out-edges in kind order, then the blocks the entry does not reach
/// in ascending order.
///
/// Kind order rather than adjacency order keeps the positions independent of
/// the order in which edges of different kinds were added.
fn position_order(side: &Side, bound: usize) -> Vec<BlockId> {
    let mut seen = vec![false; bound];
    let mut order = Vec::with_capacity(side.blocks.len());
    let mut stack = vec![(side.entry, 0)];
    seen[side.entry.index()] = true;
    while let Some(top) = stack.last_mut() {
        let (block, next) = *top;
        if let Some(&(_, target)) = side.facts(block).out_edges.get(next) {
            top.1 += 1;
            if !seen[target.index()] {
                seen[target.index()] = true;
                stack.push((target, 0));
            }
        } else {
            order.push(block);
            stack.pop();
        }
    }
    order.reverse();
    order.extend(
        side.blocks
            .keys()
            .copied()
            .filter(|block| !seen[block.index()]),
    );
    order
}

/// [`pair_nearest_with`] for a key both sides compute alike.
fn pair_nearest<K: Ord>(
    (left, right): (&Keys, &Keys),
    matching: &mut Matching,
    pending: &mut VecDeque<(BlockId, BlockId)>,
    key: impl Fn(&Keys, BlockId) -> K,
) {
    pair_nearest_with(
        (left, right),
        matching,
        pending,
        |block| key(left, block),
        |block| key(right, block),
    );
}

/// Pairs the free blocks that share a key. Each left block, in position
/// order, takes the free right block of its key whose relative position is
/// the nearest to its own, ties to the lower id. Each new pair is queued
/// for the walk.
fn pair_nearest_with<K: Ord>(
    (left, right): (&Keys, &Keys),
    matching: &mut Matching,
    pending: &mut VecDeque<(BlockId, BlockId)>,
    left_key: impl Fn(BlockId) -> K,
    right_key: impl Fn(BlockId) -> K,
) {
    // `a / left_count` against `b / right_count` compares as
    // `a * right_count` against `b * left_count`, exactly.
    let scale = |position: usize, count: usize| position as u128 * count as u128;
    let (left_count, right_count) = (left.order.len(), right.order.len());
    // Per key: the free left blocks in position order, and the free right
    // blocks by scaled position, which is distinct per block.
    let mut buckets: BTreeMap<K, (Vec<BlockId>, BTreeMap<u128, BlockId>)> = BTreeMap::new();
    for &block in &left.order {
        if !matching.right_of.contains_key(&block) {
            buckets.entry(left_key(block)).or_default().0.push(block);
        }
    }
    for &block in &right.order {
        if !matching.left_of.contains_key(&block)
            && let Some((_, free)) = buckets.get_mut(&right_key(block))
        {
            free.insert(scale(right.position[block.index()], left_count), block);
        }
    }
    for (blocks, mut free) in buckets.into_values() {
        for block in blocks {
            let at = scale(left.position[block.index()], right_count);
            let after = free
                .range(at..)
                .next()
                .map(|(&to, &other)| (to - at, to, other));
            let before = free
                .range(..at)
                .next_back()
                .map(|(&to, &other)| (at - to, to, other));
            // The nearer one, and on equal distance the lower id.
            let nearest = match (before, after) {
                (Some(before), Some(after)) => Some(if (after.0, after.2) < (before.0, before.2) {
                    after
                } else {
                    before
                }),
                (before, after) => before.or(after),
            };
            let Some((_, to, counterpart)) = nearest else {
                break;
            };
            free.remove(&to);
            matching.pair(block, counterpart);
            pending.push_back((block, counterpart));
        }
    }
}

#[cfg(test)]
mod tests {
    use alloc::collections::BTreeMap;
    use alloc::vec::Vec;

    use super::*;
    use crate::edge::EdgeKind;
    use crate::test_util::rng::Rng;

    /// Hashes instructions written as `opcode * 100 + operand`: `strict`
    /// over all of them in order, `loose` over the set of opcodes.
    fn hash(_: BlockId, instructions: &[u32]) -> BlockHashes {
        let strict = instructions
            .iter()
            .fold(Mixer::new(), |state, &op| state.mix(u64::from(op)))
            .finish();
        let mut opcodes: Vec<u32> = instructions.iter().map(|op| op / 100).collect();
        opcodes.sort_unstable();
        opcodes.dedup();
        let loose = opcodes
            .iter()
            .fold(Mixer::new().mix(1), |state, &opcode| {
                state.mix(u64::from(opcode))
            })
            .finish();
        BlockHashes { strict, loose }
    }

    fn cascade(left: &Cfg<u32>, right: &Cfg<u32>) -> CascadeDiff {
        CascadeDiff::compute(left, right, hash, hash)
    }

    fn block(cfg: &mut Cfg<u32>, instructions: &[u32]) -> BlockId {
        let block = cfg.new_block();
        cfg.block_mut(block)
            .instructions_mut()
            .extend_from_slice(instructions);
        block
    }

    fn right_of(diff: &CfgDiff, left: BlockId) -> Option<BlockId> {
        diff.matched
            .iter()
            .find(|pair| pair.left == left)
            .map(|pair| pair.right)
    }

    /// `entry -> a -> b -> exit`, with `b` holding `b_content`.
    fn chain(b_content: u32) -> (Cfg<u32>, [BlockId; 3]) {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry()).push(1);
        let blocks = [
            block(&mut cfg, &[100]),
            block(&mut cfg, &[b_content]),
            block(&mut cfg, &[300]),
        ];
        cfg.add_edge(cfg.entry(), blocks[0], EdgeKind::Fallthrough);
        cfg.add_edge(blocks[0], blocks[1], EdgeKind::Fallthrough);
        cfg.add_edge(blocks[1], blocks[2], EdgeKind::Fallthrough);
        (cfg, blocks)
    }

    #[test]
    fn a_block_that_gains_a_predecessor_keeps_its_counterpart() {
        let (left, [a, b, exit]) = chain(200);
        let (mut right, _) = chain(200);
        let jump = block(&mut right, &[400]);
        right.add_edge(jump, b, EdgeKind::Jump);

        let structural = CfgDiff::compute(&left, &right);
        assert_eq!(structural.left_only, [b]);
        assert_eq!(structural.right_only, [b, jump]);

        let diff = cascade(&left, &right);
        assert_eq!(diff.diff.right_only, [jump]);
        assert_eq!(diff.diff.left_only, Vec::<BlockId>::new());
        assert_eq!(right_of(&diff.diff, b), Some(b));
        assert_eq!(diff.match_level(left.entry()), Some(MatchLevel::Entry));
        assert_eq!(diff.match_level(a), Some(MatchLevel::Full));
        assert_eq!(diff.match_level(b), Some(MatchLevel::Strict));
        assert_eq!(diff.match_level(exit), Some(MatchLevel::Full));
    }

    #[test]
    fn swapped_case_blocks_pair_by_content_not_crosswise() {
        // `entry -> dispatch -> {one, two} -> join`, the cases made, and
        // their switch edges added, in the opposite order on the right.
        let build = |swapped: bool| {
            let mut cfg = Cfg::new();
            let dispatch = block(&mut cfg, &[1]);
            let join = block(&mut cfg, &[2]);
            cfg.add_edge(cfg.entry(), dispatch, EdgeKind::Fallthrough);
            let contents = if swapped { [200, 100] } else { [100, 200] };
            let cases = contents.map(|content| {
                let case = block(&mut cfg, &[content]);
                cfg.add_edge(dispatch, case, EdgeKind::SwitchCase);
                cfg.add_edge(case, join, EdgeKind::Fallthrough);
                case
            });
            (cfg, cases)
        };
        let (left, [one, two]) = build(false);
        let (right, [right_two, right_one]) = build(true);

        let structural = CfgDiff::compute(&left, &right);
        assert_eq!(right_of(&structural, one), Some(right_two), "crosswise");

        let diff = cascade(&left, &right);
        assert_eq!(right_of(&diff.diff, one), Some(right_one));
        assert_eq!(right_of(&diff.diff, two), Some(right_two));
        assert_eq!(diff.match_level(one), Some(MatchLevel::Full));
        assert!(diff.diff.is_identical(), "{diff:?}");
    }

    #[test]
    fn a_changed_operand_pairs_at_loose_and_a_changed_opcode_falls_back() {
        let (left, [_, b, _]) = chain(201);
        let (right, _) = chain(202);
        let diff = cascade(&left, &right);
        assert_eq!(right_of(&diff.diff, b), Some(b));
        assert_eq!(diff.match_level(b), Some(MatchLevel::Loose));

        // A new opcode leaves only the wiring: the walk pairs `b`.
        let (right, _) = chain(900);
        let diff = cascade(&left, &right);
        assert_eq!(right_of(&diff.diff, b), Some(b));
        assert_eq!(diff.match_level(b), Some(MatchLevel::Walk));

        // An unreachable block of new content pairs by its shape.
        let (mut left, _) = chain(201);
        let (mut right, _) = chain(201);
        let orphan = block(&mut left, &[500]);
        block(&mut right, &[600]);
        let diff = cascade(&left, &right);
        assert_eq!(diff.match_level(orphan), Some(MatchLevel::Shape));
        assert!(diff.diff.left_only.is_empty() && diff.diff.right_only.is_empty());
    }

    #[test]
    fn equal_keys_pair_by_the_nearest_relative_position_ties_to_the_lower_id() {
        // Unreachable blocks sit in ascending order after the entry. `late`
        // is at 3/4 on the left; `first` at 1/5 and `last` at 4/5 on the
        // right.
        let mut left = Cfg::new();
        block(&mut left, &[1]);
        block(&mut left, &[2]);
        let late = block(&mut left, &[700]);
        let mut right = Cfg::new();
        let first = block(&mut right, &[700]);
        block(&mut right, &[1]);
        block(&mut right, &[2]);
        let last = block(&mut right, &[700]);
        let diff = cascade(&left, &right);
        assert_eq!(right_of(&diff.diff, late), Some(last));
        assert_eq!(diff.diff.right_only, [first]);

        // At 1/2 on the left, between 1/3 and 2/3: equally near both.
        let mut left = Cfg::new();
        let middle = block(&mut left, &[700]);
        let mut right = Cfg::new();
        let (lower, upper) = (block(&mut right, &[700]), block(&mut right, &[700]));
        let diff = cascade(&left, &right);
        assert_eq!(right_of(&diff.diff, middle), Some(lower));
        assert_eq!(diff.diff.right_only, [upper]);
    }

    #[test]
    fn removed_blocks_and_a_lone_entry_take_no_part() {
        let entry_only = Cfg::<u32>::new();
        let diff = cascade(&entry_only, &entry_only);
        assert_eq!(diff.levels, [MatchLevel::Entry]);
        assert_eq!(diff.match_level(BlockId::from_index(1)), None);

        let (mut left, [a, b, _]) = chain(200);
        let (right, _) = chain(200);
        left.remove_block(b);
        let diff = cascade(&left, &right);
        assert_eq!(diff.diff.right_only, [b]);
        assert_eq!(diff.match_level(b), None);
        assert_eq!(diff.match_level(a), Some(MatchLevel::Strict));
        assert_eq!(diff.levels.len(), diff.diff.matched.len());
    }

    const KINDS: [EdgeKind; 6] = [
        EdgeKind::Fallthrough,
        EdgeKind::ConditionalTrue,
        EdgeKind::ConditionalFalse,
        EdgeKind::SwitchCase,
        EdgeKind::Back,
        EdgeKind::ExceptionUnwind,
    ];

    /// An edge between two roles.
    type RoleEdge = (usize, usize, EdgeKind);

    /// Blocks as contents and edges by role, role 0 being the entry.
    struct Blueprint {
        contents: Vec<Vec<u32>>,
        edges: Vec<RoleEdge>,
    }

    impl Blueprint {
        /// Every block holds content of its own; the entry may not reach
        /// every block.
        fn random(rng: &mut Rng) -> Self {
            let count = 1 + rng.below(30);
            let contents = (0..count)
                .map(|role| vec![u32::try_from(role).expect("a small role") * 100 + 1])
                .collect();
            let edges = (0..rng.below(3 * count))
                .map(|_| {
                    (
                        rng.below(count),
                        rng.below(count),
                        KINDS[rng.below(KINDS.len())],
                    )
                })
                .collect();
            Self { contents, edges }
        }

        /// A copy edited at random: new and lost edges, changed operands
        /// and opcodes, new blocks.
        fn edited(&self, rng: &mut Rng) -> Self {
            let mut contents = self.contents.clone();
            for content in &mut contents {
                match rng.below(6) {
                    0 => content[0] += 1,
                    1 => content[0] += 5_000,
                    _ => {}
                }
            }
            for _ in 0..rng.below(4) {
                contents.push(vec![9_000 + u32::try_from(rng.below(5)).expect("small")]);
            }
            let mut edges: Vec<_> = self
                .edges
                .iter()
                .copied()
                .filter(|_| !rng.one_in(8))
                .collect();
            for _ in 0..rng.below(4) {
                let count = contents.len();
                edges.push((
                    rng.below(count),
                    rng.below(count),
                    KINDS[rng.below(KINDS.len())],
                ));
            }
            Self { contents, edges }
        }

        /// The CFG with role `i` made as block `slots[i]` (`slots[0]` is 0,
        /// the entry), and the edges added in the order given.
        fn build(&self, slots: &[usize], edges: &[RoleEdge]) -> Cfg<u32> {
            let mut cfg = Cfg::new();
            for _ in 1..self.contents.len() {
                cfg.new_block();
            }
            let id = |role: usize| BlockId::from_index(slots[role]);
            for (role, content) in self.contents.iter().enumerate() {
                cfg.block_mut(id(role))
                    .instructions_mut()
                    .extend_from_slice(content);
            }
            for &(source, target, kind) in edges {
                cfg.add_edge(id(source), id(target), kind);
            }
            cfg
        }

        fn identity(&self) -> Vec<usize> {
            (0..self.contents.len()).collect()
        }
    }

    #[test]
    fn a_renumbered_copy_is_identical_and_matched_in_full() {
        let mut rng = Rng::new(88);
        for _ in 0..200 {
            let shape = Blueprint::random(&mut rng);
            let mut slots = shape.identity();
            rng.shuffle(&mut slots[1..]);
            let original = shape.build(&shape.identity(), &shape.edges);
            let copy = shape.build(&slots, &shape.edges);

            let diff = cascade(&original, &copy);
            assert!(diff.diff.is_identical(), "{diff:?}");
            for pair in &diff.diff.matched {
                assert_eq!(pair.right.index(), slots[pair.left.index()]);
            }
            let mut levels = diff.levels.iter();
            assert_eq!(levels.next(), Some(&MatchLevel::Entry));
            assert!(levels.all(|&level| level == MatchLevel::Full), "{diff:?}");
        }
    }

    /// `edges` in a random order that keeps the edges one block has of one
    /// kind in their order.
    fn reordered(rng: &mut Rng, edges: &[RoleEdge]) -> Vec<RoleEdge> {
        let mut groups: BTreeMap<(usize, u8), VecDeque<RoleEdge>> = BTreeMap::new();
        for &edge in edges {
            groups
                .entry((edge.0, edge.2 as u8))
                .or_default()
                .push_back(edge);
        }
        let mut order = edges.to_vec();
        rng.shuffle(&mut order);
        order
            .iter()
            .map(|&(source, _, kind)| {
                groups
                    .get_mut(&(source, kind as u8))
                    .and_then(VecDeque::pop_front)
                    .expect("one slot per edge of the group")
            })
            .collect()
    }

    #[test]
    fn every_block_takes_part_once_whatever_order_the_edges_came_in() {
        let mut rng = Rng::new(89);
        for _ in 0..300 {
            let shape = Blueprint::random(&mut rng);
            let edited = shape.edited(&mut rng);
            let left = shape.build(&shape.identity(), &shape.edges);
            let right = edited.build(&edited.identity(), &edited.edges);
            let diff = cascade(&left, &right);

            let lefts: BTreeSet<BlockId> = diff.diff.matched.iter().map(|pair| pair.left).collect();
            let rights: BTreeSet<BlockId> =
                diff.diff.matched.iter().map(|pair| pair.right).collect();
            assert_eq!(lefts.len(), diff.diff.matched.len(), "one-to-one");
            assert_eq!(rights.len(), diff.diff.matched.len(), "one-to-one");
            assert!(
                lefts
                    .iter()
                    .all(|block| !diff.diff.left_only.contains(block))
            );
            assert!(
                rights
                    .iter()
                    .all(|block| !diff.diff.right_only.contains(block))
            );
            assert_eq!(lefts.len() + diff.diff.left_only.len(), left.block_count());
            assert_eq!(
                rights.len() + diff.diff.right_only.len(),
                right.block_count()
            );
            assert_eq!(diff.levels.len(), diff.diff.matched.len());

            let shuffled_left = shape.build(&shape.identity(), &reordered(&mut rng, &shape.edges));
            let shuffled_right =
                edited.build(&edited.identity(), &reordered(&mut rng, &edited.edges));
            assert_eq!(cascade(&shuffled_left, &shuffled_right), diff);
        }
    }
}
