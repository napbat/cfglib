//! Hot/cold order: a base order stable-partitioned by coldness class, with
//! the index where each cold section starts.

extern crate alloc;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::graph::coldness::{ColdClass, ColdnessClasses};
use crate::graph::view::DenseId;

/// A block order with the hot part first, and the indices where its two cold
/// sections start.
///
/// `order[..cold_start]` is the hot part: the entry, then the
/// [`Normal`](ColdClass::Normal) nodes. `order[cold_start..never_start]`
/// holds the [`Rare`](ColdClass::Rare) nodes and `order[never_start..]` the
/// [`Never`](ColdClass::Never) nodes, so
/// `cold_start <= never_start <= order.len()`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SplitOrder<N = BlockId> {
    /// Every node of the base order, hot part first.
    pub order: Vec<N>,
    /// The index of the first cold node: `order[cold_start..]` is `Rare`
    /// then `Never`.
    pub cold_start: usize,
    /// The index of the first `Never` node: `order[never_start..]` is
    /// `Never`.
    pub never_start: usize,
}

/// Stable-partitions `base` into its `Normal`, `Rare`, and `Never` nodes,
/// keeping `entry` first whatever its class.
///
/// `base` is any order: reverse postorder, a profile-guided order, or one the
/// caller chose. The result is a permutation of it in which the nodes of each
/// class keep their relative order. When `base` holds `entry`, the entry is
/// `order[0]` and part of the hot part even when it is cold: it can be `Rare`
/// when every path from it ends in a node seeded `Rare`, and it still has to
/// lead the function. A node that appears more than once in `base` appears
/// as often in the result.
///
/// Time is O(n) for `n` nodes of `base`, on top of
/// [`ColdnessClasses::compute`].
///
/// # Emitting the sections
///
/// [`linearize`](crate::linearize) needs no change to split a function. Give
/// it each part as a [`BlockOrder::Custom`](crate::BlockOrder::Custom) and it
/// returns one stream per section. The last block of each part has no next
/// block, so a layout edge that crosses the split gets an explicit jump, and
/// the cold part can be placed anywhere.
///
/// # Examples
///
/// The false arm of a branch leads to an abort that the consumer seeds
/// `Rare`; the arm and the abort move to the cold section, and the branch
/// block jumps to the arm.
///
/// ```
/// use cfglib::{
///     BlockId, BlockOrder, Cfg, ColdClass, ColdnessClasses, ColdnessGraph, EdgeKind, Emitter,
///     hot_cold_order, linearize,
/// };
///
/// /// Seeds `abort` `Rare`; every edge passes any class on.
/// struct Seeds<'a> {
///     cfg: &'a Cfg<&'static str>,
///     abort: BlockId,
/// }
///
/// impl ColdnessGraph for Seeds<'_> {
///     type NodeId = BlockId;
///     fn node_bound(&self) -> usize {
///         self.cfg.block_bound()
///     }
///     fn entry(&self) -> BlockId {
///         self.cfg.entry()
///     }
///     fn reachable(&self) -> impl Iterator<Item = BlockId> + '_ {
///         self.cfg.reverse_postorder().into_iter()
///     }
///     fn records(&self, node: BlockId) -> impl Iterator<Item = (BlockId, ColdClass)> + '_ {
///         self.cfg.successors(node).map(|target| (target, ColdClass::Normal))
///     }
///     fn incoming(&self, node: BlockId) -> impl Iterator<Item = (BlockId, ColdClass)> + '_ {
///         self.cfg.predecessors(node).map(|source| (source, ColdClass::Normal))
///     }
///     fn seed(&self, node: BlockId) -> ColdClass {
///         if node == self.abort { ColdClass::Rare } else { ColdClass::Normal }
///     }
///     fn ends(&self, _node: BlockId) -> bool {
///         false
///     }
/// }
///
/// /// Names each transfer, without its target.
/// struct Asm;
///
/// impl Emitter<&'static str> for Asm {
///     fn emit_jump(&self, _target: BlockId) -> &'static str {
///         "jmp"
///     }
///     fn emit_conditional_branch(&self, _condition: &&'static str, _target: BlockId) -> &'static str {
///         "bcc"
///     }
///     fn emit_block_start(&self, _block: BlockId) -> Option<&'static str> {
///         None
///     }
/// }
///
/// let mut cfg = Cfg::new();
/// let entry = cfg.entry();
/// let [then_arm, join, else_arm, abort] = [(); 4].map(|()| cfg.new_block());
/// cfg.block_mut(entry).push("test");
/// cfg.block_mut(then_arm).push("work");
/// cfg.block_mut(join).push("ret");
/// cfg.block_mut(else_arm).push("report");
/// cfg.block_mut(abort).push("abort");
/// cfg.add_edge(entry, then_arm, EdgeKind::ConditionalTrue);
/// cfg.add_edge(entry, else_arm, EdgeKind::ConditionalFalse);
/// cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);
/// cfg.add_edge(else_arm, abort, EdgeKind::Fallthrough);
///
/// let classes = ColdnessClasses::compute(&Seeds { cfg: &cfg, abort });
/// let split = hot_cold_order(&cfg.reverse_postorder(), &classes, entry);
/// assert_eq!(split.order, [entry, then_arm, join, else_arm, abort]);
/// assert_eq!((split.cold_start, split.never_start), (3, 5));
///
/// let hot = split.order[..split.cold_start].to_vec();
/// let cold = split.order[split.cold_start..].to_vec();
/// let hot = linearize(&cfg, BlockOrder::Custom(hot), &Asm)?;
/// let cold = linearize(&cfg, BlockOrder::Custom(cold), &Asm)?;
/// let text = |stream: &[cfglib::LinearInst<&'static str>]| -> Vec<&'static str> {
///     stream.iter().map(|line| line.inst).collect()
/// };
/// // The false edge crosses the split, so the branch block jumps to it.
/// assert_eq!(text(&hot), ["test", "bcc", "jmp", "work", "ret"]);
/// assert_eq!(text(&cold), ["report", "abort"]);
/// # Ok::<(), cfglib::LinearizeError>(())
/// ```
#[must_use]
pub fn hot_cold_order<N: DenseId>(
    base: &[N],
    classes: &ColdnessClasses<N>,
    entry: N,
) -> SplitOrder<N> {
    let entry_position = base.iter().position(|&node| node == entry);
    let of_class = |class: ColdClass| {
        base.iter()
            .enumerate()
            .filter(move |&(position, &node)| {
                Some(position) != entry_position && classes.class(node) == class
            })
            .map(|(_, &node)| node)
    };

    let mut order = Vec::with_capacity(base.len());
    order.extend(entry_position.map(|_| entry));
    order.extend(of_class(ColdClass::Normal));
    let cold_start = order.len();
    order.extend(of_class(ColdClass::Rare));
    let never_start = order.len();
    order.extend(of_class(ColdClass::Never));
    SplitOrder {
        order,
        cold_start,
        never_start,
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;
    use alloc::vec::Vec;

    use super::{SplitOrder, hot_cold_order};
    use crate::block::BlockId;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::coldness::{ColdClass, ColdnessClasses, ColdnessGraph};
    use crate::test_util::rng::Rng;
    use crate::transform::linearize::{BlockOrder, Emitter, LinearInst, linearize};

    /// A toy instruction that names the targets of the transfers.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Op {
        Plain,
        Jump(BlockId),
        Branch(BlockId),
        Label(BlockId),
    }

    struct Asm;

    impl Emitter<Op> for Asm {
        fn emit_jump(&self, target: BlockId) -> Op {
            Op::Jump(target)
        }

        fn emit_conditional_branch(&self, _condition: &Op, target: BlockId) -> Op {
            Op::Branch(target)
        }

        fn emit_block_start(&self, block: BlockId) -> Option<Op> {
            Some(Op::Label(block))
        }
    }

    /// A rooted graph over numbered nodes with node 0 as the entry.
    struct Toy {
        records: Vec<Vec<(usize, ColdClass)>>,
        incoming: Vec<Vec<(usize, ColdClass)>>,
        seeds: Vec<ColdClass>,
        ends: Vec<bool>,
        reachable: Vec<usize>,
    }

    impl Toy {
        fn new(
            records: Vec<Vec<(usize, ColdClass)>>,
            seeds: Vec<ColdClass>,
            ends: Vec<bool>,
        ) -> Self {
            let mut incoming = vec![Vec::new(); records.len()];
            for (source, list) in records.iter().enumerate() {
                for &(target, cap) in list {
                    incoming[target].push((source, cap));
                }
            }
            let mut seen = vec![false; records.len()];
            let mut reachable = Vec::new();
            let mut work = vec![0];
            while let Some(node) = work.pop() {
                if !core::mem::replace(&mut seen[node], true) {
                    reachable.push(node);
                    work.extend(records[node].iter().map(|&(target, _)| target));
                }
            }
            Self {
                records,
                incoming,
                seeds,
                ends,
                reachable,
            }
        }

        /// Every block of `cfg` with a record per edge, all uncapped, and
        /// the blocks of `rare` seeded `Rare`.
        fn of_cfg(cfg: &Cfg<Op>, rare: &[BlockId]) -> Self {
            let records = (0..cfg.block_bound())
                .map(|index| {
                    cfg.successors(BlockId::from_index(index))
                        .map(|target| (target.index(), ColdClass::Normal))
                        .collect()
                })
                .collect();
            let seeds = (0..cfg.block_bound())
                .map(|index| {
                    if rare.contains(&BlockId::from_index(index)) {
                        ColdClass::Rare
                    } else {
                        ColdClass::Normal
                    }
                })
                .collect();
            Self::new(records, seeds, vec![false; cfg.block_bound()])
        }
    }

    impl ColdnessGraph for Toy {
        type NodeId = usize;

        fn node_bound(&self) -> usize {
            self.records.len()
        }

        fn entry(&self) -> usize {
            0
        }

        fn reachable(&self) -> impl Iterator<Item = usize> + '_ {
            self.reachable.iter().copied()
        }

        fn records(&self, node: usize) -> impl Iterator<Item = (usize, ColdClass)> + '_ {
            self.records[node].iter().copied()
        }

        fn incoming(&self, node: usize) -> impl Iterator<Item = (usize, ColdClass)> + '_ {
            self.incoming[node].iter().copied()
        }

        fn seed(&self, node: usize) -> ColdClass {
            self.seeds[node]
        }

        fn ends(&self, node: usize) -> bool {
            self.ends[node]
        }
    }

    /// The classes of `graph` keyed by block, for a graph built over a CFG.
    fn block_classes(graph: &Toy) -> ColdnessClasses<BlockId> {
        struct ByBlock<'a>(&'a Toy);

        impl ColdnessGraph for ByBlock<'_> {
            type NodeId = BlockId;

            fn node_bound(&self) -> usize {
                self.0.node_bound()
            }

            fn entry(&self) -> BlockId {
                BlockId::from_index(0)
            }

            fn reachable(&self) -> impl Iterator<Item = BlockId> + '_ {
                self.0.reachable().map(BlockId::from_index)
            }

            fn records(&self, node: BlockId) -> impl Iterator<Item = (BlockId, ColdClass)> + '_ {
                self.0
                    .records(node.index())
                    .map(|(target, cap)| (BlockId::from_index(target), cap))
            }

            fn incoming(&self, node: BlockId) -> impl Iterator<Item = (BlockId, ColdClass)> + '_ {
                self.0
                    .incoming(node.index())
                    .map(|(source, cap)| (BlockId::from_index(source), cap))
            }

            fn seed(&self, node: BlockId) -> ColdClass {
                self.0.seed(node.index())
            }

            fn ends(&self, node: BlockId) -> bool {
                self.0.ends(node.index())
            }
        }

        ColdnessClasses::compute(&ByBlock(graph))
    }

    /// Checks that no jump of `stream` targets the block after its source.
    fn assert_no_jump_to_next(stream: &[LinearInst<Op>], part: &[BlockId]) {
        for line in stream {
            if let Op::Jump(target) = line.inst {
                let position = part
                    .iter()
                    .position(|&block| block == line.block)
                    .expect("every line belongs to a block of the part");
                assert_ne!(part.get(position + 1), Some(&target), "{stream:?}");
            }
        }
    }

    #[test]
    fn a_rare_false_arm_moves_to_the_cold_part_and_the_branch_jumps_to_it() {
        let mut cfg = Cfg::<Op>::new();
        let entry = cfg.entry();
        let [then_arm, join, else_arm, abort] = [(); 4].map(|()| cfg.new_block());
        for block in [entry, then_arm, join, else_arm, abort] {
            cfg.block_mut(block).push(Op::Plain);
        }
        cfg.add_edge(entry, then_arm, EdgeKind::ConditionalTrue);
        cfg.add_edge(entry, else_arm, EdgeKind::ConditionalFalse);
        cfg.add_edge(then_arm, join, EdgeKind::Fallthrough);
        cfg.add_edge(else_arm, abort, EdgeKind::Fallthrough);
        let classes = block_classes(&Toy::of_cfg(&cfg, &[abort]));
        assert_eq!(classes.class(else_arm), ColdClass::Rare);

        let split = hot_cold_order(&cfg.reverse_postorder(), &classes, entry);
        assert_eq!(split.order, [entry, then_arm, join, else_arm, abort]);
        assert_eq!((split.cold_start, split.never_start), (3, 5));

        let (hot, cold) = split.order.split_at(split.cold_start);
        let hot_stream = linearize(&cfg, BlockOrder::Custom(hot.to_vec()), &Asm).unwrap();
        let cold_stream = linearize(&cfg, BlockOrder::Custom(cold.to_vec()), &Asm).unwrap();
        let entry_transfers: Vec<Op> = hot_stream
            .iter()
            .filter(|line| line.block == entry && line.index == usize::MAX)
            .map(|line| line.inst)
            .collect();
        assert_eq!(
            entry_transfers,
            [Op::Label(entry), Op::Branch(then_arm), Op::Jump(else_arm)],
            "the false edge crosses the split"
        );
        assert_no_jump_to_next(&hot_stream, hot);
        assert_no_jump_to_next(&cold_stream, cold);
    }

    #[test]
    fn an_all_normal_graph_keeps_the_base_order() {
        let mut cfg = Cfg::<Op>::new();
        let [a, b, c] = [(); 3].map(|()| cfg.new_block());
        cfg.add_edge(cfg.entry(), a, EdgeKind::Fallthrough);
        cfg.add_edge(a, b, EdgeKind::Fallthrough);
        cfg.add_edge(cfg.entry(), c, EdgeKind::Jump);
        let classes = block_classes(&Toy::of_cfg(&cfg, &[]));

        let base = cfg.reverse_postorder();
        let split = hot_cold_order(&base, &classes, cfg.entry());
        assert_eq!(
            split,
            SplitOrder {
                cold_start: base.len(),
                never_start: base.len(),
                order: base,
            }
        );
    }

    #[test]
    fn a_cold_entry_still_leads_the_hot_part() {
        // Every path from the entry ends in the abort, so the entry is `Rare`.
        let graph = Toy::new(
            vec![vec![(1, ColdClass::Normal)], vec![]],
            vec![ColdClass::Normal, ColdClass::Rare],
            vec![false; 2],
        );
        let classes = ColdnessClasses::compute(&graph);
        assert_eq!(classes.class(0), ColdClass::Rare);

        let split = hot_cold_order(&[1, 0], &classes, 0);
        assert_eq!(split.order, [0, 1]);
        assert_eq!((split.cold_start, split.never_start), (1, 2));
    }

    #[test]
    fn an_empty_base_and_a_missing_entry_need_no_special_case() {
        let graph = Toy::new(
            vec![vec![], vec![]],
            vec![ColdClass::Normal; 2],
            vec![false; 2],
        );
        let classes = ColdnessClasses::compute(&graph);

        let empty = hot_cold_order(&[], &classes, 0);
        assert_eq!(
            (empty.order.len(), empty.cold_start, empty.never_start),
            (0, 0, 0)
        );

        // Node 1 is unreached, so `Never`; the entry is not in the base.
        let split = hot_cold_order(&[1], &classes, 0);
        assert_eq!(split.order, [1]);
        assert_eq!((split.cold_start, split.never_start), (0, 0));
    }

    /// A random graph of up to 24 nodes with random caps, seeds, and `ends`.
    fn random_graph(rng: &mut Rng) -> Toy {
        let count = 1 + rng.below(24);
        let cold = [ColdClass::Never, ColdClass::Rare];
        let pick = |rng: &mut Rng, normal_percent| {
            if rng.chance(normal_percent) {
                ColdClass::Normal
            } else {
                cold[rng.below(cold.len())]
            }
        };
        let records = (0..count)
            .map(|_| {
                (0..rng.below(4))
                    .map(|_| (rng.below(count), pick(rng, 80)))
                    .collect()
            })
            .collect();
        let seeds = (0..count).map(|_| pick(rng, 85)).collect();
        let ends = (0..count).map(|_| rng.one_in(5)).collect();
        Toy::new(records, seeds, ends)
    }

    #[test]
    fn the_order_is_a_stable_partition_of_the_base_with_the_entry_first() {
        let mut rng = Rng::new(57);
        for round in 0..400 {
            let graph = random_graph(&mut rng);
            let classes = ColdnessClasses::compute(&graph);
            let mut base: Vec<usize> = (0..graph.node_bound()).filter(|_| !rng.one_in(6)).collect();
            rng.shuffle(&mut base);

            let split = hot_cold_order(&base, &classes, 0);
            let context = alloc::format!("round {round}: base {base:?}, split {split:?}");

            let mut sorted_base = base.clone();
            sorted_base.sort_unstable();
            let mut sorted_order = split.order.clone();
            sorted_order.sort_unstable();
            assert_eq!(sorted_order, sorted_base, "{context}: a permutation");
            assert!(split.cold_start <= split.never_start, "{context}");
            assert!(split.never_start <= split.order.len(), "{context}");

            let has_entry = base.contains(&0);
            if has_entry {
                assert_eq!(split.order[0], 0, "{context}: the entry leads");
            }
            let skip = usize::from(has_entry);
            let sections = [
                (ColdClass::Normal, &split.order[skip..split.cold_start]),
                (
                    ColdClass::Rare,
                    &split.order[split.cold_start..split.never_start],
                ),
                (ColdClass::Never, &split.order[split.never_start..]),
            ];
            for (class, section) in sections {
                let expected: Vec<usize> = base
                    .iter()
                    .copied()
                    .filter(|&node| node != 0 && classes.class(node) == class)
                    .collect();
                assert_eq!(section, expected, "{context}: {class:?} keeps base order");
            }
        }
    }
}
