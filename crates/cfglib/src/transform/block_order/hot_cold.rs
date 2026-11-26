//! Hot/cold order: a base order stable-partitioned by coldness class.

extern crate alloc;
use alloc::vec::Vec;

use crate::graph::coldness::{ColdClass, ColdnessClasses};
use crate::graph::view::RootedView;

/// Stable-partitions `base` into its hot part, its [`Rare`](ColdClass::Rare)
/// nodes, and its [`Never`](ColdClass::Never) nodes, keeping the root of
/// `graph` first whatever its class.
///
/// The result is `[hot, rare, never]`: `hot` is the root, then the
/// [`Normal`](ColdClass::Normal) nodes. `base` is any order: reverse
/// postorder, a profile-guided order, or one the caller chose, and
/// `classes` are the classes of `graph`. The three parts together are a
/// permutation of `base` in which the nodes of each class keep their relative
/// order. When `base` holds the root, the root leads `hot` even when it is
/// cold: it can be `Rare` when every path from it ends in a node seeded
/// `Rare`, and it still has to lead the function. A node that appears more
/// than once in `base` appears as often in the result.
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
///     BlockId, BlockOrder, Cfg, ColdClass, ColdnessClasses, EdgeKind, Emitter, hot_cold_order,
///     linearize,
/// };
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
/// // `abort` is seeded `Rare`; every edge passes any class on.
/// let seed = |block| if block == abort { ColdClass::Rare } else { ColdClass::Normal };
/// let classes = ColdnessClasses::compute(&cfg, seed, |_| ColdClass::Normal, |_| false);
/// let [hot, rare, never] = hot_cold_order(&cfg, &cfg.reverse_postorder(), &classes);
/// assert_eq!(hot, [entry, then_arm, join]);
/// assert_eq!(rare, [else_arm, abort]);
/// assert!(never.is_empty());
///
/// let hot = linearize(&cfg, BlockOrder::Custom(hot), &Asm)?;
/// let cold = linearize(&cfg, BlockOrder::Custom(rare), &Asm)?;
/// let text = |stream: &[cfglib::LinearInst<&'static str>]| -> Vec<&'static str> {
///     stream.iter().map(|line| line.inst).collect()
/// };
/// // The false edge crosses the split, so the branch block jumps to it.
/// assert_eq!(text(&hot), ["test", "bcc", "jmp", "work", "ret"]);
/// assert_eq!(text(&cold), ["report", "abort"]);
/// # Ok::<(), cfglib::LinearizeError>(())
/// ```
#[must_use]
pub fn hot_cold_order<G: RootedView>(
    graph: &G,
    base: &[G::NodeId],
    classes: &ColdnessClasses<G::NodeId>,
) -> [Vec<G::NodeId>; 3] {
    let root = graph.root();
    let root_position = base.iter().position(|&node| node == root);
    let [mut hot, mut rare, mut never] = [Vec::with_capacity(base.len()), Vec::new(), Vec::new()];
    hot.extend(root_position.map(|_| root));
    for (position, &node) in base.iter().enumerate() {
        if Some(position) == root_position {
            continue;
        }
        match classes.class(node) {
            ColdClass::Normal => hot.push(node),
            ColdClass::Rare => rare.push(node),
            ColdClass::Never => never.push(node),
        }
    }
    [hot, rare, never]
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec;
    use alloc::vec::Vec;

    use super::hot_cold_order;
    use crate::block::BlockId;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::coldness::{ColdClass, ColdnessClasses};
    use crate::graph::store::{Graph, NodeId};
    use crate::graph::view::{DenseId, Rooted};
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

    /// The classes of `cfg` with the blocks of `rare` seeded `Rare` and every
    /// edge uncapped.
    fn block_classes(cfg: &Cfg<Op>, rare: &[BlockId]) -> ColdnessClasses {
        let seed = |block| {
            if rare.contains(&block) {
                ColdClass::Rare
            } else {
                ColdClass::Normal
            }
        };
        ColdnessClasses::compute(cfg, seed, |_| ColdClass::Normal, |_| false)
    }

    /// A graph over numbered nodes, each edge holding its cap, with node 0
    /// as the root, and its classes under `seeds` and `ends`.
    struct Toy {
        graph: Graph<(), ColdClass>,
        classes: ColdnessClasses<NodeId>,
    }

    impl Toy {
        fn new(edges: &[Vec<(usize, ColdClass)>], seeds: &[ColdClass], ends: &[bool]) -> Self {
            let mut graph = Graph::new();
            let nodes: Vec<NodeId> = edges.iter().map(|_| graph.add_node(())).collect();
            for (source, targets) in edges.iter().enumerate() {
                for &(target, cap) in targets {
                    graph.add_edge(nodes[source], nodes[target], cap);
                }
            }
            let classes = ColdnessClasses::compute(
                &Rooted::new(&graph, nodes[0]),
                |node| seeds[node.index()],
                |edge| *graph.edge(edge).payload(),
                |node| ends[node.index()],
            );
            Self { graph, classes }
        }

        fn split(&self, base: &[usize]) -> [Vec<usize>; 3] {
            let rooted = Rooted::new(&self.graph, NodeId::from_index(0));
            let base: Vec<NodeId> = base.iter().map(|&node| NodeId::from_index(node)).collect();
            hot_cold_order(&rooted, &base, &self.classes)
                .map(|part| part.into_iter().map(DenseId::index).collect())
        }

        fn class(&self, node: usize) -> ColdClass {
            self.classes.class(NodeId::from_index(node))
        }
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
        let classes = block_classes(&cfg, &[abort]);
        assert_eq!(classes.class(else_arm), ColdClass::Rare);

        let [hot, rare, never] = hot_cold_order(&cfg, &cfg.reverse_postorder(), &classes);
        assert_eq!(hot, [entry, then_arm, join]);
        assert_eq!(rare, [else_arm, abort]);
        assert_eq!(never, []);

        let hot_stream = linearize(&cfg, BlockOrder::Custom(hot.clone()), &Asm).unwrap();
        let cold_stream = linearize(&cfg, BlockOrder::Custom(rare.clone()), &Asm).unwrap();
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
        assert_no_jump_to_next(&hot_stream, &hot);
        assert_no_jump_to_next(&cold_stream, &rare);
    }

    #[test]
    fn an_all_normal_graph_keeps_the_base_order() {
        let mut cfg = Cfg::<Op>::new();
        let [a, b, c] = [(); 3].map(|()| cfg.new_block());
        cfg.add_edge(cfg.entry(), a, EdgeKind::Fallthrough);
        cfg.add_edge(a, b, EdgeKind::Fallthrough);
        cfg.add_edge(cfg.entry(), c, EdgeKind::Jump);
        let classes = block_classes(&cfg, &[]);

        let base = cfg.reverse_postorder();
        assert_eq!(
            hot_cold_order(&cfg, &base, &classes),
            [base.clone(), Vec::new(), Vec::new()]
        );
    }

    #[test]
    fn a_cold_entry_still_leads_the_hot_part() {
        // Every path from the entry ends in the abort, so the entry is `Rare`.
        let toy = Toy::new(
            &[vec![(1, ColdClass::Normal)], vec![]],
            &[ColdClass::Normal, ColdClass::Rare],
            &[false; 2],
        );
        assert_eq!(toy.class(0), ColdClass::Rare);
        assert_eq!(toy.split(&[1, 0]), [vec![0], vec![1], vec![]]);
    }

    #[test]
    fn an_empty_base_and_a_missing_entry_need_no_special_case() {
        let toy = Toy::new(&[vec![], vec![]], &[ColdClass::Normal; 2], &[false; 2]);
        assert_eq!(toy.split(&[]), [Vec::<usize>::new(), vec![], vec![]]);
        // Node 1 is unreached, so `Never`; the entry is not in the base.
        assert_eq!(toy.split(&[1]), [vec![], vec![], vec![1]]);
    }

    /// A random graph of up to 24 nodes with random caps, seeds, and `ends`.
    fn random_toy(rng: &mut Rng) -> Toy {
        let count = 1 + rng.below(24);
        let cold = [ColdClass::Never, ColdClass::Rare];
        let pick = |rng: &mut Rng, normal_percent| {
            if rng.chance(normal_percent) {
                ColdClass::Normal
            } else {
                cold[rng.below(cold.len())]
            }
        };
        let edges: Vec<Vec<(usize, ColdClass)>> = (0..count)
            .map(|_| {
                (0..rng.below(4))
                    .map(|_| (rng.below(count), pick(rng, 80)))
                    .collect()
            })
            .collect();
        let seeds: Vec<ColdClass> = (0..count).map(|_| pick(rng, 85)).collect();
        let ends: Vec<bool> = (0..count).map(|_| rng.one_in(5)).collect();
        Toy::new(&edges, &seeds, &ends)
    }

    #[test]
    fn the_order_is_a_stable_partition_of_the_base_with_the_entry_first() {
        let mut rng = Rng::new(57);
        for round in 0..400 {
            let toy = random_toy(&mut rng);
            let mut base: Vec<usize> = (0..toy.graph.node_bound())
                .filter(|_| !rng.one_in(6))
                .collect();
            rng.shuffle(&mut base);

            let split = toy.split(&base);
            let context = alloc::format!("round {round}: base {base:?}, split {split:?}");

            let mut sorted_base = base.clone();
            sorted_base.sort_unstable();
            let mut sorted_order = split.concat();
            sorted_order.sort_unstable();
            assert_eq!(sorted_order, sorted_base, "{context}: a permutation");

            let has_entry = base.contains(&0);
            if has_entry {
                assert_eq!(split[0][0], 0, "{context}: the entry leads");
            }
            let skip = usize::from(has_entry);
            let sections = [
                (ColdClass::Normal, &split[0][skip..]),
                (ColdClass::Rare, &split[1][..]),
                (ColdClass::Never, &split[2][..]),
            ];
            for (class, section) in sections {
                let expected: Vec<usize> = base
                    .iter()
                    .copied()
                    .filter(|&node| node != 0 && toy.class(node) == class)
                    .collect();
                assert_eq!(section, expected, "{context}: {class:?} keeps base order");
            }
        }
    }
}
