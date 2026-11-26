//! Irreducible-to-reducible CFG transformations: node splitting and dispatch.
//!
//! An irreducible CFG contains cycles with multiple entry points.
//! [`make_reducible`] removes them by choosing one entry of such a cycle as
//! its header and duplicating everything the other entries lead to inside the
//! cycle, so that the original blocks are entered only through the header and
//! the copies sit in front of it.
//!
//! [`make_reducible_by_dispatch`] copies nothing. It enters every
//! irreducible cycle through one new hub block that branches on a selector
//! value, which a new block on each edge into the cycle's entries sets, with
//! code a [`DispatchBuilder`] supplies. It grows polynomially, by at most the
//! edges times the irreducible nesting depth, where splitting can grow
//! exponentially, at the price of selector code instead of copies.
//!
//! Both read the irreducible cycles, and their entries, off one forest:
//! [`LoopForest::compute_cycles`], innermost cycles first.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::cmp::Reverse;
use smallvec::SmallVec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::loop_forest::LoopForest;
use crate::graph::traverse::reachable;
use crate::graph::view::{Direction, Induced};

mod dispatch;

pub use dispatch::{DispatchBuilder, DispatchReport, make_reducible_by_dispatch};

/// Transform an irreducible CFG into a reducible one by node splitting.
///
/// Returns the copies, in the order they were added. If the CFG is already
/// reducible, returns nothing and makes no changes.
///
/// Each round takes an innermost cycle of [`LoopForest::compute_cycles`]
/// with more than one entry and picks the entry whose rivals lead to the
/// fewest blocks of the cycle without passing it. Those blocks are copied, the
/// edges from outside the cycle into the other entries are redirected to the
/// copies, and the copies keep the kind, weight and payload of every edge they
/// were copied from. Cycles inside the copies and inside the original are
/// handled by later rounds.
///
/// **Caution**: node splitting can cause exponential code growth in
/// pathological cases. For most real-world binaries the duplication
/// is modest.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, DominatorTree, EdgeKind, is_reducible, make_reducible};
///
/// // A reducible CFG is left alone.
/// let mut cfg = Cfg::<u32>::new();
/// let [a, b] = [(); 2].map(|()| cfg.new_block());
/// cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
/// cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
/// assert_eq!(make_reducible(&mut cfg), []);
///
/// // Once `a` and `b` jump to each other, the cycle has two entries: one of
/// // them is copied in front of the other.
/// cfg.add_edge(a, b, EdgeKind::Jump);
/// cfg.add_edge(b, a, EdgeKind::Jump);
/// assert!(!is_reducible(&cfg, &DominatorTree::compute(&cfg)));
/// assert_eq!(make_reducible(&mut cfg).len(), 1);
/// assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)));
/// ```
pub fn make_reducible<I: Clone, E: Clone>(cfg: &mut Cfg<I, E>) -> Vec<BlockId> {
    let mut copies = Vec::new();
    loop {
        // Copies add blocks, so each round finds the cycles again.
        let forest = LoopForest::compute_cycles(cfg);
        let Some(cycle) = forest
            .loops()
            .filter(|&lp| forest.is_irreducible(lp))
            .min_by_key(|&lp| Reverse(forest.depth(lp)))
        else {
            return copies;
        };
        let mut member = vec![false; cfg.block_bound()];
        for block in forest.body(cycle) {
            member[block.index()] = true;
        }
        let mut entries = forest.entries(cycle);
        entries.sort_unstable();
        split_cycle(cfg, &member, &entries, &mut copies);
    }
}

/// The blocks of `cfg` that can be reached from `start` in `direction`
/// without entering a block outside `within`, which holds every block of
/// `start`.
fn closure<I, E>(
    cfg: &Cfg<I, E>,
    start: impl IntoIterator<Item = BlockId>,
    direction: Direction,
    within: &[bool],
) -> Vec<bool> {
    reachable(&Induced::new(cfg, within), start, direction)
}

/// Make the cycle of the blocks in `member` single-entry: keep one of its
/// `entries` (ascending) as its header, copy the blocks the other entries lead
/// to without passing the header, and send every edge from outside the cycle
/// into those entries to the copies, which are appended to `copies`.
///
/// The blocks the other entries lead to are closed under the cycle's edges
/// except for the edge back to the header, so the copies need no edge into
/// the cycle other than that one. The header is the function entry when it
/// is an entry of the cycle, since nothing can be put in front of it, and
/// otherwise the entry with the smallest set to copy.
fn split_cycle<I: Clone, E: Clone>(
    cfg: &mut Cfg<I, E>,
    member: &[bool],
    entries: &[BlockId],
    copies: &mut Vec<BlockId>,
) {
    let region_without = |header: BlockId| {
        let mut allowed = member.to_vec();
        allowed[header.index()] = false;
        let others = entries.iter().copied().filter(|&entry| entry != header);
        closure(cfg, others, Direction::Forward, &allowed)
    };
    let size = |region: &[bool]| region.iter().filter(|&&copied| copied).count();
    let (header, region) = if entries.contains(&cfg.entry()) {
        (cfg.entry(), region_without(cfg.entry()))
    } else {
        entries
            .iter()
            .map(|&header| (header, region_without(header)))
            .min_by_key(|(_, region)| size(region))
            .expect("a cycle with several entries has entries")
    };

    let mut copy_of: Vec<Option<BlockId>> = vec![None; cfg.block_bound()];
    let originals: Vec<BlockId> = cfg
        .block_ids()
        .filter(|block| region[block.index()])
        .collect();
    for &block in &originals {
        let copy = cfg.new_block();
        let instructions = cfg.block(block).instructions().to_vec();
        cfg.block_mut(copy).instructions_mut().extend(instructions);
        copy_of[block.index()] = Some(copy);
        copies.push(copy);
    }

    // Each copy takes the edges of its original in the same adjacency order,
    // with fresh identities and all the metadata of the edge it copies. An
    // edge to a copied block goes to its copy; the others (to the header, or
    // out of the cycle) go where the original did.
    for &block in &originals {
        let copy = copy_of[block.index()].expect("every copied block has a copy");
        let outgoing: Vec<(BlockId, EdgeKind, Option<f64>, E)> = cfg
            .outgoing(block)
            .map(|id| {
                let edge = cfg.edge(id);
                (
                    edge.target(),
                    edge.kind(),
                    edge.weight(),
                    edge.payload().clone(),
                )
            })
            .collect();
        for (target, kind, weight, payload) in outgoing {
            let target = copy_of[target.index()].unwrap_or(target);
            match weight {
                Some(weight) => {
                    cfg.add_weighted_edge_with_payload(copy, target, kind, weight, payload);
                }
                None => {
                    cfg.add_edge_with_payload(copy, target, kind, payload);
                }
            }
        }
    }

    // Edges from outside the cycle now enter the copies; the originals keep
    // the edges from inside the cycle, so the header is their only way in.
    for &entry in entries.iter().filter(|&&entry| entry != header) {
        let from_outside: SmallVec<[EdgeId; 4]> = cfg
            .incoming(entry)
            .filter(|&id| !member[cfg.edge(id).source().index()])
            .collect();
        let copy = copy_of[entry.index()].expect("every entry other than the header is copied");
        for id in from_outside {
            cfg.redirect_edge_target(id, copy);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::graph::structure::is_reducible;
    use crate::test_util::{MockInst, ff};

    #[test]
    fn already_reducible_is_noop() {
        // Simple diamond: entry → A → merge, entry → B → merge.
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("entry"));
        cfg.block_mut(a).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.block_mut(merge).instructions_mut().push(ff("merge"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(b, merge, EdgeKind::Fallthrough);

        let dom = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &dom));
        assert_eq!(make_reducible(&mut cfg), []);
    }

    #[test]
    fn irreducible_cycle_is_fixed() {
        // Build an irreducible CFG:
        //   entry → A, entry → B
        //   A → B, B → A   (cycle with two entries)
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("entry"));
        cfg.block_mut(a).instructions_mut().push(ff("a"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, b, EdgeKind::Fallthrough);
        cfg.add_edge(b, a, EdgeKind::Fallthrough);

        let dom = DominatorTree::compute(&cfg);
        assert!(!is_reducible(&cfg, &dom), "should be irreducible before");

        let copies = make_reducible(&mut cfg);
        assert_ne!(copies.len(), 0, "should have split at least one node");

        let dom2 = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &dom2), "should be reducible after");
    }

    /// `entry -> {q, r}`, the cycle `q -> w -> s -> {w, q}`, and `r -> s`: the
    /// cycle has two entries, `q` and `s`, and `w`, which only the cycle
    /// reaches, is where the first retreating edge of a depth-first search that
    /// goes through `q` first ends. `q_first` picks which of `q` and `r` the
    /// entry lists first, so the search meets the cycle through either.
    fn cycle_with_an_interior_witness(q_first: bool) -> Cfg<()> {
        let mut cfg: Cfg<()> = Cfg::new();
        let [q, w, s, r] = [(); 4].map(|()| cfg.new_block());
        let (first, second) = if q_first { (q, r) } else { (r, q) };
        cfg.add_edge(cfg.entry(), first, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), second, EdgeKind::ConditionalFalse);
        cfg.add_edge(q, w, EdgeKind::Fallthrough);
        cfg.add_edge(w, s, EdgeKind::Fallthrough);
        cfg.add_edge(s, w, EdgeKind::Back);
        cfg.add_edge(s, q, EdgeKind::Back);
        cfg.add_edge(r, s, EdgeKind::Fallthrough);
        cfg
    }

    #[test]
    fn a_retreating_edge_into_the_interior_of_a_cycle_still_gets_split_at_an_entry() {
        // The first retreating edge ends at `w` or at `s`, and splitting one
        // block of a cycle this dense hands the entry on to its successors:
        // `w` and `s` took turns being the witness forever. Only copying
        // everything the second entry leads to ends it.
        for q_first in [true, false] {
            let mut cfg = cycle_with_an_interior_witness(q_first);
            let before = DominatorTree::compute(&cfg);
            assert!(!is_reducible(&cfg, &before), "q first: {q_first}");
            let original_blocks = cfg.block_count();

            let splits = make_reducible(&mut cfg).len();

            assert!(splits >= 1, "q first: {q_first}");
            assert_eq!(cfg.block_count(), original_blocks + splits);
            let after = DominatorTree::compute(&cfg);
            assert!(is_reducible(&cfg, &after), "q first: {q_first}");
        }
    }

    #[test]
    fn every_irreducible_entry_is_split_in_turn() {
        // Two independent two-entry cycles: whichever the search meets first,
        // fixing it leaves the other, so the next round has to find it again.
        let mut cfg: Cfg<()> = Cfg::new();
        let entry = cfg.entry();
        for _ in 0..2 {
            let [a, b] = [(); 2].map(|()| cfg.new_block());
            cfg.add_edge(entry, a, EdgeKind::ConditionalTrue);
            cfg.add_edge(entry, b, EdgeKind::ConditionalFalse);
            cfg.add_edge(a, b, EdgeKind::Fallthrough);
            cfg.add_edge(b, a, EdgeKind::Fallthrough);
        }
        let before = DominatorTree::compute(&cfg);
        assert!(!is_reducible(&cfg, &before));
        let original_blocks = cfg.block_count();

        let splits = make_reducible(&mut cfg).len();

        assert!(splits >= 2, "each cycle needs a split, got {splits}");
        assert_eq!(cfg.block_count(), original_blocks + splits);
        let after = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &after));
    }

    #[test]
    fn a_cycle_entered_in_its_middle_is_made_reducible() {
        let mut cfg: Cfg<()> = Cfg::new();
        let cycle: Vec<_> = (0..4).map(|_| cfg.new_block()).collect();
        let external = cfg.new_block();

        cfg.add_edge(cfg.entry(), cycle[0], EdgeKind::ConditionalTrue);
        cfg.add_edge(cycle[0], cycle[1], EdgeKind::Fallthrough);
        cfg.add_edge(cycle[1], cycle[2], EdgeKind::Fallthrough);
        cfg.add_edge(cycle[2], cycle[3], EdgeKind::Fallthrough);
        cfg.add_edge(cycle[3], cycle[0], EdgeKind::Back);
        cfg.add_edge(cfg.entry(), external, EdgeKind::ConditionalFalse);
        cfg.add_edge(external, cycle[1], EdgeKind::Unconditional);

        let before = DominatorTree::compute(&cfg);
        assert!(!is_reducible(&cfg, &before));
        let original_blocks = cfg.block_count();

        let splits = make_reducible(&mut cfg).len();

        assert!(splits >= 1);
        assert_eq!(cfg.block_count(), original_blocks + splits);
        let after = DominatorTree::compute(&cfg);
        assert!(is_reducible(&cfg, &after));
    }

    #[test]
    fn splitting_preserves_outgoing_edge_metadata_and_order() {
        // entry -> {a, b} and a <-> b, every edge weighted and tagged, and each
        // cycle block also leaving to `exit`, so whichever block is copied has
        // two outgoing edges to clone and an entering edge to redirect.
        let mut cfg = Cfg::<MockInst, Tag>::with_edge_payload();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let exit = cfg.new_block();
        for (block, name) in [(cfg.entry(), "entry"), (a, "a"), (b, "b"), (exit, "exit")] {
            cfg.block_mut(block).instructions_mut().push(ff(name));
        }

        let entry = cfg.entry();
        let into_a =
            cfg.add_weighted_edge_with_payload(entry, a, EdgeKind::ConditionalTrue, 0.375, Tag(1));
        let into_b =
            cfg.add_weighted_edge_with_payload(entry, b, EdgeKind::ConditionalFalse, 0.125, Tag(2));
        cfg.add_weighted_edge_with_payload(a, b, EdgeKind::Fallthrough, 0.5, Tag(3));
        cfg.add_weighted_edge_with_payload(a, exit, EdgeKind::SwitchCase, 0.625, Tag(4));
        cfg.add_weighted_edge_with_payload(b, a, EdgeKind::Back, 0.75, Tag(5));
        cfg.add_weighted_edge_with_payload(b, exit, EdgeKind::SwitchCase, 0.25, Tag(6));
        let original_block_bound = cfg.block_bound();
        let signature = |cfg: &Cfg<MockInst, Tag>, edge| {
            let edge = cfg.edge(edge);
            (
                edge.target(),
                edge.kind(),
                edge.weight(),
                edge.payload().clone(),
            )
        };
        let before = [a, b].map(|block| {
            let edges: Vec<_> = cfg.outgoing(block).collect();
            let signatures: Vec<_> = edges.iter().map(|&edge| signature(&cfg, edge)).collect();
            (edges, signatures)
        });
        let entering_before = [into_a, into_b].map(|edge| cfg.edge(edge).weight());

        assert_eq!(make_reducible(&mut cfg).len(), 1);

        let copy = BlockId::from_index(original_block_bound);
        let which = [a, b]
            .iter()
            .position(|&block| {
                cfg.block(block).instructions()[0].1 == cfg.block(copy).instructions()[0].1
            })
            .expect("the copy holds the instructions of one of the cycle's blocks");
        let (outgoing, expected) = &before[which];
        let entering = [into_a, into_b][which];
        assert_eq!(cfg.edge(entering).target(), copy);
        assert_eq!(cfg.edge(entering).weight(), entering_before[which]);
        assert_eq!(
            *cfg.edge(entering).payload(),
            Tag(u8::try_from(which).expect("two candidates") + 1)
        );
        assert_eq!(
            cfg.outgoing([a, b][which]).collect::<Vec<_>>(),
            *outgoing,
            "the original keeps its edges and their order"
        );

        let copied: Vec<_> = cfg.outgoing(copy).collect();
        assert_eq!(copied.len(), outgoing.len());
        for ((&copied, &original), expected) in copied.iter().zip(outgoing).zip(expected) {
            assert_ne!(copied, original);
            assert_eq!(cfg.edge(copied).source(), copy);
            assert_eq!(&signature(&cfg, copied), expected);
        }
    }

    /// An edge payload that identifies the edge it was put on.
    #[derive(Debug, Clone, PartialEq, Eq, Default)]
    struct Tag(u8);

    /// Every digraph of up to four nodes, then seeded random digraphs of five
    /// to nine, as a node count and an edge list over dense indices.
    fn digraphs() -> impl Iterator<Item = (usize, Vec<(usize, usize)>)> {
        use crate::test_util::rng::Rng;

        let exhaustive = (1..=4_usize).flat_map(|count| {
            let pairs: Vec<(usize, usize)> = (0..count)
                .flat_map(|from| (0..count).map(move |to| (from, to)))
                .collect();
            (0..1_u32 << pairs.len()).map(move |mask| {
                let edges = pairs
                    .iter()
                    .enumerate()
                    .filter(|&(bit, _)| mask & (1 << bit) != 0)
                    .map(|(_, &pair)| pair)
                    .collect();
                (count, edges)
            })
        });
        let random = (0..3_000_u64).map(|seed| {
            let mut rng = Rng::new(seed);
            let count = 5 + rng.below(5);
            let percent = 150 / count + rng.below(300 / count);
            let edges = (0..count)
                .flat_map(|from| (0..count).map(move |to| (from, to)))
                .filter(|_| rng.chance(percent))
                .collect();
            (count, edges)
        });
        exhaustive.chain(random)
    }

    /// Make one digraph reducible and check the result without reading the
    /// production cycle logic. Returns whether the digraph was irreducible.
    fn check_digraph(count: usize, edges: &[(usize, usize)]) -> bool {
        // Block `i` holds the instruction `i`, which a copy repeats, so a
        // block's instruction names the original it stands for.
        let mut cfg = Cfg::<u32>::new();
        let mut blocks = alloc::vec![cfg.entry()];
        blocks.extend((1..count).map(|_| cfg.new_block()));
        for (index, &block) in blocks.iter().enumerate() {
            cfg.block_mut(block)
                .instructions_mut()
                .push(u32::try_from(index).expect("a test index fits in u32"));
        }
        for &(from, to) in edges {
            cfg.add_edge(blocks[from], blocks[to], EdgeKind::Fallthrough);
        }
        let was_reducible = is_reducible(&cfg, &DominatorTree::compute(&cfg));
        let original_blocks = cfg.block_count();
        let original_bound = cfg.block_bound();
        let label = alloc::format!("{count} nodes {edges:?}");

        let copies = make_reducible(&mut cfg);

        assert!(is_reducible(&cfg, &DominatorTree::compute(&cfg)), "{label}");
        assert_eq!(
            copies,
            (original_bound..cfg.block_bound())
                .map(BlockId::from_index)
                .collect::<Vec<_>>(),
            "{label}: the copies are the new blocks"
        );
        assert_eq!(cfg.block_count(), original_blocks + copies.len(), "{label}");
        assert_eq!(copies.is_empty(), was_reducible, "{label}");

        // The new graph covers the old one: mapping each block to the
        // original it stands for sends the outgoing edges of every block,
        // as a multiset, onto those of its original, so every path of the
        // original is a path of the result and the other way round.
        let origin = |block: BlockId| cfg.block(block).instructions()[0] as usize;
        for block in cfg.block_ids() {
            let mut now: Vec<usize> = cfg.successors(block).map(origin).collect();
            let mut then: Vec<usize> = edges
                .iter()
                .filter(|&&(from, _)| from == origin(block))
                .map(|&(_, to)| to)
                .collect();
            now.sort_unstable();
            then.sort_unstable();
            assert_eq!(
                now,
                then,
                "successors of a copy of {}: {label}",
                origin(block)
            );
        }
        assert_eq!(origin(cfg.entry()), 0, "{label}");
        !was_reducible
    }

    #[test]
    fn the_result_is_reducible_and_covers_the_original_on_every_small_digraph() {
        use std::sync::Arc;
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::time::{Duration, Instant};

        // Not returning is the failure this guards against, so the sweep runs
        // on a worker that the test stops waiting for, naming the digraph it
        // was stuck on, instead of hanging the whole run.
        let position = Arc::new(AtomicUsize::new(0));
        let progress = Arc::clone(&position);
        let worker = std::thread::spawn(move || {
            let mut irreducible = 0_usize;
            for (index, (count, edges)) in digraphs().enumerate() {
                progress.store(index, Ordering::Relaxed);
                irreducible += usize::from(check_digraph(count, &edges));
            }
            irreducible
        });
        let deadline = Instant::now() + Duration::from_secs(120);
        while !worker.is_finished() {
            let stuck = position.load(Ordering::Relaxed);
            assert!(
                Instant::now() < deadline,
                "make_reducible did not return on digraph {stuck}: {:?}",
                digraphs().nth(stuck)
            );
            std::thread::sleep(Duration::from_millis(10));
        }
        let irreducible = worker
            .join()
            .unwrap_or_else(|failure| std::panic::resume_unwind(failure));
        assert!(
            irreducible > 1_000,
            "only {irreducible} irreducible digraphs"
        );
    }
}
