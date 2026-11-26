//! Inference on hand-built profiles, and conservation on random graphs with
//! random partial estimates.

use super::*;
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::test_util::rng::Rng;

const EVEN: InferenceCosts = InferenceCosts {
    increase: 1,
    decrease: 1,
};

/// A CFG with its exits, estimates, and priors, by block and edge slot.
struct Profile {
    cfg: Cfg<u32>,
    exit: Vec<bool>,
    block: Vec<Option<u64>>,
    edge: Vec<Option<u64>>,
    prior: Vec<Option<Probability>>,
}

impl Profile {
    fn new(count: usize) -> Self {
        let mut cfg = Cfg::<u32>::new();
        for _ in 1..count {
            cfg.new_block();
        }
        Self {
            cfg,
            exit: vec![false; count],
            block: vec![None; count],
            edge: Vec::new(),
            prior: Vec::new(),
        }
    }

    fn edge(&mut self, from: usize, to: usize, prior: Option<f64>) -> EdgeId {
        self.edge.push(None);
        self.prior
            .push(prior.map(|p| Probability::from_f64(p).expect("a probability")));
        self.cfg.add_edge(
            BlockId::from_index(from),
            BlockId::from_index(to),
            EdgeKind::Jump,
        )
    }

    fn infer(&self, costs: &InferenceCosts) -> BlockFrequencies {
        BlockFrequencies::infer(
            &self.cfg,
            |block| self.exit[block.index()],
            |block| self.block[block.index()],
            |edge| self.edge[edge.index()],
            |edge| self.prior[edge.index()],
            costs,
        )
        .expect("small counts do not overflow")
    }
}

fn blocks(counts: &BlockFrequencies, count: usize) -> Vec<u64> {
    (0..count)
        .map(|block| counts.block(BlockId::from_index(block)))
        .collect()
}

#[test]
fn a_corrupted_block_in_a_loop_is_repaired_and_the_rest_kept() {
    // entry -> header; header -> first or exit; first -> second -> header.
    let mut profile = Profile::new(5);
    profile.edge(0, 1, None);
    profile.edge(1, 2, Some(0.9));
    profile.edge(1, 4, Some(0.1));
    profile.edge(2, 3, None);
    profile.edge(3, 1, None);
    profile.exit[4] = true;
    profile.block = vec![Some(1), Some(10), Some(9), Some(50), Some(1)];

    let counts = profile.infer(&EVEN);
    assert_eq!(blocks(&counts, 5), [1, 10, 9, 9, 1]);
    let edges: Vec<u64> = profile
        .cfg
        .edge_ids()
        .map(|edge| counts.edge(edge))
        .collect();
    assert_eq!(edges, [1, 9, 1, 9, 9]);
}

#[test]
fn a_diamond_counted_at_entry_and_join_splits_by_the_priors() {
    let mut profile = Profile::new(4);
    let left = profile.edge(0, 1, Some(0.25));
    let right = profile.edge(0, 2, Some(0.75));
    profile.edge(1, 3, None);
    profile.edge(2, 3, None);
    profile.exit[3] = true;
    profile.block[0] = Some(1_000);
    profile.block[3] = Some(1_000);

    let counts = profile.infer(&EVEN);
    assert_eq!(blocks(&counts, 4), [1_000, 250, 750, 1_000]);
    assert_eq!((counts.edge(left), counts.edge(right)), (250, 750));

    // Without priors the arms share the count evenly.
    profile.prior = vec![None; 4];
    let counts = profile.infer(&EVEN);
    assert_eq!(blocks(&counts, 4), [1_000, 500, 500, 1_000]);
}

#[test]
fn edge_estimates_are_honored_over_the_priors() {
    let mut profile = Profile::new(4);
    let left = profile.edge(0, 1, Some(0.9));
    profile.edge(0, 2, Some(0.1));
    profile.edge(1, 3, None);
    profile.edge(2, 3, None);
    profile.exit[3] = true;
    profile.block[0] = Some(100);
    profile.edge[left.index()] = Some(40);

    let counts = profile.infer(&EVEN);
    assert_eq!(blocks(&counts, 4), [100, 40, 60, 100]);
}

#[test]
fn without_estimates_nothing_runs() {
    let mut profile = Profile::new(3);
    profile.edge(0, 1, None);
    profile.edge(1, 2, None);
    profile.edge(1, 1, None);
    profile.exit[2] = true;
    let counts = profile.infer(&EVEN);
    assert_eq!(blocks(&counts, 3), [0, 0, 0]);
}

#[test]
fn a_removed_or_unknown_block_counts_zero() {
    let mut profile = Profile::new(3);
    profile.edge(0, 1, None);
    profile.exit[1] = true;
    profile.block[0] = Some(7);
    profile.cfg.remove_block(BlockId::from_index(2));
    let counts = profile.infer(&EVEN);
    assert_eq!(blocks(&counts, 4), [7, 7, 0, 0]);
    assert_eq!(counts.edge(EdgeId::from_index(9)), 0);
}

#[test]
fn an_estimate_beyond_the_network_range_is_an_overflow() {
    let mut profile = Profile::new(2);
    profile.edge(0, 1, None);
    profile.exit[1] = true;
    let result = BlockFrequencies::infer(
        &profile.cfg,
        |block| profile.exit[block.index()],
        |_| Some(u64::MAX),
        |_| None,
        |_| None,
        &EVEN,
    );
    assert_eq!(result, Err(FlowError::Overflow));
}

/// A random CFG of `count` blocks: each has up to two successors, the entry
/// at least one, and a block without successors is an exit, as is one in
/// five of the others.
fn random_profile(rng: &mut Rng, count: usize) -> Profile {
    let mut profile = Profile::new(count);
    for from in 0..count {
        let successors = rng.below(3).max(usize::from(from == 0));
        for _ in 0..successors {
            let to = rng.below(count);
            let prior = rng
                .chance(70)
                .then(|| crate::usize_to_f64(rng.below(11)) / 10.0);
            profile.edge(from, to, prior);
        }
        profile.exit[from] = successors == 0 || rng.chance(20);
    }
    profile
}

/// Counts of random executions that start at the entry and end at an exit.
fn random_walks(rng: &mut Rng, profile: &Profile, walks: usize) -> (Vec<u64>, Vec<u64>) {
    let cfg = &profile.cfg;
    let mut block = vec![0; profile.exit.len()];
    let mut edge = vec![0; profile.edge.len()];
    for _ in 0..walks {
        let (mut seen_blocks, mut seen_edges) = (Vec::new(), Vec::new());
        let mut at = cfg.entry();
        for _ in 0..64 {
            seen_blocks.push(at.index());
            let out: Vec<EdgeId> = cfg.outgoing(at).collect();
            if profile.exit[at.index()] && (out.is_empty() || rng.one_in(2)) {
                for &slot in &seen_blocks {
                    block[slot] += 1;
                }
                for &slot in &seen_edges {
                    edge[slot] += 1;
                }
                break;
            }
            let taken = out[rng.below(out.len())];
            seen_edges.push(taken.index());
            at = cfg.edge(taken).target();
        }
    }
    (block, edge)
}

#[test]
fn consistent_estimates_are_returned_unchanged() {
    let mut rng = Rng::new(42);
    let mut executed = 0;
    for _ in 0..300 {
        let count = 2 + rng.below(8);
        let mut profile = random_profile(&mut rng, count);
        let walks = 1 + rng.below(20);
        let (block, edge) = random_walks(&mut rng, &profile, walks);
        executed += usize::from(block[0] > 0);
        profile.block = block.iter().copied().map(Some).collect();
        profile.edge = edge.iter().copied().map(Some).collect();
        let counts = profile.infer(&EVEN);
        assert_eq!(blocks(&counts, count), block);
        let inferred: Vec<u64> = profile.cfg.edge_ids().map(|id| counts.edge(id)).collect();
        assert_eq!(inferred, edge);
    }
    assert!(executed > 150, "only {executed} profiles ran at all");
}

#[test]
fn random_partial_estimates_give_counts_that_conserve_flow() {
    let mut rng = Rng::new(65);
    let mut running = 0;
    for _ in 0..500 {
        let count = 2 + rng.below(10);
        let mut profile = random_profile(&mut rng, count);
        for estimate in &mut profile.block {
            *estimate = rng
                .chance(60)
                .then(|| u64::try_from(rng.below(100)).expect("small"));
        }
        for estimate in &mut profile.edge {
            *estimate = rng
                .chance(30)
                .then(|| u64::try_from(rng.below(100)).expect("small"));
        }
        let costs = InferenceCosts {
            increase: u32::try_from(rng.below(5)).expect("small"),
            decrease: u32::try_from(rng.below(5)).expect("small"),
        };
        let counts = profile.infer(&costs);
        running += usize::from(profile.cfg.block_ids().any(|block| counts.block(block) > 0));

        let cfg = &profile.cfg;
        for block in cfg.block_ids() {
            let into: u64 = cfg.incoming(block).map(|edge| counts.edge(edge)).sum();
            let out: u64 = cfg.outgoing(block).map(|edge| counts.edge(edge)).sum();
            let runs = counts.block(block);
            if block == cfg.entry() {
                assert!(runs >= into);
            } else {
                assert_eq!(runs, into, "{block:?} runs as often as it is entered");
            }
            if profile.exit[block.index()] {
                assert!(runs >= out);
            } else {
                assert_eq!(runs, out, "{block:?} runs as often as it is left");
            }
        }
    }
    assert!(running > 150, "only {running} inferences ran a block");
}
