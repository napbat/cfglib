extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::vec::Vec;

use super::{ExtTspConfig, ext_tsp_order, ext_tsp_score};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::traverse::reverse_postorder;
use crate::graph::view::Direction;
use crate::test_util::rng::Rng;
use crate::test_util::{MockInst, ff};
use crate::transform::linearize::{Emitter, falls_through};

/// Sizes by block index and counts by edge index.
struct Profile {
    sizes: Vec<u64>,
    counts: Vec<u64>,
}

impl Profile {
    fn uniform(cfg: &Cfg<MockInst>, size: u64) -> Self {
        Self {
            sizes: alloc::vec![size; cfg.block_bound()],
            counts: alloc::vec![0; cfg.edge_bound()],
        }
    }

    fn count(mut self, edge: EdgeId, count: u64) -> Self {
        self.counts[edge.index()] = count;
        self
    }

    fn order(&self, cfg: &Cfg<MockInst>, config: &ExtTspConfig) -> Vec<BlockId> {
        ext_tsp_order(
            cfg,
            |block| self.sizes[block.index()],
            |edge| self.counts[edge.index()],
            |_| true,
            config,
        )
    }

    fn score(&self, cfg: &Cfg<MockInst>, order: &[BlockId], config: &ExtTspConfig) -> u128 {
        ext_tsp_score(
            cfg,
            order,
            |block| self.sizes[block.index()],
            |edge| self.counts[edge.index()],
            |edge| falls(cfg, edge),
            config,
        )
    }
}

/// The fallthrough rule of the random tests: every edge but a dispatch.
fn falls(cfg: &Cfg<MockInst>, edge: EdgeId) -> bool {
    cfg.edge(edge).kind() != EdgeKind::SwitchCase
}

/// Names the transfers; can invert any branch when `inverts`.
struct Asm {
    inverts: bool,
}

impl Emitter<MockInst> for Asm {
    fn emit_jump(&self, _target: BlockId) -> MockInst {
        ff("jump")
    }
    fn emit_conditional_branch(&self, _condition: &MockInst, _target: BlockId) -> MockInst {
        ff("branch")
    }
    fn emit_block_start(&self, _block: BlockId) -> Option<MockInst> {
        None
    }
    fn emit_inverted_conditional_branch(
        &self,
        _condition: &MockInst,
        _target: BlockId,
    ) -> Option<MockInst> {
        self.inverts.then(|| ff("branch-not"))
    }
}

/// The entry branches to a true arm that runs 90 times and a false arm that
/// runs 10 times; both fall through to a join.
fn diamond() -> (Cfg<MockInst>, Profile, [BlockId; 3]) {
    let mut cfg = Cfg::new();
    let [hot, cold, join] = [(); 3].map(|()| cfg.new_block());
    cfg.block_mut(cfg.entry()).push(ff("test"));
    let edges = [
        (cfg.entry(), hot, EdgeKind::ConditionalTrue, 90),
        (cfg.entry(), cold, EdgeKind::ConditionalFalse, 10),
        (hot, join, EdgeKind::Fallthrough, 90),
        (cold, join, EdgeKind::Fallthrough, 10),
    ];
    let mut profile = Profile::uniform(&cfg, 16);
    profile.counts.resize(edges.len(), 0);
    for (source, target, kind, count) in edges {
        let edge = cfg.add_edge(source, target, kind);
        profile = profile.count(edge, count);
    }
    (cfg, profile, [hot, cold, join])
}

#[test]
fn the_hot_arm_of_a_diamond_follows_the_branch() {
    let (cfg, profile, [hot, cold, join]) = diamond();
    let config = ExtTspConfig::default();
    let entry = cfg.entry();
    assert_eq!(profile.order(&cfg, &config), [entry, hot, join, cold]);

    let size = |block: BlockId| profile.sizes[block.index()];
    let count = |edge: EdgeId| profile.counts[edge.index()];
    let cfg = &cfg;
    let falls = |inverts| move |edge| falls_through(cfg, edge, &Asm { inverts });
    let inverting = ext_tsp_order(cfg, size, count, falls(true), &config);
    assert_eq!(inverting, [entry, hot, join, cold]);

    // Without inversion the true edge is a taken branch wherever its target
    // lands, so the false arm takes the fallthrough.
    let plain = ext_tsp_order(cfg, size, count, falls(false), &config);
    assert_eq!(plain, [entry, cold, hot, join]);
}

#[test]
fn a_hot_loop_body_is_contiguous() {
    let mut cfg = Cfg::new();
    let [exit, latch, body, header] = [(); 4].map(|()| cfg.new_block());
    cfg.block_mut(header).push(ff("test"));
    let edges = [
        (cfg.entry(), header, EdgeKind::Fallthrough, 1),
        (header, exit, EdgeKind::ConditionalFalse, 1),
        (header, body, EdgeKind::ConditionalTrue, 1_000),
        (body, latch, EdgeKind::Fallthrough, 1_000),
        (latch, header, EdgeKind::Back, 1_000),
    ];
    let mut profile = Profile::uniform(&cfg, 8);
    profile.counts.resize(edges.len(), 0);
    for (source, target, kind, count) in edges {
        let edge = cfg.add_edge(source, target, kind);
        profile = profile.count(edge, count);
    }

    let order = profile.order(&cfg, &ExtTspConfig::default());
    let position = |block| order.iter().position(|&placed| placed == block).unwrap();
    let mut loop_positions = [position(header), position(body), position(latch)];
    loop_positions.sort_unstable();
    assert_eq!(
        loop_positions[2] - loop_positions[0],
        2,
        "header, body, and latch are contiguous: {order:?}"
    );
    assert_eq!(order[0], cfg.entry());
}

#[test]
fn a_graph_without_counts_keeps_its_reverse_postorder() {
    let (cfg, mut profile, _) = diamond();
    profile.counts.fill(0);
    assert_eq!(
        profile.order(&cfg, &ExtTspConfig::default()),
        reverse_postorder(&cfg, cfg.entry(), Direction::Forward)
    );
}

#[test]
fn unreachable_and_removed_blocks_are_omitted() {
    let (mut cfg, profile, [hot, cold, join]) = diamond();
    let orphan = cfg.new_block();
    cfg.add_edge(orphan, join, EdgeKind::Fallthrough);
    let mut profile = profile;
    profile.sizes.push(16);
    profile.counts.push(1_000);
    cfg.remove_block(cold);

    let order = profile.order(&cfg, &ExtTspConfig::default());
    assert_eq!(order, [cfg.entry(), hot, join]);
}

/// A random graph of up to 40 blocks with random edge kinds, some removed
/// blocks, self-loops and parallel edges, and a random profile.
fn random_graph(rng: &mut Rng) -> (Cfg<MockInst>, Profile) {
    const KINDS: [EdgeKind; 8] = [
        EdgeKind::Fallthrough,
        EdgeKind::ConditionalTrue,
        EdgeKind::ConditionalFalse,
        EdgeKind::Unconditional,
        EdgeKind::Back,
        EdgeKind::Jump,
        EdgeKind::SwitchCase,
        EdgeKind::CallReturn,
    ];
    let mut cfg = Cfg::new();
    let count = 1 + rng.below(40);
    for _ in 1..count {
        cfg.new_block();
    }
    let blocks: Vec<BlockId> = cfg.block_ids().collect();
    for &block in &blocks {
        cfg.block_mut(block).push(ff("inst"));
        for _ in 0..rng.below(4) {
            let target = blocks[rng.below(blocks.len())];
            cfg.add_edge(block, target, KINDS[rng.below(KINDS.len())]);
        }
    }
    if count > 1 && rng.one_in(3) {
        cfg.remove_block(blocks[1 + rng.below(count - 1)]);
    }
    let sizes = (0..cfg.block_bound())
        .map(|_| u64::try_from(rng.below(96)).unwrap())
        .collect();
    let counts = (0..cfg.edge_bound())
        .map(|_| {
            if rng.one_in(4) {
                0
            } else {
                u64::try_from(1 + rng.below(1_000)).unwrap()
            }
        })
        .collect();
    (cfg, Profile { sizes, counts })
}

fn random_config(rng: &mut Rng) -> ExtTspConfig {
    if rng.one_in(2) {
        return ExtTspConfig::default();
    }
    ExtTspConfig {
        forward_distance: u64::try_from(rng.below(300)).unwrap(),
        backward_distance: u64::try_from(rng.below(300)).unwrap(),
        forward_weight: u32::try_from(rng.below(1_200)).unwrap(),
        backward_weight: u32::try_from(rng.below(1_200)).unwrap(),
        chain_split_threshold: rng.below(6),
    }
}

#[test]
fn the_order_is_a_deterministic_permutation_that_scores_at_least_the_reverse_postorder() {
    let mut rng = Rng::new(50);
    for round in 0..300 {
        let (cfg, profile) = random_graph(&mut rng);
        let config = random_config(&mut rng);
        let order = |cfg: &Cfg<MockInst>| {
            ext_tsp_order(
                cfg,
                |block| profile.sizes[block.index()],
                |edge| profile.counts[edge.index()],
                |edge| falls(cfg, edge),
                &config,
            )
        };
        let result = order(&cfg);
        let reverse_postorder = reverse_postorder(&cfg, cfg.entry(), Direction::Forward);
        let context = format!("round {round}: {config:?}, order {result:?}");

        assert_eq!(result.first(), Some(&cfg.entry()), "{context}");
        assert_eq!(
            result.iter().copied().collect::<BTreeSet<_>>(),
            reverse_postorder.iter().copied().collect::<BTreeSet<_>>(),
            "{context}: the reachable blocks"
        );
        assert_eq!(result.len(), reverse_postorder.len(), "{context}");
        assert!(
            profile.score(&cfg, &result, &config)
                >= profile.score(&cfg, &reverse_postorder, &config),
            "{context}"
        );
        assert_eq!(order(&cfg), result, "{context}: deterministic");
    }
}

#[test]
fn the_cfg_order_follows_the_emitter_and_is_a_permutation() {
    let mut rng = Rng::new(51);
    for round in 0..100 {
        let (cfg, profile) = random_graph(&mut rng);
        let inverts = rng.one_in(2);
        let result = ext_tsp_order(
            &cfg,
            |block| profile.sizes[block.index()],
            |edge| profile.counts[edge.index()],
            |edge| falls_through(&cfg, edge, &Asm { inverts }),
            &ExtTspConfig::default(),
        );
        let mut sorted = result.clone();
        sorted.sort_unstable();
        let mut reachable = reverse_postorder(&cfg, cfg.entry(), Direction::Forward);
        reachable.sort_unstable();
        assert_eq!(sorted, reachable, "round {round}");
        assert_eq!(result[0], cfg.entry(), "round {round}");
    }
}

/// The score by its definition, with plain arithmetic for small counts.
fn reference_score(
    cfg: &Cfg<MockInst>,
    order: &[BlockId],
    profile: &Profile,
    config: &ExtTspConfig,
) -> u128 {
    let mut address = alloc::vec![None; cfg.block_bound()];
    let mut next = 0;
    for (position, &block) in order.iter().enumerate() {
        address[block.index()] = Some((position, next));
        next += profile.sizes[block.index()];
    }
    let mut score = 0;
    for edge in cfg.edge_ids() {
        let (source, target) = (cfg.edge(edge).source(), cfg.edge(edge).target());
        let (Some((source_position, source_start)), Some((target_position, target_start))) =
            (address[source.index()], address[target.index()])
        else {
            continue;
        };
        let count = u128::from(profile.counts[edge.index()]);
        let end = source_start + profile.sizes[source.index()];
        score += if target_position == source_position + 1 && falls(cfg, edge) {
            count * 1_000
        } else if target_start >= end && target_start - end <= config.forward_distance {
            let reach = u128::from(config.forward_distance - (target_start - end));
            count * u128::from(config.forward_weight) * reach
                / u128::from(config.forward_distance).max(1)
        } else if target_start < end && end - target_start <= config.backward_distance {
            let reach = u128::from(config.backward_distance - (end - target_start));
            count * u128::from(config.backward_weight) * reach
                / u128::from(config.backward_distance).max(1)
        } else {
            0
        };
    }
    score
}

#[test]
fn the_score_follows_its_definition() {
    let mut rng = Rng::new(52);
    for round in 0..300 {
        let (cfg, profile) = random_graph(&mut rng);
        let mut config = random_config(&mut rng);
        config.forward_distance = config.forward_distance.max(1);
        config.backward_distance = config.backward_distance.max(1);
        let mut order: Vec<BlockId> = cfg.block_ids().filter(|_| !rng.one_in(5)).collect();
        rng.shuffle(&mut order);
        assert_eq!(
            profile.score(&cfg, &order, &config),
            reference_score(&cfg, &order, &profile, &config),
            "round {round}: {config:?}, {order:?}"
        );
    }
}

#[test]
fn a_zero_distance_credits_only_an_adjacent_jump_and_large_counts_do_not_overflow() {
    let mut cfg = Cfg::<MockInst>::new();
    let [next, far] = [(); 2].map(|()| cfg.new_block());
    let near_jump = cfg.add_edge(cfg.entry(), next, EdgeKind::Jump);
    let far_jump = cfg.add_edge(cfg.entry(), far, EdgeKind::Jump);
    let profile = Profile::uniform(&cfg, 4)
        .count(near_jump, u64::MAX)
        .count(far_jump, u64::MAX);
    let order = [cfg.entry(), next, far];
    let score = |config: &ExtTspConfig| {
        ext_tsp_score(
            &cfg,
            &order,
            |block| profile.sizes[block.index()],
            |edge| profile.counts[edge.index()],
            |_| false,
            config,
        )
    };
    let count = u128::from(u64::MAX);

    let zero = ExtTspConfig {
        forward_distance: 0,
        forward_weight: 1_000,
        ..ExtTspConfig::default()
    };
    assert_eq!(score(&zero), count * 1_000, "only the adjacent jump scores");

    // The far jump covers half of an 8-byte reach.
    let short = ExtTspConfig {
        forward_distance: 8,
        forward_weight: u32::MAX,
        ..ExtTspConfig::default()
    };
    let full = count * u128::from(u32::MAX);
    assert_eq!(score(&short), full + full / 2);
}
