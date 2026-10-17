use cfglib::{
    BlockId, Cfg, EdgeId, EdgeKind, contract_edge, make_reducible, merge_blocks,
    remove_empty_blocks,
};

use super::NODE_COUNT;
use crate::fixtures::{
    empty_chain_cfg, fixture_u32, high_fan_in_cfg, irreducible_cfg, linear_cfg,
    weighted_high_fan_out_cfg, weighted_irreducible_cfg,
};
use crate::harness::{BenchmarkSuite, benchmark_case};
use crate::structural_oracles::{
    FanOut, assert_cfg_shape, assert_empty_chain, assert_high_fan_in, assert_irreducible_fixture,
    assert_linear_cfg, assert_split_weighted_fan_out, assert_weighted_fan_out,
    assert_weighted_irreducible,
};

fn register_chains(suite: &mut BenchmarkSuite<'_>) {
    let linear = linear_cfg(NODE_COUNT / 2);
    let empty_chain = empty_chain_cfg(NODE_COUNT / 2);
    benchmark_case!(
        suite,
        "cfg_clone_linear",
        || linear.clone(),
        |result: &Cfg<u32>| {
            assert_linear_cfg(result, NODE_COUNT / 2);
        }
    );
    benchmark_case!(
        suite,
        "cfg_clone_merge_linear",
        || {
            let mut cloned = linear.clone();
            let merged = merge_blocks(&mut cloned);
            (cloned, merged)
        },
        |(result, merged): &(Cfg<u32>, usize)| {
            assert_eq!(*merged, NODE_COUNT / 2 - 1);
            // Merging a linear chain leaves ONE live block holding every
            // instruction; the consumed blocks are gone, and only their
            // identity slots survive until a compact.
            assert_cfg_shape(result, 1, 0);
            assert_eq!(result.block_bound(), NODE_COUNT / 2);
            assert_eq!(
                result.block_ids().collect::<Vec<_>>(),
                [result.entry()],
                "only the merge target stays live"
            );
            assert_eq!(
                result.block(result.entry()).instructions(),
                &(0..fixture_u32(NODE_COUNT) / 2).collect::<Vec<_>>()
            );
        }
    );
    benchmark_case!(
        suite,
        "cfg_clone_empty_chain",
        || empty_chain.clone(),
        |result: &Cfg<u32>| assert_empty_chain(result, NODE_COUNT / 2)
    );
    benchmark_case!(
        suite,
        "cfg_clone_remove_empty_chain",
        || {
            let mut cloned = empty_chain.clone();
            let removed = remove_empty_blocks(&mut cloned);
            (cloned, removed)
        },
        |(result, removed): &(Cfg<u32>, usize)| {
            assert_eq!(*removed, NODE_COUNT / 2 - 2);
            // Bypassing the empty middle removes it: only the instruction
            // carriers stay live, joined by the surviving first edge.
            assert_cfg_shape(result, 2, 1);
            assert_eq!(result.block_bound(), NODE_COUNT / 2);
            assert_eq!(result.block(result.entry()).instructions(), &[0]);
            let last = BlockId::from_index(NODE_COUNT / 2 - 1);
            assert_eq!(
                result.block_ids().collect::<Vec<_>>(),
                [result.entry(), last]
            );
            assert_eq!(
                result.block(last).instructions(),
                &[fixture_u32(NODE_COUNT / 2 - 1)]
            );
            let edge = result.edge(EdgeId::from_raw(0));
            assert_eq!(edge.source(), result.entry());
            assert_eq!(edge.target(), last);
            assert_eq!(edge.kind(), EdgeKind::Fallthrough);
            assert!(edge.weight().is_none());
        }
    );
}

fn register_fan_in(suite: &mut BenchmarkSuite<'_>) {
    let (high_fan_in, old_target, new_target) = high_fan_in_cfg(NODE_COUNT);
    benchmark_case!(
        suite,
        "cfg_clone_high_fan_in",
        || high_fan_in.clone(),
        |result: &Cfg<u32>| assert_high_fan_in(result, NODE_COUNT, old_target, new_target, false,)
    );
    benchmark_case!(
        suite,
        "cfg_clone_redirect_high_fan_in",
        || {
            let mut cloned = high_fan_in.clone();
            cloned.redirect_edges_to(old_target, new_target);
            cloned
        },
        |result: &Cfg<u32>| assert_high_fan_in(result, NODE_COUNT, old_target, new_target, true,)
    );
}

fn register_fan_out(suite: &mut BenchmarkSuite<'_>) {
    let (weighted_high_fan_out, fan_out_source, fan_out_target) =
        weighted_high_fan_out_cfg(NODE_COUNT);
    benchmark_case!(
        suite,
        "cfg_clone_weighted_high_fan_out",
        || { weighted_high_fan_out.clone() },
        |result: &Cfg<u32>| assert_weighted_fan_out(
            result,
            NODE_COUNT,
            fan_out_source,
            fan_out_target,
            FanOut::Intact,
        )
    );
    benchmark_case!(
        suite,
        "cfg_clone_split_weighted_high_fan_out",
        || {
            let mut cloned = weighted_high_fan_out.clone();
            let split = cloned.split_block(fan_out_target, 1);
            (cloned, split)
        },
        |(result, split): &(Cfg<u32>, BlockId)| assert_split_weighted_fan_out(
            result,
            NODE_COUNT,
            fan_out_source,
            fan_out_target,
            *split,
        )
    );
    benchmark_case!(
        suite,
        "cfg_clone_merge_weighted_high_fan_out",
        || {
            let mut cloned = weighted_high_fan_out.clone();
            let merged = merge_blocks(&mut cloned);
            (cloned, merged)
        },
        |(result, merged): &(Cfg<u32>, usize)| {
            assert_eq!(*merged, 1);
            assert_weighted_fan_out(
                result,
                NODE_COUNT,
                fan_out_source,
                fan_out_target,
                FanOut::Merged,
            );
        }
    );
    benchmark_case!(
        suite,
        "cfg_clone_contract_weighted_high_fan_out",
        || {
            let mut cloned = weighted_high_fan_out.clone();
            let contracted = contract_edge(&mut cloned, fan_out_source, fan_out_target);
            (cloned, contracted)
        },
        |(result, contracted): &(Cfg<u32>, bool)| {
            assert!(*contracted);
            assert_weighted_fan_out(
                result,
                NODE_COUNT,
                fan_out_source,
                fan_out_target,
                FanOut::Contracted,
            );
        }
    );
}

fn register_irreducible(suite: &mut BenchmarkSuite<'_>) {
    let irreducible_small = irreducible_cfg(2, 512);
    let irreducible_large = irreducible_cfg(512, 512);
    let weighted_irreducible = weighted_irreducible_cfg();
    benchmark_case!(
        suite,
        "cfg_clone_irreducible_small",
        || irreducible_small.clone(),
        |result: &Cfg<u32>| assert_irreducible_fixture(result, 2, 512, 0)
    );
    benchmark_case!(
        suite,
        "cfg_clone_make_reducible_small",
        || {
            let mut cloned = irreducible_small.clone();
            let splits = make_reducible(&mut cloned);
            (cloned, splits)
        },
        |(result, splits): &(Cfg<u32>, usize)| {
            assert_eq!(*splits, 1);
            assert_irreducible_fixture(result, 2, 512, 1);
        }
    );
    benchmark_case!(
        suite,
        "cfg_clone_irreducible_large",
        || irreducible_large.clone(),
        |result: &Cfg<u32>| assert_irreducible_fixture(result, 512, 512, 0)
    );
    benchmark_case!(
        suite,
        "cfg_clone_make_reducible_large",
        || {
            let mut cloned = irreducible_large.clone();
            let splits = make_reducible(&mut cloned);
            (cloned, splits)
        },
        |(result, splits): &(Cfg<u32>, usize)| {
            assert_eq!(*splits, 511);
            assert_irreducible_fixture(result, 512, 512, 511);
        }
    );
    benchmark_case!(
        suite,
        "cfg_clone_weighted_irreducible",
        || weighted_irreducible.clone(),
        |result: &Cfg<u32>| assert_weighted_irreducible(result, false)
    );
    benchmark_case!(
        suite,
        "cfg_clone_make_reducible_weighted",
        || {
            let mut cloned = weighted_irreducible.clone();
            let splits = make_reducible(&mut cloned);
            (cloned, splits)
        },
        |(result, splits): &(Cfg<u32>, usize)| {
            assert_eq!(*splits, 1);
            assert_weighted_irreducible(result, true);
        }
    );
}

pub(super) fn register(suite: &mut BenchmarkSuite<'_>) {
    register_chains(suite);
    register_fan_in(suite);
    register_fan_out(suite);
    register_irreducible(suite);
}
