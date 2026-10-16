extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use super::{follow, follow_path};

#[test]
fn follow_stops_at_the_end_the_bound_and_the_cycle() {
    let chain = |node: &u32| (*node < 3).then(|| node + 1);
    assert_eq!(follow(0, 16, chain), 3, "runs out of steps to take");
    assert_eq!(follow(0, 2, chain), 2, "the hop bound stops it");
    assert_eq!(
        follow(9, 16, chain),
        9,
        "a seed with no step is its own answer"
    );

    // The guard is the whole path, not the seed: `0 -> 1 -> 2 -> 3 -> 1`
    // closes on a middle node and never returns to where it started.
    let lasso = |node: &u32| Some(if *node == 3 { 1 } else { node + 1 });
    assert_eq!(follow(0, 16, lasso), 3);
    assert_eq!(follow_path(0, 16, lasso), vec![0, 1, 2, 3]);
}

#[test]
fn follow_path_returns_the_chain_including_the_seed() {
    let chain = |node: &u32| (*node < 3).then(|| node + 1);
    assert_eq!(follow_path(0, 16, chain), vec![0, 1, 2, 3]);
    assert_eq!(follow_path(0, 2, chain), vec![0, 1, 2]);
    assert_eq!(follow_path(0, 0, chain), vec![0]);
    assert_eq!(follow_path(9, 16, chain), vec![9]);
    // `follow` is the last node of exactly this chain.
    for hops in 0..5 {
        assert_eq!(
            Some(follow(0, hops, chain)),
            follow_path(0, hops, chain).pop()
        );
    }
}

#[test]
fn a_chase_longer_than_its_inline_chain_keeps_every_node_and_its_guard() {
    // Twenty nodes: past the eight a chase holds inline.
    let long = |node: &u32| (*node < 19).then(|| node + 1);
    assert_eq!(follow(0, 64, long), 19);
    assert_eq!(follow_path(0, 64, long), (0..20).collect::<Vec<_>>());
    assert_eq!(
        follow(0, 12, long),
        12,
        "the hop bound cuts a spilled chase"
    );

    // The cycle closes on a node that the spill moved to the heap.
    let lasso = |node: &u32| Some(if *node == 19 { 2 } else { node + 1 });
    assert_eq!(follow(0, 64, lasso), 19);
    assert_eq!(follow_path(0, 64, lasso), (0..20).collect::<Vec<_>>());
}
