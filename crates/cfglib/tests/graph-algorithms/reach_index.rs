//! `ReachIndex` answers every pair the way a breadth-first walk does.

use cfglib::{NodeId, ReachIndex, ReachIndexConfig, ReachScratch};

use crate::Lcg;
use crate::digraph::Digraph;

#[test]
fn every_pair_matches_the_breadth_first_distances_in_both_modes() {
    let configs = [
        ReachIndexConfig::default(),
        ReachIndexConfig::new().with_orders(0).with_supportive(0),
        ReachIndexConfig::new().with_orders(5).with_supportive(64),
        ReachIndexConfig::new().with_exact_max_chains(usize::MAX),
        ReachIndexConfig::new().with_exact_max_chains(3),
    ];
    let mut scratch = ReachScratch::default();
    for seed in 1..=24 {
        let nodes = 8 + usize::try_from(seed).expect("a small seed fits in usize") * 2;
        let digraph = Digraph::random(&mut Lcg(seed + 130), nodes, nodes + nodes / 2);
        let graph = digraph.graph();
        let distances: Vec<_> = (0..nodes)
            .map(|source| digraph.distances(source, true, None))
            .collect();
        for config in configs {
            let index = ReachIndex::compute(&graph, config);
            for (source, row) in distances.iter().enumerate() {
                for (target, distance) in row.iter().enumerate() {
                    assert_eq!(
                        index.reaches_in(
                            &mut scratch,
                            NodeId::from_index(source),
                            NodeId::from_index(target),
                        ),
                        distance.is_some(),
                        "{source} -> {target} under {config:?} in {:?}",
                        digraph.edges
                    );
                }
            }
        }
    }
}
