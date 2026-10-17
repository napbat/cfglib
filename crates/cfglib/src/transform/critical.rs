//! Critical-edge splitting with payload and identity preservation.

extern crate alloc;
use alloc::vec::Vec;

use crate::cfg::Cfg;
use crate::edge::{Edge, EdgeId, EdgeKind};
use crate::rewrite::Rewrite;

/// Split every critical edge and return the number split.
///
/// Synthetic edges take the payload's [`Default`]; use
/// [`split_critical_edges_with`] to choose them. The original edge identity
/// remains on the first half of each split.
pub fn split_critical_edges<I, E>(cfg: &mut Cfg<I, E>) -> usize
where
    E: Default,
{
    split_critical_edges_mapped(cfg).0
}

/// Split critical edges using default payloads for synthetic fallthroughs.
pub fn split_critical_edges_mapped<I, E>(cfg: &mut Cfg<I, E>) -> (usize, Rewrite)
where
    E: Default,
{
    split_critical_edges_with(cfg, |_, _| E::default())
}

/// Split critical edges with a consumer-defined payload for each synthetic
/// fallthrough.
///
/// `payload_for` sees the original stable edge before mutation. The original
/// edge is redirected to the inserted block without changing its identity,
/// weight, or payload, and keeps its kind with one exception: a
/// [`Back`](EdgeKind::Back) edge is the edge that enters a loop header
/// (`find_back_edges_tagged` reads the tag that way), so the tag moves to the
/// second half, which is the one that now enters the header, and the first
/// half becomes [`Unconditional`](EdgeKind::Unconditional). Its mapping
/// contains both halves in execution order; only the second half receives the
/// generated payload.
pub fn split_critical_edges_with<I, E>(
    cfg: &mut Cfg<I, E>,
    mut payload_for: impl FnMut(EdgeId, &Edge<E>) -> E,
) -> (usize, Rewrite) {
    let mut critical = Vec::new();
    for block_id in cfg.block_ids() {
        let source = block_id;
        if cfg.outgoing(source).count() < 2 {
            continue;
        }
        for edge in cfg.outgoing(source) {
            let target = cfg.edge(edge).target();
            if cfg.incoming(target).count() >= 2 {
                critical.push(edge);
            }
        }
    }

    let mut mapping = Rewrite::new();
    for &edge in &critical {
        let (target, kind, payload) = {
            let original = cfg.edge(edge);
            (
                original.target(),
                original.kind(),
                payload_for(edge, original.data()),
            )
        };
        let middle = cfg.new_block();
        cfg.redirect_edge_target(edge, middle);
        let second_kind = if kind == EdgeKind::Back {
            cfg.edge_mut(edge).set_kind(EdgeKind::Unconditional);
            EdgeKind::Back
        } else {
            EdgeKind::Fallthrough
        };
        let second = cfg.add_edge_with_payload(middle, target, second_kind, payload);

        mapping.record_edge(edge, [edge, second]);
        mapping.record_created_block(middle);
        mapping.record_created_edge(second);
    }
    (critical.len(), mapping)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_util::{diamond_cfg, ff};
    use crate::{DominatorTree, detect_loops_tagged};

    #[test]
    fn split_critical_edges_on_diamond() {
        let mut cfg = diamond_cfg();
        let split = split_critical_edges(&mut cfg);
        assert_eq!(split, 0, "basic diamond has no critical edges");
    }

    #[test]
    fn split_critical_edges_inserts_block() {
        let mut cfg = crate::cfg::Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(ff("entry"));
        cfg.block_mut(a).push(ff("a"));
        cfg.block_mut(b).push(ff("b"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        let c = cfg.new_block();
        cfg.block_mut(c).push(ff("c"));
        cfg.add_edge(c, a, EdgeKind::ConditionalTrue);
        cfg.add_edge(c, b, EdgeKind::ConditionalFalse);

        let orig_blocks = cfg.block_count();
        let split = split_critical_edges(&mut cfg);
        assert_eq!(split, 4);
        assert_eq!(cfg.block_count(), orig_blocks + 4);
    }

    #[test]
    fn split_preserves_original_metadata_and_maps_both_halves() {
        let mut cfg = Cfg::<(), &'static str>::with_edge_payload();
        let left = cfg.new_block();
        let right = cfg.new_block();
        let other_source = cfg.new_block();
        let original = cfg.add_weighted_edge_with_payload(
            cfg.entry(),
            left,
            EdgeKind::ConditionalTrue,
            0.75,
            "case 7",
        );
        cfg.add_edge_with_payload(cfg.entry(), right, EdgeKind::ConditionalFalse, "default");
        cfg.add_edge_with_payload(other_source, left, EdgeKind::Jump, "other");

        let (count, mapping) = split_critical_edges_with(&mut cfg, |_, edge| *edge.payload());
        assert_eq!(count, 1);
        let replacements = mapping.edges(original).unwrap();
        assert_eq!(replacements.len(), 2);
        assert_eq!(replacements[0], original);
        assert_eq!(cfg.edge(original).kind(), EdgeKind::ConditionalTrue);
        assert_eq!(cfg.edge(original).weight(), Some(0.75));
        assert_eq!(cfg.edge(original).payload(), &"case 7");
        assert_eq!(cfg.edge(replacements[1]).payload(), &"case 7");
    }

    #[test]
    fn splitting_a_back_edge_leaves_one_loop_with_its_own_header() {
        let mut cfg = Cfg::<u32>::new();
        let entry = cfg.entry();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.add_edge(entry, header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        let back = cfg.add_weighted_edge(body, header, EdgeKind::Back, 0.5);
        cfg.add_edge(body, exit, EdgeKind::ConditionalFalse);

        let before = detect_loops_tagged(&cfg, &DominatorTree::compute(&cfg));
        assert_eq!(before.len(), 1);
        let (count, mapping) = split_critical_edges_mapped(&mut cfg);
        assert_eq!(count, 3);

        let halves = mapping.edges(back).unwrap();
        assert_eq!(halves[0], back);
        assert_eq!(cfg.edge(back).weight(), Some(0.5), "the weight stays put");
        assert_ne!(cfg.edge(back).kind(), EdgeKind::Back);
        assert_eq!(cfg.edge(halves[1]).kind(), EdgeKind::Back);
        assert_eq!(cfg.edge(halves[1]).target(), header);
        let after = detect_loops_tagged(&cfg, &DominatorTree::compute(&cfg));
        assert_eq!(after.len(), 1, "{after:?}");
        assert_eq!(after[0].header, header);
        assert!(!after[0].body.contains(&entry));
    }

    #[test]
    fn critical_edges_split_on_a_cfg_with_edge_payloads() {
        let mut cfg = Cfg::<(), &'static str>::with_edge_payload();
        let left = cfg.new_block();
        let right = cfg.new_block();
        let other_source = cfg.new_block();
        cfg.add_edge_with_payload(cfg.entry(), left, EdgeKind::ConditionalTrue, "case");
        cfg.add_edge_with_payload(cfg.entry(), right, EdgeKind::ConditionalFalse, "default");
        cfg.add_edge_with_payload(other_source, left, EdgeKind::Jump, "other");
        // `&str` is `Default`, so synthetic edges carry the empty string.
        assert_eq!(split_critical_edges(&mut cfg), 1);
        assert_eq!(cfg.block_count(), 5);
    }
}
