//! Structural oracles that judge a transform from outside: they know what
//! the documents promise about identities and graph shape, and recompute
//! reachability, adjacency, and liveness by brute force instead of asking the
//! library.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    BlockId, Cfg, CfgRenumbering, EdgeId, EdgeKind, EffectInfo, InstrInfo, Rewrite, verify,
};

use crate::toy::{Inst, Payload, Reg};

pub fn assert_valid<E>(cfg: &Cfg<Inst, E>, context: &str) {
    let report = verify(cfg);
    assert!(report.is_ok(), "{context}: {:?}", report.errors);
}

pub fn live_blocks<I, E>(cfg: &Cfg<I, E>) -> BTreeSet<BlockId> {
    cfg.block_ids().collect()
}

/// Blocks reachable from the entry by explicit graph search.
pub fn reachable_blocks<I, E>(cfg: &Cfg<I, E>) -> BTreeSet<BlockId> {
    let mut seen = BTreeSet::from([cfg.entry()]);
    let mut pending = vec![cfg.entry()];
    while let Some(block) = pending.pop() {
        for successor in cfg.successors(block) {
            if seen.insert(successor) {
                pending.push(successor);
            }
        }
    }
    seen
}

/// Edges a basic-block-graph reader calls critical: they leave a block with
/// several successors and enter a block with several predecessors.
pub fn critical_edges<I, E>(cfg: &Cfg<I, E>) -> Vec<EdgeId> {
    cfg.edge_ids()
        .filter(|&edge| {
            let edge = cfg.edge(edge);
            cfg.outgoing(edge.source()).count() > 1 && cfg.incoming(edge.target()).count() > 1
        })
        .collect()
}

/// Panics unless no cleanup pass has anything left to do, judged only from
/// the definitions in the module documents.
pub fn assert_fully_simplified<I, E>(cfg: &Cfg<I, E>, context: &str) {
    assert_eq!(
        reachable_blocks(cfg),
        live_blocks(cfg),
        "{context}: an unreachable block survived"
    );
    for block in cfg.block_ids() {
        let outgoing: Vec<EdgeId> = cfg.outgoing(block).collect();
        let [sole] = outgoing[..] else { continue };
        let edge = cfg.edge(sole);
        let target = edge.target();
        assert!(
            target == block || target == cfg.entry() || cfg.incoming(target).count() != 1,
            "{context}: {block} -> {target} is a mergeable chain link"
        );
        let fallthrough_like =
            matches!(edge.kind(), EdgeKind::Fallthrough | EdgeKind::Unconditional);
        assert!(
            block == cfg.entry()
                || !cfg.block(block).is_empty()
                || !fallthrough_like
                || target == block,
            "{context}: empty block {block} is bypassable"
        );
    }
}

/// Edges that exist before and after keep the data the passes promise to
/// preserve: kind, weight, and consumer payload.
pub fn assert_surviving_edges_unchanged<E: Payload>(
    before: &Cfg<Inst, E>,
    after: &Cfg<Inst, E>,
    context: &str,
) {
    assert_surviving_payloads_unchanged(before, after, context);
    for edge in before.edge_ids().filter(|&edge| after.contains_edge(edge)) {
        assert_eq!(
            before.edge(edge).kind(),
            after.edge(edge).kind(),
            "{context}: kind of {edge}"
        );
    }
}

/// The weight and consumer payload of every edge that exists before and
/// after, for transforms that may retag an edge's kind.
pub fn assert_surviving_payloads_unchanged<E: Payload>(
    before: &Cfg<Inst, E>,
    after: &Cfg<Inst, E>,
    context: &str,
) {
    for edge in before.edge_ids().filter(|&edge| after.contains_edge(edge)) {
        let (old, new) = (before.edge(edge), after.edge(edge));
        assert_eq!(old.payload(), new.payload(), "{context}: payload of {edge}");
        assert_eq!(
            old.weight().map(f64::to_bits),
            new.weight().map(f64::to_bits),
            "{context}: weight of {edge}"
        );
    }
}

/// Where each identity of `before` lives after a transform that may have
/// compacted. An identity the rewrite does not mention was left alone, so it
/// only moves if a renumbering did.
pub struct Placement<'a> {
    pub rewrite: &'a Rewrite,
    pub renumbering: Option<&'a CfgRenumbering>,
}

impl Placement<'_> {
    fn block_images(&self, old: BlockId) -> Vec<BlockId> {
        match self.rewrite.blocks(old) {
            Some(images) => images.to_vec(),
            None => self.renumbered(old).into_iter().collect(),
        }
    }

    fn edge_images(&self, old: EdgeId) -> Vec<EdgeId> {
        match self.rewrite.edges(old) {
            Some(images) => images.to_vec(),
            None => match self.renumbering {
                Some(renumbering) => renumbering.edge(old).into_iter().collect(),
                None => vec![old],
            },
        }
    }

    fn renumbered(&self, old: BlockId) -> Option<BlockId> {
        match self.renumbering {
            Some(renumbering) => renumbering.node(old),
            None => Some(old),
        }
    }
}

/// Checks a [`Rewrite`] against two snapshots of the graph: an identity is
/// mapped to nothing exactly when it is gone, every replacement and every
/// created identity is live, and every identity that appeared is reported as
/// created.
pub fn assert_rewrite_accounts_for_identities<E>(
    before: &Cfg<Inst, E>,
    after: &Cfg<Inst, E>,
    placement: &Placement<'_>,
    context: &str,
) {
    let rewrite = placement.rewrite;
    for old in before.block_ids() {
        let images = placement.block_images(old);
        assert!(
            images.iter().all(|&image| after.contains_block(image)),
            "{context}: {old} is mapped to a dead block {images:?}"
        );
        if rewrite.blocks(old).is_some_and(<[BlockId]>::is_empty) {
            assert!(
                placement.renumbering.is_some() || !after.contains_block(old),
                "{context}: {old} is reported removed but is live"
            );
        }
        if images.is_empty() {
            assert!(
                placement.renumbering.is_some() || !after.contains_block(old),
                "{context}: {old} has no image but is live"
            );
        }
    }
    for old in before.edge_ids() {
        let images = placement.edge_images(old);
        assert!(
            images.iter().all(|&image| after.contains_edge(image)),
            "{context}: {old} is mapped to a dead edge {images:?}"
        );
        if placement.renumbering.is_none() {
            assert_eq!(
                images.is_empty(),
                !after.contains_edge(old),
                "{context}: {old} liveness disagrees with its mapping"
            );
        }
    }
    if placement.renumbering.is_some() {
        return;
    }
    for &created in rewrite.created_blocks() {
        assert!(
            after.contains_block(created),
            "{context}: created {created} is dead"
        );
        assert!(
            !before.contains_block(created),
            "{context}: created {created} existed"
        );
    }
    for &created in rewrite.created_edges() {
        assert!(
            after.contains_edge(created),
            "{context}: created {created} is dead"
        );
        assert!(
            !before.contains_edge(created),
            "{context}: created {created} existed"
        );
    }
    for block in after
        .block_ids()
        .filter(|&block| !before.contains_block(block))
    {
        assert!(
            rewrite.created_blocks().contains(&block),
            "{context}: new block {block} is not reported as created"
        );
    }
    for edge in after.edge_ids().filter(|&edge| !before.contains_edge(edge)) {
        assert!(
            rewrite.created_edges().contains(&edge),
            "{context}: new edge {edge} is not reported as created"
        );
    }
}

/// Every instruction a surviving old block held is still found, in order and
/// uninterrupted, in the blocks the rewrite says that block became. Only
/// valid for transforms that move instructions without deleting any.
pub fn assert_images_hold_original_instructions<E>(
    before: &Cfg<Inst, E>,
    after: &Cfg<Inst, E>,
    placement: &Placement<'_>,
    context: &str,
) {
    for old in before.block_ids() {
        let original = before.block(old).instructions();
        let images = placement.block_images(old);
        if images.is_empty() || original.is_empty() {
            continue;
        }
        let image_instructions: Vec<Inst> = images
            .iter()
            .flat_map(|&image| after.block(image).instructions().iter().cloned())
            .collect();
        assert!(
            image_instructions
                .windows(original.len())
                .any(|window| window == original),
            "{context}: the instructions of {old} are not contiguous in {images:?}"
        );
    }
}

pub fn instruction_multiset<E>(
    cfg: &Cfg<Inst, E>,
    blocks: &BTreeSet<BlockId>,
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for &block in blocks {
        for inst in cfg.block(block).instructions() {
            *counts.entry(format!("{inst:?}")).or_default() += 1;
        }
    }
    counts
}

/// Variables live at the end of each block, by backward fixed point over the
/// explicit successor lists. Independent of the library's `Liveness`.
pub fn brute_force_live_out<E>(cfg: &Cfg<Inst, E>) -> BTreeMap<BlockId, BTreeSet<Reg>> {
    brute_force_liveness(cfg).1
}

/// Variables live at the start of each block; see [`brute_force_live_out`].
pub fn brute_force_live_in<E>(cfg: &Cfg<Inst, E>) -> BTreeMap<BlockId, BTreeSet<Reg>> {
    brute_force_liveness(cfg).0
}

fn brute_force_liveness<E>(
    cfg: &Cfg<Inst, E>,
) -> (
    BTreeMap<BlockId, BTreeSet<Reg>>,
    BTreeMap<BlockId, BTreeSet<Reg>>,
) {
    let mut live_in: BTreeMap<BlockId, BTreeSet<Reg>> = cfg
        .block_ids()
        .map(|block| (block, BTreeSet::new()))
        .collect();
    let mut live_out = live_in.clone();
    loop {
        let mut changed = false;
        for block in cfg.block_ids().collect::<Vec<_>>().into_iter().rev() {
            let out: BTreeSet<Reg> = cfg
                .successors(block)
                .flat_map(|successor| live_in[&successor].iter().copied())
                .collect();
            let mut live = out.clone();
            for inst in cfg.block(block).instructions().iter().rev() {
                for def in inst.defs() {
                    live.remove(def);
                }
                live.extend(inst.uses().iter().copied());
            }
            let previous_out = live_out.insert(block, out.clone());
            let previous_in = live_in.insert(block, live.clone());
            changed |= previous_out != Some(out) || previous_in != Some(live);
        }
        if !changed {
            return (live_in, live_out);
        }
    }
}

/// What iterated classic dead-code removal leaves of the reachable blocks:
/// repeat "drop a pure instruction none of whose results is live afterwards"
/// on the recomputed liveness until nothing is left to drop. The result of
/// the unique least fixpoint, so it does not depend on removal order.
pub fn classic_dead_code_survivors<E: Clone>(cfg: &Cfg<Inst, E>) -> BTreeMap<String, usize> {
    let reachable = reachable_blocks(cfg);
    let mut work = cfg.clone();
    for block in cfg.block_ids().filter(|block| !reachable.contains(block)) {
        work.block_mut(block).instructions_mut().clear();
    }
    loop {
        let live_out = brute_force_live_out(&work);
        let mut changed = false;
        for &block in &reachable {
            let mut live = live_out[&block].clone();
            let mut kept = Vec::new();
            for inst in work.block(block).instructions().iter().rev() {
                let removable = inst.effects().is_empty()
                    && !inst.defs().is_empty()
                    && inst.defs().iter().all(|def| !live.contains(def));
                if removable {
                    changed = true;
                    continue;
                }
                for def in inst.defs() {
                    live.remove(def);
                }
                live.extend(inst.uses().iter().copied());
                kept.push(inst.clone());
            }
            kept.reverse();
            *work.block_mut(block).instructions_mut() = kept;
        }
        if !changed {
            return instruction_multiset(&work, &reachable);
        }
    }
}
