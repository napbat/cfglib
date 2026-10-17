//! Loop transformations — rotation and loop-invariant detection.
//!
//! These transforms build on the loop detection and canonicalization
//! infrastructure in [`crate::graph::structure`].

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::def_use::DefUseChains;
use crate::edge::{EdgeId, EdgeKind};
use crate::graph::structure::NaturalLoop;
use crate::region::HandlerBody;
use crate::{EffectInfo, ProgramPoint};

/// Result of loop rotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LoopRotation {
    /// The new loop header: the first body block, entered from the guard
    /// (the original header) and from `bottom_test`.
    pub new_header: BlockId,
    /// The copy of the original header that now ends every iteration.
    pub bottom_test: BlockId,
}

/// Rotate a loop from top-tested to bottom-tested form.
///
/// ```text
/// before:                    after:
/// header:  if (!cond) exit   header:      if (!cond) exit   (a guard, runs once)
/// body:    ...               body:        ...
/// latch:   goto header       latch:       goto bottom_test
///                            bottom_test: if (cond) goto body else exit
/// ```
///
/// `bottom_test` holds a copy of the header's instructions and a copy of each
/// of its two edges (kind, weight, payload, order), so it continues exactly as
/// the header did and every path through the original loop survives. The
/// latch edge keeps its identity, weight, and payload and now targets
/// `bottom_test`; a [`Back`](EdgeKind::Back) kind becomes
/// [`Fallthrough`](EdgeKind::Fallthrough) because the edge no longer reaches a
/// loop header. The rotated loop is headed by
/// [`new_header`](LoopRotation::new_header).
///
/// Rotation applies to a loop with a single latch whose header is a simple
/// conditional: exactly two non-exceptional edges, one staying in the loop and
/// one leaving it. It returns `None` otherwise.
///
/// Requires `I: Clone` to copy the header's instructions and `E: Clone` to
/// copy its edge payloads.
pub fn rotate_loop<I: Clone, E: Clone>(
    cfg: &mut Cfg<I, E>,
    lp: &NaturalLoop,
) -> Option<LoopRotation> {
    let mut latches = lp.latches.iter();
    let (Some(&latch), None) = (latches.next(), latches.next()) else {
        return None;
    };

    let header = lp.header;
    let header_edges: Vec<EdgeId> = cfg.outgoing(header).collect();
    let [first, second] = header_edges[..] else {
        return None;
    };
    if [first, second]
        .iter()
        .any(|&edge| cfg.edge(edge).kind().is_exceptional())
    {
        return None;
    }
    let targets = [first, second].map(|edge| cfg.edge(edge).target());
    let new_header = match targets.map(|target| lp.body.contains(&target)) {
        [true, false] => targets[0],
        [false, true] => targets[1],
        _ => return None,
    };

    let bottom_test = duplicate_block(cfg, header);
    let latch_edges: Vec<EdgeId> = cfg
        .outgoing(latch)
        .filter(|&edge| cfg.edge(edge).target() == header)
        .collect();
    for edge in latch_edges {
        cfg.redirect_edge_target(edge, bottom_test);
        if cfg.edge(edge).kind() == EdgeKind::Back {
            cfg.edge_mut(edge).set_kind(EdgeKind::Fallthrough);
        }
    }

    Some(LoopRotation {
        new_header,
        bottom_test,
    })
}

/// A new block holding a copy of `block`'s instructions and of its outgoing
/// edges, in the same order with kind, weight, and payload retained. The copy
/// is protected by the regions that protect `block` and belongs to the
/// handler bodies `block` belongs to.
fn duplicate_block<I: Clone, E: Clone>(cfg: &mut Cfg<I, E>, block: BlockId) -> BlockId {
    let copy = cfg.new_block();
    let instructions = cfg.block(block).instructions().to_vec();
    *cfg.block_mut(copy).instructions_mut() = instructions;
    let outgoing: Vec<_> = cfg
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
        let edge = cfg.add_edge_with_payload(copy, target, kind, payload);
        cfg.edge_mut(edge).set_weight(weight);
    }
    for region in &mut cfg.regions {
        if region.protected_blocks.contains(&block) {
            region.protected_blocks.insert(copy);
        }
        for handler in &mut region.handlers {
            if let HandlerBody::Known(blocks) = &mut handler.body
                && blocks.contains(&block)
            {
                blocks.insert(copy);
            }
        }
    }
    copy
}

/// Identify loop-invariant instructions.
///
/// An instruction is loop-invariant when it computes a value that is the same
/// on every iteration: it defines something, has no declared side effect, and
/// cannot unwind, and every operand is read from
/// - definitions that all lie outside the loop, or
/// - exactly one definition, which is itself a loop-invariant instruction
///   in the loop.
///
/// An operand that two definitions can supply, such as a variable set on one
/// arm inside the loop, is therefore not invariant even when both definitions
/// are constants.
///
/// This is the invariance test only. Whether moving an instruction out of the
/// loop is safe or profitable (its destination may have other definitions in
/// the loop, and it may not dominate every exit) remains the caller's
/// decision.
///
/// Returns indices `(block, instruction_index)` of invariant instructions.
#[must_use]
pub fn find_loop_invariants<I: EffectInfo, E>(
    cfg: &Cfg<I, E>,
    lp: &NaturalLoop,
) -> Vec<(BlockId, usize)> {
    let chains = DefUseChains::compute(cfg);
    let mut invariants: BTreeSet<ProgramPoint> = BTreeSet::new();
    let mut changed = true;

    while changed {
        changed = false;
        for &block in &lp.body {
            for (inst_idx, inst) in cfg.block(block).instructions().iter().enumerate() {
                let point = ProgramPoint { block, inst_idx };
                let computes_a_value =
                    !inst.defs().is_empty() && inst.effects().is_empty() && !inst.may_unwind();
                if !computes_a_value || invariants.contains(&point) {
                    continue;
                }
                let invariant = inst.uses().iter().all(|variable| {
                    let suppliers = chains.defs_of(point).iter().filter(|site| {
                        cfg.block(site.block).instructions()[site.inst_idx]
                            .defs()
                            .contains(variable)
                    });
                    let mut inside = suppliers
                        .clone()
                        .filter(|site| lp.body.contains(&site.block));
                    match (inside.next(), inside.next()) {
                        (None, _) => true,
                        (Some(site), None) => suppliers.count() == 1 && invariants.contains(site),
                        _ => false,
                    }
                });
                if invariant {
                    invariants.insert(point);
                    changed = true;
                }
            }
        }
    }

    invariants
        .into_iter()
        .map(|point| (point.block, point.inst_idx))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeId;
    use crate::graph::dominator::DominatorTree;
    use crate::graph::structure::{detect_loops, detect_loops_tagged};
    use crate::region::{Handler, HandlerKind, Region, RegionId};
    use crate::test_util::{
        DfInst, TestEffect, df_def, df_ff, df_impure, df_op, df_throwing, df_use,
    };

    /// `entry → header`, a top-tested loop whose body is a chain of
    /// `body_len` blocks ending in the latch, and `header → exit`.
    struct CountingLoop {
        cfg: Cfg<DfInst>,
        header: BlockId,
        first: BlockId,
        latch: BlockId,
        exit: BlockId,
        latch_edge: EdgeId,
    }

    fn counting_loop(body_len: usize) -> CountingLoop {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let header = cfg.new_block();
        let body: Vec<BlockId> = (0..body_len).map(|_| cfg.new_block()).collect();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(df_ff("init"));
        cfg.block_mut(header).instructions_mut().push(df_ff("cmp"));
        for &block in &body {
            cfg.block_mut(block).instructions_mut().push(df_ff("work"));
        }
        cfg.block_mut(exit).instructions_mut().push(df_ff("ret"));

        let first = body[0];
        let latch = body[body_len - 1];
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_weighted_edge(header, first, EdgeKind::ConditionalTrue, 0.75);
        cfg.add_weighted_edge(header, exit, EdgeKind::ConditionalFalse, 0.25);
        for pair in body.windows(2) {
            cfg.add_edge(pair[0], pair[1], EdgeKind::Fallthrough);
        }
        let latch_edge = cfg.add_weighted_edge(latch, header, EdgeKind::Back, 1.0);
        CountingLoop {
            cfg,
            header,
            first,
            latch,
            exit,
            latch_edge,
        }
    }

    fn the_loop<E>(cfg: &Cfg<DfInst, E>) -> NaturalLoop {
        let mut loops = detect_loops(cfg, &DominatorTree::compute(cfg));
        assert_eq!(loops.len(), 1, "the fixture holds exactly one loop");
        loops.remove(0)
    }

    fn exits(cfg: &Cfg<DfInst>, block: BlockId) -> Vec<(BlockId, EdgeKind, Option<f64>)> {
        cfg.outgoing(block)
            .map(|edge| {
                let edge = cfg.edge(edge);
                (edge.target(), edge.kind(), edge.weight())
            })
            .collect()
    }

    #[test]
    fn the_bottom_test_of_a_rotated_loop_leaves_through_the_headers_exit() {
        for body_len in [1, 3] {
            let CountingLoop {
                mut cfg,
                header,
                first,
                exit,
                ..
            } = counting_loop(body_len);
            let lp = the_loop(&cfg);

            let rotation = rotate_loop(&mut cfg, &lp).expect("a top-tested loop rotates");

            assert_eq!(rotation.new_header, first);
            let test = [
                (first, EdgeKind::ConditionalTrue, Some(0.75)),
                (exit, EdgeKind::ConditionalFalse, Some(0.25)),
            ];
            assert_eq!(exits(&cfg, rotation.bottom_test), test);
            assert_eq!(exits(&cfg, header), test, "the guard keeps its own edges");
            assert_eq!(cfg.block(rotation.bottom_test).instructions().len(), 1);
        }
    }

    #[test]
    fn rotation_retargets_the_latch_edge_instead_of_replacing_it() {
        let CountingLoop {
            mut cfg,
            latch,
            latch_edge,
            ..
        } = counting_loop(3);
        let edges_before = cfg.edge_count();
        let lp = the_loop(&cfg);

        let rotation = rotate_loop(&mut cfg, &lp).expect("a top-tested loop rotates");

        let edge = cfg.edge(latch_edge);
        assert_eq!(
            (edge.source(), edge.target()),
            (latch, rotation.bottom_test)
        );
        assert_eq!(edge.weight(), Some(1.0));
        assert_eq!(
            edge.kind(),
            EdgeKind::Fallthrough,
            "it no longer reaches a loop header, so it is no back edge"
        );
        assert_eq!(
            cfg.edge_count(),
            edges_before + 2,
            "only the bottom test's own two edges are new"
        );
    }

    #[test]
    fn a_rotated_loop_is_a_natural_loop_headed_by_its_first_body_block() {
        let CountingLoop {
            mut cfg,
            header,
            first,
            latch,
            ..
        } = counting_loop(3);
        let lp = the_loop(&cfg);
        let rotation = rotate_loop(&mut cfg, &lp).expect("a top-tested loop rotates");

        let dominators = DominatorTree::compute(&cfg);
        for loops in [
            detect_loops(&cfg, &dominators),
            detect_loops_tagged(&cfg, &dominators),
        ] {
            assert_eq!(loops.len(), 1);
            let rotated = &loops[0];
            assert_eq!(rotated.header, first);
            assert_eq!(rotated.latches, BTreeSet::from([rotation.bottom_test]));
            assert!(rotated.body.contains(&latch));
            assert!(rotated.body.contains(&rotation.bottom_test));
            assert!(
                !rotated.body.contains(&header),
                "the header only guards entry"
            );
        }
    }

    #[test]
    fn rotation_declines_a_loop_with_two_latches() {
        let CountingLoop {
            mut cfg,
            header,
            first,
            ..
        } = counting_loop(1);
        let second_latch = cfg.new_block();
        cfg.add_edge(first, second_latch, EdgeKind::ConditionalFalse);
        cfg.add_edge(second_latch, header, EdgeKind::Back);
        let lp = the_loop(&cfg);
        assert_eq!(lp.latches.len(), 2);

        assert_eq!(rotate_loop(&mut cfg, &lp), None);
    }

    #[test]
    fn rotation_declines_a_header_whose_other_edge_is_an_unwind() {
        // The header's two edges are the loop body and an unwind to a
        // handler outside the loop: that is not a two-way conditional.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let handler = cfg.new_block();
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::Fallthrough);
        cfg.add_edge(header, handler, EdgeKind::ExceptionUnwind);
        cfg.add_edge(body, header, EdgeKind::Back);
        let lp = the_loop(&cfg);

        assert_eq!(rotate_loop(&mut cfg, &lp), None);
    }

    #[test]
    fn the_bottom_test_joins_the_regions_and_handler_bodies_of_the_header() {
        let CountingLoop {
            mut cfg,
            header,
            first,
            ..
        } = counting_loop(1);
        let catch = cfg.new_block();
        let region = cfg.add_region(Region {
            id: RegionId::from_index(0),
            protected_blocks: BTreeSet::from([header, first]),
            handlers: Vec::from([Handler {
                entry: catch,
                body: HandlerBody::Known(BTreeSet::from([catch, header])),
                kind: HandlerKind::Catch,
            }]),
            parent: None,
        });
        let lp = the_loop(&cfg);

        let rotation = rotate_loop(&mut cfg, &lp).expect("a top-tested loop rotates");

        let region = cfg.region(region).expect("rotation keeps the region");
        assert!(region.protected_blocks.contains(&rotation.bottom_test));
        let HandlerBody::Known(body) = &region.handlers[0].body else {
            panic!("the handler body stays known");
        };
        assert!(body.contains(&rotation.bottom_test));
    }

    #[test]
    fn rotation_and_invariants_accept_consumer_edge_payloads() {
        let mut cfg = Cfg::<DfInst, &str>::with_edge_payload();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(df_def("x", 0));
        cfg.block_mut(header).push(df_ff("cmp"));
        cfg.block_mut(body).push(df_op("t", "add", 1, &[0]));
        cfg.add_edge_with_payload(cfg.entry(), header, EdgeKind::Fallthrough, "enter");
        cfg.add_edge_with_payload(header, body, EdgeKind::ConditionalTrue, "stay");
        cfg.add_edge_with_payload(header, exit, EdgeKind::ConditionalFalse, "leave");
        let latch_edge = cfg.add_edge_with_payload(body, header, EdgeKind::Back, "again");
        let lp = the_loop(&cfg);

        assert_eq!(find_loop_invariants(&cfg, &lp), [(body, 0)]);

        let rotation = rotate_loop(&mut cfg, &lp).expect("a top-tested loop rotates");
        let copied: Vec<&str> = cfg
            .outgoing(rotation.bottom_test)
            .map(|edge| *cfg.edge(edge).payload())
            .collect();
        assert_eq!(copied, ["stay", "leave"]);
        assert_eq!(cfg.edge(latch_edge).payload(), &"again");
    }

    fn instructions<const N: usize>(instructions: [DfInst; N]) -> Vec<DfInst> {
        instructions.into()
    }

    /// `entry → header ⇄ body`, `header → exit`, with the given instructions
    /// in the entry and body blocks. Returns the body block.
    fn loop_over(entry: Vec<DfInst>, body: Vec<DfInst>) -> (Cfg<DfInst>, BlockId) {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let header = cfg.new_block();
        let work = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().extend(entry);
        cfg.block_mut(header).instructions_mut().push(df_ff("cmp"));
        cfg.block_mut(work).instructions_mut().extend(body);
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, work, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(work, header, EdgeKind::Back);
        (cfg, work)
    }

    #[test]
    fn an_instruction_over_outer_definitions_is_invariant() {
        let (cfg, body) = loop_over(
            instructions([df_def("x", 0)]),
            instructions([df_op("t", "add", 1, &[0])]),
        );
        assert_eq!(find_loop_invariants(&cfg, &the_loop(&cfg)), [(body, 0)]);
    }

    #[test]
    fn invariance_chains_through_invariant_definitions() {
        let (cfg, body) = loop_over(
            instructions([df_def("x", 0)]),
            instructions([df_op("t", "add", 1, &[0]), df_op("u", "mul", 2, &[1])]),
        );
        assert_eq!(
            find_loop_invariants(&cfg, &the_loop(&cfg)),
            [(body, 0), (body, 1)]
        );
    }

    #[test]
    fn a_loop_carried_operand_is_not_invariant() {
        let (cfg, _) = loop_over(
            instructions([df_def("i", 0)]),
            instructions([df_op("step", "add", 0, &[0]), df_op("t", "mul", 1, &[0])]),
        );
        assert_eq!(find_loop_invariants(&cfg, &the_loop(&cfg)), []);
    }

    #[test]
    fn an_instruction_that_defines_nothing_is_not_invariant() {
        let (cfg, _) = loop_over(
            instructions([df_def("x", 0)]),
            instructions([
                df_ff("nop"),
                df_use("print", 0),
                df_impure("store", TestEffect::MemoryWrite),
            ]),
        );
        assert_eq!(find_loop_invariants(&cfg, &the_loop(&cfg)), []);
    }

    #[test]
    fn a_side_effecting_or_throwing_definition_is_not_invariant() {
        let mut load = df_impure("load", TestEffect::MemoryRead);
        load.defs.push(1);
        let trap = df_throwing(df_def("div", 2));
        let (cfg, _) = loop_over(Vec::new(), instructions([load, trap]));
        assert_eq!(find_loop_invariants(&cfg, &the_loop(&cfg)), []);
    }

    #[test]
    fn an_operand_with_a_conditional_loop_definition_is_not_invariant() {
        // `x = 0` before the loop; `x = 1` on one arm inside it; the join
        // reads x. Both definitions are constants, yet which one reaches
        // the join changes from one iteration to the next.
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let header = cfg.new_block();
        let branch = cfg.new_block();
        let set = cfg.new_block();
        let skip = cfg.new_block();
        let join = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(df_def("x0", 0));
        cfg.block_mut(header).instructions_mut().push(df_ff("cmp"));
        cfg.block_mut(set).instructions_mut().push(df_def("x1", 0));
        cfg.block_mut(join)
            .instructions_mut()
            .push(df_op("y", "add", 1, &[0]));
        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, branch, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(branch, set, EdgeKind::ConditionalTrue);
        cfg.add_edge(branch, skip, EdgeKind::ConditionalFalse);
        cfg.add_edge(set, join, EdgeKind::Fallthrough);
        cfg.add_edge(skip, join, EdgeKind::Fallthrough);
        cfg.add_edge(join, header, EdgeKind::Back);

        assert_eq!(
            find_loop_invariants(&cfg, &the_loop(&cfg)),
            [(set, 0)],
            "only the constant definition itself is invariant"
        );
    }
}
