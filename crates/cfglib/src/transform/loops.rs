//! Loop transformations — rotation, invariant code motion, unrolling.
//!
//! These transforms build on the loop detection and canonicalization
//! infrastructure in [`crate::graph::structure`].

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::InstrInfo;
use crate::edge::EdgeKind;
use crate::graph::structure::NaturalLoop;

/// Result of loop rotation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RotationResult {
    /// The new header block (was the first body block).
    pub new_header: BlockId,
    /// The duplicated test block (copy of original header at bottom).
    pub bottom_test: BlockId,
}

/// Rotate a loop from top-tested to bottom-tested form.
///
/// Transforms:
/// ```text
/// header:            →    header (execute once):
///   if (!cond) exit       body:
///   body                    ...
///   goto header             if (cond) goto body
/// ```
///
/// This is only valid for loops with a single latch and where the
/// header is a simple conditional. Returns `None` if rotation is
/// not applicable.
///
/// Requires `I: Clone` to duplicate the header block's instructions.
pub fn rotate_loop<I: Clone>(cfg: &mut Cfg<I>, lp: &NaturalLoop) -> Option<RotationResult> {
    // Only rotate loops with exactly one latch.
    if lp.latches.len() != 1 {
        return None;
    }

    let header = lp.header;
    let &latch = lp.latches.iter().next()?;

    // Header must have exactly two successors (conditional).
    let header_succs: Vec<BlockId> = cfg.successors(header).collect();
    if header_succs.len() != 2 {
        return None;
    }

    // One successor must be in the loop body, one must be an exit.
    let (body_succ, _exit_succ) = {
        let in_loop_0 = lp.body.contains(&header_succs[0]);
        let in_loop_1 = lp.body.contains(&header_succs[1]);
        match (in_loop_0, in_loop_1) {
            (true, false) => (header_succs[0], header_succs[1]),
            (false, true) => (header_succs[1], header_succs[0]),
            _ => return None, // both in or both out
        }
    };

    // Create a copy of the header at the bottom (new latch test).
    let bottom_test = cfg.new_block();
    let header_instrs = cfg.block(header).instructions().to_vec();
    *cfg.block_mut(bottom_test).instructions_vec_mut() = header_instrs;

    // Redirect the old latch → header edge to latch → bottom_test.
    let latch_edges: Vec<_> = cfg
        .successor_edges(latch)
        .iter()
        .copied()
        .filter(|&eid| cfg.edge(eid).target() == header)
        .collect();
    for eid in latch_edges {
        let kind = cfg.edge(eid).kind();
        cfg.remove_edge(eid);
        cfg.add_edge(latch, bottom_test, kind);
    }

    // Bottom test loops back to body_succ (not header) on continue.
    cfg.add_edge(bottom_test, body_succ, EdgeKind::Back);

    Some(RotationResult {
        new_header: body_succ,
        bottom_test,
    })
}

/// Identify loop-invariant instructions.
///
/// An instruction is loop-invariant if all of its operands are either:
/// - Defined outside the loop, or
/// - Defined by other loop-invariant instructions.
///
/// Returns indices `(block, instruction_index)` of invariant instructions.
#[must_use]
pub fn find_loop_invariants<I: InstrInfo>(cfg: &Cfg<I>, lp: &NaturalLoop) -> Vec<(BlockId, usize)> {
    let mut invariants = BTreeSet::new();
    let mut changed = true;

    // Collect all defs inside the loop.
    let mut loop_defs = BTreeSet::new();
    for &bid in &lp.body {
        for (idx, inst) in cfg.block(bid).instructions().iter().enumerate() {
            for d in inst.defs() {
                loop_defs.insert((bid, idx, d.clone()));
            }
        }
    }

    while changed {
        changed = false;
        for &bid in &lp.body {
            for (idx, inst) in cfg.block(bid).instructions().iter().enumerate() {
                if invariants.contains(&(bid, idx)) {
                    continue;
                }
                // Check: all uses are defined outside loop or by invariants.
                let all_uses_invariant = inst.uses().iter().all(|u| {
                    // Is there a def of this variable inside the loop
                    // that is NOT invariant?
                    !loop_defs.iter().any(|(db, di, variable)| {
                        variable == u && !invariants.contains(&(*db, *di))
                    })
                });
                if all_uses_invariant {
                    invariants.insert((bid, idx));
                    changed = true;
                }
            }
        }
    }

    invariants.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::dominator::DominatorTree;
    use crate::graph::structure::detect_loops;
    use crate::test_util::{DfInst, df_def, df_ff, df_use};

    #[test]
    fn find_invariants_in_simple_loop() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();

        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(df_def("def0", 0));
        cfg.block_mut(header)
            .instructions_vec_mut()
            .push(df_ff("cmp"));
        cfg.block_mut(body)
            .instructions_vec_mut()
            .push(df_use("use0", 0));
        cfg.block_mut(exit)
            .instructions_vec_mut()
            .push(df_ff("ret"));

        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Back);

        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_ne!(loops.len(), 0);

        let invs = find_loop_invariants(&cfg, &loops[0]);
        assert!(
            !invs.is_empty(),
            "instruction using only outer-defined loc should be invariant"
        );
    }

    #[test]
    fn rotate_simple_loop() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let header = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();

        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(df_ff("init"));
        cfg.block_mut(header)
            .instructions_vec_mut()
            .push(df_ff("cmp"));
        cfg.block_mut(body)
            .instructions_vec_mut()
            .push(df_ff("work"));
        cfg.block_mut(exit)
            .instructions_vec_mut()
            .push(df_ff("ret"));

        cfg.add_edge(cfg.entry(), header, EdgeKind::Fallthrough);
        cfg.add_edge(header, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(header, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, header, EdgeKind::Back);

        let dom = DominatorTree::compute(&cfg);
        let loops = detect_loops(&cfg, &dom);
        assert_eq!(loops.len(), 1);

        let result = rotate_loop(&mut cfg, &loops[0]);
        assert!(result.is_some(), "simple loop should be rotatable");
        let rot = result.unwrap();
        assert_eq!(cfg.block(rot.bottom_test).instructions().len(), 1);
    }
}
