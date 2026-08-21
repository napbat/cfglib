//! CFG transformation passes.
//!
//! All mutation passes operate in-place and return the number of
//! blocks, edges, or instructions affected.
//!
//! - [`cleanup`] — unreachable removal, block merging, empty-block bypass, combined simplify.
//! - [`critical`] — critical edge splitting (required for SSA).
//! - [`dce`] — dead code elimination via liveness analysis.
//! - [`linearize()`] — re-serialize a CFG back to a flat instruction stream.

pub mod cleanup;
pub mod coloring;
pub mod contract;
pub mod critical;
pub mod dce;
pub mod linearize;
pub mod loops;
pub mod pre;

// Re-export all pass entry points at the `transform` level for convenience.
pub use cleanup::{merge_blocks, remove_empty_blocks, remove_unreachable, simplify};
pub use critical::split_critical_edges;
pub use dce::dead_code_elimination;
pub use linearize::{BlockOrder, Emitter, LinearInst, linearize};

#[cfg(test)]
mod tests {
    extern crate alloc;
    use alloc::vec::Vec;

    use super::*;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::test_util::{MockInst, ff};

    /// Build a diamond CFG: entry → A, entry → B, A → merge, B → merge.
    fn diamond_cfg() -> Cfg<MockInst> {
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        let merge = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(a).instructions_vec_mut().push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.block_mut(merge)
            .instructions_vec_mut()
            .push(ff("merge"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        cfg.add_edge(a, merge, EdgeKind::Fallthrough);
        cfg.add_edge(b, merge, EdgeKind::Fallthrough);
        cfg
    }

    #[test]
    fn remove_unreachable_noop_when_all_reachable() {
        let mut cfg = diamond_cfg();
        let removed = remove_unreachable(&mut cfg);
        assert_eq!(removed, 0);
    }

    #[test]
    fn remove_unreachable_removes_disconnected_block() {
        let mut cfg = diamond_cfg();
        let orphan = cfg.new_block();
        cfg.block_mut(orphan)
            .instructions_vec_mut()
            .push(ff("dead"));
        let removed = remove_unreachable(&mut cfg);
        assert_eq!(removed, 1);
        assert!(cfg.block(orphan).instructions().is_empty());
    }

    #[test]
    fn merge_blocks_merges_linear_chain() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        let c = cfg.new_block();
        let d = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.block_mut(c).instructions_vec_mut().push(ff("c"));
        cfg.block_mut(d).instructions_vec_mut().push(ff("d"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        cfg.add_edge(b, c, EdgeKind::Fallthrough);
        cfg.add_edge(c, d, EdgeKind::Fallthrough);
        let merged = merge_blocks(&mut cfg);
        assert_eq!(merged, 3);
        assert_eq!(cfg.block(cfg.entry()).instructions().len(), 4);
        assert_eq!(cfg.successor_edges(cfg.entry()).len(), 0);
        assert!(cfg.block(b).instructions().is_empty());
        assert!(cfg.block(c).instructions().is_empty());
        assert!(cfg.block(d).instructions().is_empty());
    }

    #[test]
    fn merge_blocks_does_not_merge_when_multiple_predecessors() {
        let mut cfg = diamond_cfg();
        let merged = merge_blocks(&mut cfg);
        assert_eq!(merged, 0);
    }

    #[test]
    fn merge_blocks_skips_self_loop() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("a"));
        cfg.add_edge(cfg.entry(), cfg.entry(), EdgeKind::Back);
        let merged = merge_blocks(&mut cfg);
        assert_eq!(merged, 0);
    }

    #[test]
    fn remove_empty_blocks_bypasses_empty_block() {
        let mut cfg = Cfg::new();
        let first_empty = cfg.new_block();
        let second_empty = cfg.new_block();
        let target = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(target)
            .instructions_vec_mut()
            .push(ff("target"));
        cfg.add_edge(cfg.entry(), first_empty, EdgeKind::Fallthrough);
        cfg.add_edge(first_empty, second_empty, EdgeKind::Fallthrough);
        cfg.add_edge(second_empty, target, EdgeKind::Fallthrough);
        let removed = remove_empty_blocks(&mut cfg);
        assert_eq!(removed, 2);
        let succs: Vec<_> = cfg.successors(cfg.entry()).collect();
        assert_eq!(succs.len(), 1);
        assert_eq!(succs[0], target);
        assert_eq!(cfg.predecessor_edges(first_empty).len(), 0);
        assert_eq!(cfg.successor_edges(first_empty).len(), 0);
        assert_eq!(cfg.predecessor_edges(second_empty).len(), 0);
        assert_eq!(cfg.successor_edges(second_empty).len(), 0);
    }

    #[test]
    fn remove_empty_blocks_does_not_remove_entry() {
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let removed = remove_empty_blocks(&mut cfg);
        assert_eq!(removed, 0);
    }

    #[test]
    fn simplify_runs_all_passes() {
        let mut cfg = Cfg::new();
        let empty = cfg.new_block();
        let b = cfg.new_block();
        let orphan = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.block_mut(orphan)
            .instructions_vec_mut()
            .push(ff("dead"));
        cfg.add_edge(cfg.entry(), empty, EdgeKind::Fallthrough);
        cfg.add_edge(empty, b, EdgeKind::Fallthrough);
        let total = simplify(&mut cfg);
        assert!(
            total > 0,
            "simplify should perform at least 1 transformation"
        );
        assert!(cfg.block(orphan).instructions().is_empty());
    }

    #[test]
    fn split_critical_edges_on_diamond() {
        let mut cfg = diamond_cfg();
        let split = split_critical_edges(&mut cfg);
        assert_eq!(split, 0, "basic diamond has no critical edges");
    }

    #[test]
    fn split_critical_edges_inserts_block() {
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .push(ff("entry"));
        cfg.block_mut(a).instructions_vec_mut().push(ff("a"));
        cfg.block_mut(b).instructions_vec_mut().push(ff("b"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);
        let c = cfg.new_block();
        cfg.block_mut(c).instructions_vec_mut().push(ff("c"));
        cfg.add_edge(c, a, EdgeKind::ConditionalTrue);
        cfg.add_edge(c, b, EdgeKind::ConditionalFalse);

        let orig_blocks = cfg.num_blocks();
        let split = split_critical_edges(&mut cfg);
        assert_eq!(split, 4);
        assert_eq!(cfg.num_blocks(), orig_blocks + 4);
    }

    #[test]
    fn dead_code_elimination_removes_unused_def() {
        use crate::test_util::{DfInst, df_def, df_use};

        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();

        cfg.block_mut(cfg.entry())
            .instructions_vec_mut()
            .extend([df_def("dead_def", 0), df_def("live_def", 1)]);

        cfg.block_mut(exit)
            .instructions_vec_mut()
            .push(df_use("use1", 1));

        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);

        let removed = dead_code_elimination(&mut cfg);
        assert_eq!(removed, 1, "should remove the dead def of loc0");
        assert_eq!(cfg.block(cfg.entry()).instructions().len(), 1);
        assert_eq!(cfg.block(cfg.entry()).instructions()[0].name, "live_def");
    }
}
