//! Reverse (transpose) CFG construction.
//!
//! Builds a new CFG with all edges flipped. The entry of the reverse
//! CFG is an exit block of the original. If the original has multiple
//! exit blocks, a synthetic merge block is created as the new entry.

extern crate alloc;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;

/// Build the reverse (transpose) of a CFG.
///
/// Every edge `(A → B)` in the original becomes `(B → A)` in the
/// reverse. Block contents are preserved. The new entry is:
/// - The sole exit block if there is exactly one.
/// - A new synthetic block connected to all exit blocks otherwise.
///
/// # Panics
///
/// Panics if the CFG has no blocks.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, EdgeKind, reverse_cfg};
///
/// let mut cfg = Cfg::<u32>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
///
/// let rev = reverse_cfg(&cfg);
/// // In the reverse CFG, b1 is the entry and has an edge to b0.
/// let succs: Vec<_> = rev.successors(rev.entry()).collect();
/// assert_eq!(succs.len(), 1);
/// ```
#[must_use]
pub fn reverse_cfg<I: Clone>(cfg: &Cfg<I>) -> Cfg<I> {
    assert!(cfg.block_count() > 0, "cannot reverse an empty CFG");

    let mut rev = Cfg::new();

    // Copy block contents (block 0 already exists from Cfg::new).
    for _i in 1..cfg.block_count() {
        rev.new_block();
    }
    for block_id in cfg.block_ids() {
        let src = cfg.block(block_id);
        let bid = block_id;
        let dst = rev.block_mut(bid);
        *dst.instructions_mut() = src.instructions().to_vec();
        if let Some(lbl) = src.label() {
            dst.set_label(lbl);
        }
    }

    // Reverse all live edges.
    for edge in cfg.edges() {
        rev.add_edge(edge.target(), edge.source(), edge.kind());
    }

    // Set entry to exit block(s).
    let exits: alloc::vec::Vec<BlockId> = cfg.exit_blocks().collect();
    if exits.len() == 1 {
        rev.set_entry(exits[0]);
    } else if exits.len() > 1 {
        let synth = rev.new_block();
        for &ex in &exits {
            rev.add_edge(synth, ex, EdgeKind::Unconditional);
        }
        rev.set_entry(synth);
    }
    // If no exits (infinite loop), entry stays as block 0.

    rev
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use crate::test_util::ff;
    use alloc::vec::Vec;

    #[test]
    fn reverse_linear_chain() {
        // entry → b → c
        let mut cfg = Cfg::new();
        let b = cfg.new_block();
        let c = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("e"));
        cfg.block_mut(b).instructions_mut().push(ff("b"));
        cfg.block_mut(c).instructions_mut().push(ff("c"));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        cfg.add_edge(b, c, EdgeKind::Fallthrough);

        let rev = reverse_cfg(&cfg);

        // In reverse: c → b → entry. Entry of reverse is c.
        assert_eq!(rev.entry(), c);
        let rev_succs: Vec<BlockId> = rev.successors(c).collect();
        assert!(rev_succs.contains(&b));
        let rev_succs_b: Vec<BlockId> = rev.successors(b).collect();
        assert!(rev_succs_b.contains(&cfg.entry()));
    }

    #[test]
    fn reverse_preserves_instructions() {
        let mut cfg = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(ff("hello"));

        let rev = reverse_cfg(&cfg);
        assert_eq!(rev.block(cfg.entry()).instructions().len(), 1);
    }

    #[test]
    fn reverse_diamond_creates_synth_entry() {
        // entry → a, entry → b (both a,b are exits)
        let mut cfg = Cfg::new();
        let a = cfg.new_block();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry()).instructions_mut().push(ff("e"));
        cfg.add_edge(cfg.entry(), a, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), b, EdgeKind::ConditionalFalse);

        let rev = reverse_cfg(&cfg);

        // Two exits → synthetic entry block with edges to a and b.
        let synth = rev.entry();
        assert!(synth.index() >= 3); // block 3 = synthetic
        let synth_succs: Vec<BlockId> = rev.successors(synth).collect();
        assert!(synth_succs.contains(&a));
        assert!(synth_succs.contains(&b));
    }
}
