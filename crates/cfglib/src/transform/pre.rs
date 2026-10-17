//! Elimination of fully redundant computations — the dominated subset of
//! partial redundancy elimination (PRE).
//!
//! An instruction is redundant when an instruction that dominates it already
//! computed the same value, which [`ValueNumbering`] decides. A computation
//! that is available on only some paths into a block (partially redundant)
//! is left alone: removing it would need new computations on the other
//! paths.

extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::ProgramPoint;
use crate::analysis::value_numbering::{ValueNumber, ValueNumberInfo, ValueNumbering};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::copy_propagation::CopySource;
use crate::graph::dominator::DominatorTree;

/// The instructions whose value a dominating instruction already computed.
#[derive(Debug, Clone)]
pub struct PreAnalysis {
    /// `(block, instruction index, value number)` of every redundant
    /// instruction, ascending by block and then instruction. The value
    /// number is the one the dominating computation gave the value.
    pub redundant: Vec<(BlockId, usize, ValueNumber)>,
}

impl PreAnalysis {
    /// Find the redundant instructions of `cfg` by global value numbering
    /// over `dom`.
    #[must_use]
    pub fn compute<G>(cfg: &G, dom: &DominatorTree) -> Self
    where
        G: CfgView,
        G::Inst: ValueNumberInfo,
    {
        Self::from_numbering(&ValueNumbering::compute(cfg, dom))
    }

    fn from_numbering(numbering: &ValueNumbering) -> Self {
        let redundant = numbering
            .blocks
            .iter()
            .flat_map(|(&block, numbers)| {
                numbers.redundant.iter().map(move |&index| {
                    let value =
                        numbers.inst_vn[index].expect("a redundant instruction is value-numbered");
                    (block, index, value)
                })
            })
            .collect();
        Self { redundant }
    }
}

/// Remove the redundant instructions that can be replaced by the dominating
/// computation, and make every read of their definitions read the dominating
/// instruction's definitions instead.
///
/// The replacement is sound for single-assignment definitions, so a
/// redundant instruction is removed only when each of its definitions and
/// the matching definition of the dominating instruction has no other
/// definition in the graph, and every read of its definitions is dominated
/// by it. An instruction that does not qualify stays in place. The reads are
/// rewritten through [`CopySource::rewrite_use`].
///
/// Returns the number of instructions removed.
pub fn eliminate_pre<I, E>(cfg: &mut Cfg<I, E>, dom: &DominatorTree) -> usize
where
    I: ValueNumberInfo + CopySource,
{
    let numbering = ValueNumbering::compute(cfg, dom);
    let analysis = PreAnalysis::from_numbering(&numbering);
    if analysis.redundant.is_empty() {
        return 0;
    }

    // Every value is first computed by exactly one non-redundant instruction.
    let mut leaders: BTreeMap<ValueNumber, ProgramPoint> = BTreeMap::new();
    for (&block, numbers) in &numbering.blocks {
        let mut redundant = numbers.redundant.iter().peekable();
        for (inst_idx, &value) in numbers.inst_vn.iter().enumerate() {
            if redundant.next_if_eq(&&inst_idx).is_none()
                && let Some(value) = value
            {
                leaders.insert(value, ProgramPoint { block, inst_idx });
            }
        }
    }

    let mut definition_count: BTreeMap<I::Variable, usize> = BTreeMap::new();
    let mut readers: BTreeMap<I::Variable, Vec<ProgramPoint>> = BTreeMap::new();
    for block in cfg.block_ids() {
        for (inst_idx, inst) in cfg.block(block).instructions().iter().enumerate() {
            for variable in inst.defs() {
                *definition_count.entry(variable.clone()).or_default() += 1;
            }
            for variable in inst.uses() {
                readers
                    .entry(variable.clone())
                    .or_default()
                    .push(ProgramPoint { block, inst_idx });
            }
        }
    }

    let mut rewrites: Vec<(ProgramPoint, I::Variable, I::Variable)> = Vec::new();
    let mut removals: Vec<ProgramPoint> = Vec::new();
    for &(block, inst_idx, value) in &analysis.redundant {
        let point = ProgramPoint { block, inst_idx };
        let Some(leader) = leaders.get(&value) else {
            continue;
        };
        let duplicate = cfg.block(block).instructions()[inst_idx].defs();
        let original = cfg.block(leader.block).instructions()[leader.inst_idx].defs();
        let replaceable = duplicate.len() == original.len()
            && duplicate.iter().zip(original).all(|(old, new)| {
                old != new
                    && definition_count[old] == 1
                    && definition_count[new] == 1
                    && readers.get(old).is_none_or(|sites| {
                        sites.iter().all(|&site| dominates_read(dom, point, site))
                    })
            });
        if !replaceable {
            continue;
        }
        for (old, new) in duplicate.iter().zip(original) {
            for &reader in readers.get(old).into_iter().flatten() {
                rewrites.push((reader, old.clone(), new.clone()));
            }
        }
        removals.push(point);
    }

    for (reader, old, new) in rewrites {
        cfg.block_mut(reader.block).instructions_mut()[reader.inst_idx].rewrite_use(&old, &new);
    }
    // `removals` ascends by block and index; remove from the back so the
    // indices still to be removed stay valid.
    for point in removals.iter().rev() {
        cfg.block_mut(point.block)
            .instructions_mut()
            .remove(point.inst_idx);
    }
    removals.len()
}

/// Whether the instruction at `reader` runs only after the one at
/// `definition` has run.
fn dominates_read(dom: &DominatorTree, definition: ProgramPoint, reader: ProgramPoint) -> bool {
    if definition.block == reader.block {
        definition.inst_idx < reader.inst_idx
    } else {
        dom.strictly_dominates(definition.block, reader.block)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::InstrInfo;
    use crate::block::BasicBlock;
    use crate::cfg::Cfg;
    use crate::edge::EdgeKind;
    use crate::graph::dominator::DominatorTree;
    use crate::test_util::{VnInst, vn_inst as pi};

    #[test]
    fn pre_detects_within_block_redundancy() {
        // Same expression computed twice within one block.
        let mut cfg: Cfg<VnInst> = Cfg::new();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .extend([pi(1, &[0, 1], &[2]), pi(1, &[0, 1], &[3])]);
        let dom = DominatorTree::compute(&cfg);
        let result = PreAnalysis::compute(&cfg, &dom);
        assert_eq!(result.redundant.len(), 1);
    }

    #[test]
    fn pre_detects_cross_block_redundancy() {
        // Block 0 computes op1(loc0, loc1), block 1 computes the same.
        // Block 0 dominates block 1, so the second is redundant.
        let mut cfg: Cfg<VnInst> = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(pi(1, &[0, 1], &[2]));
        cfg.block_mut(b)
            .instructions_mut()
            .push(pi(1, &[0, 1], &[3]));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let dom = DominatorTree::compute(&cfg);
        let result = PreAnalysis::compute(&cfg, &dom);
        assert_eq!(result.redundant.len(), 1);
        assert_eq!(result.redundant[0].0, b);
    }

    #[test]
    fn pre_no_redundancy() {
        let mut cfg: Cfg<VnInst> = Cfg::new();
        let b = cfg.new_block();
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(pi(1, &[0, 1], &[2]));
        cfg.block_mut(b)
            .instructions_mut()
            .push(pi(2, &[0, 1], &[3]));
        cfg.add_edge(cfg.entry(), b, EdgeKind::Fallthrough);
        let dom = DominatorTree::compute(&cfg);
        let result = PreAnalysis::compute(&cfg, &dom);
        assert_eq!(result.redundant.len(), 0);
    }

    /// `defs = name(uses)`; every instruction is pure.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct Op {
        name: &'static str,
        uses: Vec<u16>,
        defs: Vec<u16>,
    }

    fn op(name: &'static str, uses: &[u16], defs: &[u16]) -> Op {
        Op {
            name,
            uses: uses.to_vec(),
            defs: defs.to_vec(),
        }
    }

    impl InstrInfo for Op {
        type Variable = u16;

        fn uses(&self) -> &[u16] {
            &self.uses
        }
        fn defs(&self) -> &[u16] {
            &self.defs
        }
    }

    impl ValueNumberInfo for Op {
        type Operator = &'static str;

        fn operator(&self) -> &'static str {
            self.name
        }
        fn is_pure(&self) -> bool {
            true
        }
    }

    impl CopySource for Op {
        fn as_copy(&self) -> Option<(u16, u16)> {
            None
        }
        fn rewrite_use(&mut self, old: &u16, new: &u16) {
            for used in &mut self.uses {
                if used == old {
                    *used = *new;
                }
            }
        }
    }

    /// Panics when an instruction reads a variable that nothing defines and
    /// that is not one of the `inputs` the function starts with.
    fn assert_every_read_is_defined<E>(cfg: &Cfg<Op, E>, inputs: &[u16]) {
        let defined: Vec<u16> = cfg
            .blocks()
            .flat_map(BasicBlock::instructions)
            .flat_map(|inst| inst.defs.iter().copied())
            .chain(inputs.iter().copied())
            .collect();
        let undefined: Vec<u16> = cfg
            .blocks()
            .flat_map(BasicBlock::instructions)
            .flat_map(|inst| inst.uses.iter().copied())
            .filter(|used| !defined.contains(used))
            .collect();
        assert!(undefined.is_empty(), "undefined reads: {undefined:?}");
    }

    #[test]
    fn a_removed_instruction_leaves_its_readers_on_the_dominating_definition() {
        let mut cfg: Cfg<Op> = Cfg::new();
        cfg.block_mut(cfg.entry()).instructions_mut().extend([
            op("add", &[0, 1], &[2]),
            op("add", &[0, 1], &[3]),
            op("use", &[3], &[]),
        ]);
        let dom = DominatorTree::compute(&cfg);

        assert_eq!(eliminate_pre(&mut cfg, &dom), 1);

        assert_eq!(
            cfg.block(cfg.entry()).instructions(),
            [op("add", &[0, 1], &[2]), op("use", &[2], &[])]
        );
        assert_every_read_is_defined(&cfg, &[0, 1]);
    }

    #[test]
    fn readers_in_dominated_blocks_follow_the_dominating_definition() {
        let mut cfg: Cfg<Op> = Cfg::new();
        let next = cfg.new_block();
        let last = cfg.new_block();
        cfg.add_edge(cfg.entry(), next, EdgeKind::Fallthrough);
        cfg.add_edge(next, last, EdgeKind::Fallthrough);
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(op("add", &[0, 1], &[2]));
        cfg.block_mut(next)
            .instructions_mut()
            .push(op("add", &[0, 1], &[3]));
        cfg.block_mut(last)
            .instructions_mut()
            .push(op("use", &[3], &[]));
        let dom = DominatorTree::compute(&cfg);

        assert_eq!(eliminate_pre(&mut cfg, &dom), 1);

        assert_eq!(cfg.block(next).instructions(), []);
        assert_eq!(cfg.block(last).instructions(), [op("use", &[2], &[])]);
        assert_every_read_is_defined(&cfg, &[0, 1]);
    }

    #[test]
    fn an_instruction_that_cannot_be_replaced_soundly_stays() {
        // The dominating `r2` is overwritten before the duplicate runs, so
        // no variable holds the value any more.
        let mut overwritten: Cfg<Op> = Cfg::new();
        overwritten
            .block_mut(overwritten.entry())
            .instructions_mut()
            .extend([
                op("add", &[0, 1], &[2]),
                op("other", &[], &[2]),
                op("add", &[0, 1], &[3]),
                op("use", &[3], &[]),
            ]);
        // `r3` is also assigned elsewhere, so its reads are not all the
        // duplicate's.
        let mut reassigned: Cfg<Op> = Cfg::new();
        reassigned
            .block_mut(reassigned.entry())
            .instructions_mut()
            .extend([
                op("add", &[0, 1], &[2]),
                op("add", &[0, 1], &[3]),
                op("other", &[], &[3]),
                op("use", &[3], &[]),
            ]);
        // The duplicate sits on one arm, so the read after the join is not
        // dominated by it.
        let mut one_arm: Cfg<Op> = Cfg::new();
        let arm = one_arm.new_block();
        let join = one_arm.new_block();
        one_arm.add_edge(one_arm.entry(), arm, EdgeKind::ConditionalTrue);
        one_arm.add_edge(one_arm.entry(), join, EdgeKind::ConditionalFalse);
        one_arm.add_edge(arm, join, EdgeKind::Fallthrough);
        one_arm
            .block_mut(one_arm.entry())
            .instructions_mut()
            .push(op("add", &[0, 1], &[2]));
        one_arm
            .block_mut(arm)
            .instructions_mut()
            .push(op("add", &[0, 1], &[3]));
        one_arm
            .block_mut(join)
            .instructions_mut()
            .push(op("use", &[3], &[]));

        for mut cfg in [overwritten, reassigned, one_arm] {
            let before: Vec<Vec<Op>> = cfg
                .blocks()
                .map(|block| block.instructions().to_vec())
                .collect();
            let dom = DominatorTree::compute(&cfg);
            assert_eq!(eliminate_pre(&mut cfg, &dom), 0);
            let after: Vec<Vec<Op>> = cfg
                .blocks()
                .map(|block| block.instructions().to_vec())
                .collect();
            assert_eq!(after, before);
        }
    }

    #[test]
    fn elimination_works_on_graphs_with_edge_payloads_and_converges() {
        let mut cfg = Cfg::<Op, &str>::with_edge_payload();
        let next = cfg.new_block();
        cfg.add_edge_with_payload(cfg.entry(), next, EdgeKind::Fallthrough, "enter");
        cfg.block_mut(cfg.entry())
            .instructions_mut()
            .push(op("add", &[0, 1], &[2]));
        cfg.block_mut(next)
            .instructions_mut()
            .extend([op("add", &[0, 1], &[3]), op("use", &[3], &[])]);

        let dom = DominatorTree::compute(&cfg);
        assert_eq!(eliminate_pre(&mut cfg, &dom), 1);
        assert_every_read_is_defined(&cfg, &[0, 1]);
        assert_eq!(eliminate_pre(&mut cfg, &dom), 0);
    }
}
