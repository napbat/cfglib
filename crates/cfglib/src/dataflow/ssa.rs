//! Generic Static Single Assignment (SSA) construction.
//!
//! The SSA representation is independent of an instruction set and of the
//! concrete instruction type stored in [`Cfg`]. An [`InstrInfo`] adapter
//! supplies its native variable identity, and [`build_ssa`] produces renamed
//! definitions, uses, and phi operands keyed by [`ProgramPoint`]. The original
//! instructions remain untouched and can be recovered from the source CFG.

extern crate alloc;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::dataflow::{InstrInfo, ProgramPoint, VariableId};
use crate::graph::dominator::{DominatorChildOrder, DominatorTree};
use crate::graph::search::EpochMarks;

/// The dominance frontier of every block.
#[derive(Debug, Clone)]
pub struct DominanceFrontiers {
    /// `frontiers[b]` is the dominance-frontier set of `b`.
    frontiers: Vec<BTreeSet<BlockId>>,
}

impl DominanceFrontiers {
    /// Compute dominance frontiers using the algorithm from Cooper, Harvey,
    /// and Kennedy.
    #[must_use]
    pub fn compute<I>(cfg: &Cfg<I>, dom: &DominatorTree) -> Self {
        let mut frontiers = vec![BTreeSet::new(); cfg.num_blocks()];

        for block in cfg.blocks() {
            if cfg.predecessor_edges(block.id()).len() < 2 {
                continue;
            }

            let frontier_root = dom.idom(block.id()).unwrap_or(block.id());
            for predecessor in cfg.predecessors(block.id()) {
                let mut runner = predecessor;
                while runner != frontier_root {
                    frontiers[runner.index()].insert(block.id());
                    let Some(parent) = dom.idom(runner) else {
                        break;
                    };
                    runner = parent;
                }
            }
        }

        Self { frontiers }
    }

    /// Return the dominance-frontier set for `block`.
    #[must_use]
    pub fn frontier(&self, block: BlockId) -> &BTreeSet<BlockId> {
        &self.frontiers[block.index()]
    }
}

/// A structural phi placement before SSA values are renamed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhiPlacement<V> {
    /// The source-IR variable merged by the phi.
    pub variable: V,
    /// CFG predecessors that contribute operands, in CFG predecessor order.
    pub predecessors: Vec<BlockId>,
}

/// Phi placements indexed by containing block.
#[derive(Debug, Clone)]
pub struct PhiPlacements<V> {
    placements: Vec<Vec<PhiPlacement<V>>>,
}

impl<V> PhiPlacements<V> {
    /// Return phi placements at `block`.
    #[must_use]
    pub fn at(&self, block: BlockId) -> &[PhiPlacement<V>] {
        &self.placements[block.index()]
    }

    /// Return the total number of placed phis.
    #[must_use]
    pub fn len(&self) -> usize {
        self.placements.iter().map(Vec::len).sum()
    }

    /// Return whether no phis were placed.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Iterate over all `(block, placement)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (BlockId, &PhiPlacement<V>)> {
        self.placements
            .iter()
            .enumerate()
            .flat_map(|(index, phis)| {
                phis.iter()
                    .map(move |phi| (BlockId::from_index(index), phi))
            })
    }
}

/// Place phis for every variable defined in the CFG.
///
/// This is the iterated-dominance-frontier phase of SSA construction. Use
/// [`build_ssa`] when renamed definitions and operands are also required
/// (and see its precondition: the entry block must not be a branch
/// target).
#[must_use]
pub fn place_phis<I: InstrInfo>(cfg: &Cfg<I>, dom: &DominatorTree) -> PhiPlacements<I::Variable> {
    let frontiers = DominanceFrontiers::compute(cfg, dom);
    let mut definition_blocks: BTreeMap<I::Variable, Vec<BlockId>> = BTreeMap::new();

    for block in cfg.blocks() {
        for instruction in block.instructions() {
            for variable in instruction.defs() {
                let blocks = definition_blocks.entry(variable.clone()).or_default();
                if blocks.last().copied() != Some(block.id()) {
                    blocks.push(block.id());
                }
            }
        }
    }

    let mut placements = vec![Vec::new(); cfg.num_blocks()];
    let mut has_phi = EpochMarks::new(cfg.num_blocks());
    let mut visited = EpochMarks::new(cfg.num_blocks());
    for (variable, definitions) in definition_blocks {
        has_phi.reset();
        visited.reset();
        for &block in &definitions {
            visited.mark(block.index());
        }
        let mut worklist = definitions;

        while let Some(block) = worklist.pop() {
            for &frontier_block in frontiers.frontier(block) {
                if has_phi.is_marked(frontier_block.index()) {
                    continue;
                }
                has_phi.mark(frontier_block.index());

                placements[frontier_block.index()].push(PhiPlacement {
                    variable: variable.clone(),
                    predecessors: cfg.predecessors(frontier_block).collect(),
                });
                if !visited.is_marked(frontier_block.index()) {
                    visited.mark(frontier_block.index());
                    worklist.push(frontier_block);
                }
            }
        }
    }

    PhiPlacements { placements }
}

/// A per-variable SSA version number.
pub type SsaVersion = usize;

/// A source-IR variable qualified by an SSA version.
///
/// Version `0` represents the value entering a dominator-tree root before any
/// definition in that root's region. Positive versions are produced by phis
/// and instruction definitions.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SsaValue<V> {
    /// Source-IR variable identity.
    pub variable: V,
    /// SSA version of the variable.
    pub version: SsaVersion,
}

impl<V> SsaValue<V> {
    /// Create an SSA value with an explicit version.
    #[must_use]
    pub const fn new(variable: V, version: SsaVersion) -> Self {
        Self { variable, version }
    }

    /// Create the version-zero live-in value for `variable`.
    #[must_use]
    pub const fn live_in(variable: V) -> Self {
        Self::new(variable, 0)
    }

    /// Return whether this is a version-zero live-in value.
    #[must_use]
    pub const fn is_live_in(&self) -> bool {
        self.version == 0
    }
}

/// A fully renamed SSA phi.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaPhi<V> {
    /// SSA value defined by the phi.
    pub result: SsaValue<V>,
    /// Incoming SSA value for each CFG predecessor.
    pub operands: Vec<(BlockId, SsaValue<V>)>,
}

/// SSA annotations for one source instruction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaInstruction<V> {
    /// Position of the original instruction in the source CFG.
    pub point: ProgramPoint,
    /// Renamed operands, preserving the adapter's use order.
    pub uses: Vec<SsaValue<V>>,
    /// Fresh definitions, preserving the adapter's definition order.
    pub defs: Vec<SsaValue<V>>,
}

/// SSA contents associated with one CFG block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaBlock<V> {
    /// Source CFG block.
    pub block: BlockId,
    /// Renamed phis at the start of the block.
    pub phis: Vec<SsaPhi<V>>,
    /// Renamed instructions in source instruction order.
    pub instructions: Vec<SsaInstruction<V>>,
}

/// An IR-neutral, renamed SSA view of a CFG.
///
/// This type deliberately stores no instruction payload. Each
/// [`SsaInstruction`] carries a [`ProgramPoint`] that maps back to the native
/// instruction in the CFG used to build the form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SsaForm<V> {
    blocks: Vec<SsaBlock<V>>,
    max_versions: BTreeMap<V, SsaVersion>,
}

impl<V: VariableId> SsaForm<V> {
    /// Return all SSA blocks in source CFG order.
    #[must_use]
    pub fn blocks(&self) -> &[SsaBlock<V>] {
        &self.blocks
    }

    /// Return the SSA block corresponding to `block`.
    #[must_use]
    pub fn block(&self, block: BlockId) -> &SsaBlock<V> {
        &self.blocks[block.index()]
    }

    /// Return SSA annotations for a source program point, if it exists.
    #[must_use]
    pub fn instruction(&self, point: ProgramPoint) -> Option<&SsaInstruction<V>> {
        self.blocks
            .get(point.block.index())?
            .instructions
            .get(point.inst_idx)
    }

    /// Iterate over all `(block, phi)` pairs.
    pub fn phis(&self) -> impl Iterator<Item = (BlockId, &SsaPhi<V>)> {
        self.blocks
            .iter()
            .flat_map(|block| block.phis.iter().map(move |phi| (block.block, phi)))
    }

    /// Return the greatest assigned version for `variable`.
    ///
    /// A result of `0` means the variable only occurs as a live-in value or
    /// does not occur in the form.
    #[must_use]
    pub fn max_version(&self, variable: &V) -> SsaVersion {
        self.max_versions.get(variable).copied().unwrap_or(0)
    }
}

#[derive(Debug)]
struct PhiDraft<V> {
    variable: V,
    predecessors: Vec<BlockId>,
    result: Option<SsaValue<V>>,
    operands: BTreeMap<BlockId, SsaValue<V>>,
}

#[derive(Debug)]
struct BlockDraft<V> {
    phis: Vec<PhiDraft<V>>,
    instructions: Vec<SsaInstruction<V>>,
}

enum RenameEvent<V> {
    Enter(BlockId),
    Exit(Vec<V>),
}

fn current_value<V: VariableId>(
    variable: &V,
    stacks: &BTreeMap<V, Vec<SsaValue<V>>>,
) -> SsaValue<V> {
    stacks
        .get(variable)
        .and_then(|stack| stack.last())
        .cloned()
        .unwrap_or_else(|| SsaValue::live_in(variable.clone()))
}

fn fresh_value<V: VariableId>(
    variable: &V,
    max_versions: &mut BTreeMap<V, SsaVersion>,
) -> SsaValue<V> {
    let version = max_versions.entry(variable.clone()).or_default();
    *version += 1;
    SsaValue::new(variable.clone(), *version)
}

fn create_drafts<I: InstrInfo>(
    cfg: &Cfg<I>,
    placements: &PhiPlacements<I::Variable>,
) -> Vec<BlockDraft<I::Variable>> {
    cfg.blocks()
        .iter()
        .map(|block| BlockDraft {
            phis: placements
                .at(block.id())
                .iter()
                .map(|placement| PhiDraft {
                    variable: placement.variable.clone(),
                    predecessors: placement.predecessors.clone(),
                    result: None,
                    operands: BTreeMap::new(),
                })
                .collect(),
            instructions: Vec::with_capacity(block.instructions().len()),
        })
        .collect()
}

fn rename_block<I: InstrInfo>(
    cfg: &Cfg<I>,
    block: BlockId,
    drafts: &mut [BlockDraft<I::Variable>],
    stacks: &mut BTreeMap<I::Variable, Vec<SsaValue<I::Variable>>>,
    max_versions: &mut BTreeMap<I::Variable, SsaVersion>,
) -> Vec<I::Variable> {
    let mut pushed_variables = Vec::new();
    for phi in &mut drafts[block.index()].phis {
        let result = fresh_value(&phi.variable, max_versions);
        phi.result = Some(result.clone());
        stacks.entry(phi.variable.clone()).or_default().push(result);
        pushed_variables.push(phi.variable.clone());
    }

    for (inst_idx, instruction) in cfg.block(block).instructions().iter().enumerate() {
        let uses = instruction
            .uses()
            .iter()
            .map(|variable| current_value(variable, stacks))
            .collect();
        let mut defs = Vec::with_capacity(instruction.defs().len());
        for variable in instruction.defs() {
            let value = fresh_value(variable, max_versions);
            stacks
                .entry(variable.clone())
                .or_default()
                .push(value.clone());
            pushed_variables.push(variable.clone());
            defs.push(value);
        }
        drafts[block.index()].instructions.push(SsaInstruction {
            point: ProgramPoint { block, inst_idx },
            uses,
            defs,
        });
    }

    for successor in cfg.successors(block) {
        let incoming: Vec<_> = drafts[successor.index()]
            .phis
            .iter()
            .map(|phi| current_value(&phi.variable, stacks))
            .collect();
        for (phi, value) in drafts[successor.index()].phis.iter_mut().zip(incoming) {
            phi.operands.insert(block, value);
        }
    }
    pushed_variables
}

fn rename_drafts<I: InstrInfo>(
    cfg: &Cfg<I>,
    dom: &DominatorTree,
    drafts: &mut [BlockDraft<I::Variable>],
    max_versions: &mut BTreeMap<I::Variable, SsaVersion>,
) {
    let mut stacks = BTreeMap::new();
    let mut visited = alloc::vec![false; cfg.num_blocks()];
    // The event stack consumes siblings in reverse, so descending links
    // preserve `DominatorTree::children`'s ascending DFS visitation.
    let children = dom.child_links(DominatorChildOrder::Descending);
    let mut roots = vec![cfg.entry()];
    roots.extend(
        cfg.blocks()
            .iter()
            .map(crate::block::BasicBlock::id)
            .filter(|&block| block != cfg.entry() && dom.idom(block).is_none()),
    );

    for root in roots {
        let mut events = vec![RenameEvent::Enter(root)];
        while let Some(event) = events.pop() {
            match event {
                RenameEvent::Enter(block) if !visited[block.index()] => {
                    visited[block.index()] = true;
                    let pushed = rename_block(cfg, block, drafts, &mut stacks, max_versions);
                    events.push(RenameEvent::Exit(pushed));
                    let mut child = children.first_child(block);
                    while let Some(next) = child {
                        events.push(RenameEvent::Enter(next));
                        child = children.next_sibling(next);
                    }
                }
                RenameEvent::Enter(_) => {}
                RenameEvent::Exit(pushed_variables) => {
                    for variable in pushed_variables.into_iter().rev() {
                        if let Some(stack) = stacks.get_mut(&variable) {
                            stack.pop();
                        }
                    }
                }
            }
        }
    }
}

fn finish_blocks<V: VariableId>(
    drafts: Vec<BlockDraft<V>>,
    max_versions: &mut BTreeMap<V, SsaVersion>,
) -> Vec<SsaBlock<V>> {
    drafts
        .into_iter()
        .enumerate()
        .map(|(index, draft)| {
            let phis = draft
                .phis
                .into_iter()
                .map(|phi| {
                    let PhiDraft {
                        variable,
                        predecessors,
                        result,
                        operands,
                    } = phi;
                    let result = result.unwrap_or_else(|| fresh_value(&variable, max_versions));
                    let operands = predecessors
                        .into_iter()
                        .map(|predecessor| {
                            let value = operands
                                .get(&predecessor)
                                .cloned()
                                .unwrap_or_else(|| SsaValue::live_in(variable.clone()));
                            (predecessor, value)
                        })
                        .collect();
                    SsaPhi { result, operands }
                })
                .collect();
            SsaBlock {
                block: BlockId::from_index(index),
                phis,
                instructions: draft.instructions,
            }
        })
        .collect()
}

/// Build a fully renamed SSA view of `cfg`.
///
/// The algorithm performs phi placement followed by classic dominator-tree
/// renaming. It is iterative rather than recursive, so deeply nested control
/// flow does not consume the host call stack. Variables read before any
/// dominating definition receive version `0`.
///
/// # Precondition
///
/// The entry block must not be a branch target. Phi operands come from
/// predecessor edges, so a phi placed AT the entry (entry doubling as a
/// loop header) has no operand for the version-`0` live-in value and the
/// value entering the function is dropped from the web. Every builder in
/// this workspace guarantees the property; direct constructions that
/// branch to the entry should canonicalize first
/// ([`insert_preheader`](crate::insert_preheader) /
/// [`split_block`](crate::Cfg::split_block)).
#[must_use]
pub fn build_ssa<I: InstrInfo>(cfg: &Cfg<I>, dom: &DominatorTree) -> SsaForm<I::Variable> {
    let placements = place_phis(cfg, dom);
    let mut drafts = create_drafts(cfg, &placements);
    let mut max_versions = BTreeMap::new();
    rename_drafts(cfg, dom, &mut drafts, &mut max_versions);
    let blocks = finish_blocks(drafts, &mut max_versions);
    SsaForm {
        blocks,
        max_versions,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builder::CfgBuilder;
    use crate::edge::EdgeKind;
    use crate::test_util::{DfInst, df_def, df_use};
    use alloc::vec;

    #[test]
    fn no_phis_in_linear_cfg() {
        let cfg = CfgBuilder::build(vec![df_def("def r0", 0), df_use("use r0", 0)]).unwrap();
        let dom = DominatorTree::compute(&cfg);
        assert!(place_phis(&cfg, &dom).is_empty());
    }

    #[test]
    fn phi_at_merge_point() {
        let mut cfg = Cfg::<DfInst>::new();
        let then_block = cfg.new_block();
        let else_block = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), else_block, EdgeKind::ConditionalFalse);
        cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
        cfg.add_edge(else_block, merge, EdgeKind::Fallthrough);
        cfg.block_mut(then_block).push(df_def("then", 0));
        cfg.block_mut(else_block).push(df_def("else", 0));
        cfg.block_mut(merge).push(df_use("use", 0));

        let dom = DominatorTree::compute(&cfg);
        let placements = place_phis(&cfg, &dom);
        assert_eq!(placements.len(), 1);
        assert_eq!(placements.at(merge)[0].variable, 0);
    }

    #[test]
    fn renaming_uses_latest_definition() {
        let mut cfg = Cfg::<DfInst>::new();
        cfg.block_mut(cfg.entry()).instructions_vec_mut().extend([
            df_def("first", 0),
            df_def("second", 0),
            df_use("use", 0),
        ]);

        let dom = DominatorTree::compute(&cfg);
        let ssa = build_ssa(&cfg, &dom);
        let instructions = &ssa.block(cfg.entry()).instructions;
        assert_eq!(instructions[0].defs[0], SsaValue::new(0, 1));
        assert_eq!(instructions[1].defs[0], SsaValue::new(0, 2));
        assert_eq!(instructions[2].uses[0], SsaValue::new(0, 2));
    }

    #[test]
    fn read_before_definition_is_live_in() {
        let cfg = CfgBuilder::build(vec![df_use("use", 7), df_def("def", 7)]).unwrap();
        let dom = DominatorTree::compute(&cfg);
        let ssa = build_ssa(&cfg, &dom);
        assert_eq!(
            ssa.block(cfg.entry()).instructions[0].uses[0],
            SsaValue::live_in(7)
        );
    }

    #[test]
    fn diamond_phi_has_renamed_result_and_operands() {
        let mut cfg = Cfg::<DfInst>::new();
        let left = cfg.new_block();
        let right = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(cfg.entry(), left, EdgeKind::ConditionalTrue);
        cfg.add_edge(cfg.entry(), right, EdgeKind::ConditionalFalse);
        cfg.add_edge(left, merge, EdgeKind::Fallthrough);
        cfg.add_edge(right, merge, EdgeKind::Fallthrough);
        cfg.block_mut(left).push(df_def("left", 0));
        cfg.block_mut(right).push(df_def("right", 0));
        cfg.block_mut(merge).push(df_use("merged", 0));

        let dom = DominatorTree::compute(&cfg);
        let ssa = build_ssa(&cfg, &dom);
        let phi = &ssa.block(merge).phis[0];
        assert_eq!(phi.result.variable, 0);
        assert_ne!(phi.result.version, 0);
        assert_eq!(phi.operands.len(), 2);
        assert!(phi.operands.iter().all(|(_, value)| value.version != 0));
        assert_ne!(phi.operands[0].1, phi.operands[1].1);
        assert_eq!(ssa.block(merge).instructions[0].uses[0], phi.result);
    }

    #[test]
    fn unreachable_block_is_still_annotated() {
        let mut cfg = Cfg::<DfInst>::new();
        let unreachable = cfg.new_block();
        cfg.block_mut(unreachable).push(df_def("dead", 3));
        let dom = DominatorTree::compute(&cfg);
        let ssa = build_ssa(&cfg, &dom);
        assert_eq!(ssa.block(unreachable).instructions.len(), 1);
        assert_eq!(ssa.block(unreachable).instructions[0].defs[0].variable, 3);
    }
}
