//! Instruction-level analyses over a `Cfg` with removed blocks, checked
//! against reference algorithms and across a compaction.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{
    BlockId, Cfg, CfgMetrics, DominatorTree, Liveness, ProgramPoint, ReachingDefs, SsaForm,
    SsaValue, SsaVersion,
};

use crate::oracle::Reference;
use crate::topology::{GENERAL, Op, Realizations, SEEDS, SSA_SAFE, Slots, Topology};

fn realize(seed: u64, shape: crate::topology::Shape) -> Realizations {
    Realizations::of_topology(Topology::random(seed, shape)).labeled(format!("seed {seed}"))
}

fn liveness_matches(cfg: &Cfg<Op>, slots: &Slots<BlockId>, reference: &Reference<'_>, label: &str) {
    let liveness = Liveness::compute(cfg);
    let (live_in, live_out) = reference.liveness();
    for block in reference.live() {
        let id = slots.id(block);
        assert_eq!(
            *liveness.live_in(id),
            live_in[block],
            "{label}: live_in({block})"
        );
        assert_eq!(
            *liveness.live_out(id),
            live_out[block],
            "{label}: live_out({block})"
        );
    }
}

#[test]
fn liveness_matches_a_fixed_point_over_the_live_blocks() {
    for seed in SEEDS {
        realize(seed, GENERAL).run_cfgs(liveness_matches);
    }
}

fn reaching_definitions_match(
    cfg: &Cfg<Op>,
    slots: &Slots<BlockId>,
    reference: &Reference<'_>,
    label: &str,
) {
    let reaching = ReachingDefs::compute(cfg);
    for block in reference.live() {
        let found: BTreeSet<(u8, usize, usize)> = reaching
            .reaching_in(slots.id(block))
            .iter()
            .map(|definition| {
                (
                    definition.variable,
                    slots.slot_of(definition.site.block),
                    definition.site.inst_idx,
                )
            })
            .collect();
        assert_eq!(
            found,
            reference.reaching_in(block),
            "{label}: reaching_in({block})"
        );
    }
}

#[test]
fn reaching_definitions_match_path_search_from_each_definition() {
    for seed in SEEDS {
        realize(seed, GENERAL).run_cfgs(reaching_definitions_match);
    }
}

fn cfg_metrics_match(
    cfg: &Cfg<Op>,
    _slots: &Slots<BlockId>,
    reference: &Reference<'_>,
    label: &str,
) {
    let metrics = CfgMetrics::compute(cfg);
    let expected = reference.metrics();
    assert_eq!(
        metrics.graph.node_count, expected.node_count,
        "{label}: node_count"
    );
    assert_eq!(
        metrics.graph.node_count,
        cfg.block_count(),
        "{label}: node_count is block_count"
    );
    assert_eq!(
        metrics.graph.edge_count, expected.edge_count,
        "{label}: edge_count"
    );
    assert_eq!(
        metrics.graph.edge_count,
        cfg.edge_count(),
        "{label}: edge_count is edge_count"
    );
    assert_eq!(
        metrics.graph.reachable_node_count, expected.reachable_node_count,
        "{label}: reachable_node_count"
    );
    assert_eq!(
        metrics.graph.unreachable_node_count, expected.unreachable_node_count,
        "{label}: unreachable_node_count"
    );
    assert_eq!(
        metrics.graph.cyclomatic_complexity, expected.cyclomatic_complexity,
        "{label}: cyclomatic_complexity"
    );
    assert_eq!(
        metrics.graph.exit_count, expected.exit_count,
        "{label}: exit_count"
    );
    assert_eq!(
        metrics.instruction_count, expected.instruction_count,
        "{label}: instruction_count"
    );
    let instructions = f64::from(u32::try_from(expected.instruction_count).expect("few"));
    let blocks = f64::from(u32::try_from(expected.node_count).expect("few"));
    assert!(
        (metrics.avg_instructions_per_block - instructions / blocks).abs() < f64::EPSILON,
        "{label}: the average divides by the live blocks it reports"
    );
}

#[test]
fn cfg_metrics_count_live_blocks_and_instructions() {
    for seed in SEEDS {
        realize(seed, GENERAL).run_cfgs(cfg_metrics_match);
    }
}

/// Where an SSA version is defined: an instruction, or a phi at a block start.
#[derive(Debug, Clone, Copy)]
enum Definition {
    Instruction(ProgramPoint),
    Phi { block: BlockId, index: usize },
}

fn definitions(ssa: &SsaForm<u8>) -> BTreeMap<(u8, SsaVersion), Definition> {
    let mut found = BTreeMap::new();
    let mut record = |value: &SsaValue<u8>, definition: Definition| {
        assert!(
            value.version >= 1,
            "{value:?}: a definition has a positive version"
        );
        let key = (value.variable, value.version);
        assert!(
            found.insert(key, definition).is_none(),
            "{value:?} is defined twice"
        );
    };
    for block in ssa.blocks() {
        for (index, phi) in block.phis.iter().enumerate() {
            record(
                &phi.result,
                Definition::Phi {
                    block: block.block,
                    index,
                },
            );
        }
        for instruction in &block.instructions {
            for value in instruction.defs() {
                record(value, Definition::Instruction(instruction.point));
            }
        }
    }
    found
}

/// The instruction sites that can supply `value`, and whether the value
/// entering the function can: version zero, through any chain of phis.
fn supplying_sites(
    ssa: &SsaForm<u8>,
    defined: &BTreeMap<(u8, SsaVersion), Definition>,
    slots: &Slots<BlockId>,
    value: &SsaValue<u8>,
) -> (BTreeSet<(usize, usize)>, bool) {
    let mut sites = BTreeSet::new();
    let mut entering = false;
    let mut seen = BTreeSet::new();
    let mut pending = vec![value.clone()];
    while let Some(value) = pending.pop() {
        if value.is_live_in() {
            entering = true;
            continue;
        }
        if !seen.insert((value.variable, value.version)) {
            continue;
        }
        match defined[&(value.variable, value.version)] {
            Definition::Instruction(point) => {
                sites.insert((slots.slot_of(point.block), point.inst_idx));
            }
            Definition::Phi { block, index } => {
                let phi = &ssa.block(block).phis[index];
                pending.extend(phi.operands().iter().map(|(_, operand)| operand.clone()));
            }
        }
    }
    (sites, entering)
}

fn ssa_reads_see_exactly_the_reaching_writes(
    cfg: &Cfg<Op>,
    slots: &Slots<BlockId>,
    reference: &Reference<'_>,
    label: &str,
) {
    let dominators = DominatorTree::compute(cfg);
    let ssa = SsaForm::compute(cfg, &dominators);
    assert_eq!(
        ssa.blocks().len(),
        cfg.block_bound(),
        "{label}: one SSA block per slot"
    );
    for index in 0..cfg.block_bound() {
        if !cfg.contains_block(BlockId::from_index(index)) {
            let retired = &ssa.blocks()[index];
            assert!(
                retired.phis.is_empty() && retired.instructions.is_empty(),
                "{label}: the retired slot {index} holds an empty SSA block"
            );
        }
    }

    let defined = definitions(&ssa);
    let mut greatest: BTreeMap<u8, SsaVersion> = BTreeMap::new();
    for &(variable, version) in defined.keys() {
        let entry = greatest.entry(variable).or_insert(0);
        *entry = (*entry).max(version);
    }
    for variable in 0..4_u8 {
        assert_eq!(
            ssa.max_version(&variable),
            greatest.get(&variable).copied().unwrap_or(0),
            "{label}: max_version({variable})"
        );
    }

    let reachable = reference.reachable();
    for &block in &reachable {
        let id = slots.id(block);
        let ssa_block = ssa.block(id);
        for phi in &ssa_block.phis {
            let sources: BTreeSet<usize> = phi
                .operands()
                .iter()
                .map(|(point, _)| slots.slot_of(point.block))
                .collect();
            let predecessors: BTreeSet<usize> =
                reference.predecessors(block).iter().copied().collect();
            assert_eq!(
                sources, predecessors,
                "{label}: phi operands of block {block}"
            );
        }
        assert_eq!(
            ssa_block.instructions.len(),
            reference.instructions(block).len(),
            "{label}: block {block} keeps every instruction"
        );
        for (index, op) in reference.instructions(block).iter().enumerate() {
            let instruction = &ssa_block.instructions[index];
            assert_eq!(
                instruction.point,
                ProgramPoint {
                    block: id,
                    inst_idx: index
                },
                "{label}: the instruction point names its source"
            );
            assert_eq!(
                instruction.uses().len(),
                op.uses.len(),
                "{label}: uses at {block}:{index}"
            );
            for (read, &variable) in instruction.uses().iter().zip(&op.uses) {
                assert_eq!(
                    read.variable, variable,
                    "{label}: use order at {block}:{index}"
                );
                assert_eq!(
                    ssa.value_at(instruction.point, &variable),
                    *read,
                    "{label}: value_at agrees with the renamed use at {block}:{index}"
                );
                let Some((sites, entering)) = reference.reaching_writes(block, index, variable)
                else {
                    continue;
                };
                assert_eq!(
                    supplying_sites(&ssa, &defined, slots, read),
                    (sites, entering),
                    "{label}: writes reaching the read of {variable} at {block}:{index}"
                );
            }
        }
    }
}

#[test]
fn ssa_reads_name_exactly_the_writes_that_reach_them() {
    for seed in SEEDS {
        realize(seed, SSA_SAFE).run_cfgs(ssa_reads_see_exactly_the_reaching_writes);
    }
}

#[test]
fn ssa_numbering_does_not_depend_on_holes() {
    for seed in SEEDS {
        let realized = realize(seed, SSA_SAFE);
        let holed = SsaForm::compute(
            &realized.cfg_holed,
            &DominatorTree::compute(&realized.cfg_holed),
        );
        let compacted = SsaForm::compute(
            &realized.cfg_compacted,
            &DominatorTree::compute(&realized.cfg_compacted),
        );
        let holed_slots = realized.cfg_slots_holed();
        let compacted_slots = realized.cfg_slots_compacted();
        let reference = Reference::new(&realized.topology);
        let operands = |ssa: &SsaForm<u8>, slots: &Slots<BlockId>, block: usize| {
            ssa.block(slots.id(block))
                .phis
                .iter()
                .map(|phi| {
                    let sources: Vec<(usize, usize, SsaValue<u8>)> = phi
                        .operands()
                        .iter()
                        .map(|(point, value)| {
                            (slots.slot_of(point.block), point.inst_idx, value.clone())
                        })
                        .collect();
                    (phi.result.clone(), sources)
                })
                .collect::<Vec<_>>()
        };
        for block in reference.live() {
            assert_eq!(
                operands(&holed, &holed_slots, block),
                operands(&compacted, &compacted_slots, block),
                "seed {seed}: phis of block {block}"
            );
            let (before, after) = (
                holed.block(holed_slots.id(block)),
                compacted.block(compacted_slots.id(block)),
            );
            for (left, right) in before.instructions.iter().zip(&after.instructions) {
                assert_eq!(
                    left.uses(),
                    right.uses(),
                    "seed {seed}: uses in block {block}"
                );
                assert_eq!(
                    left.defs(),
                    right.defs(),
                    "seed {seed}: defs in block {block}"
                );
            }
        }
        for variable in 0..4_u8 {
            assert_eq!(
                holed.max_version(&variable),
                compacted.max_version(&variable),
                "seed {seed}: max_version({variable})"
            );
        }
    }
}

/// Every instruction-level analysis, for a case built elsewhere (for example
/// after a transform). The entry block must not be a branch target.
pub fn check_cfg_analyses(realized: &Realizations) {
    realized.run_cfgs(liveness_matches);
    realized.run_cfgs(reaching_definitions_match);
    realized.run_cfgs(cfg_metrics_match);
    realized.run_cfgs(ssa_reads_see_exactly_the_reaching_writes);
}
