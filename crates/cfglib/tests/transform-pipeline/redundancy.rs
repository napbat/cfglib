//! Redundancy elimination: local and global value numbering judged against
//! brute-force expression trees, and PRE judged by running the program.

use std::collections::BTreeMap;

use cfglib::{
    BasicBlock, BlockId, BlockValueNumbers, Cfg, DominatorTree, EdgeKind, InstrInfo, PreAnalysis,
    ValueNumbering, eliminate_pre,
};

use crate::programs::{Rng, random_program, random_single_assignment_program};
use crate::toy::{BinaryOp, Builder, Inst, Kind, Reg, assert_same_behavior_observing};

/// The expression a pure instruction computes, spelled out over the block's
/// or program's inputs. Two instructions compute the same value exactly when
/// these strings are equal.
fn spell(inst: &Inst, operand: impl Fn(Reg) -> String) -> String {
    let operands: Vec<String> = inst
        .uses()
        .iter()
        .map(|&register| operand(register))
        .collect();
    format!("{:?}({})", inst.kind, operands.join(","))
}

fn is_numbered(inst: &Inst) -> bool {
    matches!(inst.kind, Kind::Constant(_) | Kind::Move | Kind::Binary(_))
}

fn tree(
    register: Reg,
    definitions: &BTreeMap<Reg, &Inst>,
    memo: &mut BTreeMap<Reg, String>,
) -> String {
    if let Some(known) = memo.get(&register) {
        return known.clone();
    }
    let spelled = match definitions.get(&register) {
        Some(inst) => {
            let operands: Vec<String> = inst
                .uses()
                .iter()
                .map(|&used| tree(used, definitions, memo))
                .collect();
            format!("{:?}({})", inst.kind, operands.join(","))
        }
        None => format!("input{register}"),
    };
    memo.insert(register, spelled.clone());
    spelled
}

/// The expression tree of every register of a single-assignment program.
fn program_trees<E>(cfg: &Cfg<Inst, E>) -> BTreeMap<Reg, String> {
    let definitions: BTreeMap<Reg, &Inst> = cfg
        .blocks()
        .flat_map(BasicBlock::instructions)
        .filter(|inst| is_numbered(inst))
        .map(|inst| (inst.defs()[0], inst))
        .collect();
    let mut memo = BTreeMap::new();
    definitions
        .keys()
        .map(|&register| (register, tree(register, &definitions, &mut memo)))
        .collect()
}

struct Site {
    block: BlockId,
    index: usize,
    tree: String,
}

fn numbered_sites<E>(cfg: &Cfg<Inst, E>) -> Vec<Site> {
    let trees = program_trees(cfg);
    let mut sites = Vec::new();
    for block in cfg.block_ids() {
        for (index, inst) in cfg.block(block).instructions().iter().enumerate() {
            if is_numbered(inst) {
                sites.push(Site {
                    block,
                    index,
                    tree: trees[&inst.defs()[0]].clone(),
                });
            }
        }
    }
    sites
}

#[test]
fn global_numbering_flags_exactly_the_recomputations_of_a_dominating_expression() {
    let mut flagged = 0;
    for seed in 0..80 {
        let cfg = random_single_assignment_program::<()>(seed);
        let dom = DominatorTree::compute(&cfg);
        let numbering = ValueNumbering::compute(&cfg, &dom);
        let sites = numbered_sites(&cfg);
        let context = format!("seed {seed}");

        for later in &sites {
            let earlier_equal = sites.iter().any(|earlier| {
                earlier.tree == later.tree
                    && ((earlier.block == later.block && earlier.index < later.index)
                        || dom.strictly_dominates(earlier.block, later.block))
            });
            let numbers = &numbering.blocks[&later.block];
            assert_eq!(
                numbers.redundant.contains(&later.index),
                earlier_equal,
                "{context}: {} #{} computing {}",
                later.block,
                later.index,
                later.tree
            );
            assert!(numbers.inst_vn[later.index].is_some(), "{context}");
            flagged += usize::from(earlier_equal);
        }
        for (position, left) in sites.iter().enumerate() {
            for right in &sites[position + 1..] {
                let left_vn = numbering.blocks[&left.block].inst_vn[left.index].unwrap();
                let right_vn = numbering.blocks[&right.block].inst_vn[right.index].unwrap();
                assert!(left_vn < numbering.value_count && right_vn < numbering.value_count);
                if left.tree != right.tree {
                    assert_ne!(
                        left_vn, right_vn,
                        "{context}: different values share a number"
                    );
                }
            }
        }
        assert_eq!(
            numbering.redundant_count(),
            sites
                .iter()
                .filter(|later| numbering.blocks[&later.block]
                    .redundant
                    .contains(&later.index))
                .count(),
            "{context}"
        );
    }
    assert!(
        flagged > 10,
        "only {flagged} redundant expressions across all programs"
    );
}

#[test]
fn pre_analysis_reports_the_pairs_global_numbering_calls_redundant() {
    for seed in 0..60 {
        let cfg = random_single_assignment_program::<()>(seed);
        let dom = DominatorTree::compute(&cfg);
        let numbering = ValueNumbering::compute(&cfg, &dom);
        let analysis = PreAnalysis::compute(&cfg, &dom);

        let mut reported: Vec<(BlockId, usize, u32)> = analysis.redundant.clone();
        reported.sort();
        let mut expected: Vec<(BlockId, usize, u32)> = numbering
            .blocks
            .iter()
            .flat_map(|(&block, numbers)| {
                numbers
                    .redundant
                    .iter()
                    .map(move |&index| (block, index, numbers.inst_vn[index].unwrap()))
            })
            .collect();
        expected.sort();
        assert_eq!(reported, expected, "seed {seed}");
    }
}

#[test]
fn local_numbering_matches_symbolic_execution_of_every_block() {
    for seed in 0..80 {
        let cfg = random_program::<()>(seed, true);
        let mut next = 0;
        for block in cfg.block_ids() {
            let (numbers, following) = BlockValueNumbers::compute(&cfg, block, next);
            let mut environment: BTreeMap<Reg, String> = BTreeMap::new();
            let mut seen: BTreeMap<String, usize> = BTreeMap::new();
            let mut expected_redundant = Vec::new();
            for (index, inst) in cfg.block(block).instructions().iter().enumerate() {
                if !is_numbered(inst) {
                    assert_eq!(numbers.inst_vn[index], None);
                    continue;
                }
                let value = spell(inst, |register| {
                    environment
                        .get(&register)
                        .cloned()
                        .unwrap_or_else(|| format!("input{register}"))
                });
                if let Some(&first) = seen.get(&value) {
                    expected_redundant.push(index);
                    assert_eq!(
                        numbers.inst_vn[index], numbers.inst_vn[first],
                        "seed {seed}"
                    );
                } else {
                    seen.insert(value.clone(), index);
                    for earlier in seen.values().filter(|&&other| other != index) {
                        assert_ne!(
                            numbers.inst_vn[index], numbers.inst_vn[*earlier],
                            "seed {seed}"
                        );
                    }
                }
                environment.insert(inst.defs()[0], value);
            }
            assert_eq!(
                numbers.redundant, expected_redundant,
                "seed {seed}, {block}"
            );
            assert!(
                numbers
                    .inst_vn
                    .iter()
                    .flatten()
                    .all(|&vn| next <= vn && vn < following)
            );
            next = following;
        }
    }
}

fn straight_line(seed: u64) -> Cfg<Inst> {
    let mut rng = Rng::new(seed);
    let pool: [Reg; 5] = [0, 2, 10, 11, 12];
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for _ in 0..16 {
        let (left, right) = (pool[rng.below(5)], pool[rng.below(5)]);
        let destination = pool[1 + rng.below(4)];
        let inst = match rng.below(5) {
            0 => Inst::constant(destination, i64::try_from(rng.below(3)).unwrap()),
            1 => Inst::mov(destination, left),
            2 => Inst::binary(BinaryOp::Multiply, destination, left, right),
            _ => Inst::add(destination, left, right),
        };
        builder.push(entry, inst);
    }
    builder.push(entry, Inst::print(10));
    builder.finish()
}

#[test]
fn global_and_local_numbering_agree_inside_one_block_with_redefinitions() {
    for seed in 0..100 {
        let cfg = straight_line(seed);
        let entry = cfg.entry();
        let global = ValueNumbering::compute(&cfg, &DominatorTree::compute(&cfg));
        let (local, _) = BlockValueNumbers::compute(&cfg, entry, 0);
        let global = &global.blocks[&entry];

        assert_eq!(global.redundant, local.redundant, "seed {seed}");
        for left in 0..local.inst_vn.len() {
            for right in 0..local.inst_vn.len() {
                assert_eq!(
                    global.inst_vn[left] == global.inst_vn[right],
                    local.inst_vn[left] == local.inst_vn[right],
                    "seed {seed}: numbers {left} and {right} are grouped differently"
                );
            }
        }
    }
}

/// `t` is computed in the entry, an arm overwrites an operand, and the join
/// recomputes. Only the arm-free path still holds the entry's value.
fn side_redefinition() -> (Cfg<Inst>, BlockId) {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for inst in [Inst::add(10, 0, 2), Inst::less(11, 0, 1), Inst::branch(11)] {
        builder.push(entry, inst);
    }
    let side = builder.block([Inst::constant(2, 9)]);
    let join = builder.block([
        Inst::add(12, 0, 2),
        Inst::print(10),
        Inst::print(12),
        Inst::ret(),
    ]);
    builder.edge(entry, side, EdgeKind::ConditionalTrue);
    builder.edge(entry, join, EdgeKind::ConditionalFalse);
    builder.edge(side, join, EdgeKind::Fallthrough);
    (builder.finish(), join)
}

#[test]
fn global_numbering_does_not_reuse_an_expression_whose_operand_a_side_path_rewrites() {
    let (cfg, join) = side_redefinition();
    let numbering = ValueNumbering::compute(&cfg, &DominatorTree::compute(&cfg));

    assert!(
        numbering.blocks[&join].redundant.is_empty(),
        "r2 is 9 on the path through the arm, so the join's r0 + r2 is not the entry's"
    );
}

#[test]
fn eliminating_redundancy_never_reuses_a_value_across_a_side_redefinition() {
    let (before, _) = side_redefinition();
    let mut cfg = before.clone();
    let dom = DominatorTree::compute(&cfg);
    eliminate_pre(&mut cfg, &dom);

    assert_same_behavior_observing(&before, &cfg, &[], "side redefinition");
}

#[test]
fn eliminating_a_dominated_recomputation_keeps_every_user_defined() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for inst in [
        Inst::add(10, 0, 2),
        Inst::add(11, 0, 2),
        Inst::print(11),
        Inst::print(10),
        Inst::ret(),
    ] {
        builder.push(entry, inst);
    }
    let before = builder.finish();
    let mut cfg = before.clone();
    let dom = DominatorTree::compute(&cfg);

    assert_eq!(eliminate_pre(&mut cfg, &dom), 1);

    assert_eq!(
        cfg.block(entry).instructions().len(),
        before.block(entry).instructions().len() - 1
    );
    assert_same_behavior_observing(&before, &cfg, &[], "dominated recomputation");
}

#[test]
fn an_expression_computed_on_one_path_only_is_not_removed_from_the_join() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for inst in [Inst::less(11, 0, 1), Inst::branch(11)] {
        builder.push(entry, inst);
    }
    let arm = builder.block([Inst::add(10, 0, 2), Inst::print(10)]);
    let join = builder.block([Inst::add(12, 0, 2), Inst::print(12), Inst::ret()]);
    builder.edge(entry, arm, EdgeKind::ConditionalTrue);
    builder.edge(entry, join, EdgeKind::ConditionalFalse);
    builder.edge(arm, join, EdgeKind::Fallthrough);
    let before = builder.finish();
    let mut cfg = before.clone();
    let dom = DominatorTree::compute(&cfg);

    assert_eq!(
        eliminate_pre(&mut cfg, &dom),
        0,
        "the join is reachable without the arm"
    );
    assert_eq!(cfg, before);
}

#[test]
fn an_operand_redefined_between_two_computations_makes_them_different_values() {
    let mut builder = Builder::<()>::new();
    let entry = builder.entry();
    for inst in [
        Inst::add(10, 0, 2),
        Inst::constant(2, 9),
        Inst::add(11, 0, 2),
        Inst::print(10),
        Inst::print(11),
        Inst::ret(),
    ] {
        builder.push(entry, inst);
    }
    let before = builder.finish();
    let mut cfg = before.clone();
    let dom = DominatorTree::compute(&cfg);

    assert_eq!(eliminate_pre(&mut cfg, &dom), 0);
    assert_eq!(cfg, before);
}

#[test]
fn eliminating_redundancy_preserves_the_output_of_random_programs() {
    let mut removed = 0;
    for seed in 0..80 {
        let before = random_program::<()>(seed, false);
        let dom = DominatorTree::compute(&before);
        let analysis = PreAnalysis::compute(&before, &dom);
        let mut cfg = before.clone();
        let count = eliminate_pre(&mut cfg, &dom);

        assert!(count <= analysis.redundant.len(), "seed {seed}");
        assert_same_behavior_observing(&before, &cfg, &[], &format!("seed {seed}"));
        removed += count;
    }
    assert!(removed > 0, "no random program had a removable redundancy");
}
