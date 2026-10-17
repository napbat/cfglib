//! `PhiWebs` against an independent union-find over the same phis.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{PhiWebs, SsaForm, SsaValue};

use crate::machine::Reg;
use crate::programs::{analyze, corpus, dead_phi_diamond};

fn root(parent: &mut [usize], mut node: usize) -> usize {
    while parent[node] != node {
        parent[node] = parent[parent[node]];
        node = parent[node];
    }
    node
}

/// The partition of `values` into the connected components of `pairs`,
/// by an independent union-find over indices.
fn components(
    values: &BTreeSet<SsaValue<Reg>>,
    pairs: &[(SsaValue<Reg>, SsaValue<Reg>)],
) -> BTreeSet<BTreeSet<SsaValue<Reg>>> {
    let list: Vec<_> = values.iter().cloned().collect();
    let mut parent: Vec<usize> = (0..list.len()).collect();
    let position = |value: &SsaValue<Reg>| {
        list.iter()
            .position(|other| other == value)
            .expect("a listed value")
    };
    for (left, right) in pairs {
        let (a, b) = (
            root(&mut parent, position(left)),
            root(&mut parent, position(right)),
        );
        parent[a] = b;
    }
    let mut groups: BTreeMap<usize, BTreeSet<SsaValue<Reg>>> = BTreeMap::new();
    for (index, value) in list.iter().enumerate() {
        let representative = root(&mut parent, index);
        groups
            .entry(representative)
            .or_default()
            .insert(value.clone());
    }
    groups.into_values().collect()
}

fn phi_pairs(
    ssa: &SsaForm<Reg>,
    keep: impl Fn(&SsaValue<Reg>) -> bool,
) -> Vec<(SsaValue<Reg>, SsaValue<Reg>)> {
    ssa.phis()
        .filter(|(_, phi)| keep(&phi.result))
        .flat_map(|(_, phi)| {
            phi.operands()
                .iter()
                .map(|(_, operand)| (phi.result.clone(), operand.clone()))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn values_of(pairs: &[(SsaValue<Reg>, SsaValue<Reg>)]) -> BTreeSet<SsaValue<Reg>> {
    pairs
        .iter()
        .flat_map(|(left, right)| [left.clone(), right.clone()])
        .collect()
}

/// The results of the phis that something reads: a value read by an
/// instruction, or by an operand of a phi that is itself read. Found by a
/// depth-first walk from the instruction uses over phi operand edges.
fn live_phi_results(ssa: &SsaForm<Reg>) -> BTreeSet<SsaValue<Reg>> {
    let operands: BTreeMap<SsaValue<Reg>, Vec<SsaValue<Reg>>> = ssa
        .phis()
        .map(|(_, phi)| {
            (
                phi.result.clone(),
                phi.operands()
                    .iter()
                    .map(|(_, value)| value.clone())
                    .collect(),
            )
        })
        .collect();
    let mut stack: Vec<SsaValue<Reg>> = ssa
        .blocks()
        .iter()
        .flat_map(|block| &block.instructions)
        .flat_map(|insn| insn.uses().iter().cloned())
        .collect();
    let mut live = BTreeSet::new();
    while let Some(value) = stack.pop() {
        if operands.contains_key(&value) && live.insert(value.clone()) {
            stack.extend(operands[&value].iter().cloned());
        }
    }
    live
}

fn webs_as_partition(webs: &PhiWebs<Reg>) -> BTreeSet<BTreeSet<SsaValue<Reg>>> {
    webs.webs.iter().map(|web| web.values.clone()).collect()
}

#[test]
fn phi_webs_partition_the_values_connected_by_phis() {
    for (name, program) in corpus() {
        let ssa = analyze(&program.cfg);
        let pairs = phi_pairs(&ssa, |_| true);
        let expected = components(&values_of(&pairs), &pairs);
        let webs = PhiWebs::compute(&ssa);
        assert_eq!(webs_as_partition(&webs), expected, "{name}");
    }
}

#[test]
fn a_web_index_names_the_web_holding_the_value() {
    for (name, program) in corpus() {
        let ssa = analyze(&program.cfg);
        for webs in [PhiWebs::compute(&ssa), PhiWebs::compute_live(&ssa)] {
            let listed: usize = webs.webs.iter().map(|web| web.values.len()).sum();
            assert_eq!(
                listed,
                webs.web_of.len(),
                "{name}: every value is in exactly one web"
            );
            for (value, &index) in &webs.web_of {
                assert!(webs.webs[index].values.contains(value), "{name}: {value:?}");
            }
            for (_, phi) in ssa.phis() {
                if let Some(&web) = webs.web_of.get(&phi.result) {
                    for (_, operand) in phi.operands() {
                        assert_eq!(
                            webs.web_of.get(operand),
                            Some(&web),
                            "{name}: operand of {:?}",
                            phi.result
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn live_phi_webs_leave_out_phis_nothing_reads() {
    for (name, program) in corpus() {
        let ssa = analyze(&program.cfg);
        let live = live_phi_results(&ssa);
        let pairs = phi_pairs(&ssa, |result| live.contains(result));
        let expected = components(&values_of(&pairs), &pairs);
        assert_eq!(
            webs_as_partition(&PhiWebs::compute_live(&ssa)),
            expected,
            "{name}"
        );
    }
}

#[test]
fn a_dead_phi_forms_a_web_only_when_dead_phis_are_included() {
    let program = dead_phi_diamond();
    let ssa = analyze(&program.cfg);
    let merge = program.cfg.block_ids().last().expect("the merge block");
    let dead = ssa
        .block(merge)
        .phis
        .iter()
        .find(|phi| phi.result.variable == Reg("dead"))
        .expect("`dead` merges at the join");

    let all = PhiWebs::compute(&ssa);
    let web = &all.webs[all.web_of[&dead.result]];
    assert_eq!(web.values.len(), 3, "the phi result and both arm values");

    let live = PhiWebs::compute_live(&ssa);
    assert!(!live.web_of.contains_key(&dead.result));
    // The live phi of `x` still forms a web of its own.
    assert_eq!(live.webs.len(), 1);
}
