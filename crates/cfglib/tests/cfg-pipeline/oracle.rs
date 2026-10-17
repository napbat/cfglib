//! Brute-force structural oracles: dominance by its definition ("remove the
//! candidate and see whether the node is still reachable"), post-dominance by
//! path existence, frontiers from those, and the SSA invariants stated in
//! terms of them.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::{BlockId, Cfg, EdgeId, EdgeKind, ProgramPoint, SsaForm, SsaValue};

use crate::toy::{Inst, Var};

/// The `(source, target)` blocks of each edge in `edges`, which is what an
/// execution trace observes.
pub fn endpoints<I>(cfg: &Cfg<I>, edges: &BTreeSet<EdgeId>) -> BTreeSet<(BlockId, BlockId)> {
    edges
        .iter()
        .map(|&edge| (cfg.edge(edge).source(), cfg.edge(edge).target()))
        .collect()
}

/// The blocks reachable from the entry without passing through `avoided`.
fn reachable_avoiding<I>(cfg: &Cfg<I>, avoided: Option<BlockId>) -> BTreeSet<BlockId> {
    let mut seen = BTreeSet::new();
    if Some(cfg.entry()) == avoided {
        return seen;
    }
    let mut stack = vec![cfg.entry()];
    while let Some(block) = stack.pop() {
        if !seen.insert(block) {
            continue;
        }
        stack.extend(cfg.successors(block).filter(|next| Some(*next) != avoided));
    }
    seen
}

/// Dominance, frontiers, and post-dominance of one CFG, each computed from
/// its definition.
pub struct Reference {
    /// The blocks reachable from the entry.
    pub reachable: BTreeSet<BlockId>,
    dominators: BTreeMap<BlockId, BTreeSet<BlockId>>,
    predecessors: BTreeMap<BlockId, Vec<BlockId>>,
    exits: Vec<BlockId>,
    successors: BTreeMap<BlockId, Vec<BlockId>>,
}

impl Reference {
    /// Compute the reference for `cfg`.
    pub fn new<I>(cfg: &Cfg<I>) -> Self {
        let reachable = reachable_avoiding(cfg, None);
        let mut dominators: BTreeMap<BlockId, BTreeSet<BlockId>> = BTreeMap::new();
        for &candidate in &reachable {
            let without = reachable_avoiding(cfg, Some(candidate));
            for &node in &reachable {
                if node == candidate || !without.contains(&node) {
                    dominators.entry(node).or_default().insert(candidate);
                }
            }
        }
        let mut predecessors: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
        let mut successors: BTreeMap<BlockId, Vec<BlockId>> = BTreeMap::new();
        for edge in cfg.edges() {
            predecessors
                .entry(edge.target())
                .or_default()
                .push(edge.source());
            successors
                .entry(edge.source())
                .or_default()
                .push(edge.target());
        }
        let exits = reachable
            .iter()
            .copied()
            .filter(|block| !successors.contains_key(block))
            .collect();
        Self {
            reachable,
            dominators,
            predecessors,
            exits,
            successors,
        }
    }

    /// Whether `dominator` dominates `node`: every path from the entry to
    /// `node` passes through `dominator`. Reflexive on reachable blocks.
    pub fn dominates(&self, dominator: BlockId, node: BlockId) -> bool {
        self.dominators
            .get(&node)
            .is_some_and(|set| set.contains(&dominator))
    }

    /// Whether `dominator` dominates `node` and differs from it.
    pub fn strictly_dominates(&self, dominator: BlockId, node: BlockId) -> bool {
        dominator != node && self.dominates(dominator, node)
    }

    /// The closest strict dominator of `node`: the one every other strict
    /// dominator also dominates.
    pub fn immediate_dominator(&self, node: BlockId) -> Option<BlockId> {
        let strict: Vec<BlockId> = self
            .dominators
            .get(&node)?
            .iter()
            .copied()
            .filter(|candidate| *candidate != node)
            .collect();
        strict
            .iter()
            .copied()
            .find(|&candidate| strict.iter().all(|&other| self.dominates(other, candidate)))
    }

    /// The dominance frontier of `block` from the definition: blocks `y` where
    /// `block` dominates a predecessor of `y` and does not strictly dominate
    /// `y`.
    pub fn frontier(&self, block: BlockId) -> BTreeSet<BlockId> {
        self.reachable
            .iter()
            .copied()
            .filter(|&target| {
                !self.strictly_dominates(block, target)
                    && self.predecessors.get(&target).is_some_and(|preds| {
                        preds.iter().any(|&pred| {
                            self.reachable.contains(&pred) && self.dominates(block, pred)
                        })
                    })
            })
            .collect()
    }

    /// The iterated dominance frontier of `definitions`: the closure of the
    /// frontier under taking the frontier of its own members.
    pub fn iterated_frontier(&self, definitions: &BTreeSet<BlockId>) -> BTreeSet<BlockId> {
        let mut result = BTreeSet::new();
        let mut pending: Vec<BlockId> = definitions
            .iter()
            .copied()
            .filter(|block| self.reachable.contains(block))
            .collect();
        while let Some(block) = pending.pop() {
            for member in self.frontier(block) {
                if result.insert(member) {
                    pending.push(member);
                }
            }
        }
        result
    }

    /// Whether `block` can reach an exit without passing through `avoided`.
    fn reaches_exit_avoiding(&self, block: BlockId, avoided: Option<BlockId>) -> bool {
        if Some(block) == avoided {
            return false;
        }
        let mut seen = BTreeSet::new();
        let mut stack = vec![block];
        while let Some(current) = stack.pop() {
            if !seen.insert(current) {
                continue;
            }
            if self.exits.contains(&current) {
                return true;
            }
            stack.extend(
                self.successors
                    .get(&current)
                    .into_iter()
                    .flatten()
                    .copied()
                    .filter(|next| Some(*next) != avoided),
            );
        }
        false
    }

    /// Whether every path from `block` to an exit exists, i.e. the block can
    /// finish at all.
    pub fn reaches_exit(&self, block: BlockId) -> bool {
        self.reaches_exit_avoiding(block, None)
    }

    /// Whether `post_dominator` post-dominates `block`: `block` reaches an
    /// exit, and no path from it to an exit avoids `post_dominator`.
    /// Meaningful only for a `block` that reaches an exit.
    pub fn post_dominates(&self, post_dominator: BlockId, block: BlockId) -> bool {
        post_dominator == block || !self.reaches_exit_avoiding(block, Some(post_dominator))
    }

    /// The reachable blocks with no successor.
    pub fn exits(&self) -> &[BlockId] {
        &self.exits
    }
}

/// Whether any edge of `cfg` is an unwind edge.
pub fn has_unwind_edges<I>(cfg: &Cfg<I>) -> bool {
    cfg.edges()
        .any(|edge| edge.kind() == EdgeKind::ExceptionUnwind)
}

/// Where an SSA value is defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Definition {
    Phi(BlockId),
    Instruction(ProgramPoint),
}

/// Every SSA invariant violation found in `ssa`, as readable messages.
///
/// Checked, each stated against brute-force dominance: a value has one
/// definition; a use is dominated by its definition (instruction order in
/// the same block, strict dominance across blocks, a phi operand at the end
/// of its predecessor); a version-zero use has no dominating definition of
/// its variable; `value_at` answers what each instruction reads; and every
/// annotation sits on the instruction it describes.
pub fn ssa_violations(
    cfg: &Cfg<Inst<Var>>,
    ssa: &SsaForm<Var>,
    reference: &Reference,
) -> Vec<String> {
    let mut violations = Vec::new();
    let mut definitions: BTreeMap<SsaValue<Var>, Definition> = BTreeMap::new();
    let mut define = |value: &SsaValue<Var>, site: Definition, violations: &mut Vec<String>| {
        if value.version == 0 {
            violations.push(format!("{value:?} is defined but version 0 is the live-in"));
        }
        if let Some(previous) = definitions.insert(value.clone(), site) {
            violations.push(format!(
                "{value:?} defined twice: {previous:?} and {site:?}"
            ));
        }
    };
    for block in cfg.block_ids() {
        let contents = ssa.block(block);
        if contents.block != block {
            violations.push(format!(
                "{block:?} holds the contents of {:?}",
                contents.block
            ));
        }
        for phi in &contents.phis {
            define(&phi.result, Definition::Phi(block), &mut violations);
        }
        for (index, annotation) in contents.instructions.iter().enumerate() {
            let point = ProgramPoint {
                block,
                inst_idx: index,
            };
            if annotation.point != point || ssa.instruction(point) != Some(annotation) {
                violations.push(format!("annotation at {point} is not indexed by it"));
            }
            for value in annotation.defs() {
                define(value, Definition::Instruction(point), &mut violations);
            }
        }
    }

    let dominates_use = |definition: Definition, user: ProgramPoint, at_departure: bool| {
        match definition {
            Definition::Phi(block) => reference.dominates(block, user.block),
            Definition::Instruction(site) if site.block == user.block => {
                site.inst_idx < user.inst_idx
            }
            Definition::Instruction(site) => {
                // A departure point names the end of the predecessor, so the
                // definition's block must still strictly dominate it.
                let _ = at_departure;
                reference.strictly_dominates(site.block, user.block)
            }
        }
    };
    let mut check_use = |value: &SsaValue<Var>, user: ProgramPoint, at_departure: bool| {
        if !reference.reachable.contains(&user.block) {
            return;
        }
        match definitions.get(value) {
            Some(&definition) => {
                if !dominates_use(definition, user, at_departure) {
                    violations.push(format!(
                        "{value:?} used at {user} is not dominated by {definition:?}"
                    ));
                }
            }
            None if value.version == 0 => {
                let shadowing = definitions
                    .iter()
                    .filter(|(other, _)| other.variable == value.variable)
                    .find(|(_, definition)| dominates_use(**definition, user, at_departure));
                if let Some((other, _)) = shadowing {
                    violations.push(format!(
                        "live-in {value:?} used at {user} although {other:?} dominates it"
                    ));
                }
            }
            None => violations.push(format!("{value:?} is used at {user} but never defined")),
        }
    };
    for block in cfg.block_ids() {
        for phi in &ssa.block(block).phis {
            for (departure, operand) in phi.operands() {
                check_use(operand, *departure, true);
            }
        }
        for annotation in &ssa.block(block).instructions {
            for value in annotation.uses() {
                check_use(value, annotation.point, false);
            }
        }
    }
    violations.extend(value_at_violations(cfg, ssa));
    violations
}

/// `SsaForm::value_at` answers, for every use, the value that use reads.
fn value_at_violations(cfg: &Cfg<Inst<Var>>, ssa: &SsaForm<Var>) -> Vec<String> {
    let mut violations = Vec::new();
    for block in cfg.block_ids() {
        for (index, inst) in cfg.block(block).instructions().iter().enumerate() {
            let point = ProgramPoint {
                block,
                inst_idx: index,
            };
            let annotation = ssa
                .instruction(point)
                .expect("every instruction is annotated");
            for (variable, value) in inst.uses.iter().zip(annotation.uses()) {
                let answered = ssa.value_at(point, variable);
                if answered != *value {
                    violations.push(format!("value_at({point}, {variable:?}) = {answered:?}, the instruction reads {value:?}"));
                }
            }
        }
    }
    violations
}

/// The departure points of the edge-sourced operands of a phi at `block`,
/// as the definition of a phi states them: one operand per incoming edge,
/// departing at the end of its source. Only valid without unwind edges.
pub fn expected_phi_departures(cfg: &Cfg<Inst<Var>>, block: BlockId) -> Vec<ProgramPoint> {
    cfg.incoming(block)
        .map(|edge| {
            let source = cfg.edge(edge).source();
            ProgramPoint {
                block: source,
                inst_idx: cfg.block(source).instructions().len(),
            }
        })
        .collect()
}
