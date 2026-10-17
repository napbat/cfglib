//! Brute-force reference answers over the neutral model.
//!
//! None of this shares code or data structures with the library: summaries
//! come from a per-root scan of the instructions, dominators from "remove a
//! block and see what the entry still reaches", frontiers from their textbook
//! definition, liveness from a backward fixed point over single instructions,
//! and placement from the iterated frontier cut down to live roots.

use std::collections::BTreeSet;

use crate::model::{Access, Ending, Inst, Model};

/// What one instruction does to the roots.
pub struct Effects {
    /// The roots whose value before the instruction can be read.
    pub reads: BTreeSet<u8>,
    pub writes: BTreeSet<u8>,
    /// The roots that the instruction overwrites completely.
    pub kills: BTreeSet<u8>,
    pub calls: bool,
}

pub fn effects(inst: &Inst) -> Effects {
    let mut effects = Effects {
        reads: BTreeSet::new(),
        writes: BTreeSet::new(),
        kills: BTreeSet::new(),
        calls: inst.ending == Ending::Call,
    };
    for access in &inst.accesses {
        match access {
            Access::Read(id) | Access::ReadSparse(id) => {
                effects.reads.insert(*id);
            }
            Access::ReadIndexed(ids) => effects.reads.extend(ids),
            Access::WriteFull(id) => {
                effects.writes.insert(*id);
                effects.kills.insert(*id);
            }
            // A write that keeps some bits reads the root as well.
            Access::WritePartial(id) | Access::WriteSparse(id) => {
                effects.reads.insert(*id);
                effects.writes.insert(*id);
            }
            Access::WriteIndexed(ids) => {
                effects.reads.extend(ids);
                effects.writes.extend(ids);
            }
            Access::RaiseBinding(_) | Access::Noise => {}
        }
    }
    effects
}

/// The root sets of one block.
#[derive(Debug, PartialEq, Eq)]
pub struct Summary {
    pub reads_before_write: BTreeSet<u8>,
    pub writes: BTreeSet<u8>,
    pub kills: BTreeSet<u8>,
}

/// The summary of live block `block`, found one root at a time.
pub fn summary(model: &Model, block: usize) -> Summary {
    let crate::model::BlockSlot::Live(live) = &model.slots[block] else {
        panic!("block {block} is removed");
    };
    let steps: Vec<Effects> = live.insts.iter().map(effects).collect();
    let mut summary = Summary {
        reads_before_write: BTreeSet::new(),
        writes: BTreeSet::new(),
        kills: BTreeSet::new(),
    };
    for root in model.named_roots() {
        // Walk until something reads the root, or overwrites it. A call reads
        // the root after the statements of its own instruction.
        for step in &steps {
            if step.reads.contains(&root) {
                summary.reads_before_write.insert(root);
                break;
            }
            if step.kills.contains(&root) {
                break;
            }
            if step.calls {
                summary.reads_before_write.insert(root);
                break;
            }
        }
        if steps
            .iter()
            .any(|step| step.writes.contains(&root) || step.calls)
        {
            summary.writes.insert(root);
        }
        if steps.iter().any(|step| step.kills.contains(&root)) {
            summary.kills.insert(root);
        }
    }
    summary
}

/// The live blocks without successors.
pub fn exits(model: &Model) -> Vec<usize> {
    model
        .live()
        .filter(|(_, block)| block.successors.is_empty())
        .map(|(index, _)| index)
        .collect()
}

/// Liveness at the start and at the end of every block. A removed block has
/// empty sets.
pub struct Liveness {
    pub live_in: Vec<BTreeSet<u8>>,
    pub live_out: Vec<BTreeSet<u8>>,
}

/// A backward fixed point at the granularity of one instruction.
///
/// `exit_seed` states what is live after each exit block.
pub fn liveness(model: &Model, exit_seed: &dyn Fn(usize) -> BTreeSet<u8>) -> Liveness {
    let named = model.named_roots();
    let count = model.slots.len();
    let mut result = Liveness {
        live_in: vec![BTreeSet::new(); count],
        live_out: vec![BTreeSet::new(); count],
    };
    loop {
        let mut changed = false;
        for (index, block) in model.live() {
            let out: BTreeSet<u8> = if block.successors.is_empty() {
                exit_seed(index)
            } else {
                block
                    .successors
                    .iter()
                    .flat_map(|&successor| result.live_in[successor].iter().copied())
                    .collect()
            };
            let mut live = out.clone();
            for inst in block.insts.iter().rev() {
                let step = effects(inst);
                if step.calls {
                    live.extend(named.iter().copied());
                }
                for killed in &step.kills {
                    live.remove(killed);
                }
                live.extend(step.reads.iter().copied());
            }
            changed |= result.live_out[index] != out || result.live_in[index] != live;
            result.live_out[index] = out;
            result.live_in[index] = live;
        }
        if !changed {
            return result;
        }
    }
}

/// Which blocks the entry reaches when `skip` is deleted.
fn reach(model: &Model, skip: Option<usize>) -> Vec<bool> {
    let mut seen = vec![false; model.slots.len()];
    if skip == Some(0) {
        return seen;
    }
    let mut stack = vec![0];
    seen[0] = true;
    while let Some(block) = stack.pop() {
        let crate::model::BlockSlot::Live(live) = &model.slots[block] else {
            continue;
        };
        for &next in &live.successors {
            if Some(next) != skip && !seen[next] {
                seen[next] = true;
                stack.push(next);
            }
        }
    }
    seen
}

pub fn reachable(model: &Model) -> Vec<bool> {
    reach(model, None)
}

/// The dominators of every reachable block, itself included: `d` dominates `n`
/// when deleting `d` leaves `n` unreachable. Other blocks get an empty set.
pub fn dominators(model: &Model) -> Vec<BTreeSet<usize>> {
    let reachable = reachable(model);
    (0..model.slots.len())
        .map(|node| {
            if !reachable[node] {
                return BTreeSet::new();
            }
            (0..model.slots.len())
                .filter(|&candidate| candidate == node || !reach(model, Some(candidate))[node])
                .collect()
        })
        .collect()
}

/// `y` is in the frontier of `x` when `x` dominates a predecessor of `y` and
/// does not strictly dominate `y`.
pub fn frontiers(model: &Model) -> Vec<BTreeSet<usize>> {
    let dominators = dominators(model);
    let reachable = reachable(model);
    (0..model.slots.len())
        .map(|x| {
            if !reachable[x] {
                return BTreeSet::new();
            }
            model
                .live()
                .map(|(y, _)| y)
                .filter(|&y| {
                    let dominates_a_predecessor = model
                        .predecessors(y)
                        .iter()
                        .any(|&predecessor| dominators[predecessor].contains(&x));
                    let strictly_dominates = x != y && dominators[y].contains(&x);
                    dominates_a_predecessor && !strictly_dominates
                })
                .collect()
        })
        .collect()
}

/// The iterated dominance frontier of `definitions`.
fn iterated(frontiers: &[BTreeSet<usize>], definitions: &BTreeSet<usize>) -> BTreeSet<usize> {
    let mut joins = BTreeSet::new();
    let mut work: Vec<usize> = definitions.iter().copied().collect();
    while let Some(block) = work.pop() {
        for &join in &frontiers[block] {
            if joins.insert(join) {
                work.push(join);
            }
        }
    }
    joins
}

/// The roots that need a parameter at each block: the iterated frontier of the
/// blocks that define the root, kept where the root is live on entry. The
/// entry block defines every root, and an unreachable block defines none.
pub fn placement(model: &Model, live_in: &[BTreeSet<u8>]) -> Vec<BTreeSet<u8>> {
    let frontiers = frontiers(model);
    let reachable = reachable(model);
    let mut placed = vec![BTreeSet::new(); model.slots.len()];
    for root in model.named_roots() {
        let mut definitions = BTreeSet::from([0]);
        for (index, block) in model.live() {
            let writes = block
                .insts
                .iter()
                .map(effects)
                .any(|step| step.writes.contains(&root) || step.calls);
            if reachable[index] && writes {
                definitions.insert(index);
            }
        }
        for join in iterated(&frontiers, &definitions) {
            if live_in[join].contains(&root) {
                placed[join].insert(root);
            }
        }
    }
    placed
}
