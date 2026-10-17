//! A consumer-side verifier of the `ir::mlil::FunctionView` contract.
//!
//! `cfglib` ships no MLIL verifier: the documentation states the contract and
//! leaves the check to the consumer. This one is written from the rules in the
//! module documentation of `ir::mlil` and runs over any view, so the same
//! checker accepts every valid function and names each violation of a broken
//! one. Block adjacency is compared against the successor records, and the
//! dominance of every use by its definition uses the crate's `DominatorTree`.

use std::collections::{BTreeMap, BTreeSet};

use cfglib::ir::mlil::{self, ValueDef};
use cfglib::{DenseId, DominatorTree, GraphView};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Violation {
    /// A live block with no instruction, so no terminator.
    EmptyBlock { block: usize },
    /// The last instruction of a block is not a terminator.
    NotATerminator { block: usize },
    /// A terminator that is not the last instruction.
    TerminatorNotLast { block: usize, position: usize },
    /// The last instruction is no terminator and still has successor records.
    SuccessorsOfNonTerminator { block: usize },
    /// A successor record whose target is not a live block.
    DanglingTarget { block: usize, record: usize },
    /// A successor record whose arguments do not match the target parameters
    /// in count.
    ArgumentCount { block: usize, record: usize },
    /// An argument whose type differs from the type of its parameter.
    ArgumentType {
        block: usize,
        record: usize,
        position: usize,
    },
    /// `def` reports another place than the one that lists the value.
    DefinitionMismatch { value: usize },
    /// A value that two places list.
    DefinedTwice { value: usize },
    /// `block_of` disagrees with the block that lists the instruction.
    BlockOfMismatch { inst: usize },
    /// An identity at or above its bound.
    OutOfBounds { what: &'static str, index: usize },
    /// An operand that no parameter or result defines.
    UndefinedOperand { value: usize },
    /// A use that its definition does not dominate.
    NotDominated { value: usize, block: usize },
    /// The graph view lists other successors than the successor records.
    AdjacencyMismatch { block: usize },
    /// The graph view lists other predecessors than the successor records.
    PredecessorMismatch { block: usize },
}

/// Checks `function` and returns every violation, in a stable order.
pub fn check<D, F>(function: &F, is_terminator: &dyn Fn(&D::Op) -> bool) -> Vec<Violation>
where
    D: mlil::Dialect,
    F: mlil::FunctionView<D>,
{
    let mut found = Vec::new();
    let blocks: Vec<F::Block> = GraphView::node_ids(function).collect();
    check_blocks::<D, F>(function, &blocks, is_terminator, &mut found);
    check_definitions::<D, F>(function, &blocks, &mut found);
    check_adjacency::<D, F>(function, &blocks, &mut found);
    // The dominator tree of a view that breaks its own graph contract is
    // unspecified, so dominance is only asked of a sound adjacency.
    let adjacency_is_sound = found.iter().all(|violation| {
        !matches!(
            violation,
            Violation::AdjacencyMismatch { .. } | Violation::PredecessorMismatch { .. }
        )
    });
    if adjacency_is_sound {
        check_uses::<D, F>(function, &blocks, &mut found);
    }
    found
}

fn check_blocks<D, F>(
    function: &F,
    blocks: &[F::Block],
    is_terminator: &dyn Fn(&D::Op) -> bool,
    found: &mut Vec<Violation>,
) where
    D: mlil::Dialect,
    F: mlil::FunctionView<D>,
{
    let live: BTreeSet<usize> = blocks.iter().map(|block| block.index()).collect();
    for &block in blocks {
        let index = block.index();
        let records = mlil::FunctionView::<D>::successors(function, block);
        let has_records = records.len() > 0;
        match function.insts(block).split_last() {
            None => found.push(Violation::EmptyBlock { block: index }),
            Some((&last, rest)) => {
                if !is_terminator(function.op(last)) {
                    found.push(Violation::NotATerminator { block: index });
                    if has_records {
                        found.push(Violation::SuccessorsOfNonTerminator { block: index });
                    }
                }
                for (position, &inst) in rest.iter().enumerate() {
                    if is_terminator(function.op(inst)) {
                        found.push(Violation::TerminatorNotLast {
                            block: index,
                            position,
                        });
                    }
                }
            }
        }
        for (record, successor) in mlil::FunctionView::<D>::successors(function, block).enumerate()
        {
            if !live.contains(&successor.target.index()) {
                found.push(Violation::DanglingTarget {
                    block: index,
                    record,
                });
                continue;
            }
            let params = function.params(successor.target);
            if params.len() != successor.args.len() {
                found.push(Violation::ArgumentCount {
                    block: index,
                    record,
                });
                continue;
            }
            for (position, (&arg, &param)) in successor.args.iter().zip(params).enumerate() {
                if function.ty(arg) != function.ty(param) {
                    found.push(Violation::ArgumentType {
                        block: index,
                        record,
                        position,
                    });
                }
            }
        }
    }
}

fn check_definitions<D, F>(function: &F, blocks: &[F::Block], found: &mut Vec<Violation>)
where
    D: mlil::Dialect,
    F: mlil::FunctionView<D>,
{
    let mut listed: BTreeMap<usize, usize> = BTreeMap::new();
    for &block in blocks {
        for (index, &value) in function.params(block).iter().enumerate() {
            *listed.entry(value.index()).or_default() += 1;
            let index = u32::try_from(index).expect("few params");
            if function.def(value) != (ValueDef::Param { block, index }) {
                found.push(Violation::DefinitionMismatch {
                    value: value.index(),
                });
            }
        }
        for &inst in function.insts(block) {
            if inst.index() >= function.inst_bound() {
                found.push(Violation::OutOfBounds {
                    what: "instruction",
                    index: inst.index(),
                });
            }
            if function.block_of(inst) != block {
                found.push(Violation::BlockOfMismatch { inst: inst.index() });
            }
            for (index, &value) in function.results(inst).iter().enumerate() {
                *listed.entry(value.index()).or_default() += 1;
                let index = u32::try_from(index).expect("few results");
                if function.def(value) != (ValueDef::Result { inst, index }) {
                    found.push(Violation::DefinitionMismatch {
                        value: value.index(),
                    });
                }
            }
        }
    }
    for (value, count) in listed {
        if count > 1 {
            found.push(Violation::DefinedTwice { value });
        }
        if value >= function.value_bound() {
            found.push(Violation::OutOfBounds {
                what: "value",
                index: value,
            });
        }
    }
}

fn check_adjacency<D, F>(function: &F, blocks: &[F::Block], found: &mut Vec<Violation>)
where
    D: mlil::Dialect,
    F: mlil::FunctionView<D>,
{
    for &block in blocks {
        let mut records: Vec<usize> = mlil::FunctionView::<D>::successors(function, block)
            .map(|successor| successor.target.index())
            .collect();
        let mut adjacency: Vec<usize> = GraphView::successors(function, block)
            .map(DenseId::index)
            .collect();
        // The contract says the adjacency agrees with the records, which is
        // as a multiset: it does not fix an order.
        records.sort_unstable();
        adjacency.sort_unstable();
        if adjacency != records {
            found.push(Violation::AdjacencyMismatch {
                block: block.index(),
            });
        }
        let mut derived: Vec<usize> = blocks
            .iter()
            .flat_map(|&source| {
                mlil::FunctionView::<D>::successors(function, source)
                    .filter(|successor| successor.target == block)
                    .map(move |_| source.index())
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut listed: Vec<usize> = GraphView::predecessors(function, block)
            .map(DenseId::index)
            .collect();
        derived.sort_unstable();
        listed.sort_unstable();
        if derived != listed {
            found.push(Violation::PredecessorMismatch {
                block: block.index(),
            });
        }
    }
}

/// Where a definition sits: the block, and the position of its instruction
/// (a parameter sits before every instruction).
fn home<D, F>(function: &F, blocks: &[F::Block]) -> BTreeMap<usize, (usize, usize)>
where
    D: mlil::Dialect,
    F: mlil::FunctionView<D>,
{
    let mut home = BTreeMap::new();
    for &block in blocks {
        for &param in function.params(block) {
            home.insert(param.index(), (block.index(), 0));
        }
        for (position, &inst) in function.insts(block).iter().enumerate() {
            for &result in function.results(inst) {
                home.insert(result.index(), (block.index(), position + 1));
            }
        }
    }
    home
}

fn check_uses<D, F>(function: &F, blocks: &[F::Block], found: &mut Vec<Violation>)
where
    D: mlil::Dialect,
    F: mlil::FunctionView<D>,
{
    let dominators = DominatorTree::compute(function);
    let home = home::<D, F>(function, blocks);
    let mut check_use = |value: F::Value, block: F::Block, position: usize| {
        let Some(&(def_block, def_position)) = home.get(&value.index()) else {
            found.push(Violation::UndefinedOperand {
                value: value.index(),
            });
            return;
        };
        // A use in a block the entry cannot reach has no dominator to ask.
        if !dominators.is_reachable(block) {
            return;
        }
        let dominated = if def_block == block.index() {
            def_position <= position
        } else {
            dominators.dominates(F::Block::from_index(def_block), block)
        };
        if !dominated {
            found.push(Violation::NotDominated {
                value: value.index(),
                block: block.index(),
            });
        }
    };
    for &block in blocks {
        let insts = function.insts(block);
        for (position, &inst) in insts.iter().enumerate() {
            for &value in function.operands(inst) {
                check_use(value, block, position);
            }
        }
        // Successor arguments are read by the terminator, at the end.
        for successor in mlil::FunctionView::<D>::successors(function, block) {
            for &arg in successor.args {
                check_use(arg, block, insts.len());
            }
        }
    }
}
