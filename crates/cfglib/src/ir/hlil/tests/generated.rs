//! Every small flat function, with handler regions and jump-only blocks,
//! lifted and verified over each toy dialect.
//!
//! The functions are every combination of terminators over three blocks,
//! each block with or without work, with a catch region over any block whose
//! pad and extent are any other blocks, reached by an exception edge or not.
//! Dead pads, pads inside loops, jump-only trampolines that the lift treats as
//! empty, and the blocks the sweep emits all occur. A function whose control
//! tree lists no unresolved label must lift and verify; one that lists a label
//! is the one case allowed to fail, and must fail: a goto to a block that has
//! no instruction has no label to find.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ControlTree;
use crate::block::BlockId;
use crate::ir::hlil::{
    FlatBuilder, FlatFunction, LiftError, LiftMetadata, lift, lift_trampolines, verify,
};
use crate::region::{Handler as RegionHandler, HandlerBody, HandlerKind, Region, RegionId};

use super::flat::{FlatKit, ToyEdge, ToyFlat, ToyFlatBuilder};
use super::flat_lift::{boolean, int};
use super::{Prim, ToyHlilBuilder};

over_toys!(every_small_function_with_a_region_lifts_or_reports_its_dangling_label);

/// How one block ends.
#[derive(Clone, Copy)]
enum Ending {
    Return,
    Jump(usize),
    Branch(usize, usize),
}

/// One block: whether it does work before it ends, and how it ends.
#[derive(Clone, Copy)]
struct Shape {
    work: bool,
    ending: Ending,
}

/// The shapes of one block over `count` blocks. A block that only jumps is a
/// trampoline.
fn shapes(count: usize) -> Vec<Shape> {
    let mut all = Vec::new();
    for work in [false, true] {
        for target in 0..count {
            all.push(Shape {
                work,
                ending: Ending::Jump(target),
            });
        }
    }
    all.push(Shape {
        work: false,
        ending: Ending::Return,
    });
    all.push(Shape {
        work: true,
        ending: Ending::Return,
    });
    for on_true in 0..count {
        for on_false in 0..count {
            all.push(Shape {
                work: true,
                ending: Ending::Branch(on_true, on_false),
            });
        }
    }
    all
}

/// The region of a function: a catch handler at `pad` with the blocks of
/// `extent`, over the block `protected`, with an exception edge into the pad
/// when `live`.
struct Guard {
    protected: usize,
    pad: usize,
    extent: Vec<usize>,
    live: bool,
}

/// Whether a block outside the pad's extent flows into it, which no handler
/// does: the pad is reached by its exception edge or from its own extent.
fn is_well_formed(shapes: &[Shape], guard: &Guard) -> bool {
    let mut owned = guard.extent.clone();
    owned.push(guard.pad);
    let enters_from_outside =
        |from: usize, to: usize| owned.contains(&to) && !owned.contains(&from);
    // The entry arrives from the synthetic root, outside every extent.
    if owned.contains(&0) {
        return false;
    }
    !shapes
        .iter()
        .enumerate()
        .any(|(from, shape)| match shape.ending {
            Ending::Return => false,
            Ending::Jump(target) => enters_from_outside(from, target),
            Ending::Branch(on_true, on_false) => {
                enters_from_outside(from, on_true) || enters_from_outside(from, on_false)
            }
        })
}

fn flat_function<D: FlatKit>(shapes: &[Shape], guard: &Guard) -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let blocks: Vec<BlockId> = (0..shapes.len()).map(|_| flat.new_block(None)).collect();
    let i = flat.int_variable(0);
    let n = flat.int_variable(0);
    let condition = flat.int_variable(1);
    for (index, shape) in shapes.iter().enumerate() {
        let block = blocks[index];
        if shape.work {
            flat.operate(
                block,
                Prim::Add,
                &[int::<D>(i), int::<D>(i)],
                &[int::<D>(i)],
            );
        }
        match shape.ending {
            Ending::Return => flat.give_back(block, &[i]),
            Ending::Jump(target) => {
                flat.jump(block);
                flat.connect(block, blocks[target], ToyEdge::Fall);
            }
            Ending::Branch(on_true, on_false) => {
                flat.operate(
                    block,
                    Prim::LessThan,
                    &[int::<D>(i), int::<D>(n)],
                    &[boolean::<D>(condition)],
                );
                flat.branch(block, condition);
                flat.connect(block, blocks[on_true], ToyEdge::True);
                flat.connect(block, blocks[on_false], ToyEdge::False);
            }
        }
    }
    flat.start(blocks[0]);
    if guard.live {
        flat.connect(blocks[guard.protected], blocks[guard.pad], ToyEdge::Except);
    }
    flat.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [blocks[guard.protected]].into_iter().collect(),
        handlers: vec![RegionHandler {
            entry: blocks[guard.pad],
            body: HandlerBody::known(
                core::iter::once(guard.pad)
                    .chain(guard.extent.iter().copied())
                    .map(|block| blocks[block]),
            ),
            kind: HandlerKind::CatchAll,
        }],
        parent: None,
    });
    flat.takes(&[i, n]);
    flat.finish_unchecked()
}

fn every_small_function_with_a_region_lifts_or_reports_its_dangling_label<D: FlatKit>() {
    let count = 3;
    let all = shapes(count);
    let mut choice = vec![0_usize; count];
    let mut lifted = 0_usize;
    let mut dangling = 0_usize;
    loop {
        let picked: Vec<Shape> = choice.iter().map(|&index| all[index]).collect();
        for protected in 0..count {
            for pad in (0..count).filter(|&block| block != protected) {
                let rest: Vec<usize> = (0..count)
                    .filter(|&block| block != protected && block != pad)
                    .collect();
                for subset in 0..(1_u32 << rest.len()) {
                    let extent: Vec<usize> = rest
                        .iter()
                        .enumerate()
                        .filter(|&(bit, _)| subset & (1 << bit) != 0)
                        .map(|(_, &block)| block)
                        .collect();
                    for live in [false, true] {
                        let guard = Guard {
                            protected,
                            pad,
                            extent: extent.clone(),
                            live,
                        };
                        if !is_well_formed(&picked, &guard) {
                            continue;
                        }
                        let flat = flat_function::<D>(&picked, &guard);
                        let trampolines = lift_trampolines::<D, _>(&flat);
                        let (_, report) = ControlTree::compute_with_report_treating_empty(
                            flat.cfg(),
                            &trampolines,
                        );
                        let result =
                            lift::<D, _, _>(&flat, ToyHlilBuilder::<D>::new(), LiftMetadata::Omit);
                        if report.unresolved_labels.is_empty() {
                            let lifted_function = result.unwrap_or_else(|error| {
                                panic!("lift failed for {}: {error}\n{report:?}", flat.dump())
                            });
                            assert!(
                                verify(&lifted_function.function).is_ok(),
                                "{}\n{}",
                                flat.dump(),
                                verify(&lifted_function.function)
                            );
                            lifted += 1;
                        } else {
                            assert!(
                                matches!(result, Err(LiftError::Builder(_))),
                                "a dangling label must fail verification: {}\n{report:?}",
                                flat.dump()
                            );
                            dangling += 1;
                        }
                    }
                }
            }
        }
        let mut digit = 0;
        while digit < count {
            choice[digit] += 1;
            if choice[digit] < all.len() {
                break;
            }
            choice[digit] = 0;
            digit += 1;
        }
        if digit == count {
            break;
        }
    }
    // Both outcomes occur, so neither branch of the check is vacuous.
    assert!(
        lifted > 0 && dangling > 0,
        "{lifted} lifted, {dangling} dangling"
    );
}
