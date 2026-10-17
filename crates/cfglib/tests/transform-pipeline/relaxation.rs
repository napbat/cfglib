//! Width relaxation of a linearized function: the branch-encoding fixed point
//! of a toy assembler, judged by brute-force enumeration of layouts.

use std::collections::BTreeMap;

use cfglib::{BlockOrder, LinearInst, RelaxError, relax_layout};

use crate::linearize::linearized;
use crate::programs::random_program;
use crate::toy::{Inst, Kind};

/// Bytes of one plain instruction; large enough that a few of them put a
/// branch target beyond a signed byte.
const PLAIN_WIDTH: usize = 24;

#[derive(Debug, Clone, PartialEq, Eq)]
struct Item {
    inst: Inst,
    wide: bool,
}

fn item_width(item: &Item) -> usize {
    match (item.inst.kind, item.wide) {
        (Kind::Label(_), _) => 0,
        (Kind::Jump(_) | Kind::BranchIf(_), false) => 2,
        (Kind::Jump(_), true) => 5,
        (Kind::BranchIf(_), true) => 6,
        _ => PLAIN_WIDTH,
    }
}

fn target_of(item: &Item) -> Option<usize> {
    match item.inst.kind {
        Kind::Jump(block) | Kind::BranchIf(block) => Some(block),
        _ => None,
    }
}

/// Whether a branch whose displacement is measured from its own end reaches
/// its label with a signed byte.
fn reaches_narrowly(offset: usize, target: usize) -> bool {
    let end = isize::try_from(offset + 2).unwrap();
    i8::try_from(isize::try_from(target).unwrap() - end).is_ok()
}

struct Assembled {
    items: Vec<Item>,
    offsets: Vec<usize>,
    total: usize,
    labels: BTreeMap<usize, usize>,
}

fn label_positions(items: &[Item]) -> BTreeMap<usize, usize> {
    items
        .iter()
        .enumerate()
        .filter_map(|(position, item)| match item.inst.kind {
            Kind::Label(block) => Some((block, position)),
            _ => None,
        })
        .collect()
}

fn resolve(positions: &BTreeMap<usize, usize>, offsets: &[usize]) -> BTreeMap<usize, usize> {
    positions
        .iter()
        .map(|(&block, &position)| (block, offsets[position]))
        .collect()
}

fn assemble(stream: &[LinearInst<Inst>]) -> Assembled {
    let mut items: Vec<Item> = stream
        .iter()
        .map(|entry| Item {
            inst: entry.inst.clone(),
            wide: false,
        })
        .collect();
    let positions = label_positions(&items);
    let (offsets, total, labels) = relax_layout(
        &mut items,
        |item, _offset| Ok::<usize, ()>(item_width(item)),
        |offsets, _total| Ok(resolve(&positions, offsets)),
        |item, offset, labels: &BTreeMap<usize, usize>| {
            let Some(block) = target_of(item) else {
                return Ok(false);
            };
            if !item.wide && !reaches_narrowly(offset, labels[&block]) {
                item.wide = true;
                return Ok(true);
            }
            Ok(false)
        },
    )
    .expect("the toy hooks cannot fail");
    Assembled {
        items,
        offsets,
        total,
        labels,
    }
}

fn assert_consistent(assembled: &Assembled, context: &str) {
    let Assembled {
        items,
        offsets,
        total,
        labels,
    } = assembled;
    let mut running = 0;
    for (item, &offset) in items.iter().zip(offsets) {
        assert_eq!(
            offset, running,
            "{context}: offsets are prefix sums of the final widths"
        );
        running += item_width(item);
    }
    assert_eq!(*total, running, "{context}");
    assert_eq!(
        labels,
        &resolve(&label_positions(items), offsets),
        "{context}: the returned labels were built from the final offsets"
    );
    for (item, &offset) in items.iter().zip(offsets) {
        if let (Some(block), false) = (target_of(item), item.wide) {
            assert!(
                reaches_narrowly(offset, labels[&block]),
                "{context}: narrow branch is out of range"
            );
        }
    }
}

/// Whether the assignment `wide` (one flag per branch, in stream order) is a
/// consistent layout: every branch left narrow reaches its label.
fn is_consistent_assignment(stream: &[LinearInst<Inst>], wide: &[bool]) -> bool {
    let mut branch = 0;
    let items: Vec<Item> = stream
        .iter()
        .map(|entry| {
            let is_branch = matches!(entry.inst.kind, Kind::Jump(_) | Kind::BranchIf(_));
            let item = Item {
                inst: entry.inst.clone(),
                wide: is_branch && wide[branch],
            };
            branch += usize::from(is_branch);
            item
        })
        .collect();
    let mut offsets = Vec::new();
    let mut running = 0;
    for item in &items {
        offsets.push(running);
        running += item_width(item);
    }
    let labels = resolve(&label_positions(&items), &offsets);
    items
        .iter()
        .zip(&offsets)
        .all(|(item, &offset)| match target_of(item) {
            Some(block) => item.wide || reaches_narrowly(offset, labels[&block]),
            None => true,
        })
}

#[test]
fn relaxing_a_linearized_function_yields_a_consistent_layout() {
    let mut widened = 0;
    for seed in 0..40 {
        let cfg = random_program::<()>(seed, false);
        let stream = linearized(&cfg, BlockOrder::ReversePostorder);
        let assembled = assemble(&stream);
        assert_consistent(&assembled, &format!("seed {seed}"));
        widened += assembled.items.iter().filter(|item| item.wide).count();
    }
    assert!(widened > 0, "the toy widths never forced a wide branch");
}

#[test]
fn relaxation_widens_the_fewest_branches_that_any_consistent_layout_allows() {
    let mut compared = 0;
    for seed in 0..120 {
        let cfg = random_program::<()>(seed, false);
        let stream = linearized(&cfg, BlockOrder::ReversePostorder);
        let branches = stream
            .iter()
            .filter(|entry| matches!(entry.inst.kind, Kind::Jump(_) | Kind::BranchIf(_)))
            .count();
        if branches > 12 {
            continue;
        }
        let assembled = assemble(&stream);
        let result: Vec<bool> = assembled
            .items
            .iter()
            .filter(|item| target_of(item).is_some())
            .map(|item| item.wide)
            .collect();
        for mask in 0_u32..(1 << branches) {
            let candidate: Vec<bool> = (0..branches).map(|bit| mask >> bit & 1 == 1).collect();
            if is_consistent_assignment(&stream, &candidate) {
                assert!(
                    result
                        .iter()
                        .zip(&candidate)
                        .all(|(&mine, &other)| !mine || other),
                    "seed {seed}: relaxation widened a branch that layout {candidate:?} keeps narrow"
                );
            }
        }
        compared += 1;
    }
    assert!(
        compared > 10,
        "only {compared} programs were small enough to enumerate"
    );
}

#[test]
fn widening_one_branch_can_push_an_earlier_one_out_of_reach() {
    #[derive(Clone, Copy)]
    enum Part {
        Branch { label: usize, wide: bool },
        Padding(usize),
    }
    let mut items = [
        Part::Branch {
            label: 0,
            wide: false,
        },
        Part::Branch {
            label: 1,
            wide: false,
        },
        Part::Padding(123),
        Part::Padding(1),
        Part::Padding(10),
    ];
    // Label 0 binds to the item after the big padding, label 1 to the end.
    let bindings = [3_usize, 5];
    let (offsets, total, labels) = relax_layout(
        &mut items,
        |part, _| {
            Ok::<_, ()>(match part {
                Part::Branch { wide: false, .. } => 2,
                Part::Branch { wide: true, .. } => 5,
                Part::Padding(width) => *width,
            })
        },
        |offsets, total| {
            Ok(bindings.map(|position| offsets.get(position).copied().unwrap_or(total)))
        },
        |part, offset, labels: &[usize; 2]| match part {
            Part::Branch { label, wide: false } if !reaches_narrowly(offset, labels[*label]) => {
                *part = Part::Branch {
                    label: *label,
                    wide: true,
                };
                Ok(true)
            }
            _ => Ok(false),
        },
    )
    .unwrap();

    assert!(
        items
            .iter()
            .take(2)
            .all(|part| matches!(part, Part::Branch { wide: true, .. }))
    );
    assert_eq!(offsets, [0, 5, 10, 133, 134]);
    assert_eq!(total, 144);
    assert_eq!(labels, [133, 144]);
}

#[test]
fn an_items_width_may_depend_on_where_it_lands() {
    let mut items = [3_usize, 0, 5];
    let (offsets, total, ()) = relax_layout(
        &mut items,
        |&alignment_or_size, offset| {
            // Item 1 pads to the next multiple of eight; others have fixed size.
            Ok::<_, ()>(if alignment_or_size == 0 {
                (8 - offset % 8) % 8
            } else {
                alignment_or_size
            })
        },
        |_, _| Ok(()),
        |_, _, ()| Ok(false),
    )
    .unwrap();

    assert_eq!(offsets, [0, 3, 8]);
    assert_eq!(total, 13);
}

#[test]
fn relaxing_nothing_returns_an_empty_layout_and_the_callers_context() {
    let mut items: [u8; 0] = [];
    let (offsets, total, context) = relax_layout(
        &mut items,
        |_, _| Ok::<usize, ()>(1),
        |offsets, total| Ok((offsets.len(), total)),
        |_, _, _| Ok(false),
    )
    .unwrap();
    assert_eq!(offsets, Vec::<usize>::new());
    assert_eq!((total, context), (0, (0, 0)));
}

#[test]
fn a_layout_longer_than_the_address_space_is_an_overflow() {
    let mut items = [0_u8; 2];
    let result = relax_layout(
        &mut items,
        |_, _| Ok::<usize, ()>(usize::MAX / 2 + 1),
        |_, _| Ok(()),
        |_, _, ()| Ok(false),
    );
    assert_eq!(result.unwrap_err(), RelaxError::Overflow);
}

#[test]
fn the_first_failing_hook_ends_relaxation_with_its_error() {
    let mut prepared = 0;
    let mut items = [1_usize, 2, 3];
    let by_width = relax_layout(
        &mut items,
        |&width, _| {
            if width == 2 {
                Err("width of item 2")
            } else {
                Ok(width)
            }
        },
        |_, _| {
            prepared += 1;
            Ok(())
        },
        |_, _, ()| Ok(false),
    );
    assert_eq!(by_width.unwrap_err(), RelaxError::Item("width of item 2"));
    assert_eq!(
        prepared, 0,
        "the context is never built from a failed layout"
    );

    let by_prepare = relax_layout(
        &mut items,
        |&width, _| Ok::<_, &str>(width),
        |_, _| Err::<(), _>("labels"),
        |_, _, ()| Err("widen"),
    );
    assert_eq!(by_prepare.unwrap_err(), RelaxError::Item("labels"));

    let by_widen = relax_layout(
        &mut items,
        |&width, _| Ok::<_, &str>(width),
        |_, _| Ok(()),
        |width, _, ()| {
            if *width == 2 {
                Err("widen item 2")
            } else {
                Ok(false)
            }
        },
    );
    assert_eq!(by_widen.unwrap_err(), RelaxError::Item("widen item 2"));

    let boxed: Box<dyn std::error::Error> = Box::new(RelaxError::Item(String::from("boom")));
    assert_ne!(boxed.to_string(), "");
}
