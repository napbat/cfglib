extern crate alloc;

use alloc::format;
use alloc::vec;
use alloc::vec::Vec;
use core::cell::Cell;
use core::convert::Infallible;

use super::{RelaxMode, SpanItem, relax_spans};
use crate::test_util::rng::Rng;
use crate::transform::layout::{RelaxError, relax_layout};

/// A toy assembler item: a branch whose 2-byte short form reaches a signed
/// byte from its end and whose 5-byte long form reaches anything, padding,
/// or a block aligned to `align` bytes that then takes `width` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Item {
    Branch { target: usize, long: bool },
    Padding(usize),
    Align { align: usize, width: usize },
}

impl SpanItem for Item {
    fn width(&self, offset: usize) -> usize {
        match *self {
            Item::Branch { long: false, .. } => 2,
            Item::Branch { long: true, .. } => 5,
            Item::Padding(width) => width,
            Item::Align { align, width } => offset.next_multiple_of(align) - offset + width,
        }
    }

    fn span_target(&self) -> Option<usize> {
        match *self {
            Item::Branch { target, .. } => Some(target),
            _ => None,
        }
    }

    fn is_aligned(&self) -> bool {
        matches!(self, Item::Align { .. })
    }

    fn refit(&mut self, displacement: isize, mode: RelaxMode) -> bool {
        let Item::Branch { long, .. } = self else {
            return false;
        };
        // x86 measures from the end of the branch. A forward target moves
        // with the branch's own width, so its distance from the end does
        // not depend on the form; a backward one does.
        let width = if *long { 5 } else { 2 };
        let short_from_end = if displacement > 0 {
            displacement - width
        } else {
            displacement - 2
        };
        let reaches_short = i8::try_from(short_from_end).is_ok();
        let refit = match mode {
            RelaxMode::Grow => !*long && !reaches_short,
            RelaxMode::Shrink => *long && reaches_short,
        };
        *long ^= refit;
        refit
    }
}

fn displacement(from: usize, to: usize) -> isize {
    isize::try_from(to).unwrap() - isize::try_from(from).unwrap()
}

/// The fixed point of `relax_layout` with `refit` as the widening.
fn relaxed_by_layout(items: &mut [Item]) -> (Vec<usize>, usize) {
    let (offsets, total, _) = relax_layout(
        items,
        |item, offset| Ok::<_, Infallible>(item.width(offset)),
        |offsets, total| Ok((offsets.to_vec(), total)),
        |item, offset, (offsets, total): &(Vec<usize>, usize)| {
            Ok(item.span_target().is_some_and(|target| {
                let to = offsets.get(target).copied().unwrap_or(*total);
                item.refit(displacement(offset, to), RelaxMode::Grow)
            }))
        },
    )
    .unwrap();
    (offsets, total)
}

/// Checks that the offsets lay the items out and every span fits its form.
fn assert_fixed_point(items: &[Item], offsets: &[usize], total: usize, context: &str) {
    let mut offset = 0;
    for (index, item) in items.iter().enumerate() {
        assert_eq!(offsets[index], offset, "{context}: offset of item {index}");
        offset += item.width(offset);
    }
    assert_eq!(total, offset, "{context}: total");
    for (index, item) in items.iter().enumerate() {
        if let Some(target) = item.span_target() {
            let to = offsets.get(target).copied().unwrap_or(total);
            let mut probe = item.clone();
            assert!(
                !probe.refit(displacement(offsets[index], to), RelaxMode::Grow),
                "{context}: item {index} does not reach its target"
            );
        }
    }
}

/// Up to 60 items: branches in the form `long`, padding, and aligned blocks
/// when `aligned`. A branch targets any item or the end, or only a later one
/// when `forward`.
fn random_items(rng: &mut Rng, long: bool, aligned: bool, forward: bool) -> Vec<Item> {
    let count = rng.below(60);
    (0..count)
        .map(|index| match rng.below(if aligned { 10 } else { 9 }) {
            0..=4 => Item::Branch {
                target: if forward {
                    index + 1 + rng.below(count - index)
                } else {
                    rng.below(count + 1)
                },
                long,
            },
            5..=8 => Item::Padding(rng.below(48)),
            _ => Item::Align {
                align: [4, 8, 16, 64][rng.below(4)],
                width: rng.below(8),
            },
        })
        .collect()
}

fn relax(items: &mut [Item], mode: RelaxMode) -> (Vec<usize>, usize) {
    relax_spans(items, mode).expect("toy layouts stay small")
}

#[test]
fn grow_mode_reaches_the_fixed_point_of_relax_layout() {
    let mut rng = Rng::new(71);
    for round in 0..1_000 {
        let aligned = round % 2 == 1;
        let original = random_items(&mut rng, false, aligned, false);
        let mut spans = original.clone();
        let mut reference = original.clone();
        let context = format!("round {round}: {original:?}");

        let relaxed = relax(&mut spans, RelaxMode::Grow);
        assert_eq!(relaxed, relaxed_by_layout(&mut reference), "{context}");
        assert_eq!(spans, reference, "{context}: the same forms");
        assert_fixed_point(&spans, &relaxed.0, relaxed.1, &context);
    }
}

/// A chain of `count` branches, each over the next one, then a padding of
/// `tail` bytes. A branch reaches 3 bytes past its own end in its short
/// form, so it needs the long form exactly when the next branch is long.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Chained {
    Branch { target: usize, long: bool },
    Tail(usize),
}

impl Chained {
    fn chain(count: usize, long: bool, tail: usize) -> Vec<Self> {
        let mut items: Vec<Self> = (0..count)
            .map(|index| Chained::Branch {
                target: index + 2,
                long,
            })
            .collect();
        items.push(Chained::Tail(tail));
        items
    }

    fn width(&self) -> usize {
        match *self {
            Chained::Branch { long, .. } => {
                if long {
                    5
                } else {
                    2
                }
            }
            Chained::Tail(width) => width,
        }
    }

    fn refit(&mut self, displacement: isize, mode: RelaxMode) -> bool {
        let width = isize::try_from(self.width()).unwrap();
        let Chained::Branch { long, .. } = self else {
            return false;
        };
        let reaches_short = displacement - width <= 3;
        let refit = match mode {
            RelaxMode::Grow => !*long && !reaches_short,
            RelaxMode::Shrink => *long && reaches_short,
        };
        *long ^= refit;
        refit
    }
}

/// A [`Chained`] item that counts the calls of `width` and `refit`.
struct Counted<'c> {
    item: Chained,
    calls: &'c Cell<usize>,
}

impl SpanItem for Counted<'_> {
    fn width(&self, _offset: usize) -> usize {
        self.calls.set(self.calls.get() + 1);
        self.item.width()
    }

    fn span_target(&self) -> Option<usize> {
        match self.item {
            Chained::Branch { target, .. } => Some(target),
            Chained::Tail(_) => None,
        }
    }

    fn is_aligned(&self) -> bool {
        false
    }

    fn refit(&mut self, displacement: isize, mode: RelaxMode) -> bool {
        self.calls.set(self.calls.get() + 1);
        self.item.refit(displacement, mode)
    }
}

/// Relaxes a chain of `count` branches with `relax_spans` and returns the
/// final forms and the number of calls.
fn counted_spans(count: usize, mode: RelaxMode) -> (Vec<Chained>, usize) {
    let calls = Cell::new(0);
    let (long, tail) = match mode {
        RelaxMode::Grow => (false, 100),
        RelaxMode::Shrink => (true, 1),
    };
    let mut items: Vec<Counted<'_>> = Chained::chain(count, long, tail)
        .into_iter()
        .map(|item| Counted {
            item,
            calls: &calls,
        })
        .collect();
    relax_spans(&mut items, mode).unwrap();
    (
        items.into_iter().map(|item| item.item).collect(),
        calls.get(),
    )
}

#[test]
fn a_cascade_costs_n_log_n_calls_where_relax_layout_costs_n_squared() {
    const COUNT: usize = 1_024;
    let bound = 2 * COUNT * usize::try_from(COUNT.ilog2()).unwrap();
    for (mode, final_long) in [(RelaxMode::Grow, true), (RelaxMode::Shrink, false)] {
        let (items, calls) = counted_spans(COUNT, mode);
        assert!(
            items
                .iter()
                .all(|item| !matches!(item, Chained::Branch { long, .. } if *long != final_long)),
            "{mode:?}: every branch changes form in the cascade"
        );
        assert!(calls <= bound, "{mode:?}: {calls} calls, bound {bound}");
    }

    // The baseline: relax_layout lays out and offers to widen every item in
    // each of the COUNT rounds of the cascade.
    let calls = Cell::new(0);
    let mut items = Chained::chain(COUNT, false, 100);
    relax_layout(
        &mut items,
        |item, _| {
            calls.set(calls.get() + 1);
            Ok::<_, Infallible>(item.width())
        },
        |offsets, total| {
            let mut labels = offsets.to_vec();
            labels.push(total);
            Ok(labels)
        },
        |item, offset, labels: &Vec<usize>| {
            calls.set(calls.get() + 1);
            let Chained::Branch { target, .. } = *item else {
                return Ok(false);
            };
            Ok(item.refit(displacement(offset, labels[target]), RelaxMode::Grow))
        },
    )
    .unwrap();
    let (spans, _) = counted_spans(COUNT, RelaxMode::Grow);
    assert_eq!(items, spans, "the same fixed point");
    assert!(
        calls.get() >= COUNT * COUNT,
        "relax_layout made {} calls",
        calls.get()
    );
}

#[test]
fn shrink_mode_reaches_a_valid_layout_no_larger_than_the_long_one() {
    let mut rng = Rng::new(72);
    for round in 0..1_000 {
        let forward = round % 2 == 1;
        let long = random_items(&mut rng, true, false, forward);
        let short: Vec<Item> = long
            .iter()
            .map(|item| match *item {
                Item::Branch { target, .. } => Item::Branch {
                    target,
                    long: false,
                },
                ref other => other.clone(),
            })
            .collect();
        let context = format!("round {round}: {long:?}");
        let long_total: usize = long.iter().map(|item| item.width(0)).sum();

        let mut shrunk = long.clone();
        let (offsets, total) = relax(&mut shrunk, RelaxMode::Shrink);
        assert_fixed_point(&shrunk, &offsets, total, &context);
        let mut grown = short;
        let grown_layout = relax(&mut grown, RelaxMode::Grow);
        assert!(
            grown_layout.1 <= total && total <= long_total,
            "{context}: {} <= {total} <= {long_total}",
            grown_layout.1
        );
        if forward {
            // A forward span depends only on later items, so the fixed point
            // is unique.
            assert_eq!((offsets, total), grown_layout, "{context}");
            assert_eq!(shrunk, grown, "{context}");
        }
    }
}

#[test]
fn shrink_mode_with_alignment_terminates_at_a_fixed_point() {
    let mut rng = Rng::new(73);
    for round in 0..1_000 {
        let mut items = random_items(&mut rng, true, true, false);
        let context = format!("round {round}: {items:?}");
        let (offsets, total) = relax(&mut items, RelaxMode::Shrink);
        assert_fixed_point(&items, &offsets, total, &context);
    }
}

#[test]
fn a_shrink_before_an_aligned_block_grows_a_span_back() {
    // Fifteen branches shrink by 3 bytes each and move the backward target
    // 45 bytes closer to the start. The aligned block absorbs the shift, so
    // the branch after it stays at 192 and no longer reaches its target.
    let mut items: Vec<Item> = (1..=15)
        .map(|next| Item::Branch {
            target: next,
            long: true,
        })
        .collect();
    items.extend([
        Item::Padding(0),
        Item::Padding(100),
        Item::Align {
            align: 64,
            width: 0,
        },
        Item::Branch {
            target: 15,
            long: true,
        },
    ]);
    let mut long = items.clone();
    let (long_offsets, _) = relax(&mut long, RelaxMode::Grow);
    assert_eq!(long, items, "every branch starts in its longest form");
    assert_eq!(
        long_offsets[18] - long_offsets[15],
        117,
        "fits short at first"
    );

    let (offsets, total) = relax(&mut items, RelaxMode::Shrink);
    assert_fixed_point(&items, &offsets, total, "regrowth");
    assert!(
        items[..15]
            .iter()
            .all(|item| matches!(item, Item::Branch { long: false, .. })),
        "{items:?}"
    );
    assert_eq!(
        items[18],
        Item::Branch {
            target: 15,
            long: true
        },
        "grown back for good"
    );
    assert_eq!((offsets[15], offsets[18], total), (30, 192, 197));
}

#[test]
fn shrink_mode_can_stop_above_the_least_layout() {
    // Each branch covers the other, and each reaches in its short form only
    // when the other is short: both stay long from the long forms.
    let layout = |long| {
        vec![
            Item::Branch { target: 3, long },
            Item::Padding(123),
            Item::Branch { target: 0, long },
        ]
    };
    let mut grown = layout(false);
    let mut shrunk = layout(true);
    let (_, grown_total) = relax(&mut grown, RelaxMode::Grow);
    let (offsets, shrunk_total) = relax(&mut shrunk, RelaxMode::Shrink);
    assert_eq!(grown, layout(false));
    assert_eq!(shrunk, layout(true));
    assert_fixed_point(&shrunk, &offsets, shrunk_total, "crossing");
    assert_eq!((grown_total, shrunk_total), (127, 133));
}

#[test]
fn empty_self_and_end_spans_need_no_special_case() {
    assert_eq!(relax(&mut [], RelaxMode::Grow), (Vec::new(), 0));
    let mut items = [
        Item::Branch {
            target: 0,
            long: true,
        },
        Item::Branch {
            target: 2,
            long: true,
        },
    ];
    let (offsets, total) = relax(&mut items, RelaxMode::Shrink);
    assert_eq!((offsets.as_slice(), total), ([0, 2].as_slice(), 4));
    assert!(matches!(
        items,
        [
            Item::Branch { long: false, .. },
            Item::Branch { long: false, .. }
        ]
    ));
}

#[test]
fn lengths_and_displacements_beyond_the_address_space_are_errors() {
    let mut too_long = [Item::Padding(usize::MAX), Item::Padding(1)];
    assert_eq!(
        relax_spans(&mut too_long, RelaxMode::Grow),
        Err(RelaxError::Overflow)
    );
    let mut too_far = [
        Item::Branch {
            target: 2,
            long: false,
        },
        Item::Padding(usize::MAX / 2 + 10),
    ];
    assert_eq!(
        relax_spans(&mut too_far, RelaxMode::Grow),
        Err(RelaxError::Overflow)
    );
}

#[test]
#[should_panic(expected = "span target 5 of item 0 is beyond the 1 items")]
fn a_span_target_beyond_the_end_panics() {
    let mut items = [Item::Branch {
        target: 5,
        long: false,
    }];
    let _ = relax_spans(&mut items, RelaxMode::Grow);
}
