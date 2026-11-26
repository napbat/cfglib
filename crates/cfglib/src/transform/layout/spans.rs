//! Incremental relaxation of span-dependent items.
//!
//! [`relax_spans`] reaches the fixed point of
//! [`relax_layout`](super::relax_layout) for items whose only dependency is
//! the displacement to one other item, without laying every item out again
//! each round. Offsets live in a Fenwick tree over the widths, and a segment
//! tree indexes the spans by the items they cover, so that a width change
//! re-checks only the spans over the changed item and re-reads only the
//! aligned items after it.

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use core::convert::Infallible;

use super::RelaxError;
use crate::graph::fanout::DenseFanout;

#[cfg(test)]
mod tests;

/// How [`relax_spans`] changes the forms of span items.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RelaxMode {
    /// Items start in the forms that the caller gives them, normally their
    /// shortest, and only grow, as in an assembler that starts every branch
    /// short.
    Grow,
    /// Items start in the forms that the caller gives them, normally their
    /// longest, and shrink where a shorter form reaches, as in linker
    /// relaxation.
    Shrink,
}

/// An item of a linear layout whose width can depend on the displacement to
/// one other item: a branch, or anything that occupies bytes.
///
/// The displacement of an item is the offset of its
/// [`span_target`](Self::span_target) minus its own offset, so it includes
/// the item's own width when the target follows it. A target form that
/// measures from the end of the item subtracts its width in
/// [`refit`](Self::refit).
pub trait SpanItem {
    /// The width of the item when it starts at `offset`.
    ///
    /// The width depends on `offset` only when
    /// [`is_aligned`](Self::is_aligned) holds; otherwise it depends on the
    /// form alone.
    fn width(&self, offset: usize) -> usize;

    /// The index of the item that the displacement is measured to, or `None`
    /// for an item without a span. The length of the item slice names the
    /// end of the layout.
    fn span_target(&self) -> Option<usize>;

    /// Whether the width depends on the offset, as for alignment padding. An
    /// aligned item is read again whenever an item before it changes width.
    fn is_aligned(&self) -> bool;

    /// Changes the form of the item to fit `displacement`, and returns
    /// whether the form changed.
    ///
    /// With [`RelaxMode::Grow`], a form that does not reach `displacement`
    /// becomes a longer one that does; a form that reaches stays. With
    /// [`RelaxMode::Shrink`], a form becomes a shorter one that reaches
    /// `displacement`, if there is one. A call must never move the other
    /// way, and a form whose width is 0 is allowed: it models a jump that a
    /// fallthrough replaced.
    fn refit(&mut self, displacement: isize, mode: RelaxMode) -> bool;
}

/// Relaxes span items to a fixed point of forms and offsets, and returns the
/// offsets and the total length.
///
/// In [`RelaxMode::Grow`] the caller gives the items their shortest forms,
/// and each round grows the span items that do not reach their targets. This
/// is the fixed point of [`relax_layout`](super::relax_layout) with `refit`
/// as its widening: both check every span against the layout of the round,
/// so they widen the same items in the same rounds. Without aligned items, a
/// wider item only lengthens the spans over it, so this is the least fixed
/// point: every layout in which all spans fit has each item at least as
/// wide.
///
/// In [`RelaxMode::Shrink`] the caller gives the items their longest forms,
/// and each round shrinks the span items that a shorter form fits. A shrink
/// before an aligned item can move a later item away from a span that
/// crosses the aligned item. Such an item grows back through `refit` with
/// [`RelaxMode::Grow`] and does not shrink again, so each item changes form
/// in one direction and then the other, and the loop ends. The result is a
/// fixed point but need not be the least one: two crossing spans that each
/// reach in a short form only when the other is short stay long.
///
/// In both modes, at the fixed point every span fits its form: `refit` with
/// [`RelaxMode::Grow`] and the final displacement changes nothing.
///
/// # Cost
///
/// A round re-checks only the span items that changed form and the spans
/// that cover an item whose width changed, and re-reads the width of those
/// items and of the aligned items after them. Offsets come from a Fenwick
/// tree and covering spans from a segment tree, so a round costs
/// O((c + s + a) log n) for `c` changed items, `s` re-checked spans, and `a`
/// re-read aligned items. A cascade in which every widening pushes one more
/// span out of range costs O(n log n) here and O(n²) callbacks in
/// [`relax_layout`](super::relax_layout), which lays out every item each
/// round. `span_target` and `is_aligned` are read once per item.
///
/// # Errors
///
/// Returns [`RelaxError::Overflow`] when the total length overflows `usize`
/// or a displacement overflows `isize`.
///
/// # Panics
///
/// Panics when an item names a span target beyond `items.len()`.
///
/// # Examples
///
/// The second jump does not reach the end of the long body in its short
/// form. Once it grows, the first jump, which jumps over it, no longer
/// reaches either.
///
/// ```
/// use cfglib::{RelaxMode, SpanItem, relax_spans};
///
/// enum Item {
///     /// A jump to an item, 2 bytes short and 5 bytes long; a short jump
///     /// reaches 127 bytes past its own end.
///     Jump { target: usize, long: bool },
///     Body(usize),
/// }
///
/// impl SpanItem for Item {
///     fn width(&self, _offset: usize) -> usize {
///         match self {
///             Item::Jump { long, .. } => if *long { 5 } else { 2 },
///             Item::Body(width) => *width,
///         }
///     }
///     fn span_target(&self) -> Option<usize> {
///         match self {
///             Item::Jump { target, .. } => Some(*target),
///             Item::Body(_) => None,
///         }
///     }
///     fn is_aligned(&self) -> bool {
///         false
///     }
///     fn refit(&mut self, displacement: isize, mode: RelaxMode) -> bool {
///         let Item::Jump { long, .. } = self else { return false };
///         let from_end = displacement - if *long { 5 } else { 2 };
///         let reaches_short = from_end <= 127;
///         let refit = match mode {
///             RelaxMode::Grow => !*long && !reaches_short,
///             RelaxMode::Shrink => *long && reaches_short,
///         };
///         *long ^= refit;
///         refit
///     }
/// }
///
/// let mut items = [
///     Item::Jump { target: 3, long: false },
///     Item::Jump { target: 4, long: false },
///     Item::Body(124),
///     Item::Body(10),
/// ];
/// let (offsets, total) = relax_spans(&mut items, RelaxMode::Grow)?;
/// assert_eq!(offsets, [0, 5, 10, 134]);
/// assert_eq!(total, 144);
/// # Ok::<(), cfglib::RelaxError<core::convert::Infallible>>(())
/// ```
pub fn relax_spans<I: SpanItem>(
    items: &mut [I],
    mode: RelaxMode,
) -> Result<(Vec<usize>, usize), RelaxError<Infallible>> {
    let count = items.len();
    let targets: Vec<Option<usize>> = items.iter().map(SpanItem::span_target).collect();
    for (item, target) in targets.iter().enumerate() {
        if let &Some(target) = target {
            assert!(
                target <= count,
                "span target {target} of item {item} is beyond the {count} items"
            );
        }
    }
    let aligned: Vec<usize> = (0..count)
        .filter(|&item| items[item].is_aligned())
        .collect();
    let covering = Covering::new(
        count,
        targets.iter().enumerate().filter_map(|(item, &target)| {
            target.map(|target| (item, item.min(target), item.max(target)))
        }),
    );

    let mut layout = Layout::new(items)?;
    let mut frozen = vec![false; count];
    let mut queued = vec![0_usize; count];
    let mut queue: Vec<usize> = (0..count).filter(|&item| targets[item].is_some()).collect();
    let mut changed = Vec::new();
    let mut moved = Vec::new();
    for round in 1.. {
        changed.clear();
        for &item in &queue {
            let target = targets[item].expect("only span items are queued");
            let displacement = layout.displacement(item, target)?;
            let item_changed = match mode {
                RelaxMode::Grow => items[item].refit(displacement, RelaxMode::Grow),
                RelaxMode::Shrink if frozen[item] => {
                    items[item].refit(displacement, RelaxMode::Grow)
                }
                RelaxMode::Shrink => {
                    if items[item].refit(displacement, RelaxMode::Shrink) {
                        true
                    } else {
                        // A shrink elsewhere moved the target out of reach:
                        // grow back, for good.
                        frozen[item] = items[item].refit(displacement, RelaxMode::Grow);
                        frozen[item]
                    }
                }
            };
            if item_changed {
                changed.push(item);
            }
        }
        if changed.is_empty() {
            break;
        }

        moved.clear();
        layout.reread(items, &changed, &aligned, &mut moved)?;
        queue.clear();
        let mut enqueue = |item: usize| {
            if queued[item] != round {
                queued[item] = round;
                queue.push(item);
            }
        };
        changed.iter().copied().for_each(&mut enqueue);
        for &position in &moved {
            covering.spans_over(position, &mut enqueue);
        }
        queue.sort_unstable();
    }
    Ok(layout.into_offsets())
}

/// The widths of the items, with their prefix sums in a Fenwick tree.
struct Layout {
    widths: Vec<usize>,
    /// `tree[k - 1]` sums the widths of items `k - k.isolate_lowest_one()`
    /// through `k - 1`.
    tree: Vec<usize>,
    total: usize,
}

impl Layout {
    /// Lays the items out from offset zero.
    fn new<I: SpanItem>(items: &[I]) -> Result<Self, RelaxError<Infallible>> {
        let mut widths = Vec::with_capacity(items.len());
        let mut total = 0_usize;
        for item in items {
            let width = item.width(total);
            widths.push(width);
            total = total.checked_add(width).ok_or(RelaxError::Overflow)?;
        }
        // Each node adds itself to its parent once: O(n) construction.
        let mut tree = widths.clone();
        for node in 1..=tree.len() {
            let parent = node + node.isolate_lowest_one();
            if parent <= tree.len() {
                tree[parent - 1] += tree[node - 1];
            }
        }
        Ok(Self {
            widths,
            tree,
            total,
        })
    }

    /// The offset of `item`; `widths.len()` names the end.
    fn offset(&self, item: usize) -> usize {
        let mut sum = 0;
        let mut node = item;
        while node > 0 {
            sum += self.tree[node - 1];
            node &= node - 1;
        }
        sum
    }

    fn displacement(&self, item: usize, target: usize) -> Result<isize, RelaxError<Infallible>> {
        let (from, to) = (self.offset(item), self.offset(target));
        let distance =
            |magnitude: usize| isize::try_from(magnitude).map_err(|_| RelaxError::Overflow);
        if to >= from {
            distance(to - from)
        } else {
            distance(from - to).map(|magnitude| -magnitude)
        }
    }

    /// Sets the width of `item`. The new total has been checked to fit.
    fn set_width(&mut self, item: usize, width: usize) {
        let old = core::mem::replace(&mut self.widths[item], width);
        let mut node = item + 1;
        while node <= self.tree.len() {
            if width >= old {
                self.tree[node - 1] += width - old;
            } else {
                self.tree[node - 1] -= old - width;
            }
            node += node.isolate_lowest_one();
        }
    }

    /// Reads again the widths of the `changed` items and of the `aligned`
    /// items after them whose offset moved, in item order, and pushes every
    /// item whose width changed onto `moved`.
    fn reread<I: SpanItem>(
        &mut self,
        items: &[I],
        changed: &[usize],
        aligned: &[usize],
        moved: &mut Vec<usize>,
    ) -> Result<(), RelaxError<Infallible>> {
        // How far the items already read again moved every offset after
        // them, wrapping in `usize`. The old and the new offset both fit
        // `usize`, so it is zero exactly when they are equal.
        let mut shift = 0_usize;
        let (mut next_changed, mut next_aligned) = (0, 0);
        loop {
            if shift == 0 {
                // No offset moved: the aligned items before the next changed
                // item keep their widths.
                let Some(&item) = changed.get(next_changed) else {
                    break;
                };
                next_aligned += aligned[next_aligned..].partition_point(|&other| other < item);
            }
            let item = match (changed.get(next_changed), aligned.get(next_aligned)) {
                (Some(&first), Some(&second)) => first.min(second),
                (Some(&item), None) | (None, Some(&item)) => item,
                (None, None) => break,
            };
            next_changed += usize::from(changed.get(next_changed) == Some(&item));
            next_aligned += usize::from(aligned.get(next_aligned) == Some(&item));

            let old = self.widths[item];
            let width = items[item].width(self.offset(item));
            if width != old {
                self.total = if width > old {
                    self.total
                        .checked_add(width - old)
                        .ok_or(RelaxError::Overflow)?
                } else {
                    self.total - (old - width)
                };
                self.set_width(item, width);
                shift = shift.wrapping_add(width).wrapping_sub(old);
                moved.push(item);
            }
        }
        Ok(())
    }

    fn into_offsets(self) -> (Vec<usize>, usize) {
        let mut offset = 0;
        let offsets = self
            .widths
            .iter()
            .map(|&width| {
                let start = offset;
                offset += width;
                start
            })
            .collect();
        (offsets, self.total)
    }
}

/// The spans indexed by the items they cover, in a segment tree: the span of
/// an item from `low` to `high` covers items `low..high`, whose width is in
/// its displacement.
struct Covering {
    /// The number of leaves, a power of two.
    leaves: usize,
    /// The spans stored at each tree node.
    spans: DenseFanout<usize>,
}

impl Covering {
    fn new(count: usize, spans: impl Iterator<Item = (usize, usize, usize)>) -> Self {
        let leaves = count.next_power_of_two();
        let mut stored = Vec::new();
        for (span, low, high) in spans {
            Self::decompose(leaves, low, high, |node| stored.push((node, span)));
        }
        Self {
            leaves,
            spans: DenseFanout::from_pairs(2 * leaves, stored),
        }
    }

    /// Calls `visit` with each tree node of the canonical cover of
    /// `low..high`.
    fn decompose(leaves: usize, low: usize, high: usize, mut visit: impl FnMut(usize)) {
        let (mut low, mut high) = (low + leaves, high + leaves);
        while low < high {
            if low & 1 == 1 {
                visit(low);
                low += 1;
            }
            if high & 1 == 1 {
                high -= 1;
                visit(high);
            }
            low >>= 1;
            high >>= 1;
        }
    }

    /// Calls `visit` with every span that covers `item`.
    fn spans_over(&self, item: usize, mut visit: impl FnMut(usize)) {
        let mut node = item + self.leaves;
        while node > 0 {
            for &span in self.spans.get(node) {
                visit(span);
            }
            node >>= 1;
        }
    }
}
