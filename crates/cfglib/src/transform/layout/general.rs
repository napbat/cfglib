//! Relaxation for items whose widths may depend on anything: the caller
//! rebuilds its label context from every round's offsets.

extern crate alloc;

use alloc::vec::Vec;

use super::RelaxError;

/// Relaxes `items` to a fixed point of widths and offsets.
///
/// Each iteration lays the items out (`width`, which may consult the item's
/// own offset for alignment), builds the caller's label context from the
/// offsets and total length (`prepare` — typically label-to-offset
/// resolution), and offers every item the chance to widen (`widen`, which
/// mutates the item's form and reports whether it changed). When an
/// iteration widens nothing, the final `(offsets, total, context)` triple
/// is returned — offsets and context are mutually consistent, so the
/// caller can materialize and encode directly.
///
/// `widen` must be monotone: a widened item may never report a smaller
/// width later, or the loop need not terminate.
///
/// # Cost
///
/// Each iteration calls `width` and `widen` once per item and `prepare`
/// once, and the loop runs until an iteration widens nothing. A cascade in
/// which each widening pushes one more span out of range takes one
/// iteration per widening: O(n²) calls for `n` items. When every item
/// depends on one displacement at most, [`relax_spans`](crate::relax_spans)
/// reaches the same fixed point and re-checks only the spans that a width
/// change moved.
///
/// # Errors
///
/// Returns [`RelaxError::Overflow`] when the total length overflows, or
/// the first caller error from any hook.
pub fn relax_layout<I, C, E>(
    items: &mut [I],
    mut width: impl FnMut(&I, usize) -> Result<usize, E>,
    mut prepare: impl FnMut(&[usize], usize) -> Result<C, E>,
    mut widen: impl FnMut(&mut I, usize, &C) -> Result<bool, E>,
) -> Result<(Vec<usize>, usize, C), RelaxError<E>> {
    loop {
        let mut offsets = Vec::with_capacity(items.len());
        let mut total = 0_usize;
        for item in items.iter() {
            offsets.push(total);
            let item_width = width(item, total).map_err(RelaxError::Item)?;
            total = total.checked_add(item_width).ok_or(RelaxError::Overflow)?;
        }
        let context = prepare(&offsets, total).map_err(RelaxError::Item)?;
        let mut changed = false;
        for (position, item) in items.iter_mut().enumerate() {
            changed |= widen(item, offsets[position], &context).map_err(RelaxError::Item)?;
        }
        if !changed {
            return Ok((offsets, total, context));
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;

    use alloc::vec::Vec;

    use super::{RelaxError, relax_layout};

    /// A branch to a bound label that reaches `i8` deltas narrowly and
    /// widens otherwise, or fixed-width padding.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Item {
        Branch { label: usize, wide: bool },
        Padding(usize),
    }

    /// Labels bind to item positions (or one past the end for code end).
    fn layout(items: &mut [Item], bindings: &[usize]) -> (Vec<usize>, usize) {
        let (offsets, total, _labels) = relax_layout(
            items,
            |item, _offset| {
                Ok::<usize, ()>(match item {
                    Item::Branch { wide: false, .. } => 2,
                    Item::Branch { wide: true, .. } => 5,
                    Item::Padding(width) => *width,
                })
            },
            |offsets, total| {
                Ok(bindings
                    .iter()
                    .map(|&bound| offsets.get(bound).copied().unwrap_or(total))
                    .collect::<Vec<_>>())
            },
            |item, offset, labels: &Vec<usize>| {
                let Item::Branch { label, wide } = item else {
                    return Ok(false);
                };
                let delta =
                    isize::try_from(labels[*label]).unwrap() - isize::try_from(offset).unwrap();
                if !*wide && i8::try_from(delta).is_err() {
                    *wide = true;
                    return Ok(true);
                }
                Ok(false)
            },
        )
        .unwrap();
        (offsets, total)
    }

    #[test]
    fn near_branches_stay_narrow() {
        let mut items = [
            Item::Branch {
                label: 0,
                wide: false,
            },
            Item::Padding(8),
        ];
        let (offsets, total) = layout(&mut items, &[2]);
        assert_eq!(offsets, [0, 2]);
        assert_eq!(total, 10);
        assert!(matches!(items[0], Item::Branch { wide: false, .. }));
    }

    #[test]
    fn widening_cascades_until_the_fixed_point() {
        // The first branch barely reaches its label while everything stays
        // narrow; the second branch must widen for its own distance, which
        // moves the first branch's label out of reach in a later iteration.
        let mut items = [
            Item::Branch {
                label: 0,
                wide: false,
            },
            Item::Branch {
                label: 1,
                wide: false,
            },
            Item::Padding(123),
            Item::Padding(10),
        ];
        // Label 0 binds after the large padding; label 1 at code end.
        let (offsets, total) = layout(&mut items, &[3, 4]);
        assert!(matches!(items[1], Item::Branch { wide: true, .. }));
        assert!(
            matches!(items[0], Item::Branch { wide: true, .. }),
            "the second branch's widening pushed the first label out of range"
        );
        assert_eq!(offsets, [0, 5, 10, 133]);
        assert_eq!(total, 143);
    }

    #[test]
    fn widths_may_depend_on_the_item_offset() {
        // Each widened item aligns its start to a multiple of four before
        // occupying six units, so its width is a function of where it lands.
        let mut items = [0_usize, 0, 0];
        let (offsets, total, ()) = relax_layout(
            &mut items,
            |item, offset| {
                Ok::<usize, ()>(if *item == 0 {
                    3
                } else {
                    (offset.next_multiple_of(4) - offset) + 6
                })
            },
            |_, _| Ok(()),
            |item, _, ()| {
                if *item == 0 {
                    *item = 1;
                    return Ok(true);
                }
                Ok(false)
            },
        )
        .unwrap();
        assert_eq!(offsets, [0, 6, 14]);
        assert_eq!(total, 22, "interior items pad to the next 4 boundary");
    }

    #[test]
    fn caller_errors_and_overflow_are_reported() {
        let mut failing = [()];
        let result = relax_layout(
            &mut failing,
            |(), _| Err::<usize, _>("no width"),
            |_, _| Ok(()),
            |(), _, ()| Ok(false),
        );
        assert_eq!(result.unwrap_err(), RelaxError::Item("no width"));

        let mut overflowing = [(), ()];
        let result = relax_layout(
            &mut overflowing,
            |(), _| Ok::<usize, &str>(usize::MAX),
            |_, _| Ok(()),
            |(), _, ()| Ok(false),
        );
        assert_eq!(result.unwrap_err(), RelaxError::Overflow);
    }
}
