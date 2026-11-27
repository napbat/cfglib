//! The steps after a bounded path enumeration: keep the complete paths that
//! no other one shadows, and index the sources by the targets their paths
//! reach.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;
use smallvec::{Array, SmallVec};

/// Keeps, in order, the items that no other item shadows.
///
/// `shadows(other, item)` says whether `other` shadows `item`; no item is
/// compared with itself. Each item is tested against every other one,
/// including the ones the filter removes, so the relation need not be
/// transitive or acyclic: two items that shadow each other both go. That
/// costs `n * (n - 1)` calls for `n` items.
pub(crate) fn retain_unshadowed<A: Array>(
    items: &mut SmallVec<A>,
    mut shadows: impl FnMut(&A::Item, &A::Item) -> bool,
) {
    let count = items.len();
    if count < 2 {
        return;
    }
    let keep: SmallVec<[bool; 16]> = (0..count)
        .map(|index| {
            !(0..count).any(|other| other != index && shadows(&items[other], &items[index]))
        })
        .collect();
    let mut keep = keep.into_iter();
    items.retain(|_| keep.next().is_some_and(|kept| kept));
}

/// For each target below `bound`, the sources whose paths reach it: each
/// source once, in the order `sources` yields them.
///
/// `sources` yields each source once, with the targets of all its paths;
/// a target may repeat within one source.
///
/// # Panics
///
/// Panics when a target is not below `bound`.
pub(crate) fn sources_by_target<T, I>(
    bound: usize,
    sources: impl IntoIterator<Item = (T, I)>,
) -> Vec<Vec<T>>
where
    T: Copy + PartialEq,
    I: IntoIterator<Item = usize>,
{
    let mut lists = vec![Vec::new(); bound];
    for (source, targets) in sources {
        for target in targets {
            let list: &mut Vec<T> = &mut lists[target];
            // The sources come one at a time, so a repeat of this source's
            // target sits at the end of its list.
            if list.last() != Some(&source) {
                list.push(source);
            }
        }
    }
    lists
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_item_survives_unless_another_one_shadows_it() {
        let mut items: SmallVec<[u32; 4]> = SmallVec::from_slice(&[3, 1, 4, 1, 5]);
        // A smaller number shadows a larger one; equal numbers keep each other.
        retain_unshadowed(&mut items, |other, item| other < item);
        assert_eq!(items.as_slice(), [1, 1]);
    }

    #[test]
    fn mutual_shadowing_removes_both_items() {
        let mut items: SmallVec<[u32; 4]> = SmallVec::from_slice(&[1, 2, 7]);
        retain_unshadowed(&mut items, |other, item| other != item && *item < 3);
        assert_eq!(items.as_slice(), [7]);
    }

    #[test]
    fn each_target_lists_each_source_once_in_source_order() {
        let lists = sources_by_target(4, [('a', vec![2, 0, 2]), ('b', vec![]), ('c', vec![2])]);
        assert_eq!(lists, [vec!['a'], vec![], vec!['a', 'c'], vec![]]);
    }
}
