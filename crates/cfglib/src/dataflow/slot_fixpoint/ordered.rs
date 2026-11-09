//! The ordered driver: it evaluates units one at a time in sweep order, and
//! each evaluation reports the slots that it read.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;
use core::mem;

use super::SlotFixpointStats;
use crate::dataflow::bits::DenseBits;

/// The slots that one evaluation read and the slots that it changed.
///
/// [`ordered_fixpoint`] gives each evaluation an empty access. A read slot
/// is a dependency: the unit runs again after a later change of the slot. A
/// changed slot queues every unit whose latest evaluation read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotAccess<K> {
    reads: Vec<K>,
    changes: Vec<K>,
}

impl<K> SlotAccess<K> {
    const fn new() -> Self {
        Self {
            reads: Vec::new(),
            changes: Vec::new(),
        }
    }

    /// Records that the evaluation read slot `key`. A repeated read counts
    /// once.
    pub fn read(&mut self, key: K) {
        self.reads.push(key);
    }

    /// Records that the evaluation changed slot `key`.
    pub fn changed(&mut self, key: K) {
        self.changes.push(key);
    }
}

/// A queue of unit indices without duplicates. The lowest index comes out
/// first. The queue is a bit row, so a push allocates nothing.
struct Queue {
    queued: DenseBits,
    /// The words below this index hold no queued unit.
    cursor: usize,
}

impl Queue {
    fn new(bound: usize) -> Self {
        Self {
            queued: DenseBits::new(bound),
            cursor: 0,
        }
    }

    fn full(bound: usize) -> Self {
        let mut queue = Self::new(bound);
        for index in 0..bound {
            queue.insert(index);
        }
        queue
    }

    fn insert(&mut self, index: usize) {
        self.cursor = self.cursor.min(index / 64);
        self.queued.insert(index);
    }

    fn pop_first(&mut self) -> Option<usize> {
        let words = self.queued.words();
        while let Some(&word) = words.get(self.cursor) {
            if word != 0 {
                let index = self.cursor * 64 + word.trailing_zeros() as usize;
                self.queued.remove(index);
                return Some(index);
            }
            self.cursor += 1;
        }
        None
    }
}

/// Runs the units `0..units` to a fixpoint in ordered sweeps.
///
/// A sweep evaluates its dirty units in ascending order, one at a time. Thus
/// every later evaluation of a sweep sees a slot that an earlier evaluation
/// of that sweep changed. In sweep 0 every unit is dirty. Each evaluation
/// records the slots that it read and changed in its [`SlotAccess`]. Its
/// reads replace the dependencies of its previous evaluation. When it
/// changes a slot, each unit whose latest evaluation read that slot becomes
/// dirty. Such a unit runs in the current sweep when it sorts after the
/// evaluated unit. Otherwise it runs in the next sweep; this includes the
/// evaluated unit.
///
/// The driver evaluates exactly the units that a full sweep in unit order
/// evaluates when it skips each unit whose latest evaluation read no slot
/// that changed since. Thus a deterministic evaluation that depends only on
/// the slots that it records gives the same slots as that full sweep, but
/// the driver never visits a clean unit. A unit whose latest evaluation
/// recorded no read does not run again.
///
/// A sweep after the first runs only when the sweep before it changed a
/// slot. It then runs also when no unit is dirty, so the sweep count is that
/// of the full sweep, which stops after its first sweep without a change.
/// Before each sweep the driver calls `proceed` with the sweep index. When
/// `proceed` returns `false`, the fixpoint stops and is not converged. Put a
/// sweep limit or a work budget in `proceed`.
///
/// # Examples
///
/// ```
/// use cfglib::ordered_fixpoint;
///
/// // Unit `u` copies slot `u + 1` into slot `u`, and slot 3 starts set. The
/// // chain runs against the sweep order, so it sets one unit in each sweep.
/// let mut slots = [false, false, false, true];
/// let stats = ordered_fixpoint(3, |_| true, |unit, access| {
///     access.read(unit + 1);
///     if !slots[unit] && slots[unit + 1] {
///         slots[unit] = true;
///         access.changed(unit);
///     }
/// });
/// assert_eq!(slots, [true; 4]);
/// assert_eq!(stats.round_count(), 4);
/// assert_eq!(stats.evaluation_count(), 5);
/// assert!(stats.converged());
/// ```
pub fn ordered_fixpoint<K: Ord + Clone>(
    units: usize,
    mut proceed: impl FnMut(usize) -> bool,
    mut evaluate: impl FnMut(usize, &mut SlotAccess<K>),
) -> SlotFixpointStats {
    let mut stats = SlotFixpointStats::default();
    // The sorted, distinct slots that the latest evaluation of each unit
    // read, and the same relation inverted.
    let mut reads: Vec<Vec<K>> = (0..units).map(|_| Vec::new()).collect();
    let mut readers: BTreeSet<(K, usize)> = BTreeSet::new();
    let mut current = Queue::full(units);
    let mut next = Queue::new(units);
    let mut access = SlotAccess::new();
    for sweep in 0.. {
        if !proceed(sweep) {
            return stats;
        }
        stats.rounds += 1;
        let mut changed = false;
        while let Some(unit) = current.pop_first() {
            access.reads.clear();
            evaluate(unit, &mut access);
            stats.evaluations += 1;
            access.reads.sort_unstable();
            access.reads.dedup();
            // Record the new reads before the changes queue their readers: a
            // unit that changes a slot that it read runs again next sweep.
            if access.reads != reads[unit] {
                for key in reads[unit].drain(..) {
                    readers.remove(&(key, unit));
                }
                readers.extend(access.reads.iter().map(|key| (key.clone(), unit)));
                mem::swap(&mut reads[unit], &mut access.reads);
            }
            for key in access.changes.drain(..) {
                changed = true;
                for &(_, reader) in readers.range((key.clone(), 0)..=(key, usize::MAX)) {
                    if reader > unit {
                        current.insert(reader);
                    } else {
                        next.insert(reader);
                    }
                }
            }
        }
        if !changed {
            break;
        }
        mem::swap(&mut current, &mut next);
    }
    stats.converged = true;
    stats
}
