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
#[derive(Debug, Clone)]
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

    /// Queues every unit that `other` holds.
    fn absorb(&mut self, other: &Self) {
        for index in other.queued.ones() {
            self.insert(index);
        }
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

/// The ordered driver with its dependency index kept between runs, so a
/// later run evaluates only the units that an edit affects.
///
/// [`ordered_fixpoint`] rebuilds the index of which unit read which slot on
/// every call and starts with every unit dirty. An editor-style consumer
/// (name resolution over files, per-declaration units) keeps one
/// `OrderedFixpoint` instead: the first [`run`](Self::run) evaluates every
/// unit, and each later run starts from the units the caller names and from
/// the readers of the slots that [`invalidate`](Self::invalidate) was told
/// about, then follows the recorded reads exactly as the first run did. This
/// is the destabilization of the incremental top-down solver (Erhard et al.,
/// STTT 2024; Stade, Tilscher, Seidl, CAV 2024) over the recorded read sets.
///
/// # Examples
///
/// ```
/// use cfglib::{OrderedFixpoint, SlotAccess};
///
/// // Unit `u` copies slot `u + 1` into slot `u`.
/// fn copy(slots: &mut [i32; 4], unit: usize, access: &mut SlotAccess<usize>) {
///     access.read(unit + 1);
///     if slots[unit] != slots[unit + 1] {
///         slots[unit] = slots[unit + 1];
///         access.changed(unit);
///     }
/// }
///
/// let mut slots = [0, 0, 0, 7];
/// let mut fixpoint = OrderedFixpoint::new(3);
/// let first = fixpoint.run([], |_| true, |unit, access| copy(&mut slots, unit, access));
/// assert_eq!((slots, first.evaluation_count()), ([7; 4], 5));
///
/// // The caller edits slot 3 outside a run: only the chain of its readers
/// // runs again, one unit each.
/// slots[3] = 9;
/// assert_eq!(fixpoint.invalidate(&[3]), [2]);
/// let again = fixpoint.run([], |_| true, |unit, access| copy(&mut slots, unit, access));
/// assert_eq!((slots, again.evaluation_count()), ([9; 4], 3));
///
/// // Nothing is dirty any more.
/// let idle = fixpoint.run([], |_| true, |unit, access| copy(&mut slots, unit, access));
/// assert_eq!(idle.evaluation_count(), 0);
/// ```
#[derive(Debug, Clone)]
pub struct OrderedFixpoint<K> {
    /// The sorted, distinct slots that the latest evaluation of each unit
    /// read.
    reads: Vec<Vec<K>>,
    /// The same relation inverted: a slot and a unit that read it.
    readers: BTreeSet<(K, usize)>,
    /// The units the next run starts with.
    dirty: Queue,
}

impl<K: Ord + Clone> OrderedFixpoint<K> {
    /// A driver over the units `0..units` with every unit dirty, so the
    /// first run evaluates them all.
    #[must_use]
    pub fn new(units: usize) -> Self {
        Self {
            reads: (0..units).map(|_| Vec::new()).collect(),
            readers: BTreeSet::new(),
            dirty: Queue::full(units),
        }
    }

    /// Runs ordered sweeps until no unit is dirty, as [`ordered_fixpoint`]
    /// does, from the units already dirty and the units in `dirty`.
    ///
    /// A unit that changes a slot makes every unit whose latest evaluation
    /// read it dirty, whether that evaluation ran in this run or an earlier
    /// one. When `proceed` stops the run, the units still dirty stay dirty
    /// for the next run.
    ///
    /// # Panics
    ///
    /// Panics when a unit in `dirty` is out of range.
    pub fn run(
        &mut self,
        dirty: impl IntoIterator<Item = usize>,
        mut proceed: impl FnMut(usize) -> bool,
        mut evaluate: impl FnMut(usize, &mut SlotAccess<K>),
    ) -> SlotFixpointStats {
        let units = self.reads.len();
        for unit in dirty {
            assert!(unit < units, "dirty unit is out of range");
            self.dirty.insert(unit);
        }
        let mut stats = SlotFixpointStats::default();
        let mut current = mem::replace(&mut self.dirty, Queue::new(units));
        let mut next = Queue::new(units);
        let mut access = SlotAccess::new();
        for sweep in 0.. {
            if !proceed(sweep) {
                current.absorb(&next);
                self.dirty = current;
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
                // Record the new reads before the changes queue their
                // readers: a unit that changes a slot that it read runs again
                // next sweep.
                if access.reads != self.reads[unit] {
                    for key in self.reads[unit].drain(..) {
                        self.readers.remove(&(key, unit));
                    }
                    let unit_reads = access.reads.iter().map(|key| (key.clone(), unit));
                    self.readers.extend(unit_reads);
                    mem::swap(&mut self.reads[unit], &mut access.reads);
                }
                for key in access.changes.drain(..) {
                    changed = true;
                    for reader in self.readers_of(key) {
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

    /// Makes dirty every unit whose latest evaluation read one of `slots`,
    /// after the caller changed them outside a run, and returns those
    /// units, ascending and distinct. The next [`run`](Self::run) evaluates
    /// them.
    pub fn invalidate(&mut self, slots: &[K]) -> Vec<usize> {
        let mut readers: Vec<usize> = slots
            .iter()
            .flat_map(|key| self.readers_of(key.clone()))
            .collect();
        readers.sort_unstable();
        readers.dedup();
        for &reader in &readers {
            self.dirty.insert(reader);
        }
        readers
    }

    /// The units whose latest evaluation read `key`, ascending.
    fn readers_of(&self, key: K) -> impl Iterator<Item = usize> + '_ {
        self.readers
            .range((key.clone(), 0)..=(key, usize::MAX))
            .map(|&(_, reader)| reader)
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
/// This is the first run of an [`OrderedFixpoint`], which keeps the reads it
/// recorded so that a later run after an edit evaluates only what the edit
/// affects.
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
    proceed: impl FnMut(usize) -> bool,
    evaluate: impl FnMut(usize, &mut SlotAccess<K>),
) -> SlotFixpointStats {
    OrderedFixpoint::new(units).run([], proceed, evaluate)
}
