//! The wave driver: it evaluates groups together against the facts that are
//! known before the wave, and it merges the outputs in group order.

extern crate alloc;

use alloc::vec::Vec;

use super::SlotFixpointStats;
use crate::dataflow::bits::DenseBits;
use crate::graph::fanout::Fanout;

/// Runs the evaluations of one wave, for example on a thread pool.
///
/// The driver starts no threads: the executor decides about parallelism.
/// Every evaluation of a wave reads the same facts, so the executor can run
/// them in any order and on any thread.
pub trait WaveExecutor {
    /// Returns the output of `evaluate` for each group of `wave`, in wave
    /// order. Returns `None` to stop the fixpoint before this wave, for
    /// example when a work budget refuses the wave.
    fn run<T, F>(&mut self, wave: &[usize], evaluate: F) -> Option<Vec<T>>
    where
        T: Send,
        F: Fn(usize) -> T + Sync;
}

/// A [`WaveExecutor`] that evaluates each wave in order on the calling
/// thread. It never stops a fixpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SequentialExecutor;

impl WaveExecutor for SequentialExecutor {
    fn run<T, F>(&mut self, wave: &[usize], evaluate: F) -> Option<Vec<T>>
    where
        T: Send,
        F: Fn(usize) -> T + Sync,
    {
        Some(wave.iter().map(|&group| evaluate(group)).collect())
    }
}

/// The slots that one merge changed. They decide the next wave.
pub struct WaveRequeue<'a, K> {
    readers: &'a Fanout<K, usize>,
    next: &'a mut DenseBits,
}

impl<'a, K: Ord> WaveRequeue<'a, K> {
    /// Records that slot `key` changed. Every group that reads the slot runs
    /// in the next wave. Returns those groups.
    ///
    /// # Panics
    ///
    /// Panics when a reader of `key` is not below the group count.
    pub fn changed(&mut self, key: &K) -> &'a [usize] {
        let readers = self.readers.get(key);
        for &group in readers {
            assert!(
                group < self.next.len(),
                "a reader names a group beyond the group count"
            );
            self.next.insert(group);
        }
        readers
    }
}

/// Runs `groups` units to a fixpoint in waves.
///
/// Wave 0 holds every group. The executor evaluates the groups of a wave
/// against the facts in `state` as they were before the wave, so the
/// evaluations of one wave are independent and can run in parallel. The
/// driver then gives each output to `merge` in ascending group order, so the
/// result does not depend on the schedule of the executor. `merge` records
/// the facts and reports each slot that it changed through [`WaveRequeue`].
/// The next wave is the ascending set of groups that `readers` lists for
/// those slots. The fixpoint ends when a wave queues no group.
///
/// A group runs in the first wave and again in each wave after one that
/// changed a slot that it reads. Thus the evaluations are at most `groups`
/// plus, for each change, the readers of the changed slot. For monotone
/// facts, where a group that sees more facts derives a superset of them, the
/// result is the fixpoint that one-at-a-time evaluation of the groups
/// reaches.
///
/// The fixpoint also ends, not converged, when the executor refuses a wave.
///
/// # Panics
///
/// Panics when the executor returns a number of outputs that is not the
/// number of groups in the wave, or when a reader is not below `groups`.
///
/// # Examples
///
/// ```
/// use cfglib::{Fanout, SequentialExecutor, wave_fixpoint};
///
/// // Each group copies the fact of the slot that it reads into its own slot:
/// // group 1 reads slot 0, group 2 reads slot 1, and slot 0 starts set.
/// let reads = [None, Some(0), Some(1)];
/// let readers = Fanout::from_pairs(vec![(0, 1), (1, 2)]);
/// let mut facts = vec![true, false, false];
/// let stats = wave_fixpoint(
///     3,
///     &readers,
///     &mut facts,
///     &mut SequentialExecutor,
///     |group, facts: &Vec<bool>| !facts[group] && reads[group].is_some_and(|slot| facts[slot]),
///     |facts, group, set, requeue| {
///         if set {
///             facts[group] = true;
///             requeue.changed(&group);
///         }
///     },
/// );
/// assert_eq!(facts, [true, true, true]);
/// assert_eq!(stats.round_count(), 2);
/// assert_eq!(stats.evaluation_count(), 4);
/// assert!(stats.converged());
/// ```
pub fn wave_fixpoint<K, S, T, X>(
    groups: usize,
    readers: &Fanout<K, usize>,
    state: &mut S,
    executor: &mut X,
    evaluate: impl Fn(usize, &S) -> T + Sync,
    mut merge: impl FnMut(&mut S, usize, T, &mut WaveRequeue<'_, K>),
) -> SlotFixpointStats
where
    K: Ord,
    S: Sync,
    T: Send,
    X: WaveExecutor,
{
    let mut wave_stats = SlotFixpointStats::default();
    let mut wave: Vec<usize> = (0..groups).collect();
    let mut next = DenseBits::new(groups);
    while !wave.is_empty() {
        let known: &S = state;
        let Some(outputs) = executor.run(&wave, |group| evaluate(group, known)) else {
            return wave_stats;
        };
        assert_eq!(
            outputs.len(),
            wave.len(),
            "the executor returns one output for each group of the wave"
        );
        wave_stats.rounds += 1;
        wave_stats.evaluations += wave.len();
        let mut requeue = WaveRequeue {
            readers,
            next: &mut next,
        };
        for (&group, output) in wave.iter().zip(outputs) {
            merge(state, group, output, &mut requeue);
        }
        wave.clear();
        wave.extend(next.ones());
        next.clear();
    }
    wave_stats.converged = true;
    wave_stats
}
