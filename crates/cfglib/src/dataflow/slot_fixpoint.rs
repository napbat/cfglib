//! Fixpoints over keyed slots that the caller owns. A unit runs again only
//! when a slot that it read changed.
//!
//! The solvers of [`fixpoint`](super::fixpoint),
//! [`node_fixpoint`](super::node_fixpoint) and
//! [`edge_fixpoint`](super::edge_fixpoint) own their facts and get the
//! dependencies of a node from the edges of a graph view. Some fixpoints have
//! no such graph: the caller keeps the facts in its own tables, a unit of
//! evaluation is a file or a declaration, and the slots that a unit reads
//! come from an index that the caller built or from what the evaluation read.
//! The two drivers here schedule such units. They own nothing else: they
//! have no problem trait, no meet and no transfer function. The caller
//! limits them through its executor or its sweep hook.
//!
//! - [`wave_fixpoint`] evaluates groups in waves against the facts that are
//!   known before the wave. A [`WaveExecutor`] of the caller runs the
//!   evaluations of a wave, possibly in parallel. The driver merges the
//!   outputs in group order. The next wave holds the groups that read a slot
//!   that the merge changed, from a static [`Fanout`](crate::Fanout) of
//!   readers.
//! - [`ordered_fixpoint`] evaluates units one at a time in a fixed sweep
//!   order, so a later evaluation of a sweep sees a slot that an earlier one
//!   changed (Gauss-Seidel order). Each evaluation records the slots that it
//!   read in a [`SlotAccess`], so the dependencies can change from one
//!   evaluation to the next.
//!
//! Both drivers stop when no unit is left to run. Each one returns its
//! rounds and evaluations in a [`SlotFixpointStats`].

mod ordered;
mod wave;

#[cfg(test)]
mod tests;

pub use ordered::{SlotAccess, ordered_fixpoint};
pub use wave::{SequentialExecutor, WaveExecutor, WaveRequeue, wave_fixpoint};

/// What a slot fixpoint did: its rounds, its evaluations, and whether it
/// reached the fixpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotFixpointStats {
    rounds: usize,
    evaluations: usize,
    converged: bool,
}

impl SlotFixpointStats {
    /// Returns the number of waves or sweeps that ran.
    #[must_use]
    pub const fn round_count(self) -> usize {
        self.rounds
    }

    /// Returns the number of unit evaluations in all rounds.
    #[must_use]
    pub const fn evaluation_count(self) -> usize {
        self.evaluations
    }

    /// Returns whether the fixpoint ran until no unit was left to run. The
    /// value is `false` when the executor or the sweep hook stopped it first.
    #[must_use]
    pub const fn converged(self) -> bool {
        self.converged
    }
}
