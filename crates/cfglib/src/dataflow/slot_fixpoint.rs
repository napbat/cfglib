//! Fixpoints over caller-owned keyed slots, re-running a unit only when a
//! slot it read changed.
//!
//! The solvers in [`fixpoint`](super::fixpoint),
//! [`node_fixpoint`](super::node_fixpoint) and
//! [`edge_fixpoint`](super::edge_fixpoint) own their facts and derive the
//! dependencies of a node from the edges of a graph view. Some fixpoints
//! have no such graph: a consumer owns the facts in its own tables, a unit
//! of evaluation is a file or a declaration, and the slots that a unit reads
//! are known only from an index the consumer built or from what the
//! evaluation itself read. The two drivers here schedule such units and own
//! nothing else. They are not members of the solver family: they have no
//! problem trait, meet or transfer, and the caller bounds them through its
//! executor or its sweep hook rather than through a `SolveConfig`.
//!
//! - [`wave_fixpoint`] evaluates groups in waves against the facts known
//!   before the wave, through a caller-supplied [`WaveExecutor`] that may run
//!   them in parallel, and merges the outputs in group order. The next wave
//!   is the groups that read a slot the merge changed, from a static
//!   [`Fanout`](crate::Fanout) of readers.
//! - [`ordered_fixpoint`] evaluates units one at a time in a fixed sweep
//!   order, so a slot changed earlier in a sweep is visible later in the same
//!   sweep (Gauss–Seidel). Each evaluation reports the slots it read through
//!   a [`SlotAccess`], so the dependencies may differ between evaluations.
//!
//! Both drivers stop when no unit is left to run and report the rounds and
//! evaluations in a [`SlotFixpointReport`].

mod ordered;
mod wave;

#[cfg(test)]
mod tests;

pub use ordered::{SlotAccess, ordered_fixpoint};
pub use wave::{SequentialExecutor, WaveExecutor, WaveRequeue, wave_fixpoint};

/// What a slot fixpoint did: its rounds, its evaluations, and whether it
/// reached the fixpoint.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SlotFixpointReport {
    rounds: usize,
    evaluations: usize,
    converged: bool,
}

impl SlotFixpointReport {
    /// The waves or sweeps that ran.
    #[must_use]
    pub const fn round_count(self) -> usize {
        self.rounds
    }

    /// The unit evaluations over all rounds.
    #[must_use]
    pub const fn evaluation_count(self) -> usize {
        self.evaluations
    }

    /// Whether the fixpoint ran until no unit was left to run. `false` when
    /// the executor or the sweep hook stopped it first.
    #[must_use]
    pub const fn converged(self) -> bool {
        self.converged
    }
}
