//! Fixpoints over keyed slots that the caller owns. A unit runs again only
//! when a slot that it read changed.
//!
//! The solvers of [`node_fixpoint`](super::node_fixpoint) and
//! [`edge_fixpoint`](super::edge_fixpoint) own their facts and get the
//! dependencies of a node from the edges of a graph view. Some fixpoints have
//! no such graph: the caller keeps the facts in its own tables, a unit of
//! evaluation is a file or a declaration, and the slots that a unit reads
//! come from an index that the caller built or from what the evaluation read.
//! The two drivers here schedule such units. They own nothing else: they
//! have no problem trait, no meet and no transfer function.
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
//!   evaluation to the next. An [`OrderedFixpoint`] keeps those reads between
//!   runs, so a run after an edit evaluates only the units that it affects.
//!
//! Both drivers stop when no unit is left to run. Both take the step limit of
//! every bounded fixpoint of the crate, counted in evaluations, and stop with
//! [`SolveError::StepLimitExceeded`](crate::SolveError::StepLimitExceeded)
//! when work is left at it. Each one returns its evaluations and rounds in a
//! [`SolveStats`](crate::SolveStats).

mod ordered;
mod wave;

#[cfg(test)]
mod tests;

pub use ordered::{OrderedFixpoint, SlotAccess, ordered_fixpoint};
pub use wave::{SequentialExecutor, WaveExecutor, WaveRequeue, wave_fixpoint};
