//! Parallel-move serialization of one multi-assignment transfer.
//!
//! Every read of a transfer observes pre-statement storage. MLIL webs can
//! later share a native home, so a serialized write must not clobber a
//! value a later sibling still reads. Readers of a storage therefore come
//! before its writers. Only a read/write cycle gets a synthetic copy.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;

use super::super::dialect::Lift;
use super::super::error::{Error, Result};
use super::super::expr::{Expr, Place};
use super::super::statement::{Lane, StatementId};
use super::super::template::{LiftedStatement, VarExpr};
use super::{Defined, Emitter, PendingAssign};

impl<D: Lift> Emitter<'_, D> {
    /// Tells whether a serialized write by `writer` clobbers native lanes of
    /// pre-state web `read`. Compares native storage, not web identity: a
    /// straight-line definition starts a fresh web, while its sibling still
    /// reads the older web in the same location.
    fn clobbers(&self, writer: &PendingAssign<D>, read: usize) -> bool {
        let read_info = &self.webs[read];
        let target_info = &self.webs[writer.target];
        target_info.storage.is_some()
            && target_info.storage == read_info.storage
            && writer.positions.iter().any(|&position| {
                target_info
                    .lanes
                    .get(usize::from(position))
                    .is_some_and(|lane| read_info.lanes.contains(lane))
            })
    }

    /// Tells whether assignment `writer` must wait for some other pending
    /// assignment that still reads state `writer` clobbers.
    fn blocked(&self, lifted: &[Option<PendingAssign<D>>], writer: usize) -> bool {
        let Some(pending) = &lifted[writer] else {
            return false;
        };
        lifted.iter().enumerate().any(|(index, reader)| {
            index != writer
                && reader.as_ref().is_some_and(|reader| {
                    reader
                        .reads
                        .iter()
                        .any(|&read| self.clobbers(pending, read))
                })
        })
    }

    /// Serializes one parallel transfer: rebuild every assignment against
    /// pre-statement state, then sequentialize it as a parallel move.
    /// Readers of a storage come before its writers. Only a cycle gets a
    /// synthetic pre-state copy. The first emitted instruction carries the
    /// statement effects, the throw flag, and exceptional successors.
    #[expect(clippy::too_many_arguments, reason = "one slot per statement facet")]
    pub(super) fn transfer(
        &mut self,
        source: usize,
        id: StatementId,
        assignments: &[(Place<D>, Expr<D>)],
        effects: &[<D as Vocabulary>::Effect],
        may_throw: bool,
        has_exceptional_successors: bool,
        spans: &[<D as Vocabulary>::SourceSpan],
        annotation: &crate::SsaInstruction<Lane<D>>,
    ) -> Result<()> {
        let mut use_cursor = 0usize;
        let mut def_cursor = 0usize;
        let mut lifted: Vec<PendingAssign<D>> = Vec::new();
        for (place, value) in assignments {
            let mut reads = Vec::new();
            let value = self.rebuild(value, &annotation.uses, &mut use_cursor, &mut reads)?;
            let target = annotation
                .defs
                .get(def_cursor)
                .ok_or_else(|| Error::Lifting("SSA lost a definition".into()))?;
            let target = self.resolver.web(target)?;
            let mut positions = Vec::with_capacity(place.lanes.len());
            for offset in 0..place.lanes.len() {
                let def = &annotation.defs[def_cursor + offset];
                positions.push(self.position(target, def.variable.1)?);
            }
            def_cursor += place.lanes.len();
            lifted.push(PendingAssign {
                target,
                positions,
                value,
                reads,
            });
        }
        let mut pending: Vec<Option<PendingAssign<D>>> = lifted.into_iter().map(Some).collect();
        let mut first = true;
        while let Some(start) = pending.iter().position(Option::is_some) {
            // Pick the earliest assignment no remaining sibling waits on.
            // In a cycle, pick the earliest one and copy what it clobbers.
            let next = (start..pending.len())
                .find(|&index| pending[index].is_some() && !self.blocked(&pending, index));
            let next = if let Some(next) = next {
                next
            } else {
                self.break_cycle(
                    source,
                    id,
                    &mut pending,
                    start,
                    spans,
                    effects,
                    may_throw,
                    has_exceptional_successors,
                    first,
                )?;
                first = false;
                start
            };
            let Some(assign) = pending[next].take() else {
                unreachable!("selected assignment is pending");
            };
            let width = self.webs[assign.target].shape.lanes;
            let merges = assign.positions.len() < usize::from(width);
            let statement_effects = if first { effects.to_vec() } else { Vec::new() };
            let throws = first && may_throw;
            let exceptional = first && has_exceptional_successors;
            first = false;
            self.hand_off(
                source,
                id,
                LiftedStatement::Assign {
                    positions: assign.positions,
                    width,
                    merges,
                    value: assign.value,
                    effects: statement_effects,
                },
                &assign.reads,
                Defined::Target {
                    web: assign.target,
                    merge: merges,
                },
                throws,
                exceptional,
                spans.to_vec(),
            )?;
        }
        Ok(())
    }

    /// Breaks a cycle at assignment `writer`: copies every pre-state web
    /// that another pending sibling reads and `writer` clobbers into a
    /// synthetic temporary, and redirects those reads to the copy.
    #[expect(clippy::too_many_arguments, reason = "one slot per statement facet")]
    fn break_cycle(
        &mut self,
        source: usize,
        id: StatementId,
        pending: &mut [Option<PendingAssign<D>>],
        writer: usize,
        spans: &[<D as Vocabulary>::SourceSpan],
        effects: &[<D as Vocabulary>::Effect],
        may_throw: bool,
        has_exceptional_successors: bool,
        mut first: bool,
    ) -> Result<()> {
        let Some(writing) = &pending[writer] else {
            return Ok(());
        };
        let mut hazards = BTreeSet::new();
        for (index, reader) in pending.iter().enumerate() {
            if index == writer {
                continue;
            }
            if let Some(reader) = reader {
                for &read in &reader.reads {
                    if self.clobbers(writing, read) {
                        hazards.insert(read);
                    }
                }
            }
        }
        for hazard in hazards {
            let shape = self.webs[hazard].shape.clone();
            let temporary = self.declare_temporary(shape.clone())?;
            let all: Vec<u8> = (0..shape.lanes).collect();
            let statement_effects = if first { effects.to_vec() } else { Vec::new() };
            let throws = first && may_throw;
            let exceptional = first && has_exceptional_successors;
            first = false;
            self.hand_off(
                source,
                id,
                LiftedStatement::Assign {
                    positions: all.clone(),
                    width: shape.lanes,
                    merges: false,
                    value: VarExpr::Read {
                        positions: all,
                        scalar: shape.scalar,
                    },
                    effects: statement_effects,
                },
                &[hazard],
                Defined::Target {
                    web: temporary,
                    merge: false,
                },
                throws,
                exceptional,
                spans.to_vec(),
            )?;
            for (index, reader) in pending.iter_mut().enumerate() {
                if index == writer {
                    continue;
                }
                for web in reader.iter_mut().flat_map(|reader| reader.reads.iter_mut()) {
                    if *web == hazard {
                        *web = temporary;
                    }
                }
            }
        }
        Ok(())
    }
}
