//! The edges of a built graph: the normal edges of each block end, and the
//! unwind edges of the handler table.

extern crate alloc;

use alloc::vec::Vec;

use super::{
    AddressBuildError, AddressEdgeInfo, AddressHandler, AddressInstruction, AddressTransfers,
    CallPolicy, GivenEdge, GivenEdges, HandlerIndex, ends_block,
};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::{EdgeRole, Flow, Transfer, UnresolvedTransfer};

/// Where the normal edges of a build come from.
pub(super) enum Normal<A, E> {
    /// The flow of each instruction.
    Flow,
    /// The edges that the caller gives for the instructions in the map.
    /// Each other instruction continues at its end address.
    Given(GivenEdges<A, E>),
}

/// Adds the normal edges of each block end in `terminators`.
///
/// # Errors
///
/// With given edges, returns [`AddressBuildError::MissingContinuation`] when
/// an instruction without given edges continues at an address that starts
/// no instruction.
#[expect(
    clippy::too_many_arguments,
    reason = "the emitter reads each part of the populated graph"
)]
pub(super) fn add_normal_edges<I: AddressInstruction, E, F>(
    cfg: &mut Cfg<I, E>,
    addresses: &[I::Address],
    transfers: &AddressTransfers<I>,
    blocks_by_instruction: &[BlockId],
    terminators: &[(BlockId, usize)],
    policy: CallPolicy,
    normal: Normal<I::Address, E>,
    edge_payload: &mut F,
) -> Result<Vec<UnresolvedTransfer<I::Address>>, AddressBuildError<I::Address>>
where
    F: for<'k> FnMut(AddressEdgeInfo<'k, I::Address, I::CaseKey>) -> E,
{
    let mut emitter = AddressEdgeEmitter {
        cfg,
        addresses,
        blocks_by_instruction,
        policy,
        edge_payload,
        unresolved: Vec::new(),
    };
    let mut given = match normal {
        Normal::Flow => None,
        Normal::Given(given) => Some(given),
    };
    for &(block, position) in terminators {
        let source = addresses[position];
        if let Some(edges) = given.as_mut().and_then(|given| given.remove(&source)) {
            for edge in edges {
                emitter.add_given(block, edge);
            }
            continue;
        }
        let flow = transfers
            .binary_search_by_key(&position, |(position, _)| *position)
            .ok()
            .map(|transfer| &transfers[transfer].1);
        emitter.emit(block, position, flow, given.is_some())?;
    }
    Ok(emitter.unresolved)
}

struct AddressEdgeEmitter<'a, I: AddressInstruction, E, F> {
    cfg: &'a mut Cfg<I, E>,
    addresses: &'a [I::Address],
    blocks_by_instruction: &'a [BlockId],
    policy: CallPolicy,
    edge_payload: &'a mut F,
    unresolved: Vec<UnresolvedTransfer<I::Address>>,
}

impl<I: AddressInstruction, E, F> AddressEdgeEmitter<'_, I, E, F>
where
    F: for<'k> FnMut(AddressEdgeInfo<'k, I::Address, I::CaseKey>) -> E,
{
    /// Adds the edges that leave `block`, whose last instruction sits at
    /// `position` and transfers by `flow`.
    ///
    /// A block whose last instruction has no transfer of its own ended at a
    /// leader, so it continues sequentially. The continuation is the
    /// instruction at the block's last end address, which can differ from
    /// the next instruction in the stream when paths overlap.
    ///
    /// A `strict` emitter rejects a continuation whose target starts no
    /// instruction. Without `strict`, such a continuation has no edge and no
    /// report.
    fn emit(
        &mut self,
        block: BlockId,
        position: usize,
        flow: Option<&Flow<I::Address, I::CaseKey>>,
        strict: bool,
    ) -> Result<(), AddressBuildError<I::Address>> {
        let source = self.addresses[position];
        let end = self
            .cfg
            .block(block)
            .instructions()
            .last()
            .and_then(AddressInstruction::end_address)
            .expect("validated block terminator end");
        let Some(flow) = flow.filter(|flow| ends_block(flow, self.policy)) else {
            return self.add_continuation(
                block,
                source,
                Transfer {
                    target: Some(end),
                    role: EdgeRole::Sequential,
                },
                strict,
            );
        };
        for transfer in flow.transfers(source, Some(end)) {
            if matches!(
                transfer.role,
                EdgeRole::Sequential | EdgeRole::CallContinuation { .. }
            ) {
                self.add_continuation(block, source, transfer, strict)?;
            } else {
                self.add_transfer(block, source, transfer);
            }
        }
        Ok(())
    }

    /// Adds the edge of the continuation `transfer` of the block end at
    /// `source`.
    ///
    /// A `strict` emitter rejects a continuation whose target starts no
    /// instruction. Without `strict`, such a continuation has no edge.
    fn add_continuation(
        &mut self,
        block: BlockId,
        source: I::Address,
        transfer: Transfer<'_, I::Address, I::CaseKey>,
        strict: bool,
    ) -> Result<(), AddressBuildError<I::Address>> {
        if strict
            && let Some(target) = transfer.target
            && self.addresses.binary_search(&target).is_err()
        {
            return Err(AddressBuildError::MissingContinuation { source, target });
        }
        self.add_transfer(block, source, transfer);
        Ok(())
    }

    /// Adds the given `edge` that leaves `block`. The leaders checked that
    /// its target starts an instruction.
    fn add_given(&mut self, block: BlockId, edge: GivenEdge<I::Address, E>) {
        let target = self
            .addresses
            .binary_search(&edge.target)
            .expect("a given target is an instruction start");
        self.cfg.add_edge_with_payload(
            block,
            self.blocks_by_instruction[target],
            edge.kind,
            edge.payload,
        );
    }

    fn add_transfer(
        &mut self,
        block: BlockId,
        source: I::Address,
        transfer: Transfer<'_, I::Address, I::CaseKey>,
    ) {
        let Some(target) = transfer.target else {
            if let Some(role) = transfer.unresolved_role() {
                self.unresolved.push(UnresolvedTransfer {
                    source,
                    target: None,
                    role,
                });
            }
            return;
        };
        let Ok(target_position) = self.addresses.binary_search(&target) else {
            if let Some(role) = transfer.unresolved_role() {
                self.unresolved.push(UnresolvedTransfer {
                    source,
                    target: Some(target),
                    role,
                });
            }
            return;
        };
        let kind = transfer.role.kind();
        let payload = (self.edge_payload)(AddressEdgeInfo {
            source,
            target,
            kind,
            role: transfer.role,
        });
        self.cfg.add_edge_with_payload(
            block,
            self.blocks_by_instruction[target_position],
            kind,
            payload,
        );
    }
}

pub(super) fn add_exception_edges<I: AddressInstruction, E>(
    cfg: &mut Cfg<I, E>,
    handlers: &[AddressHandler<I::Address>],
    handler_index: &HandlerIndex,
    addresses: &[I::Address],
    blocks_by_instruction: &[BlockId],
    edge_payload: &mut impl FnMut(AddressEdgeInfo<'_, I::Address, I::CaseKey>) -> E,
) {
    for (index, handler) in handlers.iter().enumerate() {
        let target = blocks_by_instruction[handler_index.entries[index]];
        for position in handler_index.ranges[index].clone() {
            let source = blocks_by_instruction[position];
            let instruction = cfg
                .block(source)
                .instructions()
                .first()
                .expect("a protected instruction leads its block");
            debug_assert!(instruction.address() == addresses[position]);
            if instruction.retains_exception_edge() {
                let payload = edge_payload(AddressEdgeInfo {
                    source: addresses[position],
                    target: handler.entry,
                    kind: EdgeKind::ExceptionUnwind,
                    role: EdgeRole::Unwind { handler: index },
                });
                cfg.add_edge_with_payload(source, target, EdgeKind::ExceptionUnwind, payload);
            }
        }
    }
}

#[cfg(test)]
mod tests;
