//! Ordered funclet chains of address handlers.
//!
//! A runtime that unwinds a protected region can run several cleanup funclets
//! before it reaches a catch clause or propagates. It enters them one after
//! the other: each funclet returns to the unwinder, and the unwinder enters
//! the next. A [`HandlerChain`] states that order for one handler of the
//! handler table. This module validates the chains and adds their edges to a
//! graph that the address builder populated.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use super::{
    AddressBuildError, AddressEdgeInfo, AddressHandler, AddressInstruction, AddressSpace,
    HandlerIndex,
};
use crate::block::BlockId;
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::{EdgeRole, Flow};
use crate::region::HandlerKind;

/// The funclets that the unwinder enters after the entry of one handler.
///
/// The entry of `handlers[handler]` is the first funclet of the chain.
/// [`then`](Self::then) lists the entries of the funclets that follow, in the
/// order in which the unwinder enters them.
///
/// Each return of a funclet is an edge to the entry of the next funclet.
/// Each return of the last funclet is an edge to the entry of each catch
/// handler of the same protected range ([`HandlerKind::Catch`],
/// [`HandlerKind::CatchAll`] and [`HandlerKind::Filter`]). A return still
/// ends its block, so the graph also keeps the path on which the unwinder
/// propagates the exception out of the function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandlerChain<A> {
    /// Position in the handler table of the handler that starts the chain.
    pub handler: usize,
    /// Entry addresses of the funclets that follow the handler entry.
    pub then: Vec<A>,
}

/// Checks the chains and marks the entry of each funclet as a leader.
pub(super) fn validate_chains<A: AddressSpace>(
    chains: &[HandlerChain<A>],
    handler_count: usize,
    addresses: &[A],
    leaders: &mut [bool],
) -> Result<(), AddressBuildError<A>> {
    for chain in chains {
        if chain.handler >= handler_count {
            return Err(AddressBuildError::UnknownChainHandler {
                handler: chain.handler,
            });
        }
        for &address in &chain.then {
            let position = addresses
                .binary_search(&address)
                .map_err(|_| AddressBuildError::MissingChainEntry { address })?;
            leaders[position] = true;
        }
    }
    Ok(())
}

/// Adds the edge from each return of a funclet to the next step.
///
/// The funclets of a chain are blocks that the builder already made. A return
/// of a funclet is a block that control reaches from the entry of the funclet
/// along normal edges, without entering the entry of another funclet, and
/// whose last instruction returns.
pub(super) fn add_chain_edges<I: AddressInstruction, E>(
    cfg: &mut Cfg<I, E>,
    handlers: &[AddressHandler<I::Address>],
    handler_index: &HandlerIndex,
    chains: &[HandlerChain<I::Address>],
    addresses: &[I::Address],
    blocks_by_instruction: &[BlockId],
    edge_payload: &mut impl FnMut(AddressEdgeInfo<'_, I::Address, I::CaseKey>) -> E,
) {
    let entry_block = |address: I::Address| {
        let position = addresses
            .binary_search(&address)
            .expect("`validate_chains` checked that a funclet entry is an instruction start");
        blocks_by_instruction[position]
    };
    let funclet_blocks: BTreeSet<BlockId> = chains
        .iter()
        .flat_map(|chain| {
            core::iter::once(blocks_by_instruction[handler_index.entries[chain.handler]])
                .chain(chain.then.iter().map(|&address| entry_block(address)))
        })
        .collect();
    for chain in chains {
        let handler = &handlers[chain.handler];
        let mut steps: Vec<(I::Address, BlockId)> = Vec::with_capacity(chain.then.len() + 1);
        steps.push((
            handler.entry,
            blocks_by_instruction[handler_index.entries[chain.handler]],
        ));
        steps.extend(
            chain
                .then
                .iter()
                .map(|&address| (address, entry_block(address))),
        );
        for (position, &(_, block)) in steps.iter().enumerate() {
            let targets: Vec<(I::Address, BlockId)> = if let Some(&next) = steps.get(position + 1) {
                Vec::from([next])
            } else {
                handlers
                    .iter()
                    .enumerate()
                    .filter(|(_, other)| {
                        other.protected == handler.protected
                            && matches!(
                                other.kind,
                                HandlerKind::Catch
                                    | HandlerKind::CatchAll
                                    | HandlerKind::Filter { .. }
                            )
                    })
                    .map(|(index, other)| {
                        (
                            other.entry,
                            blocks_by_instruction[handler_index.entries[index]],
                        )
                    })
                    .collect()
            };
            for source in returns(cfg, block, &funclet_blocks) {
                let source_address = cfg
                    .block(source)
                    .instructions()
                    .last()
                    .expect("a block holds an instruction")
                    .address();
                for &(target_address, target) in &targets {
                    let payload = edge_payload(AddressEdgeInfo {
                        source: source_address,
                        target: target_address,
                        kind: EdgeKind::ExceptionUnwind,
                        role: EdgeRole::Resume {
                            handler: chain.handler,
                        },
                    });
                    cfg.add_edge_with_payload(source, target, EdgeKind::ExceptionUnwind, payload);
                }
            }
        }
    }
}

/// Returns the blocks that end a funclet with a return, in block order.
fn returns<I: AddressInstruction, E>(
    cfg: &Cfg<I, E>,
    entry: BlockId,
    funclets: &BTreeSet<BlockId>,
) -> Vec<BlockId> {
    let mut seen = BTreeSet::from([entry]);
    let mut pending = Vec::from([entry]);
    let mut found = BTreeSet::new();
    while let Some(block) = pending.pop() {
        let returns = cfg
            .block(block)
            .instructions()
            .last()
            .is_some_and(|instruction| matches!(instruction.flow(), Flow::Return));
        if returns {
            found.insert(block);
        }
        for edge in cfg.outgoing(block) {
            let edge = cfg.edge(edge);
            let followed = !matches!(
                edge.data().kind(),
                EdgeKind::ExceptionUnwind | EdgeKind::Call
            );
            let target = edge.target();
            if followed && !funclets.contains(&target) && seen.insert(target) {
                pending.push(target);
            }
        }
    }
    found.into_iter().collect()
}

#[cfg(test)]
mod tests;
