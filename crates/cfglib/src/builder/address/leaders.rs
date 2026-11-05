//! The leaders of a built graph: the instructions that start a block.

use super::{
    AddressBuildError, AddressInstruction, AddressSpace, AddressTransfers, CallPolicy, GivenEdge,
    GivenEdges, ends_block,
};
use crate::flow::Flow;

/// Marks the leaders that the sequence of the stream makes: the instruction
/// after each instruction whose end is not the next start, and the
/// instruction at that end.
fn collect_sequence_leaders<I: AddressInstruction>(
    instructions: &[I],
    addresses: &[I::Address],
    leaders: &mut [bool],
) {
    for position in 0..addresses.len() {
        let end = instructions[position]
            .end_address()
            .expect("validated instruction end");
        let next = addresses.get(position + 1);
        let continues_to_next = next.is_some_and(|next| end == *next);
        if !continues_to_next {
            if next.is_some() {
                leaders[position + 1] = true;
            }
            mark_target(addresses, end, leaders);
        }
    }
}

pub(super) fn collect_flow_leaders<I: AddressInstruction>(
    instructions: &[I],
    addresses: &[I::Address],
    transfers: &AddressTransfers<I>,
    policy: CallPolicy,
    leaders: &mut [bool],
) {
    collect_sequence_leaders(instructions, addresses, leaders);
    for &(position, ref flow) in transfers {
        if !ends_block(flow, policy) {
            // A flattened call is an operation of its block: neither its
            // callee nor its continuation leads anything.
            continue;
        }
        let end = instructions[position]
            .end_address()
            .expect("validated instruction end");
        for transfer in flow.transfers(addresses[position], Some(end)) {
            if let Some(target) = transfer.target {
                mark_target(addresses, target, leaders);
            }
        }
        if matches!(flow, Flow::Exceptional { .. }) {
            // The exceptional edge must be exact to this instruction: an IR
            // built on the graph gives it to one throwing statement.
            leaders[position] = true;
        }
        if position + 1 < addresses.len() {
            leaders[position + 1] = true;
        }
    }
}

/// Marks the leaders of a stream whose normal edges the caller gives.
///
/// An instruction with given edges ends its block, and the target of each
/// of them leads a block. Each other instruction continues at its end
/// address: its flow can only continue or call. A call that ends its block by
/// `policy` marks its leaders as in [`collect_flow_leaders`].
///
/// The flows come from `transfers`, the block-ending flows that the build
/// classified once. An instruction that `transfers` does not hold continues
/// (`Next`).
///
/// # Errors
///
/// Returns [`AddressBuildError::MissingGivenSource`] for a given source,
/// and [`AddressBuildError::MissingGivenTarget`] for a given target, that
/// starts no instruction. Returns
/// [`AddressBuildError::TransferWithoutGivenEdges`] for an instruction
/// without given edges whose flow transfers other than by its continuation
/// and its call. The first error in address order wins.
pub(super) fn collect_given_leaders<I: AddressInstruction, E>(
    instructions: &[I],
    addresses: &[I::Address],
    transfers: &AddressTransfers<I>,
    given: &GivenEdges<I::Address, E>,
    policy: CallPolicy,
    leaders: &mut [bool],
) -> Result<(), AddressBuildError<I::Address>> {
    if let Some(&address) = given
        .keys()
        .find(|source| addresses.binary_search(source).is_err())
    {
        return Err(AddressBuildError::MissingGivenSource { address });
    }
    collect_sequence_leaders(instructions, addresses, leaders);
    // Both sequences are in address order, so one merge visits each
    // instruction with given edges and each block-ending flow once.
    let mut transfers = transfers.iter().peekable();
    for (&address, edges) in given {
        let position = addresses
            .binary_search(&address)
            .expect("a given source is an instruction start");
        while let Some((earlier, flow)) = transfers.next_if(|(other, _)| *other < position) {
            mark_flow_leaders(instructions, addresses, *earlier, flow, policy, leaders)?;
        }
        let flow = transfers
            .next_if(|(other, _)| *other == position)
            .map(|(_, flow)| flow);
        mark_given_leaders(addresses, position, edges, flow, leaders)?;
    }
    for (position, flow) in transfers {
        mark_flow_leaders(instructions, addresses, *position, flow, policy, leaders)?;
    }
    Ok(())
}

/// Marks the leaders of the instruction at `position`, which has the given
/// `edges` and ends its block by the block-ending `flow`, if any.
fn mark_given_leaders<A: AddressSpace, K, E>(
    addresses: &[A],
    position: usize,
    edges: &[GivenEdge<A, E>],
    flow: Option<&Flow<A, K>>,
    leaders: &mut [bool],
) -> Result<(), AddressBuildError<A>> {
    for target in edges.iter().map(|edge| edge.target) {
        if addresses.binary_search(&target).is_err() {
            return Err(AddressBuildError::MissingGivenTarget {
                source: addresses[position],
                target,
            });
        }
        mark_target(addresses, target, leaders);
    }
    if matches!(flow, Some(Flow::Exceptional { .. })) {
        leaders[position] = true;
    }
    if position + 1 < addresses.len() {
        leaders[position + 1] = true;
    }
    Ok(())
}

/// Marks the leaders of the instruction at `position`, which has no given
/// edges and ends its block by `flow`.
fn mark_flow_leaders<I: AddressInstruction>(
    instructions: &[I],
    addresses: &[I::Address],
    position: usize,
    flow: &Flow<I::Address, I::CaseKey>,
    policy: CallPolicy,
    leaders: &mut [bool],
) -> Result<(), AddressBuildError<I::Address>> {
    let address = addresses[position];
    if !flow.is_call() {
        return Err(AddressBuildError::TransferWithoutGivenEdges { address });
    }
    if ends_block(flow, policy) {
        let end = instructions[position]
            .end_address()
            .expect("validated instruction end");
        for target in flow
            .transfers(address, Some(end))
            .filter_map(|transfer| transfer.target)
        {
            mark_target(addresses, target, leaders);
        }
        if position + 1 < addresses.len() {
            leaders[position + 1] = true;
        }
    }
    Ok(())
}

pub(super) fn mark_target<A: AddressSpace>(addresses: &[A], target: A, leaders: &mut [bool]) {
    if let Ok(position) = addresses.binary_search(&target) {
        leaders[position] = true;
    }
}
