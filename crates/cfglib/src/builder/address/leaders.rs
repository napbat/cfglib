//! The leaders of a built graph: the instructions that start a block.

use super::{
    AddressBuildError, AddressInstruction, AddressSpace, AddressTransfers, CallPolicy, StatedEdges,
    ends_block,
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

/// Marks the leaders of a stream whose normal edges the caller states.
///
/// An instruction with stated edges ends its block, and the target of each
/// of them leads a block. Each other instruction continues at its end
/// address: its flow can only continue or call. A call that ends its block by
/// `policy` marks its leaders as in [`collect_flow_leaders`].
///
/// # Errors
///
/// Returns [`AddressBuildError::MissingStatedSource`] for a stated source,
/// and [`AddressBuildError::MissingStatedTarget`] for a stated target, that
/// starts no instruction. Returns [`AddressBuildError::UnstatedTransfer`] for
/// an instruction without stated edges whose flow transfers other than by
/// its continuation and its call.
pub(super) fn collect_stated_leaders<I: AddressInstruction, E>(
    instructions: &[I],
    addresses: &[I::Address],
    stated: &StatedEdges<I::Address, E>,
    policy: CallPolicy,
    leaders: &mut [bool],
) -> Result<(), AddressBuildError<I::Address>> {
    if let Some(&address) = stated
        .keys()
        .find(|source| addresses.binary_search(source).is_err())
    {
        return Err(AddressBuildError::MissingStatedSource { address });
    }
    collect_sequence_leaders(instructions, addresses, leaders);
    for (position, instruction) in instructions.iter().enumerate() {
        let address = addresses[position];
        let flow = instruction.flow();
        if let Some(edges) = stated.get(&address) {
            for target in edges.iter().map(|edge| edge.target) {
                if addresses.binary_search(&target).is_err() {
                    return Err(AddressBuildError::MissingStatedTarget {
                        source: address,
                        target,
                    });
                }
                mark_target(addresses, target, leaders);
            }
            if matches!(flow, Flow::Exceptional { .. }) {
                leaders[position] = true;
            }
            if position + 1 < addresses.len() {
                leaders[position + 1] = true;
            }
            continue;
        }
        if flow.ends_block() && !flow.is_call() {
            return Err(AddressBuildError::UnstatedTransfer { address });
        }
        if ends_block(&flow, policy) {
            let end = instruction
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
    }
    Ok(())
}

pub(super) fn mark_target<A: AddressSpace>(addresses: &[A], target: A, leaders: &mut [bool]) {
    if let Ok(position) = addresses.binary_search(&target) {
        leaders[position] = true;
    }
}
