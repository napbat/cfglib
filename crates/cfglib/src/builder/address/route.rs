//! Exact unwind routes of protected instructions.
//!
//! When an instruction throws, a runtime does not jump to every handler of
//! the table. It takes one of a few ways: it enters cleanup funclets one
//! after the other, each of which returns to the runtime, and it ends at a
//! catch handler, at the instruction itself, or outside the function. The
//! caller states these ways as [`UnwindRoute`] values. This module validates
//! them for the address builder (phase 1) and adds the graph edges that state
//! the ways exactly (phase 2,
//! [`AddressGraph::add_unwind_routes`](crate::AddressGraph::add_unwind_routes)).
//!
//! A funclet that two ways enter in different contexts cannot be one set of
//! blocks: the return of the funclet leads to the next step of the way that
//! entered it, and a shared set of blocks would also lead to the next steps
//! of the other ways. Phase 2 therefore gives each distinct remaining way its
//! own instance of the funclet: the first instance uses the original blocks,
//! and each further instance uses a copy of them.

extern crate alloc;

use alloc::vec::Vec;
use core::ops::Range;

use super::{AddressBuildError, AddressInstruction, AddressSpace, AddressTransfers, BlockId, Flow};

mod graph;
mod trie;

#[cfg(test)]
mod tests;

/// The ways that the runtime takes when an instruction of a range throws.
///
/// Each instruction of `protected` has all `paths`. When two routes protect
/// one instruction, it has the paths of both.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnwindRoute<A> {
    /// The half-open address range of the protected instructions. It starts
    /// and ends on instruction starts, and it can end at the end of the code.
    pub protected: Range<A>,
    /// The ways that the runtime takes, each one an ordered list of steps
    /// and an end.
    pub paths: Vec<UnwindPath<A>>,
}

/// One way that the runtime takes after a protected instruction throws.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnwindPath<A> {
    /// The actions that the runtime does first, in order.
    pub steps: Vec<UnwindStep<A>>,
    /// Where the way ends after the last step.
    pub end: UnwindEnd<A>,
}

/// One action of an [`UnwindPath`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnwindStep<A> {
    /// The runtime enters the funclet at this address; the funclet returns to
    /// the runtime.
    Funclet(A),
    /// An action that is no block of the graph, such as a destructor call or
    /// a filter call. The tag is opaque: the builder keeps it, and the
    /// consumer gives it a meaning.
    Call(u32),
}

/// The end of an [`UnwindPath`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum UnwindEnd<A> {
    /// The runtime enters the handler block at this address: a catch clause
    /// or an `__except` block.
    Handler(A),
    /// The exception leaves the function.
    Propagate,
    /// The execution resumes at the protected instruction itself: a filter
    /// that returns "continue execution".
    ///
    /// A path that ends with `Resume` has no [`UnwindStep::Funclet`]. The
    /// runtime resumes the execution before it unwinds, and one funclet
    /// instance would serve many protected instructions, so it could not
    /// return to each of them.
    Resume,
}

/// One way out of a block that holds a protected instruction or the return
/// of a funclet instance.
///
/// [`AddressGraph::add_unwind_routes`](crate::AddressGraph::add_unwind_routes)
/// returns one hop for each source block and end. The graph keeps no record
/// of the opaque calls, so the hop tells the consumer which of them the
/// runtime can run before it takes the way.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnwindHop {
    /// The block that the way leaves.
    pub source: BlockId,
    /// Where the way goes.
    pub end: HopEnd,
    /// The distinct lists of [`UnwindStep::Call`] tags, each one in path
    /// order, that the runtime can run before it takes this hop. The lists
    /// are sorted and distinct. A list can be empty.
    pub calls: Vec<Vec<u32>>,
}

/// The end of an [`UnwindHop`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum HopEnd {
    /// The way goes to this block through an edge of the graph. The block is
    /// the source block itself for [`UnwindEnd::Resume`].
    Block(BlockId),
    /// The exception leaves the function ([`UnwindEnd::Propagate`]). The
    /// graph has no edge for this hop.
    Leave,
}

/// Tests whether `flow` can end a funclet: a return, or a jump to code that
/// the stream does not hold. The jump is a tail call, so the callee returns to
/// the runtime for the funclet.
pub(super) fn ends_funclet<A: AddressSpace, K>(flow: &Flow<A, K>, addresses: &[A]) -> bool {
    match flow {
        Flow::Return => true,
        Flow::Jump { target } => addresses.binary_search(target).is_err(),
        _ => false,
    }
}

/// Resolves a protected range to the positions of its instructions.
///
/// `code_end` is the greatest exclusive end address of the stream.
pub(super) fn range_positions<A: AddressSpace>(
    range: &Range<A>,
    code_end: Option<A>,
    addresses: &[A],
) -> Result<Range<usize>, AddressBuildError<A>> {
    let (start, end) = (range.start, range.end);
    if end <= start {
        return Err(AddressBuildError::EmptyProtectedRange { start, end });
    }
    let start_position = addresses
        .binary_search(&start)
        .map_err(|_| AddressBuildError::MissingRangeStart { address: start })?;
    let end_position = if code_end.is_some_and(|code_end| end == code_end) {
        addresses.len()
    } else {
        addresses
            .binary_search(&end)
            .map_err(|_| AddressBuildError::MissingRangeEnd { address: end })?
    };
    Ok(start_position..end_position)
}

/// Rejects a route that has a path with a funclet step and a resume end.
fn check_resume<A>(index: usize, route: &UnwindRoute<A>) -> Result<(), AddressBuildError<A>> {
    let invalid = route.paths.iter().any(|path| {
        matches!(path.end, UnwindEnd::Resume)
            && path
                .steps
                .iter()
                .any(|step| matches!(step, UnwindStep::Funclet(_)))
    });
    if invalid {
        Err(AddressBuildError::ResumeAfterFunclet { route: index })
    } else {
        Ok(())
    }
}

/// Checks the routes against the stream, marks each protected instruction,
/// each funclet entry and each handler as a leader and, when a route exists,
/// marks each instruction that can end a funclet as a leader.
pub(super) fn validate_routes<I: AddressInstruction>(
    routes: &[UnwindRoute<I::Address>],
    code_end: Option<I::Address>,
    addresses: &[I::Address],
    transfers: &AddressTransfers<I>,
    leaders: &mut [bool],
) -> Result<(), AddressBuildError<I::Address>> {
    for (index, route) in routes.iter().enumerate() {
        let positions = range_positions(&route.protected, code_end, addresses)?;
        check_resume(index, route)?;
        if positions.end < leaders.len() {
            leaders[positions.end] = true;
        }
        for position in positions {
            leaders[position] = true;
        }
        for path in &route.paths {
            let funclets = path.steps.iter().filter_map(|step| match step {
                UnwindStep::Funclet(address) => Some(*address),
                UnwindStep::Call(_) => None,
            });
            let handler = match path.end {
                UnwindEnd::Handler(address) => Some(address),
                UnwindEnd::Propagate | UnwindEnd::Resume => None,
            };
            for address in funclets.chain(handler) {
                let position = addresses
                    .binary_search(&address)
                    .map_err(|_| AddressBuildError::MissingRouteEntry { address })?;
                leaders[position] = true;
            }
        }
    }
    if !routes.is_empty() {
        for (position, flow) in transfers {
            if ends_funclet(flow, addresses) {
                leaders[*position] = true;
            }
        }
    }
    Ok(())
}
