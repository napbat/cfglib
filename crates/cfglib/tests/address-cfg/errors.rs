//! Every [`AddressBuildError`] variant with the input that triggers it, the
//! empty stream, and the error's `Display` and `Error` behavior.
//!
//! The streams here are decoder output a position-indexed assembler cannot
//! spell: spans with chosen starts and ends. `Span` is the minimum
//! instruction a decoder could hand over, generic in the address type so each
//! case runs over both `u32` and `u64`.

use core::fmt::{Debug, Display};
use std::error::Error;

use cfglib::{
    AddressBuildError, AddressCfgOptions, AddressHandler, AddressInstruction, AddressSpace,
    CallPolicy, Flow, HandlerKind, verify,
};

use crate::bytecode::{Bytecode, Opcode};
use crate::harness::{assert_empty, plain_options as no_options, try_build};
use crate::machine::{Machine, Mnemonic};

/// What the error scenarios need of an address type.
trait Width: AddressSpace + From<u8> + Into<u64> + Debug + Display + 'static {}

impl Width for u32 {}
impl Width for u64 {}

/// A decoded span of code that falls through.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Span<A> {
    address: A,
    end: Option<A>,
}

impl<A: AddressSpace> AddressInstruction for Span<A> {
    type Address = A;
    type CaseKey = i8;

    fn address(&self) -> A {
        self.address
    }

    fn end_address(&self) -> Option<A> {
        self.end
    }

    fn flow(&self) -> Flow<A, i8> {
        Flow::Next
    }
}

/// The address `n`, which fits every address type used here.
fn at<A: Width>(n: u8) -> A {
    A::from(n)
}

fn span<A: Width>(start: u8, end: u8) -> Span<A> {
    Span {
        address: at(start),
        end: Some(at(end)),
    }
}

/// Two adjacent four-byte spans, `0x10..0x14` and `0x14..0x18`.
fn two_spans<A: Width>() -> Vec<Span<A>> {
    vec![span(0x10, 0x14), span(0x14, 0x18)]
}

fn catch_range<A: Width>(start: u8, end: u8, entry: u8) -> AddressHandler<A> {
    AddressHandler {
        protected: at(start)..at(end),
        entry: at(entry),
        kind: HandlerKind::Catch,
    }
}

fn entry_at<A: Width>(entry: u8) -> AddressCfgOptions<A> {
    AddressCfgOptions {
        entry: Some(at(entry)),
        calls: CallPolicy::Edges,
    }
}

fn error_of<A: Width>(
    instructions: Vec<Span<A>>,
    handlers: &[AddressHandler<A>],
    options: AddressCfgOptions<A>,
) -> AddressBuildError<A> {
    match try_build(instructions, handlers, options) {
        Ok(_) => panic!("the stream was accepted"),
        Err(error) => error,
    }
}

fn a_zero_size_instruction_is_rejected<A: Width>() {
    let stream = vec![span(0x10, 0x14), span(0x14, 0x14)];
    assert_eq!(
        error_of::<A>(stream, &[], no_options()),
        AddressBuildError::ZeroSizeInstruction { address: at(0x14) }
    );
}

fn instruction_starts_must_strictly_increase<A: Width>() {
    let descending = vec![span(0x20, 0x24), span(0x10, 0x14)];
    assert_eq!(
        error_of::<A>(descending, &[], no_options()),
        AddressBuildError::UnorderedInstruction {
            previous: at(0x20),
            address: at(0x10),
        }
    );
    let duplicate = vec![span(0x10, 0x14), span(0x10, 0x18)];
    assert_eq!(
        error_of::<A>(duplicate, &[], no_options()),
        AddressBuildError::UnorderedInstruction {
            previous: at(0x10),
            address: at(0x10),
        }
    );
}

fn an_instruction_that_ends_outside_the_address_space_is_rejected<A: Width>() {
    let stream = vec![
        span(0x10, 0x14),
        Span {
            address: at(0x14),
            end: None,
        },
    ];
    assert_eq!(
        error_of::<A>(stream, &[], no_options()),
        AddressBuildError::AddressOverflow { address: at(0x14) }
    );
}

fn an_entry_that_is_not_an_instruction_start_is_rejected<A: Width>() {
    for entry in [0x08, 0x11, 0x18, 0x40] {
        assert_eq!(
            error_of::<A>(two_spans(), &[], entry_at(entry)),
            AddressBuildError::MissingEntry { address: at(entry) },
            "entry {entry:#x}"
        );
    }
    assert!(try_build(two_spans::<A>(), &[], entry_at(0x14)).is_ok());
}

fn an_empty_protected_range_is_rejected<A: Width>() {
    for (start, end) in [(0x14, 0x14), (0x14, 0x10)] {
        assert_eq!(
            error_of::<A>(two_spans(), &[catch_range(start, end, 0x10)], no_options()),
            AddressBuildError::EmptyProtectedRange {
                start: at(start),
                end: at(end),
            }
        );
    }
}

fn a_protected_range_must_start_on_an_instruction<A: Width>() {
    for start in [0x11, 0x40] {
        assert_eq!(
            error_of::<A>(two_spans(), &[catch_range(start, 0x50, 0x10)], no_options()),
            AddressBuildError::MissingRangeStart { address: at(start) }
        );
    }
}

fn a_protected_range_must_end_on_an_instruction_or_at_the_code_end<A: Width>() {
    assert_eq!(
        error_of::<A>(two_spans(), &[catch_range(0x10, 0x15, 0x14)], no_options()),
        AddressBuildError::MissingRangeEnd { address: at(0x15) }
    );
    // The end of the code is not an instruction start, and is a valid end.
    let graph = try_build(
        two_spans(),
        &[catch_range::<A>(0x10, 0x18, 0x14)],
        no_options(),
    )
    .expect("a range may end at the end of the code");
    assert_eq!(graph.cfg.regions().len(), 1);
    assert!(verify(&graph.cfg).is_ok());
}

fn a_handler_entry_must_be_an_instruction_start<A: Width>() {
    for entry in [0x11, 0x18, 0x40] {
        assert_eq!(
            error_of::<A>(two_spans(), &[catch_range(0x10, 0x14, entry)], no_options()),
            AddressBuildError::MissingHandlerEntry { address: at(entry) }
        );
    }
}

fn an_empty_stream_builds_one_empty_entry_block<A: Width>() {
    let graph = try_build::<Span<A>>(Vec::new(), &[], no_options()).expect("empty is valid");
    assert_eq!(graph.cfg.block_count(), 1);
    assert!(graph.cfg.block(graph.cfg.entry()).is_empty());
    assert_eq!(graph.cfg.edge_count(), 0);
    assert!(graph.instruction_blocks.is_empty());
    assert_empty(&graph.handler_refs);
    assert_empty(&graph.unresolved_transfers);
    assert_empty(&graph.calls);
    assert_empty(graph.cfg.regions());
    assert!(verify(&graph.cfg).is_ok());
}

fn an_empty_stream_has_no_entry_and_no_protectable_range<A: Width>() {
    assert_eq!(
        error_of::<A>(Vec::<Span<A>>::new(), &[], entry_at(0x10)),
        AddressBuildError::MissingEntry { address: at(0x10) }
    );
    assert_eq!(
        error_of::<A>(
            Vec::<Span<A>>::new(),
            &[catch_range(0x10, 0x14, 0x10)],
            no_options()
        ),
        AddressBuildError::MissingRangeStart { address: at(0x10) }
    );
}

/// Every variant, each triggered by its own input, with the addresses its
/// message must mention.
fn one_of_each_error<A: Width>() -> Vec<(AddressBuildError<A>, Vec<&'static str>)> {
    let overflow = Span {
        address: at(0x10),
        end: None,
    };
    vec![
        (
            error_of::<A>(vec![span(0x10, 0x10)], &[], no_options()),
            vec!["16"],
        ),
        (
            error_of::<A>(vec![span(0x20, 0x24), span(0x10, 0x14)], &[], no_options()),
            vec!["32", "16"],
        ),
        (error_of::<A>(vec![overflow], &[], no_options()), vec!["16"]),
        (error_of::<A>(two_spans(), &[], entry_at(0x11)), vec!["17"]),
        (
            error_of::<A>(two_spans(), &[catch_range(0x10, 0x10, 0x10)], no_options()),
            vec!["16"],
        ),
        (
            error_of::<A>(two_spans(), &[catch_range(0x11, 0x18, 0x10)], no_options()),
            vec!["17"],
        ),
        (
            error_of::<A>(two_spans(), &[catch_range(0x10, 0x15, 0x10)], no_options()),
            vec!["21"],
        ),
        (
            error_of::<A>(two_spans(), &[catch_range(0x10, 0x14, 0x11)], no_options()),
            vec!["17"],
        ),
    ]
}

fn error_messages_name_the_offending_addresses_and_differ_by_variant<A: Width>() {
    let errors = one_of_each_error::<A>();
    assert_eq!(errors.len(), 8);
    let mut seen: Vec<String> = Vec::new();
    for (error, addresses) in &errors {
        let message = error.to_string();
        for address in addresses {
            assert!(message.contains(address), "{message:?} names {address}");
        }
        assert!(!seen.contains(&message), "two variants share {message:?}");
        seen.push(message);
    }
}

fn build_errors_convert_into_boxed_errors<A: Width>() -> Result<(), Box<dyn Error>> {
    let result = try_build(
        two_spans::<A>(),
        &[catch_range(0x10, 0x14, 0x11)],
        no_options(),
    );
    let error: &dyn Error = result.as_ref().expect_err("the handler entry is invalid");
    assert!(error.source().is_none());
    result?;
    Ok(())
}

macro_rules! over_both_widths {
    ($($scenario:ident),+ $(,)?) => {
        $(
            mod $scenario {
                #[test]
                fn u32_addresses() {
                    super::$scenario::<u32>();
                }

                #[test]
                fn u64_addresses() {
                    super::$scenario::<u64>();
                }
            }
        )+
    };
}

over_both_widths!(
    a_zero_size_instruction_is_rejected,
    instruction_starts_must_strictly_increase,
    an_instruction_that_ends_outside_the_address_space_is_rejected,
    an_entry_that_is_not_an_instruction_start_is_rejected,
    an_empty_protected_range_is_rejected,
    a_protected_range_must_start_on_an_instruction,
    a_protected_range_must_end_on_an_instruction_or_at_the_code_end,
    a_handler_entry_must_be_an_instruction_start,
    an_empty_stream_builds_one_empty_entry_block,
    an_empty_stream_has_no_entry_and_no_protectable_range,
    error_messages_name_the_offending_addresses_and_differ_by_variant,
);

#[test]
fn a_rejected_build_converts_into_a_boxed_error_over_u32() {
    let error = build_errors_convert_into_boxed_errors::<u32>().expect_err("the handler is bad");
    assert!(error.to_string().contains("17"));
}

#[test]
fn a_rejected_build_converts_into_a_boxed_error_over_u64() {
    let error = build_errors_convert_into_boxed_errors::<u64>().expect_err("the handler is bad");
    assert!(error.to_string().contains("17"));
}

#[test]
fn an_instruction_ending_at_the_top_of_the_u32_space_is_valid_and_one_past_it_overflows() {
    let last = |offset: u32| Bytecode {
        offset,
        opcode: Opcode::Ret,
    };
    // `Ret` is one byte: it ends at `u32::MAX`, the last representable
    // exclusive end.
    let graph = try_build(vec![last(u32::MAX - 1)], &[], no_options()).unwrap();
    assert_eq!(graph.cfg.block_count(), 1);
    assert_eq!(
        try_build(vec![last(u32::MAX)], &[], no_options()).err(),
        Some(AddressBuildError::AddressOverflow { address: u32::MAX })
    );
}

#[test]
fn an_instruction_ending_at_the_top_of_the_u64_space_is_valid_and_one_past_it_overflows() {
    let last = |address: u64| Machine {
        address,
        length: 1,
        mnemonic: Mnemonic::Ret,
        faults: false,
    };
    let graph = try_build(vec![last(u64::MAX - 1)], &[], no_options()).unwrap();
    assert_eq!(graph.cfg.block_count(), 1);
    assert_eq!(
        try_build(vec![last(u64::MAX)], &[], no_options()).err(),
        Some(AddressBuildError::AddressOverflow { address: u64::MAX })
    );
}
