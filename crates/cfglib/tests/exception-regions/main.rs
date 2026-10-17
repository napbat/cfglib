//! Consumer-style tests of exception regions, modeling symtree-cfg's try/finally
//! lowering and the table-driven frontends (dasm, elevator) that name only a
//! handler entry: regions and handlers on a `Cfg`, cleanup routes, handler
//! metadata, extent recovery and promotion, `EhModel` classification, the text
//! form, and how compaction and the cleanup transforms treat all of it.
//!
//! The transform tests state the contract any sane pass owes a graph with
//! regions, rather than whatever the passes do today: afterwards every region
//! and handler reference names a live block, `compact` does not panic, and a
//! handler is still entered through an unwind edge from exactly the code it
//! protected.

/// The package's golden-file comparison, shared with the unit tests.
#[path = "../../src/test_util/golden.rs"]
mod golden;

mod cleanup_transforms;
mod cleanups;
mod compaction;
mod eh_model;
mod extents;
mod handler_metadata;
mod regions;
mod shapes;
mod text_round_trip;
