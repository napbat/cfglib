//! The line-oriented text form of a control-flow graph.
//!
//! This is the CFG's own reading of [`graph::text`](crate::graph::text): a
//! block is a header line and its instructions, an edge names its
//! [`EdgeKind`], and an exception region names the blocks it protects and
//! the handlers attached to it.
//!
//! ```text
//! entry bb0
//! bb0: preheader
//!     compare i, n
//!     branch
//! bb1:
//! bb0 -> bb1 true
//! bb0 -> bb2 false
//! region r0 protected=bb0,bb1
//!     handler catch entry=bb2 body=bb2
//! ```
//!
//! Instruction lines are the only indented lines inside a block, and handler
//! lines the only indented lines inside a region, so the reader never has to
//! guess what an indented line belongs to. Labels and instruction text are
//! escaped exactly as the store's text form escapes them, so an instruction
//! that starts with `#` or with whitespace still reads back as that
//! instruction rather than as a comment, a blank line, or indentation.
//!
//! # What it is not
//!
//! Edge weights, consumer edge payloads, and cleanup records are not in the
//! projection; the `serde` representation of the graph is the fidelity path.
//! So is the order of a block's edges when a transform redirected one: edges
//! are written, and read back, in identity order, so a redirected edge returns
//! to the place its identity gives it.
//! An instruction whose label renders empty is not representable either,
//! because an empty line carries nothing to read back.

extern crate alloc;
use alloc::string::String;
use core::fmt::{self, Write as _};

use crate::block::BlockId;
use crate::display::{DisplayInstr, IndentedWriter};
use crate::graph::text::{EscapingSink, write_label};
use crate::region::{Handler, HandlerBody, Region, RegionId};

use super::Cfg;

mod parse;

pub use parse::parse_cfg_text;

impl<I: DisplayInstr, E> Cfg<I, E> {
    /// Render this CFG in the line-oriented text form.
    ///
    /// # Panics
    ///
    /// Panics only if writing to an in-memory [`String`] unexpectedly fails.
    #[must_use]
    pub fn to_text(&self) -> String {
        let mut out = String::new();
        self.write_text(&mut out)
            .expect("writing text to a String cannot fail");
        out
    }

    /// Write this CFG in the line-oriented text form into any sink.
    ///
    /// # Errors
    ///
    /// Returns the sink's formatting error if a write fails.
    pub fn write_text(&self, sink: &mut dyn fmt::Write) -> fmt::Result {
        let mut printer = IndentedWriter::new(sink);
        writeln!(printer, "entry {}", self.entry())?;

        for id in self.block_ids() {
            let block = self.block(id);
            write!(printer, "{id}:")?;
            if let Some(name) = block.label() {
                write_label(&mut printer, name)?;
            }
            printer.end_line()?;
            printer.nested(|printer| {
                for instruction in block.instructions() {
                    printer.optional_line(|line| {
                        instruction.write_mnemonic(&mut EscapingSink::new(line))
                    })?;
                }
                Ok(())
            })?;
        }

        for id in self.edge_ids() {
            let edge = self.edge(id);
            writeln!(
                printer,
                "{} -> {} {}",
                edge.source(),
                edge.target(),
                edge.kind().name()
            )?;
        }

        for (id, region) in self.regions() {
            write_region(&mut printer, id, region, &|block| {
                self.contains_block(block)
            })?;
        }
        Ok(())
    }
}

/// Write one region. A set of blocks lists only the live ones, as
/// [`compact`](Cfg::compact) would keep: a removed block stays named in a
/// region until then, and the text form has no slot to put it in.
fn write_region(
    printer: &mut IndentedWriter<'_>,
    id: RegionId,
    region: &Region,
    live: &impl Fn(BlockId) -> bool,
) -> fmt::Result {
    write!(printer, "region r{} protected=", id.index())?;
    write_blocks(
        printer,
        region
            .protected_blocks
            .iter()
            .copied()
            .filter(|&block| live(block)),
    )?;
    if let Some(parent) = region.parent {
        write!(printer, " parent=r{}", parent.index())?;
    }
    printer.end_line()?;
    printer.nested(|printer| {
        for handler in &region.handlers {
            write_handler(printer, handler, live)?;
        }
        Ok(())
    })
}

fn write_handler(
    printer: &mut IndentedWriter<'_>,
    handler: &Handler,
    live: &impl Fn(BlockId) -> bool,
) -> fmt::Result {
    printer.indent()?;
    write!(
        printer,
        "handler {} entry={}",
        handler.kind.name(),
        handler.entry
    )?;
    if let Some(filter_block) = handler.kind.filter_block() {
        write!(printer, " filter={filter_block}")?;
    }
    printer.write_str(" body=")?;
    match handler.body.blocks() {
        None => printer.write_str("?")?,
        Some(blocks) => write_blocks(printer, blocks.iter().copied().filter(|&block| live(block)))?,
    }
    printer.end_line()
}

fn write_blocks(
    printer: &mut IndentedWriter<'_>,
    blocks: impl IntoIterator<Item = BlockId>,
) -> fmt::Result {
    for (position, block) in blocks.into_iter().enumerate() {
        if position != 0 {
            printer.write_str(",")?;
        }
        write!(printer, "{block}")?;
    }
    Ok(())
}

/// Reading a handler body back: `?` is an unknown extent, anything else the
/// complete one, the empty list included.
pub(crate) fn parse_handler_body(
    value: &str,
    mut block: impl FnMut(&str) -> Option<BlockId>,
) -> Option<HandlerBody> {
    if value == "?" {
        return Some(HandlerBody::Unknown);
    }
    if value.is_empty() {
        return Some(HandlerBody::known([]));
    }
    value
        .split(',')
        .map(&mut block)
        .collect::<Option<alloc::vec::Vec<_>>>()
        .map(HandlerBody::known)
}

#[cfg(test)]
mod tests;
