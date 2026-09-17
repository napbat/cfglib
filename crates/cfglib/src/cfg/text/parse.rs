//! Reading the line-oriented text form back into a control-flow graph.

extern crate alloc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use crate::block::BlockId;
use crate::edge::EdgeKind;
use crate::graph::text::{
    TextError, TextErrorKind, parse_index, significant_lines, split_tokens, unescape,
};
use crate::region::{Handler, Region, RegionId};

use super::super::Cfg;
use super::{HandlerShape, parse_handler_body};

/// Read a CFG back from the text form, parsing each instruction line with
/// `parse_instruction`.
///
/// Instruction syntax belongs to the consumer, so the closure owns it: it
/// receives one unescaped instruction line and returns the instruction or a
/// message explaining why the line is not one.
///
/// Block indices are reproduced exactly, holes included, so the graph round
/// trips: writing the result again yields the input.
///
/// # Errors
///
/// Returns the first malformed line, with its number and what was wrong.
///
/// # Examples
///
/// ```
/// use cfglib::parse_cfg_text;
///
/// let cfg = parse_cfg_text(
///     "entry bb0\nbb0:\n    nop\nbb1:\nbb0 -> bb1 fallthrough\n",
///     |line| Ok::<_, String>(line.to_owned()),
/// )
/// .unwrap();
///
/// assert_eq!(cfg.block_count(), 2);
/// assert_eq!(cfg.block(cfg.entry()).instructions(), ["nop"]);
/// ```
pub fn parse_cfg_text<I>(
    text: &str,
    parse_instruction: impl Fn(&str) -> Result<I, String>,
) -> Result<Cfg<I>, TextError> {
    let mut reader = Reader {
        cfg: Cfg::new(),
        declared: vec![false],
        next: 0,
        entry: None,
        context: Context::Nothing,
        pending: None,
    };
    reader.read(text, &parse_instruction)?;
    reader.finish()
}

/// What an indented line belongs to.
#[derive(Clone, Copy)]
enum Context {
    Nothing,
    Block(BlockId),
    Region,
}

struct Reader<I> {
    cfg: Cfg<I>,
    /// Whether each block slot was declared, rather than filling a hole.
    declared: Vec<bool>,
    /// The lowest index a block header may still declare.
    next: usize,
    entry: Option<usize>,
    context: Context,
    pending: Option<Region>,
}

impl<I> Reader<I> {
    fn read(
        &mut self,
        text: &str,
        parse_instruction: &impl Fn(&str) -> Result<I, String>,
    ) -> Result<(), TextError> {
        for (line, content) in significant_lines(text) {
            let trimmed = content.trim_start();
            if content.len() != trimmed.len() {
                match self.context {
                    Context::Region => self.read_handler(trimmed, line)?,
                    Context::Block(block) => {
                        let text = unescape(trimmed, line)?;
                        let instruction = parse_instruction(&text)
                            .map_err(|why| TextError::new(line, TextErrorKind::Instruction(why)))?;
                        self.cfg.block_mut(block).push(instruction);
                    }
                    Context::Nothing => {
                        return Err(TextError::new(line, TextErrorKind::Unrecognized));
                    }
                }
                continue;
            }

            self.flush_region();
            if let Some(rest) = trimmed.strip_prefix("entry ") {
                self.entry = Some(parse_index(rest.trim(), "bb", line)?);
                self.context = Context::Nothing;
            } else if let Some(rest) = trimmed.strip_prefix("region ") {
                self.read_region(rest, line)?;
            } else {
                self.read_block_or_edge(trimmed, line)?;
            }
        }
        self.flush_region();
        Ok(())
    }

    fn read_block_or_edge(&mut self, content: &str, line: usize) -> Result<(), TextError> {
        let (tokens, tail) = split_tokens(content, 1)
            .ok_or_else(|| TextError::new(line, TextErrorKind::Unrecognized))?;

        if let Some(rest) = tail.strip_prefix("-> ") {
            let (target_tokens, _) = split_tokens(rest, 2)
                .ok_or_else(|| TextError::new(line, TextErrorKind::Unrecognized))?;
            let source = self.live_block(parse_index(tokens[0], "bb", line)?, line)?;
            let target = self.live_block(parse_index(target_tokens[0], "bb", line)?, line)?;
            let kind = EdgeKind::from_name(target_tokens[1])
                .ok_or_else(|| TextError::new(line, TextErrorKind::EdgeKind))?;
            self.cfg.add_edge(source, target, kind);
            self.context = Context::Nothing;
            return Ok(());
        }

        let identifier = tokens[0]
            .strip_suffix(':')
            .ok_or_else(|| TextError::new(line, TextErrorKind::Unrecognized))?;
        let index = parse_index(identifier, "bb", line)?;
        let label = unescape(tail, line)?;
        self.declare_block(index, label, line)
    }

    fn declare_block(&mut self, index: usize, label: String, line: usize) -> Result<(), TextError> {
        if index < self.next {
            return Err(TextError::new(
                line,
                TextErrorKind::NodeOrder {
                    expected: self.next,
                },
            ));
        }
        while self.declared.len() <= index {
            self.cfg.new_block();
            self.declared.push(false);
        }
        let block = BlockId::from_index(index);
        if !label.is_empty() {
            self.cfg.block_mut(block).set_label(label);
        }
        self.declared[index] = true;
        self.next = index + 1;
        self.context = Context::Block(block);
        Ok(())
    }

    fn read_region(&mut self, content: &str, line: usize) -> Result<(), TextError> {
        let mut fields = content.split_whitespace();
        let index = fields
            .next()
            .ok_or_else(|| TextError::new(line, TextErrorKind::Region))
            .and_then(|token| parse_index(token, "r", line))?;
        if index != self.cfg.regions().len() {
            return Err(TextError::new(line, TextErrorKind::Region));
        }

        let mut region = Region {
            id: RegionId::from_raw(0),
            protected_blocks: alloc::collections::BTreeSet::new(),
            handlers: Vec::new(),
            parent: None,
        };
        for field in fields {
            let (key, value) = split_field(field, line)?;
            match key {
                "protected" => {
                    for token in value.split(',').filter(|token| !token.is_empty()) {
                        region
                            .protected_blocks
                            .insert(self.live_block(parse_index(token, "bb", line)?, line)?);
                    }
                }
                "parent" => {
                    region.parent = Some(RegionId::from_raw(
                        u32::try_from(parse_index(value, "r", line)?)
                            .map_err(|_| TextError::new(line, TextErrorKind::Region))?,
                    ));
                }
                _ => return Err(TextError::new(line, TextErrorKind::Region)),
            }
        }
        self.pending = Some(region);
        self.context = Context::Region;
        Ok(())
    }

    fn read_handler(&mut self, content: &str, line: usize) -> Result<(), TextError> {
        let rest = content
            .strip_prefix("handler ")
            .ok_or_else(|| TextError::new(line, TextErrorKind::Unrecognized))?;
        let mut fields = rest.split_whitespace();
        let kind_name = fields
            .next()
            .ok_or_else(|| TextError::new(line, TextErrorKind::Region))?;
        let shape = HandlerShape::from_name(kind_name)
            .ok_or_else(|| TextError::new(line, TextErrorKind::Region))?;

        let mut entry = None;
        let mut filter_block = None;
        let mut body = None;
        for field in fields {
            let (key, value) = split_field(field, line)?;
            let block = |token: &str| -> Result<BlockId, TextError> {
                let index = parse_index(token, "bb", line)?;
                self.live_block(index, line)
            };
            match key {
                "entry" => entry = Some(block(value)?),
                "filter" => filter_block = Some(block(value)?),
                "body" => {
                    body = Some(
                        parse_handler_body(value, |token| {
                            let index = parse_index(token, "bb", line).ok()?;
                            self.live_block(index, line).ok()
                        })
                        .ok_or_else(|| TextError::new(line, TextErrorKind::Region))?,
                    );
                }
                _ => return Err(TextError::new(line, TextErrorKind::Region)),
            }
        }

        let malformed = || TextError::new(line, TextErrorKind::Region);
        let handler = Handler {
            entry: entry.ok_or_else(malformed)?,
            body: body.ok_or_else(malformed)?,
            kind: shape.kind(filter_block).ok_or_else(malformed)?,
        };
        self.pending
            .as_mut()
            .ok_or_else(malformed)?
            .handlers
            .push(handler);
        Ok(())
    }

    fn flush_region(&mut self) {
        if let Some(region) = self.pending.take() {
            self.cfg.add_region(region);
        }
    }

    fn live_block(&self, index: usize, line: usize) -> Result<BlockId, TextError> {
        if self.declared.get(index).copied() == Some(true) {
            return Ok(BlockId::from_index(index));
        }
        Err(TextError::new(line, TextErrorKind::UnknownNode { index }))
    }

    fn finish(mut self) -> Result<Cfg<I>, TextError> {
        let entry = self.entry.unwrap_or(0);
        let block = self.live_block(entry, 0)?;
        self.cfg.set_entry(block);
        for index in 0..self.declared.len() {
            if !self.declared[index] {
                self.cfg.remove_block(BlockId::from_index(index));
            }
        }
        Ok(self.cfg)
    }
}

fn split_field(field: &str, line: usize) -> Result<(&str, &str), TextError> {
    field
        .split_once('=')
        .ok_or_else(|| TextError::new(line, TextErrorKind::Region))
}
