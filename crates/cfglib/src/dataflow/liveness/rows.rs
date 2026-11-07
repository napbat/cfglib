//! The bit-row solve that both liveness results read.

extern crate alloc;
use alloc::collections::BTreeSet;
use alloc::vec::Vec;

use crate::InstrOperands;
use crate::block::BlockId;
use crate::dataflow::bits::{bit_slot, flat_row, ones_of};
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::unwind::is_unwind_edge;

/// The variables of one solve, sorted, so that a variable's position is its
/// bit in every row.
pub(super) struct Universe<V> {
    variables: Vec<V>,
}

impl<V: Ord + Clone> Universe<V> {
    /// The variables that `cfg` reads or writes through `operands`, and those
    /// of `seeds`.
    fn of<G, O>(cfg: &G, operands: &O, seeds: &[Vec<V>]) -> Self
    where
        G: CfgView,
        O: InstrOperands<G::Inst, Variable = V>,
    {
        let mut variables: Vec<V> = seeds.iter().flatten().cloned().collect();
        for block in cfg.node_ids() {
            for instruction in cfg.instructions(block) {
                variables.extend(operands.uses(instruction).iter().cloned());
                variables.extend(operands.defs(instruction).iter().cloned());
            }
        }
        variables.sort();
        variables.dedup();
        Self { variables }
    }

    /// The number of variables, which is the length of a row in bits.
    pub(super) fn len(&self) -> usize {
        self.variables.len()
    }

    /// The words of one row.
    fn words(&self) -> usize {
        self.variables.len().div_ceil(64)
    }

    /// Sets the bit of `variable` in `row`.
    fn insert(&self, row: &mut [u64], variable: &V) {
        if let Ok(index) = self.variables.binary_search(variable) {
            let (word, bit) = bit_slot(index);
            row[word] |= bit;
        }
    }

    /// Clears the bit of `variable` in `row`.
    fn remove(&self, row: &mut [u64], variable: &V) {
        if let Ok(index) = self.variables.binary_search(variable) {
            let (word, bit) = bit_slot(index);
            row[word] &= !bit;
        }
    }

    /// Returns the variables of `row`.
    pub(super) fn set(&self, row: &[u64]) -> BTreeSet<V> {
        ones_of(row)
            .map(|index| self.variables[index].clone())
            .collect()
    }

    /// The variables, in the order of their bits.
    pub(super) fn into_variables(self) -> Vec<V> {
        self.variables
    }
}

/// What each block does to liveness, one bit row per block in a flat buffer.
struct BlockRows {
    /// The words of one row.
    words: usize,
    /// The upward-exposed reads of each block.
    reads: Vec<u64>,
    /// The writes of each block.
    writes: Vec<u64>,
    /// The seed of each block.
    seeded: Vec<u64>,
    /// Per block, whether an instruction may unwind. Empty when no edge of
    /// the graph unwinds.
    unwinding: Vec<bool>,
    /// Per block, the writes before the first possible unwind: the only
    /// writes every unwind departure has completed. Empty when no edge of
    /// the graph unwinds.
    before_unwind: Vec<u64>,
}

impl BlockRows {
    fn of<G, O>(
        cfg: &G,
        operands: &O,
        universe: &Universe<O::Variable>,
        seeds: &[Vec<O::Variable>],
        has_unwind: bool,
    ) -> Self
    where
        G: CfgView,
        O: InstrOperands<G::Inst>,
    {
        let bound = cfg.node_bound();
        let words = universe.words();
        let mut rows = Self {
            words,
            reads: alloc::vec![0; bound * words],
            writes: alloc::vec![0; bound * words],
            seeded: alloc::vec![0; bound * words],
            unwinding: alloc::vec![false; if has_unwind { bound } else { 0 }],
            before_unwind: alloc::vec![0; if has_unwind { bound * words } else { 0 }],
        };
        for block in cfg.node_ids() {
            let index = block.index();
            let range = flat_row(words, index);
            for variable in &seeds[index] {
                universe.insert(&mut rows.seeded[range.clone()], variable);
            }
            if has_unwind {
                for instruction in cfg.instructions(block) {
                    if operands.may_unwind(instruction) {
                        rows.unwinding[index] = true;
                        break;
                    }
                    for variable in operands.defs(instruction) {
                        universe.insert(&mut rows.before_unwind[range.clone()], variable);
                    }
                }
            }
            for instruction in cfg.instructions(block).iter().rev() {
                for variable in operands.defs(instruction) {
                    universe.remove(&mut rows.reads[range.clone()], variable);
                    universe.insert(&mut rows.writes[range.clone()], variable);
                }
                for variable in operands.uses(instruction) {
                    universe.insert(&mut rows.reads[range.clone()], variable);
                }
            }
        }
        rows
    }
}

/// The facts of one liveness solve, one bit row per block slot in a flat
/// buffer. A slot without a block has an empty row.
pub(super) struct LiveRows<V> {
    /// The variables that the bits of a row stand for.
    pub(super) universe: Universe<V>,
    /// The node bound of the solved view: the number of rows.
    pub(super) bound: usize,
    /// The words of one row.
    pub(super) words: usize,
    /// The variables live at the entry of each block.
    pub(super) live_in: Vec<u64>,
    /// The variables live after normal completion of each block, with the
    /// seed of the block joined in.
    pub(super) live_out: Vec<u64>,
    /// The number of worklist entries the solve processed.
    pub(super) steps: usize,
}

impl<V> LiveRows<V> {
    /// The live-in row of each block slot, in slot order.
    pub(super) fn live_in_rows(&self) -> impl Iterator<Item = &[u64]> {
        let words = self.words;
        (0..self.bound).map(move |block| &self.live_in[flat_row(words, block)])
    }

    /// The live-out row of each block slot, in slot order.
    pub(super) fn live_out_rows(&self) -> impl Iterator<Item = &[u64]> {
        let words = self.words;
        (0..self.bound).map(move |block| &self.live_out[flat_row(words, block)])
    }
}

/// Solves liveness over the operands `operands` reports and returns the rows
/// of every block, with the seed of a block in its live-out row.
///
/// Normal departures use all completed definitions. Unwind departures use
/// only definitions before the first possible unwind. The block's reads
/// join both paths. Bit rows keep the repeated joins allocation-free.
pub(super) fn solve<V, O>(
    cfg: &V,
    operands: &O,
    live_out: impl Fn(BlockId) -> Vec<O::Variable>,
) -> LiveRows<O::Variable>
where
    V: CfgView,
    O: InstrOperands<V::Inst>,
{
    let bound = cfg.node_bound();
    let mut seeds: Vec<Vec<O::Variable>> = alloc::vec![Vec::new(); bound];
    for block in cfg.node_ids() {
        seeds[block.index()] = live_out(block);
    }
    let universe = Universe::of(cfg, operands, &seeds);
    let has_unwind = cfg.edge_ids().any(|edge| is_unwind_edge(cfg.edge(edge)));
    let rows = BlockRows::of(cfg, operands, &universe, &seeds, has_unwind);
    let words = rows.words;

    let mut live_in = alloc::vec![0u64; bound * words];
    let mut flowing_out = alloc::vec![0u64; bound * words];
    let mut queued = alloc::vec![false; bound];
    let mut worklist: Vec<BlockId> = cfg.node_ids().collect();
    for block in &worklist {
        queued[block.index()] = true;
    }
    let mut next = alloc::vec![0u64; words];
    let mut unwind_out = alloc::vec![0u64; if has_unwind { words } else { 0 }];
    let mut steps = 0;
    while let Some(block) = worklist.pop() {
        steps += 1;
        let index = block.index();
        queued[index] = false;
        let out = &mut flowing_out[flat_row(words, index)];
        out.fill(0);
        unwind_out.fill(0);
        for edge in cfg.outgoing(block) {
            let edge = cfg.edge(edge);
            let destination = if has_unwind && rows.unwinding[index] && is_unwind_edge(edge) {
                &mut unwind_out[..]
            } else {
                &mut out[..]
            };
            for (word, from) in destination
                .iter_mut()
                .zip(&live_in[flat_row(words, edge.target().index())])
            {
                *word |= *from;
            }
        }
        for (position, word) in next.iter_mut().enumerate() {
            let at = index * words + position;
            *word = ((out[position] | rows.seeded[at]) & !rows.writes[at]) | rows.reads[at];
            if has_unwind {
                *word |= unwind_out[position] & !rows.before_unwind[at];
            }
        }
        if next[..] != live_in[flat_row(words, index)] {
            live_in[flat_row(words, index)].copy_from_slice(&next);
            for predecessor in cfg.predecessors(block) {
                if !queued[predecessor.index()] {
                    queued[predecessor.index()] = true;
                    worklist.push(predecessor);
                }
            }
        }
    }

    for (word, seed) in flowing_out.iter_mut().zip(&rows.seeded) {
        *word |= *seed;
    }
    LiveRows {
        universe,
        bound,
        words,
        live_in,
        live_out: flowing_out,
        steps,
    }
}
