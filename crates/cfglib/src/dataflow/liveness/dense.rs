//! Liveness with a dense bit row as the facts of each block.

extern crate alloc;
use alloc::vec::Vec;

use super::rows::solve;
use crate::block::BlockId;
use crate::dataflow::bits::{DenseRow, FlatRows};
use crate::dataflow::cfg_view::CfgView;
use crate::{InstrInfo, VariableId};

/// Result of a liveness analysis with a [`DenseRow`] as the facts of each
/// block.
///
/// The solve is the one of [`Liveness`](super::Liveness), with the same
/// answers and the same unwind rule: an unwind departure uses only the writes
/// before the first instruction of its block that may unwind, and the reads of
/// the block join both paths. `Liveness` turns the row of every block into a
/// `BTreeSet`; this result keeps the solve's two flat buffers of rows, so the
/// facts of every block cost one bit per variable and no allocation of their
/// own.
///
/// Bit `i` of a row stands for [`variable(i)`](Self::variable). The variables
/// are the ones the graph reads or writes and the seeds name, sorted, so the
/// [`ones`](DenseRow::ones) of a row visit the live variables in the order a
/// `BTreeSet` of them iterates.
///
/// # Examples
///
/// ```
/// # use cfglib::{Cfg, EdgeKind, InstrInfo};
/// # #[derive(Debug, Clone)]
/// # struct Inst { uses: Vec<u16>, defs: Vec<u16> }
/// # impl InstrInfo for Inst {
/// #     type Variable = u16;
/// #     fn uses(&self) -> &[u16] { &self.uses }
/// #     fn defs(&self) -> &[u16] { &self.defs }
/// # }
/// use cfglib::DenseLiveness;
///
/// let mut cfg = Cfg::<Inst>::new();
/// let b0 = cfg.entry();
/// let b1 = cfg.new_block();
/// cfg.add_edge(b0, b1, EdgeKind::Fallthrough);
/// cfg.block_mut(b0).push(Inst { uses: vec![7], defs: vec![3] });
/// cfg.block_mut(b1).push(Inst { uses: vec![3], defs: vec![] });
///
/// let live = DenseLiveness::compute(&cfg);
/// let r3 = live.index_of(&3).expect("the graph names r3");
/// assert!(live.live_out(b0).get(r3));
/// let live_in: Vec<u16> = live.live_in(b0).ones().map(|bit| *live.variable(bit)).collect();
/// assert_eq!(live_in, [7]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DenseLiveness<V> {
    /// The variables that the bits of a row stand for, sorted.
    variables: Vec<V>,
    /// The variables live at the entry of each block slot.
    live_in: FlatRows,
    /// The variables live after normal completion of each block slot.
    live_out: FlatRows,
}

impl<V: VariableId> DenseLiveness<V> {
    /// Runs liveness on `cfg`.
    #[must_use]
    pub fn compute<G>(cfg: &G) -> Self
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        Self::compute_with_exits(cfg, |_| Vec::new())
    }

    /// Runs liveness seeded with what leaves each block.
    ///
    /// The seed is the one of
    /// [`Liveness::compute_with_exits`](super::Liveness::compute_with_exits):
    /// the variables of `live_out(block)` are live at the end of `block`, and
    /// [`live_out`](Self::live_out) holds them. A seeded variable that the
    /// graph never names still has a bit.
    #[must_use]
    pub fn compute_with_exits<G>(cfg: &G, live_out: impl Fn(BlockId) -> Vec<V>) -> Self
    where
        G: CfgView,
        G::Inst: InstrInfo<Variable = V>,
    {
        let rows = solve(cfg, live_out);
        let len = rows.universe.len();
        Self {
            variables: rows.universe.into_variables(),
            live_in: FlatRows::new(len, rows.bound, rows.live_in),
            live_out: FlatRows::new(len, rows.bound, rows.live_out),
        }
    }

    /// The variables that the bits of a row stand for, in bit order.
    ///
    /// Every row has one bit for each, so the length of this slice is the
    /// universe of every row.
    #[must_use]
    pub fn variables(&self) -> &[V] {
        &self.variables
    }

    /// The variable that bit `index` of a row stands for.
    ///
    /// # Panics
    ///
    /// Panics when `index` is not below the length of
    /// [`variables`](Self::variables).
    #[must_use]
    pub fn variable(&self, index: usize) -> &V {
        &self.variables[index]
    }

    /// The bit that stands for `variable`, or `None` for a variable that the
    /// graph never names and no seed states, which is never live.
    #[must_use]
    pub fn index_of(&self, variable: &V) -> Option<usize> {
        self.variables.binary_search(variable).ok()
    }

    /// The variables live at the entry of `block`.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn live_in(&self, block: BlockId) -> DenseRow<'_> {
        self.live_in.row(block.index())
    }

    /// The variables live after normal completion of `block`, with its seed.
    ///
    /// An unwind that leaves before a throwing instruction does not read this
    /// row, as for [`Liveness::live_out`](super::Liveness::live_out).
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn live_out(&self, block: BlockId) -> DenseRow<'_> {
        self.live_out.row(block.index())
    }

    /// Whether `variable` is live at the entry of `block`.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn is_live_in(&self, variable: &V, block: BlockId) -> bool {
        self.index_of(variable)
            .is_some_and(|index| self.live_in(block).get(index))
    }

    /// Whether `variable` is live after normal completion of `block`.
    ///
    /// # Panics
    ///
    /// Panics when `block` is not below the node bound of the view that
    /// computed the facts.
    #[must_use]
    pub fn is_live_out(&self, variable: &V, block: BlockId) -> bool {
        self.index_of(variable)
            .is_some_and(|index| self.live_out(block).get(index))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::Cfg;
    use crate::dataflow::liveness::Liveness;
    use crate::edge::EdgeKind;
    use crate::test_util::protected_region;
    use crate::test_util::{DfInst, df_copy, df_def as def, df_ff, df_throwing, df_use as use_};
    use alloc::collections::BTreeSet;
    use alloc::vec;

    /// The variables of a row, through the universe of `live`.
    fn set(live: &DenseLiveness<u16>, row: DenseRow<'_>) -> BTreeSet<u16> {
        row.ones().map(|index| *live.variable(index)).collect()
    }

    fn assert_rows_match_the_sets(
        cfg: &Cfg<DfInst>,
        dense: &DenseLiveness<u16>,
        sets: &Liveness<u16>,
    ) {
        for block in cfg.block_ids() {
            assert_eq!(
                &set(dense, dense.live_in(block)),
                sets.live_in(block),
                "{block}"
            );
            assert_eq!(
                &set(dense, dense.live_out(block)),
                sets.live_out(block),
                "{block}"
            );
        }
    }

    /// A loop that carries a variable above 64 around its back edge, so the
    /// rows hold more than one word, with a seed at the exit that the graph
    /// never names.
    #[test]
    fn the_rows_hold_the_sets_across_a_loop_and_a_seed() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let head = cfg.new_block();
        let body = cfg.new_block();
        let exit = cfg.new_block();
        cfg.block_mut(cfg.entry()).push(def("def_r0", 0));
        cfg.block_mut(cfg.entry()).push(def("def_r70", 70));
        cfg.block_mut(head).push(use_("use_r0", 0));
        cfg.block_mut(body).push(df_copy("copy_r1_r70", 1, 70));
        cfg.block_mut(body).push(df_copy("copy_r0_r1", 0, 1));
        cfg.block_mut(exit).push(def("def_r2", 2));
        cfg.add_edge(cfg.entry(), head, EdgeKind::Fallthrough);
        cfg.add_edge(head, body, EdgeKind::ConditionalTrue);
        cfg.add_edge(head, exit, EdgeKind::ConditionalFalse);
        cfg.add_edge(body, head, EdgeKind::Fallthrough);
        let seed = |block| {
            if block == exit {
                vec![2_u16, 99]
            } else {
                Vec::new()
            }
        };

        let dense = DenseLiveness::compute_with_exits(&cfg, seed);
        assert_rows_match_the_sets(&cfg, &dense, &Liveness::compute_with_exits(&cfg, seed));
        assert_eq!(dense.variables(), [0, 1, 2, 70, 99]);
        assert!(dense.is_live_in(&70, head), "the loop carries r70");
        assert!(
            dense.is_live_in(&99, exit),
            "a seed nothing defines stays live"
        );
        assert!(
            dense.is_live_out(&99, exit),
            "the seed is in the live-out row"
        );
        assert!(!dense.is_live_out(&2, cfg.entry()));
        assert_eq!(dense.index_of(&5), None, "the graph never names r5");
        assert!(!dense.is_live_in(&5, head));
        assert_eq!(dense.live_in(head).len(), 5, "one bit per variable");
    }

    #[test]
    fn an_unwind_reads_only_the_writes_completed_before_the_throw() {
        let (mut cfg, [protected, normal, handler]) = protected_region([
            def("first", 0),
            df_throwing(df_ff("call")),
            def("second", 0),
            def("result", 1),
        ]);
        cfg.block_mut(normal).push(use_("success", 1));
        cfg.block_mut(handler).push(use_("catch", 0));

        let dense = DenseLiveness::compute(&cfg);
        assert_rows_match_the_sets(&cfg, &dense, &Liveness::compute(&cfg));
        assert!(
            !dense.is_live_in(&0, protected),
            "the first write precedes every unwind"
        );
        assert!(
            !dense.is_live_in(&1, protected),
            "the normal path writes r1 first"
        );
        assert!(dense.is_live_out(&1, protected));
        assert!(
            !dense.is_live_out(&0, protected),
            "the normal path never reads r0"
        );
    }

    #[test]
    fn a_graph_without_variables_has_empty_rows() {
        let mut cfg: Cfg<DfInst> = Cfg::new();
        let exit = cfg.new_block();
        cfg.add_edge(cfg.entry(), exit, EdgeKind::Fallthrough);
        assert!(cfg.remove_block(exit));

        let dense = DenseLiveness::compute(&cfg);
        assert_eq!(dense.variables(), [0_u16; 0]);
        assert_eq!(dense.live_in(cfg.entry()).len(), 0, "a row has no bit");
        assert_eq!(dense.live_out(exit).len(), 0, "a removed slot keeps a row");
        assert_eq!(dense.index_of(&0), None);
    }

    #[test]
    #[should_panic(expected = "row 1 outside 1 rows")]
    fn a_block_beyond_the_solved_view_panics() {
        let cfg: Cfg<DfInst> = Cfg::new();
        let _ = DenseLiveness::compute(&cfg).live_in(BlockId::from_index(1));
    }
}
