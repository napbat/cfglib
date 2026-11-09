//! A small backward problem that exercises the block fixpoint solver.

extern crate alloc;

use alloc::collections::BTreeSet;

use crate::InstrInfo;
use crate::block::BlockId;
use crate::dataflow::cfg_view::CfgView;
use crate::dataflow::fixpoint::{Direction, Problem};

use super::DfInst;

/// Block-level live variables of [`DfInst`] code.
///
/// The transfer walks the block in reverse: a definition kills its variable
/// and a use makes it live. Unwind edges get no special treatment, so the
/// facts are those of plain textbook liveness.
pub struct LiveVariables;

impl<V: CfgView<Inst = DfInst>> Problem<V> for LiveVariables {
    type Fact = BTreeSet<u16>;

    fn direction(&self) -> Direction {
        Direction::Backward
    }

    fn bottom(&self, _view: &V) -> Self::Fact {
        BTreeSet::new()
    }

    fn boundary(&self, _view: &V) -> Self::Fact {
        BTreeSet::new()
    }

    fn meet(&self, a: &Self::Fact, b: &Self::Fact) -> Self::Fact {
        a.union(b).copied().collect()
    }

    fn transfer(&self, view: &V, block: BlockId, live_out: &Self::Fact) -> Self::Fact {
        let mut live = live_out.clone();
        for inst in view.instructions(block).iter().rev() {
            for def in inst.defs() {
                live.remove(def);
            }
            live.extend(inst.uses().iter().copied());
        }
        live
    }
}
