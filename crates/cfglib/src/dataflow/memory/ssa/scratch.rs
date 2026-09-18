//! Caller-owned working storage for memory SSA.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::dataflow::ssa::SsaScratch;
use crate::graph::dominator::DominatorScratch;
use crate::memory::MemoryTrace;
use crate::union_find::DisjointSet;

use super::MemoryClassId;

/// Every buffer [`MemorySSA::compute_in`](super::MemorySSA::compute_in)
/// fills, owned by the caller.
///
/// Memory SSA is three analyses in a trench coat: it traces the procedure's
/// memory events, merges their locations into alias classes, builds a shadow
/// CFG whose instructions are those events over those classes, and then runs
/// an ordinary dominator tree and an ordinary SSA form over the shadow. Every
/// one of those allocates its own working storage, and a whole-codebase pass
/// pays the lot once per callable.
///
/// This holds the trace, the location and union-find buffers of the class
/// merge, and the [`DominatorScratch`] and [`SsaScratch`] the shadow's own two
/// analyses need, so the only allocation left per call is the answer — plus
/// the shadow CFG itself, which is a [`Cfg`](crate::Cfg) and so owns storage
/// this cannot lend it.
///
/// # One scratch, many procedures
///
/// The type parameters are the location, variable, and fence vocabularies, so
/// one scratch serves every CFG whose adapter reports memory the same way. It
/// is [`Send`] when they are, so a worker thread owns one for the whole corpus
/// it is handed.
///
/// # Sizing
///
/// Nothing is fixed at construction and nothing is released by a call, so a
/// scratch used on a large procedure and then on a small one is correct and
/// the space it holds is the high-water mark of the sequence.
///
/// # Examples
///
/// ```
/// use cfglib::{Cfg, ExactMemoryAlias, MemorySSA, MemorySsaScratch};
/// # use cfglib::{InstrInfo, MemoryAccess, MemoryEvent, MemoryEventInfo};
/// # #[derive(Clone)]
/// # struct Inst(Vec<MemoryEvent<u32, u8, ()>>);
/// # impl InstrInfo for Inst {
/// #     type Variable = u8;
/// #     fn uses(&self) -> &[u8] { &[] }
/// #     fn defs(&self) -> &[u8] { &[] }
/// # }
/// # impl MemoryEventInfo for Inst {
/// #     type Location = u32;
/// #     type Fence = ();
/// #     fn memory_events(&self) -> impl Iterator<Item = MemoryEvent<u32, u8, ()>> {
/// #         self.0.iter().cloned()
/// #     }
/// # }
/// let mut cfg = Cfg::<Inst>::new();
/// cfg.block_mut(cfg.entry())
///     .push(Inst(vec![MemoryEvent::Access(MemoryAccess::write(7, [1]))]));
///
/// let mut scratch = MemorySsaScratch::new();
/// let memory: MemorySSA<u32, u8, ()> =
///     MemorySSA::compute_in(&mut scratch, &cfg, &ExactMemoryAlias);
/// assert_eq!(memory.events().len(), 1);
/// ```
#[derive(Debug)]
pub struct MemorySsaScratch<L, V, F> {
    /// The procedure's memory events, in structural order.
    pub(super) trace: MemoryTrace<L, V, F>,
    /// Every reported location, sorted and deduplicated.
    pub(super) locations: Vec<L>,
    /// The may-alias merge over those locations.
    pub(super) union_find: DisjointSet,
    /// Each merged root's assigned class, which is what numbers the classes
    /// in ascending location order.
    pub(super) class_by_root: BTreeMap<usize, MemoryClassId>,
    /// The shadow CFG's dominator buffers.
    pub(super) dominators: DominatorScratch,
    /// The shadow CFG's SSA buffers, whose variable is the alias class.
    pub(super) ssa: SsaScratch<MemoryClassId>,
}

impl<L, V, F> Default for MemorySsaScratch<L, V, F> {
    fn default() -> Self {
        Self {
            trace: MemoryTrace::empty(),
            locations: Vec::new(),
            union_find: DisjointSet::new(0),
            class_by_root: BTreeMap::new(),
            dominators: DominatorScratch::new(),
            ssa: SsaScratch::new(),
        }
    }
}

impl<L, V, F> MemorySsaScratch<L, V, F> {
    /// Scratch holding nothing yet.
    ///
    /// Every buffer is sized by the first procedure it sees, so there is no
    /// event or block count to state here.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Empty the buffers a call reads before it writes them.
    ///
    /// The trace and the union-find are refilled wholesale by the call and so
    /// are not listed; everything that is appended to or looked up in is.
    pub(super) fn reset(&mut self) {
        self.locations.clear();
        self.class_by_root.clear();
    }
}
