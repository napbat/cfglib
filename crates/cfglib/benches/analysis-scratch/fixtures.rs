//! Procedure-shaped CFGs at the three scales a whole-codebase flow build
//! meets, plus the instruction adapter they carry.
//!
//! The shape is a chain of if/else diamonds: the smallest one is the four
//! blocks that most real callables actually are, and the larger ones are the
//! same shape repeated, so a size change moves the block count without also
//! moving the density of phis, memory events, or variables per block.

use cfglib::{Cfg, EdgeKind, InstrInfo, MemoryAccess, MemoryEvent, MemoryEventInfo};

/// One reported memory location. Real stack slots are few and repeat, and the
/// alias oracle is quadratic in distinct locations, so the fixture keeps the
/// vocabulary small on purpose.
pub(crate) type Slot = u32;

/// Ordinary data flow plus explicit memory events, which is the adapter
/// surface all four analyses read.
#[derive(Debug, Clone)]
pub(crate) struct FlowInst {
    uses: Vec<u32>,
    defs: Vec<u32>,
    events: Vec<MemoryEvent<Slot, u32, ()>>,
}

impl FlowInst {
    fn new(
        uses: impl IntoIterator<Item = u32>,
        defs: impl IntoIterator<Item = u32>,
        events: impl IntoIterator<Item = MemoryEvent<Slot, u32, ()>>,
    ) -> Self {
        Self {
            uses: uses.into_iter().collect(),
            defs: defs.into_iter().collect(),
            events: events.into_iter().collect(),
        }
    }
}

impl InstrInfo for FlowInst {
    type Variable = u32;

    fn uses(&self) -> &[u32] {
        &self.uses
    }

    fn defs(&self) -> &[u32] {
        &self.defs
    }
}

impl MemoryEventInfo for FlowInst {
    type Location = Slot;
    type Fence = ();

    fn memory_events(
        &self,
    ) -> impl Iterator<Item = MemoryEvent<Self::Location, Self::Variable, Self::Fence>> {
        self.events.iter().cloned()
    }
}

/// How many distinct slots the fixture's loads and stores name.
const SLOTS: Slot = 8;

/// How many ordinary variables one region reuses, so renaming stacks and
/// version maps hold a realistic number of keys.
const VARIABLES: u32 = 6;

/// The three scales, as the region count each is built from. A region is one
/// diamond: three blocks plus the merge's successor shared with the next.
pub(crate) const SMALL_REGIONS: usize = 1;
pub(crate) const MEDIUM_REGIONS: usize = 16;
pub(crate) const LARGE_REGIONS: usize = 666;

/// A chain of `regions` if/else diamonds, `3 * regions + 1` blocks.
///
/// Every region defines the same variables on both arms and reads them after
/// the merge, so each merge block takes real phis, and stores and loads name
/// slots that repeat across regions, so memory SSA sees phis too.
pub(crate) fn diamond_chain(regions: usize) -> Cfg<FlowInst> {
    let mut cfg = Cfg::new();
    let mut current = cfg.entry();
    for index in 0..regions {
        let then_block = cfg.new_block();
        let else_block = cfg.new_block();
        let merge = cfg.new_block();
        cfg.add_edge(current, then_block, EdgeKind::ConditionalTrue);
        cfg.add_edge(current, else_block, EdgeKind::ConditionalFalse);
        cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);
        cfg.add_edge(else_block, merge, EdgeKind::Fallthrough);

        let base = u32::try_from(index).expect("region index fits in u32") * VARIABLES;
        let slot = u32::try_from(index).expect("region index fits in u32") % SLOTS;
        fill_condition(&mut cfg, current, base, slot);
        fill_arm(&mut cfg, then_block, base, slot);
        fill_arm(&mut cfg, else_block, base, (slot + 1) % SLOTS);
        fill_merge(&mut cfg, merge, base, slot);
        current = merge;
    }
    cfg
}

/// The block that computes the branch condition: a load and two transfers.
fn fill_condition(cfg: &mut Cfg<FlowInst>, block: cfglib::BlockId, base: u32, slot: Slot) {
    let address = base % VARIABLES;
    cfg.block_mut(block)
        .push(FlowInst::new([], [address], Vec::new()));
    cfg.block_mut(block).push(FlowInst::new(
        [address],
        [base + 1],
        [MemoryEvent::Access(
            MemoryAccess::read(slot, [base + 1]).with_address_uses([address]),
        )],
    ));
    cfg.block_mut(block)
        .push(FlowInst::new([base + 1], [base + 2], Vec::new()));
}

/// One arm of a diamond: a store of the arm's own value into the slot.
fn fill_arm(cfg: &mut Cfg<FlowInst>, block: cfglib::BlockId, base: u32, slot: Slot) {
    cfg.block_mut(block)
        .push(FlowInst::new([base + 2], [base + 3], Vec::new()));
    cfg.block_mut(block).push(FlowInst::new(
        [base + 3],
        [],
        [MemoryEvent::Access(MemoryAccess::write(slot, [base + 3]))],
    ));
}

/// The merge: the block whose phis every one of the four analyses has to
/// place, followed by a read/modify/write so memory SSA has a use after its
/// own phi.
fn fill_merge(cfg: &mut Cfg<FlowInst>, block: cfglib::BlockId, base: u32, slot: Slot) {
    cfg.block_mut(block)
        .push(FlowInst::new([base + 3], [base + 4], Vec::new()));
    cfg.block_mut(block).push(FlowInst::new(
        [base + 4],
        [base + 5],
        [MemoryEvent::Access(MemoryAccess::read_modify_write(
            slot,
            [base + 4],
            [base + 5],
        ))],
    ));
}
