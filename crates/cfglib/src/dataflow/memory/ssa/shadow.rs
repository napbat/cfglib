//! The shadow CFG memory SSA runs ordinary SSA over, and the alias classes
//! that are its variables.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use crate::cfg::Cfg;
use crate::dataflow::InstrInfo;
use crate::memory::{MemoryEvent, MemoryTraceEntry};

use super::scratch::MemorySsaScratch;
use super::{MemoryAlias, MemoryClassId, MemoryEventSite, MemoryLocationClass};

/// One memory event as an instruction over its alias class.
///
/// A read uses the class, a write both uses and defines it, and a fence does
/// neither. The class is held once rather than in a use vector and a
/// definition vector, because those two allocations per event are most of
/// what building the shadow costs: [`Option::as_slice`] gives
/// [`InstrInfo`](crate::InstrInfo) the slices it asks for without either.
#[derive(Clone)]
pub(super) struct ShadowInstruction<L, V, F> {
    pub(super) site: MemoryEventSite,
    pub(super) event: MemoryEvent<L, V, F>,
    pub(super) class: Option<MemoryClassId>,
    /// Whether this event replaces its class's state as well as reading it.
    writes: bool,
}

impl<L, V, F> InstrInfo for ShadowInstruction<L, V, F> {
    type Variable = MemoryClassId;

    fn uses(&self) -> &[Self::Variable] {
        self.class.as_slice()
    }

    fn defs(&self) -> &[Self::Variable] {
        if self.writes {
            self.class.as_slice()
        } else {
            &[]
        }
    }
}

/// Build the shadow CFG: `cfg`'s blocks and edges, carrying one instruction
/// per memory event instead of the source instructions.
pub(super) fn build_shadow_cfg<I, E, L, V, F>(
    cfg: &Cfg<I, E>,
    entries: &[MemoryTraceEntry<L, V, F>],
    class_by_location: &BTreeMap<L, MemoryClassId>,
) -> Cfg<ShadowInstruction<L, V, F>>
where
    L: Clone + Ord,
    V: Clone,
    F: Clone,
{
    let mut shadow = Cfg::new();
    for expected_index in 1..cfg.block_bound() {
        let block = shadow.new_block();
        debug_assert_eq!(block.index(), expected_index);
    }
    shadow.set_entry(cfg.entry());
    for edge in cfg.edges() {
        shadow.add_edge(edge.source(), edge.target(), edge.kind());
    }

    for entry in entries {
        let site = MemoryEventSite::new(entry.point(), entry.event_index());
        let event = entry.event().clone();
        let class = match &event {
            MemoryEvent::Access(access) => Some(
                *class_by_location
                    .get(access.location())
                    .expect("every access location must have an alias class"),
            ),
            MemoryEvent::Fence(_) => None,
        };
        let writes = event.writes();
        shadow.block_mut(site.point.block).push(ShadowInstruction {
            site,
            event,
            class,
            writes,
        });
    }
    shadow
}

/// Merge every reported location into transitive may-alias classes.
///
/// The locations are sorted and deduplicated in one reused vector rather than
/// gathered through a set, which is the same answer — the classes are numbered
/// in ascending location order either way — without a tree node per location.
pub(super) fn build_location_classes<L, V, F, A>(
    scratch: &mut MemorySsaScratch<L, V, F>,
    alias: &A,
) -> (Vec<MemoryLocationClass<L>>, BTreeMap<L, MemoryClassId>)
where
    L: Clone + Ord,
    A: MemoryAlias<L> + ?Sized,
{
    let MemorySsaScratch {
        trace,
        locations,
        union_find,
        class_by_root,
        ..
    } = scratch;

    locations.extend(
        trace
            .entries()
            .iter()
            .filter_map(|entry| match entry.event() {
                MemoryEvent::Access(access) => Some(access.location().clone()),
                MemoryEvent::Fence(_) => None,
            }),
    );
    locations.sort_unstable();
    locations.dedup();

    union_find.reset(locations.len());
    for left in 0..locations.len() {
        for right in left + 1..locations.len() {
            if alias.may_alias(&locations[left], &locations[right])
                || alias.may_alias(&locations[right], &locations[left])
            {
                union_find.union_toward_min(left, right);
            }
        }
    }

    let mut classes: Vec<MemoryLocationClass<L>> = Vec::new();
    let mut class_by_location = BTreeMap::new();
    for (location_index, location) in locations.drain(..).enumerate() {
        let root = union_find.find(location_index);
        let id = *class_by_root.entry(root).or_insert_with(|| {
            let id = MemoryClassId(classes.len());
            classes.push(MemoryLocationClass {
                id,
                locations: Vec::new(),
            });
            id
        });
        classes[id.index()].locations.push(location.clone());
        class_by_location.insert(location, id);
    }
    (classes, class_by_location)
}
