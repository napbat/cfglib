//! Seeded topologies with removed slots, built as a `Cfg` and as a `Graph`.

use std::collections::BTreeSet;

use cfglib::{
    BlockId, Cfg, CfgRenumbering, DenseId, EdgeId, EdgeKind, Graph, InstrInfo, NodeId, Renumbering,
    Rooted, RootedView,
};

use crate::oracle::Reference;

/// A toy instruction: at most one definition and two reads of small variables.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Op {
    pub defs: Vec<u8>,
    pub uses: Vec<u8>,
}

impl InstrInfo for Op {
    type Variable = u8;

    fn uses(&self) -> &[u8] {
        &self.uses
    }

    fn defs(&self) -> &[u8] {
        &self.defs
    }
}

/// A small deterministic generator, so a failing seed reproduces anywhere.
pub struct Lcg(u64);

impl Lcg {
    pub const fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xD1B5_4A32_D192_ED03)
    }

    fn next_word(&mut self) -> u32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        u32::try_from(self.0 >> 32).expect("a 64-bit state shifted by 32 fits in 32 bits")
    }

    pub fn below(&mut self, bound: usize) -> usize {
        usize::try_from(self.next_word()).expect("a u32 fits in usize") % bound
    }

    pub fn percent(&mut self, chance: usize) -> bool {
        self.below(100) < chance
    }
}

/// What a generated topology is allowed to look like.
#[derive(Debug, Clone, Copy)]
pub struct Shape {
    /// Slots allocated before anything is removed; at least five.
    pub slots: usize,
    /// Whether an edge may target the entry block. SSA requires it not to.
    pub entry_has_predecessors: bool,
    /// Whether every block the entry reaches can reach an exit, which is what
    /// makes post-dominance exact rather than relative to a cut.
    pub every_block_reaches_exit: bool,
}

/// The plain data of a control-flow graph with removed slots.
///
/// Node `0` is the entry. Slots are identities before any compaction, so an
/// edge may name a removed node and a removed edge keeps its position.
#[derive(Debug, Clone)]
pub struct Topology {
    pub slots: usize,
    pub edges: Vec<(usize, usize)>,
    pub blocks: Vec<Vec<Op>>,
    pub removed_nodes: BTreeSet<usize>,
    pub removed_edges: BTreeSet<usize>,
}

const VARIABLES: usize = 4;

impl Topology {
    pub fn random(seed: u64, shape: Shape) -> Self {
        assert!(shape.slots >= 5, "a topology needs room for holes");
        let mut rng = Lcg::new(seed);
        let slots = shape.slots;
        let exit = slots - 1;

        let mut removed_nodes: BTreeSet<usize> = (1..exit).filter(|_| rng.percent(25)).collect();
        if removed_nodes.is_empty() {
            removed_nodes.insert(1);
        }

        let mut edges = Vec::new();
        for source in 0..exit {
            for _ in 0..rng.below(3) {
                let mut target = rng.below(slots);
                if target == 0 && !shape.entry_has_predecessors {
                    target = 1 + rng.below(slots - 1);
                }
                edges.push((source, target));
            }
        }
        let removed_edges = (0..edges.len()).filter(|_| rng.percent(12)).collect();

        let blocks = (0..slots)
            .map(|_| (0..rng.below(4)).map(|_| random_op(&mut rng)).collect())
            .collect();

        let mut topology = Self {
            slots,
            edges,
            blocks,
            removed_nodes,
            removed_edges,
        };
        if shape.every_block_reaches_exit {
            topology.connect_stranded_blocks_to(exit);
        }
        topology
    }

    fn connect_stranded_blocks_to(&mut self, exit: usize) {
        let stranded: Vec<usize> = {
            let reference = Reference::new(self);
            let sinks = reference.sinks();
            reference
                .live()
                .filter(|&block| reference.reach(block, None).is_disjoint(&sinks))
                .collect()
        };
        self.edges
            .extend(stranded.into_iter().map(|block| (block, exit)));
    }

    /// Read a `Cfg`'s live structure back as plain data, keeping its slots.
    pub fn from_cfg(cfg: &Cfg<Op>) -> Self {
        let slots = cfg.block_bound();
        let mut removed_nodes = BTreeSet::new();
        let blocks = (0..slots)
            .map(|slot| {
                let block = BlockId::from_index(slot);
                if cfg.contains_block(block) {
                    cfg.block(block).instructions().to_vec()
                } else {
                    removed_nodes.insert(slot);
                    Vec::new()
                }
            })
            .collect();

        let mut removed_edges = BTreeSet::new();
        let edges = (0..cfg.edge_bound())
            .map(|slot| {
                let edge = EdgeId::from_index(slot);
                if cfg.contains_edge(edge) {
                    let edge = cfg.edge(edge);
                    (edge.source().index(), edge.target().index())
                } else {
                    removed_edges.insert(slot);
                    (0, 0)
                }
            })
            .collect();

        Self {
            slots,
            edges,
            blocks,
            removed_nodes,
            removed_edges,
        }
    }

    pub fn is_live(&self, slot: usize) -> bool {
        slot < self.slots && !self.removed_nodes.contains(&slot)
    }

    /// Whether an edge survives: it is not removed and neither end was.
    pub fn is_live_edge(&self, index: usize) -> bool {
        let (source, target) = self.edges[index];
        !self.removed_edges.contains(&index) && self.is_live(source) && self.is_live(target)
    }

    /// The topology as a `Cfg`: edges and blocks added first, removals after,
    /// so every identity is its slot.
    pub fn build_cfg(&self) -> Cfg<Op> {
        let mut cfg = Cfg::<Op>::new();
        let mut blocks = vec![cfg.entry()];
        blocks.extend((1..self.slots).map(|_| cfg.new_block()));
        for (slot, instructions) in self.blocks.iter().enumerate() {
            for instruction in instructions {
                cfg.block_mut(blocks[slot]).push(instruction.clone());
            }
        }
        let edges: Vec<EdgeId> = self
            .edges
            .iter()
            .map(|&(source, target)| {
                cfg.add_edge(blocks[source], blocks[target], EdgeKind::Unconditional)
            })
            .collect();
        for &edge in &self.removed_edges {
            assert!(cfg.remove_edge(edges[edge]), "edge {edge} was live");
        }
        for &block in &self.removed_nodes {
            assert!(cfg.remove_block(blocks[block]), "block {block} was live");
        }
        cfg
    }

    pub fn build_graph(&self) -> Graph<(), ()> {
        let mut graph = Graph::<(), ()>::new();
        let nodes: Vec<NodeId> = (0..self.slots).map(|_| graph.add_node(())).collect();
        let edges: Vec<_> = self
            .edges
            .iter()
            .map(|&(source, target)| graph.add_edge(nodes[source], nodes[target], ()))
            .collect();
        for &edge in &self.removed_edges {
            assert!(graph.remove_edge(edges[edge]), "edge {edge} was live");
        }
        for &node in &self.removed_nodes {
            assert!(graph.remove_node(nodes[node]), "node {node} was live");
        }
        graph
    }
}

fn random_op(rng: &mut Lcg) -> Op {
    let variable = |rng: &mut Lcg| {
        u8::try_from(rng.below(VARIABLES)).expect("a variable index below four fits in u8")
    };
    let defs = if rng.percent(50) {
        vec![variable(rng)]
    } else {
        Vec::new()
    };
    let uses = (0..rng.below(3)).map(|_| variable(rng)).collect();
    Op { defs, uses }
}

/// The identity each topology slot has in one store, `None` where removed.
pub struct Slots<N> {
    ids: Vec<Option<N>>,
}

impl<N: DenseId> Slots<N> {
    /// Before compaction a slot is its own identity.
    pub fn holed(topology: &Topology) -> Self {
        Self {
            ids: (0..topology.slots)
                .map(|slot| topology.is_live(slot).then(|| N::from_index(slot)))
                .collect(),
        }
    }

    /// After compaction a slot is whatever the `Renumbering` answers.
    pub fn compacted(topology: &Topology, renumber: impl Fn(N) -> Option<N>) -> Self {
        Self {
            ids: (0..topology.slots)
                .map(|slot| renumber(N::from_index(slot)))
                .collect(),
        }
    }

    pub fn id(&self, slot: usize) -> N {
        self.ids[slot].unwrap_or_else(|| panic!("slot {slot} has no identity: it was removed"))
    }

    pub fn slot_of(&self, id: N) -> usize {
        self.ids
            .iter()
            .position(|candidate| *candidate == Some(id))
            .unwrap_or_else(|| panic!("{id:?} is not the identity of any slot"))
    }
}

/// An analysis over any rooted view, checked against the reference.
pub trait ViewCheck {
    fn check<G: RootedView>(
        &self,
        view: &G,
        slots: &Slots<G::NodeId>,
        reference: &Reference<'_>,
        label: &str,
    );
}

/// One topology realized in both stores, with and without holes.
pub struct Realizations {
    label: String,
    pub topology: Topology,
    pub cfg_holed: Cfg<Op>,
    pub cfg_compacted: Cfg<Op>,
    pub cfg_renumbering: CfgRenumbering,
    pub graph_holed: Graph<(), ()>,
    pub graph_compacted: Graph<(), ()>,
    pub graph_renumbering: Renumbering,
}

impl Realizations {
    pub fn of_topology(topology: Topology) -> Self {
        let cfg = topology.build_cfg();
        Self::of(topology, cfg)
    }

    /// Realize whatever holes a `Cfg` has right now, for example after a
    /// transform that has not been followed by a compaction.
    pub fn of_cfg(cfg: Cfg<Op>) -> Self {
        Self::of(Topology::from_cfg(&cfg), cfg)
    }

    fn of(topology: Topology, cfg_holed: Cfg<Op>) -> Self {
        let mut cfg_compacted = cfg_holed.clone();
        let cfg_renumbering = cfg_compacted.compact();
        let graph_holed = topology.build_graph();
        let mut graph_compacted = graph_holed.clone();
        let graph_renumbering = graph_compacted.compact();
        Self {
            label: String::new(),
            topology,
            cfg_holed,
            cfg_compacted,
            cfg_renumbering,
            graph_holed,
            graph_compacted,
            graph_renumbering,
        }
    }

    /// Name this case in every failure message, for example by its seed.
    #[must_use]
    pub fn labeled(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    pub fn cfg_slots_holed(&self) -> Slots<BlockId> {
        Slots::holed(&self.topology)
    }

    pub fn cfg_slots_compacted(&self) -> Slots<BlockId> {
        Slots::compacted(&self.topology, |old| self.cfg_renumbering.node(old))
    }

    pub fn graph_slots_holed(&self) -> Slots<NodeId> {
        Slots::holed(&self.topology)
    }

    pub fn graph_slots_compacted(&self) -> Slots<NodeId> {
        Slots::compacted(&self.topology, |old| self.graph_renumbering.node(old))
    }

    /// Run `check` on the `Cfg` and on the `Graph`, holed and compacted.
    pub fn run_views(&self, check: &impl ViewCheck) {
        let reference = Reference::new(&self.topology);
        let label = |store: &str| format!("{} {store}", self.label);
        check.check(
            &self.cfg_holed,
            &self.cfg_slots_holed(),
            &reference,
            &label("cfg with holes"),
        );
        check.check(
            &self.cfg_compacted,
            &self.cfg_slots_compacted(),
            &reference,
            &label("cfg compacted"),
        );
        let holed_slots = self.graph_slots_holed();
        let holed = Rooted::new(&self.graph_holed, holed_slots.id(0));
        check.check(&holed, &holed_slots, &reference, &label("graph with holes"));
        let compacted_slots = self.graph_slots_compacted();
        let compacted = Rooted::new(&self.graph_compacted, compacted_slots.id(0));
        check.check(
            &compacted,
            &compacted_slots,
            &reference,
            &label("graph compacted"),
        );
    }

    /// Run `check` on the `Cfg`, holed and compacted.
    pub fn run_cfgs(&self, check: impl Fn(&Cfg<Op>, &Slots<BlockId>, &Reference<'_>, &str)) {
        let reference = Reference::new(&self.topology);
        let label = |store: &str| format!("{} {store}", self.label);
        check(
            &self.cfg_holed,
            &self.cfg_slots_holed(),
            &reference,
            &label("cfg with holes"),
        );
        check(
            &self.cfg_compacted,
            &self.cfg_slots_compacted(),
            &reference,
            &label("cfg compacted"),
        );
    }
}

/// Seeds every randomized test runs, so the suite stays deterministic.
pub const SEEDS: std::ops::Range<u64> = 0..48;

pub const GENERAL: Shape = Shape {
    slots: 12,
    entry_has_predecessors: true,
    every_block_reaches_exit: false,
};

pub const EXIT_COMPLETE: Shape = Shape {
    slots: 12,
    entry_has_predecessors: true,
    every_block_reaches_exit: true,
};

pub const SSA_SAFE: Shape = Shape {
    slots: 12,
    entry_has_predecessors: false,
    every_block_reaches_exit: false,
};
