//! Checked specialization of proved conditional control flow.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::vec;
use alloc::vec::Vec;

use crate::{BlockId, EdgeId, EdgeKind, FlowControl, FlowEffect, HandlerKind, ProgramPoint};

use super::{
    Dialect, EntityId, Error, Function, Instruction, InstructionId, ProvenanceMap, Result,
    VerifyDialect,
};

/// Exact current-graph correspondences after conditional specialization.
///
/// Blocks retain their identities, including emptied unreachable blocks.
/// Surviving edges retain their identities. Instructions compact in graph
/// order, as in canonical dead-code elimination. Source coordinates embedded
/// in operations are never interpreted as current graph coordinates.
#[derive(Debug, Clone, Default)]
pub struct ControlFlowMap {
    instructions: Vec<Option<InstructionId>>,
    points: BTreeMap<ProgramPoint, ProgramPoint>,
    edges: BTreeSet<EdgeId>,
}

impl ControlFlowMap {
    /// Returns the current identity of a surviving original instruction.
    #[must_use]
    pub fn instruction(&self, original: InstructionId) -> Option<InstructionId> {
        self.instructions.get(original.index()).copied().flatten()
    }

    /// Returns the current point corresponding to a surviving original point.
    #[must_use]
    pub fn point(&self, original: ProgramPoint) -> Option<ProgramPoint> {
        self.points.get(&original).copied()
    }

    /// Returns an edge's unchanged identity, or `None` if it was retired.
    #[must_use]
    pub fn edge(&self, original: EdgeId) -> Option<EdgeId> {
        self.edges.contains(&original).then_some(original)
    }
}

/// A checked specialized function and its current-graph correspondences.
#[derive(Debug, Clone)]
pub struct ControlFlowReduction<D: Dialect> {
    /// The rebuilt and verified function.
    pub function: Function<D>,
    /// Original-to-current graph coordinates, not original source coordinates.
    pub mapping: ControlFlowMap,
    /// Blocks emptied because no entry or exception/cleanup root reaches them.
    pub unreachable: BTreeSet<BlockId>,
    /// Number of conditional transfers specialized.
    pub specialized: usize,
    /// Number of instructions removed, including decided terminators.
    pub removed: usize,
}

impl<D: VerifyDialect> Function<D> {
    /// Specializes only conditional transfers that `decide` proves constant.
    ///
    /// A proof must also establish that evaluating the condition is total and
    /// effect-free; it must not merely predict its value after normal completion.
    /// Selected terminators must declare no effects, definitions or throwing
    /// behavior. Earlier evaluations and all exceptional outgoing edges stay.
    /// Unknown transfers remain unchanged.
    ///
    /// `edge` receives every surviving payload, the exact current-coordinate
    /// map, and whether this is a selected conditional edge. It must explicitly
    /// remap payload-owned graph identities and convert selected edges into
    /// unconditional/fallthrough metadata, or return an error. No payload is
    /// guessed or discarded. Other edges retain their structural classification.
    ///
    /// All region-referenced blocks (including known bodies and protected sets),
    /// handler/filter roots, and cleanup resume/continuation blocks are roots.
    /// Regions, cleanups, block identities/labels, variables and signature stay
    /// exact. Unreachable non-metadata blocks are emptied and lose incident
    /// edges, preventing dead predecessors from entering SSA without changing
    /// caller block-keyed facts. Dropped instructions/edges lose their provenance;
    /// all surviving provenance remaps explicitly. Operations, including original
    /// source-point seals, are retained verbatim.
    /// Returns `None` without cloning when no terminator is decided.
    ///
    /// # Errors
    /// Rejects invalid input, unsupported branch topology, effects or throwing
    /// ownership on a decided terminator, unsupported payload remapping, and
    /// any invalid completed function. The source function is never modified.
    pub fn reduce_constant_control_flow(
        &self,
        mut decide: impl FnMut(&Instruction<D>) -> Option<bool>,
        mut edge: impl FnMut(&D::Edge, &ControlFlowMap, bool) -> Result<D::Edge>,
    ) -> Result<Option<ControlFlowReduction<D>>> {
        let mut selected = BTreeSet::new();
        let mut dropped_edges = BTreeSet::new();
        let mut terminators = BTreeSet::new();
        for block in self.cfg.block_ids() {
            let Some(last) = self.cfg.block(block).instructions().last() else {
                continue;
            };
            let Some(taken) = decide(last) else {
                continue;
            };
            if last.flow_effect() != FlowEffect::ConditionalJump
                || !last.effects().is_empty()
                || !last.defs().is_empty()
                || last.may_throw()
            {
                return Err(Error::InvalidConstruction(
                    "constant branch has effects, results or throwing ownership".into(),
                ));
            }
            let mut true_edge = None;
            let mut false_edge = None;
            for id in self.cfg.outgoing(block) {
                let slot = match self.cfg.edge(id).kind() {
                    EdgeKind::ConditionalTrue => &mut true_edge,
                    EdgeKind::ConditionalFalse => &mut false_edge,
                    kind if kind.is_exceptional() => continue,
                    _ => {
                        return Err(Error::InvalidConstruction(
                            "constant branch has an unsupported ordinary edge".into(),
                        ));
                    }
                };
                if slot.replace(id).is_some() {
                    return Err(Error::InvalidConstruction(
                        "constant branch has duplicate outcome edges".into(),
                    ));
                }
            }
            let (Some(true_edge), Some(false_edge)) = (true_edge, false_edge) else {
                return Err(Error::InvalidConstruction(
                    "constant branch requires exactly one true and one false ordinary edge".into(),
                ));
            };
            let (kept, dropped) = if taken {
                (true_edge, false_edge)
            } else {
                (false_edge, true_edge)
            };
            selected.insert(kept);
            dropped_edges.insert(dropped);
            terminators.insert(last.id());
        }
        if selected.is_empty() {
            return Ok(None);
        }
        let report = self.verify();
        if !report.is_ok() {
            return Err(report.into());
        }
        let mut function = self.clone();
        for &id in &dropped_edges {
            function.cfg.remove_edge(id);
        }
        let mut reachable = BTreeSet::new();
        let mut walking = vec![function.cfg.entry()];
        for region in function.cfg.regions() {
            walking.extend(region.protected_blocks.iter().copied());
            for handler in &region.handlers {
                walking.push(handler.entry);
                if let Some(body) = handler.body.blocks() {
                    walking.extend(body.iter().copied());
                }
                if let HandlerKind::Filter { filter_block } = handler.kind {
                    walking.push(filter_block);
                }
            }
        }
        for cleanup in function.cfg.cleanups() {
            walking.extend(cleanup.resume_from);
            walking.extend(cleanup.continuations.iter().map(|route| route.resume));
        }
        while let Some(block) = walking.pop() {
            if reachable.insert(block) {
                walking.extend(
                    function
                        .cfg
                        .outgoing(block)
                        .map(|id| function.cfg.edge(id).target()),
                );
            }
        }
        let unreachable: BTreeSet<_> = function
            .cfg
            .block_ids()
            .filter(|block| !reachable.contains(block))
            .collect();
        let dead_edges: Vec<_> = function
            .cfg
            .edges()
            .filter(|edge| {
                unreachable.contains(&edge.source()) || unreachable.contains(&edge.target())
            })
            .map(|edge| edge.id())
            .collect();
        for id in dead_edges {
            function.cfg.remove_edge(id);
        }
        let mut mapping = ControlFlowMap {
            instructions: vec![None; self.instruction_count()],
            points: BTreeMap::new(),
            edges: function.cfg.edge_ids().collect(),
        };
        let mut next = 0usize;
        let blocks: Vec<_> = function.cfg.block_ids().collect();
        for block in blocks {
            let instructions = function.cfg.block_mut(block).instructions_mut();
            instructions.retain(|instruction| {
                !unreachable.contains(&block) && !terminators.contains(&instruction.id())
            });
            for (inst_idx, instruction) in instructions.iter_mut().enumerate() {
                let original = instruction.id();
                let id = InstructionId::from_raw(u32::try_from(next).map_err(|_| {
                    Error::InvalidConstruction("instruction identity exceeds u32::MAX".into())
                })?);
                next += 1;
                mapping.instructions[original.index()] = Some(id);
                if let Some(point) = self.instruction_point(original) {
                    mapping
                        .points
                        .insert(point, ProgramPoint { block, inst_idx });
                }
                instruction.set_id(id);
            }
        }
        let edges: Vec<_> = function.cfg.edge_ids().collect();
        for id in edges {
            let payload = edge(
                function.cfg.edge(id).payload(),
                &mapping,
                selected.contains(&id),
            )?;
            let kind = D::edge_kind(&payload);
            if (selected.contains(&id)
                && !matches!(kind, EdgeKind::Fallthrough | EdgeKind::Unconditional))
                || (!selected.contains(&id) && kind != function.cfg.edge(id).kind())
                || D::is_entry_edge(&payload) != D::is_entry_edge(function.cfg.edge(id).payload())
            {
                return Err(Error::InvalidConstruction(
                    "constant control rewrite changed unsupported edge metadata".into(),
                ));
            }
            let slot = function.cfg.edge_mut(id);
            slot.set_kind(kind);
            *slot.payload_mut() = payload;
        }
        function.instruction_points = vec![None; next];
        function.reindex_instructions();
        let mut provenance = ProvenanceMap::new(self.source().clone());
        for entry in self.provenance.entries() {
            let entity = match entry.entity {
                EntityId::Instruction(original) => match mapping.instruction(original) {
                    Some(id) => EntityId::Instruction(id),
                    None => continue,
                },
                EntityId::Edge(original) if mapping.edge(original).is_none() => continue,
                other => other,
            };
            provenance.insert(entry.source.clone(), entity)?;
        }
        function.provenance = provenance;
        let report = function.verify();
        if !report.is_ok() {
            return Err(report.into());
        }
        Ok(Some(ControlFlowReduction {
            function,
            mapping,
            unreachable,
            specialized: selected.len(),
            removed: self.instruction_count() - next,
        }))
    }
}
