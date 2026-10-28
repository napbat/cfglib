//! Phase 2: the unwind edges, with one funclet instance for each context.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::String;
use alloc::vec::Vec;

use super::trie::{Node, Tries};
use super::{HopEnd, UnwindEnd, UnwindHop, UnwindRoute, check_resume, range_positions};
use crate::block::{BasicBlock, BlockId};
use crate::builder::address::{
    AddressBuildError, AddressEdgeInfo, AddressGraph, AddressInstruction,
};
use crate::cfg::Cfg;
use crate::edge::EdgeKind;
use crate::flow::{EdgeRole, Flow};

/// The blocks of one funclet: the original blocks, which one instance uses.
struct FuncletRegion {
    /// The blocks, in block order.
    blocks: Vec<BlockId>,
    /// The blocks that return to the runtime, in block order.
    returns: Vec<BlockId>,
}

/// The entry and the returns of one instance of a funclet.
struct Instance {
    entry: BlockId,
    returns: Vec<BlockId>,
}

impl<I: AddressInstruction + Clone, E: Clone> AddressGraph<I, E> {
    /// Adds the unwind edges of `routes`, exactly.
    ///
    /// Run this after the caller finished the other edits of the graph: it
    /// finds each address through [`instruction_blocks`](Self::instruction_blocks),
    /// and it reads the returns of the funclets from the edges that exist.
    /// Blocks can be split before the call, but each protected instruction,
    /// each funclet entry and each handler must lead its block.
    ///
    /// Each distinct list of paths that follows a funclet gets its own
    /// instance of the funclet. The first instance uses the original blocks.
    /// Each further instance uses new blocks that hold copies of the
    /// instructions and copies of every edge of the original blocks, with each
    /// target that lies in the funclet moved to the copy. The copies belong
    /// to no region and are not in [`instruction_blocks`](Self::instruction_blocks),
    /// which keeps the original block of each address. A copy of a block that
    /// holds a protected instruction has no unwind edge: the protected
    /// instructions of a funclet are a rare case that this call does not
    /// model.
    ///
    /// The funclet of an instance is the set of blocks that control reaches
    /// from the entry along edges that are not unwind edges or call edges,
    /// without entering the entry of another funclet that a route names. A
    /// block of it that returns, or that ends in a tail jump with no edge,
    /// is a return of the instance. A funclet that never returns has no
    /// return: its instance has no outgoing unwind edge, and that is no
    /// error.
    ///
    /// For each protected instruction that
    /// [retains an exception edge](AddressInstruction::retains_exception_edge)
    /// the call adds an [`EdgeRole::Route`] edge to the first instance of each
    /// way, and an edge from each return of an instance to the next instance
    /// or to the handler block. There is one edge for each pair of source
    /// and target block. The `route` of the edge names the first route that
    /// states it. An [`UnwindEnd::Resume`] edge goes from the protected block
    /// to itself, and [`UnwindEnd::Propagate`] adds no edge.
    ///
    /// The result lists the hops of each source block that has a way, so the
    /// consumer can also see the ways that have no edge, and the opaque calls
    /// of each way. It is sorted by source block and end.
    ///
    /// # Errors
    ///
    /// Returns an [`AddressBuildError`] when a protected range is empty or
    /// does not start and end on instruction starts, when a funclet entry or a
    /// handler is no instruction start
    /// ([`AddressBuildError::MissingRouteEntry`]), when a protected
    /// instruction, a funclet entry or a handler does not lead its block
    /// ([`AddressBuildError::RouteAddressNotBlockStart`]), and when a path
    /// has a funclet step and ends with [`UnwindEnd::Resume`]
    /// ([`AddressBuildError::ResumeAfterFunclet`]). The graph is not changed
    /// when the call fails.
    pub fn add_unwind_routes(
        &mut self,
        routes: &[UnwindRoute<I::Address>],
        edge_payload: impl FnMut(AddressEdgeInfo<'_, I::Address, I::CaseKey>) -> E,
    ) -> Result<Vec<UnwindHop>, AddressBuildError<I::Address>> {
        let protected = self.protected_blocks(routes)?;
        let tries = Tries::build(routes);
        let mut funclet_entries = BTreeMap::new();
        let mut handler_blocks = BTreeMap::new();
        for node in &tries.nodes {
            if let Some(address) = node.funclet
                && !funclet_entries.contains_key(&address)
            {
                funclet_entries.insert(address, self.leading_block(address)?);
            }
            for (_, end) in &node.ends {
                if let UnwindEnd::Handler(address) = *end
                    && !handler_blocks.contains_key(&address)
                {
                    handler_blocks.insert(address, self.leading_block(address)?);
                }
            }
        }

        let entry_blocks: BTreeSet<BlockId> = funclet_entries.values().copied().collect();
        let regions: BTreeMap<I::Address, FuncletRegion> = funclet_entries
            .iter()
            .map(|(&address, &entry)| (address, region_of(&self.cfg, entry, &entry_blocks)))
            .collect();
        let mut used = BTreeSet::new();
        let mut instances = Vec::with_capacity(tries.nodes.len());
        for node in &tries.nodes {
            instances.push(node.funclet.map(|address| {
                let region = &regions[&address];
                let entry = funclet_entries[&address];
                if used.insert(address) {
                    Instance {
                        entry,
                        returns: region.returns.clone(),
                    }
                } else {
                    clone_region(&mut self.cfg, region, entry)
                }
            }));
        }

        let first_routes = tries.first_routes();
        let mut emitter = Emitter {
            cfg: &mut self.cfg,
            edge_payload,
            added: BTreeSet::new(),
            hops: BTreeMap::new(),
            instances: &instances,
            handler_blocks: &handler_blocks,
        };
        for (route, blocks) in protected.iter().enumerate() {
            let root = &tries.nodes[tries.roots[route]];
            for &source in blocks {
                emitter.ways(source, route, root);
            }
        }
        for (id, node) in tries.nodes.iter().enumerate() {
            if let Some(instance) = &instances[id] {
                for &source in &instance.returns {
                    emitter.ways(source, first_routes[id], node);
                }
            }
        }
        Ok(emitter.hops())
    }

    /// The block of each protected instruction that keeps an exception edge,
    /// by route.
    fn protected_blocks(
        &self,
        routes: &[UnwindRoute<I::Address>],
    ) -> Result<Vec<Vec<BlockId>>, AddressBuildError<I::Address>> {
        let addresses: Vec<I::Address> = self.instruction_blocks.keys().copied().collect();
        let code_end = self
            .cfg
            .blocks()
            .flat_map(BasicBlock::instructions)
            .filter_map(I::end_address)
            .max();
        let mut protected = Vec::with_capacity(routes.len());
        for (index, route) in routes.iter().enumerate() {
            let positions = range_positions(&route.protected, code_end, &addresses)?;
            check_resume(index, route)?;
            let mut blocks = Vec::new();
            for &address in &addresses[positions] {
                let block = self.leading_block(address)?;
                let retains = self
                    .cfg
                    .block(block)
                    .instructions()
                    .first()
                    .is_some_and(AddressInstruction::retains_exception_edge);
                if retains {
                    blocks.push(block);
                }
            }
            protected.push(blocks);
        }
        Ok(protected)
    }

    /// The block that starts at `address`.
    fn leading_block(&self, address: I::Address) -> Result<BlockId, AddressBuildError<I::Address>> {
        let block = *self
            .instruction_blocks
            .get(&address)
            .ok_or(AddressBuildError::MissingRouteEntry { address })?;
        let leads = self
            .cfg
            .block(block)
            .instructions()
            .first()
            .is_some_and(|instruction| instruction.address() == address);
        if leads {
            Ok(block)
        } else {
            Err(AddressBuildError::RouteAddressNotBlockStart { address })
        }
    }
}

/// Finds the blocks of the funclet that starts at `entry`, and its returns.
fn region_of<I: AddressInstruction, E>(
    cfg: &Cfg<I, E>,
    entry: BlockId,
    funclet_entries: &BTreeSet<BlockId>,
) -> FuncletRegion {
    let mut seen = BTreeSet::from([entry]);
    let mut pending = Vec::from([entry]);
    while let Some(block) = pending.pop() {
        for edge in cfg.outgoing(block) {
            let edge = cfg.edge(edge);
            let followed = !matches!(
                edge.data().kind(),
                EdgeKind::ExceptionUnwind | EdgeKind::Call
            );
            let target = edge.target();
            if followed && !funclet_entries.contains(&target) && seen.insert(target) {
                pending.push(target);
            }
        }
    }
    let returns = seen
        .iter()
        .copied()
        .filter(|&block| returns_to_runtime(cfg, block))
        .collect();
    FuncletRegion {
        blocks: seen.into_iter().collect(),
        returns,
    }
}

/// Tests whether the last instruction of `block` returns to the runtime: a
/// return, or a jump that has no edge but unwind and call edges. That jump
/// leaves the function, so it is a tail call.
fn returns_to_runtime<I: AddressInstruction, E>(cfg: &Cfg<I, E>, block: BlockId) -> bool {
    cfg.block(block)
        .instructions()
        .last()
        .is_some_and(|instruction| match instruction.flow() {
            Flow::Return => true,
            Flow::Jump { .. } => cfg.outgoing(block).all(|edge| {
                matches!(
                    cfg.edge(edge).data().kind(),
                    EdgeKind::ExceptionUnwind | EdgeKind::Call
                )
            }),
            _ => false,
        })
}

/// Adds copies of the blocks of `region`, with copies of their edges.
fn clone_region<I: Clone, E: Clone>(
    cfg: &mut Cfg<I, E>,
    region: &FuncletRegion,
    entry: BlockId,
) -> Instance {
    let copies: BTreeMap<BlockId, BlockId> = region
        .blocks
        .iter()
        .map(|&block| (block, cfg.new_block()))
        .collect();
    let mut edges = Vec::new();
    for (&original, &copy) in &copies {
        let block = cfg.block(original);
        let instructions = block.instructions().to_vec();
        let label = block.label().map(String::from);
        cfg.block_mut(copy).instructions_mut().extend(instructions);
        if let Some(label) = label {
            cfg.block_mut(copy).set_label(label);
        }
        for id in cfg.outgoing(original) {
            let edge = cfg.edge(id);
            let data = edge.data();
            let target = edge.target();
            edges.push((
                copy,
                copies.get(&target).copied().unwrap_or(target),
                data.kind(),
                data.weight(),
                data.payload().clone(),
            ));
        }
    }
    for (source, target, kind, weight, payload) in edges {
        match weight {
            Some(weight) => {
                cfg.add_weighted_edge_with_payload(source, target, kind, weight, payload);
            }
            None => {
                cfg.add_edge_with_payload(source, target, kind, payload);
            }
        }
    }
    Instance {
        entry: copies[&entry],
        returns: region.returns.iter().map(|block| copies[block]).collect(),
    }
}

/// Adds the route edges and collects the hops.
struct Emitter<'a, I: AddressInstruction, E, F> {
    cfg: &'a mut Cfg<I, E>,
    edge_payload: F,
    added: BTreeSet<(BlockId, BlockId)>,
    hops: BTreeMap<(BlockId, HopEnd), BTreeSet<Vec<u32>>>,
    instances: &'a [Option<Instance>],
    handler_blocks: &'a BTreeMap<I::Address, BlockId>,
}

impl<I, E, F> Emitter<'_, I, E, F>
where
    I: AddressInstruction,
    F: for<'k> FnMut(AddressEdgeInfo<'k, I::Address, I::CaseKey>) -> E,
{
    /// Adds the edges and the hops from `source` for the ways of `node`.
    fn ways(&mut self, source: BlockId, route: usize, node: &Node<I::Address>) {
        for (calls, child) in &node.edges {
            let entry = self.instances[*child]
                .as_ref()
                .expect("a child node names a funclet")
                .entry;
            self.way(source, route, calls, HopEnd::Block(entry));
        }
        for (calls, end) in &node.ends {
            let end = match end {
                UnwindEnd::Handler(address) => HopEnd::Block(self.handler_blocks[address]),
                UnwindEnd::Propagate => HopEnd::Leave,
                UnwindEnd::Resume => HopEnd::Block(source),
            };
            self.way(source, route, calls, end);
        }
    }

    fn way(&mut self, source: BlockId, route: usize, calls: &[u32], end: HopEnd) {
        self.hops
            .entry((source, end))
            .or_default()
            .insert(calls.to_vec());
        let HopEnd::Block(target) = end else {
            return;
        };
        if !self.added.insert((source, target)) {
            return;
        }
        let source_address = self
            .cfg
            .block(source)
            .instructions()
            .last()
            .expect("a block holds an instruction")
            .address();
        let target_address = self
            .cfg
            .block(target)
            .instructions()
            .first()
            .expect("a block holds an instruction")
            .address();
        let payload = (self.edge_payload)(AddressEdgeInfo {
            source: source_address,
            target: target_address,
            kind: EdgeKind::ExceptionUnwind,
            role: EdgeRole::Route { route },
        });
        self.cfg
            .add_edge_with_payload(source, target, EdgeKind::ExceptionUnwind, payload);
    }

    fn hops(self) -> Vec<UnwindHop> {
        self.hops
            .into_iter()
            .map(|((source, end), calls)| UnwindHop {
                source,
                end,
                calls: calls.into_iter().collect(),
            })
            .collect()
    }
}
