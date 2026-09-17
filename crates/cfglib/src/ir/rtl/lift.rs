//! Lifting of RTL functions into typed MLIL through def-use webs.
//!
//! Storage reuse dissolves here: per-lane SSA versions unite into webs —
//! *live* φ families (a header φ nothing reads must not fuse the
//! lifetimes on either side of it), lanes co-written by one assignment
//! with each other — and every web becomes one typed MLIL variable. A
//! register reused for a float and later a counter becomes two variables
//! with honest constraints, so the consumer renders reinterpretations
//! only where a web genuinely mixes representations. Parallel transfers
//! serialize safely because reads reference prior versions; when a
//! target web is also read by a sibling assignment, the pre-state is
//! copied into a synthetic temporary first.
//!
//! Emission is dialect-driven through [`super::Emission`]; edges lift
//! *after* every instruction exists, so
//! [`Lift::lift_edge`](super::Lift::lift_edge) can bake exact emitted
//! throw-site identities into exceptional edge payloads, and the
//! returned [`Lifting`] carries the complete maps.

extern crate alloc;

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::string::ToString;
use alloc::vec::Vec;

use crate::ir::dialect::Vocabulary;
use crate::ir::mlil::{FunctionBuilder as MlilBuilder, Signature as MlilSignature};
use crate::region::{Handler, HandlerBody, HandlerKind, Region};
use crate::union_find::DisjointSet;
use crate::{BlockId, DominatorTree, PhiWebs, SsaForm, SsaValue};

use super::dialect::{Dialect, Lift, MlilBridge};
use super::emission::{EdgeContext, Emitter, LiftMaps, Lifting, Resolver};
use super::error::{Error, Result};
use super::function::Function;
use super::render::Webs;
use super::statement::{Lane, Statement};
use super::template::WebInfo;
use super::types::{Constraint, Inference, Shape};

struct WebFact<D: Dialect> {
    storage: <D as Vocabulary>::NativeVariable,
    lanes: BTreeSet<u8>,
    scalar: Inference<D::Constraint>,
    live_in: bool,
}

/// Lifts one RTL function into a typed MLIL builder.
///
/// `context` is the dialect's [`Constraint::Context`] — the class
/// hierarchy or type table constraint merging consults; `&()` for
/// self-contained domains like [`ScalarType`](super::ScalarType).
///
/// # Errors
///
/// Returns an error when SSA annotations disagree with statement
/// structure, the dialect's emission violates read alignment or
/// exceptional placement, or MLIL construction fails.
#[expect(clippy::too_many_lines, reason = "one linear pass per phase")]
pub fn lift<D: Lift>(
    function: &Function<D>,
    context: &<D::Constraint as Constraint>::Context,
) -> Result<Lifting<D>> {
    let cfg = &function.cfg;
    let dominators = DominatorTree::compute(cfg);
    let ssa: SsaForm<Lane<D>> = SsaForm::compute(cfg, &dominators);

    // Phase 1: dense ids for every SSA value, then unite φ families and
    // co-written lanes into webs. Only *live* φ families unite: SSA
    // placement is not liveness-pruned, and a dead header φ would fuse
    // the unrelated lifetimes on either side of a full overwrite.
    let mut ids: BTreeMap<(Lane<D>, usize), usize> = BTreeMap::new();
    let mut union = DisjointSet::default();
    let mut id_of = |value: &SsaValue<Lane<D>>, union: &mut DisjointSet| -> usize {
        let key = (value.variable.clone(), value.version);
        if let Some(&id) = ids.get(&key) {
            return id;
        }
        let id = union.push();
        ids.insert(key, id);
        id
    };
    for web in &PhiWebs::compute_live(&ssa).webs {
        let mut anchor: Option<usize> = None;
        for value in &web.values {
            let id = id_of(value, &mut union);
            match anchor {
                Some(anchor) => {
                    union.union_toward_min(anchor, id);
                }
                None => anchor = Some(id),
            }
        }
    }
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        let annotations = ssa.block(block_id);
        for (index, node) in block.instructions().iter().enumerate() {
            let annotation = annotations
                .instructions
                .get(index)
                .ok_or_else(|| Error::Lifting("SSA lost an instruction".into()))?;
            for value in &annotation.uses {
                id_of(value, &mut union);
            }
            if let Statement::Transfer { assignments, .. } = node.statement() {
                let mut cursor = 0usize;
                for (place, _) in assignments {
                    let defs = annotation
                        .defs
                        .get(cursor..cursor + place.lanes.len())
                        .ok_or_else(|| Error::Lifting("SSA lost a definition".into()))?;
                    let first = id_of(&defs[0], &mut union);
                    for value in &defs[1..] {
                        let other = id_of(value, &mut union);
                        union.union_toward_min(first, other);
                    }
                    cursor += place.lanes.len();
                }
            }
        }
    }

    // An unused source parameter still needs a live-in web so the
    // semantic signature remains complete.
    for place in &function.signature().parameters {
        for &lane in &place.lanes {
            let value = SsaValue::live_in((place.storage.clone(), lane));
            id_of(&value, &mut union);
        }
    }

    // The version-zero values of one storage are all the same live-in
    // value: unite them so an input register read lane-by-lane stays one
    // variable.
    let mut previous: Option<(
        &<D as crate::ir::dialect::Vocabulary>::NativeVariable,
        usize,
    )> = None;
    for ((lane, version), &id) in &ids {
        if *version != 0 {
            continue;
        }
        if let Some((storage, first)) = previous
            && *storage == lane.0
        {
            union.union_toward_min(first, id);
        } else {
            previous = Some((&lane.0, id));
        }
    }

    // Phase 2: resolve roots and collect per-web facts.
    let roots: Vec<usize> = (0..union.len()).map(|id| union.find(id)).collect();
    let mut facts: Vec<Option<WebFact<D>>> =
        core::iter::repeat_with(|| None).take(roots.len()).collect();
    for ((lane, version), &id) in &ids {
        let root = roots[id];
        let fact = facts[root].get_or_insert_with(|| WebFact {
            storage: lane.0.clone(),
            lanes: BTreeSet::new(),
            scalar: Inference::Unseen,
            live_in: false,
        });
        fact.lanes.insert(lane.1);
        fact.live_in |= *version == 0;
    }

    // Phase 3: infer each web's constraint from read wants and
    // assignment value shapes.
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        let annotations = ssa.block(block_id);
        for (index, node) in block.instructions().iter().enumerate() {
            let annotation = &annotations.instructions[index];
            let mut cursor = 0usize;
            let mut missing = false;
            node.statement().for_each_read(&mut |_, lanes, scalar| {
                for _ in lanes {
                    let Some(value) = annotation.uses.get(cursor) else {
                        missing = true;
                        return;
                    };
                    cursor += 1;
                    let key = (value.variable.clone(), value.version);
                    if let Some(&id) = ids.get(&key)
                        && let Some(fact) = facts[roots[id]].as_mut()
                    {
                        fact.scalar.observe(scalar, context);
                    }
                }
            });
            if missing {
                return Err(Error::Lifting("SSA lost a use".into()));
            }
            if let Statement::Transfer { assignments, .. } = node.statement() {
                let mut def_cursor = 0usize;
                for (place, value) in assignments {
                    let target = &annotation.defs[def_cursor];
                    def_cursor += place.lanes.len();
                    let key = (target.variable.clone(), target.version);
                    if let Some(&id) = ids.get(&key)
                        && let Some(fact) = facts[roots[id]].as_mut()
                    {
                        fact.scalar.observe(&value.shape().scalar, context);
                    }
                }
            }
        }
    }

    // Phase 4: declare one typed MLIL variable per web, in deterministic
    // first-appearance order over sorted SSA values.
    let mut builder = MlilBuilder::<<D as MlilBridge>::Mlil>::new(function.source.clone());
    let mut web_of_root = alloc::vec![None; roots.len()];
    let mut webs: Vec<WebInfo<D>> = Vec::new();
    for &id in ids.values() {
        let root = roots[id];
        if web_of_root[root].is_some() {
            continue;
        }
        let fact = facts[root]
            .as_ref()
            .ok_or_else(|| Error::Lifting("web lost its inferred facts".into()))?;
        let lanes: Vec<u8> = fact.lanes.iter().copied().collect();
        let shape = Shape {
            scalar: fact.scalar.resolve(),
            lanes: u8::try_from(lanes.len()).unwrap_or(u8::MAX),
        };
        let parameter = function
            .signature()
            .parameters
            .iter()
            .position(|place| place.storage == fact.storage && place.lanes == lanes);
        let role = parameter.map_or_else(
            || D::web_role(Some(&fact.storage)),
            |ordinal| D::parameter_role(u16::try_from(ordinal).unwrap_or(u16::MAX), &fact.storage),
        );
        let variable = builder
            .declare_variable(role, D::native_variable(&fact.storage, function.source()))
            .map_err(|error| Error::Lifting(error.to_string()))?;
        web_of_root[root] = Some(webs.len());
        webs.push(WebInfo {
            variable,
            storage: Some(fact.storage.clone()),
            lanes,
            shape,
            live_in: fact.live_in,
        });
    }
    let parameters = function
        .signature()
        .parameters
        .iter()
        .map(|place| {
            webs.iter()
                .find(|web| {
                    web.live_in
                        && web.storage.as_ref() == Some(&place.storage)
                        && web.lanes == place.lanes
                })
                .map(|web| web.variable)
                .ok_or_else(|| Error::Lifting("signature parameter lost its live-in web".into()))
        })
        .collect::<Result<Vec<_>>>()?;
    builder
        .set_signature(MlilSignature::<<D as MlilBridge>::Mlil>::new(
            parameters,
            function.signature().returns.clone(),
        ))
        .map_err(|error| Error::Lifting(error.to_string()))?;

    // Phase 5: mirror blocks. Edges wait until every instruction exists,
    // so edge lifting can reference emitted instruction identities.
    let mut block_map: Vec<BlockId> = Vec::with_capacity(cfg.block_count());
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        if block_id == cfg.entry() {
            block_map.push(builder.entry());
        } else {
            let label = block.label().unwrap_or("b").to_string();
            block_map.push(builder.new_block(label));
        }
    }

    // Phase 6: emit MLIL instructions through the dialect's hook, one
    // serialized statement at a time.
    let resolver = Resolver {
        ids: &ids,
        roots: &roots,
        web_of_root: &web_of_root,
    };
    let statement_count = function.statement_count();
    let mut emitter = Emitter {
        builder,
        web_index: webs
            .iter()
            .enumerate()
            .map(|(index, web)| {
                debug_assert_eq!(web.variable.index(), index);
                Some(index)
            })
            .collect(),
        webs,
        resolver,
        maps: LiftMaps {
            blocks: block_map.clone(),
            block_expansions: block_map
                .iter()
                .copied()
                .map(|block| alloc::vec![block])
                .collect(),
            instructions: alloc::vec![Vec::new(); statement_count],
            throw_sites: alloc::vec![None; statement_count],
            edges: Vec::new(),
        },
        tail: block_map.clone(),
        throw_blocks: alloc::vec![None; statement_count],
        current: None,
        native_defined: false,
    };
    for block_id in cfg.block_ids() {
        let block = cfg.block(block_id);
        let has_exceptional_successor = cfg
            .outgoing(block_id)
            .any(|edge| cfg.edge(edge).kind().is_exceptional());
        let annotations = ssa.block(block_id);
        for (index, node) in block.instructions().iter().enumerate() {
            let annotation = &annotations.instructions[index];
            let spans = function
                .provenance()
                .mappings_to(node.id())
                .map(|entry| entry.source.clone())
                .collect();
            emitter.statement(
                block_id.index(),
                node.id(),
                node.statement(),
                has_exceptional_successor && node.statement().may_throw(),
                spans,
                annotation,
            )?;
        }
    }

    // Phase 7: lift edges with the finished statement-to-instruction
    // mappings in hand. Normal edges leave the source chain's final
    // block; exceptional edges leave the block holding their owning
    // statement's throw site, so a continuation split keeps the throw
    // site terminal in its block.
    for edge in cfg.edges() {
        let source = cfg.block(edge.source());
        let exceptional = edge.kind().is_exceptional();
        let owner = if exceptional {
            source
                .instructions()
                .iter()
                .find(|node| node.statement().may_throw())
                .map(super::statement::StatementNode::id)
        } else {
            source
                .instructions()
                .last()
                .filter(|node| node.statement().is_terminator())
                .map(super::statement::StatementNode::id)
        };
        let context = EdgeContext {
            maps: &emitter.maps,
            owner,
        };
        let metadata = D::lift_edge(edge.payload(), &context)?;
        let mlil_source = if exceptional {
            owner
                .and_then(|statement| emitter.throw_blocks[statement.index()])
                .unwrap_or(emitter.tail[edge.source().index()])
        } else {
            emitter.tail[edge.source().index()]
        };
        let lifted = emitter
            .builder
            .add_edge(
                mlil_source,
                block_map[edge.target().index()],
                metadata,
                None,
            )
            .map_err(|error| Error::Lifting(error.to_string()))?;
        emitter.maps.edges.push(lifted);
    }

    for region in cfg.regions() {
        emitter
            .builder
            .add_region(map_region(region, &emitter.maps)?)
            .map_err(|error| Error::Lifting(error.to_string()))?;
    }

    Ok(Lifting {
        builder: emitter.builder,
        webs: Webs::new(emitter.webs),
        maps: emitter.maps,
    })
}

fn map_region(region: &Region, maps: &LiftMaps) -> Result<Region> {
    let protected_blocks = region
        .protected_blocks
        .iter()
        .flat_map(|&block| maps.blocks(block).iter().copied())
        .collect();
    let handlers = region
        .handlers
        .iter()
        .map(|handler| {
            let entry = maps
                .block(handler.entry)
                .ok_or_else(|| Error::Lifting("handler entry lost during block lifting".into()))?;
            let body = match &handler.body {
                HandlerBody::Unknown => HandlerBody::Unknown,
                HandlerBody::Known(blocks) => HandlerBody::Known(
                    blocks
                        .iter()
                        .flat_map(|&block| maps.blocks(block).iter().copied())
                        .collect(),
                ),
            };
            let kind = match handler.kind {
                HandlerKind::Filter { filter_block } => HandlerKind::Filter {
                    filter_block: maps.block(filter_block).ok_or_else(|| {
                        Error::Lifting("handler filter lost during block lifting".into())
                    })?,
                },
                other => other,
            };
            Ok(Handler { entry, body, kind })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Region {
        id: region.id,
        protected_blocks,
        handlers,
        parent: region.parent,
    })
}
