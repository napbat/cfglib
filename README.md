# cfglib

Generic, `no_std` graph and dataflow framework for code intelligence, program analysis, decompilation, and compiler infrastructure.

`cfglib` has purpose-built storage for arbitrary directed relations, language-parametric scope graphs, file-incremental stack graphs, and `Cfg<I, E = ()>` graphs. A `Cfg` has basic blocks, typed control-flow edges, caller-owned edge metadata, and exception regions.

`cfglib::ir` holds IR traits and the generic algorithms over them. `ir::rtl` is the register-transfer contract. `ir::mlil` is the read-only view contract of a flat function in static single assignment form with block parameters. `ir::hlil` is the structured contract: a read-only view of statement and expression trees, a builder trait that the consumer implements, and the lift, lower, and recovery algorithms that drive it. The consumer owns every IR storage, builder implementation, and editing pass. The consumer also owns the value analyses over an MLIL function. The generic algorithms read the IR through views and return facts. An algorithm that creates IR drives a builder trait that the consumer implements. No algorithm changes the function that it reads.

Small read-only node and edge view contracts let consumer-owned stores and zero-copy filtered views reuse the algorithms. `CfgView` extends them for the instruction-level analyses, so a consumer block store runs the same dataflow as `Cfg`. Every instruction-adjacent axis is consumer-typed. Dataflow variables, constants, expression operators, side-effect vocabularies, branch targets, call targets, and edge payloads all come from the adapter. x86 registers and flags, shader register components, bytecode locals, compiler IR values, and source-language symbols do not need a library-owned numbering scheme. String literals and symbol ids flow through the analyses as naturally as machine words and addresses.

On top of that, cfglib has a compiler middle-end toolkit: dominator trees, renamed SSA construction, location-aware memory SSA, dataflow analyses, value numbering, explicit alias sets, loop transforms, dead-code elimination, partial redundancy elimination, and graph coloring.

Named transformations compose through `PassPipeline<T, E>`. A pipeline keeps
the caller's declared order, supports closure-backed or stateful trait passes,
reports which executions changed the target, and attributes the first failure
while retaining the completed-prefix report. Pass selection and canonical
schedules remain consumer policy, so one framework can drive a CFG or any
consumer-owned compilation context without coupling their dialects.

Everything is `no_std + alloc`. The core graph store keeps adjacency in compressed arrays and removes entities with a liveness bit, so identities stay stable until an explicit `compact()`.

## Quick start

### Direct construction (the primary front door)

No trait is required to build, verify, analyze, or render a CFG. Source
frontends translate their syntax trees straight into blocks:

```rust
use cfglib::{Cfg, DominatorTree, EdgeKind, ReachingDefs, verify};

let mut cfg = Cfg::<Stmt>::new();
let then_block = cfg.new_block();
let merge = cfg.new_block();
cfg.block_mut(cfg.entry()).push(stmt_if);
cfg.block_mut(then_block).push(stmt_assign);
cfg.add_edge(cfg.entry(), then_block, EdgeKind::ConditionalTrue);
cfg.add_edge(cfg.entry(), merge, EdgeKind::ConditionalFalse);
cfg.add_edge(then_block, merge, EdgeKind::Fallthrough);

assert!(verify(&cfg).is_ok());
let dominators = DominatorTree::compute(&cfg);
let reaching = ReachingDefs::compute(&cfg); // needs InstrInfo on Stmt
```

### Builder construction (structured instruction streams)

Frontends with a flat, structured stream (shader bytecode, structured ISAs)
implement `FlowControl` and use `CfgBuilder`; explicit gotos are wired
afterwards through the opt-in `JumpTargets` trait:

```rust
use cfglib::{CfgBuilder, FlowControl, FlowEffect, resolve_jump_edges};

impl FlowControl for Inst {
    fn flow_effect(&self) -> FlowEffect {
        match self.opcode {
            Op::If    => FlowEffect::ConditionalOpen,
            Op::Else  => FlowEffect::ConditionalAlternate,
            Op::EndIf => FlowEffect::ConditionalClose,
            Op::Loop  => FlowEffect::LoopOpen,
            Op::End   => FlowEffect::LoopClose,
            Op::Ret   => FlowEffect::Return,
            _         => FlowEffect::Fallthrough,
        }
    }
}

let mut cfg = CfgBuilder::build(instructions).unwrap();
let resolution = resolve_jump_edges(&mut cfg); // wires goto/label edges
```

## Feature overview

### Generic graph core

`Graph<N, E>` is the single owned storage type for consumer-defined node and edge payloads, stable identities, and forward/reverse adjacency. `GraphView` lets existing graph stores use node algorithms without first migrating their storage; `EdgeView` additionally exposes stable edge identity, view-oriented endpoints, payloads, and ordered adjacency. `FilteredEdges` borrows either representation through an edge predicate without cloning or renumbering anything—for example, the same CFG can be viewed as normal-only flow or full normal-plus-exception flow. `RootedView` adds a distinguished entry node for algorithms that need one (dominance, reachability metrics, loop and interval analysis), and the `Rooted` adapter roots any plain view at a chosen node. This layer is suitable for symbol/reference graphs, value-flow graphs, call graphs, type relations, import graphs, grammar dependencies, and analysis-derived relations; it has no instruction or binary-analysis concepts.

```rust
use cfglib::{DominatorTree, Graph, Rooted, TraversalDirection, shortest_path};

let mut graph = Graph::new();
let source = graph.add_node("definition");
let target = graph.add_node("call result");
let edge = graph.add_edge(source, target, ("return", "src/lib.rs", 42));

assert_eq!(graph.edge(edge).payload().0, "return");
assert_eq!(
    shortest_path(&graph, source, target, TraversalDirection::Outgoing),
    Some(vec![source, target]),
);
let dominators = DominatorTree::compute(&Rooted::new(&graph, source));
```

`Graph` keeps a compressed sparse-row base beside an appendable delta in one type: adding a node or an edge is constant time with no per-node allocation, removal clears a bit in a liveness bitset, `out_degree`/`in_degree` answer from the compressed run's two offsets plus a walk of whatever was appended since, redirecting one endpoint keeps the edge's identity, and `compact()` folds the delta back into the base and hands back a `Renumbering` describing every identity it moved. That is what makes whole-codebase graphs affordable — a million nodes and four million edges build in roughly half the time an arena of per-node adjacency containers takes, without the inline slack and the per-node allocation that arena spends. The module documentation states the design and its costs; `benches/graph-store.rs` measures it.

`Fanout<K, V>`, `DenseFanout<V, K>`, and `SortedMap<K, V>` are that compressed base without the edges. They are built-once tables — from pairs or from grouped values, read-only afterwards — holding every key's run of values end to end in one array, keyed by a sparse ordered key, by a dense `0..bound` index (any `DenseId`, so a tagged `Id<T>` keys one directly), or, in the single-value case, by a sparse key to one value. A reverse index from a symbol to its reference sites, a scope to the rows declared in it, or a file to its flow sites is the same shape as adjacency and should cost the same: three allocations for the whole table rather than one per key, `get` returning a slice, `iter()` walking keys ascending, and `heap_bytes()` reporting exactly what the columns hold.

Identities are tagged — `Id<NodeTag>`, `Id<EdgeTag>`, `BlockId = Id<BlockTag>`, `ScopeId`, `StackNodeId` — so a block identity and a symbol identity stay distinct types without a newtype and a pair of conversion shims for each. One trait, `DenseId`, covers them all, and a view reports two numbers rather than one: `node_bound()` is the exclusive upper bound that sizes a side table, and `node_ids()` yields the live nodes. A removed node is not a node of the view, so an analysis runs over a mutated store without compacting it first and without seeing a phantom singleton.

### Scope graphs

`graph::scope::ScopeGraph<S, L, R, D, Q>` separates language facts from
language policy. It stores consumer-defined scope metadata, labeled
scope-to-scope edges, relation-tagged declarations, and references. A `ScopeGraphQuery`
implementation supplies the path-language state machine, declaration matching,
label and data orders, optional indexed candidate visitation and decisive-match
pruning, and—when needed—a full path/data shadowing order. This keeps lexical parents, imports,
inheritance, namespaces, ordered declarations, accessibility, redirects, and
opacity in the frontend's vocabulary.

`ScopeResolution::compute` resolves a stored reference.
`ScopeResolution::compute_from` runs the same algorithm for an ephemeral query,
which supports completion and speculative refactoring without modifying the
graph. `ScopeResolutionIndex` resolves all stored references and provides the
reverse definition-to-reference relation used by rename and find-references.
Paths are cycle-free, deterministic, and optionally bounded; truncated results
report which bound was reached. Scope storage supports `serde` when the feature
is enabled.

### Stack graphs

`graph::stack::StackGraph<F, S, N, E>` implements the standard root,
exported/internal scope, symbol push/pop, scoped-symbol push/pop, drop-scopes,
and jump-to-scope nodes. Symbols, file records, node metadata, and edge metadata
are consumer types.
Concrete paths maintain symbol and scope stacks transactionally, scoped symbols
can pause one lookup while another resolves, and edge precedence removes
shadowed complete paths while retaining genuine ambiguity.

Every ordinary node belongs to one file partition. Direct cross-file edges are
rejected; independently constructed files compose through the root or exported
scope stitching endpoints. `StackPartialPathSet` extracts reusable structural
routes per file, `StackPartialPathDatabase` replaces one file at a time while
leaving other identities stable, and `StackGraph::clear_file` retires one
changed partition with node/edge tombstones before it is rebuilt. Then
`StackResolution::compute_from_partials` replays compatible summaries with the
same semantics as direct search. Graphs and partial-path databases are
serializable with `serde`, so unchanged file summaries can be persisted and
reused. Both direct and stitched indexes expose reverse
definition-to-reference bindings.

### Control-flow graph (`Cfg<I>`)

| Feature | Description |
|---|---|
| Generic `Cfg<I, E = ()>` | Parameterised over any instruction and caller-owned edge payload; unit payload preserves the original API |
| `no_std` + `alloc` | Runs in embedded, kernel, and WASM environments |
| `CfgBuilder` | Builds a CFG from a flat structured instruction stream (`if/else/endif`, `loop/endloop`, `switch/case/endswitch`, `break`, `continue`) |
| Goto wiring | `resolve_jump_edges` + `JumpTargets` (consumer-typed targets: labels, addresses, syntax nodes) |
| Compressed adjacency | `Cfg` stores its blocks and edges in `Graph`: a compressed base plus an appendable delta, with no per-block adjacency allocation |
| Edge removal | `remove_edge()` clears the liveness bit of the edge; existing `EdgeId`s stay stable until `compact()` renumbers the survivors |
| Edge metadata | Stable `EdgeId`, endpoints, `EdgeKind`, optional weight, and a caller-owned payload; `ExceptionFlow<M>` standardizes search/unwind phase and execute/search/continue disposition while retaining platform metadata |
| Regions | Try/catch/finally regions with explicit `HandlerBody::Known`/`Unknown` completeness and `HandlerKind` (Catch, CatchAll, Finally, Fault, Filter); `install_clr_region` and `install_seh_region` import normalized platform clauses |
| Cleanup continuations | `add_continuation` / `set_cleanup_resume` → `Cleanup` (`Continuation` + `CompletionReason`): which route out of a single shared `finally` block a resume edge belongs to |
| Handler metadata | `HandlerMetadata<M>` side table (`HandlerFilters<F>` / `HandlerTypes<T>`) — consumer-typed filter predicates and CLR caught-type tokens keyed by `HandlerRef`, with no type parameter on `Cfg` |
| Native handler state | `SehRegistrationChain<F, H>` models per-thread x86 frame registrations; `VehModel<H>` separately models ordered process-wide vectored exception/continue handlers |
| Subgraph extraction | `subgraph()` or `subgraph_mapped()` with dense O(1) block-id remapping and payload cloning |
| Block splitting | `split_block()`, mapped payload-aware variants, and validated multi-point splitting with automatic stable edge transfer |
| `serde` feature | Optional serialization support |
| Leader-based construction | `build_address_cfg` with `AddressCfgOptions` / `CallPolicy`, the `AddressInstruction` / `AddressSpace` traits, `Flow`, and `AddressHandler`. Machine/bytecode streams: caller-selected function entry; calls kept as edges or flattened into reported call sites; leaders at targets, after terminators, at exceptional instructions, and at exception boundaries; known successors retained beside explicit unresolved transfers; typed normal, instruction-exceptional, and unwind edges through a consumer payload hook, every kind chosen by `EdgeRole::kind`; nested regions registered enclosing-first with parents wired |

### IR trait contracts (`ir`)

`cfglib::ir` holds IR traits and the generic algorithms over them. It has no IR storage, no builder implementation, and no editing code. The consumer owns the dense arenas, builds its own records, and applies every change. The generic algorithms read the IR through views and return facts.

A fact is a side table about one state of a function. No algorithm changes the function that it reads. After an edit, the consumer computes the facts again. A root liveness table is a fact, and cfglib stores no liveness inside a function. An algorithm that creates a function does so only through a consumer-implemented builder trait, and the consumer owns the storage behind it.

| Level | Module | The consumer implements | cfglib provides |
|---|---|---|---|
| Register transfer (RTL) | `ir::rtl` | `Dialect`, `InstructionBuilder`, `RecordView`, `FunctionView`, `Fold` | `RootTable`, `BlockSummaries`, `RootLiveness`, `ParamPlacement`, `SelectionResolution` |
| Medium level (MLIL) | `ir::mlil` | `Dialect`, `FunctionView` | Nothing. Value analyses over the view belong to the consumer. |
| Structured (HLIL) | `ir::hlil` | `Dialect`, `FunctionView`, `FunctionBuilder`, `FlatBuilder`, `FlatFunction`, `LiftInstruction`, `LiftEdge`, `LiftDialect`, `RecoverDialect`, `VerifyDialect`, `Pseudocode` | `lift`, `lower`, `recover_structure`, `VariablePlacements`, `verify`, `pseudocode` |

Every dialect type is plain data that the consumer defines. The traits contain no language, runtime, calling convention, opcode, or source-location type.

cfglib no longer holds IR storage. The removal covers the RTL, MLIL, and HLIL function types and builder implementations, tail duplication, the shared vocabulary trait, the provenance map, and the value analyses over MLIL (use index, liveness, constant propagation, and dead-instruction marking). A consumer that needs one of them keeps it in its own crate. Lift and lower between a variable-based flat IR and HLIL are generic algorithms over consumer-implemented builder traits.

#### Register-transfer contract (`ir::rtl`)

One machine instruction is one atomic state update:

- Every read sees the single state before the instruction.
- Pure computation, loads, stores, and intrinsics follow in a stated order.
- Writes collect into one delta. The delta commits at the instruction boundary.
- One control boundary ends the instruction.

A fault is a typed outcome that an effect or a raise declares. It is not a control-flow successor. A raised outcome commits nothing of the delta. A raise can publish state of its own, its bindings: some architectures set state before they raise, such as the status flags of a floating-point exception that the program does not mask. An instruction has no control flow of its own. A repeated instruction branches to its own address, and a choice inside one instruction is a select or a guarded write.

Every RTL value is a signless bit vector. Signedness, a floating format, and a lane structure are properties of an operator, not of a value.

| Item | Description |
|---|---|
| `Bits`, `BitSlice` | `Bits` is the one value type: a width in `1..=Bits::MAX_WIDTH` (8192). `Bits::BOOL` is the one-bit type. A `BitSlice` is a bit range inside one root. |
| `Dialect` | The vocabulary of one instruction set, normally a unit marker type. The associated types are `Root`, `Operator`, `EffectOp`, `WritePolicy`, `Outcome`, and `NondetDomain`. `Dialect::root_width` gives the width of a root. Roots never alias: a register view is a `BitSlice` of one root. `Place<D>` names such a range. |
| `InstructionBuilder<D>` | The write-only protocol through which instruction semantics emit one record into consumer storage. The producer calls `read`, `constant`, `apply`, `nondet`, `effect`, `raise`, and `write` in statement order, then `finish` with a `Control`. `finish` seals the record. The protocol cannot read a statement back or change it. |
| `RecordView<D>` | One sealed record: its `statements`, its `control`, and the `width` of each value. A sealed record is read-only. |
| `FunctionView<D>` | A `RootedView` over the decoded machine control flow. The root is the entry block. It adds `inst_bound`, `value_bound`, the `instructions` of a block, and the `record` of an instruction. |
| `Fold` | Constant evaluation of the operators of a dialect. `Fold::fold` writes the results of an operator over `ConstBits` operands to a buffer. It returns `false` when the operator has no constant evaluation for the operands. |

The builder takes these plain-data types, and a record yields them. Every slice inside them is flat, so a consumer keeps them in dense pools:

- `Selection` says where a read takes its bits: `Direct` (one `Place`), `FiniteIndexed` (the candidate that an index value picks), or `SparseIndexed` (an element of a sparse bank, such as a model-specific register).
- `WriteTarget` says what a write changes: `Root`, `FiniteIndexed`, or `SparseIndexed`. A `Root` write carries a complete coverage plan: ascending `SpanAction` values that start at bit zero, end at the root width, and leave no gap. Each span has an `Action`: `PreservePreState`, `TakeInput`, or `Nondeterministic`.
- `Raise` names an outcome, an optional one-bit guard, a payload, and its bindings. A raised outcome publishes no state of the delta. A binding pairs a root with the new value of the whole root. The value comes from the state before the instruction, like every other value. A binding is not a write: on the normal path the root keeps its value.
- `Control` ends the instruction: `Advance`, `Jump`, `Branch`, `Call`, `Return`, `Dispatch`, or `Terminating`. A `Dispatch` names a scrutinee. The case values belong to the decoded control flow, not to the record.
- `Statement` is one statement of a sealed record: `Read`, `Constant`, `Apply`, `Nondet`, `Effect`, `Raise`, or `Write`. A `ValueRange` is a run of consecutive results.

`InstructionBuilder` states a contract. The consumer checks it when `finish` seals the record, and reports a violation through `InstructionBuilder::Error`:

- Every operand is a handle that the builder returned earlier.
- A `WriteTarget::Root` plan covers its whole root. Two writes of one instruction never cover the same bit.
- A guard or condition has type `Bits::BOOL`.
- An unconditional `Raise` is followed by `Control::Terminating`, and no statement follows it. The record holds no write: the delta of such a record never commits.
- The bindings of one `Raise` name each root once. Each bound value has the width of its root.

#### RTL facts

Every RTL fact is a side table. The algorithms read the function through `FunctionView` and never change it. The facts build on each other in the order of this table. A bit set in a fact holds dense indices of one `RootTable`, so the facts of one function share the table of that function.

| Fact | Entry point | Description |
|---|---|---|
| Root table | `RootTable::collect` | The roots that the function names, in ascending order. A raise binding names its root. The position of a root is its dense index. |
| Block summaries | `BlockSummaries::compute`; `BlockSummaries::block` → `BlockSummary` | Per block: `reads_before_write`, `writes`, and `kills`. A root is a whole unit here. A partial write reads its root and writes it. An indexed or sparse write is a may-write and never kills. A raise binding is not a write, and it does not read its root. |
| Root liveness | `RootLiveness::compute`, `RootLiveness::compute_seeded` with `ExitSeed`; `live_in`, `live_out` | Whole-root liveness. `compute` makes every named root live after each exit block. `compute_seeded` calls a closure for each exit block, and the closure states the live roots through `ExitSeed::insert` or `ExitSeed::insert_all`. |
| Parameter placement | `ParamPlacement::compute`; `roots_at`, `param_count` | The roots that need a block parameter at each block: the pruned iterated dominance frontier of the blocks that write the root. The call takes the `DominatorTree`, the `DominanceFrontiers`, the root table, the summaries, and the liveness. |
| Selection resolution | `SelectionResolution::compute` with `RootConstant` entry values; `candidate`, `resolved` → `ResolvedSelection` | The indexed reads and writes whose index is a constant. The analysis follows root constants across blocks and evaluates the index with `Fold`. After a call, every root is unknown. |

```rust
use cfglib::ir::rtl::{BlockSummaries, ParamPlacement, RootLiveness, RootTable};
use cfglib::{DominanceFrontiers, DominatorTree};

// `function` implements `ir::rtl::FunctionView<D>`.
let roots = RootTable::collect(&function);
let summaries = BlockSummaries::compute(&function, &roots);
let liveness = RootLiveness::compute(&function, &roots, &summaries);
let dominators = DominatorTree::compute(&function);
let frontiers = DominanceFrontiers::compute(&function, &dominators);
let placement = ParamPlacement::compute(
    &function, &dominators, &frontiers, &roots, &summaries, &liveness,
);
let needed_at_join = placement.roots_at(join);
```

#### Medium-level contract (`ir::mlil`)

An MLIL function is flat and in static single assignment form:

- Every instruction defines zero or more values and reads values by handle.
- There is no expression tree and no mutable variable.
- A block receives its join values as ordered parameters. The parameters of the entry block are the parameters of the function.
- The terminator of a block owns its successor records. A record holds the target block, the values that pass to the parameters of the target, and the edge metadata.
- The function never stores predecessors. They come from the successor records.

| Item | Description |
|---|---|
| `Dialect` | The vocabulary of one medium-level dialect. The associated types are `Op` (an operation with its immediate attributes), `Type` (the type of a value), and `Edge` (the metadata of a successor record, such as the case value of a switch arm). |
| `FunctionView<D>` | A `RootedView` over the blocks. The methods `params`, `insts`, `operands`, and `results` return slices, so the consumer keeps them in dense pools. The view also gives `op`, `successors`, `block_of`, `def`, `ty`, `inst_bound`, and `value_bound`. |
| `ValueDef`, `Successor` | `ValueDef` says where a value is defined: `Result` of an instruction or `Param` of a block. `Successor` is one successor record: `target`, `args`, and `edge`. |

A `FunctionView` follows these rules:

- Each live block ends with exactly one terminator, which is the last instruction of `insts`. Only a terminator has successors.
- Every value has exactly one definition, which `def` reports.
- The arguments of each successor record match the parameters of the target block in count and type.
- The adjacency of the `RootedView` agrees with `successors`.
- An identity below a bound that the view does not yield is a removed entity.

`ir::mlil` has no analysis of its own. A consumer that needs use lists, liveness, constant propagation, or dead-instruction marking builds them over `FunctionView`. The dialect of the consumer owns the semantics that those analyses need, such as effects and constant folding.

### Graph algorithms

| Algorithm | Function / Type | Description |
|---|---|---|
| DFS / BFS | `depth_first_preorder`, `depth_first_postorder`, `breadth_first`, and matching `Cfg` methods | Direction-selectable traversals over `GraphView` |
| Edge-aware traversal | `breadth_first_edges[_with]`, `depth_first_edges[_with]`, `shortest_path_edges`, `EdgeStep` | Every distinct edge once, with identity and endpoints, over any edge view; parallel edges stay distinct |
| Configurable search | `search` + `SearchConfig` (order, visited policy, direction, depth bound) | First-match, pruning (`Visit::Skip`), early exit (`ControlFlow::Break`), and backtracking as configuration; `VisitedPolicy::Path` un-marks on unwind so every route to a node is reported |
| Reusable search marks | `search_with_marks` + `EpochMarks` | The same search with its visited marks in a caller-owned epoch-stamped buffer: a per-root pass allocates marks once instead of an O(node count) buffer per root, and each search still starts from a clean set (epoch bump, O(1)) |
| Readable dense set | `EpochSet` | The same epoch stamp with its members kept: constant-time `insert`/`contains`, a constant-time `clear`, and `ones` in one pass over the answer rather than the universe (insertion order, not ascending) |
| Reusable search scratch | `search_with_scratch` + `SearchScratch` | The same search with the marks *and* the call's own buffers — seeds, frontier, adjacency — caller-owned, for a pass whose searches are small enough that the call is the cost; marks and buffers both reset on entry |
| Disjoint sets | `DisjointSet` (`union` by rank, `union_toward_min` for deterministic minimum representatives, growable via `push`) | The union-find backing phi webs and may-alias location classes, usable directly by consumers |
| Open post-order fold | `open_fold_post_order` + `OpenFold` trait (`FoldEnter::Leaf`/`Fold`, `MarkScope::Shared`/`Isolated`) | Child-answers-to-parent folding over a lazily discovered space: caller-keyed cycle guard (nodes need no `Ord`), per-node mark scoping (exact per-path marking or persistent pruning), early exit, depth bound — C++-style member lookup and type-algebra evaluators without hand-written frame stacks |
| Forward-only view support | `scan_predecessors` | The honest O(nodes × edges) `predecessors` for views that store only successors, replacing per-consumer scan stubs |
| Traversal events | `breadth_first_events` → `BfsEvent`; `depth_first_events` → `DfsEvent` | Breadth-first discovery with tree/non-tree edges, or depth-first discover/tree/back/forward-or-cross/finish events in pinned order |
| Open-graph search | `open_search` + `OpenSearchConfig` | The same disciplines over a lazily discovered node space: successors come from a closure, no dense ids (import/re-export chases, ordered emission walks) |
| Open-graph events | `open_breadth_first_events` → `OpenBfsEvent`; `open_depth_first_events` → `OpenDfsEvent` | Breadth-first discovery/refusal events or depth-first discover/finish/refusal events over a lazily discovered node space; path policy can revisit a shared node once per route |
| Open-graph route enumeration | `open_breadth_first_paths` → `OpenPathsEvent` | Simple routes shortest-first over a lazily discovered space — no global marks, per-route cycle guard; exponential worst case, so bound with depth, `Skip`, or `Break` |
| Alias chase | `follow`, `follow_path` | Out-degree ≤ 1 chase with a hop bound and a full-path cycle guard; the chain, or just its end |
| Shortest path | `shortest_path` (nodes), `shortest_path_edges` (edge witness) | Forward or reverse unweighted witness path |
| Minimum-label relaxation | `min_label_relaxation` | Edge-defined label transfer to a minimum fixpoint; nodes re-expand when their label improves |
| Multi-source reachability | `reachable` | Dense `Vec<bool>` from a seed set, forward or reverse; order-insensitive |
| Nearest common ancestor | `nearest_common_ancestor` | Bidirectional BFS meet of two nodes; smallest combined distance, ties by smallest node id |
| All common ancestors | `common_ancestors` → `Vec<CommonAncestor>` | Every shared node with both hop counts, depth-bounded, in `b`'s BFS discovery order — the consumer-rankable form (MRO linearization, overload preference) |
| Horn-clause derivability | `HornClauses` | AND-OR closure (`head <- b1 & b2`): nullability, all-arguments-constant, all-callers-dead |
| Topological sort | `topological_sort` | Stable ordering or cycle detection |
| Dominator tree | `DominatorTree::compute` (rooted views), `compute_from` (explicit root) | Simple Lengauer-Tarjan over any graph view, O(m log n) with explicit stacks. The tree is numbered once in preorder, so `dominates` is O(1) and `subtree`, `preorder`, `level`, `children`, and `nearest_common_dominator` need no extra tables |
| Reusable dominator scratch | `DominatorTree::compute_in`, `compute_from_in` + `DominatorScratch` | The same tree with the call's working buffers caller-owned, for a pass computing one tree per procedure over a codebase; the scratch holds dense indices, so one instance serves every graph and node-id type |
| Post-dominator tree | `PostDominatorTree::compute` (rooted views) → `PostDominatorTree<N>`, `PostNode` | Over a virtual exit that every exit joins. Exit-free regions are cut at the blocks closing each loop, so every reachable block has facts independent of successor order. `post_dominates` is O(1); `ipdom`, `nearest_common_post_dominator`, `exits`, `cut_roots`, `reaches_exit`, `subtree`, `children`, `frontier` (the control-dependence sources of a block), and `control_dependents` (the blocks control dependent on one arm) answer from the one numbered tree |
| Reusable post-dominator scratch | `PostDominatorTree::compute_in` + `PostDominatorScratch` | The same tree with the call's working buffers caller-owned, parallel to `DominatorScratch` |
| Dominance frontiers | `DominanceFrontiers::compute`; `frontier` → `&[N]` | Every block's frontier in one flat run table, read back as a slice in ascending order. For SSA φ-placement |
| Dominator recompute diff | `DominatorTree::compute_with_diff` | Recompute after a graph edit, reporting the nodes whose idom changed |
| Strongly connected components | `tarjan_scc` → `SccDecomposition<N>`, `condensation` → component DAG of owned `Scc<N>` payloads | Generic iterative Tarjan algorithm, reverse-topological order (leaves first); components are ascending slices of one flat array (`components()`, `component(node)` → `Scc` with `nodes()`, `contains`, `is_singleton`) |
| SCC in topological order | `kosaraju_scc` → `SccDecomposition<N>` | The same partition numbered sources first (`index(u) < index(v)` across every edge); the classic deterministic two-pass algorithm, for budgeted forward closures over the condensation |
| Condensation of a given decomposition | `condensation_of(graph, &SccDecomposition)` → `Graph<(), ()>` | The component DAG whose node index **is** the given decomposition's component index, either algorithm's; deduplicated edges, and in-degrees plus dependents straight off the graph (the one-pass fixpoint shape) |
| Back-edge detection | `find_back_edges` (dominance, any view), `find_back_edges_tagged` (CFG, honors `Back` tags) | |
| Natural loop detection | `detect_loops` / `detect_loops_tagged` → `Vec<NaturalLoop<N>>` | Header, body, latches, nesting depth |
| Loop nesting tree | `LoopNestingTree::compute` | Parent/child loop hierarchy |
| Control dependence graph | `control_dependence_graph` → `Graph<N, ()>` | From post-dominator tree, over any view; a block that branches back to itself has the edge to itself |
| Program dependence graph | `program_dependence_graph` → `Graph<DependenceNode, DependenceKind>` | Control + def-use edges; reverse traversal performs backward slicing |
| Interval analysis | `IntervalAnalysis::compute` | T1-T2 reduction over rooted views; reducibility test |
| Reducibility transform | `make_reducible` | Node splitting for irreducible CFGs |
| Reverse CFG | `reverse_cfg` | Flip all edges, swap entry/exits |
| Call graph | `call_graph` + `CallInfo`, `propagate_summaries` (callee-first SCC fixpoint) | Consumer-typed callees; interprocedural summary scaffold |
| CFG diff | `CfgDiff::compute` | Structural comparison (bindiff-style fingerprinting), no trait bounds |
| Exception handling model | `EhModel::compute` | Payload-generic CFG input; stable source `EdgeId`, exact handler/unwind/leave/resume/continue kinds, landing pads, cleanup/resume blocks, handler identities, protected-by mapping, and cleanup continuations |
| Integrity verification | `verify`, `verify_view`, `verify_edge_view`; `verify_with` + `SemanticValidator` | Structural node/edge-view checks plus deterministic typed consumer hooks for cardinality, ordering, and edge-payload rules |
| DOT export | `write_dot` / `to_dot` + `DotStyle` (any node- and edge-bearing view); `Cfg::to_dot` (`DisplayInstr`), `Cfg::to_dot_with` (bound-free), `Graph::to_dot` | One writer; the style supplies node labels, edge attributes, graph name, and layout direction. Labels are escaped and left-justified |
| Text form | `write_text` / `to_text` + `TextStyle`, `parse_text`; `Cfg::to_text`, `parse_cfg_text` | One line per node and per edge, ids are the dense indices. Round trips: write → parse → write is a fixed point, so two graphs compare as text |

### Dataflow framework

Instruction-level analyses read the graph through `CfgView`. The graph-shaped solvers read a rooted, node, or edge view instead.

#### Instruction-level view (`CfgView`)

`CfgView` is the one read-only trait that the instruction-level algorithms take. It joins the graph views with the facts that these algorithms need and a graph view does not give: the ordered instructions of each block, the `EdgeKind` of each edge, and the optional branch weight of each edge.

| Part | Description |
|---|---|
| Supertraits | `RootedView<NodeId = BlockId>` and `EdgeView<EdgeId = EdgeId>`. The edge data implements `KindedEdge`. The root is the entry block. |
| `Inst`, `instructions` | The instruction type of the blocks, and the instructions of one block in program order. |
| `edge_kind` | The kind of an edge. The default reads it from the `KindedEdge` data. |
| `edge_weight` | The branch weight of an edge: a probability or an execution count. The default is `None`, which means that the view has no profile. |

`Cfg` implements `CfgView`, so every caller that holds a `Cfg` keeps working. A consumer with its own block and edge storage implements the trait and runs the same analyses without a conversion. The unit tests of the trait run each analysis on a vector-backed view and compare the answer with the one on a `Cfg`.

The contract has two rules. `instructions` accepts every identity that `node_ids` yields, and it returns the same slice each time. The index of an instruction in that slice is the instruction index of the `ProgramPoint` that an analysis reports.

The view has no separate exception model. An edge of kind `EdgeKind::ExceptionUnwind` leaves its source block before a throwing instruction. `InstrInfo::may_unwind` marks the throwing instructions of a block. `dataflow::departs_before_throws` combines the two.

These analyses take `&G` where `G: CfgView`: `ReachingDefs`, `Liveness`, `DefUseChains`, `SsaForm`, `PhiPlacements`, `constant_propagation`, `SccpAnalysis`, `abstract_interpret`, `MemoryTrace`, `MemorySSA`, `DeadCode`, `recover_expressions`, `ValueNumbering`, `BlockValueNumbers`, `cfg_purity`, `CfgMetrics`, `detect_cfg_patterns`, `CfgProfile`, `detect_tail_calls`, and `detect_switch_tables`. The problems `LivenessProblem`, `SeededLivenessProblem`, `ReachingDefsProblem`, and `ConstPropProblem` also run on every `CfgView`. A transform that edits the graph still takes `&mut Cfg`.

| Analysis | Function / Type | Description |
|---|---|---|
| Generic fixpoint solver | `solve_problem[_from][_with_config]`, `Problem` / `TryProblem` traits | Forward or backward, any lattice type; every solver carries the same seeded/bounded/fallible matrix over shared `SolveConfig` and `SolveError`; `meet_options` lifts any lattice to `Option<F>` with `None` as unreachable bottom |
| Node-level fixpoint | `solve_node_problem[_from][_with_config]`, `NodeProblem` / `TryNodeProblem` traits | Per-node facts over any graph view (taint, reachability-with-facts) |
| Edge-sensitive fixpoint | `solve_edge_problem`, `solve_edge_problem_from`, `EdgeProblem` trait | Full or seeded per-edge transfer over any edge view; stable id/data plus physical node pre/post states and deterministic bounded-solve errors |
| Fallible edge-sensitive fixpoint | `try_solve_edge_problem`, `try_solve_edge_problem_from`, `TryEdgeProblem` trait | Preserves consumer boundary, merge, node-transfer, and edge-transfer errors separately from solver limits |
| Reachability-lifted edge analysis | `Reachable` + `ReachableEdgeProblem` trait | Verification-style analyses over `Option` facts: `None` is unreached bottom, transfers short-circuit, entry facts start the flow, and each edge chooses pre- or post-state (exceptional edges observe the state the node received) |
| Dense bit-set facts | `DenseBits` | Set-of-indices lattice element packed 64 per word; `union_with` is the join and reports change for convergence checks; `clear` is a word fill, `words` exposes the packed row, and `ones`/`intersection` step one set index at a time rather than one universe index |
| Reaching definitions | `ReachingDefs::compute` | Which writes reach each point; an `ExceptionUnwind` edge carries only the writes completed before each instruction that `InstrInfo::may_unwind` |
| Liveness | `Liveness::compute` | Live-in / normal-completion live-out at each block; `live_on_unwind` is what unwind successors read before each throwing instruction; `live_before_instructions` / `live_after_instructions` replay the block transfer for instruction-granular sets (dead-store detection) |
| Def-use / use-def chains | `DefUseChains::compute` | Bidirectional def↔use links; dead-def detection |
| SSA construction | `SsaForm::compute` | IDF phi placement plus full dominator-forest renaming, including disconnected handler/dead-code components; each phi operand names its departure point, so an unwind contributes the value before each throwing instruction |
| Reusable SSA scratch | `SsaForm::compute_in` + `SsaScratch` | The same form with every working buffer caller-owned — frontiers, definition lists, renaming stacks, phi drafts, the walk's own stacks — leaving only the form itself allocated per procedure |
| Phi placement | `PhiPlacements::compute` | Structural IDF phase for consumers that only need placement |
| SSA deconstruction | `eliminate_phis`, `copies_by_predecessor` | φ-to-copy lowering |
| Phi webs | `PhiWebs::compute` | Congruence classes for register coalescing |
| Constant propagation | `constant_propagation`, `ConstantFolder` (associated `Const`) | Top/Const/Bottom lattice over a consumer constant domain — machine words, strings, bools, float bits; an unwind carries the constants known before each throwing instruction |
| Sparse conditional constant propagation | `SccpAnalysis::compute` | SSA-based, marks unreachable edges |
| Copy and value-alias propagation | `copy_propagation[_with_exits]`, `alias_propagation`, `CopySource` trait | Guarded chain resolution and dead transfer removal; pairwise aliases may refine types or metadata without changing runtime values. The `_with_exits` form keeps a copy whose definition is live at an exit — a function whose result is only ever a copy of an input otherwise loses the definition a caller reads back — while still rewriting that copy's readers inside the function. A transfer at or after a throwing instruction never substitutes into code its unwind reaches |
| Memory-event trace | `MemoryTrace::compute`, `MemoryEventInfo` trait | Ordered, location-typed reads, writes, read/modify/write accesses, address-variable dependencies, and fences; instruction summaries distinguish separate read+write from compound modification |
| Reusable memory-SSA scratch | `MemorySSA::compute_in` + `MemorySsaScratch` | The same answer with the event trace, the alias merge, and the shadow CFG's dominator and SSA buffers caller-owned |
| Memory SSA | `dataflow::memory::MemorySSA::compute`, `MemoryAlias` trait | Event-driven SSA per may-alias location class: loop/branch φ-nodes, reaching writes and clobbers, bidirectional def-use chains, transitive readers, and ordinary-SSA address inputs; an unwind observes the memory state before each throwing instruction |
| Reusable value-flow scratch | `MemoryValueFlow::compute_in` + `MemoryValueFlowScratch` | The same graph with the per-event address/stored/loaded resolution buffers caller-owned |
| Memory value flow | `dataflow::memory::MemoryValueFlow::compute` | One graph over ordinary SSA values, versioned memory states, and exact events; typed address/store/read/write/load and ordinary/memory-phi edges retain the complete transfer path |
| Abstract interpretation | `abstract_interpret`, `AbstractDomain` trait | Generic abstract domain framework on the edge-sensitive solver; an unwind carries the state before each throwing instruction |

### Higher-level analyses

| Analysis | Function / Type | Description |
|---|---|---|
| Expression tree recovery | `recover_expressions`, `ExprInstr` (associated `Operator` + `Const`) | Rebuild expression DAGs from flat instructions |
| Value numbering (local) | `BlockValueNumbers::compute` | Per-block hash-consing |
| Value numbering (global) | `ValueNumbering::compute`, `ValueNumberInfo` (associated `Operator`) | Dominator-scoped GVN over any operation identity; a child an unwind reaches never reuses what its parent computes or writes from the first throwing instruction on |
| Redundancy counting | `ValueNumbering::redundant_count` | From GVN results |
| Explicit alias sets | `AliasSets::new` / `merge`; `MemoryAlias` trait | Caller-populated union-find classes usable directly as the alias oracle for memory SSA; an unmerged pair is a proof of disjointness |
| Index-path aliasing | `index_paths_may_overlap` | The `base[i][j]` kernel for alias oracles over indexed storage: unequal arity or unknown components stay conservative, fully known disagreement proves disjointness |
| Exclusive handler extents | `recover_exclusive_extents[_with]`, `promote_exclusive_extents` | Evidence-reporting sibling of `promote_handler_extents`: blocks exclusively reachable from one handler entry along normal edges, boundary and ambiguity issues, and atomic per-region promotion that refuses overlap |
| Dead code analysis | `DeadCode::compute`, `DeadCode::compute_with_exits` | Liveness-dead instructions (effect-guarded) and unreachable blocks, reported without mutating — the analysis `dead_code_elimination` applies. The seeded form takes what leaves each block (a returned value, storage a caller reads again, state a non-returning exit hands on), which a graph read from the inside cannot state; `SeededLivenessProblem` carries the seed into the solve |
| Purity classification | `cfg_purity`, `block_purity`, `EffectInfo` (associated `Effect`) | Consumer effect vocabularies — machine memory/IO, allocation, panics |
| Metrics | `GraphMetrics::compute` (any rooted view); `CfgMetrics::compute` | Node/edge counts, cyclomatic complexity, nesting depth, instruction density |
| Pattern detection | `detect_patterns` (any view), `detect_cfg_patterns` (adds trampolines + arm orientation) | Diamond, chain, self-loop, empty trampoline |
| Profiling | `CfgProfile::from_edge_weights`, `set_uniform_edge_weights` | Edge probabilities propagated from the entry into per-block frequencies (loops solved to a fixpoint), hot/cold block queries |
| Tail call detection | `detect_tail_calls` (heuristic), `detect_explicit_tail_calls` (`CallInfo` markers) | A call that ends its block and flows into a block that returns at once; explicit markers reported with their position |
| Switch table recovery | `detect_switch_tables` (`SwitchSource`), `recover_switch_tables` | Consumer-typed targets: addresses, syntax nodes; dispatch → structured switch |
| Control tree | `ControlTree::compute`, `compute_with_report` (→ `LiftReport`), `compute_predicated[_with_report]`, `compute_with_report_treating_empty` | The structured form of a `Cfg`: `if/else`, pre-tested, post-tested, and endless loops, `switch` with case edges and a default, `try/catch` over declared handler extents, `break`/`continue` (labeled across loops), `goto`/`label`, and guarded runs of predicated instructions. A payload-free fact like `DominatorTree`: a flat arena of `ControlNode`s that name `BlockId`, `EdgeId`, `RegionId`, and `BlockSpan` instruction ranges, so it holds no instruction and never borrows the graph. Merge points are immediate post-dominators; an exit-free region has none. No owned code is dropped — the blocks the entry reaches and the blocks a handler owns, a catch pad no call unwinds into and its extent among them; a protected block the entry cannot reach is dead code — and `LiftReport` lists every goto, swept block, unstructured region, and dangling label |
| Control tree traversal | `ControlTree::preorder`, `walk` (`ControlVisitor`), `child_bodies`, `for_each_instruction`, `write_pseudocode` / `to_pseudocode` | Non-recursive preorder and enter/leave walks, the child bodies of any node in execution order, every instruction in program-structure order, and indented pseudocode whose instruction text the caller supplies |

### Transforms

`PassPipeline` composes heterogeneous named passes over any caller-owned target.
`pass_fn` adapts a closure; implementing `Pass` supports stateful reusable
passes. Schedules retain insertion order and report the completed prefix when a
fallible pass stops execution.

| Transform | Function | Description |
|---|---|---|
| Simplify (all-in-one) | `simplify`, `simplify_mapped` | Unreachable removal + block merging + empty bypass until stable; mapped form composes identity changes |
| Remove unreachable | `remove_unreachable` | Reachability pruning; blocks named by exception metadata are kept |
| Merge blocks | `merge_blocks` | Coalesce single-succ/single-pred chains inside one protected region |
| Remove empty blocks | `remove_empty_blocks` | Bypass empty fallthrough blocks that no handler or cleanup names |
| Critical edge splitting | `split_critical_edges`, `split_critical_edges_with` | Insert blocks on multi-succ → multi-pred edges while retaining the original edge identity/payload and mapping both halves |
| Dead code elimination | `dead_code_elimination[_with_exits]` (instructions), `remove_dead_code[_mapped]` (plus the structure left dead) | Liveness-based; requires `EffectInfo` so side-effecting code is never silently deleted. The `_with_exits` form seeds what leaves each block |
| Edge contraction | `contract_edge`, `contract_edge_mapped` | Merge two blocks connected by a single edge; mapped form preserves surviving edge identities/payloads. Refuses exception edges and merges across a protection boundary |
| Node splitting | `Cfg::split_block`, `Cfg::split_block_at_points[_with_payloads]` | Split at one or several validated consumer-selected instruction boundaries; the new blocks stay in the original's regions |
| Loop rotation | `rotate_loop` | Top-tested → bottom-tested loop form |
| Loop invariant detection | `find_loop_invariants` | Identify hoistable instructions |
| Partial redundancy elimination | `PreAnalysis::compute`, `eliminate_pre` | GVN-based PRE |
| Graph coloring | `interference_graph`, `color_graph` | Interference builder returns a `Graph`; coloring accepts any graph view |
| Linearization | `linearize`, `Emitter` trait, `BlockOrder` | Re-serialize CFG to a flat stream; emitters speak `BlockId`, naming is theirs |
| Width relaxation | `relax_layout` | The assembler fixed point for symbolic layouts: monotone branch widening against caller-resolved labels, offset-dependent (alignment) widths, converged offsets plus the final label context |

The structural passes (`simplify`, `merge_blocks`, `remove_empty_blocks`,
`remove_unreachable`, `contract_edge`) keep a CFG's exception metadata true, so
`Cfg::compact` and `EhModel` stay usable after any sequence of them. A block a
handler, filter, or cleanup names by a single reference is never removed or
merged away (`remove_unreachable` treats such blocks as roots), and two blocks
merge only when the same regions protect both. `Cfg::split_block` is the
inverse: the new tail joins every region and handler body of the original.
Call `Cfg::remove_unreachable_regions` first to drop regions nothing can raise
into together with their handler blocks.

### Output

Every renderer writes into a `&mut dyn fmt::Write` sink. The `to_*` forms are the
allocating counterparts. `Cfg::write_text` indents through [`IndentedWriter`],
which consumer printers can also use.

| Form | Entry points | Notes |
|---|---|---|
| DOT | `write_dot` / `to_dot` over a `DotStyle`; `Cfg::write_dot`, `Cfg::write_dot_with`, `Graph::write_dot` | `DotStyle` carries the node-label hook, the edge-attribute hook (`DotEdgeAttributes`: label, color, style, penwidth), the graph name, the node-identifier prefix, and `DotRankDir`. `control_flow_edge_attributes` is the CFG's kind-to-color mapping; `plain_edge_attributes` decorates nothing |
| Text | `write_text` / `to_text` over a `TextStyle`, read back by `parse_text`; `Cfg::write_text` / `Cfg::to_text`, read back by `parse_cfg_text` | One fact per line. `n3 label`, `n3 -> n5 label`, `#` comments; a CFG adds `entry bb0`, `bb3:` headers with indented instructions, `bb3 -> bb5 kind`, and `region` / `handler` lines. Escapes are `\\`, `\n`, `\r`, and a leading `\-` |

Both label hooks are ordinary closures or function items of type
`Label<Id, Data>`; `no_label` and `display_label` cover the common answers,
and `bind_label` / `bind_edge_attributes` state the higher-ranked signature
for a closure that would otherwise infer one fixed payload lifetime.

The text form is a readable projection — identity, topology, and label text.
Edge weights, non-text consumer payloads, and cleanup records live in the
`serde` representation of the store, which remains the fidelity path.

[`IndentedWriter`]: https://docs.rs/cfglib/latest/cfglib/struct.IndentedWriter.html

## Extension contracts

The generic graph has no consumer trait requirement when it owns the storage. `Graph`, `Cfg`, and `KeyedGraph` expose both node and edge views directly. Implement `DenseId` and the four required `GraphView` methods (`node_bound`, `node_ids`, `successors`, and `predecessors`) only when adapting another graph store; implement `EdgeView` when algorithms must observe stable edge identities or payloads; add `RootedView` (or use `Rooted`) for entry-requiring algorithms. Implement `CfgView` when the instruction-level analyses must read your own block and edge storage. Instruction traits are opt-in according to which CFG and dataflow features an adapter needs — every associated type is the consumer's own:

```text
Graph<N, E>           (owned arbitrary graph; no adapter trait)
GraphView             (existing consumer-owned graph storage)
├── NodeView          (graph-owned node payloads)
├── EdgeView          (stable edge identity, endpoints, data, adjacency)
└── RootedView        (adds a distinguished entry node; `Rooted` adapts)

CfgView                   (RootedView + EdgeView + block instruction slices;
                           `Cfg` implements it; every instruction-level
                           analysis reads it — associated Inst)

FlowControl               (required only by CfgBuilder)
└── JumpTargets           (goto/label wiring — associated Target)

InstrInfo<Variable = V>   (optional — native IR variables, defs/uses)
├── EffectInfo            (purity, DCE — associated Effect)
├── Predicated            (guard variable of a guarded instruction)
├── CopySource            (copy propagation)
├── ConstantFolder        (constant propagation, SCCP — associated Const)
├── ExprInstr             (expression trees — associated Operator, Const)
├── ValueNumberInfo       (value numbering, PRE — associated Operator)
└── MemoryEventInfo       (locations, address uses, access kinds, and fences — associated Location, Fence)

MemoryAlias<Location>     (optional may-alias oracle consumed by MemorySSA)

DisplayInstr              (rendering only — DOT and text)
CallInfo                  (call graphs, explicit tail calls — associated Callee)
SwitchSource              (switch table recovery — associated Target)

ir::rtl::Dialect          (instruction-set vocabulary — Root, Operator, EffectOp,
│                          WritePolicy, Outcome, NondetDomain)
└── ir::rtl::Fold         (constant evaluation of operators)
ir::rtl::InstructionBuilder<D>  (write-only emission into consumer storage)
ir::rtl::RecordView<D>    (one sealed instruction record)
ir::rtl::FunctionView<D>  (RootedView + blocks of instruction records)

ir::mlil::Dialect         (Op, Type, Edge)
ir::mlil::FunctionView<D> (RootedView + blocks of flat SSA instructions)

ir::hlil::Dialect         (Op, Constant, Type, Variable)
ir::hlil::FunctionView<D> (read-only statement and expression trees)
```

The `ir` traits follow the same rule. The consumer implements `ir::rtl::FunctionView`, `ir::mlil::FunctionView`, or `ir::hlil::FunctionView` on its own dense storage. The RTL and MLIL views are `RootedView`s, so the graph algorithms, such as `DominatorTree::compute`, run on them directly. The HLIL view is a pair of trees, not a graph.

`MemoryEventInfo` is the single instruction-side source of truth for memory.
Directional `MemoryAccess::read`, `write`, and `read_modify_write`
constructors keep loaded definitions, stored uses, and address uses distinct
(`MemoryAccess::opaque` covers table-driven kinds with unknown values);
events also retain exact locations, atomicity, and ordered fences. `MemorySSA`
in `dataflow::memory` merges locations through the caller-provided
`MemoryAlias` relation, then exposes loop-correct reaching definitions,
clobbered states, users, and transitive readers. `MemoryValueFlow` combines it
with the matching ordinary `SsaForm`, making
`value -> store -> memory state -> load -> value` explicit. Returning `false`
from an alias oracle promises the locations cannot overlap at runtime;
`ConservativeMemoryAlias` is the safe fallback when that cannot be proven.

## Workspace

| Crate | Description |
|---|---|
| **cfglib** | Generic graph, CFG, SSA, and dataflow framework with the RTL and MLIL IR trait contracts |

Adapters live with the language they adapt, next to their decoders and test
corpora. The SM4/SM5 decompiler is in
[d3dasm](https://github.com/coconutbird/d3dasm). Its `cfg` module also exposes
a raw CFG and component-granular SSA adapter. The Java/DEX decompiler is in
cafe.

A language picks its entry level. A machine-shaped language emits `ir::rtl`
records through `InstructionBuilder` into its own storage. A variable-shaped
language implements `ir::mlil::FunctionView` directly. In both cases the
storage, the builders, and the passes that edit them stay with the language.
cfglib deliberately defines no generic LLIL. Low-level IRs are language-owned,
and the minimal traits that they share (`FlowControl`, `InstrInfo`, `CfgView`,
`Cfg`, `SsaForm`) are their common surface.

## Adapting a language, IR, or existing graph

For symbol, reference, value-flow, type-relation, import, or grammar graphs, store domain objects directly in `Graph<N, E>`, or in `KeyedGraph<K, N, E>` when the domain identities are sparse keys. Projects that already own adjacency lists can instead implement `GraphView` for their store, then use traversal, shortest-path, topological-sort, SCC, dominance, loop-detection, metrics, and pattern algorithms without migrating their data. Dense `u32` and `usize` handles work directly; custom handles implement `DenseId`.

For a control-flow and SSA adapter:

1. Build `Cfg<I>` directly with `new_block()` / `add_edge()`, or use `Cfg<I, E>::with_edge_payload()` plus `add_edge_with_payload()` when branch labels, handler order, continuation/call-site identity, or source locations must survive. Structured unit-payload streams can instead implement `FlowControl` and use `CfgBuilder::build()`. A consumer with its own block and edge storage implements `CfgView` instead of building a `Cfg`.
2. Optionally implement `InstrInfo` with a native `Variable` identity (and its sub-traits) for dataflow analyses.
3. Implement `DisplayInstr` when you want DOT or text output.

`Cfg<I, E>` is a `Graph<BasicBlock<I>, Edge<E>, BlockTag, EdgeTag>` plus the entry block and the exception metadata that make it a control-flow graph; `E` defaults to `()` for the unannotated form. Two records describe what a change did, and they are not the same thing: `Rewrite` is a transform's sparse account — a missing entry means unchanged, an empty replacement list means removed, one replacement means retained or redirected, several ordered replacements mean a split — while `Renumbering` is the total account `Cfg::compact` gives when it renumbers the survivors. `Rewrite::then_renumbered` carries the first across the second. Stable parallel `EdgeId`s plus caller payloads are also the generic continuation/call-site mechanism; cfglib does not impose a VM-specific continuation type.

The variable type only needs `Clone + Ord` — never `Copy`, never numeric. It can be an architecture enum such as `Register(Rax)` / `Flag(Zero)`, a shader structure such as `(register file, index, component)`, an interned source symbol, or an existing IR value handle. Adapters decide the atomic aliasing unit; for overlapping resources such as x86 subregisters, expose canonical units or every affected unit.

```rust
use cfglib::{Cfg, DominatorTree, InstrInfo, SsaForm};

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
enum X86Variable { Register(u8), Flag(u8), StackSlot(i32) }

struct X86Instruction {
    uses: Vec<X86Variable>,
    defs: Vec<X86Variable>,
}

impl InstrInfo for X86Instruction {
    type Variable = X86Variable;

    fn uses(&self) -> &[Self::Variable] { &self.uses }
    fn defs(&self) -> &[Self::Variable] { &self.defs }
}

let cfg = Cfg::<X86Instruction>::new();
let dominators = DominatorTree::compute(&cfg);
let ssa = SsaForm::compute(&cfg, &dominators);
```

`SsaForm<V>` is a non-mutating view over the source CFG. It stores renamed phi results, operands, instruction uses, and instruction definitions as `SsaValue<V>`, while each `SsaInstruction` keeps a `ProgramPoint` back to the native instruction. Read the lists through `SsaPhi::operands`, `SsaInstruction::uses`, and `SsaInstruction::defs`, which return slices; a record holds a short list inline, so a typical instruction or two-way phi costs no allocation of its own. `PhiPlacement::predecessors` follows the same shape. Version `0` denotes a live-in or otherwise undefined incoming value.

For an IR adapter, implement the `ir::rtl` or `ir::mlil` traits on your own dense storage. Keep every arena, builder, and editing pass in your crate. Call the facts of cfglib on your function view, and apply their answers yourself. A machine-shaped language emits `ir::rtl` records through `InstructionBuilder`. A consumer that builds a block-parameter form from those records uses `ParamPlacement` to find where each root needs a parameter. A variable-shaped language implements `ir::mlil::FunctionView` directly. The consumer writes its own value analyses over that view, because they depend on the semantics of its dialect, and its own passes delete or rewrite instructions.

The concrete shader-bytecode adapter (the d3dasm decompiler's `cfg`
module) derives native register-component identities from decoded masks
and swizzles, retains relative index expressions, classifies multi-result
and UAV read-modify-write operations, and reports observable shader effects
through its own effect vocabulary.

The `tests/source-cfg.rs` integration test is the executable specification of
the source-language side: interned symbol variables, string/bool constants
through constant propagation, enum operators in expression trees, goto wiring
by label token, and switch recovery over syntax-node targets.

## Development

Install the git hooks once with `prek install` (or `pre-commit install`); both
tools read the same `.pre-commit-config.yaml`. Hygiene checks, the repository
policy script, `cargo fmt`, and pedantic Clippy run at commit time; the full
test and documentation suites run at push time. CI runs the same gates with
default features as well, plus an MSRV check against the `rust-version` in
`Cargo.toml` (the one place the minimum Rust version is written down) and
`no_std` target builds. The complete gate list and the workspace's naming and
layout conventions live in `AGENTS.md`.

## License

MIT
