# cfglib

Generic, `no_std` graph and dataflow framework for code intelligence, program analysis, decompilation, and compiler infrastructure.

`cfglib` has purpose-built storage for arbitrary directed relations, language-parametric scope graphs, file-incremental stack graphs, and `Cfg<I, E = ()>` graphs. A `Cfg` has basic blocks, typed control-flow edges, caller-owned edge metadata, and exception regions.

`cfglib::ir` holds IR traits. `ir::rtl` is the register-transfer contract. `ir::mlil` is the read-only view contract of a flat function in static single assignment form with block parameters. `ir::hlil` is the structured contract: a read-only view of statement and expression trees and a builder trait that the consumer implements. The consumer owns every IR storage, builder implementation, editing pass, and IR algorithm (lifting, lowering, verification, value analyses, walks over the structured shapes, and the facts over an RTL function). The RTL and MLIL views are rooted graph views, so the graph algorithms of cfglib run on them and return facts about the graph.

Small read-only node and edge view contracts let consumer-owned stores and zero-copy filtered views reuse the algorithms. `CfgView` extends them for the instruction-level analyses of a consumer, so the generic solvers run the same way over a consumer block store as over `Cfg`. Every instruction-adjacent axis is consumer-typed. Dataflow variables, constants, expression operators, side-effect vocabularies, branch targets, call targets, and edge payloads all come from the adapter. x86 registers and flags, shader register components, bytecode locals, compiler IR values, and source-language symbols do not need a library-owned numbering scheme. String literals and symbol ids flow through the solvers as naturally as machine words and addresses.

On top of that, cfglib has a compiler middle-end toolkit: dominator trees, dominance frontiers, and generic dataflow solvers. Instruction-level analyses such as reaching definitions, SSA construction, and memory SSA live in the `il` crate.

Named transformations compose through `PassPipeline<T, E>`. A pipeline keeps
the caller's declared order, supports closure-backed or stateful trait passes,
reports which executions changed the target, and attributes the first failure
while retaining the completed-prefix report. Pass selection and canonical
schedules remain consumer policy, so one framework can drive a CFG or any
consumer-owned compilation context without coupling their dialects.

Everything is `no_std + alloc`. The core graph store keeps adjacency in compressed arrays and removes entities with a liveness bit, so identities stay stable until an explicit `compact`.

## Quick start

### Direct construction (the primary front door)

No trait is required to build, verify, analyze, or render a CFG. Source
frontends translate their syntax trees straight into blocks:

```rust
use cfglib::{Cfg, DominatorTree, EdgeKind, PostDominatorTree, verify};

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
let post_dominators = PostDominatorTree::compute(&cfg);
```

### Builder construction (structured instruction streams)

Frontends with a flat, structured stream (shader bytecode, structured ISAs)
implement `FlowControl` and use `CfgBuilder`. An instruction is a structured
marker, a label, or a `Flow` — the same vocabulary `build_address_cfg` reads —
and `resolve_jump_edges` wires the targets those flows name afterwards:

```rust
use cfglib::{CfgBuilder, Flow, FlowControl, FlowEffect, resolve_jump_edges};

impl FlowControl for Inst {
    type Target = Label;
    type CaseKey = ();

    fn flow_effect(&self) -> FlowEffect<Label, ()> {
        match self.opcode {
            Op::If          => FlowEffect::ConditionalOpen,
            Op::Else        => FlowEffect::ConditionalAlternate,
            Op::EndIf       => FlowEffect::ConditionalClose,
            Op::Loop        => FlowEffect::LoopOpen,
            Op::End         => FlowEffect::LoopClose,
            Op::Label(name) => FlowEffect::Label(name),
            Op::Goto(name)  => FlowEffect::Flow(Flow::Jump { target: name }),
            Op::Ret         => FlowEffect::Flow(Flow::Return),
            _               => FlowEffect::Flow(Flow::Next),
        }
    }
}

let mut cfg = CfgBuilder::build(instructions).unwrap();
let resolution = resolve_jump_edges(&mut cfg); // wires goto/label edges
```

## Feature overview

### Generic graph core

`Graph<N, E>` is the single owned storage type for consumer-defined node and edge payloads, stable identities, and forward/reverse adjacency. `GraphView` lets existing graph stores use node algorithms without first migrating their storage; `EdgeView` additionally exposes stable edge identity, view-oriented endpoints, payloads, and ordered adjacency. `FilteredEdges` borrows either representation through an edge predicate without cloning or renumbering anything—for example, the same CFG can be viewed as normal-only flow or full normal-plus-exception flow. `RootedView` adds a distinguished entry node for algorithms that need one (dominance, reachability metrics, loop and interval analysis); the `Rooted` adapter roots any plain view at a chosen node, `Reversed` reverses every edge, `Induced` confines a view to a node set, and `VirtualRoot` puts one extra node in front of a list of roots, so a post-dominator or multi-entry question composes views instead of copying the graph. This layer is suitable for symbol/reference graphs, value-flow graphs, call graphs, type relations, import graphs, grammar dependencies, and analysis-derived relations; it has no instruction or binary-analysis concepts.

```rust
use cfglib::{Direction, DominatorTree, Graph, Rooted, shortest_path};

let mut graph = Graph::new();
let source = graph.add_node("definition");
let target = graph.add_node("call result");
let edge = graph.add_edge(source, target, ("return", "src/lib.rs", 42));

assert_eq!(graph.edge(edge).payload().0, "return");
assert_eq!(
    shortest_path(&graph, source, target, Direction::Forward),
    Some(vec![source, target]),
);
let dominators = DominatorTree::compute(&Rooted::new(&graph, source));
```

`Graph` keeps a compressed sparse-row base beside an appendable delta in one type: adding a node or an edge is constant time with no per-node allocation, removal clears a bit in a liveness bitset, `out_degree`/`in_degree` answer from the compressed run's two offsets plus a walk of whatever was appended since, redirecting one endpoint keeps the edge's identity, and `compact(order)` folds the delta back into the base and hands back a `Renumbering` describing every identity it moved. That is what makes whole-codebase graphs affordable: a node costs its payload and a bit, with no inline adjacency slack and no per-node allocation, and `benches/graph-store.rs` measures the build, the incremental appends, and the compaction at a million nodes and four million edges. The module documentation states the design and its costs; `benches/graph-store.rs` measures it. `compact(NodeOrder::Insertion)` keeps the nodes in the order they were added; the other `NodeOrder`s — breadth-first or reverse postorder from roots, or the caller's own sequence — number them so the neighbors in a graph built in discovery or file order get nearby identities, while edge identities and every node's adjacency order come out the same under every order. The same benchmark compares the orders on a structured graph whose ids were shuffled.

`Fanout<K, V>`, `DenseFanout<V, K>`, and `SortedMap<K, V>` are that compressed base without the edges. They are built-once tables — from pairs or from grouped values, read-only afterwards — holding every key's run of values end to end in one array, keyed by a sparse ordered key, by a dense `0..bound` index (any `DenseId`, so a tagged `Id<T>` keys one directly), or, in the single-value case, by a sparse key to one value. A reverse index from a symbol to its reference sites, a scope to the rows declared in it, or a file to its flow sites is the same shape as adjacency and should cost the same: three allocations for the whole table rather than one per key, `get` returning a slice, and `iter()` walking keys ascending.

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

`ScopeResolution::compute` resolves a `ScopeQuery`: a stored reference
(`ScopeQuery::from_reference`) or an ephemeral query (`ScopeQuery::new`), which
supports completion and speculative refactoring without modifying the graph.
`ScopeResolutionIndex` resolves all stored references and provides the
reverse definition-to-reference relation used by rename and find-references.
Paths are cycle-free, deterministic, and optionally bounded by a
`PathSearchConfig`; the `PathSearchStats` of a result report which bound
refused work. Scope storage supports `serde` when the feature is enabled. The
store of scopes and edges is `ScopeGraph::graph`, and the scope graph is a
`GraphView` and an `EdgeView`.

`ScopeResolutionIndex::compute` takes a `ScopeIndexConfig`: the
`PathSearchConfig` of each resolution, and `track_reads`, which also records
the scopes each resolution read: the ends of the path prefixes it expanded
(the index is the same either way; the record costs memory and some time).
After the graph is edited in place, `ScopeResolutionIndex::update`
takes a `ScopeGraphChanges` (the scopes whose payload, data, outgoing edges or
edge labels changed, and the references added or edited), re-resolves only the
references that read a changed scope, and patches the reverse index; the result
equals a fresh `compute`, and a `ScopeIndexUpdate` reports how many references
it resolved and which ones changed. This holds when the query's hooks read only
the scopes on the path and the query input, as the `ScopeGraphQuery`
documentation spells out. An index computed without `track_reads` records
nothing, so its first update resolves every reference and records them from
then on.

`ScopeFirstResolution::try_compute` follows the one upward path that a
`ScopeCursorQuery` admits and stops at the first accepted datum. Its hooks see
a `ScopeCursor` (the start, the end, and the hop count of a path) instead of a
stored path, so a walk of `D` scopes costs `O(D)` hook calls and allocates
nothing. `ScopeCursorPolicy` runs the same policy under `ScopeResolution`,
where a cursor policy shadows no candidate.

A `ScopeEnvironmentCache` resolves queries under a `ScopeEnvironmentQuery`,
the memo-safe query form, and `ScopeEnvironmentCache::index` resolves every
reference through one cache. Its hooks see the scope where a path ends and the
path-language state there, never the path or the reference, and a key
projects each datum to what shadowing and lookup compare (typically its name).
The environment of a scope in a state (every visible datum by key, with its
path, after lexicographic label-word shadowing within each key) is then the
same for every reference, so it is computed once and the references of a
block share the walk up their parents and imports: the cache keeps the table
of a memoizing `open_fold_in` whose answers are the environments. On a graph
without cycles each reference sees the data that `compute` shows with the same
hooks. Scopes on a cycle (found once with `SccDecomposition::compute`) are
resolved anew on each reach under a (scope, state) path guard and not cached,
so a path language that comes back to a scope in another state can reach data
that `compute`'s scope guard cuts off. Policies that need the whole path, such
as accessibility, stay `ScopeGraphQuery`s or run afterwards on the stored
candidate paths. The cache trades memory for time: it holds every key of every
environment it computed. An index from the cache records no reads, so its
first `update` resolves every reference.

### Stack graphs

`graph::stack::StackGraph<F, S, N, E>` implements the standard root,
exported/internal scope, symbol push/pop, scoped-symbol push/pop, drop-scopes,
and jump-to-scope nodes. The two singletons come with the graph, and every
other node is added through `StackGraph::add_node` with its `StackNodeKind`.
Symbols, file records, node metadata, and edge metadata are consumer types.
Concrete paths maintain symbol and scope stacks transactionally, scoped symbols
can pause one lookup while another resolves, and edge precedence removes
shadowed complete paths while retaining genuine ambiguity. Both graphs share
one misuse policy: an identity that is not part of the graph panics, and a
broken structural rule, a branch of a linear walk, or a stale summary is an
error.

Every ordinary node belongs to one file partition. Direct cross-file edges are
rejected; independently constructed files compose through the root or exported
scope stitching endpoints. `StackPartialPathSet` extracts reusable structural
routes per file, `StackPartialPathDatabase` replaces one file at a time while
leaving other identities stable, and `StackGraph::clear_file` retires one
changed partition with node/edge tombstones before it is rebuilt. Then
`StackResolution::try_compute_stitched` replays compatible summaries with the
same semantics as direct search, and reports a database that no longer
matches the graph. Every search takes the same `PathSearchConfig`. Graphs and
partial-path databases are serializable with `serde`, so unchanged file
summaries can be persisted and reused. Direct, linear, and stitched
`StackResolutionIndex`es expose reverse definition-to-reference bindings.

### Control-flow graph (`Cfg<I>`)

| Feature | Description |
|---|---|
| Generic `Cfg<I, E = ()>` | Parameterised over any instruction and caller-owned edge payload; unit payload preserves the original API |
| `no_std` + `alloc` | Runs in embedded, kernel, and WASM environments |
| `CfgBuilder` | Builds a CFG from a flat structured instruction stream (`if/else/endif`, `loop/endloop`, `switch/case/endswitch`, `break`, `continue`) |
| Goto wiring | `resolve_jump_edges` wires the targets that the `Flow` of a `FlowEffect::Flow` instruction names (jumps, conditional branches, switches, exceptional paths) to the blocks of `FlowEffect::Label` instructions, with consumer-typed targets (labels, addresses, syntax nodes); `JumpResolution::unresolved` reports the rest as `UnresolvedTransfer`s |
| Compressed adjacency | `Cfg` stores its blocks and edges in `Graph`: a compressed base plus an appendable delta, with no per-block adjacency allocation; `compact(NodeOrder::ReversePostorder(..))` can renumber the blocks in reverse postorder from the entry |
| Edge removal | `remove_edge()` clears the liveness bit of the edge; existing `EdgeId`s stay stable until `compact` renumbers the survivors |
| Edge metadata | Stable `EdgeId`, endpoints, `EdgeKind`, an optional branch `Probability` (`numerator / 2^31`, printed as the shortest decimal that reads back as it), and a caller-owned payload; `ExceptionFlow<M>` adds the search/unwind phase and execute/search/continue disposition the `EdgeKind` does not carry, while retaining platform metadata |
| Regions | Try/catch/finally regions with explicit `HandlerBody::Known`/`Unknown` completeness and `HandlerKind` (Catch, CatchAll, Finally, Fault, Filter — a CLR filter or an SEH `__except`); a region's id is its position (`add_region` returns it, `regions()` yields it); `handlers()`/`handlers_at(block)` enumerate handlers; `RegionIndex` answers the innermost protecting region; one `RegionError` from the fallible mutators and `check_regions` |
| Cleanup continuations | `add_continuation` / `set_cleanup_exit` → `Cleanup` (`Continuation` + `CompletionReason`): which route out of a single shared `finally` block a continuation edge belongs to |
| Handler metadata | `HandlerMetadata<M>` side table — consumer-typed filter predicates and CLR caught-type tokens (`set_all` after `add_region`) keyed by `HandlerRef`, with no type parameter on `Cfg` |
| Subgraph extraction | `subgraph_mapped()` with dense O(1) block-id remapping, payload cloning, and the old-to-new `Rewrite` |
| Block splitting | `split_block()` (one boundary, caller-chosen inserted edge) and `split_block_at_points()` (validated multi-point splitting that also returns the `Rewrite`), with automatic stable edge transfer |
| `serde` feature | Optional serialization support |
| Leader-based construction | `build_address_cfg` with `AddressCfgOptions` (entry, `CallPolicy`, routes, given edges), the `AddressInstruction` / `AddressSpace` traits, `Flow`, and `AddressHandler` with an address-space `AddressHandlerKind` (a filter is named by its address and resolved to its block). Machine/bytecode streams: caller-selected function entry; calls kept as edges or flattened into reported call sites; leaders at targets, after terminators, at exceptional instructions, and at exception boundaries; known successors retained beside explicit unresolved transfers; typed normal, instruction-exceptional, and unwind edges through a consumer payload hook, every kind chosen by `EdgeRole::kind`; nested regions registered enclosing-first with parents wired |
| Exact unwind routes | `AddressCfgOptions::routes` with `UnwindRoute`, then `AddressGraph::add_unwind_routes`: the caller states the ways that the runtime takes when a protected instruction throws (funclets in order, opaque calls, and a catch handler, a resume or a propagation as the end). The graph gets one `EdgeRole::Route` edge for each step of each way, and one instance of a funclet for each distinct way on from it, so a shared funclet never leads into the next step of another way. The result lists the hops with the opaque calls of each way |
| Given normal edges | `AddressCfgOptions::given_edges` with `GivenEdges`: a recovery that already decided the control flow gives the normal edges of each instruction that ends one of its paths, each with its kind and payload. The graph takes no normal edge from the `Flow` of such an instruction, so it can hold a continuation that the flow lacks (the return of a trap) and leave out one that the flow has (the continuation of a call that never returns). The builder does not check a given kind against the flow. Every other instruction continues at its end address, and a missing continuation is an error, not a silent drop. A call without given edges keeps the edges of its flow, so the builder cannot detect a call that never returns and has no given edges. Handler regions and unwind routes work the same with or without given edges |

### IR trait contracts (`ir`)

`cfglib::ir` holds IR traits. It has no IR storage, no builder implementation, no editing code, and no algorithm that reads the operands, values, or effects of instructions. The consumer owns the dense arenas, builds its own records, applies every change, and owns its IR algorithms. The graph algorithms of cfglib read the RTL and MLIL views as graphs and return facts.

A fact is a side table about one state of a function. No algorithm changes the function that it reads. After an edit, the consumer computes the facts again. cfglib stores no fact inside a function.

| Level | Module | The consumer implements | cfglib provides |
|---|---|---|---|
| Low level (LLIL), editable | `ir::llil` | `Instruction` (address, length, bytes and `Flow` of one instruction in the literal form of its instruction set), `Semantics` | `Semantics::capture`, the capture of one record into `effects::Captured` |
| Register transfer (RTL) | `ir::rtl` | `Dialect`, `InstructionBuilder`, `RecordView`, `FunctionView` | Nothing. The facts over an RTL function and the constant evaluation of its operators belong to the consumer. |
| Medium level (MLIL) | `ir::mlil` | `Dialect`, `FunctionView` | Nothing. Value analyses over the view belong to the consumer. |
| Structured (HLIL) | `ir::hlil` | `Dialect`, `FunctionView`, `FunctionBuilder` | Nothing. The walks over the shapes belong to the consumer. |

Every dialect type is plain data that the consumer defines. The traits contain no language, runtime, calling convention, opcode, or source-location type.

cfglib no longer holds IR storage or IR algorithms. The removal covers the RTL, MLIL, and HLIL function types and builder implementations, tail duplication, the shared vocabulary trait, the provenance map, the facts over an RTL function (root table, block summaries, and the rest, now in `il::rtl::facts`), and the value analyses over MLIL (use index, liveness, constant propagation, and dead-instruction marking). A consumer that needs one of them keeps it in its own crate; the `il` crate holds them for its IR.

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
| `Bits`, `BitSlice`, `ConstBits` | `Bits` is the one value type: a width in `1..=Bits::MAX_WIDTH` (8192). `Bits::BOOL` is the one-bit type. A `BitSlice` is a bit range inside one root. A `ConstBits` is a known constant value: little-endian words and a `Bits`. It is the one form of a constant operand that a consumer's executor and evaluator pass; the constant evaluation of the operators belongs to the consumer. |
| `Dialect` | The vocabulary of one instruction set, normally a unit marker type. The associated types are `Root`, `Operator`, `EffectOp`, `WritePolicy`, `Outcome`, and `NondetDomain`. `Dialect::root_width` gives the width of a root. Roots never alias: a register view is a `BitSlice` of one root. `Place<R>` names such a range of a root of type `R`, and `Place::whole::<D>(root)` covers the whole root. |
| `InstructionBuilder<D>` | The write-only protocol through which instruction semantics emit one record into consumer storage. The producer calls `read`, `constant`, `apply`, `nondet`, `effect`, `raise`, and `write` in statement order, then `finish` with a `Control`. `finish` seals the record. The protocol cannot read a statement back or change it. |
| `RecordView<D>` | One sealed record: its `statements`, its `control`, and the `width` of each value. A sealed record is read-only. |
| `FunctionView<D>` | A `RootedView` over the decoded machine control flow. The root is the entry block. It adds `inst_bound`, `value_bound`, the `instructions` of a block, and the `record` of an instruction. |

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

#### Structured contract (`ir::hlil`)

An HLIL function is a pair of trees over declared variables:

- Statements form trees: assignments, `if`, `while`, `do`/`while`, `loop`, `for`, `switch`, `break` and `continue` (optionally labeled), `return`, labeled groups with `goto`, `try` with handlers and a `finally` body, and a dialect `Region` that wraps a body.
- Values are typed expression trees of variable reads, dialect constants, and dialect operations.
- The statement and expression shapes (`StmtShape`, `ExprShape`) are universal and belong to the library. A dialect supplies the operations, constants, value types, and variable metadata.
- A variable holds differently typed values at different points, so each occurrence carries its own type.

| Item | Description |
|---|---|
| `Dialect`, `FunctionView<D>` | The vocabulary of one structured dialect, and the read-only view that lends statements, expressions, switch arms, and handlers as slices from the consumer's dense storage. |
| `FunctionBuilder<D>` | The write-only protocol that builds a function, children before parents. The consumer seals the function in `finish`. |
| `Entity`, `StmtShape`, `ExprShape`, `SwitchArm`, `Handler`, `HandlerKind` | The shapes that the view lends, and the name of one statement, expression, or variable. |

A function follows these rules:

- Every statement and expression has exactly one parent, and a child has a lower index than its parent.
- A `continue` with a label targets a `Labeled` group whose body is exactly one loop, and a `break` or `goto` can target any label.
- Control that completes must end in a `return` or iterate forever.
- A region's operands are evaluated once, on entry, and the end of the region releases the values they had then.

The algorithms over this contract (lift from a flat function, lowering, structure recovery, verification, pseudocode, and declaration placement) belong to the consumers of the contract, not to cfglib.

### Graph algorithms

| Algorithm | Function / Type | Description |
|---|---|---|
| DFS / BFS | `depth_first_preorder`, `depth_first_postorder`, `reverse_postorder`, `breadth_first`, and matching `Cfg` methods | Traversals over any `GraphView` along a `Direction`; the crate's walks run on one breadth-first and one depth-first core, apart from a few hot loops that measured faster on their own |
| Graph views | `Rooted`, `Reversed`, `Induced`, `VirtualRoot` | Reshape a view without copying it: root it at a node, reverse every edge, confine it to a node set, or put one extra node in front of a list of roots (caller order, optional membership mask for constant-time reverse adjacency). Compose them, e.g. a virtual exit over `Reversed` |
| Edge-aware traversal | `breadth_first_edges`, `depth_first_edges`, `shortest_path_edges[_in]` | Every distinct edge once, as edge ids, over any edge view; parallel edges stay distinct. Restrict the edges a walk may take with `FilteredEdges` |
| Configurable search | `search` + `SearchConfig` (order, visited policy, depth bound) along a `Direction` | First-match, pruning (`Visit::Skip`), early exit (`ControlFlow::Break`), and backtracking as configuration; `VisitedPolicy::Path` un-marks on unwind so every route to a node is reported |
| Reusable search scratch | `search_in` + `SearchScratch` | The same search with the visited marks *and* the call's own buffers caller-owned, for a pass that searches once per root over one node space; the scratch starts empty, grows on entry, and resets its marks by an epoch bump |
| Reusable visited marks | `EpochMarks` (`mark`, `is_marked`, `reset`, `grow_to`) | An epoch-stamped visited buffer: clearing is a bump of the current epoch, so a hand-written walk per root marks and resets one buffer in O(1) |
| Disjoint sets | `DisjointSet` (`union` by rank, `union_toward_min` for deterministic minimum representatives, growable via `push`, `reset` to reuse the storage) | A union-find, usable directly by consumers |
| Bidirected Dyck reachability | `bidirected_dyck_classes` → `BidirectedDyck` (`insert`, `insert_plain`, `same_class`, `class_of`) | Classes of nodes joined by balanced-parenthesis paths when every labeled edge `u --(k--> v` implies `v --)k--> u`: field-sensitive alias and data-dependence analysis. Union-find with one source per label and class (Chatterjee et al. 2018), O((m + n) log n) for a constant alphabet; edges can be inserted after the build, and the minimum member represents each class |
| Open post-order fold | `open_fold` + `OpenFold` trait (`FoldEnter::Leaf`/`Fold`, `MarkScope::Shared`/`Isolated`) + `OpenFoldConfig` (`max_depth`, `revisit`: `Revisit::Refuse`/`Memoize`); `open_fold_in` + `OpenFoldMemo` | Child-answers-to-parent folding over a lazily discovered space: caller-keyed cycle guard (nodes need no `Ord`), per-node mark scoping (exact per-path marking or persistent pruning), early exit, depth bound — C++-style member lookup and type-algebra evaluators without hand-written frame stacks. `Revisit::Memoize` is the third choice for a node reached again: it folds each mark once and hands the finished answer to every later route, so a diamond's base reaches the fold along both routes at the cost of one fold, provided a node's answer does not depend on the route to it (answers that absorbed a refusal of a still-open node on a cycle are not cached, nor are those of nodes whose `OpenFold::memoize` hook says no). An answer the depth bound cut short is not stored, so a stored answer is whole and serves every later route at any depth. `open_fold_in` keeps the finished answers in a caller-owned table across calls, whatever bounds they use |
| Open fixpoint fold | `open_fold` with `Revisit::Iterate { max_rounds }` → `OpenFoldError::RoundLimitExceeded`/`UnmarkedCycle` | The same `OpenFold` over a cyclic space whose answers travel around cycles (names exported through `export *` cycles, members visible through cyclic includes): one answer per mark, cycles of marked nodes found with Tarjan lowlinks during the walk and re-folded until a round changes nothing, so monotone hooks reach the least fixpoint from any root; on a space without cycles it walks as `Revisit::Memoize` does |
| Forward-only view support | `scan_predecessors` | The honest O(nodes × edges) `predecessors` for views that store only successors, replacing per-consumer scan stubs |
| Traversal events | `depth_first_events` → `DfsEvent` | Depth-first discover/tree/back/forward-or-cross/finish events in pinned order |
| Open-graph search | `open_search` + `SearchConfig` | The same disciplines over a lazily discovered node space: successors come from a closure, no dense ids (import/re-export chases, ordered emission walks) |
| Open-graph events | `open_breadth_first_events` → `OpenBfsEvent`; `open_depth_first_events` → `OpenDfsEvent` | Breadth-first discovery/refusal events or depth-first discover/finish/refusal events over a lazily discovered node space; path policy can revisit a shared node once per route |
| Open-graph route enumeration | `open_breadth_first_paths` → `OpenPathsEvent` | Simple routes shortest-first over a lazily discovered space — no global marks, per-route cycle guard; exponential worst case, so bound with depth, `Skip`, or `Break` |
| Alias chase | `follow`, `follow_path` | Out-degree ≤ 1 chase with a hop bound and a full-path cycle guard; the chain, or just its end |
| Shortest path | `shortest_path`, `shortest_path_in` (nodes), `shortest_path_edges`, `shortest_path_edges_in` (edge witness) + `ShortestPathScratch` | Unweighted witness path from two searches, one from each end, alternating one edge examination per side and stopping once the best join is no longer than the sum of the two search depths. The `_in` forms keep the tables in a caller-owned, epoch-stamped scratch, so a repeated point query costs O(edges examined) with no per-call O(n) setup. Among equally short paths the witness is the first join the alternation finds, in adjacency order. The backward search reads `predecessors` (`incoming`), so give a forward-only view real reverse adjacency first |
| Minimum-label relaxation | `min_label_relaxation` | Edge-defined label transfer to a minimum fixpoint over any edge view, forward or backward, expanded smallest label first; nodes re-expand when their label improves, and with extensive transfers (nonnegative costs) each node expands once, O((m + s) log(m + s)) |
| Path expressions | `path_expressions` → `PathExpressions` (`to`, `arena` → `PathExprArena` of `PathExprNode`), `interpret` + `PathAlgebra`, `GenKillAlgebra` | Tarjan's single-source path expressions: for each node, a hash-consed regular expression over edge ids denoting exactly the paths from the root, unambiguous so non-idempotent algebras (path counting) work. Bottom-up over the dominator tree with path compression; reducible views need no elimination, and each strongly connected sibling group of an irreducible view costs Gaussian elimination, O(r³). `interpret` evaluates every shared subexpression once in a consumer semiring with star; `GenKillAlgebra` is the forward union-meet bit-vector algebra, equal to the node fixpoint |
| Multi-source reachability | `reachable` | Dense `Vec<bool>` from a seed set, forward or reverse; order-insensitive; confine it to a region with `Induced` |
| Pairwise reachability index | `ReachIndex::compute` + `ReachIndexConfig`, `reaches(source, target)`, `reaches_in(&mut ReachScratch, source, target)` | For repeated single-pair queries on one static graph only (set closures stay with `reachable`): over the sources-first condensation, O'Reach-style observations (weak components, topological levels, extended DFS orders, up to 64 supportive components) answer most pairs in O(1), and a pruned bidirectional search the rest; every answer is exact. An opt-in chain-cover mode answers every pair with one table read when the condensation splits into few chains. Deterministic; invalid after an edit |
| Nearest common ancestor | `nearest_common_ancestor` | Bidirectional BFS meet of two nodes that stops once nothing unseen can beat the best meet; smallest combined distance, ties by smallest node id |
| All common ancestors | `common_ancestors` → `Vec<CommonAncestor>` | Every shared node with both hop counts, depth-bounded, in `b`'s BFS discovery order — the consumer-rankable form (MRO linearization, overload preference) |
| Horn-clause derivability | `HornClauses`; `saturate` → `HornSaturation` (`update`, `is_derived`, `derived`), `grow_facts` | AND-OR closure (`head <- b1 & b2`): nullability, all-arguments-constant, all-callers-dead. `derivable` solves from scratch into a `DenseBits` (a saturation that keeps no state); a `HornSaturation` keeps its counters, so clauses (and facts) added in batches cost O(S + F) over every `update` together instead of O(S + F) per batch |
| Topological sort | `topological_sort`, `topological_sort_or_cycle` | Stable ordering or cycle detection; `topological_sort_or_cycle` returns the same order, or one cycle as the witness (successor order, from its smallest node) in the same O(n + m) pass |
| Dominator tree | `DominatorTree::compute` (rooted views; another root through `Rooted`, the reverse graph through `Reversed`) | Simple Lengauer-Tarjan over any graph view, O(m log n) with explicit stacks. The tree is numbered once in preorder, so `dominates` is O(1) and `subtree`, `preorder`, `level`, `children`, `roots`, `nearest_common_dominator` (of two nodes, `None` when either is unreachable) and `nearest_common_dominator_among` (of a set, skipping its unreachable nodes) need no extra tables. Edge dominance (every path to a node takes one edge) is O(indegree) from the same tree: `edge_dominates`, `dominating_in_edge` (the bridge into a node), and `edge_id_dominates` for views with stable edge ids, where parallel edges are distinct |
| Reusable dominator scratch | `DominatorTree::compute_in` + `DominatorScratch` | The same tree with the call's working buffers caller-owned, for a pass computing one tree per procedure over a codebase; the scratch holds dense indices, so one instance serves every graph and node-id type, and every `_in` function of the dominator family (certified, incremental, forest, post-dominator, iterated frontiers) takes the same one |
| Dominator forest | `DominatorTree::complete_forest_in` + `DominatorScratch` | Extends a tree over the code its root misses: each source component of unreachable code roots a tree of its own, and every frontier query reads the forest like a tree. `None`, at the cost of one scan, when the root reaches every node |
| Dominator certificate | `DominatorTree::compute_certified_in` → `(DominatorTree, LowHighOrder)`, `verify_certificate` → `Result<(), CertificateError>`; the same pair on `PostDominatorTree` | A low-high order (Georgiadis-Tarjan) built in linear time from the construction's semidominators, and a checker that accepts a tree exactly when it is the dominator tree: the parent property plus the low-high property, one O(n + m) pass with the tree's interval test. A tree that is too shallow, which the parent property alone accepts, is rejected with any certificate. About twice a plain construction to build, about one to check; debug builds check every tree they build |
| Post-dominator tree | `PostDominatorTree::compute` (rooted views) → `PostDominatorTree<N>`, `PostNode` | Over a virtual exit that every exit joins. Exit-free regions are cut at the blocks closing each loop, so every reachable block has facts independent of successor order. `post_dominates` is O(1); `ipdom`, `nearest_common_post_dominator`, `nearest_common_post_dominator_among`, `exits`, `cut_roots`, `reaches_exit`, `subtree`, `children`, `frontier` (the control-dependence sources of a block), `control_dependents` (the blocks control dependent on one arm), and `edge_post_dominates` (every path from a block to the exit takes one edge) answer from the one numbered tree |
| Post-dominator sink classes | `PostDominatorTree::compute_in(scratch, graph, policy)` + `ExitPolicy` (`CutLoops`, what `compute` builds, or `SinkComponents`); `sink_class` → `Option<SinkId>`, `sink_members` | The other standard reading of an exit-free region: each sink component of the blocks that reach no exit (a strongly connected component no edge leaves) is one class joined to the virtual exit, whose blocks post-dominate each other; the sink post-dominance of Hecker et al. (TOPLAS 2021), checked against its fixpoint definition on every small graph. A class is one node named by its lowest block, so `post_dominates` stays O(1); `control_dependents`, `frontier`, `iterated_frontier` and `control_dependence_graph` skip the arms inside a class and give nontermination-insensitive control dependence. Same O(m log n) construction plus O(n + m) for the sinks; on graphs where every block reaches an exit the tree is `==` to the default. The scratch is the one `DominatorScratch` |
| Dominance frontiers | `DominanceFrontiers::compute(graph, tree)`, `recompute` (into the same storage) for a `DominatorTree` or a `PostDominatorTree` (`FrontierTree`); `frontier` → `&[N]`; `iterated_frontier`, `iterated_frontier_in` + `DominatorScratch` | Every block's frontier in one flat run table, read back as a slice in ascending dense index, the order of the per-query `frontier` of either tree. Over a post-dominator tree the frontiers are the control-dependence sources. `iterated_frontier` closes a set of definitions under the table, with an optional live-in filter, for SSA φ-placement with many queries per graph; `DominatorTree::iterated_frontier` and `PostDominatorTree::iterated_frontier` answer the same query without a table, for a few queries |
| Dominator updates | `DominatorTree::apply_updates_in` + `DominatorUpdate` (in place, from named edits) | Brings a tree up to date after a batch of edge insertions and deletions, reporting the nodes whose idom changed (Georgiadis et al.'s depth-based search for insertions, a subtree recomputation for deletions). An edit that changes the tree costs O(n) to renumber it plus what it touches, so one insertion runs about 2× faster than a full rebuild and diff at 4K nodes and about 90× at 2²⁰. A batch is finished with one full construction once replaying it stops paying: more than 64 edits, a subtree past a quarter of the tree, or more renumbering than one pass per 2¹⁴ nodes |
| Strongly connected components | `SccDecomposition::compute(graph, SccNumbering)` → `SccDecomposition<N>` | `LeavesFirst`: generic iterative Tarjan algorithm in Pearce's single-array form, reverse-topological order; `SourcesFirst`: the same partition numbered sources first (`index(u) <= index(v)` across every edge), the classic deterministic two-pass Kosaraju sequence, for budgeted forward closures over the condensation. Components are ascending slices of one flat array (`components()`, `component(node)` → `Scc` with `nodes()`, `contains`, `is_singleton`, `is_cyclic`); `cycle_through(graph, node)` → a shortest cycle through a node, searched inside its component only; removed node slots belong to no component, and querying one panics |
| SCC under edge edits | `IncrementalScc::compute(graph)`; `insert_edge`, `insert_edges`, `remove_edge` → `SccChange` (`merged`, `split`, `reordered`); `component_id`, `position`, `decomposition` | Components and a sources-first order kept current as the caller adds and removes edges, the graph being the only adjacency. An edge that agrees with the order costs O(1); one against it walks only the nodes its target reaches between its ends, merges any cycle it closed, and shifts that window; a batch takes one walk; removing an edge inside a component re-splits that component alone, in place. Ids are each component's smallest node; `decomposition()` keeps the sources-first contract but may number differently from a fresh one. The node set is fixed at `compute` |
| Condensation | `SccDecomposition::condensation(graph)` → `Graph<(), ()>` | The component DAG whose node index **is** the decomposition's component index, either numbering's; deduplicated edges, and in-degrees plus dependents straight off the graph (the one-pass fixpoint shape) |
| Strong articulation points and bridges | `StrongCuts::compute(graph, component)`, `compute_in(&mut StrongCutScratch, graph, component)` → `StrongCuts<N>` | The nodes and edges one strongly connected component (a slice of `components()`) needs to stay strongly connected, read off two dominator trees (of it and of its reverse, from its lowest node) plus two searches, over a view of that component alone: O(m log n). A parallel edge is never a strong bridge; both lists are sorted by dense index |
| Back-edge detection | `find_back_edges` (any edge view) | Edges whose target dominates their source, and edges tagged `Back` that close a cycle dominance cannot prove; out of code the root reaches only, so unreachable code has no back-edges. The `()` edges of a plain graph carry no tag |
| Feedback arc set | `linear_arrangement` → `Vec<N>`; `feedback_arc_set(graph, FeedbackConfig)` → `Vec<EdgeId>` | An acyclic view of any graph, no root or dominators needed: components sources first, each cyclic one ordered by Eades–Lin–Smyth in O(n + m); the set is every edge pointing backward in that order plus self-loops, so removing it leaves the graph acyclic. `minimal` re-admits each edge whose return closes no cycle. Deterministic; small, not minimum (NP-hard) |
| Loop forest | `LoopForest::compute` / `compute_cycles` → `loop_of_header`, `parent`, `depth`, `latches`, `entries`, `exits`, `is_irreducible`, `contains`, `encloses`, `innermost`, `enclosing`, `is_nested`, `body`, `body_count`, `loop_count`, `node_depths` | One loop per header of a `find_back_edges` edge, in near-linear space and time, whatever the nesting: every block is stored once under its innermost loop, and `contains` and `encloses` are constant time for every loop dominance finds. A tagged loop whose header does not dominate a latch (an irreducible cycle) keeps an explicit set of blocks, costs its own size, and is answered from that set; `encloses` and `enclosing` are exact where `parent` holds only the header. `compute_cycles` needs no tag: it builds Havlak's cycle nesting forest of the code the root reaches (Havlak–Tarjan with Ramalingam's bound, O(m α(n))), in which every cycle, irreducible or not, is properly nested and headed by the entry a depth-first walk in adjacency order visits first; on a reducible graph it holds the natural loops. `entries` lists a loop's header and its other entries. `GraphMetrics` and `CfgMetrics` read the forest directly |
| Loop preheaders | `insert_preheader`, `canonicalize_loops` → `Vec<CanonicalLoop>` | A dedicated block before a loop header that takes over the header's entering edges in place; canonicalization gives every loop of `LoopForest::compute` one, outermost first, and reports the exits once every preheader is in |
| Weak topological order | `WeakTopologicalOrder::compute` over a rooted view (`Rooted`, and `Reversed` under it for a backward order) → `order`, `rank`, `is_head`, `heads`, `head_of` | Bourdoncle's hierarchical order of the nodes the root reaches: nested components, each contiguous and head first, so every edge runs forward or closes a loop onto the head of a component that contains its source, and the heads cut every cycle. The components are the cycles of `LoopForest::compute_cycles`, from the same construction: one depth-first walk and a union-find pass, O(m α(n)); the stored adjacency order decides ties |
| Coldness classes | `ColdnessClasses::compute` over a rooted edge view with `seed`, `cap`, and `ends` closures → `class`, `edge` (`ColdClass`: `Never < Rare < Normal`) | The greatest assignment of three classes under a seed per node, a cap per edge, and the rule that a node is no hotter than the hottest edge it can take or that can reach it. O(n + m) with per-node counters and an explicit queue, seeded in reverse postorder from the root; independent of node and edge order |
| Block frequencies | `BlockFrequencies::compute` over a rooted edge view and its `LoopForest` → `block`, `edge`; edge `Probability`, `FrequencyConfig` (`max_loop_scale`, default 4096), `FREQUENCY_SHIFT` | Wu–Larus static profile: execution counts relative to one entry from branch probabilities (over a CFG, `\|edge\| cfg.edge(edge).weight()`). Loops are propagated innermost first, each scaled by `1 / (1 - cp)` capped at `max_loop_scale`; missing probabilities share what the stated ones leave, and each source is renormalized. Integer fixed point with `u128` intermediates, so results are identical on every platform; saturates instead of overflowing. O(n + m) after the loop forest when exits leave one loop at a time. Exact up to rounding on reducible graphs; irreducible cycles are approximated as documented |
| Network flow | `FlowNetwork` (`add_node`, `add_arc` → `ArcId`, `max_flow`, `min_cost_flow` → `FlowSolution`, `flow`), `FlowError` | A caller-built integer network: `i64` capacities and costs, parallel arcs, self-loops and negative costs, residual capacities kept per half arc with no copied graph. `max_flow` is Dinic (BFS levels, blocking flows by an explicit-stack walk with current-arc pointers), O(V²E) and O(E·min(√E, V^(2/3))) at unit capacities; the value saturates at `i64::MAX`. `min_cost_flow` finds a flow of value `min(limit, maximum)` of least cost, cycles included (limit `0` is a minimum-cost circulation), by capacity-scaled successive shortest paths: Dijkstra over Johnson potentials, stopped at the first node that can absorb, with negative-cost arcs saturated as the scale reaches them. Scans follow arc insertion order and ties go to the smaller node index; integer-only, and an overflow is a `FlowError`, not a wrap |
| Minimum cuts and Menger paths | `EdgeCut::compute`, `VertexCut::compute`, `edge_disjoint_paths` | Unit-capacity separators between a set of sources and a set of sinks over any view (a virtual super-source and super-sink): the fewest edges, or the fewest other nodes, whose removal leaves no source-to-sink path, and as many edge-disjoint paths as the edge cut has edges. Dinic on the shared `FlowNetwork`, with nodes split into `in`/`out` for vertex cuts: O(m·min(√m, n^(2/3))) for edges, O(√n·m) for vertices; `limit: Some(k)` stops after `k + 1` paths, O(k·m). The canonical cut is the one closest to the sources (the residual-reachable set), the same for every maximum flow and adjacency order; parallel edges count separately. `None` when a source is a sink, or for vertex cuts adjacent to one |
| Profile inference | `BlockFrequencies::infer` + `InferenceCosts` (`increase`, `decrease`, default `1`) → `block`, `edge` | Flow-consistent whole block and edge counts from noisy, partial estimates (He et al., POPL 2022; edge estimates as in Ayupov et al., CC 2024), in the same table type as the static frequencies: a minimum-cost circulation over split blocks where each unit above or below an estimate costs `increase` or `decrease`. Among the circulations of least cost it returns the one closest to the expected counts of `BlockFrequencies::compute` under the branch priors, exactly (a second solve over the arcs that the first solve's potentials leave free), so unsampled successors share evenly. Counts conserve flow at every block that is neither the entry nor an exit |
| Control dependence graph | `control_dependence_graph` → `Graph<N, ()>`; `nontermination_sensitive_controllers` → `Vec<N>`, `nontermination_sensitive_controllers_in` + `NonterminationScratch`; `weak_control_closure` → `Vec<N>`, `weak_control_closure_in` + `ControlClosureScratch` | From post-dominator tree, over any view; a block that branches back to itself has the edge to itself. Exit-free regions follow the tree's `ExitPolicy`: relative to the cut, or nontermination-insensitive under `SinkComponents`. `nontermination_sensitive_controllers` gives the nontermination-sensitive reading of one block without a tree (Chalupa et al., CAV 2021): the branches with one arm that must reach the block and one that may avoid it, by leaving or by looping forever, in O(n + m) per block; the whole relation can be quadratic, as on a ladder, so there is no builder of it. `weak_control_closure` closes a set of blocks for slicing without a tree (Danicic et al., TCS 2011): it adds every block the set reaches that chooses which block of the set comes next, by lDFS (Léchenet et al., TCS 2023) in O(n·m), O(n²) for bounded out-degree. With the entry and every exit among the seeds of a graph whose blocks all reach an exit, it is the seeds with their iterated post-dominance frontier. The `_in` forms reuse a caller-owned scratch across calls |
| SESE regions | `ProgramStructureTree::compute` → `region_count`, `regions`, `entry_edge`, `exit_edge`, `parent`, `innermost`, `contains`, `blocks` (`SeseRegionId`, `SeseEdge`); `CycleEquivalence::compute` → `class` | The canonical single-entry single-exit regions nested into the program structure tree of Johnson, Pearson and Pingali (PLDI 1994), from cycle equivalence by one undirected DFS with bracket lists: O(n + m) on explicit stacks, irreducible graphs included. The graph is closed through a virtual exit with the post-dominator cut, so a boundary is a graph edge, the virtual edge into the root, or a block's virtual edge to the exit. A region's blocks are one slice and `contains` is O(1). These are not the exception regions of `region` |
| Reducibility | `is_reducible` (exact test over a dominator tree), `make_reducible` (CFG node splitting → the copies), `make_reducible_by_dispatch` (dispatch hubs, `DispatchBuilder` → `DispatchReport`) | Both work over the cycles of `LoopForest::compute_cycles`, innermost first. Splitting copies what an irreducible cycle's other entries lead to in front of the entry it keeps, which can take exponentially many copies. Dispatch copies nothing: every edge into an entry of an irreducible cycle goes through a new selector block to one new hub, which branches to the entries by `SwitchCase` edges and becomes the cycle's header. The consumer's `DispatchBuilder` writes all code: `select` sets the hub's own selector value and ends in the jump to the hub, `dispatch` branches on it, and `case`/`jump` give the payloads of the new edges. Each irreducible cycle adds one hub and one selector per redirected edge, so the growth is at most the edges times the irreducible nesting depth |
| Tree decomposition | `TreeDecomposition::compute` + `EliminationHeuristic` (`MinDegree`, `MinFill`) → `bag_count`, `bag`, `parent`, `bag_of`, `width` | Greedy elimination over the undirected simple graph underlying any view (directions, parallel edges and self-loops ignored), for graphs with gotos or irreducible loops that `SplDecomposition::compute` refuses. One bag per node in elimination order, a forest over components, ties by smallest dense index, independent of adjacency order. A heuristic with no width guarantee: on the chain, diamond, nested-branch and nested-loop test fixtures it finds width 1 or 2 |
| Call graph | `call_graph` over `FlowControl` instructions (callees from `Flow::Call` and `Flow::TailCall`), `call_graph_indexed` and `FunctionIndex` (key → node), `recursive_functions` (every function from one SCC pass), `propagate_summaries` + `SummaryProblem` + `SummaryConfig` (`widening`, `narrowing`) | Consumer-typed callees; interprocedural summary scaffold over any graph view. `find_function` and `is_recursive_function` answer one query per call; the batch forms answer all of them at once. The summary propagation keeps its own callee-first SCC schedule: inside a recursive component, a worklist recomputes only the callers of a changed summary. With `widening` it widens at a feedback vertex set of the component (the callers that read a callee not yet computed in the callee-first order), so summaries of infinite height converge; with `narrowing` it then recomputes each recursive component, narrowing at the same points, before its callers read it, which recovers the precision the widening gave up. The hooks have the shape of the dataflow problems' `widen` and `narrow` |
| CFG diff | `CfgDiff::compute`; `CascadeDiff::compute` + `BlockHashes` → `diff`, `levels`, `match_level` (`MatchLevel`) | Structural comparison (bindiff-style): blocks are matched by wiring, from the entries along edges of equal kind, over every live block; a matched block reports changed instruction counts and changed edges. Any edge payload, no trait bounds. `CascadeDiff` also matches by content through two hashes the consumer computes per block (`strict`: instructions and operands in order; `loose`: survives operand changes), for example with `Mixer`. It pairs the entries, then by `full` (strict plus one WL round over the neighbors' loose hashes), `strict`, `loose` and shape, each within a bucket to the nearest relative reverse-postorder position, running the walk after each key; each pair reports the first key it shares, or `Walk` / `Shape` when it shares none |
| CFG diff of split and merged blocks | `RegionDiff::compute` → `regions` (`RegionMatch` with its member `pairs`), `left_only`, `right_only` | n:m correspondences: each CFG is cut, read-only, into maximal chains (each link the only out-edge of its source and the only in-edge of its target, a fallthrough, unconditional, or jump edge; never through an exceptional edge or into the entry). The chains are matched by wiring like `CfgDiff::compute`, and each matched pair is aligned by cumulative instruction count into the smallest regions no correspondence crosses: a split block is a 1:2 region, a merge a 2:1 one, unchanged blocks 1:1. Splits across branches are not covered |
| Structural fingerprints | `wl_colors`, `multiset_fingerprint`, `canonical_order`, `canonical_hash`, `Mixer` | Deterministic 64-bit structure hashes from one fixed SplitMix64 mixer (`Mixer`, public so consumers hash their content the same way), the same across runs, platforms, and versions. `wl_colors` refines caller-supplied node labels by the labeled multisets of in- and out-neighbors (1-WL), O(n + m log d) a round; `multiset_fingerprint` folds colors whatever their numbering; `canonical_order` numbers the nodes the root reaches by a preorder over out-edges sorted by a caller key, and `canonical_hash` hashes the graph in that order, counting unreached nodes by their WL colors. Labels and keys come from the caller, so content stays the consumer's. Equal fingerprints are evidence, not proof, of isomorphism |
| Exception handling model | `EhModel::compute` | Payload-generic CFG input; block roles (landing pad, cleanup, catch dispatch, resume point) with `blocks_of(kind)`, the exception-control `EdgeId`s, and the protected-by mapping; handlers and cleanup records stay on the `Cfg` (`handlers_at`, `cleanup`) |
| Integrity verification | `verify`, `verify_view`, `verify_edge_view`; `verify_semantics` + `SemanticValidator`; `VerifyReport`, `VerifyError` | Structural node/edge-view checks plus deterministic typed consumer hooks for cardinality, ordering, and edge-payload rules. One `VerifyReport` holds the structural errors and the consumer's typed errors |
| DOT export | `write_dot` / `to_dot` + `DotStyle` (any node- and edge-bearing view); `Cfg::write_dot` (caller label hook), `Cfg::to_dot` (`DisplayInstr`), `Graph::to_dot` | One writer; the style supplies node labels, edge attributes, graph name, and layout direction. Labels are escaped and left-justified; the graph name, node prefix, color and style are quoted when DOT requires it, and a NaN or infinite pen width is not written |
| Text form | `write_text` / `to_text` + `TextStyle`, `parse_text`; `Graph::write_text` / `Graph::to_text`. The CFG's own listing is a different form: `Cfg::write_listing` / `Cfg::to_listing`, `parse_cfg_listing` | One line per node and per edge, ids are the dense indices. Round trips: write → parse → write is a fixed point, so two graphs compare as text |

### Dataflow framework

Instruction-level analyses read the graph through `CfgView`. The graph-shaped solvers read a rooted, node, or edge view instead.

#### Instruction-level view (`CfgView`)

`CfgView` is the one read-only trait that the instruction-level algorithms of a consumer, and the CFG analyses of cfglib, take. It joins the graph views with the facts that these algorithms need and a graph view does not give: the ordered instructions of each block, the `EdgeKind` of each edge, and the optional branch weight of each edge.

| Part | Description |
|---|---|
| Supertraits | `RootedView<NodeId = BlockId>` and `EdgeView<EdgeId = EdgeId>`. The edge data implements `KindedEdge`. The root is the entry block. |
| `Inst`, `instructions` | The instruction type of the blocks, and the instructions of one block in program order. |
| `edge_weight` | The branch `Probability` of an edge, relative to the other edges leaving its source. The default is `None`, which means that the view has no profile. |

`Cfg` implements `CfgView`, so every caller that holds a `Cfg` keeps working. A consumer with its own block and edge storage implements the trait and runs the same analyses without a conversion. The unit tests of the trait run the block fixpoint solver on a vector-backed view and compare the answer with the one on a `Cfg`.

The contract has two rules. `instructions` accepts every identity that `node_ids` yields, and it returns the same slice each time. The index of an instruction in that slice is the instruction index of the `ProgramPoint` that an analysis reports.

The view has no separate exception model. An edge of kind `EdgeKind::ExceptionUnwind` leaves its source block before a throwing instruction. `InstrInfo::may_unwind` marks the throwing instructions of a block.

These analyses take `&G` where `G: CfgView`: `CfgMetrics`, `Pattern::orient_diamonds` and `empty_trampolines`, and every per-block `NodeProblem` solved by `solve_node_problem`. A transform that edits the graph still takes `&mut Cfg`. Live variables, reaching definitions, SSA construction, memory SSA, and memory value flow are in the `il` crate.

| Analysis | Function / Type | Description |
|---|---|---|
| Fixpoint solvers | `solve_node_problem` (`NodeProblem` / `TryNodeProblem`), `solve_edge_problem` (`EdgeProblem` / `TryEdgeProblem`), each taking one `Solve { seeds, config }`; `Facts`, `EdgeFacts`, `SolveStats` (`step_count`, `round_count`, `peak_count`) | Forward or backward, any lattice type, over any graph view (per-block problems of a CFG, taint or reachability-with-facts over a value-flow graph) or edge view (per-edge transfer with the node's physical pre and post states). `Seeds::Live` queues every live node, `Seeds::Reachable(root)` what the root reaches with the root given the boundary fact (the rooted solve of a CFG), `Seeds::Of` a dirty region. Every infallible problem is a fallible one with `Error = Infallible`, so one entry point serves both; consumer errors stay separate from the deterministic step limit of `SolveConfig` (`TrySolveError`, `SolveError`). Node facts come in one program-order `Facts` type (a backward node's IN is what its transfer produced); every problem names its starting fact `boundary`; `meet_options` and `try_meet_options` lift a lattice to `Option<F>` with `None` as unreachable bottom |
| Iteration order and widening | `SolveConfig { max_steps, order, narrowing_passes }`, `IterationOrder` (`Index`, `ReversePostorder`, `WeakTopological`), `widen` / `narrow` on every problem trait | Every solver pops its worklist by dense index unless the config picks reverse postorder or a weak topological order, walked from the solve's queued nodes along the analysis direction; the facts of a finite-height problem do not change, the steps do. In the weak topological order the solver widens the fact flowing into each head, so intervals and other infinite-height lattices terminate, and narrows each outermost loop in bounded descending passes as soon as it has converged, before the code after it runs; the defaults of `widen` and `narrow` change nothing |
| Streamed solve | `stream_node_problem`, `stream_edge_problem` → `SolveStats` | The same solve, from the same `Solve`, in the weak topological order whatever the config names, handing each node's IN and OUT facts to a visitor once, as soon as the outermost loop that holds it is final, then freeing them: a node's IN after its visit, its OUT once its last reader's loop is final, an edge's fact once its target is visited. Peak memory follows the widest loops instead of the graph's size; for checks, diagnostics, or a projection into the consumer's own table |
| Resumed solve | `resume_node_problem`, `resume_edge_problem` + `Resume { changed, mode }`, `ResumeMode` (`Reset`, `Grow`) | Re-solve after an edit from the previous `Facts` / `EdgeFacts`, reused in place, given the nodes the edit changed and the `Solve` of the solve it continues: `Reset` resets and re-solves only what they reach along the analysis direction and equals a full solve of the edited graph; `Grow` climbs from the previous facts when the edit can only add facts. Slots follow the edited bound, removed nodes and edges go back to bottom and empty, and an added edge's writer runs again |
| Interprocedural IFDS | `solve_ifds` + `IfdsConfig { max_steps, collect_path_edges }`, `Supergraph` / `IfdsProblem` traits → `IfdsResults` (`facts_at`, `holds`, `summaries`, `stats`); `CfgSupergraph` | Context-sensitive, realizable-path dataflow by the tabulation of Reps, Horwitz and Sagiv over a consumer supergraph and distributive flow functions on dense facts (fact 0 is the zero fact, whose identity the solver adds). O(E·D³) worst case; deterministic (node, fact) worklist with the shared step limit, counted in path edges. Optional fact-level collection of finished contexts' path edges (He et al., ISSTA 2023) lowers the peak without changing the answer. `CfgSupergraph` adapts keyed CFGs whose calls come from the `Flow` of their instructions, one node per program point |
| Slot fixpoint drivers | `wave_fixpoint` (`WaveExecutor`, `SequentialExecutor`, `WaveRequeue`), `ordered_fixpoint` (`SlotAccess`), `OrderedFixpoint` (`run`, `invalidate`) → `SolveStats` | Schedule units over slots that the caller owns, without a graph view or a problem trait: waves evaluated against the facts known before the wave and merged in group order, or ordered (Gauss-Seidel) sweeps whose evaluations record the slots that they read. Both take the shared step limit, counted in evaluations. `OrderedFixpoint` keeps those reads between runs, so a run after an edit evaluates only the units it names, the readers of slots it invalidates, and their transitive readers |
| Dense bit-set facts | `DenseBits` | Set-of-indices lattice element packed 64 per word; `union_with` is the join and reports change for convergence checks; `remove` and `clear` clear one index or the whole row, `words` exposes the packed row, and `ones`/`intersection` step one set index at a time rather than one universe index |

### Higher-level analyses

| Analysis | Function / Type | Description |
|---|---|---|
| Exclusive handler extents | `recover_exclusive_extents(cfg, is_normal)`, `promote_exclusive_extents` | Evidence-reporting sibling of `promote_handler_extents`: blocks exclusively reachable from one handler entry along the edges `is_normal` accepts, boundary and ambiguity issues; both promotions share one atomic per-region core that refuses ambiguous and overlapping bodies |
| Metrics | `GraphMetrics::compute` (any rooted edge view); `CfgMetrics::compute` (the same graph metrics plus instruction counts) | Live node and edge counts (removed slots are not nodes), reachable nodes, cyclomatic complexity, nesting depth (0 for no loops, 1 for one loop; the loops of `LoopForest::compute`, `Back` tags honored), instruction density |
| Pattern detection | `detect_patterns` (any view) → `Pattern`; `Pattern::orient_diamonds` (true arm first); `empty_trampolines` | Diamond (arms in successor order), chain, self-loop; empty trampoline blocks |
| Control tree | `ControlTree::compute`, `compute_treating_empty`, `report` (→ `StructuringReport`) | The structured form of a `Cfg`: `if/else`, pre-tested, post-tested, and endless loops, `switch` with case edges and a default, `try/catch` over declared handler extents, `break`/`continue` (labeled across loops), and `goto`/`label`. A payload-free fact like `DominatorTree`: a flat arena of `ControlNode`s that name `BlockId`, `EdgeId`, `RegionId`, and `BlockSpan` instruction ranges, so it holds no instruction and never borrows the graph. Merge points are immediate post-dominators; an exit-free region has none. Beside an arm that throws or returns, the arms that complete rejoin where they meet: before the stop of the walk that holds the branch, or, in a walk without one, before the exits that two or more arms reach. No owned code is dropped — the blocks the entry reaches and the blocks a handler owns, a catch pad no call unwinds into and its extent among them; a protected block the entry cannot reach is dead code — and the tree's `StructuringReport` lists every goto, swept block, unstructured region, and dangling label. Structuring keeps its own stacks, so the nesting depth of an input is bounded by memory, not by the call stack |
| Control tree traversal | `ControlTree::preorder`, `walk` (`ControlVisitor`), `child_bodies`, `for_each_instruction`, `write_pseudocode` / `to_pseudocode` | Non-recursive preorder and enter/leave walks, the child bodies of any node in execution order, every instruction in program-structure order, and indented pseudocode whose instruction text the caller supplies (the printer keeps its own stack too) |
| Series-parallel-loop decomposition | `SplDecomposition::compute(&tree, &cfg)` → `SplDecomposition` (`SplNode`, `SplNodeId`, children as `ControlRun<SplNodeId>`) or `SplError`; `path_decomposition` → `PathDecomposition` | The control tree as a parse tree of the grammar of structured goto-free programs (`ε`, `break`, `continue`, `;`, `if`, `while`; Cai, Goharshady, Hitarth, Lam 2026), over the graph's blocks: every block in one leaf, the empty blocks and loop tests the tree has no leaf for included, and checked against the graph by the grammar's CFG construction, O(n + m). Loops are `while (true)` with their tests as branches that break. `goto`, labels, `try`, multi-level jumps, early returns, and more than one exit have no decomposition, and the error names the first. The path decomposition has width at most `2·d + 1` for the decomposition's nesting depth `d` (nested `Parallel` and `Loop` nodes), and its bags hold O(n·d) blocks |

### Transforms

`PassPipeline` composes heterogeneous named passes over any caller-owned target.
`pass_fn` adapts a closure; implementing `Pass` supports stateful reusable
passes. Schedules retain insertion order and report the completed prefix when a
fallible pass stops execution: a `PassError` names the failed pass and keeps
that report.

| Transform | Function | Description |
|---|---|---|
| Simplify (all-in-one) | `simplify_mapped` | Unreachable removal + block merging + empty bypass until stable; composes the identity changes of every pass |
| Remove unreachable | `remove_unreachable_mapped` | Reachability pruning; blocks named by exception metadata are kept |
| Merge blocks | `merge_blocks`, `merge_blocks_mapped` | Coalesce single-succ/single-pred chains inside one protected region |
| Remove empty blocks | `remove_empty_blocks`, `remove_empty_blocks_mapped` | Bypass empty fallthrough blocks that no handler or cleanup names |
| Critical edge splitting | `split_critical_edges_mapped` | Insert blocks on multi-succ → multi-pred edges while retaining the original edge identity/payload and mapping both halves; a closure builds each synthetic edge's payload |
| Edge contraction | `contract_edge`, `contract_edge_mapped` | Merge two blocks connected by a single edge; the mapped form reports the surviving edge identities/payloads. Refuses exception edges and merges across a protection boundary |
| Node splitting | `Cfg::split_block`, `Cfg::split_block_at_points` | Split at one or several validated consumer-selected instruction boundaries; the new blocks stay in the original's regions |
| Ext-TSP order | `ext_tsp_order`, `ext_tsp_score`, `ExtTspConfig` | Profile-guided block order (Newell–Pupyrev greedy chain merging, the BOLT and LLVM layout) from consumer block sizes and edge counts: a fallthrough earns an edge's count, a short jump a share that falls with distance. Integer scores, entry first, unreachable blocks omitted, never below reverse postorder. Over a CFG, `falls_through` supplies the fallthrough rule of the `Emitter`: layout edges always, true edges when it inverts branches, jumps when it elides them |
| Hot/cold order | `hot_cold_order` → `[hot, rare, never]` | Stable-partition any base order (reverse postorder, a caller order) by `ColdnessClasses` into the hot part (the view's root first, then `Normal`) and the `Rare` and `Never` sections. Linearizing each part as a `BlockOrder::Custom` gives one stream per section, and a layout edge that crosses the split gets its jump. O(n) |
| Linearization | `linearize`, `falls_through`, `Emitter` trait, `BlockOrder`, `LinearizeError` | Re-serialize CFG to a flat stream; emitters speak `BlockId`, naming is theirs. Exceptional, call, and dispatch edges emit nothing; a conditional edge out of an empty block, or a custom order naming a dead block, is an error. Optional emitter hooks invert a branch whose true target is next (`emit_inverted_conditional_branch`) and drop a jump to the next block (`elides_jump_to_next`); the defaults keep every transfer. `falls_through` says whether an edge needs no transfer when its target is next |
| Width relaxation | `relax_layout`, `relax_spans` (`SpanItem`, `RelaxMode`) | The assembler fixed point for symbolic layouts: monotone branch widening against caller-resolved labels, offset-dependent (alignment) widths, converged offsets plus the final label context; O(n) callbacks per round, O(n²) for a cascade. `relax_spans` reaches the same fixed point for items that depend on one displacement, re-checking only the spans over a changed item through a Fenwick tree and an interval tree, O((n + c) log n) for `c` re-checks; `RelaxMode::Shrink` starts long and shrinks as a linker does, growing back for good a span that alignment pushed out of reach |

Every pass has a `_mapped` form that returns its count and the `Rewrite` of what
it did. Block merging, empty-block bypass and edge contraction also have a plain
form that records nothing, because their record costs as much as the edge moves
themselves. The structural passes (`simplify_mapped`, `merge_blocks`,
`remove_empty_blocks`, `remove_unreachable_mapped`, `contract_edge`) keep a
CFG's exception metadata true, so
`Cfg::compact` and `EhModel` stay usable after any sequence of them. A block a
handler, filter, or cleanup names by a single reference is never removed or
merged away (unreachable removal treats such blocks as roots), and two blocks
merge only when the same regions protect both. `Cfg::split_block` is the
inverse: the new tail joins every region and handler body of the original.
Call `Cfg::remove_unreachable_regions` first to drop regions nothing can raise
into together with their handler blocks.

### Output

Every renderer writes into a `&mut dyn fmt::Write` sink. The `to_*` forms are the
allocating counterparts, and `render` turns any writer into a `String`.
`Cfg::write_listing` indents through [`IndentedWriter`], which consumer printers
can also use.

| Form | Entry points | Notes |
|---|---|---|
| DOT | `write_dot` / `to_dot` over a `DotStyle`; `Cfg::write_dot`, `Graph::write_dot` | `DotStyle` carries the node-label hook, the edge-attribute hook (`DotEdgeAttributes`: label, color, style, penwidth), the graph name, the node-identifier prefix, and `DotRankDir`. `control_flow_edge_attributes` is the CFG's kind-to-color mapping; `plain_edge_attributes` decorates nothing |
| Text | `write_text` / `to_text` over a `TextStyle`, read back by `parse_text`; `Graph::write_text` / `Graph::to_text`; the CFG listing `Cfg::write_listing` / `Cfg::to_listing`, read back by `parse_cfg_listing` | One fact per line. `n3 label`, `n3 -> n5 label`, `#` comments; the CFG listing has `entry bb0`, `bb3:` headers with indented instructions, `bb3 -> bb5 kind`, and `region` / `handler` lines. Escapes are `\\`, `\n`, `\r`, and, on the first character only, `\-`, `\#`, and `\u{20}`-style for whitespace. A document may leave at most 2^20 removed slots (`TextErrorKind::TooManyHoles`). Edges replay in identity order, so the adjacency order a redirect produced is only in the serde form |

Both label hooks are ordinary closures or function items. `DotStyle::new` and
`TextStyle::new` state the higher-ranked signature the writers need, so a
closure passed straight in may return text that borrows its payload;
`no_label` and `display_label` cover the common answers, and `Label<Id, Data>`
names the function-pointer type the default styles use.

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

FlowControl               (CfgBuilder, resolve_jump_edges, call graphs — a
                           structured marker, a label or a Flow; associated
                           Target, CaseKey)

InstrInfo<Variable = V>   (optional — native IR variables, defs/uses)
└── MemoryEventInfo       (locations, address uses, access kinds, and fences — associated Location, Fence)

DisplayInstr              (rendering only — DOT and text)

ir::llil::Instruction     (AddressInstruction + length and bytes of one
                           instruction in its literal form)
ir::llil::Semantics       (emits or captures the RTL record of one instruction)

ir::rtl::Dialect          (instruction-set vocabulary — Root, Operator, EffectOp,
                           WritePolicy, Outcome, NondetDomain)
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
events also retain exact locations, ordered fences, and the `AccessMode`
(direction and atomicity) of each access, which `MemoryEvent::access` lends.
The same `AccessMode` describes the accesses of an `effects::Effect`, whose
`Effect::compute` names each memory location as a `MemoryAddress` (an address
space and an address expression). cfglib owns this vocabulary only: memory
SSA, alias oracles, and memory value flow over it live in the `il` crate.

## Workspace

| Crate | Description |
|---|---|
| **cfglib** | Generic graph, CFG, and dataflow framework with the RTL, MLIL, and HLIL IR trait contracts |

Adapters live with the language they adapt, next to their decoders and test
corpora. The SM4/SM5 decompiler is in
[d3dasm](https://github.com/coconutbird/d3dasm). Its `cfg` module also exposes
a raw CFG and component-granular SSA adapter. The Java/DEX decompiler is in
cafe.

A language picks its entry level. A machine-shaped language emits `ir::rtl`
records through `InstructionBuilder` into its own storage. A variable-shaped
language implements `ir::mlil::FunctionView` directly. A language with source
structure implements `ir::hlil::FunctionView` and `FunctionBuilder`. In
every case the storage, the builders, and the passes that edit them stay with
the language.
cfglib deliberately defines no generic LLIL. Low-level IRs are language-owned,
and the minimal traits that they share (`FlowControl`, `InstrInfo`, `CfgView`,
`Cfg`) are their common surface.

## Adapting a language, IR, or existing graph

For symbol, reference, value-flow, type-relation, import, or grammar graphs, store domain objects directly in `Graph<N, E>`, or in `KeyedGraph<K, N, E>` when the domain identities are sparse keys. Projects that already own adjacency lists can instead implement `GraphView` for their store, then use traversal, shortest-path, topological-sort, SCC, dominance, loop-detection, metrics, and pattern algorithms without migrating their data. Dense `u32` and `usize` handles work directly; custom handles implement `DenseId`.

For a control-flow adapter:

1. Build `Cfg<I>` directly with `new_block()` / `add_edge()`, or use `Cfg<I, E>::with_edge_payload()` plus `add_edge(source, target, Edge::new(kind).with_payload(payload))` when branch labels, handler order, continuation/call-site identity, or source locations must survive. Structured unit-payload streams can instead implement `FlowControl` and use `CfgBuilder::build()`. A consumer with its own block and edge storage implements `CfgView` instead of building a `Cfg`.
2. Optionally implement `InstrInfo` with a native `Variable` identity (and its sub-traits) for dataflow analyses.
3. Implement `DisplayInstr` when you want DOT or text output.

`Cfg<I, E>` is a `Graph<BasicBlock<I>, Edge<E>, BlockTag, EdgeTag>` plus the entry block and the exception metadata that make it a control-flow graph; `E` defaults to `()` for the unannotated form. Two records describe what a change did, and they are not the same thing: `Rewrite` is a transform's sparse account — a missing entry means unchanged, an empty replacement list means removed, one replacement means retained or redirected, several ordered replacements mean a split — while `Renumbering` is the total account `Cfg::compact` gives when it renumbers the survivors. Both compose with `compose(&next)`, and `rewrite.compose(&Rewrite::from(&renumbering))` carries the first across the second. Stable parallel `EdgeId`s plus caller payloads are also the generic continuation/call-site mechanism; cfglib does not impose a VM-specific continuation type.

The variable type only needs `Clone + Ord` — never `Copy`, never numeric. It can be an architecture enum such as `Register(Rax)` / `Flag(Zero)`, a shader structure such as `(register file, index, component)`, an interned source symbol, or an existing IR value handle. Adapters decide the atomic aliasing unit; for overlapping resources such as x86 subregisters, expose canonical units or every affected unit.

```rust
use cfglib::{Cfg, DominatorTree, InstrInfo};

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
```

SSA construction over the same `InstrInfo` adapter lives in the `il` crate.

For an IR adapter, implement the `ir::rtl` or `ir::mlil` traits on your own dense storage. Keep every arena, builder, and editing pass in your crate. Run the graph algorithms of cfglib (dominators, post-dominators, loops, and the rest) on your function view, and apply their answers yourself; the facts over an RTL function, such as the root table and block summaries, are in `il::rtl::facts`. A machine-shaped language emits `ir::rtl` records through `InstructionBuilder`. A variable-shaped language implements `ir::mlil::FunctionView` directly. The consumer writes its own value analyses over that view, because they depend on the semantics of its dialect, and its own passes delete or rewrite instructions.

The concrete shader-bytecode adapter (the d3dasm decompiler's `cfg`
module) derives native register-component identities from decoded masks
and swizzles, retains relative index expressions, classifies multi-result
and UAV read-modify-write operations, and reports observable shader effects
through its own effect vocabulary.

The `tests/source-cfg.rs` integration test is the executable specification of
the source-language side: interned symbol variables, dominators, rendering,
and goto wiring by label token.

Every other target in `crates/cfglib/tests/` pins one public area through the
crate facade only, the way a consumer calls it, and judges the library against
a reference written in the test (a brute-force definition, an interpreter run
before and after a transform, a round-trip law) rather than against the
library's own logic. A directory target (`tests/<area>/main.rs`) holds the
larger ones.

| Area | Targets |
|---|---|
| Graph store, tables, traversal | `graph-store`, `fanout-tables`, `open-search`, `graph-algorithms`, `tombstones` (every analysis on a store with removed slots, before and after `compact`) |
| Control flow and exceptions | `control-tree`, `exception-regions`, `exception-models`, `address-cfg` |
| Dataflow | `dataflow-solvers` |
| Transforms and analyses | `transform-pipeline`, `cfg-analyses` |
| IR contracts, each over its own toy dialect | `rtl-contract`, `mlil-contract` |
| Formats | `text-dot`, `serde-round-trip` (needs `--features serde`), `no-std-facade` |

## Development

`mise install` provides the toolchain, prek and nextest; `mise run ci` runs
every check; `prek install` installs the pre-push hooks (fmt, clippy, file
hygiene). The hooks live in `prek.toml` and run at push time only: every gate
in `AGENTS.md` must still pass before a change is complete. CI runs those gates
in two parallel jobs, plus a `no_std` build for `x86_64-unknown-none`.
`AGENTS.md` also holds the workspace's naming and layout conventions.

## License

MIT
