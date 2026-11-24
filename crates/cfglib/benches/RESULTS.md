# Local optimization results

Measured on Windows on 2026-08-24, starting from revision `43d3f07`. These are
synthetic regression fixtures, not claims about every consumer workload. This
file is a dated record of that session, not a description of the current tree:
later changes have moved the numbers, and the harness's `coverage:` line, not
this page, states how many cases and facade functions a run covers. Rerun the
commands in [README.md](README.md) for current figures. The graph-store
baseline and the ordered-compaction record at the end are later sessions, each
with its own date and revision.

## Screening comparison

The table compares the unmodified starting revision with the optimized working
tree on the same machine. Both used the direct `System` allocator, seven timing
samples, and `CFGLIB_BENCH_MS=1`. That short target is useful for ranking large
regressions while iterating, but the absolute values are intentionally treated
as approximate. Mutation timings include the fixture clone named by each case.

| Case | Before (ns/op) | After (ns/op) | Speedup |
| --- | ---: | ---: | ---: |
| `cfg_dominators` | 222,400 | 49,066 | 4.53x |
| `cfg_control_dependence_graph` | 2,347,700 | 75,800 | 30.97x |
| `directed_detect_loops_multilatch` | 18,842,000 | 578,750 | 32.56x |
| `cfg_interval_analysis` | 45,103,800 | 137,013 | 329.19x |
| `directed_interval_reverse_id_chain` | 851,031,600 | 110,340 | 7,712.81x |
| `directed_node_fixpoint_bool` | 444,833 | 21,611 | 20.58x |
| `directed_breadth_first_edges` | 100,560 | 14,963 | 6.72x |
| `directed_shortest_path_edges` | 99,655 | 15,906 | 6.27x |
| `cfg_clone_merge_linear` | 131,924,300 | 110,867 | 1,189.94x |
| `cfg_clone_remove_empty_chain` | 47,257,400 | 84,708 | 557.88x |
| `cfg_clone_redirect_high_fan_in` | 6,900,500 | 82,150 | 84.00x |
| `cfg_clone_split_weighted_high_fan_out` | 407,700 | 7,536 | 54.10x |
| `cfg_clone_merge_weighted_high_fan_out` | 218,995,500 | 12,593 | 17,390.26x |
| `cfg_clone_make_reducible_large` | 44,566,300 | 12,070,300 | 3.69x |

The starting weighted reducibility transformation failed its semantic oracle
because copied outgoing edges lost their weights; the optimized version retains
edge kind, weight, identity rules, and adjacency order.

## Verification run

A longer run of the optimized tree with `CFGLIB_BENCH_MS=25` completed every
case and all semantic oracles. The registry verified benchmark coverage for
every root-facade free function. The allocation-instrumented build also
completed every case and verified that each measured operation returned to its
starting live-byte baseline.

The zero-allocation reusable-scratch search is a useful harness sanity check:
`api_search_with_scratch` reports no allocations where the allocating
`api_search` entry point reports 11 on the same 1,024-node workload, which
holds in the current tree. Use the commands in [README.md](README.md) to
reproduce either mode; pinning an otherwise-idle machine and increasing the
target duration is recommended before making small-factor decisions. The
per-case figures of that run are not kept here, because the allocation counts
of most cases have moved since.

## Graph-store baseline

Measured on Linux on 2026-11-24 at revision `94ca44a`, the commit that added
the structured inputs, on an AMD Ryzen AI MAX+ 395. Each figure is the
fastest of five repetitions within a run and of three runs, pinned to one
core. Other processes kept the load average between 6 and 17 of 32 hardware
threads, and the uniform rows moved by up to 2× between runs, so read the
ratios inside one table rather than the absolute values. There is no earlier
record of this target to compare with.

Locality of each input, as N2N AID; lower is more local:

| Input | Successors | Predecessors |
| --- | ---: | ---: |
| `uniform` | 151,584.3 | 143,726.3 |
| `structured` | 107,466.8 | 8,324.4 |
| `structured_relabeled` | 165,740.4 | 160,311.5 |

The structured input's successor distance is high because half of its blocks
call a procedure entry anywhere in the graph; its predecessors are mostly
the blocks just before them.

One million nodes, 100,000 appends (milliseconds):

| Input | Arrival | `build` | `compact_after_build` | `successor_scan` | `predecessor_scan` | `append_and_scan` | `compact` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `uniform` | shuffled | 151.4 | 56.7 | 102.1 | 98.0 | 109.3 | 69.6 |
| `uniform` | grouped | 170.1 | 46.5 | 20.2 | 100.3 | 24.9 | 62.9 |
| `structured` | shuffled | 53.4 | 27.6 | 26.4 | 23.6 | 40.5 | 33.6 |
| `structured` | grouped | 28.5 | 20.4 | 13.2 | 18.9 | 16.8 | 28.7 |
| `structured_relabeled` | shuffled | 46.8 | 27.2 | 33.4 | 18.3 | 42.0 | 35.2 |
| `structured_relabeled` | grouped | 43.5 | 22.8 | 13.6 | 29.1 | 17.5 | 30.3 |

The procedure-scale table read 13.4 ms for `build_scan` and 20.7 ms for
`build_compact_scan`.

What the inputs separate:

- Arrival order dominates successor scans. Grouped arrival puts a node's
  edge records next to each other, so walking successors costs a fifth of
  the shuffled walk on `uniform` and half on `structured`. On `uniform` the
  predecessor walk stays scattered whatever the arrival, because a node's
  incoming edges come from everywhere.
- Numbering shows where the edge records are already in order. With
  grouped arrival, relabeling the structured input slows the predecessor
  walk by 1.5× and the build by 1.5×, while the successor walk, whose
  neighbor payload reads are dominated by far calls on both inputs, barely
  moves. Under shuffled arrival the scattered edge-record loads hide the
  numbering: the two structured inputs traded places between runs.
- These are the rows to judge #86 (ordered compaction) and #87 (neighbor
  ids beside the edge ids) against: a gain should appear on
  `structured_relabeled` and on predecessor walks, and the uniform rows bound
  its worst case.

## Ordered compaction

Measured on Linux on 2026-11-24 on revision `e570e9c`, the commit that added
`compact_ordered`, plus the benchmark rows that came with this record, on the
same machine, pinned to one core. Each figure is the fastest of five
repetitions within a run and of three runs. Other processes kept the load
average between 29 and 66 of 32 hardware threads, and one of the three runs
read up to 3× slower throughout, so read the ratios inside one table. The
searches start from the nodes with no predecessor, which are found before the
clock starts.

Locality of each numbering of `structured_relabeled`, as N2N AID for
successors / predecessors; lower is more local. The `structured` input it was
shuffled from reads 107,466.8 / 8,324.4.

| Order | Shuffled arrival | Grouped arrival |
| --- | ---: | ---: |
| `compact` | 165,740.4 / 160,311.5 | 165,740.4 / 160,311.5 |
| `first_touch` | 191,955.7 / 151,207.0 | 173,006.5 / 148,394.6 |
| `breadth_first` | 150,257.7 / 42,096.5 | 150,256.2 / 42,091.1 |
| `reverse_postorder` | 131,304.2 / 12,696.5 | 139,466.5 / 21,651.8 |

One million nodes, 1,849,803 edges (milliseconds):

| Arrival | Order | `compaction` | `successor_scan` | `predecessor_scan` | `tarjan_scc` |
| --- | --- | ---: | ---: | ---: | ---: |
| shuffled | `compact` | 20.9 | 22.1 | 16.1 | 159.3 |
| shuffled | `first_touch` | 40.8 | 20.2 | 15.3 | 143.3 |
| shuffled | `breadth_first` | 152.5 | 17.5 | 13.8 | 126.4 |
| shuffled | `reverse_postorder` | 200.2 | 18.8 | 14.1 | 68.5 |
| grouped | `compact` | 19.7 | 13.6 | 18.4 | 162.6 |
| grouped | `first_touch` | 42.7 | 20.6 | 15.0 | 147.0 |
| grouped | `breadth_first` | 137.9 | 16.5 | 13.3 | 136.2 |
| grouped | `reverse_postorder` | 194.2 | 19.2 | 13.7 | 72.8 |

What the orders buy on this input:

- Reverse postorder recovers most of the predecessor locality the shuffle
  took, and halves `tarjan_scc` under both arrivals. Breadth-first recovers
  less and gains less.
- Predecessor walks are 12 to 28% faster under the two searches and 5 to
  18% faster under first touch.
- Successor walks are 9 to 21% faster under shuffled arrival, but 21 to
  51% slower under grouped arrival. Grouped arrival had laid each node's
  outgoing edge records out together in the old node order. Only nodes move,
  so a walk in the new order visits those runs out of order; neighbor ids
  beside the edge ids (#87) would remove that load.
- The searches cost 7× (breadth-first) and 10× (reverse postorder) a plain
  compaction, because they walk a freshly built store's delta chains in a
  scattered order. First touch costs 2×.
- First touch is the weakest order here: neither arrival gives it a
  meaningful edge order to follow, since one is random and the other is
  sorted by the shuffled ids.
