//! A persisted CFG: the document a consumer writes, loads in another
//! process, and keeps working on.

use std::fmt::Debug;

use cfglib::{
    BasicBlock, BlockId, Cfg, DisplayInstr, DominatorTree, Edge, EdgeId, EdgeKind, ExceptionFlow,
    ExceptionPhase, verify,
};
use serde_json::{Value, json};

use crate::consumer::{Op, Route, rich_cfg, small_cfg};
use crate::golden::assert_golden;
use crate::json::{decode, document, encode, load, round_trip};

/// The decoded CFG checked against the original through every public
/// accessor, including the slots of removed blocks and edges, and through
/// the analyses a consumer runs next.
fn assert_same_cfg<I, E>(actual: &Cfg<I, E>, expected: &Cfg<I, E>)
where
    I: PartialEq + Debug + DisplayInstr,
    E: PartialEq + Debug,
{
    assert_eq!(actual.entry(), expected.entry());
    assert_eq!(actual.block_bound(), expected.block_bound());
    assert_eq!(actual.block_count(), expected.block_count());
    assert_eq!(actual.edge_bound(), expected.edge_bound());
    assert_eq!(actual.edge_count(), expected.edge_count());
    assert_eq!(actual.is_compact(), expected.is_compact());

    for index in 0..expected.block_bound() {
        let block = BlockId::from_index(index);
        assert_eq!(
            actual.contains_block(block),
            expected.contains_block(block),
            "{block}"
        );
        assert_eq!(
            actual.block(block),
            expected.block(block),
            "{block}: a removed block's contents stay readable until compaction"
        );
    }
    for block in expected.block_ids() {
        assert_eq!(
            actual.outgoing(block).collect::<Vec<_>>(),
            expected.outgoing(block).collect::<Vec<_>>(),
            "{block}"
        );
        assert_eq!(
            actual.incoming(block).collect::<Vec<_>>(),
            expected.incoming(block).collect::<Vec<_>>(),
            "{block}"
        );
        assert_eq!(
            actual.successors(block).collect::<Vec<_>>(),
            expected.successors(block).collect::<Vec<_>>()
        );
        assert_eq!(
            actual.predecessors(block).collect::<Vec<_>>(),
            expected.predecessors(block).collect::<Vec<_>>()
        );
    }
    for index in 0..expected.edge_bound() {
        let edge = EdgeId::from_index(index);
        assert_eq!(
            actual.contains_edge(edge),
            expected.contains_edge(edge),
            "{edge}"
        );
    }
    for edge in expected.edge_ids() {
        let (actual, expected) = (actual.edge(edge), expected.edge(edge));
        assert_eq!(actual.source(), expected.source(), "{edge}");
        assert_eq!(actual.target(), expected.target(), "{edge}");
        assert_eq!(actual.kind(), expected.kind(), "{edge}");
        assert_eq!(
            actual.weight().map(f64::to_bits),
            expected.weight().map(f64::to_bits),
            "{edge}"
        );
        assert_eq!(actual.payload(), expected.payload(), "{edge}");
    }
    assert_eq!(actual.regions(), expected.regions());
    assert_eq!(actual.cleanups(), expected.cleanups());

    assert_eq!(
        DominatorTree::compute(actual),
        DominatorTree::compute(expected)
    );
    assert_eq!(actual.to_text(), expected.to_text());
    assert_eq!(verify(actual).is_ok(), verify(expected).is_ok());
}

#[test]
fn a_cfg_with_every_kind_of_state_decodes_equal() {
    let original = rich_cfg();
    assert!(!original.is_compact(), "the fixture keeps removed slots");
    assert_eq!(original.cleanups().len(), 2);

    let decoded = round_trip(&original);
    assert_eq!(decoded, original);
}

#[test]
fn the_decoded_cfg_agrees_with_the_original_through_every_accessor_and_analysis() {
    let original = rich_cfg();
    let decoded: Cfg<Op, Route> = round_trip(&original);
    assert_same_cfg(&decoded, &original);
}

#[test]
fn writing_a_decoded_cfg_reproduces_the_document_byte_for_byte() {
    let first = encode(&rich_cfg());
    let second = encode(&decode::<Cfg<Op, Route>>(&first));
    assert_eq!(first, second);
}

/// Where each old slot of a CFG went through a compaction, by index.
type Renumbered = Vec<Option<usize>>;

/// Apply one fixed edit script and report where every old block and edge went.
fn evolve(mut cfg: Cfg<Op, Route>) -> (Cfg<Op, Route>, Renumbered, Renumbered) {
    let late = cfg.block_ids().last().expect("the fixture has blocks");
    let fresh = cfg.new_block();
    cfg.block_mut(fresh).push(Op::Call("fresh".to_owned()));
    cfg.add_weighted_edge_with_payload(cfg.entry(), fresh, EdgeKind::Call, 0.5, Route::case(9));
    cfg.remove_block(late);

    let (block_bound, edge_bound) = (cfg.block_bound(), cfg.edge_bound());
    let renumbering = cfg.compact();
    let blocks = (0..block_bound)
        .map(|index| {
            renumbering
                .node(BlockId::from_index(index))
                .map(BlockId::index)
        })
        .collect();
    let edges = (0..edge_bound)
        .map(|index| {
            renumbering
                .edge(EdgeId::from_index(index))
                .map(EdgeId::index)
        })
        .collect();
    (cfg, blocks, edges)
}

#[test]
fn the_decoded_cfg_keeps_evolving_exactly_like_the_original() {
    let (expected, expected_blocks, expected_edges) = evolve(rich_cfg());
    let (actual, actual_blocks, actual_edges) = evolve(round_trip(&rich_cfg()));

    assert_same_cfg(&actual, &expected);
    assert_eq!(
        actual_blocks, expected_blocks,
        "the same renumbering of blocks"
    );
    assert_eq!(
        actual_edges, expected_edges,
        "the same renumbering of edges"
    );
    assert!(actual.is_compact());
}

#[test]
fn removed_blocks_and_edges_stay_removed_and_keep_their_slots() {
    let original = rich_cfg();
    let decoded: Cfg<Op, Route> = round_trip(&original);

    let removed_blocks: Vec<_> = (0..original.block_bound())
        .map(BlockId::from_index)
        .filter(|&block| !original.contains_block(block))
        .collect();
    assert!(!removed_blocks.is_empty(), "the fixture removed a block");
    for block in removed_blocks {
        assert!(!decoded.contains_block(block), "{block}");
    }
    assert!(decoded.block_bound() > decoded.block_count());
    assert!(decoded.edge_bound() > decoded.edge_count());
}

#[test]
fn a_compacted_cfg_round_trips_and_stays_compact() {
    let (compacted, ..) = evolve(rich_cfg());
    let decoded: Cfg<Op, Route> = round_trip(&compacted);
    assert!(decoded.is_compact());
    assert_same_cfg(&decoded, &compacted);
}

#[test]
fn an_empty_cfg_round_trips() {
    let cfg = Cfg::<Op>::new();
    let decoded: Cfg<Op> = round_trip(&cfg);
    assert_same_cfg(&decoded, &cfg);
    assert_eq!(decoded.block_count(), 1);
}

#[test]
fn the_fidelity_path_keeps_what_the_text_form_cannot() {
    // Weights, edge payloads and cleanup records are outside the text form;
    // this is the path that carries them.
    let original = rich_cfg();
    let decoded: Cfg<Op, Route> = round_trip(&original);

    let weights: Vec<_> = decoded.edges().filter_map(|edge| edge.weight()).collect();
    assert_eq!(weights, [0.75, 0.25, 0.5]);
    assert!(decoded.edges().any(|edge| edge.payload().case == Some(3)));
    assert!(
        decoded
            .edges()
            .any(|edge| edge.payload().provenance == ["loop condition"])
    );
    assert_eq!(decoded.cleanups().len(), 2);
}

#[test]
fn a_cfg_edge_payload_of_the_exception_model_round_trips() {
    let mut cfg = Cfg::<Op, ExceptionFlow<String>>::with_edge_payload();
    let handler = cfg.new_block();
    cfg.add_edge_with_payload(
        cfg.entry(),
        handler,
        EdgeKind::ExceptionUnwind,
        ExceptionFlow::exceptional(ExceptionPhase::Search, None, "filter #2".to_owned()),
    );
    cfg.add_edge_with_payload(
        cfg.entry(),
        handler,
        EdgeKind::Fallthrough,
        ExceptionFlow::normal("scope table".to_owned()),
    );

    let decoded: Cfg<Op, ExceptionFlow<String>> = round_trip(&cfg);
    assert_same_cfg(&decoded, &cfg);
    let kinds: Vec<_> = decoded.edges().map(|edge| edge.payload().phase()).collect();
    assert_eq!(kinds, [Some(ExceptionPhase::Search), None]);
}

#[test]
fn a_block_and_an_edge_round_trip_on_their_own() {
    let cfg = rich_cfg();
    for block in cfg.blocks() {
        let decoded: BasicBlock<Op> = round_trip(block);
        assert_eq!(&decoded, block);
    }
    for edge in cfg.edges() {
        let decoded: Edge<Route> = round_trip(edge.data());
        assert_eq!(&decoded, edge.data());
    }
}

#[test]
fn every_edge_kind_is_written_as_its_variant_name() {
    for kind in EdgeKind::ALL {
        let value = document(&kind);
        assert_eq!(value, Value::String(format!("{kind:?}")));
        assert_eq!(load::<EdgeKind>(value).expect("the name loads"), kind);
    }
    assert!(load::<EdgeKind>(Value::String("Sideways".to_owned())).is_err());
    assert!(load::<EdgeKind>(Value::String("fallthrough".to_owned())).is_err());
}

#[test]
fn the_small_cfg_document_is_stable() {
    let json = serde_json::to_string_pretty(&small_cfg(true)).expect("the value serializes");
    assert_golden("serde-round-trip/small-cfg.json", &json);
}

/// The document of `small_cfg(false)` as a release without cleanup records
/// wrote it, written out by hand: the `cleanups` field is not there.
fn older_document() -> Value {
    json!({
        "graph": {
            "base_nodes": [],
            "base_edges": [],
            "out_offsets": [],
            "out_edges": [],
            "in_offsets": [],
            "in_edges": [],
            "delta_nodes": [
                {"instructions": ["x = f()"], "label": null},
                {"instructions": [], "label": "cleanup"}
            ],
            "delta_edges": [
                {
                    "source": 0,
                    "target": 1,
                    "payload": {"kind": "ExceptionUnwind", "weight": null, "payload": null}
                }
            ],
            "chains": {"last": [[0, 4_294_967_295_u32], [4_294_967_295_u32, 0]], "next": [[0, 0]]},
            "live_nodes": {"words": [3], "len": 2},
            "live_edges": {"words": [1], "len": 1},
            "live_node_count": 2,
            "live_edge_count": 1
        },
        "entry": 0,
        "regions": [
            {
                "id": 0,
                "protected_blocks": [0],
                "handlers": [
                    {"entry": 1, "body": {"Known": [1]}, "kind": "Finally"}
                ],
                "parent": null
            }
        ]
    })
}

#[test]
fn a_document_written_before_cleanup_records_existed_still_loads_with_none() {
    let decoded: Cfg<String> = load(older_document()).expect("`cleanups` has a default");
    assert_eq!(decoded, small_cfg(false));
    assert_eq!(decoded.cleanups(), []);
}

#[test]
fn dropping_the_cleanups_field_from_a_current_document_leaves_everything_else() {
    let mut value = document(&rich_cfg());
    value
        .as_object_mut()
        .expect("a CFG is an object")
        .remove("cleanups")
        .expect("a current document has the field");

    let decoded: Cfg<Op, Route> = load(value).expect("`cleanups` has a default");
    let original = rich_cfg();
    assert_eq!(decoded.cleanups(), []);
    assert_eq!(decoded.regions(), original.regions());
    assert_eq!(decoded.to_text(), original.to_text());
    assert_eq!(decoded.edge_count(), original.edge_count());
    assert_eq!(decoded.block_bound(), original.block_bound());
}

#[test]
fn an_explicitly_empty_cleanups_field_is_the_same_as_an_absent_one() {
    let mut explicit = older_document();
    explicit["cleanups"] = json!([]);
    let decoded: Cfg<String> = load(explicit).expect("an empty list loads");
    assert_eq!(decoded, small_cfg(false));
}
