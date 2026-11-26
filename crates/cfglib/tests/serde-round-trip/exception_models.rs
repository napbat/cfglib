//! Normalized CLR and SEH regions with their catch-type side table, and the
//! exception-flow edge annotations, persisted by a binary analyzer between
//! runs.

use cfglib::{
    BlockId, Cfg, EdgeKind, ExceptionDisposition, ExceptionFlow, ExceptionPhase, Handler,
    HandlerBody, HandlerKind, HandlerMetadata, Region,
};
use serde_json::json;

use crate::consumer::Op;
use crate::json::{document, load, round_trip};

/// A CFG with room for the blocks the regions below name.
fn cfg_with_blocks() -> (Cfg<Op>, Vec<BlockId>) {
    let mut cfg = Cfg::<Op>::new();
    let mut blocks = vec![cfg.entry()];
    for _ in 0..5 {
        blocks.push(cfg.new_block());
    }
    (cfg, blocks)
}

/// A CLR clause table with every clause kind, and the type token of its one
/// typed catch, keyed by the clause's position.
fn clr_region(blocks: &[BlockId]) -> (Region, Vec<(usize, u32)>) {
    let region = Region {
        protected_blocks: [blocks[0], blocks[1]].into_iter().collect(),
        handlers: vec![
            Handler {
                entry: blocks[2],
                body: HandlerBody::known([blocks[2]]),
                kind: HandlerKind::Catch,
            },
            Handler {
                entry: blocks[2],
                body: HandlerBody::Unknown,
                kind: HandlerKind::CatchAll,
            },
            Handler {
                entry: blocks[3],
                body: HandlerBody::known([blocks[3]]),
                kind: HandlerKind::Finally,
            },
            Handler {
                entry: blocks[3],
                body: HandlerBody::known([]),
                kind: HandlerKind::Fault,
            },
            Handler {
                entry: blocks[4],
                body: HandlerBody::Unknown,
                kind: HandlerKind::Filter {
                    filter_block: blocks[5],
                },
            },
        ],
        parent: None,
    };
    (region, vec![(0, 0x0200_0001)])
}

#[test]
fn a_clr_region_and_its_type_tokens_decode_equal_and_register_like_the_original() {
    let (mut original_cfg, blocks) = cfg_with_blocks();
    let (mut decoded_cfg, _) = cfg_with_blocks();
    let (region, tokens) = clr_region(&blocks);
    let decoded: Region = round_trip(&region);
    assert_eq!(decoded, region);

    let original_id = original_cfg.add_region(region).unwrap();
    let decoded_id = decoded_cfg.add_region(decoded).unwrap();
    let mut original_types = HandlerMetadata::new();
    original_types.set_all(original_id, tokens);
    let decoded_types: HandlerMetadata<u32> = round_trip(&original_types);

    assert_eq!(decoded_id, original_id);
    assert_eq!(
        decoded_cfg.regions().collect::<Vec<_>>(),
        original_cfg.regions().collect::<Vec<_>>()
    );
    assert_eq!(decoded_types, original_types);
    assert_eq!(
        decoded_types.len(),
        1,
        "only the typed catch has a type token"
    );
}

#[test]
fn an_seh_scope_table_decodes_equal_and_registers_like_the_original() {
    let (mut original_cfg, blocks) = cfg_with_blocks();
    let (mut decoded_cfg, _) = cfg_with_blocks();
    // `__except (filter)` then `__finally`, in dispatch order.
    let region = Region {
        protected_blocks: [blocks[0]].into_iter().collect(),
        handlers: vec![
            Handler {
                entry: blocks[2],
                body: HandlerBody::known([blocks[2], blocks[3]]),
                kind: HandlerKind::Filter {
                    filter_block: blocks[1],
                },
            },
            Handler {
                entry: blocks[4],
                body: HandlerBody::Unknown,
                kind: HandlerKind::Finally,
            },
        ],
        parent: None,
    };
    let decoded: Region = round_trip(&region);
    assert_eq!(decoded, region);

    let original_id = original_cfg.add_region(region).unwrap();
    let decoded_id = decoded_cfg.add_region(decoded).unwrap();
    assert_eq!(decoded_id, original_id);
    assert_eq!(
        decoded_cfg.regions().collect::<Vec<_>>(),
        original_cfg.regions().collect::<Vec<_>>()
    );
}

#[test]
fn every_exception_flow_variant_round_trips_with_its_metadata() {
    let phases = [ExceptionPhase::Search, ExceptionPhase::Unwind];
    let dispositions = [
        None,
        Some(ExceptionDisposition::ExecuteHandler),
        Some(ExceptionDisposition::ContinueSearch),
        Some(ExceptionDisposition::ContinueExecution),
    ];
    let mut flows = vec![ExceptionFlow::new("scope".to_owned())];
    for phase in phases {
        for disposition in dispositions {
            flows.push(ExceptionFlow {
                phase: Some(phase),
                disposition,
                metadata: "record".to_owned(),
            });
        }
    }

    for flow in flows {
        let decoded: ExceptionFlow<String> = round_trip(&flow);
        assert_eq!(decoded, flow);
        assert_eq!(decoded.phase, flow.phase);
        assert_eq!(decoded.disposition, flow.disposition);
        assert_eq!(decoded.metadata, flow.metadata);
    }
    let unit: ExceptionFlow = round_trip(&ExceptionFlow::new(()));
    assert_eq!(unit.phase, None);
}

#[test]
fn unknown_exception_vocabulary_is_rejected() {
    assert!(load::<ExceptionPhase>(json!("Dispatch")).is_err());
    assert!(load::<ExceptionDisposition>(json!("Abort")).is_err());
    assert!(
        load::<ExceptionFlow<()>>(json!({
            "phase": "Sideways",
            "disposition": null,
            "metadata": null
        }))
        .is_err()
    );
    assert!(
        load::<ExceptionFlow<()>>(json!({
            "phase": "Search",
            "disposition": "Abort",
            "metadata": null
        }))
        .is_err()
    );
}

#[test]
fn the_phase_and_disposition_names_are_the_variant_names() {
    assert_eq!(document(&ExceptionPhase::Search), json!("Search"));
    assert_eq!(
        document(&ExceptionDisposition::ContinueExecution),
        json!("ContinueExecution")
    );
    assert_eq!(
        document(&EdgeKind::ExceptionContinue),
        json!("ExceptionContinue")
    );
}
