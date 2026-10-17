//! Normalized CLR and SEH regions, the vectored-handler lists, and the
//! exception-flow edge annotations, persisted by a binary analyzer between
//! runs.

use cfglib::{
    BlockId, Cfg, ClrExceptionRegion, ClrHandler, ClrHandlerKind, EdgeKind, ExceptionDisposition,
    ExceptionFlow, ExceptionPhase, HandlerBody, HandlerTypes, SehExceptionRegion, SehHandler,
    SehHandlerKind, SehRegistration, SehRegistrationChain, VectoredExceptionModel,
    VectoredHandlerId, VectoredHandlerKind, VectoredHandlerOrder, install_clr_region,
    install_seh_region,
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

fn clr_region(blocks: &[BlockId]) -> ClrExceptionRegion<u32> {
    ClrExceptionRegion {
        protected_blocks: [blocks[0], blocks[1]].into_iter().collect(),
        handlers: vec![
            ClrHandler {
                entry: blocks[2],
                body: HandlerBody::known([blocks[2]]),
                kind: ClrHandlerKind::Catch { ty: 0x0200_0001 },
            },
            ClrHandler {
                entry: blocks[2],
                body: HandlerBody::Unknown,
                kind: ClrHandlerKind::CatchAll,
            },
            ClrHandler {
                entry: blocks[3],
                body: HandlerBody::known([blocks[3]]),
                kind: ClrHandlerKind::Finally,
            },
            ClrHandler {
                entry: blocks[3],
                body: HandlerBody::known([]),
                kind: ClrHandlerKind::Fault,
            },
            ClrHandler {
                entry: blocks[4],
                body: HandlerBody::Unknown,
                kind: ClrHandlerKind::Filter {
                    filter_block: blocks[5],
                },
            },
        ],
        parent: None,
    }
}

#[test]
fn a_clr_region_decodes_equal_and_installs_like_the_original() {
    let (mut original_cfg, blocks) = cfg_with_blocks();
    let (mut decoded_cfg, _) = cfg_with_blocks();
    let region = clr_region(&blocks);
    let decoded: ClrExceptionRegion<u32> = round_trip(&region);
    assert_eq!(decoded, region);

    let mut original_types = HandlerTypes::new();
    let mut decoded_types = HandlerTypes::new();
    let original_id = install_clr_region(&mut original_cfg, &mut original_types, region);
    let decoded_id = install_clr_region(&mut decoded_cfg, &mut decoded_types, decoded);

    assert_eq!(decoded_id, original_id);
    assert_eq!(decoded_cfg.regions(), original_cfg.regions());
    assert_eq!(decoded_types, original_types);
    assert_eq!(
        decoded_types.len(),
        1,
        "only the typed catch has a type token"
    );
}

#[test]
fn every_clr_handler_kind_round_trips() {
    for kind in [
        ClrHandlerKind::Catch {
            ty: "System.Exception".to_owned(),
        },
        ClrHandlerKind::CatchAll,
        ClrHandlerKind::Finally,
        ClrHandlerKind::Fault,
        ClrHandlerKind::Filter {
            filter_block: BlockId::from_raw(4),
        },
    ] {
        assert_eq!(round_trip(&kind), kind);
    }
}

#[test]
fn an_seh_scope_table_decodes_equal_and_installs_like_the_original() {
    let (mut original_cfg, blocks) = cfg_with_blocks();
    let (mut decoded_cfg, _) = cfg_with_blocks();
    let region = SehExceptionRegion {
        protected_blocks: [blocks[0]].into_iter().collect(),
        handlers: vec![
            SehHandler {
                entry: blocks[2],
                body: HandlerBody::known([blocks[2], blocks[3]]),
                kind: SehHandlerKind::Except {
                    filter_block: blocks[1],
                },
            },
            SehHandler {
                entry: blocks[4],
                body: HandlerBody::Unknown,
                kind: SehHandlerKind::Finally,
            },
        ],
        parent: None,
    };
    let decoded: SehExceptionRegion = round_trip(&region);
    assert_eq!(decoded, region);

    let original_id = install_seh_region(&mut original_cfg, region);
    let decoded_id = install_seh_region(&mut decoded_cfg, decoded);
    assert_eq!(decoded_id, original_id);
    assert_eq!(decoded_cfg.regions(), original_cfg.regions());
}

#[test]
fn a_registration_chain_keeps_its_newest_first_dispatch_order() {
    let mut chain = SehRegistrationChain::<u32, u64>::new();
    for (frame, handler) in [
        (0x1000, 0x40_1000),
        (0x0F00, 0x40_2000),
        (0x0E00, 0x40_3000),
    ] {
        chain.push(SehRegistration {
            frame,
            handler,
            regions: vec![cfglib::RegionId::from_raw(frame % 7)],
        });
    }
    chain.pop();

    let mut decoded: SehRegistrationChain<u32, u64> = round_trip(&chain);
    assert_eq!(decoded, chain);
    assert_eq!(decoded.len(), 2);
    assert_eq!(
        decoded
            .dispatch_order()
            .map(|entry| entry.frame)
            .collect::<Vec<_>>(),
        [0x0F00, 0x1000]
    );

    // A chain loaded from a document keeps behaving like a live one.
    decoded.push(SehRegistration {
        frame: 0x0D00,
        handler: 0x40_4000,
        regions: Vec::new(),
    });
    assert_eq!(
        decoded.dispatch_order().next().map(|entry| entry.frame),
        Some(0x0D00)
    );
    assert_eq!(decoded.pop().map(|entry| entry.frame), Some(0x0D00));
}

/// A process-wide model after some history: both lists in use, one handler
/// registered first, one removed.
fn vectored_model() -> VectoredExceptionModel<String> {
    let mut model = VectoredExceptionModel::new();
    let tail = model.register(
        VectoredHandlerKind::Exception,
        VectoredHandlerOrder::Last,
        "tail".to_owned(),
    );
    model.register(
        VectoredHandlerKind::Exception,
        VectoredHandlerOrder::First,
        "head".to_owned(),
    );
    model.register(
        VectoredHandlerKind::Continue,
        VectoredHandlerOrder::Last,
        "resume".to_owned(),
    );
    model.remove(tail);
    model
}

fn names(model: &VectoredExceptionModel<String>, kind: VectoredHandlerKind) -> Vec<String> {
    model
        .handlers(kind)
        .map(|entry| entry.handler.clone())
        .collect()
}

#[test]
fn a_vectored_model_keeps_call_order_and_hands_out_fresh_identities() {
    let original = vectored_model();
    let mut decoded: VectoredExceptionModel<String> = round_trip(&original);
    assert_eq!(decoded, original);

    assert_eq!(names(&decoded, VectoredHandlerKind::Exception), ["head"]);
    assert_eq!(names(&decoded, VectoredHandlerKind::Continue), ["resume"]);
    assert_eq!(decoded.len(), 2);

    // The next identity must continue the original's numbering, or a decoded
    // model could hand out an identity that is already registered.
    let mut continued = original.clone();
    let fresh_original = continued.register(
        VectoredHandlerKind::Exception,
        VectoredHandlerOrder::First,
        "late".to_owned(),
    );
    let fresh_decoded = decoded.register(
        VectoredHandlerKind::Exception,
        VectoredHandlerOrder::First,
        "late".to_owned(),
    );
    assert_eq!(fresh_decoded, fresh_original);
    assert_eq!(fresh_decoded, VectoredHandlerId::from_raw(3));
    assert_eq!(decoded, continued);
    assert_eq!(
        names(&decoded, VectoredHandlerKind::Exception),
        ["late", "head"]
    );
}

#[test]
fn a_decoded_vectored_model_still_removes_by_identity() {
    let original = vectored_model();
    let mut decoded: VectoredExceptionModel<String> = round_trip(&original);
    let removed = decoded.remove(VectoredHandlerId::from_raw(2));
    assert_eq!(
        removed.map(|entry| entry.handler),
        Some("resume".to_owned())
    );
    assert_eq!(
        decoded
            .remove(VectoredHandlerId::from_raw(0))
            .map(|entry| entry.handler),
        None
    );
    assert_eq!(decoded.len(), 1);
}

#[test]
fn list_selectors_and_identities_round_trip() {
    for kind in [
        VectoredHandlerKind::Exception,
        VectoredHandlerKind::Continue,
    ] {
        assert_eq!(round_trip(&kind), kind);
    }
    for order in [VectoredHandlerOrder::First, VectoredHandlerOrder::Last] {
        assert_eq!(round_trip(&order), order);
    }
    let id = VectoredHandlerId::from_raw(u64::MAX);
    assert_eq!(
        round_trip(&id).raw(),
        u64::MAX,
        "the identity space is 64 bits wide"
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
    let mut flows = vec![ExceptionFlow::normal("scope".to_owned())];
    for phase in phases {
        for disposition in dispositions {
            flows.push(ExceptionFlow::exceptional(
                phase,
                disposition,
                "record".to_owned(),
            ));
        }
    }

    for flow in flows {
        let decoded: ExceptionFlow<String> = round_trip(&flow);
        assert_eq!(decoded, flow);
        assert_eq!(decoded.phase(), flow.phase());
        assert_eq!(decoded.disposition(), flow.disposition());
        assert_eq!(decoded.metadata(), flow.metadata());
    }
    let unit: ExceptionFlow = round_trip(&ExceptionFlow::normal(()));
    assert_eq!(unit.phase(), None);
}

#[test]
fn unknown_exception_vocabulary_is_rejected() {
    assert!(load::<ExceptionPhase>(json!("Dispatch")).is_err());
    assert!(load::<ExceptionDisposition>(json!("Abort")).is_err());
    assert!(load::<VectoredHandlerKind>(json!("Frame")).is_err());
    assert!(load::<VectoredHandlerOrder>(json!("Middle")).is_err());
    assert!(load::<ExceptionFlow<()>>(json!({"Sideways": {"metadata": null}})).is_err());
    assert!(
        load::<ExceptionFlow<()>>(json!({"Exceptional": {"metadata": null}})).is_err(),
        "an exceptional edge must say which phase it is in"
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
