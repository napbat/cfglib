//! The exception-region family in a document: regions, handlers, bodies,
//! cleanup records, and the consumer-owned handler metadata side table.

use cfglib::{
    BlockId, Cleanup, CompletionReason, Continuation, Handler, HandlerBody, HandlerFilters,
    HandlerKind, HandlerRef, HandlerTypes, Region, RegionId,
};
use serde_json::{Value, json};

use crate::consumer::{handler, region};
use crate::json::{document, encode, load, round_trip};

fn block(raw: u32) -> BlockId {
    BlockId::from_raw(raw)
}

#[test]
fn a_region_with_every_handler_kind_and_body_shape_round_trips() {
    let original = region(
        &[block(1), block(2)],
        vec![
            handler(block(5), HandlerKind::Catch, HandlerBody::known([block(5)])),
            handler(block(5), HandlerKind::CatchAll, HandlerBody::known([])),
            handler(
                block(5),
                HandlerKind::Filter {
                    filter_block: block(4),
                },
                HandlerBody::Unknown,
            ),
            handler(
                block(6),
                HandlerKind::Finally,
                HandlerBody::known([block(6), block(7)]),
            ),
            handler(block(6), HandlerKind::Fault, HandlerBody::Unknown),
        ],
        Some(RegionId::from_raw(0)),
    );

    let decoded: Region = round_trip(&original);
    assert_eq!(decoded, original);
    for (decoded, original) in decoded.handlers.iter().zip(&original.handlers) {
        assert_eq!(decoded.body.blocks(), original.body.blocks());
        assert_eq!(decoded.body.is_known(), original.body.is_known());
    }
}

#[test]
fn a_handler_body_has_its_documented_externally_tagged_form() {
    let known = HandlerBody::known([block(1), block(2)]);
    assert_eq!(encode(&known), r#"{"Known":[1,2]}"#);
    assert_eq!(encode(&HandlerBody::known([])), r#"{"Known":[]}"#);
    assert_eq!(encode(&HandlerBody::Unknown), r#""Unknown""#);

    assert_eq!(round_trip(&known), known);
    let empty = round_trip(&HandlerBody::known([]));
    assert!(empty.is_known(), "an empty complete body stays complete");
    assert_eq!(round_trip(&HandlerBody::unknown()), HandlerBody::Unknown);
}

#[test]
fn a_handler_body_stored_as_a_plain_block_sequence_no_longer_loads() {
    // The doc on `HandlerBody` says snapshots from before the type existed
    // do not deserialize; they must fail cleanly rather than load as empty.
    assert!(load::<HandlerBody>(json!([1, 2])).is_err());
    assert!(load::<Handler>(json!({"entry": 3, "body": [3], "kind": "Catch"})).is_err());
}

#[test]
fn every_handler_kind_and_completion_reason_round_trips_by_name() {
    for kind in [
        HandlerKind::Catch,
        HandlerKind::CatchAll,
        HandlerKind::Finally,
        HandlerKind::Fault,
        HandlerKind::Filter {
            filter_block: block(9),
        },
    ] {
        assert_eq!(round_trip(&kind), kind);
    }
    for reason in [
        CompletionReason::Normal,
        CompletionReason::Transfer,
        CompletionReason::Return,
        CompletionReason::Unwind,
    ] {
        assert_eq!(document(&reason), Value::String(format!("{reason:?}")));
        assert_eq!(round_trip(&reason), reason);
    }
    assert!(load::<HandlerKind>(json!("Rethrow")).is_err());
    assert!(load::<CompletionReason>(json!("Exceptional")).is_err());
}

#[test]
fn a_cleanup_record_and_its_continuations_round_trip_in_recorded_order() {
    let cleanup = Cleanup {
        handler: HandlerRef::new(RegionId::from_raw(2), 1),
        resume_from: Some(block(8)),
        continuations: vec![
            Continuation {
                reason: CompletionReason::Return,
                resume: block(11),
            },
            Continuation {
                reason: CompletionReason::Normal,
                resume: block(9),
            },
            Continuation {
                reason: CompletionReason::Return,
                resume: block(10),
            },
        ],
    };

    let decoded: Cleanup = round_trip(&cleanup);
    assert_eq!(decoded, cleanup);
    assert_eq!(
        decoded
            .resumes_for(CompletionReason::Return)
            .collect::<Vec<_>>(),
        [block(11), block(10)],
        "the recorded order is the order a consumer reads routes in"
    );

    let diverging = Cleanup {
        resume_from: None,
        ..cleanup
    };
    assert_eq!(round_trip(&diverging).resume_from, None);
}

#[test]
fn a_handler_reference_keeps_its_region_and_position() {
    let reference = HandlerRef::new(RegionId::from_raw(7), 3);
    let decoded: HandlerRef = round_trip(&reference);
    assert_eq!(
        (decoded.region(), decoded.index()),
        (RegionId::from_raw(7), 3)
    );
}

#[test]
fn identities_are_plain_unsigned_integers() {
    assert_eq!(document(&block(3)), json!(3));
    assert_eq!(document(&RegionId::from_raw(4)), json!(4));
    assert_eq!(
        load::<BlockId>(json!(3)).expect("a small id loads"),
        block(3)
    );
    for hostile in [
        json!(-1),
        json!(4_294_967_296_u64),
        json!("bb3"),
        json!(1.5),
        json!(null),
    ] {
        assert!(load::<BlockId>(hostile.clone()).is_err(), "{hostile}");
        assert!(load::<RegionId>(hostile.clone()).is_err(), "{hostile}");
    }
}

fn handler_ref(region: u32, index: usize) -> HandlerRef {
    HandlerRef::new(RegionId::from_raw(region), index)
}

#[test]
fn handler_metadata_survives_a_format_that_only_admits_string_keys() {
    let mut types = HandlerTypes::<String>::new();
    types.set(handler_ref(1, 0), "System.IO.IOException".to_owned());
    types.set(handler_ref(0, 2), "System.Exception".to_owned());
    types.set(handler_ref(0, 0), "System.ArgumentException".to_owned());

    // JSON objects have string keys only, so a table keyed by a struct would
    // fail to encode at all.
    let decoded: HandlerTypes<String> = round_trip(&types);
    assert_eq!(decoded, types);
    assert_eq!(decoded.len(), 3);
    assert_eq!(
        decoded.iter().map(|(key, _)| key).collect::<Vec<_>>(),
        [handler_ref(0, 0), handler_ref(0, 2), handler_ref(1, 0)],
        "iteration stays in handler order"
    );
    assert_eq!(
        decoded.metadata(handler_ref(0, 2)).map(String::as_str),
        Some("System.Exception")
    );
    assert_eq!(decoded.metadata(handler_ref(0, 1)), None);
}

#[test]
fn a_decoded_metadata_table_keeps_accepting_edits_in_order() {
    let mut filters = HandlerFilters::<u32>::new();
    filters.set(handler_ref(0, 0), 10);
    filters.set(handler_ref(2, 0), 30);
    let mut decoded: HandlerFilters<u32> = round_trip(&filters);

    assert_eq!(decoded.set(handler_ref(1, 0), 20), None);
    assert_eq!(decoded.set(handler_ref(2, 0), 31), Some(30));
    assert_eq!(decoded.remove(handler_ref(0, 0)), Some(10));
    assert_eq!(
        decoded
            .iter()
            .map(|(key, value)| (key, *value))
            .collect::<Vec<_>>(),
        [(handler_ref(1, 0), 20), (handler_ref(2, 0), 31)]
    );
}

#[test]
fn a_hand_written_metadata_table_out_of_handler_order_never_misreports_a_key() {
    // `HandlerMetadata` answers lookups by binary search over entries sorted
    // by key. A document that lists them out of order breaks that invariant:
    // the load must either be rejected or still find every entry it holds.
    let keys = [handler_ref(2, 0), handler_ref(0, 0), handler_ref(1, 0)];
    let mut sorted = HandlerFilters::<u32>::new();
    for (index, key) in keys.into_iter().enumerate() {
        sorted.set(key, u32::try_from(index).expect("a small index"));
    }
    let mut value = document(&sorted);
    let entries = value
        .as_object_mut()
        .and_then(|object| object.values_mut().next())
        .and_then(Value::as_array_mut)
        .expect("the table is one sequence of entries");
    entries.reverse();

    if let Ok(decoded) = load::<HandlerFilters<u32>>(value) {
        for (index, key) in keys.into_iter().enumerate() {
            assert_eq!(
                decoded.metadata(key).copied(),
                Some(u32::try_from(index).expect("a small index")),
                "{key} is in the document but the table cannot find it"
            );
        }
    }
}
