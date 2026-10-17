//! Process-wide vectored exception and continue handlers.

use std::collections::{HashSet, VecDeque};

use VectoredHandlerKind::{Continue, Exception};
use VectoredHandlerOrder::{First, Last};
use cfglib::{
    VectoredExceptionModel, VectoredHandler, VectoredHandlerId, VectoredHandlerKind,
    VectoredHandlerOrder, VehModel,
};

fn exception_order(model: &VehModel<&'static str>) -> Vec<&'static str> {
    model
        .handlers(VectoredHandlerKind::Exception)
        .map(|registration| registration.handler)
        .collect()
}

fn continue_order(model: &VehModel<&'static str>) -> Vec<&'static str> {
    model
        .handlers(VectoredHandlerKind::Continue)
        .map(|registration| registration.handler)
        .collect()
}

#[test]
fn a_later_first_registration_precedes_every_earlier_one_in_its_own_list() {
    let mut model = VehModel::<&'static str>::new();
    assert!(model.is_empty());
    model.register(Exception, Last, "last-1");
    model.register(Exception, First, "first-1");
    model.register(Continue, Last, "continue-last-1");
    model.register(Exception, First, "first-2");
    model.register(Exception, Last, "last-2");
    model.register(Continue, First, "continue-first-1");

    assert_eq!(
        exception_order(&model),
        ["first-2", "first-1", "last-1", "last-2"],
        "First pushes to the front, Last to the back"
    );
    // The continue list is independent of the exception list.
    assert_eq!(
        continue_order(&model),
        ["continue-first-1", "continue-last-1"]
    );
    assert_eq!(model.len(), 6);
    assert!(!model.is_empty());
    assert_eq!(model.handlers(Exception).len(), 4);
}

#[test]
fn vectored_ids_are_unique_stale_ids_remove_nothing_and_removal_spans_both_lists() {
    let mut model = VectoredExceptionModel::new();
    let tail = model.register(
        VectoredHandlerKind::Exception,
        VectoredHandlerOrder::Last,
        "tail",
    );
    let observer = model.register(
        VectoredHandlerKind::Continue,
        VectoredHandlerOrder::Last,
        "observer",
    );
    let head = model.register(
        VectoredHandlerKind::Exception,
        VectoredHandlerOrder::First,
        "head",
    );
    let ids = [tail, observer, head];
    assert_eq!(
        ids.iter().collect::<HashSet<_>>().len(),
        3,
        "ids are distinct"
    );
    for id in ids {
        assert_eq!(VectoredHandlerId::from_raw(id.raw()), id);
    }

    // Removal finds a registration in either list and returns all of it.
    assert_eq!(
        model.remove(observer),
        Some(VectoredHandler {
            id: observer,
            kind: VectoredHandlerKind::Continue,
            handler: "observer",
        })
    );
    assert_eq!(model.len(), 2);
    assert_eq!(
        model.remove(observer),
        None,
        "a registration is removed once"
    );

    // A later registration never inherits a removed id, so a stale id cannot
    // unregister somebody else's handler.
    let replacement = model.register(
        VectoredHandlerKind::Continue,
        VectoredHandlerOrder::Last,
        "replacement",
    );
    assert_ne!(replacement, observer);
    assert_eq!(model.remove(observer), None);
    assert_eq!(model.len(), 3);
    assert_eq!(model.remove(head).map(|entry| entry.handler), Some("head"));
    assert_eq!(model.remove(tail).map(|entry| entry.handler), Some("tail"));
    assert_eq!(
        model.remove(replacement).map(|entry| entry.handler),
        Some("replacement")
    );
    assert!(model.is_empty());
    assert_eq!(model.len(), 0);

    let default: VectoredExceptionModel<u8> = VectoredExceptionModel::default();
    assert!(default.is_empty());
}

#[test]
fn the_vectored_model_matches_a_deque_model_under_random_registrations_and_removals() {
    let mut model = VehModel::new();
    // Reference: one deque per list, with the registration id and handler.
    let mut exception: VecDeque<(VectoredHandlerId, u32)> = VecDeque::new();
    let mut continuing: VecDeque<(VectoredHandlerId, u32)> = VecDeque::new();
    let mut issued: Vec<VectoredHandlerId> = Vec::new();
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    for step in 0..600_u32 {
        let roll = next();
        if roll.is_multiple_of(4) && !issued.is_empty() {
            // Remove a random id, which may already be gone.
            let id = issued[usize::try_from(next() % issued.len() as u64).unwrap()];
            let expected = [&mut exception, &mut continuing]
                .into_iter()
                .find_map(|list| {
                    let at = list.iter().position(|&(listed, _)| listed == id)?;
                    list.remove(at)
                });
            assert_eq!(
                model.remove(id).map(|entry| (entry.id, entry.handler)),
                expected
            );
        } else {
            let (kind, list) = if roll.is_multiple_of(2) {
                (VectoredHandlerKind::Exception, &mut exception)
            } else {
                (VectoredHandlerKind::Continue, &mut continuing)
            };
            let order = if roll.is_multiple_of(3) {
                VectoredHandlerOrder::First
            } else {
                VectoredHandlerOrder::Last
            };
            let id = model.register(kind, order, step);
            assert!(!issued.contains(&id), "id {id:?} was issued twice");
            issued.push(id);
            match order {
                VectoredHandlerOrder::First => list.push_front((id, step)),
                VectoredHandlerOrder::Last => list.push_back((id, step)),
            }
        }
        let listed = |kind| -> Vec<(VectoredHandlerId, u32)> {
            model
                .handlers(kind)
                .map(|entry| (entry.id, entry.handler))
                .collect()
        };
        assert_eq!(
            listed(VectoredHandlerKind::Exception),
            Vec::from(exception.clone())
        );
        assert_eq!(
            listed(VectoredHandlerKind::Continue),
            Vec::from(continuing.clone())
        );
        assert_eq!(model.len(), exception.len() + continuing.len());
    }
}
