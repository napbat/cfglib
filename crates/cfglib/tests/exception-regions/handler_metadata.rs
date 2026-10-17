//! Consumer-owned handler data: the side table that keeps catch types and
//! filter predicates beside the CFG, keyed by handler identity.

use cfglib::{
    ControlNode, ControlTree, EdgeKind, HandlerFilters, HandlerKind, HandlerMetadata, HandlerRef,
    HandlerTypes, RegionId,
};

use crate::shapes::{Stmt, block_of, catch, protect, try_catch};

fn key(region: u32, index: usize) -> HandlerRef {
    HandlerRef::new(RegionId::from_raw(region), index)
}

#[test]
fn setting_a_payload_returns_the_one_it_replaced() {
    let mut types = HandlerTypes::new();

    assert_eq!(types.set(key(0, 0), "IOException"), None);
    assert_eq!(types.set(key(0, 1), "ArgumentException"), None);
    assert_eq!(
        types.set(key(0, 0), "FileNotFoundException"),
        Some("IOException")
    );

    assert_eq!(types.len(), 2);
    assert!(!types.is_empty());
    assert_eq!(types.metadata(key(0, 0)), Some(&"FileNotFoundException"));
    assert_eq!(types.metadata(key(1, 0)), None);
}

#[test]
fn iteration_is_in_handler_order_whatever_the_order_of_insertion() {
    let mut filters = HandlerFilters::new();
    for (region, index, filter) in [
        (1, 0, "g"),
        (0, 2, "c"),
        (0, 0, "a"),
        (1, 3, "h"),
        (0, 1, "b"),
    ] {
        filters.set(key(region, index), filter);
    }

    let in_order: Vec<(HandlerRef, &str)> = filters
        .iter()
        .map(|(handler, filter)| (handler, *filter))
        .collect();

    assert_eq!(
        in_order,
        [
            (key(0, 0), "a"),
            (key(0, 1), "b"),
            (key(0, 2), "c"),
            (key(1, 0), "g"),
            (key(1, 3), "h"),
        ]
    );
}

#[test]
fn a_payload_is_edited_in_place_and_removed_by_handler() {
    let mut types = HandlerMetadata::<String>::default();
    assert!(types.is_empty());
    types.set(key(0, 0), String::from("IOException"));

    types
        .metadata_mut(key(0, 0))
        .expect("present")
        .push_str("Subtype");
    assert!(types.metadata_mut(key(0, 1)).is_none());

    assert_eq!(
        types.metadata(key(0, 0)).map(String::as_str),
        Some("IOExceptionSubtype")
    );
    assert_eq!(
        types.remove(key(0, 0)).as_deref(),
        Some("IOExceptionSubtype")
    );
    assert_eq!(types.remove(key(0, 0)), None);
    assert!(types.is_empty());
}

#[test]
fn the_filter_and_type_aliases_are_the_same_table() {
    let mut filters: HandlerFilters<u32> = HandlerFilters::new();
    filters.set(key(0, 0), 17);
    let as_types: HandlerTypes<u32> = filters.clone();

    assert_eq!(as_types, filters);
}

#[test]
fn a_consumer_reads_a_catch_type_back_through_the_tree_by_handler_identity() {
    // try { risky() } catch (IOException e) { .. } catch (ArgumentException e) { .. }
    let mut cfg = cfglib::Cfg::<Stmt>::new();
    let entry = cfg.entry();
    cfg.block_mut(entry).push(Stmt::text("risky()"));
    let io = block_of(&mut cfg, &["on_io(e)"]);
    let argument = block_of(&mut cfg, &["on_argument(e)"]);
    let after = block_of(&mut cfg, &["return"]);
    cfg.add_edge(entry, after, EdgeKind::Fallthrough);
    cfg.add_edge(entry, io, EdgeKind::ExceptionHandler);
    cfg.add_edge(entry, argument, EdgeKind::ExceptionHandler);
    cfg.add_edge(io, after, EdgeKind::Fallthrough);
    cfg.add_edge(argument, after, EdgeKind::Fallthrough);
    let region = protect(&mut cfg, &[entry], vec![catch(io), catch(argument)]);
    let mut types = HandlerTypes::new();
    types.set(HandlerRef::new(region, 0), "IOException");
    types.set(HandlerRef::new(region, 1), "ArgumentException");

    let tree = ControlTree::compute(&cfg);

    let handlers = tree
        .preorder()
        .find_map(|id| match *tree.node(id) {
            ControlNode::TryCatch { handlers, .. } => Some(handlers),
            _ => None,
        })
        .expect("the region is a try/catch");
    let caught: Vec<(&str, &str)> = tree
        .handlers(handlers)
        .iter()
        .map(|arm| {
            assert_eq!(arm.kind, HandlerKind::Catch);
            let ty = *types.metadata(arm.handler).expect("every arm is typed");
            let first = cfg.block(arm.entry).instructions()[0].as_str();
            (ty, first)
        })
        .collect();
    assert_eq!(
        caught,
        [
            ("IOException", "on_io(e)"),
            ("ArgumentException", "on_argument(e)")
        ]
    );
}

#[test]
fn a_filtered_catch_with_no_funclet_is_a_plain_catch_carrying_its_predicate() {
    // C# `catch (E e) when (cond)`: the predicate runs in the pad, so the
    // handler is a `Catch` and the predicate is the frontend's own handle.
    let fixture = try_catch(&["log(e)"]);
    let handler = HandlerRef::new(fixture.region, 0);
    let mut filters = HandlerFilters::new();

    filters.set(handler, "expr#17");

    assert_eq!(
        fixture.cfg.handler(handler).map(|found| found.kind),
        Some(HandlerKind::Catch)
    );
    assert_eq!(filters.metadata(handler), Some(&"expr#17"));
}
