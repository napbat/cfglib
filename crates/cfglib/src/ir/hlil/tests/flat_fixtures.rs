//! Flat functions that several test modules lift, lower, and lift again.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::region::{Handler as RegionHandler, HandlerBody, HandlerKind, Region, RegionId};

use super::Prim;
use super::flat::{FlatKit, ToyEdge, ToyFlat, ToyFlatBuilder, ToyForm};
use super::flat_lift::{boolean, int};
use crate::ir::hlil::FlatBuilder;

/// `while (i < n) { i = i + 1 } return i;` in flat form.
pub(crate) fn counting_loop<D: FlatKit>() -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let header = flat.new_block(Some("header"));
    let body = flat.new_block(Some("body"));
    let exit = flat.new_block(Some("exit"));
    let i = flat.int_variable(0);
    let n = flat.int_variable(0);
    let condition = flat.int_variable(1);
    let one = flat.int_variable(1);

    flat.operate(
        header,
        Prim::LessThan,
        &[int::<D>(i), int::<D>(n)],
        &[boolean::<D>(condition)],
    );
    flat.branch(header, condition);
    flat.literal(body, 1, one);
    flat.operate(
        body,
        Prim::Add,
        &[int::<D>(i), int::<D>(one)],
        &[int::<D>(i)],
    );
    flat.jump(body);
    flat.give_back(exit, &[i]);
    flat.start(header);
    flat.connect(header, body, ToyEdge::True);
    flat.connect(header, exit, ToyEdge::False);
    flat.connect(body, header, ToyEdge::Fall);
    flat.takes(&[i, n]);
    flat.finish().expect("the counting loop is well formed")
}

/// `protected: x = call() [throws]; after: return x`, with a catch-all
/// handler `pad: return 7`.
pub(crate) fn guarded_call<D: FlatKit>() -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let protected = flat.new_block(Some("protected"));
    let pad = flat.new_block(Some("pad"));
    let after = flat.new_block(Some("after"));
    let x = flat.int_variable(0);
    let fallback = flat.int_variable(0);

    flat.operate(protected, Prim::Call, &[], &[int::<D>(x)]);
    flat.literal(pad, 7, fallback);
    flat.give_back(pad, &[fallback]);
    flat.give_back(after, &[x]);
    flat.start(protected);
    flat.connect(protected, after, ToyEdge::Fall);
    flat.connect(protected, pad, ToyEdge::Except);
    flat.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [protected].into_iter().collect(),
        handlers: vec![RegionHandler {
            entry: pad,
            body: HandlerBody::known([pad]),
            kind: HandlerKind::CatchAll,
        }],
        parent: None,
    });
    flat.finish().expect("the guarded call is well formed")
}

/// A pre-tested loop whose conditional break routes through a block holding
/// an explicit jump instruction, with that block claimed by an exception
/// region's protected set: javac shapes both ways.
pub(crate) fn claimed_trampoline_loop<D: FlatKit>() -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let header = flat.new_block(Some("header"));
    let body = flat.new_block(Some("body"));
    let latch = flat.new_block(Some("latch"));
    let tramp = flat.new_block(Some("tramp"));
    let exit = flat.new_block(Some("exit"));
    let pad = flat.new_block(Some("pad"));
    let i = flat.int_variable(0);
    let n = flat.int_variable(0);
    let keep_going = flat.int_variable(1);
    let leave = flat.int_variable(1);
    let one = flat.int_variable(1);

    flat.operate(
        header,
        Prim::LessThan,
        &[int::<D>(i), int::<D>(n)],
        &[boolean::<D>(keep_going)],
    );
    flat.branch(header, keep_going);
    flat.operate(
        body,
        Prim::LessThan,
        &[int::<D>(n), int::<D>(i)],
        &[boolean::<D>(leave)],
    );
    flat.branch(body, leave);
    flat.literal(latch, 1, one);
    flat.operate(
        latch,
        Prim::Add,
        &[int::<D>(i), int::<D>(one)],
        &[int::<D>(i)],
    );
    flat.jump(tramp);
    flat.give_back(exit, &[i]);
    flat.give_back(pad, &[i]);
    flat.start(header);
    flat.connect(header, body, ToyEdge::True);
    flat.connect(header, exit, ToyEdge::False);
    flat.connect(body, tramp, ToyEdge::True);
    flat.connect(body, latch, ToyEdge::False);
    flat.connect(latch, header, ToyEdge::Fall);
    flat.connect(tramp, exit, ToyEdge::Jump);
    flat.add_region(Region {
        id: RegionId::from_raw(0),
        protected_blocks: [tramp].into_iter().collect(),
        handlers: vec![RegionHandler {
            entry: pad,
            body: HandlerBody::Unknown,
            kind: HandlerKind::CatchAll,
        }],
        parent: None,
    });
    flat.takes(&[i, n]);
    flat.finish_unchecked()
}

/// A dispatch on `v0` whose arms assign `v1`: the values 1 and 2 through two
/// case edges to one arm, 3 to a second arm, and a default arm.
pub(crate) fn dispatch_with_default<D: FlatKit>() -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let dispatch = flat.new_block(Some("dispatch"));
    let case_one = flat.new_block(Some("one"));
    let case_two = flat.new_block(Some("two"));
    let fallback = flat.new_block(Some("fallback"));
    let merge = flat.new_block(Some("merge"));
    let selector = flat.int_variable(0);
    let result = flat.int_variable(0);
    flat.emit(dispatch, ToyForm::Switch, &[int::<D>(selector)], &[]);
    for (block, value) in [(case_one, 10), (case_two, 20), (fallback, 30)] {
        flat.literal(block, value, result);
        flat.connect(block, merge, ToyEdge::Fall);
    }
    flat.give_back(merge, &[result]);
    flat.start(dispatch);
    for (target, value) in [(case_one, 1), (case_one, 2), (case_two, 3)] {
        flat.connect(dispatch, target, ToyEdge::Case(vec![D::constant(value)]));
    }
    flat.connect(dispatch, fallback, ToyEdge::Default);
    flat.takes(&[selector]);
    flat.finish().expect("the dispatch is well formed")
}

/// `if (c) { if (c) { ... out = 7 ... } }` with `depth` ifs, and `return out`
/// after them, in flat form: each head branches into the next head, or into
/// the assignment at the innermost, and to the shared exit otherwise.
pub(crate) fn nested_ifs<D: FlatKit>(depth: usize) -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let condition = flat.int_variable(0);
    let out = flat.int_variable(1);
    let exit = flat.new_block(None);
    let core = flat.new_block(None);
    let heads: Vec<_> = (0..depth).map(|_| flat.new_block(None)).collect();
    flat.literal(core, 7, out);
    flat.jump(core);
    flat.give_back(exit, &[out]);
    for (index, &head) in heads.iter().enumerate() {
        flat.branch(head, condition);
        flat.connect(
            head,
            heads.get(index + 1).copied().unwrap_or(core),
            ToyEdge::True,
        );
        flat.connect(head, exit, ToyEdge::False);
    }
    flat.connect(core, exit, ToyEdge::Fall);
    flat.start(heads[0]);
    flat.takes(&[condition]);
    flat.finish_unchecked()
}

/// `while (c) { while (c) { ... out = 7 ... } }` with `depth` loops, and
/// `return out` after them, in flat form: each header branches into the next
/// header, or into the assignment at the innermost, and to a jump block that
/// returns to the header outside it.
pub(crate) fn nested_loops<D: FlatKit>(depth: usize) -> ToyFlat<D> {
    let mut flat = ToyFlatBuilder::<D>::new();
    let condition = flat.int_variable(0);
    let out = flat.int_variable(1);
    let exit = flat.new_block(None);
    let core = flat.new_block(None);
    let heads: Vec<_> = (0..depth).map(|_| flat.new_block(None)).collect();
    flat.literal(core, 7, out);
    flat.jump(core);
    flat.give_back(exit, &[out]);
    for (index, &head) in heads.iter().enumerate() {
        let leave = flat.new_block(None);
        flat.branch(head, condition);
        flat.jump(leave);
        flat.connect(
            head,
            heads.get(index + 1).copied().unwrap_or(core),
            ToyEdge::True,
        );
        flat.connect(head, leave, ToyEdge::False);
        let outside = index.checked_sub(1).map_or(exit, |outer| heads[outer]);
        flat.connect(leave, outside, ToyEdge::Fall);
    }
    flat.connect(core, heads[depth - 1], ToyEdge::Fall);
    flat.start(heads[0]);
    flat.takes(&[condition]);
    flat.finish_unchecked()
}
