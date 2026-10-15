//! The pseudocode writer.

extern crate alloc;

use alloc::string::String;

use crate::ir::hlil::{FunctionBuilder, Handler, HandlerKind, pseudocode, write_pseudocode};
use crate::test_util::golden::assert_golden;

use super::fixtures::{bump, every_statement_shape};
use super::{Imp, Kit, Prim, ToyHlilBuilder, Ty};

over_toys!(
    builder_constructs_verifies_and_renders,
    every_statement_shape_renders,
    pseudocode_writes_into_any_sink,
);

fn builder_constructs_verifies_and_renders<D: Kit>() {
    let function = bump::<D>();

    assert_golden(
        "hlil/builder-if-loop.pseudo",
        &pseudocode::<D, _>(&function),
    );
}

fn pseudocode_writes_into_any_sink<D: Kit>() {
    let function = bump::<D>();
    let mut sink = String::new();

    write_pseudocode::<D, _>(&function, &mut sink).unwrap();

    assert_eq!(sink, pseudocode::<D, _>(&function));
}

fn every_statement_shape_renders<D: Kit>() {
    let function = every_statement_shape::<D>();

    assert_golden(
        "hlil/every-statement-shape.pseudo",
        &pseudocode::<D, _>(&function),
    );
}

#[test]
fn caught_types_print_in_debug_form() {
    let mut builder = ToyHlilBuilder::<Imp>::new();
    let work = builder.effect(Prim::Call, &[]);
    let handler = Handler {
        kind: HandlerKind::Catch,
        binding: None,
        caught_types: &[Imp::ty(Ty::Integer), Imp::ty(Ty::Boolean)],
        body: &[],
    };
    let guarded = builder.try_(&[work], &[handler], &[]);
    let function = builder.finish(&[guarded]).unwrap();

    assert_eq!(
        pseudocode::<Imp, _>(&function),
        "try {\n    call();\n} catch (Integer | Boolean) {\n}\n"
    );
}
