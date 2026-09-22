//! Read-resolver tests: pairing lifted reads with their HLIL operands.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::ir::hlil::{self, ExpressionKind, StatementKind, lift_function as lift_hlil};
use crate::ir::mlil;

use super::super::{ReadResolver, ResolvedRead, Webs, referenced_webs};
use super::{
    Edge, Effect, EffectOp, FunctionBuilder, LiftedStatement, Operator, ScalarType,
    SemanticDialect, Statement, TestDialect, ValueShape, VarExpr, apply, assign, constant, lift,
    read, vector_const,
};

/// Lifts a single-block RTL body through MLIL into HLIL.
fn structured(
    statements: Vec<Statement<TestDialect>>,
) -> (hlil::LiftedFunction<SemanticDialect>, Webs<TestDialect>) {
    let mut builder = FunctionBuilder::<TestDialect>::new("test".into());
    let entry = builder.entry();
    let body = builder.new_block("body");
    builder.add_edge(entry, body, Edge::Entry).unwrap();
    for statement in statements {
        builder.append(body, statement, None).unwrap();
    }
    builder
        .append(body, Statement::Return { values: Vec::new() }, None)
        .unwrap();
    let function = builder.finish().unwrap();
    let lifting = lift(&function, &()).unwrap();
    let webs = lifting.webs;
    let function = lifting.builder.finish().unwrap();
    let lifted = lift_hlil(&function).unwrap();
    assert!(lifted.report.is_fully_structured());
    (lifted, webs)
}

fn emit_twice(storage: u8, scalar: ScalarType) -> Statement<TestDialect> {
    Statement::Effect {
        writes: Vec::new(),
        operation: EffectOp::Emit,
        operands: vec![read(storage, &[0], scalar), read(storage, &[0], scalar)],
        effects: vec![Effect::Emit],
        may_throw: false,
    }
}

/// A read of an inlined producer resolves with the position remap the
/// producer's written order dictates.
#[test]
fn read_resolver_remaps_through_an_inlined_producer() {
    // r0.xy ← add (single use, inlines); r1.xy ← r0.yx (swapped read).
    let (lifted, webs) = structured(vec![
        assign(
            0,
            &[0, 1],
            apply(
                Operator::Add,
                vec![
                    vector_const(&[1, 2], ScalarType::F32),
                    vector_const(&[3, 4], ScalarType::F32),
                ],
                ValueShape::vector(ScalarType::F32, 2),
            ),
        ),
        assign(1, &[0, 1], read(0, &[1, 0], ScalarType::F32)),
        emit_twice(1, ScalarType::F32),
    ]);
    let assignment = lifted
        .function
        .statements()
        .iter()
        .find_map(|statement| match statement.kind() {
            StatementKind::Assign { value, .. } => lifted.function.expression(*value),
            _ => None,
        })
        .expect("the r1 assignment survives as a statement");
    let ExpressionKind::Operation {
        operation:
            LiftedStatement::Assign {
                value: VarExpr::Read { positions, .. },
                ..
            },
        operands,
    } = assignment.kind()
    else {
        panic!("expected an assignment of a read");
    };
    assert_eq!(positions.as_slice(), &[1, 0]);
    let mut reads = ReadResolver::new(&lifted.function, operands);
    let ResolvedRead::Inlined { value, remap, .. } = reads.resolve(positions).unwrap() else {
        panic!("single-use producer inlines");
    };
    assert!(matches!(
        value,
        VarExpr::Apply {
            operator: Operator::Add,
            ..
        }
    ));
    assert_eq!(remap, Some(vec![1, 0]));
    // The inlined producer's target never renders: it is not referenced.
    let referenced = referenced_webs(&lifted.function);
    let r0_web = webs.iter().find(|web| web.storage == Some(0)).unwrap();
    let r1_web = webs.iter().find(|web| web.storage == Some(1)).unwrap();
    assert!(!referenced.contains(&r0_web.variable));
    assert!(referenced.contains(&r1_web.variable));
}

/// A read of a multi-use producer resolves as a variable occurrence, and
/// the webs resolve by both levels' variable identities.
#[test]
fn read_resolver_names_multi_use_producers() {
    let (lifted, webs) = structured(vec![
        assign(
            0,
            &[0],
            apply(
                Operator::Add,
                vec![constant(1, ScalarType::U32), constant(2, ScalarType::U32)],
                ValueShape::scalar(ScalarType::U32),
            ),
        ),
        assign(1, &[0], read(0, &[0], ScalarType::U32)),
        assign(2, &[0], read(0, &[0], ScalarType::U32)),
        emit_twice(1, ScalarType::U32),
        emit_twice(2, ScalarType::U32),
    ]);
    let mut resolved = Vec::new();
    for statement in lifted.function.statements() {
        let StatementKind::Assign { value, .. } = statement.kind() else {
            continue;
        };
        let Some(expression) = lifted.function.expression(*value) else {
            continue;
        };
        let ExpressionKind::Operation {
            operation:
                LiftedStatement::Assign {
                    value: VarExpr::Read { positions, .. },
                    ..
                },
            operands,
        } = expression.kind()
        else {
            continue;
        };
        let mut reads = ReadResolver::new(&lifted.function, operands);
        resolved.push(reads.resolve(positions).unwrap());
    }
    let r0_web = webs.iter().find(|web| web.storage == Some(0)).unwrap();
    assert_eq!(resolved.len(), 2, "both reads of r0 stay variable reads");
    for entry in resolved {
        let ResolvedRead::Variable(variable) = entry else {
            panic!("multi-use producer must not inline");
        };
        assert_eq!(webs.of_lifted(variable).unwrap().variable, r0_web.variable);
        assert_eq!(
            webs.of(mlil::VariableId::from_raw(variable.raw()))
                .unwrap()
                .variable,
            r0_web.variable
        );
    }
    let referenced = referenced_webs(&lifted.function);
    assert!(referenced.contains(&r0_web.variable));
}
