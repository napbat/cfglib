//! `StatementTree`: expression trees, shared values, and select-of-self,
//! checked through a small renderer of the kind that a plugin writes.

use std::fmt::Write as _;

use cfglib::ir::rtl::{Bits, Control, InstructionBuilder, Place, Raise, Selection};
use cfglib::{StatementTree, TreeItem, TreeNode, TreeNodeId, TreeSpanValue};

use crate::wren::{
    Eff, Fault, Op, Reg, Space, Undef, Wren, WrenEffects, bits, capture, constant, keep, part,
    take, undefined, write_plan, write_whole,
};
use cfglib::{CapturedValue, DenseId};

/// Renders one node: the name of a named node, or the node inline.
fn node(tree: &StatementTree<'_, Wren>, id: TreeNodeId) -> String {
    if tree.is_named(id) {
        return match tree.node(id) {
            TreeNode::Effect {
                ordinal, result, ..
            } => format!("e{ordinal}.{result}"),
            _ => format!("t{}", id.index()),
        };
    }
    inline(tree, id)
}

/// Renders the definition of one node.
fn inline(tree: &StatementTree<'_, Wren>, id: TreeNodeId) -> String {
    match tree.node(id) {
        TreeNode::Read(Selection::Direct(place)) => place_name(place),
        TreeNode::Read(Selection::FiniteIndexed { index, .. }) => {
            format!("slots[{}]", node(tree, index))
        }
        TreeNode::Read(Selection::SparseIndexed { key, .. }) => {
            format!("bank[{}]", node(tree, key))
        }
        TreeNode::Const { words, .. } => format!("{}", words[0]),
        TreeNode::Apply {
            operator,
            args,
            result,
            ..
        } => {
            let args: Vec<String> = args.iter().map(|&arg| node(tree, arg)).collect();
            let name = format!("{operator:?}").to_lowercase();
            if matches!(operator, Op::DivMod) {
                format!("{name}.{result}({})", args.join(", "))
            } else {
                format!("{name}({})", args.join(", "))
            }
        }
        TreeNode::Nondet { .. } => "undefined".to_owned(),
        TreeNode::Effect {
            ordinal, result, ..
        } => format!("e{ordinal}.{result}"),
    }
}

fn place_name(place: Place<Wren>) -> String {
    if place == Place::whole(place.root) {
        format!("{:?}", place.root)
    } else {
        format!(
            "{:?}[{}+:{}]",
            place.root,
            place.slice.offset,
            place.slice.width.width()
        )
    }
}

/// Renders every item, one line each.
fn render(tree: &StatementTree<'_, Wren>) -> Vec<String> {
    let mut lines = Vec::new();
    for item in tree.items() {
        let mut line = String::new();
        match item {
            TreeItem::Let(id) => write!(line, "let t{} = {}", id.index(), inline(tree, id)),
            TreeItem::Effect {
                operation,
                args,
                results,
                ..
            } => {
                let args: Vec<String> = args.iter().map(|&arg| node(tree, arg)).collect();
                let results: Vec<String> =
                    results.iter().map(|&result| node(tree, result)).collect();
                write!(
                    line,
                    "[{}] = {operation:?}({})",
                    results.join(", "),
                    args.join(", ")
                )
            }
            TreeItem::Raise { outcome, guard, .. } => match guard {
                Some(guard) => write!(line, "raise {outcome:?} if {}", node(tree, guard)),
                None => write!(line, "raise {outcome:?}"),
            },
            TreeItem::Write { root, spans, .. } => {
                let parts: Vec<String> = spans
                    .iter()
                    .filter_map(|span| {
                        let target = place_name(Place {
                            root,
                            slice: span.span,
                        });
                        match span.value {
                            TreeSpanValue::Keep => None,
                            TreeSpanValue::Assign { value, .. } => {
                                Some(format!("{target} = {}", node(tree, value)))
                            }
                            TreeSpanValue::Conditional { guard, value, .. } => Some(format!(
                                "if {} {target} = {}",
                                node(tree, guard),
                                node(tree, value)
                            )),
                            TreeSpanValue::Undefined(_) => Some(format!("{target} = undefined")),
                        }
                    })
                    .collect();
                write!(line, "{}", parts.join("; "))
            }
            TreeItem::FiniteWrite { index, value, .. } => {
                write!(line, "slots[{}] = {}", node(tree, index), node(tree, value))
            }
            TreeItem::SparseWrite { key, value, .. } => {
                write!(line, "bank[{}] = {}", node(tree, key), node(tree, value))
            }
            TreeItem::Control(control) => match control {
                Control::Advance => write!(line, "next"),
                Control::Branch {
                    condition, taken, ..
                } => write!(
                    line,
                    "if {} goto {}",
                    node(tree, condition),
                    node(tree, taken)
                ),
                Control::Call { target } => write!(line, "call {}", node(tree, target)),
                other => write!(line, "{other:?}"),
            },
        }
        .expect("a string accepts every write");
        lines.push(line);
    }
    lines
}

/// Captures one instruction and renders its tree.
fn lines_of(
    emit: impl FnOnce(&mut &mut cfglib::Captured<Wren>) -> Control<CapturedValue>,
) -> Vec<String> {
    let captured = capture(emit);
    render(&StatementTree::compute_record(&WrenEffects, &captured))
}

#[test]
fn a_value_with_one_use_stays_inline() {
    let lines = lines_of(|b| {
        let a = b.read(Selection::Direct(Place::whole(Reg::A)));
        let one = constant(b, 1, 32);
        let sum = b.apply(&Op::Add, &[a, one], &[bits(32)]).first;
        write_whole(b, Reg::A, sum);
        Control::Advance
    });
    assert_eq!(lines, ["A = add(A, 1)", "next"]);
}

#[test]
fn a_shared_value_gets_a_let_before_its_first_use_after_the_lets_inside_it() {
    let lines = lines_of(|b| {
        let a = b.read(Selection::Direct(Place::whole(Reg::A)));
        let c = b.read(Selection::Direct(Place::whole(Reg::B)));
        let sum = b.apply(&Op::Add, &[a, c], &[bits(32)]).first;
        let difference = b.apply(&Op::Sub, &[sum, sum], &[bits(32)]).first;
        let low = b.apply(&Op::And, &[difference, a], &[bits(32)]).first;
        write_whole(b, Reg::A, difference);
        write_whole(b, Reg::B, difference);
        write_plan(b, Reg::Slot(0), &[take(0, 16, low, 0)]);
        Control::Advance
    });
    assert_eq!(
        lines,
        [
            "let t2 = add(A, B)",
            "let t3 = sub(t2, t2)",
            "A = t3",
            "B = t3",
            "Slot(0) = and(t3, A)",
            "next",
        ]
    );
}

#[test]
fn a_shared_read_and_a_shared_constant_stay_inline() {
    let lines = lines_of(|b| {
        let a = b.read(Selection::Direct(part(Reg::A, 0, 16)));
        let two = constant(b, 2, 16);
        let sum = b.apply(&Op::Add, &[a, two], &[bits(16)]).first;
        let product = b.apply(&Op::Sub, &[a, two], &[bits(16)]).first;
        write_plan(b, Reg::Slot(0), &[take(0, 16, sum, 0)]);
        write_plan(b, Reg::Slot(1), &[take(0, 16, product, 0)]);
        Control::Advance
    });
    assert_eq!(
        lines,
        [
            "Slot(0) = add(A[0+:16], 2)",
            "Slot(1) = sub(A[0+:16], 2)",
            "next",
        ]
    );
}

#[test]
fn a_shared_choice_keeps_one_name() {
    let lines = lines_of(|b| {
        let choice = b.nondet(Bits::BOOL, Undef::Flags);
        write_whole(b, Reg::Z, choice);
        write_whole(b, Reg::C, choice);
        write_plan(b, Reg::Check, &[undefined(0, 1)]);
        Control::Advance
    });
    assert_eq!(
        lines,
        [
            "let t0 = undefined",
            "Z = t0",
            "C = t0",
            "Check = undefined",
            "next",
        ]
    );
}

#[test]
fn a_select_of_the_old_bits_is_a_conditional_assignment() {
    let captured = capture(|b| {
        let guard = b.read(Selection::Direct(Place::whole(Reg::Z)));
        let new = b.read(Selection::Direct(part(Reg::B, 0, 16)));
        let old = b.read(Selection::Direct(part(Reg::A, 0, 16)));
        let chosen = b.apply(&Op::Select, &[guard, new, old], &[bits(16)]).first;
        let zero = constant(b, 0, 16);
        write_plan(b, Reg::A, &[take(0, 16, chosen, 0), take(16, 16, zero, 0)]);
        Control::Advance
    });
    let tree = StatementTree::compute_record(&WrenEffects, &captured);
    assert_eq!(
        render(&tree),
        ["if Z A[0+:16] = B[0+:16]; A[16+:16] = 0", "next"]
    );
    // The select and the read of the old bits are not uses.
    let select = TreeNodeId::from_index(3);
    let old = TreeNodeId::from_index(2);
    assert!(matches!(
        tree.node(select),
        TreeNode::Apply {
            operator: Op::Select,
            ..
        }
    ));
    assert_eq!(tree.use_count(select), 0);
    assert_eq!(tree.use_count(old), 0);
    let Some(TreeItem::Write { spans, .. }) = tree.items().next() else {
        panic!("the first item is the write");
    };
    assert_eq!(
        spans[0].value,
        TreeSpanValue::Conditional {
            guard: TreeNodeId::from_index(0),
            value: TreeNodeId::from_index(1),
            offset: 0,
        }
    );
}

#[test]
fn effects_name_their_results_and_items_keep_program_order() {
    let lines = lines_of(|b| {
        let base = b.read(Selection::Direct(Place::whole(Reg::A)));
        let four = constant(b, 4, 32);
        let address = b.apply(&Op::Add, &[base, four], &[bits(32)]).first;
        let loaded = b
            .effect(
                &Eff::Load(Space::Data),
                &[address],
                &[bits(32)],
                &[Fault::Page],
            )
            .first;
        let zero = constant(b, 0, 32);
        let is_zero = b.apply(&Op::Eq, &[loaded, zero], &[Bits::BOOL]).first;
        b.raise(Raise {
            outcome: Fault::Divide,
            guard: Some(is_zero),
            payload: &[],
            binds: &[],
        });
        let pair = b.apply(&Op::DivMod, &[base, loaded], &[bits(32), bits(32)]);
        write_whole(b, Reg::A, pair.first);
        write_whole(b, Reg::B, pair.get(1).expect("two results"));
        b.effect(&Eff::Store(Space::Stack), &[address, base], &[], &[]);
        Control::Call { target: address }
    });
    assert_eq!(
        lines,
        [
            "let t2 = add(A, 4)",
            "[e0.0] = Load(Data)(t2)",
            "raise Divide if eq(e0.0, 0)",
            "A = divmod.0(A, e0.0)",
            "B = divmod.1(A, e0.0)",
            "[] = Store(Stack)(t2, A)",
            "call t2",
        ]
    );
}

#[test]
fn indexed_writes_and_a_branch_render_their_trees() {
    let candidates = [Place::whole(Reg::Slot(0)), Place::whole(Reg::Slot(1))];
    let lines = lines_of(|b| {
        let index = b.read(Selection::Direct(Place::whole(Reg::Z)));
        let slot = b.read(Selection::FiniteIndexed {
            index,
            candidates: &candidates,
        });
        let key = b.read(Selection::Direct(Place::whole(Reg::B)));
        let element = b.read(Selection::SparseIndexed {
            bank: Reg::Bank,
            key,
            slice: cfglib::ir::rtl::BitSlice::new(0, bits(16)),
        });
        b.write(
            cfglib::ir::rtl::WriteTarget::FiniteIndexed {
                index,
                candidates: &candidates,
                value: element,
            },
            crate::wren::Policy::Merge,
        );
        b.write(
            cfglib::ir::rtl::WriteTarget::SparseIndexed {
                bank: Reg::Bank,
                key,
                slice: cfglib::ir::rtl::BitSlice::new(16, bits(16)),
                value: slot,
            },
            crate::wren::Policy::Merge,
        );
        write_plan(b, Reg::A, &[keep(0, 32)]);
        let not = b.apply(&Op::Not, &[index], &[Bits::BOOL]).first;
        Control::Branch {
            condition: not,
            taken: key,
            not_taken: None,
        }
    });
    assert_eq!(
        lines,
        [
            "slots[Z] = bank[B]",
            "bank[B] = slots[Z]",
            "",
            "if not(Z) goto B",
        ]
    );
}
