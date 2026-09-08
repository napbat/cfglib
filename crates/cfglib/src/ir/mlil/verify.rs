//! Structural verification shared by every MLIL dialect.

extern crate alloc;

use alloc::collections::BTreeSet;
use alloc::format;
use alloc::vec::Vec;

use crate::{EdgeId, FlowControl, FlowEffect, ProgramPoint};

use super::{
    Dialect, EntityId, Function, Instruction, VerificationIssue, VerificationReport, VerifyDialect,
};

pub(super) fn verify_function<D: VerifyDialect>(function: &Function<D>) -> VerificationReport {
    let mut issues = Vec::new();
    verify_cfg(function, &mut issues);
    verify_variables(function, &mut issues);
    verify_signature(function, &mut issues);
    verify_instructions(function, &mut issues);
    verify_edges(function, &mut issues);
    verify_regions(function, &mut issues);
    verify_provenance(function, &mut issues);
    D::verify(function, &mut issues);
    VerificationReport::new(super::error::LEVEL, issues)
}

fn verify_signature<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    for message in function
        .signature
        .parameter_issues(|&parameter| function.variable(parameter).is_some())
    {
        issue(issues, message);
    }
}

fn verify_regions<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    let block_count = function.cfg.block_count();
    let entry = function.cfg.entry();
    let check_block = |issues: &mut Vec<VerificationIssue>,
                       block: crate::BlockId,
                       region: crate::RegionId,
                       role: &str| {
        if block.index() >= block_count {
            issue(issues, format!("{region} {role} {block} does not exist"));
        } else if block == entry {
            issue(
                issues,
                format!("{region} {role} {block} is the synthetic root"),
            );
        }
    };
    for (position, region) in function.cfg.regions().iter().enumerate() {
        if region.id.index() != position {
            issue(
                issues,
                format!(
                    "region table slot {position} contains non-dense identity {}",
                    region.id
                ),
            );
        }
        if region.protected_blocks.is_empty() {
            issue(issues, format!("{} protects no blocks", region.id));
        }
        for &block in &region.protected_blocks {
            check_block(issues, block, region.id, "protected block");
        }
        for handler in &region.handlers {
            check_block(issues, handler.entry, region.id, "handler entry");
            if let Some(blocks) = handler.body.blocks() {
                for &block in blocks {
                    check_block(issues, block, region.id, "handler body block");
                }
                if !blocks.contains(&handler.entry) {
                    issue(
                        issues,
                        format!(
                            "{} handler body omits its own entry {}",
                            region.id, handler.entry
                        ),
                    );
                }
            }
            if let crate::HandlerKind::Filter { filter_block } = handler.kind {
                check_block(issues, filter_block, region.id, "filter block");
            }
        }
        if let Some(parent) = region.parent {
            if parent.index() >= position {
                issue(
                    issues,
                    format!("{} parent {parent} was not added before it", region.id),
                );
            }
        }
    }
}

fn verify_cfg<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    for error in crate::verify(&function.cfg).errors {
        issue(issues, format!("control-flow graph: {}", error.message));
    }

    let entry = function.cfg.entry();
    if !function.cfg.predecessor_edges(entry).is_empty() {
        issue(issues, "synthetic root has incoming edges");
    }
    if !function.cfg.block(entry).is_empty() {
        issue(issues, "synthetic root contains semantic instructions");
    }
    let outgoing = function.cfg.successor_edges(entry);
    if outgoing.len() != 1 {
        issue(
            issues,
            format!(
                "synthetic root has {} outgoing edges instead of one",
                outgoing.len()
            ),
        );
    } else if !D::is_entry_edge(function.cfg.edge(outgoing[0]).payload()) {
        issue(issues, "synthetic root edge is not an entry edge");
    }

    for block in function.cfg.blocks() {
        if block.id() != entry
            && block.is_empty()
            && !function.cfg.successor_edges(block.id()).is_empty()
        {
            issue(
                issues,
                format!(
                    "semantic block {} is empty but has outgoing edges",
                    block.id()
                ),
            );
        }
    }
}

fn verify_variables<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    for (index, variable) in function.variables.iter().enumerate() {
        if variable.id.index() != index {
            issue(
                issues,
                format!(
                    "variable table slot {index} contains non-dense identity {}",
                    variable.id
                ),
            );
        }
    }
}

fn verify_instructions<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    for instruction in function.instructions() {
        // Dead-code analysis keeps instructions alive only through their
        // declared effects, so an observably-throwing instruction with
        // none would be silently deletable.
        if instruction.may_throw() && instruction.effects().is_empty() {
            issue(
                issues,
                format!(
                    "throwing instruction {} declares no effect",
                    instruction.id()
                ),
            );
        }
    }
    let mut seen = BTreeSet::new();
    let mut count = 0usize;
    for block in function.cfg.blocks() {
        for (inst_idx, instruction) in block.instructions().iter().enumerate() {
            count += 1;
            let id = instruction.id();
            if !seen.insert(id) {
                issue(issues, format!("duplicate instruction identity {id}"));
            }
            let expected_point = ProgramPoint {
                block: block.id(),
                inst_idx,
            };
            match function.instruction_points.get(id.index()) {
                Some(point) if *point == expected_point => {}
                Some(point) => issue(
                    issues,
                    format!("instruction {id} is at {expected_point} but indexed at {point}"),
                ),
                None => issue(
                    issues,
                    format!("instruction {id} has no identity-table entry"),
                ),
            }
            verify_instruction(function, instruction, issues);
        }
    }
    if count != function.instruction_points.len() {
        issue(
            issues,
            format!(
                "instruction table has {} entries for {count} stored instructions",
                function.instruction_points.len()
            ),
        );
    }
    for (index, point) in function.instruction_points.iter().enumerate() {
        let valid = function
            .cfg
            .blocks()
            .get(point.block.index())
            .and_then(|block| block.instructions().get(point.inst_idx))
            .is_some_and(|instruction| instruction.id().index() == index);
        if !valid {
            issue(
                issues,
                format!("instruction table entry i{index} points outside its instruction"),
            );
        }
    }
}

fn verify_instruction<D: Dialect>(
    function: &Function<D>,
    instruction: &Instruction<D>,
    issues: &mut Vec<VerificationIssue>,
) {
    if instruction.uses().len() != instruction.use_types().len() {
        issue(
            issues,
            format!(
                "{} has mismatched use and use-type counts",
                instruction.id()
            ),
        );
    }
    if instruction.defs().len() != instruction.def_types().len() {
        issue(
            issues,
            format!(
                "{} has mismatched definition and definition-type counts",
                instruction.id()
            ),
        );
    }
    for variable in instruction.uses().iter().chain(instruction.defs()) {
        if function.variable(*variable).is_none() {
            issue(
                issues,
                format!("{} names undeclared variable {variable}", instruction.id()),
            );
        }
    }
    let distinct_definitions = instruction.defs().iter().copied().collect::<BTreeSet<_>>();
    if distinct_definitions.len() != instruction.defs().len() {
        issue(
            issues,
            format!("{} defines one variable more than once", instruction.id()),
        );
    }
    if instruction.flow_effect() == FlowEffect::MayThrow && !instruction.may_throw() {
        issue(
            issues,
            format!(
                "{} has may-throw control flow without exceptional semantics",
                instruction.id()
            ),
        );
    }
}

fn verify_edges<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    for edge in function.cfg.edges() {
        let expected = D::edge_kind(edge.payload());
        if edge.kind() != expected {
            issue(
                issues,
                format!(
                    "edge {} kind {} disagrees with dialect kind {expected}",
                    edge.id(),
                    edge.kind()
                ),
            );
        }
        if D::is_entry_edge(edge.payload()) && edge.source() != function.cfg.entry() {
            issue(
                issues,
                format!(
                    "entry edge {} does not originate at the synthetic root",
                    edge.id()
                ),
            );
        }
    }
}

fn verify_provenance<D: Dialect>(function: &Function<D>, issues: &mut Vec<VerificationIssue>) {
    let live_edges: BTreeSet<EdgeId> = function.cfg.edges().map(crate::Edge::id).collect();
    for entry in function.provenance.entries() {
        if D::span_is_empty(&entry.source) {
            issue(issues, "provenance contains an empty source span");
        }
        let valid = match entry.entity {
            EntityId::Block(block) => block.index() < function.cfg.block_count(),
            EntityId::Edge(edge) => live_edges.contains(&edge),
            EntityId::Instruction(instruction) => function.instruction(instruction).is_some(),
            EntityId::Variable(variable) => function.variable(variable).is_some(),
        };
        if !valid {
            issue(
                issues,
                format!("provenance names missing entity {:?}", entry.entity),
            );
        }
    }
}

fn issue(issues: &mut Vec<VerificationIssue>, message: impl Into<alloc::string::String>) {
    issues.push(VerificationIssue::new(message));
}
