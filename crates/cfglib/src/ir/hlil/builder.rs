//! Checked construction of generic HLIL functions.

extern crate alloc;

use alloc::format;
use alloc::vec::Vec;

use super::statement::{child_bodies, expression_references};
use super::{
    Dialect, EntityId, Error, Expression, ExpressionId, ExpressionKind, Function, ProvenanceMap,
    Result, Signature, Statement, StatementId, StatementKind, Variable, VariableId, VerifyDialect,
};

/// Incremental bottom-up builder that assigns dense stable HLIL identities.
///
/// Children are created before parents: expressions before the statements
/// that reference them, inner statements before the compound statements
/// whose bodies carry them, and finally [`Self::set_body`] roots the tree.
pub struct FunctionBuilder<D: Dialect> {
    variables: Vec<Variable<D>>,
    signature: Signature<D>,
    expressions: Vec<Expression<D>>,
    statements: Vec<Statement<D>>,
    body: Vec<StatementId>,
    provenance: ProvenanceMap<D>,
}

impl<D: Dialect> FunctionBuilder<D> {
    /// Creates an empty builder for one source function.
    #[must_use]
    pub fn new(source: D::Source) -> Self {
        Self {
            variables: Vec::new(),
            signature: Signature::<D>::default(),
            expressions: Vec::new(),
            statements: Vec::new(),
            body: Vec::new(),
            provenance: ProvenanceMap::new(source),
        }
    }

    /// Declares one variable.
    ///
    /// # Errors
    ///
    /// Returns an error if the function exceeds the compact identity space.
    pub fn declare_variable(
        &mut self,
        role: D::VariableRole,
        native: Option<D::NativeVariable>,
        declared_type: Option<D::ValueType>,
    ) -> Result<VariableId> {
        let raw = u32::try_from(self.variables.len())
            .map_err(|_| Error::InvalidConstruction("variable count exceeds u32::MAX".into()))?;
        let id = VariableId::from_raw(raw);
        self.variables.push(Variable {
            id,
            role,
            native,
            declared_type,
        });
        Ok(id)
    }

    /// Declares the ordered parameter and return signature.
    ///
    /// # Errors
    ///
    /// Returns an error when a parameter is undeclared or repeated.
    pub fn set_signature(&mut self, signature: Signature<D>) -> Result<()> {
        let declared = self.variables.len();
        if let Some(issue) = signature
            .parameter_issues(|parameter| parameter.index() < declared)
            .into_iter()
            .next()
        {
            return Err(Error::InvalidConstruction(issue));
        }
        self.signature = signature;
        Ok(())
    }

    /// Adds one typed expression node.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression names an undeclared variable or
    /// a not-yet-created operand, or the identity space is exhausted.
    pub fn add_expression(
        &mut self,
        kind: ExpressionKind<D>,
        value_type: D::ValueType,
    ) -> Result<ExpressionId> {
        match &kind {
            ExpressionKind::Variable(variable) => self.require_variable(*variable)?,
            ExpressionKind::Constant(_) => {}
            ExpressionKind::Operation { operands, .. } => {
                for &operand in operands {
                    self.require_expression(operand)?;
                }
            }
        }
        let raw = u32::try_from(self.expressions.len())
            .map_err(|_| Error::InvalidConstruction("expression count exceeds u32::MAX".into()))?;
        let id = ExpressionId::from_raw(raw);
        self.expressions.push(Expression {
            id,
            kind,
            value_type,
        });
        Ok(id)
    }

    /// Adds one statement over already-created children.
    ///
    /// # Errors
    ///
    /// Returns an error when the statement references a not-yet-created
    /// child, assigns into a constant, declares an empty case-value list,
    /// binds an undeclared handler variable, or exhausts the identity space.
    pub fn add_statement(
        &mut self,
        kind: StatementKind<D>,
        source: Option<D::SourceSpan>,
    ) -> Result<StatementId> {
        self.require_statement_children(&kind)?;
        let raw = u32::try_from(self.statements.len())
            .map_err(|_| Error::InvalidConstruction("statement count exceeds u32::MAX".into()))?;
        let id = StatementId::from_raw(raw);
        self.statements.push(Statement { id, kind });
        if let Some(span) = source {
            self.provenance.insert(span, EntityId::Statement(id))?;
        }
        Ok(id)
    }

    /// Replaces the operation of one operation expression in place,
    /// keeping its operands and type — exact negation rewrites a
    /// comparison without wrapping it.
    ///
    /// # Errors
    ///
    /// Returns an error when the expression does not exist or is not an
    /// operation.
    pub fn replace_operation(
        &mut self,
        expression: ExpressionId,
        operation: D::Operation,
    ) -> Result<()> {
        let node = self
            .expressions
            .get_mut(expression.index())
            .ok_or_else(|| Error::InvalidConstruction("unknown expression".into()))?;
        let ExpressionKind::Operation {
            operation: existing,
            ..
        } = &mut node.kind
        else {
            return Err(Error::InvalidConstruction(
                "only operation expressions carry an operation".into(),
            ));
        };
        *existing = operation;
        Ok(())
    }

    /// Looks up one already-created expression.
    #[must_use]
    pub fn expression(&self, id: ExpressionId) -> Option<&Expression<D>> {
        self.expressions.get(id.index())
    }

    /// Records an additional many-to-many source correspondence.
    ///
    /// # Errors
    ///
    /// Returns an error when `source` is empty or reversed.
    pub fn map_entity(&mut self, source: D::SourceSpan, entity: EntityId) -> Result<bool> {
        Ok(self.provenance.insert(source, entity)?)
    }

    /// Roots the statement tree at the given top-level sequence.
    ///
    /// # Errors
    ///
    /// Returns an error when a root statement does not exist.
    pub fn set_body(&mut self, body: Vec<StatementId>) -> Result<()> {
        for &statement in &body {
            self.require_statement(statement)?;
        }
        self.body = body;
        Ok(())
    }

    fn require_variable(&self, variable: VariableId) -> Result<()> {
        if variable.index() < self.variables.len() {
            Ok(())
        } else {
            Err(Error::InvalidConstruction(format!(
                "expression names undeclared variable {variable}"
            )))
        }
    }

    fn require_expression(&self, expression: ExpressionId) -> Result<()> {
        if expression.index() < self.expressions.len() {
            Ok(())
        } else {
            Err(Error::InvalidConstruction(format!(
                "reference to not-yet-created expression {expression}"
            )))
        }
    }

    fn require_statement(&self, statement: StatementId) -> Result<()> {
        if statement.index() < self.statements.len() {
            Ok(())
        } else {
            Err(Error::InvalidConstruction(format!(
                "reference to not-yet-created statement {statement}"
            )))
        }
    }

    fn require_statement_children(&self, kind: &StatementKind<D>) -> Result<()> {
        let mut expression_error = Ok(());
        expression_references(kind, &mut |expression| {
            if expression_error.is_ok() {
                expression_error = self.require_expression(expression);
            }
        });
        expression_error?;
        for body in child_bodies(kind) {
            for &statement in body {
                self.require_statement(statement)?;
            }
        }
        if let StatementKind::Assign { target, .. } = kind
            && let Some(expression) = self.expressions.get(target.index())
            && matches!(expression.kind(), ExpressionKind::Constant(_))
        {
            return Err(Error::InvalidConstruction(format!(
                "assignment target {target} is a constant"
            )));
        }
        if let StatementKind::Switch { cases, .. } = kind {
            for case in cases {
                if case.values.is_empty() {
                    return Err(Error::InvalidConstruction(
                        "switch arm declares no case values".into(),
                    ));
                }
            }
        }
        if let StatementKind::Try { handlers, .. } = kind {
            for handler in handlers {
                if let Some(binding) = handler.binding
                    && binding.index() >= self.variables.len()
                {
                    return Err(Error::InvalidConstruction(format!(
                        "handler binds undeclared variable {binding}"
                    )));
                }
            }
        }
        Ok(())
    }
}

impl<D: VerifyDialect> FunctionBuilder<D> {
    /// Completes and strictly verifies the function.
    ///
    /// # Errors
    ///
    /// Returns every discovered invariant violation as one report.
    pub fn finish(self) -> Result<Function<D>> {
        let function = Function {
            variables: self.variables,
            signature: self.signature,
            expressions: self.expressions,
            statements: self.statements,
            body: self.body,
            provenance: self.provenance,
        };
        let report = function.verify();
        if report.is_ok() {
            Ok(function)
        } else {
            Err(report.into())
        }
    }
}
