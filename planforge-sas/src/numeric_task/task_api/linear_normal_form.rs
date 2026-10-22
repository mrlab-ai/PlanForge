use super::*;
use crate::axioms::{ComparisonOperator, NumericExpression};
use crate::utils::{float_tolerance, linear_effects::LinearizationError};

impl NumericRootTask {
    /// Hoffmann's linear normal form, established before any task consumer.
    /// PDDL arithmetic is real arithmetic. We approximate the algebraically
    /// equivalent comparison in f64: canonical coefficients and threshold,
    /// and one canonicalization after evaluating the shared linear sum.
    /// Differences from operation-by-operation rounding within the existing
    /// canonicalization tolerance are intentional; the tolerance itself is
    /// unchanged. Nonlinear comparisons keep their expression definitions.
    pub(super) fn normalize_linear_comparisons(&mut self) {
        self.fold_arithmetic_constants();
        let mut parts = std::collections::HashMap::new();
        let mut replaced = vec![false; self.numeric_variables.len()];
        let definitions = self
            .assignment_axiom_lookup()
            .expect("unique arithmetic definitions");
        fn mark(
            variable: VariableIndex,
            replaced: &mut [bool],
            definitions: &[Option<AxiomIndex>],
            axioms: &[AssignmentAxiom],
        ) {
            if replaced[variable.index()] {
                return;
            }
            if let Some(definition) = definitions[variable.index()] {
                replaced[variable.index()] = true;
                for dependency in axioms[definition.index()].expression.dependencies() {
                    mark(dependency, replaced, definitions, axioms);
                }
            }
        }
        for axiom in &self.assignment_axioms {
            if let NumericExpression::Linear(terms) = &axiom.expression {
                let key: Vec<_> = terms
                    .iter()
                    .map(|(var, coefficient)| (*var, float_tolerance::canonical_bits(*coefficient)))
                    .collect();
                parts.entry(key).or_insert(axiom.affected_var_id);
            }
        }
        let mut constants = std::collections::HashMap::new();
        for (id, variable) in self.numeric_variables.iter().enumerate() {
            if variable.get_type() == &NumericType::Constant {
                constants
                    .entry(float_tolerance::canonical_bits(self.numeric_state[id]))
                    .or_insert(VariableIndex::from_usize(id));
            }
        }
        for id in 0..self.comparison_axioms.len() {
            let comparison = &self.comparison_axioms[id];
            let expand = |variable| match self.linearize_numeric_var(variable) {
                Ok(expression) => Some(expression),
                Err(LinearizationError::NonLinearAssignmentAxiom { .. }) => None,
                Err(error) => panic!("cannot normalize comparison {id}: {error}"),
            };
            let (Some(left), Some(right)) = (
                expand(comparison.left_hand_side),
                expand(comparison.right_hand_side),
            ) else {
                continue;
            };
            mark(
                comparison.left_hand_side,
                &mut replaced,
                &definitions,
                &self.assignment_axioms,
            );
            mark(
                comparison.right_hand_side,
                &mut replaced,
                &definitions,
                &self.assignment_axioms,
            );
            let expression = left.subtract(&right);
            let mut terms: Vec<_> = expression
                .coefficients
                .iter()
                .enumerate()
                .map(|(var, &coefficient)| {
                    (
                        VariableIndex::from_usize(var),
                        NumericValue::new(coefficient),
                    )
                })
                .filter(|(_, coefficient)| *coefficient != ZERO_VALUE)
                .collect();
            if terms.iter().any(|(var, _)| {
                self.numeric_variables[var.index()].get_type() != &NumericType::Regular
            }) {
                continue;
            }
            assert!(
                expression.constant.is_finite()
                    && terms.iter().all(|(_, coefficient)| coefficient.is_finite()),
                "comparison {id} has non-finite linear coefficients"
            );
            let mut operator = comparison.operator.clone();
            let scale = terms
                .first()
                .map_or(ONE_VALUE, |(_, coefficient)| *coefficient);
            if scale < ZERO_VALUE {
                operator = match operator {
                    ComparisonOperator::LessThan => ComparisonOperator::GreaterThan,
                    ComparisonOperator::LessThanOrEqual => ComparisonOperator::GreaterThanOrEqual,
                    ComparisonOperator::GreaterThan => ComparisonOperator::LessThan,
                    ComparisonOperator::GreaterThanOrEqual => ComparisonOperator::LessThanOrEqual,
                    other => other,
                };
            }
            // Dividing by the leading coefficient identifies positive scalar
            // multiples as well as opposite orientations of the same sum.
            for (_, coefficient) in &mut terms {
                *coefficient = float_tolerance::canonicalize(*coefficient / scale);
                assert!(
                    *coefficient != ZERO_VALUE && coefficient.is_finite(),
                    "normalizing comparison {id} loses a term"
                );
            }
            let threshold =
                float_tolerance::canonicalize(NumericValue::new(-expression.constant) / scale);
            assert!(
                threshold.is_finite(),
                "comparison {id} has non-finite threshold"
            );
            let left = match terms.as_slice() {
                [] => self.normal_form_constant(ZERO_VALUE, &mut constants),
                [(variable, _)] => *variable,
                _ => {
                    let key: Vec<_> = terms
                        .iter()
                        .map(|(var, coefficient)| {
                            (*var, float_tolerance::canonical_bits(*coefficient))
                        })
                        .collect();
                    *parts.entry(key).or_insert_with(|| {
                        let variable = VariableIndex::from_usize(self.numeric_variables.len());
                        self.numeric_variables.push(NumericVariable::new(
                            format!("linear-part-{}", variable.index()),
                            NumericType::Derived,
                            Some(0),
                        ));
                        self.numeric_state.push(ZERO_VALUE);
                        self.assignment_axioms
                            .push(AssignmentAxiom::linear(variable, terms));
                        variable
                    })
                }
            };
            let right = self.normal_form_constant(threshold, &mut constants);
            let comparison = &mut self.comparison_axioms[id];
            comparison.left_hand_side = left;
            comparison.right_hand_side = right;
            comparison.operator = operator;
        }
        replaced.resize(self.numeric_variables.len(), false);
        self.remove_unused_arithmetic(&replaced);
    }

    fn fold_arithmetic_constants(&mut self) {
        let definitions = self
            .assignment_axiom_lookup()
            .expect("unique arithmetic definitions");
        fn fold(
            variable: VariableIndex,
            task: &mut NumericRootTask,
            definitions: &[Option<AxiomIndex>],
            visiting: &mut [bool],
        ) -> bool {
            match task.numeric_variables[variable.index()].get_type() {
                NumericType::Constant => return true,
                NumericType::Regular | NumericType::Cost => return false,
                NumericType::Derived => {}
            }
            assert!(!visiting[variable.index()], "cyclic arithmetic definition");
            visiting[variable.index()] = true;
            let definition = definitions[variable.index()]
                .expect("derived variable has an arithmetic definition");
            let expression = task.assignment_axioms[definition.index()]
                .expression
                .clone();
            let constant = expression
                .dependencies()
                .all(|dependency| fold(dependency, task, definitions, visiting));
            visiting[variable.index()] = false;
            if constant && task.metric.var_id() != Some(variable) {
                let value = expression
                    .evaluate(&task.numeric_state)
                    .expect("valid constant expression");
                task.numeric_variables[variable.index()] = NumericVariable::new(
                    task.numeric_variables[variable.index()].name().to_string(),
                    NumericType::Constant,
                    None,
                );
                task.numeric_state[variable.index()] = value;
                true
            } else {
                false
            }
        }
        let mut visiting = vec![false; self.numeric_variables.len()];
        for id in 0..self.numeric_variables.len() {
            fold(
                VariableIndex::from_usize(id),
                self,
                &definitions,
                &mut visiting,
            );
        }
        self.assignment_axioms.retain(|axiom| {
            self.numeric_variables[axiom.affected_var_id.index()].get_type()
                == &NumericType::Derived
        });
    }

    fn normal_form_constant(
        &mut self,
        value: NumericValue,
        constants: &mut std::collections::HashMap<u64, VariableIndex>,
    ) -> VariableIndex {
        *constants
            .entry(float_tolerance::canonical_bits(value))
            .or_insert_with(|| {
                let variable = VariableIndex::from_usize(self.numeric_variables.len());
                self.numeric_variables.push(NumericVariable::new(
                    format!("constant-{value}"),
                    NumericType::Constant,
                    None,
                ));
                self.numeric_state.push(value);
                variable
            })
    }

    fn remove_unused_arithmetic(&mut self, replaced: &[bool]) {
        let definitions = self
            .assignment_axiom_lookup()
            .expect("unique arithmetic definitions");
        let mut retained: Vec<_> = replaced.iter().map(|replaced| !replaced).collect();
        fn retain(
            variable: VariableIndex,
            retained: &mut [bool],
            definitions: &[Option<AxiomIndex>],
            axioms: &[AssignmentAxiom],
        ) {
            if retained[variable.index()] {
                return;
            }
            retained[variable.index()] = true;
            if let Some(definition) = definitions[variable.index()] {
                for operand in axioms[definition.index()].expression.dependencies() {
                    retain(operand, retained, definitions, axioms);
                }
            }
        }
        let mut roots = Vec::new();
        for axiom in &self.assignment_axioms {
            if !replaced[axiom.affected_var_id.index()] {
                roots.extend(axiom.expression.dependencies());
            }
        }
        for comparison in &self.comparison_axioms {
            roots.extend([comparison.left_hand_side, comparison.right_hand_side]);
        }
        roots.extend(self.metric.var_id());
        for operator in &self.operators {
            for effect in operator.assignment_effects() {
                roots.extend([effect.affected_var_id(), effect.var_id()]);
            }
        }
        for root in roots {
            retain(root, &mut retained, &definitions, &self.assignment_axioms);
        }
        let mut mapping = vec![None; retained.len()];
        let mut next = 0;
        for (id, keep) in retained.iter().enumerate() {
            if *keep {
                mapping[id] = Some(VariableIndex::from_usize(next));
                next += 1;
            }
        }
        let mapped = |variable: VariableIndex| {
            mapping[variable.index()].expect("retained arithmetic dependency")
        };
        self.metric = Metric::new(self.metric.is_min(), self.metric.var_id().map(mapped));
        self.numeric_variables = self
            .numeric_variables
            .iter()
            .enumerate()
            .filter(|(id, _)| retained[*id])
            .map(|(_, var)| var.clone())
            .collect();
        self.numeric_state = self
            .numeric_state
            .iter()
            .enumerate()
            .filter(|(id, _)| retained[*id])
            .map(|(_, value)| *value)
            .collect();
        self.assignment_axioms
            .retain(|axiom| retained[axiom.affected_var_id.index()]);
        for axiom in &mut self.assignment_axioms {
            axiom.affected_var_id = mapped(axiom.affected_var_id);
            match &mut axiom.expression {
                NumericExpression::Binary { left, right, .. } => {
                    *left = mapped(*left);
                    *right = mapped(*right);
                }
                NumericExpression::Linear(terms) => {
                    for (variable, _) in terms {
                        *variable = mapped(*variable);
                    }
                }
            }
        }
        for comparison in &mut self.comparison_axioms {
            comparison.left_hand_side = mapped(comparison.left_hand_side);
            comparison.right_hand_side = mapped(comparison.right_hand_side);
        }
        self.operators = self
            .operators
            .iter()
            .map(|operator| {
                Operator::new(
                    operator.name().to_string(),
                    operator.preconditions().clone(),
                    operator.effects().clone(),
                    operator
                        .assignment_effects()
                        .iter()
                        .map(|effect| {
                            AssignmentEffect::new(
                                mapped(effect.affected_var_id()),
                                effect.operation().clone(),
                                mapped(effect.var_id()),
                                effect.is_conditional(),
                                effect.conditions().clone(),
                            )
                        })
                        .collect(),
                    operator.cost(),
                )
            })
            .collect();
        // Rebuild arithmetic layers, preserving propositional layer distances.
        let definitions = self
            .assignment_axiom_lookup()
            .expect("unique arithmetic definitions");
        fn layer(
            variable: VariableIndex,
            definitions: &[Option<AxiomIndex>],
            axioms: &[AssignmentAxiom],
            visiting: &mut [bool],
            layers: &mut [Option<usize>],
        ) -> Option<usize> {
            let definition = definitions[variable.index()]?;
            if let Some(layer) = layers[variable.index()] {
                return Some(layer);
            }
            assert!(!visiting[variable.index()], "cyclic arithmetic definition");
            visiting[variable.index()] = true;
            let depth = axioms[definition.index()]
                .expression
                .dependencies()
                .filter_map(|var| layer(var, definitions, axioms, visiting, layers))
                .max()
                .map_or(0, |depth| depth + 1);
            layers[variable.index()] = Some(depth);
            visiting[variable.index()] = false;
            Some(depth)
        }
        let mut layers = vec![None; self.numeric_variables.len()];
        let mut visiting = vec![false; layers.len()];
        for id in 0..layers.len() {
            layer(
                VariableIndex::from_usize(id),
                &definitions,
                &self.assignment_axioms,
                &mut visiting,
                &mut layers,
            );
        }
        for (variable, &layer) in self.numeric_variables.iter_mut().zip(&layers) {
            *variable =
                NumericVariable::new(variable.name().to_string(), *variable.get_type(), layer);
        }
        self.assignment_axioms
            .sort_by_key(|axiom| layers[axiom.affected_var_id.index()]);
        let comparison_layer = layers.iter().flatten().max().map_or(0, |depth| depth + 1);
        if let Some(old_first) = self
            .variables
            .iter()
            .filter_map(ExplicitVariable::axiom_layer)
            .min()
        {
            for variable in &mut self.variables {
                if let Some(old) = variable.axiom_layer() {
                    *variable = variable.with_axiom_layer(Some(comparison_layer + old - old_first));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
