#[cfg(test)]
mod tests;

use std::sync::Arc;

use planforge_sas::numeric_conditions::{CompOp, ConditionNode, ConditionValue, NumericCondition};
use planforge_sas::numeric_task::{
    AbstractNumericTask, AssignmentEffect, AssignmentOperation, AxiomIndex, ExplicitFact,
    INF_VALUE, NEG_INF_VALUE, NumericTaskExt, NumericType, NumericValue, Operator, OperatorIndex,
    VariableIndex, ZERO_VALUE,
};
use planforge_sas::utils::float_tolerance;
use planforge_sas::utils::interval::Interval;
use planforge_sas::utils::linear_effects::{
    LinearExpression, LinearizationError, linearize_numeric_var,
};

/// A derived affine expression whose value changes by a deterministic constant
/// under every operator. Such an expression is an exact numeric coordinate of
/// the original task and can therefore be partitioned without compiling a new
/// task or losing correlations between its regular numeric dependencies.
#[derive(Clone, Debug)]
pub(crate) struct AdditiveNumericView {
    expression: LinearExpression,
    operator_deltas: Vec<NumericValue>,
}

impl AdditiveNumericView {
    /// The constant every application of `operator_id` adds to the view.
    pub(crate) fn operator_delta(&self, operator_id: OperatorIndex) -> NumericValue {
        self.operator_deltas[operator_id.index()]
    }

    pub(crate) fn evaluate(&self, numeric_values: &[NumericValue]) -> NumericValue {
        self.expression.evaluate(numeric_values)
    }
}

/// The value of coordinate `var_id` in a state with `numeric` values: its
/// additive view's for a derived coordinate, else its own, canonicalized as
/// split boundaries are.
pub(crate) fn coordinate_value(
    views: &[Option<AdditiveNumericView>],
    numeric: &[NumericValue],
    var_id: VariableIndex,
) -> NumericValue {
    float_tolerance::canonicalize(match &views[var_id.index()] {
        Some(view) => view.evaluate(numeric),
        None => numeric[var_id.index()],
    })
}

/// The additive view of numeric variable `numeric_var_id`, if it is a
/// derived affine expression of at least two regular variables that every
/// operator changes by a constant. `definitions` is the assignment axiom of
/// every derived variable. An affine expression of one variable is no
/// coordinate of its own: it is an image of that variable, so a comparison
/// on it is a threshold on the variable, as in the restricted task.
fn analyze_additive_numeric_view(
    task: &dyn AbstractNumericTask,
    definitions: &[Option<AxiomIndex>],
    numeric_var_id: VariableIndex,
) -> Option<AdditiveNumericView> {
    if task.numeric_variables()[numeric_var_id.index()].get_type() != &NumericType::Derived {
        return None;
    }
    if !defined_without_cost(task, definitions, numeric_var_id) {
        return None;
    }
    let expression = linear_expression(task, numeric_var_id)?;
    if expression.is_constant()
        || !expression.constant.is_finite()
        || expression
            .coefficients
            .iter()
            .any(|coefficient| !coefficient.is_finite())
    {
        return None;
    }
    let dependencies = regular_dependencies(task, &expression)?;
    if dependencies.len() < 2 {
        return None;
    }

    let operator_deltas = task
        .get_operators()
        .iter()
        .map(|operator| additive_view_delta_for_operator(task, &expression, operator))
        .collect::<Option<Vec<_>>>()?;
    Some(AdditiveNumericView {
        expression,
        operator_deltas,
    })
}

/// The variables `expression` depends on, or `None` if one is not regular.
fn regular_dependencies(
    task: &dyn AbstractNumericTask,
    expression: &LinearExpression,
) -> Option<Vec<VariableIndex>> {
    expression
        .coefficients
        .iter()
        .enumerate()
        .filter(|(_, coefficient)| coefficient.abs() >= float_tolerance::ABS_EPSILON)
        .map(|(var_id, _)| {
            (task.numeric_variables()[var_id].get_type() == &NumericType::Regular)
                .then_some(VariableIndex::from_usize(var_id))
        })
        .collect()
}

/// The coordinate numeric variable `numeric_var_id` is refined along: itself
/// if it is a coordinate, its variable if it is an affine expression of one
/// regular variable, else none.
fn operand_coordinate(
    task: &dyn AbstractNumericTask,
    definitions: &[Option<AxiomIndex>],
    views: &[Option<AdditiveNumericView>],
    numeric_var_id: VariableIndex,
) -> Option<VariableIndex> {
    if is_coordinate(task, views, numeric_var_id) {
        return Some(numeric_var_id);
    }
    if task.numeric_variables()[numeric_var_id.index()].get_type() != &NumericType::Derived
        || !defined_without_cost(task, definitions, numeric_var_id)
    {
        return None;
    }
    let expression = linear_expression(task, numeric_var_id)?;
    match regular_dependencies(task, &expression)?.as_slice() {
        &[variable] => Some(variable),
        _ => None,
    }
}

/// Whether numeric variable `numeric_var_id` has a value without reading a
/// cost variable: it is regular or constant, or derived by an assignment
/// axiom over such variables. A derived variable without an assignment
/// axiom has no definition. Panics on an operand that is no numeric
/// variable and on a cyclic definition.
fn defined_without_cost(
    task: &dyn AbstractNumericTask,
    definitions: &[Option<AxiomIndex>],
    numeric_var_id: VariableIndex,
) -> bool {
    fn visit(
        task: &dyn AbstractNumericTask,
        definitions: &[Option<AxiomIndex>],
        numeric_var_id: VariableIndex,
        visiting: &mut [bool],
    ) -> bool {
        let variable = task
            .numeric_variables()
            .get(numeric_var_id.index())
            .unwrap_or_else(|| {
                panic!("an assignment axiom reads the missing numeric variable {numeric_var_id}")
            });
        match variable.get_type() {
            NumericType::Cost => false,
            NumericType::Regular | NumericType::Constant => true,
            NumericType::Derived => {
                assert!(
                    !visiting[numeric_var_id.index()],
                    "derived numeric variable {} is defined cyclically",
                    variable.name()
                );
                let Some(axiom_id) = definitions[numeric_var_id.index()] else {
                    return false;
                };
                let axiom = &task.assignment_axioms()[axiom_id.index()];
                visiting[numeric_var_id.index()] = true;
                let defined = visit(task, definitions, axiom.get_left_var_id(), visiting)
                    && visit(task, definitions, axiom.get_right_var_id(), visiting);
                visiting[numeric_var_id.index()] = false;
                defined
            }
        }
    }

    visit(
        task,
        definitions,
        numeric_var_id,
        &mut vec![false; task.numeric_variables().len()],
    )
}

/// The affine expression numeric variable `numeric_var_id` stands for, or
/// `None` if it reads a derived variable without an assignment axiom, or
/// its definition multiplies or divides by a variable or divides by zero.
/// Panics on a malformed definition.
fn linear_expression(
    task: &dyn AbstractNumericTask,
    numeric_var_id: VariableIndex,
) -> Option<LinearExpression> {
    match linearize_numeric_var(task, numeric_var_id) {
        Ok(expression) => Some(expression),
        Err(
            LinearizationError::MissingAssignmentAxiom { .. }
            | LinearizationError::NonLinearAssignmentAxiom { .. }
            | LinearizationError::DivisionByZeroConstant { .. },
        ) => None,
        Err(error) => {
            panic!("numeric variable {numeric_var_id} has a malformed definition: {error}")
        }
    }
}

/// Every refined coordinate that can constrain evaluation of `tree`.
pub(crate) fn active_comparison_dimensions(
    tree: &NumericCondition,
    numeric_domain_sizes: &[usize],
    additive_views: &[Option<AdditiveNumericView>],
) -> Vec<VariableIndex> {
    let mut dimensions = tree
        .regular_numeric_var_dependencies()
        .iter()
        .copied()
        .filter(|&numeric_var_id| numeric_domain_sizes[numeric_var_id.index()] > 1)
        .collect::<Vec<_>>();
    for node in tree.nodes() {
        let ConditionNode::Arith {
            result_numeric_var_id,
            ..
        } = node
        else {
            continue;
        };
        if numeric_domain_sizes[result_numeric_var_id.index()] > 1
            && additive_views[result_numeric_var_id.index()].is_some()
        {
            dimensions.push(*result_numeric_var_id);
        }
    }
    dimensions.sort_unstable();
    dimensions.dedup();
    dimensions
}

fn additive_view_delta_for_operator(
    task: &dyn AbstractNumericTask,
    expression: &LinearExpression,
    operator: &Operator,
) -> Option<NumericValue> {
    let mut delta: f64 = 0.0;
    for (numeric_var_id, &coefficient) in expression.coefficients.iter().enumerate() {
        if coefficient.abs() < float_tolerance::ABS_EPSILON {
            continue;
        }
        delta += coefficient
            * regular_additive_delta_for_operator(
                task,
                VariableIndex::from_usize(numeric_var_id),
                operator,
            )?
            .value();
    }
    delta.is_finite().then_some(NumericValue::new(delta))
}

fn regular_additive_delta_for_operator(
    task: &dyn AbstractNumericTask,
    numeric_var_id: VariableIndex,
    operator: &Operator,
) -> Option<NumericValue> {
    let mut matching = operator
        .assignment_effects()
        .iter()
        .filter(|effect| effect.affected_var_id() == numeric_var_id);
    let Some(effect) = matching.next() else {
        return Some(ZERO_VALUE);
    };
    if matching.next().is_some() || effect.is_conditional() || !effect.conditions().is_empty() {
        return None;
    }
    constant_effect_delta(task, effect)
}

fn constant_effect_delta(
    task: &dyn AbstractNumericTask,
    effect: &AssignmentEffect,
) -> Option<NumericValue> {
    if task.numeric_variables()[effect.var_id().index()].get_type() != &NumericType::Constant {
        return None;
    }
    let rhs = task.get_initial_numeric_state_values()[effect.var_id().index()];
    if !rhs.is_finite() {
        return None;
    }
    match effect.operation() {
        AssignmentOperation::Plus => Some(rhs),
        AssignmentOperation::Minus => Some(-rhs),
        AssignmentOperation::Assign | AssignmentOperation::Times | AssignmentOperation::Divide => {
            None
        }
    }
}

/// The coordinates of a task: its regular numeric variables, and the derived
/// numeric variables whose value every operator changes by a constant.
/// Abstractions split coordinates, and a condition over one coordinate is an
/// interval of it, so regression over coordinates is exact.
pub(crate) struct Coordinates<'task> {
    task: &'task dyn AbstractNumericTask,
    views: Arc<[Option<AdditiveNumericView>]>,
    /// Per numeric variable, the coordinate a comparison on it is refined
    /// along, if any.
    operand_coordinates: Vec<Option<VariableIndex>>,
    /// Per numeric condition, its requirement on the one coordinate it
    /// constrains when it holds and when it does not, if it constrains one;
    /// indexed by [`ConditionValue`].
    requirements: Vec<[Option<CoordinateRequirement>; 2]>,
}

/// The values a condition allows for the one coordinate it constrains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct CoordinateRequirement {
    pub(crate) coordinate: VariableIndex,
    pub(crate) values: Interval,
}

impl<'task> Coordinates<'task> {
    pub(crate) fn new(task: &'task dyn AbstractNumericTask) -> Self {
        let definitions = task
            .assignment_axiom_lookup()
            .expect("a task defines every numeric variable by at most one assignment axiom");
        let views: Arc<[_]> = (0..task.numeric_variables().len())
            .map(|var_id| {
                analyze_additive_numeric_view(task, &definitions, VariableIndex::from_usize(var_id))
            })
            .collect();
        let operand_coordinates = (0..task.numeric_variables().len())
            .map(|var_id| {
                operand_coordinate(
                    task,
                    &definitions,
                    &views,
                    VariableIndex::from_usize(var_id),
                )
            })
            .collect();
        let is_coordinate = |var_id: VariableIndex| is_coordinate(task, &views, var_id);
        let requirements = task
            .numeric_conditions()
            .iter()
            .map(|condition| {
                [ConditionValue::True, ConditionValue::False]
                    .map(|truth| condition_requirement(task, condition, truth, &is_coordinate))
            })
            .collect();
        Self {
            task,
            views,
            operand_coordinates,
            requirements,
        }
    }

    /// The additive view of every numeric variable that has one.
    pub(crate) fn views(&self) -> &[Option<AdditiveNumericView>] {
        &self.views
    }

    /// The views, shared with an abstraction that outlives the task borrow.
    pub(crate) fn shared_views(&self) -> Arc<[Option<AdditiveNumericView>]> {
        Arc::clone(&self.views)
    }

    /// Whether numeric variable `var_id` is a coordinate: regular, or derived
    /// with an additive view.
    pub(crate) fn is_coordinate(&self, var_id: VariableIndex) -> bool {
        is_coordinate(self.task, &self.views, var_id)
    }

    /// The numeric variables a split refines `condition` along: the
    /// coordinates its two operands are refined along, or its regular leaves
    /// if neither operand has one.
    pub(crate) fn refinement_dimensions(&self, condition: &NumericCondition) -> Vec<VariableIndex> {
        let mut direct = [
            condition.left_numeric_var_id(),
            condition.right_numeric_var_id(),
        ]
        .into_iter()
        .filter_map(|numeric_var_id| self.operand_coordinates[numeric_var_id.index()])
        .collect::<Vec<_>>();
        direct.sort_unstable();
        direct.dedup();
        if !direct.is_empty() {
            return direct;
        }
        condition.regular_numeric_var_dependencies().to_vec()
    }

    /// The values of the task's numeric variables in its initial state, with
    /// every derived coordinate evaluated through its view.
    pub(crate) fn initial_values(&self) -> Vec<NumericValue> {
        let mut values = self.task.get_initial_numeric_state_values().to_vec();
        for (numeric_var_id, view) in self.views.iter().enumerate() {
            if let Some(view) = view {
                // A view reads regular variables only, which keep their values.
                values[numeric_var_id] = view.evaluate(&values);
            }
        }
        values
    }

    /// The requirement of the comparison fact `fact` on its coordinate, or
    /// `None` if `fact` is not a comparison over exactly one coordinate.
    pub(crate) fn requirement(&self, fact: &ExplicitFact) -> Option<CoordinateRequirement> {
        let condition_id = self
            .task
            .numeric_conditions()
            .id_for_var(fact.var_index())?;
        let truth = ConditionValue::from_usize(fact.value())
            .unwrap_or_else(|| panic!("comparison fact {fact:?} has no truth value"));
        self.requirements[condition_id][truth.as_usize()]
    }

    /// The states whose coordinate `coordinate` `op_id` maps into `target`,
    /// or `None` if there are none: the inverse of the operator's effects on
    /// it, composed in reverse order. An assignment's preimage is everything
    /// if `target` contains the assigned value and nothing otherwise.
    pub(crate) fn preimage(
        &self,
        target: Interval,
        op_id: OperatorIndex,
        coordinate: VariableIndex,
    ) -> Option<Interval> {
        let mut preimage = target;
        if let Some(view) = &self.views[coordinate.index()] {
            let delta = float_tolerance::canonicalize(view.operator_delta(op_id));
            preimage.apply_reverse_op(&AssignmentOperation::Plus, &Interval::singleton(delta));
            return Some(preimage.canonicalized());
        }
        let initial = self.task.get_initial_numeric_state_values();
        for effect in self.task.get_operators()[op_id.index()]
            .assignment_effects()
            .iter()
            .filter(|effect| effect.affected_var_id() == coordinate)
            .rev()
        {
            let rhs = float_tolerance::canonicalize(initial[effect.var_id().index()]);
            match effect.operation() {
                AssignmentOperation::Assign => {
                    if !preimage.contains(rhs) {
                        return None;
                    }
                    preimage = Interval::unbounded();
                }
                AssignmentOperation::Times if rhs == ZERO_VALUE => {
                    if !preimage.contains(ZERO_VALUE) {
                        return None;
                    }
                    preimage = Interval::unbounded();
                }
                operation => preimage.apply_reverse_op(operation, &Interval::singleton(rhs)),
            }
            preimage = preimage.canonicalized();
        }
        Some(preimage)
    }

    /// The values `op_id` maps coordinate `coordinate` to from `source`.
    pub(crate) fn image(
        &self,
        source: Interval,
        op_id: OperatorIndex,
        coordinate: VariableIndex,
    ) -> Interval {
        coordinate_image(self.task, &self.views, source, op_id, coordinate)
    }
}

/// The values `op_id` maps coordinate `coordinate` to from `source`, in
/// `task` with the additive views `views`: its view's delta added for a
/// derived coordinate, else the operator's effects on it applied in order,
/// canonicalized after every step as concrete states are.
pub(crate) fn coordinate_image(
    task: &dyn AbstractNumericTask,
    views: &[Option<AdditiveNumericView>],
    source: Interval,
    op_id: OperatorIndex,
    coordinate: VariableIndex,
) -> Interval {
    let mut image = source;
    if let Some(view) = &views[coordinate.index()] {
        let delta = float_tolerance::canonicalize(view.operator_delta(op_id));
        image.apply_op(&AssignmentOperation::Plus, &Interval::singleton(delta));
        return image.canonicalized();
    }
    let initial = task.get_initial_numeric_state_values();
    for effect in task.get_operators()[op_id.index()]
        .assignment_effects()
        .iter()
        .filter(|effect| effect.affected_var_id() == coordinate)
    {
        let rhs = Interval::singleton(float_tolerance::canonicalize(
            initial[effect.var_id().index()],
        ));
        image.apply_op(effect.operation(), &rhs);
        image = image.canonicalized();
    }
    image
}

fn is_coordinate(
    task: &dyn AbstractNumericTask,
    views: &[Option<AdditiveNumericView>],
    var_id: VariableIndex,
) -> bool {
    match task.numeric_variables()[var_id.index()].get_type() {
        NumericType::Regular => true,
        NumericType::Derived => views[var_id.index()].is_some(),
        NumericType::Constant | NumericType::Cost => false,
    }
}

/// The requirement of `tree` on its one coordinate where its comparison has
/// `truth`, if it constrains exactly one coordinate.
fn condition_requirement(
    task: &dyn AbstractNumericTask,
    tree: &NumericCondition,
    truth: ConditionValue,
    is_coordinate: &impl Fn(VariableIndex) -> bool,
) -> Option<CoordinateRequirement> {
    let required_op = required_comparison_op(tree.op(), truth);
    let left = linear_expression(task, tree.left_numeric_var_id())?;
    let right = linear_expression(task, tree.right_numeric_var_id())?;
    let num_numeric_vars = task.numeric_variables().len();
    let requirement = |(coordinate, values): (usize, Interval)| CoordinateRequirement {
        coordinate: VariableIndex::from_usize(coordinate),
        values,
    };
    if is_coordinate(tree.left_numeric_var_id()) && right.is_constant() {
        let expression =
            LinearExpression::variable(num_numeric_vars, tree.left_numeric_var_id()).subtract(
                &LinearExpression::constant(num_numeric_vars, right.constant),
            );
        if let Some(found) =
            single_var_interval_for_linear_zero_comparison(&expression, required_op)
        {
            return Some(requirement(found));
        }
    }
    if is_coordinate(tree.right_numeric_var_id()) && left.is_constant() {
        let expression = LinearExpression::constant(num_numeric_vars, left.constant).subtract(
            &LinearExpression::variable(num_numeric_vars, tree.right_numeric_var_id()),
        );
        if let Some(found) =
            single_var_interval_for_linear_zero_comparison(&expression, required_op)
        {
            return Some(requirement(found));
        }
    }
    let expression = left.subtract(&right);
    single_var_interval_for_linear_zero_comparison(&expression, required_op).map(requirement)
}

/// The comparison that holds where comparison `op` has `truth`.
fn required_comparison_op(op: CompOp, truth: ConditionValue) -> CompOp {
    match truth {
        ConditionValue::True => op,
        ConditionValue::False => match op {
            CompOp::Lt => CompOp::Ge,
            CompOp::Le => CompOp::Gt,
            CompOp::Gt => CompOp::Le,
            CompOp::Ge => CompOp::Lt,
            CompOp::Eq => CompOp::Ne,
            CompOp::Ne => CompOp::Eq,
        },
    }
}

fn single_var_interval_for_linear_zero_comparison(
    expression: &LinearExpression,
    op: CompOp,
) -> Option<(usize, Interval)> {
    if op == CompOp::Ne {
        return None;
    }

    let mut non_zero_coefficients = expression
        .coefficients
        .iter()
        .enumerate()
        .filter(|(_, coefficient)| coefficient.abs() >= float_tolerance::ABS_EPSILON);
    let (numeric_var_id, coefficient) = non_zero_coefficients.next()?;
    if non_zero_coefficients.next().is_some() {
        return None;
    }

    let threshold = NumericValue::new(-expression.constant / *coefficient);
    if !threshold.is_finite() {
        return None;
    }

    let interval = match (op, coefficient.is_sign_positive()) {
        (CompOp::Lt, true) | (CompOp::Gt, false) => {
            Interval::new(NEG_INF_VALUE, threshold, false, false)
        }
        (CompOp::Le, true) | (CompOp::Ge, false) => {
            Interval::new(NEG_INF_VALUE, threshold, false, true)
        }
        (CompOp::Gt, true) | (CompOp::Lt, false) => {
            Interval::new(threshold, INF_VALUE, false, false)
        }
        (CompOp::Ge, true) | (CompOp::Le, false) => {
            Interval::new(threshold, INF_VALUE, true, false)
        }
        (CompOp::Eq, _) => Interval::singleton(threshold),
        (CompOp::Ne, _) => return None,
    };
    Some((numeric_var_id, interval))
}
