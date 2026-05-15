use super::numeric_helper::NumericTaskHelper;
use planners_sas::numeric::numeric_task::{AbstractNumericTask, AssignmentOperation};
use planners_sas::numeric::utils::linear_effects::LinearExpression;

#[derive(Debug, Clone)]
struct BoundCondition {
    coefficients: Vec<f64>,
    constant: f64,
    epsilon: f64,
}

#[derive(Debug, Clone)]
struct PreparedLinearEffect {
    lhs: usize,
    coefficients: Vec<f64>,
    constant: f64,
}

#[derive(Debug, Clone, Default)]
pub struct NumericBound {
    initialized: bool,
    precision: f64,
    epsilon: f64,
    numeric_helper: NumericTaskHelper,
    numeric_variable_ids: Vec<usize>,
    num_numeric_variables: usize,
    num_actions: usize,
    operator_conditions: Vec<Vec<BoundCondition>>,
    prepared_simple_effects: Vec<Vec<Option<f64>>>,
    prepared_linear_effects: Vec<Vec<PreparedLinearEffect>>,
    variable_has_ub: Vec<bool>,
    variable_has_lb: Vec<bool>,
    variable_ub: Vec<f64>,
    variable_lb: Vec<f64>,
    effect_has_ub: Vec<Vec<bool>>,
    effect_has_lb: Vec<Vec<bool>>,
    effect_ub: Vec<Vec<f64>>,
    effect_lb: Vec<Vec<f64>>,
    assignment_has_ub: Vec<Vec<bool>>,
    assignment_has_lb: Vec<Vec<bool>>,
    assignment_ub: Vec<Vec<f64>>,
    assignment_lb: Vec<Vec<f64>>,
    variable_before_action_has_ub: Vec<Vec<bool>>,
    variable_before_action_has_lb: Vec<Vec<bool>>,
    variable_before_action_ub: Vec<Vec<f64>>,
    variable_before_action_lb: Vec<Vec<f64>>,
}

impl NumericBound {
    pub fn new(task: &dyn AbstractNumericTask, precision: f64, epsilon: f64) -> Self {
        let mut bound = Self::default();
        bound.initialize(task, precision, epsilon);
        bound
    }

    pub fn initialize(&mut self, task: &dyn AbstractNumericTask, precision: f64, epsilon: f64) {
        assert!(
            precision >= 0.0,
            "numeric bound precision must be non-negative"
        );
        assert!(epsilon >= 0.0, "numeric bound epsilon must be non-negative");
        self.numeric_helper = NumericTaskHelper::new_lmcut(task, precision, epsilon, false);
        self.numeric_variable_ids = task.regular_numeric_variable_ids();
        self.num_numeric_variables = self.numeric_variable_ids.len();
        self.num_actions = task.get_operators().len();
        self.operator_conditions = (0..self.num_actions)
            .map(|operator_id| self.extract_operator_conditions(task, operator_id))
            .collect();
        self.prepared_simple_effects =
            vec![vec![None; self.num_numeric_variables]; self.num_actions];
        self.prepared_linear_effects = vec![Vec::new(); self.num_actions];
        self.extract_operator_effects(task);
        self.variable_has_ub = vec![false; self.num_numeric_variables];
        self.variable_has_lb = vec![false; self.num_numeric_variables];
        self.variable_ub = vec![f64::MAX; self.num_numeric_variables];
        self.variable_lb = vec![f64::MIN; self.num_numeric_variables];
        self.effect_has_ub = vec![vec![false; self.num_numeric_variables]; self.num_actions];
        self.effect_has_lb = vec![vec![false; self.num_numeric_variables]; self.num_actions];
        self.effect_ub = vec![vec![f64::MAX; self.num_numeric_variables]; self.num_actions];
        self.effect_lb = vec![vec![f64::MIN; self.num_numeric_variables]; self.num_actions];
        self.assignment_has_ub = vec![vec![false; self.num_numeric_variables]; self.num_actions];
        self.assignment_has_lb = vec![vec![false; self.num_numeric_variables]; self.num_actions];
        self.assignment_ub = vec![vec![f64::MAX; self.num_numeric_variables]; self.num_actions];
        self.assignment_lb = vec![vec![f64::MIN; self.num_numeric_variables]; self.num_actions];
        self.variable_before_action_has_ub =
            vec![vec![false; self.num_actions]; self.num_numeric_variables];
        self.variable_before_action_has_lb =
            vec![vec![false; self.num_actions]; self.num_numeric_variables];
        self.variable_before_action_ub =
            vec![vec![f64::MAX; self.num_actions]; self.num_numeric_variables];
        self.variable_before_action_lb =
            vec![vec![f64::MIN; self.num_actions]; self.num_numeric_variables];
        self.initialized = true;
        self.precision = precision;
        self.epsilon = epsilon;
    }

    pub fn calculate_bounds(&mut self, _state: &[f64], iterations: usize) {
        assert!(
            self.initialized,
            "numeric bound must be initialized before use"
        );
        assert!(
            iterations <= i32::MAX as usize,
            "bound iterations exceed supported range"
        );
        self.prepare();
        self.update_before_action_bounds();
        let mut completed_iterations = 0usize;

        while completed_iterations < iterations
            && (self.update_variable_bounds(_state)
                || self.update_action_bounds()
                || self.update_before_action_bounds())
        {
            completed_iterations += 1;
        }
    }

    pub fn precision(&self) -> f64 {
        self.precision
    }

    pub fn get_variable_before_action_has_ub(&self, var_id: usize, op_id: usize) -> bool {
        self.variable_before_action_has_ub[var_id][op_id]
    }

    pub fn get_variable_before_action_has_lb(&self, var_id: usize, op_id: usize) -> bool {
        self.variable_before_action_has_lb[var_id][op_id]
    }

    pub fn get_variable_before_action_ub(&self, var_id: usize, op_id: usize) -> f64 {
        self.variable_before_action_ub[var_id][op_id]
    }

    pub fn get_variable_before_action_lb(&self, var_id: usize, op_id: usize) -> f64 {
        self.variable_before_action_lb[var_id][op_id]
    }

    pub fn get_effect_has_ub(&self, op_id: usize, var_id: usize) -> bool {
        self.effect_has_ub[op_id][var_id]
    }

    pub fn get_effect_has_lb(&self, op_id: usize, var_id: usize) -> bool {
        self.effect_has_lb[op_id][var_id]
    }

    pub fn get_effect_ub(&self, op_id: usize, var_id: usize) -> f64 {
        self.effect_ub[op_id][var_id]
    }

    pub fn get_effect_lb(&self, op_id: usize, var_id: usize) -> f64 {
        self.effect_lb[op_id][var_id]
    }

    pub fn get_assignment_has_ub(&self, op_id: usize, var_id: usize) -> bool {
        self.assignment_has_ub[op_id][var_id]
    }

    pub fn get_assignment_has_lb(&self, op_id: usize, var_id: usize) -> bool {
        self.assignment_has_lb[op_id][var_id]
    }

    pub fn get_assignment_ub(&self, op_id: usize, var_id: usize) -> f64 {
        self.assignment_ub[op_id][var_id]
    }

    pub fn get_assignment_lb(&self, op_id: usize, var_id: usize) -> f64 {
        self.assignment_lb[op_id][var_id]
    }

    pub fn has_no_increasing_assignment_effect(&self, op_id: usize, var_id: usize) -> bool {
        self.get_variable_before_action_has_lb(var_id, op_id)
            && self.get_assignment_has_ub(op_id, var_id)
            && self.get_variable_before_action_lb(var_id, op_id)
                >= self.get_assignment_ub(op_id, var_id)
    }

    pub fn has_no_decreasing_assignment_effect(&self, op_id: usize, var_id: usize) -> bool {
        self.get_variable_before_action_has_ub(var_id, op_id)
            && self.get_assignment_has_lb(op_id, var_id)
            && self.get_variable_before_action_ub(var_id, op_id)
                <= self.get_assignment_lb(op_id, var_id)
    }

    fn prepare(&mut self) {
        for var_id in 0..self.num_numeric_variables {
            self.variable_has_ub[var_id] = false;
            self.variable_has_lb[var_id] = false;
            self.variable_ub[var_id] = f64::MAX;
            self.variable_lb[var_id] = f64::MIN;
        }

        for op_id in 0..self.num_actions {
            for var_id in 0..self.num_numeric_variables {
                self.effect_has_ub[op_id][var_id] = true;
                self.effect_has_lb[op_id][var_id] = true;
                self.effect_ub[op_id][var_id] = 0.0;
                self.effect_lb[op_id][var_id] = 0.0;
                self.assignment_has_ub[op_id][var_id] = false;
                self.assignment_has_lb[op_id][var_id] = false;
                self.assignment_ub[op_id][var_id] = f64::MAX;
                self.assignment_lb[op_id][var_id] = f64::MIN;

                if let Some(simple_effect) = self.prepared_simple_effects[op_id][var_id]
                    && simple_effect.abs() >= self.precision
                {
                    self.effect_ub[op_id][var_id] = simple_effect;
                    self.effect_lb[op_id][var_id] = simple_effect;
                }
            }

            for linear_effect in &self.prepared_linear_effects[op_id] {
                let lhs = linear_effect.lhs;
                self.effect_has_ub[op_id][lhs] = false;
                self.effect_has_lb[op_id][lhs] = false;
                self.effect_ub[op_id][lhs] = f64::MAX;
                self.effect_lb[op_id][lhs] = f64::MIN;
                self.assignment_has_ub[op_id][lhs] = false;
                self.assignment_has_lb[op_id][lhs] = false;
                self.assignment_ub[op_id][lhs] = f64::MAX;
                self.assignment_lb[op_id][lhs] = f64::MIN;
            }
        }

        for var_id in 0..self.num_numeric_variables {
            for op_id in 0..self.num_actions {
                self.variable_before_action_has_ub[var_id][op_id] = false;
                self.variable_before_action_has_lb[var_id][op_id] = false;
                self.variable_before_action_ub[var_id][op_id] = f64::MAX;
                self.variable_before_action_lb[var_id][op_id] = f64::MIN;
            }
        }
    }

    fn update_before_action_bounds(&mut self) -> bool {
        let mut change = false;

        for var_id in 0..self.num_numeric_variables {
            for op_id in 0..self.num_actions {
                let mut upper_bounded = self.variable_has_ub[var_id];
                let mut lower_bounded = self.variable_has_lb[var_id];
                let mut ub = if upper_bounded {
                    self.variable_ub[var_id]
                } else {
                    f64::MAX
                };
                let mut lb = if lower_bounded {
                    self.variable_lb[var_id]
                } else {
                    f64::MIN
                };

                for condition in &self.operator_conditions[op_id] {
                    let weight = condition.coefficients[var_id];
                    let mut k = condition.constant - condition.epsilon;

                    if weight.abs() < self.precision {
                        continue;
                    }

                    let mut condition_bounded = true;
                    for another_id in 0..self.num_numeric_variables {
                        if another_id == var_id {
                            continue;
                        }

                        let another_weight = condition.coefficients[another_id];
                        if another_weight >= self.precision {
                            if self.variable_before_action_has_ub[another_id][op_id] {
                                k += another_weight
                                    * self.variable_before_action_ub[another_id][op_id];
                            } else {
                                condition_bounded = false;
                                break;
                            }
                        } else if another_weight <= -self.precision {
                            if self.variable_before_action_has_lb[another_id][op_id] {
                                k += another_weight
                                    * self.variable_before_action_lb[another_id][op_id];
                            } else {
                                condition_bounded = false;
                                break;
                            }
                        }
                    }

                    if !condition_bounded {
                        continue;
                    }

                    if weight <= -self.precision {
                        if !upper_bounded {
                            upper_bounded = true;
                        }
                        ub = ub.min(-k / weight);
                    }

                    if weight >= self.precision {
                        if !lower_bounded {
                            lower_bounded = true;
                        }
                        lb = lb.max(-k / weight);
                    }
                }

                if upper_bounded
                    && (!self.variable_before_action_has_ub[var_id][op_id]
                        || (self.variable_before_action_ub[var_id][op_id] - ub).abs()
                            >= self.precision)
                {
                    change = true;
                    self.variable_before_action_has_ub[var_id][op_id] = true;
                    self.variable_before_action_ub[var_id][op_id] = ub;
                }

                if lower_bounded
                    && (!self.variable_before_action_has_lb[var_id][op_id]
                        || (self.variable_before_action_lb[var_id][op_id] - lb).abs()
                            >= self.precision)
                {
                    change = true;
                    self.variable_before_action_has_lb[var_id][op_id] = true;
                    self.variable_before_action_lb[var_id][op_id] = lb;
                }
            }
        }

        change
    }

    fn update_variable_bounds(&mut self, state: &[f64]) -> bool {
        let mut change = false;

        for var_id in 0..self.num_numeric_variables {
            let mut has_ub = true;
            let mut has_lb = true;
            let actual_var_id = self.numeric_variable_ids[var_id];
            let mut ub = *state.get(actual_var_id).unwrap_or_else(|| {
                panic!("numeric bound state is missing numeric variable {actual_var_id}")
            });
            let mut lb = ub;

            for op_id in 0..self.num_actions {
                if has_ub {
                    let mut upper_bounded = false;
                    let mut has_effect = false;
                    let mut local_ub = f64::MAX;

                    if self.effect_has_ub[op_id][var_id] {
                        let increment = self.effect_ub[op_id][var_id];
                        if increment < self.precision {
                            upper_bounded = true;
                        } else if self.variable_before_action_has_ub[var_id][op_id] {
                            upper_bounded = true;
                            has_effect = true;
                            local_ub = increment + self.variable_before_action_ub[var_id][op_id];
                        }
                    }

                    if (!upper_bounded || has_effect) && self.assignment_has_ub[op_id][var_id] {
                        upper_bounded = true;
                        has_effect = true;
                        local_ub = local_ub.min(self.assignment_ub[op_id][var_id]);
                    }

                    if has_effect {
                        ub = ub.max(local_ub);
                    }
                    if !upper_bounded {
                        has_ub = false;
                    }
                }

                if has_lb {
                    let mut lower_bounded = false;
                    let mut has_effect = false;
                    let mut local_lb = f64::MIN;

                    if self.effect_has_lb[op_id][var_id] {
                        let increment = self.effect_lb[op_id][var_id];
                        if increment > -self.precision {
                            lower_bounded = true;
                        } else if self.variable_before_action_has_lb[var_id][op_id] {
                            lower_bounded = true;
                            has_effect = true;
                            local_lb = increment + self.variable_before_action_lb[var_id][op_id];
                        }
                    }

                    if (!lower_bounded || has_effect) && self.assignment_has_lb[op_id][var_id] {
                        lower_bounded = true;
                        has_effect = true;
                        local_lb = local_lb.max(self.assignment_lb[op_id][var_id]);
                    }

                    if has_effect {
                        lb = lb.min(local_lb);
                    }
                    if !lower_bounded {
                        has_lb = false;
                    }
                }

                if !has_ub && !has_lb {
                    break;
                }
            }

            if has_ub {
                if !change
                    && (!self.variable_has_ub[var_id]
                        || (self.variable_ub[var_id] - ub).abs() >= self.precision)
                {
                    change = true;
                }
                self.variable_has_ub[var_id] = true;
                self.variable_ub[var_id] = ub;
            }

            if has_lb {
                if !change
                    && (!self.variable_has_lb[var_id]
                        || (self.variable_lb[var_id] - lb).abs() >= self.precision)
                {
                    change = true;
                }
                self.variable_has_lb[var_id] = true;
                self.variable_lb[var_id] = lb;
            }
        }

        change
    }

    #[allow(clippy::needless_range_loop)]
    fn update_action_bounds(&mut self) -> bool {
        let mut change = false;

        for op_id in 0..self.num_actions {
            for linear_effect in self.prepared_linear_effects[op_id].clone() {
                let lhs = linear_effect.lhs;
                let coefficients = linear_effect.coefficients;
                let constant = linear_effect.constant;
                let mut has_lb = true;
                let mut has_ub = true;
                let mut ub = constant;
                let mut lb = constant;

                for var_id in 0..self.num_numeric_variables {
                    if var_id == lhs {
                        continue;
                    }

                    let weight = coefficients[var_id];
                    if has_ub {
                        if weight >= self.precision
                            && self.variable_before_action_has_ub[var_id][op_id]
                        {
                            ub += weight * self.variable_before_action_ub[var_id][op_id];
                        } else if weight <= -self.precision
                            && self.variable_before_action_has_lb[var_id][op_id]
                        {
                            ub += weight * self.variable_before_action_lb[var_id][op_id];
                        } else if weight.abs() >= self.precision {
                            has_ub = false;
                        }
                    }

                    if has_lb {
                        if weight >= self.precision
                            && self.variable_before_action_has_lb[var_id][op_id]
                        {
                            lb += weight * self.variable_before_action_lb[var_id][op_id];
                        } else if weight <= -self.precision
                            && self.variable_before_action_has_ub[var_id][op_id]
                        {
                            lb += weight * self.variable_before_action_ub[var_id][op_id];
                        } else if weight.abs() >= self.precision {
                            has_lb = false;
                        }
                    }

                    if !has_ub && !has_lb {
                        break;
                    }
                }

                let mut new_assignment_has_ub = false;
                let mut new_assignment_has_lb = false;
                let mut new_assignment_ub = f64::MAX;
                let mut new_assignment_lb = f64::MIN;
                let mut new_effect_has_ub = false;
                let mut new_effect_has_lb = false;
                let mut new_effect_ub = f64::MAX;
                let mut new_effect_lb = f64::MIN;

                if has_ub {
                    if coefficients[lhs].abs() < self.precision {
                        new_assignment_has_ub = true;
                        new_assignment_ub = ub;
                    } else if coefficients[lhs] >= self.precision
                        && self.variable_before_action_has_ub[lhs][op_id]
                    {
                        new_assignment_has_ub = true;
                        new_assignment_ub =
                            ub + coefficients[lhs] * self.variable_before_action_ub[lhs][op_id];
                    } else if coefficients[lhs] <= -self.precision
                        && self.variable_before_action_has_lb[lhs][op_id]
                    {
                        new_assignment_has_ub = true;
                        new_assignment_ub =
                            ub + coefficients[lhs] * self.variable_before_action_lb[lhs][op_id];
                    }

                    let increment_coefficient = coefficients[lhs] - 1.0;
                    if increment_coefficient.abs() < self.precision {
                        new_effect_has_ub = true;
                        new_effect_ub = ub;
                    } else if increment_coefficient >= self.precision
                        && self.variable_before_action_has_ub[lhs][op_id]
                    {
                        new_effect_has_ub = true;
                        // PARITY(numeric-fd): the reference implementation uses the boolean
                        // `get_variable_before_action_has_ub(lhs, op_id)` value in this branch
                        // rather than the numeric upper bound itself.
                        new_effect_ub = ub
                            + increment_coefficient
                                * f64::from(self.variable_before_action_has_ub[lhs][op_id]);
                    } else if increment_coefficient <= -self.precision
                        && self.variable_before_action_has_lb[lhs][op_id]
                    {
                        new_effect_has_ub = true;
                        new_effect_ub =
                            ub + increment_coefficient * self.variable_before_action_lb[lhs][op_id];
                    }
                }

                if has_lb {
                    if coefficients[lhs].abs() < self.precision {
                        new_assignment_has_lb = true;
                        new_assignment_lb = lb;
                    } else if coefficients[lhs] >= self.precision
                        && self.variable_before_action_has_lb[lhs][op_id]
                    {
                        new_assignment_has_lb = true;
                        new_assignment_lb =
                            lb + coefficients[lhs] * self.variable_before_action_lb[lhs][op_id];
                    } else if coefficients[lhs] <= -self.precision
                        && self.variable_before_action_has_ub[lhs][op_id]
                    {
                        new_assignment_has_lb = true;
                        new_assignment_lb =
                            lb + coefficients[lhs] * self.variable_before_action_ub[lhs][op_id];
                    }

                    let increment_coefficient = coefficients[lhs] - 1.0;
                    if increment_coefficient.abs() < self.precision {
                        new_effect_has_lb = true;
                        new_effect_lb = lb;
                    } else if increment_coefficient >= self.precision
                        && self.variable_before_action_has_lb[lhs][op_id]
                    {
                        new_effect_has_lb = true;
                        // PARITY(numeric-fd): same reference quirk as the upper-bound branch:
                        // C++ multiplies by the boolean `get_variable_before_action_has_lb(...)`
                        // instead of the numeric lower bound value.
                        new_effect_lb = lb
                            + increment_coefficient
                                * f64::from(self.variable_before_action_has_lb[lhs][op_id]);
                    } else if increment_coefficient <= -self.precision
                        && self.variable_before_action_has_ub[lhs][op_id]
                    {
                        new_effect_has_lb = true;
                        new_effect_lb =
                            lb + increment_coefficient * self.variable_before_action_ub[lhs][op_id];
                    }
                }

                let assignment_result =
                    self.check_coefficient_in_preconditions(&coefficients, op_id);
                if assignment_result.0.0 {
                    new_assignment_has_ub = true;
                    new_assignment_ub = new_assignment_ub.min(assignment_result.1.0 + constant);
                }
                if assignment_result.0.1 {
                    new_assignment_has_lb = true;
                    new_assignment_lb = new_assignment_lb.max(assignment_result.1.1 + constant);
                }

                let mut increment_coefficients = coefficients.clone();
                increment_coefficients[lhs] -= 1.0;
                let increment_result =
                    self.check_coefficient_in_preconditions(&increment_coefficients, op_id);
                if increment_result.0.0 {
                    new_effect_has_ub = true;
                    new_effect_ub = new_effect_ub.min(increment_result.1.0 + constant);
                }
                if increment_result.0.1 {
                    new_effect_has_lb = true;
                    new_effect_lb = new_effect_lb.max(increment_result.1.1 + constant);
                }

                if new_assignment_has_ub
                    && (!self.assignment_has_ub[op_id][lhs]
                        || (new_assignment_ub - self.assignment_ub[op_id][lhs]).abs()
                            >= self.precision)
                {
                    change = true;
                    self.assignment_has_ub[op_id][lhs] = true;
                    self.assignment_ub[op_id][lhs] = new_assignment_ub;
                }

                if new_assignment_has_lb
                    && (!self.assignment_has_lb[op_id][lhs]
                        || (new_assignment_lb - self.assignment_lb[op_id][lhs]).abs()
                            >= self.precision)
                {
                    change = true;
                    self.assignment_has_lb[op_id][lhs] = true;
                    self.assignment_lb[op_id][lhs] = new_assignment_lb;
                }

                if new_effect_has_ub
                    && (!self.effect_has_ub[op_id][lhs]
                        || (new_effect_ub - self.effect_ub[op_id][lhs]).abs() >= self.precision)
                {
                    change = true;
                    self.effect_has_ub[op_id][lhs] = true;
                    self.effect_ub[op_id][lhs] = new_effect_ub;
                }

                if new_effect_has_lb
                    && (!self.effect_has_lb[op_id][lhs]
                        || (new_effect_lb - self.effect_lb[op_id][lhs]).abs() >= self.precision)
                {
                    change = true;
                    self.effect_has_lb[op_id][lhs] = true;
                    self.effect_lb[op_id][lhs] = new_effect_lb;
                }
            }
        }

        change
    }

    #[allow(clippy::needless_range_loop)]
    fn check_coefficient_in_preconditions(
        &self,
        coefficients: &[f64],
        op_id: usize,
    ) -> ((bool, bool), (f64, f64)) {
        let mut has_ub = false;
        let mut has_lb = false;
        let mut ub = f64::MAX;
        let mut lb = f64::MIN;

        for condition in &self.operator_conditions[op_id] {
            let mut has_scale = true;
            let mut scale_initialized = false;
            let mut scale = 0.0;

            for n_id in 0..self.num_numeric_variables {
                let coefficient = coefficients[n_id];
                let condition_coefficient = condition.coefficients[n_id];
                if coefficient.abs() >= self.precision
                    && condition_coefficient.abs() >= self.precision
                {
                    let new_scale = coefficient / condition_coefficient;
                    if !scale_initialized {
                        scale = new_scale;
                        scale_initialized = true;
                    } else if (new_scale - scale).abs() >= self.precision {
                        has_scale = false;
                        break;
                    }
                } else if coefficient.abs() >= self.precision
                    || condition_coefficient.abs() >= self.precision
                {
                    has_scale = false;
                    break;
                }
            }

            if has_scale {
                if scale >= self.precision {
                    has_lb = true;
                    lb = lb.max((-condition.constant + condition.epsilon) * scale);
                } else if scale <= -self.precision {
                    has_ub = true;
                    ub = ub.min((-condition.constant + condition.epsilon) * scale);
                }
            }
        }

        ((has_ub, has_lb), (ub, lb))
    }

    fn extract_operator_effects(&mut self, task: &dyn AbstractNumericTask) {
        for operator_id in 0..self.num_actions {
            let operator = task.get_operators().get(operator_id).unwrap_or_else(|| {
                panic!("operator id {operator_id} is out of bounds for numeric bound effects")
            });
            let linearized_effects = task
                .linearized_assignment_effects(operator_id)
                .unwrap_or_else(|error| {
                    panic!(
                        "failed to linearize numeric bound effects for operator {operator_id}: {error}"
                    )
                });

            for (assignment_effect_id, linearized_effect) in linearized_effects.iter().enumerate() {
                let assignment_effect = operator.assignment_effects().get(assignment_effect_id).unwrap_or_else(|| {
                    panic!(
                        "assignment effect id {assignment_effect_id} is out of bounds for operator {operator_id}"
                    )
                });
                let Some(lhs) = self.local_numeric_var_id(linearized_effect.affected_var_id) else {
                    continue;
                };
                let assignment_expression = self.assignment_expression(linearized_effect, lhs);

                if self.is_simple_effect(assignment_effect, &assignment_expression, lhs) {
                    self.prepared_simple_effects[operator_id][lhs] =
                        Some(assignment_expression.constant);
                    continue;
                }

                match assignment_effect.operation() {
                    AssignmentOperation::Assign
                    | AssignmentOperation::Plus
                    | AssignmentOperation::Minus => {}
                    AssignmentOperation::Times | AssignmentOperation::Divide => todo!(
                        "numeric bound propagation for multiplicative assignment effect {} on operator {operator_id} is not implemented yet",
                        assignment_effect_id
                    ),
                }

                self.prepared_linear_effects[operator_id].push(PreparedLinearEffect {
                    lhs,
                    coefficients: assignment_expression.coefficients,
                    constant: assignment_expression.constant,
                });
            }
        }
    }

    fn assignment_expression(
        &self,
        linearized_effect: &planners_sas::numeric::utils::linear_effects::LinearNumericEffect,
        lhs: usize,
    ) -> LinearExpression {
        LinearExpression::variable(self.num_numeric_variables, lhs).add(&LinearExpression {
            coefficients: self.project_coefficients(&linearized_effect.delta.coefficients),
            constant: linearized_effect.delta.constant,
        })
    }

    fn is_simple_effect(
        &self,
        assignment_effect: &planners_sas::numeric::numeric_task::AssignmentEffect,
        assignment_expression: &LinearExpression,
        lhs: usize,
    ) -> bool {
        if !matches!(
            assignment_effect.operation(),
            AssignmentOperation::Plus | AssignmentOperation::Minus
        ) {
            return false;
        }

        if assignment_expression.constant.abs() < self.precision {
            return false;
        }

        for (var_id, &coefficient) in assignment_expression.coefficients.iter().enumerate() {
            let expected = if var_id == lhs { 1.0 } else { 0.0 };
            if (coefficient - expected).abs() >= self.precision {
                return false;
            }
        }

        true
    }

    fn extract_operator_conditions(
        &self,
        task: &dyn AbstractNumericTask,
        operator_id: usize,
    ) -> Vec<BoundCondition> {
        let operator = task.get_operators().get(operator_id).unwrap_or_else(|| {
            panic!("operator id {operator_id} is out of bounds for numeric bound extraction")
        });
        let mut conditions = Vec::new();

        for precondition in operator.preconditions() {
            if precondition.value() > 0 {
                continue;
            }

            if let Some(helper_conditions) = self
                .numeric_helper
                .comparison_fact_conditions(precondition.var(), 0)
            {
                conditions.extend(helper_conditions.iter().map(|condition| BoundCondition {
                    coefficients: self.project_coefficients(&condition.coefficients),
                    constant: condition.constant,
                    epsilon: if condition.is_strictly_greater {
                        self.epsilon
                    } else {
                        0.0
                    },
                }));
            }
        }

        conditions
    }

    fn local_numeric_var_id(&self, actual_numeric_var_id: usize) -> Option<usize> {
        self.numeric_variable_ids
            .iter()
            .position(|&numeric_var_id| numeric_var_id == actual_numeric_var_id)
    }

    fn project_coefficients(&self, coefficients: &[f64]) -> Vec<f64> {
        self.numeric_variable_ids
            .iter()
            .map(|&numeric_var_id| coefficients.get(numeric_var_id).copied().unwrap_or(0.0))
            .collect()
    }
}
