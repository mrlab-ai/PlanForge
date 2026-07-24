use std::collections::BTreeMap;
use std::path::Path;

use planforge_cplex::{
    Constraint, Model, ObjectiveSense, SolveStatus, Variable, assert_unrestricted_license,
};
use planforge_sas::numeric_task::AbstractNumericTask;
use planforge_sas::state_registry::{ConcreteState, StateRegistry};

use super::{NumericPotentialConfig, NumericPotentialFunction, PotentialTask};

#[derive(Debug)]
pub enum OptimizationOutcome {
    Optimal {
        value: f64,
        function: NumericPotentialFunction,
    },
    Infeasible,
    Unbounded {
        primal_ray: Vec<f64>,
    },
    ResourceLimit(SolveStatus),
}

#[derive(Debug, Clone)]
pub(super) struct PotentialSystem {
    pub(super) variables: Vec<Variable>,
    pub(super) constraints: Vec<Constraint>,
    pub(super) fact_columns: Vec<Vec<usize>>,
    pub(super) constant_column: Option<usize>,
    pub(super) weight_columns: Vec<usize>,
    pub(super) operator_rows: Vec<Option<usize>>,
    pub(super) conditioned_goal: Option<(usize, usize)>,
}

#[derive(Debug)]
pub struct NumericPotentialOptimizer {
    task: PotentialTask,
    system: PotentialSystem,
    model: Model,
    max_potential: f64,
    ignore_numeric_variables: bool,
    prop_scratch: Vec<usize>,
    numeric_scratch: Vec<f64>,
}

impl NumericPotentialOptimizer {
    pub fn new(
        task: &dyn AbstractNumericTask,
        config: &NumericPotentialConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        assert_unrestricted_license()
            .map_err(|error| format!("numeric_potential requires unrestricted CPLEX: {error}"))?;
        let task = PotentialTask::build(
            task,
            config.precision,
            config.epsilon,
            config.ignore_numeric_variables,
            config.bounds,
            config.simple_action_bounds,
        )?;
        if task.goal_unsatisfiable {
            return Ok(Self {
                system: PotentialSystem {
                    variables: Vec::new(),
                    constraints: Vec::new(),
                    fact_columns: Vec::new(),
                    constant_column: None,
                    weight_columns: Vec::new(),
                    operator_rows: vec![None; task.operators.len()],
                    conditioned_goal: None,
                },
                model: Model::new("numeric_potential_early_dead_end")
                    .map_err(|error| error.to_string())?,
                task,
                max_potential: config.max_potential,
                ignore_numeric_variables: config.ignore_numeric_variables,
                prop_scratch: Vec::new(),
                numeric_scratch: Vec::new(),
            });
        }
        let system = build_plain_system(&task, config.max_potential, false, f64::INFINITY)?;
        let mut model = Model::new("numeric_potential").map_err(|error| error.to_string())?;
        model
            .load(
                ObjectiveSense::Maximize,
                &system.variables,
                &system.constraints,
            )
            .map_err(|error| error.to_string())?;
        if config.dump_lp {
            model
                .write(Path::new("numeric_potential.lp"))
                .map_err(|error| format!("failed to dump numeric potential LP: {error}"))?;
        }
        Ok(Self {
            task,
            system,
            model,
            max_potential: config.max_potential,
            ignore_numeric_variables: config.ignore_numeric_variables,
            prop_scratch: Vec::new(),
            numeric_scratch: Vec::new(),
        })
    }

    pub fn task(&self) -> &PotentialTask {
        &self.task
    }

    pub fn num_columns(&self) -> usize {
        self.system.variables.len()
    }

    pub fn num_rows(&self) -> usize {
        self.system.constraints.len()
    }

    pub fn validate_duality(
        &mut self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        expected: f64,
        tolerance: f64,
    ) -> Result<(f64, usize, usize), String> {
        let infinity = Model::infinity();
        let variables = self
            .task
            .operators
            .iter()
            .map(|operator| {
                Variable::new(
                    0.0,
                    if operator.reachable { infinity } else { 0.0 },
                    operator.cost,
                )
            })
            .collect::<Vec<_>>();
        let mut constraints = Vec::new();
        state.fill_state(registry, &mut self.prop_scratch);
        let mut goals = vec![None; self.task.domain_sizes.len()];
        for goal in &self.task.propositional_goals {
            goals[goal.var()] = Some(goal.value());
        }
        for (var_id, &domain_size) in self.task.domain_sizes.iter().enumerate() {
            if self.task.derived_propositional[var_id] {
                continue;
            }
            for value in 0..domain_size {
                if goals[var_id].is_none()
                    && self.task.reachable_facts.as_ref().is_some_and(|facts| {
                        !facts
                            .get(var_id)
                            .and_then(|values| values.get(value))
                            .copied()
                            .unwrap_or(true)
                    })
                {
                    continue;
                }
                let mut coefficients = BTreeMap::new();
                for (operator_id, operator) in self.task.operators.iter().enumerate() {
                    if operator.effects.contains(&(var_id, value)) {
                        *coefficients.entry(operator_id).or_insert(0.0) += 1.0;
                    }
                    if operator
                        .effects
                        .iter()
                        .any(|(effect_var, _)| *effect_var == var_id)
                        && operator
                            .preconditions
                            .iter()
                            .any(|fact| fact.var() == var_id && fact.value() == value)
                    {
                        *coefficients.entry(operator_id).or_insert(0.0) -= 1.0;
                    }
                }
                let coefficients = nonzero_coefficients(coefficients);
                if coefficients.is_empty() {
                    continue;
                }
                let lower = f64::from(goals[var_id] == Some(value))
                    - f64::from(self.prop_scratch[var_id] == value);
                constraints.push(Constraint::new(lower, infinity, coefficients));
            }
        }
        let initial_features =
            self.task
                .feature_values(state, registry, &mut self.numeric_scratch)?;
        for (feature_id, bounds) in self.task.feature_goal_bounds.iter().enumerate() {
            let coefficients = self
                .task
                .operators
                .iter()
                .enumerate()
                .filter_map(|(operator_id, operator)| {
                    let delta = operator.numeric_delta(feature_id);
                    (delta != 0.0).then_some((operator_id, delta))
                })
                .collect::<Vec<_>>();
            if bounds.lower.is_finite() {
                constraints.push(Constraint::new(
                    bounds.lower - initial_features[feature_id],
                    infinity,
                    coefficients.clone(),
                ));
            }
            if bounds.upper.is_finite() {
                constraints.push(Constraint::new(
                    -infinity,
                    bounds.upper - initial_features[feature_id],
                    coefficients,
                ));
            }
        }
        let columns = variables.len();
        let rows = constraints.len();
        let mut model =
            Model::new("numeric_potential_duality").map_err(|error| error.to_string())?;
        model
            .load(ObjectiveSense::Minimize, &variables, &constraints)
            .map_err(|error| error.to_string())?;
        let status = model.solve().map_err(|error| error.to_string())?;
        if status != SolveStatus::Optimal {
            return Err(format!(
                "numeric potential duality validation failed: operator-counting LP status is {status:?} while h_pot(s_0)={expected}"
            ));
        }
        let optimum = model.objective_value().map_err(|error| error.to_string())?;
        if (optimum - expected).abs() > tolerance {
            return Err(format!(
                "numeric potential duality validation failed: |{expected} - {optimum}|={} > {tolerance}",
                (expected - optimum).abs()
            ));
        }
        Ok((optimum, columns, rows))
    }

    pub fn prepare_for_ordinary_potential(&mut self) -> Result<(), String> {
        if self.system.conditioned_goal.is_none() {
            return Ok(());
        }
        self.system = build_plain_system(&self.task, self.max_potential, false, f64::INFINITY)?;
        self.model = Model::new("numeric_potential").map_err(|error| error.to_string())?;
        self.model
            .load(
                ObjectiveSense::Maximize,
                &self.system.variables,
                &self.system.constraints,
            )
            .map_err(|error| error.to_string())
    }

    pub fn optimize_for_state(
        &mut self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<OptimizationOutcome, String> {
        if self.task.goal_unsatisfiable {
            return Ok(OptimizationOutcome::Unbounded {
                primal_ray: Vec::new(),
            });
        }
        let objective = self.state_objective(state, registry)?;
        self.model
            .set_objective(&objective)
            .map_err(|error| error.to_string())?;
        self.solve_current_objective()
    }

    pub fn optimize_for_all_propositional_states(
        &mut self,
        numeric_reference: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<OptimizationOutcome, String> {
        let mut objective = vec![0.0; self.system.variables.len()];
        for (var_id, &domain_size) in self.task.domain_sizes.iter().enumerate() {
            if self.task.derived_propositional[var_id] {
                continue;
            }
            let probability = 1.0 / domain_size as f64;
            for value in 0..domain_size {
                objective[self.system.fact_columns[var_id][value]] = probability;
            }
        }
        if let Some(column) = self.system.constant_column {
            objective[column] = 1.0;
            let values =
                self.task
                    .feature_values(numeric_reference, registry, &mut self.numeric_scratch)?;
            for (feature_id, value) in values.into_iter().enumerate() {
                objective[self.system.weight_columns[feature_id]] = value;
            }
        }
        self.model
            .set_objective(&objective)
            .map_err(|error| error.to_string())?;
        self.solve_current_objective()
    }

    pub fn optimize_for_samples(
        &mut self,
        states: &[ConcreteState],
        registry: &StateRegistry<'_>,
    ) -> Result<OptimizationOutcome, String> {
        if states.is_empty() {
            return Err("numeric_potential cannot optimize for an empty sample set".to_string());
        }
        let mut objective = vec![0.0; self.system.variables.len()];
        for state in states {
            let state_objective = self.state_objective(state, registry)?;
            for (sum, coefficient) in objective.iter_mut().zip(state_objective) {
                *sum += coefficient;
            }
        }
        let divisor = states.len() as f64;
        for coefficient in &mut objective {
            *coefficient /= divisor;
        }
        self.model
            .set_objective(&objective)
            .map_err(|error| error.to_string())?;
        self.solve_current_objective()
    }

    pub fn conditionable_goals(&self) -> Vec<(usize, usize)> {
        if self.ignore_numeric_variables || self.task.features.is_empty() {
            return Vec::new();
        }
        self.task
            .propositional_goals
            .iter()
            .filter_map(|goal| {
                self.task
                    .operators
                    .iter()
                    .any(|operator| operator.effects.contains(&(goal.var(), goal.value())))
                    .then_some((goal.var(), goal.value()))
            })
            .collect()
    }

    pub fn goal_achievers(&self, var: usize, value: usize) -> Vec<usize> {
        let mut achievers: Vec<usize> = Vec::new();
        'operators: for (operator_id, operator) in self.task.operators.iter().enumerate() {
            if !operator.reachable || !operator.effects.contains(&(var, value)) {
                continue;
            }
            for &representative_id in &achievers {
                let representative = &self.task.operators[representative_id];
                if operator.cost.to_bits() == representative.cost.to_bits()
                    && operator.numeric_deltas == representative.numeric_deltas
                    && operator.numeric_precondition_bounds
                        == representative.numeric_precondition_bounds
                {
                    continue 'operators;
                }
            }
            achievers.push(operator_id);
        }
        achievers
    }

    pub fn optimize_for_conditioned_goal(
        &mut self,
        var: usize,
        value: usize,
        achiever: usize,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<OptimizationOutcome, String> {
        if !self.task.operators[achiever]
            .effects
            .contains(&(var, value))
        {
            return Err(format!(
                "operator {achiever} does not achieve conditioned goal {var}={value}"
            ));
        }
        self.system =
            build_conditioned_system(&self.task, self.max_potential, var, value, achiever)?;
        self.model =
            Model::new("numeric_potential_conditioned").map_err(|error| error.to_string())?;
        self.model
            .load(
                ObjectiveSense::Maximize,
                &self.system.variables,
                &self.system.constraints,
            )
            .map_err(|error| error.to_string())?;
        let objective = self.state_objective(state, registry)?;
        self.model
            .set_objective(&objective)
            .map_err(|error| error.to_string())?;
        self.solve_current_objective()
    }

    fn solve_current_objective(&mut self) -> Result<OptimizationOutcome, String> {
        let status = self.model.solve().map_err(|error| error.to_string())?;
        match status {
            SolveStatus::Optimal => {
                let value = self
                    .model
                    .objective_value()
                    .map_err(|error| error.to_string())?;
                let solution = self.model.solution().map_err(|error| error.to_string())?;
                Ok(OptimizationOutcome::Optimal {
                    value,
                    function: self.extract_function(&solution, self.system.conditioned_goal),
                })
            }
            SolveStatus::Infeasible => Ok(OptimizationOutcome::Infeasible),
            SolveStatus::Unbounded => Ok(OptimizationOutcome::Unbounded {
                primal_ray: self.model.primal_ray().unwrap_or_default(),
            }),
            SolveStatus::IterationLimit
            | SolveStatus::TimeLimit
            | SolveStatus::ObjectiveLimit
            | SolveStatus::UserAbort
            | SolveStatus::DeterministicTimeLimit
            | SolveStatus::Unknown(_) => Ok(OptimizationOutcome::ResourceLimit(status)),
        }
    }

    pub fn zero_function(&self) -> NumericPotentialFunction {
        NumericPotentialFunction::new(
            self.task
                .domain_sizes
                .iter()
                .map(|domain_size| vec![0.0; *domain_size])
                .collect(),
            vec![0.0; self.task.features.len()],
            0.0,
            None,
        )
    }

    pub(super) fn homogeneous_system(&self) -> Result<PotentialSystem, String> {
        build_plain_system(&self.task, self.max_potential, true, 1.0)
    }

    pub(super) fn objective_for_state(
        &mut self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
        system: &PotentialSystem,
    ) -> Result<Vec<f64>, String> {
        let mut objective = vec![0.0; system.variables.len()];
        state.fill_state(registry, &mut self.prop_scratch);
        for (var_id, &value) in self.prop_scratch.iter().enumerate() {
            objective[system.fact_columns[var_id][value]] += 1.0;
        }
        if let Some(column) = system.constant_column {
            objective[column] = 1.0;
            let values = self
                .task
                .feature_values(state, registry, &mut self.numeric_scratch)?;
            for (feature_id, value) in values.into_iter().enumerate() {
                objective[system.weight_columns[feature_id]] += value;
            }
        }
        Ok(objective)
    }

    pub(super) fn function_from_system_solution(
        &self,
        system: &PotentialSystem,
        solution: &[f64],
    ) -> NumericPotentialFunction {
        extract_function_for_system(&self.task, system, solution, None)
    }

    pub(super) fn ordinary_system(&self) -> &PotentialSystem {
        assert!(
            self.system.conditioned_goal.is_none(),
            "joint potential LP requires the ordinary potential system"
        );
        &self.system
    }

    fn state_objective(
        &mut self,
        state: &ConcreteState,
        registry: &StateRegistry<'_>,
    ) -> Result<Vec<f64>, String> {
        let mut objective = vec![0.0; self.system.variables.len()];
        state.fill_state(registry, &mut self.prop_scratch);
        if !self.system.fact_columns.is_empty() {
            for (var_id, &value) in self.prop_scratch.iter().enumerate() {
                objective[self.system.fact_columns[var_id][value]] += 1.0;
            }
        }
        if let Some(column) = self.system.constant_column {
            objective[column] = 1.0;
            let values = self
                .task
                .feature_values(state, registry, &mut self.numeric_scratch)?;
            for (feature_id, value) in values.into_iter().enumerate() {
                objective[self.system.weight_columns[feature_id]] += value;
            }
        }
        Ok(objective)
    }

    fn extract_function(
        &self,
        solution: &[f64],
        conditioned_goal: Option<(usize, usize)>,
    ) -> NumericPotentialFunction {
        extract_function_for_system(&self.task, &self.system, solution, conditioned_goal)
    }
}

fn extract_function_for_system(
    task: &PotentialTask,
    system: &PotentialSystem,
    solution: &[f64],
    conditioned_goal: Option<(usize, usize)>,
) -> NumericPotentialFunction {
    assert_eq!(solution.len(), system.variables.len());
    let fact_potentials = if system.fact_columns.is_empty() {
        task.domain_sizes
            .iter()
            .map(|domain_size| vec![0.0; *domain_size])
            .collect()
    } else {
        system
            .fact_columns
            .iter()
            .zip(&task.domain_sizes)
            .map(|(columns, &domain_size)| {
                columns[..domain_size]
                    .iter()
                    .map(|&column| solution[column])
                    .collect()
            })
            .collect()
    };
    let numeric_potentials = system
        .weight_columns
        .iter()
        .map(|&column| solution[column])
        .collect();
    let constant_potential = system
        .constant_column
        .map(|column| solution[column])
        .unwrap_or(0.0);
    NumericPotentialFunction::new(
        fact_potentials,
        numeric_potentials,
        constant_potential,
        conditioned_goal,
    )
}

fn build_plain_system(
    task: &PotentialTask,
    max_potential: f64,
    zero_costs: bool,
    box_bound: f64,
) -> Result<PotentialSystem, String> {
    let infinity = Model::infinity();
    let goal_bounds = if zero_costs {
        &task.ray_feature_goal_bounds
    } else {
        &task.feature_goal_bounds
    };
    let prop_upper = if max_potential.is_finite() {
        max_potential
    } else {
        infinity
    };
    let mut variables = Vec::new();
    let mut fact_columns = Vec::with_capacity(task.domain_sizes.len());
    for (var_id, &domain_size) in task.domain_sizes.iter().enumerate() {
        let fixed = task.derived_propositional[var_id];
        let mut columns = Vec::with_capacity(domain_size + 1);
        for _ in 0..=domain_size {
            columns.push(variables.len());
            let variable = if fixed {
                Variable::new(0.0, 0.0, 0.0)
            } else {
                Variable::new(-infinity, prop_upper, 0.0)
            };
            variables.push(box_variable(variable, box_bound));
        }
        fact_columns.push(columns);
    }

    let constant_column = (!task.features.is_empty()).then(|| {
        let id = variables.len();
        variables.push(box_variable(
            Variable::new(-infinity, infinity, 0.0),
            box_bound,
        ));
        id
    });
    let mut weight_columns = Vec::with_capacity(task.features.len());
    let mut m_columns = Vec::with_capacity(task.features.len());
    for (feature_id, bounds) in goal_bounds.iter().enumerate() {
        let has_lower = bounds.lower.is_finite();
        let has_upper = bounds.upper.is_finite();
        let (lower, upper) =
            if task.assignment_target_features[feature_id] || (!has_lower && !has_upper) {
                (0.0, 0.0)
            } else if has_lower && !has_upper {
                (-infinity, 0.0)
            } else if !has_lower && has_upper {
                (0.0, infinity)
            } else {
                (-infinity, infinity)
            };
        weight_columns.push(variables.len());
        variables.push(box_variable(Variable::new(lower, upper, 0.0), box_bound));
        m_columns.push((has_lower || has_upper).then(|| {
            let id = variables.len();
            variables.push(box_variable(
                Variable::new(-infinity, infinity, 0.0),
                box_bound,
            ));
            id
        }));
    }

    let mut constraints = Vec::new();
    let mut operator_rows = vec![None; task.operators.len()];
    for (operator_id, operator) in task.operators.iter().enumerate() {
        if !operator.reachable {
            continue;
        }
        let mut pre_by_var = BTreeMap::new();
        for fact in &operator.preconditions {
            pre_by_var.insert(fact.var(), fact.value());
        }
        let mut coefficients = BTreeMap::<usize, f64>::new();
        for &(var_id, post) in &operator.effects {
            if task.derived_propositional[var_id] {
                continue;
            }
            let pre = pre_by_var
                .get(&var_id)
                .copied()
                .unwrap_or(task.domain_sizes[var_id]);
            *coefficients.entry(fact_columns[var_id][pre]).or_default() += 1.0;
            *coefficients.entry(fact_columns[var_id][post]).or_default() -= 1.0;
        }
        for &(feature_id, delta) in &operator.numeric_deltas {
            *coefficients.entry(weight_columns[feature_id]).or_default() -= delta;
        }
        operator_rows[operator_id] = Some(constraints.len());
        constraints.push(Constraint::new(
            -infinity,
            if zero_costs { 0.0 } else { operator.cost },
            nonzero_coefficients(coefficients),
        ));
    }

    let mut goal_values = task.domain_sizes.clone();
    for goal in &task.propositional_goals {
        goal_values[goal.var()] = goal.value();
    }
    for (var_id, &goal_value) in goal_values.iter().enumerate() {
        if task.derived_propositional[var_id] {
            continue;
        }
        let column = fact_columns[var_id][goal_value];
        variables[column].lower = 0.0;
        variables[column].upper = 0.0;
        let undef = fact_columns[var_id][task.domain_sizes[var_id]];
        for (value, &value_column) in fact_columns[var_id]
            .iter()
            .take(task.domain_sizes[var_id])
            .enumerate()
        {
            if goal_value == task.domain_sizes[var_id]
                && task.reachable_facts.as_ref().is_some_and(|facts| {
                    !facts
                        .get(var_id)
                        .and_then(|values| values.get(value))
                        .copied()
                        .unwrap_or(true)
                })
            {
                continue;
            }
            constraints.push(Constraint::new(
                -infinity,
                0.0,
                vec![(value_column, 1.0), (undef, -1.0)],
            ));
        }
    }

    if let Some(constant) = constant_column {
        let mut awareness = vec![(constant, 1.0)];
        for (var_id, &goal_value) in goal_values.iter().enumerate() {
            if !task.derived_propositional[var_id] && goal_value == task.domain_sizes[var_id] {
                awareness.push((fact_columns[var_id][goal_value], 1.0));
            }
        }
        for column in m_columns.iter().flatten() {
            awareness.push((*column, 1.0));
        }
        awareness.sort_unstable_by_key(|(column, _)| *column);
        constraints.push(Constraint::new(-infinity, 0.0, awareness));

        for (feature_id, bounds) in goal_bounds.iter().enumerate() {
            let Some(m) = m_columns[feature_id] else {
                continue;
            };
            let weight = weight_columns[feature_id];
            if bounds.lower.is_finite() {
                constraints.push(Constraint::new(
                    0.0,
                    infinity,
                    vec![(weight, -bounds.lower), (m, 1.0)],
                ));
            }
            if bounds.upper.is_finite() {
                constraints.push(Constraint::new(
                    0.0,
                    infinity,
                    vec![(weight, -bounds.upper), (m, 1.0)],
                ));
            }
        }
    }

    Ok(PotentialSystem {
        variables,
        constraints,
        fact_columns,
        constant_column,
        weight_columns,
        operator_rows,
        conditioned_goal: None,
    })
}

fn box_variable(variable: Variable, box_bound: f64) -> Variable {
    if box_bound.is_finite() {
        Variable::new(
            variable.lower.max(-box_bound),
            variable.upper.min(box_bound),
            variable.objective,
        )
    } else {
        variable
    }
}

fn build_conditioned_system(
    task: &PotentialTask,
    _max_potential: f64,
    goal_var: usize,
    goal_value: usize,
    achiever_id: usize,
) -> Result<PotentialSystem, String> {
    let infinity = Model::infinity();
    let mut variables = vec![Variable::new(-infinity, infinity, 0.0)];
    let constant_column = 0;
    let mut weight_columns = Vec::with_capacity(task.features.len());
    for (feature_id, _) in task.features.iter().enumerate() {
        weight_columns.push(variables.len());
        variables.push(if task.assignment_target_features[feature_id] {
            Variable::new(0.0, 0.0, 0.0)
        } else {
            Variable::new(-infinity, infinity, 0.0)
        });
    }

    let mut constraints = Vec::new();
    let mut operator_rows = vec![None; task.operators.len()];
    for (operator_id, operator) in task.operators.iter().enumerate() {
        if operator_id == achiever_id || !operator.reachable {
            continue;
        }
        let coefficients = operator
            .numeric_deltas
            .iter()
            .map(|&(feature_id, delta)| (weight_columns[feature_id], -delta))
            .collect();
        operator_rows[operator_id] = Some(constraints.len());
        constraints.push(Constraint::new(-infinity, operator.cost, coefficients));
    }

    let achiever = &task.operators[achiever_id];
    let mut inequalities: Vec<(Vec<(usize, f64)>, f64)> = Vec::new();
    for equality in &task.global_linear_equalities {
        inequalities.push((equality.coefficients.clone(), equality.rhs));
        inequalities.push((
            equality
                .coefficients
                .iter()
                .map(|&(variable_id, coefficient)| (variable_id, -coefficient))
                .collect(),
            -equality.rhs,
        ));
    }
    for &(feature_id, bounds) in &achiever.numeric_precondition_bounds {
        let feature = &task.features[feature_id];
        if bounds.lower.is_finite() {
            inequalities.push((
                feature
                    .coefficients
                    .iter()
                    .map(|&(variable_id, coefficient)| (variable_id, -coefficient))
                    .collect(),
                feature.constant - bounds.lower,
            ));
        }
        if bounds.upper.is_finite() {
            inequalities.push((
                feature.coefficients.clone(),
                bounds.upper - feature.constant,
            ));
        }
    }
    let dual_columns: Vec<usize> = inequalities
        .iter()
        .map(|_| {
            let column = variables.len();
            variables.push(Variable::new(0.0, infinity, 0.0));
            column
        })
        .collect();

    let mut balance_rows = (0..task.numeric_variable_count)
        .map(|_| BTreeMap::new())
        .collect::<Vec<BTreeMap<usize, f64>>>();
    for (row_id, (row, _)) in inequalities.iter().enumerate() {
        for &(numeric_var_id, coefficient) in row {
            balance_rows[numeric_var_id].insert(dual_columns[row_id], coefficient);
        }
    }
    for (feature_id, feature) in task.features.iter().enumerate() {
        for &(numeric_var_id, coefficient) in &feature.coefficients {
            balance_rows[numeric_var_id].insert(weight_columns[feature_id], -coefficient);
        }
    }
    for coefficients in balance_rows {
        if !coefficients.is_empty() {
            constraints.push(Constraint::new(
                0.0,
                0.0,
                nonzero_coefficients(coefficients),
            ));
        }
    }

    let mut achiever_row = vec![(constant_column, 1.0)];
    for (feature_id, feature) in task.features.iter().enumerate() {
        if feature.affine && feature.constant != 0.0 {
            achiever_row.push((weight_columns[feature_id], feature.constant));
        }
    }
    for (row_id, (_, rhs)) in inequalities.iter().enumerate() {
        if *rhs != 0.0 {
            achiever_row.push((dual_columns[row_id], *rhs));
        }
    }
    achiever_row.sort_unstable_by_key(|(column, _)| *column);
    constraints.push(Constraint::new(-infinity, achiever.cost, achiever_row));

    Ok(PotentialSystem {
        variables,
        constraints,
        fact_columns: Vec::new(),
        constant_column: Some(constant_column),
        weight_columns,
        operator_rows,
        conditioned_goal: Some((goal_var, goal_value)),
    })
}

fn nonzero_coefficients(coefficients: BTreeMap<usize, f64>) -> Vec<(usize, f64)> {
    coefficients
        .into_iter()
        .filter(|(_, coefficient)| *coefficient != 0.0)
        .collect()
}
