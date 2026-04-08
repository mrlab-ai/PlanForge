use super::lm_cut_numeric_heuristic::LmCutNumericConfig;
use super::numeric_bound::NumericBound;
use ordered_float::NotNan;
use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
use planners_sas::numeric::numeric_task::{
    metric_operator_cost_from_initial_values, AbstractNumericTask, AssignmentEffect,
    AssignmentOperation, Effect, Fact, Operator,
};
use planners_sas::numeric::utils::linear_effects::LinearExpression;
use planners_sas::numeric::utils::linear_effects::LinearNumericEffect;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

#[derive(Debug, Clone)]
struct NumericCondition {
    coefficients: Vec<f64>,
    constant: f64,
    is_strictly_greater: bool,
    name: String,
}

impl NumericCondition {
    fn from_expression(
        expression: LinearExpression,
        is_strictly_greater: bool,
        name: String,
    ) -> Self {
        Self {
            coefficients: expression.coefficients,
            constant: expression.constant,
            is_strictly_greater,
            name,
        }
    }
}

fn invert_comparison_operator(operator: &ComparisonOperator) -> ComparisonOperator {
    match operator {
        ComparisonOperator::LessThan => ComparisonOperator::GreaterThanOrEqual,
        ComparisonOperator::LessThanOrEqual => ComparisonOperator::GreaterThan,
        ComparisonOperator::Equal => ComparisonOperator::UnEqual,
        ComparisonOperator::GreaterThanOrEqual => ComparisonOperator::LessThan,
        ComparisonOperator::GreaterThan => ComparisonOperator::LessThanOrEqual,
        ComparisonOperator::UnEqual => ComparisonOperator::Equal,
    }
}

fn comparison_operator_for_fact_value(
    comparison_axiom: &ComparisonAxiom,
    fact_value: i32,
) -> Option<ComparisonOperator> {
    assert!(
        fact_value == 0 || fact_value == 1,
        "comparison fact value must be boolean-like, got {fact_value}"
    );

    let operator = if fact_value == 0 {
        comparison_axiom.get_operator().clone()
    } else {
        invert_comparison_operator(comparison_axiom.get_operator())
    };

    match operator {
        ComparisonOperator::UnEqual => None,
        supported => Some(supported),
    }
}

impl NumericCondition {
    fn evaluate_slack(&self, numeric_values: &[f64], epsilon: f64) -> f64 {
        let mut net = self.constant;
        if self.is_strictly_greater {
            net -= epsilon;
        }
        for (coefficient, value) in self.coefficients.iter().zip(numeric_values.iter()) {
            net += coefficient * value;
        }
        net
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropositionStatus {
    Unreached = 0,
    Reached = 1,
    GoalZone = 2,
    BeforeGoalZone = 3,
}

#[derive(Debug, Clone)]
pub struct RelaxedProposition {
    pub precondition_of: Vec<usize>,
    pub effect_of: Vec<usize>,
    pub id: usize,
    pub status: PropositionStatus,
    pub explored: bool,
    pub is_numeric_condition: bool,
    pub id_numeric_condition: Option<usize>,
    pub h_max_cost: f64,
    pub name: String,
}

impl RelaxedProposition {
    pub fn new(id: usize, name: String) -> Self {
        Self {
            precondition_of: Vec::new(),
            effect_of: Vec::new(),
            id,
            status: PropositionStatus::Unreached,
            explored: false,
            is_numeric_condition: false,
            id_numeric_condition: None,
            h_max_cost: f64::INFINITY,
            name,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RelaxedOperator {
    pub id: usize,
    pub original_op_id_1: Option<usize>,
    pub original_op_id_2: Option<usize>,
    pub precondition_ids: Vec<usize>,
    pub effect_ids: Vec<usize>,
    pub assignment_effect_ids: Vec<usize>,
    pub linear_assignment_effects: Vec<LinearNumericEffect>,
    pub sose_constants: Vec<f64>,
    pub conditional: bool,
    pub infinite: bool,
    pub base_cost_1: f64,
    pub base_cost_2: f64,
    pub cost_1: f64,
    pub cost_2: f64,
    pub unsatisfied_preconditions: usize,
    pub h_max_supporter: Option<usize>,
    pub h_max_supporter_cost: f64,
    pub name: String,
}

impl RelaxedOperator {
    pub fn new(
        precondition_ids: Vec<usize>,
        effect_ids: Vec<usize>,
        op_id: usize,
        base_cost: f64,
        name: String,
        conditional: bool,
    ) -> Self {
        Self {
            id: op_id,
            original_op_id_1: None,
            original_op_id_2: Some(op_id),
            precondition_ids,
            effect_ids,
            assignment_effect_ids: Vec::new(),
            linear_assignment_effects: Vec::new(),
            sose_constants: Vec::new(),
            conditional,
            infinite: false,
            base_cost_1: 0.0,
            base_cost_2: base_cost,
            cost_1: 0.0,
            cost_2: base_cost,
            unsatisfied_preconditions: 0,
            h_max_supporter: None,
            h_max_supporter_cost: 0.0,
            name,
        }
    }

    pub fn assert_well_formed(&self) {
        assert!(
            self.cost_1 >= 0.0,
            "relaxed operator cost_1 must stay non-negative"
        );
        assert!(
            self.cost_2 >= 0.0,
            "relaxed operator cost_2 must stay non-negative"
        );
        assert!(
            self.base_cost_1 >= 0.0,
            "relaxed operator base_cost_1 must stay non-negative"
        );
        assert!(
            self.base_cost_2 >= 0.0,
            "relaxed operator base_cost_2 must stay non-negative"
        );
        assert!(
            self.unsatisfied_preconditions <= self.precondition_ids.len(),
            "unsatisfied preconditions exceed precondition list"
        );
    }
}

pub type Landmark = Vec<(f64, usize)>;

pub struct LandmarkCutLandmarks<'task> {
    task: &'task dyn AbstractNumericTask,
    config: LmCutNumericConfig,
    propositions: Vec<RelaxedProposition>,
    proposition_index: Vec<Vec<usize>>,
    conditions: Vec<NumericCondition>,
    epsilons: Vec<f64>,
    comparison_fact_to_condition_ids: BTreeMap<(usize, i32), Vec<usize>>,
    comparison_axiom_by_var: BTreeMap<usize, usize>,
    operator_to_simple_effects: Vec<Vec<Option<f64>>>,
    relaxed_operators: Vec<RelaxedOperator>,
    original_to_relaxed_operators: Vec<Vec<usize>>,
    artificial_precondition_id: usize,
    artificial_goal_id: usize,
    num_propositions: usize,
    num_variables: usize,
    numeric_initial_state: Vec<f64>,
    priority_queue: BinaryHeap<(Reverse<NotNan<f64>>, usize)>,
    numeric_bound: NumericBound,
    use_bounds: bool,
    initialized: bool,
}

impl<'task> LandmarkCutLandmarks<'task> {
    pub fn new(task: &'task dyn AbstractNumericTask, config: LmCutNumericConfig) -> Self {
        assert!(
            config.precision >= 0.0,
            "LM-cut precision must be non-negative"
        );
        assert!(config.epsilon >= 0.0, "LM-cut epsilon must be non-negative");

        let use_bounds = config.bound_iterations > 0;
        let numeric_bound = NumericBound::new(task, config.precision);
        let mut result = Self {
            task,
            config,
            propositions: Vec::new(),
            proposition_index: Vec::new(),
            conditions: Vec::new(),
            epsilons: Vec::new(),
            comparison_fact_to_condition_ids: BTreeMap::new(),
            comparison_axiom_by_var: BTreeMap::new(),
            operator_to_simple_effects: Vec::new(),
            relaxed_operators: Vec::new(),
            original_to_relaxed_operators: Vec::new(),
            artificial_precondition_id: 0,
            artificial_goal_id: 1,
            num_propositions: 0,
            num_variables: 0,
            numeric_initial_state: Vec::new(),
            priority_queue: BinaryHeap::new(),
            numeric_bound,
            use_bounds,
            initialized: false,
        };
        result.initialize();
        result
    }

    fn initialize(&mut self) {
        assert!(!self.initialized, "LM-cut landmarks initialized twice");
        self.propositions.clear();
        self.proposition_index.clear();
        self.conditions.clear();
        self.epsilons.clear();
        self.comparison_fact_to_condition_ids.clear();
        self.comparison_axiom_by_var.clear();
        self.operator_to_simple_effects.clear();
        self.relaxed_operators.clear();
        self.original_to_relaxed_operators.clear();
        self.propositions.push(RelaxedProposition::new(
            self.artificial_precondition_id,
            "artificial".to_string(),
        ));
        self.propositions.push(RelaxedProposition::new(
            self.artificial_goal_id,
            "goal".to_string(),
        ));
        self.num_variables = usize::try_from(self.task.get_num_variables().max(0)).unwrap_or(0);
        self.proposition_index = vec![Vec::new(); self.num_variables];

        self.num_propositions = 2;
        self.build_propositional_propositions();
        self.build_numeric_condition_propositions();
        self.build_relaxed_operators();
        self.build_goal_operator();
        self.build_original_to_relaxed_index();
        self.build_cross_references();
        self.initialized = true;
    }

    fn build_propositional_propositions(&mut self) {
        for variable_id in 0..self.num_variables {
            let domain_size = usize::try_from(
                self.task
                    .get_variable_domain_size(variable_id as i32)
                    .expect("variable id must be valid")
                    .max(0),
            )
            .expect("domain size must fit usize");
            self.proposition_index[variable_id].reserve(domain_size);
            for value in 0..domain_size {
                let fact = Fact::new(variable_id as u32, value as i32);
                let proposition_id = self.propositions.len();
                let proposition = RelaxedProposition::new(
                    proposition_id,
                    self.task.get_fact_name(&fact).to_string(),
                );
                self.propositions.push(proposition);
                self.proposition_index[variable_id].push(proposition_id);
                self.num_propositions += 1;
            }
        }
    }

    fn build_relaxed_operators(&mut self) {
        let operators = self.task.get_operators();
        for (operator_id, operator) in operators.iter().enumerate() {
            let base_cost = self.calculate_base_operator_cost(operator_id, operator);
            self.build_relaxed_operator_for_operator(operator_id, operator, base_cost)
                .expect("LM-cut numeric operator construction must succeed");
        }

        self.build_supported_sose_operators()
            .expect("LM-cut supported SOSE operator construction must succeed");

        self.build_simple_effects()
            .expect("LM-cut simple-effect construction must succeed");

        for (axiom_offset, axiom) in self.task.axioms().iter().enumerate() {
            let operator_id = operators.len() + axiom_offset;
            self.build_relaxed_operator_for_axiom(operator_id, axiom);
        }

        self.delete_noops();
    }

    fn delete_noops(&mut self) {
        self.relaxed_operators
            .retain(|operator| !operator.effect_ids.is_empty());
    }

    fn build_goal_operator(&mut self) {
        let mut goal_preconditions = Vec::new();
        let mut seen = BTreeSet::new();
        for goal_index in 0..usize::try_from(self.task.get_num_goals().max(0)).unwrap_or(0) {
            let goal = self.task.get_goal_fact(goal_index as i32);
            for proposition_id in self.goal_proposition_ids(goal) {
                if seen.insert(proposition_id) {
                    goal_preconditions.push(proposition_id);
                }
            }
        }
        if goal_preconditions.is_empty() {
            goal_preconditions.push(self.artificial_precondition_id);
        }

        let mut goal_operator = RelaxedOperator::new(
            goal_preconditions,
            vec![self.artificial_goal_id],
            usize::MAX,
            0.0,
            "goal".to_string(),
            false,
        );
        goal_operator.original_op_id_2 = None;
        goal_operator.assert_well_formed();
        self.relaxed_operators.push(goal_operator);
    }

    fn goal_proposition_ids(&self, goal: &Fact) -> Vec<usize> {
        let mut goal_preconditions = Vec::new();
        let mut seen = BTreeSet::new();

        for proposition_id in self.precondition_proposition_ids(goal) {
            if seen.insert(proposition_id) {
                goal_preconditions.push(proposition_id);
            }
        }

        for axiom in self.task.axioms() {
            if axiom.var_id() != goal.var() || axiom.effect_value() as i32 != goal.value() {
                continue;
            }

            for condition in axiom.conditions() {
                for proposition_id in self.precondition_proposition_ids(condition) {
                    if seen.insert(proposition_id) {
                        goal_preconditions.push(proposition_id);
                    }
                }
            }
        }

        goal_preconditions
    }

    fn build_cross_references(&mut self) {
        for proposition in &mut self.propositions {
            proposition.precondition_of.clear();
            proposition.effect_of.clear();
        }

        for (operator_index, operator) in self.relaxed_operators.iter().enumerate() {
            for &proposition_id in &operator.precondition_ids {
                let proposition = self
                    .propositions
                    .get_mut(proposition_id)
                    .expect("precondition proposition id must be valid");
                proposition.precondition_of.push(operator_index);
            }
            for &proposition_id in &operator.effect_ids {
                let proposition = self
                    .propositions
                    .get_mut(proposition_id)
                    .expect("effect proposition id must be valid");
                proposition.effect_of.push(operator_index);
            }
        }
    }

    fn build_original_to_relaxed_index(&mut self) {
        let operator_count = self.task.get_operators().len() + self.task.axioms().len();
        self.original_to_relaxed_operators = vec![Vec::new(); operator_count];
        for (relaxed_operator_id, operator) in self.relaxed_operators.iter().enumerate() {
            if let Some(original_id) = operator.original_op_id_1 {
                if let Some(mapped) = self.original_to_relaxed_operators.get_mut(original_id) {
                    mapped.push(relaxed_operator_id);
                }
            }
            if let Some(original_id) = operator.original_op_id_2 {
                if let Some(mapped) = self.original_to_relaxed_operators.get_mut(original_id) {
                    mapped.push(relaxed_operator_id);
                }
            }
        }
    }

    fn setup_exploration_queue(&mut self) {
        self.priority_queue.clear();

        for proposition in &mut self.propositions {
            proposition.status = PropositionStatus::Unreached;
            proposition.explored = false;
            proposition.h_max_cost = f64::INFINITY;
        }

        for operator in &mut self.relaxed_operators {
            operator.unsatisfied_preconditions = operator.precondition_ids.len();
            operator.h_max_supporter = None;
            operator.h_max_supporter_cost = f64::INFINITY;
        }
    }

    fn setup_exploration_queue_state(
        &mut self,
        propositional_values: &[i32],
        numeric_values: &[f64],
    ) -> Result<(), String> {
        assert_eq!(
            propositional_values.len(),
            self.num_variables,
            "LM-cut exploration received the wrong number of propositional values"
        );
        self.numeric_initial_state.clear();
        self.numeric_initial_state
            .resize(self.conditions.len(), 0.0);

        for (variable_id, &value) in propositional_values.iter().enumerate() {
            if value < 0 {
                return Err(format!(
                    "LM-cut exploration received negative value {value} for variable {variable_id}"
                ));
            }
            if self.is_numeric_axiom_var(variable_id) && !self.config.ignore_numeric {
                continue;
            }
            let fact = Fact::new(variable_id as u32, value);
            let proposition_id = self.get_proposition_id(&fact);
            self.enqueue_if_necessary(proposition_id, 0.0)?;
        }

        if !self.config.ignore_numeric {
            for condition_id in 0..self.conditions.len() {
                let slack = self.evaluate_numeric_condition(condition_id, numeric_values)?;
                self.numeric_initial_state[condition_id] = -slack;
                if slack > -self.config.precision {
                    let proposition_id = self.numeric_condition_proposition_id(condition_id)?;
                    self.enqueue_if_necessary(proposition_id, 0.0)?;
                }
            }
        }

        self.enqueue_if_necessary(self.artificial_precondition_id, 0.0)?;
        Ok(())
    }

    fn first_exploration(
        &mut self,
        propositional_values: &[i32],
        numeric_values: &[f64],
    ) -> Result<(), String> {
        assert!(
            self.priority_queue.is_empty(),
            "LM-cut first exploration requires an empty queue"
        );
        self.setup_exploration_queue();
        self.setup_exploration_queue_state(propositional_values, numeric_values)?;

        while let Some((Reverse(popped_cost), proposition_id)) = self.priority_queue.pop() {
            let popped_cost = popped_cost.into_inner();
            let proposition_cost = self
                .propositions
                .get(proposition_id)
                .expect("priority queue proposition id must be valid")
                .h_max_cost;
            assert!(
                proposition_cost <= popped_cost + self.config.precision,
                "LM-cut queue popped a cost smaller than the proposition h_max"
            );
            if proposition_cost + self.config.precision < popped_cost {
                continue;
            }

            self.propositions[proposition_id].explored = true;
            let triggered_operators = self.propositions[proposition_id].precondition_of.clone();

            for operator_id in triggered_operators {
                let operator = self
                    .relaxed_operators
                    .get_mut(operator_id)
                    .expect("triggered operator id must be valid");
                assert!(
                    operator.unsatisfied_preconditions > 0,
                    "LM-cut operator precondition counter underflow"
                );
                operator.unsatisfied_preconditions -= 1;

                if operator.unsatisfied_preconditions == 0 {
                    operator.h_max_supporter = Some(proposition_id);
                    operator.h_max_supporter_cost = proposition_cost;
                    let effect_ids = operator.effect_ids.clone();
                    for effect_id in effect_ids {
                        self.update_queue(
                            propositional_values,
                            numeric_values,
                            operator_id,
                            proposition_id,
                            effect_id,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    fn first_exploration_incremental(
        &mut self,
        propositional_values: &[i32],
        numeric_values: &[f64],
        cut: &[usize],
    ) -> Result<(), String> {
        assert!(
            self.priority_queue.is_empty(),
            "LM-cut incremental exploration requires an empty queue"
        );
        for &relaxed_operator_id in cut {
            let original_ids = {
                let operator =
                    self.relaxed_operators
                        .get(relaxed_operator_id)
                        .ok_or_else(|| {
                            format!("LM-cut cut operator id {relaxed_operator_id} is invalid")
                        })?;
                [operator.original_op_id_1, operator.original_op_id_2]
            };

            for original_id in original_ids.into_iter().flatten() {
                let mapped = self
                    .original_to_relaxed_operators
                    .get(original_id)
                    .ok_or_else(|| format!("LM-cut original operator id {original_id} is invalid"))?
                    .clone();
                for mapped_operator_id in mapped {
                    let operator =
                        self.relaxed_operators
                            .get(mapped_operator_id)
                            .ok_or_else(|| {
                                format!(
                                "LM-cut mapped relaxed operator id {mapped_operator_id} is invalid"
                            )
                            })?;
                    if operator.unsatisfied_preconditions == 0 {
                        let supporter_id = operator.h_max_supporter.ok_or_else(|| {
                            format!(
                                "LM-cut reachable operator {} must have an h_max supporter",
                                operator.name
                            )
                        })?;
                        let effect_ids = self.relaxed_operators[mapped_operator_id]
                            .effect_ids
                            .clone();
                        for effect_id in effect_ids {
                            self.update_queue(
                                propositional_values,
                                numeric_values,
                                mapped_operator_id,
                                supporter_id,
                                effect_id,
                            )?;
                        }
                    }
                }
            }
        }

        while let Some((Reverse(popped_cost), proposition_id)) = self.priority_queue.pop() {
            let popped_cost = popped_cost.into_inner();
            let proposition_cost = self
                .propositions
                .get(proposition_id)
                .expect("priority queue proposition id must be valid")
                .h_max_cost;
            assert!(
                proposition_cost <= popped_cost + self.config.precision,
                "LM-cut incremental queue popped a cost smaller than the proposition h_max"
            );
            if proposition_cost + self.config.precision < popped_cost {
                continue;
            }

            let triggered_operators = self.propositions[proposition_id].precondition_of.clone();
            for operator_id in triggered_operators {
                let update = {
                    let operator = self
                        .relaxed_operators
                        .get(operator_id)
                        .ok_or_else(|| format!("LM-cut operator id {operator_id} is invalid"))?;
                    if operator.h_max_supporter == Some(proposition_id) {
                        let new_supporter = self.max_supporter_for_operator(operator_id)?;
                        if let Some((new_supporter_id, new_cost)) = new_supporter {
                            if new_cost + self.config.precision < operator.h_max_supporter_cost {
                                Some((new_supporter_id, new_cost, operator.effect_ids.clone()))
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                };

                if let Some((new_supporter_id, new_cost, effect_ids)) = update {
                    let operator = self
                        .relaxed_operators
                        .get_mut(operator_id)
                        .expect("operator id already validated");
                    operator.h_max_supporter = Some(new_supporter_id);
                    operator.h_max_supporter_cost = new_cost;
                    for effect_id in effect_ids {
                        self.update_queue(
                            propositional_values,
                            numeric_values,
                            operator_id,
                            new_supporter_id,
                            effect_id,
                        )?;
                    }
                }
            }
        }

        Ok(())
    }

    fn max_supporter_for_operator(
        &self,
        operator_id: usize,
    ) -> Result<Option<(usize, f64)>, String> {
        let operator = self
            .relaxed_operators
            .get(operator_id)
            .ok_or_else(|| format!("LM-cut operator id {operator_id} is invalid"))?;
        if operator.unsatisfied_preconditions != 0 {
            return Ok(None);
        }
        let mut best: Option<(usize, f64)> = None;
        for &precondition_id in &operator.precondition_ids {
            let proposition = self
                .propositions
                .get(precondition_id)
                .ok_or_else(|| format!("LM-cut proposition id {precondition_id} is invalid"))?;
            if proposition.status == PropositionStatus::Unreached {
                return Ok(None);
            }
            match best {
                None => best = Some((precondition_id, proposition.h_max_cost)),
                Some((_, best_cost)) if proposition.h_max_cost > best_cost => {
                    best = Some((precondition_id, proposition.h_max_cost))
                }
                _ => {}
            }
        }
        Ok(best)
    }

    fn mark_goal_plateau(
        &mut self,
        propositional_values: &[i32],
        numeric_values: &[f64],
        proposition_id: usize,
    ) -> Result<(), String> {
        if self
            .propositions
            .get(proposition_id)
            .ok_or_else(|| format!("LM-cut proposition id {proposition_id} is invalid"))?
            .status
            == PropositionStatus::GoalZone
        {
            return Ok(());
        }

        self.propositions[proposition_id].status = PropositionStatus::GoalZone;
        let achievers = self.propositions[proposition_id].effect_of.clone();
        for achiever_id in achievers {
            let recurse_to = {
                let achiever = self
                    .relaxed_operators
                    .get(achiever_id)
                    .ok_or_else(|| format!("LM-cut achiever id {achiever_id} is invalid"))?;
                if achiever.cost_1 < self.config.precision
                    && achiever.cost_2 < self.config.precision
                    && achiever.unsatisfied_preconditions == 0
                {
                    let ms = self.calculate_numeric_times(
                        propositional_values,
                        numeric_values,
                        proposition_id,
                        achiever_id,
                    )?;
                    if self.multiplier_allows_traversal(achiever_id, ms) {
                        achiever.h_max_supporter
                    } else {
                        None
                    }
                } else {
                    None
                }
            };
            if let Some(supporter_id) = recurse_to {
                self.mark_goal_plateau(propositional_values, numeric_values, supporter_id)?;
            }
        }
        Ok(())
    }

    fn second_exploration(
        &mut self,
        propositional_values: &[i32],
        numeric_values: &[f64],
        cut: &mut Vec<usize>,
        m_list: &mut Vec<(f64, f64)>,
    ) -> Result<(), String> {
        assert!(
            cut.is_empty(),
            "LM-cut second exploration requires empty cut"
        );
        assert!(
            m_list.is_empty(),
            "LM-cut second exploration requires empty multiplier list"
        );

        let mut queue = Vec::new();
        self.propositions[self.artificial_precondition_id].status =
            PropositionStatus::BeforeGoalZone;
        queue.push(self.artificial_precondition_id);

        for (variable_id, &value) in propositional_values.iter().enumerate() {
            if self.is_numeric_axiom_var(variable_id) && !self.config.ignore_numeric {
                continue;
            }
            let fact = Fact::new(variable_id as u32, value);
            let proposition_id = self.get_proposition_id(&fact);
            if self.propositions[proposition_id].status != PropositionStatus::BeforeGoalZone {
                self.propositions[proposition_id].status = PropositionStatus::BeforeGoalZone;
                queue.push(proposition_id);
            }
        }

        if !self.config.ignore_numeric {
            for condition_id in 0..self.conditions.len() {
                if self.numeric_initial_state[condition_id] < self.config.precision {
                    let proposition_id = self.numeric_condition_proposition_id(condition_id)?;
                    if self.propositions[proposition_id].status != PropositionStatus::BeforeGoalZone
                    {
                        self.propositions[proposition_id].status =
                            PropositionStatus::BeforeGoalZone;
                        queue.push(proposition_id);
                    }
                }
            }
        }

        while let Some(proposition_id) = queue.pop() {
            let triggered_operators = self.propositions[proposition_id].precondition_of.clone();
            for operator_id in triggered_operators {
                let should_process = {
                    let operator = self
                        .relaxed_operators
                        .get(operator_id)
                        .ok_or_else(|| format!("LM-cut operator id {operator_id} is invalid"))?;
                    operator.h_max_supporter == Some(proposition_id) && !cut.contains(&operator_id)
                };
                if !should_process {
                    continue;
                }

                let effect_ids = self.relaxed_operators[operator_id].effect_ids.clone();
                let mut min_cut_cost = f64::INFINITY;

                for &effect_id in &effect_ids {
                    let effect_status = self.propositions[effect_id].status;
                    if effect_status == PropositionStatus::GoalZone {
                        let ms = self.calculate_numeric_times(
                            propositional_values,
                            numeric_values,
                            effect_id,
                            operator_id,
                        )?;
                        if self.multiplier_allows_traversal(operator_id, ms) {
                            let edge_cost = self.edge_cost(operator_id, ms)?;
                            cut.push(operator_id);
                            m_list.push(ms);
                            min_cut_cost = min_cut_cost.min(edge_cost);
                        }
                    }
                }

                for &effect_id in &effect_ids {
                    let effect_status = self.propositions[effect_id].status;
                    if effect_status == PropositionStatus::BeforeGoalZone
                        || effect_status == PropositionStatus::GoalZone
                    {
                        continue;
                    }
                    let ms = self.calculate_numeric_times(
                        propositional_values,
                        numeric_values,
                        effect_id,
                        operator_id,
                    )?;
                    if self.multiplier_allows_traversal(operator_id, ms) {
                        let edge_cost = self.edge_cost(operator_id, ms)?;
                        if edge_cost + self.config.precision < min_cut_cost {
                            assert_eq!(effect_status, PropositionStatus::Reached);
                            self.propositions[effect_id].status = PropositionStatus::BeforeGoalZone;
                            queue.push(effect_id);
                        }
                    }
                }
            }
        }

        Ok(())
    }

    fn calculate_numeric_times(
        &self,
        propositional_values: &[i32],
        numeric_values: &[f64],
        effect_id: usize,
        operator_id: usize,
    ) -> Result<(f64, f64), String> {
        let effect = self
            .propositions
            .get(effect_id)
            .ok_or_else(|| format!("LM-cut effect proposition id {effect_id} is invalid"))?;
        if effect.is_numeric_condition {
            let operator = self
                .relaxed_operators
                .get(operator_id)
                .ok_or_else(|| format!("LM-cut operator id {operator_id} is invalid"))?;
            if operator.infinite {
                return Ok((0.0, 1.0));
            }
            let condition_id = effect.id_numeric_condition.ok_or_else(|| {
                format!("LM-cut numeric proposition {effect_id} is missing its condition id")
            })?;
            if operator.original_op_id_1.is_some() {
                let c_u = *operator
                    .sose_constants
                    .get(condition_id)
                    .ok_or_else(|| {
                        format!(
                            "LM-cut SOSE operator {operator_id} is missing condition constant {condition_id}"
                        )
                    })?;
                if c_u < self.config.precision {
                    return Ok((-1.0, -1.0));
                }
                if operator.cost_1 < self.config.precision {
                    return Ok((1.0, 1.0));
                }

                let effect_expression =
                    self.operator_condition_delta_expression(operator_id, condition_id)?;
                let c = effect_expression.constant;
                let s_u = effect_expression.evaluate(numeric_values) - c;

                if operator.cost_2 < self.config.precision {
                    if (c + s_u).abs() < self.config.precision {
                        return Ok((1.0, 1.0));
                    }
                    if c + s_u > 0.0 {
                        return Ok((-1.0, -1.0));
                    }
                    let mut m_1 = -(c + s_u) / c_u;
                    if self.config.ceiling_less_than_one {
                        m_1 = m_1.max(1.0);
                    }
                    return Ok((m_1, 1.0));
                }

                let u_target = (self.numeric_initial_state[condition_id] * c_u * operator.cost_2
                    / operator.cost_1)
                    .sqrt()
                    - c;
                if u_target - s_u < self.config.precision || c + u_target < self.config.precision {
                    return Ok((-1.0, -1.0));
                }

                let mut m_1 = (u_target - s_u) / c_u;
                let mut m_2 = self.numeric_initial_state[condition_id] / (c + u_target);
                if self.config.ceiling_less_than_one {
                    m_1 = m_1.max(1.0);
                    m_2 = m_2.max(1.0);
                }
                return Ok((m_1, m_2));
            }
            let mut net = operator
                .original_op_id_2
                .filter(|&original_id| original_id < self.operator_to_simple_effects.len())
                .and_then(|original_id| {
                    self.operator_to_simple_effects[original_id]
                        .get(condition_id)
                        .copied()
                        .flatten()
                })
                .unwrap_or(0.0);
            if net < self.config.precision {
                return Ok((-1.0, -1.0));
            }
            let mut m = self.numeric_initial_state[condition_id] / net;
            if m < self.config.precision {
                return Ok((0.0, 0.0));
            }
            if self.config.ceiling_less_than_one {
                m = m.max(1.0);
            }
            Ok((0.0, m))
        } else {
            Ok((0.0, 1.0))
        }
    }

    fn multiplier_allows_traversal(&self, operator_id: usize, ms: (f64, f64)) -> bool {
        let operator = &self.relaxed_operators[operator_id];
        match operator.original_op_id_1 {
            Some(_) => ms.0 >= self.config.precision,
            None => ms.1 >= self.config.precision,
        }
    }

    fn edge_cost(&self, operator_id: usize, ms: (f64, f64)) -> Result<f64, String> {
        let operator = self
            .relaxed_operators
            .get(operator_id)
            .ok_or_else(|| format!("LM-cut operator id {operator_id} is invalid"))?;
        let mut edge_cost = ms.1 * operator.cost_2;
        if operator.original_op_id_1.is_some() {
            edge_cost += ms.0 * operator.cost_1;
        }
        Ok(edge_cost)
    }

    fn reset_goal_zone_statuses(&mut self) {
        for proposition in &mut self.propositions {
            if proposition.status == PropositionStatus::GoalZone
                || proposition.status == PropositionStatus::BeforeGoalZone
            {
                proposition.status = PropositionStatus::Reached;
            }
        }
    }

    fn update_queue(
        &mut self,
        propositional_values: &[i32],
        numeric_values: &[f64],
        operator_id: usize,
        supporter_id: usize,
        effect_id: usize,
    ) -> Result<(), String> {
        let effect = self
            .propositions
            .get(effect_id)
            .ok_or_else(|| format!("LM-cut effect proposition id {effect_id} is invalid"))?;
        if effect.is_numeric_condition {
            let ms = self.calculate_numeric_times(
                propositional_values,
                numeric_values,
                effect_id,
                operator_id,
            )?;
            if self.multiplier_allows_traversal(operator_id, ms) {
                let target_cost =
                    self.propositions[supporter_id].h_max_cost + self.edge_cost(operator_id, ms)?;
                self.enqueue_if_necessary(effect_id, target_cost)?;
            }
            return Ok(());
        }
        let target_cost =
            self.propositions[supporter_id].h_max_cost + self.relaxed_operators[operator_id].cost_2;
        self.enqueue_if_necessary(effect_id, target_cost)?;
        Ok(())
    }

    fn enqueue_if_necessary(&mut self, proposition_id: usize, cost: f64) -> Result<bool, String> {
        assert!(cost >= 0.0, "LM-cut enqueue cost must be non-negative");
        let proposition = self
            .propositions
            .get_mut(proposition_id)
            .ok_or_else(|| format!("LM-cut proposition id {proposition_id} is invalid"))?;
        if proposition.status == PropositionStatus::Unreached
            || proposition.h_max_cost > cost + self.config.precision
        {
            proposition.status = PropositionStatus::Reached;
            proposition.h_max_cost = cost;
            self.priority_queue.push((
                Reverse(
                    NotNan::new(cost)
                        .map_err(|_| "LM-cut enqueue cost must not be NaN".to_string())?,
                ),
                proposition_id,
            ));
            return Ok(true);
        }
        Ok(false)
    }

    fn calculate_base_operator_cost(&self, operator_id: usize, operator: &Operator) -> f64 {
        assert!(
            operator_id < self.task.get_operators().len(),
            "base operator cost is only defined for concrete operators"
        );
        metric_operator_cost_from_initial_values(self.task, operator).max(0.0)
    }

    fn build_relaxed_operator_for_operator(
        &mut self,
        operator_id: usize,
        operator: &Operator,
        base_cost: f64,
    ) -> Result<(), String> {
        let linearized_assignment_effects = self
            .task
            .linearized_assignment_effects(operator_id)
            .map_err(|error| {
                format!(
                    "LM-cut failed to linearize numeric effects for operator {operator_id}: {error}"
                )
            })?;
        let precondition_ids = self.build_precondition_ids(operator.preconditions());
        let unconditional_assignment_effect_ids = operator
            .assignment_effects()
            .iter()
            .enumerate()
            .filter_map(|(assignment_effect_id, assignment_effect)| {
                if assignment_effect.is_conditional() || !assignment_effect.conditions().is_empty()
                {
                    None
                } else {
                    Some(assignment_effect_id)
                }
            })
            .collect::<Vec<_>>();

        for effect in operator.effects() {
            if effect.conditions().is_empty() {
                continue;
            }
            let mut extended_preconditions = precondition_ids.clone();
            let mut seen: BTreeSet<usize> = extended_preconditions.iter().copied().collect();
            for condition in effect.conditions() {
                for proposition_id in self.precondition_proposition_ids(condition) {
                    if seen.insert(proposition_id) {
                        extended_preconditions.push(proposition_id);
                    }
                }
            }
            let conditional_name = format!(
                "{} {}",
                operator.name(),
                self.proposition_name_for_effect(effect)
            );
            let mut conditional_operator = RelaxedOperator::new(
                extended_preconditions,
                vec![self.get_proposition_id_for_effect(effect)],
                operator_id,
                base_cost,
                conditional_name,
                true,
            );
            conditional_operator.assert_well_formed();
            self.relaxed_operators.push(conditional_operator);
        }

        let base_effect_ids = operator
            .effects()
            .iter()
            .filter(|effect| effect.conditions().is_empty())
            .map(|effect| self.get_proposition_id_for_effect(effect))
            .collect::<Vec<_>>();

        let mut effect_ids = base_effect_ids;
        self.extend_numeric_effect_ids(
            &linearized_assignment_effects,
            &unconditional_assignment_effect_ids,
            &mut effect_ids,
        )?;

        let mut relaxed_operator = RelaxedOperator::new(
            if precondition_ids.is_empty() {
                vec![self.artificial_precondition_id]
            } else {
                precondition_ids.clone()
            },
            effect_ids,
            operator_id,
            base_cost,
            operator.name().to_string(),
            false,
        );
        relaxed_operator.assignment_effect_ids = unconditional_assignment_effect_ids.clone();
        relaxed_operator.linear_assignment_effects = unconditional_assignment_effect_ids
            .iter()
            .map(|&assignment_effect_id| {
                linearized_assignment_effects
                    .get(assignment_effect_id)
                    .cloned()
                    .expect("linearized assignment effect id must be valid")
            })
            .collect();
        relaxed_operator.assert_well_formed();
        self.relaxed_operators.push(relaxed_operator);

        for (assignment_effect_id, assignment_effect) in
            operator.assignment_effects().iter().enumerate()
        {
            if !assignment_effect.is_conditional() && assignment_effect.conditions().is_empty() {
                continue;
            }

            let mut extended_preconditions = precondition_ids.clone();
            let mut seen: BTreeSet<usize> = extended_preconditions.iter().copied().collect();
            for condition in assignment_effect.conditions() {
                for proposition_id in self.precondition_proposition_ids(condition) {
                    if seen.insert(proposition_id) {
                        extended_preconditions.push(proposition_id);
                    }
                }
            }

            let conditional_assignment_effect_ids = unconditional_assignment_effect_ids
                .iter()
                .copied()
                .chain(std::iter::once(assignment_effect_id))
                .collect::<Vec<_>>();
            let mut conditional_effect_ids = Vec::new();
            self.extend_numeric_effect_ids(
                &linearized_assignment_effects,
                &conditional_assignment_effect_ids,
                &mut conditional_effect_ids,
            )?;

            let mut conditional_operator = RelaxedOperator::new(
                if extended_preconditions.is_empty() {
                    vec![self.artificial_precondition_id]
                } else {
                    extended_preconditions
                },
                conditional_effect_ids,
                operator_id,
                base_cost,
                format!(
                    "{} numeric {}",
                    operator.name(),
                    assignment_effect.affected_var_id()
                ),
                true,
            );
            conditional_operator.assignment_effect_ids = conditional_assignment_effect_ids;
            conditional_operator.linear_assignment_effects = conditional_operator
                .assignment_effect_ids
                .iter()
                .map(|&effect_id| {
                    let mut linear_effect = linearized_assignment_effects
                        .get(effect_id)
                        .cloned()
                        .expect("linearized assignment effect id must be valid");
                    if effect_id == assignment_effect_id {
                        linear_effect.is_conditional = false;
                        linear_effect.conditions.clear();
                    }
                    linear_effect
                })
                .collect();
            conditional_operator.assert_well_formed();
            self.relaxed_operators.push(conditional_operator);
        }

        self.build_infinite_operators_for_operator(
            operator_id,
            operator,
            base_cost,
            &precondition_ids,
            &linearized_assignment_effects,
        )?;
        Ok(())
    }

    fn build_simple_effects(&mut self) -> Result<(), String> {
        let operator_count = self.task.get_operators().len();
        self.operator_to_simple_effects = vec![vec![None; self.conditions.len()]; operator_count];

        for relaxed_operator_id in 0..self.relaxed_operators.len() {
            let original_op_id = {
                let relaxed_operator = &self.relaxed_operators[relaxed_operator_id];
                if relaxed_operator.conditional
                    || relaxed_operator.infinite
                    || relaxed_operator.original_op_id_1.is_some()
                {
                    continue;
                }
                match relaxed_operator.original_op_id_2 {
                    Some(original_id) if original_id < operator_count => original_id,
                    _ => continue,
                }
            };

            let mut additional_effect_ids = Vec::new();
            let mut seen: BTreeSet<usize> = self.relaxed_operators[relaxed_operator_id]
                .effect_ids
                .iter()
                .copied()
                .collect();

            for condition_id in 0..self.conditions.len() {
                let expression =
                    self.operator_condition_delta_expression(relaxed_operator_id, condition_id)?;
                if !expression.is_constant() {
                    continue;
                }
                let simple_effect = expression.constant;
                if simple_effect < self.config.precision {
                    continue;
                }

                self.operator_to_simple_effects[original_op_id][condition_id] = Some(simple_effect);
                let proposition_id = self.numeric_condition_proposition_id(condition_id)?;
                if seen.insert(proposition_id) {
                    additional_effect_ids.push(proposition_id);
                }
            }

            if !additional_effect_ids.is_empty() {
                self.relaxed_operators[relaxed_operator_id]
                    .effect_ids
                    .extend(additional_effect_ids);
            }
        }

        Ok(())
    }

    fn build_infinite_operators_for_operator(
        &mut self,
        operator_id: usize,
        operator: &Operator,
        base_cost: f64,
        base_precondition_ids: &[usize],
        linearized_assignment_effects: &[LinearNumericEffect],
    ) -> Result<(), String> {
        for (assignment_effect_id, assignment_effect) in
            operator.assignment_effects().iter().enumerate()
        {
            let linear_effect = linearized_assignment_effects
                .get(assignment_effect_id)
                .cloned()
                .ok_or_else(|| {
                    format!(
                        "LM-cut linearized assignment effect {assignment_effect_id} for operator {operator_id} is missing"
                    )
                })?;

            let mut precondition_ids = base_precondition_ids.to_vec();
            let mut seen: BTreeSet<usize> = precondition_ids.iter().copied().collect();
            for condition in assignment_effect.conditions() {
                for proposition_id in self.precondition_proposition_ids(condition) {
                    if seen.insert(proposition_id) {
                        precondition_ids.push(proposition_id);
                    }
                }
            }
            if precondition_ids.is_empty() {
                precondition_ids.push(self.artificial_precondition_id);
            }

            let mut plus_effect_ids = Vec::new();
            let mut minus_effect_ids = Vec::new();
            for condition_id in 0..self.conditions.len() {
                let condition = self
                    .conditions
                    .get(condition_id)
                    .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;
                let weight = condition
                    .coefficients
                    .get(linear_effect.affected_var_id)
                    .copied()
                    .unwrap_or(0.0);
                let weighted_delta = linear_effect.delta.scale(weight);
                if weighted_delta.is_constant() {
                    continue;
                }
                if weight > self.config.precision {
                    plus_effect_ids.push(self.numeric_condition_proposition_id(condition_id)?);
                } else if weight < -self.config.precision {
                    minus_effect_ids.push(self.numeric_condition_proposition_id(condition_id)?);
                }
            }

            if !plus_effect_ids.is_empty() {
                let guard_expression = linear_effect.delta.clone();
                let guard_proposition_id = self.add_numeric_condition_proposition(
                    NumericCondition::from_expression(
                        guard_expression,
                        true,
                        format!(
                            "numeric ({} {} +inf guard)",
                            operator.name(),
                            linear_effect.affected_var_id
                        ),
                    ),
                );
                let mut relaxed_operator = RelaxedOperator::new(
                    {
                        let mut guarded_preconditions = precondition_ids.clone();
                        guarded_preconditions.push(guard_proposition_id);
                        guarded_preconditions
                    },
                    plus_effect_ids,
                    operator_id,
                    base_cost,
                    format!("{} {} +inf", operator.name(), linear_effect.affected_var_id),
                    true,
                );
                relaxed_operator.infinite = true;
                relaxed_operator.assignment_effect_ids = vec![assignment_effect_id];
                relaxed_operator.linear_assignment_effects = vec![linear_effect.clone()];
                relaxed_operator.assert_well_formed();
                self.relaxed_operators.push(relaxed_operator);
            }

            if !minus_effect_ids.is_empty() {
                let guard_expression = linear_effect.delta.scale(-1.0);
                let guard_proposition_id = self.add_numeric_condition_proposition(
                    NumericCondition::from_expression(
                        guard_expression,
                        true,
                        format!(
                            "numeric ({} {} -inf guard)",
                            operator.name(),
                            linear_effect.affected_var_id
                        ),
                    ),
                );
                let mut relaxed_operator = RelaxedOperator::new(
                    {
                        let mut guarded_preconditions = precondition_ids;
                        guarded_preconditions.push(guard_proposition_id);
                        guarded_preconditions
                    },
                    minus_effect_ids,
                    operator_id,
                    base_cost,
                    format!("{} {} -inf", operator.name(), linear_effect.affected_var_id),
                    true,
                );
                relaxed_operator.infinite = true;
                relaxed_operator.assignment_effect_ids = vec![assignment_effect_id];
                relaxed_operator.linear_assignment_effects = vec![linear_effect];
                relaxed_operator.assert_well_formed();
                self.relaxed_operators.push(relaxed_operator);
            }
        }

        Ok(())
    }

    fn extend_numeric_effect_ids(
        &self,
        linearized_assignment_effects: &[LinearNumericEffect],
        assignment_effect_ids: &[usize],
        effect_ids: &mut Vec<usize>,
    ) -> Result<(), String> {
        let mut seen: BTreeSet<usize> = effect_ids.iter().copied().collect();
        for condition_id in 0..self.conditions.len() {
            if self.assignment_effects_may_support_condition(
                linearized_assignment_effects,
                assignment_effect_ids,
                condition_id,
            )? {
                let proposition_id = self.numeric_condition_proposition_id(condition_id)?;
                if seen.insert(proposition_id) {
                    effect_ids.push(proposition_id);
                }
            }
        }
        Ok(())
    }

    fn assignment_effects_may_support_condition(
        &self,
        linearized_assignment_effects: &[LinearNumericEffect],
        assignment_effect_ids: &[usize],
        condition_id: usize,
    ) -> Result<bool, String> {
        let condition = self
            .conditions
            .get(condition_id)
            .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;
        let mut expression = LinearExpression::zero(self.task.numeric_variables().len());
        for &assignment_effect_id in assignment_effect_ids {
            let linear_effect = linearized_assignment_effects
                .get(assignment_effect_id)
                .ok_or_else(|| {
                    format!("LM-cut assignment effect id {assignment_effect_id} is invalid")
                })?;
            let target_coefficient = condition
                .coefficients
                .get(linear_effect.affected_var_id)
                .copied()
                .unwrap_or(0.0);
            if target_coefficient.abs() < self.config.precision {
                continue;
            }
            expression = expression.add(&linear_effect.delta.scale(target_coefficient));
        }
        Ok(!expression.is_constant() || expression.constant.abs() >= self.config.precision)
    }

    fn operator_weighted_delta_expression(
        &self,
        relaxed_operator_id: usize,
        coefficients: &[f64],
    ) -> Result<LinearExpression, String> {
        let relaxed_operator =
            self.relaxed_operators
                .get(relaxed_operator_id)
                .ok_or_else(|| {
                    format!("LM-cut relaxed operator id {relaxed_operator_id} is invalid")
                })?;
        let mut expression = LinearExpression::zero(self.task.numeric_variables().len());
        for linear_effect in &relaxed_operator.linear_assignment_effects {
            let weight = coefficients
                .get(linear_effect.affected_var_id)
                .copied()
                .unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            expression = expression.add(&linear_effect.delta.scale(weight));
        }
        Ok(expression)
    }

    fn build_supported_sose_operators(&mut self) -> Result<(), String> {
        if !self.config.use_second_order_simple {
            return Ok(());
        }

        let operator_count = self.task.get_operators().len();
        let mut relaxed_by_original = vec![Vec::new(); operator_count];
        for (relaxed_operator_id, operator) in self.relaxed_operators.iter().enumerate() {
            if operator.original_op_id_1.is_some() {
                continue;
            }
            if let Some(original_id) = operator.original_op_id_2.filter(|&id| id < operator_count) {
                relaxed_by_original[original_id].push(relaxed_operator_id);
            }
        }

        let mut new_operators = Vec::new();
        for op2_id in 0..operator_count {
            for &op2_relaxed_id in &relaxed_by_original[op2_id] {
                let op2 = &self.relaxed_operators[op2_relaxed_id];
                if op2
                    .linear_assignment_effects
                    .iter()
                    .any(|effect| effect.is_conditional || !effect.conditions.is_empty())
                {
                    continue;
                }

                let mut supporter_to_effects: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
                for condition_id in 0..self.conditions.len() {
                    let base_expression =
                        self.operator_condition_delta_expression(op2_relaxed_id, condition_id)?;
                    if base_expression.is_constant() {
                        continue;
                    }

                    let self_loop = self.operator_weighted_delta_expression(
                        op2_relaxed_id,
                        &base_expression.coefficients,
                    )?;
                    if !self_loop.is_constant() || self_loop.constant.abs() >= self.config.precision
                    {
                        continue;
                    }

                    let mut condition_supporters = Vec::new();
                    for op1_id in 0..operator_count {
                        for &op1_relaxed_id in &relaxed_by_original[op1_id] {
                            let op1 = &self.relaxed_operators[op1_relaxed_id];
                            if op1.linear_assignment_effects.iter().any(|effect| {
                                effect.is_conditional || !effect.conditions.is_empty()
                            }) {
                                continue;
                            }

                            let support_expression = self.operator_weighted_delta_expression(
                                op1_relaxed_id,
                                &base_expression.coefficients,
                            )?;
                            if !support_expression.is_constant() {
                                continue;
                            }
                            if support_expression.constant < self.config.precision {
                                continue;
                            }

                            let parallel_expression = self.operator_condition_delta_expression(
                                op1_relaxed_id,
                                condition_id,
                            )?;
                            if !parallel_expression.is_constant()
                                || parallel_expression.constant.abs() >= self.config.precision
                            {
                                continue;
                            }

                            condition_supporters
                                .push((op1_relaxed_id, support_expression.constant));
                        }
                    }

                    if condition_supporters.is_empty() {
                        continue;
                    }

                    for (op1_relaxed_id, sose_constant) in condition_supporters {
                        supporter_to_effects
                            .entry(op1_relaxed_id)
                            .or_default()
                            .push((condition_id, sose_constant));
                    }
                }

                for (op1_relaxed_id, effects) in supporter_to_effects {
                    let op1 = self.relaxed_operators[op1_relaxed_id].clone();
                    let op2 = self.relaxed_operators[op2_relaxed_id].clone();
                    let original_op_id_1 = op1
                        .original_op_id_2
                        .filter(|&id| id < operator_count)
                        .ok_or_else(|| {
                            format!(
                                "LM-cut SOSE supporter relaxed operator {op1_relaxed_id} is missing its concrete operator id"
                            )
                        })?;
                    let original_op_id_2 = op2
                        .original_op_id_2
                        .filter(|&id| id < operator_count)
                        .ok_or_else(|| {
                            format!(
                                "LM-cut SOSE target relaxed operator {op2_relaxed_id} is missing its concrete operator id"
                            )
                        })?;

                    let mut precondition_ids = op1.precondition_ids.clone();
                    let mut seen: BTreeSet<usize> = precondition_ids.iter().copied().collect();
                    for proposition_id in op2.precondition_ids {
                        if seen.insert(proposition_id) {
                            precondition_ids.push(proposition_id);
                        }
                    }
                    if precondition_ids.is_empty() {
                        precondition_ids.push(self.artificial_precondition_id);
                    }

                    let mut effect_ids = Vec::new();
                    let mut sose_constants = vec![0.0; self.conditions.len()];
                    for (condition_id, sose_constant) in effects {
                        effect_ids.push(self.numeric_condition_proposition_id(condition_id)?);
                        sose_constants[condition_id] = sose_constant;
                    }

                    let mut sose_operator = RelaxedOperator::new(
                        precondition_ids,
                        effect_ids,
                        original_op_id_2,
                        op2.base_cost_2,
                        format!("{} {}", op1.name, op2.name),
                        false,
                    );
                    sose_operator.original_op_id_1 = Some(original_op_id_1);
                    sose_operator.original_op_id_2 = Some(original_op_id_2);
                    sose_operator.base_cost_1 = op1.base_cost_2;
                    sose_operator.base_cost_2 = op2.base_cost_2;
                    sose_operator.cost_1 = sose_operator.base_cost_1;
                    sose_operator.cost_2 = sose_operator.base_cost_2;
                    sose_operator.sose_constants = sose_constants;
                    sose_operator.linear_assignment_effects = op2.linear_assignment_effects.clone();
                    sose_operator.assert_well_formed();
                    new_operators.push(sose_operator);
                }
            }
        }

        self.relaxed_operators.extend(new_operators);
        Ok(())
    }

    fn operator_condition_delta_expression(
        &self,
        relaxed_operator_id: usize,
        condition_id: usize,
    ) -> Result<LinearExpression, String> {
        let relaxed_operator =
            self.relaxed_operators
                .get(relaxed_operator_id)
                .ok_or_else(|| {
                    format!("LM-cut relaxed operator id {relaxed_operator_id} is invalid")
                })?;
        let condition = self
            .conditions
            .get(condition_id)
            .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;

        let mut expression = LinearExpression::zero(self.task.numeric_variables().len());
        for linear_effect in &relaxed_operator.linear_assignment_effects {
            let target_coefficient = condition
                .coefficients
                .get(linear_effect.affected_var_id)
                .copied()
                .unwrap_or(0.0);
            if target_coefficient.abs() < self.config.precision {
                continue;
            }
            expression = expression.add(&linear_effect.delta.scale(target_coefficient));
        }
        Ok(expression)
    }

    fn numeric_net_effect_for_operator(
        &self,
        propositional_values: &[i32],
        numeric_values: &[f64],
        relaxed_operator_id: usize,
        condition_id: usize,
    ) -> Result<f64, String> {
        let relaxed_operator =
            self.relaxed_operators
                .get(relaxed_operator_id)
                .ok_or_else(|| {
                    format!("LM-cut relaxed operator id {relaxed_operator_id} is invalid")
                })?;
        let condition = self
            .conditions
            .get(condition_id)
            .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;

        let expression =
            self.operator_condition_delta_expression(relaxed_operator_id, condition_id)?;
        let mut net = 0.0;
        for linear_effect in &relaxed_operator.linear_assignment_effects {
            if !self.numeric_effect_conditions_hold(propositional_values, &linear_effect.conditions)
            {
                continue;
            }
            let affected = linear_effect.affected_var_id;
            let target_coefficient = condition.coefficients.get(affected).copied().unwrap_or(0.0);
            if target_coefficient.abs() < self.config.precision {
                continue;
            }

            let delta_value = linear_effect.delta.evaluate(numeric_values);
            net += target_coefficient * delta_value;
        }
        let unconditional_net = expression.evaluate(numeric_values);
        assert!(
            (net - unconditional_net).abs() < self.config.precision
                || relaxed_operator
                    .linear_assignment_effects
                    .iter()
                    .any(|effect| !effect.conditions.is_empty()),
            "LM-cut conditional-free numeric effect mismatch between decomposed and aggregated evaluation"
        );
        Ok(net)
    }

    fn numeric_effect_conditions_hold(
        &self,
        propositional_values: &[i32],
        conditions: &[Fact],
    ) -> bool {
        conditions.iter().all(|condition| {
            propositional_values.get(condition.var() as usize).copied() == Some(condition.value())
        })
    }

    fn build_relaxed_operator_for_axiom(&mut self, operator_id: usize, axiom: &PropositionalAxiom) {
        let precondition_ids = self.build_precondition_ids(axiom.conditions());
        let effect_fact = Fact::new(axiom.var_id(), axiom.effect_value() as i32);
        let mut relaxed_operator = RelaxedOperator::new(
            if precondition_ids.is_empty() {
                vec![self.artificial_precondition_id]
            } else {
                precondition_ids
            },
            vec![self.get_proposition_id(&effect_fact)],
            operator_id,
            0.0,
            format!("axiom {}", self.task.get_fact_name(&effect_fact)),
            false,
        );
        relaxed_operator.assert_well_formed();
        self.relaxed_operators.push(relaxed_operator);
    }

    fn build_precondition_ids(&self, preconditions: &[Fact]) -> Vec<usize> {
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        for precondition in preconditions {
            for proposition_id in self.precondition_proposition_ids(precondition) {
                if seen.insert(proposition_id) {
                    result.push(proposition_id);
                }
            }
        }
        result
    }

    fn build_numeric_condition_propositions(&mut self) {
        if self.config.ignore_numeric {
            return;
        }

        for (comparison_axiom_id, comparison_axiom) in
            self.task.comparison_axioms().iter().enumerate()
        {
            let affected_var_id = usize::try_from(comparison_axiom.get_affected_var_id())
                .expect("comparison axiom affected variable must be non-negative");
            self.comparison_axiom_by_var
                .insert(affected_var_id, comparison_axiom_id);
            for fact_value in [0_i32, 1_i32] {
                let Some(conditions) = self.build_numeric_conditions_for_fact_value(
                    comparison_axiom,
                    affected_var_id,
                    fact_value,
                ) else {
                    continue;
                };
                let mut condition_ids = Vec::new();
                for condition in conditions {
                    let proposition_id = self.add_numeric_condition_proposition(condition);
                    let condition_id = self.propositions[proposition_id]
                        .id_numeric_condition
                        .expect("new numeric condition proposition must reference its condition");
                    condition_ids.push(condition_id);
                }
                self.comparison_fact_to_condition_ids
                    .insert((affected_var_id, fact_value), condition_ids);
            }
        }
    }

    fn add_numeric_condition_proposition(&mut self, condition: NumericCondition) -> usize {
        let condition_id = self.conditions.len();
        let proposition_id = self.propositions.len();
        let mut proposition = RelaxedProposition::new(proposition_id, condition.name.clone());
        proposition.is_numeric_condition = true;
        proposition.id_numeric_condition = Some(condition_id);
        self.propositions.push(proposition);
        self.conditions.push(condition);
        self.epsilons.push(self.config.epsilon);
        self.num_propositions += 1;
        proposition_id
    }

    fn build_numeric_conditions_for_fact_value(
        &self,
        comparison_axiom: &ComparisonAxiom,
        affected_var_id: usize,
        fact_value: i32,
    ) -> Option<Vec<NumericCondition>> {
        let operator = comparison_operator_for_fact_value(comparison_axiom, fact_value)?;
        let lhs = usize::try_from(comparison_axiom.get_left_var_id())
            .expect("comparison lhs numeric var must be non-negative");
        let rhs = usize::try_from(comparison_axiom.get_right_var_id())
            .expect("comparison rhs numeric var must be non-negative");
        let fact = Fact::new(affected_var_id as u32, fact_value);
        let fact_name = self.task.get_fact_name(&fact).to_string();

        match operator {
            ComparisonOperator::GreaterThan | ComparisonOperator::GreaterThanOrEqual => {
                Some(vec![self.build_numeric_condition(
                    lhs,
                    rhs,
                    matches!(operator, ComparisonOperator::GreaterThan),
                    fact_name,
                )])
            }
            ComparisonOperator::LessThan | ComparisonOperator::LessThanOrEqual => Some(vec![self
                .build_numeric_condition(
                    rhs,
                    lhs,
                    matches!(operator, ComparisonOperator::LessThan),
                    fact_name,
                )]),
            ComparisonOperator::Equal => Some(vec![
                self.build_numeric_condition(lhs, rhs, false, fact_name.clone()),
                self.build_numeric_condition(rhs, lhs, false, fact_name),
            ]),
            ComparisonOperator::UnEqual => unreachable!(
                "unsupported disequality facts must be filtered before building conditions"
            ),
        }
    }

    fn build_numeric_condition(
        &self,
        positive_var_id: usize,
        negative_var_id: usize,
        is_strictly_greater: bool,
        name: String,
    ) -> NumericCondition {
        let positive_expression = self
            .task
            .linearize_numeric_var(positive_var_id)
            .unwrap_or_else(|error| {
                panic!(
                "LM-cut failed to linearize comparison lhs numeric var {positive_var_id}: {error}"
            )
            });
        let negative_expression = self
            .task
            .linearize_numeric_var(negative_var_id)
            .unwrap_or_else(|error| {
                panic!(
                "LM-cut failed to linearize comparison rhs numeric var {negative_var_id}: {error}"
            )
            });
        let expression = positive_expression.subtract(&negative_expression);
        NumericCondition::from_expression(
            expression,
            is_strictly_greater,
            format!("numeric ({name})"),
        )
    }

    fn numeric_condition_proposition_id(&self, condition_id: usize) -> Result<usize, String> {
        self.propositions
            .iter()
            .find(|proposition| proposition.id_numeric_condition == Some(condition_id))
            .map(|proposition| proposition.id)
            .ok_or_else(|| format!("LM-cut numeric condition {condition_id} has no proposition"))
    }

    fn evaluate_numeric_condition(
        &self,
        condition_id: usize,
        numeric_values: &[f64],
    ) -> Result<f64, String> {
        let condition = self
            .conditions
            .get(condition_id)
            .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;
        let epsilon = *self
            .epsilons
            .get(condition_id)
            .ok_or_else(|| format!("LM-cut epsilon for condition {condition_id} is missing"))?;
        Ok(condition.evaluate_slack(numeric_values, epsilon))
    }

    fn precondition_proposition_ids(&self, fact: &Fact) -> Vec<usize> {
        if !self.config.ignore_numeric {
            let var_id = usize::try_from(fact.var()).expect("fact variable id must fit usize");
            if let Some(condition_ids) = self
                .comparison_fact_to_condition_ids
                .get(&(var_id, fact.value()))
            {
                return condition_ids
                    .iter()
                    .map(|&condition_id| {
                        self.numeric_condition_proposition_id(condition_id)
                            .expect("comparison fact condition must have proposition")
                    })
                    .collect();
            }
        }

        if self.is_numeric_axiom_var(fact.var() as usize) {
            let fact_name = self.task.get_fact_name(fact);
            panic!(
                "LM-cut numeric conditions do not support disequality comparison fact `{fact_name}`"
            );
        }

        vec![self.get_proposition_id(fact)]
    }

    fn get_proposition_id_for_effect(&self, effect: &Effect) -> usize {
        let fact = Fact::new(effect.var_id(), effect.value() as i32);
        self.get_proposition_id(&fact)
    }

    fn proposition_name_for_effect(&self, effect: &Effect) -> String {
        let fact = Fact::new(effect.var_id(), effect.value() as i32);
        self.task.get_fact_name(&fact).to_string()
    }

    fn get_proposition_id(&self, fact: &Fact) -> usize {
        let variable_id = usize::try_from(fact.var()).expect("fact var id must fit usize");
        let value_id = usize::try_from(fact.value()).expect("fact value must be non-negative");
        let proposition_ids = self
            .proposition_index
            .get(variable_id)
            .expect("fact variable must exist in proposition index");
        *proposition_ids
            .get(value_id)
            .expect("fact value must exist in proposition index")
    }

    fn is_numeric_axiom_var(&self, variable_id: usize) -> bool {
        self.comparison_axiom_by_var.contains_key(&variable_id)
    }

    pub fn compute_landmarks(
        &mut self,
        propositional_values: &[i32],
        state_buffer_len: usize,
        numeric_values: &[f64],
    ) -> Result<(bool, f64, Vec<Landmark>), String> {
        assert!(
            self.initialized,
            "LM-cut landmarks used before initialization"
        );
        assert!(
            state_buffer_len > 0,
            "LM-cut scaffold requires a non-empty packed state buffer"
        );
        assert_eq!(
            numeric_values.len(),
            self.task.numeric_variables().len(),
            "LM-cut scaffold received the wrong number of numeric values"
        );
        if self.use_bounds {
            self.numeric_bound
                .calculate_bounds(numeric_values, self.config.bound_iterations);
        }

        for operator in &mut self.relaxed_operators {
            operator.cost_1 = operator.base_cost_1;
            operator.cost_2 = operator.base_cost_2;
        }

        self.first_exploration(propositional_values, numeric_values)?;
        if self.propositions[self.artificial_goal_id].status == PropositionStatus::Unreached {
            return Ok((true, f64::INFINITY, Vec::new()));
        }

        let mut total_cost = 0.0;
        let mut landmarks = Vec::new();
        let mut cut = Vec::new();
        let mut m_list = Vec::new();

        while self.propositions[self.artificial_goal_id].h_max_cost >= self.config.precision {
            self.mark_goal_plateau(
                propositional_values,
                numeric_values,
                self.artificial_goal_id,
            )?;
            self.second_exploration(propositional_values, numeric_values, &mut cut, &mut m_list)?;
            assert!(!cut.is_empty(), "LM-cut must find a non-empty cut");

            let mut cut_cost = f64::INFINITY;
            let mut operator_to_min_cut_cost: BTreeMap<usize, f64> = BTreeMap::new();
            let mut operator_to_m: BTreeMap<usize, f64> = BTreeMap::new();

            for (cut_index, &operator_id) in cut.iter().enumerate() {
                let multiplier = m_list[cut_index];
                let current_cut_cost = self.edge_cost(operator_id, multiplier)?;
                let operator = &self.relaxed_operators[operator_id];
                if let Some(original_id) = operator.original_op_id_1 {
                    let entry = operator_to_min_cut_cost
                        .entry(original_id)
                        .or_insert(current_cut_cost);
                    *entry = entry.min(current_cut_cost);
                }
                if let Some(original_id) = operator.original_op_id_2 {
                    let entry = operator_to_min_cut_cost
                        .entry(original_id)
                        .or_insert(current_cut_cost);
                    *entry = entry.min(current_cut_cost);
                }
                cut_cost = cut_cost.min(current_cut_cost);
            }

            if !cut_cost.is_finite() || cut_cost < self.config.precision {
                let cut_details = cut
                    .iter()
                    .zip(m_list.iter())
                    .map(|(&operator_id, &multiplier)| {
                        let operator = &self.relaxed_operators[operator_id];
                        let edge_cost = self.edge_cost(operator_id, multiplier).unwrap_or(f64::NAN);
                        format!(
                            "id={operator_id} name={} edge_cost={} m=({},{}) cost=({},{}) orig=({:?},{:?}) effects={:?}",
                            operator.name,
                            edge_cost,
                            multiplier.0,
                            multiplier.1,
                            operator.cost_1,
                            operator.cost_2,
                            operator.original_op_id_1,
                            operator.original_op_id_2,
                            operator.effect_ids,
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" | ");
                return Err(format!(
                    "LM-cut produced a non-progressing cut: cut_cost={cut_cost}, cut_size={}, goal_h_max={}, cut=[{}].",
                    cut.len(),
                    self.propositions[self.artificial_goal_id].h_max_cost,
                    cut_details,
                ));
            }

            total_cost += cut_cost;
            let mut progress_made = false;
            for (original_id, min_cost) in operator_to_min_cut_cost {
                if min_cost < self.config.precision {
                    continue;
                }
                let mapped = self.original_to_relaxed_operators[original_id].clone();
                for relaxed_operator_id in mapped {
                    let relaxed_operator = &mut self.relaxed_operators[relaxed_operator_id];
                    let mut multiplier = min_cost;
                    if relaxed_operator.original_op_id_1 == Some(original_id)
                        && relaxed_operator.cost_1 >= self.config.precision
                    {
                        multiplier /= relaxed_operator.cost_1;
                        let previous_cost = relaxed_operator.cost_1;
                        relaxed_operator.cost_1 =
                            (relaxed_operator.cost_1 - cut_cost / multiplier).max(0.0);
                        if relaxed_operator.cost_1 + self.config.precision < previous_cost {
                            progress_made = true;
                        }
                        operator_to_m.insert(original_id, multiplier);
                    }
                    if relaxed_operator.original_op_id_2 == Some(original_id)
                        && relaxed_operator.cost_2 >= self.config.precision
                    {
                        multiplier /= relaxed_operator.cost_2;
                        let previous_cost = relaxed_operator.cost_2;
                        relaxed_operator.cost_2 =
                            (relaxed_operator.cost_2 - cut_cost / multiplier).max(0.0);
                        if relaxed_operator.cost_2 + self.config.precision < previous_cost {
                            progress_made = true;
                        }
                        operator_to_m.insert(original_id, multiplier);
                    }
                }
            }

            if !progress_made {
                return Err(format!(
                    "LM-cut failed to reduce any relaxed-operator cost despite a positive cut: cut_cost={cut_cost}, cut_size={}, goal_h_max={}.",
                    cut.len(),
                    self.propositions[self.artificial_goal_id].h_max_cost,
                ));
            }

            landmarks.push(
                operator_to_m
                    .into_iter()
                    .map(|(operator_id, multiplier)| (multiplier, operator_id))
                    .collect(),
            );

            self.first_exploration_incremental(propositional_values, numeric_values, &cut)?;
            cut.clear();
            m_list.clear();
            self.reset_goal_zone_statuses();
            self.propositions[self.artificial_goal_id].status = PropositionStatus::Reached;
            self.propositions[self.artificial_precondition_id].status = PropositionStatus::Reached;
        }

        return Ok((false, total_cost, landmarks));
    }

    pub fn task(&self) -> &'task dyn AbstractNumericTask {
        self.task
    }

    pub fn propositions(&self) -> &[RelaxedProposition] {
        &self.propositions
    }

    pub fn relaxed_operators(&self) -> &[RelaxedOperator] {
        &self.relaxed_operators
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator, PropositionalAxiom};
    use planners_sas::numeric::numeric_task::{
        AssignmentEffect, AssignmentOperation, ExplicitVariable, Fact, Metric, NumericRootTask,
        NumericType, NumericVariable, Operator,
    };

    fn simple_var(name: &str, values: &[&str], axiom_layer: i32) -> ExplicitVariable {
        planners_sas::numeric::numeric_task::ExplicitVariable::new(
            values.len() as u32,
            name.to_string(),
            values.iter().map(|value| value.to_string()).collect(),
            axiom_layer,
            0,
        )
    }

    fn proposition_task() -> NumericRootTask {
        use planners_sas::numeric::numeric_task::{Effect, Fact, Operator};

        let variables = vec![
            simple_var("v0", &["v0-0", "v0-1"], -1),
            simple_var("v1", &["v1-0", "v1-1"], 0),
        ];
        let goals = vec![Fact::new(0, 1)];
        let operators = vec![Operator::new(
            "flip".to_string(),
            vec![Fact::new(0, 0)],
            vec![
                Effect::new(vec![], 0, 0, 1),
                Effect::new(vec![Fact::new(0, 0)], 1, 0, 1),
            ],
            vec![],
            1,
        )];
        let axioms = vec![PropositionalAxiom::new(vec![Fact::new(0, 1)], 1, 0, 1)];
        NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            vec![],
            goals,
            vec![],
            vec![0, 0],
            vec![],
            operators,
            axioms,
            vec![],
            vec![],
            (0, 0),
        )
    }

    #[test]
    fn initializes_propositions_and_relaxed_operators_for_propositional_task() {
        let task = proposition_task();
        let landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        assert_eq!(landmarks.propositions.len(), 6);
        assert_eq!(landmarks.relaxed_operators.len(), 4);

        let goal_operator = landmarks
            .relaxed_operators
            .last()
            .expect("goal operator should exist");
        assert_eq!(goal_operator.effect_ids, vec![landmarks.artificial_goal_id]);
        assert!(goal_operator.precondition_ids.contains(&3));

        let flip = landmarks
            .relaxed_operators
            .iter()
            .find(|op| op.name == "flip")
            .expect("base operator should exist");
        assert_eq!(flip.precondition_ids, vec![2]);
        assert_eq!(flip.effect_ids, vec![3]);

        let conditional = landmarks
            .relaxed_operators
            .iter()
            .find(|op| op.conditional)
            .expect("conditional relaxed operator should exist");
        assert!(conditional.precondition_ids.contains(&2));
        assert_eq!(conditional.effect_ids, vec![5]);
    }

    #[test]
    fn enqueue_ignores_subprecision_cost_improvements() {
        let task = proposition_task();
        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());
        landmarks.setup_exploration_queue();

        let proposition_id = landmarks.get_proposition_id(&Fact::new(0, 0));
        assert!(landmarks.enqueue_if_necessary(proposition_id, 1.0).unwrap());
        assert!(!landmarks
            .enqueue_if_necessary(
                proposition_id,
                1.0 - (landmarks.config.precision * 0.5),
            )
            .unwrap());
        assert_eq!(landmarks.propositions[proposition_id].h_max_cost, 1.0);
    }

    #[test]
    fn first_exploration_reaches_artificial_goal_for_supported_state() {
        let task = proposition_task();
        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        landmarks
            .first_exploration(&[0, 0], &[])
            .expect("first exploration should succeed for propositional task");

        assert_eq!(
            landmarks.propositions[landmarks.artificial_goal_id].status,
            PropositionStatus::Reached
        );
        assert_eq!(
            landmarks.propositions[landmarks.artificial_goal_id].h_max_cost,
            1.0
        );
    }

    #[test]
    fn compute_landmarks_reports_dead_end_when_goal_is_unreachable() {
        use planners_sas::numeric::numeric_task::Fact;

        let variables = vec![simple_var("v0", &["v0-0", "v0-1"], -1)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            vec![],
            vec![Fact::new(0, 1)],
            vec![],
            vec![0],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            (0, 0),
        );
        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[0], 1, &[])
            .expect("dead-end detection should finish before later LM-cut phases");

        assert!(dead_end);
        assert!(total_cost.is_infinite());
        assert!(cuts.is_empty());
    }

    #[test]
    fn numeric_goal_condition_is_seeded_from_numeric_state() {
        use planners_sas::numeric::numeric_task::{Fact, NumericType, NumericVariable};

        let variables = vec![simple_var("cmp", &["lt", "ge"], 0)];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
        ];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 0)],
            vec![],
            vec![1],
            vec![1.0, 2.0],
            vec![],
            vec![],
            comparison_axioms,
            vec![],
            (0, 0),
        );
        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[1], 1, &[1.0, 2.0])
            .expect("satisfied numeric goal condition should produce zero LM-cut cost");

        assert!(!dead_end);
        assert_eq!(total_cost, 0.0);
        assert!(cuts.is_empty());
        assert_eq!(
            landmarks.propositions[landmarks.artificial_goal_id].status,
            PropositionStatus::Reached
        );
    }

    #[test]
    fn goal_operator_compiles_numeric_goal_axiom_conditions() {
        use planners_sas::numeric::numeric_task::{Fact, NumericType, NumericVariable};

        let variables = vec![
            simple_var("cmp", &["lt", "ge"], 0),
            simple_var("goal", &["not-done", "done"], 1),
        ];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
        ];
        let axioms = vec![PropositionalAxiom::new(vec![Fact::new(0, 0)], 1, 0, 1)];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(1, 1)],
            vec![],
            vec![1, 0],
            vec![1.0, 2.0],
            vec![],
            axioms,
            comparison_axioms,
            vec![],
            (0, 0),
        );
        let landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        let goal_operator = landmarks
            .relaxed_operators
            .last()
            .expect("goal operator should exist");
        let derived_goal_proposition_id = landmarks.get_proposition_id(&Fact::new(1, 1));

        assert!(goal_operator
            .precondition_ids
            .contains(&derived_goal_proposition_id));
        assert!(goal_operator.precondition_ids.iter().any(|&id| {
            landmarks.propositions[id].is_numeric_condition
        }));
    }

    #[test]
    fn numeric_equality_goal_condition_is_seeded_from_numeric_state() {
        use planners_sas::numeric::numeric_task::{Fact, NumericType, NumericVariable};

        let variables = vec![simple_var("cmp", &["eq", "neq"], 0)];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
        ];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::Equal)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 0)],
            vec![],
            vec![0],
            vec![2.0, 2.0],
            vec![],
            vec![],
            comparison_axioms,
            vec![],
            (0, 0),
        );
        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[0], 1, &[2.0, 2.0])
            .expect("satisfied equality goal condition should produce zero LM-cut cost");

        assert!(!dead_end);
        assert_eq!(total_cost, 0.0);
        assert!(cuts.is_empty());
        assert_eq!(
            landmarks.propositions[landmarks.artificial_goal_id].status,
            PropositionStatus::Reached
        );
    }

    #[test]
    #[should_panic(expected = "do not support disequality comparison fact")]
    fn rejects_disequality_goal_fact_for_equality_axiom() {
        use planners_sas::numeric::numeric_task::{Fact, NumericType, NumericVariable};

        let variables = vec![simple_var("cmp", &["eq", "neq"], 0)];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
        ];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::Equal)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 1)],
            vec![],
            vec![1],
            vec![2.0, 3.0],
            vec![],
            vec![],
            comparison_axioms,
            vec![],
            (0, 0),
        );

        let _ = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());
    }

    #[test]
    fn compute_landmarks_returns_propositional_cut_cost() {
        let task = proposition_task();
        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[0, 0], 2, &[])
            .expect("propositional LM-cut slice should compute a cut cost");

        assert!(!dead_end);
        assert_eq!(total_cost, 1.0);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0], vec![(1.0, 0)]);
    }

    #[test]
    fn compute_landmarks_returns_numeric_cut_cost_for_assignment_effect() {
        let variables = vec![
            simple_var("cmp", &["false", "true"], 0),
            simple_var("ready", &["yes", "no"], -1),
        ];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
            NumericVariable::new("inc".to_string(), NumericType::Constant, -1),
        ];
        let operators = vec![Operator::new(
            "increase-y".to_string(),
            vec![Fact::new(1, 0)],
            vec![],
            vec![AssignmentEffect::new(
                1,
                AssignmentOperation::Plus,
                2,
                false,
                vec![],
            )],
            1,
        )];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 0)],
            vec![],
            vec![1, 0],
            vec![2.0, 1.0, 2.0],
            operators,
            vec![],
            comparison_axioms,
            vec![],
            (0, 0),
        );
        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[1, 0], 2, &[2.0, 1.0, 2.0])
            .expect("simple numeric achiever should produce a finite numeric cut");

        assert!(!dead_end);
        assert_eq!(total_cost, 0.5);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0], vec![(0.5, 0)]);
    }

    #[test]
    fn conditional_propositional_effect_compiles_numeric_guard() {
        use planners_sas::numeric::numeric_task::{Effect, Fact, NumericType, NumericVariable};

        let variables = vec![
            simple_var("cmp", &["lt", "ge"], 0),
            simple_var("done", &["no", "yes"], -1),
        ];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
        ];
        let operators = vec![Operator::new(
            "finish-when-x-lt-y".to_string(),
            vec![],
            vec![Effect::new(vec![Fact::new(0, 0)], 1, 0, 1)],
            vec![],
            1,
        )];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(1, 1)],
            vec![],
            vec![1, 0],
            vec![1.0, 2.0],
            operators,
            vec![],
            comparison_axioms,
            vec![],
            (0, 0),
        );
        let landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        let done_yes_proposition_id = landmarks.get_proposition_id(&Fact::new(1, 1));
        let conditional_operator = landmarks
            .relaxed_operators
            .iter()
            .find(|operator| operator.conditional && operator.effect_ids == vec![done_yes_proposition_id])
            .expect("conditional relaxed operator should exist");
        let comparison_fact_proposition_id = landmarks.get_proposition_id(&Fact::new(0, 0));

        assert!(conditional_operator.precondition_ids.iter().all(|&id| {
            landmarks.propositions[id].is_numeric_condition
        }));
        assert!(!conditional_operator
            .precondition_ids
            .contains(&comparison_fact_proposition_id));
        assert_eq!(conditional_operator.effect_ids, vec![done_yes_proposition_id]);
    }

    #[test]
    fn compute_landmarks_uses_linearized_derived_assignment_effects() {
        let variables = vec![
            simple_var("cmp", &["false", "true"], 0),
            simple_var("ready", &["yes", "no"], -1),
        ];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
            NumericVariable::new("sum".to_string(), NumericType::Derived, -1),
            NumericVariable::new("z".to_string(), NumericType::Regular, -1),
        ];
        let operators = vec![Operator::new(
            "increase-z".to_string(),
            vec![Fact::new(1, 0)],
            vec![],
            vec![AssignmentEffect::new(
                3,
                AssignmentOperation::Plus,
                2,
                false,
                vec![],
            )],
            1,
        )];
        let assignment_axioms = vec![planners_sas::numeric::axioms::AssignmentAxiom::new(
            2,
            planners_sas::numeric::axioms::CalOperator::Sum,
            0,
            1,
        )];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 3, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 0)],
            vec![],
            vec![1, 0],
            vec![2.0, 1.0, 3.0, 0.0],
            operators,
            vec![],
            comparison_axioms,
            assignment_axioms,
            (0, 0),
        );
        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[1, 0], 2, &[2.0, 1.0, 3.0, 0.0])
            .expect("derived linear source expressions should support numeric LM-cut");

        assert!(!dead_end);
        assert_eq!(total_cost, 1.0);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0], vec![(1.0, 0)]);
    }

    #[test]
    fn compute_landmarks_uses_linearized_derived_numeric_conditions() {
        let variables = vec![simple_var("cmp", &["false", "true"], 0)];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
            NumericVariable::new("sum".to_string(), NumericType::Derived, -1),
            NumericVariable::new("target".to_string(), NumericType::Regular, -1),
            NumericVariable::new("inc".to_string(), NumericType::Constant, -1),
        ];
        let operators = vec![Operator::new(
            "increase-x".to_string(),
            vec![],
            vec![],
            vec![AssignmentEffect::new(
                0,
                AssignmentOperation::Plus,
                4,
                false,
                vec![],
            )],
            1,
        )];
        let assignment_axioms = vec![planners_sas::numeric::axioms::AssignmentAxiom::new(
            2,
            planners_sas::numeric::axioms::CalOperator::Sum,
            0,
            1,
        )];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 3, 2, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 0)],
            vec![],
            vec![1],
            vec![0.0, 1.0, 1.0, 3.0, 1.0],
            operators,
            vec![],
            comparison_axioms,
            assignment_axioms,
            (0, 0),
        );
        let mut landmarks = LandmarkCutLandmarks::new(&task, LmCutNumericConfig::default());

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[1], 1, &[0.0, 1.0, 1.0, 3.0, 1.0])
            .expect("derived numeric conditions should be linearized to base variables");

        assert!(!dead_end);
        assert_eq!(total_cost, 2.0);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0], vec![(2.0, 0)]);
    }

    #[test]
    fn compute_landmarks_builds_supported_sose_cut() {
        let variables = vec![simple_var("cmp", &["false", "true"], 0)];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
            NumericVariable::new("z".to_string(), NumericType::Regular, -1),
            NumericVariable::new("inc".to_string(), NumericType::Constant, -1),
        ];
        let operators = vec![
            Operator::new(
                "increase-z".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    2,
                    AssignmentOperation::Plus,
                    3,
                    false,
                    vec![],
                )],
                1,
            ),
            Operator::new(
                "increase-y-by-z".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    2,
                    false,
                    vec![],
                )],
                1,
            ),
        ];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 0)],
            vec![],
            vec![1],
            vec![5.0, 1.0, 1.0, 1.0],
            operators,
            vec![],
            comparison_axioms,
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.use_second_order_simple = true;
        let mut landmarks = LandmarkCutLandmarks::new(&task, config);

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[1], 1, &[5.0, 1.0, 1.0, 1.0])
            .expect("supported SOSE case should compute a finite numeric cut");

        assert!(!dead_end);
        assert_eq!(total_cost, 1.0);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0], vec![(3.0, 0), (1.0, 1)]);
    }

    #[test]
    fn compute_landmarks_builds_supported_sose_cut_for_conditional_relaxed_target() {
        let variables = vec![
            simple_var("cmp", &["false", "true"], 0),
            simple_var("ready", &["false", "true"], -1),
        ];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
            NumericVariable::new("z".to_string(), NumericType::Regular, -1),
            NumericVariable::new("inc".to_string(), NumericType::Constant, -1),
        ];
        let operators = vec![
            Operator::new(
                "increase-z".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    2,
                    AssignmentOperation::Plus,
                    3,
                    false,
                    vec![],
                )],
                1,
            ),
            Operator::new(
                "increase-y-by-z-when-ready".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    2,
                    true,
                    vec![Fact::new(1, 1)],
                )],
                1,
            ),
        ];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 0)],
            vec![],
            vec![1, 1],
            vec![5.0, 1.0, 1.0, 1.0],
            operators,
            vec![],
            comparison_axioms,
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.use_second_order_simple = true;
        let mut landmarks = LandmarkCutLandmarks::new(&task, config);

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[1, 1], 2, &[5.0, 1.0, 1.0, 1.0])
            .expect("conditional relaxed SOSE target should compute a finite numeric cut");

        assert!(!dead_end);
        assert_eq!(total_cost, 1.0);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0], vec![(3.0, 0), (1.0, 1)]);
    }

    #[test]
    fn compute_landmarks_builds_supported_sose_cut_for_conditional_supporter() {
        let variables = vec![
            simple_var("cmp", &["false", "true"], 0),
            simple_var("ready", &["false", "true"], -1),
        ];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
            NumericVariable::new("z".to_string(), NumericType::Regular, -1),
            NumericVariable::new("inc".to_string(), NumericType::Constant, -1),
        ];
        let operators = vec![
            Operator::new(
                "increase-z-when-ready".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    2,
                    AssignmentOperation::Plus,
                    3,
                    true,
                    vec![Fact::new(1, 1)],
                )],
                1,
            ),
            Operator::new(
                "increase-y-by-z".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    2,
                    false,
                    vec![],
                )],
                1,
            ),
        ];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 0)],
            vec![],
            vec![1, 1],
            vec![5.0, 1.0, 1.0, 1.0],
            operators,
            vec![],
            comparison_axioms,
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.use_second_order_simple = true;
        let mut landmarks = LandmarkCutLandmarks::new(&task, config);

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[1, 1], 2, &[5.0, 1.0, 1.0, 1.0])
            .expect("conditional SOSE supporter should compute a finite numeric cut");

        assert!(!dead_end);
        assert_eq!(total_cost, 1.0);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0], vec![(3.0, 0), (1.0, 1)]);
    }

    #[test]
    fn compute_landmarks_ignores_non_simple_supporters_when_valid_supporter_exists() {
        let variables = vec![simple_var("cmp", &["false", "true"], 0)];
        let numeric_variables = vec![
            NumericVariable::new("x".to_string(), NumericType::Regular, -1),
            NumericVariable::new("y".to_string(), NumericType::Regular, -1),
            NumericVariable::new("z".to_string(), NumericType::Regular, -1),
            NumericVariable::new("inc".to_string(), NumericType::Constant, -1),
        ];
        let operators = vec![
            Operator::new(
                "increase-z-by-inc".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    2,
                    AssignmentOperation::Plus,
                    3,
                    false,
                    vec![],
                )],
                1,
            ),
            Operator::new(
                "increase-z-by-x".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    2,
                    AssignmentOperation::Plus,
                    0,
                    false,
                    vec![],
                )],
                1,
            ),
            Operator::new(
                "increase-y-by-z".to_string(),
                vec![],
                vec![],
                vec![AssignmentEffect::new(
                    1,
                    AssignmentOperation::Plus,
                    2,
                    false,
                    vec![],
                )],
                1,
            ),
        ];
        let comparison_axioms = vec![ComparisonAxiom::new(0, 0, 1, ComparisonOperator::LessThan)];
        let task = NumericRootTask::new(
            3,
            Metric::new(true, -1),
            variables,
            numeric_variables,
            vec![Fact::new(0, 0)],
            vec![],
            vec![1],
            vec![5.0, 1.0, 1.0, 1.0],
            operators,
            vec![],
            comparison_axioms,
            vec![],
            (0, 0),
        );
        let mut config = LmCutNumericConfig::default();
        config.use_second_order_simple = true;
        let mut landmarks = LandmarkCutLandmarks::new(&task, config);

        let (dead_end, total_cost, cuts) = landmarks
            .compute_landmarks(&[1], 1, &[5.0, 1.0, 1.0, 1.0])
            .expect("valid simple supporter should still enable SOSE when another supporter is non-simple");

        assert!(!dead_end);
        assert_eq!(total_cost, 1.0);
        assert_eq!(cuts.len(), 1);
        assert_eq!(cuts[0], vec![(3.0, 0), (1.0, 2)]);
    }
}
