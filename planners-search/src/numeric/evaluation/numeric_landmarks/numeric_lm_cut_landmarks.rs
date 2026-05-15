use super::lm_cut_numeric_heuristic::LmCutNumericConfig;
use super::numeric_bound::NumericBound;
use super::numeric_helper::{LinearNumericCondition as NumericCondition, NumericTaskHelper};
use crate::numeric::evaluation::domain_abstractions::transition_cost_partitioning::{
    LmCutResidualOperatorCostPartition, StateRegion,
};
use planners_sas::numeric::axioms::PropositionalAxiom;
use planners_sas::numeric::numeric_task::{
    AbstractNumericTask, Effect, ExplicitFact, Operator, metric_operator_cost_from_initial_values,
};
use planners_sas::numeric::utils::linear_effects::LinearExpression;
use planners_sas::numeric::utils::linear_effects::LinearNumericEffect;
use std::collections::{BTreeMap, BTreeSet};
use tracing::debug;

#[derive(Debug, Clone, Copy)]
struct QueueEntry {
    cost: f64,
    proposition_id: usize,
}

#[derive(Debug, Default, Clone)]
struct PriorityQueue {
    entries: Vec<QueueEntry>,
}

impl PriorityQueue {
    #[inline(always)]
    fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    #[inline(always)]
    fn clear(&mut self) {
        self.entries.clear();
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    #[inline(always)]
    fn push(&mut self, entry: QueueEntry) {
        self.entries.push(entry);
        let mut index = self.entries.len() - 1;
        while index > 0 {
            let parent = (index - 1) / 2;
            if self.entries[parent].cost <= self.entries[index].cost {
                break;
            }
            self.entries.swap(parent, index);
            index = parent;
        }
    }

    #[inline(always)]
    fn pop(&mut self) -> Option<QueueEntry> {
        let last = self.entries.pop()?;
        if self.entries.is_empty() {
            return Some(last);
        }

        let result = std::mem::replace(&mut self.entries[0], last);
        let len = self.entries.len();
        let mut index = 0;
        loop {
            let left = (2 * index) + 1;
            if left >= len {
                break;
            }

            let right = left + 1;
            let mut child = left;
            if right < len && self.entries[right].cost < self.entries[left].cost {
                child = right;
            }

            if self.entries[index].cost <= self.entries[child].cost {
                break;
            }

            self.entries.swap(index, child);
            index = child;
        }

        Some(result)
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
            h_max_supporter_cost: f64::INFINITY,
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

#[derive(Debug, Clone, Copy)]
struct PropositionRuntime {
    status: PropositionStatus,
    h_max_cost: f64,
    mark: u32,
    goal_zone_touched: bool,
}

impl Default for PropositionRuntime {
    fn default() -> Self {
        Self {
            status: PropositionStatus::Unreached,
            h_max_cost: f64::INFINITY,
            mark: 0,
            goal_zone_touched: false,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct OperatorRuntime {
    cost_1: f64,
    cost_2: f64,
    unsatisfied_preconditions: usize,
    h_max_supporter: Option<usize>,
    h_max_supporter_cost: f64,
    mark: u32,
}

impl OperatorRuntime {
    fn new(operator: &RelaxedOperator) -> Self {
        Self {
            cost_1: operator.base_cost_1,
            cost_2: operator.base_cost_2,
            unsatisfied_preconditions: operator.precondition_ids.len(),
            h_max_supporter: None,
            h_max_supporter_cost: f64::INFINITY,
            mark: 0,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct OperatorConditionEval {
    simple_effect: Option<f64>,
    has_sose: bool,
    composite_expression: Option<LinearExpression>,
    has_upper_bound: bool,
    upper_bound: f64,
}

pub type Landmark = Vec<(f64, usize)>;

type ComputeLandmarksResult = (bool, f64, Option<Vec<Landmark>>);

pub struct LandmarkCutLandmarks<'task> {
    task: &'task dyn AbstractNumericTask,
    config: LmCutNumericConfig,
    fixed_operator_costs: Option<Vec<f64>>,
    residual_operator_cost_partitions: Option<Vec<LmCutResidualOperatorCostPartition>>,
    residual_variant_precondition_ids: Vec<Vec<Vec<usize>>>,
    propositions: Vec<RelaxedProposition>,
    proposition_runtime: Vec<PropositionRuntime>,
    proposition_precondition_of_data: Vec<usize>,
    proposition_precondition_of_ranges: Vec<(usize, usize)>,
    proposition_effect_of_data: Vec<usize>,
    proposition_effect_of_ranges: Vec<(usize, usize)>,
    proposition_index: Vec<Vec<usize>>,
    numeric_condition_proposition_ids: Vec<usize>,
    conditions: Vec<NumericCondition>,
    epsilons: Vec<f64>,
    numeric_helper: NumericTaskHelper,
    comparison_fact_to_condition_ids: BTreeMap<(usize, usize), Vec<usize>>,
    linear_effect_to_conditions_plus: Vec<Vec<Vec<usize>>>,
    linear_effect_to_conditions_minus: Vec<Vec<Vec<usize>>>,
    operator_condition_eval: Vec<Vec<OperatorConditionEval>>,
    relaxed_operators: Vec<RelaxedOperator>,
    operator_runtime: Vec<OperatorRuntime>,
    proposition_runtime_epoch: u32,
    touched_proposition_ids: Vec<usize>,
    goal_zone_proposition_ids: Vec<usize>,
    operator_runtime_epoch: u32,
    touched_operator_ids: Vec<usize>,
    regular_numeric_variable_ids: Vec<usize>,
    operator_precondition_id_data: Vec<usize>,
    operator_precondition_id_ranges: Vec<(usize, usize)>,
    operator_effect_id_data: Vec<usize>,
    operator_effect_id_ranges: Vec<(usize, usize)>,
    original_to_relaxed_operators: Vec<Vec<usize>>,
    goal_precondition_ids: Vec<usize>,
    artificial_precondition_id: usize,
    artificial_goal_id: usize,
    num_propositions: usize,
    num_variables: usize,
    numeric_initial_state: Vec<f64>,
    priority_queue: PriorityQueue,
    cut_marks: Vec<u32>,
    cut_mark_epoch: u32,
    original_operator_min_cut_costs: Vec<f64>,
    original_operator_min_cut_cost_marks: Vec<u32>,
    original_operator_min_cut_cost_epoch: u32,
    original_operator_multipliers: Vec<f64>,
    original_operator_multiplier_marks: Vec<u32>,
    original_operator_multiplier_epoch: u32,
    touched_original_operator_ids: Vec<usize>,
    landmark_original_operator_ids: Vec<usize>,
    incremental_original_operator_marks: Vec<u32>,
    incremental_original_operator_epoch: u32,
    incremental_original_operator_ids_scratch: Vec<usize>,
    composite_expression_values: Vec<Vec<f64>>,
    composite_expression_value_marks: Vec<Vec<u32>>,
    composite_expression_value_epoch: u32,
    cut_scratch: Vec<usize>,
    multiplier_scratch: Vec<(f64, f64)>,
    second_exploration_queue_scratch: Vec<usize>,
    numeric_bound: NumericBound,
    use_bounds: bool,
    initialized: bool,
}

#[allow(unused)]
impl<'task> LandmarkCutLandmarks<'task> {
    pub fn new(task: &'task dyn AbstractNumericTask, config: LmCutNumericConfig) -> Self {
        Self::new_with_fixed_operator_costs(task, config, None)
    }

    pub fn new_with_fixed_operator_costs(
        task: &'task dyn AbstractNumericTask,
        config: LmCutNumericConfig,
        fixed_operator_costs: Option<Vec<f64>>,
    ) -> Self {
        Self::new_with_residual_operator_cost_partitions(task, config, fixed_operator_costs, None)
    }

    pub fn new_with_residual_operator_cost_partitions(
        task: &'task dyn AbstractNumericTask,
        config: LmCutNumericConfig,
        fixed_operator_costs: Option<Vec<f64>>,
        residual_operator_cost_partitions: Option<Vec<LmCutResidualOperatorCostPartition>>,
    ) -> Self {
        // PARITY(numeric-fd): `random_pcf` is still blocked until the randomized supporter
        // selection path is ported. The other current feature flags are wired through the same
        // control-flow sites as the C++ implementation.
        assert!(
            config.precision >= 0.0,
            "LM-cut precision must be non-negative"
        );
        assert!(config.epsilon >= 0.0, "LM-cut epsilon must be non-negative");
        assert!(
            !config.random_pcf,
            "LM-cut random_pcf=true is not implemented yet"
        );
        let use_bounds = config.bound_iterations > 0;
        let numeric_bound = NumericBound::new(task, config.precision, config.epsilon);
        let numeric_helper = NumericTaskHelper::new_lmcut(
            task,
            config.precision,
            config.epsilon,
            config.use_constant_assignment,
        );
        let regular_numeric_variable_ids = task.regular_numeric_variable_ids();
        let mut result = Self {
            task,
            config,
            fixed_operator_costs,
            residual_operator_cost_partitions,
            residual_variant_precondition_ids: Vec::new(),
            propositions: Vec::new(),
            proposition_runtime: Vec::new(),
            proposition_precondition_of_data: Vec::new(),
            proposition_precondition_of_ranges: Vec::new(),
            proposition_effect_of_data: Vec::new(),
            proposition_effect_of_ranges: Vec::new(),
            proposition_index: Vec::new(),
            numeric_condition_proposition_ids: Vec::new(),
            conditions: Vec::new(),
            epsilons: Vec::new(),
            numeric_helper,
            comparison_fact_to_condition_ids: BTreeMap::new(),
            linear_effect_to_conditions_plus: Vec::new(),
            linear_effect_to_conditions_minus: Vec::new(),
            operator_condition_eval: Vec::new(),
            relaxed_operators: Vec::new(),
            operator_runtime: Vec::new(),
            proposition_runtime_epoch: 1,
            touched_proposition_ids: Vec::new(),
            goal_zone_proposition_ids: Vec::new(),
            operator_runtime_epoch: 1,
            touched_operator_ids: Vec::new(),
            regular_numeric_variable_ids,
            operator_precondition_id_data: Vec::new(),
            operator_precondition_id_ranges: Vec::new(),
            operator_effect_id_data: Vec::new(),
            operator_effect_id_ranges: Vec::new(),
            original_to_relaxed_operators: Vec::new(),
            goal_precondition_ids: Vec::new(),
            artificial_precondition_id: 0,
            artificial_goal_id: 1,
            num_propositions: 0,
            num_variables: 0,
            numeric_initial_state: Vec::new(),
            priority_queue: PriorityQueue::new(),
            cut_marks: Vec::new(),
            cut_mark_epoch: 1,
            original_operator_min_cut_costs: Vec::new(),
            original_operator_min_cut_cost_marks: Vec::new(),
            original_operator_min_cut_cost_epoch: 1,
            original_operator_multipliers: Vec::new(),
            original_operator_multiplier_marks: Vec::new(),
            original_operator_multiplier_epoch: 1,
            touched_original_operator_ids: Vec::new(),
            landmark_original_operator_ids: Vec::new(),
            incremental_original_operator_marks: Vec::new(),
            incremental_original_operator_epoch: 1,
            incremental_original_operator_ids_scratch: Vec::new(),
            composite_expression_values: Vec::new(),
            composite_expression_value_marks: Vec::new(),
            composite_expression_value_epoch: 1,
            cut_scratch: Vec::new(),
            multiplier_scratch: Vec::new(),
            second_exploration_queue_scratch: Vec::new(),
            numeric_bound,
            use_bounds,
            initialized: false,
        };
        result.initialize();
        result
    }

    fn initialize(&mut self) {
        assert!(!self.initialized, "LM-cut landmarks initialized twice");
        let debug_summary = std::env::var_os("LMCUT_DEBUG_SUMMARY").is_some();
        self.propositions.clear();
        self.proposition_runtime.clear();
        self.proposition_precondition_of_data.clear();
        self.proposition_precondition_of_ranges.clear();
        self.proposition_effect_of_data.clear();
        self.proposition_effect_of_ranges.clear();
        self.proposition_index.clear();
        self.numeric_condition_proposition_ids.clear();
        self.conditions.clear();
        self.epsilons.clear();
        self.comparison_fact_to_condition_ids.clear();
        self.linear_effect_to_conditions_plus.clear();
        self.linear_effect_to_conditions_minus.clear();
        self.operator_condition_eval.clear();
        self.composite_expression_values.clear();
        self.composite_expression_value_marks.clear();
        self.relaxed_operators.clear();
        self.operator_runtime.clear();
        self.touched_proposition_ids.clear();
        self.goal_zone_proposition_ids.clear();
        self.touched_operator_ids.clear();
        self.operator_precondition_id_data.clear();
        self.operator_precondition_id_ranges.clear();
        self.operator_effect_id_data.clear();
        self.operator_effect_id_ranges.clear();
        self.original_to_relaxed_operators.clear();
        self.residual_variant_precondition_ids.clear();
        self.goal_precondition_ids.clear();
        self.propositions.push(RelaxedProposition::new(
            self.artificial_precondition_id,
            "artificial".to_string(),
        ));
        self.propositions.push(RelaxedProposition::new(
            self.artificial_goal_id,
            "goal".to_string(),
        ));
        self.num_variables = self.task.get_num_variables();
        self.proposition_index = vec![Vec::new(); self.num_variables];

        self.num_propositions = 2;
        self.build_propositional_propositions();
        if debug_summary {
            debug!(
                "LMCUT_DEBUG_STAGE after_props prop={} numeric_conditions={}",
                self.num_propositions,
                self.conditions.len()
            );
        }
        self.build_numeric_conditions();
        self.build_residual_variant_conditions();
        if debug_summary {
            debug!(
                "LMCUT_DEBUG_STAGE after_numeric_conditions prop={} numeric_conditions={}",
                self.num_propositions,
                self.conditions.len()
            );
        }
        self.build_comparison_fact_condition_ids();
        self.add_linear_conditions();
        if debug_summary {
            debug!(
                "LMCUT_DEBUG_STAGE after_linear_conditions prop={} numeric_conditions={}",
                self.num_propositions,
                self.conditions.len()
            );
        }
        self.prepare_goal_preconditions();
        self.build_relaxed_operators();
        self.build_goal_operator();
        self.build_original_to_relaxed_index();
        self.build_cross_references();
        self.build_packed_runtime_adjacency();
        self.proposition_runtime
            .resize(self.propositions.len(), PropositionRuntime::default());
        self.operator_runtime = self
            .relaxed_operators
            .iter()
            .map(OperatorRuntime::new)
            .collect();
        self.composite_expression_values = self
            .operator_condition_eval
            .iter()
            .map(|row| vec![0.0; row.len()])
            .collect();
        self.composite_expression_value_marks = self
            .operator_condition_eval
            .iter()
            .map(|row| vec![0; row.len()])
            .collect();
        self.original_operator_min_cut_costs
            .resize(self.original_to_relaxed_operators.len(), 0.0);
        self.original_operator_min_cut_cost_marks
            .resize(self.original_to_relaxed_operators.len(), 0);
        self.original_operator_multipliers
            .resize(self.original_to_relaxed_operators.len(), 0.0);
        self.original_operator_multiplier_marks
            .resize(self.original_to_relaxed_operators.len(), 0);
        self.incremental_original_operator_marks
            .resize(self.original_to_relaxed_operators.len(), 0);
        self.cut_marks.resize(self.relaxed_operators.len(), 0);
        if self.use_bounds {
            let initial_numeric_values = self.task.get_initial_numeric_state_values();
            self.numeric_bound
                .calculate_bounds(&initial_numeric_values, self.config.bound_iterations);
        }
        if debug_summary {
            let infinite_operators = self
                .relaxed_operators
                .iter()
                .filter(|operator| operator.infinite)
                .count();
            let second_order_simple_operators = self
                .relaxed_operators
                .iter()
                .filter(|operator| operator.original_op_id_1.is_some())
                .count();
            debug!(
                "LMCUT_DEBUG_SUMMARY infinite={} sose={} ops={} prop={} numeric_conditions={}",
                infinite_operators,
                second_order_simple_operators,
                self.task.get_operators().len() + self.task.axioms().len(),
                self.num_propositions,
                self.conditions.len()
            );
        }
        self.initialized = true;
    }

    fn start_cut_marking(&mut self) {
        if self.cut_mark_epoch == u32::MAX {
            self.cut_marks.fill(0);
            self.cut_mark_epoch = 1;
        } else {
            self.cut_mark_epoch += 1;
        }
    }

    fn is_cut_marked(&self, operator_id: usize) -> bool {
        self.cut_marks.get(operator_id).copied().unwrap_or(0) == self.cut_mark_epoch
    }

    fn mark_cut(&mut self, operator_id: usize) {
        if let Some(mark) = self.cut_marks.get_mut(operator_id) {
            *mark = self.cut_mark_epoch;
        }
    }

    fn advance_epoch(epoch: &mut u32, marks: &mut [u32]) {
        if *epoch == u32::MAX {
            marks.fill(0);
            *epoch = 1;
        } else {
            *epoch += 1;
        }
    }

    fn start_cut_iteration_tracking(&mut self) {
        Self::advance_epoch(
            &mut self.original_operator_min_cut_cost_epoch,
            &mut self.original_operator_min_cut_cost_marks,
        );
        Self::advance_epoch(
            &mut self.original_operator_multiplier_epoch,
            &mut self.original_operator_multiplier_marks,
        );
        self.touched_original_operator_ids.clear();
        self.landmark_original_operator_ids.clear();
    }

    fn start_state_numeric_tracking(&mut self) {
        if self.composite_expression_value_epoch == u32::MAX {
            for row in &mut self.composite_expression_value_marks {
                row.fill(0);
            }
            self.composite_expression_value_epoch = 1;
        } else {
            self.composite_expression_value_epoch += 1;
        }
    }

    fn start_exploration_runtime_tracking(&mut self) {
        if self.proposition_runtime_epoch == u32::MAX {
            for runtime in &mut self.proposition_runtime {
                runtime.mark = 0;
            }
            self.proposition_runtime_epoch = 1;
        } else {
            self.proposition_runtime_epoch += 1;
        }
        if self.operator_runtime_epoch == u32::MAX {
            for runtime in &mut self.operator_runtime {
                runtime.mark = 0;
            }
            self.operator_runtime_epoch = 1;
        } else {
            self.operator_runtime_epoch += 1;
        }
        self.touched_proposition_ids.clear();
        self.touched_operator_ids.clear();
    }

    #[inline(always)]
    fn mark_proposition_runtime_touched(&mut self, proposition_id: usize) {
        if self.proposition_runtime[proposition_id].mark != self.proposition_runtime_epoch {
            self.proposition_runtime[proposition_id].mark = self.proposition_runtime_epoch;
            self.touched_proposition_ids.push(proposition_id);
        }
    }

    #[inline(always)]
    fn mark_operator_runtime_touched(&mut self, operator_id: usize) {
        if self.operator_runtime[operator_id].mark != self.operator_runtime_epoch {
            self.operator_runtime[operator_id].mark = self.operator_runtime_epoch;
            self.touched_operator_ids.push(operator_id);
        }
    }

    #[inline(always)]
    fn proposition_status(&self, proposition_id: usize) -> PropositionStatus {
        self.proposition_runtime[proposition_id].status
    }

    #[inline(always)]
    fn set_proposition_status(&mut self, proposition_id: usize, status: PropositionStatus) {
        self.mark_proposition_runtime_touched(proposition_id);
        let runtime = &mut self.proposition_runtime[proposition_id];
        if matches!(
            status,
            PropositionStatus::GoalZone | PropositionStatus::BeforeGoalZone
        ) && !runtime.goal_zone_touched
        {
            runtime.goal_zone_touched = true;
            self.goal_zone_proposition_ids.push(proposition_id);
        }
        runtime.status = status;
    }

    #[inline(always)]
    fn proposition_h_max_cost(&self, proposition_id: usize) -> f64 {
        self.proposition_runtime[proposition_id].h_max_cost
    }

    #[inline(always)]
    fn set_proposition_h_max_cost(&mut self, proposition_id: usize, cost: f64) {
        self.mark_proposition_runtime_touched(proposition_id);
        self.proposition_runtime[proposition_id].h_max_cost = cost;
    }

    #[inline(always)]
    fn operator_unsatisfied_preconditions(&self, operator_id: usize) -> usize {
        self.operator_runtime[operator_id].unsatisfied_preconditions
    }

    #[inline(always)]
    fn set_operator_unsatisfied_preconditions(&mut self, operator_id: usize, count: usize) {
        self.mark_operator_runtime_touched(operator_id);
        self.operator_runtime[operator_id].unsatisfied_preconditions = count;
    }

    #[inline(always)]
    fn decrement_operator_unsatisfied_preconditions(&mut self, operator_id: usize) -> usize {
        self.mark_operator_runtime_touched(operator_id);
        self.operator_runtime[operator_id].unsatisfied_preconditions -= 1;
        self.operator_runtime[operator_id].unsatisfied_preconditions
    }

    #[inline(always)]
    fn set_operator_h_max_supporter(&mut self, operator_id: usize, supporter_id: Option<usize>) {
        self.mark_operator_runtime_touched(operator_id);
        self.operator_runtime[operator_id].h_max_supporter = supporter_id;
    }

    #[inline(always)]
    fn set_operator_h_max_supporter_cost(&mut self, operator_id: usize, cost: f64) {
        self.mark_operator_runtime_touched(operator_id);
        self.operator_runtime[operator_id].h_max_supporter_cost = cost;
    }

    #[inline(always)]
    fn operator_cost_1(&self, operator_id: usize) -> f64 {
        self.operator_runtime[operator_id].cost_1
    }

    #[inline(always)]
    fn operator_cost_2(&self, operator_id: usize) -> f64 {
        self.operator_runtime[operator_id].cost_2
    }

    #[inline(always)]
    fn operator_h_max_supporter(&self, operator_id: usize) -> Option<usize> {
        self.operator_runtime[operator_id].h_max_supporter
    }

    #[inline(always)]
    fn operator_h_max_supporter_cost(&self, operator_id: usize) -> f64 {
        self.operator_runtime[operator_id].h_max_supporter_cost
    }

    #[inline(always)]
    fn cached_composite_expression_value(
        &mut self,
        operator_id: usize,
        condition_id: usize,
        numeric_values: &[f64],
    ) -> f64 {
        let Some(expression) = self.operator_condition_eval[operator_id][condition_id]
            .composite_expression
            .as_ref()
        else {
            return 0.0;
        };
        if self.composite_expression_value_marks[operator_id][condition_id]
            != self.composite_expression_value_epoch
        {
            self.composite_expression_value_marks[operator_id][condition_id] =
                self.composite_expression_value_epoch;
            self.composite_expression_values[operator_id][condition_id] =
                expression.evaluate(numeric_values);
        }
        self.composite_expression_values[operator_id][condition_id]
    }

    fn update_original_operator_min_cut_cost(&mut self, original_id: usize, cut_cost: f64) {
        if self.original_operator_min_cut_cost_marks[original_id]
            != self.original_operator_min_cut_cost_epoch
        {
            self.original_operator_min_cut_cost_marks[original_id] =
                self.original_operator_min_cut_cost_epoch;
            self.original_operator_min_cut_costs[original_id] = cut_cost;
            self.touched_original_operator_ids.push(original_id);
        } else {
            self.original_operator_min_cut_costs[original_id] =
                self.original_operator_min_cut_costs[original_id].min(cut_cost);
        }
    }

    fn record_original_operator_multiplier(&mut self, original_id: usize, multiplier: f64) {
        if self.original_operator_multiplier_marks[original_id]
            != self.original_operator_multiplier_epoch
        {
            self.original_operator_multiplier_marks[original_id] =
                self.original_operator_multiplier_epoch;
            self.landmark_original_operator_ids.push(original_id);
        }
        self.original_operator_multipliers[original_id] = multiplier;
    }

    fn build_propositional_propositions(&mut self) {
        for variable_id in 0..self.num_variables {
            let domain_size = self
                .task
                .get_variable_domain_size(variable_id)
                .expect("variable id must be valid");
            self.proposition_index[variable_id].reserve(domain_size);
            for value in 0..domain_size {
                let helper_proposition_id = self
                    .numeric_helper
                    .get_proposition(variable_id, value)
                    .expect("helper proposition id must exist");
                let proposition_id = self.propositions.len();
                let proposition = RelaxedProposition::new(
                    proposition_id,
                    self.numeric_helper
                        .get_proposition_name(helper_proposition_id)
                        .unwrap_or("")
                        .to_string(),
                );
                self.propositions.push(proposition);
                self.proposition_index[variable_id].push(proposition_id);
                self.num_propositions += 1;
            }
        }
    }

    fn build_numeric_conditions(&mut self) {
        if self.config.ignore_numeric {
            return;
        }

        for condition_id in 0..self.numeric_helper.get_n_numeric_conditions() {
            let condition = self
                .numeric_helper
                .get_condition(condition_id)
                .cloned()
                .expect("helper numeric condition must exist");
            let epsilon = self.numeric_helper.get_epsilon(condition_id).unwrap_or(
                if condition.is_strictly_greater {
                    self.config.epsilon
                } else {
                    0.0
                },
            );
            self.add_numeric_condition_proposition_with_epsilon(condition, epsilon);
        }
    }

    fn build_residual_variant_conditions(&mut self) {
        self.residual_variant_precondition_ids.clear();
        let Some(partitions) = self.residual_operator_cost_partitions.clone() else {
            return;
        };
        self.residual_variant_precondition_ids = Vec::with_capacity(partitions.len());
        for (operator_id, partition) in partitions.iter().enumerate() {
            let mut operator_variants = Vec::with_capacity(partition.variants.len());
            for (variant_id, variant) in partition.variants.iter().enumerate() {
                operator_variants.push(self.residual_region_precondition_ids(
                    operator_id,
                    variant_id,
                    &variant.source_region,
                ));
            }
            self.residual_variant_precondition_ids.push(operator_variants);
        }
    }

    fn residual_region_precondition_ids(
        &mut self,
        operator_id: usize,
        variant_id: usize,
        source_region: &StateRegion,
    ) -> Vec<usize> {
        let mut ids = Vec::new();
        let mut seen = BTreeSet::new();
        for (var, values) in source_region.propositions.iter().enumerate() {
            if values.len() == 1 {
                let fact = ExplicitFact::new(var, values[0]);
                for proposition_id in self.precondition_proposition_ids(&fact) {
                    if seen.insert(proposition_id) {
                        ids.push(proposition_id);
                    }
                }
            }
        }
        let num_numeric_vars = self.task.numeric_variables().len();
        for (numeric_var_id, interval) in source_region.numeric.iter().enumerate() {
            if interval.lower.is_finite() {
                let mut expression = LinearExpression::zero(num_numeric_vars);
                expression.coefficients[numeric_var_id] = 1.0;
                expression.constant = -interval.lower;
                let proposition_id =
                    self.add_numeric_condition_proposition(NumericCondition::from_expression(
                        expression,
                        !interval.lower_closed,
                        format!(
                            "fillSCP residual op {operator_id} variant {variant_id} n{numeric_var_id} lower"
                        ),
                    ));
                if seen.insert(proposition_id) {
                    ids.push(proposition_id);
                }
            }
            if interval.upper.is_finite() {
                let mut expression = LinearExpression::zero(num_numeric_vars);
                expression.coefficients[numeric_var_id] = -1.0;
                expression.constant = interval.upper;
                let proposition_id =
                    self.add_numeric_condition_proposition(NumericCondition::from_expression(
                        expression,
                        !interval.upper_closed,
                        format!(
                            "fillSCP residual op {operator_id} variant {variant_id} n{numeric_var_id} upper"
                        ),
                    ));
                if seen.insert(proposition_id) {
                    ids.push(proposition_id);
                }
            }
        }
        ids
    }

    fn build_relaxed_operators(&mut self) {
        let operators = self.task.get_operators();
        for (operator_id, operator) in operators.iter().enumerate() {
            let base_cost = self.calculate_base_operator_cost(operator_id, operator);
            self.build_relaxed_operator_for_operator(operator_id, operator, base_cost)
                .expect("LM-cut numeric operator construction must succeed");
        }

        for (axiom_offset, axiom) in self.task.axioms().iter().enumerate() {
            let operator_id = operators.len() + axiom_offset;
            self.build_relaxed_operator_for_axiom(operator_id, axiom);
        }

        self.build_supported_sose_operators()
            .expect("LM-cut supported SOSE operator construction must succeed");
        self.prune_infinite_effects_for_supported_sose();

        self.build_simple_effects()
            .expect("LM-cut simple-effect construction must succeed");

        self.delete_noops();
    }

    fn delete_noops(&mut self) {
        self.relaxed_operators
            .retain(|operator| !operator.effect_ids.is_empty());
    }

    fn prune_infinite_effects_for_supported_sose(&mut self) {
        let operator_count = self.task.get_operators().len();
        for relaxed_operator in &mut self.relaxed_operators {
            if !relaxed_operator.infinite {
                continue;
            }

            let Some(original_op_id) = relaxed_operator
                .original_op_id_2
                .filter(|&operator_id| operator_id < operator_count)
            else {
                continue;
            };

            relaxed_operator.effect_ids.retain(|&effect_id| {
                let Some(condition_id) = self
                    .propositions
                    .get(effect_id)
                    .and_then(|proposition| proposition.id_numeric_condition)
                else {
                    return true;
                };

                !self.operator_condition_eval[original_op_id][condition_id].has_sose
            });
        }
    }

    fn build_goal_operator(&mut self) {
        let mut goal_preconditions = self.goal_precondition_ids.clone();
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

    fn prepare_goal_preconditions(&mut self) {
        let mut goal_preconditions = Vec::new();
        let mut seen = BTreeSet::new();
        for goal_index in 0..self.task.get_num_goals() {
            let goal = self.task.get_goal_fact(goal_index);
            let helper_propositional_ids = self
                .numeric_helper
                .get_propositional_goals(goal_index)
                .map(|goals| self.build_precondition_ids(goals))
                .unwrap_or_default();
            let helper_numeric_condition_ids = self.numeric_helper.get_numeric_goals(goal_index);
            let helper_numeric_ids = helper_numeric_condition_ids
                .iter()
                .map(|&condition_id| {
                    self.get_numeric_proposition_id(condition_id).expect(
                        "helper goal numeric condition must already have a canonical LM-cut proposition",
                    )
                })
                .collect::<Vec<_>>();

            if helper_propositional_ids.is_empty() && helper_numeric_ids.is_empty() {
                if self.is_numeric_axiom_var(goal.var()) {
                    let _ = self.precondition_proposition_ids(goal);
                    continue;
                }
                for proposition_id in self.precondition_proposition_ids(goal) {
                    if seen.insert(proposition_id) {
                        goal_preconditions.push(proposition_id);
                    }
                }
                continue;
            }

            // PARITY(numeric-fd): when a goal is compiled through numeric helper axioms,
            // the C++ goal operator uses the helper propositional/numeric conditions produced by
            // `build_numeric_goals()` rather than also keeping the derived goal fact itself as a
            // precondition. Keeping both over-constrains the goal operator and can create false
            // dead ends.
            // The shared helper now owns the reference-side `fact_to_axiom_map = -2`
            // goal classification, so later state/precondition handling skips the direct fact.

            for proposition_id in helper_propositional_ids
                .iter()
                .chain(helper_numeric_ids.iter())
                .copied()
            {
                if seen.insert(proposition_id) {
                    goal_preconditions.push(proposition_id);
                }
            }
        }
        self.goal_precondition_ids = goal_preconditions;
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

    fn build_packed_runtime_adjacency(&mut self) {
        self.proposition_precondition_of_data.clear();
        self.proposition_precondition_of_ranges.clear();
        self.proposition_precondition_of_ranges
            .reserve(self.propositions.len());
        for proposition in &self.propositions {
            let start = self.proposition_precondition_of_data.len();
            self.proposition_precondition_of_data
                .extend_from_slice(&proposition.precondition_of);
            let end = self.proposition_precondition_of_data.len();
            self.proposition_precondition_of_ranges.push((start, end));
        }

        self.proposition_effect_of_data.clear();
        self.proposition_effect_of_ranges.clear();
        self.proposition_effect_of_ranges
            .reserve(self.propositions.len());
        for proposition in &self.propositions {
            let start = self.proposition_effect_of_data.len();
            self.proposition_effect_of_data
                .extend_from_slice(&proposition.effect_of);
            let end = self.proposition_effect_of_data.len();
            self.proposition_effect_of_ranges.push((start, end));
        }

        self.operator_precondition_id_data.clear();
        self.operator_precondition_id_ranges.clear();
        self.operator_precondition_id_ranges
            .reserve(self.relaxed_operators.len());
        self.operator_effect_id_data.clear();
        self.operator_effect_id_ranges.clear();
        self.operator_effect_id_ranges
            .reserve(self.relaxed_operators.len());
        for operator in &self.relaxed_operators {
            let precondition_start = self.operator_precondition_id_data.len();
            self.operator_precondition_id_data
                .extend_from_slice(&operator.precondition_ids);
            let precondition_end = self.operator_precondition_id_data.len();
            self.operator_precondition_id_ranges
                .push((precondition_start, precondition_end));

            let effect_start = self.operator_effect_id_data.len();
            self.operator_effect_id_data
                .extend_from_slice(&operator.effect_ids);
            let effect_end = self.operator_effect_id_data.len();
            self.operator_effect_id_ranges
                .push((effect_start, effect_end));
        }
    }

    #[inline(always)]
    fn proposition_precondition_of_range(&self, proposition_id: usize) -> (usize, usize) {
        self.proposition_precondition_of_ranges[proposition_id]
    }

    #[inline(always)]
    fn proposition_effect_of_range(&self, proposition_id: usize) -> (usize, usize) {
        self.proposition_effect_of_ranges[proposition_id]
    }

    #[inline(always)]
    fn operator_precondition_id_range(&self, operator_id: usize) -> (usize, usize) {
        self.operator_precondition_id_ranges[operator_id]
    }

    #[inline(always)]
    fn operator_effect_id_range(&self, operator_id: usize) -> (usize, usize) {
        self.operator_effect_id_ranges[operator_id]
    }

    fn build_original_to_relaxed_index(&mut self) {
        let operator_count = self.task.get_operators().len() + self.task.axioms().len();
        self.original_to_relaxed_operators = vec![Vec::new(); operator_count];
        for (relaxed_operator_id, operator) in self.relaxed_operators.iter().enumerate() {
            if let Some(original_id) = operator.original_op_id_1
                && let Some(mapped) = self.original_to_relaxed_operators.get_mut(original_id)
            {
                mapped.push(relaxed_operator_id);
            }
            if let Some(original_id) = operator.original_op_id_2
                && let Some(mapped) = self.original_to_relaxed_operators.get_mut(original_id)
            {
                mapped.push(relaxed_operator_id);
            }
        }
    }

    fn setup_exploration_queue(&mut self) {
        self.priority_queue.clear();
        for &proposition_id in &self.touched_proposition_ids {
            let runtime = &mut self.proposition_runtime[proposition_id];
            runtime.status = PropositionStatus::Unreached;
            runtime.h_max_cost = f64::INFINITY;
        }

        for &operator_id in &self.touched_operator_ids {
            self.operator_runtime[operator_id] =
                OperatorRuntime::new(&self.relaxed_operators[operator_id]);
        }

        self.start_exploration_runtime_tracking();
    }

    fn setup_exploration_queue_state(
        &mut self,
        propositional_values: &[usize],
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
            if self.is_numeric_axiom_var(variable_id) && !self.config.ignore_numeric {
                continue;
            }
            let fact = ExplicitFact::new(variable_id, value);
            let proposition_id = self.get_proposition_id(&fact);
            self.enqueue_if_necessary(proposition_id, 0.0);
        }

        if !self.config.ignore_numeric {
            for condition_id in 0..self.conditions.len() {
                let slack = self.evaluate_numeric_condition(condition_id, numeric_values)?;
                self.numeric_initial_state[condition_id] = -slack;
                if slack > -self.config.precision {
                    let proposition_id = self.get_numeric_proposition_id(condition_id)?;
                    self.enqueue_if_necessary(proposition_id, 0.0);
                }
            }
        }

        self.enqueue_if_necessary(self.artificial_precondition_id, 0.0);
        Ok(())
    }

    fn first_exploration(
        &mut self,
        propositional_values: &[usize],
        numeric_values: &[f64],
    ) -> Result<(), String> {
        assert!(
            self.priority_queue.is_empty(),
            "LM-cut first exploration requires an empty queue"
        );
        self.setup_exploration_queue();
        self.setup_exploration_queue_state(propositional_values, numeric_values)?;

        while let Some(entry) = self.priority_queue.pop() {
            let popped_cost = entry.cost;
            let proposition_id = entry.proposition_id;
            let proposition_cost = self.proposition_h_max_cost(proposition_id);
            assert!(
                proposition_cost <= popped_cost,
                "LM-cut queue popped a cost smaller than the proposition h_max"
            );
            if proposition_cost < popped_cost {
                continue;
            }

            let (triggered_start, triggered_end) =
                self.proposition_precondition_of_range(proposition_id);
            for triggered_index in triggered_start..triggered_end {
                let operator_id = self.proposition_precondition_of_data[triggered_index];
                let effect_count = {
                    assert!(
                        self.operator_unsatisfied_preconditions(operator_id) > 0,
                        "LM-cut operator precondition counter underflow"
                    );
                    let remaining = self.decrement_operator_unsatisfied_preconditions(operator_id);

                    if remaining == 0 {
                        self.set_operator_h_max_supporter(operator_id, Some(proposition_id));
                        self.set_operator_h_max_supporter_cost(operator_id, proposition_cost);
                        self.relaxed_operators[operator_id].effect_ids.len()
                    } else {
                        0
                    }
                };

                if effect_count > 0 {
                    let (effect_start, _) = self.operator_effect_id_range(operator_id);
                    for effect_index in effect_start..(effect_start + effect_count) {
                        let effect_id = self.operator_effect_id_data[effect_index];
                        self.update_queue(
                            propositional_values,
                            numeric_values,
                            operator_id,
                            proposition_id,
                            effect_id,
                        );
                    }
                }
            }
        }

        Ok(())
    }

    fn first_exploration_incremental(
        &mut self,
        propositional_values: &[usize],
        numeric_values: &[f64],
        cut: &[usize],
    ) -> Result<(), String> {
        assert!(
            self.priority_queue.is_empty(),
            "LM-cut incremental exploration requires an empty queue"
        );
        let mut original_ids_to_update =
            std::mem::take(&mut self.incremental_original_operator_ids_scratch);
        original_ids_to_update.clear();
        Self::advance_epoch(
            &mut self.incremental_original_operator_epoch,
            &mut self.incremental_original_operator_marks,
        );
        let result = (|| {
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
                    let mark = self
                        .incremental_original_operator_marks
                        .get_mut(original_id)
                        .ok_or_else(|| {
                            format!("LM-cut original operator id {original_id} is invalid")
                        })?;
                    if *mark != self.incremental_original_operator_epoch {
                        *mark = self.incremental_original_operator_epoch;
                        original_ids_to_update.push(original_id);
                    }
                }
            }

            for &original_id in &original_ids_to_update {
                let mapped_operator_count = self
                    .original_to_relaxed_operators
                    .get(original_id)
                    .ok_or_else(|| format!("LM-cut original operator id {original_id} is invalid"))?
                    .len();
                for mapped_index in 0..mapped_operator_count {
                    let mapped_operator_id =
                        self.original_to_relaxed_operators[original_id][mapped_index];
                    let operator = self.relaxed_operators.get(mapped_operator_id).ok_or_else(
                        || {
                            format!(
                                "LM-cut mapped relaxed operator id {mapped_operator_id} is invalid"
                            )
                        },
                    )?;
                    if self.operator_unsatisfied_preconditions(mapped_operator_id) == 0 {
                        let supporter_id = self
                            .operator_h_max_supporter(mapped_operator_id)
                            .ok_or_else(|| {
                                format!(
                                    "LM-cut reachable operator {} must have an h_max supporter",
                                    operator.name
                                )
                            })?;
                        let (effect_start, effect_end) =
                            self.operator_effect_id_range(mapped_operator_id);
                        for effect_index in effect_start..effect_end {
                            let effect_id = self.operator_effect_id_data[effect_index];
                            self.update_queue(
                                propositional_values,
                                numeric_values,
                                mapped_operator_id,
                                supporter_id,
                                effect_id,
                            );
                        }
                    }
                }
            }

            while let Some(entry) = self.priority_queue.pop() {
                let popped_cost = entry.cost;
                let proposition_id = entry.proposition_id;
                let proposition_cost = self.proposition_h_max_cost(proposition_id);
                assert!(
                    proposition_cost <= popped_cost,
                    "LM-cut incremental queue popped a cost smaller than the proposition h_max"
                );
                if proposition_cost < popped_cost {
                    continue;
                }

                let (triggered_start, triggered_end) =
                    self.proposition_precondition_of_range(proposition_id);
                for triggered_index in triggered_start..triggered_end {
                    let operator_id = self.proposition_precondition_of_data[triggered_index];
                    let update = {
                        if self.operator_h_max_supporter(operator_id) == Some(proposition_id) {
                            let old_supporter_cost =
                                self.operator_h_max_supporter_cost(operator_id);
                            if old_supporter_cost > proposition_cost {
                                let new_supporter = self.update_h_max_supporter(operator_id);
                                if let Some((new_supporter_id, new_cost)) = new_supporter {
                                    Some((
                                        new_supporter_id,
                                        new_cost,
                                        new_cost != old_supporter_cost,
                                    ))
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

                    if let Some((new_supporter_id, new_cost, needs_effect_update)) = update {
                        self.set_operator_h_max_supporter(operator_id, Some(new_supporter_id));
                        self.set_operator_h_max_supporter_cost(operator_id, new_cost);
                        if needs_effect_update {
                            let (effect_start, effect_end) =
                                self.operator_effect_id_range(operator_id);
                            for effect_index in effect_start..effect_end {
                                let effect_id = self.operator_effect_id_data[effect_index];
                                self.update_queue(
                                    propositional_values,
                                    numeric_values,
                                    operator_id,
                                    new_supporter_id,
                                    effect_id,
                                );
                            }
                        }
                    }
                }
            }

            Ok(())
        })();
        original_ids_to_update.clear();
        self.incremental_original_operator_ids_scratch = original_ids_to_update;
        result
    }

    #[inline(always)]
    fn update_h_max_supporter(&mut self, operator_id: usize) -> Option<(usize, f64)> {
        debug_assert!(operator_id < self.relaxed_operators.len());
        if self.operator_unsatisfied_preconditions(operator_id) != 0 {
            return None;
        }

        let mut best_supporter = self.operator_h_max_supporter(operator_id);
        let mut best_cost = best_supporter
            .map(|supporter_id| self.proposition_h_max_cost(supporter_id))
            .unwrap_or(f64::NEG_INFINITY);

        if let Some(supporter_id) = best_supporter
            && self.proposition_status(supporter_id) == PropositionStatus::Unreached
        {
            return None;
        }

        let (precondition_start, precondition_end) =
            self.operator_precondition_id_range(operator_id);
        for precondition_index in precondition_start..precondition_end {
            let precondition_id = self.operator_precondition_id_data[precondition_index];
            if self.proposition_status(precondition_id) == PropositionStatus::Unreached {
                return None;
            }
            let precondition_cost = self.proposition_h_max_cost(precondition_id);
            if best_supporter.is_none() || precondition_cost > best_cost {
                best_supporter = Some(precondition_id);
                best_cost = precondition_cost;
            }
        }

        if let Some(supporter_id) = best_supporter {
            self.set_operator_h_max_supporter(operator_id, Some(supporter_id));
            self.set_operator_h_max_supporter_cost(operator_id, best_cost);
            Some((supporter_id, best_cost))
        } else {
            None
        }
    }

    fn mark_goal_plateau(
        &mut self,
        propositional_values: &[usize],
        numeric_values: &[f64],
        proposition_id: usize,
    ) {
        if self.proposition_status(proposition_id) == PropositionStatus::GoalZone {
            return;
        }

        self.set_proposition_status(proposition_id, PropositionStatus::GoalZone);
        let (achiever_start, achiever_end) = self.proposition_effect_of_range(proposition_id);
        for achiever_index in achiever_start..achiever_end {
            let achiever_id = self.proposition_effect_of_data[achiever_index];
            let (is_zero_cost_applicable, achiever_supporter) = {
                let _achiever = self
                    .relaxed_operators
                    .get(achiever_id)
                    .unwrap_or_else(|| panic!("LM-cut achiever id {achiever_id} is invalid"));
                (
                    self.operator_cost_1(achiever_id) < self.config.precision
                        && self.operator_cost_2(achiever_id) < self.config.precision
                        && self.operator_unsatisfied_preconditions(achiever_id) == 0,
                    self.operator_h_max_supporter(achiever_id),
                )
            };
            let recurse_to = if is_zero_cost_applicable {
                let ms = self.calculate_numeric_times(
                    propositional_values,
                    numeric_values,
                    proposition_id,
                    achiever_id,
                    !self.config.disable_ma,
                );
                if self.multiplier_allows_traversal(achiever_id, ms) {
                    achiever_supporter
                } else {
                    None
                }
            } else {
                None
            };
            if let Some(supporter_id) = recurse_to {
                self.mark_goal_plateau(propositional_values, numeric_values, supporter_id);
            }
        }
    }

    fn second_exploration(
        &mut self,
        propositional_values: &[usize],
        numeric_values: &[f64],
        cut: &mut Vec<usize>,
        m_list: &mut Vec<(f64, f64)>,
    ) {
        assert!(
            cut.is_empty(),
            "LM-cut second exploration requires empty cut"
        );
        assert!(
            m_list.is_empty(),
            "LM-cut second exploration requires empty multiplier list"
        );
        self.start_cut_marking();
        let mut queue = std::mem::take(&mut self.second_exploration_queue_scratch);
        queue.clear();
        self.set_proposition_status(
            self.artificial_precondition_id,
            PropositionStatus::BeforeGoalZone,
        );
        queue.push(self.artificial_precondition_id);

        for (variable_id, &value) in propositional_values.iter().enumerate() {
            if self.is_numeric_axiom_var(variable_id) && !self.config.ignore_numeric {
                continue;
            }
            let fact = ExplicitFact::new(variable_id, value);
            let proposition_id = self.get_proposition_id(&fact);
            if self.proposition_status(proposition_id) != PropositionStatus::BeforeGoalZone {
                self.set_proposition_status(proposition_id, PropositionStatus::BeforeGoalZone);
                queue.push(proposition_id);
            }
        }

        if !self.config.ignore_numeric {
            for condition_id in 0..self.conditions.len() {
                if self.numeric_initial_state[condition_id] < self.config.precision {
                    let proposition_id = self.get_numeric_proposition_id_infallible(condition_id);
                    if self.proposition_status(proposition_id) != PropositionStatus::BeforeGoalZone
                    {
                        self.set_proposition_status(
                            proposition_id,
                            PropositionStatus::BeforeGoalZone,
                        );
                        queue.push(proposition_id);
                    }
                }
            }
        }

        while let Some(proposition_id) = queue.pop() {
            let (triggered_start, triggered_end) =
                self.proposition_precondition_of_range(proposition_id);
            for triggered_index in triggered_start..triggered_end {
                let operator_id = self.proposition_precondition_of_data[triggered_index];
                let should_process = {
                    self.operator_h_max_supporter(operator_id) == Some(proposition_id)
                        && !self.is_cut_marked(operator_id)
                };
                if !should_process {
                    continue;
                }

                let (effect_start, effect_end) = self.operator_effect_id_range(operator_id);
                let mut min_cut_cost = f64::INFINITY;

                for effect_index in effect_start..effect_end {
                    let effect_id = self.operator_effect_id_data[effect_index];
                    let effect_status = self.proposition_status(effect_id);
                    if effect_status == PropositionStatus::GoalZone {
                        let ms = self.calculate_numeric_times(
                            propositional_values,
                            numeric_values,
                            effect_id,
                            operator_id,
                            !self.config.disable_ma,
                        );
                        let operator = &self.relaxed_operators[operator_id];
                        if (operator.original_op_id_1.is_some() && ms.0 >= self.config.precision)
                            || (operator.original_op_id_1.is_none()
                                && ms.1 >= self.config.precision)
                        {
                            let edge_cost = self.edge_cost(operator_id, ms);
                            cut.push(operator_id);
                            m_list.push(ms);
                            self.mark_cut(operator_id);
                            min_cut_cost = min_cut_cost.min(edge_cost);
                        }
                    }
                }

                for effect_index in effect_start..effect_end {
                    let effect_id = self.operator_effect_id_data[effect_index];
                    let effect_status = self.proposition_status(effect_id);
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
                        !self.config.disable_ma,
                    );
                    let operator = &self.relaxed_operators[operator_id];
                    if (operator.original_op_id_1.is_some() && ms.0 >= self.config.precision)
                        || (operator.original_op_id_1.is_none() && ms.1 >= self.config.precision)
                    {
                        let edge_cost = self.edge_cost(operator_id, ms);
                        if edge_cost < min_cut_cost {
                            assert_eq!(effect_status, PropositionStatus::Reached);
                            self.set_proposition_status(
                                effect_id,
                                PropositionStatus::BeforeGoalZone,
                            );
                            queue.push(effect_id);
                        }
                    }
                }
            }
        }
        queue.clear();
        self.second_exploration_queue_scratch = queue;
    }

    #[inline(always)]
    fn calculate_numeric_times_sose(
        &mut self,
        numeric_values: &[f64],
        condition_id: usize,
        operator_id: usize,
        operator_runtime: OperatorRuntime,
        original_op_id_1: usize,
        original_op_id_2: usize,
    ) -> (f64, f64) {
        let eval_row = &self.operator_condition_eval[original_op_id_2][condition_id];
        let c = eval_row.simple_effect.unwrap_or(0.0);
        let has_upper_bound = eval_row.has_upper_bound;
        let upper_bound = eval_row.upper_bound;
        let composite_coefficients = eval_row
            .composite_expression
            .as_ref()
            .map(|expression| expression.coefficients.as_slice());

        let mut c_u = *self.relaxed_operators[operator_id]
            .sose_constants
            .get(condition_id)
            .expect("LM-cut SOSE operator must store condition constants");
        if self.config.use_constant_assignment {
            c_u += self.calculate_constant_assignment_effect_infallible(
                original_op_id_1,
                composite_coefficients.unwrap_or_else(|| {
                    panic!(
                        "LM-cut SOSE target operator {original_op_id_2} is missing composite coefficients for condition {condition_id}"
                    )
                }),
                numeric_values,
                self.use_bounds,
            );
        }
        if c_u < self.config.precision {
            return (-1.0, -1.0);
        }
        if operator_runtime.cost_1 < self.config.precision {
            return (1.0, 1.0);
        }

        let s_u =
            self.cached_composite_expression_value(original_op_id_2, condition_id, numeric_values);

        if operator_runtime.cost_2 < self.config.precision {
            if (c + s_u).abs() < self.config.precision {
                return (1.0, 1.0);
            }
            if c + s_u > 0.0 {
                return (-1.0, -1.0);
            }
            let mut m_1 = -(c + s_u) / c_u;
            if self.config.ceiling_less_than_one {
                m_1 = m_1.max(1.0);
            }
            return (m_1, 1.0);
        }

        let mut u_target =
            (self.numeric_initial_state[condition_id] * c_u * operator_runtime.cost_2
                / operator_runtime.cost_1)
                .sqrt()
                - c;
        if self.use_bounds && has_upper_bound {
            u_target = u_target.min(upper_bound);
        }
        if u_target - s_u < self.config.precision || c + u_target < self.config.precision {
            return (-1.0, -1.0);
        }

        let mut m_1 = (u_target - s_u) / c_u;
        let mut m_2 = self.numeric_initial_state[condition_id] / (c + u_target);
        if self.config.ceiling_less_than_one {
            m_1 = m_1.max(1.0);
            m_2 = m_2.max(1.0);
        }
        (m_1, m_2)
    }

    #[inline(always)]
    fn calculate_numeric_times_simple_numeric(
        &mut self,
        numeric_values: &[f64],
        condition_id: usize,
        original_op_id_2: Option<usize>,
    ) -> (f64, f64) {
        let mut net = 0.0;
        if let Some(original_id) = original_op_id_2 {
            let eval_row = &self.operator_condition_eval[original_id][condition_id];
            net += eval_row.simple_effect.unwrap_or(0.0);
            if eval_row.composite_expression.is_some() {
                net += self.cached_composite_expression_value(
                    original_id,
                    condition_id,
                    numeric_values,
                );
            }
        }
        if self.config.use_constant_assignment {
            let original_operator_id = original_op_id_2
                .expect("LM-cut relaxed operator must store its concrete operator id");
            let has_supported_sose =
                self.operator_condition_eval[original_operator_id][condition_id].has_sose;
            net += self.calculate_constant_assignment_effect_infallible(
                original_operator_id,
                &self.conditions[condition_id].coefficients,
                numeric_values,
                self.use_bounds && !has_supported_sose,
            );
        }
        if net < self.config.precision {
            return (-1.0, -1.0);
        }

        let mut m = self.numeric_initial_state[condition_id] / net;
        if m < self.config.precision {
            return (0.0, 0.0);
        }
        if self.config.ceiling_less_than_one {
            m = m.max(1.0);
        }
        (0.0, m)
    }

    #[inline(never)]
    fn calculate_numeric_times(
        &mut self,
        _propositional_values: &[usize],
        numeric_values: &[f64],
        effect_id: usize,
        operator_id: usize,
        use_ma: bool,
    ) -> (f64, f64) {
        debug_assert!(effect_id < self.propositions.len());
        debug_assert!(operator_id < self.relaxed_operators.len());
        let effect = &self.propositions[effect_id];
        let operator_runtime = self.operator_runtime[operator_id];
        let operator = &self.relaxed_operators[operator_id];
        if !use_ma || !effect.is_numeric_condition || operator.infinite {
            return (0.0, 1.0);
        }

        let condition_id = effect
            .id_numeric_condition
            .expect("LM-cut numeric proposition must store its condition id");
        match (operator.original_op_id_1, operator.original_op_id_2) {
            (Some(original_op_id_1), Some(original_op_id_2)) => self.calculate_numeric_times_sose(
                numeric_values,
                condition_id,
                operator_id,
                operator_runtime,
                original_op_id_1,
                original_op_id_2,
            ),
            (_, original_op_id_2) => self.calculate_numeric_times_simple_numeric(
                numeric_values,
                condition_id,
                original_op_id_2,
            ),
        }
    }

    #[inline(always)]
    fn multiplier_allows_traversal(&self, operator_id: usize, ms: (f64, f64)) -> bool {
        let operator = &self.relaxed_operators[operator_id];
        (operator.original_op_id_1.is_some() && ms.0 >= self.config.precision)
            || ms.1 >= self.config.precision
    }

    fn edge_cost(&self, operator_id: usize, ms: (f64, f64)) -> f64 {
        let operator = &self.relaxed_operators[operator_id];
        let mut edge_cost = ms.1 * self.operator_cost_2(operator_id);
        if operator.original_op_id_1.is_some() {
            edge_cost += ms.0 * self.operator_cost_1(operator_id);
        }
        edge_cost
    }

    fn reset_goal_zone_statuses(&mut self) {
        for &proposition_id in &self.goal_zone_proposition_ids {
            if self.proposition_status(proposition_id) == PropositionStatus::GoalZone
                || self.proposition_status(proposition_id) == PropositionStatus::BeforeGoalZone
            {
                self.proposition_runtime[proposition_id].status = PropositionStatus::Reached;
            }
            self.proposition_runtime[proposition_id].goal_zone_touched = false;
        }
        self.goal_zone_proposition_ids.clear();
    }

    #[inline(always)]
    fn update_queue(
        &mut self,
        propositional_values: &[usize],
        numeric_values: &[f64],
        operator_id: usize,
        supporter_id: usize,
        effect_id: usize,
    ) {
        debug_assert!(effect_id < self.propositions.len());
        debug_assert!(operator_id < self.relaxed_operators.len());
        debug_assert!(supporter_id < self.propositions.len());
        let effect = &self.propositions[effect_id];
        if effect.is_numeric_condition {
            let condition_id = effect
                .id_numeric_condition
                .expect("LM-cut numeric proposition must store its condition id");
            if self.numeric_initial_state[condition_id] < self.config.precision {
                return;
            }
            let ms = self.calculate_numeric_times(
                propositional_values,
                numeric_values,
                effect_id,
                operator_id,
                !self.config.irmax,
            );
            let operator = &self.relaxed_operators[operator_id];
            if operator.original_op_id_1.is_some() {
                if ms.0 >= self.config.precision {
                    let target_cost = self.proposition_h_max_cost(supporter_id)
                        + (ms.0 * self.operator_cost_1(operator_id))
                        + (ms.1 * self.operator_cost_2(operator_id));
                    self.enqueue_if_necessary(effect_id, target_cost);
                }
            } else if ms.1 >= self.config.precision {
                let target_cost = self.proposition_h_max_cost(supporter_id)
                    + (ms.1 * self.operator_cost_2(operator_id));
                self.enqueue_if_necessary(effect_id, target_cost);
            }
            return;
        }
        let target_cost =
            self.proposition_h_max_cost(supporter_id) + self.operator_cost_2(operator_id);
        self.enqueue_if_necessary(effect_id, target_cost);
    }

    #[inline(always)]
    fn enqueue_if_necessary(&mut self, proposition_id: usize, cost: f64) -> bool {
        assert!(cost >= 0.0, "LM-cut enqueue cost must be non-negative");
        assert!(!cost.is_nan(), "LM-cut enqueue cost must not be NaN");
        debug_assert!(proposition_id < self.propositions.len());
        // PARITY(numeric-fd): C++ uses the strict comparison `h_max_cost > cost` here.
        // A `+ precision` tolerance suppresses small but real h_max decreases during
        // `first_exploration_incremental()`, which can keep `goal_h_max` artificially high
        // after zero-cost cuts and surface as false dead-end reports.
        if self.proposition_status(proposition_id) == PropositionStatus::Unreached
            || self.proposition_h_max_cost(proposition_id) > cost
        {
            self.set_proposition_status(proposition_id, PropositionStatus::Reached);
            self.set_proposition_h_max_cost(proposition_id, cost);
            self.priority_queue.push(QueueEntry {
                cost,
                proposition_id,
            });
            return true;
        }
        false
    }

    fn calculate_base_operator_cost(&self, operator_id: usize, operator: &Operator) -> f64 {
        assert!(
            operator_id < self.task.get_operators().len(),
            "base operator cost is only defined for concrete operators"
        );
        if let Some(costs) = &self.fixed_operator_costs {
            return costs.get(operator_id).copied().unwrap_or(0.0).max(0.0);
        }
        if let Some(partitions) = &self.residual_operator_cost_partitions {
            return partitions
                .get(operator_id)
                .map(|partition| partition.fallback_cost)
                .unwrap_or(0.0)
                .max(0.0);
        }
        let mut operator_cost = metric_operator_cost_from_initial_values(self.task, operator);

        if self.task.is_linear_cost_operator(operator_id) && self.use_bounds {
            let coefficients = self.task.operator_cost_coefficients(operator_id);
            operator_cost = self.task.operator_cost_constant(operator_id);

            for (numeric_var_id, &weight) in coefficients.iter().enumerate() {
                if weight >= self.config.precision
                    && self
                        .numeric_bound
                        .get_variable_before_action_has_lb(numeric_var_id, operator_id)
                {
                    operator_cost += weight
                        * self
                            .numeric_bound
                            .get_variable_before_action_lb(numeric_var_id, operator_id);
                } else if weight <= -self.config.precision
                    && self
                        .numeric_bound
                        .get_variable_before_action_has_ub(numeric_var_id, operator_id)
                {
                    operator_cost += weight
                        * self
                            .numeric_bound
                            .get_variable_before_action_ub(numeric_var_id, operator_id);
                } else if weight.abs() >= self.config.precision {
                    operator_cost = 0.0;
                    break;
                }
            }
        }

        operator_cost.max(0.0)
    }

    fn build_relaxed_operator_for_operator(
        &mut self,
        operator_id: usize,
        operator: &Operator,
        base_cost: f64,
    ) -> Result<(), String> {
        let helper_linearized_assignment_effects = self
            .numeric_helper
            .linearized_effects_for_action(operator_id, operator.assignment_effects().len())?;
        let helper_conditional_fact_effects = self
            .numeric_helper
            .get_action_conditional_fact_effects(operator_id)
            .map(|effects| effects.to_vec())
            .ok_or_else(|| {
                format!("LM-cut helper conditional fact effects {operator_id} are missing")
            })?;
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(operator_id)
            .map(|effects| effects.to_vec())
            .ok_or_else(|| format!("LM-cut helper linear effects {operator_id} are missing"))?;
        let helper_pre_list = self
            .numeric_helper
            .get_action_pre_list(operator_id)
            .expect("helper action pre-list must exist for operator");
        let helper_num_list = self
            .numeric_helper
            .get_action_num_list(operator_id)
            .map(|ids| ids.to_vec())
            .expect("helper action numeric pre-list must exist for operator");
        let precondition_groups = self.precondition_proposition_id_groups(helper_pre_list);
        let mut precondition_ids = self.flatten_precondition_groups(&precondition_groups);
        self.append_numeric_condition_propositions(&helper_num_list, &mut precondition_ids);
        let unconditional_linear_assignment_effect_ids = helper_linear_effects
            .iter()
            .filter(|effect| {
                effect.preconditions.propositional_facts.is_empty()
                    && effect.preconditions.numeric_group_ids.is_empty()
            })
            .map(|effect| effect.source_assignment_effect_id)
            .collect::<Vec<_>>();

        for conditional_effect in &helper_conditional_fact_effects {
            if self.is_numeric_axiom_var(conditional_effect.add_fact.var()) {
                continue;
            }
            let mut extended_preconditions = precondition_ids.clone();
            let mut seen: BTreeSet<usize> = extended_preconditions.iter().copied().collect();
            let effect_condition_groups = self.precondition_proposition_id_groups(
                &conditional_effect.preconditions.propositional_facts,
            );
            for group in &effect_condition_groups {
                for &proposition_id in group {
                    if seen.insert(proposition_id) {
                        extended_preconditions.push(proposition_id);
                    }
                }
            }
            self.append_numeric_condition_propositions(
                &conditional_effect.preconditions.numeric_group_ids,
                &mut extended_preconditions,
            );
            let conditional_name = format!(
                "{} {}",
                operator.name(),
                self.get_proposition_name(
                    conditional_effect.add_fact.var(),
                    conditional_effect.add_fact.value()
                )
            );
            let conditional_operator = RelaxedOperator::new(
                extended_preconditions,
                vec![self.get_proposition_id(&conditional_effect.add_fact)],
                operator_id,
                base_cost,
                conditional_name,
                true,
            );
            conditional_operator.assert_well_formed();
            self.relaxed_operators.push(conditional_operator);
        }

        let base_effect_ids = self
            .numeric_helper
            .get_action_add_list(operator_id)
            .into_iter()
            .flatten()
            .filter(|fact| !self.is_numeric_axiom_var(fact.var()))
            .map(|fact| self.get_proposition_id(fact))
            .collect::<Vec<_>>();

        let mut relaxed_operator = RelaxedOperator::new(
            if precondition_ids.is_empty() {
                vec![self.artificial_precondition_id]
            } else {
                precondition_ids.clone()
            },
            base_effect_ids.clone(),
            operator_id,
            base_cost,
            operator.name().to_string(),
            false,
        );
        relaxed_operator.assignment_effect_ids = unconditional_linear_assignment_effect_ids.clone();
        relaxed_operator.linear_assignment_effects = unconditional_linear_assignment_effect_ids
            .iter()
            .map(|&assignment_effect_id| {
                helper_linearized_assignment_effects
                    .get(assignment_effect_id)
                    .and_then(|effect| effect.clone())
                    .expect("helper linearized assignment effect id must be valid")
            })
            .collect();
        relaxed_operator.assert_well_formed();
        self.relaxed_operators.push(relaxed_operator);

        if let Some(partition) = self
            .residual_operator_cost_partitions
            .as_ref()
            .and_then(|partitions| partitions.get(operator_id))
            .cloned()
        {
            let variant_precondition_ids = self
                .residual_variant_precondition_ids
                .get(operator_id)
                .cloned()
                .unwrap_or_default();
            for (variant_id, variant) in partition.variants.iter().enumerate() {
                let mut guarded_precondition_ids = if precondition_ids.is_empty() {
                    vec![self.artificial_precondition_id]
                } else {
                    precondition_ids.clone()
                };
                let mut seen: BTreeSet<usize> =
                    guarded_precondition_ids.iter().copied().collect();
                if let Some(guards) = variant_precondition_ids.get(variant_id) {
                    for &guard_id in guards {
                        if seen.insert(guard_id) {
                            guarded_precondition_ids.push(guard_id);
                        }
                    }
                }
                let mut guarded_operator = RelaxedOperator::new(
                    guarded_precondition_ids.clone(),
                    base_effect_ids.clone(),
                    operator_id,
                    variant.cost,
                    format!("{} residual {}", operator.name(), variant_id),
                    false,
                );
                guarded_operator.assignment_effect_ids =
                    unconditional_linear_assignment_effect_ids.clone();
                guarded_operator.linear_assignment_effects = unconditional_linear_assignment_effect_ids
                    .iter()
                    .map(|&assignment_effect_id| {
                        helper_linearized_assignment_effects
                            .get(assignment_effect_id)
                            .and_then(|effect| effect.clone())
                            .expect("helper linearized assignment effect id must be valid")
                    })
                    .collect();
                guarded_operator.assert_well_formed();
                self.relaxed_operators.push(guarded_operator);
                self.build_linear_operators(
                    operator_id,
                    operator,
                    variant.cost,
                    &guarded_precondition_ids,
                    &helper_linearized_assignment_effects,
                )?;
            }
        }

        self.build_linear_operators(
            operator_id,
            operator,
            base_cost,
            &precondition_ids,
            &helper_linearized_assignment_effects,
        )?;
        Ok(())
    }

    fn build_simple_effects(&mut self) -> Result<(), String> {
        let operator_count = self.task.get_operators().len();

        for relaxed_operator_id in 0..self.relaxed_operators.len() {
            let original_op_id = {
                let relaxed_operator = &self.relaxed_operators[relaxed_operator_id];
                // PARITY(numeric-fd): keep this aligned with C++ `build_simple_effects()`,
                // which checks only `!conditional && op_id_1 == -1 && op_id_2 < n_actions`.
                // Infinite operators must still be considered here because an action with both
                // a linear effect and a simple/assignment-like effect can legitimately
                // contribute simple effects to its infinite relaxed operator.
                if relaxed_operator.conditional || relaxed_operator.original_op_id_1.is_some() {
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
                let has_supported_sose =
                    self.operator_condition_eval[original_op_id][condition_id].has_sose;
                let (has_simple_effect, simple_effect) = self.calculate_simple_effect_constant(
                    original_op_id,
                    &self.conditions[condition_id].coefficients,
                    self.use_bounds && !has_supported_sose,
                )?;
                let has_constant_assignment_effect = self.config.use_constant_assignment
                    && self.has_constant_assignment_effect(
                        original_op_id,
                        &self.conditions[condition_id].coefficients,
                        self.use_bounds && !has_supported_sose,
                    )?;

                if !has_simple_effect && !has_supported_sose && !has_constant_assignment_effect {
                    continue;
                }

                self.operator_condition_eval[original_op_id][condition_id].simple_effect =
                    Some(simple_effect);
                let proposition_id = self.get_numeric_proposition_id(condition_id)?;
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

    fn simple_effect_constant_for_operator(
        &self,
        relaxed_operator_id: usize,
        condition_id: usize,
    ) -> Result<(bool, f64), String> {
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
        let original_operator_id = relaxed_operator
            .original_op_id_2
            .filter(|&operator_id| operator_id < self.task.get_operators().len())
            .ok_or_else(|| {
                format!(
                    "LM-cut relaxed operator {relaxed_operator_id} is missing its concrete operator id"
                )
            })?;
        self.calculate_simple_effect_constant(
            original_operator_id,
            &condition.coefficients,
            self.use_bounds,
        )
    }

    fn calculate_simple_effect_constant(
        &self,
        original_operator_id: usize,
        coefficients: &[f64],
        use_bounded_linear: bool,
    ) -> Result<(bool, f64), String> {
        let helper_simple_effects = self
            .numeric_helper
            .get_action_eff_list(original_operator_id)
            .ok_or_else(|| {
                format!("LM-cut helper action eff_list {original_operator_id} is missing")
            })?;
        let helper_conditional_numeric_effects = self
            .numeric_helper
            .get_action_conditional_eff_list(original_operator_id)
            .ok_or_else(|| {
                format!(
                    "LM-cut helper conditional numeric effects {original_operator_id} are missing"
                )
            })?;
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(original_operator_id)
            .ok_or_else(|| {
                format!("LM-cut helper linear effects {original_operator_id} are missing")
            })?;
        let mut has_simple_effect = false;
        let mut net = 0.0;

        for (local_var_id, &simple_effect) in helper_simple_effects.iter().enumerate() {
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id).ok_or_else(|| {
                format!(
                    "LM-cut helper simple effect target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            net += weight * simple_effect;
        }

        for conditional_effect in helper_conditional_numeric_effects {
            let actual_var_id = *self.regular_numeric_variable_ids
                .get(conditional_effect.target_local_var_id)
                .ok_or_else(|| {
                    format!(
                        "LM-cut helper conditional numeric effect target {} is invalid for operator {original_operator_id}",
                        conditional_effect.target_local_var_id
                    )
                })?;
            let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            let contribution = weight * conditional_effect.delta;
            if contribution >= self.config.precision {
                net += contribution;
            }
        }

        for linear_effect in helper_linear_effects {
            let local_var_id = linear_effect.target_local_var_id;
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id).ok_or_else(|| {
                format!(
                    "LM-cut helper linear effect target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }

            let conditional = !linear_effect.preconditions.propositional_facts.is_empty()
                || !linear_effect.preconditions.numeric_group_ids.is_empty();
            let mut contribution = weight * linear_effect.constant;

            if use_bounded_linear
                && weight >= self.config.precision
                && self
                    .numeric_bound
                    .get_effect_has_ub(original_operator_id, local_var_id)
                && (!self.config.use_constant_assignment
                    || !self
                        .numeric_bound
                        .get_assignment_has_ub(original_operator_id, local_var_id))
            {
                contribution = weight
                    * self
                        .numeric_bound
                        .get_effect_ub(original_operator_id, local_var_id);
            } else if use_bounded_linear
                && weight <= -self.config.precision
                && self
                    .numeric_bound
                    .get_effect_has_lb(original_operator_id, local_var_id)
                && (!self.config.use_constant_assignment
                    || !self
                        .numeric_bound
                        .get_assignment_has_lb(original_operator_id, local_var_id))
            {
                contribution = weight
                    * self
                        .numeric_bound
                        .get_effect_lb(original_operator_id, local_var_id);
            } else if use_bounded_linear
                && self.config.use_constant_assignment
                && ((weight >= self.config.precision
                    && self
                        .numeric_bound
                        .get_assignment_has_ub(original_operator_id, local_var_id))
                    || (weight <= -self.config.precision
                        && self
                            .numeric_bound
                            .get_assignment_has_lb(original_operator_id, local_var_id)))
            {
                contribution = 0.0;
                has_simple_effect = true;
            }

            if !conditional || contribution >= self.config.precision {
                net += contribution;
            }
        }

        if !has_simple_effect {
            has_simple_effect = net >= self.config.precision;
        }

        Ok((has_simple_effect, net))
    }

    fn build_linear_operators(
        &mut self,
        operator_id: usize,
        operator: &Operator,
        base_cost: f64,
        base_precondition_ids: &[usize],
        linearized_assignment_effects: &[Option<LinearNumericEffect>],
    ) -> Result<(), String> {
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(operator_id)
            .map(|effects| effects.to_vec())
            .ok_or_else(|| format!("LM-cut helper linear effects {operator_id} are missing"))?;
        if self.numeric_helper.get_action_n_linear_eff(operator_id) == 0 {
            return Ok(());
        }
        for helper_linear_effect in helper_linear_effects {
            let assignment_effect_id = helper_linear_effect.source_assignment_effect_id;
            let linear_effect = linearized_assignment_effects
                .get(assignment_effect_id)
                .and_then(|effect| effect.clone())
                .ok_or_else(|| {
                    format!(
                        "LM-cut linearized assignment effect {assignment_effect_id} for operator {operator_id} is missing"
                    )
                })?;
            let helper_effect_preconditions = helper_linear_effect.preconditions.clone();

            let mut precondition_ids = base_precondition_ids.to_vec();
            let mut seen: BTreeSet<usize> = precondition_ids.iter().copied().collect();
            let assignment_condition_groups = self.precondition_proposition_id_groups(
                &helper_effect_preconditions.propositional_facts,
            );
            for group in &assignment_condition_groups {
                for &proposition_id in group {
                    if seen.insert(proposition_id) {
                        precondition_ids.push(proposition_id);
                    }
                }
            }
            self.append_numeric_condition_propositions(
                &helper_effect_preconditions.numeric_group_ids,
                &mut precondition_ids,
            );
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
                if !self.has_linear_effect(
                    operator_id,
                    &condition.coefficients,
                    self.use_bounds,
                    false,
                )? {
                    continue;
                }
                let weight = condition
                    .coefficients
                    .get(linear_effect.affected_var_id)
                    .copied()
                    .unwrap_or(0.0);
                if weight > self.config.precision {
                    plus_effect_ids.push(self.get_numeric_proposition_id(condition_id)?);
                } else if weight < -self.config.precision {
                    minus_effect_ids.push(self.get_numeric_proposition_id(condition_id)?);
                }
            }

            if !plus_effect_ids.is_empty() {
                let mut relaxed_operator = RelaxedOperator::new(
                    {
                        let mut guarded_preconditions = precondition_ids.clone();
                        for &condition_id in &self.linear_effect_to_conditions_plus[operator_id]
                            [assignment_effect_id]
                        {
                            guarded_preconditions
                                .push(self.get_numeric_proposition_id(condition_id)?);
                        }
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
                let mut relaxed_operator = RelaxedOperator::new(
                    {
                        let mut guarded_preconditions = precondition_ids;
                        for &condition_id in &self.linear_effect_to_conditions_minus[operator_id]
                            [assignment_effect_id]
                        {
                            guarded_preconditions
                                .push(self.get_numeric_proposition_id(condition_id)?);
                        }
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
                let proposition_id = self.get_numeric_proposition_id(condition_id)?;
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

    #[allow(clippy::needless_range_loop)]
    fn build_supported_sose_operators(&mut self) -> Result<(), String> {
        let operator_count = self.task.get_operators().len();
        self.operator_condition_eval =
            vec![vec![OperatorConditionEval::default(); self.conditions.len()]; operator_count];
        if !self.config.use_second_order_simple {
            return Ok(());
        }

        let mut base_relaxed_by_original = vec![None; operator_count];
        for (relaxed_operator_id, operator) in self.relaxed_operators.iter().enumerate() {
            if operator.original_op_id_1.is_some() || operator.conditional || operator.infinite {
                continue;
            }
            if let Some(original_id) = operator.original_op_id_2.filter(|&id| id < operator_count) {
                base_relaxed_by_original[original_id] = Some(relaxed_operator_id);
            }
        }

        let mut new_operators = Vec::new();
        for op2_id in 0..operator_count {
            let Some(op2_relaxed_id) = base_relaxed_by_original[op2_id] else {
                continue;
            };

            let mut supporter_to_effects: BTreeMap<usize, Vec<(usize, f64)>> = BTreeMap::new();
            for condition_id in 0..self.conditions.len() {
                let condition = self
                    .conditions
                    .get(condition_id)
                    .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;
                if !self.has_linear_effect(op2_id, &condition.coefficients, false, false)?
                    || self.has_linear_effect(op2_id, &condition.coefficients, false, true)?
                {
                    continue;
                }

                let base_expression =
                    self.original_operator_condition_delta_expression(op2_id, condition_id)?;
                let composite_expression = LinearExpression {
                    coefficients: base_expression.coefficients.clone(),
                    constant: 0.0,
                };
                if self.has_effect(op2_id, &composite_expression.coefficients)? {
                    continue;
                }

                // PARITY(numeric-fd): get_sose_supporters() iterates ALL operators for the
                // linear-supporter check. Any op with a linear effect on composite_coefficients
                // (even one that has no base relaxed op) invalidates SOSE. The supporter
                // collection then only gathers ops that also have a simple/constant-assignment
                // effect. We must not gate the linear check behind base_relaxed_by_original.
                let mut condition_supporters = Vec::new();
                let mut invalid_support = false;
                for op1_id in 0..operator_count {
                    // Linear-effect check: scan ALL operators (C++ iterates all operators here).
                    if self.has_linear_effect(
                        op1_id,
                        &composite_expression.coefficients,
                        self.use_bounds,
                        false,
                    )? {
                        invalid_support = true;
                        break;
                    }

                    // Supporter collection: only ops that have a base relaxed op can become
                    // SOSE supporters (because we need a relaxed precondition set for them).
                    let Some(op1_relaxed_id) = base_relaxed_by_original[op1_id] else {
                        continue;
                    };

                    let (has_simple_effect, simple_effect) = self
                        .calculate_simple_effect_constant(
                            op1_id,
                            &composite_expression.coefficients,
                            self.use_bounds,
                        )?;
                    let has_constant_assignment_effect = self.config.use_constant_assignment
                        && self.has_constant_assignment_effect(
                            op1_id,
                            &composite_expression.coefficients,
                            self.use_bounds,
                        )?;
                    if !has_simple_effect && !has_constant_assignment_effect {
                        continue;
                    }

                    if self.has_effect(op1_id, &condition.coefficients)? {
                        invalid_support = true;
                        break;
                    }

                    condition_supporters.push((op1_relaxed_id, simple_effect));
                }

                if invalid_support {
                    continue;
                }

                self.operator_condition_eval[op2_id][condition_id].has_sose = true;
                self.operator_condition_eval[op2_id][condition_id].composite_expression =
                    Some(composite_expression);
                if self.use_bounds {
                    let projected_coefficients = self
                        .regular_numeric_variable_ids
                        .iter()
                        .map(|&numeric_var_id| {
                            base_expression
                                .coefficients
                                .get(numeric_var_id)
                                .copied()
                                .unwrap_or(0.0)
                        })
                        .collect::<Vec<_>>();
                    let mut has_bound = true;
                    let mut upper_bound = 0.0;
                    for (regular_var_id, &weight) in projected_coefficients.iter().enumerate() {
                        if weight >= self.config.precision
                            && self
                                .numeric_bound
                                .get_variable_before_action_has_ub(regular_var_id, op2_id)
                        {
                            upper_bound += weight
                                * self
                                    .numeric_bound
                                    .get_variable_before_action_ub(regular_var_id, op2_id);
                        } else if weight <= -self.config.precision
                            && self
                                .numeric_bound
                                .get_variable_before_action_has_lb(regular_var_id, op2_id)
                        {
                            upper_bound += weight
                                * self
                                    .numeric_bound
                                    .get_variable_before_action_lb(regular_var_id, op2_id);
                        } else if weight.abs() >= self.config.precision {
                            has_bound = false;
                            break;
                        }
                    }

                    if has_bound {
                        self.operator_condition_eval[op2_id][condition_id].has_upper_bound = true;
                        self.operator_condition_eval[op2_id][condition_id].upper_bound =
                            upper_bound;
                    }
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
                precondition_ids.extend(op2.precondition_ids);
                if precondition_ids.is_empty() {
                    precondition_ids.push(self.artificial_precondition_id);
                }

                let mut effect_ids = Vec::new();
                let mut sose_constants = vec![0.0; self.conditions.len()];
                for (condition_id, sose_constant) in effects {
                    effect_ids.push(self.get_numeric_proposition_id(condition_id)?);
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

        self.relaxed_operators.extend(new_operators);
        Ok(())
    }

    fn has_linear_effect(
        &self,
        original_operator_id: usize,
        coefficients: &[f64],
        use_bounded_linear: bool,
        only_conditional: bool,
    ) -> Result<bool, String> {
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(original_operator_id)
            .ok_or_else(|| {
                format!("LM-cut helper linear effects {original_operator_id} are missing")
            })?;
        for linear_effect in helper_linear_effects {
            let actual_var_id = *self.regular_numeric_variable_ids
                .get(linear_effect.target_local_var_id)
                .ok_or_else(|| {
                    format!(
                        "LM-cut helper linear effect target {} is invalid for operator {original_operator_id}",
                        linear_effect.target_local_var_id
                    )
                })?;
            let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            if only_conditional
                && linear_effect.preconditions.propositional_facts.is_empty()
                && linear_effect.preconditions.numeric_group_ids.is_empty()
            {
                continue;
            }

            if !use_bounded_linear {
                return Ok(true);
            }

            let local_var_id = Some(linear_effect.target_local_var_id);

            if weight >= self.config.precision {
                let blocked_by_upper_bound = local_var_id
                    .map(|var_id| {
                        self.numeric_bound
                            .get_effect_has_ub(original_operator_id, var_id)
                            || (self.config.use_constant_assignment
                                && self
                                    .numeric_bound
                                    .get_assignment_has_ub(original_operator_id, var_id))
                    })
                    .unwrap_or(false);
                if !blocked_by_upper_bound {
                    return Ok(true);
                }
                continue;
            }

            if weight <= -self.config.precision {
                let blocked_by_lower_bound = local_var_id
                    .map(|var_id| {
                        self.numeric_bound
                            .get_effect_has_lb(original_operator_id, var_id)
                            || (self.config.use_constant_assignment
                                && self
                                    .numeric_bound
                                    .get_assignment_has_lb(original_operator_id, var_id))
                    })
                    .unwrap_or(false);
                if !blocked_by_lower_bound {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }
    fn has_effect(
        &self,
        original_operator_id: usize,
        coefficients: &[f64],
    ) -> Result<bool, String> {
        if self.has_linear_effect(original_operator_id, coefficients, false, false)? {
            return Ok(true);
        }

        if self.config.use_constant_assignment
            && self.has_constant_assignment_effect(original_operator_id, coefficients, false)?
        {
            return Ok(true);
        }

        let (has_simple_effect, _) =
            self.calculate_simple_effect_constant(original_operator_id, coefficients, false)?;
        Ok(has_simple_effect)
    }

    fn has_constant_assignment_effect(
        &self,
        original_operator_id: usize,
        coefficients: &[f64],
        use_bounded_linear: bool,
    ) -> Result<bool, String> {
        let helper_is_assignment = self
            .numeric_helper
            .get_action_is_assignment(original_operator_id)
            .ok_or_else(|| {
                format!("LM-cut helper action is_assignment {original_operator_id} is missing")
            })?;
        let helper_conditional_assignments = self
            .numeric_helper
            .get_action_conditional_assign_list(original_operator_id)
            .ok_or_else(|| {
                format!("LM-cut helper conditional assignments {original_operator_id} are missing")
            })?;
        for (local_var_id, &is_assignment) in helper_is_assignment.iter().enumerate() {
            if !is_assignment {
                continue;
            }
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id).ok_or_else(|| {
                format!(
                    "LM-cut helper assignment target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            if self.use_bounds
                && ((weight >= self.config.precision
                    && self
                        .numeric_bound
                        .has_no_increasing_assignment_effect(original_operator_id, local_var_id))
                    || (weight <= -self.config.precision
                        && self.numeric_bound.has_no_decreasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        )))
            {
                continue;
            }

            return Ok(true);
        }

        for conditional_assignment in helper_conditional_assignments {
            let local_var_id = conditional_assignment.target_local_var_id;
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id).ok_or_else(|| {
                format!(
                    "LM-cut helper conditional assignment target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            if self.use_bounds
                && ((weight >= self.config.precision
                    && self
                        .numeric_bound
                        .has_no_increasing_assignment_effect(original_operator_id, local_var_id))
                    || (weight <= -self.config.precision
                        && self.numeric_bound.has_no_decreasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        )))
            {
                continue;
            }
            return Ok(true);
        }

        if use_bounded_linear {
            for (local_var_id, &actual_var_id) in
                self.regular_numeric_variable_ids.iter().enumerate()
            {
                let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
                if (weight >= self.config.precision
                    && self
                        .numeric_bound
                        .get_assignment_has_ub(original_operator_id, local_var_id)
                    && (!self.use_bounds
                        || !self.numeric_bound.has_no_increasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        )))
                    || (weight <= -self.config.precision
                        && self
                            .numeric_bound
                            .get_assignment_has_lb(original_operator_id, local_var_id)
                        && (!self.use_bounds
                            || !self.numeric_bound.has_no_decreasing_assignment_effect(
                                original_operator_id,
                                local_var_id,
                            )))
                {
                    return Ok(true);
                }
            }
        }

        Ok(false)
    }

    fn calculate_constant_assignment_effect(
        &self,
        original_operator_id: usize,
        coefficients: &[f64],
        numeric_values: &[f64],
        use_bounded_linear: bool,
    ) -> Result<f64, String> {
        let helper_is_assignment = self
            .numeric_helper
            .get_action_is_assignment(original_operator_id)
            .ok_or_else(|| {
                format!("LM-cut helper action is_assignment {original_operator_id} is missing")
            })?;
        let helper_assign_list = self
            .numeric_helper
            .get_action_assign_list(original_operator_id)
            .ok_or_else(|| {
                format!("LM-cut helper action assign_list {original_operator_id} is missing")
            })?;
        let helper_conditional_assignments = self
            .numeric_helper
            .get_action_conditional_assign_list(original_operator_id)
            .ok_or_else(|| {
                format!("LM-cut helper conditional assignments {original_operator_id} are missing")
            })?;
        let mut net = 0.0;

        for (local_var_id, &is_assignment) in helper_is_assignment.iter().enumerate() {
            if !is_assignment {
                continue;
            }
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id).ok_or_else(|| {
                format!(
                    "LM-cut helper assignment target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            if self.use_bounds
                && ((weight >= self.config.precision
                    && self
                        .numeric_bound
                        .has_no_increasing_assignment_effect(original_operator_id, local_var_id))
                    || (weight <= -self.config.precision
                        && self.numeric_bound.has_no_decreasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        )))
            {
                continue;
            }

            let constant_target = helper_assign_list[local_var_id];
            let state_value = numeric_values.get(actual_var_id).copied().unwrap_or(0.0);
            if (weight >= self.config.precision && constant_target > state_value)
                || (weight <= -self.config.precision && constant_target < state_value)
            {
                net += weight * (constant_target - state_value);
            }
        }

        for conditional_assignment in helper_conditional_assignments {
            let local_var_id = conditional_assignment.target_local_var_id;
            let actual_var_id = *self.regular_numeric_variable_ids.get(local_var_id).ok_or_else(|| {
                format!(
                    "LM-cut helper conditional assignment target {local_var_id} is invalid for operator {original_operator_id}"
                )
            })?;
            let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
            if weight.abs() < self.config.precision {
                continue;
            }
            if self.use_bounds
                && ((weight >= self.config.precision
                    && self
                        .numeric_bound
                        .has_no_increasing_assignment_effect(original_operator_id, local_var_id))
                    || (weight <= -self.config.precision
                        && self.numeric_bound.has_no_decreasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        )))
            {
                continue;
            }

            let constant_target = conditional_assignment.assigned_value;
            let state_value = numeric_values.get(actual_var_id).copied().unwrap_or(0.0);
            if (weight >= self.config.precision && constant_target > state_value)
                || (weight <= -self.config.precision && constant_target < state_value)
            {
                net += weight * (constant_target - state_value);
            }
        }

        if use_bounded_linear {
            for (local_var_id, &actual_var_id) in
                self.regular_numeric_variable_ids.iter().enumerate()
            {
                let weight = coefficients.get(actual_var_id).copied().unwrap_or(0.0);
                if self.use_bounds
                    && ((weight >= self.config.precision
                        && self.numeric_bound.has_no_increasing_assignment_effect(
                            original_operator_id,
                            local_var_id,
                        ))
                        || (weight <= -self.config.precision
                            && self.numeric_bound.has_no_decreasing_assignment_effect(
                                original_operator_id,
                                local_var_id,
                            )))
                {
                    continue;
                }

                if weight >= self.config.precision
                    && self
                        .numeric_bound
                        .get_assignment_has_ub(original_operator_id, local_var_id)
                {
                    let mut contribution = (self
                        .numeric_bound
                        .get_assignment_ub(original_operator_id, local_var_id)
                        - numeric_values.get(actual_var_id).copied().unwrap_or(0.0))
                    .max(0.0);
                    if self
                        .numeric_bound
                        .get_effect_has_ub(original_operator_id, local_var_id)
                    {
                        contribution = contribution.min(
                            self.numeric_bound
                                .get_effect_ub(original_operator_id, local_var_id),
                        );
                    }
                    net += weight * contribution;
                } else if weight <= -self.config.precision
                    && self
                        .numeric_bound
                        .get_assignment_has_lb(original_operator_id, local_var_id)
                {
                    let mut contribution = (self
                        .numeric_bound
                        .get_assignment_lb(original_operator_id, local_var_id)
                        - numeric_values.get(actual_var_id).copied().unwrap_or(0.0))
                    .min(0.0);
                    if self
                        .numeric_bound
                        .get_effect_has_lb(original_operator_id, local_var_id)
                    {
                        contribution = contribution.max(
                            self.numeric_bound
                                .get_effect_lb(original_operator_id, local_var_id),
                        );
                    }
                    net += weight * contribution;
                }
            }
        }

        Ok(net)
    }

    #[inline(always)]
    fn calculate_constant_assignment_effect_infallible(
        &self,
        original_operator_id: usize,
        coefficients: &[f64],
        numeric_values: &[f64],
        use_bounded_linear: bool,
    ) -> f64 {
        self.calculate_constant_assignment_effect(
            original_operator_id,
            coefficients,
            numeric_values,
            use_bounded_linear,
        )
        .unwrap_or_else(|error| panic!("{error}"))
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

    fn original_operator_condition_delta_expression(
        &self,
        original_operator_id: usize,
        condition_id: usize,
    ) -> Result<LinearExpression, String> {
        let helper_linear_effects = self
            .numeric_helper
            .get_action_linear_effects(original_operator_id)
            .ok_or_else(|| {
                format!("LM-cut helper linear effects {original_operator_id} are missing")
            })?;
        let condition = self
            .conditions
            .get(condition_id)
            .ok_or_else(|| format!("LM-cut numeric condition {condition_id} is invalid"))?;

        let mut expression = LinearExpression::zero(self.task.numeric_variables().len());
        for linear_effect in helper_linear_effects {
            let reconstructed_linear_effect = self
                .numeric_helper
                .linearized_effect_for_action_assignment(
                    original_operator_id,
                    linear_effect.source_assignment_effect_id,
                )
                .ok_or_else(|| {
                    format!(
                        "LM-cut helper linearized effect {} is missing for operator {original_operator_id}",
                        linear_effect.source_assignment_effect_id
                    )
                })?;
            let target_coefficient = condition
                .coefficients
                .get(reconstructed_linear_effect.affected_var_id)
                .copied()
                .unwrap_or(0.0);
            if target_coefficient.abs() < self.config.precision {
                continue;
            }

            expression =
                expression.add(&reconstructed_linear_effect.delta.scale(target_coefficient));
        }
        Ok(expression)
    }

    fn numeric_net_effect_for_operator(
        &self,
        propositional_values: &[usize],
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
        propositional_values: &[usize],
        conditions: &[ExplicitFact],
    ) -> bool {
        conditions.iter().all(|condition| {
            propositional_values.get(condition.var()).copied() == Some(condition.value())
        })
    }

    fn build_relaxed_operator_for_axiom(&mut self, operator_id: usize, axiom: &PropositionalAxiom) {
        let helper_preconditions = self
            .numeric_helper
            .get_action_pre_list(operator_id)
            .expect("helper action pre-list for axiom must exist");
        let helper_num_list = self
            .numeric_helper
            .get_action_num_list(operator_id)
            .map(|ids| ids.to_vec())
            .expect("helper action numeric pre-list for axiom must exist");
        let precondition_groups = self.precondition_proposition_id_groups(helper_preconditions);
        let mut precondition_ids = self.flatten_precondition_groups(&precondition_groups);
        self.append_numeric_condition_propositions(&helper_num_list, &mut precondition_ids);
        let effect_fact = self
            .numeric_helper
            .get_action_add_list(operator_id)
            .and_then(|add_facts| add_facts.first())
            .cloned()
            .unwrap_or_else(|| ExplicitFact::new(axiom.var_id(), axiom.effect_value()));
        let effect_ids = if self.is_numeric_axiom_var(axiom.var_id()) {
            Vec::new()
        } else {
            vec![self.get_proposition_id(&effect_fact)]
        };
        let relaxed_operator = RelaxedOperator::new(
            if precondition_ids.is_empty() {
                vec![self.artificial_precondition_id]
            } else {
                precondition_ids
            },
            effect_ids,
            operator_id,
            0.0,
            format!(
                "axiom {}",
                self.get_proposition_name(effect_fact.var(), effect_fact.value())
            ),
            false,
        );
        relaxed_operator.assert_well_formed();
        self.relaxed_operators.push(relaxed_operator);
    }

    fn build_precondition_ids(&self, preconditions: &[ExplicitFact]) -> Vec<usize> {
        self.flatten_precondition_groups(&self.precondition_proposition_id_groups(preconditions))
    }

    fn precondition_proposition_id_groups(
        &self,
        preconditions: &[ExplicitFact],
    ) -> Vec<Vec<usize>> {
        preconditions
            .iter()
            .map(|precondition| self.precondition_proposition_ids(precondition))
            .filter(|group| !group.is_empty())
            .collect()
    }

    fn flatten_precondition_groups(&self, groups: &[Vec<usize>]) -> Vec<usize> {
        let mut result = Vec::new();
        let mut seen = BTreeSet::new();
        for group in groups {
            for &proposition_id in group {
                if seen.insert(proposition_id) {
                    result.push(proposition_id);
                }
            }
        }
        result
    }

    fn append_numeric_condition_propositions(
        &mut self,
        condition_group_ids: &[usize],
        target_ids: &mut Vec<usize>,
    ) {
        for condition_id in self
            .numeric_helper
            .get_condition_ids_from_group_ids(condition_group_ids)
        {
            let Ok(proposition_id) = self.get_numeric_proposition_id(condition_id) else {
                continue;
            };
            if !target_ids.contains(&proposition_id) {
                target_ids.push(proposition_id);
            }
        }
    }

    #[allow(clippy::single_element_loop)]
    fn build_comparison_fact_condition_ids(&mut self) {
        if self.config.ignore_numeric {
            return;
        }

        for comparison_axiom in self.task.comparison_axioms().iter() {
            let affected_var_id = comparison_axiom.get_affected_var_id();
            for fact_value in [0] {
                let condition_ids = self
                    .numeric_helper
                    .get_comparison_fact_condition_ids(affected_var_id, fact_value);
                if !condition_ids.is_empty() {
                    self.comparison_fact_to_condition_ids
                        .insert((affected_var_id, fact_value), condition_ids);
                }
            }
        }
    }

    fn add_linear_conditions(&mut self) {
        if self.config.ignore_numeric {
            return;
        }

        self.linear_effect_to_conditions_plus = vec![Vec::new(); self.task.get_operators().len()];
        self.linear_effect_to_conditions_minus = vec![Vec::new(); self.task.get_operators().len()];

        for (operator_id, operator) in self.task.get_operators().iter().enumerate() {
            let helper_conditional_fact_effects = self
                .numeric_helper
                .get_action_conditional_fact_effects(operator_id)
                .map(|effects| effects.to_vec())
                .expect("helper conditional fact effects must exist for operator");
            let helper_linear_effects = self
                .numeric_helper
                .get_action_linear_effects(operator_id)
                .map(|effects| effects.to_vec())
                .expect("helper linear effects must exist for operator");
            self.linear_effect_to_conditions_plus[operator_id] =
                vec![Vec::new(); operator.assignment_effects().len()];
            self.linear_effect_to_conditions_minus[operator_id] =
                vec![Vec::new(); operator.assignment_effects().len()];

            let base_precondition_groups = self.precondition_proposition_id_groups(
                self.numeric_helper
                    .get_action_pre_list(operator_id)
                    .expect("helper action pre-list must exist for operator"),
            );
            let helper_num_list = self
                .numeric_helper
                .get_action_num_list(operator_id)
                .map(|ids| ids.to_vec())
                .expect("helper action numeric pre-list must exist for operator");
            let mut expanded_base_precondition_ids =
                self.flatten_precondition_groups(&base_precondition_groups);
            self.append_numeric_condition_propositions(
                &helper_num_list,
                &mut expanded_base_precondition_ids,
            );

            if !base_precondition_groups.is_empty() {
                let mut global_base_precondition_ids =
                    self.flatten_precondition_groups(&base_precondition_groups);
                self.append_numeric_condition_propositions(
                    &helper_num_list,
                    &mut global_base_precondition_ids,
                );
                // Keep this explicit even though the local vector is not reused below:
                // helper-owned action precondition shaping still materializes the redundant
                // conditions globally through `add_numeric_condition_proposition()`, mirroring the
                // reference-side global condition growth in `numeric_helper::build_action()`.
            }

            for conditional_effect in &helper_conditional_fact_effects {
                let effect_condition_groups = self.precondition_proposition_id_groups(
                    &conditional_effect.preconditions.propositional_facts,
                );
                if effect_condition_groups.is_empty() {
                    continue;
                }

                let mut expanded_effect_condition_ids =
                    self.flatten_precondition_groups(&effect_condition_groups);
                self.append_numeric_condition_propositions(
                    &conditional_effect.preconditions.numeric_group_ids,
                    &mut expanded_effect_condition_ids,
                );
                // The expanded ids are transient here, but the redundant numeric conditions are
                // still materialized globally through `append_*_redundant_numeric_conditions()`.
            }

            for linear_effect in &helper_linear_effects {
                let assignment_effect_id = linear_effect.source_assignment_effect_id;
                let helper_effect_preconditions = linear_effect.preconditions.clone();
                let mut extended_numeric_group_ids =
                    helper_effect_preconditions.numeric_group_ids.clone();
                for &group_id in &helper_num_list {
                    if !extended_numeric_group_ids.contains(&group_id) {
                        extended_numeric_group_ids.push(group_id);
                    }
                }
                let assignment_condition_groups = self.precondition_proposition_id_groups(
                    &helper_effect_preconditions.propositional_facts,
                );
                if !assignment_condition_groups.is_empty()
                    || !helper_effect_preconditions.numeric_group_ids.is_empty()
                {
                    let mut expanded_effect_condition_ids =
                        self.flatten_precondition_groups(&assignment_condition_groups);
                    self.append_numeric_condition_propositions(
                        &helper_effect_preconditions.numeric_group_ids,
                        &mut expanded_effect_condition_ids,
                    );
                    // Same as above: the local vector is disposable, but the condition material-
                    // ization side effects still persist globally in `self.conditions`.
                }

                let mut extended_precondition_ids = expanded_base_precondition_ids.clone();
                let mut seen_preconditions: BTreeSet<usize> =
                    extended_precondition_ids.iter().copied().collect();
                for group in &assignment_condition_groups {
                    for &proposition_id in group {
                        if seen_preconditions.insert(proposition_id) {
                            extended_precondition_ids.push(proposition_id);
                        }
                    }
                }
                self.append_numeric_condition_propositions(
                    &helper_effect_preconditions.numeric_group_ids,
                    &mut extended_precondition_ids,
                );

                let reconstructed_linear_effect = self
                    .numeric_helper
                    .linearized_effect_for_action_assignment(operator_id, assignment_effect_id)
                    .unwrap_or_else(|| {
                        panic!(
                            "LM-cut helper linearized effect {assignment_effect_id} is missing for operator {operator_id}"
                        )
                    });
                let affected_var_id = reconstructed_linear_effect.affected_var_id;
                let delta_expression = reconstructed_linear_effect.delta;

                let plus_proposition_id =
                    self.add_numeric_condition_proposition(NumericCondition::from_expression(
                        delta_expression.clone(),
                        true,
                        format!(
                            "numeric ({} {} +inf guard)",
                            operator.name(),
                            affected_var_id
                        ),
                    ));
                let plus_condition_id = self.propositions[plus_proposition_id]
                    .id_numeric_condition
                    .expect("new +inf condition proposition must reference its condition");
                self.linear_effect_to_conditions_plus[operator_id][assignment_effect_id]
                    .push(plus_condition_id);

                let plus_redundant_conditions = self
                    .conditions
                    .get(plus_condition_id)
                    .map(|base_condition| {
                        let other_conditions = self
                            .representative_numeric_conditions_for_group_ids(
                                &extended_numeric_group_ids,
                            );
                        self.numeric_helper
                            .combine_condition_with_conditions(base_condition, &other_conditions)
                    })
                    .unwrap_or_default();
                for redundant_condition in plus_redundant_conditions {
                    let redundant_proposition_id =
                        self.add_numeric_condition_proposition(redundant_condition);
                    let redundant_condition_id = self.propositions[redundant_proposition_id]
                        .id_numeric_condition
                        .expect(
                            "new redundant +inf condition proposition must reference its condition",
                        );
                    self.linear_effect_to_conditions_plus[operator_id][assignment_effect_id]
                        .push(redundant_condition_id);
                }

                let minus_proposition_id =
                    self.add_numeric_condition_proposition(NumericCondition::from_expression(
                        delta_expression.scale(-1.0),
                        true,
                        format!(
                            "numeric ({} {} -inf guard)",
                            operator.name(),
                            affected_var_id
                        ),
                    ));
                let minus_condition_id = self.propositions[minus_proposition_id]
                    .id_numeric_condition
                    .expect("new -inf condition proposition must reference its condition");
                self.linear_effect_to_conditions_minus[operator_id][assignment_effect_id]
                    .push(minus_condition_id);

                let minus_redundant_conditions = self
                    .conditions
                    .get(minus_condition_id)
                    .map(|base_condition| {
                        let other_conditions = self
                            .representative_numeric_conditions_for_group_ids(
                                &extended_numeric_group_ids,
                            );
                        self.numeric_helper
                            .combine_condition_with_conditions(base_condition, &other_conditions)
                    })
                    .unwrap_or_default();
                for redundant_condition in minus_redundant_conditions {
                    let redundant_proposition_id =
                        self.add_numeric_condition_proposition(redundant_condition);
                    let redundant_condition_id = self.propositions[redundant_proposition_id]
                        .id_numeric_condition
                        .expect(
                            "new redundant -inf condition proposition must reference its condition",
                        );
                    self.linear_effect_to_conditions_minus[operator_id][assignment_effect_id]
                        .push(redundant_condition_id);
                }
            }
        }
    }

    fn get_proposition_numeric_conditions(
        &self,
        proposition_ids: &[usize],
    ) -> Option<Vec<NumericCondition>> {
        let conditions = proposition_ids
            .iter()
            .filter_map(|&proposition_id| {
                self.propositions
                    .get(proposition_id)
                    .and_then(|proposition| proposition.id_numeric_condition)
                    .and_then(|condition_id| self.conditions.get(condition_id))
                    .cloned()
            })
            .collect::<Vec<_>>();
        Some(conditions)
    }

    fn representative_numeric_conditions_for_group_ids(
        &self,
        group_ids: &[usize],
    ) -> Vec<NumericCondition> {
        group_ids
            .iter()
            .filter_map(|&group_id| {
                self.numeric_helper
                    .condition_group_representative_condition_id(group_id)
                    .and_then(|condition_id| self.numeric_helper.get_condition(condition_id))
                    .cloned()
            })
            .collect()
    }

    fn add_numeric_condition_proposition(&mut self, condition: NumericCondition) -> usize {
        let epsilon = if condition.is_strictly_greater {
            self.config.epsilon
        } else {
            0.0
        };
        self.add_numeric_condition_proposition_with_epsilon(condition, epsilon)
    }

    fn add_numeric_condition_proposition_with_epsilon(
        &mut self,
        condition: NumericCondition,
        epsilon: f64,
    ) -> usize {
        let condition_id = self.conditions.len();
        let proposition_id = self.propositions.len();
        let mut proposition = RelaxedProposition::new(proposition_id, condition.name.clone());
        proposition.is_numeric_condition = true;
        proposition.id_numeric_condition = Some(condition_id);
        self.propositions.push(proposition);
        self.numeric_condition_proposition_ids.push(proposition_id);
        self.conditions.push(condition);
        self.epsilons.push(epsilon);
        self.num_propositions += 1;
        proposition_id
    }

    fn get_numeric_proposition_id(&self, condition_id: usize) -> Result<usize, String> {
        self.numeric_condition_proposition_ids
            .get(condition_id)
            .copied()
            .ok_or_else(|| format!("LM-cut numeric condition {condition_id} has no proposition"))
    }

    #[inline(always)]
    fn get_numeric_proposition_id_infallible(&self, condition_id: usize) -> usize {
        self.numeric_condition_proposition_ids[condition_id]
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

    fn precondition_proposition_ids(&self, fact: &ExplicitFact) -> Vec<usize> {
        if !self.config.ignore_numeric {
            if self.numeric_helper.is_comparison_axiom_var(fact.var()) && fact.value() > 0 {
                return Vec::new();
            }
            if let Some(condition_ids) = self
                .comparison_fact_to_condition_ids
                .get(&(fact.var(), fact.value()))
            {
                return condition_ids
                    .iter()
                    .map(|&condition_id| {
                        self.get_numeric_proposition_id(condition_id)
                            .expect("comparison fact condition must have proposition")
                    })
                    .collect();
            }
        }

        if self.is_numeric_axiom_var(fact.var()) {
            // Reached when either `ignore_numeric=true` (numeric tracking is disabled
            // wholesale) or the fact references a numeric-axiom var that isn't a
            // comparison axiom registered in `comparison_fact_to_condition_ids`
            // (e.g. an assignment-axiom var, or a comparison var whose TRUE/value=0
            // entry produced an empty condition list at build time).
            //
            // Dropping the precondition is an admissible relaxation: an operator
            // whose numeric/axiom precondition we cannot model becomes easier to
            // apply, never harder. Reachability over-approximates and the LM-cut
            // sum of cut costs is still a valid lower bound on optimal plan cost.
            // The prior `panic!` aborted whole searches on plant-watering/6_2_2
            // and similar tasks where assignment-axiom preconditions surface.
            return Vec::new();
        }

        vec![self.get_proposition_id(fact)]
    }

    fn get_proposition_id_for_effect(&self, effect: &Effect) -> usize {
        let fact = ExplicitFact::new(effect.var_id(), effect.value());
        self.get_proposition_id(&fact)
    }

    fn get_proposition_name(&self, var_id: usize, value: usize) -> String {
        self.numeric_helper
            .get_proposition(var_id, value)
            .and_then(|helper_id| self.numeric_helper.get_proposition_name(helper_id))
            .unwrap_or("")
            .to_string()
    }

    fn proposition_name_for_effect(&self, effect: &Effect) -> String {
        self.get_proposition_name(effect.var_id(), effect.value())
    }

    fn get_proposition_id(&self, fact: &ExplicitFact) -> usize {
        self.numeric_helper
            .get_proposition(fact.var(), fact.value())
            .map(|helper_id| helper_id + 2)
            .expect("helper proposition id must exist")
    }

    fn is_numeric_axiom_var(&self, variable_id: usize) -> bool {
        self.numeric_helper.is_numeric_axiom_var(variable_id)
    }

    fn compute_landmarks_impl(
        &mut self,
        propositional_values: &[usize],
        state_buffer_len: usize,
        numeric_values: &[f64],
        debug_state: bool,
        collect_landmarks: bool,
    ) -> Result<ComputeLandmarksResult, String> {
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
        let mut cut = std::mem::take(&mut self.cut_scratch);
        cut.clear();
        let mut m_list = std::mem::take(&mut self.multiplier_scratch);
        m_list.clear();
        let result = (|| {
            self.start_state_numeric_tracking();
            self.first_exploration(propositional_values, numeric_values)?;
            if self.proposition_status(self.artificial_goal_id) == PropositionStatus::Unreached {
                return Ok((true, f64::INFINITY, collect_landmarks.then(Vec::new)));
            }

            let mut total_cost = 0.0;
            let mut landmarks = collect_landmarks.then(Vec::new);
            let debug_iterations = std::env::var_os("LMCUT_DEBUG_ITERATIONS").is_some();
            let debug_focus = std::env::var_os("LMCUT_DEBUG_FOCUS").is_some();
            let mut iteration = 0usize;

            while self.proposition_h_max_cost(self.artificial_goal_id) >= self.config.precision {
                iteration += 1;
                self.start_cut_iteration_tracking();
                self.mark_goal_plateau(
                    propositional_values,
                    numeric_values,
                    self.artificial_goal_id,
                );
                self.second_exploration(
                    propositional_values,
                    numeric_values,
                    &mut cut,
                    &mut m_list,
                );
                assert!(!cut.is_empty(), "LM-cut must find a non-empty cut");

                let mut cut_cost = f64::INFINITY;

                for (cut_index, &operator_id) in cut.iter().enumerate() {
                    let multiplier = m_list[cut_index];
                    let current_cut_cost = self.edge_cost(operator_id, multiplier);
                    let (original_op_id_1, original_op_id_2) = {
                        let operator = &self.relaxed_operators[operator_id];
                        (operator.original_op_id_1, operator.original_op_id_2)
                    };
                    if multiplier.0 >= self.config.precision
                        && let Some(original_id) = original_op_id_1
                    {
                        self.update_original_operator_min_cut_cost(original_id, current_cut_cost);
                    }
                    if let Some(original_id) = original_op_id_2 {
                        self.update_original_operator_min_cut_cost(original_id, current_cut_cost);
                    }
                    cut_cost = cut_cost.min(current_cut_cost);
                }

                if debug_iterations && (iteration <= 20 || iteration.is_multiple_of(1000)) {
                    let cut_details = if iteration <= 3 || cut_cost.abs() < self.config.precision {
                        cut
                        .iter()
                        .zip(m_list.iter())
                        .map(|(&operator_id, &(m1, m2))| {
                            let operator = &self.relaxed_operators[operator_id];
                            let effects = operator
                                .effect_ids
                                .iter()
                                .filter_map(|&effect_id| {
                                    self.propositions.get(effect_id).map(|effect| {
                                        format!("{}:{:?}", effect.name, effect.status)
                                    })
                                })
                                .collect::<Vec<_>>()
                                .join(",");
                            format!(
                                "id={} name={} orig=({:?},{:?}) cost=({},{}) m=({},{}) supporter={:?} effects=[{}]",
                                operator_id,
                                operator.name,
                                operator.original_op_id_1,
                                operator.original_op_id_2,
                                self.operator_cost_1(operator_id),
                                self.operator_cost_2(operator_id),
                                m1,
                                m2,
                                self.operator_h_max_supporter(operator_id),
                                effects
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" | ")
                    } else {
                        String::new()
                    };
                    debug!(
                        "LMCUT_DEBUG_ITER iteration={} goal_h={} cut_size={} cut_cost={}",
                        iteration,
                        self.proposition_h_max_cost(self.artificial_goal_id),
                        cut.len(),
                        cut_cost
                    );
                    if !cut_details.is_empty() {
                        debug!("LMCUT_DEBUG_ZERO_CUT {}", cut_details);
                    }
                }

                if !cut_cost.is_finite() {
                    let cut_details = cut
                    .iter()
                    .zip(m_list.iter())
                    .map(|(&operator_id, &multiplier)| {
                        let operator = &self.relaxed_operators[operator_id];
                        let edge_cost = self.edge_cost(operator_id, multiplier);
                        let supporter = operator
                            .original_op_id_2
                            .and(self.operator_h_max_supporter(operator_id))
                            .or(self.operator_h_max_supporter(operator_id))
                            .or(None)
                            .and_then(|supporter_id| self.propositions.get(supporter_id).map(|p| {
                                format!(
                                    "{}:{}:{:?}",
                                    supporter_id,
                                    p.name,
                                    self.proposition_status(supporter_id)
                                )
                            }))
                            .or_else(|| self.operator_h_max_supporter(operator_id).and_then(|supporter_id| self.propositions.get(supporter_id).map(|p| {
                                format!(
                                    "{}:{}:{:?}",
                                    supporter_id,
                                    p.name,
                                    self.proposition_status(supporter_id)
                                )
                            })))
                            .unwrap_or_else(|| "none".to_string());
                        let effects = operator
                            .effect_ids
                            .iter()
                            .filter_map(|&effect_id| {
                                self.propositions.get(effect_id).map(|p| {
                                    format!("{}:{}:{:?}", effect_id, p.name, self.proposition_status(effect_id))
                                })
                            })
                            .collect::<Vec<_>>();
                        let preconditions = operator
                            .precondition_ids
                            .iter()
                            .filter_map(|&precondition_id| {
                                self.propositions.get(precondition_id).map(|p| {
                                    format!(
                                        "{}:{}:{:?}:h={}",
                                        precondition_id,
                                        p.name,
                                        self.proposition_status(precondition_id),
                                        self.proposition_h_max_cost(precondition_id),
                                    )
                                })
                            })
                            .collect::<Vec<_>>();
                        format!(
                            "id={operator_id} name={} unsat={} edge_cost={} m=({},{}) cost=({},{}) orig=({:?},{:?}) supporter={} preconditions=[{}] effects=[{}]",
                            operator.name,
                            self.operator_unsatisfied_preconditions(operator_id),
                            edge_cost,
                            multiplier.0,
                            multiplier.1,
                            self.operator_cost_1(operator_id),
                            self.operator_cost_2(operator_id),
                            operator.original_op_id_1,
                            operator.original_op_id_2,
                            supporter,
                            preconditions.join(", "),
                            effects.join(", "),
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(" | ");
                    return Err(format!(
                        "LM-cut produced an invalid cut: cut_cost={cut_cost}, cut_size={}, goal_h_max={}, cut=[{}].",
                        cut.len(),
                        self.proposition_h_max_cost(self.artificial_goal_id),
                        cut_details,
                    ));
                }

                if debug_state {
                    let cut_details = cut
                        .iter()
                        .zip(m_list.iter())
                        .map(|(&operator_id, &(m1, m2))| {
                            let operator = &self.relaxed_operators[operator_id];
                            format!(
                                "name={} orig=({:?},{:?}) m=({},{}) cost=({},{})",
                                operator.name,
                                operator.original_op_id_1,
                                operator.original_op_id_2,
                                m1,
                                m2,
                                self.operator_cost_1(operator_id),
                                self.operator_cost_2(operator_id),
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(" | ");
                    debug!(
                        "LMCUT_DEBUG_STATE iteration={} cut_cost={} cut=[{}]",
                        iteration, cut_cost, cut_details,
                    );
                }

                total_cost += cut_cost;

                // PARITY(numeric-fd): the reference implementation has no bailout for repeated
                // zero-cost cuts here. Returning an error from this loop turns a parity bug into a
                // synthetic dead end, so the faithful port must continue the LM-cut iterations.
                for touched_index in 0..self.touched_original_operator_ids.len() {
                    let original_id = self.touched_original_operator_ids[touched_index];
                    let min_cost = self.original_operator_min_cut_costs[original_id];
                    if min_cost < self.config.precision {
                        continue;
                    }
                    let mapped_operator_count =
                        self.original_to_relaxed_operators[original_id].len();
                    for mapped_index in 0..mapped_operator_count {
                        let relaxed_operator_id =
                            self.original_to_relaxed_operators[original_id][mapped_index];
                        let mut multiplier = min_cost;
                        let mut multiplier_to_record = None;
                        {
                            let relaxed_operator = &self.relaxed_operators[relaxed_operator_id];
                            let runtime = &mut self.operator_runtime[relaxed_operator_id];
                            if relaxed_operator.original_op_id_1 == Some(original_id)
                                && runtime.cost_1 >= self.config.precision
                            {
                                multiplier /= runtime.cost_1;
                                runtime.cost_1 = (runtime.cost_1 - cut_cost / multiplier).max(0.0);
                                if collect_landmarks {
                                    multiplier_to_record = Some(multiplier);
                                }
                            }
                            if relaxed_operator.original_op_id_2 == Some(original_id)
                                && runtime.cost_2 >= self.config.precision
                            {
                                multiplier /= runtime.cost_2;
                                runtime.cost_2 = (runtime.cost_2 - cut_cost / multiplier).max(0.0);
                                if collect_landmarks {
                                    multiplier_to_record = Some(multiplier);
                                }
                            }
                        }
                        if let Some(multiplier_to_record) = multiplier_to_record {
                            self.record_original_operator_multiplier(
                                original_id,
                                multiplier_to_record,
                            );
                        }
                    }
                }

                if let Some(landmarks) = landmarks.as_mut() {
                    //self.landmark_original_operator_ids.sort_unstable(); //TODO: Figure out if that is necessary
                    landmarks.push(
                        self.landmark_original_operator_ids
                            .iter()
                            .map(|&operator_id| {
                                (self.original_operator_multipliers[operator_id], operator_id)
                            })
                            .collect(),
                    );
                }

                self.first_exploration_incremental(propositional_values, numeric_values, &cut)?;
                if debug_focus && iteration <= 3 {
                    for operator in self.relaxed_operators.iter().filter(|operator| {
                        matches!(
                            operator.name.as_str(),
                            "increase_y "
                                | "decrease_y "
                                | "increase_z "
                                | "visit x0y0z0"
                                | "visit x0y0z1"
                        )
                    }) {
                        let supporter = operator
                            .original_op_id_2
                            .and(self.operator_h_max_supporter(operator.id))
                            .or(self.operator_h_max_supporter(operator.id))
                            .and_then(|supporter_id| {
                                self.propositions.get(supporter_id).map(|supporter| {
                                    format!(
                                        "{}:{}:{:?}:h={}",
                                        supporter_id,
                                        supporter.name,
                                        self.proposition_status(supporter_id),
                                        self.proposition_h_max_cost(supporter_id),
                                    )
                                })
                            })
                            .unwrap_or_else(|| "none".to_string());
                        let preconditions = operator
                            .precondition_ids
                            .iter()
                            .filter_map(|&precondition_id| {
                                self.propositions.get(precondition_id).map(|precondition| {
                                    format!(
                                        "{}:{}:{:?}:h={}",
                                        precondition_id,
                                        precondition.name,
                                        self.proposition_status(precondition_id),
                                        self.proposition_h_max_cost(precondition_id),
                                    )
                                })
                            })
                            .collect::<Vec<_>>()
                            .join(" | ");
                        let effects = operator
                            .effect_ids
                            .iter()
                            .filter_map(|&effect_id| {
                                self.propositions.get(effect_id).map(|effect| {
                                    format!(
                                        "{}:{}:{:?}:h={}",
                                        effect_id,
                                        effect.name,
                                        self.proposition_status(effect_id),
                                        self.proposition_h_max_cost(effect_id),
                                    )
                                })
                            })
                            .collect::<Vec<_>>()
                            .join(" | ");
                        debug!(
                            "LMCUT_DEBUG_FOCUS iteration={} name={} cost=({}, {}) supporter={} preconditions=[{}] effects=[{}]",
                            iteration,
                            operator.name,
                            self.operator_cost_1(operator.id),
                            self.operator_cost_2(operator.id),
                            supporter,
                            preconditions,
                            effects,
                        );
                    }
                    for &proposition_id in &[61usize, 69usize, 80usize, 81usize, 104usize] {
                        if let Some(proposition) = self.propositions.get(proposition_id) {
                            let achievers = proposition
                                .effect_of
                                .iter()
                                .filter_map(|&achiever_id| {
                                    self.relaxed_operators.get(achiever_id).map(|achiever| {
                                        format!(
                                            "{}:{}:cost=({}, {}):supporter={:?}",
                                            achiever_id,
                                            achiever.name,
                                            self.operator_cost_1(achiever_id),
                                            self.operator_cost_2(achiever_id),
                                            self.operator_h_max_supporter(achiever_id),
                                        )
                                    })
                                })
                                .collect::<Vec<_>>()
                                .join(" | ");
                            debug!(
                                "LMCUT_DEBUG_PROP iteration={} id={} name={} status={:?} h={} achievers=[{}]",
                                iteration,
                                proposition_id,
                                proposition.name,
                                self.proposition_status(proposition_id),
                                self.proposition_h_max_cost(proposition_id),
                                achievers,
                            );
                        }
                    }
                }
                cut.clear();
                m_list.clear();
                self.reset_goal_zone_statuses();
                self.set_proposition_status(self.artificial_goal_id, PropositionStatus::Reached);
                self.set_proposition_status(
                    self.artificial_precondition_id,
                    PropositionStatus::Reached,
                );
            }

            Ok((false, total_cost, landmarks))
        })();
        cut.clear();
        m_list.clear();
        self.cut_scratch = cut;
        self.multiplier_scratch = m_list;
        result
    }

    pub fn compute_landmark_cost(
        &mut self,
        propositional_values: &[usize],
        state_buffer_len: usize,
        numeric_values: &[f64],
        debug_state: bool,
    ) -> Result<(bool, f64), String> {
        let (dead_end, total_cost, _) = self.compute_landmarks_impl(
            propositional_values,
            state_buffer_len,
            numeric_values,
            debug_state,
            false,
        )?;
        Ok((dead_end, total_cost))
    }

    pub fn compute_landmarks(
        &mut self,
        propositional_values: &[usize],
        state_buffer_len: usize,
        numeric_values: &[f64],
        debug_state: bool,
    ) -> Result<(bool, f64, Vec<Landmark>), String> {
        let (dead_end, total_cost, landmarks) = self.compute_landmarks_impl(
            propositional_values,
            state_buffer_len,
            numeric_values,
            debug_state,
            true,
        )?;
        Ok((dead_end, total_cost, landmarks.unwrap_or_default()))
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
