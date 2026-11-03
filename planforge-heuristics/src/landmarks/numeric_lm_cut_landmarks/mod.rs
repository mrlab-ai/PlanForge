//! The landmarks of numeric LM-cut: the relaxed task, its propositions and
//! operators, and the cuts that give the landmarks of a state.

mod constant_assignments;
mod cut;
mod effects;
mod exploration;
mod propositions;
mod relaxed_task;
mod sose_effects;

use super::lm_cut_numeric_heuristic::LmCutNumericConfig;
use super::numeric_bound::NumericBound;
use super::numeric_helper::{LinearNumericCondition as NumericCondition, NumericTaskHelper};
use planforge_sas::axioms::PropositionalAxiom;
use planforge_sas::default_value_axioms::{DefaultValueAxiomMode, default_value_axioms};
use planforge_sas::numeric_task::{
    ExplicitFact, ExplicitValueIndex, NumericValue, Operator, OperatorIndex, Task, VariableIndex,
};
use planforge_sas::utils::linear_effects::LinearExpression;

use super::operator_conditions::OperatorConditionTable;
use planforge_sas::utils::linear_effects::LinearNumericEffect;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// The value a comparison-axiom variable takes when the comparison is false.
/// Only this value carries numeric conditions: the true value is produced by
/// the axiom itself.
const COMPARISON_FALSE_FACT_VALUE: ExplicitValueIndex = ExplicitValueIndex::new(0);

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
    pub precondition_of: Vec<OperatorIndex>,
    pub effect_of: Vec<OperatorIndex>,
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
    pub id: OperatorIndex,
    pub original_op_id_1: Option<OperatorIndex>,
    pub original_op_id_2: Option<OperatorIndex>,
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
        op_id: OperatorIndex,
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

#[derive(Debug)]
pub struct LandmarkCutLandmarks {
    task: Arc<Task>,
    config: LmCutNumericConfig,
    propositions: Vec<RelaxedProposition>,
    proposition_runtime: Vec<PropositionRuntime>,
    /// Scratch of `mark_goal_plateau`: (proposition, next achiever, end).
    plateau_stack: Vec<(usize, usize, usize)>,
    proposition_precondition_of_data: Vec<OperatorIndex>,
    proposition_precondition_of_ranges: Vec<(usize, usize)>,
    proposition_effect_of_data: Vec<OperatorIndex>,
    proposition_effect_of_ranges: Vec<(usize, usize)>,
    proposition_index: Vec<Vec<usize>>,
    numeric_condition_proposition_ids: Vec<usize>,
    conditions: Vec<NumericCondition>,
    epsilons: Vec<f64>,
    numeric_helper: NumericTaskHelper,
    /// Condition ids of the *false* fact of each comparison axiom variable,
    /// keyed by that variable. Only value 0 is ever registered or looked up:
    /// a comparison variable being true is the axiom's own effect, which
    /// `precondition_proposition_ids` relaxes away.
    comparison_false_fact_condition_ids: BTreeMap<VariableIndex, Vec<usize>>,
    linear_effect_to_conditions_plus: Vec<Vec<Vec<usize>>>,
    linear_effect_to_conditions_minus: Vec<Vec<Vec<usize>>>,
    operator_condition_eval: OperatorConditionTable,
    relaxed_operators: Vec<RelaxedOperator>,
    operator_runtime: Vec<OperatorRuntime>,
    proposition_runtime_epoch: u32,
    touched_proposition_ids: Vec<usize>,
    goal_zone_proposition_ids: Vec<usize>,
    operator_runtime_epoch: u32,
    touched_operator_ids: Vec<OperatorIndex>,
    regular_numeric_variable_ids: Vec<VariableIndex>,
    operator_precondition_id_data: Vec<usize>,
    operator_precondition_id_ranges: Vec<(usize, usize)>,
    operator_effect_id_data: Vec<usize>,
    operator_effect_id_ranges: Vec<(usize, usize)>,
    original_to_relaxed_operators: Vec<Vec<OperatorIndex>>,
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
    touched_original_operator_ids: Vec<OperatorIndex>,
    incremental_original_operator_marks: Vec<u32>,
    incremental_original_operator_epoch: u32,
    incremental_original_operator_ids_scratch: Vec<OperatorIndex>,
    /// The value of each condition-table entry's composite expression in the
    /// evaluated state, valid where its mark is the current epoch.
    composite_expression_values: Vec<f64>,
    composite_expression_value_marks: Vec<u32>,
    composite_expression_value_epoch: u32,
    cut_scratch: Vec<OperatorIndex>,
    multiplier_scratch: Vec<(f64, f64)>,
    second_exploration_queue_scratch: Vec<usize>,
    /// The variable bounds before each operator; built only with
    /// `bound_iterations > 0`, the only case anything reads them.
    bounds: Option<NumericBound>,
    /// Every rule the relaxation reads: the task's axioms, which only ever
    /// *prove* a derived variable, followed by the rules describing how one takes
    /// its default value.
    ///
    /// The delete relaxation has no negation by failure, so without the second
    /// group a state in which a derived variable is false looks like a dead end.
    /// The task does not carry them -- they are exponential in the worst case and
    /// only a relaxation wants them -- so they are derived here, and behind an
    /// `Arc` because the build loop needs the list while it mutates the rest of
    /// the heuristic.
    relaxed_axioms: Arc<[PropositionalAxiom]>,
    use_bounds: bool,
    initialized: bool,
}

impl LandmarkCutLandmarks {
    pub fn new(task: Arc<Task>, config: LmCutNumericConfig) -> Self {
        assert!(
            config.precision >= 0.0,
            "LM-cut precision must be non-negative"
        );
        assert!(config.epsilon >= 0.0, "LM-cut epsilon must be non-negative");
        let use_bounds = config.bound_iterations > 0;
        let default_value_axioms =
            default_value_axioms(&task, DefaultValueAxiomMode::ApproximateNegativeCycles);
        let relaxed_axioms: Arc<[PropositionalAxiom]> = task
            .axioms()
            .iter()
            .cloned()
            .chain(default_value_axioms.iter().cloned())
            .collect();
        let bounds = use_bounds.then(|| {
            NumericBound::new(
                &task,
                &default_value_axioms,
                config.precision,
                config.epsilon,
            )
        });
        let numeric_helper = NumericTaskHelper::new_lmcut(
            &task,
            &default_value_axioms,
            config.precision,
            config.epsilon,
            config.use_constant_assignment,
        );
        let regular_numeric_variable_ids = task.regular_numeric_variable_ids();
        let mut result = Self {
            task,
            config,
            propositions: Vec::new(),
            proposition_runtime: Vec::new(),
            plateau_stack: Vec::new(),
            proposition_precondition_of_data: Vec::new(),
            proposition_precondition_of_ranges: Vec::new(),
            proposition_effect_of_data: Vec::new(),
            proposition_effect_of_ranges: Vec::new(),
            proposition_index: Vec::new(),
            numeric_condition_proposition_ids: Vec::new(),
            conditions: Vec::new(),
            epsilons: Vec::new(),
            numeric_helper,
            comparison_false_fact_condition_ids: BTreeMap::new(),
            linear_effect_to_conditions_plus: Vec::new(),
            linear_effect_to_conditions_minus: Vec::new(),
            operator_condition_eval: OperatorConditionTable::default(),
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
            touched_original_operator_ids: Vec::new(),
            incremental_original_operator_marks: Vec::new(),
            incremental_original_operator_epoch: 1,
            incremental_original_operator_ids_scratch: Vec::new(),
            composite_expression_values: Vec::new(),
            composite_expression_value_marks: Vec::new(),
            composite_expression_value_epoch: 1,
            cut_scratch: Vec::new(),
            multiplier_scratch: Vec::new(),
            second_exploration_queue_scratch: Vec::new(),
            bounds,
            relaxed_axioms,
            use_bounds,
            initialized: false,
        };
        result.initialize();
        result
    }

    fn initialize(&mut self) {
        assert!(!self.initialized, "LM-cut landmarks initialized twice");
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
        self.comparison_false_fact_condition_ids.clear();
        self.linear_effect_to_conditions_plus.clear();
        self.linear_effect_to_conditions_minus.clear();
        self.operator_condition_eval = OperatorConditionTable::default();
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
        self.goal_precondition_ids.clear();
        self.propositions.push(RelaxedProposition::new(
            self.artificial_precondition_id,
            "artificial".to_string(),
        ));
        self.propositions.push(RelaxedProposition::new(
            self.artificial_goal_id,
            "goal".to_string(),
        ));
        self.num_variables = self.task.variables().len();
        self.proposition_index = vec![Vec::new(); self.num_variables];

        self.num_propositions = 2;
        self.build_propositional_propositions();
        self.build_numeric_conditions();
        self.build_comparison_fact_condition_ids();
        self.add_linear_conditions();
        self.prepare_goal_preconditions();
        // Operator costs read the bounds of the variables before each operator.
        if self.use_bounds {
            let initial_numeric_values = self.task.initial_numeric_values();
            self.bounds
                .as_mut()
                .expect("bounds are built with bound_iterations > 0")
                .calculate_bounds(initial_numeric_values, self.config.bound_iterations);
        }
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
        self.composite_expression_values = vec![0.0; self.operator_condition_eval.len()];
        self.composite_expression_value_marks = vec![0; self.operator_condition_eval.len()];
        self.original_operator_min_cut_costs
            .resize(self.original_to_relaxed_operators.len(), 0.0);
        self.original_operator_min_cut_cost_marks
            .resize(self.original_to_relaxed_operators.len(), 0);
        self.incremental_original_operator_marks
            .resize(self.original_to_relaxed_operators.len(), 0);
        self.cut_marks.resize(self.relaxed_operators.len(), 0);
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

    fn is_cut_marked(&self, operator_id: OperatorIndex) -> bool {
        self.cut_marks[operator_id.index()] == self.cut_mark_epoch
    }

    fn mark_cut(&mut self, operator_id: OperatorIndex) {
        if let Some(mark) = self.cut_marks.get_mut(operator_id.index()) {
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
        self.touched_original_operator_ids.clear();
    }

    fn start_state_numeric_tracking(&mut self) {
        if self.composite_expression_value_epoch == u32::MAX {
            self.composite_expression_value_marks.fill(0);
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
    fn mark_operator_runtime_touched(&mut self, operator_id: OperatorIndex) {
        if self.operator_runtime[operator_id.index()].mark != self.operator_runtime_epoch {
            self.operator_runtime[operator_id.index()].mark = self.operator_runtime_epoch;
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
    fn operator_unsatisfied_preconditions(&self, operator_id: OperatorIndex) -> usize {
        self.operator_runtime[operator_id.index()].unsatisfied_preconditions
    }

    #[inline(always)]
    fn decrement_operator_unsatisfied_preconditions(
        &mut self,
        operator_id: OperatorIndex,
    ) -> usize {
        self.mark_operator_runtime_touched(operator_id);
        self.operator_runtime[operator_id.index()].unsatisfied_preconditions -= 1;
        self.operator_runtime[operator_id.index()].unsatisfied_preconditions
    }

    #[inline(always)]
    fn set_operator_h_max_supporter(
        &mut self,
        operator_id: OperatorIndex,
        supporter_id: Option<usize>,
    ) {
        self.mark_operator_runtime_touched(operator_id);
        self.operator_runtime[operator_id.index()].h_max_supporter = supporter_id;
    }

    #[inline(always)]
    fn set_operator_h_max_supporter_cost(&mut self, operator_id: OperatorIndex, cost: f64) {
        self.mark_operator_runtime_touched(operator_id);
        self.operator_runtime[operator_id.index()].h_max_supporter_cost = cost;
    }

    #[inline(always)]
    fn operator_cost_1(&self, operator_id: OperatorIndex) -> f64 {
        self.operator_runtime[operator_id.index()].cost_1
    }

    #[inline(always)]
    fn operator_cost_2(&self, operator_id: OperatorIndex) -> f64 {
        self.operator_runtime[operator_id.index()].cost_2
    }

    #[inline(always)]
    fn operator_h_max_supporter(&self, operator_id: OperatorIndex) -> Option<usize> {
        self.operator_runtime[operator_id.index()].h_max_supporter
    }

    #[inline(always)]
    fn operator_h_max_supporter_cost(&self, operator_id: OperatorIndex) -> f64 {
        self.operator_runtime[operator_id.index()].h_max_supporter_cost
    }

    #[inline(always)]
    fn cached_composite_expression_value(
        &mut self,
        operator_id: OperatorIndex,
        condition_id: usize,
        numeric_values: &[NumericValue],
    ) -> f64 {
        let Some(index) = self
            .operator_condition_eval
            .index(operator_id, condition_id)
        else {
            return 0.0;
        };
        let Some(expression) = self
            .operator_condition_eval
            .get(operator_id, condition_id)
            .composite_expression
            .as_ref()
        else {
            return 0.0;
        };
        if self.composite_expression_value_marks[index] != self.composite_expression_value_epoch {
            self.composite_expression_value_marks[index] = self.composite_expression_value_epoch;
            self.composite_expression_values[index] = expression.evaluate(numeric_values).value();
        }
        self.composite_expression_values[index]
    }

    fn update_original_operator_min_cut_cost(&mut self, original_id: OperatorIndex, cut_cost: f64) {
        if self.original_operator_min_cut_cost_marks[original_id.index()]
            != self.original_operator_min_cut_cost_epoch
        {
            self.original_operator_min_cut_cost_marks[original_id.index()] =
                self.original_operator_min_cut_cost_epoch;
            self.original_operator_min_cut_costs[original_id.index()] = cut_cost;
            self.touched_original_operator_ids.push(original_id);
        } else {
            self.original_operator_min_cut_costs[original_id.index()] =
                self.original_operator_min_cut_costs[original_id.index()].min(cut_cost);
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
    fn operator_precondition_id_range(&self, operator_id: OperatorIndex) -> (usize, usize) {
        self.operator_precondition_id_ranges[operator_id.index()]
    }

    #[inline(always)]
    fn operator_effect_id_range(&self, operator_id: OperatorIndex) -> (usize, usize) {
        self.operator_effect_id_ranges[operator_id.index()]
    }

    /// The bounds, which are read only with `bound_iterations > 0`.
    fn numeric_bound(&self) -> &NumericBound {
        self.bounds
            .as_ref()
            .expect("bounds are read only with bound_iterations > 0")
    }

    fn evaluate_numeric_condition(
        &self,
        condition_id: usize,
        numeric_values: &[NumericValue],
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

    fn get_proposition_name(&self, var_id: VariableIndex, value: ExplicitValueIndex) -> String {
        self.numeric_helper
            .get_proposition(var_id, value)
            .and_then(|helper_id| self.numeric_helper.get_proposition_name(helper_id))
            .unwrap_or("")
            .to_string()
    }

    fn get_proposition_id(&self, fact: &ExplicitFact) -> usize {
        self.numeric_helper
            .get_proposition(fact.var_index(), fact.value_index())
            .map(|helper_id| helper_id + 2)
            .expect("helper proposition id must exist")
    }

    fn is_numeric_axiom_var(&self, variable_id: VariableIndex) -> bool {
        self.numeric_helper.is_numeric_axiom_var(variable_id)
    }

    pub fn task(&self) -> &Task {
        &self.task
    }

    pub fn propositions(&self) -> &[RelaxedProposition] {
        &self.propositions
    }

    pub fn relaxed_operators(&self) -> &[RelaxedOperator] {
        &self.relaxed_operators
    }
}
