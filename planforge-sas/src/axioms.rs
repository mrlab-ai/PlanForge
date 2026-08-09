#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::cmp::max;
use std::sync::Arc;

use crate::numeric_task::{AbstractNumericTask, ExplicitFact, TaskRef};
use crate::utils::errors::{AxiomEvalError, InvalidIndex, WrongAxiomLayer};
use crate::utils::int_packer::IntDoublePacker;

#[derive(Debug, Clone)]
pub struct PropositionalAxiom {
    conditions: Vec<ExplicitFact>,
    var_id: usize,
    precondition_value: usize,
    effect_value: usize,
}

impl PropositionalAxiom {
    pub fn new(
        conditions: Vec<ExplicitFact>,
        var_id: usize,
        precondition_value: usize,
        effect_value: usize,
    ) -> Self {
        PropositionalAxiom {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
    }

    pub fn var_id(&self) -> usize {
        self.var_id
    }

    pub fn precondition_value(&self) -> usize {
        self.precondition_value
    }

    pub fn effect_value(&self) -> usize {
        self.effect_value
    }

    pub fn conditions(&self) -> &Vec<ExplicitFact> {
        &self.conditions
    }
}

#[derive(Debug, Clone)]
pub enum CalOperator {
    Sum,
    Difference,
    Product,
    Division,
}
#[derive(Debug, Clone)]
pub struct AssignmentAxiom {
    pub affected_var_id: usize,
    pub operator: CalOperator,
    pub left_hand_side: usize,
    pub right_hand_side: usize,
}

impl AssignmentAxiom {
    pub fn new(
        affected_var_id: usize,
        operator: CalOperator,
        left_hand_side: usize,
        right_hand_side: usize,
    ) -> Self {
        AssignmentAxiom {
            affected_var_id,
            operator,
            left_hand_side,
            right_hand_side,
        }
    }

    pub fn update_values(&self, numeric_state: &mut [f64]) -> Result<f64, InvalidIndex> {
        let left = self.left_hand_side;
        let right = self.right_hand_side;
        if left >= numeric_state.len() || right >= numeric_state.len() {
            return Err(InvalidIndex {
                length: numeric_state.len(),
                index: left,
            });
        }
        let affected = self.affected_var_id;
        if affected >= numeric_state.len() {
            return Err(InvalidIndex {
                length: numeric_state.len(),
                index: affected,
            });
        }
        let result = match self.operator {
            CalOperator::Sum => numeric_state[left] + numeric_state[right],
            CalOperator::Difference => numeric_state[left] - numeric_state[right],
            CalOperator::Product => numeric_state[left] * numeric_state[right],
            CalOperator::Division => {
                if numeric_state[right] == 0.0 {
                    return Err(InvalidIndex {
                        length: numeric_state.len(),
                        index: right,
                    });
                }
                numeric_state[left] / numeric_state[right]
            }
        };
        numeric_state[affected] = result;
        Ok(result)
    }

    pub fn get_left_var_id(&self) -> usize {
        self.left_hand_side
    }

    pub fn get_right_var_id(&self) -> usize {
        self.right_hand_side
    }

    pub fn get_affected_var_id(&self) -> usize {
        self.affected_var_id
    }

    pub fn get_operator(&self) -> &CalOperator {
        &self.operator
    }
}

#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    LessThan,
    LessThanOrEqual,
    Equal,
    GreaterThanOrEqual,
    GreaterThan,
    UnEqual,
}

impl ComparisonOperator {
    pub fn compare(&self, numeric_values: &[f64], left: usize, right: usize) -> bool {
        let (left, right) = (numeric_values[left], numeric_values[right]);
        match self {
            ComparisonOperator::LessThan => left < right,
            ComparisonOperator::LessThanOrEqual => left <= right,
            ComparisonOperator::Equal => left == right,
            ComparisonOperator::GreaterThanOrEqual => left >= right,
            ComparisonOperator::GreaterThan => left > right,
            ComparisonOperator::UnEqual => left != right,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ComparisonAxiom {
    pub affected_var_id: usize,
    pub left_hand_side: usize,
    pub right_hand_side: usize,
    pub operator: ComparisonOperator,
}

impl ComparisonAxiom {
    pub fn new(
        affected_var_id: usize,
        left_hand_side: usize,
        right_hand_side: usize,
        operator: ComparisonOperator,
    ) -> Self {
        ComparisonAxiom {
            affected_var_id,
            left_hand_side,
            right_hand_side,
            operator,
        }
    }

    pub fn is_hold(&self, numeric_state: &[f64]) -> Result<bool, InvalidIndex> {
        let left = self.left_hand_side;
        let right = self.right_hand_side;
        if left >= numeric_state.len() || right >= numeric_state.len() {
            return Err(InvalidIndex {
                length: numeric_state.len(),
                index: left,
            });
        }
        let comp_op = &self.operator;
        let result = comp_op.compare(numeric_state, self.left_hand_side, self.right_hand_side);
        Ok(result)
    }

    pub fn get_affected_var_id(&self) -> usize {
        self.affected_var_id
    }
    pub fn get_left_var_id(&self) -> usize {
        self.left_hand_side
    }
    pub fn get_right_var_id(&self) -> usize {
        self.right_hand_side
    }

    pub fn get_operator(&self) -> &ComparisonOperator {
        &self.operator
    }
}
#[derive(Debug, Clone)]
struct AxiomRule {
    condition_count: usize,
    effect_var: usize,
    effect_value: usize,
}

impl AxiomRule {
    pub fn new(cond_count: usize, eff_var: usize, eff_val: usize) -> Self {
        AxiomRule {
            condition_count: cond_count,
            effect_var: eff_var,
            effect_value: eff_val,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct AxiomLiteral {
    condition_of: Vec<usize>,
}

#[derive(Debug, Clone, Copy, Default)]
struct NegationByFailureInfo {
    var_id: usize,
    /// The variable's axiom default. It is both the value that says "nothing
    /// proved this variable" and the literal the closure then announces.
    default_value: usize,
}

impl NegationByFailureInfo {
    pub fn new(var_id: usize, default_value: usize) -> Self {
        NegationByFailureInfo {
            var_id,
            default_value,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LiteralRef {
    var_id: usize,
    value: usize,
}

#[derive(Debug, Clone)]
#[allow(unused)]
struct AxiomEvaluatorData {
    axiom_literals: Vec<Vec<AxiomLiteral>>,
    rules: Vec<AxiomRule>,
    comparison_axiom_layer: Option<usize>,
    first_propositional_axiom_layer: Option<usize>,
    last_propositional_axiom_layer: Option<usize>,
    last_arithmetic_axiom_layer: Option<usize>,
    nbf_info_by_layer: Vec<Vec<NegationByFailureInfo>>,
    /// Per variable, the value the closure resets it to. Copied out of the
    /// task once so the hot reset loop is an indexed load rather than a
    /// borrow of the task's shared initial state.
    axiom_default_values: Vec<usize>,
    has_numeric_axioms: bool,
    has_propositional_axioms: bool,
}

#[allow(clippy::needless_range_loop)]
fn build_compiled_axiom_evaluator_data(
    numeric_task: &dyn AbstractNumericTask,
) -> AxiomEvaluatorData {
    let mut axiom_literals = vec![];
    let mut nbf_info_by_layer = vec![];
    let mut rules = vec![];
    let mut comparison_axiom_layer = None;
    let mut first_propositional_axiom_layer = None;
    let mut last_propositional_axiom_layer = None;
    let mut last_arithmetic_axiom_layer = None;

    for numeric_var in numeric_task.numeric_variables().iter() {
        last_arithmetic_axiom_layer = max(last_arithmetic_axiom_layer, numeric_var.axiom_layer());
    }

    for i in 0..numeric_task.get_num_variables() {
        let axiom_layer = numeric_task.get_variable_axiom_layer(i).unwrap();
        if axiom_layer.is_none() {
            continue;
        }
        last_propositional_axiom_layer = max(last_propositional_axiom_layer, axiom_layer);
        if first_propositional_axiom_layer.is_none()
            || axiom_layer < first_propositional_axiom_layer
        {
            first_propositional_axiom_layer = axiom_layer;
        }
    }

    if first_propositional_axiom_layer.is_some() && numeric_task.get_num_cmp_axioms() > 0 {
        comparison_axiom_layer = first_propositional_axiom_layer;
        first_propositional_axiom_layer = first_propositional_axiom_layer.map(|x| x + 1);
        debug_assert_eq!(
            comparison_axiom_layer.unwrap(),
            last_arithmetic_axiom_layer.map(|x| x + 1).unwrap_or(0)
        );
    }

    for var in numeric_task.variables().iter() {
        axiom_literals.push(vec![AxiomLiteral::default(); var.domain_size()]);
    }

    for axiom in numeric_task.axioms().iter() {
        let cond_count = axiom.conditions.len();
        let eff_var = axiom.var_id;
        let eff_val = axiom.effect_value;
        rules.push(AxiomRule::new(cond_count, eff_var, eff_val));
    }

    for i in 0..numeric_task.axioms().len() {
        let axiom: &PropositionalAxiom = &numeric_task.axioms()[i];
        for condition in axiom.conditions().iter() {
            axiom_literals[condition.var()][condition.value()]
                .condition_of
                .push(i);
        }
    }

    let mut last_layer = None;
    for i in 0..numeric_task.get_num_variables() {
        last_layer = max(
            last_layer,
            numeric_task.get_variable_axiom_layer(i).unwrap(),
        );
    }
    nbf_info_by_layer.resize(last_layer.map(|x| x + 1).unwrap_or(0), vec![]);

    let axiom_default_values: Vec<usize> = (0..numeric_task.get_num_variables())
        .map(|var_id| {
            numeric_task
                .get_variable_default_axiom_value(var_id)
                .expect("variable id below the variable count is in bounds")
        })
        .collect();
    for var_id in 0..numeric_task.get_num_variables() {
        let axiom_layer = numeric_task.get_variable_axiom_layer(var_id).unwrap();
        if let Some(idx) = axiom_layer
            && axiom_layer != last_layer
        {
            let nbf_info = NegationByFailureInfo::new(var_id, axiom_default_values[var_id]);
            nbf_info_by_layer[idx].push(nbf_info);
        }
    }

    AxiomEvaluatorData {
        axiom_literals,
        rules,
        comparison_axiom_layer,
        first_propositional_axiom_layer,
        last_propositional_axiom_layer,
        last_arithmetic_axiom_layer,
        nbf_info_by_layer,
        axiom_default_values,
        has_numeric_axioms: !numeric_task.assignment_axioms().is_empty()
            || !numeric_task.comparison_axioms().is_empty(),
        has_propositional_axioms: !numeric_task.axioms().is_empty(),
    }
}

#[allow(unused)]
pub struct AxiomEvaluator<'a> {
    pub numeric_task: TaskRef<'a>,
    state_packer: Arc<IntDoublePacker>,
    axiom_literals: Vec<Vec<AxiomLiteral>>,
    rules: Vec<AxiomRule>,
    comparison_axiom_layer: Option<usize>,
    first_propositional_axiom_layer: Option<usize>,
    last_propositional_axiom_layer: Option<usize>,
    last_arithmetic_axiom_layer: Option<usize>,
    nbf_info_by_layer: Vec<Vec<NegationByFailureInfo>>,
    axiom_default_values: Vec<usize>,
    queue: RefCell<Vec<LiteralRef>>,
    unsatisfied_conditions: RefCell<Vec<usize>>,
}

impl<'a> AxiomEvaluator<'a> {
    pub fn new(numeric_task: TaskRef<'a>, state_packer: Arc<IntDoublePacker>) -> Self {
        let compiled = build_compiled_axiom_evaluator_data(&*numeric_task);
        let rule_count = compiled.rules.len();

        AxiomEvaluator {
            numeric_task,
            state_packer,
            axiom_literals: compiled.axiom_literals,
            rules: compiled.rules,
            comparison_axiom_layer: compiled.comparison_axiom_layer,
            first_propositional_axiom_layer: compiled.first_propositional_axiom_layer,
            last_propositional_axiom_layer: compiled.last_propositional_axiom_layer,
            last_arithmetic_axiom_layer: compiled.last_arithmetic_axiom_layer,
            nbf_info_by_layer: compiled.nbf_info_by_layer,
            axiom_default_values: compiled.axiom_default_values,
            queue: RefCell::new(Vec::new()),
            unsatisfied_conditions: RefCell::new(vec![0; rule_count]),
        }
    }

    pub fn evaluate_arithmetic_axioms(
        &self,
        numeric_state: &mut [f64],
    ) -> Result<(), InvalidIndex> {
        for axiom in self.numeric_task.assignment_axioms() {
            axiom.update_values(numeric_state)?;
        }

        Ok(())
    }
    pub fn affected_vars_by_arithmetic_axioms(&self, affected: &mut Vec<usize>) {
        for axiom in self.numeric_task.assignment_axioms() {
            affected.push(axiom.get_affected_var_id());
        }
    }

    pub fn evaluate_comparison_axioms(
        &self,
        buffer: &mut [u64],
        numeric_state: &mut [f64],
    ) -> Result<bool, AxiomEvalError> {
        for axiom in self.numeric_task.comparison_axioms() {
            let is_hold = axiom.is_hold(numeric_state).map_err(|e| {
                AxiomEvalError::InvalidIndex(InvalidIndex {
                    length: numeric_state.len(),
                    index: e.index,
                })
            })?;
            self.state_packer
                .set(buffer, axiom.get_affected_var_id(), !is_hold as u64);
        }

        Ok(true)
    }

    pub fn affected_vars_by_comparison_axioms(&self, affected: &mut Vec<usize>) {
        for axiom in self.numeric_task.comparison_axioms() {
            affected.push(axiom.get_affected_var_id());
        }
    }

    /// Compute the propositional axiom closure over `buffer`.
    ///
    /// Derived variables start at their default value and are proven upwards
    /// through the axiom layers: each layer runs the Horn rules to fixpoint,
    /// then admits negation-by-failure for the variables that stayed at their
    /// default, which lets the next layer use "not proven" as a condition.
    pub fn evaluate_propositional_axioms(&self, buffer: &mut [u64]) -> Result<(), AxiomEvalError> {
        if self.numeric_task.axioms().is_empty() {
            return Ok(());
        }

        let mut queue = self.queue.borrow_mut();
        queue.clear();
        let mut unsatisfied_conditions = self.unsatisfied_conditions.borrow_mut();
        if unsatisfied_conditions.len() != self.rules.len() {
            unsatisfied_conditions.resize(self.rules.len(), 0);
        }

        self.seed_queue_from_state(buffer, &mut queue)?;
        self.fire_trivial_rules(buffer, &mut queue, &mut unsatisfied_conditions);

        let layer_count = self.nbf_info_by_layer.len();
        for layer_no in 0..layer_count {
            self.propagate_horn_rules(buffer, &mut queue, &mut unsatisfied_conditions);
            // The last layer has nothing above it that could read the
            // negation-by-failure literals, so it does not produce them.
            if layer_no + 1 != layer_count {
                self.enqueue_unproven_literals(buffer, &mut queue, layer_no);
            }
        }

        Ok(())
    }

    /// Queue every literal the closure may start from, and reset the derived
    /// variables it will prove to their default values.
    ///
    /// Non-derived variables and comparison results hold their value from the
    /// state; propositional derived variables do not, because a fact proven in
    /// the predecessor state need not hold here.
    #[inline]
    fn seed_queue_from_state(
        &self,
        buffer: &mut [u64],
        queue: &mut Vec<LiteralRef>,
    ) -> Result<(), AxiomEvalError> {
        for var_id in 0..self.numeric_task.get_num_variables() {
            let axiom_layer = self.numeric_task.get_variable_axiom_layer(var_id).unwrap();
            if axiom_layer.is_none() || axiom_layer == self.comparison_axiom_layer {
                queue.push(LiteralRef {
                    var_id,
                    value: self.state_packer.get(buffer, var_id) as usize,
                });
                continue;
            }
            if axiom_layer <= self.last_arithmetic_axiom_layer
                || axiom_layer > self.last_propositional_axiom_layer
            {
                return Err(AxiomEvalError::WrongAxiomLayer(WrongAxiomLayer {
                    axiom_layer,
                    last_arithmetic_axiom_layer: self.last_arithmetic_axiom_layer,
                }));
            }
            self.state_packer
                .set(buffer, var_id, self.axiom_default_values[var_id] as u64);
        }
        Ok(())
    }

    /// Reset every rule's outstanding-condition counter and apply the rules
    /// that have no conditions at all, which hold in every state.
    #[inline]
    fn fire_trivial_rules(
        &self,
        buffer: &mut [u64],
        queue: &mut Vec<LiteralRef>,
        unsatisfied_conditions: &mut [usize],
    ) {
        for (rule_index, rule) in self.rules.iter().enumerate() {
            unsatisfied_conditions[rule_index] = rule.condition_count;
            if rule.condition_count == 0 {
                self.derive_literal(buffer, queue, rule.effect_var, rule.effect_value as u64);
            }
        }
    }

    /// Run the Horn rules to fixpoint: each dequeued literal satisfies one
    /// condition of every rule that names it, and a rule whose last condition
    /// is satisfied derives its effect.
    #[inline]
    fn propagate_horn_rules(
        &self,
        buffer: &mut [u64],
        queue: &mut Vec<LiteralRef>,
        unsatisfied_conditions: &mut [usize],
    ) {
        while let Some(literal) = queue.pop() {
            let dependent_rules = &self.axiom_literals[literal.var_id][literal.value].condition_of;
            for &rule_index in dependent_rules {
                let remaining = &mut unsatisfied_conditions[rule_index];
                *remaining -= 1;
                if *remaining == 0 {
                    let rule = &self.rules[rule_index];
                    self.derive_literal(buffer, queue, rule.effect_var, rule.effect_value as u64);
                }
            }
        }
    }

    /// Negation by failure: a derived variable still at its default value
    /// after this layer's fixpoint cannot be proven, so higher layers may
    /// assume it false.
    #[inline]
    fn enqueue_unproven_literals(
        &self,
        buffer: &mut [u64],
        queue: &mut Vec<LiteralRef>,
        layer_no: usize,
    ) {
        for info in &self.nbf_info_by_layer[layer_no] {
            if self.state_packer.get(buffer, info.var_id) == info.default_value as u64 {
                queue.push(LiteralRef {
                    var_id: info.var_id,
                    value: info.default_value,
                });
            }
        }
    }

    /// Write a derived literal and queue it, unless the buffer already says
    /// so — re-deriving a literal would make the fixpoint loop forever.
    #[inline]
    fn derive_literal(
        &self,
        buffer: &mut [u64],
        queue: &mut Vec<LiteralRef>,
        var_id: usize,
        value: u64,
    ) {
        if self.state_packer.get(buffer, var_id) == value {
            return;
        }
        self.state_packer.set(buffer, var_id, value);
        queue.push(LiteralRef {
            var_id,
            value: value as usize,
        });
    }

    pub fn evaluate(
        &self,
        buffer: &mut [u64],
        numeric_state: &mut [f64],
    ) -> Result<(), AxiomEvalError> {
        if !self.has_axioms() {
            return Ok(());
        }
        if self.has_numeric_axioms() {
            self.evaluate_comparison_axioms(buffer, numeric_state)?;
        }
        if self.has_propositional_axioms() {
            self.evaluate_propositional_axioms(buffer)?;
        }
        Ok(())
    }

    pub fn affected_propositional_vars(&self, affected_prop_vars: &mut Vec<usize>) {
        if self.has_axioms() {
            self.affected_vars_by_comparison_axioms(affected_prop_vars);
        }
    }

    pub fn affected_numeric_vars(&self, affected_numeric_vars: &mut Vec<usize>) {
        if self.has_numeric_axioms() {
            self.affected_vars_by_arithmetic_axioms(affected_numeric_vars);
        }
    }

    pub fn has_axioms(&self) -> bool {
        self.has_numeric_axioms() || self.has_propositional_axioms()
    }

    pub fn has_numeric_axioms(&self) -> bool {
        !self.numeric_task.assignment_axioms().is_empty()
            || !self.numeric_task.comparison_axioms().is_empty()
    }

    pub fn has_propositional_axioms(&self) -> bool {
        !self.numeric_task.axioms().is_empty()
    }
}
