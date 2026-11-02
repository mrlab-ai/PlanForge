#[cfg(test)]
mod tests;

use std::cell::RefCell;
use std::sync::Arc;

use crate::numeric_task::{
    ExplicitFact, ExplicitValueIndex, ExplicitVariable, NumericValue, NumericVariable, Task,
    VariableIndex, ZERO_VALUE,
};
use crate::utils::errors::AssignmentAxiomError;
use crate::utils::state_packer::StatePacker;

#[derive(Debug, Clone, PartialEq)]
pub struct PropositionalAxiom {
    conditions: Vec<ExplicitFact>,
    var_id: VariableIndex,
    precondition_value: ExplicitValueIndex,
    effect_value: ExplicitValueIndex,
}

impl PropositionalAxiom {
    pub fn new(
        conditions: Vec<ExplicitFact>,
        var_id: VariableIndex,
        precondition_value: ExplicitValueIndex,
        effect_value: ExplicitValueIndex,
    ) -> Self {
        PropositionalAxiom {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
    }

    pub fn var_id(&self) -> VariableIndex {
        self.var_id
    }

    pub fn precondition_value(&self) -> ExplicitValueIndex {
        self.precondition_value
    }

    pub fn effect_value(&self) -> ExplicitValueIndex {
        self.effect_value
    }

    pub fn conditions(&self) -> &Vec<ExplicitFact> {
        &self.conditions
    }

    /// Only for `Task::assign_fact_namespaces`, the single pass that
    /// owns fact namespace assignment.
    pub(crate) fn conditions_mut(&mut self) -> &mut Vec<ExplicitFact> {
        &mut self.conditions
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CalOperator {
    Sum,
    Difference,
    Product,
    Division,
}
/// Arithmetic definitions retain binary syntax for nonlinear expressions and
/// effect sources. Normalized comparisons use a single linear sum.
#[derive(Debug, Clone, PartialEq)]
pub enum NumericExpression {
    Binary {
        operator: CalOperator,
        left: VariableIndex,
        right: VariableIndex,
    },
    /// A constant-free sum over regular numeric variables.
    Linear(Vec<(VariableIndex, NumericValue)>),
}

impl NumericExpression {
    pub fn dependencies(&self) -> impl Iterator<Item = VariableIndex> + '_ {
        let (binary, linear) = match self {
            Self::Binary { left, right, .. } => ([Some(*left), Some(*right)], &[][..]),
            Self::Linear(terms) => ([None, None], terms.as_slice()),
        };
        binary
            .into_iter()
            .flatten()
            .chain(linear.iter().map(|(variable, _)| *variable))
    }

    /// The expression's value in `values`; the task asserts on construction
    /// that every operand indexes into them.
    pub fn evaluate(&self, values: &[NumericValue]) -> Result<NumericValue, AssignmentAxiomError> {
        let value = match self {
            Self::Linear(terms) => NumericValue::affine(
                terms
                    .iter()
                    .map(|(variable, coefficient)| (coefficient.value(), values[variable.index()])),
                0.0,
            ),
            Self::Binary {
                operator,
                left,
                right,
            } => {
                let (lhs, rhs) = (values[left.index()], values[right.index()]);
                match operator {
                    CalOperator::Sum => lhs + rhs,
                    CalOperator::Difference => lhs - rhs,
                    CalOperator::Product => lhs * rhs,
                    CalOperator::Division => {
                        if rhs == ZERO_VALUE {
                            return Err(AssignmentAxiomError::DivisionByZero {
                                divisor: right.index(),
                            });
                        }
                        lhs / rhs
                    }
                }
            }
        };
        Ok(value)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentAxiom {
    pub affected_var_id: VariableIndex,
    pub expression: NumericExpression,
}

impl AssignmentAxiom {
    pub fn new(
        affected_var_id: VariableIndex,
        operator: CalOperator,
        left: VariableIndex,
        right: VariableIndex,
    ) -> Self {
        Self {
            affected_var_id,
            expression: NumericExpression::Binary {
                operator,
                left,
                right,
            },
        }
    }

    pub fn linear(
        affected_var_id: VariableIndex,
        terms: Vec<(VariableIndex, NumericValue)>,
    ) -> Self {
        Self {
            affected_var_id,
            expression: NumericExpression::Linear(terms),
        }
    }

    pub fn update_values(
        &self,
        numeric_state: &mut [NumericValue],
    ) -> Result<NumericValue, AssignmentAxiomError> {
        let value = self.expression.evaluate(numeric_state)?;
        numeric_state[self.affected_var_id.index()] = value;
        Ok(value)
    }

    pub fn affected_var_id(&self) -> VariableIndex {
        self.affected_var_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ComparisonOperator {
    LessThan,
    LessThanOrEqual,
    Equal,
    GreaterThanOrEqual,
    GreaterThan,
    UnEqual,
}

impl ComparisonOperator {
    pub fn compare(
        &self,
        numeric_values: &[NumericValue],
        left: VariableIndex,
        right: VariableIndex,
    ) -> bool {
        let (left, right) = (numeric_values[left.index()], numeric_values[right.index()]);
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

#[derive(Debug, Clone, PartialEq)]
pub struct ComparisonAxiom {
    pub affected_var_id: VariableIndex,
    pub left_hand_side: VariableIndex,
    pub right_hand_side: VariableIndex,
    pub operator: ComparisonOperator,
}

impl ComparisonAxiom {
    pub fn new(
        affected_var_id: VariableIndex,
        left_hand_side: VariableIndex,
        right_hand_side: VariableIndex,
        operator: ComparisonOperator,
    ) -> Self {
        ComparisonAxiom {
            affected_var_id,
            left_hand_side,
            right_hand_side,
            operator,
        }
    }

    pub fn is_hold(&self, numeric_state: &[NumericValue]) -> bool {
        self.operator
            .compare(numeric_state, self.left_hand_side, self.right_hand_side)
    }

    pub fn affected_var_id(&self) -> VariableIndex {
        self.affected_var_id
    }
    pub fn left_var_id(&self) -> VariableIndex {
        self.left_hand_side
    }
    pub fn right_var_id(&self) -> VariableIndex {
        self.right_hand_side
    }

    pub fn operator(&self) -> &ComparisonOperator {
        &self.operator
    }
}
#[derive(Debug, Clone)]
struct AxiomRule {
    condition_count: usize,
    effect_var: VariableIndex,
    effect_value: ExplicitValueIndex,
}

impl AxiomRule {
    pub fn new(cond_count: usize, eff_var: VariableIndex, eff_val: ExplicitValueIndex) -> Self {
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
    var_id: VariableIndex,
    /// The variable's axiom default. It is both the value that says "nothing
    /// proved this variable" and the literal the closure then announces.
    default_value: ExplicitValueIndex,
}

impl NegationByFailureInfo {
    pub fn new(var_id: VariableIndex, default_value: ExplicitValueIndex) -> Self {
        NegationByFailureInfo {
            var_id,
            default_value,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct LiteralRef {
    var_id: VariableIndex,
    value: ExplicitValueIndex,
}

pub struct AxiomEvaluator {
    /// The arithmetic axioms, in dependency-layer order.
    assignment_axioms: Vec<AssignmentAxiom>,
    comparison_axioms: Vec<ComparisonAxiom>,
    state_packer: Arc<StatePacker>,
    axiom_literals: Vec<Vec<AxiomLiteral>>,
    rules: Vec<AxiomRule>,
    nbf_info_by_layer: Vec<Vec<NegationByFailureInfo>>,
    /// The variables whose value the closure starts from -- non-derived ones
    /// and comparison results -- restricted to those some rule reads.
    seed_variables: Vec<VariableIndex>,
    /// The derived propositional variables, with the default value the
    /// closure resets them to before proving them anew.
    reset_values: Vec<(VariableIndex, ExplicitValueIndex)>,
    has_numeric_axioms: bool,
    has_propositional_axioms: bool,
}

/// Scratch buffers for the propositional axiom closure.
///
/// The closure overwrites both buffers before it reads them, so one scratch
/// per thread is equivalent to one per evaluator — and keeps the evaluator
/// itself immutable, hence `Sync` and shareable across a parallel search.
#[derive(Debug)]
struct ClosureScratch {
    queue: Vec<LiteralRef>,
    unsatisfied_conditions: Vec<usize>,
}

thread_local! {
    static CLOSURE_SCRATCH: RefCell<ClosureScratch> = const {
        RefCell::new(ClosureScratch {
            queue: Vec::new(),
            unsatisfied_conditions: Vec::new(),
        })
    };
}

impl AxiomEvaluator {
    /// Compiles the task's axioms into the tables the closure runs on.
    ///
    /// # Panics
    /// Panics if a derived propositional variable sits at or below the last
    /// arithmetic layer: the closure runs after the arithmetic axioms.
    pub fn new(task: &Task, state_packer: Arc<StatePacker>) -> Self {
        let last_arithmetic_axiom_layer = task
            .numeric_variables()
            .iter()
            .filter_map(NumericVariable::axiom_layer)
            .max();
        let first_propositional_axiom_layer = task
            .variables()
            .iter()
            .filter_map(ExplicitVariable::axiom_layer)
            .min();
        let last_propositional_axiom_layer = task
            .variables()
            .iter()
            .filter_map(ExplicitVariable::axiom_layer)
            .max();
        // The comparison axioms occupy the lowest propositional layer.
        let comparison_axiom_layer = (!task.comparison_axioms().is_empty())
            .then_some(first_propositional_axiom_layer)
            .flatten();

        let mut axiom_literals: Vec<Vec<AxiomLiteral>> = task
            .variables()
            .iter()
            .map(|variable| vec![AxiomLiteral::default(); variable.domain_size()])
            .collect();
        // A rule's index is its position in `rules`, which is also the index
        // the literals it reads point back at, so both are built in one pass.
        let mut rules = Vec::with_capacity(task.axioms().len());
        for (rule_id, axiom) in task.axioms().iter().enumerate() {
            rules.push(AxiomRule::new(
                axiom.conditions().len(),
                axiom.var_id,
                axiom.effect_value,
            ));
            for condition in axiom.conditions() {
                axiom_literals[condition.var()][condition.value()]
                    .condition_of
                    .push(rule_id);
            }
        }

        let mut nbf_info_by_layer =
            vec![Vec::new(); last_propositional_axiom_layer.map_or(0, |layer| layer + 1)];
        let mut seed_variables = Vec::new();
        let mut reset_values = Vec::new();
        for (var_id, variable) in task.variables().iter().enumerate() {
            let var_index = VariableIndex::new(var_id as u32);
            let axiom_layer = variable.axiom_layer();
            if axiom_layer.is_none() || axiom_layer == comparison_axiom_layer {
                // A literal no rule reads would leave the queue without
                // effect.
                if axiom_literals[var_id]
                    .iter()
                    .any(|literal| !literal.condition_of.is_empty())
                {
                    seed_variables.push(var_index);
                }
                continue;
            }
            assert!(
                axiom_layer > last_arithmetic_axiom_layer,
                "derived variable {var_id} has axiom layer {axiom_layer:?}, not after the last \
                 arithmetic layer {last_arithmetic_axiom_layer:?}"
            );
            let default_value = variable.axiom_default_value();
            reset_values.push((var_index, default_value));
            // A condition variable is computed, not proven: the seed already
            // queues the verdict `evaluate_comparison_axioms` wrote for it, so
            // there is no "stayed at its default" case left to announce.
            // Queuing one anyway would announce the same literal twice and let
            // a Horn rule fire one condition short.
            if axiom_layer != last_propositional_axiom_layer {
                let layer = axiom_layer.expect("a reset variable is derived");
                nbf_info_by_layer[layer].push(NegationByFailureInfo::new(var_index, default_value));
            }
        }

        AxiomEvaluator {
            has_numeric_axioms: !task.assignment_axioms().is_empty()
                || !task.comparison_axioms().is_empty(),
            has_propositional_axioms: !task.axioms().is_empty(),
            assignment_axioms: task.assignment_axioms().to_vec(),
            comparison_axioms: task.comparison_axioms().to_vec(),
            state_packer,
            axiom_literals,
            rules,
            nbf_info_by_layer,
            seed_variables,
            reset_values,
        }
    }

    pub fn evaluate_arithmetic_axioms(
        &self,
        numeric_state: &mut [NumericValue],
    ) -> Result<(), AssignmentAxiomError> {
        for axiom in &self.assignment_axioms {
            axiom.update_values(numeric_state)?;
        }

        Ok(())
    }

    pub fn evaluate_comparison_axioms(&self, buffer: &mut [u64], numeric_state: &[NumericValue]) {
        for axiom in &self.comparison_axioms {
            let is_hold = axiom.is_hold(numeric_state);
            self.state_packer
                .set(buffer, axiom.affected_var_id().index(), !is_hold as u64);
        }
    }

    /// Compute the propositional axiom closure over `buffer`.
    ///
    /// Derived variables start at their default value and are proven upwards
    /// through the axiom layers: each layer runs the Horn rules to fixpoint,
    /// then admits negation-by-failure for the variables that stayed at their
    /// default, which lets the next layer use "not proven" as a condition.
    ///
    /// The rules the translator emits to *refute* a derived variable -- head at
    /// the variable's default value, for the heuristics that read axioms as
    /// relaxed operators -- are inert here, and only just. Three things make
    /// them so: `Self::derive_literal` returns early when the buffer already
    /// holds the value, `Self::fire_trivial_rules` runs before anything has
    /// been proven so an unconditional refutation only ever confirms the
    /// default, and an exact negation picks one negated literal out of every
    /// positive body, so it cannot hold once one of the rules it negates does.
    /// Nothing here therefore rewrites a proven literal back to its default.
    /// Mainline forbids such rules outright since issue454 and asserts it in
    /// this constructor; while they still arrive in the task, the property is
    /// argued rather than checked.
    pub fn evaluate_propositional_axioms(&self, buffer: &mut [u64]) {
        if !self.has_propositional_axioms {
            return;
        }

        CLOSURE_SCRATCH.with_borrow_mut(|scratch| {
            let ClosureScratch {
                queue,
                unsatisfied_conditions,
            } = scratch;
            queue.clear();
            unsatisfied_conditions.resize(self.rules.len(), 0);

            self.seed_queue_from_state(buffer, queue);
            self.fire_trivial_rules(buffer, queue, unsatisfied_conditions);

            let layer_count = self.nbf_info_by_layer.len();
            for layer_no in 0..layer_count {
                self.propagate_horn_rules(buffer, queue, unsatisfied_conditions);
                // The last layer has nothing above it that could read the
                // negation-by-failure literals, so it does not produce them.
                if layer_no + 1 != layer_count {
                    self.enqueue_unproven_literals(buffer, queue, layer_no);
                }
            }
        });
    }

    /// Queue every literal the closure may start from, and reset the derived
    /// variables it will prove to their default values.
    ///
    /// Non-derived variables and comparison results hold their value from the
    /// state; propositional derived variables do not, because a fact proven in
    /// the predecessor state need not hold here.
    #[inline]
    fn seed_queue_from_state(&self, buffer: &mut [u64], queue: &mut Vec<LiteralRef>) {
        for &var_id in &self.seed_variables {
            queue.push(LiteralRef {
                var_id,
                value: ExplicitValueIndex::new(self.state_packer.get(buffer, var_id.index()) as u32),
            });
        }
        for &(var_id, default_value) in &self.reset_values {
            self.state_packer
                .set(buffer, var_id.index(), default_value.index() as u64);
        }
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
                self.derive_literal(buffer, queue, rule.effect_var, rule.effect_value);
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
            let dependent_rules =
                &self.axiom_literals[literal.var_id.index()][literal.value.index()].condition_of;
            for &rule_index in dependent_rules {
                let remaining = &mut unsatisfied_conditions[rule_index];
                *remaining -= 1;
                if *remaining == 0 {
                    let rule = &self.rules[rule_index];
                    self.derive_literal(buffer, queue, rule.effect_var, rule.effect_value);
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
            if self.state_packer.get(buffer, info.var_id.index())
                == (info.default_value.index() as u64)
            {
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
        var_id: VariableIndex,
        value: ExplicitValueIndex,
    ) {
        if self.state_packer.get(buffer, var_id.index()) == (value.index() as u64) {
            return;
        }
        self.state_packer
            .set(buffer, var_id.index(), value.index() as u64);
        queue.push(LiteralRef { var_id, value });
    }

    /// Closes a state under all axioms: the arithmetic ones in
    /// `numeric_state`, then the derived facts in `buffer`.
    pub fn evaluate(
        &self,
        buffer: &mut [u64],
        numeric_state: &mut [NumericValue],
    ) -> Result<(), AssignmentAxiomError> {
        if self.has_numeric_axioms {
            self.evaluate_arithmetic_axioms(numeric_state)?;
        }
        self.evaluate_derived_facts(buffer, numeric_state);
        Ok(())
    }

    /// Writes the derived facts into `buffer`: the comparison results over
    /// `numeric_state`, whose derived values the arithmetic axioms must
    /// already hold, and the propositional closure.
    pub fn evaluate_derived_facts(&self, buffer: &mut [u64], numeric_state: &[NumericValue]) {
        if self.has_numeric_axioms {
            self.evaluate_comparison_axioms(buffer, numeric_state);
        }
        self.evaluate_propositional_axioms(buffer);
    }

    pub fn has_numeric_axioms(&self) -> bool {
        self.has_numeric_axioms
    }

    pub fn has_propositional_axioms(&self) -> bool {
        self.has_propositional_axioms
    }
}
