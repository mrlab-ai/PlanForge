use std::{fmt, hash::Hash};

#[derive(Debug, Clone, PartialEq)]
pub struct Metric {
    is_min: bool,
    var_id: Option<usize>,
}

impl Metric {
    pub fn new(is_min: bool, var_id: Option<usize>) -> Self {
        Metric { is_min, var_id }
    }

    pub fn is_min(&self) -> bool {
        self.is_min
    }

    pub fn var_id(&self) -> Option<usize> {
        self.var_id
    }

    pub fn use_metric(&self) -> bool {
        self.var_id.is_some()
    }
}

#[allow(unused)]
#[derive(Debug, Clone, PartialEq)]
pub struct ExplicitVariable {
    pub(super) domain_size: usize,
    pub(super) name: String,
    pub(super) fact_names: Vec<String>,
    pub(super) axiom_layer: Option<usize>,
    /// The value a *derived* variable holds until an axiom proves otherwise.
    ///
    /// The axiom closure resets every derived variable to this value before it
    /// runs, and reads "still at the default" as "not proven" when it admits
    /// negation-by-failure literals. The SAS format carries it in the
    /// initial-state block, which is why the parser can only fill it in once
    /// that block has been read.
    ///
    /// A non-derived variable is never reset, so this field is not read for
    /// one; it then just repeats the variable's initial value.
    pub(super) axiom_default_value: usize,
}

impl ExplicitVariable {
    pub fn new(
        domain_size: usize,
        name: String,
        fact_names: Vec<String>,
        axiom_layer: Option<usize>,
        axiom_default_value: usize,
    ) -> Self {
        ExplicitVariable {
            domain_size,
            name,
            fact_names,
            axiom_layer,
            axiom_default_value,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn axiom_layer(&self) -> Option<usize> {
        self.axiom_layer
    }

    pub fn with_axiom_layer(&self, axiom_layer: Option<usize>) -> Self {
        Self {
            axiom_layer,
            ..self.clone()
        }
    }

    pub fn domain_size(&self) -> usize {
        self.domain_size
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NumericVariable {
    name: String,
    numeric_type: NumericType,
    axiom_layer: Option<usize>,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum NumericType {
    Constant,
    Derived,
    Cost,
    Regular,
}

impl NumericVariable {
    pub fn new(name: String, numeric_type: NumericType, axiom_layer: Option<usize>) -> Self {
        NumericVariable {
            name,
            numeric_type,
            axiom_layer,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn get_type(&self) -> &NumericType {
        &self.numeric_type
    }

    pub fn axiom_layer(&self) -> Option<usize> {
        self.axiom_layer
    }
}

/// Which id space a fact's variable belongs to.
///
/// A propositional variable, a numeric condition and a domain abstraction's
/// numeric variable are different kinds of thing that happen to share the
/// `(variable, value)` shape, so a fact carries the answer instead of leaving
/// callers to rediscover it from
/// [`NumericConditions::is_condition_var`](crate::numeric_conditions::NumericConditions::is_condition_var) or
/// from an unlabelled `num_propositional_vars + numeric_var_id` offset.
#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
#[repr(u32)]
pub enum FactNamespace {
    /// The fact's variable is a genuine propositional variable.
    Propositional = 0,
    /// The fact's variable carries the truth value of a numeric condition.
    Condition = 1,
    /// The fact's variable is a *numeric* variable, addressed in a domain
    /// abstraction's own id space: propositional variables occupy
    /// `[0, num_propositional_vars)` and numeric variables follow them, so the
    /// id is `num_propositional_vars + numeric_var_id` and the value is a
    /// partition index rather than a propositional value.
    ///
    /// That id space is private to the abstraction machinery. No fact a task
    /// exposes may carry this tag, which is what
    /// [`assert_fact_namespaces`](crate::numeric_task::assert_fact_namespaces) checks.
    NumericVariable = 2,
}

impl FactNamespace {
    /// Number of tag bits [`ExplicitFact`] reserves in its variable id.
    const BITS: u32 = 4;

    #[inline(always)]
    const fn from_tag(tag: u32) -> Self {
        match tag {
            0 => FactNamespace::Propositional,
            1 => FactNamespace::Condition,
            2 => FactNamespace::NumericVariable,
            // Only `ExplicitFact`'s constructors write the tag, and they only
            // ever write a discriminant of this enum.
            _ => unreachable!(),
        }
    }
}

/// Variable/value pair, tagged with the variable's [`FactNamespace`].
///
/// `u32` fields halve the storage per fact compared to `usize` on 64-bit
/// targets (16 B → 8 B). The namespace occupies the top
/// `FactNamespace::BITS` bits of the variable id, leaving a hard
/// [`Self::MAX_VAR_ID`] ceiling — vastly above anything realistic planning
/// tasks reach, and checked at construction.
///
/// The tag lives in the *high* bits on purpose: [`Self::var`] is a plain mask
/// of the low bits, so every existing var-major sort and dense per-variable
/// sweep keeps its meaning.
#[derive(Clone, Copy)]
pub struct ExplicitFact {
    /// `namespace << (32 - FactNamespace::BITS) | var`.
    var_id: u32,
    value_id: u32,
}

impl ExplicitFact {
    const VAR_BITS: u32 = u32::BITS - FactNamespace::BITS;
    const VAR_MASK: u32 = (1 << Self::VAR_BITS) - 1;

    /// Largest variable id a fact can name.
    pub const MAX_VAR_ID: usize = Self::VAR_MASK as usize;

    /// Fact on a genuine propositional variable.
    #[inline]
    pub fn propositional(var: usize, value: usize) -> Self {
        Self::in_namespace(FactNamespace::Propositional, var, value)
    }

    /// Fact on the variable carrying a numeric condition's truth value.
    #[inline]
    pub fn condition(var: usize, value: usize) -> Self {
        Self::in_namespace(FactNamespace::Condition, var, value)
    }

    /// Fact on a numeric variable in a domain abstraction's id space.
    ///
    /// `abstraction_var` is the abstraction id, not the numeric variable id;
    /// see [`FactNamespace::NumericVariable`] for the offset encoding. `value`
    /// is a partition index of that numeric variable.
    #[inline]
    pub fn numeric_variable(abstraction_var: usize, value: usize) -> Self {
        Self::in_namespace(FactNamespace::NumericVariable, abstraction_var, value)
    }

    /// Constructors accept `usize` to minimize call-site churn; values are
    /// narrowed here, and out-of-range arguments fail rather than silently
    /// aliasing another variable or bleeding into the namespace tag.
    pub fn in_namespace(namespace: FactNamespace, var: usize, value: usize) -> Self {
        assert!(
            var <= Self::MAX_VAR_ID,
            "ExplicitFact var {var} exceeds the {} packed variable-id bits",
            Self::VAR_BITS
        );
        debug_assert!(
            value <= u32::MAX as usize,
            "ExplicitFact value {value} > u32::MAX"
        );
        ExplicitFact {
            var_id: (namespace as u32) << Self::VAR_BITS | var as u32,
            value_id: value as u32,
        }
    }

    /// The same fact re-tagged. Used by the one pass that owns namespace
    /// assignment, `NumericRootTask::assign_fact_namespaces`.
    #[inline]
    #[must_use]
    pub fn with_namespace(self, namespace: FactNamespace) -> Self {
        ExplicitFact {
            var_id: (namespace as u32) << Self::VAR_BITS | (self.var_id & Self::VAR_MASK),
            value_id: self.value_id,
        }
    }

    #[inline(always)]
    pub fn namespace(&self) -> FactNamespace {
        FactNamespace::from_tag(self.var_id >> Self::VAR_BITS)
    }

    #[inline(always)]
    pub fn is_condition(&self) -> bool {
        self.namespace() == FactNamespace::Condition
    }

    #[inline(always)]
    pub fn var(&self) -> usize {
        (self.var_id & Self::VAR_MASK) as usize
    }
    #[inline(always)]
    pub fn value(&self) -> usize {
        self.value_id as usize
    }
}

/// Identity, ordering and hashing are all over `(variable, value)` and ignore
/// the namespace tag.
///
/// The tag is a function of the variable, so dropping it loses nothing — and
/// it means the tag can never split one fact into two that compare unequal but
/// order equal, which would quietly break `sort` + `dedup` pairs and mutex
/// lookups. Ordering stays variable-major, which is what the successor
/// generator's decision tree is built from.
impl PartialEq for ExplicitFact {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value_id == other.value_id && self.var() == other.var()
    }
}

impl Eq for ExplicitFact {}

impl std::hash::Hash for ExplicitFact {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.var().hash(state);
        self.value_id.hash(state);
    }
}

impl Ord for ExplicitFact {
    #[inline]
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.var(), self.value_id).cmp(&(other.var(), other.value_id))
    }
}

impl PartialOrd for ExplicitFact {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl fmt::Debug for ExplicitFact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.namespace() {
            FactNamespace::Propositional => {
                write!(f, "Fact(var: {}, value: {})", self.var(), self.value())
            }
            FactNamespace::Condition => {
                write!(f, "Fact(cond: {}, value: {})", self.var(), self.value())
            }
            FactNamespace::NumericVariable => {
                write!(f, "Fact(num: {}, partition: {})", self.var(), self.value())
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Effect {
    pub(super) conditions: Vec<ExplicitFact>,
    var_id: usize,
    precondition_value: Option<usize>,
    effect_value: usize,
}

impl Effect {
    pub fn new(
        conditions: Vec<ExplicitFact>,
        var_id: usize,
        precondition_value: Option<usize>,
        effect_value: usize,
    ) -> Self {
        Effect {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
    }

    pub fn var_id(&self) -> usize {
        self.var_id
    }

    pub fn precondition_value(&self) -> Option<usize> {
        self.precondition_value
    }

    pub fn conditions(&self) -> &Vec<ExplicitFact> {
        &self.conditions
    }

    pub fn value(&self) -> usize {
        self.effect_value
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum AssignmentOperation {
    Assign,
    Plus,
    Minus,
    Times,
    Divide,
}

impl AssignmentOperation {
    pub fn apply(left: f64, operation: &AssignmentOperation, right: f64) -> f64 {
        match operation {
            AssignmentOperation::Assign => right,
            AssignmentOperation::Plus => left + right,
            AssignmentOperation::Minus => left - right,
            AssignmentOperation::Times => left * right,
            AssignmentOperation::Divide => {
                if right == 0.0 {
                    panic!("Division by zero is not allowed");
                }
                left / right
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentEffect {
    affected_var_id: usize,
    operation: AssignmentOperation,
    var_id: usize,
    is_conditional: bool,
    pub(super) conditions: Vec<ExplicitFact>,
}

impl AssignmentEffect {
    pub fn new(
        affected_var_id: usize,
        operation: AssignmentOperation,
        var_id: usize,
        is_conditional: bool,
        conditions: Vec<ExplicitFact>,
    ) -> Self {
        AssignmentEffect {
            affected_var_id,
            operation,
            var_id,
            is_conditional,
            conditions,
        }
    }

    pub fn affected_var_id(&self) -> usize {
        self.affected_var_id
    }
    pub fn var_id(&self) -> usize {
        self.var_id
    }

    pub fn operation(&self) -> &AssignmentOperation {
        &self.operation
    }

    pub fn is_conditional(&self) -> bool {
        self.is_conditional
    }

    pub fn conditions(&self) -> &Vec<ExplicitFact> {
        &self.conditions
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Operator {
    name: Box<str>,
    pub(super) preconditions: Vec<ExplicitFact>,
    pub(super) effects: Vec<Effect>,
    pub(super) assignment_effects: Vec<AssignmentEffect>,
    repeated_assignment_targets: Box<[RepeatedTarget]>,
    cost: u64,
}

/// Whether an assignment effect is the first write to its target within its
/// operator or a further additive write.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RepeatedTarget {
    First,
    Additive,
}

impl Operator {
    pub fn new(
        name: String,
        preconditions: Vec<ExplicitFact>,
        effects: Vec<Effect>,
        assignment_effects: Vec<AssignmentEffect>,
        cost: u64,
    ) -> Self {
        let mut repeated_assignment_targets = Vec::with_capacity(assignment_effects.len());
        let mut target_is_additive = std::collections::HashMap::new();
        for effect in &assignment_effects {
            let affected_var_id = effect.affected_var_id();
            let is_additive = matches!(
                effect.operation(),
                AssignmentOperation::Plus | AssignmentOperation::Minus
            );
            match target_is_additive.get_mut(&affected_var_id) {
                None => {
                    target_is_additive.insert(affected_var_id, is_additive);
                    repeated_assignment_targets.push(RepeatedTarget::First);
                }
                Some(previous_are_additive) if *previous_are_additive && is_additive => {
                    repeated_assignment_targets.push(RepeatedTarget::Additive);
                }
                Some(_) => {
                    panic!(
                        "operator {name} writes numeric variable {affected_var_id} more than once \
                         with a non-additive assignment, which has no order-independent result"
                    );
                }
            }
        }

        // `Box<str>` is two words (ptr + len) vs `String`'s three words
        // (ptr + len + cap) and drops spare capacity from any growth steps
        // during parsing. For tasks with 10^6 operators this trims the
        // task-loading peak by 20-30 MB. Names are immutable so we never
        // need the `cap` field again.
        Operator {
            name: name.into_boxed_str(),
            preconditions,
            effects,
            assignment_effects,
            repeated_assignment_targets: repeated_assignment_targets.into_boxed_slice(),
            cost,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn effects(&self) -> &Vec<Effect> {
        &self.effects
    }

    pub fn assignment_effects(&self) -> &Vec<AssignmentEffect> {
        &self.assignment_effects
    }

    pub(crate) fn repeated_assignment_targets(&self) -> &[RepeatedTarget] {
        &self.repeated_assignment_targets
    }

    pub fn preconditions(&self) -> &Vec<ExplicitFact> {
        &self.preconditions
    }

    pub fn cost(&self) -> u64 {
        self.cost
    }
}
