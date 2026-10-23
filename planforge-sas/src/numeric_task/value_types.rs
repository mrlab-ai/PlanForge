use std::{
    fmt::{self, Display},
    hash::Hash,
};

use super::value::{NumericValue, ZERO_VALUE};

pub const ZERO_OP_COST: OperatorCost = OperatorCost(0);

#[derive(Debug, Clone, PartialEq)]
pub struct Metric {
    is_min: bool,
    var_id: Option<VariableIndex>,
}

impl Metric {
    pub fn new(is_min: bool, var_id: Option<VariableIndex>) -> Self {
        Metric { is_min, var_id }
    }

    pub fn is_min(&self) -> bool {
        self.is_min
    }

    pub fn var_id(&self) -> Option<VariableIndex> {
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
    pub(super) axiom_default_value: ExplicitValueIndex,
}

impl ExplicitVariable {
    pub fn new(
        domain_size: usize,
        name: String,
        fact_names: Vec<String>,
        axiom_layer: Option<usize>,
        axiom_default_value: ExplicitValueIndex,
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

/// Defines a `u32`-backed index newtype.
///
/// Indices are stored as `u32` to halve their footprint on 64-bit targets.
/// [`from_usize`](VariableIndex::from_usize) is the one narrowing point, and it
/// panics instead of truncating: an index past `u32::MAX` would otherwise alias
/// another entity.
macro_rules! u32_index {
    ($(#[$meta:meta])* $name:ident) => {
        $(#[$meta])*
        #[derive(Clone, Copy, Debug, Default, Eq, Ord, PartialEq, PartialOrd, Hash)]
        pub struct $name(u32);

        impl $name {
            #[inline(always)]
            pub const fn new(index: u32) -> Self {
                Self(index)
            }

            #[inline(always)]
            pub const fn from_usize(index: usize) -> Self {
                assert!(
                    index <= u32::MAX as usize,
                    concat!(stringify!($name), " exceeds u32::MAX")
                );
                Self(index as u32)
            }

            #[inline(always)]
            pub const fn index(&self) -> usize {
                self.0 as usize
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}", self.0)
            }
        }
    };
}

u32_index!(
    /// Index of a variable in its id space.
    VariableIndex
);

u32_index!(
    /// Index of a value in the domain of a variable.
    ExplicitValueIndex
);

u32_index!(
    /// Index of an axiom.
    AxiomIndex
);

u32_index!(
    /// Index of an operator.
    OperatorIndex
);

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
    packed_var: u32,
    value_id: ExplicitValueIndex,
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
        ExplicitFact {
            packed_var: (namespace as u32) << Self::VAR_BITS | var as u32,
            value_id: ExplicitValueIndex::from_usize(value),
        }
    }

    /// The same fact re-tagged. Used by the one pass that owns namespace
    /// assignment, `NumericRootTask::assign_fact_namespaces`.
    #[inline]
    #[must_use]
    pub fn with_namespace(self, namespace: FactNamespace) -> Self {
        ExplicitFact {
            packed_var: (namespace as u32) << Self::VAR_BITS | (self.packed_var & Self::VAR_MASK),
            value_id: self.value_id,
        }
    }

    #[inline(always)]
    pub fn namespace(&self) -> FactNamespace {
        FactNamespace::from_tag(self.packed_var >> Self::VAR_BITS)
    }

    #[inline(always)]
    pub fn is_condition(&self) -> bool {
        self.namespace() == FactNamespace::Condition
    }

    #[inline(always)]
    pub const fn var(&self) -> usize {
        (self.packed_var & Self::VAR_MASK) as usize
    }
    #[inline(always)]
    pub const fn value(&self) -> usize {
        self.value_id.index()
    }
    #[inline(always)]
    pub const fn var_index(&self) -> VariableIndex {
        VariableIndex(self.packed_var & Self::VAR_MASK)
    }
    #[inline(always)]
    pub const fn value_index(&self) -> ExplicitValueIndex {
        self.value_id
    }

    /// Whether the fact holds in the propositional values `propositional`,
    /// which cover every variable.
    #[inline]
    pub fn holds_in(&self, propositional: &[ExplicitValueIndex]) -> bool {
        propositional[self.var()] == self.value_id
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
    var_id: VariableIndex,
    precondition_value: Option<ExplicitValueIndex>,
    effect_value: ExplicitValueIndex,
}

impl Effect {
    pub fn new(
        conditions: Vec<ExplicitFact>,
        var_id: VariableIndex,
        precondition_value: Option<ExplicitValueIndex>,
        effect_value: ExplicitValueIndex,
    ) -> Self {
        Effect {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
    }

    pub fn var_id(&self) -> VariableIndex {
        self.var_id
    }

    pub fn precondition_value(&self) -> Option<ExplicitValueIndex> {
        self.precondition_value
    }

    pub fn conditions(&self) -> &Vec<ExplicitFact> {
        &self.conditions
    }

    pub fn value(&self) -> ExplicitValueIndex {
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
    pub fn apply(
        left: NumericValue,
        operation: &AssignmentOperation,
        right: NumericValue,
    ) -> NumericValue {
        match operation {
            AssignmentOperation::Assign => right,
            AssignmentOperation::Plus => left + right,
            AssignmentOperation::Minus => left - right,
            AssignmentOperation::Times => left * right,
            AssignmentOperation::Divide => {
                if right == ZERO_VALUE {
                    panic!("Division by zero is not allowed");
                }
                left / right
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct AssignmentEffect {
    pub(super) affected_var_id: VariableIndex,
    operation: AssignmentOperation,
    pub(super) var_id: VariableIndex,
    is_conditional: bool,
    pub(super) conditions: Vec<ExplicitFact>,
}

impl AssignmentEffect {
    pub fn new(
        affected_var_id: VariableIndex,
        operation: AssignmentOperation,
        var_id: VariableIndex,
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

    pub fn affected_var_id(&self) -> VariableIndex {
        self.affected_var_id
    }
    pub fn var_id(&self) -> VariableIndex {
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

/// Struct used to avoid confusions with other numeric values, representing
/// the cost of an operator.
#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd, Hash)]
pub struct OperatorCost(u64);

impl OperatorCost {
    #[inline(always)]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    #[inline(always)]
    pub const fn from_usize(value: usize) -> Self {
        // Lossless: `usize` is at most 64 bits on every supported target.
        Self(value as u64)
    }

    #[inline(always)]
    pub const fn value(&self) -> u64 {
        self.0
    }
}

impl Display for OperatorCost {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Operator {
    name: Box<str>,
    pub(super) preconditions: Vec<ExplicitFact>,
    pub(super) effects: Vec<Effect>,
    pub(super) assignment_effects: Vec<AssignmentEffect>,
    repeated_assignment_targets: Box<[RepeatedTarget]>,
    cost: OperatorCost,
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
        cost: OperatorCost,
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
                        "operator {name} writes numeric variable {} more than once \
                         with a non-additive assignment, which has no order-independent result",
                        affected_var_id.index()
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

    /// Writes the successor of the explicit state (`propositional`,
    /// `numeric`) under this operator into `successor_propositional` and
    /// `successor_numeric`, before axioms.
    ///
    /// Effects apply simultaneously, as in
    /// [`StateRegistry`](crate::state_registry::StateRegistry): effect
    /// conditions and assignment operands read the parent state, and further
    /// additive writes to one variable add up. Preconditions are the caller's
    /// to check.
    pub fn apply_to_explicit_state(
        &self,
        propositional: &[ExplicitValueIndex],
        numeric: &[NumericValue],
        successor_propositional: &mut Vec<ExplicitValueIndex>,
        successor_numeric: &mut Vec<NumericValue>,
    ) {
        let holds = |conditions: &[ExplicitFact]| {
            conditions
                .iter()
                .all(|condition| condition.holds_in(propositional))
        };
        successor_propositional.clear();
        successor_propositional.extend_from_slice(propositional);
        successor_numeric.clear();
        successor_numeric.extend_from_slice(numeric);
        for effect in &self.effects {
            if holds(effect.conditions()) {
                successor_propositional[effect.var_id().index()] = effect.value();
            }
        }
        for (effect, repeated_target) in self
            .assignment_effects
            .iter()
            .zip(&self.repeated_assignment_targets)
        {
            if !holds(effect.conditions()) {
                continue;
            }
            let affected = effect.affected_var_id().index();
            let left = match repeated_target {
                RepeatedTarget::First => numeric[affected],
                RepeatedTarget::Additive => successor_numeric[affected],
            };
            successor_numeric[affected] = AssignmentOperation::apply(
                left,
                effect.operation(),
                numeric[effect.var_id().index()],
            );
        }
    }

    pub fn preconditions(&self) -> &Vec<ExplicitFact> {
        &self.preconditions
    }

    pub fn cost(&self) -> OperatorCost {
        self.cost
    }
}
