//! What kind of task this engine will accept.
//!
//! The transcription this crate builds is purely propositional: finite-domain
//! variables, conditional effects, no numeric state. Numeric tasks are rejected
//! at the boundary with a reason, never silently reinterpreted.
//!
//! The load-bearing condition is that the task has no comparison axioms.
//!
//! *Numeric-inertness.* Comparison axioms are the only channel through which the
//! numeric substate can affect anything propositional. Applicability reads only
//! `Operator::preconditions`, which are propositional facts. Propositional effect
//! firing reads only propositional conditions. In the axiom evaluator the
//! comparison pass is what turns numeric values into propositional facts, and the
//! propositional pass reads only propositional values; the assignment pass writes
//! only numeric variables. So with no comparison axioms the numeric substate is
//! write-only: the propositional projection is an exact bisimulation of the task,
//! and plans transfer between them one for one.
//!
//! That is what lets us ignore the `Cost` variable. This engine solves the
//! satisficing feasibility problem — any valid plan is acceptable and plan
//! quality is not part of the objective — so dropping a write-only cost counter
//! from the transcription is semantically correct rather than a convenient
//! default. Costs are still reported, computed by the exact verifier.

use planforge_sas::numeric_task::{AbstractNumericTask, NumericType};

/// Why a task cannot be transcribed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NotClassical {
    /// The task has compiled numeric conditions.
    NumericConditions { count: usize },
    /// The task computes derived numeric values.
    NumericArithmetic { count: usize },
    /// A numeric variable holds genuine numeric state rather than a constant or
    /// a cost counter.
    NumericStateVariable { var_id: usize, name: String },
    /// An assignment axiom writes something other than a cost variable.
    NumericAxiomWrite { affected_var_id: usize },
    /// An operator writes a numeric variable that is not a cost variable, or
    /// writes a cost variable conditionally.
    NumericEffect {
        operator: String,
        affected_var_id: usize,
        reason: NumericEffectReason,
    },
    /// A derived predicate whose truth actually depends on the state.
    ConditionedAxiom { var_id: usize, conditions: usize },
    /// An operator effect writes an axiom-derived variable, so the derived
    /// values are not state-independent and cannot be constant-folded.
    EffectOnDerivedVariable { operator: String, var_id: usize },
    /// A variable with an empty domain.
    EmptyVariableDomain { var_id: usize },
    /// A fact refers to a value outside its variable's domain.
    ValueOutOfRange {
        var_id: usize,
        value: usize,
        domain_size: usize,
    },
    /// The parser is expected to hoist an effect's `precondition_value` onto the
    /// operator's preconditions. If that ever stops happening the transcription
    /// would become weaker than the semantics it encodes, which is the one
    /// failure direction that yields wrong plans rather than missed ones.
    UnhoistedEffectPrecondition {
        operator: String,
        var_id: usize,
        value: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericEffectReason {
    /// Affects a variable that is not of type `Cost`.
    NotACostVariable,
    /// The right-hand side is not a constant.
    NonConstantOperand,
    /// A cost update guarded by conditions, so cost is not write-only.
    Conditional,
}

impl std::fmt::Display for NotClassical {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NumericConditions { count } => write!(
                f,
                "task has {count} numeric condition(s) (comparison axioms); \
                 the sgd engine supports propositional tasks only"
            ),
            Self::NumericArithmetic { count } => write!(
                f,
                "task has {count} numeric assignment axiom(s); \
                 the sgd engine supports propositional tasks only"
            ),
            Self::NumericStateVariable { var_id, name } => write!(
                f,
                "numeric variable {var_id} ({name}) holds numeric state; \
                 the sgd engine supports propositional tasks only"
            ),
            Self::NumericAxiomWrite { affected_var_id } => write!(
                f,
                "an assignment axiom writes numeric variable {affected_var_id}, \
                 which is not a cost variable"
            ),
            Self::NumericEffect {
                operator,
                affected_var_id,
                reason,
            } => {
                let detail = match reason {
                    NumericEffectReason::NotACostVariable => "which is not a cost variable",
                    NumericEffectReason::NonConstantOperand => "by a non-constant amount",
                    NumericEffectReason::Conditional => "conditionally",
                };
                write!(
                    f,
                    "operator {operator:?} writes numeric variable \
                     {affected_var_id} {detail}"
                )
            }
            Self::ConditionedAxiom { var_id, conditions } => write!(
                f,
                "variable {var_id} is derived by an axiom with {conditions} condition(s); \
                 state-dependent derived predicates are not supported yet"
            ),
            Self::EffectOnDerivedVariable { operator, var_id } => write!(
                f,
                "operator {operator:?} writes axiom-derived variable {var_id}"
            ),
            Self::EmptyVariableDomain { var_id } => {
                write!(f, "variable {var_id} has an empty domain")
            }
            Self::ValueOutOfRange {
                var_id,
                value,
                domain_size,
            } => write!(
                f,
                "value {value} is out of range for variable {var_id} \
                 (domain size {domain_size})"
            ),
            Self::UnhoistedEffectPrecondition {
                operator,
                var_id,
                value,
            } => write!(
                f,
                "operator {operator:?} has an effect requiring var{var_id}={value} \
                 that is not among its preconditions; the SAS parser is expected \
                 to hoist it"
            ),
        }
    }
}

impl std::error::Error for NotClassical {}

/// Check that `task` is a propositional task this crate can transcribe.
///
/// Returns every reason it is not, so a rejection is diagnosable in one pass
/// rather than one error at a time.
pub fn check_classical<T: AbstractNumericTask + ?Sized>(task: &T) -> Result<(), Vec<NotClassical>> {
    let mut problems = Vec::new();

    // 1. No compiled numeric conditions. This is the load-bearing condition.
    if !task.comparison_axioms().is_empty() {
        problems.push(NotClassical::NumericConditions {
            count: task.comparison_axioms().len(),
        });
    }

    // 2. No numeric arithmetic.
    if !task.assignment_axioms().is_empty() {
        problems.push(NotClassical::NumericArithmetic {
            count: task.assignment_axioms().len(),
        });
    }

    // 3. Numeric variables may only be constants or cost counters. A pure STRIPS
    //    task still has both: a `derived!1.0()` constant and `total-cost`.
    let is_cost: Vec<bool> = task
        .numeric_variables()
        .iter()
        .map(|v| matches!(v.get_type(), NumericType::Cost))
        .collect();
    let is_constant: Vec<bool> = task
        .numeric_variables()
        .iter()
        .map(|v| matches!(v.get_type(), NumericType::Constant))
        .collect();
    for (var_id, numeric) in task.numeric_variables().iter().enumerate() {
        match numeric.get_type() {
            NumericType::Constant | NumericType::Cost => {}
            NumericType::Regular | NumericType::Derived => {
                problems.push(NotClassical::NumericStateVariable {
                    var_id,
                    name: numeric.name().to_string(),
                });
            }
        }
    }

    // 4. Assignment axioms and effects may only touch cost variables, by a
    //    constant, unconditionally. Note the check is on the *affected*
    //    variable: rejecting all assignment effects would reject every task,
    //    because each operator increments `total-cost`.
    for axiom in task.assignment_axioms() {
        let affected = axiom.get_affected_var_id();
        if !is_cost.get(affected).copied().unwrap_or(false) {
            problems.push(NotClassical::NumericAxiomWrite {
                affected_var_id: affected,
            });
        }
    }
    for operator in task.get_operators() {
        for effect in operator.assignment_effects() {
            let affected = effect.affected_var_id();
            if !is_cost.get(affected).copied().unwrap_or(false) {
                problems.push(NotClassical::NumericEffect {
                    operator: operator.name().to_string(),
                    affected_var_id: affected,
                    reason: NumericEffectReason::NotACostVariable,
                });
            } else if !is_constant.get(effect.var_id()).copied().unwrap_or(false) {
                problems.push(NotClassical::NumericEffect {
                    operator: operator.name().to_string(),
                    affected_var_id: affected,
                    reason: NumericEffectReason::NonConstantOperand,
                });
            } else if effect.is_conditional() || !effect.conditions().is_empty() {
                problems.push(NotClassical::NumericEffect {
                    operator: operator.name().to_string(),
                    affected_var_id: affected,
                    reason: NumericEffectReason::Conditional,
                });
            }
        }
    }

    // 5. Derived predicates must be state-independent, so that they can be
    //    constant-folded. The translator injects one unconditional axiom into
    //    every task, so a task with *no* axioms does not exist and rejecting
    //    non-empty `axioms()` would reject everything.
    for axiom in task.axioms() {
        if !axiom.conditions().is_empty() {
            problems.push(NotClassical::ConditionedAxiom {
                var_id: axiom.var_id(),
                conditions: axiom.conditions().len(),
            });
        }
    }
    for operator in task.get_operators() {
        for effect in operator.effects() {
            let derived = task
                .get_variable_axiom_layer(effect.var_id())
                .ok()
                .flatten()
                .is_some();
            if derived {
                problems.push(NotClassical::EffectOnDerivedVariable {
                    operator: operator.name().to_string(),
                    var_id: effect.var_id(),
                });
            }
        }
    }

    // 6. Structural sanity: domains non-empty, every referenced value in range,
    //    and the parser's hoisting invariant.
    let domain_size = |var_id: usize| task.get_variable_domain_size(var_id).ok();
    for var_id in 0..task.get_num_variables() {
        if domain_size(var_id) == Some(0) {
            problems.push(NotClassical::EmptyVariableDomain { var_id });
        }
    }
    let check_fact = |var_id: usize, value: usize, problems: &mut Vec<NotClassical>| {
        if let Some(size) = domain_size(var_id)
            && value >= size
        {
            problems.push(NotClassical::ValueOutOfRange {
                var_id,
                value,
                domain_size: size,
            });
        }
    };
    for index in 0..task.get_num_goals() {
        let goal = task.get_goal_fact(index);
        check_fact(goal.var(), goal.value(), &mut problems);
    }
    for operator in task.get_operators() {
        for fact in operator.preconditions() {
            check_fact(fact.var(), fact.value(), &mut problems);
        }
        for effect in operator.effects() {
            check_fact(effect.var_id(), effect.value(), &mut problems);
            for fact in effect.conditions() {
                check_fact(fact.var(), fact.value(), &mut problems);
            }
            if let Some(required) = effect.precondition_value() {
                check_fact(effect.var_id(), required, &mut problems);
                let hoisted = operator
                    .preconditions()
                    .iter()
                    .any(|pre| pre.var() == effect.var_id() && pre.value() == required);
                if !hoisted {
                    problems.push(NotClassical::UnhoistedEffectPrecondition {
                        operator: operator.name().to_string(),
                        var_id: effect.var_id(),
                        value: required,
                    });
                }
            }
        }
    }

    if problems.is_empty() {
        Ok(())
    } else {
        Err(problems)
    }
}
