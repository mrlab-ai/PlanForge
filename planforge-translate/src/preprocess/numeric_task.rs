//! Building the search's task from the reordered one, without going through
//! text.
//!
//! The mirror image of [`crate::preprocess::output`]: the same reordered task,
//! read through the same level mapping and the same token tables, phrased as the
//! [`NumericRootTask`] the search takes rather than as the SAS+ file another
//! planner takes. What the format leaves implicit is established by
//! [`NumericRootTask::from_sas_parts`], which the file's reader goes through as
//! well. That is what keeps the two ways in from drifting apart, and
//! `tests/src/task_equivalence_tests` is what holds them to it.

use planforge_sas::axioms::{
    AssignmentAxiom, CalOperator, ComparisonAxiom, ComparisonOperator, PropositionalAxiom,
};
use planforge_sas::numeric_task::{
    AssignmentEffect, AssignmentOperation, Effect, ExplicitFact, Metric, NumericRootTask,
    NumericVariable, Operator,
};
use planforge_sas::sas_format::{
    SAS_FILE_VERSION, SasTaskParts, SasVariable, axiom_layer_from_sas,
    effect_precondition_from_sas, operator_cost_from_sas,
};
use tracing::info;

use crate::preprocess::{PreprocessedTask, ReorderedTask};
use crate::sas_tasks::{SasFact, assignment_operator};

pub fn build(reordered: &ReorderedTask) -> NumericRootTask {
    let ReorderedTask {
        task: PreprocessedTask { sas, metric, .. },
        prop_order,
        numeric_order,
    } = reordered;

    info!("Building the search task...");

    let fact = |pair: &SasFact| {
        let (var, value) = reordered.prop_fact(pair);
        ExplicitFact::propositional(var, value)
    };
    let facts = |pairs: &[SasFact]| pairs.iter().map(fact).collect();

    let variables = prop_order
        .iter()
        .map(|&var| SasVariable {
            domain_size: sas.variables.ranges[var],
            // The file names a variable after the id it had before the
            // reordering, and the reader takes the name from the file.
            name: format!("var{var}"),
            fact_names: sas.variables.value_names[var].clone(),
            axiom_layer: axiom_layer_from_sas(sas.variables.axiom_layers[var]),
        })
        .collect();

    let numeric_variables = numeric_order
        .iter()
        .map(|&var| {
            let (numeric_type, layer, name) = reordered.numeric_variable(var);
            // The name is written as a line of its own and read back trimmed,
            // so a name with padding denotes the trimmed one either way.
            NumericVariable::new(
                name.trim().to_string(),
                numeric_type,
                axiom_layer_from_sas(layer),
            )
        })
        .collect();

    let state = prop_order
        .iter()
        .map(|&var| {
            let value = sas.init.values[var];
            usize::try_from(value)
                .unwrap_or_else(|_| panic!("variable {var} has no initial value (got {value})"))
        })
        .collect();
    let numeric_state = numeric_order
        .iter()
        .map(|&var| sas.init.num_values[var])
        .collect();

    let operators = sas
        .operators
        .iter()
        .map(|op| build_operator(reordered, op))
        .collect();

    let axioms = sas
        .axioms
        .iter()
        .map(|axiom| {
            let (effect_var, effect_value) = axiom.effect;
            // A derived variable is binary and defaults to the value the axiom
            // does not derive, so the rule requires the value it overwrites.
            PropositionalAxiom::new(
                facts(&axiom.condition),
                reordered.prop_level(effect_var),
                1 - effect_value,
                effect_value,
            )
        })
        .collect();

    let comparison_axioms = sas
        .comp_axioms
        .iter()
        .map(|axiom| {
            ComparisonAxiom::new(
                reordered.prop_level(axiom.effect),
                reordered.numeric_level(axiom.parts[0]),
                reordered.numeric_level(axiom.parts[1]),
                ComparisonOperator::from_sas(&axiom.comp)
                    .unwrap_or_else(|| panic!("{:?} is not a comparator", axiom.comp)),
            )
        })
        .collect();

    let assignment_axioms = sas
        .numeric_axioms
        .iter()
        .map(|axiom| {
            let operator = assignment_operator(&axiom.op);
            AssignmentAxiom::new(
                reordered.numeric_level(axiom.effect),
                CalOperator::from_sas(operator).unwrap_or_else(|| {
                    panic!("{operator:?} does not combine a numeric axiom's operands")
                }),
                reordered.numeric_level(axiom.parts[0]),
                reordered.numeric_level(axiom.parts[1]),
            )
        })
        .collect();

    let task = NumericRootTask::from_sas_parts(SasTaskParts {
        version: SAS_FILE_VERSION,
        metric: Metric::from_sas(metric.optimization_criterion, metric.index).unwrap_or_else(
            || {
                panic!(
                    "{:?} is not an optimization criterion",
                    metric.optimization_criterion
                )
            },
        ),
        variables,
        numeric_variables,
        mutexes: sas
            .mutexes
            .iter()
            .map(|mutex| facts(&mutex.facts))
            .collect(),
        state,
        numeric_state,
        goals: reordered
            .ordered_goal()
            .iter()
            .map(|&(var, value)| ExplicitFact::propositional(var, value))
            .collect(),
        operators,
        axioms,
        comparison_axioms,
        assignment_axioms,
        global_constraint: fact(&sas.global_constraint),
    });
    info!("done");
    task
}

fn build_operator(reordered: &ReorderedTask, op: &crate::sas_tasks::SASOperator) -> Operator {
    let fact = |pair: &SasFact| {
        let (var, value) = reordered.prop_fact(pair);
        ExplicitFact::propositional(var, value)
    };

    // An effect that requires a value of the variable it writes contributes
    // that requirement to the operator, after the prevail conditions. The
    // order is the order the file lists them in, and an operator's
    // preconditions are held in that order rather than sorted.
    let mut preconditions: Vec<ExplicitFact> = op.prevail.iter().map(fact).collect();
    let mut effects = Vec::with_capacity(op.pre_post.len());
    for (var, precondition_field, effect_value, conditions) in &op.pre_post {
        let var_id = reordered.prop_level(*var);
        let precondition_value = effect_precondition_from_sas(*precondition_field);
        if let Some(precondition_value) = precondition_value {
            preconditions.push(ExplicitFact::propositional(var_id, precondition_value));
        }
        effects.push(Effect::new(
            conditions.iter().map(fact).collect(),
            var_id,
            precondition_value,
            *effect_value,
        ));
    }

    // An assignment effect's own variables are numeric; only its guard names
    // propositional ones.
    let assignment_effects = op
        .assign_effects
        .iter()
        .map(|(var, operator, operand, conditions)| {
            let operator = assignment_operator(operator);
            AssignmentEffect::new(
                reordered.numeric_level(*var),
                AssignmentOperation::from_sas(operator)
                    .unwrap_or_else(|| panic!("{operator:?} is not an assignment operator")),
                reordered.numeric_level(*operand),
                !conditions.is_empty(),
                conditions.iter().map(fact).collect(),
            )
        })
        .collect();

    Operator::new(
        op.output_name().to_string(),
        preconditions,
        effects,
        assignment_effects,
        operator_cost_from_sas(op.cost),
    )
}
