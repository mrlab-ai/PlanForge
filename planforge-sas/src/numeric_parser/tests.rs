use super::*;
use crate::numeric_conditions::ConditionValue;
use crate::numeric_task::{AbstractNumericTask, OperatorCost};

/// One assignment effect guarded by a single condition: `if var5 == 1 then
/// var3 += var2`. Threading the input incorrectly through the condition
/// loop silently yields `var1 += var5` here, which is why this pins every
/// field rather than just the condition.
#[test]
fn conditional_assignment_effect_parses_all_fields() {
    let input = "begin_operator\nmove\n0\n0\n1\n1 5 1 3 + 2\n7\nend_operator\n";

    let (rest, operator) = parse_operator(input).expect("operator parses");

    assert_eq!(rest, "");
    assert_eq!(operator.name, "move");
    assert_eq!(operator.cost, OperatorCost::new(7));

    let effects = &operator.assignment_effects;
    assert_eq!(effects.len(), 1);
    let effect = &effects[0];
    assert!(effect.is_conditional());
    assert_eq!(
        effect.conditions(),
        &vec![ExplicitFact::propositional(5, 1)]
    );
    assert_eq!(effect.affected_var_id(), VariableIndex::new(3));
    assert_eq!(effect.operation(), &AssignmentOperation::Plus);
    assert_eq!(effect.var_id(), VariableIndex::new(2));
}

/// Two conditions, to catch an off-by-one in how the loop advances.
#[test]
fn multi_condition_assignment_effect_parses_all_conditions() {
    let input = "begin_operator\nmove\n0\n0\n1\n2 5 1 6 0 3 + 2\n7\nend_operator\n";

    let (_, operator) = parse_operator(input).expect("operator parses");

    let effect = &operator.assignment_effects[0];
    assert_eq!(
        effect.conditions(),
        &vec![
            ExplicitFact::propositional(5, 1),
            ExplicitFact::propositional(6, 0)
        ]
    );
    assert_eq!(effect.affected_var_id(), VariableIndex::new(3));
    assert_eq!(effect.var_id(), VariableIndex::new(2));
}

/// The unconditional case must keep working unchanged.
#[test]
fn unconditional_assignment_effect_parses() {
    let input = "begin_operator\nmove\n0\n0\n1\n0 3 + 2\n7\nend_operator\n";

    let (_, operator) = parse_operator(input).expect("operator parses");

    let effect = &operator.assignment_effects[0];
    assert!(!effect.is_conditional());
    assert!(effect.conditions().is_empty());
    assert_eq!(effect.affected_var_id(), VariableIndex::new(3));
    assert_eq!(effect.var_id(), VariableIndex::new(2));
}

/// A task whose numeric conditions are *interleaved* with its genuine
/// propositional variables, which is how a SAS file writes them: `cond_ge`
/// on var1 and `cond_lt` on var3, with `a`, `b`, `c` and the derived `d`
/// around them. Every place a propositional variable id can hide is used
/// exactly once, so a site that reads an id from the wrong place shows up
/// here.
///
/// Numerically: `x = 5`, `three = 3`, so `cond_ge` (`x >= three`) holds and
/// `cond_lt` (`x < three`) does not. Both condition variables are written in
/// the legacy three-valued form, `<none of those>` and all, so that building
/// a task out of this file exercises the narrowing to
/// [`ConditionValue::DOMAIN_SIZE`].
const INTERLEAVED_CONDITIONS_SAS: &str = "\
begin_version
4
end_version
begin_metric
< 2
end_metric
6
begin_variable
var0
-1
2
Atom a()
NegatedAtom a()
end_variable
begin_variable
var1
0
3
Atom cond_ge()
NegatedAtom cond_ge()
<none of those>
end_variable
begin_variable
var2
-1
2
Atom b()
NegatedAtom b()
end_variable
begin_variable
var3
0
3
Atom cond_lt()
NegatedAtom cond_lt()
<none of those>
end_variable
begin_variable
var4
-1
2
Atom c()
NegatedAtom c()
end_variable
begin_variable
var5
1
2
Atom d()
NegatedAtom d()
end_variable
3
begin_numeric_variables
R -1 x
C -1 three
I -1 total_cost
end_numeric_variables
1
begin_mutex_group
2
1 0
3 0
end_mutex_group
begin_state
0
2
0
2
0
1
end_state
begin_numeric_state
5
3
0
end_numeric_state
begin_goal
2
0 1
5 0
end_goal
2
begin_operator
raise_x
1
1 0
1
0 4 0 1
1
0 0 + 1
1
end_operator
begin_operator
guarded
0
1
1 3 0 2 -1 1
1
1 1 0 0 + 1
1
end_operator
1
begin_rule
1
1 0
5 1 0
end_rule
2
begin_comparison_axioms
1 >= 0 1
3 < 0 1
end_comparison_axioms
0
begin_numeric_axioms
end_numeric_axioms
begin_global_constraint
5 0
end_global_constraint
begin_SG
";

/// A parsed task numbers its propositional variables exactly as the file
/// does, conditions interleaved and all, and every site that names a
/// variable reads the id the file wrote.
///
/// Only *some* of these sites are covered by a plan cost: nothing in the
/// search reads a mutex group, so a mutex group parsed against the wrong
/// ids would leave every benchmark's plan intact and silently mislead the
/// potential heuristic, which is the one consumer of `are_facts_mutex`.
#[test]
fn parsing_a_sas_task_keeps_the_file_s_variable_order() {
    let (rest, task) =
        parse_numeric_sas_output(INTERLEAVED_CONDITIONS_SAS).expect("the fixture parses");
    assert_eq!(rest, "");

    let names: Vec<&str> = (0..task.get_num_variables() as u32)
        .map(|var_id| {
            task.get_variable_name(VariableIndex::new(var_id))
                .expect("variable in range")
        })
        .collect();
    assert_eq!(names, ["var0", "var1", "var2", "var3", "var4", "var5"]);
    let conditions = task.numeric_conditions();
    assert_eq!(conditions.len(), 2);
    assert_eq!(
        conditions
            .iter()
            .map(|condition| condition.prop_var_id())
            .collect::<Vec<_>>(),
        [VariableIndex::new(1), VariableIndex::new(3)]
    );
    for var_id in 0..task.variables().len() as u32 {
        assert_eq!(
            conditions.is_condition_var(VariableIndex::new(var_id)),
            var_id == 1 || var_id == 3,
            "variable {var_id} is taken for the wrong kind"
        );
    }

    // Each variable keeps its own metadata, except that the file's third
    // condition value is narrowed away: `var1` carries a comparison, so it
    // is two-valued and its `<none of those>` default collapses onto
    // `False`. `var5` is the derived one.
    assert_eq!(
        task.get_variable_domain_size(VariableIndex::new(1)),
        Ok(ConditionValue::DOMAIN_SIZE)
    );
    assert_eq!(
        task.get_variable_default_axiom_value(VariableIndex::new(1)),
        Ok(ExplicitValueIndex::new(ConditionValue::False.as_u32()))
    );
    assert_eq!(
        task.get_variable_axiom_layer(VariableIndex::new(5)),
        Ok(Some(1))
    );
    assert_eq!(task.get_variable_name(VariableIndex::new(5)), Ok("var5"));

    // The initial state is the file's, closed under the axioms: `cond_ge`
    // holds, `cond_lt` does not, and `d` is proven by the rule that reads
    // `cond_ge`.
    assert_eq!(
        task.get_initial_propositional_state_values(),
        // a=0, cond_ge=true, b=0, cond_lt=false, c=0, d=true
        [
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(1),
            ExplicitValueIndex::new(0),
            ExplicitValueIndex::new(0)
        ]
    );

    // Goals, mutex groups and the global constraint.
    let goals: Vec<ExplicitFact> = (0..task.get_num_goals())
        .map(|goal_id| *task.get_goal_fact(goal_id))
        .collect();
    assert_eq!(
        goals,
        [
            ExplicitFact::propositional(0, 1),
            ExplicitFact::propositional(5, 0),
        ]
    );
    // Read through `are_facts_mutex`, the only consumer there is: the group
    // is `{cond_ge = true, cond_lt = true}`.
    assert!(task.are_facts_mutex(
        &ExplicitFact::condition(1, 0),
        &ExplicitFact::condition(3, 0)
    ));
    assert!(!task.are_facts_mutex(
        &ExplicitFact::propositional(0, 0),
        &ExplicitFact::propositional(2, 0)
    ));
    assert_eq!(task.global_constraint(), &ExplicitFact::propositional(5, 0));

    // Operator preconditions and effects, including the effect condition and
    // the guard of an assignment effect. `raise_x` gains a precondition on
    // its effect variable from the `0` precondition value in `0 4 0 1`.
    let raise_x = &task.get_operators()[0];
    assert_eq!(raise_x.name(), "raise_x");
    assert_eq!(
        raise_x.preconditions(),
        &vec![
            ExplicitFact::condition(1, 0),
            ExplicitFact::propositional(4, 0),
        ]
    );
    assert_eq!(raise_x.effects()[0].var_id(), VariableIndex::new(4));
    assert_eq!(
        raise_x.assignment_effects()[0].affected_var_id(),
        VariableIndex::new(0)
    );

    let guarded = &task.get_operators()[1];
    assert_eq!(
        guarded.effects()[0].conditions(),
        &vec![ExplicitFact::condition(3, 0)]
    );
    assert_eq!(guarded.effects()[0].var_id(), VariableIndex::new(2));
    assert_eq!(
        guarded.assignment_effects()[0].conditions(),
        &vec![ExplicitFact::condition(1, 0)]
    );

    // The propositional axiom's head and its condition.
    let rule = &task.axioms()[0];
    assert_eq!(rule.var_id(), VariableIndex::new(5));
    assert_eq!(rule.conditions(), &vec![ExplicitFact::condition(1, 0)]);

    // The comparison axioms name the variables they write.
    let heads: Vec<VariableIndex> = task
        .comparison_axioms()
        .iter()
        .map(|axiom| axiom.get_affected_var_id())
        .collect();
    assert_eq!(heads, [VariableIndex::new(1), VariableIndex::new(3)]);
}
