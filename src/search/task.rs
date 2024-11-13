use crate::parser::{Fact, Operator, RootTask};

trait AbstractTask {

    fn new() -> Self where Self: Sized;
    fn get_num_variables(&self) -> i32;
    fn get_variable_name(&self, index: i32) -> &str;
    fn get_variable_domain_size(&self, index: i32) -> i32;
    fn get_variable_axiom_layer(&self, index: i32) -> i32;
    fn get_variable_default_axiom_value(&self, index: i32) -> i32;
    fn get_fact_name(&self, fact: &Fact) -> &str;

    fn are_facts_mutex(&self, fact1: &Fact, fact2: &Fact) -> bool;
    fn get_operator_cost(&self, index: i32, is_axiom: bool) -> i32;
    fn get_operator_name(&self, index: i32, is_axiom: bool) -> &str;
    fn get_num_operators(&self) -> i32;
    fn get_num_operator_preconditions(&self, index: i32, is_axiom: bool) -> i32;
    fn get_operator_precondition(&self, index: i32, precond_index: i32, is_axiom: bool) -> &Fact;
    fn get_num_operator_effects(&self, index: i32, is_axiom: bool) -> i32;
    fn get_num_operator_effect_conditions(&self, index: i32, eff_index: i32, is_axiom: bool) -> i32;
    fn get_operator_effect_condition(&self, index: i32, eff_index: i32, cond_index: i32, is_axiom: bool) -> &Fact;
    fn get_operator_effect(&self, index: i32, eff_index: i32, is_axiom: bool) -> &Fact;

    fn convert_operator_index(&self, index: i32, ancestor_task: &dyn AbstractTask);

    fn get_num_axioms(&self) -> i32;
    fn get_num_goals(&self) -> i32;
    fn get_goal_fact(&self, index: i32) -> &Fact;

    fn get_initial_state_values(&self) -> Vec<i32>;

    fn convert_ancestor_state_values(&self, ancestor_state_values: &Vec<i32>, ancestor_task: &dyn AbstractTask) -> Vec<i32>;

}

trait MyTrait {
    fn do_something(&self, other: &dyn MyTrait);
    fn another_function(&self);
}

struct MyStruct;




struct TaskProxy {}

impl AbstractTask for TaskProxy {
    fn new() -> Self {
        TaskProxy {}
    }

    fn get_num_variables(&self) -> i32 {
        0
    }

    fn get_variable_name(&self, index: i32) -> &str {
        ""
    }

    fn get_variable_domain_size(&self, index: i32) -> i32 {
        0
    }

    fn get_variable_axiom_layer(&self, index: i32) -> i32 {
        0
    }

    fn get_variable_default_axiom_value(&self, index: i32) -> i32 {
        0
    }

    fn get_fact_name(&self, fact: &Fact) -> &str {
        ""
    }

    fn are_facts_mutex(&self, fact1: &Fact, fact2: &Fact) -> bool {
        false
    }

    fn get_operator_cost(&self, index: i32, is_axiom: bool) -> i32 {
        0
    }

    fn get_operator_name(&self, index: i32, is_axiom: bool) -> &str {
        ""
    }

    fn get_num_operators(&self) -> i32 {
        0
    }

    fn get_num_operator_preconditions(&self, index: i32, is_axiom: bool) -> i32 {
        0
    }

    fn get_operator_precondition(&self, index: i32, precond_index: i32, is_axiom: bool) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_num_operator_effects(&self, index: i32, is_axiom: bool) -> i32 {
        0
    }

    fn get_num_operator_effect_conditions(&self, index: i32, eff_index: i32, is_axiom: bool) -> i32 {
        0
    }

    fn get_operator_effect_condition(&self, index: i32, eff_index: i32, cond_index: i32, is_axiom: bool) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_operator_effect(&self, index: i32, eff_index: i32, is_axiom: bool) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn convert_operator_index(&self, index: i32, ancestor_task: &dyn AbstractTask) {
    }

    fn get_num_axioms(&self) -> i32 {
        0
    }

    fn get_num_goals(&self) -> i32 {
        0
    }

    fn get_goal_fact(&self, index: i32) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_initial_state_values(&self) -> Vec<i32> {
        vec![]
    }

    fn convert_ancestor_state_values(&self, ancestor_state_values: &Vec<i32>, ancestor_task: &dyn AbstractTask) -> Vec<i32> {
        vec![]
    }


}