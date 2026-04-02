pub trait AbstractTask {
    fn get_num_variables(&self) -> usize;
    fn get_variable_name(&self, index: usize) -> Result<&str, &str>;
    fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str>;
    fn get_variable_axiom_layer(&self, index: usize) -> Result<i32, &str>;
    fn get_variable_default_axiom_value(&self, index: usize) -> Result<u32, &str>;
    fn get_fact_name(&self, fact: &Fact) -> &str;

    fn are_facts_mutex(&self, fact1: &Fact, fact2: &Fact) -> bool;
    fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64;
    fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str;
    fn get_num_operators(&self) -> usize;
    fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize;
    fn get_operator_precondition(
        &self,
        index: usize,
        precond_index: usize,
        is_axiom: bool,
    ) -> &Fact;
    fn get_num_operator_effects(&self, index: usize, is_axiom: bool) -> usize;
    fn get_num_operator_effect_conditions(
        &self,
        index: usize,
        eff_index: usize,
        is_axiom: bool,
    ) -> i32;
    fn get_operator_effect_condition(
        &self,
        index: usize,
        eff_index: usize,
        cond_index: usize,
        is_axiom: bool,
    ) -> &Fact;
    fn get_operator_effect(&self, index: usize, eff_index: usize, is_axiom: bool) -> &Fact;

    fn convert_operator_index(&self, index: usize, ancestor_task: &dyn AbstractTask);

    fn get_num_axioms(&self) -> usize;
    fn get_num_goals(&self) -> usize;
    fn get_goal_fact(&self, index: usize) -> &Fact;

    fn get_initial_state_values(&self) -> Vec<usize>;

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &[usize],
        ancestor_task: &dyn AbstractTask,
    ) -> Vec<usize>;
}

#[derive(Debug)]
pub struct ExplicitVariable {
    pub domain_size: usize,
    pub name: String,
    pub fact_names: Vec<String>,
    pub axiom_layer: i32,
    pub axiom_default_value: u32,
}

impl ExplicitVariable {
    pub fn new(
        domain_size: usize,
        name: String,
        fact_names: Vec<String>,
        axiom_layer: i32,
        axiom_default_value: u32,
    ) -> Self {
        ExplicitVariable {
            domain_size,
            name,
            fact_names,
            axiom_layer,
            axiom_default_value,
        }
    }
}

#[derive(Debug)]
pub struct Fact {
    pub name: usize,
    pub value: usize,
}

impl Fact {
    pub fn new(name: usize, value: usize) -> Self {
        Fact { name, value }
    }
}

#[derive(Debug)]
pub struct Effect {
    pub conditions: Vec<Fact>,
    pub var_id: usize,
    pub precondition_value: i32,
    pub effect_value: usize,
}

impl Effect {
    pub fn new(
        conditions: Vec<Fact>,
        var_id: usize,
        precondition_value: i32,
        effect_value: usize,
    ) -> Self {
        Effect {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
    }
}

#[derive(Debug)]
pub struct Operator {
    pub name: String,
    pub effects: Vec<Effect>,
    pub cost: u64,
}

impl Operator {
    pub fn new(name: String, effects: Vec<Effect>, cost: u64) -> Self {
        Operator {
            name,
            effects,
            cost,
        }
    }
}

#[derive(Debug)]
pub struct Axiom {
    pub conditions: Vec<Fact>,
    pub var_id: usize,
    pub precondition_value: i32,
    pub effect_value: usize,
}

impl Axiom {
    pub fn new(
        conditions: Vec<Fact>,
        var_id: usize,
        precondition_value: i32,
        effect_value: usize,
    ) -> Self {
        Axiom {
            conditions,
            var_id,
            precondition_value,
            effect_value,
        }
    }
}

#[derive(Debug)]
pub struct RootTask {
    pub version: u32,
    pub metric: bool,
    pub variables: Vec<ExplicitVariable>,
    pub goals: Vec<Fact>,
    pub mutexes: Vec<Vec<Fact>>,
    pub states: Vec<usize>,
    pub operators: Vec<Operator>,
    pub axioms: Vec<Axiom>,
}

impl RootTask {
    pub fn new(
        version: u32,
        metric: bool,
        variables: Vec<ExplicitVariable>,
        goals: Vec<Fact>,
        mutexes: Vec<Vec<Fact>>,
        states: Vec<usize>,
        operators: Vec<Operator>,
        axioms: Vec<Axiom>,
    ) -> Self {
        RootTask {
            version,
            metric,
            variables,
            goals,
            mutexes,
            states,
            operators,
            axioms,
        }
    }
}

impl AbstractTask for RootTask {
    fn get_num_variables(&self) -> usize {
        self.variables.len()
    }

    fn get_variable_name(&self, index: usize) -> Result<&str, &str> {
        if index >= self.variables.len() {
            return Err("Index out of bounds");
        }
        Ok(&self.variables[index].name)
    }

    fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str> {
        if index >= self.variables.len() {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index].domain_size)
    }

    fn get_variable_axiom_layer(&self, index: usize) -> Result<i32, &str> {
        if index >= self.variables.len() {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index].axiom_layer)
    }

    fn get_variable_default_axiom_value(&self, index: usize) -> Result<u32, &str> {
        if index >= self.variables.len() {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index].axiom_default_value)
    }

    fn get_fact_name(&self, _fact: &Fact) -> &str {
        ""
    }

    fn are_facts_mutex(&self, _fact1: &Fact, _fact2: &Fact) -> bool {
        false
    }

    fn get_operator_cost(&self, _index: usize, _is_axiom: bool) -> u64 {
        0
    }

    fn get_operator_name(&self, _index: usize, _is_axiom: bool) -> &str {
        ""
    }

    fn get_num_operators(&self) -> usize {
        0
    }

    fn get_num_operator_preconditions(&self, _index: usize, _is_axiom: bool) -> usize {
        0
    }

    fn get_operator_precondition(
        &self,
        _index: usize,
        _precond_index: usize,
        _is_axiom: bool,
    ) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_num_operator_effects(&self, _index: usize, _is_axiom: bool) -> usize {
        0
    }

    fn get_num_operator_effect_conditions(
        &self,
        _index: usize,
        _eff_index: usize,
        _is_axiom: bool,
    ) -> i32 {
        0
    }

    fn get_operator_effect_condition(
        &self,
        _index: usize,
        _eff_index: usize,
        _cond_index: usize,
        _is_axiom: bool,
    ) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_operator_effect(&self, _index: usize, _eff_index: usize, _is_axiom: bool) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn convert_operator_index(&self, _index: usize, _ancestor_task: &dyn AbstractTask) {}

    fn get_num_axioms(&self) -> usize {
        0
    }

    fn get_num_goals(&self) -> usize {
        0
    }

    fn get_goal_fact(&self, _index: usize) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_initial_state_values(&self) -> Vec<usize> {
        vec![]
    }

    fn convert_ancestor_state_values(
        &self,
        _ancestor_state_values: &[usize],
        _ancestor_task: &dyn AbstractTask,
    ) -> Vec<usize> {
        vec![]
    }
}
