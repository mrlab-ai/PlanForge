trait AbstractNumericTask {
    fn get_num_variables(&self) -> i32;
    fn get_variable_name(&self, index: i32) -> Result<&str, &str>;
    fn get_variable_domain_size(&self, index: i32) -> Result<i32, &str>;
    fn get_variable_axiom_layer(&self, index: i32) -> Result<i32, &str>;
    fn get_variable_default_axiom_value(&self, index: i32) -> Result<i32, &str>;
    fn get_fact_name(&self, fact: &Fact) -> &str;

    fn are_facts_mutex(&self, fact1: &Fact, fact2: &Fact) -> bool;
    fn get_operator_cost(&self, index: i32, is_axiom: bool) -> i32;
    fn get_operator_name(&self, index: i32, is_axiom: bool) -> &str;
    fn get_num_operators(&self) -> i32;
    fn get_num_operator_preconditions(&self, index: i32, is_axiom: bool) -> i32;
    fn get_operator_precondition(&self, index: i32, precond_index: i32, is_axiom: bool) -> &Fact;
    fn get_num_operator_effects(&self, index: i32, is_axiom: bool) -> i32;
    fn get_num_operator_effect_conditions(&self, index: i32, eff_index: i32, is_axiom: bool)
        -> i32;
    fn get_operator_effect_condition(
        &self,
        index: i32,
        eff_index: i32,
        cond_index: i32,
        is_axiom: bool,
    ) -> &Fact;
    fn get_operator_effect(&self, index: i32, eff_index: i32, is_axiom: bool) -> &Fact;

    fn convert_operator_index(&self, index: i32, ancestor_task: &dyn AbstractNumericTask);

    fn get_num_axioms(&self) -> i32;
    fn get_num_goals(&self) -> i32;
    fn get_goal_fact(&self, index: i32) -> &Fact;

    fn get_initial_state_values(&self) -> Vec<i32>;

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &Vec<i32>,
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<i32>;
}

#[derive(Debug)]
pub struct ExplicitVariable {
    domain_size: u32,
    name: String,
    fact_names: Vec<String>,
    axiom_layer: i32,
    axiom_default_value: u32,
}

impl ExplicitVariable {
    pub fn new(
        domain_size: u32,
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
pub struct NumericVariable {
    name: String,
    numeric_type: NumericType,
    axiom_layer: i32,
}

impl NumericVariable {
    pub fn new(name: String, numeric_type: NumericType, axiom_layer: i32) -> Self {
        NumericVariable {
            name,
            numeric_type,
            axiom_layer,
        }
    }
}

#[derive(Debug)]
pub struct Fact {
    name: u32,
    value: u32,
}

impl Fact {
    pub fn new(name: u32, value: u32) -> Self {
        Fact { name, value }
    }
}

#[derive(Debug)]
pub struct Effect {
    conditions: Vec<Fact>,
    var_id: u32,
    precondition_value: i32,
    effect_value: u32,
}

impl Effect {
    pub fn new(
        conditions: Vec<Fact>,
        var_id: u32,
        precondition_value: i32,
        effect_value: u32,
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
pub enum PlusMinus {
    Plus,
    Minus,
}

#[derive(Debug)]
pub struct AssignmentEffect {
    var_id: u32,
    operation: PlusMinus,
    effect_value: u32,
}

impl AssignmentEffect {
    pub fn new(var_id: u32, operation: PlusMinus, effect_value: u32) -> Self {
        AssignmentEffect {
            var_id,
            operation,
            effect_value,
        }
    }
}

#[derive(Debug)]
pub struct Operator {
    name: String,
    effects: Vec<Effect>,
    cost: u32,
}

impl Operator {
    pub fn new(name: String, effects: Vec<Effect>, cost: u32) -> Self {
        Operator {
            name,
            effects,
            cost,
        }
    }
}

#[derive(Debug)]
pub struct Axiom {
    conditions: Vec<Fact>,
    var_id: u32,
    precondition_value: u32,
    effect_value: u32,
}

impl Axiom {
    pub fn new(
        conditions: Vec<Fact>,
        var_id: u32,
        precondition_value: u32,
        effect_value: u32,
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
pub struct ComparisonAxiom {
    affected_var_id: u32,
    comparison_operator: ComparisonOperator,
    left_hand_side: u32,
    right_hand_side: u32,
}

impl ComparisonAxiom {
    pub fn new(
        affected_var_id: u32,
        comparison_operator: ComparisonOperator,
        left_hand_side: u32,
        right_hand_side: u32,
    ) -> Self {
        ComparisonAxiom {
            affected_var_id,
            comparison_operator,
            left_hand_side,
            right_hand_side,
        }
    }
}

#[derive(Debug)]
pub enum CalOperator {
    Sum, 
    Difference,
    Product,
    Division,
}

#[derive(Debug)]
pub struct AssignmentAxiom {
    affected_var_id: u32,
    operator: CalOperator,
    left_hand_side: u32,
    right_hand_side: u32,
}

impl AssignmentAxiom {
    pub fn new(
        affected_var_id: u32,
        operator: CalOperator,
        left_hand_side: u32,
        right_hand_side: u32,
    ) -> Self {
        AssignmentAxiom {
            affected_var_id,
            operator,
            left_hand_side,
            right_hand_side,
        }
    }
}

#[derive(Debug)]
pub struct NumericRootTask {
    version: u32,
    metric: bool,
    variables: Vec<ExplicitVariable>,
    numeric_variables: Vec<NumericVariable>,
    goals: Vec<Fact>,
    mutexes: Vec<Vec<Fact>>,
    state: Vec<i32>,
    numeric_state: Vec<f64>,
    operators: Vec<Operator>,
    axioms: Vec<Axiom>,
    comparison_axioms: Vec<ComparisonAxiom>,
    assignment_axioms: Vec<AssignmentAxiom>,
}

impl NumericRootTask {
    pub fn new(
        version: u32,
        metric: bool,
        variables: Vec<ExplicitVariable>,
        numeric_variables: Vec<NumericVariable>,
        goals: Vec<Fact>,
        mutexes: Vec<Vec<Fact>>,
        state: Vec<i32>,
        numeric_state: Vec<f64>,
        operators: Vec<Operator>,
        axioms: Vec<Axiom>,
        comparison_axioms: Vec<ComparisonAxiom>,
        assignment_axioms: Vec<AssignmentAxiom>,
    ) -> Self {
        NumericRootTask {
            version,
            metric,
            variables,
            numeric_variables,
            goals,
            mutexes,
            state,
            numeric_state,
            operators,
            axioms,
            comparison_axioms,
            assignment_axioms,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum NumericType {
    Constant,
    Derived,
    Implicit,
    Root, // not sure if Root is correct
}

#[derive(Debug, PartialEq)]
pub enum ComparisonOperator {
    LessThan,
    LessThanOrEqual,
    Equal,
    GreaterThanOrEqual,
    GreaterThan,
    UnEqual,
}

impl AbstractNumericTask for NumericRootTask {
    fn get_num_variables(&self) -> i32 {
        self.variables.len() as i32
    }

    fn get_variable_name(&self, index: i32) -> Result<&str, &str> {
        if index < 0 || index >= self.variables.len() as i32 {
            return Err("Index out of bounds");
        }
        Ok(&self.variables[index as usize].name)
    }

    fn get_variable_domain_size(&self, index: i32) -> Result<i32, &str> {
        if index < 0 || index >= self.variables.len() as i32 {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index as usize].domain_size as i32)
    }

    fn get_variable_axiom_layer(&self, index: i32) -> Result<i32, &str> {
        if index < 0 || index >= self.variables.len() as i32 {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index as usize].axiom_layer)
    }

    fn get_variable_default_axiom_value(&self, index: i32) -> Result<i32, &str> {
        if index < 0 || index >= self.variables.len() as i32 {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index as usize].axiom_default_value as i32)
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

    fn get_num_operator_effect_conditions(
        &self,
        index: i32,
        eff_index: i32,
        is_axiom: bool,
    ) -> i32 {
        0
    }

    fn get_operator_effect_condition(
        &self,
        index: i32,
        eff_index: i32,
        cond_index: i32,
        is_axiom: bool,
    ) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn get_operator_effect(&self, index: i32, eff_index: i32, is_axiom: bool) -> &Fact {
        unimplemented!("This function is not yet implemented");
    }

    fn convert_operator_index(&self, index: i32, ancestor_task: &dyn AbstractNumericTask) {}

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

    fn convert_ancestor_state_values(
        &self,
        ancestor_state_values: &Vec<i32>,
        ancestor_task: &dyn AbstractNumericTask,
    ) -> Vec<i32> {
        vec![]
    }
}
