pub trait AbstractNumericTask {
    fn numeric_variables(&self) -> &Vec<NumericVariable>;
    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom>;
    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom>;

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
    fn get_num_operator_effect_conditions(&self, index: i32, eff_index: i32, is_axiom: bool) -> i32;
    fn get_operator_effect_condition(
        &self,
        index: i32,
        eff_index: i32,
        cond_index: i32,
        is_axiom: bool
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
        ancestor_task: &dyn AbstractNumericTask
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
        axiom_default_value: u32
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

    pub fn get_type(&self) -> &NumericType {
        &self.numeric_type
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
        effect_value: u32
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
    conditions: Vec<GlobalCondition>,
}

impl AssignmentEffect {
    pub fn new(
        var_id: u32,
        operation: PlusMinus,
        effect_value: u32,
        conditions: Vec<GlobalCondition>
    ) -> Self {
        AssignmentEffect {
            var_id,
            operation,
            effect_value,
            conditions,
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
        effect_value: u32
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
pub enum ComparisonAxiom {
    LessThan {
        affected_var_id: i32,
        left_hand_side: i32,
        right_hand_side: i32,
    },
    LessThanOrEqual {
        affected_var_id: i32,
        left_hand_side: i32,
        right_hand_side: i32,
    },
    Equal {
        affected_var_id: i32,
        left_hand_side: i32,
        right_hand_side: i32,
    },
    GreaterThanOrEqual {
        affected_var_id: i32,
        left_hand_side: i32,
        right_hand_side: i32,
    },
    GreaterThan {
        affected_var_id: i32,
        left_hand_side: i32,
        right_hand_side: i32,
    },
    UnEqual {
        affected_var_id: i32,
        left_hand_side: i32,
        right_hand_side: i32,
    },
}

impl ComparisonAxiom {
    pub fn new(
        affected_var_id: i32,
        left_hand_side: i32,
        right_hand_side: i32,
        operator: &str
    ) -> Self {
        match operator {
            "<" =>
                ComparisonAxiom::LessThan {
                    affected_var_id,
                    left_hand_side,
                    right_hand_side,
                },
            "<=" =>
                ComparisonAxiom::LessThanOrEqual {
                    affected_var_id,
                    left_hand_side,
                    right_hand_side,
                },
            "==" =>
                ComparisonAxiom::Equal {
                    affected_var_id,
                    left_hand_side,
                    right_hand_side,
                },
            ">=" =>
                ComparisonAxiom::GreaterThanOrEqual {
                    affected_var_id,
                    left_hand_side,
                    right_hand_side,
                },
            ">" =>
                ComparisonAxiom::GreaterThan {
                    affected_var_id,
                    left_hand_side,
                    right_hand_side,
                },
            "!=" =>
                ComparisonAxiom::UnEqual {
                    affected_var_id,
                    left_hand_side,
                    right_hand_side,
                },
            _ => panic!("Unknown comparison operator: {}", operator),
        }
    }

    pub fn evaluate(&self, numeric_state: &Vec<f64>) -> bool {
        match self {
            ComparisonAxiom::LessThan { left_hand_side, right_hand_side, .. } =>
                left_hand_side < right_hand_side,
            ComparisonAxiom::LessThanOrEqual { left_hand_side, right_hand_side, .. } =>
                left_hand_side <= right_hand_side,
            ComparisonAxiom::Equal { left_hand_side, right_hand_side, .. } =>
                left_hand_side == right_hand_side,
            ComparisonAxiom::GreaterThanOrEqual { left_hand_side, right_hand_side, .. } =>
                left_hand_side >= right_hand_side,
            ComparisonAxiom::GreaterThan { left_hand_side, right_hand_side, .. } =>
                left_hand_side > right_hand_side,
            ComparisonAxiom::UnEqual { left_hand_side, right_hand_side, .. } =>
                left_hand_side != right_hand_side,
        }
    }

    pub fn get_affected_var_id(&self) -> i32 {
        match self {
            ComparisonAxiom::LessThan { affected_var_id, .. } => *affected_var_id,
            ComparisonAxiom::LessThanOrEqual { affected_var_id, .. } => *affected_var_id,
            ComparisonAxiom::Equal { affected_var_id, .. } => *affected_var_id,
            ComparisonAxiom::GreaterThanOrEqual { affected_var_id, .. } => *affected_var_id,
            ComparisonAxiom::GreaterThan { affected_var_id, .. } => *affected_var_id,
            ComparisonAxiom::UnEqual { affected_var_id, .. } => *affected_var_id,
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
        right_hand_side: u32
    ) -> Self {
        AssignmentAxiom {
            affected_var_id,
            operator,
            left_hand_side,
            right_hand_side,
        }
    }

    pub fn get_left_var_id(&self) -> u32 {
        self.left_hand_side
    }

    pub fn get_right_var_id(&self) -> u32 {
        self.right_hand_side
    }

    pub fn get_affected_var_id(&self) -> u32 {
        self.affected_var_id
    }

    pub fn get_operator(&self) -> &CalOperator {
        &self.operator
    }
}

#[derive(Debug)]
pub struct GlobalCondition {
    var_id: u32,
    value: u32,
}

impl GlobalCondition {
    pub fn new(var_id: u32, value: u32) -> Self {
        GlobalCondition { var_id, value }
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
    global_constraint: (u32, u32),
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
        global_constraint: (u32, u32)
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
            global_constraint,
        }
    }
}

#[derive(Debug, PartialEq)]
pub enum NumericType {
    Constant,
    Derived,
    Instrumentation,
    Regular, // not sure if Root is correct
    Unknown, //TODO: Remove that somehow
}

impl AbstractNumericTask for NumericRootTask {
    fn numeric_variables(&self) -> &Vec<NumericVariable> {
        &self.numeric_variables
    }

    fn assignment_axioms(&self) -> &Vec<AssignmentAxiom> {
        &self.assignment_axioms
    }

    fn comparison_axioms(&self) -> &Vec<ComparisonAxiom> {
        &self.comparison_axioms
    }

    fn get_num_variables(&self) -> i32 {
        self.variables.len() as i32
    }

    fn get_variable_name(&self, index: i32) -> Result<&str, &str> {
        if index < 0 || index >= (self.variables.len() as i32) {
            return Err("Index out of bounds");
        }
        Ok(&self.variables[index as usize].name)
    }

    fn get_variable_domain_size(&self, index: i32) -> Result<i32, &str> {
        if index < 0 || index >= (self.variables.len() as i32) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index as usize].domain_size as i32)
    }

    fn get_variable_axiom_layer(&self, index: i32) -> Result<i32, &str> {
        if index < 0 || index >= (self.variables.len() as i32) {
            return Err("Index out of bounds");
        }
        Ok(self.variables[index as usize].axiom_layer)
    }

    fn get_variable_default_axiom_value(&self, index: i32) -> Result<i32, &str> {
        if index < 0 || index >= (self.variables.len() as i32) {
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
        is_axiom: bool
    ) -> i32 {
        0
    }

    fn get_operator_effect_condition(
        &self,
        index: i32,
        eff_index: i32,
        cond_index: i32,
        is_axiom: bool
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
        ancestor_task: &dyn AbstractNumericTask
    ) -> Vec<i32> {
        vec![]
    }
}
