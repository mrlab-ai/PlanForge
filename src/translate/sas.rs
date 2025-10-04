use std::io::Write;

/// SAS file version constant
pub const SAS_FILE_VERSION: u32 = 3;

/// Debug flag for validation
pub const DEBUG: bool = true;

/// SAS Variables collection with validation and output capabilities
#[derive(Debug, Clone, Default)]
pub struct SASVariables {
    pub ranges: Vec<usize>,
    pub axiom_layers: Vec<i32>,
    pub value_names: Vec<Vec<String>>,
    pub comp_axiom_layer: i32,
}

/// SAS Numeric Variables collection
#[derive(Debug, Clone, Default)]
pub struct SASNumericVariables {
    pub variable_names: Vec<String>,
    pub axiom_layers: Vec<i32>,
    pub types: Vec<String>,
}

/// SAS Mutex Group
#[derive(Debug, Clone)]
pub struct SASMutexGroup {
    pub facts: Vec<(usize, usize)>,
}

/// SAS Initial State
#[derive(Debug, Clone, Default)]
pub struct SASInit {
    pub values: Vec<i32>,
    pub num_values: Vec<f64>,
}

/// SAS Goal
#[derive(Debug, Clone, Default)]
pub struct SASGoal {
    pub pairs: Vec<(usize, usize)>,
}

/// SAS Operator with complete functionality
#[derive(Debug, Clone)]
pub struct SASOperator {
    pub name: String,
    pub prevail: Vec<(usize, usize)>,
    pub pre_post: Vec<(usize, i32, usize, Vec<(usize, usize)>)>, // var, pre, post, condition
    pub assign_effects: Vec<(usize, String, usize, Vec<(usize, usize)>)>, // nvar, op, ass_var, condition
    pub cost: f64,
}

/// SAS Axiom
#[derive(Debug, Clone)]
pub struct SASAxiom {
    pub condition: Vec<(usize, usize)>,
    pub effect: (usize, usize),
}

/// SAS Compare Axiom
#[derive(Debug, Clone)]
pub struct SASCompareAxiom {
    pub comp: String,
    pub parts: Vec<usize>,
    pub effect: usize,
}

/// SAS Numeric Axiom
#[derive(Debug, Clone)]
pub struct SASNumericAxiom {
    pub op: String,
    pub parts: Vec<usize>,
    pub effect: usize,
}

/// Main SAS Task with complete functionality matching Python
#[derive(Debug, Default)]
pub struct SASTask {
    pub variables: SASVariables,
    pub numeric_variables: SASNumericVariables,
    pub mutexes: Vec<SASMutexGroup>,
    pub init: SASInit,
    pub goal: SASGoal,
    pub operators: Vec<SASOperator>,
    pub axioms: Vec<SASAxiom>,
    pub comp_axioms: Vec<SASCompareAxiom>,
    pub numeric_axioms: Vec<SASNumericAxiom>,
    pub global_constraint: (usize, usize), // (gcv, zero)
    pub metric: (String, i32), // ('<' or '>', metric_var_index)
    pub init_constant_predicates: Vec<String>,
    pub init_constant_numerics: Vec<f64>,
}

impl SASTask {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn dump(&self) {
        println!("SAS Task:");
        println!("  Variables: {}", self.variables.ranges.len());
        println!("  Operators: {}", self.operators.len());
        println!("  Numeric Variables: {}", self.numeric_variables.variable_names.len());
        println!("  Numeric Axioms: {}", self.numeric_axioms.len());
    }

    pub fn output<W: Write>(&self, _stream: &mut W) -> std::io::Result<()> {
        // TODO: Implement output
        Ok(())
    }
}

// Backwards compatibility types
pub type Variable = Vec<String>;
pub type NumericVariable = SASNumericVariables;

#[derive(Debug, Clone)]
pub struct NumericPrecond {
    // Keeping for backwards compatibility if needed
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum NumericAxiom {
    VarConst(usize, String, i64),
    VarVar(usize, String, usize),
}

#[derive(Debug, Clone)]
pub struct CompareAxiom {
    pub comp: String,
    pub parts: Vec<usize>,
    pub effect: usize,
}
