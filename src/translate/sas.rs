#[derive(Debug, Clone)]
pub struct Variable {
    // list of value names for this finite-domain variable
    pub value_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CanonicalVariable {
    pub name: String,
    pub axiom_layer: i32,
    pub values: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct CanonicalEffect {
    pub var: usize,
    pub pre: Option<usize>,
    pub post: usize,
    pub condition: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub enum CanonicalAssignRhs {
    Variable(usize),
    Constant(i64),
}

#[derive(Debug, Clone)]
pub struct CanonicalAssignEffect {
    pub target: usize,
    pub op: String,
    pub rhs: CanonicalAssignRhs,
    pub condition: Vec<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct CanonicalOperator {
    pub name: String,
    pub prevail: Vec<(usize, usize)>,
    pub pre_post: Vec<CanonicalEffect>,
    pub assign_effects: Vec<CanonicalAssignEffect>,
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub struct NumericVariable {
    pub name: String,
    pub initial: Option<i64>,
    pub ntype: String,
    pub axiom_layer: i32,
}

#[derive(Debug, Clone)]
pub struct SASOperator {
    pub name: String,
    // prevail conditions (var->val)
    pub prevails: Vec<(usize, usize)>,
    // effects (var, pre, post, condition)
    pub effects: Vec<(usize, usize, usize, Vec<(usize, usize)>)>,
    // numeric effects: (num_var_index, op, rhs_var, condition)
    pub numeric_effects: Vec<(usize, String, usize, Vec<(usize, usize)>)>,
    // deprecated cost (from Python)
    pub cost: f64,
}

/// Propositional axiom
#[derive(Debug, Clone)]
pub struct SASAxiom {
    pub condition: Vec<(usize, usize)>,
    pub effect: (usize, usize),
}

#[derive(Debug, Default)]
pub struct SASTask {
    pub variables: Vec<Variable>,
    pub operators: Vec<SASOperator>,
    pub numeric_variables: Vec<NumericVariable>,
    pub numeric_axioms: Vec<NumericAxiom>,
    // comparison axioms (propositional encoding of comparisons)
    pub comparison_axioms: Vec<CompareAxiom>,
    // propositional axioms
    pub axioms: Vec<SASAxiom>,
    // initial values for numeric variables
    pub numeric_init: Vec<f64>,
    // mutex groups over propositional variables as pairs (var, val)
    pub mutex_groups: Vec<Vec<(usize, usize)>>,
    // ranges for each propositional variable (domain size)
    pub ranges: Vec<usize>,
    // axiom layers for each variable (-1 for non-derived)
    pub axiom_layers: Vec<i32>,
    // initial state for propositional variables
    pub init: Vec<i32>,
    // goal as (var, value) pairs
    pub goal: Vec<(usize, usize)>,
    // translation key: for each variable, list of value names
    pub translation_key: Vec<Vec<String>>,
    // Canonical descriptors mirroring the Python translator output
    pub canonical_variables: Vec<CanonicalVariable>,
    pub canonical_operators: Vec<CanonicalOperator>,
    pub canonical_metric: Option<(String, isize)>,
    // metric: (direction, variable_index) where direction is '<' (minimize) or '>' (maximize)
    pub metric: (String, isize),
    pub global_constraint: Option<(usize, usize)>,
    // Comparison axiom layer (the layer at which comparison axioms are evaluated)
    pub comp_axiom_layer: i32,
}

#[derive(Debug, Clone)]
pub enum NumericPrecond {
    VarConst(usize, String, i64),
    VarVar(usize, String, usize),
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct NumericAxiom {
    pub op: String,
    pub parts: Vec<usize>,
    pub effect: usize,
}

#[derive(Debug, Clone)]
pub struct CompareAxiom {
    pub comp: String,
    pub parts: Vec<usize>,
    pub effect_var: usize,
}
