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
    // effects (var, pre, post)
    pub effects: Vec<(usize, Option<usize>, usize)>,
    // numeric effects: (num_var_index, delta)
    pub numeric_effects: Vec<(usize, i64)>,
    // numeric preconditions: indices into SASTask.numeric_axioms
    pub numeric_preconds: Vec<usize>,
}

#[derive(Debug, Default)]
pub struct SASTask {
    pub variables: Vec<Variable>,
    pub operators: Vec<SASOperator>,
    pub numeric_variables: Vec<NumericVariable>,
    pub numeric_axioms: Vec<NumericAxiom>,
    // comparison axioms (propositional encoding of comparisons)
    pub comparison_axioms: Vec<CompareAxiom>,
    // initial values for numeric variables
    pub numeric_init: Vec<i64>,
    // mutex groups over propositional variables as pairs (var, val)
    pub mutex_groups: Vec<Vec<(usize, usize)>>,
    // initial state for propositional variables (index into value_names, or -1)
    pub init: Vec<i32>,
    // goal as (var, value) pairs
    pub goal: Vec<(usize, usize)>,
    // Canonical descriptors mirroring the Python translator output
    pub canonical_variables: Vec<CanonicalVariable>,
    pub canonical_operators: Vec<CanonicalOperator>,
    pub canonical_metric: Option<(String, isize)>,
    pub global_constraint: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub enum NumericPrecond {
    VarConst(usize, String, i64),
    VarVar(usize, String, usize),
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
