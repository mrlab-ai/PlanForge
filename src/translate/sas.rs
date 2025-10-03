#[derive(Debug, Clone)]
pub struct Variable {
    // list of value names for this finite-domain variable
    pub value_names: Vec<String>,
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
