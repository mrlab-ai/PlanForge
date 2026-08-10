/// SAS+ task representation for the planner output format.
///
/// The task is written out by [`crate::preprocess::output`], after the causal
/// graph has reordered its variables; nothing here formats itself.
use tracing::debug;

pub const SAS_FILE_VERSION: i32 = 4;

/// The comparator that holds exactly when `comp` does not.
pub fn inverted_comparator(comp: &str) -> &'static str {
    match comp {
        "<" => ">=",
        "<=" => ">",
        "=" => "!=",
        ">=" => "<",
        ">" => "<=",
        "!=" => "=",
        other => panic!("unknown comparator: {other:?}"),
    }
}

/// `op` itself, once it is one of the five PDDL assignment operators. An effect
/// that names anything else must not reach the SAS file, where it would be read
/// back as a different effect or not at all.
pub fn assignment_operator(op: &str) -> &str {
    assert!(
        matches!(op, "=" | "+" | "-" | "*" | "/"),
        "unknown assignment operator: {op:?}"
    );
    op
}

/// Planning task in finite-domain representation.
#[derive(Debug, Clone)]
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
    pub global_constraint: (usize, usize), // (var, value=0)
    pub metric: (String, i64),             // ('<' or '>', metric_var_index) where -1 = unit cost
    pub init_constant_predicates: Vec<super::pddl::Atom>,
    pub init_constant_numerics: Vec<super::pddl::FunctionAssignment>,
}

impl SASTask {
    /// Orders the operators and axioms, so that the file written for a task
    /// depends on the task and not on the order the translation produced them
    /// in. Every constructor of a `SASTask` calls this.
    pub fn canonicalize(&mut self) {
        self.operators.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.prevail.cmp(&b.prevail))
                .then_with(|| a.pre_post.cmp(&b.pre_post))
        });
        self.axioms.sort_by(|a, b| {
            a.condition
                .cmp(&b.condition)
                .then_with(|| a.effect.cmp(&b.effect))
        });
    }

    pub fn validate(&self) {
        self.variables.validate();
        for mutex in &self.mutexes {
            mutex.validate(&self.variables);
        }
        self.init.validate(&self.variables);
        self.goal.validate(&self.variables);
        for op in &self.operators {
            op.validate(&self.variables);
        }
        for axiom in &self.axioms {
            axiom.validate(&self.variables, &self.init);
        }
        assert!(
            self.metric.0 == "<" || self.metric.0 == ">",
            "Invalid metric direction: {}",
            self.metric.0
        );
        assert!(self.global_constraint.1 == 0);
    }
}

// ============================================================
// SASVariables
// ============================================================

#[derive(Debug, Clone)]
pub struct SASVariables {
    pub ranges: Vec<usize>,
    pub axiom_layers: Vec<i32>,
    pub value_names: Vec<Vec<String>>,
    pub comp_axiom_layer: i32,
}

impl SASVariables {
    pub fn new(
        ranges: Vec<usize>,
        axiom_layers: Vec<i32>,
        value_names: Vec<Vec<String>>,
        comp_axiom_layer: i32,
    ) -> Self {
        SASVariables {
            ranges,
            axiom_layers,
            value_names,
            comp_axiom_layer,
        }
    }

    pub fn validate(&self) {
        assert_eq!(self.ranges.len(), self.axiom_layers.len());
        assert_eq!(self.ranges.len(), self.value_names.len());
        for (i, ((var_range, layer), var_value_names)) in self
            .ranges
            .iter()
            .zip(self.axiom_layers.iter())
            .zip(self.value_names.iter())
            .enumerate()
        {
            assert_eq!(
                *var_range,
                var_value_names.len(),
                "var {}: range {} != value_names len {}",
                i,
                var_range,
                var_value_names.len()
            );
            assert!(*var_range >= 2, "var {}: range {} < 2", i, var_range);
            assert!(
                *layer == -1 || *layer >= 0,
                "var {}: invalid layer {}",
                i,
                layer
            );
            if *layer > self.comp_axiom_layer {
                // logic axiom: must be binary
                assert_eq!(
                    *var_range, 2,
                    "var {}: logic axiom layer {} but range {}",
                    i, layer, var_range
                );
            }
        }
    }

    pub fn validate_fact(&self, fact: (usize, usize)) {
        let (var, value) = fact;
        assert!(
            var < self.ranges.len(),
            "var {} out of range (max {})",
            var,
            self.ranges.len()
        );
        assert!(
            value < self.ranges[var],
            "value {} out of range for var {} (max {})",
            value,
            var,
            self.ranges[var]
        );
    }

    pub fn validate_condition(&self, condition: &[(usize, usize)]) {
        let mut last_var: Option<usize> = None;
        for &(var, value) in condition {
            self.validate_fact((var, value));
            if let Some(lv) = last_var {
                assert!(var > lv, "condition not sorted: var {} <= {}", var, lv);
            }
            last_var = Some(var);
        }
    }
}

// ============================================================
// SASNumericVariables
// ============================================================

#[derive(Debug, Clone)]
pub struct SASNumericVariables {
    pub variable_names: Vec<String>,
    pub axiom_layers: Vec<i32>,
    pub types: Vec<String>,
}

impl SASNumericVariables {
    pub fn new(variable_names: Vec<String>, axiom_layers: Vec<i32>, types: Vec<String>) -> Self {
        SASNumericVariables {
            variable_names,
            axiom_layers,
            types,
        }
    }
}

// ============================================================
// SASMutexGroup
// ============================================================

#[derive(Debug, Clone)]
pub struct SASMutexGroup {
    pub facts: Vec<(usize, usize)>,
}

impl SASMutexGroup {
    pub fn new(mut facts: Vec<(usize, usize)>) -> Self {
        facts.sort();
        SASMutexGroup { facts }
    }

    pub fn validate(&self, variables: &SASVariables) {
        for &fact in &self.facts {
            variables.validate_fact(fact);
        }
        let mut sorted_unique = self.facts.clone();
        sorted_unique.sort();
        sorted_unique.dedup();
        assert_eq!(self.facts, sorted_unique);
    }
}

// ============================================================
// SASInit
// ============================================================

#[derive(Debug, Clone)]
pub struct SASInit {
    pub values: Vec<i32>,
    pub num_values: Vec<f64>,
}

impl SASInit {
    pub fn new(values: Vec<i32>, num_values: Vec<f64>) -> Self {
        SASInit { values, num_values }
    }

    pub fn validate(&self, variables: &SASVariables) {
        assert_eq!(
            self.values.len(),
            variables.ranges.len(),
            "init values len {} != variable ranges len {}",
            self.values.len(),
            variables.ranges.len()
        );
        for (var, val) in self.values.iter().enumerate() {
            if *val >= 0 {
                variables.validate_fact((var, *val as usize));
            }
        }
    }
}

// ============================================================
// SASGoal
// ============================================================

#[derive(Debug, Clone)]
pub struct SASGoal {
    pub pairs: Vec<(usize, usize)>,
}

impl SASGoal {
    pub fn new(mut pairs: Vec<(usize, usize)>) -> Self {
        pairs.sort();
        SASGoal { pairs }
    }

    pub fn validate(&self, variables: &SASVariables) {
        assert!(!self.pairs.is_empty(), "Empty goal");
        variables.validate_condition(&self.pairs);
    }
}

// ============================================================
// SASOperator
// ============================================================

/// A variable/value pair: one SAS fact.
pub type SasFact = (usize, usize);

/// An effect on a propositional variable: `(variable, precondition, new value,
/// effect condition)`, where a precondition of `-1` means the effect applies
/// whatever the variable's value is.
pub type PrePost = (usize, i32, usize, Vec<SasFact>);

/// An effect on a numeric variable: `(variable, operator, argument variable,
/// effect condition)`.
pub type AssignEffect = (usize, String, usize, Vec<SasFact>);

#[derive(Debug, Clone)]
pub struct SASOperator {
    pub name: String,
    pub prevail: Vec<SasFact>,
    pub pre_post: Vec<PrePost>,
    pub assign_effects: Vec<AssignEffect>,
    pub cost: f64,
}

impl SASOperator {
    pub fn new(
        name: String,
        mut prevail: Vec<SasFact>,
        pre_post: Vec<PrePost>,
        mut assign_effects: Vec<AssignEffect>,
        cost: f64,
    ) -> Self {
        prevail.sort();
        assign_effects.sort();
        let pre_post = Self::canonical_pre_post(pre_post);
        SASOperator {
            name,
            prevail,
            pre_post,
            assign_effects,
            cost,
        }
    }

    /// Effects in a canonical order, so that two operators that differ only
    /// in the order their effects were collected compare equal.
    fn canonical_pre_post(mut pre_post: Vec<PrePost>) -> Vec<PrePost> {
        pre_post.sort();
        pre_post.dedup();
        pre_post
    }

    pub fn validate(&self, variables: &SASVariables) {
        variables.validate_condition(&self.prevail);
        let prevail_vars: std::collections::HashSet<usize> =
            self.prevail.iter().map(|(v, _)| *v).collect();
        let mut pre_values: std::collections::HashMap<usize, i32> =
            std::collections::HashMap::new();
        for (var, pre, post, cond) in &self.pre_post {
            variables.validate_condition(cond);
            assert!(
                !prevail_vars.contains(var),
                "var {} in both prevail and pre_post",
                var
            );
            if *pre != -1 {
                variables.validate_fact((*var, *pre as usize));
            }
            variables.validate_fact((*var, *post));
            assert_eq!(
                variables.axiom_layers[*var], -1,
                "pre_post effect on derived var {}",
                var
            );
            if let Some(existing_pre) = pre_values.get(var) {
                assert_eq!(
                    *existing_pre, *pre,
                    "var {} has multiple preconditions",
                    var
                );
            } else {
                pre_values.insert(*var, *pre);
            }
        }
        for (_, _, _, cond) in &self.pre_post {
            for (cvar, _) in cond {
                assert!(
                    !pre_values.contains_key(cvar) || pre_values[cvar] == -1,
                    "effect condition var {} also has pre",
                    cvar
                );
                assert!(
                    !prevail_vars.contains(cvar),
                    "effect condition var {} also in prevail",
                    cvar
                );
            }
        }
        if self.pre_post.is_empty() {
            assert!(
                !self.assign_effects.is_empty(),
                "operator {} has no effects",
                self.name
            );
        }
    }

    /// The name the SAS file carries: grounded operator names are parenthesized
    /// in PDDL, the file format is not.
    pub fn output_name(&self) -> &str {
        self.name
            .strip_prefix('(')
            .and_then(|name| name.strip_suffix(')'))
            .unwrap_or(&self.name)
    }

    pub fn get_encoding_size(&self) -> usize {
        let mut size = 1 + self.prevail.len();
        for (_, pre, _, cond) in &self.pre_post {
            size += 1 + cond.len();
            if *pre != -1 {
                size += 1;
            }
        }
        size
    }

    pub fn get_applicability_conditions(&self) -> Vec<(usize, usize)> {
        let mut conditions: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        for (var, val) in &self.prevail {
            assert!(!conditions.contains_key(var));
            conditions.insert(*var, *val);
        }
        for (var, pre, _, _) in &self.pre_post {
            if *pre != -1 {
                let pre_val = *pre as usize;
                assert!(!conditions.contains_key(var) || conditions[var] == pre_val);
                conditions.insert(*var, pre_val);
            }
        }
        let mut result: Vec<(usize, usize)> = conditions.into_iter().collect();
        result.sort();
        result
    }
}

// ============================================================
// SASAxiom
// ============================================================

#[derive(Debug, Clone)]
pub struct SASAxiom {
    pub condition: Vec<(usize, usize)>,
    pub effect: (usize, usize),
}

impl SASAxiom {
    pub fn new(mut condition: Vec<(usize, usize)>, effect: (usize, usize)) -> Self {
        condition.sort();
        assert!(effect.1 == 0 || effect.1 == 1);
        for (_, val) in &condition {
            assert!(*val < usize::MAX, "negative value in axiom condition");
        }
        SASAxiom { condition, effect }
    }

    pub fn validate(&self, variables: &SASVariables, init: &SASInit) {
        variables.validate_condition(&self.condition);
        variables.validate_fact(self.effect);
        let (eff_var, eff_value) = self.effect;
        let eff_layer = variables.axiom_layers[eff_var];
        assert!(
            eff_layer >= 0,
            "axiom effect var {} not a derived variable (layer {})",
            eff_var,
            eff_layer
        );
        let eff_init_value = init.values[eff_var];
        for &(cond_var, cond_value) in &self.condition {
            let cond_layer = variables.axiom_layers[cond_var];
            if cond_layer != -1 {
                assert!(
                    cond_layer <= eff_layer,
                    "axiom condition layer {} > effect layer {}",
                    cond_layer,
                    eff_layer
                );
                if cond_layer == eff_layer {
                    let cond_init_value = init.values[cond_var];
                    if eff_value as i32 != eff_init_value {
                        assert!(cond_value as i32 != cond_init_value);
                    } else {
                        assert!(cond_value as i32 == cond_init_value);
                    }
                }
            }
        }
    }

    pub fn dump(&self) {
        debug!("Condition:");
        for (var, val) in &self.condition {
            debug!("  v{}: {}", var, val);
        }
        debug!("Effect:");
        let (var, val) = self.effect;
        debug!("  v{}: {}", var, val);
    }

    pub fn get_encoding_size(&self) -> usize {
        1 + self.condition.len()
    }
}

// ============================================================
// SASCompareAxiom
// ============================================================

#[derive(Debug, Clone)]
pub struct SASCompareAxiom {
    pub comp: String,
    pub parts: Vec<usize>,
    pub effect: usize,
}

impl SASCompareAxiom {
    pub fn new(comp: String, parts: Vec<usize>, effect: usize) -> Self {
        SASCompareAxiom {
            comp,
            parts,
            effect,
        }
    }

    pub fn invert_comparator(&self) -> SASCompareAxiom {
        SASCompareAxiom::new(
            inverted_comparator(&self.comp).to_string(),
            self.parts.clone(),
            self.effect,
        )
    }
}

impl std::fmt::Display for SASCompareAxiom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {} {}", self.comp, self.parts[0], self.parts[1])
    }
}

// ============================================================
// SASNumericAxiom
// ============================================================

#[derive(Debug, Clone)]
pub struct SASNumericAxiom {
    pub op: String,
    pub parts: Vec<usize>,
    pub effect: usize,
}

impl SASNumericAxiom {
    pub fn new(op: String, parts: Vec<usize>, effect: usize) -> Self {
        SASNumericAxiom { op, parts, effect }
    }
}

#[cfg(test)]
mod tests {
    use super::{SASOperator, assignment_operator, inverted_comparator};

    /// The SAS file spells operator names without the PDDL parentheses.
    #[test]
    fn the_output_name_drops_the_pddl_parentheses() {
        let op = SASOperator::new(
            "(move a b)".to_string(),
            Vec::new(),
            Vec::new(),
            vec![(0, "+".to_string(), 1, Vec::new())],
            1.0,
        );

        assert_eq!(op.output_name(), "move a b");
    }

    /// A comparison variable's two facts are named after the comparison and its
    /// negation, so the two have to be exact opposites.
    #[test]
    fn inverting_a_comparator_twice_is_the_identity() {
        for comp in ["<", "<=", "=", ">=", ">", "!="] {
            assert_eq!(inverted_comparator(inverted_comparator(comp)), comp);
        }
    }

    #[test]
    #[should_panic(expected = "unknown comparator")]
    fn inverting_rejects_a_comparator_that_is_not_one() {
        inverted_comparator("=<");
    }

    /// An unknown assignment operator must fail loudly rather than being folded
    /// into a default one, which would silently rewrite the effect.
    #[test]
    #[should_panic(expected = "unknown assignment operator")]
    fn an_unknown_assignment_operator_is_rejected() {
        assignment_operator("^");
    }
}
