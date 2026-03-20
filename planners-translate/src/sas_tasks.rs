/// Port of sas_tasks.py
/// SAS+ task representation for the planner output format.
use std::io::Write;

pub const SAS_FILE_VERSION: i32 = 4;

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
    pub fn new(
        variables: SASVariables,
        numeric_variables: SASNumericVariables,
        mutexes: Vec<SASMutexGroup>,
        init: SASInit,
        goal: SASGoal,
        mut operators: Vec<SASOperator>,
        mut axioms: Vec<SASAxiom>,
        comp_axioms: Vec<SASCompareAxiom>,
        numeric_axioms: Vec<SASNumericAxiom>,
        global_constraint: (usize, usize),
        metric: (String, i64),
        init_constant_predicates: Vec<super::pddl::Atom>,
        init_constant_numerics: Vec<super::pddl::FunctionAssignment>,
    ) -> Self {
        // Sort operators by (name, prevail, pre_post) as Python does
        operators.sort_by(|a, b| {
            a.name
                .cmp(&b.name)
                .then_with(|| a.prevail.cmp(&b.prevail))
                .then_with(|| a.pre_post.cmp(&b.pre_post))
        });
        // Sort axioms by (condition, effect)
        axioms.sort_by(|a, b| {
            a.condition
                .cmp(&b.condition)
                .then_with(|| a.effect.cmp(&b.effect))
        });
        SASTask {
            variables,
            numeric_variables,
            mutexes,
            init,
            goal,
            operators,
            axioms,
            comp_axioms,
            numeric_axioms,
            global_constraint,
            metric,
            init_constant_predicates,
            init_constant_numerics,
        }
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

    pub fn dump(&self) {
        println!("variables:");
        self.variables.dump();
        println!("{} mutex groups:", self.mutexes.len());
        for mutex in &self.mutexes {
            println!("group:");
            mutex.dump();
        }
        println!("init:");
        self.init.dump();
        println!("goal:");
        self.goal.dump();
        println!("{} operators:", self.operators.len());
        for op in &self.operators {
            op.dump();
        }
        println!("{} axioms:", self.axioms.len());
        for axiom in &self.axioms {
            axiom.dump();
        }
        println!("metric: ({}, {})", self.metric.0, self.metric.1);
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        writeln!(stream, "begin_version")?;
        writeln!(stream, "{}", SAS_FILE_VERSION)?;
        writeln!(stream, "end_version")?;
        writeln!(stream, "begin_metric")?;
        writeln!(stream, "{} {}", self.metric.0, self.metric.1)?;
        writeln!(stream, "end_metric")?;
        self.variables.output(stream)?;
        self.numeric_variables.output(stream)?;
        writeln!(stream, "{}", self.mutexes.len())?;
        for mutex in &self.mutexes {
            mutex.output(stream)?;
        }
        self.init.output(stream)?;
        self.goal.output(stream)?;
        writeln!(stream, "{}", self.operators.len())?;
        for op in &self.operators {
            op.output(stream)?;
        }
        writeln!(stream, "{}", self.axioms.len())?;
        for axiom in &self.axioms {
            axiom.output(stream)?;
        }
        writeln!(stream, "{}", self.comp_axioms.len())?;
        writeln!(stream, "begin_comparison_axioms")?;
        for cax in &self.comp_axioms {
            cax.output(stream)?;
        }
        writeln!(stream, "end_comparison_axioms")?;
        writeln!(stream, "{}", self.numeric_axioms.len())?;
        writeln!(stream, "begin_numeric_axioms")?;
        for nax in &self.numeric_axioms {
            nax.output(stream)?;
        }
        writeln!(stream, "end_numeric_axioms")?;
        writeln!(stream, "begin_global_constraint")?;
        writeln!(
            stream,
            "{} {}",
            self.global_constraint.0, self.global_constraint.1
        )?;
        writeln!(stream, "end_global_constraint")?;
        Ok(())
    }

    pub fn get_encoding_size(&self) -> usize {
        let mut task_size = 0;
        task_size += self.variables.get_encoding_size();
        for mutex in &self.mutexes {
            task_size += mutex.get_encoding_size();
        }
        task_size += self.goal.get_encoding_size();
        for op in &self.operators {
            task_size += op.get_encoding_size();
        }
        for axiom in &self.axioms {
            task_size += axiom.get_encoding_size();
        }
        for cax in &self.comp_axioms {
            task_size += cax.get_encoding_size();
        }
        for nax in &self.numeric_axioms {
            task_size += nax.get_encoding_size();
        }
        task_size
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

    pub fn dump(&self) {
        for (var, ((rang, names), axiom_layer)) in self
            .ranges
            .iter()
            .zip(self.value_names.iter())
            .zip(self.axiom_layers.iter())
            .enumerate()
        {
            let axiom_str = if *axiom_layer != -1 {
                format!(" [axiom layer {}]", axiom_layer)
            } else {
                String::new()
            };
            let vals: Vec<String> = (0..*rang)
                .zip(names.iter())
                .map(|(i, n)| format!("{}:{}", i, n))
                .collect();
            println!("v{} in {{{}}}{}", var, vals.join(", "), axiom_str);
        }
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        writeln!(stream, "{}", self.ranges.len())?;
        for (var, ((rang, axiom_layer), values)) in self
            .ranges
            .iter()
            .zip(self.axiom_layers.iter())
            .zip(self.value_names.iter())
            .enumerate()
        {
            writeln!(stream, "begin_variable")?;
            writeln!(stream, "var{}", var)?;
            writeln!(stream, "{}", axiom_layer)?;
            writeln!(stream, "{}", rang)?;
            assert_eq!(*rang, values.len());
            for value in values {
                writeln!(stream, "{}", value)?;
            }
            writeln!(stream, "end_variable")?;
        }
        Ok(())
    }

    pub fn get_encoding_size(&self) -> usize {
        self.ranges.len() + self.ranges.iter().sum::<usize>()
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

    pub fn dump(&self) {
        for (v, nv) in self.variable_names.iter().enumerate() {
            println!("numv{}: {}", v, nv);
        }
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        writeln!(stream, "{}", self.variable_names.len())?;
        writeln!(stream, "begin_numeric_variables")?;
        for (v, nv) in self.variable_names.iter().enumerate() {
            writeln!(stream, "{} {} {}", self.types[v], self.axiom_layers[v], nv)?;
        }
        writeln!(stream, "end_numeric_variables")?;
        Ok(())
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

    pub fn dump(&self) {
        for (var, val) in &self.facts {
            println!("v{}: {}", var, val);
        }
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        writeln!(stream, "begin_mutex_group")?;
        writeln!(stream, "{}", self.facts.len())?;
        for (var, val) in &self.facts {
            writeln!(stream, "{} {}", var, val)?;
        }
        writeln!(stream, "end_mutex_group")?;
        Ok(())
    }

    pub fn get_encoding_size(&self) -> usize {
        self.facts.len()
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

    pub fn dump(&self) {
        for (var, val) in self.values.iter().enumerate() {
            if *val != -1 {
                println!("v{}: {}", var, val);
            }
        }
        for (var, val) in self.num_values.iter().enumerate() {
            println!("nv{}: {}", var, val);
        }
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        writeln!(stream, "begin_state")?;
        for val in &self.values {
            writeln!(stream, "{}", val)?;
        }
        writeln!(stream, "end_state")?;
        writeln!(stream, "begin_numeric_state")?;
        for val in &self.num_values {
            writeln!(stream, "{}", val)?;
        }
        writeln!(stream, "end_numeric_state")?;
        Ok(())
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

    pub fn dump(&self) {
        for (var, val) in &self.pairs {
            println!("v{}: {}", var, val);
        }
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        writeln!(stream, "begin_goal")?;
        writeln!(stream, "{}", self.pairs.len())?;
        for (var, val) in &self.pairs {
            writeln!(stream, "{} {}", var, val)?;
        }
        writeln!(stream, "end_goal")?;
        Ok(())
    }

    pub fn get_encoding_size(&self) -> usize {
        self.pairs.len()
    }
}

// ============================================================
// SASOperator
// ============================================================

#[derive(Debug, Clone)]
pub struct SASOperator {
    pub name: String,
    pub prevail: Vec<(usize, usize)>,
    pub pre_post: Vec<(usize, i32, usize, Vec<(usize, usize)>)>, // (var, pre, post, cond) where pre=-1 means no precondition
    pub assign_effects: Vec<(usize, String, usize, Vec<(usize, usize)>)>, // (nvar, op, ass_var, cond)
    pub cost: f64,
}

impl SASOperator {
    pub fn new(
        name: String,
        mut prevail: Vec<(usize, usize)>,
        pre_post: Vec<(usize, i32, usize, Vec<(usize, usize)>)>,
        mut assign_effects: Vec<(usize, String, usize, Vec<(usize, usize)>)>,
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

    fn canonical_pre_post(
        pre_post: Vec<(usize, i32, usize, Vec<(usize, usize)>)>,
    ) -> Vec<(usize, i32, usize, Vec<(usize, usize)>)> {
        // Tuplify -> sort -> dedup -> listify
        let mut tupled: Vec<(usize, i32, usize, Vec<(usize, usize)>)> = pre_post;
        tupled.sort_by(|a, b| {
            a.0.cmp(&b.0)
                .then_with(|| a.1.cmp(&b.1))
                .then_with(|| a.2.cmp(&b.2))
                .then_with(|| a.3.cmp(&b.3))
        });
        tupled.dedup();
        tupled
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

    pub fn dump(&self) {
        println!("{}", self.name);
        println!("Prevail:");
        for (var, val) in &self.prevail {
            println!("  v{}: {}", var, val);
        }
        println!("Pre/Post:");
        for (var, pre, post, cond) in &self.pre_post {
            let cond_str = if cond.is_empty() {
                String::new()
            } else {
                let parts: Vec<String> = cond
                    .iter()
                    .map(|(cv, cv2)| format!("{}: {}", cv, cv2))
                    .collect();
                format!(" [{}]", parts.join(", "))
            };
            println!("  v{}: {} -> {}{}", var, pre, post, cond_str);
        }
        for (var, ass_op, ass_var, cond) in &self.assign_effects {
            let cond_str = if cond.is_empty() {
                String::new()
            } else {
                let parts: Vec<String> = cond
                    .iter()
                    .map(|(cv, cv2)| format!("{}: {}", cv, cv2))
                    .collect();
                format!(" [{}]", parts.join(", "))
            };
            println!("  nv{}: {} nv{}{}", var, ass_op, ass_var, cond_str);
        }
        println!("Cost:");
        println!("  {}", self.cost);
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        writeln!(stream, "begin_operator")?;
        // Python: print(self.name[1:-1]) — strips outer parens
        let name = if self.name.starts_with('(') && self.name.ends_with(')') {
            &self.name[1..self.name.len() - 1]
        } else {
            &self.name
        };
        writeln!(stream, "{}", name)?;
        writeln!(stream, "{}", self.prevail.len())?;
        for (var, val) in &self.prevail {
            writeln!(stream, "{} {}", var, val)?;
        }
        writeln!(stream, "{}", self.pre_post.len())?;
        for (var, pre, post, cond) in &self.pre_post {
            write!(stream, "{} ", cond.len())?;
            for (cvar, cval) in cond {
                write!(stream, "{} {} ", cvar, cval)?;
            }
            writeln!(stream, "{} {} {}", var, pre, post)?;
        }
        writeln!(stream, "{}", self.assign_effects.len())?;
        for (nvar, op, ass_var, cond) in &self.assign_effects {
            write!(stream, "{} ", cond.len())?;
            for (cvar, cval) in cond {
                write!(stream, "{} {} ", cvar, cval)?;
            }
            writeln!(stream, "{} {} {}", nvar, op, ass_var)?;
        }
        writeln!(stream, "{}", self.cost)?;
        writeln!(stream, "end_operator")?;
        Ok(())
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
        println!("Condition:");
        for (var, val) in &self.condition {
            println!("  v{}: {}", var, val);
        }
        println!("Effect:");
        let (var, val) = self.effect;
        println!("  v{}: {}", var, val);
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        writeln!(stream, "begin_rule")?;
        writeln!(stream, "{}", self.condition.len())?;
        for (var, val) in &self.condition {
            writeln!(stream, "{} {}", var, val)?;
        }
        let (var, val) = self.effect;
        writeln!(stream, "{} {} {}", var, 1 - val, val)?;
        writeln!(stream, "end_rule")?;
        Ok(())
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
        let inv_comp = match self.comp.as_str() {
            ">=" => "<",
            "<" => ">=",
            "<=" => ">",
            ">" => "<=",
            "=" => "!=",
            "!=" => "=",
            _ => panic!("Unknown comparator: {}", self.comp),
        };
        SASCompareAxiom::new(inv_comp.to_string(), self.parts.clone(), self.effect)
    }

    pub fn dump(&self) {
        let parts_str = self
            .parts
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("v{}: {} {}", self.effect, self.comp, parts_str);
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        let parts_str = self
            .parts
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(stream, "{} {} {}", self.effect, self.comp, parts_str)?;
        Ok(())
    }

    pub fn get_encoding_size(&self) -> usize {
        1 + self.parts.len()
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

    pub fn dump(&self) {
        let parts_str = self
            .parts
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        println!("nv{}: {} {}", self.effect, self.op, parts_str);
    }

    pub fn output<W: Write>(&self, stream: &mut W) -> std::io::Result<()> {
        let parts_str = self
            .parts
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(stream, "{} {} {}", self.effect, self.op, parts_str)?;
        Ok(())
    }

    pub fn get_encoding_size(&self) -> usize {
        1 + self.parts.len()
    }
}

// ============================================================
// Conversion from internal representation
// ============================================================

/// Python: Called from main as sas_tasks.from_internal(&sastask)
/// In this port, SASTask is already the final form, so this is identity.
pub fn from_internal(task: &SASTask) -> &SASTask {
    task
}
