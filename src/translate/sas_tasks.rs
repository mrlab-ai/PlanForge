use std::io::Write;

pub const SAS_FILE_VERSION: i32 = 4;
pub const DEBUG: bool = false;

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
    pub global_constraint: (usize, usize),
    pub metric: (String, isize),
    pub init_constant_predicates: Vec<String>,
    pub init_constant_numerics: Vec<String>,
}

impl SASTask {
    pub fn new(
        mut operators: Vec<SASOperator>,
        mut axioms: Vec<SASAxiom>,
        variables: SASVariables,
        numeric_variables: SASNumericVariables,
        mutexes: Vec<SASMutexGroup>,
        init: SASInit,
        goal: SASGoal,
        comp_axioms: Vec<SASCompareAxiom>,
        numeric_axioms: Vec<SASNumericAxiom>,
        global_constraint: (usize, usize),
        metric: (String, isize),
        init_constant_predicates: Vec<String>,
        init_constant_numerics: Vec<String>,
    ) -> Self {
        operators.sort_by(|a, b| (a.name.clone(), a.prevail.clone(), a.pre_post.clone()).cmp(&(
            b.name.clone(),
            b.prevail.clone(),
            b.pre_post.clone(),
        )));
        axioms.sort_by(|a, b| (a.condition.clone(), a.effect).cmp(&(b.condition.clone(), b.effect)));
        Self {
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

    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
        writeln!(stream, "begin_version")?;
        writeln!(stream, "{}", SAS_FILE_VERSION)?;
        writeln!(stream, "end_version")?;
        writeln!(stream, "begin_metric")?;
        writeln!(stream, "{} {}", self.metric.0, self.metric.1)?;
        writeln!(stream, "end_metric")?;
        self.variables.output(&mut stream)?;
        self.numeric_variables.output(&mut stream)?;
        writeln!(stream, "{}", self.mutexes.len())?;
        for mutex in &self.mutexes {
            mutex.output(&mut stream)?;
        }
        self.init.output(&mut stream)?;
        self.goal.output(&mut stream)?;
        writeln!(stream, "{}", self.operators.len())?;
        for op in &self.operators {
            op.output(&mut stream)?;
        }
        writeln!(stream, "{}", self.axioms.len())?;
        for axiom in &self.axioms {
            axiom.output(&mut stream)?;
        }
        writeln!(stream, "{}", self.comp_axioms.len())?;
        writeln!(stream, "begin_comparison_axioms")?;
        for cax in &self.comp_axioms {
            cax.output(&mut stream)?;
        }
        writeln!(stream, "end_comparison_axioms")?;
        writeln!(stream, "{}", self.numeric_axioms.len())?;
        writeln!(stream, "begin_numeric_axioms")?;
        for nax in &self.numeric_axioms {
            nax.output(&mut stream)?;
        }
        writeln!(stream, "end_numeric_axioms")?;
        writeln!(stream, "begin_global_constraint")?;
        writeln!(stream, "{} {}", self.global_constraint.0, self.global_constraint.1)?;
        writeln!(stream, "end_global_constraint")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SASVariables {
    pub ranges: Vec<usize>,
    pub axiom_layers: Vec<i32>,
    pub value_names: Vec<Vec<String>>,
    pub comp_axiom_layer: i32,
}

impl SASVariables {
    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
        writeln!(stream, "{}", self.ranges.len())?;
        for var in 0..self.ranges.len() {
            let rang = self.ranges[var];
            let axiom_layer = self.axiom_layers[var];
            let values = &self.value_names[var];
            writeln!(stream, "begin_variable")?;
            writeln!(stream, "var{}", var)?;
            writeln!(stream, "{}", axiom_layer)?;
            writeln!(stream, "{}", rang)?;
            for value in values {
                writeln!(stream, "{}", value)?;
            }
            writeln!(stream, "end_variable")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SASNumericVariables {
    pub variable_names: Vec<String>,
    pub axiom_layers: Vec<i32>,
    pub types: Vec<String>,
}

impl SASNumericVariables {
    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
        writeln!(stream, "{}", self.variable_names.len())?;
        writeln!(stream, "begin_numeric_variables")?;
        for (idx, name) in self.variable_names.iter().enumerate() {
            let t = self.types.get(idx).cloned().unwrap_or_else(|| "U".to_string());
            let layer = self.axiom_layers.get(idx).copied().unwrap_or(-1);
            writeln!(stream, "{} {} {}", t, layer, name)?;
        }
        writeln!(stream, "end_numeric_variables")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SASMutexGroup {
    pub facts: Vec<(usize, usize)>,
}

impl SASMutexGroup {
    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
        writeln!(stream, "begin_mutex_group")?;
        writeln!(stream, "{}", self.facts.len())?;
        for (var, val) in &self.facts {
            writeln!(stream, "{} {}", var, val)?;
        }
        writeln!(stream, "end_mutex_group")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SASInit {
    pub values: Vec<i32>,
    pub num_values: Vec<f64>,
}

impl SASInit {
    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
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

#[derive(Debug, Clone)]
pub struct SASGoal {
    pub pairs: Vec<(usize, usize)>,
}

impl SASGoal {
    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
        writeln!(stream, "begin_goal")?;
        writeln!(stream, "{}", self.pairs.len())?;
        for (var, val) in &self.pairs {
            writeln!(stream, "{} {}", var, val)?;
        }
        writeln!(stream, "end_goal")?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SASOperator {
    pub name: String,
    pub prevail: Vec<(usize, usize)>,
    pub pre_post: Vec<(usize, i32, usize, Vec<(usize, usize)>)>,
    pub assign_effects: Vec<(usize, String, usize, Vec<(usize, usize)>)>,
    pub cost: f64,
}

impl SASOperator {
    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
        writeln!(stream, "begin_operator")?;
        let printed_name = if self.name.len() >= 2
            && self.name.starts_with('(')
            && self.name.ends_with(')')
        {
            self.name[1..self.name.len() - 1].to_string()
        } else {
            self.name.clone()
        };
        writeln!(stream, "{}", printed_name)?;
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
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SASAxiom {
    pub condition: Vec<(usize, usize)>,
    pub effect: (usize, usize),
}

impl SASAxiom {
    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
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
}

#[derive(Debug, Clone)]
pub struct SASCompareAxiom {
    pub comp: String,
    pub parts: Vec<usize>,
    pub effect: usize,
}

impl SASCompareAxiom {
    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
        writeln!(
            stream,
            "{} {} {} {}",
            self.effect, self.comp, self.parts[0], self.parts[1]
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct SASNumericAxiom {
    pub op: String,
    pub parts: Vec<usize>,
    pub effect: usize,
}

impl SASNumericAxiom {
    pub fn output<W: Write>(&self, mut stream: W) -> std::io::Result<()> {
        let parts = self
            .parts
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(" ");
        writeln!(stream, "{} {} {}", self.effect, self.op, parts)?;
        Ok(())
    }
}

pub fn from_internal(task: &crate::translate::sas::SASTask) -> SASTask {
    let (ranges, axiom_layers, value_names) = if !task.canonical_variables.is_empty() {
        (
            task.canonical_variables
                .iter()
                .map(|v| v.values.len())
                .collect::<Vec<_>>(),
            task.canonical_variables
                .iter()
                .map(|v| v.axiom_layer)
                .collect::<Vec<_>>(),
            task.canonical_variables
                .iter()
                .map(|v| v.values.clone())
                .collect::<Vec<_>>(),
        )
    } else {
        (
            task.ranges.clone(),
            task.axiom_layers.clone(),
            task.variables
                .iter()
                .map(|v| v.value_names.clone())
                .collect::<Vec<_>>(),
        )
    };
    let variables = SASVariables {
        ranges,
        axiom_layers,
        value_names,
        comp_axiom_layer: task.comp_axiom_layer,
    };

    let numeric_variables = SASNumericVariables {
        variable_names: task
            .numeric_variables
            .iter()
            .map(|nv| nv.name.clone())
            .collect(),
        axiom_layers: task
            .numeric_variables
            .iter()
            .map(|nv| nv.axiom_layer)
            .collect(),
        types: task
            .numeric_variables
            .iter()
            .map(|nv| nv.ntype.clone())
            .collect(),
    };

    let mutexes = task
        .mutex_groups
        .iter()
        .map(|g| SASMutexGroup { facts: g.clone() })
        .collect::<Vec<_>>();

    let init = SASInit {
        values: task.init.clone(),
        num_values: task.numeric_init.clone(),
    };

    let goal = SASGoal {
        pairs: task.goal.clone(),
    };

    let operators = if !task.canonical_operators.is_empty() {
        task.canonical_operators
            .iter()
            .map(|op| SASOperator {
                name: op.name.clone(),
                prevail: op.prevail.clone(),
                pre_post: op
                    .pre_post
                    .iter()
                    .map(|e| {
                        (
                            e.var,
                            e.pre.map(|p| p as i32).unwrap_or(-1),
                            e.post,
                            e.condition.clone(),
                        )
                    })
                    .collect(),
                assign_effects: op
                    .assign_effects
                    .iter()
                    .map(|e| {
                        let rhs = match e.rhs {
                            crate::translate::sas::CanonicalAssignRhs::Variable(v) => v,
                            crate::translate::sas::CanonicalAssignRhs::Constant(c) => c as usize,
                        };
                        (e.target, e.op.clone(), rhs, e.condition.clone())
                    })
                    .collect(),
                cost: op.cost,
            })
            .collect()
    } else {
        task.operators
            .iter()
            .map(|op| SASOperator {
                name: op.name.clone(),
                prevail: op.prevails.clone(),
                pre_post: op
                    .effects
                    .iter()
                    .map(|(var, pre, post, cond)| (*var, *pre as i32, *post, cond.clone()))
                    .collect(),
                assign_effects: op
                    .numeric_effects
                    .iter()
                    .map(|(nvar, opstr, rhs, cond)| (*nvar, opstr.clone(), *rhs, cond.clone()))
                    .collect(),
                cost: op.cost,
            })
            .collect()
    };

    let axioms = task
        .axioms
        .iter()
        .map(|ax| SASAxiom {
            condition: ax.condition.clone(),
            effect: ax.effect,
        })
        .collect::<Vec<_>>();

    let comp_axioms = task
        .comparison_axioms
        .iter()
        .map(|ax| SASCompareAxiom {
            comp: ax.comp.clone(),
            parts: ax.parts.clone(),
            effect: ax.effect_var,
        })
        .collect::<Vec<_>>();

    let numeric_axioms = task
        .numeric_axioms
        .iter()
        .map(|ax| SASNumericAxiom {
            op: ax.op.clone(),
            parts: ax.parts.clone(),
            effect: ax.effect,
        })
        .collect::<Vec<_>>();

    let global_constraint = task.global_constraint.unwrap_or((0, 0));

    let metric = task.metric.clone();

    SASTask::new(
        operators,
        axioms,
        variables,
        numeric_variables,
        mutexes,
        init,
        goal,
        comp_axioms,
        numeric_axioms,
        global_constraint,
        metric,
        Vec::new(),
        Vec::new(),
    )
}
