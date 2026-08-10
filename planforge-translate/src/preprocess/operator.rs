use std::io::Write;

use tracing::debug;

use super::variable::{ExplicitVariable, NumType, NumericVariable};
use crate::sas_tasks::{SASOperator, SasFact};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FOperator {
    Assign = 0,
    ScaleUp = 1,
    ScaleDown = 2,
    Increase = 3,
    Decrease = 4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompOperator {
    Lt = 0,
    Le = 1,
    Eq = 2,
    Ge = 3,
    Gt = 4,
    Ne = 5,
}

impl FOperator {
    pub fn from_string(s: &str) -> Self {
        match s {
            "=" => FOperator::Assign,
            "+" => FOperator::Increase,
            "-" => FOperator::Decrease,
            "*" => FOperator::ScaleUp,
            "/" => FOperator::ScaleDown,
            _ => panic!("Unknown assignment operator : '{}'", s),
        }
    }
}

impl std::fmt::Display for FOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FOperator::Assign => write!(f, "="),
            FOperator::ScaleUp => write!(f, "*"),
            FOperator::ScaleDown => write!(f, "/"),
            FOperator::Increase => write!(f, "+"),
            FOperator::Decrease => write!(f, "-"),
        }
    }
}

impl CompOperator {
    pub fn from_string(s: &str) -> Self {
        match s {
            "<" => CompOperator::Lt,
            "<=" => CompOperator::Le,
            "=" => CompOperator::Eq,
            ">=" => CompOperator::Ge,
            ">" => CompOperator::Gt,
            "!=" => CompOperator::Ne,
            _ => panic!("Unknown comparison operator: '{}'", s),
        }
    }
}

impl std::fmt::Display for CompOperator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompOperator::Lt => write!(f, "<"),
            CompOperator::Le => write!(f, "<="),
            CompOperator::Eq => write!(f, "="),
            CompOperator::Ge => write!(f, ">="),
            CompOperator::Gt => write!(f, ">"),
            CompOperator::Ne => write!(f, "!="),
        }
    }
}

pub fn stringify(cop: CompOperator) -> (String, String) {
    match cop {
        CompOperator::Lt => ("<".to_string(), ">=".to_string()),
        CompOperator::Le => ("<=".to_string(), ">".to_string()),
        CompOperator::Eq => ("=".to_string(), "!=".to_string()),
        CompOperator::Ge => (">=".to_string(), "<".to_string()),
        CompOperator::Gt => (">".to_string(), "<=".to_string()),
        CompOperator::Ne => ("!=".to_string(), "=".to_string()),
    }
}

#[derive(Debug, Clone)]
pub struct Prevail {
    pub var: usize,
    pub prev: usize,
}

impl Prevail {
    pub fn new(var: usize, prev: usize) -> Self {
        Self { var, prev }
    }
}

#[derive(Debug, Clone)]
pub struct EffCond {
    pub var: usize,
    pub cond: usize,
}

impl EffCond {
    pub fn new(var: usize, cond: usize) -> Self {
        Self { var, cond }
    }
}

#[derive(Debug, Clone)]
pub struct PrePost {
    pub var: usize,
    pub pre: Option<usize>,
    pub post: usize,
    pub is_conditional_effect: bool,
    pub effect_conds: Vec<EffCond>,
}

impl PrePost {
    pub fn new(var: usize, pre: Option<usize>, post: usize) -> Self {
        Self {
            var,
            pre,
            post,
            is_conditional_effect: false,
            effect_conds: Vec::new(),
        }
    }

    pub fn new_conditional(
        var: usize,
        effect_conds: Vec<EffCond>,
        pre: Option<usize>,
        post: usize,
    ) -> Self {
        Self {
            var,
            pre,
            post,
            is_conditional_effect: true,
            effect_conds,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NumericEffect {
    pub var: usize,
    pub effect_conds: Vec<EffCond>,
    pub fop: FOperator,
    pub foperand: usize,
    pub is_conditional_effect: bool,
}

impl NumericEffect {
    pub fn new(var: usize, fop: FOperator, foperand: usize) -> Self {
        Self {
            var,
            effect_conds: Vec::new(),
            fop,
            foperand,
            is_conditional_effect: false,
        }
    }

    pub fn new_conditional(
        var: usize,
        effect_conds: Vec<EffCond>,
        fop: FOperator,
        foperand: usize,
    ) -> Self {
        Self {
            var,
            effect_conds,
            fop,
            foperand,
            is_conditional_effect: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Operator {
    name: String,
    prevail: Vec<Prevail>,
    pre_post: Vec<PrePost>,
    assign_effects: Vec<NumericEffect>,
    cost: f64,
}

impl Operator {
    pub fn from_sas(op: &SASOperator) -> Self {
        fn effect_conditions(conditions: &[SasFact]) -> Vec<EffCond> {
            conditions
                .iter()
                .map(|&(var, value)| EffCond::new(var, value))
                .collect()
        }

        let prevail = op
            .prevail
            .iter()
            .map(|&(var, value)| Prevail::new(var, value))
            .collect();

        let pre_post = op
            .pre_post
            .iter()
            .map(|(var, pre, post, conditions)| {
                let pre = usize::try_from(*pre).ok();
                if conditions.is_empty() {
                    PrePost::new(*var, pre, *post)
                } else {
                    PrePost::new_conditional(*var, effect_conditions(conditions), pre, *post)
                }
            })
            .collect();

        let assign_effects = op
            .assign_effects
            .iter()
            .map(|(var, fop, operand, conditions)| {
                let fop = FOperator::from_string(fop);
                if conditions.is_empty() {
                    NumericEffect::new(*var, fop, *operand)
                } else {
                    NumericEffect::new_conditional(
                        *var,
                        effect_conditions(conditions),
                        fop,
                        *operand,
                    )
                }
            })
            .collect();

        Self {
            name: op.output_name().to_owned(),
            prevail,
            pre_post,
            assign_effects,
            cost: op.cost,
        }
    }

    pub fn strip_unimportant_effects(
        &mut self,
        vars: &[ExplicitVariable],
        num_vars: &[NumericVariable],
    ) {
        self.pre_post.retain(|eff| vars[eff.var].get_level() != -1);

        self.assign_effects
            .retain(|eff| num_vars[eff.var].get_level() != -1);
    }

    pub fn is_redundant(&self, num_vars: &[NumericVariable]) -> bool {
        if self.pre_post.is_empty() {
            for ass_eff in &self.assign_effects {
                if num_vars[ass_eff.var].get_type() == NumType::Regular {
                    debug!(
                        "Operator {} is not redundant because of effect on {}",
                        self.name,
                        num_vars[ass_eff.var].get_name()
                    );
                    return false;
                }
            }
            debug!("Operator {} is redundant", self.name);
            true
        } else {
            false
        }
    }

    pub fn dump(&self, vars: &[ExplicitVariable], num_vars: &[NumericVariable]) {
        debug!("{}:", self.name);
        debug!("prevail:");
        for prev in &self.prevail {
            debug!("  {} := {}", vars[prev.var].get_name(), prev.prev);
        }
        debug!("");
        debug!("pre-post:");
        for eff in &self.pre_post {
            if eff.is_conditional_effect {
                debug!("  if (");
                for cond in &eff.effect_conds {
                    debug!("{} := {}", vars[cond.var].get_name(), cond.cond);
                }
                debug!(") then");
            }
            debug!(
                " {} : {:?} -> {}",
                vars[eff.var].get_name(),
                eff.pre,
                eff.post
            );
        }
        for eff in &self.assign_effects {
            debug!("conds:");
            for cond in &eff.effect_conds {
                debug!(" {}={}", num_vars[cond.var].get_name(), cond.cond);
            }
            debug!("effect:");
            debug!(
                " {} {} {}",
                num_vars[eff.var].get_name(),
                eff.fop,
                num_vars[eff.foperand].get_name()
            );
        }
        debug!("");
    }

    pub fn get_encoding_size(&self) -> usize {
        let mut size = 1 + self.prevail.len();
        for eff in &self.pre_post {
            size += 1 + eff.effect_conds.len();
            if eff.pre.is_some() {
                size += 1;
            }
        }
        size
    }

    pub fn to_sas<W: Write>(
        &self,
        out: &mut W,
        vars: &[ExplicitVariable],
        num_vars: &[NumericVariable],
    ) {
        writeln!(out, "begin_operator").unwrap();
        writeln!(out, "{}", self.name).unwrap();

        writeln!(out, "{}", self.prevail.len()).unwrap();
        for prev in &self.prevail {
            assert!(vars[prev.var].get_level() != -1);
            if vars[prev.var].get_level() != -1 {
                writeln!(out, "{} {}", vars[prev.var].get_level(), prev.prev).unwrap();
            }
        }

        writeln!(out, "{}", self.pre_post.len()).unwrap();
        for eff in &self.pre_post {
            assert!(vars[eff.var].get_level() != -1);
            write!(out, "{}", eff.effect_conds.len()).unwrap();
            for cond in &eff.effect_conds {
                write!(out, " {} {}", vars[cond.var].get_level(), cond.cond).unwrap();
            }
            writeln!(
                out,
                " {} {:?} {}",
                vars[eff.var].get_level(),
                eff.pre.map_or(-1, |x| x as i32),
                eff.post
            )
            .unwrap();
        }

        writeln!(out, "{}", self.assign_effects.len()).unwrap();
        for eff in &self.assign_effects {
            write!(out, "{}", eff.effect_conds.len()).unwrap();
            for cond in &eff.effect_conds {
                // An assignment effect's conditions are propositional facts,
                // exactly like a `pre_post` effect's, so they index `vars` and
                // not `num_vars`.
                assert!(
                    vars[cond.var].get_level() != -1,
                    "operator {} guards an assignment effect on a pruned variable {}",
                    self.name,
                    cond.var,
                );
                write!(out, " {} {}", vars[cond.var].get_level(), cond.cond).unwrap();
            }
            writeln!(
                out,
                " {} {} {}",
                num_vars[eff.var].get_level(),
                eff.fop,
                num_vars[eff.foperand].get_level()
            )
            .unwrap();
        }

        writeln!(out, "{}", self.cost).unwrap();
        writeln!(out, "end_operator").unwrap();
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn get_prevail(&self) -> &Vec<Prevail> {
        &self.prevail
    }

    pub fn get_pre_post(&self) -> &Vec<PrePost> {
        &self.pre_post
    }

    pub fn get_num_eff(&self) -> &Vec<NumericEffect> {
        &self.assign_effects
    }
}
