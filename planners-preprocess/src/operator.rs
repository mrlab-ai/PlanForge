use std::io::Write;

use crate::DEBUG;
use crate::helper_functions::{InputStream, check_magic};
use crate::variable::{ExplicitVariable, NumType, NumericVariable};

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
    pub fn from_stream(stream: &mut InputStream) -> Self {
        check_magic(stream, "begin_operator");
        stream.skip_ws();
        let name = stream.read_line();

        let mut prevail: Vec<Prevail> = Vec::new();
        let count = stream.read_usize();
        for _ in 0..count {
            let var_no = stream.read_usize();
            let val = stream.read_usize();
            prevail.push(Prevail::new(var_no, val));
        }

        let mut pre_post: Vec<PrePost> = Vec::new();
        let count = stream.read_usize();
        for _ in 0..count {
            let eff_conds = stream.read_usize();
            let mut ecs: Vec<EffCond> = Vec::new();
            for _ in 0..eff_conds {
                let var = stream.read_usize();
                let value = stream.read_usize();
                ecs.push(EffCond::new(var, value));
            }
            let var_no = stream.read_usize();
            let val = stream.read_i32();
            let pre = if val >= 0 { Some(val as usize) } else { None };
            let new_val = stream.read_usize();
            if eff_conds != 0 {
                pre_post.push(PrePost::new_conditional(var_no, ecs, pre, new_val));
            } else {
                pre_post.push(PrePost::new(var_no, pre, new_val));
            }
        }

        let mut assign_effects: Vec<NumericEffect> = Vec::new();
        let count = stream.read_usize();
        for _ in 0..count {
            let eff_conds = stream.read_usize();
            let mut ecs: Vec<EffCond> = Vec::new();
            for _ in 0..eff_conds {
                let var = stream.read_usize();
                let value = stream.read_usize();
                ecs.push(EffCond::new(var, value));
            }
            let af_var = stream.read_usize();
            let op_str = stream.read_token();
            let operator = FOperator::from_string(&op_str);
            let ex_var = stream.read_usize();
            stream.skip_ws();

            let _ = ecs;
            assign_effects.push(NumericEffect::new(af_var, operator, ex_var));
        }

        let cost_str = stream.read_token();
        let cost = cost_str.parse::<f64>().unwrap_or(0.0);
        stream.skip_ws();
        check_magic(stream, "end_operator");

        Self {
            name,
            prevail,
            pre_post,
            assign_effects,
            cost,
        }
    }

    pub fn strip_unimportant_effects(&mut self, num_vars: &[NumericVariable]) {
        self.pre_post
            .retain(|eff| num_vars[eff.var].get_level() != -1);

        self.assign_effects
            .retain(|eff| num_vars[eff.var].get_level() != -1);
    }

    pub fn is_redundant(&self, num_vars: &[NumericVariable]) -> bool {
        if self.pre_post.is_empty() {
            for ass_eff in &self.assign_effects {
                if num_vars[ass_eff.var].get_type() == NumType::Regular {
                    if DEBUG {
                        println!(
                            "Operator {} is not redundant because of effect on {}",
                            self.name,
                            num_vars[ass_eff.var].get_name()
                        );
                    }
                    return false;
                }
            }
            if DEBUG {
                println!("Operator {} is redundant", self.name);
            }
            true
        } else {
            false
        }
    }

    pub fn dump(&self, vars: &[ExplicitVariable], num_vars: &[NumericVariable]) {
        println!("{}:", self.name);
        print!("prevail:");
        for prev in &self.prevail {
            print!("  {} := {}", vars[prev.var].get_name(), prev.prev);
        }
        println!();
        println!("pre-post:");
        for eff in &self.pre_post {
            if eff.is_conditional_effect {
                print!("  if (");
                for cond in &eff.effect_conds {
                    print!("{} := {}", vars[cond.var].get_name(), cond.cond);
                }
                print!(") then");
            }
            print!(
                " {} : {:?} -> {}",
                vars[eff.var].get_name(),
                eff.pre,
                eff.post
            );
        }
        for eff in &self.assign_effects {
            println!("conds:");
            for cond in &eff.effect_conds {
                print!(" {}={}", num_vars[cond.var].get_name(), cond.cond);
            }
            println!("effect:");
            println!(
                " {} {} {}",
                num_vars[eff.var].get_name(),
                eff.fop,
                num_vars[eff.foperand].get_name()
            );
        }
        println!();
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
                eff.pre,
                eff.post
            )
            .unwrap();
        }

        writeln!(out, "{}", self.assign_effects.len()).unwrap();
        for eff in &self.assign_effects {
            write!(out, "{}", eff.effect_conds.len()).unwrap();
            for cond in &eff.effect_conds {
                write!(out, " {} {}", num_vars[cond.var].get_level(), cond.cond).unwrap();
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

    pub fn get_cost(&self) -> f64 {
        self.cost
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
