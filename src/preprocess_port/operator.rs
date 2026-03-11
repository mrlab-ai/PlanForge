use std::io::Write;

use crate::preprocess_port::helper_functions::{check_magic, FOperator, InputStream, DEBUG};
use crate::preprocess_port::variable::{NumType, NumericVariable, Variable};

#[derive(Debug, Clone)]
pub struct Prevail {
    pub var: *const Variable,
    pub prev: i32,
}

impl Prevail {
    pub fn new(var: *const Variable, prev: i32) -> Self {
        Self { var, prev }
    }
}

#[derive(Debug, Clone)]
pub struct EffCond {
    pub var: *const Variable,
    pub cond: i32,
}

impl EffCond {
    pub fn new(var: *const Variable, cond: i32) -> Self {
        Self { var, cond }
    }
}

#[derive(Debug, Clone)]
pub struct PrePost {
    pub var: *const Variable,
    pub pre: i32,
    pub post: i32,
    pub is_conditional_effect: bool,
    pub effect_conds: Vec<EffCond>,
}

impl PrePost {
    pub fn new(var: *const Variable, pre: i32, post: i32) -> Self {
        Self {
            var,
            pre,
            post,
            is_conditional_effect: false,
            effect_conds: Vec::new(),
        }
    }

    pub fn new_conditional(
        var: *const Variable,
        effect_conds: Vec<EffCond>,
        pre: i32,
        post: i32,
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
    pub var: *const NumericVariable,
    pub effect_conds: Vec<EffCond>,
    pub fop: FOperator,
    pub foperand: *const NumericVariable,
    pub is_conditional_effect: bool,
}

impl NumericEffect {
    pub fn new(
        var: *const NumericVariable,
        fop: FOperator,
        foperand: *const NumericVariable,
    ) -> Self {
        Self {
            var,
            effect_conds: Vec::new(),
            fop,
            foperand,
            is_conditional_effect: false,
        }
    }

    pub fn new_conditional(
        var: *const NumericVariable,
        effect_conds: Vec<EffCond>,
        fop: FOperator,
        foperand: *const NumericVariable,
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
    pub fn from_stream(
        stream: &mut InputStream,
        variables: &Vec<*mut Variable>,
        numeric_variables: &Vec<*mut NumericVariable>,
    ) -> Self {
        check_magic(stream, "begin_operator");
        stream.skip_ws();
        let name = stream.read_line();

        let mut prevail: Vec<Prevail> = Vec::new();
        let count = stream.read_i32();
        for _ in 0..count {
            let var_no = stream.read_i32();
            let val = stream.read_i32();
            prevail.push(Prevail::new(
                variables[var_no as usize] as *const Variable,
                val,
            ));
        }

        let mut pre_post: Vec<PrePost> = Vec::new();
        let count = stream.read_i32();
        for _ in 0..count {
            let eff_conds = stream.read_i32();
            let mut ecs: Vec<EffCond> = Vec::new();
            for _ in 0..eff_conds {
                let var = stream.read_i32();
                let value = stream.read_i32();
                ecs.push(EffCond::new(
                    variables[var as usize] as *const Variable,
                    value,
                ));
            }
            let var_no = stream.read_i32();
            let val = stream.read_i32();
            let new_val = stream.read_i32();
            if eff_conds != 0 {
                pre_post.push(PrePost::new_conditional(
                    variables[var_no as usize] as *const Variable,
                    ecs,
                    val,
                    new_val,
                ));
            } else {
                pre_post.push(PrePost::new(
                    variables[var_no as usize] as *const Variable,
                    val,
                    new_val,
                ));
            }
        }

        let mut assign_effects: Vec<NumericEffect> = Vec::new();
        let count = stream.read_i32();
        for _ in 0..count {
            let eff_conds = stream.read_i32();
            let mut ecs: Vec<EffCond> = Vec::new();
            for _ in 0..eff_conds {
                let var = stream.read_i32();
                let value = stream.read_i32();
                ecs.push(EffCond::new(
                    variables[var as usize] as *const Variable,
                    value,
                ));
            }
            let af_var = stream.read_i32();
            let op_str = stream.read_token();
            let operato = FOperator::from_str(&op_str);
            let ex_var = stream.read_i32();
            stream.skip_ws();

            let _ = ecs;
            assign_effects.push(NumericEffect::new(
                numeric_variables[af_var as usize] as *const NumericVariable,
                operato,
                numeric_variables[ex_var as usize] as *const NumericVariable,
            ));
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

    pub fn strip_unimportant_effects(&mut self) {
        self.pre_post.retain(|eff| {
            let var = unsafe { &*eff.var };
            var.get_level() != -1
        });

        self.assign_effects.retain(|eff| {
            let var = unsafe { &*eff.var };
            var.get_level() != -1
        });
    }

    pub fn is_redundant(&self) -> bool {
        if self.pre_post.is_empty() {
            for ass_eff in &self.assign_effects {
                let var = unsafe { &*ass_eff.var };
                if var.get_type() == NumType::Regular {
                    if DEBUG {
                        println!(
                            "Operator {} is not redundant because of effect on {}",
                            self.name,
                            var.get_name()
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

    pub fn dump(&self) {
        println!("{}:", self.name);
        print!("prevail:");
        for prev in &self.prevail {
            let var = unsafe { &*prev.var };
            print!("  {} := {}", var.get_name(), prev.prev);
        }
        println!();
        print!("pre-post:");
        for eff in &self.pre_post {
            let var = unsafe { &*eff.var };
            if eff.is_conditional_effect {
                print!("  if (");
                for cond in &eff.effect_conds {
                    let cvar = unsafe { &*cond.var };
                    print!("{} := {}", cvar.get_name(), cond.cond);
                }
                print!(") then");
            }
            print!(" {} : {} -> {}", var.get_name(), eff.pre, eff.post);
        }
        println!();
    }

    pub fn get_encoding_size(&self) -> i32 {
        let mut size = 1 + self.prevail.len() as i32;
        for eff in &self.pre_post {
            size += 1 + eff.effect_conds.len() as i32;
            if eff.pre != -1 {
                size += 1;
            }
        }
        size
    }

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        writeln!(out, "begin_operator").unwrap();
        writeln!(out, "{}", self.name).unwrap();

        writeln!(out, "{}", self.prevail.len()).unwrap();
        for prev in &self.prevail {
            let var = unsafe { &*prev.var };
            assert!(var.get_level() != -1);
            if var.get_level() != -1 {
                writeln!(out, "{} {}", var.get_level(), prev.prev).unwrap();
            }
        }

        writeln!(out, "{}", self.pre_post.len()).unwrap();
        for eff in &self.pre_post {
            let var = unsafe { &*eff.var };
            assert!(var.get_level() != -1);
            write!(out, "{}", eff.effect_conds.len()).unwrap();
            for cond in &eff.effect_conds {
                let cvar = unsafe { &*cond.var };
                write!(out, " {} {}", cvar.get_level(), cond.cond).unwrap();
            }
            writeln!(out, " {} {} {}", var.get_level(), eff.pre, eff.post).unwrap();
        }

        writeln!(out, "{}", self.assign_effects.len()).unwrap();
        for eff in &self.assign_effects {
            let var = unsafe { &*eff.var };
            write!(out, "{}", eff.effect_conds.len()).unwrap();
            for cond in &eff.effect_conds {
                let cvar = unsafe { &*cond.var };
                write!(out, " {} {}", cvar.get_level(), cond.cond).unwrap();
            }
            let operand = unsafe { &*eff.foperand };
            writeln!(
                out,
                " {} {} {}",
                var.get_level(),
                eff.fop,
                operand.get_level()
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

pub fn strip_operators(operators: &mut Vec<Operator>) {
    let old_count = operators.len();
    for op in operators.iter_mut() {
        op.strip_unimportant_effects();
    }
    operators.retain(|op| !op.is_redundant());
    println!("{} of {} operators necessary.", operators.len(), old_count);
}

#[cfg(test)]
mod tests {
    use super::Operator;
    use crate::preprocess_port::helper_functions::InputStream;
    use crate::preprocess_port::variable::{NumericVariable, Variable};

    #[test]
    fn from_stream_preserves_conditional_numeric_effects() {
        let mut variable_stream = InputStream::new(
            "begin_variable\nv0\n-1\n2\na\nb\nend_variable\n\
begin_variable\nv1\n-1\n2\nc\nd\nend_variable\n"
                .to_string(),
        );
        let mut variables_storage = vec![
            Variable::from_stream(&mut variable_stream),
            Variable::from_stream(&mut variable_stream),
        ];
        let variables = variables_storage
            .iter_mut()
            .map(|var| var as *mut Variable)
            .collect::<Vec<_>>();

        let mut numeric_stream = InputStream::new("R -1 n0\nR -1 n1\n".to_string());
        let mut numeric_storage = vec![
            NumericVariable::from_stream(&mut numeric_stream),
            NumericVariable::from_stream(&mut numeric_stream),
        ];
        let numeric_variables = numeric_storage
            .iter_mut()
            .map(|var| var as *mut NumericVariable)
            .collect::<Vec<_>>();

        let input = "begin_operator\nop\n0\n0\n1\n1 1 0 0 + 1\n0\nend_operator\n".to_string();
        let mut stream = InputStream::new(input);

        let op = Operator::from_stream(&mut stream, &variables, &numeric_variables);
        let num_eff = &op.get_num_eff()[0];

        assert!(!num_eff.is_conditional_effect);
        assert!(num_eff.effect_conds.is_empty());
    }
}
