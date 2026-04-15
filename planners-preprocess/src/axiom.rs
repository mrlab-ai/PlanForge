use std::io::Write;

use crate::helper_functions::{InputStream, check_magic};
use crate::operator::{CompOperator, FOperator, stringify};
use crate::variable::{ExplicitVariable, NumericVariable};

#[derive(Debug, Clone)]
pub struct AxiomRelationalCondition {
    pub var: usize,
    pub cond: usize,
}

impl AxiomRelationalCondition {
    pub fn new(var: usize, cond: usize) -> Self {
        Self { var, cond }
    }
}

#[derive(Debug, Clone)]
pub struct AxiomRelational {
    effect_var: usize,
    old_val: usize,
    effect_val: usize,
    conditions: Vec<AxiomRelationalCondition>,
}

impl AxiomRelational {
    pub fn from_stream(stream: &mut InputStream) -> Self {
        check_magic(stream, "begin_rule");
        let count = stream.read_i32();
        let mut conditions = Vec::new();
        for _ in 0..count {
            let var_no = stream.read_usize();
            let val = stream.read_usize();
            conditions.push(AxiomRelationalCondition::new(var_no, val));
        }
        let var_no = stream.read_usize();
        let old_val = stream.read_usize();
        let new_val = stream.read_usize();
        check_magic(stream, "end_rule");
        Self {
            effect_var: var_no,
            old_val,
            effect_val: new_val,
            conditions,
        }
    }

    pub fn is_redundant(&self, vars: &[ExplicitVariable]) -> bool {
        vars[self.effect_var].get_level() == -1
    }

    pub fn str_repr(&self, vars: &[ExplicitVariable]) -> String {
        let mut buf = String::new();
        let effect_level = vars[self.effect_var].get_level();
        buf.push_str(&format!("[AX: {} := ", effect_level));
        for cond in &self.conditions {
            let level = vars[cond.var].get_level();
            buf.push_str(&format!("{} & ", level));
        }
        if buf.ends_with(" & ") {
            let len = buf.len();
            buf.replace_range(len - 3..len, "");
        }
        buf.push(']');
        buf
    }

    pub fn dump(&self, vars: &[ExplicitVariable]) {
        println!("axiom:");
        print!("conditions:");
        for cond in &self.conditions {
            print!("  {} := {}", vars[cond.var].get_name(), cond.cond);
        }
        println!();
        println!("derived:");
        println!(
            "{} -> {}",
            vars[self.effect_var].get_name(),
            self.effect_val
        );
        println!();
    }

    pub fn get_encoding_size(&self) -> usize {
        1 + self.conditions.len()
    }

    pub fn to_sas<W: Write>(&self, out: &mut W, vars: &[ExplicitVariable]) {
        assert!(vars[self.effect_var].get_level() != -1);
        writeln!(out, "begin_rule").unwrap();
        writeln!(out, "{}", self.conditions.len()).unwrap();
        for cond in &self.conditions {
            if vars[cond.var].get_level() != -1 {
                writeln!(out, "{} {}", vars[cond.var].get_level(), cond.cond).unwrap();
            }
        }
        writeln!(
            out,
            "{} {} {}",
            vars[self.effect_var].get_level(),
            self.old_val,
            self.effect_val
        )
        .unwrap();
        writeln!(out, "end_rule").unwrap();
    }

    pub fn get_conditions(&self) -> &Vec<AxiomRelationalCondition> {
        &self.conditions
    }

    pub fn get_effect_var(&self) -> usize {
        self.effect_var
    }

    pub fn get_old_val(&self) -> usize {
        self.old_val
    }

    pub fn get_effect_val(&self) -> usize {
        self.effect_val
    }
}

#[derive(Debug, Clone)]
pub struct AxiomFunctionalComparison {
    effect_var: usize,
    left_var: usize,
    right_var: usize,
    pub cop: CompOperator,
}

impl AxiomFunctionalComparison {
    pub fn from_stream(
        stream: &mut InputStream,
        variables: &mut [ExplicitVariable],
        numeric_variables: &[NumericVariable],
    ) -> Self {
        let var_no = stream.read_usize();
        let coper_str = stream.read_token();
        let coper = CompOperator::from_string(&coper_str);
        let var_no1 = stream.read_usize();
        let var_no2 = stream.read_usize();
        stream.skip_ws();

        assert!(variables.len() > var_no);
        assert!(numeric_variables.len() > var_no1);
        assert!(numeric_variables.len() > var_no2);

        variables[var_no].set_comparison();

        let left_var = &numeric_variables[var_no1];
        let right_var = &numeric_variables[var_no2];

        let (comp_string, reverse_comp_string) = stringify(coper);
        let left_name = left_var.get_name();
        let right_name = right_var.get_name();
        variables[var_no]
            .set_fact_name(0, format!("{} {}, {}", comp_string, left_name, right_name));
        variables[var_no].set_fact_name(
            1,
            format!("{} {}, {}", reverse_comp_string, left_name, right_name),
        );

        Self {
            effect_var: var_no,
            left_var: var_no1,
            right_var: var_no2,
            cop: coper,
        }
    }

    pub fn is_redundant(
        &self,
        vars: &[ExplicitVariable],
        numeric_vars: &[NumericVariable],
    ) -> bool {
        vars[self.effect_var].get_level() == -1
            || numeric_vars[self.left_var].get_level() == -1
            || numeric_vars[self.right_var].get_level() == -1
    }

    pub fn str_repr(&self, vars: &[ExplicitVariable], numeric_vars: &[NumericVariable]) -> String {
        let effect_level = vars[self.effect_var].get_level();
        let left_level = numeric_vars[self.left_var].get_level();
        let right_level = numeric_vars[self.right_var].get_level();
        format!(
            "[AX: {} := {} {} {}]",
            effect_level, left_level, self.cop, right_level
        )
    }

    pub fn dump(&self, vars: &[ExplicitVariable], numeric_vars: &[NumericVariable]) {
        let effect_var = self.effect_var;
        let left_var = self.left_var;
        let right_var = self.right_var;
        println!("functional comparison axiom:");
        println!(
            "{} := {} {} {}",
            vars[effect_var].get_name(),
            numeric_vars[left_var].get_name(),
            self.cop,
            numeric_vars[right_var].get_name()
        );
    }

    pub fn get_encoding_size(&self) -> usize {
        2
    }

    pub fn to_sas<W: Write>(
        &self,
        out: &mut W,
        vars: &[ExplicitVariable],
        numeric_vars: &[NumericVariable],
    ) {
        let effect_var = self.effect_var;
        let left_var = self.left_var;
        let right_var = self.right_var;
        assert!(vars[effect_var].get_level() != -1);
        assert!(numeric_vars[left_var].get_level() != -1);
        assert!(numeric_vars[right_var].get_level() != -1);
        writeln!(
            out,
            "{} {} {} {}",
            vars[effect_var].get_level(),
            self.cop,
            numeric_vars[left_var].get_level(),
            numeric_vars[right_var].get_level()
        )
        .unwrap();
    }

    pub fn get_effect_var(&self) -> usize {
        self.effect_var
    }

    pub fn get_left_var(&self) -> usize {
        self.left_var
    }

    pub fn get_right_var(&self) -> usize {
        self.right_var
    }
}

#[derive(Debug, Clone)]
pub struct AxiomNumericComputation {
    effect_var: usize,
    left_var: usize,
    right_var: usize,
    pub fop: FOperator,
}

impl AxiomNumericComputation {
    pub fn from_stream(
        stream: &mut InputStream,
        numeric_variables: &mut [NumericVariable],
    ) -> Self {
        let var_no = stream.read_usize();
        let fop_str = stream.read_token();
        let foper = FOperator::from_string(&fop_str);
        let var_no1 = stream.read_usize();
        let var_no2 = stream.read_usize();
        stream.skip_ws();

        assert!(numeric_variables.len() > var_no);
        assert!(numeric_variables.len() > var_no1);
        assert!(numeric_variables.len() > var_no2);

        {
            numeric_variables[var_no].set_subterm();
        }
        Self {
            effect_var: var_no,
            left_var: var_no1,
            right_var: var_no2,
            fop: foper,
        }
    }

    pub fn is_redundant(&self, num_vars: &[NumericVariable]) -> bool {
        num_vars[self.effect_var].get_level() == -1
            || num_vars[self.left_var].get_level() == -1
            || num_vars[self.right_var].get_level() == -1
    }

    pub fn str_repr(&self, num_vars: &[NumericVariable]) -> String {
        let effect_level = num_vars[self.effect_var].get_level();
        let left_level = num_vars[self.left_var].get_level();
        let right_level = num_vars[self.right_var].get_level();
        format!(
            "[AX: {} := {} {} {}]",
            effect_level, left_level, self.fop, right_level
        )
    }

    pub fn dump(&self, num_vars: &[NumericVariable]) {
        let effect_var = self.effect_var;
        let left_var = self.left_var;
        let right_var = self.right_var;
        println!("functional assignment axiom:");
        println!(
            "{} := {} {} {}",
            num_vars[effect_var].get_name(),
            num_vars[left_var].get_name(),
            self.fop,
            num_vars[right_var].get_name()
        );
    }

    pub fn get_encoding_size(&self) -> usize {
        2
    }

    pub fn to_sas<W: Write>(&self, out: &mut W, num_vars: &[NumericVariable]) {
        let effect_var = self.effect_var;
        let left_var = self.left_var;
        let right_var = self.right_var;
        assert!(num_vars[effect_var].get_level() != -1);
        assert!(num_vars[left_var].get_level() != -1);
        assert!(num_vars[right_var].get_level() != -1);
        writeln!(
            out,
            "{} {} {} {}",
            num_vars[effect_var].get_level(),
            self.fop,
            num_vars[left_var].get_level(),
            num_vars[right_var].get_level()
        )
        .unwrap();
    }

    pub fn get_effect_var(&self) -> usize {
        self.effect_var
    }

    pub fn get_left_var(&self) -> usize {
        self.left_var
    }

    pub fn get_right_var(&self) -> usize {
        self.right_var
    }
}
