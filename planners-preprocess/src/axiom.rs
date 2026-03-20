use std::io::Write;

use crate::helper_functions::{
    check_magic, stringify, CompOperator, FOperator, InputStream,
};
use crate::variable::{NumericVariable, Variable};

#[derive(Debug, Clone)]
pub struct AxiomRelationalCondition {
    pub var: *const Variable,
    pub cond: i32,
}

impl AxiomRelationalCondition {
    pub fn new(var: *const Variable, cond: i32) -> Self {
        Self { var, cond }
    }
}

#[derive(Debug, Clone)]
pub struct AxiomRelational {
    effect_var: *const Variable,
    old_val: i32,
    effect_val: i32,
    conditions: Vec<AxiomRelationalCondition>,
}

impl AxiomRelational {
    pub fn from_stream(stream: &mut InputStream, variables: &Vec<*mut Variable>) -> Self {
        check_magic(stream, "begin_rule");
        let count = stream.read_i32();
        let mut conditions = Vec::new();
        for _ in 0..count {
            let var_no = stream.read_i32();
            let val = stream.read_i32();
            conditions.push(AxiomRelationalCondition::new(
                variables[var_no as usize] as *const Variable,
                val,
            ));
        }
        let var_no = stream.read_i32();
        let old_val = stream.read_i32();
        let new_val = stream.read_i32();
        let effect_var = variables[var_no as usize] as *const Variable;
        check_magic(stream, "end_rule");
        Self {
            effect_var,
            old_val,
            effect_val: new_val,
            conditions,
        }
    }

    pub fn is_redundant(&self) -> bool {
        unsafe { &*self.effect_var }.get_level() == -1
    }

    pub fn str_repr(&self) -> String {
        let mut buf = String::new();
        let effect_level = unsafe { &*self.effect_var }.get_level();
        buf.push_str(&format!("[AX: {} := ", effect_level));
        for cond in &self.conditions {
            let level = unsafe { &*cond.var }.get_level();
            buf.push_str(&format!("{} & ", level));
        }
        if buf.ends_with(" & ") {
            let len = buf.len();
            buf.replace_range(len - 3..len, "");
        }
        buf.push(']');
        buf
    }

    pub fn dump(&self) {
        println!("axiom:");
        print!("conditions:");
        for cond in &self.conditions {
            let var = unsafe { &*cond.var };
            print!("  {} := {}", var.get_name(), cond.cond);
        }
        println!();
        println!("derived:");
        let var = unsafe { &*self.effect_var };
        println!("{} -> {}", var.get_name(), self.effect_val);
        println!();
    }

    pub fn get_encoding_size(&self) -> i32 {
        1 + self.conditions.len() as i32
    }

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        let effect_var = unsafe { &*self.effect_var };
        assert!(effect_var.get_level() != -1);
        writeln!(out, "begin_rule").unwrap();
        writeln!(out, "{}", self.conditions.len()).unwrap();
        for cond in &self.conditions {
            let var = unsafe { &*cond.var };
            assert!(var.get_level() != -1);
            writeln!(out, "{} {}", var.get_level(), cond.cond).unwrap();
        }
        writeln!(
            out,
            "{} {} {}",
            effect_var.get_level(),
            self.old_val,
            self.effect_val
        )
        .unwrap();
        writeln!(out, "end_rule").unwrap();
    }

    pub fn get_conditions(&self) -> &Vec<AxiomRelationalCondition> {
        &self.conditions
    }

    pub fn get_effect_var(&self) -> *const Variable {
        self.effect_var
    }

    pub fn get_old_val(&self) -> i32 {
        self.old_val
    }

    pub fn get_effect_val(&self) -> i32 {
        self.effect_val
    }
}

#[derive(Debug, Clone)]
pub struct AxiomFunctionalComparison {
    effect_var: *const Variable,
    left_var: *const NumericVariable,
    right_var: *const NumericVariable,
    pub cop: CompOperator,
}

impl AxiomFunctionalComparison {
    pub fn from_stream(
        stream: &mut InputStream,
        variables: &mut Vec<*mut Variable>,
        numeric_variables: &Vec<*mut NumericVariable>,
    ) -> Self {
        let var_no = stream.read_i32();
        let coper_str = stream.read_token();
        let coper = CompOperator::from_str(&coper_str);
        let var_no1 = stream.read_i32();
        let var_no2 = stream.read_i32();
        stream.skip_ws();

        assert!(variables.len() > var_no as usize);
        assert!(numeric_variables.len() > var_no1 as usize);
        assert!(numeric_variables.len() > var_no2 as usize);

        let effect_var = variables[var_no as usize] as *mut Variable;
        unsafe { &mut *effect_var }.set_comparison();

        let left_var = numeric_variables[var_no1 as usize] as *const NumericVariable;
        let right_var = numeric_variables[var_no2 as usize] as *const NumericVariable;

        let (comp_string, reverse_comp_string) = stringify(coper);
        let left_name = unsafe { &*left_var }.get_name();
        let right_name = unsafe { &*right_var }.get_name();
        unsafe { &mut *effect_var }
            .set_fact_name(0, format!("{} {}, {}", comp_string, left_name, right_name));
        unsafe { &mut *effect_var }.set_fact_name(
            1,
            format!("{} {}, {}", reverse_comp_string, left_name, right_name),
        );

        Self {
            effect_var: effect_var as *const Variable,
            left_var,
            right_var,
            cop: coper,
        }
    }

    pub fn is_redundant(&self) -> bool {
        unsafe { &*self.effect_var }.get_level() == -1
    }

    pub fn str_repr(&self) -> String {
        let effect_level = unsafe { &*self.effect_var }.get_level();
        let left_level = unsafe { &*self.left_var }.get_level();
        let right_level = unsafe { &*self.right_var }.get_level();
        format!(
            "[AX: {} := {} {} {}]",
            effect_level, left_level, self.cop, right_level
        )
    }

    pub fn dump(&self) {
        let effect_var = unsafe { &*self.effect_var };
        let left_var = unsafe { &*self.left_var };
        let right_var = unsafe { &*self.right_var };
        println!("functional comparison axiom:");
        println!(
            "{} := {} {} {}",
            effect_var.get_name(),
            left_var.get_name(),
            self.cop,
            right_var.get_name()
        );
    }

    pub fn set_relevant(&self) {
        let _ = self;
    }

    pub fn get_encoding_size(&self) -> i32 {
        2
    }

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        let effect_var = unsafe { &*self.effect_var };
        let left_var = unsafe { &*self.left_var };
        let right_var = unsafe { &*self.right_var };
        assert!(effect_var.get_level() != -1);
        writeln!(
            out,
            "{} {} {} {}",
            effect_var.get_level(),
            self.cop,
            left_var.get_level(),
            right_var.get_level()
        )
        .unwrap();
    }

    pub fn get_effect_var(&self) -> *const Variable {
        self.effect_var
    }

    pub fn get_left_var(&self) -> *const NumericVariable {
        self.left_var
    }

    pub fn get_right_var(&self) -> *const NumericVariable {
        self.right_var
    }
}

#[derive(Debug, Clone)]
pub struct AxiomNumericComputation {
    effect_var: *const NumericVariable,
    left_var: *const NumericVariable,
    right_var: *const NumericVariable,
    pub fop: FOperator,
}

impl AxiomNumericComputation {
    pub fn from_stream(
        stream: &mut InputStream,
        numeric_variables: &mut Vec<*mut NumericVariable>,
    ) -> Self {
        let var_no = stream.read_i32();
        let fop_str = stream.read_token();
        let foper = FOperator::from_str(&fop_str);
        let var_no1 = stream.read_i32();
        let var_no2 = stream.read_i32();
        stream.skip_ws();

        assert!(numeric_variables.len() > var_no as usize);
        assert!(numeric_variables.len() > var_no1 as usize);
        assert!(numeric_variables.len() > var_no2 as usize);

        let effect_var = numeric_variables[var_no as usize] as *mut NumericVariable;
        unsafe { &mut *effect_var }.set_subterm();
        let left_var = numeric_variables[var_no1 as usize] as *const NumericVariable;
        let right_var = numeric_variables[var_no2 as usize] as *const NumericVariable;

        Self {
            effect_var: effect_var as *const NumericVariable,
            left_var,
            right_var,
            fop: foper,
        }
    }

    pub fn is_redundant(&self) -> bool {
        unsafe { &*self.effect_var }.get_level() == -1
    }

    pub fn str_repr(&self) -> String {
        let effect_level = unsafe { &*self.effect_var }.get_level();
        let left_level = unsafe { &*self.left_var }.get_level();
        let right_level = unsafe { &*self.right_var }.get_level();
        format!(
            "[AX: {} := {} {} {}]",
            effect_level, left_level, self.fop, right_level
        )
    }

    pub fn dump(&self) {
        let effect_var = unsafe { &*self.effect_var };
        let left_var = unsafe { &*self.left_var };
        let right_var = unsafe { &*self.right_var };
        println!("functional assignment axiom:");
        println!(
            "{} := {} {} {}",
            effect_var.get_name(),
            left_var.get_name(),
            self.fop,
            right_var.get_name()
        );
    }

    pub fn get_encoding_size(&self) -> i32 {
        2
    }

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        let effect_var = unsafe { &*self.effect_var };
        let left_var = unsafe { &*self.left_var };
        let right_var = unsafe { &*self.right_var };
        assert!(effect_var.get_level() != -1);
        assert!(left_var.get_level() != -1);
        assert!(right_var.get_level() != -1);
        writeln!(
            out,
            "{} {} {} {}",
            effect_var.get_level(),
            self.fop,
            left_var.get_level(),
            right_var.get_level()
        )
        .unwrap();
    }

    pub fn get_effect_var(&self) -> *const NumericVariable {
        self.effect_var
    }

    pub fn get_left_var(&self) -> *const NumericVariable {
        self.left_var
    }

    pub fn get_right_var(&self) -> *const NumericVariable {
        self.right_var
    }
}

pub fn strip_axiom_relationals(axioms: &mut Vec<AxiomRelational>) {
    let old_count = axioms.len();
    axioms.retain(|axiom| !axiom.is_redundant());
    println!("{} of {} axiom rules necessary.", axioms.len(), old_count);
}

pub fn strip_axiom_functional_assignment(axioms: &mut Vec<AxiomNumericComputation>) {
    let old_count = axioms.len();
    axioms.retain(|axiom| !axiom.is_redundant());
    println!(
        "{} of {} axiom_functional assignment rules necessary.",
        axioms.len(),
        old_count
    );
}

pub fn strip_axiom_functional_comparisons(axioms: &mut Vec<AxiomFunctionalComparison>) {
    let old_count = axioms.len();
    axioms.retain(|axiom| !axiom.is_redundant());
    println!(
        "{} of {} axiom_functional comparison rules necessary.",
        axioms.len(),
        old_count
    );
}
