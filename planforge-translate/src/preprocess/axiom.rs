use std::io::Write;

use tracing::debug;

use super::operator::{CompOperator, FOperator, stringify};
use super::variable::{ExplicitVariable, NumericVariable};
use crate::sas_tasks::{SASAxiom, SASCompareAxiom, SASNumericAxiom};

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
    pub fn from_sas(axiom: &SASAxiom) -> Self {
        let conditions = axiom
            .condition
            .iter()
            .map(|&(var, value)| AxiomRelationalCondition::new(var, value))
            .collect();
        let (effect_var, effect_val) = axiom.effect;
        Self {
            effect_var,
            old_val: 1 - effect_val,
            effect_val,
            conditions,
        }
    }

    pub fn is_redundant(&self, vars: &[ExplicitVariable]) -> bool {
        vars[self.effect_var].get_level() == -1
    }

    pub fn dump(&self, vars: &[ExplicitVariable]) {
        debug!("axiom:");
        debug!("conditions:");
        for cond in &self.conditions {
            debug!("  {} := {}", vars[cond.var].get_name(), cond.cond);
        }
        debug!("");
        debug!("derived:");
        debug!(
            "{} -> {}",
            vars[self.effect_var].get_name(),
            self.effect_val
        );
        debug!("");
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
}

#[derive(Debug, Clone)]
pub struct AxiomFunctionalComparison {
    effect_var: usize,
    left_var: usize,
    right_var: usize,
    pub cop: CompOperator,
}

impl AxiomFunctionalComparison {
    /// Renaming the effect variable's two facts is not cosmetic: a comparison
    /// variable arrives from the translation named after its own index, and the
    /// SAS file is expected to spell out the comparison it stands for.
    pub fn from_sas(
        axiom: &SASCompareAxiom,
        variables: &mut [ExplicitVariable],
        numeric_variables: &[NumericVariable],
    ) -> Self {
        let var_no = axiom.effect;
        let coper = CompOperator::from_string(&axiom.comp);
        assert_eq!(
            axiom.parts.len(),
            2,
            "comparison axiom for var {var_no} compares {} operands, not 2",
            axiom.parts.len()
        );
        let var_no1 = axiom.parts[0];
        let var_no2 = axiom.parts[1];

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

    pub fn dump(&self, vars: &[ExplicitVariable], numeric_vars: &[NumericVariable]) {
        let effect_var = self.effect_var;
        let left_var = self.left_var;
        let right_var = self.right_var;
        debug!("functional comparison axiom:");
        debug!(
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
    pub fn from_sas(axiom: &SASNumericAxiom, numeric_variables: &mut [NumericVariable]) -> Self {
        let var_no = axiom.effect;
        let foper = FOperator::from_string(&axiom.op);
        assert_eq!(
            axiom.parts.len(),
            2,
            "numeric axiom for var {var_no} combines {} operands, not 2",
            axiom.parts.len()
        );
        let var_no1 = axiom.parts[0];
        let var_no2 = axiom.parts[1];

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

    pub fn dump(&self, num_vars: &[NumericVariable]) {
        let effect_var = self.effect_var;
        let left_var = self.left_var;
        let right_var = self.right_var;
        debug!("functional assignment axiom:");
        debug!(
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
