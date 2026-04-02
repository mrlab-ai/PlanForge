use std::io::{Read, Write};

use crate::axiom::{AxiomFunctionalComparison, AxiomNumericComputation, AxiomRelational};
use crate::domain_transition_graph::DomainTransitionGraph;
use crate::fact::ExplicitFact;
use crate::mutex_group::MutexGroup;
use crate::operator::Operator;
use crate::state::State;
use crate::variable::{ExplicitVariable, NumericVariable};
use crate::{DEBUG, GlobalConstraint, Metric, PRE_FILE_VERSION, SAS_FILE_VERSION};

pub struct InputStream {
    input: String,
    pos: usize,
}

impl InputStream {
    pub fn from_reader<R: Read>(reader: &mut R) -> std::io::Result<Self> {
        let mut input = String::new();
        reader.read_to_string(&mut input)?;
        Ok(Self { input, pos: 0 })
    }

    pub fn new(input: String) -> Self {
        Self { input, pos: 0 }
    }

    pub fn skip_ws(&mut self) {
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
    }

    pub fn read_token(&mut self) -> String {
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.peek_char() {
            if ch.is_whitespace() {
                break;
            }
            self.pos += ch.len_utf8();
        }
        self.input[start..self.pos].to_string()
    }

    pub fn read_line(&mut self) -> String {
        if self.pos >= self.input.len() {
            return String::new();
        }
        let bytes = self.input.as_bytes();
        let mut end = self.pos;
        while end < bytes.len() && bytes[end] != b'\n' {
            end += 1;
        }
        let mut line = self.input[self.pos..end].to_string();
        if line.ends_with('\r') {
            line.pop();
        }
        self.pos = if end < bytes.len() { end + 1 } else { end };
        line
    }

    pub fn read_until(&mut self, delim: char) -> String {
        let bytes = self.input.as_bytes();
        let mut end = self.pos;
        while end < bytes.len() && bytes[end] != delim as u8 && bytes[end] != b'\n' {
            end += 1;
        }
        let s = self.input[self.pos..end].to_string();
        if end < bytes.len() && bytes[end] == delim as u8 {
            self.pos = end + 1;
        } else {
            self.pos = end;
        }
        s
    }

    pub fn read_char(&mut self) -> char {
        self.skip_ws();
        let ch = self.peek_char().unwrap_or('\0');
        self.pos += ch.len_utf8();
        ch
    }

    pub fn read_i32(&mut self) -> i32 {
        let tok = self.read_token();
        tok.parse::<i32>().unwrap_or(0)
    }

    pub fn read_usize(&mut self) -> usize {
        let tok = self.read_token();
        tok.parse::<usize>().unwrap_or(0)
    }

    fn peek_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }
}

pub fn check_magic(stream: &mut InputStream, magic: &str) {
    let word = stream.read_token();
    if word != magic {
        eprintln!("Failed to match magic word '{}'.", magic);
        eprintln!("Got '{}'.", word);
        if magic == "begin_version" {
            eprintln!(
                "Possible cause: you are running the preprocessor on a translator file from an"
            );
            eprintln!("older version.");
        }
        std::process::exit(1);
    }
}

fn read_and_verify_version(stream: &mut InputStream) {
    check_magic(stream, "begin_version");
    let version = stream.read_i32();
    check_magic(stream, "end_version");
    if version != SAS_FILE_VERSION {
        eprintln!(
            "Expected translator file version {}, got {}.",
            SAS_FILE_VERSION, version
        );
        eprintln!("Exiting.");
        std::process::exit(1);
    }
}

fn read_metric(stream: &mut InputStream) -> Metric {
    check_magic(stream, "begin_metric");
    let metric = Metric {
        optimization_criterion: stream.read_char(),
        index: stream.read_usize(),
    };
    check_magic(stream, "end_metric");

    metric
}

fn read_variables(stream: &mut InputStream) -> Vec<ExplicitVariable> {
    let count = stream.read_usize();
    let mut variables = Vec::with_capacity(count);
    for i in 0..count {
        variables.push(ExplicitVariable::from_stream(stream, i));
    }

    variables
}

fn read_numeric_variables(stream: &mut InputStream) -> Vec<NumericVariable> {
    let count = stream.read_usize();
    let mut numeric_variables = Vec::with_capacity(count);
    check_magic(stream, "begin_numeric_variables");
    for i in 0..count {
        numeric_variables.push(NumericVariable::from_stream(stream, i));
    }
    check_magic(stream, "end_numeric_variables");

    numeric_variables
}

fn read_mutexes(stream: &mut InputStream) -> Vec<MutexGroup> {
    let count = stream.read_usize();
    let mut mutexes = Vec::with_capacity(count);
    for _ in 0..count {
        mutexes.push(MutexGroup::from_stream(stream));
    }

    mutexes
}

fn read_global_constraint(stream: &mut InputStream) -> Option<GlobalConstraint> {
    check_magic(stream, "begin_global_constraint");
    let index = stream.read_i32();
    let val = stream.read_i32();
    let gc = if index >= 0 {
        Some(GlobalConstraint {
            var: index as usize,
            value: val as usize,
        })
    } else {
        None
    };
    println!("read global constraint at var {} value {}", index, val);
    check_magic(stream, "end_global_constraint");

    gc
}

fn read_goal(stream: &mut InputStream) -> Vec<ExplicitFact> {
    check_magic(stream, "begin_goal");
    let count = stream.read_usize();
    let mut goals = Vec::with_capacity(count);
    for _ in 0..count {
        let var_no = stream.read_usize();
        let val = stream.read_usize();
        goals.push(ExplicitFact {
            var: var_no,
            value: val,
        });
    }
    check_magic(stream, "end_goal");

    goals
}

fn dump_goal(goals: &[ExplicitFact], variables: &[ExplicitVariable]) {
    println!("Goal Conditions:");
    for goal in goals {
        println!("  {}: {}", variables[goal.var].get_name(), goal.value);
    }
}

fn read_operators(stream: &mut InputStream) -> Vec<Operator> {
    let count = stream.read_usize();
    let mut operators = Vec::with_capacity(count);
    for _ in 0..count {
        operators.push(Operator::from_stream(stream));
    }

    operators
}

fn read_axioms_rel(
    stream: &mut InputStream,
    variables: &[ExplicitVariable],
) -> Vec<AxiomRelational> {
    let count = stream.read_usize();
    let mut axioms_rel = Vec::with_capacity(count);
    for _ in 0..count {
        axioms_rel.push(AxiomRelational::from_stream(stream));
    }
    axioms_rel.sort_by(|a, b| {
        variables[a.get_effect_var()]
            .get_layer()
            .cmp(&variables[b.get_effect_var()].get_layer())
    });

    axioms_rel
}

fn read_axioms_func_comp(
    stream: &mut InputStream,
    variables: &mut [ExplicitVariable],
    numeric_variables: &[NumericVariable],
) -> Vec<AxiomFunctionalComparison> {
    let count = stream.read_usize();
    let mut axioms_func_comp = Vec::with_capacity(count);
    check_magic(stream, "begin_comparison_axioms");
    for _ in 0..count {
        axioms_func_comp.push(AxiomFunctionalComparison::from_stream(
            stream,
            variables,
            numeric_variables,
        ));
    }
    check_magic(stream, "end_comparison_axioms");
    axioms_func_comp.sort_by(|a, b| {
        variables[a.get_effect_var()]
            .get_layer()
            .cmp(&variables[b.get_effect_var()].get_layer())
    });

    axioms_func_comp
}

fn read_axioms_numeric(
    stream: &mut InputStream,
    numeric_variables: &mut [NumericVariable],
) -> Vec<AxiomNumericComputation> {
    let count = stream.read_usize();
    let mut axioms_numeric = Vec::with_capacity(count);
    check_magic(stream, "begin_numeric_axioms");
    for _ in 0..count {
        axioms_numeric.push(AxiomNumericComputation::from_stream(
            stream,
            numeric_variables,
        ));
    }
    check_magic(stream, "end_numeric_axioms");
    axioms_numeric.sort_by(|a, b| {
        numeric_variables[a.get_effect_var()]
            .get_layer()
            .cmp(&numeric_variables[b.get_effect_var()].get_layer())
    });

    axioms_numeric
}

pub fn read_preprocessed_problem_description(
    stream: &mut InputStream,
) -> (
    Metric,
    Vec<ExplicitVariable>,
    Vec<NumericVariable>,
    Vec<MutexGroup>,
    State,
    Vec<ExplicitFact>,
    Vec<Operator>,
    Vec<AxiomRelational>,
    Vec<AxiomFunctionalComparison>,
    Vec<AxiomNumericComputation>,
    Option<GlobalConstraint>,
) {
    if DEBUG {
        println!("reading version...");
    }
    read_and_verify_version(stream);
    if DEBUG {
        println!("reading metric...");
    }
    let metric = read_metric(stream);
    if DEBUG {
        println!("reading variables...");
    }
    let mut variables = read_variables(stream);
    if DEBUG {
        println!("reading numeric variables...");
    }
    let mut numeric_variables = read_numeric_variables(stream);
    if DEBUG {
        println!("reading mutexes...");
    }
    let mutexes = read_mutexes(stream);
    if DEBUG {
        println!("reading initial state...");
    }
    let initial_state = State::from_stream(stream, &variables, &numeric_variables);
    let goal = read_goal(stream);
    if DEBUG {
        println!("reading operators...");
    }
    let operators = read_operators(stream);
    if DEBUG {
        println!("reading propositional axioms...");
    }
    let axioms_rel = read_axioms_rel(stream, &variables);
    if DEBUG {
        println!("reading functional comparison axioms...");
    }
    let axioms_func_comp = read_axioms_func_comp(stream, &mut variables, &numeric_variables);
    if DEBUG {
        println!("reading functional assignment axioms...");
    }
    let axioms_numeric = read_axioms_numeric(stream, &mut numeric_variables);
    if DEBUG {
        println!("reading global constraint");
    }
    let global_constraint = read_global_constraint(stream);

    (
        metric,
        variables,
        numeric_variables,
        mutexes,
        initial_state,
        goal,
        operators,
        axioms_rel,
        axioms_func_comp,
        axioms_numeric,
        global_constraint,
    )
}

pub fn dump_preprocessed_problem_description(
    variables: &[ExplicitVariable],
    numeric_variables: &[NumericVariable],
    initial_state: &State,
    goals: &[ExplicitFact],
    operators: &Vec<Operator>,
    axioms_rel: &Vec<AxiomRelational>,
    axioms_func_ass: &Vec<AxiomNumericComputation>,
    axioms_func_comp: &Vec<AxiomFunctionalComparison>,
) {
    println!("Variables ({}):", variables.len());
    for var in variables {
        var.dump();
    }
    println!("Numeric variables ({}):", numeric_variables.len());
    for var in numeric_variables {
        var.dump();
    }
    println!("Initial State:");
    initial_state.dump(variables, numeric_variables);
    dump_goal(goals, variables);
    for op in operators {
        op.dump(variables, numeric_variables);
    }
    for ax in axioms_rel {
        ax.dump(variables);
    }
    for ax in axioms_func_ass {
        ax.dump(numeric_variables);
    }
    for ax in axioms_func_comp {
        ax.dump(variables, numeric_variables);
    }
}

pub fn dump_dtgs(ordering: &[ExplicitVariable], transition_graphs: &mut [DomainTransitionGraph]) {
    let num_graphs = transition_graphs.len();
    for i in 0..num_graphs {
        let name = ordering[i].get_name();
        println!("Domain transition graph for {}:", name);
        transition_graphs[i].dump(ordering);
    }
}

pub fn to_sas(
    orig_vars: &[ExplicitVariable],
    orig_numeric_vars: &[NumericVariable],
    ordered_vars: &[ExplicitVariable],
    ordered_numeric_vars: &[NumericVariable],
    metric: &Metric,
    mutexes: &Vec<MutexGroup>,
    initial_state: &State,
    goals: &Vec<ExplicitFact>,
    operators: &Vec<Operator>,
    axioms_rel: &Vec<AxiomRelational>,
    axioms_func_ass: &Vec<AxiomNumericComputation>,
    axioms_func_comp: &Vec<AxiomFunctionalComparison>,
    constraint: &Option<GlobalConstraint>,
) {
    to_sas_at_path(
        orig_vars,
        orig_numeric_vars,
        ordered_vars,
        ordered_numeric_vars,
        metric,
        mutexes,
        initial_state,
        goals,
        operators,
        axioms_rel,
        axioms_func_ass,
        axioms_func_comp,
        constraint,
        std::path::Path::new("output"),
    );
}

pub fn to_sas_at_path(
    orig_vars: &[ExplicitVariable],
    orig_numeric_vars: &[NumericVariable],
    ordered_vars: &[ExplicitVariable],
    ordered_numeric_vars: &[NumericVariable],
    metric: &Metric,
    mutexes: &Vec<MutexGroup>,
    initial_state: &State,
    goals: &Vec<ExplicitFact>,
    operators: &Vec<Operator>,
    axioms_rel: &Vec<AxiomRelational>,
    axioms_func_ass: &Vec<AxiomNumericComputation>,
    axioms_func_comp: &Vec<AxiomFunctionalComparison>,
    constraint: &Option<GlobalConstraint>,
    output_path: &std::path::Path,
) {
    let mut outfile = std::fs::File::create(output_path)
        .unwrap_or_else(|_| panic!("open output {}", output_path.display()));

    writeln!(outfile, "begin_version").unwrap();
    writeln!(outfile, "{}", PRE_FILE_VERSION).unwrap();
    writeln!(outfile, "end_version").unwrap();

    writeln!(outfile, "begin_metric").unwrap();
    writeln!(
        outfile,
        "{} {}",
        metric.optimization_criterion, metric.index
    )
    .unwrap();
    writeln!(outfile, "end_metric").unwrap();

    let num_vars = ordered_vars.len();
    writeln!(outfile, "{}", num_vars).unwrap();
    if DEBUG {
        println!("Variables in output are: ");
    }
    for var in ordered_vars {
        var.to_sas(&mut outfile);
        if DEBUG {
            print!("{} {{", var.get_name());
            for i in 0..var.get_range() {
                print!("{}, ", var.get_fact_name(i));
            }
            println!("}}");
            println!("Initial value = {}", initial_state.get(var.index));
        }
    }

    if DEBUG {
        println!("Numeric Variables in output are: ");
    }
    writeln!(outfile, "{}", ordered_numeric_vars.len()).unwrap();
    writeln!(outfile, "begin_numeric_variables").unwrap();
    for numeric_var in ordered_numeric_vars {
        if DEBUG {
            numeric_var.dump();
        }
        numeric_var.to_sas(&mut outfile);
    }
    writeln!(outfile, "end_numeric_variables").unwrap();

    writeln!(outfile, "{}", mutexes.len()).unwrap();
    for mutex in mutexes {
        mutex.to_sas(&mut outfile, orig_vars);
    }

    writeln!(outfile, "begin_state").unwrap();
    for var in ordered_vars {
        writeln!(outfile, "{}", initial_state.get(var.index)).unwrap();
    }
    writeln!(outfile, "end_state").unwrap();
    writeln!(outfile, "begin_numeric_state").unwrap();
    for numvar in ordered_numeric_vars {
        writeln!(outfile, "{}", initial_state.get_nv(numvar.index)).unwrap();
    }
    writeln!(outfile, "end_numeric_state").unwrap();

    let mut ordered_goal_values: Vec<i32> = vec![-1; num_vars];
    for goal in goals {
        let var_index = orig_vars[goal.var].get_level();
        if var_index > -1 {
            ordered_goal_values[var_index as usize] = goal.value as i32;
        }
    }
    writeln!(outfile, "begin_goal").unwrap();
    writeln!(outfile, "{}", goals.len()).unwrap();
    for i in 0..num_vars {
        if ordered_goal_values[i] != -1 {
            writeln!(outfile, "{} {}", i, ordered_goal_values[i]).unwrap();
        }
    }
    writeln!(outfile, "end_goal").unwrap();

    writeln!(outfile, "{}", operators.len()).unwrap();
    for op in operators {
        op.to_sas(&mut outfile, orig_vars, orig_numeric_vars);
    }

    writeln!(outfile, "{}", axioms_rel.len()).unwrap();
    for ax in axioms_rel {
        ax.to_sas(&mut outfile, orig_vars);
    }

    writeln!(outfile, "{}", axioms_func_comp.len()).unwrap();
    writeln!(outfile, "begin_comparison_axioms").unwrap();
    for ax in axioms_func_comp {
        ax.to_sas(&mut outfile, orig_vars, orig_numeric_vars);
    }
    writeln!(outfile, "end_comparison_axioms").unwrap();

    writeln!(outfile, "{}", axioms_func_ass.len()).unwrap();
    writeln!(outfile, "begin_numeric_axioms").unwrap();
    for ax in axioms_func_ass {
        ax.to_sas(&mut outfile, orig_numeric_vars);
    }
    writeln!(outfile, "end_numeric_axioms").unwrap();

    if let Some(gc) = constraint {
        writeln!(outfile, "begin_global_constraint").unwrap();
        writeln!(outfile, "{} {}", orig_vars[gc.var].get_level(), gc.value).unwrap();
        writeln!(outfile, "end_global_constraint").unwrap();
    }

    writeln!(outfile, "begin_SG").unwrap();
}
