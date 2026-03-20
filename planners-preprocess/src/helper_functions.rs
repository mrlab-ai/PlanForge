use std::cmp::max;
use std::io::{Read, Write};

use crate::variable::{NumericVariable, Variable};

pub const SAS_FILE_VERSION: i32 = 4;
pub const PRE_FILE_VERSION: i32 = SAS_FILE_VERSION;

pub const DEBUG: bool = false;

#[derive(Debug, Clone)]
pub struct Metric {
    pub optimization_criterion: char,
    pub index: i32,
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalConstraint {
    pub var: *mut crate::variable::Variable,
    pub val: i32,
}

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
    Ue = 5,
}

impl FOperator {
    pub fn from_str(s: &str) -> Self {
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
    pub fn from_str(s: &str) -> Self {
        match s {
            "<" => CompOperator::Lt,
            "<=" => CompOperator::Le,
            "=" => CompOperator::Eq,
            ">=" => CompOperator::Ge,
            ">" => CompOperator::Gt,
            "!=" => CompOperator::Ue,
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
            CompOperator::Ue => write!(f, "!="),
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
        CompOperator::Ue => ("!=".to_string(), "=".to_string()),
    }
}

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

pub fn check_and_repair_empty_axiom_layers(
    numeric_variables: &[*mut NumericVariable],
    variables: &mut [*mut Variable],
) {
    let mut max_num_index_before = -1;
    let mut max_num_index_after = -1;
    for nvar in numeric_variables {
        let nvar_ref = unsafe { &**nvar };
        max_num_index_before = max(max_num_index_before, nvar_ref.get_layer());
        if nvar_ref.is_necessary() {
            max_num_index_after = max(max_num_index_after, nvar_ref.get_layer());
        }
    }
    if max_num_index_before != max_num_index_after {
        if DEBUG {
            println!(
                "index before = {} after = {}",
                max_num_index_before, max_num_index_after
            );
        }
        let decrement = max_num_index_before - max_num_index_after;
        for var in variables {
            let var_ref = unsafe { &mut **var };
            var_ref.decrement_layer(decrement);
            assert!(var_ref.get_layer() == -1 || var_ref.get_layer() > max_num_index_after);
        }
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

fn read_metric(stream: &mut InputStream, metric: &mut Metric) {
    check_magic(stream, "begin_metric");
    metric.optimization_criterion = stream.read_char();
    metric.index = stream.read_i32();
    assert!(metric.index >= 0);
    check_magic(stream, "end_metric");
}

fn read_variables(
    stream: &mut InputStream,
    internal_variables: &mut Vec<Variable>,
    variables: &mut Vec<*mut Variable>,
) {
    let count = stream.read_i32();
    internal_variables.reserve(count as usize);
    for _ in 0..count {
        internal_variables.push(Variable::from_stream(stream));
        let ptr: *mut Variable = internal_variables.last_mut().unwrap();
        variables.push(ptr);
    }
}

fn read_numeric_variables(
    stream: &mut InputStream,
    internal_numeric_variables: &mut Vec<NumericVariable>,
    numeric_variables: &mut Vec<*mut NumericVariable>,
) {
    let count = stream.read_i32();
    internal_numeric_variables.reserve(count as usize);
    check_magic(stream, "begin_numeric_variables");
    for _ in 0..count {
        internal_numeric_variables.push(NumericVariable::from_stream(stream));
        let ptr: *mut NumericVariable = internal_numeric_variables.last_mut().unwrap();
        numeric_variables.push(ptr);
    }
    check_magic(stream, "end_numeric_variables");
}

fn read_mutexes(
    stream: &mut InputStream,
    mutexes: &mut Vec<crate::mutex_group::MutexGroup>,
    variables: &Vec<*mut Variable>,
) {
    let count = stream.read_i32();
    for _ in 0..count {
        mutexes
            .push(crate::mutex_group::MutexGroup::from_stream(stream, variables));
    }
}

fn read_global_constraint(
    stream: &mut InputStream,
    variables: &Vec<*mut Variable>,
    gc: &mut GlobalConstraint,
) {
    check_magic(stream, "begin_global_constraint");
    let index = stream.read_i32();
    let val = stream.read_i32();
    gc.var = variables[index as usize];
    gc.val = val;
    println!("read global constraint at var {} value {}", index, val);
    unsafe { &*gc.var }.dump();
    check_magic(stream, "end_global_constraint");
}

fn read_goal(
    stream: &mut InputStream,
    variables: &Vec<*mut Variable>,
    goals: &mut Vec<(*mut Variable, i32)>,
) {
    check_magic(stream, "begin_goal");
    let count = stream.read_i32();
    for _ in 0..count {
        let var_no = stream.read_i32();
        let val = stream.read_i32();
        goals.push((variables[var_no as usize], val));
    }
    check_magic(stream, "end_goal");
}

fn dump_goal(goals: &Vec<(*mut Variable, i32)>) {
    println!("Goal Conditions:");
    for goal in goals {
        let var = unsafe { &*goal.0 };
        println!("  {}: {}", var.get_name(), goal.1);
    }
}

fn read_operators(
    stream: &mut InputStream,
    variables: &Vec<*mut Variable>,
    numeric_variables: &Vec<*mut NumericVariable>,
    operators: &mut Vec<crate::operator::Operator>,
) {
    let count = stream.read_i32();
    for _ in 0..count {
        operators.push(crate::operator::Operator::from_stream(
            stream,
            variables,
            numeric_variables,
        ));
    }
}

fn read_axioms_rel(
    stream: &mut InputStream,
    variables: &Vec<*mut Variable>,
    axioms_rel: &mut Vec<crate::axiom::AxiomRelational>,
) {
    let count = stream.read_i32();
    for _ in 0..count {
        axioms_rel.push(crate::axiom::AxiomRelational::from_stream(
            stream, variables,
        ));
    }
    axioms_rel.sort_by(|a, b| {
        let la = unsafe { &*a.get_effect_var() }.get_layer();
        let lb = unsafe { &*b.get_effect_var() }.get_layer();
        la.cmp(&lb)
    });
}

fn read_axioms_func_comp(
    stream: &mut InputStream,
    variables: &mut Vec<*mut Variable>,
    numeric_variables: &Vec<*mut NumericVariable>,
    axioms_func_comp: &mut Vec<crate::axiom::AxiomFunctionalComparison>,
) {
    let count = stream.read_i32();
    check_magic(stream, "begin_comparison_axioms");
    for _ in 0..count {
        axioms_func_comp.push(
            crate::axiom::AxiomFunctionalComparison::from_stream(
                stream,
                variables,
                numeric_variables,
            ),
        );
    }
    check_magic(stream, "end_comparison_axioms");
    axioms_func_comp.sort_by(|a, b| {
        let la = unsafe { &*a.get_effect_var() }.get_layer();
        let lb = unsafe { &*b.get_effect_var() }.get_layer();
        la.cmp(&lb)
    });
}

fn read_axioms_numeric(
    stream: &mut InputStream,
    numeric_variables: &mut Vec<*mut NumericVariable>,
    axioms_numeric: &mut Vec<crate::axiom::AxiomNumericComputation>,
) {
    let count = stream.read_i32();
    check_magic(stream, "begin_numeric_axioms");
    for _ in 0..count {
        axioms_numeric.push(
            crate::axiom::AxiomNumericComputation::from_stream(
                stream,
                numeric_variables,
            ),
        );
    }
    check_magic(stream, "end_numeric_axioms");
    axioms_numeric.sort_by(|a, b| {
        let la = unsafe { &*a.get_effect_var() }.get_layer();
        let lb = unsafe { &*b.get_effect_var() }.get_layer();
        la.cmp(&lb)
    });
}

pub fn read_preprocessed_problem_description(
    stream: &mut InputStream,
    metric: &mut Metric,
    internal_variables: &mut Vec<Variable>,
    variables: &mut Vec<*mut Variable>,
    internal_numeric_variables: &mut Vec<NumericVariable>,
    numeric_variables: &mut Vec<*mut NumericVariable>,
    mutexes: &mut Vec<crate::mutex_group::MutexGroup>,
    initial_state: &mut crate::state::State,
    goals: &mut Vec<(*mut Variable, i32)>,
    operators: &mut Vec<crate::operator::Operator>,
    axioms_rel: &mut Vec<crate::axiom::AxiomRelational>,
    axioms_func_ass: &mut Vec<crate::axiom::AxiomNumericComputation>,
    axioms_func_comp: &mut Vec<crate::axiom::AxiomFunctionalComparison>,
    gconstraint: &mut GlobalConstraint,
) {
    if DEBUG {
        println!("reading version...");
    }
    read_and_verify_version(stream);
    if DEBUG {
        println!("reading metric...");
    }
    read_metric(stream, metric);
    if DEBUG {
        println!("reading variables...");
    }
    read_variables(stream, internal_variables, variables);
    if DEBUG {
        println!("reading numeric variables...");
    }
    read_numeric_variables(stream, internal_numeric_variables, numeric_variables);
    if DEBUG {
        println!("reading mutexes...");
    }
    read_mutexes(stream, mutexes, variables);
    if DEBUG {
        println!("reading initial state...");
    }
    *initial_state =
        crate::state::State::from_stream(stream, variables, numeric_variables);
    read_goal(stream, variables, goals);
    if DEBUG {
        println!("reading operators...");
    }
    read_operators(stream, variables, numeric_variables, operators);
    if DEBUG {
        println!("reading propositional axioms...");
    }
    read_axioms_rel(stream, variables, axioms_rel);
    if DEBUG {
        println!("reading functional comparison axioms...");
    }
    read_axioms_func_comp(stream, variables, numeric_variables, axioms_func_comp);
    if DEBUG {
        println!("reading functional assignment axioms...");
    }
    read_axioms_numeric(stream, numeric_variables, axioms_func_ass);
    if DEBUG {
        println!("reading global constraint");
    }
    read_global_constraint(stream, variables, gconstraint);
}

pub fn dump_preprocessed_problem_description(
    variables: &Vec<*mut Variable>,
    initial_state: &crate::state::State,
    goals: &Vec<(*mut Variable, i32)>,
    operators: &Vec<crate::operator::Operator>,
    axioms_rel: &Vec<crate::axiom::AxiomRelational>,
    axioms_func_ass: &Vec<crate::axiom::AxiomNumericComputation>,
    axioms_func_comp: &Vec<crate::axiom::AxiomFunctionalComparison>,
) {
    println!("Variables ({}):", variables.len());
    for var in variables {
        unsafe { &**var }.dump();
    }
    println!("Initial State:");
    initial_state.dump();
    dump_goal(goals);
    for op in operators {
        op.dump();
    }
    for ax in axioms_rel {
        ax.dump();
    }
    for ax in axioms_func_ass {
        ax.dump();
    }
    for ax in axioms_func_comp {
        ax.dump();
    }
}

pub fn dump_dtgs(
    ordering: &Vec<*mut Variable>,
    transition_graphs: &mut Vec<
        crate::domain_transition_graph::DomainTransitionGraph,
    >,
) {
    let num_graphs = transition_graphs.len();
    for i in 0..num_graphs {
        let name = unsafe { &*ordering[i] }.get_name();
        println!("Domain transition graph for {}:", name);
        transition_graphs[i].dump();
    }
}

pub fn generate_cpp_input(
    solveable_in_poly_time: bool,
    ordered_vars: &Vec<*mut Variable>,
    numeric_vars: &Vec<*mut NumericVariable>,
    metric: &Metric,
    mutexes: &Vec<crate::mutex_group::MutexGroup>,
    initial_state: &crate::state::State,
    goals: &Vec<(*mut Variable, i32)>,
    operators: &Vec<crate::operator::Operator>,
    axioms_rel: &Vec<crate::axiom::AxiomRelational>,
    axioms_func_ass: &Vec<crate::axiom::AxiomNumericComputation>,
    axioms_func_comp: &Vec<crate::axiom::AxiomFunctionalComparison>,
    constraint: &GlobalConstraint,
) {
    let mut outfile = std::fs::File::create("output").expect("open output");

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
        unsafe { &**var }.generate_cpp_input(&mut outfile);
        if DEBUG {
            let var_ref = unsafe { &**var };
            print!("{} {{", var_ref.get_name());
            for i in 0..var_ref.get_range() {
                print!("{}, ", var_ref.get_fact_name(i as usize));
            }
            println!("}}");
            println!("Initial value = {}", initial_state.get(*var));
        }
    }

    if DEBUG {
        println!("Numeric Variables in output are: ");
    }
    writeln!(outfile, "{}", numeric_vars.len()).unwrap();
    writeln!(outfile, "begin_numeric_variables").unwrap();
    for numeric_var in numeric_vars {
        if DEBUG {
            unsafe { &**numeric_var }.dump();
        }
        unsafe { &**numeric_var }.generate_cpp_input(&mut outfile);
    }
    writeln!(outfile, "end_numeric_variables").unwrap();

    writeln!(outfile, "{}", mutexes.len()).unwrap();
    for mutex in mutexes {
        mutex.generate_cpp_input(&mut outfile);
    }

    writeln!(outfile, "begin_state").unwrap();
    for var in ordered_vars {
        writeln!(outfile, "{}", initial_state.get(*var)).unwrap();
    }
    writeln!(outfile, "end_state").unwrap();
    writeln!(outfile, "begin_numeric_state").unwrap();
    for numvar in numeric_vars {
        writeln!(outfile, "{}", initial_state.get_nv(*numvar)).unwrap();
    }
    writeln!(outfile, "end_numeric_state").unwrap();

    let mut ordered_goal_values: Vec<i32> = vec![-1; num_vars];
    for goal in goals {
        let var_index = unsafe { &*goal.0 }.get_level();
        ordered_goal_values[var_index as usize] = goal.1;
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
        op.generate_cpp_input(&mut outfile);
    }

    writeln!(outfile, "{}", axioms_rel.len()).unwrap();
    for ax in axioms_rel {
        ax.generate_cpp_input(&mut outfile);
    }

    writeln!(outfile, "{}", axioms_func_comp.len()).unwrap();
    writeln!(outfile, "begin_comparison_axioms").unwrap();
    for ax in axioms_func_comp {
        ax.generate_cpp_input(&mut outfile);
    }
    writeln!(outfile, "end_comparison_axioms").unwrap();

    writeln!(outfile, "{}", axioms_func_ass.len()).unwrap();
    writeln!(outfile, "begin_numeric_axioms").unwrap();
    for ax in axioms_func_ass {
        ax.generate_cpp_input(&mut outfile);
    }
    writeln!(outfile, "end_numeric_axioms").unwrap();

    writeln!(outfile, "begin_global_constraint").unwrap();
    let gc_var = unsafe { &*constraint.var };
    writeln!(outfile, "{} {}", gc_var.get_level(), constraint.val).unwrap();
    writeln!(outfile, "end_global_constraint").unwrap();

    writeln!(outfile, "begin_SG").unwrap();

    let _ = solveable_in_poly_time;
}
