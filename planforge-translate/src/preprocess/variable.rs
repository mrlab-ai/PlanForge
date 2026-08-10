use std::fmt;
use std::io::Write;

use tracing::debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumType {
    Unknown = 0,
    Constant = 1,
    Derived = 2,
    Instrumentation = 3,
    Regular = 4,
}

impl fmt::Display for NumType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NumType::Constant => write!(f, "C"),
            NumType::Regular => write!(f, "R"),
            NumType::Derived => write!(f, "D"),
            NumType::Instrumentation => write!(f, "I"),
            NumType::Unknown => panic!("Type of numeric variable not recognized"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct ExplicitVariable {
    pub index: usize,
    name: String,
    values: Vec<String>,
    layer: i32,
    level: i32,
    necessary: bool,
    comparison: bool,
}

impl PartialEq for ExplicitVariable {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for ExplicitVariable {}

impl PartialOrd for ExplicitVariable {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ExplicitVariable {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.index < other.index {
            std::cmp::Ordering::Less
        } else if self.index > other.index {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

impl ExplicitVariable {
    /// The variable is named after the position it had before the causal graph
    /// reordered it, which is how the SAS file keeps a variable identifiable
    /// across the reordering.
    pub fn new(index: usize, layer: i32, values: Vec<String>) -> Self {
        Self {
            index,
            name: format!("var{index}"),
            values,
            layer,
            level: -1,
            necessary: false,
            comparison: false,
        }
    }

    pub fn set_level(&mut self, level: i32) {
        assert_eq!(self.level, -1);
        self.level = level;
    }

    pub fn set_necessary(&mut self) {
        assert!(!self.necessary);
        self.necessary = true;
    }

    pub fn get_level(&self) -> i32 {
        self.level
    }

    pub fn is_necessary(&self) -> bool {
        self.necessary
    }

    pub fn get_range(&self) -> usize {
        self.values.len()
    }

    pub fn set_comparison(&mut self) {
        self.comparison = true;
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn get_layer(&self) -> i32 {
        self.layer
    }

    pub fn decrement_layer(&mut self, decrement: i32) {
        if self.layer != -1 {
            self.layer -= decrement;
        }
    }

    pub fn is_derived(&self) -> bool {
        self.layer != -1
    }

    pub fn to_sas<W: Write>(&self, out: &mut W) {
        writeln!(out, "begin_variable").unwrap();
        writeln!(out, "{}", self.name).unwrap();
        writeln!(out, "{}", self.layer).unwrap();
        writeln!(out, "{}", self.values.len()).unwrap();
        for v in &self.values {
            writeln!(out, "{}", v).unwrap();
        }
        writeln!(out, "end_variable").unwrap();
    }

    pub fn dump(&self) {
        debug!("{} [range {}", self.name, self.get_range());
        if self.level != -1 {
            debug!("; level {}", self.level);
        }
        if self.is_derived() {
            debug!("; derived; layer: {}", self.layer);
        }
        debug!("] {{");
        for fact in &self.values {
            debug!("{}, ", fact);
        }
        debug!("}}");
    }

    pub fn get_fact_name(&self, value: usize) -> String {
        self.values[value].clone()
    }

    pub fn set_fact_name(&mut self, value: usize, new_name: String) {
        assert!(value < self.values.len());
        self.values[value] = new_name;
    }
}

#[derive(Debug, Clone)]
pub struct NumericVariable {
    pub index: usize,
    name: String,
    layer: i32,
    level: i32,
    necessary: bool,
    subterm: bool,
    ntype: NumType,
}

impl PartialEq for NumericVariable {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl Eq for NumericVariable {}

impl PartialOrd for NumericVariable {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NumericVariable {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.index < other.index {
            std::cmp::Ordering::Less
        } else if self.index > other.index {
            std::cmp::Ordering::Greater
        } else {
            std::cmp::Ordering::Equal
        }
    }
}

impl NumericVariable {
    /// `R` and `I` deliberately arrive as `Unknown`: which numeric variables
    /// are regular and which are instrumentation is decided again here, from
    /// the metric and the assignment axioms that feed it, not taken from the
    /// translation.
    pub fn new(index: usize, sas_type: &str, layer: i32, name: String) -> Self {
        let ntype = match sas_type {
            "C" => NumType::Constant,
            "D" => NumType::Derived,
            "R" | "I" => NumType::Unknown,
            other => panic!("numeric variable {index} has an unknown type {other:?}"),
        };

        Self {
            index,
            name,
            layer,
            level: -1,
            necessary: false,
            subterm: false,
            ntype,
        }
    }

    pub fn set_level(&mut self, new_level: i32) {
        assert_eq!(self.level, -1);
        self.level = new_level;
    }

    pub fn set_necessary(&mut self) {
        assert!(!self.necessary);
        self.necessary = true;
        if self.ntype == NumType::Unknown {
            self.ntype = NumType::Regular;
        }
    }

    pub fn set_instrumentation(&mut self) {
        assert!(!self.necessary);
        self.necessary = true;
        if self.ntype == NumType::Unknown {
            self.ntype = NumType::Instrumentation;
        }
    }

    pub fn is_necessary(&self) -> bool {
        self.necessary
    }

    pub fn get_level(&self) -> i32 {
        self.level
    }

    pub fn set_subterm(&mut self) {
        self.subterm = true;
    }

    pub fn get_name(&self) -> String {
        self.name.clone()
    }

    pub fn get_layer(&self) -> i32 {
        self.layer
    }

    pub fn is_derived(&self) -> bool {
        self.ntype == NumType::Derived
    }

    pub fn get_type(&self) -> NumType {
        self.ntype
    }

    pub fn to_sas<W: Write>(&self, out: &mut W) {
        assert!(self.necessary);
        assert!(self.layer >= -1);
        writeln!(out, "{} {} {}", self.ntype, self.layer, self.name).unwrap();
    }

    pub fn dump(&self) {
        debug!("nv{} : >{}", self.level, self.name);
        if self.level != -1 {
            debug!("; level {}", self.level);
        }
        if self.is_derived() {
            debug!("; derived; layer: {}", self.layer);
        }
        debug!("<");
    }
}
