use std::fmt;
use std::io::Write;

use crate::helper_functions::check_magic;
use crate::helper_functions::InputStream;

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
pub struct Variable {
    values: Vec<String>,
    name: String,
    layer: i32,
    level: i32,
    necessary: bool,
    comparison: bool,
}

impl Variable {
    pub fn from_stream(stream: &mut InputStream) -> Self {
        check_magic(stream, "begin_variable");
        let name = stream.read_token();
        let layer = stream.read_i32();
        let range = stream.read_i32();
        stream.skip_ws();
        let mut values = Vec::with_capacity(range as usize);
        for _ in 0..range {
            values.push(stream.read_line());
        }
        check_magic(stream, "end_variable");
        Self {
            values,
            name,
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

    pub fn get_range(&self) -> i32 {
        self.values.len() as i32
    }

    pub fn is_comparison(&self) -> bool {
        self.comparison
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

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W) {
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
        print!("{} [range {}", self.name, self.get_range());
        if self.level != -1 {
            print!("; level {}", self.level);
        }
        if self.is_derived() {
            print!("; derived; layer: {}", self.layer);
        }
        print!("] {{");
        for fact in &self.values {
            print!("{}, ", fact);
        }
        println!("}}");
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
    name: String,
    value: f64,
    layer: i32,
    level: i32,
    necessary: bool,
    subterm: bool,
    ntype: NumType,
}

impl NumericVariable {
    pub fn from_stream(stream: &mut InputStream) -> Self {
        let nvtype = stream.read_char();
        stream.skip_ws();
        let layer_str = stream.read_until(' ');
        let layer = layer_str.parse::<i32>().unwrap_or(0);
        let name = stream.read_line();

        let mut ntype = NumType::Unknown;
        if nvtype == 'C' {
            ntype = NumType::Constant;
        }
        if nvtype == 'D' {
            ntype = NumType::Derived;
        }

        Self {
            name,
            value: 0.0,
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

    pub fn set_value(&mut self, new_value: f64) {
        self.value = new_value;
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

    pub fn set_layer(&mut self, new_layer: i32) {
        self.layer = new_layer;
    }

    pub fn is_necessary(&self) -> bool {
        self.necessary
    }

    pub fn get_level(&self) -> i32 {
        self.level
    }

    pub fn is_subterm(&self) -> bool {
        self.subterm
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

    pub fn generate_cpp_input<W: Write>(&self, out: &mut W) {
        assert!(self.necessary);
        assert!(self.layer >= -1);
        writeln!(out, "{} {} {}", self.ntype, self.layer, self.name).unwrap();
    }

    pub fn dump(&self) {
        print!("nv{} : >{}", self.level, self.name);
        if self.level != -1 {
            print!("; level {}", self.level);
        }
        if self.is_derived() {
            print!("; derived; layer: {}", self.layer);
        }
        println!("<");
    }
}
