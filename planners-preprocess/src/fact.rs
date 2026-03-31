#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct ExplicitFact {
    pub var: usize,
    pub value: usize,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
pub struct NumericFact {
    pub var: usize,
    pub value: f64,
}
