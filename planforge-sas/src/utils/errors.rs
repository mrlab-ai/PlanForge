/// The one way evaluating the arithmetic axioms in a state can fail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AssignmentAxiomError {
    DivisionByZero { divisor: usize },
}

#[derive(Debug)]
pub struct ConstructError {
    pub message: String,
}

impl std::fmt::Display for AssignmentAxiomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DivisionByZero { divisor } => write!(
                f,
                "assignment axiom divides by numeric variable {divisor}, which is zero"
            ),
        }
    }
}

impl std::error::Error for AssignmentAxiomError {}

#[derive(Debug)]
pub struct StateNotFoundError {
    pub index: usize,
}

#[derive(Debug)]
pub struct StateInsertError {
    pub message: String,
}

impl std::fmt::Display for StateInsertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for StateInsertError {}
