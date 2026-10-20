#[derive(Debug)]
pub struct InvalidIndex {
    pub length: usize,
    pub index: usize,
}

#[derive(Debug)]
pub enum AssignmentAxiomError {
    InvalidIndex(InvalidIndex),
    DivisionByZero { divisor: usize },
}

#[derive(Debug)]
pub struct ConstructError {
    pub message: String,
}

#[derive(Debug)]
pub struct WrongAxiomLayer {
    pub axiom_layer: Option<usize>,
    pub last_arithmetic_axiom_layer: Option<usize>,
}

#[derive(Debug)]
pub enum AxiomEvalError {
    InvalidIndex(InvalidIndex),
    Assignment(AssignmentAxiomError),
    WrongAxiomLayer(WrongAxiomLayer),
}

impl std::fmt::Display for InvalidIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "index {} is out of bounds for length {}",
            self.index, self.length
        )
    }
}

impl std::fmt::Display for AssignmentAxiomError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIndex(error) => write!(f, "assignment axiom reads {error}"),
            Self::DivisionByZero { divisor } => write!(
                f,
                "assignment axiom divides by numeric variable {divisor}, which is zero"
            ),
        }
    }
}

impl std::fmt::Display for AxiomEvalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidIndex(error) => write!(f, "axiom evaluation reads {error}"),
            Self::Assignment(error) => error.fmt(f),
            Self::WrongAxiomLayer(error) => write!(
                f,
                "a derived variable has axiom layer {:?}, outside the propositional layers after the last arithmetic layer {:?}",
                error.axiom_layer, error.last_arithmetic_axiom_layer
            ),
        }
    }
}

impl std::error::Error for AssignmentAxiomError {}

impl std::error::Error for AxiomEvalError {}

#[derive(Debug)]
pub struct StateNotFoundError {
    pub index: usize,
}

#[derive(Debug)]
pub struct StateInsertError {
    pub message: String,
}
