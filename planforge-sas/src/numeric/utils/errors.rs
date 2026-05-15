#[derive(Debug)]
pub struct InvalidIndex {
    pub length: usize,
    pub index: usize,
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
    WrongAxiomLayer(WrongAxiomLayer),
}

#[derive(Debug)]
pub struct StateNotFoundError {
    pub index: usize,
}

#[derive(Debug)]
pub struct StateInsertError {
    pub message: String,
}
