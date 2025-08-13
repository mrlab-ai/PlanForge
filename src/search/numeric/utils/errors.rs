#[derive(Debug)]
pub struct InvalidIndex {
    pub length: u32,
    pub index: u32,
}

#[derive(Debug)]
pub struct ConstructError {
    pub message: String,
}

#[derive(Debug)]
pub struct WrongAxiomLayer {
    pub axiom_layer: i32,
    pub last_arithmetic_axiom_layer: i32,
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

pub struct StateInsertError {
    pub message: String,
}