
pub struct InvalidIndex {
    pub length: u32,
    pub index: u32,
}

pub struct WrongAxiomLayer {
    pub axiom_layer: i32,
    pub last_arithmetic_axiom_layer: i32,
}

pub enum AxiomEvalError {
    InvalidIndex(InvalidIndex),
    WrongAxiomLayer(WrongAxiomLayer),
}