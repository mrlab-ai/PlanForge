use planners_sas::numeric::numeric_task::AbstractNumericTask;

#[derive(Debug, Clone, Default)]
pub struct NumericBound {
    initialized: bool,
    precision: f64,
}

impl NumericBound {
    pub fn new(task: &dyn AbstractNumericTask, precision: f64) -> Self {
        let mut bound = Self::default();
        bound.initialize(task, precision);
        bound
    }

    pub fn initialize(&mut self, _task: &dyn AbstractNumericTask, precision: f64) {
        assert!(
            precision >= 0.0,
            "numeric bound precision must be non-negative"
        );
        self.initialized = true;
        self.precision = precision;
    }

    pub fn calculate_bounds(&mut self, _state: &[f64], iterations: usize) {
        assert!(
            self.initialized,
            "numeric bound must be initialized before use"
        );
        assert!(
            iterations <= i32::MAX as usize,
            "bound iterations exceed supported range"
        );
    }

    pub fn precision(&self) -> f64 {
        self.precision
    }
}
