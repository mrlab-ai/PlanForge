//! A planning task as a Python object: translation and loading, its
//! variables, operators and states, successor generation and search.

use super::*;

#[pyclass(unsendable)]
pub(crate) struct Task {
    task: Arc<planforge_sas::numeric_task::Task>,
    registry: RefCell<StateRegistry>,
    succ: StandardGroundedTree,
}

#[pymethods]
impl Task {
    #[staticmethod]
    #[pyo3(signature = (text, transforms=Vec::new()))]
    fn from_sas_text(text: &str, transforms: Vec<String>) -> PyResult<Self> {
        let task = transform_numeric_task(
            planforge_sas::numeric_task::Task::try_from_str(text)
                .map_err(|error| ParseError::new_err(error.to_string()))?,
            &transforms,
            &TaskRequirements::default(),
        )
        .map_err(PlanforgeError::new_err)?;
        Ok(Self::build(Arc::new(task)))
    }

    #[staticmethod]
    #[pyo3(signature = (path, transforms=Vec::new()))]
    fn from_sas(path: PathBuf, transforms: Vec<String>) -> PyResult<Self> {
        let text = std::fs::read_to_string(&path).map_err(|e| match e.kind() {
            std::io::ErrorKind::NotFound => {
                PyFileNotFoundError::new_err(format!("{}: {e}", path.display()))
            }
            _ => ParseError::new_err(format!("failed to read {}: {e}", path.display())),
        })?;
        Self::from_sas_text(&text, transforms)
    }

    #[staticmethod]
    #[pyo3(signature = (domain, problem, transforms=Vec::new()))]
    fn from_pddl(
        py: Python<'_>,
        domain: PathBuf,
        problem: PathBuf,
        transforms: Vec<String>,
    ) -> PyResult<Self> {
        let task = py
            .allow_threads(|| -> Result<planforge_sas::numeric_task::Task, String> {
                translate_pddl_to_task(&domain, &problem)
            })
            .map_err(TranslateError::new_err)?;
        let task = transform_numeric_task(task, &transforms, &TaskRequirements::default())
            .map_err(PlanforgeError::new_err)?;
        Ok(Self::build(Arc::new(task)))
    }

    #[getter]
    fn num_variables(&self) -> usize {
        self.task.variables().len()
    }

    #[getter]
    fn num_numeric_variables(&self) -> usize {
        self.task.numeric_variables().len()
    }

    #[getter]
    fn num_operators(&self) -> usize {
        self.task.operators().len()
    }

    /// The number of ground atoms: every value of every finite-domain variable.
    #[getter]
    fn num_atoms(&self) -> usize {
        self.task
            .variables()
            .iter()
            .map(|variable| variable.domain_size())
            .sum()
    }

    #[getter]
    fn num_goals(&self) -> usize {
        self.task.goals().len()
    }

    #[getter]
    fn goals(&self) -> Vec<(usize, usize)> {
        self.task
            .goals()
            .iter()
            .map(|f| (f.var(), f.value()))
            .collect()
    }

    #[getter]
    fn metric(&self) -> bool {
        self.task.metric().use_metric()
    }

    #[getter]
    fn variable_names(&self) -> Vec<String> {
        (0..self.task.variables().len())
            .map(|i| {
                self.task
                    .variable(VariableIndex::from_usize(i))
                    .name()
                    .to_string()
            })
            .collect()
    }

    #[getter]
    fn variable_domain_sizes(&self) -> Vec<usize> {
        self.task
            .variables()
            .iter()
            .map(|variable| variable.domain_size())
            .collect()
    }

    #[getter]
    fn numeric_variable_names(&self) -> Vec<String> {
        self.task
            .numeric_variables()
            .iter()
            .map(|variable| variable.name().to_string())
            .collect()
    }

    #[getter]
    fn numeric_variable_types(&self) -> Vec<String> {
        self.task
            .numeric_variables()
            .iter()
            .map(|variable| numeric_type_name(variable.numeric_type()).to_string())
            .collect()
    }

    #[getter]
    fn registered_states(&self) -> usize {
        self.registry.borrow().num_registered_states()
    }

    /// Every ground atom of the task, ordered by variable and then by value.
    fn atoms(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAtom>>> {
        self.task
            .variables()
            .iter()
            .enumerate()
            .flat_map(|(variable, explicit_variable)| {
                (0..explicit_variable.domain_size()).map(move |value| (variable, value))
            })
            .map(|(variable, value)| {
                let name = self
                    .task
                    .fact_name(&ExplicitFact::propositional(variable, value))
                    .to_string();
                let (negated, predicate, arguments) = split_fact_name(&name);
                Py::new(
                    py,
                    PyAtom {
                        variable: variable as u32,
                        value: value as u32,
                        name,
                        predicate,
                        arguments,
                        negated,
                    },
                )
            })
            .collect()
    }

    /// Every ground action of the task, ordered by operator id.
    fn operators(&self, py: Python<'_>) -> Vec<Py<PyOperator>> {
        let task_id = self.registry.borrow().id();
        self.task
            .operators()
            .iter()
            .enumerate()
            .map(|(operator_id, operator)| {
                operator_to_py(
                    py,
                    operator,
                    Some(operator_id),
                    Some(task_id),
                    self.task
                        .operator_cost(OperatorIndex::from_usize(operator_id)),
                )
                .expect("creating a Python Operator should not fail")
            })
            .collect()
    }

    fn initial_state(&self) -> State {
        let mut reg = self.registry.borrow_mut();
        let s = reg.register_initial_state();
        State::snapshot(&s, &reg)
    }

    fn is_goal(&self, state: &State) -> PyResult<bool> {
        let reg = self.registry.borrow();
        let cstate = self.lookup(state, &reg)?;
        Ok(reg.view(&cstate).holds_all(self.task.goals()))
    }

    /// Return every operator applicable in `state`.
    ///
    /// This is the Python-driven prototyping path: crossing the FFI boundary
    /// once per expansion is deliberate. Pure Rust search never calls it.
    fn applicable_operators(&self, py: Python<'_>, state: &State) -> PyResult<Vec<Py<PyOperator>>> {
        let reg = self.registry.borrow();
        let cstate = self.lookup(state, &reg)?;
        let ids = self.applicable_operator_ids(&cstate, &reg);
        let task_id = reg.id();
        ids.into_iter()
            .map(|operator_id| {
                let operator = self.task.operators().get(operator_id).unwrap_or_else(|| {
                    panic!("successor generator returned invalid operator id {operator_id}")
                });
                operator_to_py(
                    py,
                    operator,
                    Some(operator_id),
                    Some(task_id),
                    self.task
                        .operator_cost(OperatorIndex::from_usize(operator_id)),
                )
            })
            .collect()
    }

    /// Apply an applicable operator, including numeric effects and axiom closure.
    fn apply(&self, state: &State, operator: PyRef<'_, PyOperator>) -> PyResult<State> {
        self.apply_operator_snapshot(state, &operator)
            .map(|(successor, _cost)| successor)
    }

    /// Apply an operator and also return its state-dependent transition cost.
    fn apply_with_cost(
        &self,
        state: &State,
        operator: PyRef<'_, PyOperator>,
    ) -> PyResult<(State, f64)> {
        self.apply_operator_snapshot(state, &operator)
    }

    /// Enumerate the complete reachable graph in Rust and compute exact h*.
    ///
    /// Every bound is mandatory. Hitting one raises `EnumerationError`; no
    /// partial graph and no plausible-but-wrong h* array is returned.
    #[pyo3(signature = (*, max_states, max_transitions, max_time))]
    fn enumerate_state_space(
        &self,
        py: Python<'_>,
        max_states: usize,
        max_transitions: usize,
        max_time: f64,
    ) -> PyResult<PyStateSpace> {
        let limits = EnumerationLimits {
            max_states,
            max_transitions,
            max_time: checked_duration(max_time, "max_time")?,
        };
        let task = GilReleasedTask(self.task.clone());
        let graph = py
            .allow_threads(move || task.enumerate(limits))
            .map_err(|error| EnumerationError::new_err(error.to_string()))?;
        state_space_to_py(graph, &self.task)
    }

    /// (operator, successor_state, transition_cost) for every applicable operator.
    fn successors(
        &self,
        py: Python<'_>,
        state: &State,
    ) -> PyResult<Vec<(Py<PyOperator>, State, f64)>> {
        let mut reg = self.registry.borrow_mut();
        let cstate = self.lookup(state, &reg)?;
        let ids = self.applicable_operator_ids(&cstate, &reg);
        let operators = self.task.operators();
        let task_id = reg.id();
        let mut out = Vec::with_capacity(ids.len());
        for op_id in ids {
            let operator_id = op_id;
            let op = operators.get(operator_id).unwrap_or_else(|| {
                panic!("successor generator returned invalid operator id {operator_id}")
            });
            let (succ, cost) = reg.successor(&cstate, op).map_err(|e| {
                PlanforgeError::new_err(format!(
                    "successor generation failed for {}: {e:?}",
                    op.name()
                ))
            })?;
            let py_op = operator_to_py(
                py,
                op,
                Some(operator_id),
                Some(task_id),
                self.task
                    .operator_cost(OperatorIndex::from_usize(operator_id)),
            )?;
            let snap = State::snapshot(&succ, &reg);
            out.push((py_op, snap, cost));
        }
        Ok(out)
    }

    /// Full search reusing this parsed task; delegates to the same pipeline as
    /// the module-level `solve()`.
    #[pyo3(signature = (search=None, max_time=None, max_memory=None))]
    fn solve(
        &self,
        py: Python<'_>,
        search: Option<String>,
        max_time: Option<f64>,
        max_memory: Option<u64>,
    ) -> PyResult<PySearchResult> {
        let search = search.unwrap_or_else(|| "astar(blind())".to_string());
        let spec = planforge_heuristics::features()
            .parse_search(&search)
            .map_err(|error| SpecError::new_err(error.to_string()))?;
        let time_limit = max_time.map(Duration::from_secs_f64);
        let task = GilReleasedTask(self.task.clone());
        let result = py
            .allow_threads(move || task.solve(&spec, time_limit, max_memory))
            .map_err(|error| PlanforgeError::new_err(search_failure(error)))?;
        Ok(search_result_to_py(py, result))
    }

    /// Run A* or greedy best-first search with a Python heuristic callback.
    ///
    /// The callback receives a value snapshot of each evaluated State for
    /// inspection via `.values` and `.numeric_values`. The snapshot belongs to
    /// the search's internal registry, so `task.successors(state)` rejects it;
    /// guidance heuristics should read state values, not re-explore.
    #[pyo3(signature = (heuristic, greedy=false, max_time=None, max_memory=None))]
    fn search_with_heuristic(
        &self,
        py: Python<'_>,
        heuristic: Py<PyAny>,
        greedy: bool,
        max_time: Option<f64>,
        max_memory: Option<u64>,
    ) -> PyResult<PySearchResult> {
        let error = Arc::new(Mutex::new(None));
        let heur: Box<dyn Heuristic> = Box::new(PyHeuristic {
            callable: heuristic.clone_ref(py),
            error: error.clone(),
            name: "python".to_string(),
        });
        // GIL is held for the whole search: the heuristic calls back into Python
        // once per evaluated state. This is intentionally NOT allow_threads.
        let params = SearchParams {
            limits: SearchLimits {
                time: max_time.map(Duration::from_secs_f64),
                memory_bytes: max_memory,
            },
            ..SearchParams::new(heur)
        };
        let result = if greedy {
            GreedyBestFirstSearch::new(self.task.clone(), params).run()
        } else {
            AStarSearch::new(self.task.clone(), params).run()
        };
        if let Some(err) = error
            .lock()
            .expect("the error slot is never poisoned")
            .take()
        {
            return Err(err);
        }
        let result =
            result.map_err(|error| PlanforgeError::new_err(format!("search failed: {error:#}")))?;
        Ok(search_result_to_py(py, result))
    }
}

impl Task {
    fn build(task: Arc<planforge_sas::numeric_task::Task>) -> Self {
        let registry = StateRegistry::for_task(task.clone());
        let succ = StandardGroundedTree::new(&task);
        Task {
            task,
            registry: RefCell::new(registry),
            succ,
        }
    }

    /// Resolve a State to a ConcreteState in this task's registry, asserting the
    /// state actually came from this task.
    fn lookup(&self, state: &State, reg: &StateRegistry) -> PyResult<ConcreteState> {
        if state.registry_id != reg.id() {
            return Err(PyValueError::new_err("State does not belong to this Task"));
        }
        reg.lookup_state(state.state_id)
            .map_err(|e| PlanforgeError::new_err(format!("state lookup failed: {e:?}")))
    }

    fn applicable_operator_ids(
        &self,
        state: &ConcreteState,
        registry: &StateRegistry,
    ) -> Vec<usize> {
        let mut values = Vec::new();
        state.fill_propositional_values(registry, &mut values);
        let mut ids = Vec::new();
        self.succ.applicable_operators(&values, &mut ids);
        ids.iter().map(OperatorIndex::index).collect()
    }

    fn validate_operator(&self, operator: &PyOperator, task_id: usize) -> PyResult<usize> {
        if operator.task_id != Some(task_id) {
            return Err(PyValueError::new_err(
                "Operator does not belong to this Task",
            ));
        }
        let operator_id = operator.id.ok_or_else(|| {
            PyValueError::new_err("plan-result Operator cannot be applied to a Task")
        })?;
        if operator_id >= self.task.operators().len() {
            return Err(PyValueError::new_err(format!(
                "operator id {operator_id} is out of bounds for {} operators",
                self.task.operators().len()
            )));
        }
        Ok(operator_id)
    }

    fn apply_operator_snapshot(
        &self,
        state: &State,
        operator: &PyOperator,
    ) -> PyResult<(State, f64)> {
        let mut registry = self.registry.borrow_mut();
        let concrete = self.lookup(state, &registry)?;
        let operator_id = self.validate_operator(operator, registry.id())?;
        let applicable = self.applicable_operator_ids(&concrete, &registry);
        if !applicable.contains(&(operator_id)) {
            return Err(PyValueError::new_err(format!(
                "operator {} ({}) is not applicable in this state",
                operator_id, operator.name
            )));
        }
        let rust_operator = &self.task.operators()[operator_id];
        let (successor, cost) = registry
            .successor(&concrete, rust_operator)
            .map_err(|error| {
                PlanforgeError::new_err(format!(
                    "successor generation failed for {}: {error:?}",
                    rust_operator.name()
                ))
            })?;
        Ok((State::snapshot(&successor, &registry), cost))
    }
}

pub(crate) struct GilReleasedTask(pub(crate) Arc<planforge_sas::numeric_task::Task>);

// PyO3's stable `allow_threads` bound requires captured values to be `Send`,
// even though the closure is executed synchronously on the current thread with
// only the GIL released. Keep this private and use it only for that handoff.
unsafe impl Send for GilReleasedTask {}

impl GilReleasedTask {
    fn solve(
        self,
        spec: &SearchConfig,
        time_limit: Option<Duration>,
        max_memory: Option<u64>,
    ) -> Result<SearchResult, SearchError> {
        spec.solve(
            self.0,
            planforge_search::search::SearchLimits {
                time: time_limit,
                memory_bytes: max_memory,
            },
        )
    }

    fn enumerate(
        self,
        limits: EnumerationLimits,
    ) -> Result<OwnedStateSpace, RustStateSpaceEnumerationError> {
        enumerate_state_space(self.0, limits)
    }
}
