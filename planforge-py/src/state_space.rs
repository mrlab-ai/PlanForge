//! The explicit state space of a task as a Python object: its states,
//! transitions and goal distances, with random walks and sampling over it.

use super::*;

impl PyStateSpace {
    /// The numpy views, built on first access.
    fn arrays(&self, py: Python<'_>) -> PyResult<&StateSpaceArrays> {
        if let Some(arrays) = self.array_views.get() {
            return Ok(arrays);
        }
        let numpy = py.import("numpy").map_err(|error| {
            PyErr::new::<pyo3::exceptions::PyImportError, _>(format!(
                "state-space arrays require numpy (declared by planforge-py's package metadata): {error}"
            ))
        })?;
        let state_count = self.graph.num_states();
        let prop_values: Vec<usize> = self
            .graph
            .propositional_values
            .iter()
            .map(ExplicitValueIndex::index)
            .collect();
        let num_values: Vec<f64> = self
            .graph
            .numeric_values
            .iter()
            .map(NumericValue::value)
            .collect();
        let ops: Vec<usize> = self
            .graph
            .transition_operator_ids
            .iter()
            .map(OperatorIndex::index)
            .collect();
        let built = StateSpaceArrays {
            propositional_values: numpy
                .call_method1("asarray", (prop_values,))?
                .call_method1(
                    "reshape",
                    ((state_count, self.graph.num_propositional_variables),),
                )?
                .unbind(),
            numeric_values: numpy
                .call_method1("asarray", (num_values,))?
                .call_method1(
                    "reshape",
                    ((state_count, self.graph.num_numeric_variables),),
                )?
                .unbind(),
            transition_offsets: numpy
                .call_method1("asarray", (&self.graph.transition_offsets,))?
                .unbind(),
            transition_operator_ids: numpy.call_method1("asarray", (ops,))?.unbind(),
            transition_successor_ids: numpy
                .call_method1("asarray", (&self.graph.transition_successor_ids,))?
                .unbind(),
            transition_costs: numpy
                .call_method1("asarray", (&self.graph.transition_costs,))?
                .unbind(),
            goal_states: numpy
                .call_method1("asarray", (&self.graph.goal_states,))?
                .unbind(),
            h_star: numpy
                .call_method1("asarray", (&self.graph.h_star,))?
                .unbind(),
        };
        // Another thread may have won the race; either value is equivalent.
        let _ = self.array_views.set(built);
        Ok(self
            .array_views
            .get()
            .expect("the array views were just initialised"))
    }

    /// The source state of a transition, found from the CSR offsets.
    fn source_of(&self, transition_id: usize) -> usize {
        self.graph
            .transition_offsets
            .partition_point(|&offset| offset as usize <= transition_id)
            - 1
    }

    fn edge(&self, py: Python<'_>, source: usize, transition_id: usize) -> PyResult<Py<PyEdge>> {
        let operator_id = self.graph.transition_operator_ids[transition_id].index();
        let target = self.graph.transition_successor_ids[transition_id] as usize;
        Py::new(
            py,
            PyEdge {
                source,
                target,
                operator_id,
                operator_name: self
                    .operator_names
                    .get(operator_id)
                    .cloned()
                    .unwrap_or_default(),
                cost: self.graph.transition_costs[transition_id],
                source_label: self.get_state_label(source)?,
                target_label: self.get_state_label(target)?,
            },
        )
    }

    /// A uniform index below `bound`, which must be positive.
    fn next_index(&self, bound: usize) -> usize {
        let mut seed = self
            .rng
            .lock()
            .expect("the sampling seed mutex is never held across a panic");
        (next_random(&mut seed) % bound as u64) as usize
    }

    fn group_states_by_goal_distance(&self) -> BTreeMap<NotNanKey, Vec<u32>> {
        let mut grouped = BTreeMap::<NotNanKey, Vec<u32>>::new();
        for (state, &distance) in self.graph.h_star.iter().enumerate() {
            if !distance.is_finite() {
                continue;
            }
            grouped
                .entry(distance.to_bits())
                .or_default()
                .push(u32::try_from(state).expect("state count fits u32"));
        }
        grouped
    }
}

/// Seed the state-space sampler starts from, so repeated runs of a script draw
/// the same states. Call `set_seed` for a different sequence.
pub(crate) const DEFAULT_SAMPLING_SEED: u64 = 0;

/// One transition of an enumerated state space, with both endpoints labelled.
#[pyclass(name = "Edge", frozen, get_all)]
pub(crate) struct PyEdge {
    source: usize,
    target: usize,
    operator_id: usize,
    operator_name: String,
    cost: f64,
    source_label: String,
    target_label: String,
}

#[pymethods]
impl PyEdge {
    fn __repr__(&self) -> String {
        format!(
            "Edge({} -> {}, {:?}, cost={})",
            self.source, self.target, self.operator_name, self.cost
        )
    }
}

/// The numpy views of an enumeration, built together on first access.
pub(crate) struct StateSpaceArrays {
    propositional_values: Py<PyAny>,
    numeric_values: Py<PyAny>,
    transition_offsets: Py<PyAny>,
    transition_operator_ids: Py<PyAny>,
    transition_successor_ids: Py<PyAny>,
    transition_costs: Py<PyAny>,
    goal_states: Py<PyAny>,
    h_star: Py<PyAny>,
}

/// Predecessors of every state, in the same compressed layout the forward
/// transitions use.
///
/// Enumeration only records outgoing transitions, so answering "which states
/// reach this one" means inverting the graph. That costs one pass over the
/// transitions, so it is done on first use rather than for every enumeration.
pub(crate) struct ReverseTransitions {
    offsets: Vec<u64>,
    transition_ids: Vec<u32>,
}

impl ReverseTransitions {
    fn build(graph: &OwnedStateSpace) -> Self {
        let num_states = graph.num_states();
        let mut counts = vec![0u64; num_states + 1];
        for &target in &graph.transition_successor_ids {
            counts[target as usize + 1] += 1;
        }
        for index in 1..counts.len() {
            counts[index] += counts[index - 1];
        }
        let offsets = counts;
        let mut cursor = offsets.clone();
        let mut transition_ids = vec![0u32; graph.num_transitions()];
        for (transition_id, &target) in graph.transition_successor_ids.iter().enumerate() {
            let slot = &mut cursor[target as usize];
            transition_ids[*slot as usize] = u32::try_from(transition_id)
                .expect("transition count fits u32 because successor ids do");
            *slot += 1;
        }
        Self {
            offsets,
            transition_ids,
        }
    }

    fn incoming(&self, state: usize) -> &[u32] {
        let start = self.offsets[state] as usize;
        let end = self.offsets[state + 1] as usize;
        &self.transition_ids[start..end]
    }
}

#[pyclass(name = "StateSpace", frozen)]
pub(crate) struct PyStateSpace {
    state_count: usize,
    transition_count: usize,
    goal_state_count: usize,
    dead_end_count: usize,
    diameter: Option<f64>,
    h_star_histogram: Vec<(f64, usize)>,
    /// The array views, built on first access.
    ///
    /// The enumeration is retained anyway so that single states can be queried,
    /// and a numpy view is a second copy of the same numbers: on an eight-block
    /// space the propositional values alone are around 139 MB. Building them
    /// only when a caller reads them means a script that queries states one at a
    /// time never pays for the copy.
    array_views: OnceLock<StateSpaceArrays>,
    /// The enumeration itself, kept so single states can be queried without
    /// going back through the array views.
    graph: Arc<OwnedStateSpace>,
    /// Atom names indexed by variable then value, for labelling states.
    fact_names: Arc<Vec<Vec<String>>>,
    operator_names: Arc<Vec<String>>,
    reverse: OnceLock<ReverseTransitions>,
    /// Sampling state. Seeded deterministically so two runs of the same script
    /// draw the same states; `set_seed` re-seeds it.
    rng: Mutex<u64>,
    /// States grouped by their exact goal distance, built on first sample.
    states_by_goal_distance: OnceLock<BTreeMap<NotNanKey, Vec<u32>>>,
}

/// A goal distance used as a map key. `h*` values are finite here because dead
/// ends are excluded before grouping.
pub(crate) type NotNanKey = u64;

/// SplitMix64: a seedable generator with no dependency and no shared state.
pub(crate) fn next_random(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[pymethods]
impl PyStateSpace {
    /// A state's index, checked against the enumeration.
    fn checked_state(&self, state: usize) -> PyResult<usize> {
        if state >= self.graph.num_states() {
            return Err(PyIndexError::new_err(format!(
                "state {state} out of range for {} states",
                self.graph.num_states()
            )));
        }
        Ok(state)
    }

    /// Whether no goal is reachable from `state`.
    fn is_dead_end_state(&self, state: usize) -> PyResult<bool> {
        let state = self.checked_state(state)?;
        Ok(!self.graph.h_star[state].is_finite())
    }

    /// Whether `state` satisfies the goal.
    fn is_goal_state(&self, state: usize) -> PyResult<bool> {
        let state = self.checked_state(state)?;
        Ok(self.graph.goal_states[state])
    }

    /// Whether `state` is the state the enumeration started from. Enumeration is
    /// a forward search from the initial state, so that is index 0.
    fn is_initial_state(&self, state: usize) -> PyResult<bool> {
        Ok(self.checked_state(state)? == 0)
    }

    /// Cheapest cost of reaching a goal from `state`, or `None` for a dead end.
    ///
    /// This is a cost, not a step count: the enumeration computes `h*` over the
    /// task's operator costs, so it equals the number of steps only when every
    /// operator costs one.
    fn get_cost_to_goal(&self, state: usize) -> PyResult<Option<f64>> {
        let state = self.checked_state(state)?;
        let distance = self.graph.h_star[state];
        Ok(distance.is_finite().then_some(distance))
    }

    /// The largest finite goal distance in the space, or `None` if no goal is
    /// reachable.
    fn get_max_cost_to_goal(&self) -> Option<f64> {
        self.diameter
    }

    fn get_num_states(&self) -> usize {
        self.graph.num_states()
    }

    fn get_num_dead_end_states(&self) -> usize {
        self.dead_end_count
    }

    /// States from which a goal is reachable.
    fn get_num_alive_states(&self) -> usize {
        self.graph.num_states() - self.dead_end_count
    }

    /// Every state index, so the space can be iterated.
    fn get_states(&self) -> Vec<usize> {
        (0..self.graph.num_states()).collect()
    }

    fn __len__(&self) -> usize {
        self.graph.num_states()
    }

    /// The values `state` assigns, as `(variable, value)` pairs.
    fn get_assignment(&self, state: usize) -> PyResult<Vec<PyFact>> {
        let state = self.checked_state(state)?;
        let width = self.graph.num_propositional_variables;
        let row = &self.graph.propositional_values[state * width..(state + 1) * width];
        Ok(row
            .iter()
            .enumerate()
            .map(|(variable, &value)| (variable, value.index()))
            .collect())
    }

    /// The numeric values `state` assigns, indexed by numeric variable.
    fn get_numeric_variables(&self, state: usize) -> PyResult<Vec<f64>> {
        let state = self.checked_state(state)?;
        let width = self.graph.num_numeric_variables;
        Ok(
            self.graph.numeric_values[state * width..(state + 1) * width]
                .iter()
                .map(NumericValue::value)
                .collect(),
        )
    }

    /// The atoms true in `state`, by name.
    fn get_atoms(&self, state: usize) -> PyResult<Vec<String>> {
        let state = self.checked_state(state)?;
        let width = self.graph.num_propositional_variables;
        let row = &self.graph.propositional_values[state * width..(state + 1) * width];
        Ok(row
            .iter()
            .enumerate()
            .filter_map(|(variable, &value)| {
                let name = self.fact_names.get(variable)?.get(value.index())?;
                // A variable's negated and sentinel values are not atoms that
                // hold; only the positive ones label the state.
                (!name.is_empty() && !name.starts_with("NegatedAtom") && !name.starts_with('<'))
                    .then(|| name.clone())
            })
            .collect())
    }

    /// A readable label for `state`: its index and the atoms that hold.
    fn get_state_label(&self, state: usize) -> PyResult<String> {
        let atoms = self.get_atoms(state)?;
        Ok(format!("s{state}{{{}}}", atoms.join(", ")))
    }

    /// Whether every literal holds in `state`.
    ///
    /// A literal is `(variable, value)` for "this variable takes this value", or
    /// `(variable, value, False)` for its negation.
    fn literal_holds(&self, state: usize, literals: Vec<Bound<'_, PyAny>>) -> PyResult<bool> {
        let state = self.checked_state(state)?;
        let width = self.graph.num_propositional_variables;
        let row = &self.graph.propositional_values[state * width..(state + 1) * width];
        for literal in literals {
            let (variable, value, expected): (usize, usize, bool) = match literal.len()? {
                2 => {
                    let (variable, value): (usize, usize) = literal.extract()?;
                    (variable, value, true)
                }
                3 => literal.extract()?,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "a literal is (variable, value) or (variable, value, holds), got {other} items"
                    )));
                }
            };
            let actual = row
                .get(variable)
                .ok_or_else(|| {
                    PyIndexError::new_err(format!(
                        "literal names variable {variable} but the task has {width}"
                    ))
                })?
                .index();
            if (actual == value) != expected {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Transitions leaving `state`, as `(operator_id, operator_name, successor, cost)`.
    fn get_forward_transitions(&self, py: Python<'_>, state: usize) -> PyResult<Vec<Py<PyEdge>>> {
        let state = self.checked_state(state)?;
        let start = self.graph.transition_offsets[state] as usize;
        let end = self.graph.transition_offsets[state + 1] as usize;
        (start..end).map(|id| self.edge(py, state, id)).collect()
    }

    /// Transitions entering `state`.
    ///
    /// Enumeration records only outgoing transitions, so the first call inverts
    /// the graph and later calls reuse that.
    fn get_backward_transitions(&self, py: Python<'_>, state: usize) -> PyResult<Vec<Py<PyEdge>>> {
        let state = self.checked_state(state)?;
        let reverse = self
            .reverse
            .get_or_init(|| ReverseTransitions::build(&self.graph));
        reverse
            .incoming(state)
            .iter()
            .map(|&transition_id| {
                let source = self.source_of(transition_id as usize);
                self.edge(py, source, transition_id as usize)
            })
            .collect()
    }

    /// Re-seed the sampler.
    fn set_seed(&self, seed: u64) {
        *self
            .rng
            .lock()
            .expect("the sampling seed mutex is never held across a panic") = seed;
    }

    /// A uniformly drawn state.
    fn sample_state(&self) -> PyResult<usize> {
        let count = self.graph.num_states();
        if count == 0 {
            return Err(PlanforgeError::new_err("the state space has no states"));
        }
        Ok(self.next_index(count))
    }

    /// A uniformly drawn state whose goal distance is exactly `cost`.
    ///
    /// The distance is a cost, matching [`Self::get_cost_to_goal`].
    fn sample_state_at_cost_to_goal(&self, cost: f64) -> PyResult<usize> {
        let grouped = self
            .states_by_goal_distance
            .get_or_init(|| self.group_states_by_goal_distance());
        let states = grouped
            .get(&cost.to_bits())
            .ok_or_else(|| PlanforgeError::new_err(format!("no state has goal distance {cost}")))?;
        Ok(states[self.next_index(states.len())] as usize)
    }

    /// Every state whose goal distance is exactly `cost`.
    fn get_states_at_cost_to_goal(&self, cost: f64) -> Vec<usize> {
        let grouped = self
            .states_by_goal_distance
            .get_or_init(|| self.group_states_by_goal_distance());
        grouped
            .get(&cost.to_bits())
            .map(|states| states.iter().map(|&state| state as usize).collect())
            .unwrap_or_default()
    }

    /// A uniformly drawn state from which no goal is reachable.
    fn sample_dead_end_state(&self) -> PyResult<usize> {
        let dead_ends = self
            .graph
            .h_star
            .iter()
            .enumerate()
            .filter(|(_, distance)| !distance.is_finite())
            .map(|(state, _)| state)
            .collect::<Vec<_>>();
        if dead_ends.is_empty() {
            return Err(PlanforgeError::new_err(
                "the state space has no dead-end states",
            ));
        }
        Ok(dead_ends[self.next_index(dead_ends.len())])
    }

    #[getter]
    fn state_count(&self) -> usize {
        self.state_count
    }

    #[getter]
    fn transition_count(&self) -> usize {
        self.transition_count
    }

    #[getter]
    fn goal_state_count(&self) -> usize {
        self.goal_state_count
    }

    #[getter]
    fn dead_end_count(&self) -> usize {
        self.dead_end_count
    }

    #[getter]
    fn diameter(&self) -> Option<f64> {
        self.diameter
    }

    #[getter]
    fn h_star_histogram(&self) -> Vec<(f64, usize)> {
        self.h_star_histogram.clone()
    }

    #[getter]
    fn propositional_values(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays(py)?.propositional_values.clone_ref(py))
    }

    #[getter]
    fn numeric_values(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays(py)?.numeric_values.clone_ref(py))
    }

    #[getter]
    fn transition_offsets(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays(py)?.transition_offsets.clone_ref(py))
    }

    #[getter]
    fn transition_operator_ids(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays(py)?.transition_operator_ids.clone_ref(py))
    }

    #[getter]
    fn transition_successor_ids(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays(py)?.transition_successor_ids.clone_ref(py))
    }

    #[getter]
    fn transition_costs(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays(py)?.transition_costs.clone_ref(py))
    }

    #[getter]
    fn goal_states(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays(py)?.goal_states.clone_ref(py))
    }

    #[getter]
    fn h_star(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(self.arrays(py)?.h_star.clone_ref(py))
    }

    fn __repr__(&self) -> String {
        format!(
            "StateSpace(states={}, transitions={}, goals={}, dead_ends={})",
            self.state_count, self.transition_count, self.goal_state_count, self.dead_end_count
        )
    }
}

pub(crate) fn state_space_to_py(
    graph: OwnedStateSpace,
    task: &planforge_sas::numeric_task::Task,
) -> PyResult<PyStateSpace> {
    // Names are copied out now: the enumeration outlives the borrow of the task.
    let fact_names = task
        .variables()
        .iter()
        .enumerate()
        .map(|(variable, explicit_variable)| {
            (0..explicit_variable.domain_size())
                .map(|value| {
                    task.fact_name(&ExplicitFact::propositional(variable, value))
                        .to_string()
                })
                .collect()
        })
        .collect::<Vec<Vec<String>>>();
    let operator_names = task
        .operators()
        .iter()
        .map(|operator| operator.name().to_string())
        .collect::<Vec<String>>();
    let summary = graph.summary();

    Ok(PyStateSpace {
        state_count: summary.state_count,
        transition_count: summary.transition_count,
        goal_state_count: summary.goal_state_count,
        dead_end_count: summary.dead_end_count,
        diameter: summary.diameter,
        h_star_histogram: summary.h_star_histogram,
        array_views: OnceLock::new(),
        graph: Arc::new(graph),
        fact_names: Arc::new(fact_names),
        operator_names: Arc::new(operator_names),
        reverse: OnceLock::new(),
        rng: Mutex::new(DEFAULT_SAMPLING_SEED),
        states_by_goal_distance: OnceLock::new(),
    })
}
