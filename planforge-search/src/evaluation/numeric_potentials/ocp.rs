use std::cell::RefCell;

use planforge_cplex::{Constraint, Model, ObjectiveSense, SolveStatus, Variable};
use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_sas::state_registry::StateRegistry;
use tracing::{info, warn};

use crate::evaluation::domain_abstractions::domain_abstraction_factory::OcpTransitionSystemBuild;
use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstraction;
use crate::evaluation::domain_abstractions::domain_abstraction_heuristic::DomainAbstractionHeuristic;
use crate::evaluation::{EvaluationError, EvaluationState, Heuristic};

use super::{
    NumericPotentialConfig, NumericPotentialFunction, NumericPotentialOptimizer,
    OptimizationOutcome, PotentialTask,
};

/// One numeric potential and one domain abstraction optimized jointly over
/// per-operator cost shares. The LP is solved once; state evaluation is one
/// abstract lookup and one potential dot product.
pub struct PotentialAbstractionOcpHeuristic {
    lookup: DomainAbstractionHeuristic,
    potential_task: PotentialTask,
    potential: NumericPotentialFunction,
    abstract_potentials: Option<Vec<f64>>,
    dead_end_certified: bool,
    prop_scratch: RefCell<Vec<usize>>,
    numeric_scratch: RefCell<Vec<f64>>,
}

impl PotentialAbstractionOcpHeuristic {
    pub fn new(
        task: &dyn AbstractNumericTask,
        task_ref: TaskRef<'_>,
        abstraction: DomainAbstraction,
        potential_config: NumericPotentialConfig,
        nonnegative: bool,
        max_recorded_transitions: usize,
    ) -> Result<Self, String> {
        let transition_build = abstraction
            .factory
            .build_ocp_transition_system_from_operators_with_deadline(
                task,
                abstraction.combine_labels,
                &abstraction.abstract_operators,
                None,
                max_recorded_transitions,
            )
            .map_err(|error| format!("pot_da_ocp could not record transitions: {error:#}"))?;
        let (transition_system, recorded_transition_count) = match transition_build {
            OcpTransitionSystemBuild::Complete(system) => {
                let count = system
                    .transitions
                    .iter()
                    .map(|transition| transition.concrete_op_ids.len())
                    .sum::<usize>();
                (Some(system), count)
            }
            OcpTransitionSystemBuild::ConcreteLabelCapExceeded { required_at_least } => {
                (None, required_at_least)
            }
        };
        let initial_abstract_state = abstraction.distance_table.initial_state_hash;
        let abstraction_initial_h = abstraction.distance_table.distances[initial_abstract_state];
        let lookup =
            DomainAbstractionHeuristic::new(Some("pot_da_ocp_abstraction".into()), abstraction);

        let mut optimizer = NumericPotentialOptimizer::new(task, &potential_config)?;
        let potential_task = optimizer.task().clone();
        let mut registry = StateRegistry::for_task(task_ref);
        let initial = registry.get_initial_state();
        let (plain_initial_h, plain_potential, dead_end_certified) = match optimizer
            .optimize_for_state(&initial, &registry)?
        {
            OptimizationOutcome::Optimal { value, function } => (value, function, false),
            OptimizationOutcome::Unbounded { .. } => {
                (f64::INFINITY, optimizer.zero_function(), true)
            }
            OptimizationOutcome::Infeasible | OptimizationOutcome::ResourceLimit(_) => {
                warn!(
                    "pot_da_ocp plain potential did not solve optimally; retaining its admissible zero fallback"
                );
                (0.0, optimizer.zero_function(), false)
            }
        };
        if dead_end_certified {
            return Ok(Self {
                lookup,
                potential_task,
                potential: plain_potential,
                abstract_potentials: None,
                dead_end_certified: true,
                prop_scratch: RefCell::new(Vec::new()),
                numeric_scratch: RefCell::new(Vec::new()),
            });
        }
        let Some(transition_system) = transition_system else {
            warn!(
                "pot_da_ocp transition enumeration exceeded its cap of {max_recorded_transitions} concrete-label transitions (at least {recorded_transition_count} required); continuing with the independently certified potential"
            );
            return Ok(Self {
                lookup,
                potential_task,
                potential: plain_potential,
                abstract_potentials: None,
                dead_end_certified: false,
                prop_scratch: RefCell::new(Vec::new()),
                numeric_scratch: RefCell::new(Vec::new()),
            });
        };
        if abstraction_initial_h.is_infinite() {
            return Ok(Self {
                lookup,
                potential_task,
                potential: plain_potential,
                abstract_potentials: None,
                dead_end_certified: true,
                prop_scratch: RefCell::new(Vec::new()),
                numeric_scratch: RefCell::new(Vec::new()),
            });
        }

        let infinity = Model::infinity();
        let operator_count = potential_task.operators.len();
        let state_count = lookup.abstraction().distance_table.distances.len();
        let mut variables = Vec::new();
        for operator in &potential_task.operators {
            variables.push(if nonnegative {
                Variable::new(0.0, operator.cost, 0.0)
            } else {
                Variable::new(-infinity, infinity, 0.0)
            });
        }
        let mut is_goal = vec![false; state_count];
        for &goal in &transition_system.goal_state_hashes {
            if goal < state_count {
                is_goal[goal] = true;
            }
        }
        for goal in is_goal {
            variables.push(Variable::new(
                -infinity,
                if goal { 0.0 } else { infinity },
                0.0,
            ));
        }
        let mut constraints = Vec::new();
        for transition in &transition_system.transitions {
            for &operator_id in &transition.concrete_op_ids {
                if operator_id >= operator_count
                    || transition.source_hash >= state_count
                    || transition.target_hash >= state_count
                {
                    return Err(format!(
                        "pot_da_ocp recorded invalid transition {}: {} --{}--> {}",
                        transition.transition_id,
                        transition.source_hash,
                        operator_id,
                        transition.target_hash
                    ));
                }
                let mut coefficients = vec![(operator_id, -1.0)];
                if transition.source_hash != transition.target_hash {
                    coefficients.push((operator_count + transition.source_hash, 1.0));
                    coefficients.push((operator_count + transition.target_hash, -1.0));
                }
                coefficients.sort_unstable_by_key(|(column, _)| *column);
                constraints.push(Constraint::new(-infinity, 0.0, coefficients));
            }
        }

        let system = optimizer.ordinary_system().clone();
        let potential_base = variables.len();
        variables.extend(system.variables.iter().copied());
        let mut operator_by_row = vec![None; system.constraints.len()];
        for (operator_id, row) in system.operator_rows.iter().enumerate() {
            if let Some(row) = row {
                operator_by_row[*row] = Some(operator_id);
            }
        }
        for (row_id, row) in system.constraints.iter().enumerate() {
            let mut coefficients = row
                .coefficients
                .iter()
                .map(|(column, coefficient)| (potential_base + *column, *coefficient))
                .collect::<Vec<_>>();
            if let Some(operator_id) = operator_by_row[row_id] {
                coefficients.push((operator_id, 1.0));
            }
            coefficients.sort_unstable_by_key(|(column, _)| *column);
            constraints.push(Constraint::new(row.lower, row.upper, coefficients));
        }
        let mut objective = vec![0.0; variables.len()];
        objective[operator_count + initial_abstract_state] = 1.0;
        let potential_objective = optimizer.objective_for_state(&initial, &registry, &system)?;
        for (column, coefficient) in potential_objective.into_iter().enumerate() {
            objective[potential_base + column] = coefficient;
        }
        let mut model = Model::new("pot_da_ocp").map_err(|error| error.to_string())?;
        model
            .load(ObjectiveSense::Maximize, &variables, &constraints)
            .map_err(|error| error.to_string())?;
        model
            .set_objective(&objective)
            .map_err(|error| error.to_string())?;
        if potential_config.dump_lp {
            model
                .write(std::path::Path::new("pot_da_ocp.lp"))
                .map_err(|error| error.to_string())?;
        }
        info!(
            "pot_da_ocp joint LP: {state_count} states, {} transitions, {} columns, {} rows",
            recorded_transition_count,
            variables.len(),
            constraints.len()
        );
        let status = model.solve().map_err(|error| error.to_string())?;
        info!("pot_da_ocp joint LP status: {status:?}");
        let (potential, abstract_potentials, dead_end_certified) = match status {
            SolveStatus::Optimal => {
                let solution = model.solution().map_err(|error| error.to_string())?;
                let potential = optimizer.function_from_system_solution(
                    &system,
                    &solution[potential_base..potential_base + system.variables.len()],
                );
                let abstract_potentials =
                    solution[operator_count..operator_count + state_count].to_vec();
                let joint = model.objective_value().map_err(|error| error.to_string())?;
                if joint + 1e-6 < plain_initial_h.max(abstraction_initial_h) {
                    return Err(format!(
                        "pot_da_ocp joint optimum {joint} does not dominate component values \
                         {plain_initial_h} and {abstraction_initial_h}"
                    ));
                }
                info!(
                    "pot_da_ocp: {state_count} states, {} transitions, {} columns, {} rows, h(s_0)={joint}, h_da(s_0)={abstraction_initial_h}, h_pot(s_0)={plain_initial_h}",
                    recorded_transition_count,
                    variables.len(),
                    constraints.len()
                );
                (potential, Some(abstract_potentials), false)
            }
            SolveStatus::Unbounded => (plain_potential, None, true),
            other => {
                warn!(
                    "pot_da_ocp joint LP stopped with {other:?}; continuing with the independently certified potential"
                );
                (plain_potential, None, false)
            }
        };
        Ok(Self {
            lookup,
            potential_task,
            potential,
            abstract_potentials,
            dead_end_certified,
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
        })
    }
}

impl Heuristic for PotentialAbstractionOcpHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        if self.dead_end_certified {
            return Ok(f64::INFINITY);
        }
        let registry = eval_state.state_registry();
        let potential = self
            .potential
            .value(
                eval_state.state(),
                registry,
                &self.potential_task,
                &mut self.prop_scratch.borrow_mut(),
                &mut self.numeric_scratch.borrow_mut(),
            )
            .map_err(EvaluationError::ComputationFailed)?;
        let Some(abstract_potentials) = &self.abstract_potentials else {
            return Ok(potential.max(0.0));
        };
        let hash = self.lookup.abstract_state_hash(eval_state)?;
        let abstract_value = abstract_potentials.get(hash).copied().ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "pot_da_ocp abstract state {hash} is outside the joint LP"
            ))
        })?;
        Ok((abstract_value + potential).max(0.0))
    }

    fn dead_ends_are_reliable(&self) -> bool {
        true
    }

    fn heuristic_name(&self) -> &str {
        "pot_da_ocp"
    }
}
