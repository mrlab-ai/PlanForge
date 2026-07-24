use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::path::Path;
use std::time::Instant;

use planforge_cplex::{Constraint, Model, ObjectiveSense, SolveStatus, Variable};
use planforge_sas::numeric_task::{AbstractNumericTask, TaskRef};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use tracing::{debug, info, warn};

use crate::evaluation::{EvaluationError, EvaluationState, Heuristic};

use super::rays::{Ray, RayGenerator};
use super::sampling::RandomWalkSampler;
use super::{
    DiverseFallback, NumericPotentialConfig, NumericPotentialFunction, NumericPotentialOptimizer,
    OptimizationOutcome, OptimizeFor, PotentialTask,
};

pub struct NumericPotentialHeuristic {
    task: PotentialTask,
    functions: RefCell<Vec<NumericPotentialFunction>>,
    conditioned_groups: Vec<Vec<NumericPotentialFunction>>,
    goal_cost_partitions: Vec<Vec<f64>>,
    goal_group_has_additive_share: Vec<bool>,
    rays: RefCell<Vec<Ray>>,
    ray_hit_counts: RefCell<Vec<usize>>,
    states_pruned_by_rays: Cell<usize>,
    ray_generator: RefCell<Option<RayGenerator>>,
    max_rays: usize,
    ray_epsilon: f64,
    cache_estimates: bool,
    invalidate_online_cache_on_growth: bool,
    dead_end_certified: bool,
    online_optimizer: RefCell<Option<NumericPotentialOptimizer>>,
    online_state: RefCell<OnlineState>,
    revision: Cell<u64>,
    prop_scratch: RefCell<Vec<usize>>,
    numeric_scratch: RefCell<Vec<f64>>,
}

struct OnlineState {
    max_functions: usize,
    base_interval: usize,
    current_interval: usize,
    max_consecutive_misses: usize,
    max_misses: usize,
    max_lp_solves: usize,
    new_states_only: bool,
    seen_states: HashSet<usize>,
    evaluations_since_solve: usize,
    consecutive_misses: usize,
    misses: usize,
    lp_solves: usize,
    functions_added: usize,
    trigger_states: usize,
    dead_end_certificates: usize,
    clean_cache_states: HashSet<usize>,
    cache_invalidations: usize,
    cache_entries_examined: usize,
    cache_entries_invalidated: usize,
}

impl NumericPotentialHeuristic {
    pub fn from_config(
        task: &dyn AbstractNumericTask,
        task_ref: TaskRef<'_>,
        config: NumericPotentialConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let mut optimizer = NumericPotentialOptimizer::new(task, &config)?;
        let potential_task = optimizer.task().clone();
        let ordinary_columns = optimizer.num_columns();
        let ordinary_rows = optimizer.num_rows();
        let mut registry = StateRegistry::for_task(task_ref.clone());
        let initial_state = registry.get_initial_state();
        let mut sampler = RandomWalkSampler::new(&*task_ref, 2011);
        let mut ray_generator = (config.rays > 0)
            .then(|| RayGenerator::new(&optimizer, config.ray_epsilon))
            .transpose()?;
        let mut rays = Vec::new();
        let mut ray_certified_initial_state = if let Some(generator) = ray_generator.as_mut() {
            if let Some(ray) = generator.try_certify(&mut optimizer, &initial_state, &registry)? {
                if generator.emit_certificate(
                    &ray,
                    &mut optimizer,
                    &initial_state,
                    &registry,
                    Path::new(&config.ray_certificate_file),
                )? {
                    rays.push(ray);
                    true
                } else {
                    false
                }
            } else {
                false
            }
        } else {
            false
        };
        let ray_certified_before_optimization = ray_certified_initial_state;

        // C++ always solves the initial-state objective first. Besides
        // seeding portfolios, this is the pre-search unboundedness check.
        let initial_outcome = if ray_certified_initial_state {
            None
        } else {
            Some(optimizer.optimize_for_state(&initial_state, &registry)?)
        };
        let (initial_function, dead_end_certified, initial_value, initial_was_optimal) =
            match initial_outcome {
                None => (optimizer.zero_function(), true, f64::INFINITY, false),
                Some(OptimizationOutcome::Optimal { value, function }) => {
                    (function, false, value.max(0.0), true)
                }
                Some(OptimizationOutcome::Unbounded { primal_ray }) => {
                    if let Some(generator) = ray_generator.as_mut() {
                        let mut candidate = if primal_ray.is_empty() {
                            None
                        } else {
                            generator.certify_native(
                                &mut optimizer,
                                primal_ray,
                                &initial_state,
                                &registry,
                            )?
                        };
                        if candidate.is_none() {
                            candidate =
                                generator.try_certify(&mut optimizer, &initial_state, &registry)?;
                        }
                        if let Some(ray) = candidate {
                            if generator.emit_certificate(
                                &ray,
                                &mut optimizer,
                                &initial_state,
                                &registry,
                                Path::new(&config.ray_certificate_file),
                            )? {
                                rays.push(ray);
                                ray_certified_initial_state = true;
                                (optimizer.zero_function(), true, f64::INFINITY, false)
                            } else {
                                warn!(
                                    "numeric potential exact ray artifact validation failed; retaining the admissible zero potential"
                                );
                                (optimizer.zero_function(), false, 0.0, false)
                            }
                        } else {
                            warn!(
                                "numeric potential unboundedness had no exact ray; retaining the admissible zero potential"
                            );
                            (optimizer.zero_function(), false, 0.0, false)
                        }
                    } else {
                        (optimizer.zero_function(), true, f64::INFINITY, false)
                    }
                }
                Some(OptimizationOutcome::Infeasible) => {
                    warn!(
                        "numeric potential LP was infeasible; retaining the certified zero potential"
                    );
                    (optimizer.zero_function(), false, 0.0, false)
                }
                Some(OptimizationOutcome::ResourceLimit(status)) => {
                    warn!(
                        "numeric potential LP stopped with {status:?}; retaining the certified zero potential"
                    );
                    (optimizer.zero_function(), false, 0.0, false)
                }
            };
        if config.rays > 0 {
            info!(
                "Numeric ray s_0 certificate: {}",
                if ray_certified_initial_state {
                    "yes"
                } else {
                    "no"
                }
            );
        }
        if config.validate_duality && initial_was_optimal {
            let (optimum, columns, rows) =
                optimizer.validate_duality(&initial_state, &registry, initial_value, 1e-6)?;
            info!(
                "Numeric potential duality validation: h_pot(s_0)={initial_value}, operator-counting optimum={optimum} ({columns} columns, {rows} rows)"
            );
        }

        let functions = if dead_end_certified {
            potential_task
                .goal_unsatisfiable
                .then_some(initial_function)
                .into_iter()
                .collect()
        } else {
            build_static_portfolio(
                &mut optimizer,
                task_ref.clone(),
                &mut registry,
                &initial_state,
                initial_function,
                initial_value,
                &config,
                &mut sampler,
            )?
        };
        let conditioned_groups = if dead_end_certified || !config.goal_conditioned {
            Vec::new()
        } else {
            build_conditioned_groups(
                &mut optimizer,
                &task_ref,
                &mut registry,
                &initial_state,
                initial_value,
                &config,
                &mut sampler,
            )?
        };
        if !dead_end_certified && config.rays > rays.len() && config.max_ray_generation_time > 0.0 {
            optimizer.prepare_for_ordinary_potential()?;
            let ray_start = Instant::now();
            let mut ray_samples = Vec::new();
            while ray_samples.len() < config.num_samples
                && ray_start.elapsed().as_secs_f64() < config.max_ray_generation_time
            {
                let chunk = (config.num_samples - ray_samples.len()).min(16);
                ray_samples.extend(sampler.sample(
                    &task_ref,
                    &mut registry,
                    chunk,
                    initial_value,
                    average_operator_cost(&potential_task),
                )?);
            }
            let generator = ray_generator
                .as_mut()
                .expect("positive ray count must construct a ray generator");
            for state in ray_samples {
                if rays.len() >= config.rays
                    || ray_start.elapsed().as_secs_f64() >= config.max_ray_generation_time
                {
                    break;
                }
                let covered = rays
                    .iter()
                    .map(|ray| {
                        ray.value(
                            &state,
                            &registry,
                            &potential_task,
                            &mut Vec::new(),
                            &mut Vec::new(),
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?
                    .into_iter()
                    .any(|value| value > config.ray_epsilon);
                if covered {
                    continue;
                }
                if let Some(candidate) = generator.try_certify(&mut optimizer, &state, &registry)?
                    && !generator.is_duplicate(&candidate, &rays, &potential_task)
                {
                    rays.push(candidate);
                }
            }
        }
        let goal_cost_partitions = if dead_end_certified
            || !config.goal_cost_partitioning
            || conditioned_groups.is_empty()
        {
            Vec::new()
        } else {
            build_goal_cost_partitions(
                &potential_task,
                &conditioned_groups,
                &task_ref,
                &mut registry,
                &initial_state,
                initial_value,
                &config,
                &mut sampler,
            )?
        };
        let mut goal_group_has_additive_share = vec![false; conditioned_groups.len()];
        for partition in &goal_cost_partitions {
            for (group_id, share) in partition.iter().enumerate() {
                goal_group_has_additive_share[group_id] |= *share > 0.0;
            }
        }
        let mut max_initial_value = functions
            .iter()
            .map(|function| {
                function_value(function, &initial_state, &potential_task, &registry)
                    .map(|value| value.max(0.0))
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .fold(0.0_f64, f64::max);
        let mut initial_group_values = Vec::with_capacity(conditioned_groups.len());
        for group in &conditioned_groups {
            let group_value = group
                .iter()
                .map(|function| {
                    function_value(function, &initial_state, &potential_task, &registry)
                })
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .fold(f64::INFINITY, f64::min)
                .max(0.0);
            initial_group_values.push(group_value);
            max_initial_value = max_initial_value.max(group_value);
        }
        for partition in &goal_cost_partitions {
            max_initial_value = max_initial_value.max(
                partition
                    .iter()
                    .zip(&initial_group_values)
                    .map(|(share, value)| share * value)
                    .sum(),
            );
        }
        let enable_online =
            config.opt == OptimizeFor::DiverseSamples && config.max_online_functions > 0;
        if enable_online {
            optimizer.prepare_for_ordinary_potential()?;
        }
        if !ray_certified_before_optimization {
            info!("Numeric potential LP: {ordinary_columns} columns, {ordinary_rows} rows");
        }
        info!(
            "Numeric bounds statistics: provider={} finite_lower={} finite_upper={} finite_both={}",
            config.bounds.as_str(),
            potential_task
                .global_feature_bounds
                .iter()
                .filter(|bounds| bounds.lower.is_finite())
                .count(),
            potential_task
                .global_feature_bounds
                .iter()
                .filter(|bounds| bounds.upper.is_finite())
                .count(),
            potential_task
                .global_feature_bounds
                .iter()
                .filter(|bounds| bounds.lower.is_finite() && bounds.upper.is_finite())
                .count()
        );
        info!(
            "Numeric bounds dropped action rows: {}",
            potential_task
                .operators
                .iter()
                .filter(|operator| !operator.reachable)
                .count()
        );
        if !ray_certified_before_optimization {
            info!(
                "Exact signed-column numeric invariants: {}",
                potential_task.global_linear_equalities.len()
            );
            for (feature_id, feature) in potential_task.features.iter().enumerate() {
                debug!(
                    "Numeric potential feature {feature_id}: `{}` coefficients={:?} goal=[{}, {}]",
                    feature.name,
                    feature.coefficients,
                    potential_task.feature_goal_bounds[feature_id].lower,
                    potential_task.feature_goal_bounds[feature_id].upper
                );
            }
            for (operator_id, operator) in potential_task.operators.iter().enumerate() {
                for &(feature_id, bounds) in &operator.numeric_precondition_bounds {
                    debug!(
                        "Numeric potential operator {operator_id} feature {feature_id} bounds=[{}, {}]",
                        bounds.lower, bounds.upper
                    );
                }
            }
            info!(
                "Numeric potential functions: {} ({} conditioned); max-ensemble h(s_0): {}",
                functions.len() + conditioned_groups.iter().map(Vec::len).sum::<usize>(),
                conditioned_groups.iter().map(Vec::len).sum::<usize>(),
                max_initial_value
            );
        }
        if config.rays > 0 {
            info!("Numeric rays generated: {}", rays.len());
        }
        let ray_count = rays.len();
        Ok(Self {
            task: potential_task,
            functions: RefCell::new(functions),
            conditioned_groups,
            goal_cost_partitions,
            goal_group_has_additive_share,
            rays: RefCell::new(rays),
            ray_hit_counts: RefCell::new(vec![0; ray_count]),
            states_pruned_by_rays: Cell::new(0),
            ray_generator: RefCell::new(if enable_online && config.rays > 0 {
                Some(ray_generator.expect("online rays requested without a ray generator"))
            } else {
                None
            }),
            max_rays: config.rays,
            ray_epsilon: config.ray_epsilon,
            cache_estimates: config.cache_estimates,
            invalidate_online_cache_on_growth: config.invalidate_online_cache_on_growth
                && config.cache_estimates,
            dead_end_certified,
            online_optimizer: RefCell::new(enable_online.then_some(optimizer)),
            online_state: RefCell::new(OnlineState {
                max_functions: if enable_online {
                    config.max_online_functions
                } else {
                    0
                },
                base_interval: config.online_reoptimization_interval,
                current_interval: config.online_reoptimization_interval,
                max_consecutive_misses: config.max_consecutive_online_misses,
                max_misses: config.max_online_misses,
                max_lp_solves: config.max_online_lp_solves,
                new_states_only: config.online_reoptimization_on_new_states_only,
                seen_states: HashSet::new(),
                evaluations_since_solve: 0,
                consecutive_misses: 0,
                misses: 0,
                lp_solves: 0,
                functions_added: 0,
                trigger_states: 0,
                dead_end_certificates: 0,
                clean_cache_states: HashSet::new(),
                cache_invalidations: 0,
                cache_entries_examined: 0,
                cache_entries_invalidated: 0,
            }),
            revision: Cell::new(0),
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
        })
    }
}

impl Heuristic for NumericPotentialHeuristic {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        if self.dead_end_certified {
            return Ok(f64::INFINITY);
        }
        let registry = eval_state.state_registry().ok_or_else(|| {
            EvaluationError::InvalidState("numeric_potential requires a state registry".to_string())
        })?;
        let mut prop_scratch = self.prop_scratch.borrow_mut();
        let mut numeric_scratch = self.numeric_scratch.borrow_mut();
        for (ray_id, ray) in self.rays.borrow().iter().enumerate() {
            let candidate = ray
                .value(
                    eval_state.state(),
                    registry,
                    &self.task,
                    &mut prop_scratch,
                    &mut numeric_scratch,
                )
                .map_err(EvaluationError::ComputationFailed)?;
            if candidate > self.ray_epsilon {
                self.ray_hit_counts.borrow_mut()[ray_id] += 1;
                self.states_pruned_by_rays
                    .set(self.states_pruned_by_rays.get() + 1);
                return Ok(f64::INFINITY);
            }
        }
        let mut value: f64 = 0.0;
        for function in self.functions.borrow().iter() {
            let candidate = function
                .value(
                    eval_state.state(),
                    registry,
                    &self.task,
                    &mut prop_scratch,
                    &mut numeric_scratch,
                )
                .map_err(EvaluationError::ComputationFailed)?;
            value = value.max(candidate);
        }
        let mut group_values = Vec::with_capacity(self.conditioned_groups.len());
        for (group_id, group) in self.conditioned_groups.iter().enumerate() {
            let mut group_value = f64::INFINITY;
            for function in group {
                let candidate = function
                    .value(
                        eval_state.state(),
                        registry,
                        &self.task,
                        &mut prop_scratch,
                        &mut numeric_scratch,
                    )
                    .map_err(EvaluationError::ComputationFailed)?;
                group_value = group_value.min(candidate);
                if !self.goal_group_has_additive_share[group_id] && group_value <= value {
                    break;
                }
            }
            let group_value = group_value.max(0.0);
            group_values.push(group_value);
            value = value.max(group_value);
        }
        for partition in &self.goal_cost_partitions {
            let additive_value = partition
                .iter()
                .zip(&group_values)
                .map(|(share, group_value)| share * group_value)
                .sum::<f64>();
            value = value.max(additive_value);
        }
        drop(prop_scratch);
        drop(numeric_scratch);
        value = self.maybe_add_online_function(eval_state, value)?;
        self.track_online_cache_entry(eval_state.state().get_id());
        Ok(value)
    }

    fn dead_ends_are_reliable(&self) -> bool {
        true
    }

    fn heuristic_name(&self) -> String {
        "numeric_potential".to_string()
    }

    fn revision(&self) -> u64 {
        self.revision.get()
    }

    fn reevaluate_on_every_pop(&self) -> bool {
        !self.cache_estimates
    }
}

impl NumericPotentialHeuristic {
    fn maybe_add_online_function(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        mut envelope_value: f64,
    ) -> Result<f64, EvaluationError> {
        if self.online_optimizer.borrow().is_none() {
            return Ok(envelope_value);
        }
        let should_solve = {
            let mut online = self.online_state.borrow_mut();
            let advance = if online.new_states_only {
                online.seen_states.insert(eval_state.state().get_id())
            } else {
                true
            };
            if advance {
                online.evaluations_since_solve += 1;
                online.trigger_states += 1;
            }
            if online.evaluations_since_solve < online.current_interval {
                false
            } else {
                online.evaluations_since_solve = 0;
                online.lp_solves += 1;
                true
            }
        };
        if !should_solve {
            return Ok(envelope_value);
        }
        let registry = eval_state.state_registry().ok_or_else(|| {
            EvaluationError::InvalidState("numeric_potential requires a state registry".to_string())
        })?;
        let outcome = self
            .online_optimizer
            .borrow_mut()
            .as_mut()
            .expect("online optimizer disappeared during evaluation")
            .optimize_for_state(eval_state.state(), registry)
            .map_err(EvaluationError::ComputationFailed)?;
        let mut useful = false;
        match outcome {
            OptimizationOutcome::Optimal { function, .. } => {
                let online_value = function
                    .value(
                        eval_state.state(),
                        registry,
                        &self.task,
                        &mut self.prop_scratch.borrow_mut(),
                        &mut self.numeric_scratch.borrow_mut(),
                    )
                    .map_err(EvaluationError::ComputationFailed)?
                    .max(0.0);
                let tolerance = 1e-6 * online_value.abs().max(1.0);
                if online_value > envelope_value + tolerance {
                    envelope_value = online_value;
                    self.functions.borrow_mut().push(function);
                    self.invalidate_online_cache();
                    useful = true;
                }
            }
            OptimizationOutcome::Unbounded { primal_ray } => {
                if self.max_rays == 0 {
                    return Ok(f64::INFINITY);
                }
                let mut optimizer = self.online_optimizer.borrow_mut();
                let optimizer = optimizer
                    .as_mut()
                    .expect("online optimizer disappeared during ray certification");
                let mut generator = self.ray_generator.borrow_mut();
                let generator = generator
                    .as_mut()
                    .expect("ray-enabled online optimizer has no exact verifier");
                let mut candidate = if primal_ray.is_empty() {
                    None
                } else {
                    generator
                        .certify_native(optimizer, primal_ray, eval_state.state(), registry)
                        .map_err(EvaluationError::ComputationFailed)?
                };
                if candidate.is_none() {
                    candidate = generator
                        .try_certify(optimizer, eval_state.state(), registry)
                        .map_err(EvaluationError::ComputationFailed)?;
                }
                if let Some(ray) = candidate {
                    let mut rays = self.rays.borrow_mut();
                    if rays.len() < self.max_rays
                        && !generator.is_duplicate(&ray, &rays, &self.task)
                    {
                        rays.push(ray);
                        self.ray_hit_counts.borrow_mut().push(0);
                        self.invalidate_online_cache();
                    }
                    self.online_state.borrow_mut().dead_end_certificates += 1;
                    return Ok(f64::INFINITY);
                }
            }
            OptimizationOutcome::Infeasible | OptimizationOutcome::ResourceLimit(_) => {}
        }

        let stop = {
            let mut online = self.online_state.borrow_mut();
            if useful {
                online.functions_added += 1;
                online.consecutive_misses = 0;
                online.current_interval = online.base_interval;
            } else {
                online.misses += 1;
                online.consecutive_misses += 1;
                if online.max_consecutive_misses > 0
                    && online.consecutive_misses >= online.max_consecutive_misses
                {
                    online.current_interval = (online.current_interval * 10).min(10_000);
                    online.consecutive_misses = 0;
                }
            }
            online.functions_added >= online.max_functions
                || (online.max_misses > 0 && online.misses >= online.max_misses)
                || (online.max_lp_solves > 0 && online.lp_solves >= online.max_lp_solves)
        };
        if stop {
            self.online_optimizer.borrow_mut().take();
        }
        Ok(envelope_value)
    }

    fn track_online_cache_entry(&self, state_id: usize) {
        if self.invalidate_online_cache_on_growth {
            self.online_state
                .borrow_mut()
                .clean_cache_states
                .insert(state_id);
        }
    }

    fn invalidate_online_cache(&self) {
        if !self.invalidate_online_cache_on_growth {
            return;
        }
        let mut online = self.online_state.borrow_mut();
        let examined = online.clean_cache_states.len();
        online.cache_invalidations += 1;
        online.cache_entries_examined += examined;
        online.cache_entries_invalidated += examined;
        online.clean_cache_states.clear();
        self.revision.set(self.revision.get() + 1);
    }
}

impl Drop for NumericPotentialHeuristic {
    fn drop(&mut self) {
        if self.max_rays > 0 {
            info!("Numeric rays retained: {}", self.rays.get_mut().len());
            info!(
                "States pruned by numeric rays: {}",
                self.states_pruned_by_rays.get()
            );
            for (ray_id, hits) in self.ray_hit_counts.get_mut().iter().enumerate() {
                info!("Numeric ray {ray_id} hits: {hits}");
            }
        }
        let online = self.online_state.get_mut();
        if online.max_functions > 0 {
            info!(
                "Online potential functions added: {}",
                online.functions_added
            );
            info!("Online potential LP solves: {}", online.lp_solves);
            info!("Online potential LP misses: {}", online.misses);
            info!(
                "Online potential cache invalidations: {}",
                online.cache_invalidations
            );
            info!(
                "Online potential cache entries examined: {}",
                online.cache_entries_examined
            );
            info!(
                "Online potential cache entries invalidated: {}",
                online.cache_entries_invalidated
            );
            info!("Online potential trigger states: {}", online.trigger_states);
            info!(
                "Online potential dead-end certificates: {}",
                online.dead_end_certificates
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn build_goal_cost_partitions(
    task: &PotentialTask,
    groups: &[Vec<NumericPotentialFunction>],
    task_ref: &TaskRef<'_>,
    registry: &mut StateRegistry<'_>,
    initial_state: &ConcreteState,
    initial_h: f64,
    config: &NumericPotentialConfig,
    sampler: &mut RandomWalkSampler,
) -> Result<Vec<Vec<f64>>, String> {
    let mut objective_states = vec![initial_state.clone()];
    if config.num_goal_cost_partitions > 1 {
        objective_states.extend(sampler.sample(
            task_ref,
            registry,
            config.num_goal_cost_partitions - 1,
            initial_h,
            average_operator_cost(task),
        )?);
    }
    let mut partitions = Vec::new();
    for state in objective_states {
        let partition = compute_goal_cost_partition(task, groups, &state, registry)?;
        if !partitions.iter().any(|existing: &Vec<f64>| {
            existing.len() == partition.len()
                && existing
                    .iter()
                    .zip(&partition)
                    .all(|(left, right)| (*left - *right).abs() <= 1e-9)
        }) {
            partitions.push(partition);
        }
    }
    Ok(partitions)
}

fn compute_goal_cost_partition(
    task: &PotentialTask,
    groups: &[Vec<NumericPotentialFunction>],
    objective_state: &ConcreteState,
    registry: &StateRegistry<'_>,
) -> Result<Vec<f64>, String> {
    let infinity = Model::infinity();
    let mut variables = Vec::with_capacity(groups.len());
    for group in groups {
        let group_value = group
            .iter()
            .map(|function| function_value(function, objective_state, task, registry))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .fold(f64::INFINITY, f64::min)
            .max(0.0);
        variables.push(Variable::new(0.0, infinity, group_value));
    }
    let mut constraints = Vec::with_capacity(task.operators.len());
    for (operator_id, operator) in task.operators.iter().enumerate() {
        let mut coefficients = Vec::new();
        for (group_id, group) in groups.iter().enumerate() {
            let (goal_var, goal_value) = group[0]
                .conditioned_goal()
                .expect("conditioned group must contain conditioned functions");
            let required = if operator.effects.contains(&(goal_var, goal_value)) {
                operator.cost
            } else {
                group
                    .iter()
                    .map(|function| -function.numeric_delta_for_operator(task, operator_id))
                    .fold(0.0_f64, f64::max)
            };
            if required > 0.0 {
                coefficients.push((group_id, required));
            }
        }
        constraints.push(Constraint::new(-infinity, operator.cost, coefficients));
    }
    let mut model =
        Model::new("numeric_potential_goal_cost_partition").map_err(|error| error.to_string())?;
    model
        .load(ObjectiveSense::Maximize, &variables, &constraints)
        .map_err(|error| error.to_string())?;
    match model.solve().map_err(|error| error.to_string())? {
        SolveStatus::Optimal => model.solution().map_err(|error| error.to_string()),
        status => {
            warn!(
                "goal-group cost partitioning stopped with {status:?}; retaining independent admissible groups"
            );
            Ok(vec![0.0; groups.len()])
        }
    }
}

fn build_conditioned_groups(
    optimizer: &mut NumericPotentialOptimizer,
    task_ref: &TaskRef<'_>,
    registry: &mut StateRegistry<'_>,
    initial_state: &ConcreteState,
    initial_h: f64,
    config: &NumericPotentialConfig,
    sampler: &mut RandomWalkSampler,
) -> Result<Vec<Vec<NumericPotentialFunction>>, String> {
    let start = Instant::now();
    let mut initial_props = Vec::new();
    initial_state.fill_state(registry, &mut initial_props);
    let mut goals: Vec<_> = optimizer
        .conditionable_goals()
        .into_iter()
        .filter(|(var, value)| initial_props[*var] != *value)
        .map(|(var, value)| {
            let achievers = optimizer.goal_achievers(var, value);
            (var, value, achievers)
        })
        .filter(|(_, _, achievers)| !achievers.is_empty())
        .collect();
    goals.sort_by_key(|(var, value, achievers)| (achievers.len(), *var, *value));

    let mut sample_sets = vec![Vec::new(); config.num_goal_conditioned_heuristics - 1];
    if !sample_sets.is_empty() {
        let samples = sampler.sample(
            task_ref,
            registry,
            config.num_goal_conditioned_samples * sample_sets.len(),
            initial_h,
            average_operator_cost(optimizer.task()),
        )?;
        let set_count = sample_sets.len();
        for (sample_id, sample) in samples.into_iter().enumerate() {
            sample_sets[sample_id % set_count].push(sample);
        }
    }

    let mut groups = Vec::new();
    for (var, value, achievers) in goals {
        if !within_fraction(start, config.max_conditioned_generation_time, 1.0) {
            break;
        }
        let mut portfolios: Vec<Vec<NumericPotentialFunction>> = (0..config
            .num_goal_conditioned_heuristics)
            .map(|_| Vec::new())
            .collect();
        let mut complete = vec![true; portfolios.len()];
        for achiever in achievers.iter().copied() {
            if !within_fraction(start, config.max_conditioned_generation_time, 1.0) {
                complete.fill(false);
                break;
            }
            match optimizer.optimize_for_conditioned_goal(
                var,
                value,
                achiever,
                initial_state,
                registry,
            )? {
                OptimizationOutcome::Optimal {
                    value: objective_value,
                    function,
                } => {
                    debug!(
                        "Goal-conditioned potential for goal {var}={value}, achiever {achiever}: h(s_0)={objective_value}"
                    );
                    portfolios[0].push(function);
                }
                _ => {
                    complete[0] = false;
                    break;
                }
            }
            for portfolio_id in 1..portfolios.len() {
                if !complete[portfolio_id] {
                    continue;
                }
                match optimizer.optimize_for_samples(&sample_sets[portfolio_id - 1], registry)? {
                    OptimizationOutcome::Optimal { function, .. } => {
                        portfolios[portfolio_id].push(function);
                    }
                    _ => complete[portfolio_id] = false,
                }
            }
        }
        for (portfolio, complete) in portfolios.into_iter().zip(complete) {
            if complete && portfolio.len() == achievers.len() {
                groups.push(portfolio);
            }
        }
    }
    Ok(groups)
}

#[allow(clippy::too_many_arguments)]
fn build_static_portfolio(
    optimizer: &mut NumericPotentialOptimizer,
    task_ref: TaskRef<'_>,
    registry: &mut StateRegistry<'_>,
    initial_state: &ConcreteState,
    initial_function: NumericPotentialFunction,
    initial_h: f64,
    config: &NumericPotentialConfig,
    sampler: &mut RandomWalkSampler,
) -> Result<Vec<NumericPotentialFunction>, String> {
    match config.opt {
        OptimizeFor::InitialState => Ok(vec![initial_function]),
        OptimizeFor::AllStates => {
            match optimizer.optimize_for_all_propositional_states(initial_state, registry)? {
                OptimizationOutcome::Optimal { function, .. } => Ok(vec![function]),
                _ => Ok(vec![initial_function]),
            }
        }
        OptimizeFor::Samples => {
            let samples = sampler.sample(
                &task_ref,
                registry,
                config.num_samples,
                initial_h,
                average_operator_cost(optimizer.task()),
            )?;
            match optimizer.optimize_for_samples(&samples, registry)? {
                OptimizationOutcome::Optimal { function, .. } => Ok(vec![function]),
                _ => Ok(vec![initial_function]),
            }
        }
        OptimizeFor::DiverseSamples => build_diverse_portfolio(
            optimizer,
            task_ref,
            registry,
            initial_state,
            initial_function,
            initial_h,
            config,
            sampler,
        ),
    }
}

struct SampleTarget {
    state: ConcreteState,
    optimum: NumericPotentialFunction,
}

#[allow(clippy::too_many_arguments)]
fn build_diverse_portfolio(
    optimizer: &mut NumericPotentialOptimizer,
    task_ref: TaskRef<'_>,
    registry: &mut StateRegistry<'_>,
    initial_state: &ConcreteState,
    initial_function: NumericPotentialFunction,
    initial_h: f64,
    config: &NumericPotentialConfig,
    sampler: &mut RandomWalkSampler,
) -> Result<Vec<NumericPotentialFunction>, String> {
    let start = Instant::now();
    let mut functions = Vec::new();
    let mut initial_function = Some(initial_function);
    if config.include_initial_state_potential {
        functions.push(initial_function.take().unwrap());
    }
    if config.include_all_states_potential
        && functions.len() < config.num_heuristics
        && let OptimizationOutcome::Optimal { function, .. } =
            optimizer.optimize_for_all_propositional_states(initial_state, registry)?
    {
        functions.push(function);
    }
    if functions.len() >= config.num_heuristics || config.max_diverse_generation_time <= 0.0 {
        if functions.is_empty() {
            functions.push(initial_function.take().unwrap());
        }
        return Ok(functions);
    }

    let mut samples = Vec::new();
    while samples.len() < config.num_samples
        && within_fraction(start, config.max_diverse_generation_time, 0.25)
    {
        let chunk = (config.num_samples - samples.len()).min(16);
        samples.extend(sampler.sample(
            &task_ref,
            registry,
            chunk,
            initial_h,
            average_operator_cost(optimizer.task()),
        )?);
    }
    if samples.is_empty() {
        if functions.is_empty() {
            functions.push(initial_function.take().unwrap());
        }
        return Ok(functions);
    }

    let mut seen = HashSet::new();
    let mut targets = Vec::new();
    for state in samples {
        if !seen.is_empty() && !within_fraction(start, config.max_diverse_generation_time, 0.75) {
            break;
        }
        if !seen.insert(state.get_id()) {
            continue;
        }
        if let OptimizationOutcome::Optimal { function, .. } =
            optimizer.optimize_for_state(&state, registry)?
        {
            targets.push(SampleTarget {
                state,
                optimum: function,
            });
        }
    }

    for function in &functions {
        remove_covered(function, &mut targets, optimizer.task(), registry)?;
    }
    while !targets.is_empty()
        && functions.len() < config.num_heuristics
        && within_fraction(start, config.max_diverse_generation_time, 1.0)
    {
        let states: Vec<_> = targets.iter().map(|target| target.state.clone()).collect();
        let OptimizationOutcome::Optimal {
            function: mut candidate,
            ..
        } = optimizer.optimize_for_samples(&states, registry)?
        else {
            break;
        };
        let covered = remove_covered(&candidate, &mut targets, optimizer.task(), registry)?;
        if covered == 0 {
            let target_id = match config.diverse_fallback {
                DiverseFallback::Random => sampler.choose_index(targets.len()),
                DiverseFallback::LargestGap => {
                    largest_gap_target(&functions, &targets, optimizer.task(), registry)?
                }
            };
            // C++ uses vector::erase here. Preserve the order of all
            // remaining targets because the same global RNG stream is reused
            // by later random fallback choices.
            candidate = targets.remove(target_id).optimum;
            remove_covered(&candidate, &mut targets, optimizer.task(), registry)?;
        }
        functions.push(candidate);
    }
    if functions.is_empty() {
        functions.push(initial_function.take().unwrap());
    }
    Ok(functions)
}

fn within_fraction(start: Instant, seconds: f64, fraction: f64) -> bool {
    !seconds.is_finite() || start.elapsed().as_secs_f64() < seconds * fraction
}

fn average_operator_cost(task: &PotentialTask) -> f64 {
    if task.operators.is_empty() {
        return 0.0;
    }
    task.operators
        .iter()
        .map(|operator| operator.cost)
        .sum::<f64>()
        / task.operators.len() as f64
}

fn function_value(
    function: &NumericPotentialFunction,
    state: &ConcreteState,
    task: &PotentialTask,
    registry: &StateRegistry<'_>,
) -> Result<f64, String> {
    function.value(state, registry, task, &mut Vec::new(), &mut Vec::new())
}

fn remove_covered(
    function: &NumericPotentialFunction,
    targets: &mut Vec<SampleTarget>,
    task: &PotentialTask,
    registry: &StateRegistry<'_>,
) -> Result<usize, String> {
    let before = targets.len();
    let mut kept = Vec::with_capacity(targets.len());
    for target in targets.drain(..) {
        let optimum = function_value(&target.optimum, &target.state, task, registry)?.max(0.0);
        let value = function_value(function, &target.state, task, registry)?.max(0.0);
        let tolerance = 1e-6 * optimum.abs().max(1.0);
        if value + tolerance < optimum {
            kept.push(target);
        }
    }
    *targets = kept;
    Ok(before - targets.len())
}

fn largest_gap_target(
    functions: &[NumericPotentialFunction],
    targets: &[SampleTarget],
    task: &PotentialTask,
    registry: &StateRegistry<'_>,
) -> Result<usize, String> {
    let mut best = 0;
    let mut largest_gap = f64::NEG_INFINITY;
    for (target_id, target) in targets.iter().enumerate() {
        let optimum = function_value(&target.optimum, &target.state, task, registry)?.max(0.0);
        let mut ensemble: f64 = 0.0;
        for function in functions {
            ensemble =
                ensemble.max(function_value(function, &target.state, task, registry)?.max(0.0));
        }
        if optimum - ensemble > largest_gap {
            largest_gap = optimum - ensemble;
            best = target_id;
        }
    }
    Ok(best)
}
