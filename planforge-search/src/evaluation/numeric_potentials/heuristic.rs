//! Numeric potential heuristics, behind the `cplex` feature.
//!
//! Linking needs a proprietary CPLEX installation; without one,
//! `DOCS_RS=1 cargo clippy -p planforge-search --features cplex` type-checks
//! and lints this module.

use std::collections::HashSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use planforge_cplex::{Constraint, Model, ObjectiveSense, SolveStatus, Variable};
use planforge_sas::numeric_task::{ExplicitValueIndex, NumericValue, Task};
use planforge_sas::state_registry::{ConcreteState, StateRegistry};
use planforge_sas::utils::float_tolerance;
use tracing::{debug, info, warn};

use crate::evaluation::{EvaluationError, EvaluationState, Heuristic};

use super::rays::{Ray, RayGenerator};
use super::sampling::{RandomWalkSampler, SampleSpace};
use super::{
    DiverseFallback, NumericPotentialConfig, NumericPotentialFunction, NumericPotentialOptimizer,
    OptimizationOutcome, OptimizeFor, PotentialTask,
};

pub struct NumericPotentialHeuristic {
    task: PotentialTask,
    functions: Vec<NumericPotentialFunction>,
    conditioned_groups: Vec<Vec<NumericPotentialFunction>>,
    goal_cost_partitions: Vec<Vec<f64>>,
    goal_group_has_additive_share: Vec<bool>,
    rays: Vec<Ray>,
    ray_hit_counts: Vec<usize>,
    states_pruned_by_rays: usize,
    ray_generator: Option<RayGenerator>,
    max_rays: usize,
    ray_epsilon: f64,
    cache_estimates: bool,
    invalidate_online_cache_on_growth: bool,
    dead_end_certified: bool,
    online_optimizer: Option<NumericPotentialOptimizer>,
    online_state: OnlineState,
    revision: u64,
    prop_scratch: Vec<ExplicitValueIndex>,
    numeric_scratch: Vec<NumericValue>,
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
        task_ref: Arc<Task>,
        config: NumericPotentialConfig,
    ) -> Result<Self, String> {
        config.validate()?;
        let task = &*task_ref;
        let mut optimizer = NumericPotentialOptimizer::new(task, &config)?;
        let potential_task = optimizer.task().clone();
        let ordinary_columns = optimizer.num_columns();
        let ordinary_rows = optimizer.num_rows();
        let mut registry = StateRegistry::for_task(task_ref.clone());
        let initial_state = registry.get_initial_state();
        // The exact verifier of unboundedness. It is built when the first LP
        // turns out unbounded, unless rays are requested from the start: a
        // floating-point unboundedness verdict alone never declares a dead
        // end.
        let mut ray_generator = (config.rays > 0)
            .then(|| RayGenerator::new(&optimizer, config.ray_epsilon))
            .transpose()?;
        let mut rays = Vec::new();
        let mut ray_certified_initial_state = if config.rays > 0 {
            let generator = ray_generator
                .as_mut()
                .expect("a positive ray count builds the verifier up front");
            if let Some(ray) =
                generator.try_certify(&mut optimizer, registry.view(&initial_state))?
            {
                if generator.emit_certificate(
                    &ray,
                    &mut optimizer,
                    registry.view(&initial_state),
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
            Some(optimizer.optimize_for_state(registry.view(&initial_state))?)
        };
        let (initial_function, dead_end_certified, initial_value, initial_was_optimal) =
            match initial_outcome {
                None => (optimizer.zero_function(), true, f64::INFINITY, false),
                Some(OptimizationOutcome::Optimal { value, function }) => {
                    (function, false, value.max(0.0), true)
                }
                Some(OptimizationOutcome::Unbounded { primal_ray }) => {
                    let generator =
                        exact_verifier(&mut ray_generator, &optimizer, config.ray_epsilon)?;
                    let mut candidate = if primal_ray.is_empty() {
                        None
                    } else {
                        generator.certify_native(
                            &mut optimizer,
                            primal_ray,
                            registry.view(&initial_state),
                        )?
                    };
                    if candidate.is_none() {
                        candidate =
                            generator.try_certify(&mut optimizer, registry.view(&initial_state))?;
                    }
                    if let Some(ray) = candidate {
                        if generator.emit_certificate(
                            &ray,
                            &mut optimizer,
                            registry.view(&initial_state),
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
        if ray_generator.is_some() {
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
            let (optimum, columns, rows) = optimizer.validate_duality(
                registry.view(&initial_state),
                initial_value,
                float_tolerance::LP_EPSILON,
            )?;
            info!(
                "Numeric potential duality validation: h_pot(s_0)={initial_value}, operator-counting optimum={optimum} ({columns} columns, {rows} rows)"
            );
        }

        let mut space = SampleSpace {
            sampler: RandomWalkSampler::new(
                task_ref.clone(),
                initial_value,
                average_operator_cost(&potential_task),
            ),
            registry,
            initial_state,
        };
        let functions = if dead_end_certified {
            potential_task
                .goal_unsatisfiable
                .then_some(initial_function)
                .into_iter()
                .collect()
        } else {
            build_static_portfolio(&mut optimizer, &mut space, &config, initial_function)?
        };
        let conditioned_groups = if dead_end_certified || !config.goal_conditioned {
            Vec::new()
        } else {
            build_conditioned_groups(&mut optimizer, &mut space, &config)?
        };
        if !dead_end_certified
            && config.rays > rays.len()
            && config.max_ray_generation_time.seconds() > 0.0
        {
            optimizer.prepare_for_ordinary_potential()?;
            let ray_start = Instant::now();
            let mut ray_samples = Vec::new();
            while ray_samples.len() < config.num_samples
                && ray_start.elapsed().as_secs_f64() < config.max_ray_generation_time.seconds()
            {
                let chunk = (config.num_samples - ray_samples.len()).min(16);
                ray_samples.extend(space.sample(chunk)?);
            }
            let generator = ray_generator
                .as_mut()
                .expect("positive ray count must construct a ray generator");
            for state in ray_samples {
                if rays.len() >= config.rays
                    || ray_start.elapsed().as_secs_f64() >= config.max_ray_generation_time.seconds()
                {
                    break;
                }
                let covered = rays
                    .iter()
                    .map(|ray| {
                        ray.value(
                            space.registry.view(&state),
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
                if let Some(candidate) =
                    generator.try_certify(&mut optimizer, space.registry.view(&state))?
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
            build_goal_cost_partitions(&potential_task, &conditioned_groups, &mut space, &config)?
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
                function_value(
                    function,
                    space.registry.view(&space.initial_state),
                    &potential_task,
                )
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
                    function_value(
                        function,
                        space.registry.view(&space.initial_state),
                        &potential_task,
                    )
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
            config.bounds,
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
            functions,
            conditioned_groups,
            goal_cost_partitions,
            goal_group_has_additive_share,
            rays,
            ray_hit_counts: vec![0; ray_count],
            states_pruned_by_rays: 0,
            ray_generator: if enable_online { ray_generator } else { None },
            max_rays: config.rays,
            ray_epsilon: config.ray_epsilon,
            cache_estimates: config.cache_estimates,
            invalidate_online_cache_on_growth: config.invalidate_online_cache_on_growth
                && config.cache_estimates,
            dead_end_certified,
            online_optimizer: enable_online.then_some(optimizer),
            online_state: OnlineState {
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
            },
            revision: 0,
            prop_scratch: Vec::new(),
            numeric_scratch: Vec::new(),
        })
    }
}

impl Heuristic for NumericPotentialHeuristic {
    fn evaluate(&mut self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        if self.dead_end_certified {
            return Ok(f64::INFINITY);
        }
        let registry = eval_state.state_registry();
        let state = registry.view(eval_state.state());
        let prop_scratch = &mut self.prop_scratch;
        let numeric_scratch = &mut self.numeric_scratch;
        for (ray_id, ray) in self.rays.iter().enumerate() {
            let candidate = ray
                .value(state, &self.task, prop_scratch, numeric_scratch)
                .map_err(EvaluationError::ComputationFailed)?;
            if candidate > self.ray_epsilon {
                self.ray_hit_counts[ray_id] += 1;
                self.states_pruned_by_rays += 1;
                return Ok(f64::INFINITY);
            }
        }
        let mut value: f64 = 0.0;
        for function in &self.functions {
            let candidate = function
                .value(state, &self.task, prop_scratch, numeric_scratch)
                .map_err(EvaluationError::ComputationFailed)?;
            value = value.max(candidate);
        }
        let mut group_values = Vec::with_capacity(self.conditioned_groups.len());
        for (group_id, group) in self.conditioned_groups.iter().enumerate() {
            let mut group_value = f64::INFINITY;
            for function in group {
                let candidate = function
                    .value(state, &self.task, prop_scratch, numeric_scratch)
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
        value = self.maybe_add_online_function(eval_state, value)?;
        self.track_online_cache_entry(eval_state.state().get_id());
        Ok(value)
    }

    fn name(&self) -> &str {
        "numeric_potential"
    }

    fn revision(&self) -> u64 {
        self.revision
    }

    fn reevaluate_on_every_pop(&self) -> bool {
        !self.cache_estimates
    }
}

impl NumericPotentialHeuristic {
    fn maybe_add_online_function(
        &mut self,
        eval_state: &EvaluationState<'_, '_>,
        mut envelope_value: f64,
    ) -> Result<f64, EvaluationError> {
        let Some(optimizer) = self.online_optimizer.as_mut() else {
            return Ok(envelope_value);
        };
        let online = &mut self.online_state;
        let advance =
            !online.new_states_only || online.seen_states.insert(eval_state.state().get_id());
        if advance {
            online.evaluations_since_solve += 1;
            online.trigger_states += 1;
        }
        if online.evaluations_since_solve < online.current_interval {
            return Ok(envelope_value);
        }
        online.evaluations_since_solve = 0;
        online.lp_solves += 1;

        let state = eval_state.state_registry().view(eval_state.state());
        let outcome = optimizer
            .optimize_for_state(state)
            .map_err(EvaluationError::ComputationFailed)?;
        let mut useful = false;
        match outcome {
            OptimizationOutcome::Optimal { function, .. } => {
                let online_value = function
                    .value(
                        state,
                        &self.task,
                        &mut self.prop_scratch,
                        &mut self.numeric_scratch,
                    )
                    .map_err(EvaluationError::ComputationFailed)?
                    .max(0.0);
                let tolerance = float_tolerance::LP_EPSILON * online_value.abs().max(1.0);
                if online_value > envelope_value + tolerance {
                    envelope_value = online_value;
                    self.functions.push(function);
                    self.invalidate_online_cache();
                    useful = true;
                }
            }
            OptimizationOutcome::Unbounded { primal_ray } => {
                let generator =
                    exact_verifier(&mut self.ray_generator, optimizer, self.ray_epsilon)
                        .map_err(EvaluationError::ComputationFailed)?;
                let mut candidate = if primal_ray.is_empty() {
                    None
                } else {
                    generator
                        .certify_native(optimizer, primal_ray, state)
                        .map_err(EvaluationError::ComputationFailed)?
                };
                if candidate.is_none() {
                    candidate = generator
                        .try_certify(optimizer, state)
                        .map_err(EvaluationError::ComputationFailed)?;
                }
                if let Some(ray) = candidate {
                    if self.rays.len() < self.max_rays
                        && !generator.is_duplicate(&ray, &self.rays, &self.task)
                    {
                        self.rays.push(ray);
                        self.ray_hit_counts.push(0);
                        self.invalidate_online_cache();
                    }
                    self.online_state.dead_end_certificates += 1;
                    return Ok(f64::INFINITY);
                }
            }
            OptimizationOutcome::Infeasible | OptimizationOutcome::ResourceLimit(_) => {}
        }

        let online = &mut self.online_state;
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
        if online.functions_added >= online.max_functions
            || (online.max_misses > 0 && online.misses >= online.max_misses)
            || (online.max_lp_solves > 0 && online.lp_solves >= online.max_lp_solves)
        {
            self.online_optimizer = None;
        }
        Ok(envelope_value)
    }

    fn track_online_cache_entry(&mut self, state_id: usize) {
        if self.invalidate_online_cache_on_growth {
            self.online_state.clean_cache_states.insert(state_id);
        }
    }

    fn invalidate_online_cache(&mut self) {
        if !self.invalidate_online_cache_on_growth {
            return;
        }
        let online = &mut self.online_state;
        let examined = online.clean_cache_states.len();
        online.cache_invalidations += 1;
        online.cache_entries_examined += examined;
        online.cache_entries_invalidated += examined;
        online.clean_cache_states.clear();
        self.revision += 1;
    }
}

/// The exact verifier of unboundedness in `slot`, built on first use.
fn exact_verifier<'slot>(
    slot: &'slot mut Option<RayGenerator>,
    optimizer: &NumericPotentialOptimizer,
    epsilon: f64,
) -> Result<&'slot mut RayGenerator, String> {
    if slot.is_none() {
        *slot = Some(RayGenerator::new(optimizer, epsilon)?);
    }
    Ok(slot.as_mut().expect("the verifier was just built"))
}

impl Drop for NumericPotentialHeuristic {
    fn drop(&mut self) {
        if self.max_rays > 0 {
            info!("Numeric rays retained: {}", self.rays.len());
            info!(
                "States pruned by numeric rays: {}",
                self.states_pruned_by_rays
            );
            for (ray_id, hits) in self.ray_hit_counts.iter().enumerate() {
                info!("Numeric ray {ray_id} hits: {hits}");
            }
        }
        let online = &self.online_state;
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

fn build_goal_cost_partitions(
    task: &PotentialTask,
    groups: &[Vec<NumericPotentialFunction>],
    space: &mut SampleSpace,
    config: &NumericPotentialConfig,
) -> Result<Vec<Vec<f64>>, String> {
    let mut objective_states = vec![space.initial_state.clone()];
    if config.num_goal_cost_partitions > 1 {
        objective_states.extend(space.sample(config.num_goal_cost_partitions - 1)?);
    }
    let mut partitions = Vec::new();
    for state in objective_states {
        let partition = compute_goal_cost_partition(task, groups, space.registry.view(&state))?;
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
    objective_state: planforge_sas::state_registry::ConcreteStateView<'_>,
) -> Result<Vec<f64>, String> {
    let infinity = Model::infinity();
    let mut variables = Vec::with_capacity(groups.len());
    for group in groups {
        let group_value = group
            .iter()
            .map(|function| function_value(function, objective_state, task))
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
    space: &mut SampleSpace,
    config: &NumericPotentialConfig,
) -> Result<Vec<Vec<NumericPotentialFunction>>, String> {
    let start = Instant::now();
    let mut initial_props = Vec::new();
    space
        .registry
        .view(&space.initial_state)
        .fill_propositional(&mut initial_props);
    let mut goals: Vec<_> = optimizer
        .conditionable_goals()
        .into_iter()
        .filter(|(var, value)| initial_props[var.index()] != *value)
        .map(|(var, value)| {
            let achievers = optimizer.goal_achievers(var, value);
            (var, value, achievers)
        })
        .filter(|(_, _, achievers)| !achievers.is_empty())
        .collect();
    goals.sort_by_key(|(var, value, achievers)| (achievers.len(), *var, *value));

    let mut sample_sets = vec![Vec::new(); config.num_goal_conditioned_heuristics - 1];
    if !sample_sets.is_empty() {
        let samples = space.sample(config.num_goal_conditioned_samples * sample_sets.len())?;
        let set_count = sample_sets.len();
        for (sample_id, sample) in samples.into_iter().enumerate() {
            sample_sets[sample_id % set_count].push(sample);
        }
    }

    let mut groups = Vec::new();
    for (var, value, achievers) in goals {
        if !within_fraction(start, config.max_conditioned_generation_time.seconds(), 1.0) {
            break;
        }
        let mut portfolios: Vec<Vec<NumericPotentialFunction>> = (0..config
            .num_goal_conditioned_heuristics)
            .map(|_| Vec::new())
            .collect();
        let mut complete = vec![true; portfolios.len()];
        for achiever in achievers.iter().copied() {
            if !within_fraction(start, config.max_conditioned_generation_time.seconds(), 1.0) {
                complete.fill(false);
                break;
            }
            match optimizer.optimize_for_conditioned_goal(
                var,
                value,
                achiever,
                space.registry.view(&space.initial_state),
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
                match optimizer
                    .optimize_for_samples(&sample_sets[portfolio_id - 1], &space.registry)?
                {
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

fn build_static_portfolio(
    optimizer: &mut NumericPotentialOptimizer,
    space: &mut SampleSpace,
    config: &NumericPotentialConfig,
    initial_function: NumericPotentialFunction,
) -> Result<Vec<NumericPotentialFunction>, String> {
    match config.opt {
        OptimizeFor::InitialState => Ok(vec![initial_function]),
        OptimizeFor::AllStates => {
            match optimizer
                .optimize_for_all_propositional_states(space.registry.view(&space.initial_state))?
            {
                OptimizationOutcome::Optimal { function, .. } => Ok(vec![function]),
                _ => Ok(vec![initial_function]),
            }
        }
        OptimizeFor::Samples => {
            let samples = space.sample(config.num_samples)?;
            match optimizer.optimize_for_samples(&samples, &space.registry)? {
                OptimizationOutcome::Optimal { function, .. } => Ok(vec![function]),
                _ => Ok(vec![initial_function]),
            }
        }
        OptimizeFor::DiverseSamples => {
            build_diverse_portfolio(optimizer, space, config, initial_function)
        }
    }
}

struct SampleTarget {
    state: ConcreteState,
    optimum: NumericPotentialFunction,
}

fn build_diverse_portfolio(
    optimizer: &mut NumericPotentialOptimizer,
    space: &mut SampleSpace,
    config: &NumericPotentialConfig,
    initial_function: NumericPotentialFunction,
) -> Result<Vec<NumericPotentialFunction>, String> {
    let start = Instant::now();
    let mut functions = Vec::new();
    let mut initial_function = Some(initial_function);
    if config.include_initial_state_potential {
        functions.push(initial_function.take().unwrap());
    }
    if config.include_all_states_potential
        && functions.len() < config.num_heuristics
        && let OptimizationOutcome::Optimal { function, .. } = optimizer
            .optimize_for_all_propositional_states(space.registry.view(&space.initial_state))?
    {
        functions.push(function);
    }
    if functions.len() >= config.num_heuristics
        || config.max_diverse_generation_time.seconds() <= 0.0
    {
        if functions.is_empty() {
            functions.push(initial_function.take().unwrap());
        }
        return Ok(functions);
    }

    let mut samples = Vec::new();
    while samples.len() < config.num_samples
        && within_fraction(start, config.max_diverse_generation_time.seconds(), 0.25)
    {
        let chunk = (config.num_samples - samples.len()).min(16);
        samples.extend(space.sample(chunk)?);
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
        if !seen.is_empty()
            && !within_fraction(start, config.max_diverse_generation_time.seconds(), 0.75)
        {
            break;
        }
        if !seen.insert(state.get_id()) {
            continue;
        }
        if let OptimizationOutcome::Optimal { function, .. } =
            optimizer.optimize_for_state(space.registry.view(&state))?
        {
            targets.push(SampleTarget {
                state,
                optimum: function,
            });
        }
    }

    for function in &functions {
        remove_covered(function, &mut targets, optimizer.task(), &space.registry)?;
    }
    while !targets.is_empty()
        && functions.len() < config.num_heuristics
        && within_fraction(start, config.max_diverse_generation_time.seconds(), 1.0)
    {
        let states: Vec<_> = targets.iter().map(|target| target.state.clone()).collect();
        let OptimizationOutcome::Optimal {
            function: mut candidate,
            ..
        } = optimizer.optimize_for_samples(&states, &space.registry)?
        else {
            break;
        };
        let covered = remove_covered(&candidate, &mut targets, optimizer.task(), &space.registry)?;
        if covered == 0 {
            let target_id = match config.diverse_fallback {
                DiverseFallback::Random => space.sampler.choose_index(targets.len()),
                DiverseFallback::LargestGap => {
                    largest_gap_target(&functions, &targets, optimizer.task(), &space.registry)?
                }
            };
            // C++ uses vector::erase here. Preserve the order of all
            // remaining targets because the same global RNG stream is reused
            // by later random fallback choices.
            candidate = targets.remove(target_id).optimum;
            remove_covered(&candidate, &mut targets, optimizer.task(), &space.registry)?;
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
    state: planforge_sas::state_registry::ConcreteStateView<'_>,
    task: &PotentialTask,
) -> Result<f64, String> {
    function.value(state, task, &mut Vec::new(), &mut Vec::new())
}

fn remove_covered(
    function: &NumericPotentialFunction,
    targets: &mut Vec<SampleTarget>,
    task: &PotentialTask,
    registry: &StateRegistry,
) -> Result<usize, String> {
    let before = targets.len();
    let mut kept = Vec::with_capacity(targets.len());
    for target in targets.drain(..) {
        let state = registry.view(&target.state);
        let optimum = function_value(&target.optimum, state, task)?.max(0.0);
        let value = function_value(function, state, task)?.max(0.0);
        let tolerance = float_tolerance::LP_EPSILON * optimum.abs().max(1.0);
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
    registry: &StateRegistry,
) -> Result<usize, String> {
    let mut best = 0;
    let mut largest_gap = f64::NEG_INFINITY;
    for (target_id, target) in targets.iter().enumerate() {
        let state = registry.view(&target.state);
        let optimum = function_value(&target.optimum, state, task)?.max(0.0);
        let mut ensemble: f64 = 0.0;
        for function in functions {
            ensemble = ensemble.max(function_value(function, state, task)?.max(0.0));
        }
        if optimum - ensemble > largest_gap {
            largest_gap = optimum - ensemble;
            best = target_id;
        }
    }
    Ok(best)
}
