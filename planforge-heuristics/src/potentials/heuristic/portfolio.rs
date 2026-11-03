//! How a potential heuristic chooses its potential functions: per-goal cost
//! partitions, goal-conditioned groups, and static and diverse portfolios.

use super::*;

pub(super) fn build_goal_cost_partitions(
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
                    .all(|(left, right)| (*left - *right).abs() <= float_tolerance::SEARCH_EPSILON)
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

pub(super) fn build_conditioned_groups(
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

pub(super) fn build_static_portfolio(
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
        functions.push(
            initial_function
                .take()
                .expect("the initial-state function joins the portfolio once"),
        );
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
            functions.push(
                initial_function
                    .take()
                    .expect("the initial-state function joins the portfolio once"),
            );
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
            functions.push(
                initial_function
                    .take()
                    .expect("the initial-state function joins the portfolio once"),
            );
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
        if !seen.insert(state.id()) {
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
        functions.push(
            initial_function
                .take()
                .expect("the initial-state function joins the portfolio once"),
        );
    }
    Ok(functions)
}

fn within_fraction(start: Instant, seconds: f64, fraction: f64) -> bool {
    !seconds.is_finite() || start.elapsed().as_secs_f64() < seconds * fraction
}

/// `None` for a task without operators, whose average is undefined.
pub(super) fn average_operator_cost(task: &PotentialTask) -> Option<f64> {
    (!task.operators.is_empty()).then(|| {
        task.operators
            .iter()
            .map(|operator| operator.cost)
            .sum::<f64>()
            / task.operators.len() as f64
    })
}

pub(super) fn function_value(
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
