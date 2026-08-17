use super::*;

pub(crate) fn debug_print_abstraction_stats(
    iteration: usize,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
) {
    let prop_vars = domain_sizes.len();
    let num_vars = numeric_domain_sizes.len();
    let refined_props = domain_sizes.iter().filter(|&&s| s > 1).count();
    let refined_nums = numeric_domain_sizes.iter().filter(|&&s| s > 1).count();
    let size = compute_abstraction_size_u128(domain_sizes, numeric_domain_sizes)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "<overflow>".to_string());

    let prop_max = domain_sizes.iter().copied().max().unwrap_or(0);
    let num_max = numeric_domain_sizes.iter().copied().max().unwrap_or(0);

    debug!(
        "[CEGAR] iteration {iteration}: abstract_states={size} (prop_vars={prop_vars}, num_vars={num_vars}, refined_prop={refined_props}, refined_num={refined_nums}, max_prop_size={prop_max}, max_num_parts={num_max})"
    );
}

pub(crate) fn debug_print_refinement_summary(
    before: Option<u128>,
    after: Option<u128>,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    refined: bool,
) {
    let before_s = before
        .map(|v| v.to_string())
        .unwrap_or_else(|| "<overflow>".to_string());
    let after_s = after
        .map(|v| v.to_string())
        .unwrap_or_else(|| "<overflow>".to_string());
    debug!("[Refine] refined={refined} abstract_states: {before_s} -> {after_s}");

    let mut refined_props: Vec<(usize, usize)> = domain_sizes
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| (s > 1).then_some((i, s)))
        .collect();
    refined_props.sort_by_key(|(i, _)| *i);
    let refined_nums: Vec<(usize, usize)> = numeric_domain_sizes
        .iter()
        .enumerate()
        .filter_map(|(i, &s)| (s > 1).then_some((i, s)))
        .collect();

    if !refined_props.is_empty() {
        let preview = 30usize;
        let mut line = String::new();
        let _ = write!(
            &mut line,
            "[Refine] propositional splits: {} vars",
            refined_props.len()
        );
        for (i, s) in refined_props.iter().take(preview) {
            let _ = write!(&mut line, " v{i}=>{s}");
        }
        if refined_props.len() > preview {
            let _ = write!(&mut line, " ...");
        }
        debug!("{line}");
    }
    if !refined_nums.is_empty() {
        let preview = 30usize;
        let mut line = String::new();
        let _ = write!(
            &mut line,
            "[Refine] numeric splits: {} vars",
            refined_nums.len()
        );
        for (i, s) in refined_nums.iter().take(preview) {
            let _ = write!(&mut line, " n{i}=>{s}");
        }
        if refined_nums.len() > preview {
            let _ = write!(&mut line, " ...");
        }
        debug!("{line}");
    }
}

pub(crate) fn debug_print_flaws(flaws: &[Flaw]) {
    debug!("[Flaws] count={}", flaws.len());
    let max = 200usize;
    let shown = flaws.len().min(max);
    for (i, flaw) in flaws.iter().take(shown).enumerate() {
        match flaw {
            Flaw::Propositional(pf) => {
                debug!(
                    "  {i}: PropFlaw fact=(var={}, val={}) deps={}",
                    pf.fact.var(),
                    pf.fact.value(),
                    pf.dependent_numeric_flaws.len()
                );
                for (j, nf) in pf.dependent_numeric_flaws.iter().enumerate() {
                    debug!(
                        "      - dep[{j}]: NumericFlaw var={} value={} include_in_lower={}",
                        nf.numeric_var_id, nf.value, nf.include_in_lower
                    );
                }
            }
            Flaw::Numeric(nf) => {
                debug!(
                    "  {i}: NumericFlaw var={} value={} include_in_lower={}",
                    nf.numeric_var_id, nf.value, nf.include_in_lower
                );
            }
        }
    }
    if flaws.len() > max {
        debug!("[Flaws] (truncated: showing {shown} of {})", flaws.len());
    }
}

pub(crate) fn debug_print_wildcard_plan(
    task: &dyn AbstractNumericTask,
    plan: &WildcardPlanResult,
    domain_sizes: &[usize],
    numeric_domain_sizes: &[usize],
    partitions: &NumericPartitions,
) -> Result<()> {
    let steps = plan.wildcard_plan.len();
    debug!("[Abstract Plan] steps={steps}");

    let max_steps = 200usize;
    let shown_steps = steps.min(max_steps);
    if steps > max_steps {
        debug!("[Abstract Plan] (truncated to first {shown_steps} steps)");
    }

    if let Some(prop0) = plan.abstract_prop_states.first() {
        debug!(
            "  s0 props: {}",
            fmt_nontrivial_props(prop0, domain_sizes, 100)
        );
    }
    if let Some(num0) = plan.abstract_numeric_states.first() {
        debug!(
            "  s0 nums:  {}",
            fmt_nontrivial_nums(num0, numeric_domain_sizes, partitions, 100)
        );
    }

    let ops = task.get_operators();
    let mut representative: Vec<String> = Vec::with_capacity(shown_steps);

    for i in 0..shown_steps {
        let choices = &plan.wildcard_plan[i];
        let choice_count = choices.len();
        let rep = choices
            .first()
            .and_then(|&id| ops.get(id).map(|op| op.name().to_string()))
            .unwrap_or_else(|| "<none>".to_string());
        representative.push(rep);

        let mut line = String::new();
        let _ = write!(&mut line, "  step {i}: options={choice_count}");
        let preview = 10usize;
        for &op_id in choices.iter().take(preview) {
            let name = ops.get(op_id).map(|op| op.name()).unwrap_or("<bad-op-id>");
            let _ = write!(&mut line, " [{op_id}:{name}]");
        }
        if choice_count > preview {
            let _ = write!(&mut line, " ...");
        }
        debug!("{line}");

        if i + 1 < plan.abstract_prop_states.len() {
            let prev = &plan.abstract_prop_states[i];
            let cur = &plan.abstract_prop_states[i + 1];
            let delta = fmt_delta_i32(prev, cur, 50);
            if !delta.is_empty() {
                debug!("    props Δ: {delta}");
            }
        }
        if i + 1 < plan.abstract_numeric_states.len() {
            let prev = &plan.abstract_numeric_states[i];
            let cur = &plan.abstract_numeric_states[i + 1];
            let delta = fmt_delta_numeric_partitions(prev, cur, partitions, 50);
            if !delta.is_empty() {
                debug!("    nums  Δ: {delta}");
            }
        }
    }

    debug!("[Plan] {}", representative.join(" -> "));
    debug_print_concrete_trace(task, plan, partitions, shown_steps)?;
    Ok(())
}

fn debug_print_concrete_trace(
    task: &dyn AbstractNumericTask,
    plan: &WildcardPlanResult,
    partitions: &NumericPartitions,
    shown_steps: usize,
) -> Result<()> {
    let state_packer = std::sync::Arc::new(make_prop_state_packer(task));
    let axiom_evaluator = AxiomEvaluator::new(std::sync::Arc::new(task), state_packer.clone());

    let mut buffer = vec![0u64; state_packer.num_bins()];
    set_initial_prop_values(task, &state_packer, &mut buffer);
    let mut numeric_state: Vec<f64> = task.get_initial_numeric_state_values().to_vec();

    axiom_evaluator
        .evaluate(&mut buffer, &mut numeric_state)
        .map_err(|error| anyhow!("failed to evaluate initial-state axioms: {error:?}"))?;

    let (prop_scope, num_scope) = trace_variable_scope(task, plan, shown_steps);
    debug!(
        "[Concrete Trace] scope: props={} nums={}",
        prop_scope.len(),
        num_scope.len()
    );
    debug!(
        "  s0 props: {}",
        fmt_concrete_props(task, &state_packer, &buffer, &prop_scope, 200)
    );
    debug!(
        "  s0 nums:  {}",
        fmt_concrete_nums(&numeric_state, &num_scope, partitions, 200)
    );

    let max_tries_per_step = 30usize;
    for step in 0..shown_steps {
        if step + 1 >= plan.abstract_numeric_states.len() {
            break;
        }
        let expected_abs_numeric_succ = &plan.abstract_numeric_states[step + 1];
        let choices = plan
            .wildcard_plan
            .get(step)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);

        let mut chosen: Option<(usize, Vec<u64>, Vec<f64>)> = None;
        let mut tries = 0usize;
        for &op_id in choices.iter() {
            if tries >= max_tries_per_step {
                debug!("  step {step}: ... (tried first {max_tries_per_step} options)");
                break;
            }
            let Some(op) = task.get_operators().get(op_id) else {
                continue;
            };
            tries += 1;

            // Debug-trace only; we use Forward direction which does not
            // consult `deltas`, so an empty map is fine here.
            let deltas: HashMap<usize, Vec<f64>> = HashMap::new();
            let applicable = get_progression_precondition_flaws(
                PartitionedTask {
                    task,
                    partitions,
                    deltas: &deltas,
                },
                op,
                ConcreteStateView::from_decoded(&state_packer, &buffer, &numeric_state),
                step,
                SplitDirection::Forward,
            )
            .is_empty();
            if !applicable {
                continue;
            }

            let mut cand_buffer = buffer.clone();
            let mut cand_numeric = numeric_state.clone();
            progress_concrete_state(
                op,
                &axiom_evaluator,
                &state_packer,
                &mut cand_buffer,
                &mut cand_numeric,
            )
            .expect("Error applying operator");

            let deviation_flaws = get_progression_numeric_deviation_flaws(
                task,
                op,
                NumericTransitionStates {
                    current: &numeric_state,
                    successor: &cand_numeric,
                    abstract_successor: expected_abs_numeric_succ,
                },
                partitions,
                step,
                SplitDirection::Forward,
            );

            if deviation_flaws.is_empty() {
                debug!("  step {step}: choose [{op_id}:{}]", op.name());
                chosen = Some((op_id, cand_buffer, cand_numeric));
                break;
            } else {
                debug!(
                    "  step {step}: try    [{op_id}:{}] (reject: numeric deviation)",
                    op.name()
                );
                debug!(
                    "    s{}' props: {}",
                    step + 1,
                    fmt_concrete_props(task, &state_packer, &cand_buffer, &prop_scope, 80)
                );
                debug!(
                    "    s{}' nums:  {}",
                    step + 1,
                    fmt_concrete_nums(&cand_numeric, &num_scope, partitions, 80)
                );
            }
        }

        let Some((_op_id, next_buffer, next_numeric)) = chosen else {
            debug!("  step {step}: no applicable concrete operator found for wildcard options");
            break;
        };
        buffer = next_buffer;
        numeric_state = next_numeric;

        debug!(
            "  s{} props: {}",
            step + 1,
            fmt_concrete_props(task, &state_packer, &buffer, &prop_scope, 200)
        );
        debug!(
            "  s{} nums:  {}",
            step + 1,
            fmt_concrete_nums(&numeric_state, &num_scope, partitions, 200)
        );
    }
    Ok(())
}

fn trace_variable_scope(
    task: &dyn AbstractNumericTask,
    plan: &WildcardPlanResult,
    shown_steps: usize,
) -> (Vec<usize>, Vec<usize>) {
    let ops = task.get_operators();
    let mut prop_vars: BTreeSet<usize> = BTreeSet::new();
    let mut num_vars: BTreeSet<usize> = BTreeSet::new();

    for choices in plan.wildcard_plan.iter().take(shown_steps) {
        for &op_id in choices.iter() {
            let Some(op) = ops.get(op_id) else {
                continue;
            };
            for pre in op.preconditions().iter() {
                prop_vars.insert(pre.var());
            }
            for eff in op.effects().iter() {
                prop_vars.insert(eff.var_id());
                for c in eff.conditions().iter() {
                    prop_vars.insert(c.var());
                }
            }
            for neff in op.assignment_effects().iter() {
                num_vars.insert(neff.var_id());
                num_vars.insert(neff.affected_var_id());
                for c in neff.conditions().iter() {
                    prop_vars.insert(c.var());
                }
            }
        }
    }

    (
        prop_vars.into_iter().collect(),
        num_vars.into_iter().collect(),
    )
}

fn fmt_concrete_props(
    task: &dyn AbstractNumericTask,
    packer: &IntDoublePacker,
    buffer: &[u64],
    var_ids: &[usize],
    max_items: usize,
) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    for &var_id in var_ids.iter() {
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        let dom = task
            .variables()
            .get(var_id)
            .map(|v| v.domain_size())
            .unwrap_or(0);
        if dom <= 1 {
            continue;
        }
        if shown > 0 {
            out.push(' ');
        }
        let val = packer.get(buffer, var_id);
        let _ = write!(&mut out, "v{var_id}={val}");
        shown += 1;
    }
    if out.is_empty() {
        "<empty>".to_string()
    } else {
        out
    }
}

fn fmt_concrete_nums(
    numeric_state: &[f64],
    var_ids: &[usize],
    partitions: &NumericPartitions,
    max_items: usize,
) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    for &num_id in var_ids.iter() {
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        let Some(&v) = numeric_state.get(num_id) else {
            continue;
        };
        if shown > 0 {
            out.push(' ');
        }
        let mut part_s = String::new();
        if let Some(parts) = partitions.partitions(num_id)
            && let Some(pid) = partition_for_value(parts, v)
        {
            let iv_s = partitions
                .partition_interval(num_id, pid)
                .map(fmt_interval)
                .unwrap_or_else(|| "<missing-interval>".to_string());
            part_s = format!(" p{pid}:{iv_s}");
        }
        let _ = write!(&mut out, "n{num_id}={}{}", fmt_f64_compact(v), part_s);
        shown += 1;
    }
    if out.is_empty() {
        "<empty>".to_string()
    } else {
        out
    }
}

fn fmt_delta_i32(prev: &[usize], cur: &[usize], max_items: usize) -> String {
    let mut out = String::new();
    let len = prev.len().min(cur.len());
    let mut shown = 0usize;
    for i in 0..len {
        let a = prev[i];
        let b = cur[i];
        if a == b {
            continue;
        }
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        if shown > 0 {
            out.push(' ');
        }
        let _ = write!(&mut out, "{i}:{a}->{b}");
        shown += 1;
    }
    out
}

fn fmt_nontrivial_props(values: &[usize], domain_sizes: &[usize], max_items: usize) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    let len = values.len().min(domain_sizes.len());
    for var_id in 0..len {
        if domain_sizes[var_id] <= 1 {
            continue;
        }
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        if shown > 0 {
            out.push(' ');
        }
        let _ = write!(&mut out, "v{var_id}:{}", values[var_id]);
        shown += 1;
    }
    if out.is_empty() {
        "<no-nontrivial-vars>".to_string()
    } else {
        out
    }
}

fn fmt_nontrivial_nums(
    values: &[usize],
    numeric_domain_sizes: &[usize],
    partitions: &NumericPartitions,
    max_items: usize,
) -> String {
    let mut out = String::new();
    let mut shown = 0usize;
    let len = values.len().min(numeric_domain_sizes.len());
    for num_id in 0..len {
        if numeric_domain_sizes[num_id] <= 1 {
            continue;
        }
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        if shown > 0 {
            out.push(' ');
        }
        let part = values[num_id];
        let iv_s = partitions
            .partition_interval(num_id, part)
            .map(fmt_interval)
            .unwrap_or_else(|| "<missing-interval>".to_string());
        let _ = write!(&mut out, "n{num_id}=p{part}:{iv_s}");
        shown += 1;
    }
    if out.is_empty() {
        "<no-nontrivial-vars>".to_string()
    } else {
        out
    }
}

fn fmt_delta_numeric_partitions(
    prev: &[usize],
    cur: &[usize],
    partitions: &NumericPartitions,
    max_items: usize,
) -> String {
    let mut out = String::new();
    let len = prev.len().min(cur.len());
    let mut shown = 0usize;
    for num_id in 0..len {
        let a = prev[num_id];
        let b = cur[num_id];
        if a == b {
            continue;
        }
        if shown >= max_items {
            let _ = write!(&mut out, " ...");
            break;
        }
        if shown > 0 {
            out.push(' ');
        }
        let a_s = partitions
            .partition_interval(num_id, a)
            .map(fmt_interval)
            .unwrap_or_else(|| "<missing-interval>".to_string());
        let b_s = partitions
            .partition_interval(num_id, b)
            .map(fmt_interval)
            .unwrap_or_else(|| "<missing-interval>".to_string());
        let _ = write!(&mut out, "n{num_id}:p{a}:{a_s}->p{b}:{b_s}");
        shown += 1;
    }
    out
}

#[allow(unused)]
pub(crate) fn debug_print_evaluate_state(
    prop_str: &str,
    num_str_vec: &[String],
    abs_prop_str: &[String],
    abs_num_str: &[String],
    dist: f64,
) {
    debug!("[Evaluate State]");
    debug!("  concrete props: {}", prop_str);
    debug!("  concrete nums:  {}", num_str_vec.join(" "));
    debug!("  abstract props: {}", abs_prop_str.join(" "));
    debug!("  abstract nums:  {}", abs_num_str.join(" "));
    debug!("  distance:       {}", dist);
}

pub(crate) fn dump_distances(
    factory: &DomainAbstractionFactory,
    task: &dyn AbstractNumericTask,
    table: &AbstractDistanceTable,
) {
    let num_states = table.distances.len();
    debug!("\n=== TABLE OF CORE VARIABLES FOR ALL {num_states} STATES ===\n");

    let num_prop_vars = factory.domain_sizes().len();
    if table.hash_multipliers.len() < num_prop_vars + table.numeric_domain_sizes.len() {
        debug!(
            "[dump_distances] invalid hash_multipliers len={} (expected >= {})",
            table.hash_multipliers.len(),
            num_prop_vars + table.numeric_domain_sizes.len()
        );
        return;
    }

    let mut is_axiom_var: Vec<bool> = vec![false; num_prop_vars];
    for ax in task.axioms().iter() {
        let v = ax.var_id();
        if v < is_axiom_var.len() {
            is_axiom_var[v] = true;
        }
    }

    let refined_numeric_vars: Vec<usize> = table
        .numeric_domain_sizes
        .iter()
        .enumerate()
        .filter_map(|(n, &parts)| (parts > 1).then_some(n))
        .collect();

    let non_axiom_vars: Vec<usize> = factory
        .domain_sizes()
        .iter()
        .enumerate()
        .filter_map(|(v, &dom)| {
            if dom > 1 && !is_axiom_var.get(v).copied().unwrap_or(false) {
                Some(v)
            } else {
                None
            }
        })
        .collect();

    if !refined_numeric_vars.is_empty() || !non_axiom_vars.is_empty() {
        debug!("=== ABSTRACT DOMAINS ===");
    }

    if !refined_numeric_vars.is_empty() {
        debug!("[NumericPartitions]");
        for &num_var_id in &refined_numeric_vars {
            let name = task
                .numeric_variables()
                .get(num_var_id)
                .map(|v| v.name())
                .unwrap_or("<unknown>");
            let parts = factory.partitions().partitions(num_var_id).unwrap_or(&[]);
            debug!("  n{num_var_id}({name}) parts={}", parts.len());
            for (pid, iv) in parts.iter().enumerate() {
                debug!("    p{pid}: {}", fmt_interval(*iv));
            }
        }
    }
}
