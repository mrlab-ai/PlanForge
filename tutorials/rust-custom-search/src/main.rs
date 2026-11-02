//! A search algorithm of your own, named by the `--search` grammar: enforced
//! hill-climbing (Hoffmann and Nebel 2001), built from the search building
//! blocks `SearchSpace` and `SearchProgress`.
//!
//! From the current state, a breadth-first search looks for a state with a
//! lower heuristic estimate, which becomes the next current state. As in Fast
//! Downward, the search space is shared by all breadth-first searches, so a
//! state is evaluated once, and the plan is traced back through it.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, bail};
use planforge_sas::numeric_task::Task;
use planforge_search::config::ApplyOptions;
use planforge_search::evaluation::Heuristic;
use planforge_search::heuristic_factory::{BuildContext, HeuristicConfig};
use planforge_search::search::{
    Reached, SEARCH_ALGORITHMS, SearchFeature, SearchLimits, SearchOutcome, SearchProgress,
    SearchResult, SearchSpace, entry, parse_search_spec,
};
use planforge_search::successor_generator::SuccessorGeneratorConfig;

/// `ehc(heuristic, successor_generator=decision_tree())`; the heuristic
/// defaults to blind.
#[derive(Debug, Default, ApplyOptions)]
struct EhcConfig {
    #[option(positional)]
    heuristic: HeuristicConfig,
    successor_generator: SuccessorGeneratorConfig,
}

struct EnforcedHillClimbing;

impl SearchFeature for EnforcedHillClimbing {
    const NAME: &'static str = "ehc";
    type Config = EhcConfig;

    fn heuristics(config: &EhcConfig) -> Vec<&HeuristicConfig> {
        vec![&config.heuristic]
    }

    fn run(
        config: &EhcConfig,
        task: Arc<Task>,
        limits: SearchLimits,
    ) -> std::io::Result<SearchResult> {
        let heuristic = config
            .heuristic
            .build(&BuildContext::new(&task))
            .map_err(|error| error.into_io_error())?;
        let space = SearchSpace::new(task.clone(), config.successor_generator.build(&task));
        enforced_hill_climbing(space, heuristic, limits)
            .map_err(|error| std::io::Error::other(format!("{error:#}")))
    }
}

fn enforced_hill_climbing(
    mut space: SearchSpace,
    mut heuristic: Box<dyn Heuristic + '_>,
    limits: SearchLimits,
) -> Result<SearchResult> {
    let mut progress = SearchProgress::start(limits, heuristic.name());
    let mut successors = Vec::new();
    let mut queue = VecDeque::new();

    let initial = space.initial_state();
    assert_eq!(space.reach(&initial, None, 0.0), Reached::New);
    let mut best_h = heuristic.evaluate(&space.evaluation_state(&initial))?;
    progress.statistics.nodes.evaluated += 1;
    progress.statistics.evaluations += 1;
    progress.report_initial_h(best_h);
    if best_h == f64::INFINITY {
        return Ok(progress.finish(SearchOutcome::Unsolvable, &space));
    }
    queue.push_back(initial);

    let outcome = 'search: loop {
        if let Some(outcome) = progress.limit_reached() {
            break outcome;
        }
        let Some(state) = queue.pop_front() else {
            bail!("enforced hill-climbing found no state with an estimate below {best_h}");
        };
        let state_id = state.get_id();
        if space.is_goal(state_id) {
            break space.solution(state_id);
        }
        space.close(state_id);
        progress.statistics.nodes.expanded += 1;
        let g = space.g(state_id);
        space.generate_successors(&state, &mut successors)?;
        for successor in &successors {
            progress.statistics.nodes.generated += 1;
            let path = Some((state_id, successor.operator));
            if space.reach(&successor.state, path, g + successor.cost) != Reached::New {
                // Each state is evaluated, and expanded, at most once.
                continue;
            }
            let successor_id = successor.state.get_id();
            let is_goal = space.is_goal(successor_id);
            let h = heuristic.evaluate(&space.evaluation_state(&successor.state))?;
            progress.statistics.nodes.evaluated += 1;
            progress.statistics.evaluations += 1;
            if h == f64::INFINITY {
                space.mark_dead_end(successor_id);
                progress.statistics.dead_ends += 1;
            } else if h < best_h || is_goal {
                // A better state: the breadth-first search starts over from it.
                best_h = h;
                progress.report_best_h(h, g + successor.cost);
                queue.clear();
                queue.push_back(successor.state.clone());
                continue 'search;
            } else {
                queue.push_back(successor.state.clone());
            }
        }
    };
    Ok(progress.finish(outcome, &space))
}

fn main() -> Result<()> {
    // A binary registers its search algorithms once, before it parses a
    // configuration; `planforge::run_with` does so for the real CLI.
    SEARCH_ALGORITHMS
        .register(vec![entry::<EnforcedHillClimbing>()])
        .map_err(anyhow::Error::msg)?;
    let search = parse_search_spec("ehc(blind())").map_err(anyhow::Error::msg)?;

    let task: Arc<Task> = Arc::new(Task::from_file("tests/assets/numeric_sas/example5.sas"));
    let limits = SearchLimits {
        time: Some(Duration::from_secs(10)),
        memory_bytes: None,
    };
    let result = search.solve(task, limits)?;
    let SearchOutcome::Solved { plan, cost } = &result.outcome else {
        bail!("{search} found no plan: {:?}", result.outcome);
    };
    println!(
        "{search} found a plan of {} steps and cost {cost} after {} expansions",
        plan.len(),
        result.statistics.nodes.expanded
    );
    Ok(())
}
