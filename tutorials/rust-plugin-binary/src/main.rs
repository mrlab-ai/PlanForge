use std::cmp::Ordering;
use std::collections::BinaryHeap;

use ordered_float::OrderedFloat;
use planforge_search::evaluation::{EvaluationError, EvaluationState, Heuristic};
use planforge_search::heuristic_factory::{
    BuildContext, EmptyConfig, HeuristicBuildError, HeuristicFeature, entry,
};
use planforge_search::search::best_first::{self, OpenEntry, OpenList, OpenListFeature};

struct GoalCountHeuristic;

impl Heuristic for GoalCountHeuristic {
    fn evaluate(&mut self, eval_state: &EvaluationState<'_, '_>) -> Result<f64, EvaluationError> {
        let state = eval_state.state_registry().view(eval_state.state());
        let unsatisfied = eval_state
            .task()
            .goals()
            .iter()
            .filter(|goal| !goal.is_held(state))
            .count();
        Ok(unsatisfied as f64)
    }

    fn name(&self) -> &str {
        "goalcount"
    }
}

struct GoalCountFeature;
impl HeuristicFeature for GoalCountFeature {
    const NAME: &'static str = "goalcount";
    type Config = EmptyConfig;
    fn build<'a>(
        _: &Self::Config,
        _: &BuildContext<'a>,
    ) -> Result<Box<dyn Heuristic + 'a>, HeuristicBuildError> {
        Ok(Box::new(GoalCountHeuristic))
    }
}

/// An open list of your own: one heap by `g + h`, then `h`, then insertion
/// order. Under A* it expands exactly as the built-in `tiebreaking()` does,
/// so `astar(blind(), open_list=f_heap())` measures what the search pays for
/// a registered list (`Box<dyn OpenList>`) over a statically dispatched one.
#[derive(Default)]
struct FHeap(BinaryHeap<ByF>);

struct ByF(OpenEntry);

impl ByF {
    fn key(&self) -> (OrderedFloat<f64>, OrderedFloat<f64>, u32) {
        let entry = self.0;
        let f = entry.g() + entry.h();
        (
            OrderedFloat(f),
            OrderedFloat(entry.h()),
            entry.insertion_order(),
        )
    }
}

impl PartialEq for ByF {
    fn eq(&self, other: &Self) -> bool {
        self.key() == other.key()
    }
}
impl Eq for ByF {}
impl PartialOrd for ByF {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ByF {
    /// Reversed: `BinaryHeap` pops the greatest element.
    fn cmp(&self, other: &Self) -> Ordering {
        other.key().cmp(&self.key())
    }
}

impl OpenList for FHeap {
    fn uses_preferred_operators(&self) -> bool {
        false
    }
    fn push(&mut self, entry: OpenEntry) {
        self.0.push(ByF(entry));
    }
    fn pop(&mut self) -> Option<OpenEntry> {
        self.0.pop().map(|entry| entry.0)
    }
    fn len(&self) -> usize {
        self.0.len()
    }
}

struct FHeapFeature;
impl OpenListFeature for FHeapFeature {
    const NAME: &'static str = "f_heap";
    type Config = EmptyConfig;
    fn build(_: &EmptyConfig) -> Box<dyn OpenList> {
        Box::new(FHeap::default())
    }
}

fn main() -> std::io::Result<()> {
    planforge::run_with(planforge::Extensions {
        heuristics: vec![entry::<GoalCountFeature>()],
        open_lists: vec![best_first::entry::<FHeapFeature>()],
        ..Default::default()
    })
}
