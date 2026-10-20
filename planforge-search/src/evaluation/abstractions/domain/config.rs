//! Configuration of domain-abstraction CEGAR.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;
use planforge_sas::numeric_task::VariableIndex;

use crate::evaluation::abstractions::common::cegar::{
    FlawKind, FlawScope, SplitKind, SplitSelection,
};

#[derive(Debug, Clone)]
pub struct DomainAbstractionConfig {
    pub max_states: usize,
    pub max_iterations: usize,
    pub max_time: Option<Duration>,
    pub use_wildcard_plans: bool,
    pub combine_labels: bool,
    pub debug: bool,
    pub random_seed: Option<u64>,
    pub flaw_kind: FlawKind,
    pub flaw_scope: FlawScope,
    pub split_kind: SplitKind,
    pub split_selection: SplitSelection,
    pub init_split_var_ids: Option<HashSet<VariableIndex>>,
    pub blacklisted_prop_var_ids: HashSet<VariableIndex>,
    pub blacklisted_numeric_var_ids: HashSet<VariableIndex>,
    /// When false, `DomainAbstractionGenerator::generate` skips building the
    /// abstract operators' source sets. Only saturated cost partitioning
    /// reads them; heuristics that read only the distance table, such as the
    /// canonical and maximum heuristics, turn this off, since the source sets
    /// can be large (~12 GB of `CartesianSet`s on
    /// minecraft-sword-advanced/prob_30x30_5).
    pub compute_source_sets: bool,
    /// Which of the tied best splits to take: `split_selection` scores the
    /// candidates, the best-scoring ones are sorted by a deterministic key,
    /// and the one at `rank % ties` is taken. `None` breaks ties with the
    /// random seed.
    pub flaw_selection_rank: Option<usize>,
}

impl Default for DomainAbstractionConfig {
    fn default() -> Self {
        Self {
            max_states: 100_000,
            max_iterations: usize::MAX,
            max_time: None,
            use_wildcard_plans: false,
            combine_labels: true,
            debug: false,
            random_seed: Some(2011),
            flaw_kind: FlawKind::Progression,
            flaw_scope: FlawScope::First,
            split_kind: SplitKind::Concrete,
            split_selection: SplitSelection::Random,
            init_split_var_ids: None,
            blacklisted_prop_var_ids: HashSet::new(),
            blacklisted_numeric_var_ids: HashSet::new(),
            compute_source_sets: true,
            flaw_selection_rank: None,
        }
    }
}

/// `domain_abstraction(...)` and friends expose a curated subset of the fields:
/// the rest are internal runtime flags the collection generators set
/// themselves, so this is written by hand rather than derived.
impl crate::config::ApplyOptions for DomainAbstractionConfig {
    fn apply_options(&mut self, args: &[crate::config::ConfigArg]) -> Result<(), String> {
        use crate::config::{ConfigValue, FromOptionValue, for_each_option};

        /// Type-inferring shortcut for `<T as FromOptionValue>::from_option_value`.
        fn parse<T: FromOptionValue>(value: &ConfigValue) -> Result<T, String> {
            T::from_option_value(value)
        }

        const ORDER: &[&str] = &[
            "max_states",
            "max_iterations",
            "max_time",
            "use_wildcard_plans",
            "combine_labels",
            "random_seed",
            "flaw_kind",
            "flaw_scope",
            "split_kind",
            "split_selection",
        ];
        for_each_option(args, ORDER, |key, value| {
            match key {
                "max_states" => self.max_states = parse(value)?,
                "max_iterations" => self.max_iterations = parse(value)?,
                "max_time" => {
                    let seconds = f64::from_option_value(value)?;
                    self.max_time = if seconds.is_infinite() {
                        None
                    } else {
                        Some(Duration::try_from_secs_f64(seconds).map_err(|error| {
                            format!("invalid domain-abstraction max_time {seconds}: {error}")
                        })?)
                    };
                }
                "use_wildcard_plans" => self.use_wildcard_plans = parse(value)?,
                "combine_labels" => self.combine_labels = parse(value)?,
                "random_seed" => self.random_seed = parse(value)?,
                "flaw_kind" => self.flaw_kind = parse(value)?,
                "flaw_scope" => self.flaw_scope = parse(value)?,
                "split_kind" => self.split_kind = parse(value)?,
                "split_selection" => self.split_selection = parse(value)?,
                other => {
                    return Err(format!("unknown option `{other}` for `domain_abstraction`"));
                }
            }
            Ok(())
        })
    }
}
