//! Configuration of domain-abstraction CEGAR.

use std::collections::HashSet;
use std::time::Duration;

use anyhow::Result;
use planforge_sas::numeric_task::VariableIndex;

use super::flaws::SplitDirection;
use super::split_selection::{FlawTreatmentVariants, InitSplitMethod};
use crate::evaluation::abstractions::common::cegar::FlawKind;

#[derive(Debug, Clone)]
pub struct CegarConfig {
    pub max_abstraction_size: usize,
    pub max_iterations: usize,
    pub max_time: Option<Duration>,
    pub use_wildcard_plans: bool,
    pub combine_labels: bool,
    pub debug: bool,
    pub random_seed: Option<u64>,
    pub flaw_kind: FlawKind,
    pub flaw_treatment: FlawTreatmentVariants,
    pub init_split_method: InitSplitMethod,
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
    /// How numeric flaw split values are chosen: `Forward` splits next to the
    /// concrete value; `Backward` places splits at the boundary derived
    /// from the regressed-target / required interval. When `None`, the flaw
    /// kind's default ([`FlawKind::default_split_direction`]) is used.
    pub split_direction: Option<SplitDirection>,
    /// Maximum number of comparison-axiom propositional vars this CEGAR run
    /// may refine into its pattern. `None` = unbounded. When the cap is
    /// reached, the refinement loop refuses to introduce additional
    /// comparison-axiom prop vars: `max_refined_single_atom` (and its
    /// siblings) fall through to alternative split candidates (typically
    /// a numeric split on a comparison-axiom dependency). If no eligible
    /// candidates remain, the CEGAR run terminates and the collection
    /// generator starts the next iteration with a different init seed.
    /// Exists to preserve canonical-DA additive-subset diversity (see
    /// `DomainAbstractionCollectionGeneratorMultipleCegarConfig::max_refined_comparison_vars_per_abstraction`
    /// for the longer rationale).
    pub max_refined_comparison_vars_per_abstraction: Option<usize>,
    /// Which of the tied best flaws to refine: `flaw_treatment` scores the
    /// flaws, the best-scoring ones are sorted by a deterministic key, and the
    /// one at `rank % ties` is refined. `None` breaks ties with the random
    /// seed. Treatments that refine every flaw have no tie to break.
    pub flaw_selection_rank: Option<usize>,
}

impl Default for CegarConfig {
    fn default() -> Self {
        Self {
            max_abstraction_size: 100_000,
            max_iterations: usize::MAX,
            max_time: None,
            use_wildcard_plans: false,
            combine_labels: true,
            debug: false,
            random_seed: Some(2011),
            flaw_kind: FlawKind::Progression,
            flaw_treatment: FlawTreatmentVariants::RandomSingleAtom,
            init_split_method: InitSplitMethod::InitValue,
            init_split_var_ids: None,
            blacklisted_prop_var_ids: HashSet::new(),
            blacklisted_numeric_var_ids: HashSet::new(),
            split_direction: None,
            compute_source_sets: true,
            max_refined_comparison_vars_per_abstraction: None,
            flaw_selection_rank: None,
        }
    }
}

impl CegarConfig {
    /// The split direction refinement uses: the configured one, or concrete
    /// splits.
    pub fn effective_split_direction(&self) -> SplitDirection {
        self.split_direction.unwrap_or(SplitDirection::Forward)
    }
}

/// `domain_abstraction(...)` and friends expose a curated subset of the fields:
/// the rest are internal runtime flags the collection generators set
/// themselves, so this is written by hand rather than derived.
impl crate::config::ApplyOptions for CegarConfig {
    fn apply_options(&mut self, args: &[crate::config::ConfigArg]) -> Result<(), String> {
        use crate::config::{ConfigValue, FromOptionValue, for_each_option};

        /// Type-inferring shortcut for `<T as FromOptionValue>::from_option_value`.
        fn parse<T: FromOptionValue>(value: &ConfigValue) -> Result<T, String> {
            T::from_option_value(value)
        }

        const ORDER: &[&str] = &[
            "max_abstraction_size",
            "max_iterations",
            "max_time",
            "use_wildcard_plans",
            "combine_labels",
            "random_seed",
            "flaw_treatment",
            "flaw_kind",
            "init_split_method",
            "split_direction",
        ];
        for_each_option(args, ORDER, |key, value| {
            match key {
                "max_abstraction_size" => self.max_abstraction_size = parse(value)?,
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
                "flaw_treatment" => self.flaw_treatment = parse(value)?,
                "flaw_kind" => self.flaw_kind = parse(value)?,
                "init_split_method" => self.init_split_method = parse(value)?,
                "split_direction" => self.split_direction = parse(value)?,
                other => {
                    return Err(format!("unknown option `{other}` for `domain_abstraction`"));
                }
            }
            Ok(())
        })
    }
}
