//! Options of the ICAPS 2026 Cartesian abstraction. Everything else is fixed
//! to the artifact's choices: progression flaws on the first flawed plan,
//! desired-region split candidates, wanted-value numeric splits, and the
//! artifact's stable abstract A* plan.

use super::*;

#[derive(Debug, Clone)]
pub struct Icaps26CartesianConfig {
    pub max_states: usize,
    pub max_time: Option<Duration>,
    pub combine_labels: bool,
    pub compute_source_sets: bool,
    /// Retain the explicit graph required by cost partitioning.
    pub retain_transition_system: bool,
    /// Seed of the artifact's `std::mt19937`, read by `pick=random`.
    pub random_seed: u32,
    pub pick: Icaps26SplitSelection,
    pub debug: bool,
}

impl Default for Icaps26CartesianConfig {
    fn default() -> Self {
        Self {
            max_states: usize::MAX,
            max_time: Some(Duration::from_secs(900)),
            combine_labels: false,
            compute_source_sets: true,
            retain_transition_system: true,
            random_seed: 2011,
            pick: Icaps26SplitSelection::MaxUnwanted,
            debug: false,
        }
    }
}

impl planforge_search::config::ApplyOptions for Icaps26CartesianConfig {
    fn apply_options(
        &mut self,
        args: &[planforge_search::config::ConfigArg],
    ) -> Result<(), planforge_search::config::ConfigError> {
        use planforge_search::config::FromOptionValue;

        let mut seen = HashSet::new();
        for arg in args {
            let key = arg
                .key()
                .ok_or_else(|| "all options for `icaps26_cartesian` must be named".to_string())?;
            if !seen.insert(key) {
                return Err(format!("duplicate option `{key}` for `icaps26_cartesian`").into());
            }
            match key {
                "pick" => {
                    let value = String::from_option_value(arg.value())?;
                    self.pick = match value.to_ascii_lowercase().as_str() {
                        "random" => Icaps26SplitSelection::Random,
                        "min_unwanted" => Icaps26SplitSelection::MinUnwanted,
                        "max_unwanted" => Icaps26SplitSelection::MaxUnwanted,
                        _ => {
                            return Err(format!(
                                "invalid ICAPS 2026 split selector `{value}`; expected random, min_unwanted, or max_unwanted"
                            ).into());
                        }
                    };
                }
                "max_states" => self.max_states = usize::from_option_value(arg.value())?,
                "max_time" => {
                    self.max_time =
                        planforge_search::config::TimeLimit::from_option_value(arg.value())?
                            .duration()?;
                }
                "random_seed" => {
                    let seed = u64::from_option_value(arg.value())?;
                    self.random_seed = u32::try_from(seed).map_err(|_| {
                        format!("ICAPS 2026 random_seed {seed} exceeds the artifact's 32-bit seed")
                    })?;
                }
                "combine_labels" => self.combine_labels = bool::from_option_value(arg.value())?,
                "debug" => self.debug = bool::from_option_value(arg.value())?,
                other => {
                    return Err(format!("unknown option `{other}`").into());
                }
            }
        }
        Ok(())
    }

    fn describe_options(&self, out: &mut Vec<planforge_search::config::OptionDoc>) {
        use planforge_search::config::{FromOptionValue, TimeLimit};

        let pick = match self.pick {
            Icaps26SplitSelection::Random => "random",
            Icaps26SplitSelection::MinUnwanted => "min_unwanted",
            Icaps26SplitSelection::MaxUnwanted => "max_unwanted",
        };
        for (name, default, doc) in [
            (
                "pick",
                pick.to_string(),
                "split selector: random, min_unwanted or max_unwanted",
            ),
            ("max_states", self.max_states.option_text(), ""),
            ("max_time", TimeLimit::from(self.max_time).option_text(), ""),
            (
                "random_seed",
                u64::from(self.random_seed).option_text(),
                "seed of the artifact's std::mt19937, read by pick=random",
            ),
            ("combine_labels", self.combine_labels.option_text(), ""),
            ("debug", self.debug.option_text(), ""),
        ] {
            out.push(planforge_search::config::OptionDoc {
                name: name.to_string(),
                positional: false,
                default,
                doc: doc.to_string(),
            });
        }
    }
}
