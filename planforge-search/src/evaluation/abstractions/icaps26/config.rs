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

impl crate::config::ApplyOptions for Icaps26CartesianConfig {
    fn apply_options(&mut self, args: &[crate::config::ConfigArg]) -> Result<(), String> {
        use crate::config::FromOptionValue;

        let mut seen = HashSet::new();
        for arg in args {
            let key = arg
                .key()
                .ok_or_else(|| "all options for `icaps26_cartesian` must be named".to_string())?;
            if !seen.insert(key) {
                return Err(format!("duplicate option `{key}` for `icaps26_cartesian`"));
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
                            ));
                        }
                    };
                }
                "max_states" => self.max_states = usize::from_option_value(arg.value())?,
                "max_time" => {
                    let seconds = f64::from_option_value(arg.value())?;
                    self.max_time = if seconds.is_infinite() {
                        None
                    } else {
                        Some(Duration::try_from_secs_f64(seconds).map_err(|error| {
                            format!("invalid ICAPS 2026 max_time {seconds}: {error}")
                        })?)
                    };
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
                    return Err(format!("unknown option `{other}`"));
                }
            }
        }
        Ok(())
    }
}
