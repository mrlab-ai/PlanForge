use serde::{Deserialize, Serialize};
use std::fmt;

use nom::{
    IResult,
    branch::alt,
    bytes::complete::{tag_no_case, take_while1},
    character::complete::{char, multispace0, one_of},
    combinator::{all_consuming, cut, map, map_res, opt},
    error::{VerboseError, convert_error},
    multi::separated_list0,
    sequence::{delimited, terminated, tuple},
};

use planners_search::numeric::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::{
    DomainAbstractionCollectionGeneratorMultipleCegarConfig, ExecEntirePlanMode,
    FlawTreatment, InitSplitMethod, InitSplitQuantity, NumericSplitStrategy, VariableSubset,
};
use planners_search::numeric::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::LmCutNumericConfig;
use planners_search::numeric::evaluation::pattern_databases::canonical_pdb_heuristic::CanonicalNumericPdbConfig;
use planners_search::numeric::evaluation::pattern_databases::pattern_database::PdbInternalHeuristic;
use planners_search::numeric::evaluation::pattern_databases::pattern_generator_greedy::GreedyPatternGeneratorConfig;
use planners_search::numeric::evaluation::pattern_databases::variable_order_finder::GreedyVariableOrderType;

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
pub struct DomainAbstractionConfig {
    pub max_abstraction_size: usize,
    pub use_wildcard_plans: bool,
    pub combine_labels: bool,
    pub random_seed: i32,
    pub flaw_treatment: FlawTreatment,
    pub init_split_method: InitSplitMethod,
    pub exec_entire_plan: ExecEntirePlanMode,
}

impl Default for DomainAbstractionConfig {
    fn default() -> Self {
        Self {
            max_abstraction_size: i64::MAX as usize,
            use_wildcard_plans: true,
            combine_labels: false,
            random_seed: -1,
            flaw_treatment: FlawTreatment::RandomSingleAtom,
            init_split_method: InitSplitMethod::InitValue,
            exec_entire_plan: ExecEntirePlanMode::StopAtFirstFlaw,
        }
    }
}

impl fmt::Display for DomainAbstractionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            concat!(
                "max_abstraction_size={}, ",
                "use_wildcard_plans={}, ",
                "combine_labels={}, ",
                "random_seed={}, ",
                "flaw_treatment={}, ",
                "init_split_method={}, ",
                "exec_entire_plan={}"
            ),
            self.max_abstraction_size,
            self.use_wildcard_plans,
            self.combine_labels,
            self.random_seed,
            self.flaw_treatment,
            self.init_split_method,
            self.exec_entire_plan,
        )
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum HeuristicSpec {
    Blind,
    #[serde(rename = "canonical_domain_abstractions")]
    CanonicalDomainAbstractions(DomainAbstractionCollectionGeneratorMultipleCegarConfig),
    #[serde(rename = "domain_abstraction")]
    DomainAbstraction(DomainAbstractionConfig),
    #[serde(rename = "canonical_numeric_pdb")]
    CanonicalNumericPdb(CanonicalNumericPdbConfig),
    #[serde(rename = "greedy_numeric_pdb")]
    GreedyNumericPdb(GreedyPatternGeneratorConfig),
    #[serde(rename = "lmcutnumeric")]
    Lmcutnumeric(LmCutNumericConfig),
    #[serde(rename = "multi_domain_abstractions")]
    MultiDomainAbstractions(DomainAbstractionCollectionGeneratorMultipleCegarConfig),
}

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SearchSpec {
    Astar(HeuristicSpec),
    #[serde(rename = "da_debug")]
    DaDebug,
    #[serde(rename = "astar_da_debug")]
    AstarDaDebug,
}

impl fmt::Display for HeuristicSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            HeuristicSpec::Blind => write!(f, "blind()"),
            HeuristicSpec::CanonicalDomainAbstractions(config) => {
                if *config == DomainAbstractionCollectionGeneratorMultipleCegarConfig::default() {
                    write!(f, "canonical_domain_abstractions()")
                } else {
                    write!(f, "canonical_domain_abstractions({config})")
                }
            }
            HeuristicSpec::DomainAbstraction(config) => {
                if *config == DomainAbstractionConfig::default() {
                    write!(f, "domain_abstraction()")
                } else {
                    write!(f, "domain_abstraction({config})")
                }
            }
            HeuristicSpec::CanonicalNumericPdb(config) => {
                if *config == CanonicalNumericPdbConfig::default() {
                    write!(f, "canonical_numeric_pdb()")
                } else {
                    write!(f, "canonical_numeric_pdb({config})")
                }
            }
            HeuristicSpec::GreedyNumericPdb(config) => {
                if *config == GreedyPatternGeneratorConfig::default() {
                    write!(f, "greedy_numeric_pdb()")
                } else {
                    write!(f, "greedy_numeric_pdb({config})")
                }
            }
            HeuristicSpec::Lmcutnumeric(config) => {
                if *config == LmCutNumericConfig::default() {
                    write!(f, "lmcutnumeric()")
                } else {
                    write!(f, "lmcutnumeric({config})")
                }
            }
            HeuristicSpec::MultiDomainAbstractions(config) => {
                write!(f, "multi_domain_abstractions({config})")
            }
        }
    }
}

impl fmt::Display for SearchSpec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SearchSpec::Astar(h) => write!(f, "astar({h})"),
            SearchSpec::DaDebug => write!(f, "da_debug()"),
            SearchSpec::AstarDaDebug => write!(f, "astar_da_debug()"),
        }
    }
}

pub fn parse_search_spec(raw: &str) -> Result<SearchSpec, String> {
    let input = raw;
    match all_consuming(ws(terminated(search_spec, opt(ws(one_of(".;"))))))(input) {
        Ok((_, spec)) => Ok(spec),
        Err(nom::Err::Error(e)) | Err(nom::Err::Failure(e)) => Err(format!(
            "Invalid --search config:\n{}",
            convert_error(input, e)
        )),
        Err(nom::Err::Incomplete(_)) => Err("Invalid --search config: incomplete input".into()),
    }
}

type Res<'a, T> = IResult<&'a str, T, VerboseError<&'a str>>;

fn ws<'a, F, O>(inner: F) -> impl FnMut(&'a str) -> Res<'a, O>
where
    F: FnMut(&'a str) -> Res<'a, O>,
{
    delimited(multispace0, inner, multispace0)
}

fn empty_parens(input: &str) -> Res<'_, ()> {
    map(delimited(ws(char('(')), multispace0, ws(char(')'))), |_| ())(input)
}

fn scalar_value(input: &str) -> Res<'_, String> {
    map(take_while1(|c: char| c != ',' && c != ')'), |raw: &str| {
        raw.trim().to_string()
    })(input)
}

fn identifier(input: &str) -> Res<'_, String> {
    map(
        take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_'),
        str::to_string,
    )(input)
}

fn key_value_argument(input: &str) -> Res<'_, (String, String)> {
    tuple((ws(identifier), ws(char('=')), ws(scalar_value)))(input)
        .map(|(next, (key, _, value))| (next, (key, value)))
}

fn parse_bool(value: &str) -> Result<bool, String> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(format!("expected boolean, got `{value}`")),
    }
}

fn parse_usize(value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .map_err(|_| format!("expected non-negative integer, got `{value}`"))
}

fn parse_i32(value: &str) -> Result<i32, String> {
    value
        .parse::<i32>()
        .map_err(|_| format!("expected integer, got `{value}`"))
}

fn parse_u64(value: &str) -> Result<u64, String> {
    value
        .parse::<u64>()
        .map_err(|_| format!("expected non-negative integer, got `{value}`"))
}

fn parse_greedy_variable_order_type(value: &str) -> Result<GreedyVariableOrderType, String> {
    match value {
        "cg_goal_level" => Ok(GreedyVariableOrderType::CgGoalLevel),
        "cg_goal_random" => Ok(GreedyVariableOrderType::CgGoalRandom),
        "goal_cg_level" => Ok(GreedyVariableOrderType::GoalCgLevel),
        _ => Err(format!("invalid GreedyVariableOrderType `{value}`")),
    }
}

fn parse_pdb_internal_heuristic(value: &str) -> Result<PdbInternalHeuristic, String> {
    match value {
        "blind" => Ok(PdbInternalHeuristic::Blind),
        "lmcut" => Ok(PdbInternalHeuristic::Lmcut),
        _ => Err(format!("invalid PdbInternalHeuristic `{value}`")),
    }
}

fn parse_f64_or_infinity(value: &str) -> Result<f64, String> {
    if value.eq_ignore_ascii_case("infinity") {
        Ok(f64::INFINITY)
    } else {
        value
            .parse::<f64>()
            .map_err(|_| format!("expected float or infinity, got `{value}`"))
    }
}

fn build_lmcutnumeric_config(args: Vec<(String, String)>) -> Result<LmCutNumericConfig, String> {
    let mut config = LmCutNumericConfig::default();
    let mut seen = std::collections::BTreeSet::new();

    for (key, value) in args {
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate option `{key}`"));
        }

        match key.as_str() {
            "ceiling_less_than_one" => config.ceiling_less_than_one = parse_bool(&value)?,
            "ignore_numeric" => config.ignore_numeric = parse_bool(&value)?,
            "random_pcf" => config.random_pcf = parse_bool(&value)?,
            "irmax" => config.irmax = parse_bool(&value)?,
            "disable_ma" => config.disable_ma = parse_bool(&value)?,
            "use_second_order_simple" => config.use_second_order_simple = parse_bool(&value)?,
            "use_constant_assignment" => config.use_constant_assignment = parse_bool(&value)?,
            "bound_iterations" => config.bound_iterations = parse_usize(&value)?,
            "precision" => config.precision = parse_f64_or_infinity(&value)?,
            "epsilon" => config.epsilon = parse_f64_or_infinity(&value)?,
            _ => return Err(format!("unknown option `{key}`")),
        }
    }

    Ok(config)
}

fn parse_variable_subset(value: &str) -> Result<VariableSubset, String> {
    match value {
        "goals" => Ok(VariableSubset::Goals),
        "non_goals" => Ok(VariableSubset::NonGoals),
        "all" => Ok(VariableSubset::All),
        _ => Err(format!("invalid VariableSubset `{value}`")),
    }
}

fn parse_init_split_quantity(value: &str) -> Result<InitSplitQuantity, String> {
    match value {
        "none" => Ok(InitSplitQuantity::None),
        "single" => Ok(InitSplitQuantity::Single),
        "all" => Ok(InitSplitQuantity::All),
        _ => Err(format!("invalid InitSplitQuantity `{value}`")),
    }
}

fn parse_flaw_treatment(value: &str) -> Result<FlawTreatment, String> {
    match value {
        "random_single_atom" => Ok(FlawTreatment::RandomSingleAtom),
        "one_split_per_atom" => Ok(FlawTreatment::OneSplitPerAtom),
        "one_split_per_variable" => Ok(FlawTreatment::OneSplitPerVariable),
        "max_refined_single_atom" => Ok(FlawTreatment::MaxRefinedSingleAtom),
        _ => Err(format!("invalid FlawTreatment `{value}`")),
    }
}

fn parse_init_split_method(value: &str) -> Result<InitSplitMethod, String> {
    match value {
        "goal_value" => Ok(InitSplitMethod::GoalValue),
        "goal_value_or_random_if_non_goal" => Ok(InitSplitMethod::GoalValueOrRandomIfNonGoal),
        "init_value" => Ok(InitSplitMethod::InitValue),
        "random_value" => Ok(InitSplitMethod::RandomValue),
        "random_partition" => Ok(InitSplitMethod::RandomPartition),
        "random_binary_partition_separating_init_goal" => {
            Ok(InitSplitMethod::RandomBinaryPartitionSeparatingInitGoal)
        }
        "identity" => Ok(InitSplitMethod::Identity),
        _ => Err(format!("invalid InitSplitMethod `{value}`")),
    }
}

fn parse_numeric_split_strategy(value: &str) -> Result<NumericSplitStrategy, String> {
    match value {
        "standard" => Ok(NumericSplitStrategy::Standard),
        "exclusion" => Ok(NumericSplitStrategy::Exclusion),
        _ => Err(format!("invalid NumericSplitStrategy `{value}`")),
    }
}

fn parse_exec_entire_plan_mode(value: &str) -> Result<ExecEntirePlanMode, String> {
    match value {
        "stop_at_first_flaw" => Ok(ExecEntirePlanMode::StopAtFirstFlaw),
        "execute_entire_plan" => Ok(ExecEntirePlanMode::ExecuteEntirePlan),
        _ => Err(format!("invalid ExecEntirePlanMode `{value}`")),
    }
}

fn build_domain_abstraction_config(
    args: Vec<(String, String)>,
) -> Result<DomainAbstractionConfig, String> {
    let mut config = DomainAbstractionConfig::default();
    let mut seen = std::collections::BTreeSet::new();

    for (key, value) in args {
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate option `{key}`"));
        }

        match key.as_str() {
            "max_abstraction_size" => config.max_abstraction_size = parse_usize(&value)?,
            "use_wildcard_plans" => config.use_wildcard_plans = parse_bool(&value)?,
            "combine_labels" => config.combine_labels = parse_bool(&value)?,
            "random_seed" => config.random_seed = parse_i32(&value)?,
            "flaw_treatment" => config.flaw_treatment = parse_flaw_treatment(&value)?,
            "init_split_method" => config.init_split_method = parse_init_split_method(&value)?,
            "exec_entire_plan" => config.exec_entire_plan = parse_exec_entire_plan_mode(&value)?,
            _ => return Err(format!("unknown option `{key}`")),
        }
    }

    Ok(config)
}

fn build_multi_domain_abstractions_config(
    args: Vec<(String, String)>,
) -> Result<DomainAbstractionCollectionGeneratorMultipleCegarConfig, String> {
    let mut config = DomainAbstractionCollectionGeneratorMultipleCegarConfig::default();
    let mut seen = std::collections::BTreeSet::new();

    for (key, value) in args {
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate option `{key}`"));
        }

        match key.as_str() {
            "max_abstraction_size" => config.max_abstraction_size = parse_usize(&value)?,
            "max_collection_size" => config.max_collection_size = parse_usize(&value)?,
            "abstraction_generation_max_time" => {
                config.abstraction_generation_max_time = parse_f64_or_infinity(&value)?
            }
            "total_max_time" => config.total_max_time = parse_f64_or_infinity(&value)?,
            "stagnation_limit" => config.stagnation_limit = parse_f64_or_infinity(&value)?,
            "blacklist_trigger_percentage" => {
                config.blacklist_trigger_percentage = parse_f64_or_infinity(&value)?
            }
            "enable_blacklist_on_stagnation" => {
                config.enable_blacklist_on_stagnation = parse_bool(&value)?
            }
            "blacklist_option" => config.blacklist_option = parse_variable_subset(&value)?,
            "init_split_candidates" => {
                config.init_split_candidates = parse_variable_subset(&value)?
            }
            "init_split_quantity" => {
                config.init_split_quantity = parse_init_split_quantity(&value)?
            }
            "random_seed" => config.random_seed = parse_i32(&value)?,
            "use_wildcard_plans" => config.use_wildcard_plans = parse_bool(&value)?,
            "combine_labels" => config.combine_labels = parse_bool(&value)?,
            "deviation_flaws" => config.deviation_flaws = parse_bool(&value)?,
            "flaw_treatment" => config.flaw_treatment = parse_flaw_treatment(&value)?,
            "init_split_method" => config.init_split_method = parse_init_split_method(&value)?,
            "numeric_split_strategy" => {
                config.numeric_split_strategy = parse_numeric_split_strategy(&value)?
            }
            "exec_entire_plan" => config.exec_entire_plan = parse_exec_entire_plan_mode(&value)?,
            _ => return Err(format!("unknown option `{key}`")),
        }
    }

    Ok(config)
}

fn build_greedy_numeric_pdb_config(
    args: Vec<(String, String)>,
) -> Result<GreedyPatternGeneratorConfig, String> {
    let mut config = GreedyPatternGeneratorConfig::default();
    let mut seen = std::collections::BTreeSet::new();

    for (key, value) in args {
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate option `{key}`"));
        }

        match key.as_str() {
            "max_pdb_states" => config.max_pdb_states = parse_usize(&value)?,
            "numeric_first" => config.numeric_first = parse_bool(&value)?,
            "random_seed" => config.random_seed = parse_u64(&value)?,
            "variable_order_type" => {
                config.variable_order_type = parse_greedy_variable_order_type(&value)?
            }
            "exploration_heuristic" => {
                config.exploration_heuristic = parse_pdb_internal_heuristic(&value)?
            }
            "frontier_heuristic" => {
                config.frontier_heuristic = parse_pdb_internal_heuristic(&value)?
            }
            "failed_lookup_heuristic" => {
                config.failed_lookup_heuristic = parse_pdb_internal_heuristic(&value)?
            }
            _ => return Err(format!("unknown option `{key}`")),
        }
    }

    Ok(config)
}

fn build_canonical_numeric_pdb_config(
    args: Vec<(String, String)>,
) -> Result<CanonicalNumericPdbConfig, String> {
    let mut config = CanonicalNumericPdbConfig::default();
    let mut seen = std::collections::BTreeSet::new();

    for (key, value) in args {
        if !seen.insert(key.clone()) {
            return Err(format!("duplicate option `{key}`"));
        }

        match key.as_str() {
            "max_pdb_states" => config.max_pdb_states = parse_usize(&value)?,
            "max_pattern_size" => config.max_pattern_size = parse_usize(&value)?,
            "only_interesting_patterns" => config.only_interesting_patterns = parse_bool(&value)?,
            "random_seed" => config.random_seed = parse_i32(&value)?,
            "variable_order_type" => {
                config.variable_order_type = parse_greedy_variable_order_type(&value)?
            }
            "exploration_heuristic" => {
                config.exploration_heuristic = parse_pdb_internal_heuristic(&value)?
            }
            "frontier_heuristic" => {
                config.frontier_heuristic = parse_pdb_internal_heuristic(&value)?
            }
            "failed_lookup_heuristic" => {
                config.failed_lookup_heuristic = parse_pdb_internal_heuristic(&value)?
            }
            _ => return Err(format!("unknown option `{key}`")),
        }
    }

    Ok(config)
}

fn multi_domain_abstractions_parens(
    input: &str,
) -> Res<'_, DomainAbstractionCollectionGeneratorMultipleCegarConfig> {
    map_res(
        delimited(
            ws(char('(')),
            terminated(
                separated_list0(ws(char(',')), key_value_argument),
                opt(ws(char(','))),
            ),
            ws(char(')')),
        ),
        build_multi_domain_abstractions_config,
    )(input)
}

fn domain_abstraction_parens(input: &str) -> Res<'_, DomainAbstractionConfig> {
    map_res(
        delimited(
            ws(char('(')),
            terminated(
                separated_list0(ws(char(',')), key_value_argument),
                opt(ws(char(','))),
            ),
            ws(char(')')),
        ),
        build_domain_abstraction_config,
    )(input)
}

fn greedy_numeric_pdb_parens(input: &str) -> Res<'_, GreedyPatternGeneratorConfig> {
    map_res(
        delimited(
            ws(char('(')),
            terminated(
                separated_list0(ws(char(',')), key_value_argument),
                opt(ws(char(','))),
            ),
            ws(char(')')),
        ),
        build_greedy_numeric_pdb_config,
    )(input)
}

fn canonical_numeric_pdb_parens(input: &str) -> Res<'_, CanonicalNumericPdbConfig> {
    map_res(
        delimited(
            ws(char('(')),
            terminated(
                separated_list0(ws(char(',')), key_value_argument),
                opt(ws(char(','))),
            ),
            ws(char(')')),
        ),
        build_canonical_numeric_pdb_config,
    )(input)
}

fn lmcutnumeric_parens(input: &str) -> Res<'_, LmCutNumericConfig> {
    map_res(
        delimited(
            ws(char('(')),
            terminated(
                separated_list0(ws(char(',')), key_value_argument),
                opt(ws(char(','))),
            ),
            ws(char(')')),
        ),
        build_lmcutnumeric_config,
    )(input)
}

fn heuristic_spec(input: &str) -> Res<'_, HeuristicSpec> {
    let blind = map(
        tuple((ws(tag_no_case("blind")), opt(ws(empty_parens)))),
        |_| HeuristicSpec::Blind,
    );

    let domain_abstraction = map(
        tuple((
            ws(tag_no_case("domain_abstraction")),
            opt(ws(alt((
                map(empty_parens, |_| DomainAbstractionConfig::default()),
                domain_abstraction_parens,
            )))),
        )),
        |(_, config)| HeuristicSpec::DomainAbstraction(config.unwrap_or_default()),
    );

    let canonical_domain_abstractions = map(
        tuple((
            ws(tag_no_case("canonical_domain_abstractions")),
            opt(ws(alt((
                map(
                    empty_parens,
                    |_| DomainAbstractionCollectionGeneratorMultipleCegarConfig::default(),
                ),
                multi_domain_abstractions_parens,
            )))),
        )),
        |(_, config)| HeuristicSpec::CanonicalDomainAbstractions(config.unwrap_or_default()),
    );

    let canonical_numeric_pdb = map(
        tuple((
            ws(tag_no_case("canonical_numeric_pdb")),
            opt(ws(alt((
                map(empty_parens, |_| CanonicalNumericPdbConfig::default()),
                canonical_numeric_pdb_parens,
            )))),
        )),
        |(_, config)| HeuristicSpec::CanonicalNumericPdb(config.unwrap_or_default()),
    );

    let greedy_numeric_pdb = map(
        tuple((
            ws(tag_no_case("greedy_numeric_pdb")),
            opt(ws(alt((
                map(empty_parens, |_| GreedyPatternGeneratorConfig::default()),
                greedy_numeric_pdb_parens,
            )))),
        )),
        |(_, config)| HeuristicSpec::GreedyNumericPdb(config.unwrap_or_default()),
    );

    let lmcutnumeric = map(
        tuple((
            ws(tag_no_case("lmcutnumeric")),
            opt(ws(alt((
                map(empty_parens, |_| LmCutNumericConfig::default()),
                lmcutnumeric_parens,
            )))),
        )),
        |(_, config)| HeuristicSpec::Lmcutnumeric(config.unwrap_or_default()),
    );

    let multi_domain_abstractions = map(
        tuple((
            ws(tag_no_case("multi_domain_abstractions")),
            opt(ws(multi_domain_abstractions_parens)),
        )),
        |(_, config)| HeuristicSpec::MultiDomainAbstractions(config.unwrap_or_default()),
    );

    ws(alt((
        canonical_domain_abstractions,
        multi_domain_abstractions,
        lmcutnumeric,
        canonical_numeric_pdb,
        greedy_numeric_pdb,
        domain_abstraction,
        blind,
    )))(input)
}

fn search_spec(input: &str) -> Res<'_, SearchSpec> {
    let da_debug = map(
        tuple((ws(tag_no_case("da_debug")), opt(ws(empty_parens)))),
        |_| SearchSpec::DaDebug,
    );

    let astar_da_debug = map(
        tuple((ws(tag_no_case("astar_da_debug")), opt(ws(empty_parens)))),
        |_| SearchSpec::AstarDaDebug,
    );

    let astar = map(
        tuple((
            ws(tag_no_case("astar")),
            ws(char('(')),
            cut(heuristic_spec),
            ws(char(')')),
        )),
        |(_, _, h, _)| SearchSpec::Astar(h),
    );
    ws(alt((astar, astar_da_debug, da_debug)))(input)
}

#[cfg(test)]
mod tests;
