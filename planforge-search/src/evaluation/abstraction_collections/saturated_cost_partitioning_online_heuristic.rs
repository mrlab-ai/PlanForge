use std::cell::{Ref, RefCell};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::time::{Duration, Instant};

use planforge_sas::numeric_task::{
    AbstractNumericTask, TaskRef, metric_operator_cost_from_initial_values,
};
use planforge_sas::state_registry::{ConcreteState, ExpansionContext, StateRegistry};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use serde::{Deserialize, Serialize};
use tracing::{Level, debug, enabled, info};

use crate::evaluation::cartesian_abstractions::{
    CartesianAbstraction, CartesianAbstractionHeuristic, CartesianRefinementDirection,
    CartesianRefinementHierarchy,
};
use crate::evaluation::evaluator::{EvaluationError, EvaluationState};
use crate::evaluation::heuristic::Heuristic;
use crate::evaluation::numeric_landmarks::lm_cut_numeric_heuristic::{
    LandmarkCutNumericHeuristic, LmCutNumericConfig,
};
use crate::evaluation::pattern_databases::pattern_database::{
    PatternDatabase, PdbHeuristicConfig, PdbInternalHeuristic,
};
use crate::successor_generator::SuccessorTree;

use crate::evaluation::domain_abstractions::abstract_operator_generator::AbstractOperator;
use crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::DomainAbstractionCollectionGeneratorMultipleCegarConfig;
use crate::evaluation::domain_abstractions::domain_abstraction_factory::AbstractDistanceTable;
use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstraction;
use crate::evaluation::domain_abstractions::domain_abstraction_heuristic::{
    DomainAbstractionHeuristic, DomainAbstractionLookupScratch,
    compute_collection_abstract_state_ids,
};
use super::transition_cost_partitioning::{
    AbstractOperatorCostFunction, AbstractOperatorFootprint, LmCutResidualOperatorCostPartition,
    TransitionResidualCosts,
    build_explicit_abstract_operator_cost_partitioning_table,
    build_explicit_label_cost_partitioning_table,
};
use super::component::AbstractionComponent;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScoringFunction {
    MaxHeuristic,
    MinStolenCosts,
    MaxHeuristicPerStolenCosts,
}

impl fmt::Display for ScoringFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScoringFunction::MaxHeuristic => write!(f, "max_heuristic"),
            ScoringFunction::MinStolenCosts => write!(f, "min_stolen_costs"),
            ScoringFunction::MaxHeuristicPerStolenCosts => {
                write!(f, "max_heuristic_per_stolen_costs")
            }
        }
    }
}

impl crate::config::sealed::Sealed for ScoringFunction {}

impl crate::config::FromOptionValue for ScoringFunction {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "max_heuristic" => Ok(Self::MaxHeuristic),
            "min_stolen_costs" => Ok(Self::MinStolenCosts),
            "max_heuristic_per_stolen_costs" => Ok(Self::MaxHeuristicPerStolenCosts),
            other => Err(format!("invalid ScoringFunction `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OrderGenerator {
    Greedy,
    DynamicGreedy,
    Random,
    Diverse,
}

impl fmt::Display for OrderGenerator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OrderGenerator::Greedy => write!(f, "greedy_orders"),
            OrderGenerator::DynamicGreedy => write!(f, "dynamic_greedy_orders"),
            OrderGenerator::Random => write!(f, "random_orders"),
            OrderGenerator::Diverse => write!(f, "diverse_orders"),
        }
    }
}

impl crate::config::sealed::Sealed for OrderGenerator {}

impl crate::config::FromOptionValue for OrderGenerator {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "greedy_orders" | "greedy_orders()" => Ok(Self::Greedy),
            "dynamic_greedy_orders" | "dynamic_greedy_orders()" => Ok(Self::DynamicGreedy),
            "random_orders" | "random_orders()" => Ok(Self::Random),
            "diverse_orders" | "diverse_orders()" => Ok(Self::Diverse),
            other => Err(format!("invalid OrderGenerator `{other}`")),
        }
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Saturator {
    All,
    Perim,
    Perimstar,
}

impl fmt::Display for Saturator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Saturator::All => write!(f, "all"),
            Saturator::Perim => write!(f, "perim"),
            Saturator::Perimstar => write!(f, "perimstar"),
        }
    }
}

impl crate::config::sealed::Sealed for Saturator {}

impl crate::config::FromOptionValue for Saturator {
    fn from_option_value(value: &crate::config::ConfigValue) -> Result<Self, String> {
        match crate::config::atom(value)? {
            "all" => Ok(Self::All),
            "perim" => Ok(Self::Perim),
            "perimstar" => Ok(Self::Perimstar),
            other => Err(format!("invalid Saturator `{other}`")),
        }
    }
}

#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct ScpOnlineConfig {
    /// Whether to rebuild cost partitions during search. When false, all cost
    /// partitions are built before search and construction-only abstraction
    /// data is released immediately afterwards.
    pub online: bool,
    pub max_time: f64,
    pub table_construction_max_time: f64,
    pub max_size: usize,
    /// Build a Scorpion-style offline portfolio from random-walk samples.
    /// This option is only valid with `online=false`.
    pub diversify: bool,
    /// Number of reachable concrete states used to judge whether a candidate
    /// cost partition adds value to the offline portfolio.
    pub samples: usize,
    /// Maximum number of diversified cost partitions retained offline.
    pub max_orders: usize,
    pub interval: usize,
    /// Mirrored into `collection_config.combine_labels` so `combine_labels=true`
    /// sets both. To set them independently, use the nested `collection=…`
    /// form: `scp_online(combine_labels=true, collection=…(combine_labels=false))`.
    #[option(also_sets = "collection_config.combine_labels")]
    pub combine_labels: bool,
    /// Catch-all: flat collection keys (`scp_online(max_collection_size=…)`)
    /// route here. Explicit `collection=multi_domain_abstractions(…)` form
    /// also routes here via the `nested` arm.
    #[option(flatten, nested = "collection")]
    pub collection_config: DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    pub use_numeric_pdbs: bool,
    pub max_pdb_states: usize,
    pub max_pattern_size: usize,
    pub only_interesting_patterns: bool,
    pub pdb_exploration_heuristic: PdbInternalHeuristic,
    pub pdb_frontier_heuristic: PdbInternalHeuristic,
    pub pdb_failed_lookup_heuristic: PdbInternalHeuristic,
    pub scoring_function: ScoringFunction,
    #[option(rename = "orders")]
    pub order_generator: OrderGenerator,
    /// Time reserved for the bounded initial order portfolio when offline
    /// diversification is enabled. This is independent of hill climbing.
    pub initial_order_generation_max_time: f64,
    pub order_optimization_max_time: f64,
    pub saturator: Saturator,
    /// Additional traversals over the same abstraction order using the
    /// remaining regional transition costs.
    pub residual_sweeps: usize,
    #[option(also_sets = "collection_config.random_seed")]
    pub random_seed: Option<u64>,
    pub use_abstract_operator_cost_partitioning: bool,
}

#[derive(
    Debug, Clone, Deserialize, Serialize, PartialEq, planforge_search::config::ApplyOptions,
)]
pub struct FillScpConfig {
    pub table_construction_max_time: f64,
    #[option(also_sets = "collection_config.combine_labels")]
    pub combine_labels: bool,
    #[option(flatten, nested = "collection")]
    pub collection_config: DomainAbstractionCollectionGeneratorMultipleCegarConfig,
    pub scoring_function: ScoringFunction,
    #[option(rename = "orders")]
    pub order_generator: OrderGenerator,
    pub order_optimization_max_time: f64,
    pub saturator: Saturator,
    #[option(also_sets = "collection_config.random_seed")]
    pub random_seed: Option<u64>,
    pub use_abstract_operator_cost_partitioning: bool,
    /// Flattened so `precision`, `epsilon`, etc. reach the nested LMcut config.
    /// SCP/fillSCP both flatten collection_config, but this `flatten` only
    /// applies if `collection_config` does not — only one flatten per struct.
    /// Here we use `nested = "lmcut"` instead, plus per-key forwarding via the
    /// hand-written wrapper (see `apply_fill_scp_options`). Actually — since
    /// `collection_config` is the catch-all, the LMcut fields must be named
    /// explicitly. `nested = "lmcut"` lets `lmcut=lmcutnumeric(precision=…)`
    /// work cleanly.
    #[option(nested = "lmcut")]
    pub lmcut_config: LmCutNumericConfig,
}

impl Default for FillScpConfig {
    fn default() -> Self {
        let collection_config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            combine_labels: false,
            portfolio_strategy:
                crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::PortfolioStrategy::Standard,
            ..Default::default()
        };
        let random_seed = collection_config.random_seed;
        Self {
            table_construction_max_time: 30.0,
            combine_labels: false,
            collection_config,
            scoring_function: ScoringFunction::MaxHeuristicPerStolenCosts,
            order_generator: OrderGenerator::Greedy,
            order_optimization_max_time: 5.0,
            saturator: Saturator::All,
            random_seed,
            use_abstract_operator_cost_partitioning: false,
            lmcut_config: LmCutNumericConfig::default(),
        }
    }
}

impl FillScpConfig {
    pub fn force_full_goal_tasks(&mut self) {
        self.collection_config.portfolio_strategy =
            crate::evaluation::domain_abstractions::domain_abstraction_collection_generator_multiple_cegar::PortfolioStrategy::Standard;
        self.collection_config.combine_labels = self.combine_labels;
        self.random_seed = self.collection_config.random_seed;
        // Label-mode fillSCP only consumes per-abstraction distance tables — it never
        // touches `ConcreteOperatorFootprint`. Building those footprints during CEGAR
        // is pure memory bloat (the same per-concrete-op `StateRegion` cost that
        // canonical/max already skip via 468f06a). Disable it unconditionally for
        // the label-CP path.
        if !self.use_abstract_operator_cost_partitioning {
            self.collection_config.compute_operator_footprints = false;
        }
    }

    fn as_scp_online_config(&self) -> ScpOnlineConfig {
        ScpOnlineConfig {
            online: false,
            max_time: 0.0,
            table_construction_max_time: self.table_construction_max_time,
            max_size: usize::MAX,
            diversify: false,
            samples: 1_000,
            max_orders: usize::MAX,
            interval: usize::MAX,
            combine_labels: self.combine_labels,
            collection_config: self.collection_config.clone(),
            use_numeric_pdbs: false,
            max_pdb_states: 0,
            max_pattern_size: 0,
            only_interesting_patterns: true,
            pdb_exploration_heuristic: PdbInternalHeuristic::Blind,
            pdb_frontier_heuristic: PdbInternalHeuristic::Zero,
            pdb_failed_lookup_heuristic: PdbInternalHeuristic::Zero,
            scoring_function: self.scoring_function,
            order_generator: self.order_generator,
            initial_order_generation_max_time: 10.0,
            order_optimization_max_time: self.order_optimization_max_time,
            saturator: self.saturator,
            residual_sweeps: 0,
            random_seed: self.random_seed,
            use_abstract_operator_cost_partitioning: self.use_abstract_operator_cost_partitioning,
        }
    }
}

impl Default for ScpOnlineConfig {
    fn default() -> Self {
        let collection_config = DomainAbstractionCollectionGeneratorMultipleCegarConfig {
            combine_labels: false,
            ..Default::default()
        };
        let random_seed = collection_config.random_seed;
        Self {
            online: true,
            max_time: 200.0,
            table_construction_max_time: 30.0,
            max_size: usize::MAX,
            diversify: false,
            samples: 1_000,
            max_orders: usize::MAX,
            // Default: build the SCP heuristic once at evaluation 0 and never
            // rebuild during search. Periodic rebuilds proved expensive enough
            // to dominate per-state cost on label-SCP and abstract-op CP alike.
            // Configure a finite `interval` only when targeted state-specific
            // re-orderings are worth the rebuild time (rarely, in practice).
            interval: usize::MAX,
            combine_labels: false,
            collection_config,
            use_numeric_pdbs: false,
            max_pdb_states: 50_000,
            max_pattern_size: 2,
            only_interesting_patterns: true,
            pdb_exploration_heuristic: PdbInternalHeuristic::Blind,
            pdb_frontier_heuristic: PdbInternalHeuristic::Zero,
            pdb_failed_lookup_heuristic: PdbInternalHeuristic::Zero,
            scoring_function: ScoringFunction::MaxHeuristicPerStolenCosts,
            order_generator: OrderGenerator::Greedy,
            initial_order_generation_max_time: 10.0,
            // Improve the best initial order after constructing the bounded
            // initial candidate portfolio. The table-construction deadline
            // still bounds the complete preprocessing phase.
            order_optimization_max_time: 5.0,
            saturator: Saturator::All,
            residual_sweeps: 0,
            random_seed,
            use_abstract_operator_cost_partitioning: false,
        }
    }
}

impl ScpOnlineConfig {
    pub fn pdb_heuristic_config(&self) -> PdbHeuristicConfig {
        PdbHeuristicConfig {
            exploration_heuristic: self.pdb_exploration_heuristic,
            frontier_heuristic: self.pdb_frontier_heuristic,
            failed_lookup_heuristic: self.pdb_failed_lookup_heuristic,
        }
    }
}

// ---------------------------------------------------------------------------
// Lookup tables and CP heuristic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct LookupTable {
    abstraction_id: usize,
    distances: Vec<f64>,
    unknown_value: f64,
}

#[derive(Debug, Clone, Default, PartialEq)]
struct CostPartitioningHeuristic {
    lookup_tables: Vec<LookupTable>,
    specialist_goal_id: Option<usize>,
}

struct CandidateCostPartitions {
    partitions: Vec<CostPartitioningHeuristic>,
    best_index: usize,
}

impl CostPartitioningHeuristic {
    fn is_empty(&self) -> bool {
        self.lookup_tables.is_empty()
    }

    fn add_h_values(&mut self, abstraction_id: usize, distances: Vec<f64>) {
        if distances.iter().any(|value| *value > 0.0) {
            self.lookup_tables.push(LookupTable {
                abstraction_id,
                distances,
                unknown_value: f64::INFINITY,
            });
        }
    }

    fn add_pdb_h_values(&mut self, abstraction_id: usize, distances: Vec<f64>) {
        if distances.iter().any(|value| *value > 0.0) {
            self.lookup_tables.push(LookupTable {
                abstraction_id,
                distances,
                unknown_value: 0.0,
            });
        }
    }

    fn compute_heuristic(&self, abstract_state_ids: &[Option<usize>]) -> f64 {
        let mut sum = 0.0;
        for table in &self.lookup_tables {
            let Some(state_id) = abstract_state_ids
                .get(table.abstraction_id)
                .copied()
                .flatten()
            else {
                sum += table.unknown_value;
                continue;
            };
            let Some(&value) = table.distances.get(state_id) else {
                sum += table.unknown_value;
                continue;
            };
            if value.is_infinite() && value.is_sign_positive() {
                return f64::INFINITY;
            }
            sum += value;
        }
        sum
    }

    fn estimate_size_in_kb(&self) -> usize {
        let values = self
            .lookup_tables
            .iter()
            .map(|table| table.distances.len())
            .sum::<usize>();
        values.saturating_mul(std::mem::size_of::<f64>()) / 1024
    }
}

// ---------------------------------------------------------------------------
// Main heuristic
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct ScpOnlineState {
    start_time: Instant,
    evaluated_states: usize,
    improve_heuristic: bool,
    size_kb: usize,
    cp_heuristics: Vec<CostPartitioningHeuristic>,
    h_values_by_abstraction: Vec<Vec<f64>>,
    stolen_costs_by_abstraction: Vec<f64>,
    rng: SmallRng,
    improvement_ended: bool,
    required_lookup_ids: Vec<usize>,
    offline_sample_ids: Vec<Vec<Option<usize>>>,
}

impl ScpOnlineState {
    fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos() as u64)
                .unwrap_or(0x5C9_0A11_u64)
        });
        Self {
            start_time: Instant::now(),
            evaluated_states: 0,
            improve_heuristic: true,
            size_kb: 0,
            cp_heuristics: Vec::new(),
            h_values_by_abstraction: Vec::new(),
            stolen_costs_by_abstraction: Vec::new(),
            rng: SmallRng::seed_from_u64(seed),
            improvement_ended: false,
            required_lookup_ids: Vec::new(),
            offline_sample_ids: Vec::new(),
        }
    }
}

pub struct SaturatedCostPartitioningOnlineHeuristic<'task> {
    name: String,
    task: &'task dyn AbstractNumericTask,
    abstractions: RefCell<Option<Vec<DomainAbstraction>>>,
    abstraction_heuristics: Vec<DomainAbstractionHeuristic>,
    cartesian_abstractions: RefCell<Option<Vec<CartesianAbstraction>>>,
    cartesian_hierarchies: Vec<CartesianRefinementHierarchy>,
    pdbs: Vec<PatternDatabase<'task>>,
    config: ScpOnlineConfig,
    original_operator_costs: Vec<f64>,
    state: RefCell<ScpOnlineState>,
    lookup_scratch: RefCell<DomainAbstractionLookupScratch>,
    component_ids_scratch: RefCell<Vec<Option<usize>>>,
    prop_scratch: RefCell<Vec<usize>>,
    numeric_scratch: RefCell<Vec<f64>>,
    sampling_task: RefCell<Option<TaskRef<'task>>>,
}

pub struct FillScpHeuristic<'task> {
    name: String,
    abstraction_heuristics: Vec<DomainAbstractionHeuristic>,
    cartesian_heuristics: Vec<CartesianAbstractionHeuristic>,
    cp_heuristic: CostPartitioningHeuristic,
    lmcut_heuristic: LandmarkCutNumericHeuristic<'task>,
    lookup_scratch: RefCell<DomainAbstractionLookupScratch>,
    component_ids_scratch: RefCell<Vec<Option<usize>>>,
}

impl<'task> FillScpHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        config: FillScpConfig,
        task: &'task dyn AbstractNumericTask,
    ) -> Result<Self, EvaluationError> {
        Self::new_with_cartesian(name, abstractions, Vec::new(), config, task)
    }

    pub fn new_with_cartesian(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        cartesian_abstractions: Vec<CartesianAbstraction>,
        mut config: FillScpConfig,
        task: &'task dyn AbstractNumericTask,
    ) -> Result<Self, EvaluationError> {
        config.force_full_goal_tasks();
        let scp_config = config.as_scp_online_config();
        let temp = SaturatedCostPartitioningOnlineHeuristic::new_with_cartesian(
            Some("fillSCP_scp_builder".to_string()),
            abstractions.clone(),
            cartesian_abstractions.clone(),
            Vec::new(),
            scp_config,
            task,
        )?;
        let num_domain_abstractions = abstractions.len();
        let mut abstract_state_ids: Vec<Option<usize>> = abstractions
            .iter()
            .map(|abstraction| Some(abstraction.distance_table.initial_state_hash))
            .collect();
        abstract_state_ids.extend(
            cartesian_abstractions
                .iter()
                .map(|abstraction| Some(abstraction.transition_system.initial_state_hash)),
        );
        let deadline = config
            .table_construction_max_time
            .is_finite()
            .then(|| Instant::now() + Duration::from_secs_f64(config.table_construction_max_time));

        let original_costs = temp.original_operator_costs.clone();
        let mut order = {
            let mut state = temp.state.borrow_mut();
            temp.compute_order_for_state(
                task,
                &mut state,
                &abstract_state_ids,
                &abstractions,
                num_domain_abstractions,
                deadline,
            )?
        };
        let standalone_current_h = {
            let state = temp.state.borrow();
            standalone_current_h_values(&state, &abstract_state_ids, num_domain_abstractions)
        };
        let (mut cp_heuristic, mut residual_costs, mut residual_partitions) =
            if config.use_abstract_operator_cost_partitioning {
                let (cp, costs, partitions) = temp.build_abstract_operator_fill_scp(
                    task,
                    &abstractions,
                    &order,
                    &abstract_state_ids,
                    &standalone_current_h,
                    num_domain_abstractions,
                    &original_costs,
                    deadline,
                    config.saturator,
                )?;
                (cp, costs, Some(partitions))
            } else {
                let (cp, costs) = temp.build_label_fill_scp(
                    task,
                    &abstractions,
                    &order,
                    &abstract_state_ids,
                    num_domain_abstractions,
                    &original_costs,
                    deadline,
                )?;
                (cp, costs, None)
            };
        if config.order_optimization_max_time > 0.0 {
            let optimization_deadline = config.order_optimization_max_time.is_finite().then(|| {
                Instant::now() + Duration::from_secs_f64(config.order_optimization_max_time)
            });
            temp.optimize_order_with_hill_climbing(
                task,
                &abstractions,
                &standalone_current_h,
                num_domain_abstractions,
                &original_costs,
                &abstract_state_ids,
                &mut order,
                &mut cp_heuristic,
                optimization_deadline,
            )?;
            (cp_heuristic, residual_costs, residual_partitions) =
                if config.use_abstract_operator_cost_partitioning {
                    let (cp, costs, partitions) = temp.build_abstract_operator_fill_scp(
                        task,
                        &abstractions,
                        &order,
                        &abstract_state_ids,
                        &standalone_current_h,
                        num_domain_abstractions,
                        &original_costs,
                        deadline,
                        config.saturator,
                    )?;
                    (cp, costs, Some(partitions))
                } else {
                    let (cp, costs) = temp.build_label_fill_scp(
                        task,
                        &abstractions,
                        &order,
                        &abstract_state_ids,
                        num_domain_abstractions,
                        &original_costs,
                        deadline,
                    )?;
                    (cp, costs, None)
                };
        }
        let lmcut_heuristic =
            LandmarkCutNumericHeuristic::from_config_with_residual_operator_cost_partitions(
                task,
                config.lmcut_config,
                residual_partitions.is_none().then_some(residual_costs),
                residual_partitions,
            )
            .map_err(EvaluationError::ComputationFailed)?;
        let abstraction_heuristics = abstractions
            .into_iter()
            .enumerate()
            .map(|(index, mut abstraction)| {
                abstraction.discard_transition_data();
                DomainAbstractionHeuristic::new(Some(format!("fillSCP_{index}")), abstraction)
            })
            .collect();
        let cartesian_heuristics = cartesian_abstractions
            .into_iter()
            .enumerate()
            .map(|(index, mut abstraction)| {
                abstraction.discard_transition_data();
                CartesianAbstractionHeuristic::new(
                    Some(format!("fillSCP_cartesian_{index}")),
                    abstraction,
                )
            })
            .collect();

        Ok(Self {
            name: name.unwrap_or_else(|| "fillSCP".to_string()),
            abstraction_heuristics,
            cartesian_heuristics,
            cp_heuristic,
            lmcut_heuristic,
            lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
            component_ids_scratch: RefCell::new(Vec::new()),
        })
    }

    fn compute_abstract_state_ids_into(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        ids.clear();
        let num_domain = self.abstraction_heuristics.len();
        ids.resize(num_domain + self.cartesian_heuristics.len(), None);
        let mut scratch = self.lookup_scratch.borrow_mut();
        compute_collection_abstract_state_ids(
            &self.abstraction_heuristics,
            eval_state,
            None,
            &mut scratch,
        )?;
        for (id, abstract_id) in scratch.abstract_state_ids.iter().copied().enumerate() {
            ids[id] = abstract_id;
        }
        for (cartesian_id, heuristic) in self.cartesian_heuristics.iter().enumerate() {
            ids[num_domain + cartesian_id] = Some(heuristic.abstract_state_id(eval_state)?);
        }
        Ok(())
    }
}

impl Heuristic for FillScpHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let mut component_ids = self.component_ids_scratch.borrow_mut();
        self.compute_abstract_state_ids_into(eval_state, &mut component_ids)?;
        let cp_h = self.cp_heuristic.compute_heuristic(&component_ids);
        if cp_h.is_infinite() && cp_h.is_sign_positive() {
            return Ok(cp_h);
        }
        let lmcut_h = self.lmcut_heuristic.compute_heuristic(eval_state)?;
        Ok(cp_h + lmcut_h)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }

    fn dead_ends_are_reliable(&self) -> bool {
        true
    }
}

impl<'task> SaturatedCostPartitioningOnlineHeuristic<'task> {
    pub fn new(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        pdbs: Vec<PatternDatabase<'task>>,
        config: ScpOnlineConfig,
        task: &'task dyn AbstractNumericTask,
    ) -> Result<Self, EvaluationError> {
        Self::new_with_cartesian(name, abstractions, Vec::new(), pdbs, config, task)
    }

    pub fn new_with_cartesian(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        cartesian_abstractions: Vec<CartesianAbstraction>,
        pdbs: Vec<PatternDatabase<'task>>,
        config: ScpOnlineConfig,
        task: &'task dyn AbstractNumericTask,
    ) -> Result<Self, EvaluationError> {
        Self::new_with_cartesian_and_sampling_task(
            name,
            abstractions,
            cartesian_abstractions,
            pdbs,
            config,
            task,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_cartesian_and_sampling_task(
        name: Option<String>,
        abstractions: Vec<DomainAbstraction>,
        cartesian_abstractions: Vec<CartesianAbstraction>,
        pdbs: Vec<PatternDatabase<'task>>,
        config: ScpOnlineConfig,
        task: &'task dyn AbstractNumericTask,
        sampling_task: Option<TaskRef<'task>>,
    ) -> Result<Self, EvaluationError> {
        if abstractions.is_empty() && cartesian_abstractions.is_empty() && pdbs.is_empty() {
            return Err(EvaluationError::ComputationFailed(
                "SCP requires at least one abstraction component".to_string(),
            ));
        }
        if config.online && config.interval == 0 {
            return Err(EvaluationError::ComputationFailed(
                "online SCP interval must be greater than zero".to_string(),
            ));
        }
        if config.online && config.diversify {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires online=false".to_string(),
            ));
        }
        if config.order_generator == OrderGenerator::Diverse && !config.diversify {
            return Err(EvaluationError::ComputationFailed(
                "diverse SCP orders require diversify=true".to_string(),
            ));
        }
        if config.diversify && config.samples == 0 {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires samples > 0".to_string(),
            ));
        }
        if config.diversify && config.max_orders == 0 {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires max_orders > 0".to_string(),
            ));
        }
        if config.diversify
            && (config.initial_order_generation_max_time.is_nan()
                || config.initial_order_generation_max_time < 0.0)
        {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires initial_order_generation_max_time >= 0"
                    .to_string(),
            ));
        }
        if config.diversify && sampling_task.is_none() {
            return Err(EvaluationError::ComputationFailed(
                "offline SCP diversification requires an owned task reference for sampling"
                    .to_string(),
            ));
        }
        if config.max_size == 0 {
            return Err(EvaluationError::ComputationFailed(
                "SCP max_size must be greater than zero".to_string(),
            ));
        }
        let abstraction_heuristics = abstractions
            .iter()
            .enumerate()
            .map(|(index, abstraction)| {
                DomainAbstractionHeuristic::new(
                    Some(format!("scp_online_{index}")),
                    abstraction.lookup_clone(),
                )
            })
            .collect();
        let cartesian_hierarchies = cartesian_abstractions
            .iter()
            .map(|abstraction| abstraction.hierarchy.clone())
            .collect();

        let original_costs: Vec<f64> = task
            .get_operators()
            .iter()
            .map(|op| metric_operator_cost_from_initial_values(task, op))
            .collect();

        let num_abstractions = abstractions.len();
        let num_cartesian_abstractions = cartesian_abstractions.len();
        let pdbs_count = pdbs.len();
        let total_components = num_abstractions + num_cartesian_abstractions + pdbs_count;

        let mut h_values: Vec<Vec<f64>> = Vec::with_capacity(total_components);
        let mut saturated_costs_by_abstraction: Vec<Vec<f64>> =
            Vec::with_capacity(total_components);
        let mut debug_initial_h_values = Vec::new();

        for (abstraction_id, abstraction) in abstractions.iter().enumerate() {
            let abstraction_task = abstraction.task_for_factory(task);
            let goal_facts = &abstraction.distance_table.goal_facts;
            let table = abstraction
                .factory
                .build_goal_distances_for_goals(
                    abstraction_task,
                    config.combine_labels,
                    &original_costs,
                    goal_facts,
                )
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute goal distances for order generator: {error:#}"
                    ))
                })?;
            let (_, saturated) = abstraction
                .factory
                .build_cost_partitioned_distance_table_for_goals(
                    abstraction_task,
                    config.combine_labels,
                    &original_costs,
                    false,
                    goal_facts,
                )
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute saturated costs for order generator: {error:#}"
                    ))
                })?;
            if config.collection_config.debug {
                let initial_h = table
                    .distances
                    .get(table.initial_state_hash)
                    .copied()
                    .ok_or_else(|| {
                        EvaluationError::InvalidState(format!(
                            "domain abstraction initial state {} out of bounds for {} states",
                            table.initial_state_hash,
                            table.distances.len()
                        ))
                    })?;
                debug_initial_h_values.push(initial_h);
                info!(
                    "scp_online debug: collection abstraction {abstraction_id}: original_initial_h={initial_h}, states={}, goal_facts={}",
                    abstraction_state_count(abstraction),
                    goal_facts.len()
                );
            }
            h_values.push(table.distances);
            saturated_costs_by_abstraction.push(saturated);
        }
        for (cartesian_id, abstraction) in cartesian_abstractions.iter().enumerate() {
            let (distances, saturated) = build_explicit_label_cost_partitioning_table(
                &abstraction.transition_system,
                &original_costs,
                None,
                None,
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute Cartesian order-generator table {cartesian_id}: {error:#}"
                ))
            })?;
            if config.collection_config.debug {
                let initial_h = distances
                    .get(abstraction.transition_system.initial_state_hash)
                    .copied()
                    .ok_or_else(|| {
                        EvaluationError::InvalidState(format!(
                            "Cartesian initial state {} out of bounds for {} states",
                            abstraction.transition_system.initial_state_hash,
                            distances.len()
                        ))
                    })?;
                debug_initial_h_values.push(initial_h);
                info!(
                    "scp_online debug: Cartesian abstraction {cartesian_id}: original_initial_h={initial_h}, states={}",
                    abstraction.num_states()
                );
            }
            h_values.push(distances);
            saturated_costs_by_abstraction.push(saturated);
        }

        for (pdb_id, pdb) in pdbs.iter().enumerate() {
            let distances = pdb.build_goal_distances(&original_costs).map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute PDB goal distances for order generator: {error}"
                ))
            })?;
            let (_, saturated) = pdb
                .build_cost_partitioned_distance_table(&original_costs)
                .map_err(|error| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to compute PDB saturated costs for order generator: {error}"
                    ))
                })?;
            if config.collection_config.debug {
                let initial_h = distances.first().copied().ok_or_else(|| {
                    EvaluationError::InvalidState(format!(
                        "PDB {pdb_id} has no initial abstract state"
                    ))
                })?;
                debug_initial_h_values.push(initial_h);
                info!(
                    "scp_online debug: PDB {pdb_id}: original_initial_h={initial_h}, states={}",
                    pdb.num_states()
                );
            }
            h_values.push(distances);
            saturated_costs_by_abstraction.push(saturated);
        }

        if config.collection_config.debug {
            let max_initial_h = debug_initial_h_values
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            info!(
                "scp_online debug: collection max original-cost initial h before cost partitioning = {max_initial_h}"
            );
        }

        let surplus_costs =
            compute_all_surplus_costs(&original_costs, &saturated_costs_by_abstraction);
        let stolen_costs: Vec<f64> = saturated_costs_by_abstraction
            .iter()
            .map(|saturated| compute_costs_stolen_by_heuristic(saturated, &surplus_costs))
            .collect();

        let mut st = ScpOnlineState::new(config.random_seed);
        st.h_values_by_abstraction = h_values;
        st.stolen_costs_by_abstraction = stolen_costs;

        Ok(Self {
            name: name.unwrap_or_else(|| "scp_online".to_string()),
            task,
            abstractions: RefCell::new(Some(abstractions)),
            abstraction_heuristics,
            cartesian_abstractions: RefCell::new(Some(cartesian_abstractions)),
            cartesian_hierarchies,
            pdbs,
            config,
            original_operator_costs: original_costs,
            state: RefCell::new(st),
            lookup_scratch: RefCell::new(DomainAbstractionLookupScratch::new()),
            component_ids_scratch: RefCell::new(Vec::new()),
            prop_scratch: RefCell::new(Vec::new()),
            numeric_scratch: RefCell::new(Vec::new()),
            sampling_task: RefCell::new(sampling_task),
        })
    }

    pub fn from_components(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpOnlineConfig,
        task: &'task dyn AbstractNumericTask,
    ) -> Result<Self, EvaluationError> {
        let mut domain_abstractions = Vec::new();
        let mut cartesian_abstractions = Vec::new();
        let mut pdbs = Vec::new();
        for component in components {
            match component {
                AbstractionComponent::Domain(heuristic) => {
                    domain_abstractions.push((*heuristic).into_abstraction());
                }
                AbstractionComponent::Cartesian(heuristic) => {
                    cartesian_abstractions.push((*heuristic).into_abstraction());
                }
                AbstractionComponent::PatternDatabase(pdb) => pdbs.push(*pdb),
            }
        }
        Self::new_with_cartesian(
            name,
            domain_abstractions,
            cartesian_abstractions,
            pdbs,
            config,
            task,
        )
    }

    pub fn from_components_with_sampling_task(
        name: Option<String>,
        components: Vec<AbstractionComponent<'task>>,
        config: ScpOnlineConfig,
        task: &'task dyn AbstractNumericTask,
        sampling_task: TaskRef<'task>,
    ) -> Result<Self, EvaluationError> {
        let mut domain_abstractions = Vec::new();
        let mut cartesian_abstractions = Vec::new();
        let mut pdbs = Vec::new();
        for component in components {
            match component {
                AbstractionComponent::Domain(heuristic) => {
                    domain_abstractions.push((*heuristic).into_abstraction());
                }
                AbstractionComponent::Cartesian(heuristic) => {
                    cartesian_abstractions.push((*heuristic).into_abstraction());
                }
                AbstractionComponent::PatternDatabase(pdb) => pdbs.push(*pdb),
            }
        }
        Self::new_with_cartesian_and_sampling_task(
            name,
            domain_abstractions,
            cartesian_abstractions,
            pdbs,
            config,
            task,
            Some(sampling_task),
        )
    }

    fn cartesian_abstraction_for_component(
        &self,
        component_id: usize,
        num_domain_abstractions: usize,
    ) -> Option<Ref<'_, CartesianAbstraction>> {
        let index = component_id.checked_sub(num_domain_abstractions)?;
        if index >= self.cartesian_hierarchies.len() {
            return None;
        }
        Some(Ref::map(
            self.cartesian_abstractions.borrow(),
            |collection| {
                collection
                    .as_ref()
                    .expect(
                        "Cartesian abstractions were released while SCP construction was active",
                    )
                    .get(index)
                    .expect("Cartesian abstraction component index must be valid")
            },
        ))
    }

    fn pdb_offset(&self, num_domain_abstractions: usize) -> usize {
        num_domain_abstractions + self.cartesian_hierarchies.len()
    }

    fn compute_order_for_state(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        abstractions: &[DomainAbstraction],
        num_domain_abstractions: usize,
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        let order = self.compute_state_dependent_order(
            task,
            state,
            abstract_state_ids,
            abstractions,
            num_domain_abstractions,
            deadline,
        )?;
        if self.config.use_abstract_operator_cost_partitioning
            && let Some(goal_cover_order) = self.cartesian_goal_cover_order(
                &order,
                num_domain_abstractions,
                &standalone_current_h_values(state, abstract_state_ids, num_domain_abstractions),
                true,
            )
        {
            if state.evaluated_states == 0 {
                info!(
                    "scp_online: using goal-cover transition-SCP order, first_components={:?}",
                    goal_cover_order.iter().take(24).collect::<Vec<_>>()
                );
            }
            return Ok(goal_cover_order);
        }
        Ok(order)
    }

    fn compute_state_dependent_order(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        abstractions: &[DomainAbstraction],
        num_domain_abstractions: usize,
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        match self.config.order_generator {
            OrderGenerator::Greedy => Ok(Self::compute_greedy_order_for_state(
                state,
                abstract_state_ids,
                self.config.scoring_function,
                abstractions,
                self.config.use_abstract_operator_cost_partitioning,
            )),
            OrderGenerator::Random => {
                let total = state.h_values_by_abstraction.len();
                let mut order: Vec<usize> = (0..total).collect();
                order.shuffle(&mut state.rng);
                Ok(order)
            }
            OrderGenerator::DynamicGreedy => self.compute_dynamic_greedy_order_for_state(
                task,
                state,
                abstract_state_ids,
                abstractions,
                num_domain_abstractions,
                deadline,
            ),
            OrderGenerator::Diverse => Ok(Self::compute_greedy_order_for_state(
                state,
                abstract_state_ids,
                self.config.scoring_function,
                abstractions,
                self.config.use_abstract_operator_cost_partitioning,
            )),
        }
    }

    fn compute_diversification_orders(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        abstractions: &[DomainAbstraction],
        num_domain_abstractions: usize,
        deadline: Option<Instant>,
    ) -> Result<Vec<Vec<usize>>, EvaluationError> {
        if self.config.order_generator != OrderGenerator::Diverse {
            return self
                .compute_state_dependent_order(
                    task,
                    state,
                    abstract_state_ids,
                    abstractions,
                    num_domain_abstractions,
                    deadline,
                )
                .map(|order| vec![order]);
        }

        let greedy = Self::compute_greedy_order_for_state(
            state,
            abstract_state_ids,
            self.config.scoring_function,
            abstractions,
            self.config.use_abstract_operator_cost_partitioning,
        );
        let mut random = (0..state.h_values_by_abstraction.len()).collect::<Vec<_>>();
        random.shuffle(&mut state.rng);
        Ok(deduplicate_orders(vec![greedy, random]))
    }

    fn cartesian_goal_cover_order(
        &self,
        base_order: &[usize],
        num_domain_abstractions: usize,
        standalone_current_h: &[f64],
        require_pure_cartesian_collection: bool,
    ) -> Option<Vec<usize>> {
        let collection = self.cartesian_abstractions.borrow();
        let collection = collection
            .as_ref()
            .expect("Cartesian abstractions were released while SCP order construction was active");
        cartesian_goal_cover_order(
            base_order,
            num_domain_abstractions,
            collection,
            standalone_current_h,
            require_pure_cartesian_collection,
            GoalCoverOrderVariant::default(),
        )
    }

    fn compact_cartesian_goal_cover_orders(
        &self,
        base_order: &[usize],
        num_domain_abstractions: usize,
        standalone_current_h: &[f64],
    ) -> Vec<(usize, Vec<usize>)> {
        let collection = self.cartesian_abstractions.borrow();
        let collection = collection
            .as_ref()
            .expect("Cartesian abstractions were released while SCP order construction was active");
        let mut variants_by_goal = HashMap::<usize, usize>::new();
        for abstraction in collection {
            if let Some(goal_id) = abstraction.metadata.collection_goal_id {
                *variants_by_goal.entry(goal_id).or_default() += 1;
            }
        }
        let progressive_roots = collection
            .iter()
            .any(|abstraction| abstraction.metadata.progressive_refinement_root);
        let goal_count = variants_by_goal
            .values()
            .filter(|&&variant_count| progressive_roots || variant_count >= 2)
            .count();
        let max_variants = variants_by_goal.values().copied().max().unwrap_or(0);
        if goal_count == 0 || (!progressive_roots && max_variants < 2) {
            return Vec::new();
        }

        // Rotate the anchor goal before varying its construction variant. This
        // guarantees coverage of every goal when the 64-order cap is large
        // enough, including states where that goal is the last one remaining.
        let variants = compact_goal_cover_variants(goal_count, max_variants, progressive_roots);

        let mut seen = HashSet::new();
        variants
            .into_iter()
            .filter_map(|variant| {
                let order = cartesian_goal_cover_order(
                    base_order,
                    num_domain_abstractions,
                    collection,
                    standalone_current_h,
                    false,
                    variant,
                )?;
                let first_component = *order
                    .first()
                    .expect("compact goal-cover order must not be empty");
                let goal_id = collection[first_component - num_domain_abstractions]
                    .metadata
                    .collection_goal_id
                    .expect("compact goal-cover order must start with its anchor goal");
                seen.insert(order.clone()).then_some((goal_id, order))
            })
            .collect()
    }

    fn cartesian_specialist_goal_for_order(
        &self,
        order: &[usize],
        num_domain_abstractions: usize,
    ) -> Option<usize> {
        if num_domain_abstractions != 0 {
            return None;
        }
        let collection = self.cartesian_abstractions.borrow();
        let collection = collection.as_ref()?;
        if order.len() != collection.len()
            || !order.iter().copied().all(|id| {
                collection
                    .get(id)
                    .is_some_and(|abstraction| abstraction.metadata.collection_goal_id.is_some())
            })
        {
            return None;
        }
        if !collection[order[0]].metadata.progressive_refinement_root {
            return None;
        }
        order
            .first()
            .and_then(|&id| collection[id].metadata.collection_goal_id)
    }

    fn prefixed_cartesian_goal_cover_order(
        &self,
        base_order: &[usize],
        num_domain_abstractions: usize,
        standalone_current_h: &[f64],
    ) -> Option<Vec<usize>> {
        let collection = self.cartesian_abstractions.borrow();
        let collection = collection
            .as_ref()
            .expect("Cartesian abstractions were released while SCP order construction was active");
        cartesian_goal_cover_order(
            base_order,
            num_domain_abstractions,
            collection,
            standalone_current_h,
            false,
            GoalCoverOrderVariant {
                non_goal_prefix: true,
                ..Default::default()
            },
        )
    }

    fn compute_greedy_order_for_state(
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        scoring_function: ScoringFunction,
        abstractions: &[DomainAbstraction],
        use_abstract_operator_cost_partitioning: bool,
    ) -> Vec<usize> {
        let total = state.h_values_by_abstraction.len();
        let mut order: Vec<usize> = (0..total).collect();
        order.shuffle(&mut state.rng);
        let current_h: Vec<f64> = (0..total)
            .map(|abs_id| {
                abstract_state_ids
                    .get(abs_id)
                    .copied()
                    .flatten()
                    .and_then(|sid| {
                        state
                            .h_values_by_abstraction
                            .get(abs_id)
                            .and_then(|values| values.get(sid))
                            .copied()
                    })
                    .unwrap_or(0.0)
            })
            .collect();

        let scores: Vec<f64> = (0..total)
            .map(|abs_id| {
                let h = current_h[abs_id];
                let stolen = state
                    .stolen_costs_by_abstraction
                    .get(abs_id)
                    .copied()
                    .unwrap_or(0.0);
                if use_abstract_operator_cost_partitioning
                    && let Some(abstraction) = abstractions.get(abs_id)
                {
                    compute_abstract_operator_order_score(h, stolen, scoring_function, abstraction)
                } else {
                    compute_score(h, stolen, scoring_function)
                }
            })
            .collect();

        order.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        order
    }

    fn compute_dynamic_greedy_order_for_state(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        abstractions: &[DomainAbstraction],
        num_domain_abstractions: usize,
        deadline: Option<Instant>,
    ) -> Result<Vec<usize>, EvaluationError> {
        if self.config.use_abstract_operator_cost_partitioning {
            return Err(EvaluationError::ComputationFailed(
                "dynamic_greedy_orders is only implemented for label SCP; abstract-operator SCP needs residual abstract-operator scoring, not label-order scoring".to_string(),
            ));
        }

        let total = state.h_values_by_abstraction.len();
        let mut remaining_components: Vec<usize> = (0..total).collect();
        let mut remaining_costs = self.original_operator_costs.clone();
        let mut order = Vec::with_capacity(total);

        'selection: while !remaining_components.is_empty() {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }

            remaining_components.shuffle(&mut state.rng);
            let mut candidate_saturated_costs = Vec::with_capacity(remaining_components.len());
            let mut candidate_h_values = Vec::with_capacity(remaining_components.len());

            for &pos in &remaining_components {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    break;
                }
                let result = if pos < num_domain_abstractions {
                    let abstraction = &abstractions[pos];
                    let abstraction_task = abstraction.task_for_factory(task);
                    Self::compute_domain_cp_entry(
                        abstraction,
                        abstraction_task,
                        self.config.combine_labels,
                        &remaining_costs,
                        deadline,
                    )
                } else if let Some(abstraction) =
                    self.cartesian_abstraction_for_component(pos, num_domain_abstractions)
                {
                    build_explicit_label_cost_partitioning_table(
                        &abstraction.transition_system,
                        &remaining_costs,
                        None,
                        deadline,
                    )
                    .map_err(|error| {
                        Self::construction_error(
                            &format!(
                                "failed to compute Cartesian dynamic-order table for component {pos}"
                            ),
                            error,
                        )
                    })
                } else {
                    let pdb_id = pos - self.pdb_offset(num_domain_abstractions);
                    let pdb = self.pdbs.get(pdb_id).ok_or_else(|| {
                        EvaluationError::ComputationFailed(format!(
                            "dynamic order requested missing PDB component {pdb_id}"
                        ))
                    })?;
                    pdb.build_cost_partitioned_distance_table(&remaining_costs)
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute PDB dynamic-order table {pdb_id}: {error}"
                            ))
                        })
                };
                let (distances, saturated) = match result {
                    Ok(entry) => entry,
                    Err(error) if Self::is_online_deadline_error_eval(&error) => {
                        info!("scp_online: dynamic greedy order stopped at deadline");
                        break 'selection;
                    }
                    Err(error) => return Err(error),
                };
                let h = current_h_for_distances(pos, &distances, abstract_state_ids);
                candidate_h_values.push(h);
                candidate_saturated_costs.push(saturated);
            }

            if candidate_saturated_costs.len() != remaining_components.len() {
                break;
            }

            let surplus_costs =
                compute_all_surplus_costs(&remaining_costs, &candidate_saturated_costs);
            let mut best_index = None;
            let mut best_score = f64::NEG_INFINITY;
            for rem_index in 0..remaining_components.len() {
                let stolen = compute_costs_stolen_by_heuristic(
                    &candidate_saturated_costs[rem_index],
                    &surplus_costs,
                );
                let score = compute_score(
                    candidate_h_values[rem_index],
                    stolen,
                    self.config.scoring_function,
                );
                if best_index.is_none() || score > best_score {
                    best_index = Some(rem_index);
                    best_score = score;
                }
            }
            let best_index = best_index.ok_or_else(|| {
                EvaluationError::ComputationFailed(
                    "dynamic order generator had no remaining candidate".to_string(),
                )
            })?;
            order.push(remaining_components[best_index]);
            reduce_costs(&mut remaining_costs, &candidate_saturated_costs[best_index])?;
            remaining_components.swap_remove(best_index);
        }

        order.extend(remaining_components);
        Ok(order)
    }

    fn compute_abstract_state_ids_into(
        &self,
        eval_state: &EvaluationState<'_, '_>,
        required_ids: Option<&[usize]>,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<usize, EvaluationError> {
        let num_domain = self.abstraction_heuristics.len();
        let num_cartesian = self.cartesian_hierarchies.len();
        let pdb_offset = num_domain + num_cartesian;
        let total_components = pdb_offset + self.pdbs.len();
        ids.clear();
        ids.resize(total_components, None);
        let needs_all = required_ids.is_none();
        {
            let mut scratch = self.lookup_scratch.borrow_mut();
            compute_collection_abstract_state_ids(
                &self.abstraction_heuristics,
                eval_state,
                required_ids,
                &mut scratch,
            )?;
            for (id, abstract_id) in scratch.abstract_state_ids.iter().copied().enumerate() {
                ids[id] = abstract_id;
            }
        }

        let cartesian_required = !self.cartesian_hierarchies.is_empty()
            && (needs_all
                || required_ids.is_some_and(|ids| {
                    ids.iter().any(|&id| (num_domain..pdb_offset).contains(&id))
                }));
        let pdb_required = !self.pdbs.is_empty()
            && (needs_all
                || required_ids.is_some_and(|ids| ids.iter().any(|&id| id >= pdb_offset)));
        if cartesian_required || pdb_required {
            let registry = eval_state.state_registry().ok_or_else(|| {
                EvaluationError::InvalidState(
                    "SCP Cartesian/PDB lookup requires state registry".to_string(),
                )
            })?;
            let mut numeric = self.numeric_scratch.borrow_mut();
            registry
                .fill_numeric_vars(eval_state.state(), &mut numeric)
                .map_err(|err| {
                    EvaluationError::ComputationFailed(format!(
                        "failed to read numeric state: {err:?}"
                    ))
                })?;
            let mut prop = self.prop_scratch.borrow_mut();
            eval_state.state().fill_state(registry, &mut prop);

            if cartesian_required {
                for (cartesian_id, hierarchy) in self.cartesian_hierarchies.iter().enumerate() {
                    let component_id = num_domain + cartesian_id;
                    if needs_all || required_ids.is_some_and(|ids| ids.contains(&component_id)) {
                        ids[component_id] =
                            Some(hierarchy.map_state(&prop, &numeric).map_err(|error| {
                                EvaluationError::ComputationFailed(error.to_string())
                            })?);
                    }
                }
            }
            if pdb_required {
                for (pdb_id, pdb) in self.pdbs.iter().enumerate() {
                    let sid = pdb
                        .abstract_state_id_from_source_state_values(&prop, &numeric)
                        .map_err(EvaluationError::ComputationFailed)?;
                    ids[pdb_offset + pdb_id] = sid;
                }
            }
        }

        Ok(num_domain)
    }

    fn compute_max_h(state: &ScpOnlineState, ids: &[Option<usize>]) -> f64 {
        state
            .cp_heuristics
            .iter()
            .map(|cp| cp.compute_heuristic(ids))
            .fold(0.0, f64::max)
    }

    fn required_lookup_ids(state: &ScpOnlineState) -> Vec<usize> {
        let mut ids: Vec<usize> = state
            .cp_heuristics
            .iter()
            .flat_map(|cp| cp.lookup_tables.iter().map(|table| table.abstraction_id))
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    fn is_online_deadline_error(error: &anyhow::Error) -> bool {
        crate::resource_limits::is_deadline_exceeded(error)
    }

    fn is_online_deadline_error_eval(error: &EvaluationError) -> bool {
        matches!(error, EvaluationError::ConstructionDeadlineExceeded)
    }

    fn construction_error(context: &str, error: anyhow::Error) -> EvaluationError {
        if Self::is_online_deadline_error(&error) {
            EvaluationError::ConstructionDeadlineExceeded
        } else {
            EvaluationError::ComputationFailed(format!("{context}: {error:#}"))
        }
    }

    fn reduce_abstract_operator_costs(
        remaining_costs: &mut TransitionResidualCosts,
        abstraction_id: usize,
        footprints: &[AbstractOperatorFootprint],
        tcf: &AbstractOperatorCostFunction,
        deadline: Option<Instant>,
        context: &str,
    ) -> Result<bool, EvaluationError> {
        match remaining_costs.reduce_by_abstract_operator_footprints_with_deadline(
            abstraction_id,
            footprints,
            tcf,
            deadline,
        ) {
            Ok(()) => Ok(true),
            Err(error) if Self::is_online_deadline_error(&error) => Ok(false),
            Err(error) => Err(EvaluationError::ComputationFailed(format!(
                "{context}: {error:#}"
            ))),
        }
    }

    fn update_improvement_status(&self, state: &mut ScpOnlineState) {
        if !self.config.online {
            return;
        }
        let time_limit_reached = self.config.max_time.is_finite()
            && state.start_time.elapsed() >= Duration::from_secs_f64(self.config.max_time);

        if state.improve_heuristic && (time_limit_reached || state.size_kb >= self.config.max_size)
        {
            state.improve_heuristic = false;
        }
    }

    fn release_abstractions_if_finished(&self, state: &mut ScpOnlineState) {
        if !state.improve_heuristic && !state.improvement_ended {
            let domain_released = self.abstractions.borrow_mut().take().is_some();
            let cartesian_released = self.cartesian_abstractions.borrow_mut().take().is_some();
            assert!(
                domain_released || cartesian_released,
                "SCP construction data must exist until improvement ends"
            );
            state.improvement_ended = true;
        }
    }

    fn should_build_cp(&self, state: &ScpOnlineState) -> bool {
        state.improve_heuristic
            && (state.evaluated_states == 0
                || (self.config.online
                    && state.evaluated_states.is_multiple_of(self.config.interval)))
    }

    fn maybe_build_cp(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
    ) -> Result<Vec<CostPartitioningHeuristic>, EvaluationError> {
        if !self.should_build_cp(state) {
            return Ok(Vec::new());
        }

        let abstractions_guard = self.abstractions.borrow();
        let abstractions: &[DomainAbstraction] = match &*abstractions_guard {
            Some(abs) => abs.as_slice(),
            None => &[],
        };
        if abstractions.is_empty() && self.cartesian_hierarchies.is_empty() && self.pdbs.is_empty()
        {
            return Ok(Vec::new());
        }
        let original_costs = self.original_operator_costs.as_slice();
        let deadline = self
            .config
            .table_construction_max_time
            .is_finite()
            .then(|| {
                Instant::now() + Duration::from_secs_f64(self.config.table_construction_max_time)
            });
        let standalone_current_h =
            standalone_current_h_values(state, abstract_state_ids, num_domain_abstractions);
        let mut order = self.compute_order_for_state(
            task,
            state,
            abstract_state_ids,
            abstractions,
            num_domain_abstractions,
            deadline,
        )?;
        let mode = if self.config.use_abstract_operator_cost_partitioning {
            "abstract-operator"
        } else {
            "label"
        };
        info!(
            "scp_online: building {mode} CP at evaluation {}, stored_cps={}, current_h={}, size={} KiB, order_len={}, saturator={}, elapsed={:.3}s",
            state.evaluated_states,
            state.cp_heuristics.len(),
            Self::compute_max_h(state, abstract_state_ids),
            state.size_kb,
            order.len(),
            self.config.saturator,
            state.start_time.elapsed().as_secs_f64(),
        );
        if self.config.collection_config.debug {
            log_abstraction_candidate_report(
                mode,
                state,
                abstractions,
                &order,
                abstract_state_ids,
                self.config.scoring_function,
                self.config.use_abstract_operator_cost_partitioning,
            );
        }

        let initial_order_generation_max_time = if self.config.diversify {
            self.config.initial_order_generation_max_time
        } else {
            self.config.order_optimization_max_time
        };
        let mut candidates = if self.config.use_abstract_operator_cost_partitioning {
            self.build_best_abstract_operator_cp_from_candidate_orders(
                task,
                abstractions,
                &mut order,
                abstract_state_ids,
                &standalone_current_h,
                num_domain_abstractions,
                original_costs,
                deadline,
                initial_order_generation_max_time,
            )?
        } else {
            self.build_best_label_cp_from_candidate_orders(
                task,
                abstractions,
                &mut order,
                abstract_state_ids,
                &standalone_current_h,
                num_domain_abstractions,
                original_costs,
                deadline,
                initial_order_generation_max_time,
            )?
        };

        if self.config.order_optimization_max_time > 0.0 {
            let local_deadline = optimization_deadline(self.config.order_optimization_max_time);
            self.optimize_order_with_hill_climbing(
                task,
                abstractions,
                &standalone_current_h,
                num_domain_abstractions,
                original_costs,
                abstract_state_ids,
                &mut order,
                &mut candidates.partitions[candidates.best_index],
                earliest_deadline(deadline, local_deadline),
            )?;
        }

        if self.config.online {
            let best = candidates.partitions.swap_remove(candidates.best_index);
            if best.is_empty() {
                info!("scp_online: {mode} CP attempt produced no lookup tables");
                return Ok(Vec::new());
            }
            return Ok(vec![best]);
        }
        if self.config.diversify {
            let best = candidates.partitions.swap_remove(candidates.best_index);
            candidates.partitions.retain(|cp| !cp.is_empty());
            if best.is_empty() && candidates.partitions.is_empty() {
                info!("scp_online: {mode} CP attempt produced no positive lookup tables");
                return Ok(Vec::new());
            }
            let mut initial_candidates = Vec::with_capacity(candidates.partitions.len() + 1);
            if !best.is_empty() {
                initial_candidates.push(best);
            }
            initial_candidates.extend(candidates.partitions);
            return self.build_offline_diversified_portfolio(
                task,
                state,
                abstractions,
                num_domain_abstractions,
                abstract_state_ids,
                initial_candidates,
                deadline,
            );
        }
        candidates.partitions.retain(|cp| !cp.is_empty());
        if candidates.partitions.is_empty() {
            info!("scp_online: {mode} CP attempt produced no lookup tables");
            return Ok(Vec::new());
        }
        info!(
            "scp_online: retaining {} offline SCP order partitions",
            candidates.partitions.len()
        );
        Ok(candidates.partitions)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_offline_diversified_portfolio(
        &self,
        task: &dyn AbstractNumericTask,
        state: &mut ScpOnlineState,
        abstractions: &[DomainAbstraction],
        num_domain_abstractions: usize,
        initial_abstract_state_ids: &[Option<usize>],
        initial_candidates: Vec<CostPartitioningHeuristic>,
        table_deadline: Option<Instant>,
    ) -> Result<Vec<CostPartitioningHeuristic>, EvaluationError> {
        assert!(!self.config.online);
        assert!(self.config.diversify);
        assert!(!initial_candidates.is_empty());
        assert!(!initial_candidates[0].is_empty());

        let diversification_deadline = self
            .config
            .max_time
            .is_finite()
            .then(|| Instant::now() + Duration::from_secs_f64(self.config.max_time.max(0.0)));
        let deadline = earliest_deadline(table_deadline, diversification_deadline);
        let initial_h = initial_candidates[0].compute_heuristic(initial_abstract_state_ids);
        self.generate_offline_samples(state, initial_h, deadline)?;
        assert!(
            !state.offline_sample_ids.is_empty(),
            "offline diversification must retain at least the initial state sample"
        );

        let mut sample_best = vec![f64::NEG_INFINITY; state.offline_sample_ids.len()];
        let mut portfolio = Vec::new();
        let mut portfolio_size_kb = 0usize;
        let mut evaluated_orders = initial_candidates.len();
        let mandatory_indices =
            mandatory_goal_specialist_indices(&initial_candidates, &state.offline_sample_ids);
        if mandatory_indices.len() > self.config.max_orders {
            return Err(EvaluationError::ComputationFailed(format!(
                "offline SCP requires {} orders to retain the global best and configured specialists per represented goal, exceeding max_orders={}",
                mandatory_indices.len(),
                self.config.max_orders,
            )));
        }
        let represented_goal_count = mandatory_indices
            .iter()
            .filter_map(|&index| initial_candidates[index].specialist_goal_id)
            .collect::<HashSet<_>>()
            .len();
        let specialist_count = mandatory_indices
            .iter()
            .filter(|&&index| initial_candidates[index].specialist_goal_id.is_some())
            .count();
        let mut initial_candidates = initial_candidates.into_iter().map(Some).collect::<Vec<_>>();
        for index in mandatory_indices {
            let candidate = initial_candidates[index]
                .take()
                .expect("mandatory SCP candidate indices must be unique");
            retain_mandatory_partition(
                candidate,
                &state.offline_sample_ids,
                &mut sample_best,
                &mut portfolio,
                &mut portfolio_size_kb,
                self.config.max_size,
            )
            .map_err(EvaluationError::ComputationFailed)?;
        }
        info!(
            "scp_online: retained {specialist_count} mandatory specialists across {represented_goal_count} goals plus the global best"
        );

        for candidate in initial_candidates.into_iter().flatten() {
            if portfolio.len() >= self.config.max_orders
                || portfolio_size_kb >= self.config.max_size
                || deadline.is_some_and(|end| Instant::now() >= end)
            {
                break;
            }
            retain_if_sample_improving(
                candidate,
                &state.offline_sample_ids,
                &mut sample_best,
                &mut portfolio,
                &mut portfolio_size_kb,
                self.config.max_size,
            );
        }
        assert!(
            !portfolio.is_empty(),
            "the global best SCP must be retained"
        );

        let original_costs = self.original_operator_costs.as_slice();
        for sample_index in 1..state.offline_sample_ids.len() {
            if portfolio.len() >= self.config.max_orders
                || portfolio_size_kb >= self.config.max_size
                || deadline.is_some_and(|end| Instant::now() >= end)
            {
                break;
            }
            let sample_ids = state.offline_sample_ids[sample_index].clone();
            let standalone_h =
                standalone_current_h_values(state, &sample_ids, num_domain_abstractions);
            let orders = self.compute_diversification_orders(
                task,
                state,
                &sample_ids,
                abstractions,
                num_domain_abstractions,
                deadline,
            )?;
            for mut order in orders {
                if portfolio.len() >= self.config.max_orders
                    || portfolio_size_kb >= self.config.max_size
                    || deadline.is_some_and(|end| Instant::now() >= end)
                {
                    break;
                }
                let candidate = if self.config.use_abstract_operator_cost_partitioning {
                    self.build_abstract_operator_cp(
                        task,
                        abstractions,
                        &order,
                        &sample_ids,
                        &standalone_h,
                        num_domain_abstractions,
                        original_costs,
                        deadline,
                        self.config.saturator,
                    )
                } else {
                    self.build_label_cp(
                        task,
                        abstractions,
                        &order,
                        &sample_ids,
                        num_domain_abstractions,
                        original_costs,
                        deadline,
                    )
                };
                let mut candidate = match candidate {
                    Ok(candidate) => candidate,
                    Err(error) if Self::is_online_deadline_error_eval(&error) => break,
                    Err(error) => return Err(error),
                };
                evaluated_orders += 1;

                if self.config.order_optimization_max_time > 0.0 {
                    let local_deadline =
                        optimization_deadline(self.config.order_optimization_max_time);
                    self.optimize_order_with_hill_climbing(
                        task,
                        abstractions,
                        &standalone_h,
                        num_domain_abstractions,
                        original_costs,
                        &sample_ids,
                        &mut order,
                        &mut candidate,
                        earliest_deadline(deadline, local_deadline),
                    )?;
                }

                retain_if_sample_improving(
                    candidate,
                    &state.offline_sample_ids,
                    &mut sample_best,
                    &mut portfolio,
                    &mut portfolio_size_kb,
                    self.config.max_size,
                );
            }
        }

        let sample_count = state.offline_sample_ids.len();
        info!(
            "scp_online: offline diversification retained {} of {} evaluated partitions over {} samples ({} KiB)",
            portfolio.len(),
            evaluated_orders,
            sample_count,
            portfolio_size_kb,
        );
        state.offline_sample_ids.clear();
        state.offline_sample_ids.shrink_to_fit();
        Ok(portfolio)
    }

    fn generate_offline_samples(
        &self,
        state: &mut ScpOnlineState,
        initial_h: f64,
        deadline: Option<Instant>,
    ) -> Result<(), EvaluationError> {
        if !state.offline_sample_ids.is_empty() {
            return Ok(());
        }
        let sampling_task = self
            .sampling_task
            .borrow_mut()
            .take()
            .expect("offline diversification was validated to have an owned sampling task");
        let mut registry = StateRegistry::for_task(sampling_task.clone());
        let successor_generator = SuccessorTree::new(&*sampling_task);
        let initial_state = registry.get_initial_state();
        let average_cost = if self.original_operator_costs.is_empty() {
            0.0
        } else {
            self.original_operator_costs.iter().sum::<f64>()
                / self.original_operator_costs.len() as f64
        };
        let mut applicable = Vec::new();
        let mut propositional = Vec::new();
        let mut successor_numeric = Vec::new();
        let mut successor_cost = Vec::new();
        let mut expansion_context = ExpansionContext::default();
        let mut ids = Vec::new();

        self.map_sample_state(&initial_state, &registry, self.task, &mut ids)?;
        state.offline_sample_ids.push(ids.clone());

        while state.offline_sample_ids.len() < self.config.samples {
            if deadline.is_some_and(|end| Instant::now() >= end) {
                info!(
                    "scp_online: offline sampling deadline reached after {} samples",
                    state.offline_sample_ids.len()
                );
                break;
            }
            let walk_length = random_walk_length(initial_h, average_cost, &mut state.rng)?;
            let mut current = initial_state.clone();
            for _ in 0..walk_length {
                if deadline.is_some_and(|end| Instant::now() >= end) {
                    break;
                }
                current.fill_state(&registry, &mut propositional);
                applicable.clear();
                successor_generator.get_applicable_operators(&propositional, &mut applicable);
                let Some(&operator_id) = applicable.choose(&mut state.rng) else {
                    break;
                };
                registry
                    .build_expansion_context(&current, &mut expansion_context)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to build random-walk expansion context: {error:?}"
                        ))
                    })?;
                let operator = sampling_task
                    .get_operators()
                    .get(operator_id as usize)
                    .expect("successor generator returned an invalid operator id");
                let (successor, _) = registry
                    .apply_operator_in_context(
                        &current,
                        operator,
                        &expansion_context,
                        &mut successor_numeric,
                        &mut successor_cost,
                    )
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to apply random-walk operator {}: {error:?}",
                            operator.name()
                        ))
                    })?;
                current = successor;
            }
            self.map_sample_state(&current, &registry, self.task, &mut ids)?;
            state.offline_sample_ids.push(ids.clone());
        }
        info!(
            "scp_online: generated {} offline random-walk samples",
            state.offline_sample_ids.len()
        );
        Ok(())
    }

    fn map_sample_state(
        &self,
        concrete_state: &ConcreteState,
        registry: &StateRegistry<'task>,
        task: &'task dyn AbstractNumericTask,
        ids: &mut Vec<Option<usize>>,
    ) -> Result<(), EvaluationError> {
        let eval_state =
            EvaluationState::new_with_registry(concrete_state, 0.0, false, task, registry);
        self.compute_abstract_state_ids_into(&eval_state, None, ids)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_best_label_cp_from_candidate_orders(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        incumbent_order: &mut Vec<usize>,
        abstract_state_ids: &[Option<usize>],
        standalone_current_h: &[f64],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        baseline_deadline: Option<Instant>,
        optimization_max_time: f64,
    ) -> Result<CandidateCostPartitions, EvaluationError> {
        let mut best_order = incumbent_order.clone();
        let baseline = self.build_label_cp(
            task,
            abstractions,
            &best_order,
            abstract_state_ids,
            num_domain_abstractions,
            original_costs,
            baseline_deadline,
        )?;
        let mut best_h = baseline.compute_heuristic(abstract_state_ids);
        let mut partitions = vec![baseline];
        let mut best_index = 0;
        let candidate_deadline = earliest_deadline(
            baseline_deadline,
            optimization_deadline(optimization_max_time),
        );

        for candidate in optimization_max_time
            .is_sign_positive()
            .then(|| self.candidate_label_orders(incumbent_order, standalone_current_h))
            .into_iter()
            .flatten()
        {
            if candidate_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if candidate == best_order {
                continue;
            }
            let candidate_cp = match self.build_label_cp(
                task,
                abstractions,
                &candidate,
                abstract_state_ids,
                num_domain_abstractions,
                original_costs,
                candidate_deadline,
            ) {
                Ok(cp) => cp,
                Err(error) if Self::is_online_deadline_error_eval(&error) => {
                    info!(
                        "scp_online: label candidate order stopped while computing table (deadline)"
                    );
                    break;
                }
                Err(error) => return Err(error),
            };
            let candidate_h = candidate_cp.compute_heuristic(abstract_state_ids);
            let candidate_index = partitions.len();
            partitions.push(candidate_cp);
            if candidate_h > best_h
                || (candidate_h == best_h
                    && partitions[best_index].is_empty()
                    && !partitions[candidate_index].is_empty())
            {
                if self.config.collection_config.debug {
                    info!("scp_online: label candidate order improved h {best_h} -> {candidate_h}");
                }
                best_h = candidate_h;
                best_order = candidate;
                best_index = candidate_index;
            }
        }

        *incumbent_order = best_order;
        Ok(CandidateCostPartitions {
            partitions,
            best_index,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn build_best_abstract_operator_cp_from_candidate_orders(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        incumbent_order: &mut Vec<usize>,
        abstract_state_ids: &[Option<usize>],
        standalone_current_h: &[f64],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        baseline_deadline: Option<Instant>,
        optimization_max_time: f64,
    ) -> Result<CandidateCostPartitions, EvaluationError> {
        let mut best_order = incumbent_order.clone();
        let mut baseline = self.build_abstract_operator_cp(
            task,
            abstractions,
            &best_order,
            abstract_state_ids,
            standalone_current_h,
            num_domain_abstractions,
            original_costs,
            baseline_deadline,
            self.config.saturator,
        )?;
        baseline.specialist_goal_id =
            self.cartesian_specialist_goal_for_order(&best_order, num_domain_abstractions);
        let mut best_h = baseline.compute_heuristic(abstract_state_ids);
        let mut partitions = vec![baseline];
        let mut best_index = 0;
        let candidate_deadline = earliest_deadline(
            baseline_deadline,
            optimization_deadline(optimization_max_time),
        );

        for (specialist_goal_id, candidate) in optimization_max_time
            .is_sign_positive()
            .then(|| {
                self.candidate_abstract_operator_orders(
                    incumbent_order,
                    abstractions,
                    standalone_current_h,
                )
            })
            .into_iter()
            .flatten()
        {
            if candidate_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if candidate == best_order {
                continue;
            }
            let mut candidate_cp = match self.build_abstract_operator_cp(
                task,
                abstractions,
                &candidate,
                abstract_state_ids,
                standalone_current_h,
                num_domain_abstractions,
                original_costs,
                candidate_deadline,
                self.config.saturator,
            ) {
                Ok(cp) => cp,
                Err(error) if Self::is_online_deadline_error_eval(&error) => {
                    info!(
                        "scp_online: abstract-operator candidate order stopped while computing table (deadline)"
                    );
                    break;
                }
                Err(error) => return Err(error),
            };
            candidate_cp.specialist_goal_id = specialist_goal_id.or_else(|| {
                self.cartesian_specialist_goal_for_order(&candidate, num_domain_abstractions)
            });
            let candidate_h = candidate_cp.compute_heuristic(abstract_state_ids);
            let candidate_index = partitions.len();
            partitions.push(candidate_cp);
            if candidate_h > best_h
                || (candidate_h == best_h
                    && partitions[best_index].is_empty()
                    && !partitions[candidate_index].is_empty())
            {
                if self.config.collection_config.debug {
                    info!("scp_online: candidate order improved h {best_h} -> {candidate_h}");
                }
                best_h = candidate_h;
                best_order = candidate;
                best_index = candidate_index;
            }
        }

        *incumbent_order = best_order;
        Ok(CandidateCostPartitions {
            partitions,
            best_index,
        })
    }

    fn candidate_abstract_operator_orders(
        &self,
        base_order: &[usize],
        abstractions: &[DomainAbstraction],
        standalone_current_h: &[f64],
    ) -> Vec<(Option<usize>, Vec<usize>)> {
        let mut orders = Vec::new();
        orders.push((None, base_order.to_vec()));

        let mut declaration_order = base_order.to_vec();
        declaration_order.sort_unstable();
        orders.push((None, declaration_order));

        if let Some(prefixed_goal_cover_order) = self.prefixed_cartesian_goal_cover_order(
            base_order,
            abstractions.len(),
            standalone_current_h,
        ) {
            orders.push((None, prefixed_goal_cover_order));
        }

        orders.extend(
            self.compact_cartesian_goal_cover_orders(
                base_order,
                abstractions.len(),
                standalone_current_h,
            )
            .into_iter()
            .map(|(goal_id, order)| (Some(goal_id), order)),
        );

        if let Some(goal_cover_order) = self.cartesian_goal_cover_order(
            base_order,
            abstractions.len(),
            standalone_current_h,
            false,
        ) {
            orders.push((None, goal_cover_order));
        }

        orders.push((
            None,
            max_heuristic_greedy_order(base_order, standalone_current_h),
        ));

        let mut by_collection = base_order.to_vec();
        by_collection.sort_by_key(|&id| abstraction_collection_iteration(abstractions, id));
        orders.push((None, by_collection));

        let mut progression_first = base_order.to_vec();
        progression_first.sort_by(|&left, &right| {
            abstraction_is_target_centered(abstractions, left)
                .cmp(&abstraction_is_target_centered(abstractions, right))
                .then_with(|| {
                    standalone_current_h
                        .get(right)
                        .copied()
                        .unwrap_or(0.0)
                        .total_cmp(&standalone_current_h.get(left).copied().unwrap_or(0.0))
                })
                .then_with(|| left.cmp(&right))
        });
        orders.push((None, progression_first));

        let mut target_first = base_order.to_vec();
        target_first.sort_by(|&left, &right| {
            abstraction_is_target_centered(abstractions, right)
                .cmp(&abstraction_is_target_centered(abstractions, left))
                .then_with(|| {
                    standalone_current_h
                        .get(right)
                        .copied()
                        .unwrap_or(0.0)
                        .total_cmp(&standalone_current_h.get(left).copied().unwrap_or(0.0))
                })
                .then_with(|| left.cmp(&right))
        });
        orders.push((None, target_first));

        for seed_offset in 0..3 {
            let mut random_order = base_order.to_vec();
            random_order.shuffle(&mut SmallRng::seed_from_u64(
                self.config
                    .random_seed
                    .unwrap_or(0x5C9_0A11)
                    .wrapping_add(seed_offset),
            ));
            orders.push((None, random_order));
        }

        deduplicate_specialist_orders(orders)
    }

    fn candidate_label_orders(
        &self,
        base_order: &[usize],
        standalone_current_h: &[f64],
    ) -> Vec<Vec<usize>> {
        deduplicate_orders(vec![
            base_order.to_vec(),
            max_heuristic_greedy_order(base_order, standalone_current_h),
        ])
    }

    #[allow(clippy::too_many_arguments)]
    fn optimize_order_with_hill_climbing(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        standalone_current_h: &[f64],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        abstract_state_ids: &[Option<usize>],
        incumbent_order: &mut [usize],
        incumbent_cp: &mut CostPartitioningHeuristic,
        optimization_deadline: Option<Instant>,
    ) -> Result<(), EvaluationError> {
        let mut incumbent_h = incumbent_cp.compute_heuristic(abstract_state_ids);
        if self.config.collection_config.debug {
            info!("scp_online: order optimization incumbent_h={incumbent_h}");
        }

        loop {
            if optimization_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            let mut improved = false;
            for i in 0..incumbent_order.len() {
                for j in (i + 1)..incumbent_order.len() {
                    if optimization_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                        return Ok(());
                    }

                    incumbent_order.swap(i, j);
                    let neighbor_result = if self.config.use_abstract_operator_cost_partitioning {
                        self.build_abstract_operator_cp(
                            task,
                            abstractions,
                            incumbent_order,
                            abstract_state_ids,
                            standalone_current_h,
                            num_domain_abstractions,
                            original_costs,
                            optimization_deadline,
                            self.config.saturator,
                        )
                    } else {
                        self.build_label_cp(
                            task,
                            abstractions,
                            incumbent_order,
                            abstract_state_ids,
                            num_domain_abstractions,
                            original_costs,
                            optimization_deadline,
                        )
                    };
                    let neighbor_cp = match neighbor_result {
                        Ok(cp) => cp,
                        Err(error) if Self::is_online_deadline_error_eval(&error) => {
                            incumbent_order.swap(i, j);
                            info!(
                                "scp_online: order optimization stopped while computing table (deadline)"
                            );
                            return Ok(());
                        }
                        Err(error) => {
                            incumbent_order.swap(i, j);
                            return Err(error);
                        }
                    };
                    let neighbor_h = neighbor_cp.compute_heuristic(abstract_state_ids);
                    if neighbor_h > incumbent_h {
                        if self.config.collection_config.debug {
                            info!(
                                "scp_online: order optimization swapped positions {i}/{j}, h {incumbent_h} -> {neighbor_h}"
                            );
                        }
                        *incumbent_cp = neighbor_cp;
                        incumbent_h = neighbor_h;
                        improved = true;
                        break;
                    }

                    incumbent_order.swap(i, j);
                }
                if improved {
                    break;
                }
            }
            if !improved {
                break;
            }
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_abstract_operator_cp(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        order: &[usize],
        abstract_state_ids: &[Option<usize>],
        standalone_current_h: &[f64],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        deadline: Option<Instant>,
        saturator: Saturator,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = TransitionResidualCosts::from_operator_costs(original_costs);

        for sweep in 0..=self.config.residual_sweeps {
            if sweep > 0 {
                info!(
                    "scp_online: starting regional residual abstraction sweep {sweep}/{}",
                    self.config.residual_sweeps
                );
            }
            for &pos in order {
                if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    break;
                }
                if pos < num_domain_abstractions {
                    let abstraction = &abstractions[pos];
                    debug!(
                        "scp_online: abstract-operator CP step abstraction {pos}, abstract_states={}, standalone_h={}, metadata={}",
                        abstraction_state_count(abstraction),
                        standalone_current_h.get(pos).copied().unwrap_or(0.0),
                        abstraction_metadata_summary(abstraction),
                    );
                    log_abstract_operator_footprint_summary(
                        pos,
                        &abstraction.abstract_operator_footprints,
                    );
                    let abstraction_task = abstraction.task_for_factory(task);
                    match saturator {
                        Saturator::All => {
                            self.log_abstract_operator_label_diagnostic(
                                abstraction,
                                abstraction_task,
                                pos,
                                abstract_state_ids,
                                &remaining_costs,
                            )?;
                            let (table, tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                                    abstraction_task,
                                    abstraction.combine_labels,
                                    &abstraction.abstract_operators,
                                    &abstraction.abstract_operator_footprints,
                                    &remaining_costs,
                                    pos,
                                    abstract_state_ids.get(pos).copied().flatten(),
                                    None,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator all abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        break;
                                    }
                                    Err(error) => {
                                        return Err(EvaluationError::ComputationFailed(format!(
                                        "failed to compute abstract-operator SCP table: {error:#}"
                                    )));
                                    }
                                };
                            log_transition_table_summary(
                                "all",
                                pos,
                                &table.distances,
                                &tcf.operator_costs,
                                abstract_state_ids,
                            );
                            log_movement_abstract_operator_costs(
                                pos,
                                abstraction,
                                abstraction_task,
                                &tcf,
                                &remaining_costs,
                            );
                            if should_skip_zero_current_table(
                                "all",
                                pos,
                                &table.distances,
                                abstract_state_ids,
                            ) {
                                continue;
                            }
                            if !Self::reduce_abstract_operator_costs(
                                &mut remaining_costs,
                                pos,
                                &abstraction.abstract_operator_footprints,
                                &tcf,
                                deadline,
                                "failed to reduce abstract-operator residual costs",
                            )? {
                                info!(
                                    "scp_online: abstract-operator abstraction {pos} stopped while reducing residual costs (deadline)"
                                );
                                break;
                            }
                            cp.add_h_values(pos, table.distances);
                            log_transition_residual_summary(&remaining_costs);
                        }
                        Saturator::Perim => {
                            let cap_state_id = abstract_state_ids.get(pos).copied().flatten();
                            self.log_abstract_operator_label_diagnostic(
                                abstraction,
                                abstraction_task,
                                pos,
                                abstract_state_ids,
                                &remaining_costs,
                            )?;
                            let (table, tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                                    abstraction_task,
                                    abstraction.combine_labels,
                                    &abstraction.abstract_operators,
                                    &abstraction.abstract_operator_footprints,
                                    &remaining_costs,
                                    pos,
                                    cap_state_id,
                                    cap_state_id,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator perim abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        break;
                                    }
                                    Err(error) => {
                                        return Err(EvaluationError::ComputationFailed(format!(
                                        "failed to compute abstract-operator PERIM table: {error:#}"
                                    )));
                                    }
                                };
                            log_transition_table_summary(
                                "perim",
                                pos,
                                &table.distances,
                                &tcf.operator_costs,
                                abstract_state_ids,
                            );
                            if should_skip_zero_current_table(
                                "perim",
                                pos,
                                &table.distances,
                                abstract_state_ids,
                            ) {
                                continue;
                            }
                            if !Self::reduce_abstract_operator_costs(
                                &mut remaining_costs,
                                pos,
                                &abstraction.abstract_operator_footprints,
                                &tcf,
                                deadline,
                                "failed to reduce abstract-operator PERIM residual costs",
                            )? {
                                info!(
                                    "scp_online: abstract-operator PERIM abstraction {pos} stopped while reducing residual costs (deadline)"
                                );
                                break;
                            }
                            cp.add_h_values(pos, table.distances);
                            log_transition_residual_summary(&remaining_costs);
                        }
                        Saturator::Perimstar => {
                            let cap_state_id = abstract_state_ids.get(pos).copied().flatten();
                            self.log_abstract_operator_label_diagnostic(
                                abstraction,
                                abstraction_task,
                                pos,
                                abstract_state_ids,
                                &remaining_costs,
                            )?;
                            let (perim_table, perim_tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                                    abstraction_task,
                                    abstraction.combine_labels,
                                    &abstraction.abstract_operators,
                                    &abstraction.abstract_operator_footprints,
                                    &remaining_costs,
                                    pos,
                                    cap_state_id,
                                    cap_state_id,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator perimstar/perim abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        break;
                                    }
                                    Err(error) => {
                                        return Err(EvaluationError::ComputationFailed(format!(
                                        "failed to compute abstract-operator Perim step for Perimstar: {error:#}"
                                    )));
                                    }
                                };
                            log_transition_table_summary(
                                "perimstar/perim",
                                pos,
                                &perim_table.distances,
                                &perim_tcf.operator_costs,
                                abstract_state_ids,
                            );
                            if should_skip_zero_current_table(
                                "perimstar/perim",
                                pos,
                                &perim_table.distances,
                                abstract_state_ids,
                            ) {
                                continue;
                            }
                            if !Self::reduce_abstract_operator_costs(
                                &mut remaining_costs,
                                pos,
                                &abstraction.abstract_operator_footprints,
                                &perim_tcf,
                                deadline,
                                "failed to reduce abstract-operator Perim residual costs",
                            )? {
                                info!(
                                    "scp_online: abstract-operator Perim abstraction {pos} stopped while reducing residual costs (deadline)"
                                );
                                break;
                            }
                            cp.add_h_values(pos, perim_table.distances);
                            log_transition_residual_summary(&remaining_costs);

                            let (all_table, all_tcf) = match abstraction
                                .factory
                                .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                                    abstraction_task,
                                            abstraction.combine_labels,
                                            &abstraction.abstract_operators,
                                            &abstraction.abstract_operator_footprints,
                                            &remaining_costs,
                                            pos,
                                            cap_state_id,
                                    None,
                                    deadline,
                                ) {
                                    Ok(result) => result,
                                    Err(error) if Self::is_online_deadline_error(&error) => {
                                        info!(
                                            "scp_online: abstract-operator perimstar/all abstraction {pos} stopped while computing table (deadline)"
                                        );
                                        break;
                                    }
                                    Err(error) => {
                                        return Err(EvaluationError::ComputationFailed(format!(
                                        "failed to compute abstract-operator All step for Perimstar: {error:#}"
                                    )));
                                    }
                                };
                            log_transition_table_summary(
                                "perimstar/all",
                                pos,
                                &all_table.distances,
                                &all_tcf.operator_costs,
                                abstract_state_ids,
                            );
                            if should_skip_zero_current_table(
                                "perimstar/all",
                                pos,
                                &all_table.distances,
                                abstract_state_ids,
                            ) {
                                continue;
                            }
                            if !Self::reduce_abstract_operator_costs(
                                &mut remaining_costs,
                                pos,
                                &abstraction.abstract_operator_footprints,
                                &all_tcf,
                                deadline,
                                "failed to reduce abstract-operator All residual costs",
                            )? {
                                info!(
                                    "scp_online: abstract-operator All abstraction {pos} stopped while reducing residual costs (deadline)"
                                );
                                break;
                            }
                            cp.add_h_values(pos, all_table.distances);
                            log_transition_residual_summary(&remaining_costs);
                        }
                    }
                } else if let Some(abstraction) =
                    self.cartesian_abstraction_for_component(pos, num_domain_abstractions)
                {
                    let result = self.add_transition_cartesian_step(
                        &mut cp,
                        &mut remaining_costs,
                        pos,
                        &abstraction,
                        abstract_state_ids,
                        deadline,
                    );
                    match result {
                        Ok(true) => {}
                        Ok(false) => {
                            info!(
                                "scp_online: Cartesian abstract-operator abstraction {pos} stopped while reducing residual costs (deadline)"
                            );
                            break;
                        }
                        Err(error) if Self::is_online_deadline_error_eval(&error) => {
                            info!(
                                "scp_online: Cartesian abstract-operator abstraction {pos} stopped while computing table (deadline)"
                            );
                            break;
                        }
                        Err(error) => return Err(error),
                    }
                } else {
                    self.add_transition_pdb_step(
                        &mut cp,
                        &mut remaining_costs,
                        pos,
                        order,
                        abstract_state_ids,
                        self.pdb_offset(num_domain_abstractions),
                    )?;
                }
            }
        }

        Ok(cp)
    }

    fn log_abstract_operator_label_diagnostic(
        &self,
        abstraction: &DomainAbstraction,
        abstraction_task: &dyn AbstractNumericTask,
        abstraction_id: usize,
        abstract_state_ids: &[Option<usize>],
        remaining_costs: &TransitionResidualCosts,
    ) -> Result<(), EvaluationError> {
        if !self.config.collection_config.debug || !enabled!(Level::INFO) {
            return Ok(());
        }
        let label_remaining_costs = remaining_costs.operator_costs_for_label_cp();
        let (label_distances, label_saturated) = Self::compute_domain_cp_entry(
            abstraction,
            abstraction_task,
            self.config.combine_labels,
            &label_remaining_costs,
            None,
        )?;
        let label_h = current_h_for_distances(abstraction_id, &label_distances, abstract_state_ids);
        let (positive_labels, total_label_saturated) = positive_cost_stats(&label_saturated);
        let stats = abstract_operator_footprint_stats(&abstraction.abstract_operator_footprints);
        info!(
            "scp_online: abstract-operator label diagnostic abstraction {abstraction_id}: label_equivalent_h={label_h}, positive_saturated_labels={positive_labels}, total_label_saturated={total_label_saturated:.6}, footprint_labels={}, bounded_footprint_labels={}",
            stats.total_labels, stats.bounded_labels,
        );
        log_positive_label_footprint_diagnostics(
            abstraction_id,
            abstraction_task,
            &abstraction.abstract_operator_footprints,
            &label_saturated,
        );
        Ok(())
    }

    fn add_transition_cartesian_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut TransitionResidualCosts,
        pos: usize,
        abstraction: &CartesianAbstraction,
        abstract_state_ids: &[Option<usize>],
        deadline: Option<Instant>,
    ) -> Result<bool, EvaluationError> {
        let cap_state_id = abstract_state_ids
            .get(pos)
            .copied()
            .flatten()
            .ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "missing Cartesian abstract state id for abstract-operator CP component {pos}"
                ))
            })?;

        let build = |residual_costs: &TransitionResidualCosts, cap| {
            build_explicit_abstract_operator_cost_partitioning_table(
                &abstraction.transition_system,
                &abstraction.abstract_operator_footprints,
                residual_costs,
                pos,
                cap,
                deadline,
            )
            .map_err(|error| {
                Self::construction_error(
                    &format!(
                        "failed to compute Cartesian abstract-operator CP table for component {pos}"
                    ),
                    error,
                )
            })
        };
        let reduce = |residual_costs: &mut TransitionResidualCosts,
                      tcf: &AbstractOperatorCostFunction| {
            Self::reduce_abstract_operator_costs(
                residual_costs,
                pos,
                &abstraction.abstract_operator_footprints,
                tcf,
                deadline,
                &format!(
                    "failed to reduce Cartesian abstract-operator residual costs for component {pos}"
                ),
            )
        };

        match self.config.saturator {
            Saturator::All => {
                let (distances, tcf) = build(remaining_costs, None)?;
                if !should_skip_zero_current_table(
                    "Cartesian abstract-operator all",
                    pos,
                    &distances,
                    abstract_state_ids,
                ) {
                    if !reduce(remaining_costs, &tcf)? {
                        return Ok(false);
                    }
                    cp.add_h_values(pos, distances);
                }
            }
            Saturator::Perim => {
                let (distances, tcf) = build(remaining_costs, Some(cap_state_id))?;
                if !should_skip_zero_current_table(
                    "Cartesian abstract-operator perim",
                    pos,
                    &distances,
                    abstract_state_ids,
                ) {
                    if !reduce(remaining_costs, &tcf)? {
                        return Ok(false);
                    }
                    cp.add_h_values(pos, distances);
                }
            }
            Saturator::Perimstar => {
                let (perim_distances, perim_tcf) = build(remaining_costs, Some(cap_state_id))?;
                if !should_skip_zero_current_table(
                    "Cartesian abstract-operator perimstar/perim",
                    pos,
                    &perim_distances,
                    abstract_state_ids,
                ) {
                    if !reduce(remaining_costs, &perim_tcf)? {
                        return Ok(false);
                    }
                    cp.add_h_values(pos, perim_distances);
                }
                let (all_distances, all_tcf) = build(remaining_costs, None)?;
                if !should_skip_zero_current_table(
                    "Cartesian abstract-operator perimstar/all",
                    pos,
                    &all_distances,
                    abstract_state_ids,
                ) {
                    if !reduce(remaining_costs, &all_tcf)? {
                        return Ok(false);
                    }
                    cp.add_h_values(pos, all_distances);
                }
            }
        }
        Ok(true)
    }

    fn add_transition_pdb_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut TransitionResidualCosts,
        pos: usize,
        _order: &[usize],
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
    ) -> Result<(), EvaluationError> {
        let pdb_id = pos.checked_sub(num_domain_abstractions).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "PDB component {pos} precedes PDB offset {num_domain_abstractions}"
            ))
        })?;
        let pdb = self.pdbs.get(pdb_id).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "abstract-operator CP component {pos} references missing PDB {pdb_id}"
            ))
        })?;

        let mut remaining_operator_costs = remaining_costs.operator_costs_for_label_cp();
        match self.config.saturator {
            Saturator::All => {
                let (distances, saturated) = pdb
                    .build_cost_partitioned_distance_table(&remaining_operator_costs)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB SCP table {pdb_id}: {error}"
                        ))
                    })?;
                cp.add_pdb_h_values(pos, distances);
                remaining_costs
                    .reduce_operator_costs_uniform(&saturated)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to reduce PDB residual costs: {error:#}"
                        ))
                    })?;
            }
            Saturator::Perim => {
                let h_cap = Self::pdb_current_h_cap(
                    pdb,
                    &remaining_operator_costs,
                    pos,
                    abstract_state_ids,
                )?;
                let (distances, saturated) = pdb
                    .build_cost_partitioned_distance_table_capped(&remaining_operator_costs, h_cap)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB PERIM table {pdb_id}: {error}"
                        ))
                    })?;
                cp.add_pdb_h_values(pos, distances);
                remaining_costs
                    .reduce_operator_costs_uniform(&saturated)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to reduce PDB PERIM residual costs: {error:#}"
                        ))
                    })?;
            }
            Saturator::Perimstar => {
                let h_cap = Self::pdb_current_h_cap(
                    pdb,
                    &remaining_operator_costs,
                    pos,
                    abstract_state_ids,
                )?;
                let (perim_dists, perim_sat) = pdb
                    .build_cost_partitioned_distance_table_capped(&remaining_operator_costs, h_cap)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB Perim step for Perimstar {pdb_id}: {error}"
                        ))
                    })?;
                cp.add_pdb_h_values(pos, perim_dists);
                remaining_costs
                    .reduce_operator_costs_uniform(&perim_sat)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to reduce PDB Perim residual costs: {error:#}"
                        ))
                    })?;

                remaining_operator_costs = remaining_costs.operator_costs_for_label_cp();
                let (all_dists, all_sat) = pdb
                    .build_cost_partitioned_distance_table(&remaining_operator_costs)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB All step for Perimstar {pdb_id}: {error}"
                        ))
                    })?;
                cp.add_pdb_h_values(pos, all_dists);
                remaining_costs
                    .reduce_operator_costs_uniform(&all_sat)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to reduce PDB All residual costs: {error:#}"
                        ))
                    })?;
            }
        }

        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn build_label_cp(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        order: &[usize],
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        deadline: Option<Instant>,
    ) -> Result<CostPartitioningHeuristic, EvaluationError> {
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs: Vec<f64> = original_costs.to_vec();

        for &pos in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if pos < num_domain_abstractions {
                let abstraction = &abstractions[pos];
                debug!(
                    "scp_online: label CP step abstraction {pos}, abstract_states={}",
                    abstraction_state_count(abstraction)
                );
                match self.config.saturator {
                    Saturator::All => {
                        let abstraction_task = abstraction.task_for_factory(task);
                        let (distances, saturated) = match Self::compute_domain_cp_entry(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            deadline,
                        ) {
                            Ok(entry) => entry,
                            Err(error) if Self::is_online_deadline_error_eval(&error) => {
                                info!(
                                    "scp_online: label all abstraction {pos} stopped while computing table (deadline)"
                                );
                                break;
                            }
                            Err(error) => return Err(error),
                        };
                        log_label_table_summary(
                            "all",
                            pos,
                            &distances,
                            &saturated,
                            abstract_state_ids,
                        );
                        if should_skip_zero_current_table(
                            "label all",
                            pos,
                            &distances,
                            abstract_state_ids,
                        ) {
                            continue;
                        }
                        cp.add_h_values(pos, distances);
                        reduce_costs(&mut remaining_costs, &saturated)?;
                    }
                    Saturator::Perim => {
                        let abstraction_task = abstraction.task_for_factory(task);
                        let h_cap = Self::domain_current_h_cap(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            pos,
                            abstract_state_ids,
                        )?;
                        let (distances, saturated) = Self::compute_domain_perim_entry(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            h_cap,
                        )?;
                        log_label_table_summary(
                            "perim",
                            pos,
                            &distances,
                            &saturated,
                            abstract_state_ids,
                        );
                        if should_skip_zero_current_table(
                            "label perim",
                            pos,
                            &distances,
                            abstract_state_ids,
                        ) {
                            continue;
                        }
                        cp.add_h_values(pos, distances);
                        reduce_costs(&mut remaining_costs, &saturated)?;
                    }
                    Saturator::Perimstar => {
                        let abstraction_task = abstraction.task_for_factory(task);
                        let h_cap = Self::domain_current_h_cap(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            pos,
                            abstract_state_ids,
                        )?;
                        let (perim_distances, perim_saturated) = Self::compute_domain_perim_entry(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            h_cap,
                        )?;
                        log_label_table_summary(
                            "perimstar/perim",
                            pos,
                            &perim_distances,
                            &perim_saturated,
                            abstract_state_ids,
                        );
                        if !should_skip_zero_current_table(
                            "label perimstar/perim",
                            pos,
                            &perim_distances,
                            abstract_state_ids,
                        ) {
                            cp.add_h_values(pos, perim_distances);
                            reduce_costs(&mut remaining_costs, &perim_saturated)?;
                        }

                        let (all_distances, all_saturated) = Self::compute_domain_cp_entry(
                            abstraction,
                            abstraction_task,
                            self.config.combine_labels,
                            &remaining_costs,
                            deadline,
                        )?;
                        log_label_table_summary(
                            "perimstar/all",
                            pos,
                            &all_distances,
                            &all_saturated,
                            abstract_state_ids,
                        );
                        if should_skip_zero_current_table(
                            "label perimstar/all",
                            pos,
                            &all_distances,
                            abstract_state_ids,
                        ) {
                            continue;
                        }
                        cp.add_h_values(pos, all_distances);
                        reduce_costs(&mut remaining_costs, &all_saturated)?;
                    }
                }
            } else if let Some(abstraction) =
                self.cartesian_abstraction_for_component(pos, num_domain_abstractions)
            {
                let result = self.add_label_cartesian_step(
                    &mut cp,
                    &mut remaining_costs,
                    pos,
                    &abstraction,
                    abstract_state_ids,
                    deadline,
                );
                match result {
                    Ok(()) => {}
                    Err(error) if Self::is_online_deadline_error_eval(&error) => {
                        info!(
                            "scp_online: Cartesian label abstraction {pos} stopped while computing table (deadline)"
                        );
                        break;
                    }
                    Err(error) => return Err(error),
                }
            } else {
                self.add_label_pdb_step(
                    &mut cp,
                    &mut remaining_costs,
                    pos,
                    abstract_state_ids,
                    self.pdb_offset(num_domain_abstractions),
                )?;
            }
        }

        Ok(cp)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_label_fill_scp(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        order: &[usize],
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        deadline: Option<Instant>,
    ) -> Result<(CostPartitioningHeuristic, Vec<f64>), EvaluationError> {
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs: Vec<f64> = original_costs.to_vec();

        for &pos in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if pos >= num_domain_abstractions {
                let abstraction = self
                    .cartesian_abstraction_for_component(pos, num_domain_abstractions)
                    .ok_or_else(|| {
                        EvaluationError::InvalidState(format!(
                            "fillSCP order references unsupported component {pos}"
                        ))
                    })?;
                self.add_label_cartesian_step(
                    &mut cp,
                    &mut remaining_costs,
                    pos,
                    &abstraction,
                    abstract_state_ids,
                    deadline,
                )?;
                continue;
            }
            let abstraction = &abstractions[pos];
            debug!(
                "fillSCP: label CP step abstraction {pos}, abstract_states={}",
                abstraction_state_count(abstraction)
            );
            match self.config.saturator {
                Saturator::All => {
                    let abstraction_task = abstraction.task_for_factory(task);
                    let (distances, saturated) = Self::compute_domain_cp_entry(
                        abstraction,
                        abstraction_task,
                        self.config.combine_labels,
                        &remaining_costs,
                        deadline,
                    )?;
                    log_label_table_summary(
                        "fillSCP/all",
                        pos,
                        &distances,
                        &saturated,
                        abstract_state_ids,
                    );
                    if should_skip_zero_current_table(
                        "fillSCP label all",
                        pos,
                        &distances,
                        abstract_state_ids,
                    ) {
                        continue;
                    }
                    cp.add_h_values(pos, distances);
                    reduce_costs(&mut remaining_costs, &saturated)?;
                }
                Saturator::Perim => {
                    let abstraction_task = abstraction.task_for_factory(task);
                    let h_cap = Self::domain_current_h_cap(
                        abstraction,
                        abstraction_task,
                        self.config.combine_labels,
                        &remaining_costs,
                        pos,
                        abstract_state_ids,
                    )?;
                    let (distances, saturated) = Self::compute_domain_perim_entry(
                        abstraction,
                        abstraction_task,
                        self.config.combine_labels,
                        &remaining_costs,
                        h_cap,
                    )?;
                    log_label_table_summary(
                        "fillSCP/perim",
                        pos,
                        &distances,
                        &saturated,
                        abstract_state_ids,
                    );
                    if should_skip_zero_current_table(
                        "fillSCP label perim",
                        pos,
                        &distances,
                        abstract_state_ids,
                    ) {
                        continue;
                    }
                    cp.add_h_values(pos, distances);
                    reduce_costs(&mut remaining_costs, &saturated)?;
                }
                Saturator::Perimstar => {
                    let abstraction_task = abstraction.task_for_factory(task);
                    let h_cap = Self::domain_current_h_cap(
                        abstraction,
                        abstraction_task,
                        self.config.combine_labels,
                        &remaining_costs,
                        pos,
                        abstract_state_ids,
                    )?;
                    let (perim_distances, perim_saturated) = Self::compute_domain_perim_entry(
                        abstraction,
                        abstraction_task,
                        self.config.combine_labels,
                        &remaining_costs,
                        h_cap,
                    )?;
                    if !should_skip_zero_current_table(
                        "fillSCP label perimstar/perim",
                        pos,
                        &perim_distances,
                        abstract_state_ids,
                    ) {
                        cp.add_h_values(pos, perim_distances);
                        reduce_costs(&mut remaining_costs, &perim_saturated)?;
                    }

                    let (all_distances, all_saturated) = Self::compute_domain_cp_entry(
                        abstraction,
                        abstraction_task,
                        self.config.combine_labels,
                        &remaining_costs,
                        deadline,
                    )?;
                    if should_skip_zero_current_table(
                        "fillSCP label perimstar/all",
                        pos,
                        &all_distances,
                        abstract_state_ids,
                    ) {
                        continue;
                    }
                    cp.add_h_values(pos, all_distances);
                    reduce_costs(&mut remaining_costs, &all_saturated)?;
                }
            }
        }

        Ok((cp, remaining_costs))
    }

    #[allow(clippy::too_many_arguments)]
    fn build_abstract_operator_fill_scp(
        &self,
        task: &dyn AbstractNumericTask,
        abstractions: &[DomainAbstraction],
        order: &[usize],
        abstract_state_ids: &[Option<usize>],
        _standalone_current_h: &[f64],
        num_domain_abstractions: usize,
        original_costs: &[f64],
        deadline: Option<Instant>,
        saturator: Saturator,
    ) -> Result<
        (
            CostPartitioningHeuristic,
            Vec<f64>,
            Vec<LmCutResidualOperatorCostPartition>,
        ),
        EvaluationError,
    > {
        let mut cp = CostPartitioningHeuristic::default();
        let mut remaining_costs = TransitionResidualCosts::from_operator_costs(original_costs);

        for &pos in order {
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                break;
            }
            if pos >= num_domain_abstractions {
                let abstraction = self
                    .cartesian_abstraction_for_component(pos, num_domain_abstractions)
                    .ok_or_else(|| {
                        EvaluationError::InvalidState(format!(
                            "fillSCP abstract-operator order references unsupported component {pos}"
                        ))
                    })?;
                self.add_transition_cartesian_step(
                    &mut cp,
                    &mut remaining_costs,
                    pos,
                    &abstraction,
                    abstract_state_ids,
                    deadline,
                )?;
                continue;
            }
            let abstraction = &abstractions[pos];
            debug!(
                "fillSCP: abstract-operator CP step abstraction {pos}, abstract_states={}, metadata={}",
                abstraction_state_count(abstraction),
                abstraction_metadata_summary(abstraction),
            );
            log_abstract_operator_footprint_summary(pos, &abstraction.abstract_operator_footprints);
            let abstraction_task = abstraction.task_for_factory(task);
            match saturator {
                Saturator::All => {
                    let (table, tcf) = abstraction
                        .factory
                        .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                            abstraction_task,
                            abstraction.combine_labels,
                            &abstraction.abstract_operators,
                            &abstraction.abstract_operator_footprints,
                            &remaining_costs,
                            pos,
                            abstract_state_ids.get(pos).copied().flatten(),
                            None,
                            deadline,
                        )
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute fillSCP abstract-operator table: {error:#}"
                            ))
                        })?;
                    log_transition_table_summary(
                        "fillSCP/all",
                        pos,
                        &table.distances,
                        &tcf.operator_costs,
                        abstract_state_ids,
                    );
                    if should_skip_zero_current_table(
                        "fillSCP abstract-operator all",
                        pos,
                        &table.distances,
                        abstract_state_ids,
                    ) {
                        continue;
                    }
                    if !Self::reduce_abstract_operator_costs(
                        &mut remaining_costs,
                        pos,
                        &abstraction.abstract_operator_footprints,
                        &tcf,
                        deadline,
                        "failed to reduce fillSCP abstract-operator residual costs",
                    )? {
                        info!(
                            "fillSCP: abstract-operator abstraction {pos} stopped while reducing residual costs (deadline)"
                        );
                        break;
                    }
                    cp.add_h_values(pos, table.distances);
                    log_transition_residual_summary(&remaining_costs);
                }
                Saturator::Perim => {
                    let cap_state_id = abstract_state_ids.get(pos).copied().flatten();
                    let (table, tcf) = abstraction
                        .factory
                        .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                            abstraction_task,
                            abstraction.combine_labels,
                            &abstraction.abstract_operators,
                            &abstraction.abstract_operator_footprints,
                            &remaining_costs,
                            pos,
                            cap_state_id,
                            cap_state_id,
                            deadline,
                        )
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute fillSCP abstract-operator PERIM table: {error:#}"
                            ))
                        })?;
                    log_transition_table_summary(
                        "fillSCP/perim",
                        pos,
                        &table.distances,
                        &tcf.operator_costs,
                        abstract_state_ids,
                    );
                    if should_skip_zero_current_table(
                        "fillSCP abstract-operator perim",
                        pos,
                        &table.distances,
                        abstract_state_ids,
                    ) {
                        continue;
                    }
                    if !Self::reduce_abstract_operator_costs(
                        &mut remaining_costs,
                        pos,
                        &abstraction.abstract_operator_footprints,
                        &tcf,
                        deadline,
                        "failed to reduce fillSCP abstract-operator PERIM residual costs",
                    )? {
                        info!(
                            "fillSCP: abstract-operator PERIM abstraction {pos} stopped while reducing residual costs (deadline)"
                        );
                        break;
                    }
                    cp.add_h_values(pos, table.distances);
                    log_transition_residual_summary(&remaining_costs);
                }
                Saturator::Perimstar => {
                    let cap_state_id = abstract_state_ids.get(pos).copied().flatten();
                    let (perim_table, perim_tcf) = abstraction
                        .factory
                        .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                            abstraction_task,
                            abstraction.combine_labels,
                            &abstraction.abstract_operators,
                            &abstraction.abstract_operator_footprints,
                            &remaining_costs,
                            pos,
                            cap_state_id,
                            cap_state_id,
                            deadline,
                        )
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute fillSCP abstract-operator Perim step for Perimstar: {error:#}"
                            ))
                        })?;
                    if !should_skip_zero_current_table(
                        "fillSCP abstract-operator perimstar/perim",
                        pos,
                        &perim_table.distances,
                        abstract_state_ids,
                    ) {
                        if !Self::reduce_abstract_operator_costs(
                            &mut remaining_costs,
                            pos,
                            &abstraction.abstract_operator_footprints,
                            &perim_tcf,
                            deadline,
                            "failed to reduce fillSCP abstract-operator Perim residual costs",
                        )? {
                            info!(
                                "fillSCP: abstract-operator Perim abstraction {pos} stopped while reducing residual costs (deadline)"
                            );
                            break;
                        }
                        cp.add_h_values(pos, perim_table.distances);
                        log_transition_residual_summary(&remaining_costs);
                    }

                    let (all_table, all_tcf) = abstraction
                        .factory
                        .build_abstract_operator_cost_partitioned_distance_table_with_operators_and_footprints_with_deadline(
                            abstraction_task,
                            abstraction.combine_labels,
                            &abstraction.abstract_operators,
                            &abstraction.abstract_operator_footprints,
                            &remaining_costs,
                            pos,
                            cap_state_id,
                            None,
                            deadline,
                        )
                        .map_err(|error| {
                            EvaluationError::ComputationFailed(format!(
                                "failed to compute fillSCP abstract-operator All step for Perimstar: {error:#}"
                            ))
                        })?;
                    if should_skip_zero_current_table(
                        "fillSCP abstract-operator perimstar/all",
                        pos,
                        &all_table.distances,
                        abstract_state_ids,
                    ) {
                        continue;
                    }
                    if !Self::reduce_abstract_operator_costs(
                        &mut remaining_costs,
                        pos,
                        &abstraction.abstract_operator_footprints,
                        &all_tcf,
                        deadline,
                        "failed to reduce fillSCP abstract-operator All residual costs",
                    )? {
                        info!(
                            "fillSCP: abstract-operator All abstraction {pos} stopped while reducing residual costs (deadline)"
                        );
                        break;
                    }
                    cp.add_h_values(pos, all_table.distances);
                    log_transition_residual_summary(&remaining_costs);
                }
            }
        }

        let residual_partitions = remaining_costs.operator_cost_partitions_for_lmcut(4, 4);
        Ok((
            cp,
            remaining_costs.operator_costs_for_label_cp(),
            residual_partitions,
        ))
    }

    fn add_label_cartesian_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut [f64],
        pos: usize,
        abstraction: &CartesianAbstraction,
        abstract_state_ids: &[Option<usize>],
        deadline: Option<Instant>,
    ) -> Result<(), EvaluationError> {
        let cap_state_id = abstract_state_ids
            .get(pos)
            .copied()
            .flatten()
            .ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "missing Cartesian abstract state id for CP component {pos}"
                ))
            })?;
        let build = |costs: &[f64], cap| {
            build_explicit_label_cost_partitioning_table(
                &abstraction.transition_system,
                costs,
                cap,
                deadline,
            )
            .map_err(|error| {
                Self::construction_error(
                    &format!("failed to compute Cartesian label CP table for component {pos}"),
                    error,
                )
            })
        };

        match self.config.saturator {
            Saturator::All => {
                let (distances, saturated) = build(remaining_costs, None)?;
                if !should_skip_zero_current_table(
                    "Cartesian label all",
                    pos,
                    &distances,
                    abstract_state_ids,
                ) {
                    cp.add_h_values(pos, distances);
                    reduce_costs(remaining_costs, &saturated)?;
                }
            }
            Saturator::Perim => {
                let (distances, saturated) = build(remaining_costs, Some(cap_state_id))?;
                if !should_skip_zero_current_table(
                    "Cartesian label perim",
                    pos,
                    &distances,
                    abstract_state_ids,
                ) {
                    cp.add_h_values(pos, distances);
                    reduce_costs(remaining_costs, &saturated)?;
                }
            }
            Saturator::Perimstar => {
                let (perim_distances, perim_saturated) =
                    build(remaining_costs, Some(cap_state_id))?;
                if !should_skip_zero_current_table(
                    "Cartesian label perimstar/perim",
                    pos,
                    &perim_distances,
                    abstract_state_ids,
                ) {
                    cp.add_h_values(pos, perim_distances);
                    reduce_costs(remaining_costs, &perim_saturated)?;
                }
                let (all_distances, all_saturated) = build(remaining_costs, None)?;
                if !should_skip_zero_current_table(
                    "Cartesian label perimstar/all",
                    pos,
                    &all_distances,
                    abstract_state_ids,
                ) {
                    cp.add_h_values(pos, all_distances);
                    reduce_costs(remaining_costs, &all_saturated)?;
                }
            }
        }
        Ok(())
    }

    fn add_label_pdb_step(
        &self,
        cp: &mut CostPartitioningHeuristic,
        remaining_costs: &mut [f64],
        pos: usize,
        abstract_state_ids: &[Option<usize>],
        num_domain_abstractions: usize,
    ) -> Result<(), EvaluationError> {
        let pdb_id = pos.checked_sub(num_domain_abstractions).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "PDB component {pos} precedes PDB offset {num_domain_abstractions}"
            ))
        })?;
        let pdb = self.pdbs.get(pdb_id).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "label CP component {pos} references missing PDB {pdb_id}"
            ))
        })?;

        match self.config.saturator {
            Saturator::All => {
                let (distances, saturated) = pdb
                    .build_cost_partitioned_distance_table(remaining_costs)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB SCP table {pdb_id}: {error}"
                        ))
                    })?;
                if should_skip_zero_current_table("pdb all", pos, &distances, abstract_state_ids) {
                    return Ok(());
                }
                cp.add_pdb_h_values(pos, distances);
                reduce_costs(remaining_costs, &saturated)?;
            }
            Saturator::Perim => {
                let h_cap = Self::pdb_current_h_cap(pdb, remaining_costs, pos, abstract_state_ids)?;
                let (distances, saturated) = pdb
                    .build_cost_partitioned_distance_table_capped(remaining_costs, h_cap)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB PERIM table {pdb_id}: {error}"
                        ))
                    })?;
                if should_skip_zero_current_table("pdb perim", pos, &distances, abstract_state_ids)
                {
                    return Ok(());
                }
                cp.add_pdb_h_values(pos, distances);
                reduce_costs(remaining_costs, &saturated)?;
            }
            Saturator::Perimstar => {
                let h_cap = Self::pdb_current_h_cap(pdb, remaining_costs, pos, abstract_state_ids)?;
                let (perim_dists, perim_sat) = pdb
                    .build_cost_partitioned_distance_table_capped(remaining_costs, h_cap)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB Perim step for Perimstar {pdb_id}: {error}"
                        ))
                    })?;
                if !should_skip_zero_current_table(
                    "pdb perimstar/perim",
                    pos,
                    &perim_dists,
                    abstract_state_ids,
                ) {
                    cp.add_pdb_h_values(pos, perim_dists);
                    reduce_costs(remaining_costs, &perim_sat)?;
                }

                let (all_dists, all_sat) = pdb
                    .build_cost_partitioned_distance_table(remaining_costs)
                    .map_err(|error| {
                        EvaluationError::ComputationFailed(format!(
                            "failed to compute PDB All step for Perimstar {pdb_id}: {error}"
                        ))
                    })?;
                if should_skip_zero_current_table(
                    "pdb perimstar/all",
                    pos,
                    &all_dists,
                    abstract_state_ids,
                ) {
                    return Ok(());
                }
                cp.add_pdb_h_values(pos, all_dists);
                reduce_costs(remaining_costs, &all_sat)?;
            }
        }

        Ok(())
    }

    fn retain_cp(
        state: &mut ScpOnlineState,
        cp: CostPartitioningHeuristic,
        abstract_state_ids: &[Option<usize>],
        max_h: &mut f64,
        retain_alternative: bool,
        max_size_kb: usize,
    ) {
        if state.cp_heuristics.contains(&cp) {
            info!(
                "scp_online: discarded duplicate CP with {} lookup tables",
                cp.lookup_tables.len()
            );
            return;
        }
        let new_h = cp.compute_heuristic(abstract_state_ids);
        let improves_current_state = new_h > *max_h;
        let size_kb = cp.estimate_size_in_kb();
        let fits_size_limit =
            state.cp_heuristics.is_empty() || state.size_kb.saturating_add(size_kb) <= max_size_kb;
        if (improves_current_state || retain_alternative) && fits_size_limit {
            let component_values = cp
                .lookup_tables
                .iter()
                .map(|table| {
                    let value = abstract_state_ids
                        .get(table.abstraction_id)
                        .copied()
                        .flatten()
                        .and_then(|state_id| table.distances.get(state_id))
                        .copied()
                        .unwrap_or(table.unknown_value);
                    (table.abstraction_id, value)
                })
                .collect::<Vec<_>>();
            info!(
                "scp_online: retained CP, current-state h {} -> {}, lookup_tables={}, components={:?}, size={} KiB, alternative={}",
                *max_h,
                new_h,
                cp.lookup_tables.len(),
                component_values,
                size_kb,
                !improves_current_state,
            );
            state.size_kb = state.size_kb.saturating_add(size_kb);
            state.cp_heuristics.push(cp);
            state.required_lookup_ids = Self::required_lookup_ids(state);
            *max_h = (*max_h).max(new_h);
        } else if !fits_size_limit {
            info!(
                "scp_online: discarded CP because storing {} KiB would exceed max_size={} KiB (stored={} KiB)",
                size_kb, max_size_kb, state.size_kb
            );
        } else {
            info!(
                "scp_online: rejected CP, candidate_h={} did not improve current_h={}, lookup_tables={}",
                new_h,
                *max_h,
                cp.lookup_tables.len(),
            );
        }
    }

    fn domain_current_h_cap(
        abstraction: &DomainAbstraction,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        remaining_costs: &[f64],
        component_id: usize,
        abstract_state_ids: &[Option<usize>],
    ) -> Result<f64, EvaluationError> {
        let state_id = abstract_state_ids
            .get(component_id)
            .ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "domain component {component_id} has no state-ID slot"
                ))
            })?
            .ok_or_else(|| {
                EvaluationError::InvalidState(format!(
                    "domain component {component_id} has no exact abstract state ID"
                ))
            })?;
        let table = abstraction
            .factory
            .build_goal_distances_for_goals(
                task,
                combine_labels,
                remaining_costs,
                &abstraction.distance_table.goal_facts,
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute current h cap for domain component {component_id}: {error:#}"
                ))
            })?;
        table.distances.get(state_id).copied().ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "domain component {component_id} state ID {state_id} is out of bounds for {} states",
                table.distances.len()
            ))
        })
    }

    fn pdb_current_h_cap(
        pdb: &PatternDatabase<'_>,
        remaining_costs: &[f64],
        component_id: usize,
        abstract_state_ids: &[Option<usize>],
    ) -> Result<f64, EvaluationError> {
        let state_id = *abstract_state_ids.get(component_id).ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "PDB component {component_id} has no state-ID slot"
            ))
        })?;
        let Some(state_id) = state_id else {
            // A truncated PDB need not contain the exact projected state. Its
            // standalone fallback remains admissible, but there is no finite
            // state-specific perimeter cap to apply.
            return Ok(f64::INFINITY);
        };
        let distances = pdb.build_goal_distances(remaining_costs).map_err(|error| {
            EvaluationError::ComputationFailed(format!(
                "failed to compute current h cap for PDB component {component_id}: {error}"
            ))
        })?;
        distances.get(state_id).copied().ok_or_else(|| {
            EvaluationError::InvalidState(format!(
                "PDB component {component_id} state ID {state_id} is out of bounds for {} states",
                distances.len()
            ))
        })
    }

    /// Build a single SCP for one domain abstraction using its own factory.
    fn compute_domain_cp_entry(
        abstraction: &DomainAbstraction,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        remaining_costs: &[f64],
        deadline: Option<Instant>,
    ) -> Result<(Vec<f64>, Vec<f64>), EvaluationError> {
        let start = Instant::now();
        let (table, saturated) = abstraction
            .factory
            .build_cost_partitioned_distance_table_for_goals_with_deadline(
                task,
                combine_labels,
                remaining_costs,
                false,
                &abstraction.distance_table.goal_facts,
                deadline,
            )
            .map_err(|error| Self::construction_error("failed to compute SCP table", error))?;
        debug!(
            "scp_online: label distance-table/CP construction finished in {:.3}s, states={}, saturated_costs={}",
            start.elapsed().as_secs_f64(),
            table.distances.len(),
            saturated.len()
        );
        Ok((table.distances, saturated))
    }

    /// Build a PERIM domain CP entry.
    ///
    /// Saturated costs are computed from a table where states outside the
    /// perimeter are inactive. The returned lookup table is then recomputed
    /// globally from those saturated costs, so stored online CPs remain valid
    /// for later states outside the original perimeter.
    fn compute_domain_perim_entry(
        abstraction: &DomainAbstraction,
        task: &dyn AbstractNumericTask,
        combine_labels: bool,
        remaining_costs: &[f64],
        h_cap: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), EvaluationError> {
        let (table, _) = abstraction
            .factory
            .build_cost_partitioned_distance_table_for_goals(
                task,
                combine_labels,
                remaining_costs,
                false,
                &abstraction.distance_table.goal_facts,
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute SCP table for PERIM: {error:#}"
                ))
            })?;
        let mut perim_distances = table.distances;
        if h_cap.is_finite() {
            for h in &mut perim_distances {
                if !h.is_finite() || *h > h_cap {
                    *h = f64::NEG_INFINITY;
                }
            }
        }
        let mut generator = abstraction
            .factory
            .make_operator_generator(task, combine_labels)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to create operator generator for PERIM: {error:#}"
                ))
            })?;
        let mut operators = generator.build_abstract_operators(task).map_err(|error| {
            EvaluationError::ComputationFailed(format!(
                "failed to build abstract operators for PERIM: {error:#}"
            ))
        })?;
        apply_operator_costs_from_slice(&mut operators, remaining_costs)?;
        let perim_table = AbstractDistanceTable {
            distances: perim_distances,
            generating_op_ids: table.generating_op_ids,
            initial_state_hash: table.initial_state_hash,
            goal_facts: table.goal_facts,
            hash_multipliers: table.hash_multipliers,
            numeric_domain_sizes: table.numeric_domain_sizes,
        };
        let saturated = abstraction
            .factory
            .saturated_costs_for_table(task, combine_labels, &operators, &perim_table)
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute PERIM saturated costs: {error:#}"
                ))
            })?;
        let global_table = abstraction
            .factory
            .build_goal_distances_for_goals(
                task,
                combine_labels,
                &saturated,
                &abstraction.distance_table.goal_facts,
            )
            .map_err(|error| {
                EvaluationError::ComputationFailed(format!(
                    "failed to compute global PERIM lookup table: {error:#}"
                ))
            })?;
        Ok((global_table.distances, saturated))
    }
}

fn abstraction_state_count(abstraction: &DomainAbstraction) -> u128 {
    abstraction
        .factory
        .domain_sizes()
        .iter()
        .chain(abstraction.factory.numeric_domain_sizes().iter())
        .fold(1_u128, |acc, &size| acc.saturating_mul(size as u128))
}

fn abstraction_collection_iteration(
    abstractions: &[DomainAbstraction],
    abstraction_id: usize,
) -> usize {
    abstractions
        .get(abstraction_id)
        .and_then(|abstraction| abstraction.metadata.collection_iteration)
        .unwrap_or(usize::MAX)
}

#[derive(Debug, Clone, Copy, Default)]
struct GoalCoverOrderVariant {
    anchor_goal_offset: usize,
    anchor_offset: usize,
    complementary_round: usize,
    representative_round: usize,
    non_goal_prefix: bool,
    compact: bool,
}

fn compact_goal_cover_variants(
    goal_count: usize,
    variants_per_goal: usize,
    guarantee_specialist_coverage: bool,
) -> Vec<GoalCoverOrderVariant> {
    assert!(goal_count > 0);
    assert!(variants_per_goal > 0);
    let pairwise_variant_count = goal_count
        .saturating_mul(variants_per_goal)
        .saturating_mul(variants_per_goal)
        .min(64);
    let specialist_coverage_count = guarantee_specialist_coverage
        .then(|| goal_count.saturating_mul(variants_per_goal.min(4)))
        .unwrap_or(0);
    let variant_count = pairwise_variant_count.max(specialist_coverage_count);
    (0..variant_count)
        .map(|variant_index| {
            let anchor_goal_offset = variant_index % goal_count;
            let anchor_round = variant_index / goal_count;
            let anchor_offset = anchor_round % variants_per_goal;
            let representative_round = (anchor_round / variants_per_goal) % variants_per_goal;
            GoalCoverOrderVariant {
                anchor_goal_offset,
                anchor_offset,
                complementary_round: anchor_offset.wrapping_add(representative_round),
                representative_round,
                compact: true,
                ..Default::default()
            }
        })
        .collect()
}

fn cartesian_goal_cover_order(
    base_order: &[usize],
    num_domain_abstractions: usize,
    cartesian_abstractions: &[CartesianAbstraction],
    standalone_current_h: &[f64],
    require_pure_cartesian_collection: bool,
    variant: GoalCoverOrderVariant,
) -> Option<Vec<usize>> {
    let cartesian_end = num_domain_abstractions + cartesian_abstractions.len();
    let is_goal_cartesian = |component_id: usize| {
        component_id
            .checked_sub(num_domain_abstractions)
            .and_then(|id| cartesian_abstractions.get(id))
            .is_some_and(|abstraction| abstraction.metadata.collection_goal_id.is_some())
    };
    let pure_cartesian_collection = base_order
        .iter()
        .copied()
        .all(|id| id < cartesian_end && is_goal_cartesian(id));
    if (require_pure_cartesian_collection || variant.compact) && !pure_cartesian_collection {
        return None;
    }

    let mut by_goal: HashMap<usize, Vec<usize>> = HashMap::new();
    for &component_id in base_order {
        let Some(abstraction) = component_id
            .checked_sub(num_domain_abstractions)
            .and_then(|id| cartesian_abstractions.get(id))
        else {
            continue;
        };
        let Some(goal_id) = abstraction.metadata.collection_goal_id else {
            continue;
        };
        by_goal.entry(goal_id).or_default().push(component_id);
    }
    let progressive_roots = cartesian_abstractions
        .iter()
        .any(|abstraction| abstraction.metadata.progressive_refinement_root);
    if by_goal.is_empty()
        || (!progressive_roots && by_goal.values().all(|components| components.len() < 2))
    {
        return None;
    }

    let current_h = |component_id: usize| {
        standalone_current_h
            .get(component_id)
            .copied()
            .unwrap_or(0.0)
    };
    let abstraction = |component_id: usize| {
        cartesian_abstractions
            .get(component_id - num_domain_abstractions)
            .expect("goal-cover order component must reference a Cartesian abstraction")
    };
    let goal_max_h = |components: &[usize]| {
        components
            .iter()
            .copied()
            .map(current_h)
            .fold(0.0, f64::max)
    };
    let mut sorted_goals = by_goal
        .iter()
        .filter(|(_, components)| progressive_roots || components.len() >= 2)
        .collect::<Vec<_>>();
    sorted_goals.sort_by(|(left_goal, left), (right_goal, right)| {
        goal_max_h(right)
            .total_cmp(&goal_max_h(left))
            .then_with(|| left_goal.cmp(right_goal))
    });
    let (&anchor_goal, anchor_components) =
        sorted_goals[variant.anchor_goal_offset % sorted_goals.len()];

    let compare_anchor = |&left: &usize, &right: &usize| {
        let left_abstraction = abstraction(left);
        let right_abstraction = abstraction(right);
        current_h(right)
            .total_cmp(&current_h(left))
            .then_with(|| {
                right_abstraction
                    .metadata
                    .split_selection_rank
                    .cmp(&left_abstraction.metadata.split_selection_rank)
            })
            .then_with(|| {
                (right_abstraction.metadata.refinement_direction
                    == CartesianRefinementDirection::Regression)
                    .cmp(
                        &(left_abstraction.metadata.refinement_direction
                            == CartesianRefinementDirection::Regression),
                    )
            })
            .then_with(|| left.cmp(&right))
    };
    let mut sorted_anchor_components = anchor_components.clone();
    sorted_anchor_components.sort_by(compare_anchor);
    let first_anchor = sorted_anchor_components
        .get(variant.anchor_offset % sorted_anchor_components.len())
        .copied()?;
    let first_metadata = &abstraction(first_anchor).metadata;
    let mut complementary_anchors = anchor_components
        .iter()
        .copied()
        .filter(|&component_id| component_id != first_anchor)
        .collect::<Vec<_>>();
    complementary_anchors.sort_by(|&left, &right| {
        let complement_score = |component_id: usize| {
            let metadata = &abstraction(component_id).metadata;
            (
                metadata.refinement_direction != first_metadata.refinement_direction,
                metadata.split_selection_rank == first_metadata.split_selection_rank,
            )
        };
        complement_score(right)
            .cmp(&complement_score(left))
            .then_with(|| current_h(right).total_cmp(&current_h(left)))
            .then_with(|| {
                abstraction(left)
                    .transition_system
                    .transitions
                    .len()
                    .cmp(&abstraction(right).transition_system.transitions.len())
            })
            .then_with(|| left.cmp(&right))
    });
    let complementary_anchor = (!complementary_anchors.is_empty())
        .then(|| complementary_anchors[variant.complementary_round % complementary_anchors.len()]);

    let mut order = Vec::with_capacity(base_order.len());
    let mut selected = HashSet::with_capacity(base_order.len());
    if variant.non_goal_prefix {
        for &component_id in base_order {
            if !is_goal_cartesian(component_id) {
                order.push(component_id);
                selected.insert(component_id);
            }
        }
    }
    order.push(first_anchor);
    selected.insert(first_anchor);
    if let Some(component_id) = complementary_anchor {
        order.push(component_id);
        selected.insert(component_id);
    }
    if progressive_roots {
        for component_id in sorted_anchor_components {
            if selected.insert(component_id) {
                order.push(component_id);
            }
        }
    }

    let mut other_goals = by_goal
        .iter()
        .filter(|(goal_id, _)| **goal_id != anchor_goal)
        .collect::<Vec<_>>();
    other_goals.sort_by(|(left_goal, left), (right_goal, right)| {
        goal_max_h(right)
            .total_cmp(&goal_max_h(left))
            .then_with(|| left_goal.cmp(right_goal))
    });
    for (representative_index, (_, components)) in other_goals.into_iter().enumerate() {
        let mut representatives = components.clone();
        representatives.sort_by(|&left, &right| {
            let variant_score = |component_id: usize| {
                let metadata = &abstraction(component_id).metadata;
                (
                    metadata.refinement_direction == first_metadata.refinement_direction,
                    metadata.split_selection_rank == first_metadata.split_selection_rank,
                )
            };
            variant_score(right)
                .cmp(&variant_score(left))
                .then_with(|| {
                    abstraction(left)
                        .transition_system
                        .transitions
                        .len()
                        .cmp(&abstraction(right).transition_system.transitions.len())
                })
                .then_with(|| current_h(right).total_cmp(&current_h(left)))
                .then_with(|| left.cmp(&right))
        });
        let representative_offset = variant
            .representative_round
            .wrapping_add(representative_index.wrapping_mul(variant.anchor_offset + 1))
            % representatives.len();
        let representative = representatives[representative_offset];
        order.push(representative);
        selected.insert(representative);
    }

    if variant.compact {
        debug_assert_eq!(
            order.len(),
            if progressive_roots {
                by_goal.len() + anchor_components.len() - 1
            } else {
                by_goal.len() + 1
            }
        );
        return Some(order);
    }

    let mut remaining = base_order
        .iter()
        .copied()
        .filter(|component_id| !selected.contains(component_id))
        .collect::<Vec<_>>();
    remaining.sort_by(
        |&left, &right| match (is_goal_cartesian(left), is_goal_cartesian(right)) {
            (true, true) => abstraction(left)
                .transition_system
                .transitions
                .len()
                .cmp(&abstraction(right).transition_system.transitions.len())
                .then_with(|| current_h(right).total_cmp(&current_h(left)))
                .then_with(|| left.cmp(&right)),
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            (false, false) => base_order
                .iter()
                .position(|&id| id == left)
                .cmp(&base_order.iter().position(|&id| id == right)),
        },
    );
    order.extend(remaining);
    debug_assert_eq!(order.len(), base_order.len());
    debug_assert_eq!(
        order.iter().copied().collect::<HashSet<_>>().len(),
        base_order.len()
    );
    Some(order)
}

fn max_heuristic_greedy_order(base_order: &[usize], standalone_current_h: &[f64]) -> Vec<usize> {
    let mut order = base_order.to_vec();
    order.sort_by(|&left, &right| {
        let left_h = standalone_current_h.get(left).copied().unwrap_or(0.0);
        let right_h = standalone_current_h.get(right).copied().unwrap_or(0.0);
        right_h.total_cmp(&left_h).then_with(|| left.cmp(&right))
    });
    order
}

fn optimization_deadline(max_time: f64) -> Option<Instant> {
    (max_time.is_finite() && max_time.is_sign_positive())
        .then(|| Instant::now() + Duration::from_secs_f64(max_time))
}

fn earliest_deadline(left: Option<Instant>, right: Option<Instant>) -> Option<Instant> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.min(right)),
        (Some(deadline), None) | (None, Some(deadline)) => Some(deadline),
        (None, None) => None,
    }
}

fn random_walk_length(
    initial_h: f64,
    average_operator_cost: f64,
    rng: &mut SmallRng,
) -> Result<usize, EvaluationError> {
    if !initial_h.is_finite() || initial_h < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "offline SCP sampling requires a finite non-negative initial h, got {initial_h}"
        )));
    }
    if !average_operator_cost.is_finite() || average_operator_cost < 0.0 {
        return Err(EvaluationError::ComputationFailed(format!(
            "offline SCP sampling requires a finite non-negative average operator cost, got {average_operator_cost}"
        )));
    }
    let trials_f64 = if initial_h <= f64::EPSILON || average_operator_cost <= f64::EPSILON {
        10.0
    } else {
        4.0 * (initial_h / average_operator_cost).round()
    };
    if trials_f64 > usize::MAX as f64 {
        return Err(EvaluationError::ComputationFailed(format!(
            "offline SCP random-walk trial count does not fit usize: {trials_f64}"
        )));
    }
    let trials = trials_f64 as usize;
    Ok((0..trials).filter(|_| rng.gen_bool(0.5)).count())
}

fn retain_if_sample_improving(
    candidate: CostPartitioningHeuristic,
    sample_ids: &[Vec<Option<usize>>],
    sample_best: &mut [f64],
    portfolio: &mut Vec<CostPartitioningHeuristic>,
    portfolio_size_kb: &mut usize,
    max_size_kb: usize,
) -> bool {
    assert_eq!(sample_ids.len(), sample_best.len());
    if portfolio.contains(&candidate) {
        return false;
    }
    let candidate_size = candidate.estimate_size_in_kb();
    if !portfolio.is_empty() && portfolio_size_kb.saturating_add(candidate_size) > max_size_kb {
        return false;
    }
    let values = sample_ids
        .iter()
        .map(|ids| candidate.compute_heuristic(ids))
        .collect::<Vec<_>>();
    if !values
        .iter()
        .zip(sample_best.iter())
        .any(|(&value, &best)| value > best)
    {
        return false;
    }
    for (best, value) in sample_best.iter_mut().zip(values) {
        *best = (*best).max(value);
    }
    *portfolio_size_kb = portfolio_size_kb.saturating_add(candidate_size);
    portfolio.push(candidate);
    true
}

fn mandatory_goal_specialist_indices(
    candidates: &[CostPartitioningHeuristic],
    sample_ids: &[Vec<Option<usize>>],
) -> Vec<usize> {
    assert!(!candidates.is_empty());
    const SPECIALISTS_PER_GOAL: usize = 4;
    let mut candidates_by_goal = HashMap::<usize, Vec<(usize, f64)>>::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let Some(goal_id) = candidate.specialist_goal_id else {
            continue;
        };
        let score = sample_ids
            .iter()
            .map(|ids| candidate.compute_heuristic(ids))
            .sum::<f64>();
        assert!(!score.is_nan(), "goal-specialist SCP score must not be NaN");
        candidates_by_goal
            .entry(goal_id)
            .or_default()
            .push((index, score));
    }

    let mut goals = candidates_by_goal.into_iter().collect::<Vec<_>>();
    goals.sort_by_key(|(goal_id, _)| *goal_id);
    let mut indices = vec![0];
    for (_, mut goal_candidates) in goals {
        goal_candidates.sort_by(|left, right| {
            (right.0 == 0)
                .cmp(&(left.0 == 0))
                .then_with(|| right.1.total_cmp(&left.1))
                .then_with(|| left.0.cmp(&right.0))
        });
        indices.extend(
            goal_candidates
                .into_iter()
                .take(SPECIALISTS_PER_GOAL)
                .map(|(index, _)| index)
                .filter(|&index| index != 0),
        );
    }
    indices
}

fn retain_mandatory_partition(
    candidate: CostPartitioningHeuristic,
    sample_ids: &[Vec<Option<usize>>],
    sample_best: &mut [f64],
    portfolio: &mut Vec<CostPartitioningHeuristic>,
    portfolio_size_kb: &mut usize,
    max_size_kb: usize,
) -> Result<(), String> {
    assert_eq!(sample_ids.len(), sample_best.len());
    let candidate_size = candidate.estimate_size_in_kb();
    let required_size = portfolio_size_kb.saturating_add(candidate_size);
    if required_size > max_size_kb {
        return Err(format!(
            "mandatory goal-specialist SCPs require {required_size} KiB, exceeding max_size={max_size_kb} KiB"
        ));
    }
    for (best, ids) in sample_best.iter_mut().zip(sample_ids) {
        *best = (*best).max(candidate.compute_heuristic(ids));
    }
    *portfolio_size_kb = required_size;
    portfolio.push(candidate);
    Ok(())
}

fn abstraction_is_target_centered(
    abstractions: &[DomainAbstraction],
    abstraction_id: usize,
) -> bool {
    abstractions
        .get(abstraction_id)
        .and_then(|abstraction| abstraction.metadata.flaw_kind.as_deref())
        .is_some_and(|flaw_kind| flaw_kind == "target_centered")
}

fn current_h_for_distances(
    abstraction_id: usize,
    distances: &[f64],
    abstract_state_ids: &[Option<usize>],
) -> f64 {
    abstract_state_ids
        .get(abstraction_id)
        .copied()
        .flatten()
        .and_then(|state_id| distances.get(state_id).copied())
        .unwrap_or(0.0)
}

fn should_skip_zero_current_table(
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    abstract_state_ids: &[Option<usize>],
) -> bool {
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    if current_h > 1e-9 {
        return false;
    }
    debug!("scp_online: skipping {step} abstraction {abstraction_id}: current_h=0");
    true
}

fn positive_cost_stats(costs: &[f64]) -> (usize, f64) {
    costs
        .iter()
        .copied()
        .filter(|cost| cost.is_finite() && *cost > 0.0)
        .fold((0, 0.0), |(count, total), cost| (count + 1, total + cost))
}

fn log_label_table_summary(
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    saturated_costs: &[f64],
    abstract_state_ids: &[Option<usize>],
) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    let (positive_count, total_positive) = positive_cost_stats(saturated_costs);
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    debug!(
        "scp_online: label {step} abstraction {abstraction_id}: current_h={current_h}, positive_saturated_labels={positive_count}, total_positive_saturated={total_positive:.6}"
    );
}

fn log_transition_table_summary(
    step: &str,
    abstraction_id: usize,
    distances: &[f64],
    operator_costs: &[f64],
    abstract_state_ids: &[Option<usize>],
) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    let (positive_count, total_positive) = positive_cost_stats(operator_costs);
    let current_h = current_h_for_distances(abstraction_id, distances, abstract_state_ids);
    debug!(
        "scp_online: abstract-operator {step} abstraction {abstraction_id}: current_h={current_h}, positive_saturated_abstract_ops={positive_count}, total_positive_saturated={total_positive:.6}"
    );
}

fn log_abstract_operator_footprint_summary(
    abstraction_id: usize,
    footprints: &[AbstractOperatorFootprint],
) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    let stats = abstract_operator_footprint_stats(footprints);
    debug!(
        "scp_online: abstract-operator footprints abstraction {abstraction_id}: labels={}, bounded_labels={}, bounded_numeric_dimensions={}",
        stats.total_labels, stats.bounded_labels, stats.bounded_numeric_dimensions,
    );
}

fn log_abstraction_candidate_report(
    mode: &str,
    state: &ScpOnlineState,
    abstractions: &[DomainAbstraction],
    order: &[usize],
    abstract_state_ids: &[Option<usize>],
    scoring_function: ScoringFunction,
    use_abstract_operator_cost_partitioning: bool,
) {
    let inactive = order
        .iter()
        .filter(|&&abstraction_id| {
            state
                .h_values_by_abstraction
                .get(abstraction_id)
                .map(|distances| {
                    current_h_for_distances(abstraction_id, distances, abstract_state_ids)
                })
                .unwrap_or(0.0)
                <= 1e-9
        })
        .count();
    info!(
        "scp_online: {mode} abstraction candidate report, candidates={}, inactive_current_state={inactive}, showing_top={}",
        order.len(),
        order.len().min(25),
    );

    for (rank, &abstraction_id) in order.iter().take(25).enumerate() {
        let h = state
            .h_values_by_abstraction
            .get(abstraction_id)
            .map(|distances| current_h_for_distances(abstraction_id, distances, abstract_state_ids))
            .unwrap_or(0.0);
        let stolen = state
            .stolen_costs_by_abstraction
            .get(abstraction_id)
            .copied()
            .unwrap_or(0.0);
        let Some(abstraction) = abstractions.get(abstraction_id) else {
            info!(
                "scp_online: candidate rank={rank}, id={abstraction_id}, h={h}, stolen={stolen}, missing_domain_abstraction=true"
            );
            continue;
        };
        let score = if use_abstract_operator_cost_partitioning {
            compute_abstract_operator_order_score(h, stolen, scoring_function, abstraction)
        } else {
            compute_score(h, stolen, scoring_function)
        };
        let stats = abstract_operator_footprint_stats(&abstraction.abstract_operator_footprints);
        let metadata = &abstraction.metadata;
        let seeds = truncate_for_log(&metadata.initial_seed_splits.join("|"), 220);
        info!(
            "scp_online: candidate rank={rank}, id={abstraction_id}, score={score:.6}, h={h}, stolen={stolen:.6}, states={}, abstract_ops={}, footprint_labels={}, bounded_footprint_labels={}, bounded_numeric_dimensions={}, iteration={:?}, flaw_kind={:?}, full_goal_task={:?}, seeds={seeds}",
            abstraction_state_count(abstraction),
            abstraction.abstract_operators.len(),
            stats.total_labels,
            stats.bounded_labels,
            stats.bounded_numeric_dimensions,
            metadata.collection_iteration,
            metadata.flaw_kind,
            metadata.full_goal_task,
        );
    }
}

#[derive(Debug, Clone, Copy, Default)]
struct AbstractOperatorFootprintStats {
    total_labels: usize,
    bounded_labels: usize,
    bounded_numeric_dimensions: usize,
}

fn abstract_operator_footprint_stats(
    footprints: &[AbstractOperatorFootprint],
) -> AbstractOperatorFootprintStats {
    let mut stats = AbstractOperatorFootprintStats::default();
    for label in footprints.iter().flat_map(|fp| fp.labels.iter()) {
        stats.total_labels = stats.total_labels.saturating_add(1);
        let bounded_dimensions = label
            .source_region
            .numeric
            .iter()
            .filter(|interval| interval.lower.is_finite() || interval.upper.is_finite())
            .count();
        if bounded_dimensions > 0 {
            stats.bounded_labels = stats.bounded_labels.saturating_add(1);
        }
        stats.bounded_numeric_dimensions = stats
            .bounded_numeric_dimensions
            .saturating_add(bounded_dimensions);
    }
    stats
}

#[derive(Debug, Default)]
struct LabelFootprintCounts {
    footprints: usize,
    bounded_footprints: usize,
    bounded_numeric_dimensions: usize,
}

fn log_positive_label_footprint_diagnostics(
    abstraction_id: usize,
    task: &dyn AbstractNumericTask,
    footprints: &[AbstractOperatorFootprint],
    label_saturated_costs: &[f64],
) {
    if !enabled!(Level::INFO) {
        return;
    }
    let mut counts_by_label: HashMap<usize, LabelFootprintCounts> = HashMap::new();
    for label in footprints
        .iter()
        .flat_map(|footprint| footprint.labels.iter())
    {
        let counts = counts_by_label.entry(label.concrete_op_id).or_default();
        counts.footprints += 1;
        let bounded_dimensions = label
            .source_region
            .numeric
            .iter()
            .filter(|interval| interval.lower.is_finite() || interval.upper.is_finite())
            .count();
        if bounded_dimensions > 0 {
            counts.bounded_footprints += 1;
        }
        counts.bounded_numeric_dimensions += bounded_dimensions;
    }

    let mut positive_labels = label_saturated_costs
        .iter()
        .enumerate()
        .filter_map(|(concrete_op_id, &cost)| {
            if cost.is_finite() && cost > 1e-9 {
                Some((concrete_op_id, cost))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    positive_labels.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    for (rank, (concrete_op_id, saturated_cost)) in positive_labels.into_iter().take(12).enumerate()
    {
        let counts = counts_by_label.get(&concrete_op_id);
        let (footprint_count, bounded_footprints, bounded_numeric_dimensions) = counts
            .map(|counts| {
                (
                    counts.footprints,
                    counts.bounded_footprints,
                    counts.bounded_numeric_dimensions,
                )
            })
            .unwrap_or((0, 0, 0));
        let op = task.get_operators().get(concrete_op_id);
        let op_name = op.map(|op| op.name()).unwrap_or("<missing operator>");
        let numeric_effects = op.map(|op| op.assignment_effects().len()).unwrap_or(0);
        info!(
            "scp_online: abstract-operator label diagnostic detail abstraction {abstraction_id}: rank={rank}, label={concrete_op_id}, saturated={saturated_cost:.6}, numeric_effects={numeric_effects}, footprints={footprint_count}, bounded_footprints={bounded_footprints}, bounded_numeric_dimensions={bounded_numeric_dimensions}, op={op_name}"
        );
    }
}

fn abstraction_metadata_summary(abstraction: &DomainAbstraction) -> String {
    let metadata = &abstraction.metadata;
    format!(
        "iteration={:?},strategy={:?},flaw_kind={:?},full_goal_task={:?},seeds={}",
        metadata.collection_iteration,
        metadata.portfolio_strategy,
        metadata.flaw_kind,
        metadata.full_goal_task,
        metadata.initial_seed_splits.join("|"),
    )
}

fn truncate_for_log(value: &str, max_chars: usize) -> String {
    if value.chars().count() <= max_chars {
        return value.to_string();
    }
    let mut truncated = value.chars().take(max_chars).collect::<String>();
    truncated.push_str("...");
    truncated
}

fn log_transition_residual_summary(remaining_costs: &TransitionResidualCosts) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    debug!(
        "scp_online: abstract-operator residuals now store {} region reductions",
        remaining_costs.num_reductions()
    );
}

/// Per-abstraction debug print of every abstract operator for the seven sailing
/// movement operators (or any operator whose concrete name starts with `go_`).
/// For each abstract op, logs the underlying numeric source-region preimage,
/// the residual cost the abstract op saw at distance-table time (i.e. base cost
/// minus reductions filed by abstractions processed earlier in the SCP order),
/// and the saturated cost this abstraction reserved on the abstract op.
///
/// Gated by `tracing::Level::DEBUG` so it can be turned on with
/// `--log-level debug` without changing the source. With sailing's roughly
/// 25 000+ abstract operators per abstraction, the output is large; users
/// looking at it typically `grep` for the operator name and a specific
/// abstraction id.
fn log_movement_abstract_operator_costs(
    abstraction_id: usize,
    abstraction: &DomainAbstraction,
    abstraction_task: &dyn AbstractNumericTask,
    tcf: &AbstractOperatorCostFunction,
    remaining_costs: &TransitionResidualCosts,
) {
    if !enabled!(Level::DEBUG) {
        return;
    }
    let concrete_operators = abstraction_task.get_operators();
    for (abstract_op_id, (_abstract_op, footprints)) in abstraction
        .abstract_operators
        .iter()
        .zip(abstraction.abstract_operator_footprints.iter())
        .enumerate()
    {
        let saturated = tcf
            .operator_costs
            .get(abstract_op_id)
            .copied()
            .unwrap_or(0.0);
        for footprint in &footprints.labels {
            let Some(op) = concrete_operators.get(footprint.concrete_op_id) else {
                continue;
            };
            if !op.name().starts_with("go_") {
                continue;
            }
            let residual = remaining_costs.cost_for_operator_footprint(
                abstraction_id,
                abstract_op_id,
                footprint,
            );
            let bounded_dims: Vec<String> = footprint
                .source_region
                .numeric
                .iter()
                .enumerate()
                .filter(|(_, iv)| iv.lower.is_finite() || iv.upper.is_finite())
                .map(|(var_id, iv)| {
                    format!(
                        "n{var_id}={}{},{}{}",
                        if iv.lower_closed { "[" } else { "(" },
                        iv.lower,
                        iv.upper,
                        if iv.upper_closed { "]" } else { ")" }
                    )
                })
                .collect();
            tracing::debug!(
                "scp_online debug movement op abstraction={abstraction_id} \
                 abstract_op_id={abstract_op_id} concrete_op_id={} name={} \
                 bounded_source=[{}] pre_sat_residual={:.3} saturated={:.3}",
                footprint.concrete_op_id,
                op.name(),
                bounded_dims.join(", "),
                residual,
                saturated,
            );
        }
    }
}

impl Heuristic for SaturatedCostPartitioningOnlineHeuristic<'_> {
    fn compute_heuristic(
        &self,
        eval_state: &EvaluationState<'_, '_>,
    ) -> Result<f64, EvaluationError> {
        let task = eval_state.task().ok_or_else(|| {
            EvaluationError::InvalidState(
                "SaturatedCostPartitioningOnlineHeuristic requires task".to_string(),
            )
        })?;
        let build_cp = {
            let state = self.state.borrow();
            self.should_build_cp(&state)
        };

        let mut component_ids = self.component_ids_scratch.borrow_mut();
        let num_domain_abstractions = if build_cp {
            self.compute_abstract_state_ids_into(eval_state, None, &mut component_ids)?
        } else {
            let state = self.state.borrow();
            self.compute_abstract_state_ids_into(
                eval_state,
                Some(&state.required_lookup_ids),
                &mut component_ids,
            )?
        };
        let abstract_state_ids = component_ids.as_slice();

        let mut state = self.state.borrow_mut();
        let mut max_h = Self::compute_max_h(&state, abstract_state_ids);
        if max_h.is_infinite() {
            return Ok(max_h);
        }

        if build_cp {
            self.update_improvement_status(&mut state);
            self.release_abstractions_if_finished(&mut state);
        } else if !state.improve_heuristic && !state.improvement_ended {
            self.release_abstractions_if_finished(&mut state);
        }

        let candidate_partitions = self.maybe_build_cp(
            task,
            &mut state,
            abstract_state_ids,
            num_domain_abstractions,
        )?;
        for cp in candidate_partitions {
            Self::retain_cp(
                &mut state,
                cp,
                abstract_state_ids,
                &mut max_h,
                !self.config.online,
                self.config.max_size,
            );
        }

        if build_cp && (!self.config.online || self.config.interval == usize::MAX) {
            state.improve_heuristic = false;
        }
        self.update_improvement_status(&mut state);
        self.release_abstractions_if_finished(&mut state);

        state.evaluated_states = state.evaluated_states.saturating_add(1);
        Ok(max_h)
    }

    fn heuristic_name(&self) -> String {
        self.name.clone()
    }
}

#[cfg(test)]
mod handcrafted_sailing_tests {
    use std::cell::{Ref, RefMut};
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};
    use std::time::Duration;

    use planforge_sas::axioms::{AssignmentAxiom, ComparisonAxiom, PropositionalAxiom};
    use planforge_sas::numeric_task::{
        AbstractNumericTask, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, NumericType,
        NumericVariable, Operator,
    };
    use planforge_translate::preprocess::run_preprocess_to_output;
    use planforge_translator::translate_to_sas_to_path_fast;

    use super::*;
    use crate::evaluation::cartesian_abstractions::{
        CartesianAbstraction, CartesianAbstractionCollectionConfig,
        CartesianAbstractionCollectionGenerator, CartesianAbstractionConfig,
        CartesianRefinementDirection,
    };
    use crate::evaluation::domain_abstractions::cegar::InitialSeedSplit;
    use crate::evaluation::domain_abstractions::domain_abstraction::NumericPartitions;
    use crate::evaluation::domain_abstractions::domain_abstraction_factory::DomainAbstractionFactory;
    use crate::evaluation::domain_abstractions::domain_abstraction_generator::{
        DomainAbstractionMetadata, compute_hash_multipliers,
    };
    use crate::task_restriction::build_restricted_task;

    #[test]
    #[ignore = "oracle collection report over translated sailing benchmark"]
    fn sailing_perfect_complementary_cartesian_collection_reaches_optimum_with_transition_scp() {
        let task = translated_sailing_2_2_task();
        let restricted_task = build_restricted_task(&task)
            .expect("sailing task should support restricted task")
            .expect("sailing task should have promotable roots")
            .into_task();
        let abstractions =
            CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
                abstraction: CartesianAbstractionConfig {
                    max_states: 1_000,
                    max_time: Some(Duration::from_secs(5)),
                    combine_labels: false,
                    compute_operator_footprints: true,
                    random_seed: Some(1),
                    debug: false,
                    ..Default::default()
                },
                variants_per_goal: 4,
                max_collection_states: 10_000,
                total_max_time: Some(Duration::from_secs(15)),
                progressive_goal_roots: false,
            })
            .expect("valid oracle Cartesian collection config")
            .generate(&restricted_task)
            .expect("failed to build oracle Cartesian collection");

        assert_eq!(abstractions.len(), 8);
        assert_eq!(
            abstractions
                .iter()
                .map(CartesianAbstraction::num_states)
                .sum::<usize>(),
            8_000
        );
        for goal_id in 0..2 {
            let mut modes = abstractions
                .iter()
                .filter(|abstraction| abstraction.metadata.collection_goal_id == Some(goal_id))
                .map(|abstraction| {
                    (
                        abstraction.metadata.refinement_direction,
                        abstraction.metadata.split_selection_rank,
                    )
                })
                .collect::<Vec<_>>();
            modes.sort_by_key(|(direction, rank)| {
                (
                    match direction {
                        CartesianRefinementDirection::Progression => 0,
                        CartesianRefinementDirection::Regression => 1,
                    },
                    *rank,
                )
            });
            assert_eq!(
                modes,
                vec![
                    (CartesianRefinementDirection::Progression, Some(0)),
                    (CartesianRefinementDirection::Progression, Some(1)),
                    (CartesianRefinementDirection::Regression, Some(0)),
                    (CartesianRefinementDirection::Regression, Some(1)),
                ]
            );
        }

        let standalone_max = abstractions
            .iter()
            .map(|abstraction| {
                abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash]
            })
            .fold(0.0f64, f64::max);
        let standalone_h = abstractions
            .iter()
            .map(|abstraction| {
                abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash]
            })
            .collect::<Vec<_>>();
        let goal_cover_order = cartesian_goal_cover_order(
            &(0..abstractions.len()).collect::<Vec<_>>(),
            0,
            &abstractions,
            &standalone_h,
            true,
            GoalCoverOrderVariant::default(),
        )
        .expect("the complementary sailing collection must have a goal-cover order");
        assert_eq!(&goal_cover_order[..3], &[5, 0, 1]);
        let transition_h = initial_cartesian_scp_value(&restricted_task, abstractions, true, 10.0);

        println!(
            "SAILING_PERFECT_COLLECTION standalone_max={standalone_max} transition_scp={transition_h}"
        );
        assert_eq!(standalone_max, 40.0);
        assert_eq!(transition_h, 76.0);
    }

    fn initial_cartesian_scp_value(
        task: &dyn AbstractNumericTask,
        abstractions: Vec<CartesianAbstraction>,
        use_abstract_operator_cost_partitioning: bool,
        order_optimization_max_time: f64,
    ) -> f64 {
        let initial_prop = task.get_initial_propositional_state_values();
        let initial_numeric = task.get_initial_numeric_state_values();
        let abstract_state_ids = abstractions
            .iter()
            .map(|abstraction| {
                Some(
                    abstraction
                        .hierarchy
                        .map_state(&initial_prop, &initial_numeric)
                        .expect("failed to map sailing initial state"),
                )
            })
            .collect::<Vec<_>>();
        let config = ScpOnlineConfig {
            online: true,
            table_construction_max_time: 30.0,
            order_optimization_max_time,
            max_size: 10_000_000,
            combine_labels: false,
            saturator: Saturator::All,
            residual_sweeps: 0,
            random_seed: Some(1),
            use_abstract_operator_cost_partitioning,
            ..Default::default()
        };
        let heuristic = SaturatedCostPartitioningOnlineHeuristic::new_with_cartesian(
            None,
            vec![],
            abstractions,
            vec![],
            config,
            task,
        )
        .expect("failed to construct oracle SCP heuristic");
        let mut state = heuristic.state.borrow_mut();
        let mut partitions = heuristic
            .maybe_build_cp(task, &mut state, &abstract_state_ids, 0)
            .expect("oracle SCP construction failed");
        assert_eq!(partitions.len(), 1);
        partitions
            .pop()
            .expect("one oracle SCP partition")
            .compute_heuristic(&abstract_state_ids)
    }

    #[test]
    #[ignore = "diagnostic full-task handcrafted sailing abstract-operator SCP report"]
    fn sailing_handcrafted_four_abstractions_full_task_abstract_operator_scp_initial_h_report() {
        let task = translated_sailing_2_2_task();
        let restricted_task = build_restricted_task(&task)
            .expect("sailing task should support restricted task")
            .expect("sailing task should have promotable roots")
            .into_task();
        let transformed_task = &restricted_task;
        let specs = handcrafted_full_task_specs(transformed_task);
        assert_eq!(specs.len(), 4);

        let mut abstractions = Vec::new();
        for (index, spec) in specs.iter().enumerate() {
            let single_goal_task = SingleGoalTask::new(transformed_task, spec.goal.clone());
            let mut abstraction = build_handcrafted_abstraction(&single_goal_task, spec)
                .unwrap_or_else(|error| panic!("failed to build {}: {error:#}", spec.name));
            abstraction.metadata = DomainAbstractionMetadata {
                collection_iteration: Some(index + 1),
                portfolio_strategy: Some("handcrafted_full_task_sailing".to_string()),
                flaw_kind: None,
                full_goal_task: Some(false),
                initial_seed_splits: spec.seed_splits.iter().map(seed_description).collect(),
                max_abstraction_size: Some(10_000),
                ..DomainAbstractionMetadata::default()
            };
            let states = abstraction_state_count(&abstraction);
            assert!(
                states <= 10_000,
                "{} has {states} states, expected at most 10000",
                spec.name
            );
            abstractions.push(abstraction);
        }

        let config = ScpOnlineConfig {
            online: true,
            max_time: 300.0,
            table_construction_max_time: 30.0,
            max_size: 10_000_000,
            diversify: false,
            samples: 1_000,
            max_orders: usize::MAX,
            interval: usize::MAX,
            combine_labels: false,
            collection_config: DomainAbstractionCollectionGeneratorMultipleCegarConfig {
                debug: true,
                ..Default::default()
            },
            use_numeric_pdbs: false,
            max_pdb_states: 0,
            max_pattern_size: 0,
            only_interesting_patterns: true,
            pdb_exploration_heuristic: PdbInternalHeuristic::Blind,
            pdb_frontier_heuristic: PdbInternalHeuristic::Zero,
            pdb_failed_lookup_heuristic: PdbInternalHeuristic::Zero,
            scoring_function: ScoringFunction::MaxHeuristic,
            order_generator: OrderGenerator::Greedy,
            initial_order_generation_max_time: 0.0,
            order_optimization_max_time: 0.0,
            saturator: Saturator::All,
            random_seed: Some(1),
            use_abstract_operator_cost_partitioning: true,
            residual_sweeps: 1,
        };

        let heuristic = SaturatedCostPartitioningOnlineHeuristic::new(
            None,
            abstractions,
            vec![],
            config,
            transformed_task,
        )
        .expect("failed to construct SCP heuristic");
        let abstract_state_ids = initial_abstract_state_ids(&heuristic, transformed_task);
        {
            let mut state = heuristic.state.borrow_mut();
            let mut max_h = SaturatedCostPartitioningOnlineHeuristic::compute_max_h(
                &state,
                &abstract_state_ids,
            );
            let mut partitions = heuristic
                .maybe_build_cp(
                    transformed_task,
                    &mut state,
                    &abstract_state_ids,
                    specs.len(),
                )
                .expect("initial SCP construction failed");
            assert_eq!(partitions.len(), 1);
            let cp = partitions.pop().unwrap();
            SaturatedCostPartitioningOnlineHeuristic::retain_cp(
                &mut state,
                cp,
                &abstract_state_ids,
                &mut max_h,
                true,
                usize::MAX,
            );
        }

        let state = heuristic.state.borrow();
        let initial_h =
            SaturatedCostPartitioningOnlineHeuristic::compute_max_h(&state, &abstract_state_ids);
        let abstractions_ref = heuristic.abstractions.borrow();
        let abstractions = abstractions_ref
            .as_ref()
            .expect("diagnostic keeps abstractions alive");

        println!("HANDCRAFTED_FULL_TASK_INITIAL_H {initial_h}");
        let mut contributions = vec![0.0; specs.len()];
        for cp in &state.cp_heuristics {
            for table in &cp.lookup_tables {
                let contribution = abstract_state_ids
                    .get(table.abstraction_id)
                    .copied()
                    .flatten()
                    .and_then(|state_id| table.distances.get(state_id).copied())
                    .unwrap_or(table.unknown_value);
                if let Some(total) = contributions.get_mut(table.abstraction_id) {
                    *total += contribution;
                }
            }
        }

        for (index, (spec, abstraction)) in specs.iter().zip(abstractions).enumerate() {
            let standalone_h = current_h_for_distances(
                index,
                &abstraction.distance_table.distances,
                &abstract_state_ids,
            );
            println!(
                "HANDCRAFTED_FULL_TASK_ABS index={index} name={} standalone_h={standalone_h} scp_contribution={} states={} abstract_ops={} views={}",
                spec.name,
                contributions[index],
                abstraction_state_count(abstraction),
                abstraction.abstract_operators.len(),
                partition_report(transformed_task, abstraction, &spec.view_ids),
            );
        }
        println!("HANDCRAFTED_FULL_TASK_CONTRIBUTIONS {contributions:?}");

        assert!(
            initial_h.is_finite(),
            "initial h must be finite, got {initial_h}"
        );
        assert!(initial_h > 0.0, "initial h should be positive");
        assert!(
            initial_h <= 76.0,
            "full-task diagnostic should not exceed the known optimal cost 76, got {initial_h}"
        );
    }

    #[derive(Debug, Clone)]
    struct HandcraftedSpec {
        name: String,
        goal: ExplicitFact,
        view_ids: Vec<usize>,
        seed_splits: Vec<InitialSeedSplit>,
    }

    fn handcrafted_full_task_specs(task: &dyn AbstractNumericTask) -> Vec<HandcraftedSpec> {
        [
            ("p1-u", "p1", ViewKind::Sum),
            ("p1-v", "p1", ViewKind::Difference),
            ("p0-u", "p0", ViewKind::Sum),
            ("p0-v", "p0", ViewKind::Difference),
        ]
        .into_iter()
        .map(|(name, person, view_kind)| {
            let view_ids = ["b0", "b1"]
                .into_iter()
                .map(|boat| find_sailing_view(task, boat, person, view_kind))
                .collect::<Vec<_>>();
            let seed_splits = route_seed_splits(task, &view_ids);
            HandcraftedSpec {
                name: name.to_string(),
                goal: find_saved_goal_fact(task, person),
                view_ids,
                seed_splits,
            }
        })
        .collect()
    }

    fn build_handcrafted_abstraction(
        transformed_task: &dyn AbstractNumericTask,
        spec: &HandcraftedSpec,
    ) -> anyhow::Result<DomainAbstraction> {
        let mut partitions = NumericPartitions::trivial(transformed_task);
        for seed in &spec.seed_splits {
            let InitialSeedSplit::Numeric {
                numeric_var_id,
                value,
                include_in_lower,
            } = seed
            else {
                continue;
            };
            partitions.split_at(*numeric_var_id, *value, *include_in_lower);
        }

        let goal_vars = (0..transformed_task.get_num_goals())
            .map(|goal_id| transformed_task.get_goal_fact(goal_id).var())
            .collect::<HashSet<_>>();
        let domain_mapping = (0..transformed_task.get_num_variables())
            .map(|var_id| {
                let domain_size = transformed_task
                    .get_variable_domain_size(var_id)
                    .expect("valid transformed prop var id");
                if goal_vars.contains(&var_id) {
                    (0..domain_size).collect::<Vec<_>>()
                } else {
                    vec![0; domain_size]
                }
            })
            .collect::<Vec<_>>();
        let domain_sizes = domain_mapping
            .iter()
            .map(|mapping| mapping.iter().copied().max().map_or(0, |value| value + 1))
            .collect::<Vec<_>>();
        let numeric_domain_sizes = (0..transformed_task.numeric_variables().len())
            .map(|numeric_var_id| {
                partitions
                    .partitions(numeric_var_id)
                    .expect("trivial partitions contain every numeric variable")
                    .len()
            })
            .collect::<Vec<_>>();
        let factory = DomainAbstractionFactory::new(
            transformed_task,
            domain_mapping,
            domain_sizes,
            partitions,
            numeric_domain_sizes,
        )?;
        let mut operator_generator = factory.make_operator_generator(transformed_task, false)?;
        let abstract_operators = operator_generator.build_abstract_operators(transformed_task)?;
        let abstract_operator_footprints =
            factory.build_abstract_operator_footprints(transformed_task, &abstract_operators)?;
        let distance_table = factory.build_distance_table_with_operators(
            transformed_task,
            &operator_generator,
            &abstract_operators,
            false,
        )?;
        let relevant_operator_ids = factory.relevant_operator_ids_from_operators_with_deadline(
            transformed_task,
            false,
            &abstract_operators,
            None,
        )?;
        let hash_multipliers =
            compute_hash_multipliers(factory.domain_sizes(), factory.numeric_domain_sizes())?;
        Ok(DomainAbstraction {
            factory,
            distance_table,
            hash_multipliers,
            combine_labels: false,
            relevant_operator_ids,
            abstract_operators,
            abstract_operator_footprints,
            metadata: DomainAbstractionMetadata::default(),
        })
    }

    fn initial_abstract_state_ids(
        heuristic: &SaturatedCostPartitioningOnlineHeuristic<'_>,
        task: &dyn AbstractNumericTask,
    ) -> Vec<Option<usize>> {
        let prop = task.get_initial_propositional_state_values();
        let numeric = task.get_initial_numeric_state_values();
        heuristic
            .abstraction_heuristics
            .iter()
            .map(|component| {
                Some(
                    component
                        .abstract_state_hash_from_state_values(&prop, &numeric)
                        .expect("failed to hash initial state for handcrafted abstraction"),
                )
            })
            .collect()
    }

    #[derive(Debug, Clone, Copy)]
    enum ViewKind {
        Sum,
        Difference,
    }

    fn find_sailing_view(
        task: &dyn AbstractNumericTask,
        boat: &str,
        person: &str,
        view_kind: ViewKind,
    ) -> usize {
        let tuple = format!("({boat}, {boat}, {person})");
        let candidates = task
            .numeric_variables()
            .iter()
            .enumerate()
            .filter(|(_, variable)| variable.get_type() == &NumericType::Regular)
            .filter(|(_, variable)| {
                let name = variable.name();
                name.contains(&tuple)
                    && !name.contains("25.0")
                    && match view_kind {
                        ViewKind::Sum => name.contains("derived!sum_PNE x"),
                        ViewKind::Difference => name.contains("derived!difference_PNE y"),
                    }
            })
            .map(|(id, _)| id)
            .collect::<Vec<_>>();
        assert_eq!(
            candidates.len(),
            1,
            "expected one {view_kind:?} view for {boat}/{person}, got {candidates:?}"
        );
        candidates[0]
    }

    fn find_saved_goal_fact(task: &dyn AbstractNumericTask, person: &str) -> ExplicitFact {
        let suffix = format!(" {person}");
        let mut candidates = task
            .get_operators()
            .iter()
            .filter(|operator| operator.name().starts_with("save_person "))
            .filter(|operator| operator.name().ends_with(&suffix))
            .flat_map(|operator| operator.effects().iter())
            .filter(|effect| effect.conditions().is_empty())
            .map(|effect| ExplicitFact::new(effect.var_id(), effect.value()))
            .collect::<Vec<_>>();
        candidates.sort();
        candidates.dedup();
        assert_eq!(
            candidates.len(),
            1,
            "expected one saved fact from save_person operators for {person}, got {candidates:?}"
        );
        candidates[0].clone()
    }

    fn route_seed_splits(
        task: &dyn AbstractNumericTask,
        view_ids: &[usize],
    ) -> Vec<InitialSeedSplit> {
        let initial = task.get_initial_numeric_state_values();
        let mut seeds = Vec::new();
        for &view_id in view_ids {
            add_split(&mut seeds, view_id, initial[view_id], false);
            add_split(&mut seeds, view_id, 0.0, true);
            add_split(&mut seeds, view_id, 25.0, true);
            add_route_grid_values(&mut seeds, view_id, initial[view_id], 25.0, 3.0);
        }
        seeds.sort_by(|left, right| seed_description(left).cmp(&seed_description(right)));
        seeds.dedup();
        seeds
    }

    fn add_route_grid_values(
        seeds: &mut Vec<InitialSeedSplit>,
        numeric_var_id: usize,
        start: f64,
        end: f64,
        step: f64,
    ) {
        assert!(start.is_finite() && end.is_finite() && step.is_finite() && step > 0.0);
        let direction = if start <= end { 1.0 } else { -1.0 };
        let mut value = start;
        while (end - value) * direction > step {
            value += direction * step;
            add_split(seeds, numeric_var_id, value, true);
        }
    }

    fn add_split(
        seeds: &mut Vec<InitialSeedSplit>,
        numeric_var_id: usize,
        value: f64,
        include_in_lower: bool,
    ) {
        seeds.push(InitialSeedSplit::Numeric {
            numeric_var_id,
            value,
            include_in_lower,
        });
    }

    fn partition_report(
        task: &dyn AbstractNumericTask,
        abstraction: &DomainAbstraction,
        view_ids: &[usize],
    ) -> String {
        view_ids
            .iter()
            .map(|&view_id| {
                let name = task.numeric_variables()[view_id].name();
                let num_parts = abstraction
                    .factory
                    .partitions()
                    .partitions(view_id)
                    .expect("missing partition for handcrafted view")
                    .len();
                format!("n{view_id}:{name} parts={num_parts}")
            })
            .collect::<Vec<_>>()
            .join(" || ")
    }

    fn seed_description(seed: &InitialSeedSplit) -> String {
        match seed {
            InitialSeedSplit::Propositional { var_id, value } => format!("p{var_id}={value}"),
            InitialSeedSplit::Numeric {
                numeric_var_id,
                value,
                include_in_lower,
            } => format!(
                "n{numeric_var_id}{}{}",
                if *include_in_lower { "<=" } else { "<" },
                value
            ),
        }
    }

    fn translated_sailing_2_2_task() -> NumericRootTask {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
        let domain = root.join("others/sailing/domain.pddl");
        let problem = root.join("others/sailing/prob_2_2_1229.pddl");
        assert!(domain.is_file(), "missing {}", domain.display());
        assert!(problem.is_file(), "missing {}", problem.display());
        let temp_dir = unique_temp_dir("sailing_handcrafted_full_task_scp")
            .expect("failed to create sailing diagnostic temp dir");
        let output_sas = temp_dir.join("output.sas");
        let preprocessed = temp_dir.join("output");
        translate_to_sas_to_path_fast(
            domain.to_str().expect("non-utf8 sailing domain path"),
            problem.to_str().expect("non-utf8 sailing problem path"),
            &output_sas,
        )
        .expect("sailing translation failed");
        run_preprocess_to_output(
            &[
                "preprocess".to_string(),
                output_sas.to_string_lossy().to_string(),
            ],
            &preprocessed,
        );
        NumericRootTask::from_file(&preprocessed)
    }

    fn unique_temp_dir(prefix: &str) -> std::io::Result<PathBuf> {
        let base = std::env::temp_dir().join("numeric_planneRS");
        std::fs::create_dir_all(&base)?;
        let dir = base.join(format!(
            "{prefix}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir(&dir)?;
        Ok(dir)
    }

    struct SingleGoalTask<'task> {
        base: &'task dyn AbstractNumericTask,
        goal: ExplicitFact,
    }

    impl<'task> SingleGoalTask<'task> {
        fn new(base: &'task dyn AbstractNumericTask, goal: ExplicitFact) -> Self {
            Self { base, goal }
        }
    }

    impl AbstractNumericTask for SingleGoalTask<'_> {
        fn variables(&self) -> &Vec<ExplicitVariable> {
            self.base.variables()
        }
        fn numeric_variables(&self) -> &Vec<NumericVariable> {
            self.base.numeric_variables()
        }
        fn assignment_axioms(&self) -> &Vec<AssignmentAxiom> {
            self.base.assignment_axioms()
        }
        fn comparison_axioms(&self) -> &Vec<ComparisonAxiom> {
            self.base.comparison_axioms()
        }
        fn axioms(&self) -> &Vec<PropositionalAxiom> {
            self.base.axioms()
        }
        fn metric(&self) -> &Metric {
            self.base.metric()
        }
        fn get_num_variables(&self) -> usize {
            self.base.get_num_variables()
        }
        fn get_variable_name(&self, index: usize) -> Result<&str, &str> {
            self.base.get_variable_name(index)
        }
        fn get_variable_domain_size(&self, index: usize) -> Result<usize, &str> {
            self.base.get_variable_domain_size(index)
        }
        fn get_variable_axiom_layer(&self, index: usize) -> Result<Option<usize>, &str> {
            self.base.get_variable_axiom_layer(index)
        }
        fn get_variable_default_axiom_value(&self, index: usize) -> Result<usize, &str> {
            self.base.get_variable_default_axiom_value(index)
        }
        fn get_fact_name(&self, fact: &ExplicitFact) -> &str {
            self.base.get_fact_name(fact)
        }
        fn are_facts_mutex(&self, fact1: &ExplicitFact, fact2: &ExplicitFact) -> bool {
            self.base.are_facts_mutex(fact1, fact2)
        }
        fn get_operators(&self) -> &Vec<Operator> {
            self.base.get_operators()
        }
        fn get_operator_cost(&self, index: usize, is_axiom: bool) -> u64 {
            self.base.get_operator_cost(index, is_axiom)
        }
        fn get_operator_name(&self, index: usize, is_axiom: bool) -> &str {
            self.base.get_operator_name(index, is_axiom)
        }
        fn get_num_operators(&self) -> usize {
            self.base.get_num_operators()
        }
        fn get_num_operator_preconditions(&self, index: usize, is_axiom: bool) -> usize {
            self.base.get_num_operator_preconditions(index, is_axiom)
        }
        fn get_operator_precondition(
            &self,
            index: usize,
            precond_index: usize,
            is_axiom: bool,
        ) -> &ExplicitFact {
            self.base
                .get_operator_precondition(index, precond_index, is_axiom)
        }
        fn get_num_operator_effects(&self, index: usize, is_axiom: bool) -> usize {
            self.base.get_num_operator_effects(index, is_axiom)
        }
        fn get_num_operator_effect_conditions(
            &self,
            index: usize,
            eff_index: usize,
            is_axiom: bool,
        ) -> usize {
            self.base
                .get_num_operator_effect_conditions(index, eff_index, is_axiom)
        }
        fn get_operator_effect_condition(
            &self,
            index: usize,
            eff_index: usize,
            cond_index: usize,
            is_axiom: bool,
        ) -> &ExplicitFact {
            self.base
                .get_operator_effect_condition(index, eff_index, cond_index, is_axiom)
        }
        fn get_operator_effect(
            &self,
            index: usize,
            eff_index: usize,
            is_axiom: bool,
        ) -> &ExplicitFact {
            self.base.get_operator_effect(index, eff_index, is_axiom)
        }
        fn convert_operator_index(&self, index: usize, ancestor_task: &dyn AbstractNumericTask) {
            self.base.convert_operator_index(index, ancestor_task)
        }
        fn get_num_axioms(&self) -> usize {
            self.base.get_num_axioms()
        }
        fn get_num_goals(&self) -> usize {
            1
        }
        fn get_goal_fact(&self, index: usize) -> &ExplicitFact {
            assert_eq!(index, 0, "SingleGoalTask only exposes one goal");
            &self.goal
        }
        fn get_initial_propositional_state_values(&self) -> Ref<'_, Vec<usize>> {
            self.base.get_initial_propositional_state_values()
        }
        fn get_initial_numeric_state_values(&self) -> Ref<'_, Vec<f64>> {
            self.base.get_initial_numeric_state_values()
        }
        fn get_initial_propositional_state_values_mut(&self) -> RefMut<'_, Vec<usize>> {
            self.base.get_initial_propositional_state_values_mut()
        }
        fn get_initial_numeric_state_values_mut(&self) -> RefMut<'_, Vec<f64>> {
            self.base.get_initial_numeric_state_values_mut()
        }
        fn set_initial_numeric_state_values(&self, values: Vec<f64>) {
            self.base.set_initial_numeric_state_values(values)
        }
        fn set_initial_propositional_state_values(&self, values: Vec<usize>) {
            self.base.set_initial_propositional_state_values(values)
        }
        fn convert_ancestor_state_values(
            &self,
            ancestor_state_values: &[usize],
            ancestor_task: &dyn AbstractNumericTask,
        ) -> Vec<usize> {
            self.base
                .convert_ancestor_state_values(ancestor_state_values, ancestor_task)
        }
        fn get_num_cmp_axioms(&self) -> usize {
            self.base.get_num_cmp_axioms()
        }
        fn abstract_state_values(
            &self,
            propositional_values: &[usize],
            numeric_values: &[f64],
        ) -> Result<(Vec<usize>, Vec<f64>), String> {
            self.base
                .abstract_state_values(propositional_values, numeric_values)
        }
        fn evaluated_initial_abstract_state_values(
            &self,
        ) -> Result<(Vec<usize>, Vec<f64>), String> {
            self.base.evaluated_initial_abstract_state_values()
        }
        fn abstract_operator_cost(&self, operator_id: usize) -> f64 {
            self.base.abstract_operator_cost(operator_id)
        }
    }
}

// ---------------------------------------------------------------------------
// Greedy order utilities
// ---------------------------------------------------------------------------

fn deduplicate_orders(orders: Vec<Vec<usize>>) -> Vec<Vec<usize>> {
    let mut seen = HashSet::with_capacity(orders.len());
    orders
        .into_iter()
        .filter(|order| seen.insert(order.clone()))
        .collect()
}

fn deduplicate_specialist_orders(
    orders: Vec<(Option<usize>, Vec<usize>)>,
) -> Vec<(Option<usize>, Vec<usize>)> {
    let mut index_by_order = HashMap::<Vec<usize>, usize>::with_capacity(orders.len());
    let mut unique = Vec::<(Option<usize>, Vec<usize>)>::with_capacity(orders.len());
    for (specialist_goal_id, order) in orders {
        if let Some(&index) = index_by_order.get(&order) {
            if unique[index].0.is_none() {
                unique[index].0 = specialist_goal_id;
            }
            continue;
        }
        index_by_order.insert(order.clone(), unique.len());
        unique.push((specialist_goal_id, order));
    }
    unique
}

fn compute_score(h: f64, stolen_costs: f64, scoring_function: ScoringFunction) -> f64 {
    match scoring_function {
        ScoringFunction::MaxHeuristic => h,
        ScoringFunction::MinStolenCosts => -stolen_costs,
        ScoringFunction::MaxHeuristicPerStolenCosts => h / stolen_costs.max(1.0),
    }
}

fn compute_abstract_operator_order_score(
    h: f64,
    stolen_costs: f64,
    scoring_function: ScoringFunction,
    abstraction: &DomainAbstraction,
) -> f64 {
    if h <= 1e-9 {
        return -1.0e100;
    }
    let base = compute_score(h, stolen_costs, scoring_function);
    let size_penalty = ((abstraction_state_count(abstraction) as f64) / 10_000.0)
        .max(1.0)
        .sqrt();
    base / size_penalty
}

fn standalone_current_h_values(
    state: &ScpOnlineState,
    abstract_state_ids: &[Option<usize>],
    _num_domain_abstractions: usize,
) -> Vec<f64> {
    (0..state.h_values_by_abstraction.len())
        .map(|abstraction_id| {
            state
                .h_values_by_abstraction
                .get(abstraction_id)
                .map(|distances| {
                    current_h_for_distances(abstraction_id, distances, abstract_state_ids)
                })
                .unwrap_or(0.0)
        })
        .collect()
}

fn compute_stolen_costs(wanted: f64, surplus: f64) -> f64 {
    if !wanted.is_finite() || !surplus.is_finite() {
        return 0.0;
    }
    let rest = surplus + wanted;
    if rest >= 0.0 {
        (0.0_f64).max(wanted - rest)
    } else {
        wanted.max(rest)
    }
}

fn compute_costs_stolen_by_heuristic(saturated: &[f64], surplus: &[f64]) -> f64 {
    saturated
        .iter()
        .zip(surplus.iter())
        .map(|(&s, &su)| compute_stolen_costs(s, su))
        .sum()
}

fn compute_surplus_cost(saturated_by_abs: &[Vec<f64>], op_id: usize, remaining_cost: f64) -> f64 {
    let sum: f64 = saturated_by_abs
        .iter()
        .map(|costs| costs.get(op_id).copied().unwrap_or(f64::NEG_INFINITY))
        .filter(|&w| w > f64::NEG_INFINITY)
        .sum();
    if !remaining_cost.is_finite() || !sum.is_finite() {
        return f64::INFINITY;
    }
    remaining_cost - sum
}

fn compute_all_surplus_costs(costs: &[f64], saturated_by_abs: &[Vec<f64>]) -> Vec<f64> {
    (0..costs.len())
        .map(|op_id| compute_surplus_cost(saturated_by_abs, op_id, costs[op_id]))
        .collect()
}

fn apply_operator_costs_from_slice(
    operators: &mut [AbstractOperator],
    operator_costs: &[f64],
) -> Result<(), EvaluationError> {
    for op in operators {
        if op.concrete_op_ids.is_empty() {
            return Err(EvaluationError::ComputationFailed(
                "abstract operator without concrete labels".to_string(),
            ));
        }
        let mut cost = f64::INFINITY;
        for &concrete_op_id in &op.concrete_op_ids {
            let concrete_cost = operator_costs.get(concrete_op_id).copied().ok_or_else(|| {
                EvaluationError::ComputationFailed(format!(
                    "missing residual cost for concrete operator {concrete_op_id}"
                ))
            })?;
            if !concrete_cost.is_finite() {
                return Err(EvaluationError::ComputationFailed(format!(
                    "residual cost for concrete operator {concrete_op_id} must be finite"
                )));
            }
            cost = cost.min(concrete_cost);
        }
        op.cost = cost;
    }
    Ok(())
}

fn reduce_costs(
    remaining_costs: &mut [f64],
    saturated_costs: &[f64],
) -> Result<(), EvaluationError> {
    if remaining_costs.len() != saturated_costs.len() {
        return Err(EvaluationError::ComputationFailed(format!(
            "cost vector length mismatch: remaining={}, saturated={}",
            remaining_costs.len(),
            saturated_costs.len()
        )));
    }
    for (op_id, (r, s)) in remaining_costs
        .iter_mut()
        .zip(saturated_costs.iter())
        .enumerate()
    {
        if !s.is_finite() {
            continue;
        }
        if *s <= 1e-9 {
            continue;
        }
        let new_remaining = *r - *s;
        if new_remaining < -1e-9 {
            return Err(EvaluationError::ComputationFailed(format!(
                "label residual cost underflow for operator {op_id}: remaining={r}, saturated={s}, result={new_remaining}"
            )));
        }
        *r = new_remaining.max(0.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use planforge_sas::numeric_task::{
        Effect, ExplicitFact, ExplicitVariable, Metric, NumericRootTask, Operator,
    };
    use planforge_sas::state_registry::StateRegistry;

    use crate::evaluation::cartesian_abstractions::{
        CartesianAbstractionCollectionConfig, CartesianAbstractionCollectionGenerator,
        CartesianAbstractionConfig, CartesianAbstractionGenerator,
    };
    use crate::evaluation::domain_abstractions::cegar::CegarConfig;
    use crate::evaluation::domain_abstractions::domain_abstraction_generator::DomainAbstractionGenerator;
    use crate::evaluation::pattern_databases::pattern_database::PatternDatabase;
    use crate::evaluation::pattern_databases::projected_task::{Pattern, ProjectedTask};

    fn binary_variable(name: &str) -> ExplicitVariable {
        ExplicitVariable::new(
            2,
            name.to_string(),
            vec![format!("{name}=0"), format!("{name}=1")],
            None,
            1,
        )
    }

    fn independent_goals_task() -> NumericRootTask {
        NumericRootTask::new(
            1,
            Metric::new(true, None),
            vec![binary_variable("p"), binary_variable("q")],
            vec![],
            vec![ExplicitFact::new(0, 1), ExplicitFact::new(1, 1)],
            vec![],
            vec![0, 0],
            vec![],
            vec![
                Operator::new(
                    "set-p".to_string(),
                    vec![],
                    vec![Effect::new(vec![], 0, Some(0), 1)],
                    vec![],
                    2,
                ),
                Operator::new(
                    "set-q".to_string(),
                    vec![],
                    vec![Effect::new(vec![], 1, Some(0), 1)],
                    vec![],
                    3,
                ),
            ],
            vec![],
            vec![],
            vec![],
            ExplicitFact::new(0, 0),
        )
    }

    fn cartesian_abstraction(task: &NumericRootTask) -> CartesianAbstraction {
        CartesianAbstractionGenerator::new(CartesianAbstractionConfig {
            max_states: 16,
            max_time: None,
            combine_labels: false,
            compute_operator_footprints: true,
            random_seed: None,
            debug: false,
            ..Default::default()
        })
        .unwrap()
        .generate(task)
        .unwrap()
    }

    fn scp_config(saturator: Saturator, abstract_operator: bool) -> ScpOnlineConfig {
        ScpOnlineConfig {
            max_time: 10.0,
            table_construction_max_time: 10.0,
            interval: usize::MAX,
            order_optimization_max_time: 0.0,
            saturator,
            random_seed: Some(1),
            use_abstract_operator_cost_partitioning: abstract_operator,
            ..ScpOnlineConfig::default()
        }
    }

    fn evaluate_initial(
        task: &NumericRootTask,
        heuristic: &dyn Heuristic,
    ) -> Result<f64, EvaluationError> {
        let mut registry = StateRegistry::for_task(std::sync::Arc::new(task));
        let initial_state = registry.get_initial_state();
        let eval_state =
            EvaluationState::new_with_registry(&initial_state, 0.0, false, task, &registry);
        heuristic.compute_heuristic(&eval_state)
    }

    #[test]
    fn reduce_costs_rejects_significant_underflow() {
        let mut remaining = vec![1.0];
        let saturated = vec![1.5];

        let err = reduce_costs(&mut remaining, &saturated).unwrap_err();
        assert!(format!("{err}").contains("underflow"));
    }

    #[test]
    fn reduce_costs_clamps_tiny_negative_roundoff() {
        let mut remaining = vec![1.0];
        let saturated = vec![1.0 + 1e-12];

        reduce_costs(&mut remaining, &saturated).unwrap();
        assert_eq!(remaining, vec![0.0]);
    }

    #[test]
    fn label_candidates_always_include_max_heuristic_greedy_order() {
        let base_order = vec![2, 0, 1];
        let h_values = vec![3.0, 7.0, 5.0];

        let max_order = max_heuristic_greedy_order(&base_order, &h_values);

        assert_eq!(max_order, vec![1, 2, 0]);
    }

    #[test]
    fn compact_goal_cover_orders_pair_complementary_anchor_variants() {
        let task = independent_goals_task();
        let abstractions =
            CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
                abstraction: CartesianAbstractionConfig {
                    max_states: 16,
                    max_time: None,
                    combine_labels: false,
                    compute_operator_footprints: true,
                    random_seed: Some(1),
                    debug: false,
                    ..Default::default()
                },
                variants_per_goal: 4,
                max_collection_states: 128,
                total_max_time: None,
                progressive_goal_roots: false,
            })
            .unwrap()
            .generate(&task)
            .unwrap();
        let base_order = (0..abstractions.len()).collect::<Vec<_>>();
        let standalone_h = abstractions
            .iter()
            .map(|abstraction| {
                abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash]
            })
            .collect::<Vec<_>>();
        let order = |variant| {
            cartesian_goal_cover_order(&base_order, 0, &abstractions, &standalone_h, true, variant)
                .unwrap()
        };
        let goal = |component_id: usize| {
            abstractions[component_id]
                .metadata
                .collection_goal_id
                .unwrap()
        };
        let baseline = order(GoalCoverOrderVariant {
            compact: true,
            ..Default::default()
        });
        assert_eq!(baseline.len(), 3);
        assert_eq!(goal(baseline[0]), goal(baseline[1]));
        assert_ne!(goal(baseline[0]), goal(baseline[2]));
        let first = &abstractions[baseline[0]].metadata;
        let complement = &abstractions[baseline[1]].metadata;
        assert_ne!(first.refinement_direction, complement.refinement_direction);
        assert_eq!(first.split_selection_rank, complement.split_selection_rank);

        let other_goal = order(GoalCoverOrderVariant {
            anchor_goal_offset: 1,
            compact: true,
            ..Default::default()
        });
        assert_ne!(goal(other_goal[0]), goal(baseline[0]));

        let other_anchor = order(GoalCoverOrderVariant {
            anchor_offset: 1,
            compact: true,
            ..Default::default()
        });
        assert_ne!(other_anchor[0], baseline[0]);

        let other_representative = order(GoalCoverOrderVariant {
            representative_round: 1,
            compact: true,
            ..Default::default()
        });
        assert_eq!(&other_representative[..2], &baseline[..2]);
        assert_ne!(other_representative[2], baseline[2]);

        let other_complement = order(GoalCoverOrderVariant {
            complementary_round: 1,
            compact: true,
            ..Default::default()
        });
        assert_eq!(other_complement[0], baseline[0]);
        assert_ne!(other_complement[1], baseline[1]);

        let mut mixed = abstractions.clone();
        mixed.push(cartesian_abstraction(&task));
        let mixed_order = (0..mixed.len()).collect::<Vec<_>>();
        let mixed_h = mixed
            .iter()
            .map(|abstraction| {
                abstraction.distance_table.distances[abstraction.distance_table.initial_state_hash]
            })
            .collect::<Vec<_>>();
        let structural_id = mixed.len() - 1;
        let prefixed = cartesian_goal_cover_order(
            &mixed_order,
            0,
            &mixed,
            &mixed_h,
            false,
            GoalCoverOrderVariant {
                non_goal_prefix: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(prefixed[0], structural_id);
    }

    #[test]
    fn compact_goal_cover_schedule_visits_every_anchor_goal_and_four_variants() {
        let variants = compact_goal_cover_variants(18, 8, true);
        let anchor_goals = variants
            .iter()
            .map(|variant| variant.anchor_goal_offset)
            .collect::<HashSet<_>>();

        assert_eq!(variants.len(), 72);
        assert_eq!(anchor_goals, (0..18).collect());
    }

    #[test]
    fn compact_goal_cover_schedule_supports_one_abstraction_per_goal() {
        let variants = compact_goal_cover_variants(18, 1, true);
        let anchor_goals = variants
            .iter()
            .map(|variant| variant.anchor_goal_offset)
            .collect::<HashSet<_>>();

        assert_eq!(variants.len(), 18);
        assert_eq!(anchor_goals, (0..18).collect());
    }

    #[test]
    fn offline_diversification_retains_available_specialists_per_cartesian_goal() {
        let task = std::sync::Arc::new(independent_goals_task());
        let abstractions =
            CartesianAbstractionCollectionGenerator::new(CartesianAbstractionCollectionConfig {
                abstraction: CartesianAbstractionConfig {
                    max_states: 16,
                    max_time: None,
                    combine_labels: false,
                    compute_operator_footprints: true,
                    random_seed: Some(1),
                    debug: false,
                    ..Default::default()
                },
                variants_per_goal: 4,
                max_collection_states: 128,
                total_max_time: None,
                progressive_goal_roots: true,
            })
            .unwrap()
            .generate(&*task)
            .unwrap();
        let expected_goals = abstractions
            .iter()
            .filter_map(|abstraction| abstraction.metadata.collection_goal_id)
            .collect::<HashSet<_>>();
        let components = abstractions
            .into_iter()
            .map(|abstraction| AbstractionComponent::cartesian(None, abstraction))
            .collect();
        let mut config = scp_config(Saturator::All, true);
        config.online = false;
        config.diversify = true;
        config.samples = 16;
        config.max_orders = 1 + expected_goals.len() * 4;
        config.initial_order_generation_max_time = 10.0;
        let heuristic =
            SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
                None,
                components,
                config,
                &*task,
                task.clone(),
            )
            .unwrap();

        evaluate_initial(&task, &heuristic).unwrap();
        let retained_goals = heuristic
            .state
            .borrow()
            .cp_heuristics
            .iter()
            .filter_map(|cp| cp.specialist_goal_id)
            .collect::<HashSet<_>>();
        assert_eq!(retained_goals, expected_goals);
        let mut retained_per_goal = HashMap::<usize, usize>::new();
        for goal_id in heuristic
            .state
            .borrow()
            .cp_heuristics
            .iter()
            .filter_map(|cp| cp.specialist_goal_id)
        {
            *retained_per_goal.entry(goal_id).or_default() += 1;
        }
        assert!(
            retained_per_goal
                .values()
                .all(|&count| (2..=4).contains(&count)),
            "retained specialists by goal: {retained_per_goal:?}"
        );
    }

    #[test]
    fn offline_scp_retains_an_order_that_is_weaker_only_at_the_initial_state() {
        let partition = |distances| CostPartitioningHeuristic {
            lookup_tables: vec![LookupTable {
                abstraction_id: 0,
                distances,
                unknown_value: f64::INFINITY,
            }],
            specialist_goal_id: None,
        };
        let mut state = ScpOnlineState::new(Some(1));
        let mut initial_h = 0.0;

        SaturatedCostPartitioningOnlineHeuristic::retain_cp(
            &mut state,
            partition(vec![5.0, 1.0]),
            &[Some(0)],
            &mut initial_h,
            true,
            usize::MAX,
        );
        SaturatedCostPartitioningOnlineHeuristic::retain_cp(
            &mut state,
            partition(vec![4.0, 4.0]),
            &[Some(0)],
            &mut initial_h,
            true,
            usize::MAX,
        );

        assert_eq!(state.cp_heuristics.len(), 2);
        assert_eq!(initial_h, 5.0);
        assert_eq!(
            SaturatedCostPartitioningOnlineHeuristic::compute_max_h(&state, &[Some(1)]),
            4.0
        );
    }

    #[test]
    fn online_scp_rejects_a_non_improving_state_specific_order() {
        let partition = |distances| CostPartitioningHeuristic {
            lookup_tables: vec![LookupTable {
                abstraction_id: 0,
                distances,
                unknown_value: f64::INFINITY,
            }],
            specialist_goal_id: None,
        };
        let mut state = ScpOnlineState::new(Some(1));
        let mut current_h = 0.0;

        SaturatedCostPartitioningOnlineHeuristic::retain_cp(
            &mut state,
            partition(vec![5.0]),
            &[Some(0)],
            &mut current_h,
            false,
            usize::MAX,
        );
        SaturatedCostPartitioningOnlineHeuristic::retain_cp(
            &mut state,
            partition(vec![4.0]),
            &[Some(0)],
            &mut current_h,
            false,
            usize::MAX,
        );

        assert_eq!(state.cp_heuristics.len(), 1);
        assert_eq!(current_h, 5.0);
    }

    #[test]
    fn cartesian_scp_supports_every_saturator_in_both_cost_modes() {
        let task = independent_goals_task();
        for saturator in [Saturator::All, Saturator::Perim, Saturator::Perimstar] {
            for abstract_operator in [false, true] {
                let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
                let heuristic = SaturatedCostPartitioningOnlineHeuristic::from_components(
                    None,
                    vec![component],
                    scp_config(saturator, abstract_operator),
                    &task,
                )
                .unwrap();
                assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
            }
        }
    }

    #[test]
    fn offline_scp_releases_cartesian_construction_data_after_first_evaluation() {
        let task = independent_goals_task();
        let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
        let mut config = scp_config(Saturator::All, true);
        config.online = false;
        config.interval = 1;
        let heuristic = SaturatedCostPartitioningOnlineHeuristic::from_components(
            None,
            vec![component],
            config,
            &task,
        )
        .unwrap();

        assert!(heuristic.cartesian_abstractions.borrow().is_some());
        assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
        assert!(heuristic.cartesian_abstractions.borrow().is_none());
        assert!(heuristic.state.borrow().improvement_ended);
        assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
    }

    #[test]
    fn abstract_operator_scp_combines_all_backend_types() {
        let task = independent_goals_task();
        let mut domain_config = CegarConfig::default();
        domain_config.max_abstraction_size = 16;
        domain_config.combine_labels = false;
        domain_config.compute_operator_footprints = true;
        let domain = DomainAbstractionGenerator::new(domain_config)
            .unwrap()
            .generate(&task)
            .unwrap();
        let pattern = Pattern::new(vec![1], vec![]);
        let pdb = PatternDatabase::new(ProjectedTask::new(&task, &pattern).unwrap(), 32).unwrap();
        let components = vec![
            AbstractionComponent::domain(None, domain),
            AbstractionComponent::cartesian(None, cartesian_abstraction(&task)),
            AbstractionComponent::pattern_database(pdb),
        ];
        let heuristic = SaturatedCostPartitioningOnlineHeuristic::from_components(
            None,
            components,
            scp_config(Saturator::All, true),
            &task,
        )
        .unwrap();

        assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
    }

    #[test]
    fn offline_diversification_supports_mixed_abstraction_backends() {
        let task = std::sync::Arc::new(independent_goals_task());
        let mut domain_config = CegarConfig::default();
        domain_config.max_abstraction_size = 16;
        domain_config.combine_labels = false;
        domain_config.compute_operator_footprints = true;
        let domain = DomainAbstractionGenerator::new(domain_config)
            .unwrap()
            .generate(&*task)
            .unwrap();
        let pattern = Pattern::new(vec![1], vec![]);
        let pdb = PatternDatabase::new(ProjectedTask::new(&*task, &pattern).unwrap(), 32).unwrap();
        let components = vec![
            AbstractionComponent::domain(None, domain),
            AbstractionComponent::cartesian(None, cartesian_abstraction(&task)),
            AbstractionComponent::pattern_database(pdb),
        ];
        let mut config = scp_config(Saturator::All, true);
        config.online = false;
        config.diversify = true;
        config.samples = 16;
        config.max_orders = 8;
        let heuristic =
            SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
                None,
                components,
                config,
                &*task,
                task.clone(),
            )
            .unwrap();

        assert_eq!(evaluate_initial(&task, &heuristic).unwrap(), 5.0);
        let state = heuristic.state.borrow();
        assert!(state.improvement_ended);
        assert!(state.offline_sample_ids.is_empty());
        assert!(!state.cp_heuristics.is_empty());
        assert!(state.cp_heuristics.len() <= 8);
        drop(state);
        assert!(heuristic.sampling_task.borrow().is_none());
        assert!(heuristic.cartesian_abstractions.borrow().is_none());
    }

    #[test]
    fn offline_diversification_rejects_online_construction() {
        let task = std::sync::Arc::new(independent_goals_task());
        let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
        let mut config = scp_config(Saturator::All, true);
        config.online = true;
        config.diversify = true;

        let result = SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
            None,
            vec![component],
            config,
            &*task,
            task.clone(),
        );
        let error = match result {
            Ok(_) => panic!("online diversified construction must be rejected"),
            Err(error) => error,
        };

        assert!(error.to_string().contains("requires online=false"));
    }

    #[test]
    fn diverse_orders_require_offline_diversification() {
        let task = independent_goals_task();
        let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
        let mut config = scp_config(Saturator::All, true);
        config.order_generator = OrderGenerator::Diverse;

        let result = SaturatedCostPartitioningOnlineHeuristic::from_components(
            None,
            vec![component],
            config,
            &task,
        );
        let error = match result {
            Ok(_) => panic!("diverse orders without diversification must be rejected"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("diverse SCP orders require diversify=true")
        );
    }

    #[test]
    fn offline_diversification_rejects_nan_initial_order_budget() {
        let task = std::sync::Arc::new(independent_goals_task());
        let component = AbstractionComponent::cartesian(None, cartesian_abstraction(&task));
        let mut config = scp_config(Saturator::All, true);
        config.online = false;
        config.diversify = true;
        config.initial_order_generation_max_time = f64::NAN;

        let result = SaturatedCostPartitioningOnlineHeuristic::from_components_with_sampling_task(
            None,
            vec![component],
            config,
            &*task,
            task.clone(),
        );
        let error = match result {
            Ok(_) => panic!("NaN initial-order budget must be rejected"),
            Err(error) => error,
        };

        assert!(
            error
                .to_string()
                .contains("initial_order_generation_max_time >= 0")
        );
    }

    #[test]
    fn offline_diversifier_keeps_partitions_that_improve_different_samples() {
        let partition = |distances| CostPartitioningHeuristic {
            lookup_tables: vec![LookupTable {
                abstraction_id: 0,
                distances,
                unknown_value: f64::INFINITY,
            }],
            specialist_goal_id: None,
        };
        let samples = vec![vec![Some(0)], vec![Some(1)]];
        let mut best = vec![f64::NEG_INFINITY; samples.len()];
        let mut portfolio = Vec::new();
        let mut size_kb = 0;

        assert!(retain_if_sample_improving(
            partition(vec![5.0, 1.0]),
            &samples,
            &mut best,
            &mut portfolio,
            &mut size_kb,
            usize::MAX,
        ));
        assert!(retain_if_sample_improving(
            partition(vec![4.0, 6.0]),
            &samples,
            &mut best,
            &mut portfolio,
            &mut size_kb,
            usize::MAX,
        ));
        assert!(!retain_if_sample_improving(
            partition(vec![5.0, 5.0]),
            &samples,
            &mut best,
            &mut portfolio,
            &mut size_kb,
            usize::MAX,
        ));

        assert_eq!(portfolio.len(), 2);
        assert_eq!(best, vec![5.0, 6.0]);
    }

    #[test]
    fn offline_random_walk_lengths_are_deterministic() {
        let mut left = SmallRng::seed_from_u64(7);
        let mut right = SmallRng::seed_from_u64(7);
        let left_lengths = (0..20)
            .map(|_| random_walk_length(110.0, 1.0, &mut left).unwrap())
            .collect::<Vec<_>>();
        let right_lengths = (0..20)
            .map(|_| random_walk_length(110.0, 1.0, &mut right).unwrap())
            .collect::<Vec<_>>();

        assert_eq!(left_lengths, right_lengths);
        assert!(left_lengths.iter().all(|&length| length <= 440));
    }
}
