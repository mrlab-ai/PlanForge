pub mod axiom;
pub mod causal_graph;
pub mod fact;
pub mod helper_functions;
pub mod max_dag;
pub mod mutex_group;
pub mod operator;
pub mod scc;
pub mod state;
pub mod variable;

use std::io::Write;

use tracing::{debug, info};

use self::axiom::{AxiomFunctionalComparison, AxiomNumericComputation, AxiomRelational};
use self::causal_graph::CausalGraph;
use self::fact::ExplicitFact;
use self::helper_functions::to_sas_writer;
use self::mutex_group::MutexGroup;
use self::operator::Operator;
use self::state::State;
use self::variable::{ExplicitVariable, NumericVariable};
use crate::sas_tasks::SASTask;

pub const SAS_FILE_VERSION: i32 = 4;
pub const PRE_FILE_VERSION: i32 = SAS_FILE_VERSION;

#[derive(Debug, Clone)]
pub struct Metric {
    pub optimization_criterion: char,
    pub index: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct GlobalConstraint {
    pub var: usize,
    pub value: usize,
}
pub type Condition = Vec<ExplicitFact>;

/// The translated task in the model the causal-graph analysis works on. It
/// holds the same task as [`SASTask`], plus the per-variable bookkeeping --
/// level, necessity, numeric type -- that the analysis fills in.
pub struct PreprocessedTask {
    pub metric: Metric,
    pub variables: Vec<ExplicitVariable>,
    pub numeric_variables: Vec<NumericVariable>,
    pub mutexes: Vec<MutexGroup>,
    pub initial_state: State,
    pub goals: Vec<ExplicitFact>,
    pub operators: Vec<Operator>,
    pub axioms_rel: Vec<AxiomRelational>,
    pub axioms_func_comp: Vec<AxiomFunctionalComparison>,
    pub axioms_numeric: Vec<AxiomNumericComputation>,
    pub global_constraint: Option<GlobalConstraint>,
}

impl PreprocessedTask {
    /// The axioms of each kind are grouped by the axiom layer of the variable
    /// they define, so that the analysis and the output see them stratified.
    /// The sort is stable, so axioms sharing a layer keep the order the
    /// translation produced them in.
    pub fn from_sas(task: &SASTask) -> Self {
        let sas_vars = &task.variables;
        let mut variables: Vec<ExplicitVariable> = sas_vars
            .value_names
            .iter()
            .zip(&sas_vars.axiom_layers)
            .enumerate()
            .map(|(index, (values, &layer))| ExplicitVariable::new(index, layer, values.clone()))
            .collect();

        let sas_num_vars = &task.numeric_variables;
        let mut numeric_variables: Vec<NumericVariable> = sas_num_vars
            .variable_names
            .iter()
            .zip(&sas_num_vars.axiom_layers)
            .zip(&sas_num_vars.types)
            .enumerate()
            .map(|(index, ((name, &layer), sas_type))| {
                NumericVariable::new(index, sas_type, layer, name.clone())
            })
            .collect();

        let mut axioms_rel: Vec<AxiomRelational> =
            task.axioms.iter().map(AxiomRelational::from_sas).collect();
        axioms_rel.sort_by_key(|axiom| variables[axiom.get_effect_var()].get_layer());

        let mut axioms_func_comp: Vec<AxiomFunctionalComparison> = task
            .comp_axioms
            .iter()
            .map(|axiom| {
                AxiomFunctionalComparison::from_sas(axiom, &mut variables, &numeric_variables)
            })
            .collect();
        axioms_func_comp.sort_by_key(|axiom| variables[axiom.get_effect_var()].get_layer());

        let mut axioms_numeric: Vec<AxiomNumericComputation> = task
            .numeric_axioms
            .iter()
            .map(|axiom| AxiomNumericComputation::from_sas(axiom, &mut numeric_variables))
            .collect();
        axioms_numeric.sort_by_key(|axiom| numeric_variables[axiom.get_effect_var()].get_layer());

        let (criterion, metric_index) = &task.metric;
        let mut criterion = criterion.chars();
        let optimization_criterion = criterion
            .next()
            .expect("the metric names an optimization criterion");
        assert!(
            criterion.next().is_none(),
            "the optimization criterion is a single character, got {:?}",
            task.metric.0
        );
        let index = usize::try_from(*metric_index)
            .expect("the metric names a numeric variable; unit cost is not preprocessable");

        let (gc_var, gc_value) = task.global_constraint;

        Self {
            metric: Metric {
                optimization_criterion,
                index,
            },
            variables,
            numeric_variables,
            mutexes: task.mutexes.iter().map(MutexGroup::from_sas).collect(),
            initial_state: State::from_sas(&task.init),
            goals: task
                .goal
                .pairs
                .iter()
                .map(|&(var, value)| ExplicitFact { var, value })
                .collect(),
            operators: task.operators.iter().map(Operator::from_sas).collect(),
            axioms_rel,
            axioms_func_comp,
            axioms_numeric,
            global_constraint: Some(GlobalConstraint {
                var: gc_var,
                value: gc_value,
            }),
        }
    }
}

/// The task after the causal graph has ordered and pruned its variables.
///
/// `original_variables` and `original_numeric_variables` are indexed by a
/// variable's *old* id and hold the level it was given, or `-1` if it was
/// pruned; every reference in an operator, axiom, mutex group or goal is still
/// phrased in old ids and is remapped through them on the way out.
pub struct ReorderedTask {
    pub metric: Metric,
    pub original_variables: Vec<ExplicitVariable>,
    pub original_numeric_variables: Vec<NumericVariable>,
    /// The variables that survived, in their new order.
    pub variables: Vec<ExplicitVariable>,
    pub numeric_variables: Vec<NumericVariable>,
    pub mutexes: Vec<MutexGroup>,
    pub initial_state: State,
    pub goals: Vec<ExplicitFact>,
    pub operators: Vec<Operator>,
    pub axioms_rel: Vec<AxiomRelational>,
    pub axioms_func_comp: Vec<AxiomFunctionalComparison>,
    pub axioms_numeric: Vec<AxiomNumericComputation>,
    pub global_constraint: Option<GlobalConstraint>,
}

impl ReorderedTask {
    /// The number of facts, variables, goals and effects the search will hold,
    /// which is what "how big is this task" means in the logs.
    fn encoding_size(&self) -> usize {
        let facts: usize = self.variables.iter().map(ExplicitVariable::get_range).sum();
        let mut size =
            self.variables.len() + self.numeric_variables.len() + facts + self.goals.len();
        size += self
            .mutexes
            .iter()
            .map(MutexGroup::get_encoding_size)
            .sum::<usize>();
        size += self
            .operators
            .iter()
            .map(Operator::get_encoding_size)
            .sum::<usize>();
        size += self
            .axioms_rel
            .iter()
            .map(AxiomRelational::get_encoding_size)
            .sum::<usize>();
        size += self
            .axioms_numeric
            .iter()
            .map(AxiomNumericComputation::get_encoding_size)
            .sum::<usize>();
        size += self
            .axioms_func_comp
            .iter()
            .map(AxiomFunctionalComparison::get_encoding_size)
            .sum::<usize>();
        size
    }
}

/// Orders and prunes the variables of `task` by its causal graph, and writes
/// the result as the SAS+ file the search reads.
///
/// `prune_variables` drops the variables no goal, global constraint or metric
/// depends on. Turning it off keeps every variable and only reorders.
pub fn write_reordered_sas<W: Write>(task: &SASTask, prune_variables: bool, outfile: &mut W) {
    let translated = PreprocessedTask::from_sas(task);
    let metric_index_before = translated.metric.index;

    info!("Building causal graph...");
    let reordered = CausalGraph::new(translated, prune_variables).finalize();

    debug!(
        "Metric index changed from {} to {}",
        metric_index_before, reordered.metric.index
    );
    info!(
        "Preprocessor facts: {}",
        reordered
            .variables
            .iter()
            .map(ExplicitVariable::get_range)
            .sum::<usize>()
    );
    info!(
        "Preprocessor derived variables: {}",
        reordered
            .variables
            .iter()
            .filter(|var| var.is_derived())
            .count()
    );
    info!("Preprocessor task size: {}", reordered.encoding_size());

    info!("Writing output...");
    to_sas_writer(&reordered, outfile);
    info!("done");
}
