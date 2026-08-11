//! The causal-graph pass over a translated task.
//!
//! It orders the SAS variables so that every variable comes after the ones it
//! depends on, drops the variables no goal, global constraint or metric needs,
//! and writes the result as the SAS+ file the search reads.
//!
//! The task itself stays the translator's [`SASTask`]. Everything this pass
//! works out about a variable -- whether the task still needs it, and the
//! position it ends up in -- lives beside it in [`VarState`] and
//! [`NumericVarState`], indexed by the variable's original id.

pub mod causal_graph;
pub mod max_dag;
pub mod sas_parts;

use std::io::Write;

use planforge_sas::numeric_task::NumericType;
use tracing::{debug, info};

use self::causal_graph::CausalGraph;
use crate::sas_tasks::{SASGoal, SASTask, SasFact, inverted_comparator};

/// The level of a variable the reordering has not placed: either because the
/// analysis has not run yet, or because the variable was pruned.
pub const NO_LEVEL: i32 = -1;

/// The axiom layer of a variable that no axiom derives.
pub const NO_LAYER: i32 = -1;

/// The metric the search optimizes: a direction, and the numeric variable that
/// accumulates the plan's cost.
#[derive(Debug, Clone, Copy)]
pub struct Metric {
    pub optimization_criterion: char,
    pub index: usize,
}

impl Metric {
    /// The translation spells the metric as a direction and the index of the
    /// numeric variable holding the cost.
    fn from_sas((criterion, index): &(String, i64)) -> Self {
        let mut chars = criterion.chars();
        let optimization_criterion = chars
            .next()
            .expect("the metric names an optimization criterion");
        assert!(
            chars.next().is_none(),
            "the optimization criterion is a single character, got {criterion:?}"
        );
        let index = usize::try_from(*index)
            .expect("the metric names a numeric variable; unit cost is not preprocessable");
        Self {
            optimization_criterion,
            index,
        }
    }
}

/// What a numeric variable holds, as the SAS file spells it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumType {
    /// Not classified yet: the translation only distinguishes the variables
    /// that carry a value in their own right from the derived and constant
    /// ones, and this pass decides which of the former the search needs and
    /// which merely instrument the metric.
    Unknown,
    Constant,
    Derived,
    Instrumentation,
    Regular,
}

impl NumType {
    /// The type the search knows this variable by. The two enums differ in
    /// exactly one place — [`NumType::Unknown`] is a state of the analysis and
    /// not a kind of variable — so the classification has to be finished by the
    /// time a variable reaches either consumer.
    fn as_numeric_type(self) -> NumericType {
        match self {
            NumType::Constant => NumericType::Constant,
            NumType::Derived => NumericType::Derived,
            NumType::Instrumentation => NumericType::Cost,
            NumType::Regular => NumericType::Regular,
            NumType::Unknown => {
                unreachable!("a numeric variable reached the output unclassified")
            }
        }
    }
}

/// What the pass works out about one propositional variable.
#[derive(Debug, Clone, Copy)]
pub struct VarState {
    level: i32,
    necessary: bool,
}

impl VarState {
    /// A variable the analysis has not looked at: needed by nothing, and
    /// without a position in the reordered task.
    const UNPLACED: Self = Self {
        level: NO_LEVEL,
        necessary: false,
    };

    pub fn level(self) -> i32 {
        self.level
    }

    pub fn is_necessary(self) -> bool {
        self.necessary
    }

    fn set_level(&mut self, level: i32) {
        assert_eq!(self.level, NO_LEVEL, "variable placed twice");
        self.level = level;
    }

    fn mark_necessary(&mut self) {
        assert!(!self.necessary, "variable marked necessary twice");
        self.necessary = true;
    }
}

/// What the pass works out about one numeric variable.
#[derive(Debug, Clone, Copy)]
pub struct NumericVarState {
    level: i32,
    necessary: bool,
    ntype: NumType,
}

impl NumericVarState {
    /// `R` and `I` deliberately arrive as [`NumType::Unknown`]: which numeric
    /// variables are regular and which only instrument the metric is decided
    /// here, from the metric and the assignment axioms that feed it, not taken
    /// from the translation.
    fn from_sas_type(index: usize, sas_type: &str) -> Self {
        let ntype = match sas_type {
            "C" => NumType::Constant,
            "D" => NumType::Derived,
            "R" | "I" => NumType::Unknown,
            other => panic!("numeric variable {index} has an unknown type {other:?}"),
        };
        Self {
            level: NO_LEVEL,
            necessary: false,
            ntype,
        }
    }

    pub fn level(self) -> i32 {
        self.level
    }

    pub fn is_necessary(self) -> bool {
        self.necessary
    }

    pub fn ntype(self) -> NumType {
        self.ntype
    }

    fn set_level(&mut self, level: i32) {
        assert_eq!(self.level, NO_LEVEL, "numeric variable placed twice");
        self.level = level;
    }

    fn mark_necessary(&mut self) {
        assert!(!self.necessary, "numeric variable marked necessary twice");
        self.necessary = true;
        if self.ntype == NumType::Unknown {
            self.ntype = NumType::Regular;
        }
    }

    fn mark_instrumentation(&mut self) {
        assert!(!self.necessary, "numeric variable marked necessary twice");
        self.necessary = true;
        if self.ntype == NumType::Unknown {
            self.ntype = NumType::Instrumentation;
        }
    }
}

/// A translated task prepared for the causal-graph analysis.
pub struct PreprocessedTask {
    pub sas: SASTask,
    pub metric: Metric,
    pub vars: Vec<VarState>,
    pub numeric_vars: Vec<NumericVarState>,
}

impl PreprocessedTask {
    pub fn new(mut sas: SASTask) -> Self {
        check_shape(&sas);
        name_comparison_facts(&mut sas);
        sort_axioms_by_layer(&mut sas);

        let vars = vec![VarState::UNPLACED; sas.variables.ranges.len()];
        let numeric_vars = sas
            .numeric_variables
            .types
            .iter()
            .enumerate()
            .map(|(index, sas_type)| NumericVarState::from_sas_type(index, sas_type))
            .collect();
        let metric = Metric::from_sas(&sas.metric);

        Self {
            sas,
            metric,
            vars,
            numeric_vars,
        }
    }
}

/// The pass indexes the task's variables freely, and reads a variable's range
/// from `ranges` and its facts from `value_names`, so the parallel arrays have
/// to agree before any of it runs.
fn check_shape(sas: &SASTask) {
    let variables = &sas.variables;
    let num_vars = variables.ranges.len();
    assert_eq!(variables.axiom_layers.len(), num_vars);
    assert_eq!(variables.value_names.len(), num_vars);
    assert_eq!(sas.init.values.len(), num_vars);
    for (var, ((&range, names), &value)) in variables
        .ranges
        .iter()
        .zip(&variables.value_names)
        .zip(&sas.init.values)
        .enumerate()
    {
        assert_eq!(
            range,
            names.len(),
            "variable {var} has range {range} but {} facts",
            names.len()
        );
        assert!(
            value >= 0,
            "variable {var} has no initial value (got {value})"
        );
    }

    let numeric = &sas.numeric_variables;
    let num_numeric_vars = numeric.variable_names.len();
    assert_eq!(numeric.axiom_layers.len(), num_numeric_vars);
    assert_eq!(numeric.types.len(), num_numeric_vars);
    assert_eq!(sas.init.num_values.len(), num_numeric_vars);

    // Both kinds of numeric axiom combine exactly two terms; the output names
    // the two, and a third would be dropped without a word.
    for axiom in &sas.numeric_axioms {
        assert_eq!(
            axiom.parts.len(),
            2,
            "numeric axiom for var {} combines {} operands, not 2",
            axiom.effect,
            axiom.parts.len()
        );
    }
}

/// Spells out in a comparison variable's facts which comparison it stands for.
/// This is not cosmetic: the variable arrives from the translation named after
/// its own index, and the SAS file is read by hand and by the search's own
/// diagnostics.
fn name_comparison_facts(sas: &mut SASTask) {
    let num_numeric_vars = sas.numeric_variables.variable_names.len();
    for axiom in &sas.comp_axioms {
        let effect = axiom.effect;
        assert_eq!(
            axiom.parts.len(),
            2,
            "comparison axiom for var {effect} compares {} operands, not 2",
            axiom.parts.len()
        );
        let (left, right) = (axiom.parts[0], axiom.parts[1]);
        assert!(effect < sas.variables.value_names.len());
        assert!(left < num_numeric_vars);
        assert!(right < num_numeric_vars);

        let left_name = &sas.numeric_variables.variable_names[left];
        let right_name = &sas.numeric_variables.variable_names[right];
        let facts = &mut sas.variables.value_names[effect];
        facts[0] = format!("{} {left_name}, {right_name}", axiom.comp);
        facts[1] = format!(
            "{} {left_name}, {right_name}",
            inverted_comparator(&axiom.comp)
        );
    }
}

/// Groups the axioms of each kind by the axiom layer of the variable they
/// define, so that the analysis and the output see them stratified. The sort is
/// stable, so axioms sharing a layer keep the order the translation produced
/// them in.
fn sort_axioms_by_layer(sas: &mut SASTask) {
    let SASTask {
        variables,
        numeric_variables,
        axioms,
        comp_axioms,
        numeric_axioms,
        ..
    } = sas;
    axioms.sort_by_key(|axiom| variables.axiom_layers[axiom.effect.0]);
    comp_axioms.sort_by_key(|axiom| variables.axiom_layers[axiom.effect]);
    numeric_axioms.sort_by_key(|axiom| numeric_variables.axiom_layers[axiom.effect]);
}

/// The task after the causal graph has ordered and pruned its variables.
///
/// `prop_order` and `numeric_order` list the surviving variables by their old
/// id, in their new order; every reference inside `task.sas` is still phrased
/// in old ids and is remapped through `task.vars` / `task.numeric_vars` on the
/// way out.
pub struct ReorderedTask {
    pub task: PreprocessedTask,
    pub prop_order: Vec<usize>,
    pub numeric_order: Vec<usize>,
}

impl ReorderedTask {
    /// The new id of a propositional variable reference.
    ///
    /// A reference to a variable the reordering pruned would silently change
    /// the task, so it is an error rather than something to leave out.
    pub fn prop_level(&self, var: usize) -> usize {
        placed(self.task.vars[var].level(), "variable")
    }

    /// The new id of a numeric variable reference.
    pub fn numeric_level(&self, var: usize) -> usize {
        placed(self.task.numeric_vars[var].level(), "numeric variable")
    }

    /// One fact of the reordered task.
    pub fn prop_fact(&self, &(var, value): &SasFact) -> SasFact {
        (self.prop_level(var), value)
    }

    /// What the file spells for numeric variable `var`: its type, its axiom
    /// layer as an `i32`, and its name.
    pub fn numeric_variable(&self, var: usize) -> (NumericType, i32, &str) {
        let sas = &self.task.sas;
        let state = self.task.numeric_vars[var];
        assert!(state.is_necessary());
        let layer = sas.numeric_variables.axiom_layers[var];
        assert!(layer >= NO_LAYER);
        (
            state.ntype().as_numeric_type(),
            layer,
            &sas.numeric_variables.variable_names[var],
        )
    }

    /// The goal in the new variable order.
    ///
    /// A goal variable is necessary by definition, so the reordering always gave
    /// it a level; a goal that lost its variable would silently weaken the task.
    pub fn ordered_goal(&self) -> Vec<SasFact> {
        let SASGoal { pairs } = &self.task.sas.goal;
        let mut goal: Vec<SasFact> = pairs.iter().map(|pair| self.prop_fact(pair)).collect();
        goal.sort_unstable();
        // Two goals on one variable disagree about it unless they are the same
        // goal twice, and both readings of the file take the goal to name each
        // of its variables once.
        assert!(
            goal.windows(2).all(|pair| pair[0].0 != pair[1].0),
            "the goal names one variable twice: {goal:?}"
        );
        goal
    }

    /// The number of facts, variables, goals and effects the search will hold,
    /// which is what "how big is this task" means in the logs.
    fn encoding_size(&self) -> usize {
        let PreprocessedTask { sas, .. } = &self.task;
        let facts: usize = self
            .prop_order
            .iter()
            .map(|&var| sas.variables.ranges[var])
            .sum();
        let mut size =
            self.prop_order.len() + self.numeric_order.len() + facts + sas.goal.pairs.len();
        size += sas
            .mutexes
            .iter()
            .map(|mutex| mutex.facts.len())
            .sum::<usize>();
        size += sas
            .operators
            .iter()
            .map(|op| op.get_encoding_size())
            .sum::<usize>();
        size += sas
            .axioms
            .iter()
            .map(|axiom| axiom.get_encoding_size())
            .sum::<usize>();
        // A comparison and an assignment axiom each hold their effect and the
        // pair of operands, of which the operands are shared with the terms
        // they were built from, so each counts as two.
        size += 2 * (sas.comp_axioms.len() + sas.numeric_axioms.len());
        size
    }

    fn derived_variable_count(&self) -> usize {
        self.prop_order
            .iter()
            .filter(|&&var| self.task.sas.variables.axiom_layers[var] != NO_LAYER)
            .count()
    }
}

/// The level a variable reference maps to.
fn placed(level: i32, what: &str) -> usize {
    assert_ne!(level, NO_LEVEL, "{what} was pruned");
    usize::try_from(level).expect("a placed variable's level is its position")
}

/// Orders and prunes the variables of `sas` by its causal graph, and writes the
/// result as the SAS+ file another planner reads.
pub fn write_reordered_sas<W: Write>(sas: SASTask, outfile: &mut W) -> std::io::Result<()> {
    info!("Writing output...");
    planforge_sas::sas_writer::write_sas(&sas_parts::build(&reorder(sas)), outfile)
}

/// Orders and prunes the variables of `sas` by its causal graph, and builds the
/// task the search reads straight from the result.
///
/// The default path from PDDL to a search task. It shares everything but the
/// last step with [`write_reordered_sas`]: both phrase the reordered task as
/// [`planforge_sas::sas_format::SasTaskParts`], so writing the SAS+ file and
/// reading it back produces the same task by construction —
/// `tests/src/task_equivalence_tests` holds both paths to that. The file itself
/// is for interoperating with other planners and for reading by hand, not for
/// getting a task from one part of this process to another.
pub fn reordered_numeric_task(sas: SASTask) -> planforge_sas::numeric_task::NumericRootTask {
    planforge_sas::numeric_task::NumericRootTask::from_sas_parts(sas_parts::build(&reorder(sas)))
}

/// Orders and prunes the variables of `sas` by its causal graph.
fn reorder(sas: SASTask) -> ReorderedTask {
    let task = PreprocessedTask::new(sas);
    let metric_index_before = task.metric.index;

    info!("Building causal graph...");
    let reordered = CausalGraph::new(task).finalize();

    debug!(
        "Metric index changed from {} to {}",
        metric_index_before, reordered.task.metric.index
    );
    info!(
        "Preprocessor facts: {}",
        reordered
            .prop_order
            .iter()
            .map(|&var| reordered.task.sas.variables.ranges[var])
            .sum::<usize>()
    );
    info!(
        "Preprocessor derived variables: {}",
        reordered.derived_variable_count()
    );
    info!("Preprocessor task size: {}", reordered.encoding_size());
    reordered
}
