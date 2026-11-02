//! Successor generation: a [`SuccessorGenerator`] lists the operators
//! applicable in a state, and the [`StateRegistry`] applies them and
//! registers the successors. Generators are features the configuration
//! grammar names ([`SUCCESSOR_GENERATORS`]); the [`DecisionTree`] is built in
//! and the default.
//!
//! A generator only reads the state, so it takes `&self` and writes into a
//! buffer its caller owns:
//!
//! ```
//! use planforge_sas::numeric_task::{ExplicitValueIndex, OperatorIndex, Task};
//! use planforge_search::successor_generator::{DecisionTree, SuccessorGenerator};
//!
//! /// Tests the preconditions of every operator: slow, but plainly correct.
//! struct EveryOperator {
//!     preconditions: Vec<Vec<(usize, ExplicitValueIndex)>>,
//! }
//!
//! impl EveryOperator {
//!     fn new(task: &Task) -> Self {
//!         let preconditions = task.operators().iter().map(|operator| {
//!             let facts = operator.preconditions().iter();
//!             facts.map(|fact| (fact.var(), fact.value_index())).collect()
//!         });
//!         Self { preconditions: preconditions.collect() }
//!     }
//! }
//!
//! impl SuccessorGenerator for EveryOperator {
//!     fn applicable_operators(
//!         &self,
//!         state: &[ExplicitValueIndex],
//!         applicable: &mut Vec<OperatorIndex>,
//!     ) {
//!         for (operator, preconditions) in self.preconditions.iter().enumerate() {
//!             if preconditions.iter().all(|&(var, value)| state[var] == value) {
//!                 applicable.push(OperatorIndex::from_usize(operator));
//!             }
//!         }
//!     }
//! }
//!
//! let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/assets/numeric_sas/example5.sas");
//! let task = Task::from_file(path);
//! let initial = task.initial_propositional_values();
//! let (mut every, mut tree) = (Vec::new(), Vec::new());
//! EveryOperator::new(&task).applicable_operators(initial, &mut every);
//! DecisionTree::new(&task).applicable_operators(initial, &mut tree);
//! every.sort();
//! tree.sort();
//! assert_eq!(every, tree);
//! ```

#[cfg(test)]
mod tests;

mod decision_tree;

pub use decision_tree::DecisionTree;

use std::fmt::{self, Debug};
use std::sync::Arc;

use planforge_sas::numeric_task::{
    ExplicitValueIndex, NumericValue, Operator, OperatorIndex, Task,
};
use planforge_sas::state_registry::{ConcreteState, ExpansionContext, StateRegistry};
use planforge_sas::utils::errors::StateInsertError;

use crate::config::{
    ApplyOptions, ConfigCall, ConfigError, ConfigValue, FromOptionValue, canonical_call, configure,
};
use crate::heuristic_factory::EmptyConfig;
use crate::registry::{Entry, Registry, options_of};

/// Which operators are applicable in a state.
pub trait SuccessorGenerator {
    /// Appends the operators applicable in the state whose propositional
    /// values are `state` to `applicable`. Numeric preconditions are
    /// propositional too: the comparison axioms derive them.
    fn applicable_operators(
        &self,
        state: &[ExplicitValueIndex],
        applicable: &mut Vec<OperatorIndex>,
    );
}

impl<G: SuccessorGenerator + ?Sized> SuccessorGenerator for Box<G> {
    fn applicable_operators(
        &self,
        state: &[ExplicitValueIndex],
        applicable: &mut Vec<OperatorIndex>,
    ) {
        (**self).applicable_operators(state, applicable);
    }
}

/// Forward expansion of registered states: a successor generator and the
/// scratch an expansion needs, reused from one expansion to the next.
pub struct RegistryExpander<G: SuccessorGenerator> {
    generator: G,
    propositional: Vec<ExplicitValueIndex>,
    applicable: Vec<OperatorIndex>,
    context: ExpansionContext,
    successor_numeric: Vec<NumericValue>,
    successor_cost: Vec<NumericValue>,
}

impl<G: SuccessorGenerator> RegistryExpander<G> {
    pub fn new(generator: G) -> Self {
        Self {
            generator,
            propositional: Vec::new(),
            applicable: Vec::new(),
            context: ExpansionContext::default(),
            successor_numeric: Vec::new(),
            successor_cost: Vec::new(),
        }
    }

    /// Decodes `state`, a state of `registry`, and lists the operators
    /// applicable in it, in the successor generator's order.
    pub fn expand<'e>(
        &'e mut self,
        registry: &StateRegistry,
        state: &ConcreteState,
    ) -> Result<Expansion<'e>, StateInsertError> {
        registry
            .view(state)
            .fill_propositional(&mut self.propositional);
        self.applicable.clear();
        self.generator
            .applicable_operators(&self.propositional, &mut self.applicable);
        registry.build_expansion_context(state, &mut self.context)?;
        Ok(Expansion {
            state: state.clone(),
            propositional: &self.propositional,
            applicable: &self.applicable,
            context: &self.context,
            successor_numeric: &mut self.successor_numeric,
            successor_cost: &mut self.successor_cost,
        })
    }
}

/// An expanded state: its values, its applicable operators and its
/// successors.
pub struct Expansion<'e> {
    state: ConcreteState,
    propositional: &'e [ExplicitValueIndex],
    applicable: &'e [OperatorIndex],
    context: &'e ExpansionContext,
    successor_numeric: &'e mut Vec<NumericValue>,
    successor_cost: &'e mut Vec<NumericValue>,
}

impl<'e> Expansion<'e> {
    pub fn propositional(&self) -> &'e [ExplicitValueIndex] {
        self.propositional
    }

    /// The numeric values, derived ones included.
    pub fn numeric(&self) -> &'e [NumericValue] {
        &self.context.parent_numeric
    }

    pub fn applicable_operators(&self) -> &'e [OperatorIndex] {
        self.applicable
    }

    /// The successor under `operator`, registered in the registry the state
    /// was expanded from, and the operator's metric cost there.
    pub fn successor(
        &mut self,
        registry: &mut StateRegistry,
        operator: &Operator,
    ) -> Result<(ConcreteState, f64), StateInsertError> {
        registry.apply_operator_in_context(
            &self.state,
            operator,
            self.context,
            self.successor_numeric,
            self.successor_cost,
        )
    }
}

/// A successor generator the configuration grammar names: its typed
/// configuration and how to build the generator for a task.
pub trait SuccessorGeneratorFeature: 'static {
    const NAME: &'static str;
    type Config: ApplyOptions + Default + Debug + Send + Sync + 'static;
    fn build(config: &Self::Config, task: &Task) -> Box<dyn SuccessorGenerator>;
}

trait ErasedGeneratorConfig: Debug + Send + Sync {
    fn call(&self) -> String;
    fn build(&self, task: &Task) -> Box<dyn SuccessorGenerator>;
}

struct Configured<F: SuccessorGeneratorFeature>(F::Config);

impl<F: SuccessorGeneratorFeature> Debug for Configured<F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple(F::NAME).field(&self.0).finish()
    }
}

impl<F: SuccessorGeneratorFeature> ErasedGeneratorConfig for Configured<F> {
    fn call(&self) -> String {
        canonical_call(F::NAME, &self.0)
    }
    fn build(&self, task: &Task) -> Box<dyn SuccessorGenerator> {
        F::build(&self.0, task)
    }
}

/// A configured successor generator; the decision tree by default.
#[derive(Debug, Clone)]
pub struct SuccessorGeneratorConfig(Arc<dyn ErasedGeneratorConfig>);

impl SuccessorGeneratorConfig {
    pub fn build(&self, task: &Task) -> Box<dyn SuccessorGenerator> {
        self.0.build(task)
    }
}

impl Default for SuccessorGeneratorConfig {
    fn default() -> Self {
        Self(Arc::new(Configured::<DecisionTreeFeature>(EmptyConfig {})))
    }
}

/// The canonical call, which parses back to an equal configuration.
impl fmt::Display for SuccessorGeneratorConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0.call())
    }
}

impl PartialEq for SuccessorGeneratorConfig {
    fn eq(&self, other: &Self) -> bool {
        self.0.call() == other.0.call()
    }
}

impl FromOptionValue for SuccessorGeneratorConfig {
    fn from_option_value(value: &ConfigValue) -> Result<Self, ConfigError> {
        SUCCESSOR_GENERATORS.parse(&ConfigCall::from_value(value).map_err(ConfigError::from)?)
    }

    fn option_text(&self) -> String {
        self.to_string()
    }
}

/// A registered successor generator.
pub type SuccessorGeneratorEntry = Entry<SuccessorGeneratorConfig>;

pub const fn entry<F: SuccessorGeneratorFeature>() -> SuccessorGeneratorEntry {
    Entry::new(F::NAME, parse_feature::<F>, options_of::<F::Config>)
}

fn parse_feature<F: SuccessorGeneratorFeature>(
    spec: &ConfigCall,
) -> Result<SuccessorGeneratorConfig, ConfigError> {
    let config = configure(F::NAME, F::Config::default(), &spec.args)?;
    Ok(SuccessorGeneratorConfig(Arc::new(Configured::<F>(config))))
}

/// `decision_tree()`: the [`DecisionTree`].
pub struct DecisionTreeFeature;

impl SuccessorGeneratorFeature for DecisionTreeFeature {
    const NAME: &'static str = "decision_tree";
    type Config = EmptyConfig;
    fn build(_: &EmptyConfig, task: &Task) -> Box<dyn SuccessorGenerator> {
        Box::new(DecisionTree::new(task))
    }
}

/// Every successor generator the `successor_generator` option of a search
/// names: the decision tree is built in, a binary registers others.
pub static SUCCESSOR_GENERATORS: Registry<SuccessorGeneratorConfig> =
    Registry::new("successor generator", &[entry::<DecisionTreeFeature>()]);
