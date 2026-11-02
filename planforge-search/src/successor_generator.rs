//! Successor generation: a [`SuccessorGenerator`] lists the operators
//! applicable in a state, and the [`StateRegistry`] applies them and
//! registers the successors. Generators are features the configuration
//! language names ([`Features::successor_generators`](crate::registry::Features));
//! the [`StandardGroundedTree`] is built in and the default.
//!
//! A generator only reads the state, so it takes `&self` and writes into a
//! buffer its caller owns:
//!
//! ```
//! use planforge_sas::numeric_task::{ExplicitValueIndex, OperatorIndex, Task};
//! use planforge_search::successor_generator::{StandardGroundedTree, SuccessorGenerator};
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
//! StandardGroundedTree::new(&task).applicable_operators(initial, &mut tree);
//! every.sort();
//! tree.sort();
//! assert_eq!(every, tree);
//! ```

#[cfg(test)]
mod tests;

mod standard_grounded_tree;

pub use standard_grounded_tree::StandardGroundedTree;

use std::fmt::Debug;
use std::sync::Arc;

use planforge_sas::numeric_task::{
    ExplicitValueIndex, NumericValue, Operator, OperatorIndex, Task,
};
use planforge_sas::state_registry::{ConcreteState, ExpansionContext, StateRegistry};
use planforge_sas::utils::errors::StateInsertError;

use crate::config::{
    ApplyOptions, ConfigCall, ConfigError, ConfigValue, EmptyConfig, FromOptionValue,
    canonical_call,
};
use crate::registry::{Configured, Entry, Features, Typed, options_of, parse_configured};

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
pub struct StateExpander<G: SuccessorGenerator> {
    generator: G,
    propositional: Vec<ExplicitValueIndex>,
    applicable: Vec<OperatorIndex>,
    context: ExpansionContext,
    successor_numeric: Vec<NumericValue>,
    successor_cost: Vec<NumericValue>,
}

impl<G: SuccessorGenerator> StateExpander<G> {
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

/// The interface of a configured successor generator, whatever its
/// configuration type: what [`SuccessorGeneratorConfig`] dereferences to.
pub trait ConfiguredSuccessorGenerator: Send + Sync {
    fn build(&self, task: &Task) -> Box<dyn SuccessorGenerator>;
}

impl<F: SuccessorGeneratorFeature> ConfiguredSuccessorGenerator for Typed<F, F::Config> {
    fn build(&self, task: &Task) -> Box<dyn SuccessorGenerator> {
        F::build(&self.config, task)
    }
}

/// A configured successor generator; the standard grounded tree by default.
/// It displays as its canonical call.
pub type SuccessorGeneratorConfig = Configured<dyn ConfiguredSuccessorGenerator>;

impl Default for SuccessorGeneratorConfig {
    fn default() -> Self {
        Configured::new(
            canonical_call(StandardGroundedTreeFeature::NAME, &EmptyConfig {}),
            Arc::new(Typed::<StandardGroundedTreeFeature, _>::new(EmptyConfig {})),
        )
    }
}

impl FromOptionValue for SuccessorGeneratorConfig {
    fn from_option_value(value: &ConfigValue, features: &Features) -> Result<Self, ConfigError> {
        features
            .successor_generators
            .parse(&ConfigCall::from_value(value)?, features)
    }

    fn option_text(&self) -> String {
        self.to_string()
    }
}

/// A successor generator's registry entry.
pub type SuccessorGeneratorEntry = Entry<SuccessorGeneratorConfig>;

pub const fn entry<F: SuccessorGeneratorFeature>() -> SuccessorGeneratorEntry {
    Entry::new(F::NAME, parse_feature::<F>, options_of::<F::Config>)
}

fn parse_feature<F: SuccessorGeneratorFeature>(
    spec: &ConfigCall,
    features: &Features,
) -> Result<SuccessorGeneratorConfig, ConfigError> {
    parse_configured(F::NAME, spec, features, |typed: Typed<F, F::Config>| {
        Arc::new(typed) as Arc<dyn ConfiguredSuccessorGenerator>
    })
}

/// `standard_grounded_tree()`: the [`StandardGroundedTree`].
pub struct StandardGroundedTreeFeature;

impl SuccessorGeneratorFeature for StandardGroundedTreeFeature {
    const NAME: &'static str = "standard_grounded_tree";
    type Config = EmptyConfig;
    fn build(_: &EmptyConfig, task: &Task) -> Box<dyn SuccessorGenerator> {
        Box::new(StandardGroundedTree::new(task))
    }
}
