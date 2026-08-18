//! The rules describing how a derived variable takes its *default* value.
//!
//! The translator emits only the rules that *prove* a derived variable. That is
//! all the axiom evaluator needs: it refutes a derived variable by finding it
//! still at its default at the end of its layer, never by firing a rule. A
//! heuristic that turns the axioms into relaxed operators is in a different
//! position — it has to be told how a derived variable becomes false, or a state
//! in which one is false looks like a dead end. So the negation lives here,
//! computed on demand by the consumer that needs it, which is where mainline
//! Fast Downward put it in issue454.
//!
//! Three properties of the result are worth stating up front, because callers
//! rely on all three.
//!
//! *It is an overapproximation.* A body condition on a multi-valued variable
//! negates into a disjunction — "anything but this value" — and picking one
//! disjunct per rule turns a CNF into a set of rules, which is exact, while a
//! cyclic component is refuted unconditionally, which is not. Every rule
//! produced is *implied* by the exact negation, so a relaxation built from them
//! can only be cheaper than the real task: admissibility survives, accuracy is
//! what is traded away.
//!
//! *It is deliberately incomplete.* A derived variable whose default value
//! nothing observes gets no rules at all, because the rules for it would only
//! feed rules that are themselves unobserved. `relevant_default_values` is
//! that analysis, and it is the reason a task whose only axiom is the
//! translator's global-constraint atom comes out of here empty-handed.
//!
//! *It is deterministic.* Variables are visited in id order and rule bodies are
//! collected in sorted sets, so the same task yields the same rules in the same
//! order on every run — which matters because a heuristic's tie-breaking, and
//! therefore its expansion count, depends on the order of its relaxed
//! operators.

use std::collections::{BTreeSet, VecDeque};

use crate::axioms::PropositionalAxiom;
use crate::numeric_task::{AbstractNumericTask, ExplicitFact};
use crate::utils::scc::Scc;

/// How exactly to describe a derived variable's default value.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum DefaultValueAxiomMode {
    /// Negate exactly, except for a derived variable that depends on itself
    /// through a cycle of nondefault literals: negating such a definition
    /// literal by literal produces bodies that depend on each other in a cycle
    /// and so claim the variable can never be refuted, which is wrong rather
    /// than merely imprecise (mainline's issue453). Those get the trivial
    /// `v=default <- {}` instead.
    #[default]
    ApproximateNegativeCycles,
    /// `v=default <- {}` for every derived variable whose default value is
    /// observed. Weaker, and never exponential.
    ApproximateNegative,
}

/// Which propositional variables are derived, and what they default to.
///
/// A *condition* variable carries an axiom layer too, but it is computed by the
/// comparison pass rather than proven by a Horn rule, so it is not derived in
/// the sense that matters here: nothing has to explain how it becomes false.
struct DerivedVariables {
    is_derived: Vec<bool>,
    default_value: Vec<usize>,
}

impl DerivedVariables {
    fn of(task: &dyn AbstractNumericTask) -> Self {
        let conditions = task.numeric_conditions();
        let num_variables = task.get_num_variables();
        let mut is_derived = Vec::with_capacity(num_variables);
        let mut default_value = Vec::with_capacity(num_variables);
        for var in 0..num_variables {
            let layer = task
                .get_variable_axiom_layer(var)
                .expect("variable id below the variable count is in bounds");
            let derived = layer.is_some() && !conditions.is_condition_var(var);
            if derived {
                // The negation is stated as one rule per default value, and
                // `other_value` needs the complement to be unique, so a derived
                // variable that is not binary would have no single answer.
                assert_eq!(
                    task.get_variable_domain_size(var),
                    Ok(2),
                    "derived variable {var} is not binary, so it has no single default-value rule"
                );
            }
            is_derived.push(derived);
            default_value.push(
                task.get_variable_default_axiom_value(var)
                    .expect("variable id below the variable count is in bounds"),
            );
        }
        DerivedVariables {
            is_derived,
            default_value,
        }
    }

    /// Whether `fact` reads a derived variable at its default value, or `None`
    /// when the variable is not derived.
    fn reads_default(&self, fact: &ExplicitFact) -> Option<bool> {
        self.is_derived[fact.var()].then(|| fact.value() == self.default_value[fact.var()])
    }
}

/// The value a binary derived variable holds when it is not at `value`.
fn other_value(value: usize) -> usize {
    1 - value
}

/// Which derived variables the body of a proving rule reads, split by whether it
/// reads them at their default value, plus which rules prove each variable.
///
/// The vectors are indexed by variable id over *all* propositional variables;
/// only the derived entries are ever non-empty.
struct Dependencies {
    nondefault: Vec<Vec<usize>>,
    default: Vec<Vec<usize>>,
    proving_axioms: Vec<Vec<usize>>,
}

impl Dependencies {
    fn of(task: &dyn AbstractNumericTask, derived: &DerivedVariables) -> Self {
        let num_variables = task.get_num_variables();
        let mut dependencies = Dependencies {
            nondefault: vec![Vec::new(); num_variables],
            default: vec![Vec::new(); num_variables],
            proving_axioms: vec![Vec::new(); num_variables],
        };
        for (axiom_id, axiom) in task.axioms().iter().enumerate() {
            let head = axiom.var_id();
            assert!(
                derived.is_derived[head],
                "axiom {axiom_id} writes variable {head}, which is not a derived variable"
            );
            assert_ne!(
                axiom.effect_value(),
                derived.default_value[head],
                "axiom {axiom_id} sets derived variable {head} to its default value; the rules \
                 that do that are computed here and must not already be in the task"
            );
            dependencies.proving_axioms[head].push(axiom_id);
            for condition in axiom.conditions() {
                match derived.reads_default(condition) {
                    Some(true) => dependencies.default[head].push(condition.var()),
                    Some(false) => dependencies.nondefault[head].push(condition.var()),
                    None => (),
                }
            }
        }
        for edges in dependencies
            .nondefault
            .iter_mut()
            .chain(&mut dependencies.default)
        {
            edges.sort_unstable();
            edges.dedup();
        }
        dependencies
    }
}

/// The rules describing the default value of every derived variable that needs
/// one.
///
/// Empty when the task has no axioms, and — because of the relevance analysis —
/// also when no consumer can observe a derived variable at its default value.
pub fn default_value_axioms(
    task: &dyn AbstractNumericTask,
    mode: DefaultValueAxiomMode,
) -> Vec<PropositionalAxiom> {
    if task.axioms().is_empty() {
        return Vec::new();
    }

    let derived = DerivedVariables::of(task);
    let dependencies = Dependencies::of(task, &derived);
    let refuted_unconditionally = refuted_unconditionally(&dependencies, &derived, mode);
    let needed = relevant_default_values(task, &derived, &dependencies, &refuted_unconditionally);

    let mut axioms = Vec::new();
    for var in needed {
        let default_value = derived.default_value[var];
        let precondition_value = other_value(default_value);
        if refuted_unconditionally[var] {
            axioms.push(PropositionalAxiom::new(
                Vec::new(),
                var,
                precondition_value,
                default_value,
            ));
            continue;
        }
        for body in refuting_bodies(task, &dependencies.proving_axioms[var]) {
            axioms.push(PropositionalAxiom::new(
                body.into_iter().collect(),
                var,
                precondition_value,
                default_value,
            ));
        }
    }
    axioms
}

/// Per variable, whether its default value is described by the empty body.
///
/// That is the whole of [`DefaultValueAxiomMode::ApproximateNegative`], and for
/// [`DefaultValueAxiomMode::ApproximateNegativeCycles`] it is the derived
/// variables lying on a cycle of nondefault dependencies.
///
/// Default dependencies are left out of the cycle search on purpose: a cycle
/// through one would mean the axioms are not stratifiable, which the translator
/// has already rejected.
fn refuted_unconditionally(
    dependencies: &Dependencies,
    derived: &DerivedVariables,
    mode: DefaultValueAxiomMode,
) -> Vec<bool> {
    match mode {
        DefaultValueAxiomMode::ApproximateNegative => derived.is_derived.clone(),
        DefaultValueAxiomMode::ApproximateNegativeCycles => {
            let mut cyclic = vec![false; dependencies.nondefault.len()];
            for component in Scc::new(dependencies.nondefault.clone()).get_result() {
                if component.len() > 1 {
                    for var in component {
                        cyclic[var] = true;
                    }
                }
            }
            cyclic
        }
    }
}

/// The derived variables whose default value some consumer can observe, in id
/// order.
///
/// A derived literal is observed directly when it appears in a goal, an operator
/// precondition or an operator effect condition. It is observed indirectly
/// through the rules of a variable that is itself observed: proving a head needs
/// its nondefault body literals provable and its default body literals
/// refutable, and refuting a head needs the opposite. Iterating that gives the
/// closure, and the default-value half of it is what has to be built.
///
/// The translator's global-constraint atom is deliberately not a seed. It is not
/// a goal and not an operator precondition — the search engines never consult it
/// (see `NumericRootTask::global_constraint`), only the plan verifier does, and
/// the verifier checks it against a real state rather than against a relaxation.
/// A task whose only axiom is that atom therefore needs no rules at all, which
/// is what keeps a heuristic on such a task byte-for-byte what it was.
fn relevant_default_values(
    task: &dyn AbstractNumericTask,
    derived: &DerivedVariables,
    dependencies: &Dependencies,
    refuted_unconditionally: &[bool],
) -> Vec<usize> {
    // `(variable, wants the default value)`, the pairs whose achievability some
    // consumer asks about.
    let mut needed: BTreeSet<(usize, bool)> = BTreeSet::new();
    let mut queue: VecDeque<(usize, bool)> = VecDeque::new();
    let require = |fact: &ExplicitFact,
                   needed: &mut BTreeSet<(usize, bool)>,
                   queue: &mut VecDeque<(usize, bool)>| {
        if let Some(wants_default) = derived.reads_default(fact) {
            let entry = (fact.var(), wants_default);
            if needed.insert(entry) {
                queue.push_back(entry);
            }
        }
    };

    for goal_index in 0..task.get_num_goals() {
        require(task.get_goal_fact(goal_index), &mut needed, &mut queue);
    }
    for operator in task.get_operators() {
        for precondition in operator.preconditions() {
            require(precondition, &mut needed, &mut queue);
        }
        for effect in operator.effects() {
            for condition in effect.conditions() {
                require(condition, &mut needed, &mut queue);
            }
        }
        for effect in operator.assignment_effects() {
            for condition in effect.conditions() {
                require(condition, &mut needed, &mut queue);
            }
        }
    }

    while let Some((var, wants_default)) = queue.pop_front() {
        // A default value described by the empty body depends on nothing, so
        // asking for it asks for nothing else.
        if wants_default && refuted_unconditionally[var] {
            continue;
        }
        for &dependency in &dependencies.nondefault[var] {
            let entry = (dependency, wants_default);
            if needed.insert(entry) {
                queue.push_back(entry);
            }
        }
        for &dependency in &dependencies.default[var] {
            let entry = (dependency, !wants_default);
            if needed.insert(entry) {
                queue.push_back(entry);
            }
        }
    }

    needed
        .into_iter()
        .filter_map(|(var, wants_default)| wants_default.then_some(var))
        .collect()
}

/// The bodies of the rules refuting one derived variable.
///
/// The variable is at its default exactly when *every* rule proving it fails,
/// and a rule fails when one of its body conditions reads a variable that holds
/// some other value. That is a CNF over facts, one clause per proving rule, and
/// its models are the hitting sets of the clauses. Only the non-dominated ones
/// are kept: a hitting set with a fact that no clause needs it alone for has a
/// strict subset that hits everything, and the subset is the weaker — hence
/// better — rule body.
///
/// The two degenerate cases fall out of that rather than being special-cased. An
/// unconditional proving rule contributes an *empty* clause, which nothing can
/// hit, so there are no bodies at all — nothing refutes a variable that always
/// holds. A variable with no proving rule contributes no clause, so the empty
/// hitting set survives and its default holds in every state.
fn refuting_bodies(
    task: &dyn AbstractNumericTask,
    proving_axioms: &[usize],
) -> BTreeSet<BTreeSet<ExplicitFact>> {
    let conditions = task.numeric_conditions();
    let clauses: Vec<BTreeSet<ExplicitFact>> = proving_axioms
        .iter()
        .map(|&axiom_id| {
            let mut clause = BTreeSet::new();
            for condition in task.axioms()[axiom_id].conditions() {
                let var = condition.var();
                let domain_size = task
                    .get_variable_domain_size(var)
                    .expect("an axiom condition names a variable of the task");
                for value in 0..domain_size {
                    if value != condition.value() {
                        clause.insert(conditions.fact(var, value));
                    }
                }
            }
            clause
        })
        .collect();

    let mut bodies = BTreeSet::new();
    let mut chosen = BTreeSet::new();
    let mut chosen_variables = BTreeSet::new();
    collect_non_dominated_hitting_sets(
        &clauses,
        0,
        &mut chosen,
        &mut chosen_variables,
        &mut bodies,
    );
    bodies
}

/// Depth-first enumeration of the non-dominated hitting sets of `clauses`.
///
/// Two facts of the same variable are never chosen together: a body asking a
/// variable to hold two values at once can never fire, so such a rule would be
/// dead weight rather than a weaker rule.
fn collect_non_dominated_hitting_sets(
    clauses: &[BTreeSet<ExplicitFact>],
    index: usize,
    chosen: &mut BTreeSet<ExplicitFact>,
    chosen_variables: &mut BTreeSet<usize>,
    results: &mut BTreeSet<BTreeSet<ExplicitFact>>,
) {
    let Some(clause) = clauses.get(index) else {
        // Every fact of the hitting set has to be the *only* reason some clause
        // is hit; one that is not can be dropped, and what is left still hits
        // every clause and dominates this set.
        let mut not_uniquely_used = chosen.clone();
        for clause in clauses {
            let mut hit = clause.intersection(chosen);
            if let (Some(only), None) = (hit.next(), hit.next()) {
                not_uniquely_used.remove(only);
            }
        }
        if not_uniquely_used.is_empty() {
            results.insert(chosen.clone());
        }
        return;
    };

    // Already hit, so this clause constrains nothing.
    if !clause.is_disjoint(chosen) {
        collect_non_dominated_hitting_sets(clauses, index + 1, chosen, chosen_variables, results);
        return;
    }

    for &fact in clause {
        if !chosen_variables.insert(fact.var()) {
            continue;
        }
        chosen.insert(fact);
        collect_non_dominated_hitting_sets(clauses, index + 1, chosen, chosen_variables, results);
        chosen.remove(&fact);
        chosen_variables.remove(&fact.var());
    }
}

#[cfg(test)]
mod tests;
