//! Axiom simplification, layering and negation, following mainline Fast
//! Downward's issue453.
//!
//! Layers exist for one reason: an axiom that reads a derived variable at the
//! value it *defaults* to is asking "was this never proven?", and that question
//! only has an answer once the layer below has reached its fixpoint. Reading a
//! derived variable at a value its own axioms prove needs no layer of its own,
//! because one layer's fixpoint already propagates it.
//!
//! So the layering is computed from the *negative* edges of the dependency
//! graph, over strongly connected components of that graph rather than over
//! single variables. A component with more than one variable has a cyclic
//! positive dependency, and its variables have to share a layer; a negative edge
//! inside such a component means the axioms are not stratifiable at all.
//!
//! Every derived variable defaults to false. The previous scheme let a variable
//! that was only ever read negatively default to *true* and refuted it with
//! generated axioms, and charged a layer whenever a literal's sign matched its
//! default. That is the same idea expressed per variable instead of per
//! component, and it cannot express a cyclic component: negating a cyclic
//! definition literal by literal yields rules whose bodies depend on each other
//! in a cycle, which claim a derived variable can never be refuted. See
//! issue453.

use std::collections::{BTreeSet, HashMap, HashSet};

use super::options::LayerStrategy;
use super::pddl::actions::PropositionalAction;
use super::pddl::axioms::PropositionalAxiom;
use super::pddl::conditions::*;

/// Which derived variables an axiom head depends on, and with which sign.
///
/// Built from the axioms *before* they are negated, because the negated rules
/// are a consequence of this graph rather than an input to it.
struct AxiomDependencies {
    derived_variables: HashSet<Atom>,
    positive: HashMap<Atom, HashSet<Atom>>,
    negative: HashMap<Atom, HashSet<Atom>>,
}

impl AxiomDependencies {
    fn new(axioms: &[PropositionalAxiom]) -> Self {
        let derived_variables: HashSet<Atom> = axioms.iter().map(head_of).collect();
        let mut dependencies = AxiomDependencies {
            derived_variables,
            positive: HashMap::new(),
            negative: HashMap::new(),
        };
        for axiom in axioms {
            let head = head_of(axiom);
            for body_literal in &axiom.condition {
                let Some(body_atom) = body_literal.literal_positive() else {
                    continue;
                };
                if !dependencies.derived_variables.contains(&body_atom) {
                    continue;
                }
                let edges = if body_literal.is_negated() {
                    &mut dependencies.negative
                } else {
                    &mut dependencies.positive
                };
                edges.entry(head.clone()).or_default().insert(body_atom);
            }
        }
        dependencies
    }

    /// Drops every variable neither of whose literals is needed.
    ///
    /// Whole entries can go: if a head is needed then so is everything in the
    /// bodies of its axioms, by the definition of `necessary_literals`.
    fn remove_unnecessary_variables(&mut self, necessary_literals: &HashSet<Condition>) {
        let unnecessary: Vec<Atom> = self
            .derived_variables
            .iter()
            .filter(|variable| {
                !necessary_literals.contains(&Condition::Atom((*variable).clone()))
                    && !necessary_literals.contains(&Condition::NegatedAtom(variable.negate()))
            })
            .cloned()
            .collect();
        for variable in unnecessary {
            self.derived_variables.remove(&variable);
            self.positive.remove(&variable);
            self.negative.remove(&variable);
        }
    }

    fn positive_edges(&self, head: &Atom) -> impl Iterator<Item = &Atom> {
        self.positive.get(head).into_iter().flatten()
    }

    fn negative_edges(&self, head: &Atom) -> impl Iterator<Item = &Atom> {
        self.negative.get(head).into_iter().flatten()
    }

    /// The derived variables in the order the SCC computation and every
    /// downstream iteration use, so the result does not depend on hash order.
    fn sorted_variables(&self) -> Vec<Atom> {
        let mut sorted: Vec<Atom> = self.derived_variables.iter().cloned().collect();
        sorted.sort_by(cmp_atoms);
        sorted
    }
}

/// One strongly connected component of the dependency graph, with the axioms of
/// its variables.
///
/// The variables of a cluster share an axiom layer, so a cluster is the unit
/// everything below works on: layers are assigned to clusters, and a cluster is
/// negated as a whole.
struct AxiomCluster {
    variables: Vec<Atom>,
    axioms: HashMap<Atom, Vec<PropositionalAxiom>>,
    /// Clusters holding a variable that occurs positively in the body of one of
    /// this cluster's axioms, and negatively respectively. Indices into the
    /// cluster list; a cluster is never its own positive child.
    positive_children: BTreeSet<usize>,
    negative_children: BTreeSet<usize>,
    /// Whether some literal outside the cluster needs one of its variables to be
    /// *false*, which is what makes the negated axioms worth generating.
    needed_negatively: bool,
    layer: i32,
}

impl AxiomCluster {
    fn new(variables: Vec<Atom>) -> Self {
        AxiomCluster {
            axioms: variables
                .iter()
                .map(|variable| (variable.clone(), Vec::new()))
                .collect(),
            variables,
            positive_children: BTreeSet::new(),
            negative_children: BTreeSet::new(),
            needed_negatively: false,
            layer: 0,
        }
    }

    fn axioms_of(&self, variable: &Atom) -> &Vec<PropositionalAxiom> {
        self.axioms
            .get(variable)
            .unwrap_or_else(|| panic!("{variable} is not a variable of its own cluster"))
    }
}

/// Returns the processed axioms and the layer of every derived variable.
///
/// A derived variable that survives has a layer, and the layer is at least 0;
/// non-derived variables get their `-1` elsewhere.
pub fn handle_axioms(
    operators: &[PropositionalAction],
    axioms: Vec<PropositionalAxiom>,
    goal_list: &[Condition],
    global_constraint: &Condition,
    layer_strategy: LayerStrategy,
) -> Result<(Vec<PropositionalAxiom>, HashMap<Atom, i32>), String> {
    if axioms.is_empty() {
        return Ok((vec![], HashMap::new()));
    }

    let mut clusters = compute_clusters(&axioms, operators, goal_list, global_constraint)?;
    let axiom_layers = compute_axiom_layers(&mut clusters, layer_strategy);
    compute_negative_axioms(&mut clusters);
    let processed_axioms = collect_axioms(&clusters);
    verify_layering_condition(&processed_axioms, &axiom_layers);
    Ok((processed_axioms, axiom_layers))
}

fn head_of(axiom: &PropositionalAxiom) -> Atom {
    axiom
        .effect
        .as_atom()
        .unwrap_or_else(|| panic!("an axiom head is a positive atom, got {}", axiom.effect))
        .clone()
}

/// The literals of derived variables the rest of the task can observe, closed
/// under the axiom bodies.
///
/// Signs are tracked because they decide what has to be *computed*: a variable
/// only ever read positively needs no negated axioms, and one only ever read
/// negatively needs no positive layer above it.
fn compute_necessary_literals(
    dependencies: &AxiomDependencies,
    operators: &[PropositionalAction],
    goal_list: &[Condition],
    global_constraint: &Condition,
) -> HashSet<Condition> {
    let mut necessary: HashSet<Condition> = HashSet::new();
    let mut queue: Vec<Condition> = vec![];

    let require =
        |literal: &Condition, necessary: &mut HashSet<Condition>, queue: &mut Vec<Condition>| {
            let Some(atom) = literal.literal_positive() else {
                return;
            };
            if dependencies.derived_variables.contains(&atom) && necessary.insert(literal.clone()) {
                queue.push(literal.clone());
            }
        };

    for literal in goal_list
        .iter()
        .chain(std::iter::once(global_constraint))
        .chain(operators.iter().flat_map(|op| op.precondition.iter()))
    {
        require(literal, &mut necessary, &mut queue);
    }
    // An effect condition is observed in both directions: the effect fires when
    // it holds and does not when it fails, and the code that evaluates it needs
    // an answer either way.
    for operator in operators {
        for (condition, _) in operator.add_effects.iter().chain(&operator.del_effects) {
            for literal in condition {
                require(literal, &mut necessary, &mut queue);
                require(&negate_axiom_literal(literal), &mut necessary, &mut queue);
            }
        }
    }

    while let Some(literal) = queue.pop() {
        let atom = literal
            .literal_positive()
            .expect("only literals of derived variables are queued");
        let negated = literal.is_negated();
        // Proving the head positively needs its positive body literals to be
        // provable; refuting the head needs them refutable, and vice versa for
        // the body literals it reads negatively.
        for body_atom in dependencies.positive_edges(&atom) {
            let required = signed_literal(body_atom, negated);
            require(&required, &mut necessary, &mut queue);
        }
        for body_atom in dependencies.negative_edges(&atom) {
            let required = signed_literal(body_atom, !negated);
            require(&required, &mut necessary, &mut queue);
        }
    }

    necessary
}

fn signed_literal(atom: &Atom, negated: bool) -> Condition {
    if negated {
        Condition::NegatedAtom(atom.negate())
    } else {
        Condition::Atom(atom.clone())
    }
}

/// The strongly connected components of the dependency graph, in an order in
/// which a cluster's children come after it.
///
/// Positive and negative edges are both used: variables in a positive cycle have
/// to share a layer, and a negative edge inside a component is exactly the
/// non-stratifiable case, which is easier to detect once the component is built.
fn strongly_connected_components(
    dependencies: &AxiomDependencies,
    sorted_variables: &[Atom],
) -> Vec<Vec<Atom>> {
    let index_of: HashMap<&Atom, usize> = sorted_variables
        .iter()
        .enumerate()
        .map(|(index, variable)| (variable, index))
        .collect();

    let graph: Vec<Vec<usize>> = sorted_variables
        .iter()
        .map(|variable| {
            let mut successors: Vec<usize> = dependencies
                .positive_edges(variable)
                .chain(dependencies.negative_edges(variable))
                .map(|body_atom| index_of[body_atom])
                .collect();
            successors.sort_unstable();
            successors.dedup();
            successors
        })
        .collect();

    super::preprocess::scc::Scc::new(graph)
        .get_result()
        .into_iter()
        .map(|component| {
            component
                .into_iter()
                .map(|index| sorted_variables[index].clone())
                .collect()
        })
        .collect()
}

fn compute_clusters(
    axioms: &[PropositionalAxiom],
    operators: &[PropositionalAction],
    goal_list: &[Condition],
    global_constraint: &Condition,
) -> Result<Vec<AxiomCluster>, String> {
    let mut dependencies = AxiomDependencies::new(axioms);
    let necessary_literals =
        compute_necessary_literals(&dependencies, operators, goal_list, global_constraint);
    dependencies.remove_unnecessary_variables(&necessary_literals);

    let sorted_variables = dependencies.sorted_variables();
    let components = strongly_connected_components(&dependencies, &sorted_variables);

    let mut clusters: Vec<AxiomCluster> = components.into_iter().map(AxiomCluster::new).collect();
    let mut cluster_of: HashMap<Atom, usize> = HashMap::new();
    for (index, cluster) in clusters.iter_mut().enumerate() {
        for variable in &cluster.variables {
            cluster_of.insert(variable.clone(), index);
            cluster.needed_negatively |=
                necessary_literals.contains(&Condition::NegatedAtom(variable.negate()));
        }
    }

    for axiom in axioms {
        let head = head_of(axiom);
        // The head is derived, but may have been pruned as unnecessary.
        if let Some(&index) = cluster_of.get(&head) {
            clusters[index]
                .axioms
                .get_mut(&head)
                .expect("a cluster holds an entry for each of its variables")
                .push(axiom.clone());
        }
    }

    for cluster in &mut clusters {
        for variable in &cluster.variables {
            let axioms = cluster
                .axioms
                .get_mut(variable)
                .expect("a cluster holds an entry for each of its variables");
            *axioms = simplify(std::mem::take(axioms));
        }
    }

    // `sorted_variables` rather than the dependency maps, so the error below
    // names the same pair on every run.
    for variable in &sorted_variables {
        let Some(&from) = cluster_of.get(variable) else {
            continue;
        };
        for body_atom in dependencies.positive_edges(variable) {
            let to = cluster_of[body_atom];
            if to != from {
                clusters[from].positive_children.insert(to);
            }
        }
        for body_atom in dependencies.negative_edges(variable) {
            let to = cluster_of[body_atom];
            if to == from {
                return Err(format!(
                    "axioms are not stratifiable: {variable} depends on the negation of \
                     {body_atom}, which positively depends on {variable} again"
                ));
            }
            clusters[from].negative_children.insert(to);
        }
    }

    Ok(clusters)
}

/// Lays the clusters out and reports the layer of every variable.
///
/// Both strategies live off the same property of the cluster order: a cluster's
/// children come after it, so the reverse traversal settles them first.
/// [`LayerStrategy::Min`] then reads them; [`LayerStrategy::Max`] does not need
/// to, because counting the layers down along that order already puts every
/// child strictly below its parent.
fn compute_axiom_layers(
    clusters: &mut [AxiomCluster],
    layer_strategy: LayerStrategy,
) -> HashMap<Atom, i32> {
    for index in (0..clusters.len()).rev() {
        clusters[index].layer = match layer_strategy {
            LayerStrategy::Min => lowest_allowed_layer(clusters, index),
            LayerStrategy::Max => i32::try_from(clusters.len() - 1 - index)
                .expect("a task has fewer axiom clusters than an i32 counts"),
        };
    }

    clusters
        .iter()
        .flat_map(|cluster| {
            cluster
                .variables
                .iter()
                .map(move |variable| (variable.clone(), cluster.layer))
        })
        .collect()
}

/// The lowest layer the already-laid-out children of `clusters[index]` leave it.
///
/// A positive child may share the layer, because one layer's fixpoint
/// propagates a proven literal. A negative child must sit strictly below, so
/// that "never proven" has settled before it is read.
fn lowest_allowed_layer(clusters: &[AxiomCluster], index: usize) -> i32 {
    let cluster = &clusters[index];
    cluster
        .positive_children
        .iter()
        .map(|&child| clusters[child].layer)
        .chain(
            cluster
                .negative_children
                .iter()
                .map(|&child| clusters[child].layer + 1),
        )
        .max()
        .unwrap_or(0)
}

/// Adds the axioms that *refute* the variables something reads negatively.
///
/// A cluster of one variable is negated exactly. A larger cluster has a cyclic
/// positive dependency, and negating its definition literal by literal is
/// semantically wrong — the negated bodies then depend on each other in a cycle
/// and nothing can ever refute the variable. Mainline overapproximates instead:
/// the variables of such a cluster may be false unconditionally. That is a
/// relaxation, which is what the consumers of these axioms — the heuristics —
/// need; the axiom evaluator itself refutes a derived variable by finding it
/// unproven at the end of its layer and does not use these rules at all.
fn compute_negative_axioms(clusters: &mut [AxiomCluster]) {
    for cluster in clusters
        .iter_mut()
        .filter(|cluster| cluster.needed_negatively)
    {
        if cluster.variables.len() > 1 {
            for variable in &cluster.variables {
                let axioms = cluster
                    .axioms
                    .get_mut(variable)
                    .expect("a cluster holds an entry for each of its variables");
                let name = axioms
                    .first()
                    .expect("a necessary derived variable has at least one axiom")
                    .name
                    .clone();
                axioms.push(PropositionalAxiom::new(
                    name,
                    vec![],
                    Condition::NegatedAtom(variable.negate()),
                ));
            }
        } else {
            let variable = cluster
                .variables
                .first()
                .expect("a cluster has at least one variable")
                .clone();
            let axioms = cluster
                .axioms
                .get_mut(&variable)
                .expect("a cluster holds an entry for each of its variables");
            let negated = negate(axioms);
            axioms.extend(negated);
        }
    }
}

fn collect_axioms(clusters: &[AxiomCluster]) -> Vec<PropositionalAxiom> {
    clusters
        .iter()
        .flat_map(|cluster| {
            cluster
                .variables
                .iter()
                .flat_map(|variable| cluster.axioms_of(variable).iter().cloned())
        })
        .collect()
}

/// Removes duplicate conditions, duplicate axioms and dominated axioms from the
/// axioms of one head.
///
/// An axiom whose body contains its own head can never fire productively — the
/// head starts out false — so it is dropped rather than compared.
fn simplify(mut axioms: Vec<PropositionalAxiom>) -> Vec<PropositionalAxiom> {
    for axiom in &mut axioms {
        axiom
            .condition
            .sort_by_cached_key(|condition| format!("{condition:?}"));
        axiom.condition.dedup();
    }

    let mut axioms_to_skip: HashSet<usize> = HashSet::new();
    let mut axioms_by_literal: HashMap<Condition, HashSet<usize>> = HashMap::new();
    for (index, axiom) in axioms.iter().enumerate() {
        if axiom.condition.contains(&axiom.effect) {
            axioms_to_skip.insert(index);
            continue;
        }
        for literal in &axiom.condition {
            axioms_by_literal
                .entry(literal.clone())
                .or_default()
                .insert(index);
        }
    }

    for (index, axiom) in axioms.iter().enumerate() {
        // Keeps one of several identical axioms: an axiom already skipped no
        // longer dominates anything.
        if axioms_to_skip.contains(&index) {
            continue;
        }
        if axiom.condition.is_empty() {
            return vec![axiom.clone()];
        }

        let mut literals = axiom.condition.iter();
        let first_literal = literals.next().expect("checked non-empty");
        let mut dominated_axioms = axioms_by_literal
            .get(first_literal)
            .cloned()
            .unwrap_or_default();
        for literal in literals {
            match axioms_by_literal.get(literal) {
                Some(candidates) => {
                    dominated_axioms = dominated_axioms.intersection(candidates).copied().collect()
                }
                None => {
                    dominated_axioms.clear();
                    break;
                }
            }
        }
        for dominated_axiom in dominated_axioms {
            if dominated_axiom != index {
                axioms_to_skip.insert(dominated_axiom);
            }
        }
    }

    axioms
        .into_iter()
        .enumerate()
        .filter_map(|(index, axiom)| (!axioms_to_skip.contains(&index)).then_some(axiom))
        .collect()
}

/// Negate an axiom literal.
///
/// After normalization every axiom condition and effect is an atom or a negated
/// atom, so `negate_literal` returning `None` is a broken invariant rather than
/// a case to recover from. Falling back to the literal itself would substitute
/// `L` for `¬L` and invert the axiom's meaning.
fn negate_axiom_literal(literal: &Condition) -> Condition {
    literal
        .negate_literal()
        .unwrap_or_else(|| panic!("axiom literal is not negatable: {literal:?}"))
}

/// The rules refuting the head of `axioms`, which must all have the same head.
///
/// The head is false when every one of its axioms fails, and an axiom fails when
/// one of its body literals does, so the result is the cross product of the
/// negated bodies. Sound only when the head does not positively depend on
/// itself, which is why the caller restricts this to single-variable clusters.
fn negate(axioms: &[PropositionalAxiom]) -> Vec<PropositionalAxiom> {
    assert!(!axioms.is_empty());

    let initial_effect = negate_axiom_literal(&axioms[0].effect);
    let mut result = vec![PropositionalAxiom::new(
        axioms[0].name.clone(),
        vec![],
        initial_effect,
    )];

    for axiom in axioms {
        let condition = &axiom.condition;
        if condition.is_empty() {
            // The head is proven with an empty body, so it holds in every state
            // and nothing refutes it.
            return vec![];
        } else if condition.len() == 1 {
            let new_literal = negate_axiom_literal(&condition[0]);
            for result_axiom in &mut result {
                result_axiom.condition.push(new_literal.clone());
            }
        } else {
            let mut new_result = vec![];
            for literal in condition {
                let negated_literal = negate_axiom_literal(literal);
                for result_axiom in &result {
                    let mut new_axiom = result_axiom.clone_axiom();
                    new_axiom.condition.push(negated_literal.clone());
                    new_result.push(new_axiom);
                }
            }
            result = new_result;
        }
    }

    simplify(result)
}

/// The property the layers were computed for, checked on the result.
///
/// Mainline runs this behind a debug flag; it is a linear pass over the axioms
/// and the layering is the one thing here that no downstream component
/// re-checks, so it runs unconditionally.
fn verify_layering_condition(axioms: &[PropositionalAxiom], layers: &HashMap<Atom, i32>) {
    let mut heads: HashSet<Condition> = HashSet::new();
    let mut head_variables: HashSet<Atom> = HashSet::new();
    for axiom in axioms {
        let head = axiom
            .effect
            .literal_positive()
            .unwrap_or_else(|| panic!("an axiom head is a literal, got {}", axiom.effect));
        head_variables.insert(head);
        heads.insert(axiom.effect.clone());
    }

    // 1. A variable has a layer exactly when some rule writes it, and layers
    //    are non-negative; the `-1` of a non-derived variable is set elsewhere.
    let variables_with_layers: HashSet<Atom> = layers.keys().cloned().collect();
    assert_eq!(
        head_variables, variables_with_layers,
        "every derived variable with a layer must be written by a rule, and vice versa"
    );
    for (variable, &layer) in layers {
        assert!(layer >= 0, "derived variable {variable} has layer {layer}");
    }

    for axiom in axioms {
        let head = axiom
            .effect
            .literal_positive()
            .expect("checked above that the head is a literal");
        let head_layer = layers[&head];
        for condition in &axiom.condition {
            let Some(condition_variable) = condition.literal_positive() else {
                continue;
            };
            if !head_variables.contains(&condition_variable) {
                continue;
            }
            let condition_layer = layers[&condition_variable];

            // 2. A condition on a derived variable never comes from above.
            assert!(
                condition_layer <= head_layer,
                "the rule for {head} at layer {head_layer} reads {condition_variable} at layer \
                 {condition_layer}"
            );
            // 3. A condition read at the head's own layer must be one some rule
            //    writes at that sign, i.e. it is proven within the layer rather
            //    than assumed from the absence of a proof. A derived variable
            //    can appear in heads with both signs, since the negated axioms
            //    above are rules too, which is what makes this weaker than it
            //    would be if negation were left to the search component.
            assert!(
                condition_layer < head_layer || heads.contains(condition),
                "the rule for {head} at layer {head_layer} reads {condition} at the same layer, \
                 but no rule of that layer writes it"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use crate::options::LayerStrategy;

    /// How many derived variables sit in each layer, layer by layer.
    type LayerCounts = &'static [(i32, usize)];

    /// Every `:derived` fixture, and how many derived variables each strategy
    /// puts in each layer.
    ///
    /// The counts are over the non-negative entries of the task's
    /// `axiom_layers`. `numeric-body` derives its predicate from a comparison,
    /// so its lowest layer is a numeric one and identical under both
    /// strategies; the propositional layers sit on top of the numeric ones,
    /// which is why the others start at 1 rather than 0.
    ///
    /// What the two strategies do is visible in the second column: `min`
    /// collapses a positive chain into one layer, `max` gives each link a layer
    /// of its own, and a cyclic component still shares one — the pairs of
    /// `cyclic-negation` and the groups of four of `recursive-closure` are the
    /// strongly connected components, and no strategy can split them.
    const FIXTURE_LAYERS: &[(&str, LayerCounts, LayerCounts)] = &[
        (
            "conjunctive-chain",
            &[(1, 4)],
            &[(1, 1), (2, 1), (3, 1), (4, 1)],
        ),
        ("cyclic-negation", &[(1, 5)], &[(1, 2), (2, 2), (3, 1)]),
        ("disjunctive-support", &[(1, 3)], &[(1, 1), (2, 1), (3, 1)]),
        ("goal-condition", &[(1, 3)], &[(1, 1), (2, 1), (3, 1)]),
        (
            "layered-chain",
            &[(1, 2), (2, 2), (3, 1)],
            &[(1, 1), (2, 1), (3, 1), (4, 1), (5, 1)],
        ),
        (
            "negated-dependency",
            &[(1, 2), (2, 1)],
            &[(1, 1), (2, 1), (3, 1)],
        ),
        (
            "numeric-body",
            &[(2, 1), (3, 3)],
            &[(2, 1), (3, 1), (4, 1), (5, 1)],
        ),
        (
            "recursive-closure",
            &[(1, 21)],
            &[(1, 1), (2, 4), (3, 4), (4, 4), (5, 4), (6, 4)],
        ),
    ];

    fn fixture_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/assets/derived-predicates")
    }

    /// The `axiom_layers` of the translated fixture, one entry per SAS
    /// variable.
    ///
    /// [`super::handle_axioms`] checks the layering condition on every run, so a
    /// fixture that translates at all was laid out validly.
    fn axiom_layers(fixture: &str, layer_strategy: LayerStrategy) -> Vec<i32> {
        let dir = fixture_root().join(fixture);
        let path = |name: &str| {
            dir.join(name)
                .to_str()
                .expect("fixture path is UTF-8")
                .to_owned()
        };
        crate::api::translate_to_sas_task(
            &path("domain.pddl"),
            &path("problem.pddl"),
            false,
            layer_strategy,
        )
        .unwrap_or_else(|err| panic!("translating {fixture} failed: {err}"))
        .variables
        .axiom_layers
    }

    fn derived_variables_per_layer(layers: &[i32]) -> Vec<(i32, usize)> {
        let mut counts: BTreeMap<i32, usize> = BTreeMap::new();
        for &layer in layers.iter().filter(|&&layer| layer >= 0) {
            *counts.entry(layer).or_default() += 1;
        }
        counts.into_iter().collect()
    }

    /// The table above covers the fixture tree, so a fixture added tomorrow is
    /// laid out by both strategies rather than by neither.
    #[test]
    fn the_layering_table_names_every_derived_fixture() {
        let mut found: Vec<String> = std::fs::read_dir(fixture_root())
            .expect("the derived-predicate fixtures are checked in")
            .map(|entry| {
                entry
                    .expect("readable directory entry")
                    .file_name()
                    .into_string()
                    .expect("fixture name is UTF-8")
            })
            .collect();
        found.sort();
        let mut listed: Vec<String> = FIXTURE_LAYERS
            .iter()
            .map(|&(fixture, ..)| fixture.to_owned())
            .collect();
        listed.sort();
        assert_eq!(found, listed);
    }

    #[test]
    fn both_layer_strategies_lay_out_every_derived_fixture() {
        for &(fixture, min_layers, max_layers) in FIXTURE_LAYERS {
            let min = axiom_layers(fixture, LayerStrategy::Min);
            let max = axiom_layers(fixture, LayerStrategy::Max);

            assert_eq!(derived_variables_per_layer(&min), min_layers, "{fixture}");
            assert_eq!(derived_variables_per_layer(&max), max_layers, "{fixture}");

            // A strategy decides how far apart the derived variables are laid
            // out, not which variables are derived; and spreading them out
            // never moves one down.
            assert_eq!(min.len(), max.len(), "{fixture}");
            for (var, (&min_layer, &max_layer)) in min.iter().zip(&max).enumerate() {
                assert_eq!(
                    min_layer < 0,
                    max_layer < 0,
                    "{fixture}: variable {var} is derived under one strategy only"
                );
                assert!(
                    min_layer <= max_layer,
                    "{fixture}: variable {var} sits at layer {max_layer} under max, below the \
                     {min_layer} of min"
                );
            }
        }
    }

    #[test]
    fn a_layer_strategy_is_named_as_mainline_spells_it() {
        assert_eq!("min".parse(), Ok(LayerStrategy::Min));
        assert_eq!("max".parse(), Ok(LayerStrategy::Max));
        assert_eq!(LayerStrategy::default(), LayerStrategy::Min);
        assert!("lowest".parse::<LayerStrategy>().is_err());
    }
}
