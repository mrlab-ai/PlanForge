//! Axiom simplification and layering, following mainline Fast Downward's
//! issue453 and issue454.
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
//! Every derived variable defaults to false, and every rule here *proves* one:
//! the rules that refute a derived variable are no longer produced. They were
//! never for the axiom evaluator, which refutes a variable by finding it
//! unproven at the end of its layer; they were for the heuristics that read the
//! axioms as relaxed operators, and those now derive them for themselves from
//! the SAS task — see `planforge_sas::default_value_axioms`. That is issue454,
//! and it buys three things: a task with axioms no longer pays for a negation
//! that can blow up exponentially unless a heuristic asks for it, the negation
//! is exact over SAS values rather than over PDDL literals, and every derived
//! variable now appears in the heads with one polarity only, which is what makes
//! [`verify_layering_condition`] able to insist that a negation-by-failure
//! reading comes from a strictly lower layer.

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

    /// Drops every variable that is not needed.
    ///
    /// Whole entries can go: if a head is needed then so is everything in the
    /// bodies of its axioms, by the definition of `necessary_atoms`.
    fn remove_unnecessary_variables(&mut self, necessary_atoms: &HashSet<Atom>) {
        let AxiomDependencies {
            derived_variables,
            positive,
            negative,
        } = self;
        derived_variables.retain(|variable| {
            let necessary = necessary_atoms.contains(variable);
            if !necessary {
                positive.remove(variable);
                negative.remove(variable);
            }
            necessary
        });
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
/// layers are assigned to.
struct AxiomCluster {
    variables: Vec<Atom>,
    axioms: HashMap<Atom, Vec<PropositionalAxiom>>,
    /// Clusters holding a variable that occurs positively in the body of one of
    /// this cluster's axioms, and negatively respectively. Indices into the
    /// cluster list; a cluster is never its own positive child.
    positive_children: BTreeSet<usize>,
    negative_children: BTreeSet<usize>,
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

/// The derived variables the rest of the task can observe, closed under the
/// axiom bodies.
///
/// Signs used to be tracked here, because they decided which variables needed
/// negated axioms generating. Nothing in this module generates those any more, so
/// what is left is the pruning question — may a variable be dropped from the
/// dependency graph, and hence lose its axioms and its layer — and that is
/// sign-free: a variable is kept when either of its literals is observed. Which
/// is the same set the signed closure reached, since it followed every edge in
/// both directions too and only differed in the sign it recorded.
fn compute_necessary_atoms(
    dependencies: &AxiomDependencies,
    operators: &[PropositionalAction],
    goal_list: &[Condition],
    global_constraint: &Condition,
) -> HashSet<Atom> {
    let mut necessary: HashSet<Atom> = HashSet::new();
    let mut queue: Vec<Atom> = vec![];

    let require = |literal: &Condition, necessary: &mut HashSet<Atom>, queue: &mut Vec<Atom>| {
        let Some(atom) = literal.literal_positive() else {
            return;
        };
        if dependencies.derived_variables.contains(&atom) && necessary.insert(atom.clone()) {
            queue.push(atom);
        }
    };

    for literal in goal_list
        .iter()
        .chain(std::iter::once(global_constraint))
        .chain(operators.iter().flat_map(|op| op.precondition.iter()))
        .chain(
            operators
                .iter()
                .flat_map(|op| op.add_effects.iter().chain(&op.del_effects))
                .flat_map(|(condition, _)| condition.iter()),
        )
    {
        require(literal, &mut necessary, &mut queue);
    }

    while let Some(atom) = queue.pop() {
        for body_atom in dependencies
            .positive_edges(&atom)
            .chain(dependencies.negative_edges(&atom))
        {
            let required = Condition::Atom(body_atom.clone());
            require(&required, &mut necessary, &mut queue);
        }
    }

    necessary
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

    planforge_sas::utils::scc::Scc::new(graph)
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
    let necessary_atoms =
        compute_necessary_atoms(&dependencies, operators, goal_list, global_constraint);
    dependencies.remove_unnecessary_variables(&necessary_atoms);

    let sorted_variables = dependencies.sorted_variables();
    let components = strongly_connected_components(&dependencies, &sorted_variables);

    let mut clusters: Vec<AxiomCluster> = components.into_iter().map(AxiomCluster::new).collect();
    let mut cluster_of: HashMap<Atom, usize> = HashMap::new();
    for (index, cluster) in clusters.iter().enumerate() {
        for variable in &cluster.variables {
            cluster_of.insert(variable.clone(), index);
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

/// The property the layers were computed for, checked on the result.
///
/// Mainline runs this behind a debug flag; it is a linear pass over the axioms
/// and the layering is the one thing here that no downstream component
/// re-checks, so it runs unconditionally.
fn verify_layering_condition(axioms: &[PropositionalAxiom], layers: &HashMap<Atom, i32>) {
    let head_variables: HashSet<Atom> = axioms.iter().map(head_of).collect();

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
        let head = head_of(axiom);
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
            // 3. A negation-by-failure reading — the variable at its default
            //    value, which is what a negated literal is now that every head
            //    is positive — comes from a strictly lower layer, so the absence
            //    of a proof has settled before it is read. A positive reading may
            //    share the layer, because one layer's fixpoint propagates it.
            assert!(
                !condition.is_negated() || condition_layer < head_layer,
                "the rule for {head} at layer {head_layer} reads {condition} at the same layer, \
                 but whether {condition_variable} stayed unproven is not settled until that \
                 layer is done"
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
