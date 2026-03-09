/// Port of simplify.py
/// Simplification of SAS+ tasks by removing unreachable propositions.

use std::collections::{HashMap, HashSet};

use super::sas_tasks::*;

const DEBUG: bool = false;

// ============================================================
// Exceptions
// ============================================================

#[derive(Debug)]
pub enum SimplifyError {
    Impossible,
    TriviallySolvable,
    DoesNothing,
}

impl std::fmt::Display for SimplifyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SimplifyError::Impossible => write!(f, "Impossible"),
            SimplifyError::TriviallySolvable => write!(f, "TriviallySolvable"),
            SimplifyError::DoesNothing => write!(f, "DoesNothing"),
        }
    }
}

impl std::error::Error for SimplifyError {}

// ============================================================
// Sentinel values for renaming
// ============================================================

/// Represents a value that is always false (unreachable) or always true (the only value).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenamedValue {
    Normal(usize),
    AlwaysFalse,
    AlwaysTrue,
}

// ============================================================
// DomainTransitionGraph
// ============================================================

struct DomainTransitionGraph {
    init: usize,
    size: usize,
    arcs: HashMap<usize, HashSet<usize>>,
}

impl DomainTransitionGraph {
    fn new(init: usize, size: usize) -> Self {
        DomainTransitionGraph {
            init,
            size,
            arcs: HashMap::new(),
        }
    }

    fn add_arc(&mut self, u: usize, v: usize) {
        self.arcs.entry(u).or_insert_with(HashSet::new).insert(v);
    }

    fn reachable(&self) -> HashSet<usize> {
        let mut queue = vec![self.init];
        let mut reachable: HashSet<usize> = HashSet::new();
        reachable.insert(self.init);
        while let Some(node) = queue.pop() {
            if let Some(neighbors) = self.arcs.get(&node) {
                for &n in neighbors {
                    if reachable.insert(n) {
                        queue.push(n);
                    }
                }
            }
        }
        reachable
    }
}

// ============================================================
// build_dtgs
// ============================================================

fn build_dtgs(task: &SASTask) -> Vec<DomainTransitionGraph> {
    let init_vals = &task.init.values;
    let sizes = &task.variables.ranges;

    let mut dtgs: Vec<DomainTransitionGraph> = init_vals.iter()
        .zip(sizes.iter())
        .filter(|(_, &size)| size > 0)
        .map(|(&init, &size)| DomainTransitionGraph::new(init as usize, size))
        .collect();

    let add_arc = |dtgs: &mut Vec<DomainTransitionGraph>, var_no: usize, pre_spec: i32, post: usize| {
        if pre_spec == -1 {
            for pre in 0..sizes[var_no] {
                if pre != post {
                    dtgs[var_no].add_arc(pre, post);
                }
            }
        } else {
            dtgs[var_no].add_arc(pre_spec as usize, post);
        }
    };

    let get_effective_pre = |var_no: usize,
                             conditions: &HashMap<usize, usize>,
                             effect_conditions: &[(usize, usize)]|
     -> Option<i32> {
        let mut result: i32 = *conditions.get(&var_no).map(|v| v as &usize).unwrap_or(&(usize::MAX)) as i32;
        if result == usize::MAX as i32 {
            result = -1;
        }
        for &(cond_var_no, cond_val) in effect_conditions {
            if cond_var_no == var_no {
                if result == -1 {
                    result = cond_val as i32;
                } else if cond_val as i32 != result {
                    return None; // contradictory conditions
                }
            }
        }
        Some(result)
    };

    for op in &task.operators {
        let conditions: HashMap<usize, usize> = op.get_applicability_conditions()
            .into_iter()
            .collect();
        for &(var_no, _, post, ref cond) in &op.pre_post {
            if let Some(effective_pre) = get_effective_pre(var_no, &conditions, cond) {
                add_arc(&mut dtgs, var_no, effective_pre, post);
            }
        }
    }

    for axiom in &task.axioms {
        let (var_no, val) = axiom.effect;
        add_arc(&mut dtgs, var_no, -1, val);
    }

    for cax in &task.comp_axioms {
        let eff_var = cax.effect;
        add_arc(&mut dtgs, eff_var, -1, 0);
        add_arc(&mut dtgs, eff_var, -1, 1);
    }

    dtgs
}

// ============================================================
// VarValueRenaming
// ============================================================

struct VarValueRenaming {
    new_var_nos: Vec<Option<usize>>,      // indexed by old var_no
    new_values: Vec<Vec<RenamedValue>>,    // indexed by old var_no and old value
    new_sizes: Vec<usize>,                // indexed by new var_no
    new_var_count: usize,
    num_removed_values: usize,
}

impl VarValueRenaming {
    fn new() -> Self {
        VarValueRenaming {
            new_var_nos: vec![],
            new_values: vec![],
            new_sizes: vec![],
            new_var_count: 0,
            num_removed_values: 0,
        }
    }

    fn register_variable(&mut self, old_domain_size: usize, init_value: usize, new_domain: &HashSet<usize>) {
        assert!(new_domain.len() >= 1 && new_domain.len() <= old_domain_size);
        assert!(new_domain.contains(&init_value));

        if new_domain.len() == 1 {
            // Remove this variable completely.
            let mut new_values_for_var = vec![RenamedValue::AlwaysFalse; old_domain_size];
            new_values_for_var[init_value] = RenamedValue::AlwaysTrue;
            self.new_var_nos.push(None);
            self.new_values.push(new_values_for_var);
            self.num_removed_values += old_domain_size;
        } else {
            let mut new_value_counter = 0usize;
            let mut new_values_for_var = vec![];
            for value in 0..old_domain_size {
                if new_domain.contains(&value) {
                    new_values_for_var.push(RenamedValue::Normal(new_value_counter));
                    new_value_counter += 1;
                } else {
                    self.num_removed_values += 1;
                    new_values_for_var.push(RenamedValue::AlwaysFalse);
                }
            }
            let new_size = new_value_counter;
            assert_eq!(new_size, new_domain.len());

            self.new_var_nos.push(Some(self.new_var_count));
            self.new_values.push(new_values_for_var);
            self.new_sizes.push(new_size);
            self.new_var_count += 1;
        }
    }

    fn apply_to_task(&self, task: &mut SASTask) -> Result<(), SimplifyError> {
        self.apply_to_variables(&mut task.variables);
        self.apply_to_mutexes(&mut task.mutexes);
        self.apply_to_init(&mut task.init);
        self.apply_to_goals(&mut task.goal.pairs)?;
        task.global_constraint = self.translate_global_constraint(task.global_constraint);
        self.apply_to_operators(&mut task.operators);
        self.apply_to_axioms(&mut task.axioms, &mut task.comp_axioms);
        Ok(())
    }

    fn apply_to_variables(&self, variables: &mut SASVariables) {
        variables.ranges = self.new_sizes.clone();
        let mut new_axiom_layers = vec![-1i32; self.new_var_count];
        for (old_no, new_no) in self.new_var_nos.iter().enumerate() {
            if let Some(nv) = new_no {
                new_axiom_layers[*nv] = variables.axiom_layers[old_no];
            }
        }
        variables.axiom_layers = new_axiom_layers;
        self.apply_to_value_names(&mut variables.value_names);
    }

    fn apply_to_value_names(&self, value_names: &mut Vec<Vec<String>>) {
        let mut new_value_names: Vec<Vec<String>> = self.new_sizes.iter()
            .map(|&size| vec![String::new(); size])
            .collect();

        for (var_no, values) in value_names.iter().enumerate() {
            for (value, value_name) in values.iter().enumerate() {
                let (new_var_no, new_value) = self.translate_pair(var_no, value);
                match new_value {
                    RenamedValue::AlwaysTrue => {
                        if DEBUG {
                            println!("Removed true proposition: {}", value_name);
                        }
                    }
                    RenamedValue::AlwaysFalse => {
                        if DEBUG {
                            println!("Removed false proposition: {}", value_name);
                        }
                    }
                    RenamedValue::Normal(nv) => {
                        if let Some(nvn) = new_var_no {
                            new_value_names[nvn][nv] = value_name.clone();
                        }
                    }
                }
            }
        }

        *value_names = new_value_names;
    }

    fn apply_to_mutexes(&self, mutexes: &mut Vec<SASMutexGroup>) {
        let mut new_mutexes = vec![];
        for mutex in mutexes.iter() {
            let mut new_facts = vec![];
            for &(var, val) in &mutex.facts {
                let (new_var_no, new_value) = self.translate_pair(var, val);
                if let RenamedValue::Normal(nv) = new_value {
                    if let Some(nvn) = new_var_no {
                        new_facts.push((nvn, nv));
                    }
                }
            }
            if new_facts.len() >= 2 {
                new_mutexes.push(SASMutexGroup::new(new_facts));
            }
        }
        *mutexes = new_mutexes;
    }

    fn apply_to_init(&self, init: &mut SASInit) {
        let init_pairs: Vec<(usize, usize)> = init.values.iter()
            .enumerate()
            .map(|(var, &val)| (var, val as usize))
            .collect();

        let mut new_values = vec![0i32; self.new_var_count];
        for (var, val) in init_pairs {
            let (new_var_no, new_value) = self.translate_pair(var, val);
            match new_value {
                RenamedValue::AlwaysFalse => {
                    panic!("Initial state impossible? Inconceivable!");
                }
                RenamedValue::AlwaysTrue => {
                    // Variable removed, skip
                }
                RenamedValue::Normal(nv) => {
                    if let Some(nvn) = new_var_no {
                        new_values[nvn] = nv as i32;
                    }
                }
            }
        }
        init.values = new_values;
    }

    fn apply_to_goals(&self, goals: &mut Vec<(usize, usize)>) -> Result<(), SimplifyError> {
        let mut new_goals = vec![];
        for &(var, val) in goals.iter() {
            let (new_var_no, new_value) = self.translate_pair(var, val);
            match new_value {
                RenamedValue::AlwaysFalse => {
                    return Err(SimplifyError::Impossible);
                }
                RenamedValue::AlwaysTrue => {
                    // Goal already satisfied, skip
                }
                RenamedValue::Normal(nv) => {
                    if let Some(nvn) = new_var_no {
                        new_goals.push((nvn, nv));
                    }
                }
            }
        }
        *goals = new_goals;
        if goals.is_empty() {
            return Err(SimplifyError::TriviallySolvable);
        }
        Ok(())
    }

    fn translate_global_constraint(&self, constraint: (usize, usize)) -> (usize, usize) {
        let (new_var, new_value) = self.translate_pair(constraint.0, constraint.1);
        match new_value {
            RenamedValue::AlwaysFalse => {
                panic!("Task is unsolvable, Global constraint cannot be satisfied");
            }
            RenamedValue::AlwaysTrue => {
                panic!("Regular Task without global constraint");
            }
            RenamedValue::Normal(nv) => {
                let nvn = new_var.expect("global constraint variable removed");
                println!("Simplified global constraint to new variable ordering {:?}", (nvn, nv));
                (nvn, nv)
            }
        }
    }

    fn apply_to_operators(&self, operators: &mut Vec<SASOperator>) {
        let mut new_operators = vec![];
        let mut num_removed = 0;
        for op in operators.iter() {
            match self.translate_operator(op) {
                Some(new_op) => new_operators.push(new_op),
                None => {
                    num_removed += 1;
                    if DEBUG {
                        println!("Removed operator: {}", op.name);
                    }
                }
            }
        }
        println!("{} operators removed", num_removed);
        *operators = new_operators;
    }

    fn apply_to_axioms(&self, axioms: &mut Vec<SASAxiom>, comp_axioms: &mut Vec<SASCompareAxiom>) {
        let mut new_axioms = vec![];
        let mut num_removed = 0;
        for axiom in axioms.iter() {
            match self.apply_to_axiom(axiom) {
                Ok(new_axiom) => new_axioms.push(new_axiom),
                Err(_) => {
                    num_removed += 1;
                    if DEBUG {
                        println!("Removed axiom:");
                        axiom.dump();
                    }
                }
            }
        }
        println!("{} axioms removed", num_removed);
        *axioms = new_axioms;

        for cax in comp_axioms.iter_mut() {
            if let Some(new_var) = self.new_var_nos[cax.effect] {
                cax.effect = new_var;
            }
        }
    }

    fn translate_operator(&self, op: &SASOperator) -> Option<SASOperator> {
        let mut applicability_conditions = op.get_applicability_conditions();
        match self.convert_pairs(&mut applicability_conditions) {
            Err(_) => return None, // Never applicable
            Ok(()) => {}
        }

        let conditions_dict: HashMap<usize, usize> = applicability_conditions.iter()
            .cloned()
            .collect();
        let mut new_prevail_vars: HashSet<usize> = conditions_dict.keys().cloned().collect();
        let mut new_pre_post = vec![];
        let mut new_assign_effects = vec![];

        for entry in &op.pre_post {
            if let Some(new_entry) = self.translate_pre_post(entry, &conditions_dict) {
                let new_var = new_entry.0;
                new_prevail_vars.remove(&new_var);
                new_pre_post.push(new_entry);
            }
        }

        for entry in &op.assign_effects {
            new_assign_effects.push(entry.clone());
        }

        if new_pre_post.is_empty() && new_assign_effects.is_empty() {
            return None; // No effect
        }

        let new_prevail: Vec<(usize, usize)> = conditions_dict.iter()
            .filter(|(var, _)| new_prevail_vars.contains(var))
            .map(|(&var, &val)| (var, val))
            .collect();

        Some(SASOperator::new(
            op.name.clone(),
            new_prevail,
            new_pre_post,
            new_assign_effects,
            op.cost,
        ))
    }

    fn apply_to_axiom(&self, axiom: &SASAxiom) -> Result<SASAxiom, SimplifyError> {
        let mut new_condition = axiom.condition.clone();
        self.convert_pairs(&mut new_condition)?;

        let (new_var, new_value) = self.translate_pair(axiom.effect.0, axiom.effect.1);
        match new_value {
            RenamedValue::AlwaysFalse => {
                // Python: assert not new_value is always_false
                // If the new_value is always false, then the condition must
                // have been impossible (which should have been caught above).
                panic!("axiom effect value is always_false, \
                        condition should have been impossible");
            }
            RenamedValue::AlwaysTrue => {
                return Err(SimplifyError::DoesNothing);
            }
            RenamedValue::Normal(nv) => {
                let nvn = new_var.ok_or(SimplifyError::DoesNothing)?;
                Ok(SASAxiom::new(new_condition, (nvn, nv)))
            }
        }
    }

    fn translate_pre_post(
        &self,
        entry: &(usize, i32, usize, Vec<(usize, usize)>),
        conditions_dict: &HashMap<usize, usize>,
    ) -> Option<(usize, i32, usize, Vec<(usize, usize)>)> {
        let (var_no, pre, post, ref cond) = *entry;

        let (new_var_no_opt, new_post) = self.translate_pair(var_no, post);
        if new_post == RenamedValue::AlwaysTrue {
            return None;
        }

        let new_pre = if pre == -1 {
            -1i32
        } else {
            let (_, np) = self.translate_pair(var_no, pre as usize);
            match np {
                RenamedValue::AlwaysFalse => {
                    panic!("This function should only be called for operators \
                            whose applicability conditions are deemed possible.");
                }
                RenamedValue::AlwaysTrue => {
                    // In Python: new_pre can be always_true, but it's asserted
                    // not to be at the end if the effect can fire and changes value.
                    // We'll handle this below in the equality check.
                    // Use a sentinel to track.
                    -2i32
                }
                RenamedValue::Normal(v) => v as i32,
            }
        };

        // Python: if new_post == new_pre: return None
        // For sentinels: always_true post is already handled above.
        // If both are always_true, post was handled. If both are Normal with same value, skip.
        let new_post_val = match new_post {
            RenamedValue::Normal(v) => v,
            RenamedValue::AlwaysFalse => {
                panic!("if we survived so far, this effect can trigger \
                        and then new_post cannot be always_false");
            }
            _ => return None,
        };

        if new_pre >= 0 && new_pre as usize == new_post_val {
            return None; // no-op
        }

        // -2 means always_true. Python asserts new_pre is not always_true
        // if the effect changes the value and can fire.
        assert!(new_pre != -2, "if this pre_post changes the value and can fire, \
                new_pre cannot be always_true");

        let mut new_cond = cond.clone();
        match self.convert_pairs(&mut new_cond) {
            Err(_) => return None, // Effect conditions impossible
            Ok(()) => {}
        }

        for &(cond_var, cond_value) in &new_cond {
            if let Some(&cond_dict_val) = conditions_dict.get(&cond_var) {
                if cond_dict_val != cond_value {
                    return None; // Incompatible with applicability
                }
            }
        }

        let new_var_no = new_var_no_opt.expect("post is Normal but var was removed?");

        Some((new_var_no, new_pre, new_post_val, new_cond))
    }

    fn translate_pair(&self, var_no: usize, value: usize) -> (Option<usize>, RenamedValue) {
        let new_var_no = self.new_var_nos[var_no];
        let new_value = self.new_values[var_no][value];
        (new_var_no, new_value)
    }

    fn convert_pairs(&self, pairs: &mut Vec<(usize, usize)>) -> Result<(), SimplifyError> {
        let mut new_pairs = vec![];
        for &(var, val) in pairs.iter() {
            let (new_var_no, new_value) = self.translate_pair(var, val);
            match new_value {
                RenamedValue::AlwaysFalse => {
                    return Err(SimplifyError::Impossible);
                }
                RenamedValue::AlwaysTrue => {
                    // Skip, always satisfied
                }
                RenamedValue::Normal(nv) => {
                    if let Some(nvn) = new_var_no {
                        new_pairs.push((nvn, nv));
                    }
                }
            }
        }
        *pairs = new_pairs;
        Ok(())
    }
}

// ============================================================
// build_renaming
// ============================================================

fn build_renaming(dtgs: &[DomainTransitionGraph]) -> VarValueRenaming {
    let mut renaming = VarValueRenaming::new();
    for dtg in dtgs {
        renaming.register_variable(dtg.size, dtg.init, &dtg.reachable());
    }
    renaming
}

// ============================================================
// filter_unreachable_propositions
// ============================================================

/// Python: def filter_unreachable_propositions(sas_task)
/// Simplifies the task in-place. Returns Err(Impossible) or Err(TriviallySolvable)
/// if the task is detected as unsolvable or trivially solvable.
pub fn filter_unreachable_propositions(sas_task: &mut SASTask) -> Result<(), SimplifyError> {
    if DEBUG {
        sas_task.validate();
    }

    let dtgs = build_dtgs(sas_task);
    let renaming = build_renaming(&dtgs);

    renaming.apply_to_task(sas_task)?;

    println!("{} propositions removed", renaming.num_removed_values);

    if DEBUG {
        sas_task.validate();
    }

    Ok(())
}

// ============================================================
// trivial_task (for error recovery)
// ============================================================

/// Python: def trivial_task(solvable)
/// Creates a trivial SAS task (either solvable or unsolvable).
pub fn trivial_task(solvable: bool) -> SASTask {
    let variables = SASVariables::new(
        vec![2],
        vec![-1],
        vec![vec!["Atom dummy(val1)".to_string(), "Atom dummy(val2)".to_string()]],
        0,
    );
    let num_variables = SASNumericVariables::new(
        vec!["total-cost".to_string()],
        vec![-1],
        vec!["I".to_string()],
    );
    let mutexes = vec![];
    let init = SASInit::new(vec![0], vec![0.0]);
    let goal_fact = if solvable { (0, 0) } else { (0, 1) };
    let goal = SASGoal::new(vec![goal_fact]);
    let operators = vec![];
    let axioms = vec![];
    let comp_axioms = vec![];
    let numeric_axioms = vec![];
    let global_constraint = (0, 0);
    let metric = ("<".to_string(), 0);
    let init_constant_predicates = vec![];
    let init_constant_numerics = vec![];

    SASTask::new(
        variables,
        num_variables,
        mutexes,
        init,
        goal,
        operators,
        axioms,
        comp_axioms,
        numeric_axioms,
        global_constraint,
        metric,
        init_constant_predicates,
        init_constant_numerics,
    )
}
