use std::collections::BTreeSet;

use planners_sas::numeric::numeric_task::{AbstractNumericTask, NumericType};

use super::projected_task::{
    AuxiliaryNumericVar, build_assignment_axiom_lookup, build_auxiliary_numeric_vars,
};

#[derive(Debug, Clone)]
pub(crate) struct NumericSupportContext {
    assignment_lookup: Vec<Option<usize>>,
    auxiliary_numeric_vars: Vec<AuxiliaryNumericVar>,
    derived_to_helper_id: Vec<Option<usize>>,
}

impl NumericSupportContext {
    pub(crate) fn new(task: &dyn AbstractNumericTask) -> Self {
        let assignment_lookup = build_assignment_axiom_lookup(task);
        let base_initial_numeric_values = task.get_initial_numeric_state_values().to_vec();
        let auxiliary_numeric_vars =
            build_auxiliary_numeric_vars(task, &assignment_lookup, &base_initial_numeric_values)
                .unwrap_or_default();
        let mut derived_to_helper_id = vec![None; task.numeric_variables().len()];
        for auxiliary_numeric_var in &auxiliary_numeric_vars {
            derived_to_helper_id[auxiliary_numeric_var.source_numeric_var_id] =
                Some(auxiliary_numeric_var.helper_id);
        }

        Self {
            assignment_lookup,
            auxiliary_numeric_vars,
            derived_to_helper_id,
        }
    }

    pub(crate) fn helper_space_len(&self, task: &dyn AbstractNumericTask) -> usize {
        task.numeric_variables().len() + self.auxiliary_numeric_vars.len()
    }

    pub(crate) fn auxiliary_numeric_vars(&self) -> &[AuxiliaryNumericVar] {
        &self.auxiliary_numeric_vars
    }

    pub(crate) fn assignment_axiom_id_for(&self, numeric_var_id: usize) -> Option<usize> {
        self.assignment_lookup
            .get(numeric_var_id)
            .copied()
            .flatten()
    }

    pub(crate) fn is_helper_var_id(
        &self,
        task: &dyn AbstractNumericTask,
        numeric_var_id: usize,
    ) -> bool {
        numeric_var_id >= task.numeric_variables().len()
            && numeric_var_id < self.helper_space_len(task)
    }

    pub(crate) fn helper_source_numeric_var_id(&self, helper_id: usize) -> Option<usize> {
        self.auxiliary_numeric_vars
            .iter()
            .find(|auxiliary_numeric_var| auxiliary_numeric_var.helper_id == helper_id)
            .map(|auxiliary_numeric_var| auxiliary_numeric_var.source_numeric_var_id)
    }

    pub(crate) fn helper_id_for_derived(&self, numeric_var_id: usize) -> Option<usize> {
        self.derived_to_helper_id
            .get(numeric_var_id)
            .copied()
            .flatten()
    }

    pub(crate) fn comparison_support_ids(
        &self,
        task: &dyn AbstractNumericTask,
        comparison_axiom_id: usize,
    ) -> Vec<usize> {
        let Some(comparison_axiom) = task.comparison_axioms().get(comparison_axiom_id) else {
            return Vec::new();
        };

        let mut support_ids = BTreeSet::new();
        for numeric_var_id in [
            comparison_axiom.get_left_var_id(),
            comparison_axiom.get_right_var_id(),
        ] {
            let Ok(numeric_var_id) = usize::try_from(numeric_var_id) else {
                continue;
            };
            self.collect_numeric_support_ids(
                task,
                numeric_var_id,
                &mut BTreeSet::new(),
                &mut support_ids,
            );
        }
        support_ids.into_iter().collect()
    }

    pub(crate) fn numeric_var_support_ids(
        &self,
        task: &dyn AbstractNumericTask,
        numeric_var_id: usize,
    ) -> Vec<usize> {
        let mut support_ids = BTreeSet::new();
        self.collect_numeric_support_ids(
            task,
            numeric_var_id,
            &mut BTreeSet::new(),
            &mut support_ids,
        );
        support_ids.into_iter().collect()
    }

    fn collect_numeric_support_ids(
        &self,
        task: &dyn AbstractNumericTask,
        numeric_var_id: usize,
        visiting: &mut BTreeSet<usize>,
        support_ids: &mut BTreeSet<usize>,
    ) {
        let Some(numeric_var) = task.numeric_variables().get(numeric_var_id) else {
            return;
        };

        match numeric_var.get_type() {
            NumericType::Regular => {
                support_ids.insert(numeric_var_id);
            }
            NumericType::Derived => {
                if let Some(helper_id) = self.helper_id_for_derived(numeric_var_id) {
                    support_ids.insert(helper_id);
                    return;
                }

                if !visiting.insert(numeric_var_id) {
                    return;
                }

                if let Some(axiom_id) = self
                    .assignment_lookup
                    .get(numeric_var_id)
                    .copied()
                    .flatten()
                {
                    if let Some(assignment_axiom) = task.assignment_axioms().get(axiom_id) {
                        for dependency_var_id in [
                            assignment_axiom.get_left_var_id(),
                            assignment_axiom.get_right_var_id(),
                        ] {
                            let Ok(dependency_var_id) = usize::try_from(dependency_var_id) else {
                                continue;
                            };
                            self.collect_numeric_support_ids(
                                task,
                                dependency_var_id,
                                visiting,
                                support_ids,
                            );
                        }
                    }
                }

                visiting.remove(&numeric_var_id);
            }
            NumericType::Constant | NumericType::Cost => {}
        }
    }
}
