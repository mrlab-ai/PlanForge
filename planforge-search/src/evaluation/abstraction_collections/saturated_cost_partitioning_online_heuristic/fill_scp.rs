use super::*;

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
        let components = temp.components.borrow();
        let abstract_state_ids = components
            .iter()
            .map(|component| match component {
                AbstractionComponent::Domain(heuristic) => {
                    Some(heuristic.abstraction().distance_table.initial_state_hash)
                }
                AbstractionComponent::Cartesian(heuristic) => {
                    Some(heuristic.abstraction().transition_system.initial_state_hash)
                }
                AbstractionComponent::PatternDatabase(_) => Some(0),
            })
            .collect::<Vec<_>>();
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
                &components,
                deadline,
            )?
        };
        let standalone_current_h = {
            let state = temp.state.borrow();
            standalone_current_h_values(&state, &abstract_state_ids)
        };
        let collection = PartitionedCollection {
            task,
            components: &components,
            abstract_state_ids: &abstract_state_ids,
            standalone_current_h: &standalone_current_h,
            original_costs: &original_costs,
        };
        let (mut cp_heuristic, mut residual_costs, mut residual_partitions) =
            if config.partitioning.uses_regions() {
                let (cp, costs, partitions) = temp.build_abstract_operator_fill_scp(
                    collection,
                    &order,
                    deadline,
                    config.saturator,
                )?;
                (cp, costs, Some(partitions))
            } else {
                let (cp, costs) = temp.build_label_fill_scp(collection, &order, deadline)?;
                (cp, costs, None)
            };
        if config.order_optimization_max_time > 0.0 {
            let optimization_deadline = config.order_optimization_max_time.is_finite().then(|| {
                Instant::now() + Duration::from_secs_f64(config.order_optimization_max_time)
            });
            temp.optimize_order_with_hill_climbing(
                collection,
                &mut order,
                &mut cp_heuristic,
                optimization_deadline,
            )?;
            (cp_heuristic, residual_costs, residual_partitions) =
                if config.partitioning.uses_regions() {
                    let (cp, costs, partitions) = temp.build_abstract_operator_fill_scp(
                        collection,
                        &order,
                        deadline,
                        config.saturator,
                    )?;
                    (cp, costs, Some(partitions))
                } else {
                    let (cp, costs) = temp.build_label_fill_scp(collection, &order, deadline)?;
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

    fn heuristic_name(&self) -> &str {
        &self.name
    }

    fn dead_ends_are_reliable(&self) -> bool {
        true
    }
}
