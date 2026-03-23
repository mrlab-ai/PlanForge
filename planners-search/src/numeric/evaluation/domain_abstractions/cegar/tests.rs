use super::*;

use planners_sas::numeric::numeric_task::{
	ExplicitVariable, Metric, NumericRootTask, NumericVariable, Operator,
};

#[test]
fn get_flaws_returns_empty_for_valid_wildcard_plan() {
	let variables = vec![ExplicitVariable::new(
		2,
		"v".into(),
		vec!["v0".into(), "v1".into()],
		-1,
		0,
	)];
	let numeric_variables: Vec<NumericVariable> = vec![];
	let goals = vec![Fact::new(0, 1)];
	let op = Operator::new(
		"set".into(),
		vec![Fact::new(0, 0)],
		vec![planners_sas::numeric::numeric_task::Effect::new(vec![], 0, 0, 1)],
		vec![],
		1,
	);
	let task = NumericRootTask::new(
		4,
		Metric::new(true, -1),
		variables,
		numeric_variables,
		goals,
		vec![],
		vec![0],
		vec![],
		vec![op],
		vec![],
		vec![],
		vec![],
		(0, 0),
	);

	let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
	let partitions = NumericPartitions::trivial(&task);
	let numeric_domain_sizes: Vec<usize> = vec![];
	let factory = DomainAbstractionFactory::new(
		&task,
		domain_mapping,
		domain_sizes,
		partitions,
		numeric_domain_sizes,
	)
	.unwrap();
	let plan = factory
		.compute_wildcard_plan(&task, true, false)
		.unwrap()
		.expect("plan exists");

	let cegar = Cegar::new(CegarConfig::default()).unwrap();
	let flaws = cegar.get_flaws(&task, factory.partitions(), &plan, false).unwrap();
	assert!(flaws.is_empty());
}

#[test]
fn get_flaws_reports_precondition_violation() {
	let variables = vec![ExplicitVariable::new(
		2,
		"v".into(),
		vec!["v0".into(), "v1".into()],
		-1,
		0,
	)];
	let numeric_variables: Vec<NumericVariable> = vec![];
	let goals = vec![Fact::new(0, 1)];
	let op = Operator::new(
		"set".into(),
		vec![Fact::new(0, 0)],
		vec![planners_sas::numeric::numeric_task::Effect::new(vec![], 0, 0, 1)],
		vec![],
		1,
	);
	let task = NumericRootTask::new(
		4,
		Metric::new(true, -1),
		variables,
		numeric_variables,
		goals,
		vec![],
		vec![0],
		vec![],
		vec![op],
		vec![],
		vec![],
		vec![],
		(0, 0),
	);

	let (domain_mapping, domain_sizes) = identity_domain_mapping_and_sizes(&task).unwrap();
	let partitions = NumericPartitions::trivial(&task);
	let numeric_domain_sizes: Vec<usize> = vec![];
	let factory = DomainAbstractionFactory::new(
		&task,
		domain_mapping,
		domain_sizes,
		partitions,
		numeric_domain_sizes,
	)
	.unwrap();
	let plan = factory
		.compute_wildcard_plan(&task, true, false)
		.unwrap()
		.expect("plan exists");

	// Make the stored wildcard plan invalid in the concrete initial state.
	task.set_initial_propositional_state_values(vec![1]);

	let cegar = Cegar::new(CegarConfig::default()).unwrap();
	let flaws = cegar.get_flaws(&task, factory.partitions(), &plan, false).unwrap();
	assert_eq!(flaws.len(), 1);
	match &flaws[0] {
		Flaw::Propositional(pf) => assert_eq!(pf.fact, Fact::new(0, 0)),
		_ => panic!("expected propositional flaw"),
	}
}

#[test]
fn get_flaws_reports_numeric_deviation_flaw() {
	use crate::numeric::evaluation::domain_abstractions::comparison_expression::Interval;
	use planners_sas::numeric::axioms::{ComparisonAxiom, ComparisonOperator};
	use planners_sas::numeric::numeric_task::{
		AssignmentEffect, AssignmentOperation, NumericType,
	};

	// Propositional vars: gt (comparison result), g (goal flag)
	let variables = vec![
		ExplicitVariable::new(
			3,
			"gt".into(),
			vec!["true".into(), "false".into(), "unknown".into()],
			0,
			2,
		),
		ExplicitVariable::new(2, "g".into(), vec!["g0".into(), "g1".into()], -1, 0),
	];
	let numeric_variables = vec![
		NumericVariable::new("x".into(), NumericType::Regular, -1),
		NumericVariable::new("c".into(), NumericType::Constant, -1),
		NumericVariable::new("thresh".into(), NumericType::Constant, -1),
	];
	let comparison_axioms = vec![ComparisonAxiom::new(
		0,
		0,
		2,
		ComparisonOperator::GreaterThan,
	)];
	let op0 = Operator::new(
		"inc".into(),
		vec![],
		vec![],
		vec![AssignmentEffect::new(
			0,
			AssignmentOperation::Plus,
			1,
			false,
			vec![],
		)],
		1,
	);
	let op1 = Operator::new(
		"set_g".into(),
		vec![Fact::new(0, 0)],
		vec![planners_sas::numeric::numeric_task::Effect::new(vec![], 1, 0, 1)],
		vec![],
		1,
	);
	let task = NumericRootTask::new(
		4,
		Metric::new(true, -1),
		variables,
		numeric_variables,
		vec![Fact::new(1, 1)],
		vec![],
		vec![2, 0],
		vec![-10.0, 3.0, -5.0],
		vec![op0, op1],
		vec![],
		comparison_axioms,
		vec![],
		(0, 0),
	);

	let partitions = NumericPartitions::with_partitions(vec![
		vec![
			Interval::new(f64::NEG_INFINITY, -5.0, false, true),
			Interval::new(-5.0, f64::INFINITY, false, false),
		],
		vec![Interval::singleton(3.0)],
		vec![Interval::singleton(-5.0)],
	]);

	// Hand-constructed wildcard plan:
	// - step 0 applies op0 (inc)
	// - the abstract plan (optimistically) expects x to end up in the UPPER partition (index 1)
	let plan = WildcardPlanResult {
		wildcard_plan: vec![vec![0]],
		abstract_state_hashes: vec![],
		abstract_prop_states: vec![],
		abstract_numeric_states: vec![
			vec![0, 0, 0], // initial: x in LOWER
			vec![1, 0, 0], // expected after inc: x in UPPER
		],
	};

	let cegar = Cegar::new(CegarConfig::default()).unwrap();
	let flaws = cegar.get_flaws(&task, &partitions, &plan, false).unwrap();
	assert!(
		flaws.iter().any(|f| matches!(f, Flaw::Numeric(_))),
		"expected a numeric deviation flaw"
	);
}
