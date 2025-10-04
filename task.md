# Your task. 

Port python/translate/pddl to rust. We want a semantically equivalent version. Add comprehensive testing. 

1. Analyze the file dependencies and add them here. 
   
	Dependency analysis for `python/translate/pddl/` (direct, top-level imports):
	- `actions.py` -> imports: `conditions`, `effects`, `pddl_types`, `f_expression`
	- `axioms.py` -> imports: `conditions`, `predicates`, `f_expression`
	- `conditions.py` -> imports: `f_expression`You
	- `effects.py` -> imports: `conditions`, `f_expression`
	- `f_expression.py` -> imports: (self-contained; defines FunctionalExpression, PNEs)
	- `functions.py` -> imports: (none at top-level; defines Function)
	- `pddl_types.py` -> imports: `itertools` (utility/type helpers)
	- `predicates.py` -> imports: (none at top-level; defines Predicate)
	- `tasks.py` -> imports: `actions`, `axioms`, `conditions`, `predicates`, `pddl_types`, `functions`, `f_expression`

	Recommended local port order (minimize dependencies first):
	1. `f_expression.py` (low external deps, used by many modules)
	2. `functions.py`, `predicates.py`, `pddl_types.py` (small, self-contained)
	3. `conditions.py`, `effects.py` (depend on `f_expression`)
	4. `actions.py`, `axioms.py` (depend on conditions/effects/predicates)
	5. `tasks.py` (orchestration: depends on all of the above)

2. Recreate the folder/file structure. The target rust directory is src/translate/pddl.
3. Add tests to verify correctness. 
4. Report your progress here. 