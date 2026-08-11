;; A *negated* derived goal: the task asks for `(not (alarm ?v))`, which is the
;; derived variable's axiom default rather than the value its rule proves.
;;
;; Every other fixture here - and every benchmark in `assets/numeric-pddl-files`
;; - has only positive derived goals, and for those the two possible readings of
;; a goal at a derived variable coincide closely enough to hide a bug: the body
;; of the rule proving the goal is a necessary condition for it, so an
;; abstraction may replace the goal by that body and stay admissible. At the
;; default value the two readings are *opposites*. `(not (alarm ?v))` holds
;; exactly when nothing proves `(alarm ?v)`, so substituting the body of
;; `alarm <- breach` turns "no breach" into "a breach", and an abstraction whose
;; goal is keyed by the derived *variable* instead of the derived *fact* measures
;; the distance to the negation of what was asked for.
;;
;; The domain is built so that this substitution is not merely imprecise but
;; unsound, and shows up as a *cost*:
;;
;;   * `arm` can only reopen a breach while the vault is unsealed, so from a
;;     sealed goal state the substituted goal `(breach ?v)` is unreachable and
;;     the abstraction reports h = infinity - the cheapest plan is pruned as a
;;     dead end;
;;   * the unsealed route to `(delivered ?v)` needs `crate`, `haul` and
;;     `deliver-open` instead of `seal` and `deliver-sealed`, so the goal state
;;     that survives the substitution costs one action more.
;;
;; The body is a *single* literal on purpose. `(not (alarm ?v))` is then
;; equivalent to `(not (breach ?v))`, so the negated goal has an exact
;; restatement over ordinary variables and a correct abstraction stays as
;; informative as it was; a two-literal body would only imply a disjunction and
;; force the goal to be dropped, which is admissible but says nothing.
;;
;; No action reads `(alarm ?v)` as a precondition, which is what `goal-condition`
;; covers and what this fixture deliberately does without: a domain abstraction
;; has no propositional axioms, so a derived precondition makes the whole
;; `domain_abstraction` / `canonical(domain())` family refuse to build - that is
;; why the other eight fixtures are unsolvable under it. This fixture has to be
;; *solvable* there, because that is where the goal substitution happens.
;;
;; `haul` requires the breach it hauls through. That is what puts `breach` into
;; the causal predecessors of `delivered`, and therefore into a pattern-database
;; pattern, which is the only way the projected-task goal walk can be reached by
;; a negated derived goal at all.
(define (domain derived-negated-goal)
  (:requirements :strips :typing :derived-predicates)
  (:types vault)
  (:predicates
    (breach ?v - vault)
    (sealed ?v - vault)
    (crated ?v - vault)
    (hauled ?v - vault)
    (delivered ?v - vault)
    (alarm ?v - vault))

  (:derived (alarm ?v - vault)
    (breach ?v))

  (:action defuse
    :parameters (?v - vault)
    :precondition (breach ?v)
    :effect (not (breach ?v)))

  ;; A breach can only be reopened while the vault is unsealed. This is the
  ;; asymmetry the fixture is built on: sealing is irreversible, so a sealed
  ;; state can never satisfy `(breach ?v)` again.
  (:action arm
    :parameters (?v - vault)
    :precondition (and (not (breach ?v)) (not (sealed ?v)))
    :effect (breach ?v))

  (:action seal
    :parameters (?v - vault)
    :precondition (not (sealed ?v))
    :effect (sealed ?v))

  (:action deliver-sealed
    :parameters (?v - vault)
    :precondition (and (sealed ?v) (not (delivered ?v)))
    :effect (delivered ?v))

  (:action crate
    :parameters (?v - vault)
    :precondition (not (crated ?v))
    :effect (crated ?v))

  ;; Hauled out through the breach, so this route needs the breach open.
  (:action haul
    :parameters (?v - vault)
    :precondition (and (crated ?v) (breach ?v) (not (hauled ?v)))
    :effect (hauled ?v))

  (:action deliver-open
    :parameters (?v - vault)
    :precondition (and (hauled ?v) (not (delivered ?v)))
    :effect (delivered ?v)))
