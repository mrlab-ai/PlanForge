;; A derived predicate read *negatively* although it supports itself through a
;; cycle. This is the case mainline Fast Downward's issue453 is about, and the
;; one the previous per-variable negation got wrong.
;;
;; `live` and `charged` support each other, so they form one strongly connected
;; component of the dependency graph, and `seal` reads `(not (charged ?x))`.
;; Negating the definition of `charged` literal by literal gives
;;
;;     not charged(x) <- not live(x)      and      not live(x) <- not charged(x)
;;
;; on top of `not live(x) <- not seed(x)`: two rules whose bodies are each
;; other's heads, so in any relaxation neither is ever derivable and the negated
;; atom looks unreachable. issue453 detects the component instead and refutes
;; every variable in it unconditionally, which is a relaxation rather than a
;; falsehood.
;;
;; Both signs are needed, so the component cannot be pruned: `use` reads
;; `(charged ?x)` positively and `seal` reads it negatively.
;;
;; The optimum is the same either way, because the axiom evaluator refutes a
;; derived variable by finding it unproven at the end of its layer rather than by
;; applying these rules. The heuristics that do read them are where the
;; difference shows: with the circular negation, `lmcutnumeric` and `ff` call the
;; initial state a dead end and A* reports a solvable task unsolvable.
(define (domain derived-cyclic-negation)
  (:requirements :strips :typing :derived-predicates)
  (:types node)
  (:predicates
    (seed ?x - node)
    (live ?x - node)
    (charged ?x - node)
    (sealed ?x - node)
    (used ?x - node))

  (:derived (live ?x - node) (seed ?x))
  (:derived (live ?x - node) (charged ?x))
  (:derived (charged ?x - node) (live ?x))

  (:action remove-seed
    :parameters (?x - node)
    :precondition (seed ?x)
    :effect (not (seed ?x)))

  (:action seal
    :parameters (?x - node)
    :precondition (and (not (charged ?x)) (not (sealed ?x)))
    :effect (sealed ?x))

  (:action use
    :parameters (?x - node)
    :precondition (and (charged ?x) (not (used ?x)))
    :effect (used ?x)))
