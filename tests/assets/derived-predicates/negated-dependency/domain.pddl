;; The minimal negative dependency between two derived predicates: `passable`
;; reads `(not (blocked ?r))`, and `blocked` is itself derived.
;;
;; This is the shape the translator handles by negation-by-failure. `blocked`
;; defaults to false like every derived variable, and it is put on a strictly
;; lower layer than `passable`, so that "nothing proved `blocked`" has an answer
;; by the time `passable` asks. Because something reads it negatively, the
;; translator also emits the rule refuting it,
;; `(not (blocked ?r)) <- (not (obstacle ?r))`, which the evaluator ignores and
;; the axiom-reading heuristics relax over.
;;
;; The pre-issue453 scheme instead made a variable that was only ever read
;; negatively default to *true* and refuted it with generated axioms. Nothing
;; does that any more: `derived_predicate_tests` pins zero true defaults across
;; this whole corpus.
;;
;; `layered-chain` stacks three of these; this fixture keeps a single one so a
;; failure says which mechanism broke.
(define (domain derived-negated-dependency)
  (:requirements :strips :typing :derived-predicates)
  (:types road)
  (:predicates
    (obstacle ?r - road)
    (paved ?r - road)
    (blocked ?r - road)
    (passable ?r - road)
    (crossed ?r - road))

  (:derived (blocked ?r - road)
    (obstacle ?r))

  (:derived (passable ?r - road)
    (and (paved ?r) (not (blocked ?r))))

  (:action clear
    :parameters (?r - road)
    :precondition (obstacle ?r)
    :effect (not (obstacle ?r)))

  (:action pave
    :parameters (?r - road)
    :precondition (not (paved ?r))
    :effect (paved ?r))

  (:action cross
    :parameters (?r - road)
    :precondition (and (passable ?r) (not (crossed ?r)))
    :effect (crossed ?r)))
