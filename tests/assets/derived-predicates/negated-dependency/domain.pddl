;; The minimal negative dependency between two derived predicates: `passable`
;; reads `(not (blocked ?r))`, and `blocked` is itself derived.
;;
;; This is the shape the translator handles by negation-by-failure. Because
;; `blocked` is only ever read negatively it is made to default to true, its
;; axiom is negated into `(not (blocked ?r)) <- (not (obstacle ?r))`, and it is
;; put on a lower layer than `passable` so that the layer below has settled
;; before `passable` reads it.
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
