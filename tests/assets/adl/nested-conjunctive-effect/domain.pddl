(define (domain adl-nested-conjunctive-effect)
  (:requirements :adl :typing :action-costs)
  (:types item)
  (:predicates (marked ?i - item) (spare ?i - item) (swept))
  (:functions (cost))
  ; A conjunction nested inside forall+when, which is the shape ADL domains use
  ; constantly and which no earlier fixture had: the quantified effect was always
  ; a single atom.
  (:action sweep :parameters ()
    :precondition ()
    :effect (and (swept)
                 (forall (?i - item)
                   (when (spare ?i)
                     (and (marked ?i) (not (spare ?i)))))
                 (increase (cost) 1)))
  (:action mark-one :parameters (?i - item)
    :precondition () :effect (and (marked ?i) (increase (cost) 4))))
