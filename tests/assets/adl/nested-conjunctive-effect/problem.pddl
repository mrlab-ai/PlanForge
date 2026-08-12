; Optimum 1: one sweep marks both spare items and clears their spare flag.
;
; Failure mode this detects: parse_effect had no `and` arm, so a conjunction
; inside a `when` or `forall` was read as a predicate named "and" and the parse
; crashed on the first part that was not an atom. Every real ADL domain hits it.
;
;   correct optimum   1
(define (problem nested-conjunctive-effect-1)
  (:domain adl-nested-conjunctive-effect)
  (:objects i1 i2 - item)
  (:init (spare i1) (spare i2) (= (cost) 0))
  (:goal (and (swept) (marked i1) (marked i2) (not (spare i1))))
  (:metric minimize (cost)))
