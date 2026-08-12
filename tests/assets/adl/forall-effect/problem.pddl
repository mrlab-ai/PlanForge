; Three items. Optimum 1: mark-all marks every one of them at once.
;
; Failure mode this detects: a universally quantified effect keeps its
; quantified parameters in Effect::parameters, and if instantiation never binds
; them the effect atom keeps its variable and matches no ground fact. mark-all
; then achieves nothing and the only route is three mark-one steps.
;
; The expensive alternative is deliberate. Without it the failure would only
; show as unsolvability, and a fixture that distinguishes the two by cost is
; stronger.
;
;   correct optimum          1
;   quantified effect lost   12
(define (problem forall-effect-1)
  (:domain adl-forall-effect)
  (:objects i1 i2 i3 - item)
  (:init (= (cost) 0))
  (:goal (and (marked i1) (marked i2) (marked i3)))
  (:metric minimize (cost)))
