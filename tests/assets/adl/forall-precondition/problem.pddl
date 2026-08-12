; Two items, one already clean. Optimum 2: clean i2, then finish.
;
; Failure mode this detects: if the pass that should replace forall(vars, phi)
; by not(new-axiom) rebuilds the quantifier instead, nothing downstream reads
; it and the precondition is treated as true. finish then applies immediately.
;
;   correct optimum        2
;   quantifier ignored     1
(define (problem forall-precondition-1)
  (:domain adl-forall-precondition)
  (:objects i1 i2 - item)
  (:init (clean i1) (= (cost) 0))
  (:goal (done))
  (:metric minimize (cost)))
