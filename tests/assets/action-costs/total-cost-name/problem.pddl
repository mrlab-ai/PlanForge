; Optimum 8: step-a at 5 then step-b at 3. Two steps of different cost, so a
; scale factor cannot be mistaken for a plan-length effect.
;
; Failure mode this detects: the designated cost function counted twice, once as
; the operator cost and once as an ordinary numeric effect on the same fluent.
; The name matters. `total-cost` is special-cased in the translator and `cost` is
; not, so the two take different paths and only this fixture covers this one.
;
;   correct optimum   8
;   counted twice     16
(define (problem total-cost-name-1)
  (:domain action-costs-total-cost-name)
  (:init (= (total-cost) 0))
  (:goal (goal))
  (:metric minimize (total-cost)))
