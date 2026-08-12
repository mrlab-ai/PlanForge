; One dirty item. Optimum 2: clean it, then finish.
;
; Failure mode this detects: the `not` arm of the condition parser handled only
; a negated literal or comparison, so `(not (exists ...))` was read as a literal
; whose predicate is "exists" and whose first argument is a typed parameter
; list, and the parse crashed.
;
;   correct optimum   2
(define (problem negated-quantifier-1)
  (:domain adl-negated-quantifier)
  (:objects i1 - item)
  (:init (dirty i1) (= (cost) 0))
  (:goal (done))
  (:metric minimize (cost)))
