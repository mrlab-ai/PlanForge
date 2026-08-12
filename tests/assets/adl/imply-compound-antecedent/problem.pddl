; p and q hold, r does not, so the implication is false and go is inapplicable
; until r is made true. Optimum 4: make-r then go.
;
; Failure mode this detects: (imply A B) must become (or (not A) B). If the
; antecedent is not negated the condition becomes (or (and (p) (q)) (r)), which
; holds in the initial state, and go applies at once.
;
;   correct optimum          4
;   antecedent not negated   1
(define (problem imply-compound-antecedent-1)
  (:domain adl-imply-compound-antecedent)
  (:init (p) (q) (= (cost) 0))
  (:goal (gone))
  (:metric minimize (cost)))
