; Optimum 2: grab a key, then open-with-any.
;
; Failure mode this detects: eliminating the existential moves its variables
; into the action's parameter list without raising num_external_parameters,
; which is what Fast Downward does too. Grounding then rejects every parameter
; tuple whose length exceeds that count, so open-with-any disappears and the
; only route left is force-open.
;
;   correct optimum        2
;   action dropped         9
(define (problem exists-precondition-1)
  (:domain adl-exists-precondition)
  (:objects k1 k2 - key)
  (:init (= (cost) 0))
  (:goal (opened))
  (:metric minimize (cost)))
