; Optimum 2: make-b proves (ok) through the cheaper disjunct.
;
; Failure mode this detects: split_disjunctions only rewrites action
; preconditions, so an or inside a :derived body survives to grounding and the
; derived atom is treated as unconditionally true. The empty plan then solves it.
;
; Note that two :derived blocks with the same head do work, because several
; bodies for one head already act as a disjunction. It is one body containing
; or that is mishandled, which is why this fixture uses that form.
;
;   correct optimum      2
;   axiom always true    0
(define (problem or-in-axiom-body-1)
  (:domain adl-or-in-axiom-body)
  (:init (= (cost) 0))
  (:goal (ok))
  (:metric minimize (cost)))
