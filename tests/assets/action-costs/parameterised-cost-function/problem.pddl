; Optimum 8: move l1 to l2 costs (travel l1 l2) = 7, then finish costs 1.
;
; Two failure modes this detects, both hit by every IPC domain whose action cost
; is a static function of the action's parameters, `elevators` among them.
;
; First, uniquifying an action renamed its parameters, precondition and effects
; but not its cost, so the cost still referred to the old names, grounding bound
; the new ones, and the cost reached SAS translation as `(travel ?a ?b)` with no
; numeric variable to look up.
;
; Second, `(travel l1 l1)` has no value in the initial state, so `(move l1 l1)`
; must not be generated at all. Reachability now requires an action's cost
; function to be defined, which is what Fast Downward does.
;
;   correct optimum   8
(define (problem parameterised-cost-function-1)
  (:domain costfn)
  (:objects l1 l2 - loc)
  (:init (at l1) (= (total-cost) 0) (= (travel l1 l2) 7) (= (travel l2 l1) 3))
  (:goal (done))
  (:metric minimize (total-cost)))
