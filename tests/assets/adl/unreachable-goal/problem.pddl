; `(never)` holds in no state and nothing makes it true, so `make-q` is
; unreachable and `(q)` can never hold. The task has no plan.
;
; Failure mode this detects: a goal fact with no SAS variable used to abort the
; translation. The goal is the one condition site instantiation never visits, so
; it is the only place an unreachable atom reaches SAS translation, and the right
; answer is a trivially unsolvable task. Fast Downward answers the same way, with
; a task of zero operators.
;
;   correct behaviour   no solution reported, and no crash
(define (problem unreachable-goal-1)
  (:domain adl-unreachable-goal)
  (:init (= (cost) 0))
  (:goal (and (p) (q)))
  (:metric minimize (cost)))
