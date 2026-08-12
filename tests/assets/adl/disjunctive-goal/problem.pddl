; Optimum 2: satisfy the cheaper disjunct with make-q.
;
; Failure mode this detects: if substitute_complicated_goal hides the
; disjunction behind an axiom and no later pass splits that axiom's body, the
; goal atom is never proved and the disjunction is treated as satisfied. The
; planner then reports the empty plan, cost 0.
;
;   correct optimum   2
;   goal dropped      0
(define (problem disjunctive-goal-1)
  (:domain adl-disjunctive-goal)
  (:init (= (cost) 0))
  (:goal (or (p) (q)))
  (:metric minimize (cost)))
