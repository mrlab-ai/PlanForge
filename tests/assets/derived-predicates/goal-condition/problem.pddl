;; Two switches, so both consumers of `ready` are exercised: `s1` feeds the
;; precondition of `check`, `s2` is required by the goal directly.
;;
;;     wire s1, switch-on s1, check s1, wire s2, switch-on s2   -> 5 actions
;;
;; Nothing is shared between the two switches, so the optimum is exactly the
;; four setup actions plus the one `check`. Each way of getting derived
;; predicates wrong lands on a different cost: a `true` default makes `check s1`
;; applicable immediately and `(ready s2)` already satisfied (1), losing one of
;; the two body conjuncts saves one setup action per switch (3), and losing the
;; axiom altogether leaves `(ready s2)` unreachable (no plan).
(define (problem derived-goal-condition-1)
  (:domain derived-goal-condition)
  (:objects s1 s2 - switch)
  (:init)
  (:goal (and (checked s1) (ready s2))))
