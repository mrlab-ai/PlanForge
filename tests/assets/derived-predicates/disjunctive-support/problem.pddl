;; `r1` has both a guard post and a lamp; `r2` has only a lamp, and no action
;; ever creates a post. So `r1` is cheapest to secure through the first axiom
;; (one action) and `r2` can only be secured through the second (two actions):
;;
;;     guard r1, enter r1, light r2, patrol r2, enter r2       -> 5 actions
;;
;; Both axioms are therefore needed for this optimum:
;;
;;   * without `(safe ?r) <- (guarded ?r)`, `r1` has to go the lit-and-patrolled
;;     way too and the optimum rises to 6;
;;   * without `(safe ?r) <- (lit ?r) and (patrolled ?r)`, `r2` cannot be made
;;     safe at all and there is no plan;
;;   * if `safe` defaulted to true, `enter r1, enter r2` would solve the task in
;;     2 actions.
(define (problem derived-disjunctive-support-1)
  (:domain derived-disjunctive-support)
  (:objects r1 r2 - room)
  (:init
    (has-post r1)
    (has-lamp r1)
    (has-lamp r2))
  (:goal (and (secured r1) (secured r2))))
