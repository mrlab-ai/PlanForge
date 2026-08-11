;; The vault starts breached, so `(alarm v1)` holds in the initial state and the
;; goal has to switch it off again:
;;
;;     seal v1, deliver-sealed v1, defuse v1                   -> 3 actions
;;
;; Delivery is independent of the breach and its cheapest route is `seal` plus
;; `deliver-sealed`; `defuse` is the only way to stop proving `(alarm v1)`.
;;
;; Each way of getting a negated derived goal wrong lands on a different cost:
;;
;;   * substituting the body of `alarm <- breach` for the goal - the bug this
;;     fixture exists for - makes the abstraction ask for `(breach v1)` at the
;;     goal. `arm` needs the vault unsealed, so every sealed goal state gets
;;     h = infinity and is pruned; the surviving goal state is the unsealed one,
;;     reached by `crate`, `haul`, `deliver-open` and `defuse`, cost 4;
;;   * losing the goal instead - dropping the axiom, so that nothing ever proves
;;     `(alarm v1)` and the goal holds in the initial state - drops `defuse`,
;;     cost 2;
;;   * a `true` axiom default makes `(not (alarm v1))` unachievable: no plan.
(define (problem derived-negated-goal-1)
  (:domain derived-negated-goal)
  (:objects v1 - vault)
  (:init (breach v1))
  (:goal (and (not (alarm v1)) (delivered v1))))
