;; The battery starts one unit short of a single `overdrive` away from the
;; threshold: `level` is 1, `boost` is 0, and `charged` needs their sum to reach
;; 3.
;;
;;     overdrive b1, arm b1, launch b1                          -> 3 actions
;;
;; One `overdrive` adds 2, so it does the work of two `recharge`s, which is what
;; makes the cost read the whole chain:
;;
;;   * if the comparison only saw `level`, `overdrive` would be useless and two
;;     `recharge`s would be needed instead, cost 4;
;;   * if `>=` were compiled as `>`, the sum would have to reach 4 and an extra
;;     `recharge` would be needed, cost 4;
;;   * if `ready` lost its `armed` conjunct, `arm b1` would drop out, cost 2;
;;   * if the derived predicates defaulted to true, `launch b1` alone would
;;     solve the task, cost 1;
;;   * if `charged` were evaluated before its comparison - or the comparison
;;     before the sum - it would read the "unknown" default and there would be
;;     no plan.
(define (problem derived-numeric-body-1)
  (:domain derived-numeric-body)
  (:objects b1 - battery)
  (:init
    (= (level b1) 1)
    (= (boost b1) 0))
  (:goal (launched b1)))
