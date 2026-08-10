;; Charge starts exactly on the boundary of the `> 2` guard.
;;
;;     charge, launch    -> 2 actions
;;
;; Reading `> 2` as `>= 2` would make `launch` applicable in the initial state
;; and the optimum would drop to 1.
(define (problem strict-comparison-1)
  (:domain strict-comparison)
  (:init (= (charge) 2))
  (:goal (launched)))
