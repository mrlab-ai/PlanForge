;; Three units of fuel, boosted, and two moves to make.
;;
;; Boosted moves cost two units, so `move l0 l1` leaves one unit and the second
;; move is no longer applicable. The only plan is to unboost first:
;;
;;     unboost, move l0 l1, move l1 l2      -> 3 actions
;;
;; If the `when (boosted)` guard were dropped from the second `decrease`, both
;; moves would cost one unit each and `move l0 l1, move l1 l2` would solve the
;; task in 2 actions. If the guard were applied unconditionally, no plan of any
;; length would exist. The pinned optimum of 3 therefore fails in both
;; directions.
(define (problem conditional-numeric-effect-1)
  (:domain conditional-numeric-effect)
  (:objects l0 l1 l2 - location)
  (:init
    (at l0)
    (boosted)
    (connected l0 l1)
    (connected l1 l2)
    (= (fuel) 3))
  (:goal (at l2)))
