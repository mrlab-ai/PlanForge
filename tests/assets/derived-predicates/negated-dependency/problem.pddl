;; The one road starts obstructed and unpaved, so both setup actions are
;; needed before it can be crossed:
;;
;;     pave r1, clear r1, cross r1                             -> 3 actions
;;
;; The cost is what makes the negative dependency testable. If `blocked` were
;; simply left false - the axiom dropped, or a false default that is never
;; re-derived - then `(not (blocked r1))` would hold in the initial state and
;; `pave r1, cross r1` would solve the task in 2 actions, which is exactly the
;; unsound direction: a planner that is wrong here reports a *cheaper* plan than
;; the task admits. If `blocked` were left true instead, `passable` would never
;; be derivable and there would be no plan at all.
(define (problem derived-negated-dependency-1)
  (:domain derived-negated-dependency)
  (:objects r1 - road)
  (:init (obstacle r1))
  (:goal (crossed r1)))
