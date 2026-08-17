; The written metric prefers `take-time-route`: 10 * 0 + 2 = 2 instead of
; 10 * 1 + 0 = 10. Silently substituting total-cost reverses that choice,
; because the two total-cost increments are respectively 100 and 1.
(define (domain weighted-sum-metric)
  (:requirements :strips :numeric-fluents :action-costs)
  (:predicates (done))
  (:functions
    (fuel)
    (time)
    (total-cost))

  (:action take-fuel-route
    :parameters ()
    :precondition (not (done))
    :effect (and
      (done)
      (increase (fuel) 1)
      (increase (total-cost) 1)))

  (:action take-time-route
    :parameters ()
    :precondition (not (done))
    :effect (and
      (done)
      (increase (time) 2)
      (increase (total-cost) 100))))
