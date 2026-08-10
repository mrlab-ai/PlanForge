;; A minimal domain whose optimum depends on a *guarded* numeric effect.
;;
;; `move` always burns one unit of fuel and burns a second one only while
;; `boosted` holds. Dropping the guard - i.e. applying the second `decrease`
;; unconditionally, or ignoring it - changes the optimal cost, so this domain
;; distinguishes a correct pipeline from either failure mode.
(define (domain conditional-numeric-effect)
  (:requirements :strips :typing :fluents :conditional-effects)
  (:types location)
  (:predicates
    (at ?l - location)
    (connected ?from - location ?to - location)
    (boosted))
  (:functions (fuel))

  ;; Needs two units in the tank, burns one, and burns one more while boosted.
  (:action move
    :parameters (?from - location ?to - location)
    :precondition (and (at ?from) (connected ?from ?to) (>= (fuel) 2))
    :effect (and (not (at ?from))
                 (at ?to)
                 (decrease (fuel) 1)
                 (when (boosted) (decrease (fuel) 1))))

  (:action unboost
    :parameters ()
    :precondition (boosted)
    :effect (not (boosted))))
