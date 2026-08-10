;; A minimal domain whose optimum depends on the *strictness* of a numeric
;; comparison. `launch` needs strictly more than two units of charge; `charge`
;; adds one unit. Compiling `>` as `>=` makes the initial state already
;; launchable and drops the optimum by one action.
(define (domain strict-comparison)
  (:requirements :strips :fluents)
  (:predicates (launched))
  (:functions (charge))

  (:action charge
    :parameters ()
    :effect (increase (charge) 1))

  (:action launch
    :parameters ()
    :precondition (> (charge) 2)
    :effect (launched)))
