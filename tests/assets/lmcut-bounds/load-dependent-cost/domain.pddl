; An action whose cost depends on a fluent that can only grow: delivering
; needs a load of at least 3 and costs 1 + load, and loading more raises the
; load. With numeric bounds, LM-cut prices deliver at 1 + the lower bound of
; load before it, its initial value 3.
(define (domain load-dependent-cost)
  (:requirements :strips :numeric-fluents :action-costs)
  (:predicates (delivered))
  (:functions (load) (total-cost))
  (:action deliver
    :parameters ()
    :precondition (>= (load) 3)
    :effect (and (delivered) (increase (total-cost) (+ 1 (load)))))
  (:action load-more
    :parameters ()
    :precondition ()
    :effect (and (increase (load) 1) (increase (total-cost) 1))))
