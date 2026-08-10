;; The smallest domain in which a `:derived` predicate is load-bearing: it is
;; read once as an operator precondition and once as a goal.
;;
;; `(ready ?s)` is never asserted by an action. It is proven by the axiom from
;; the two ordinary facts `(wired ?s)` and `(switched ?s)`, so each way of
;; getting derived predicates wrong shows up as a different optimum; the problem
;; file works the three costs out.
(define (domain derived-goal-condition)
  (:requirements :strips :typing :derived-predicates)
  (:types switch)
  (:predicates
    (wired ?s - switch)
    (switched ?s - switch)
    (ready ?s - switch)
    (checked ?s - switch))

  (:derived (ready ?s - switch)
    (and (wired ?s) (switched ?s)))

  (:action wire
    :parameters (?s - switch)
    :precondition (not (wired ?s))
    :effect (wired ?s))

  (:action switch-on
    :parameters (?s - switch)
    :precondition (not (switched ?s))
    :effect (switched ?s))

  ;; Reads the derived predicate as a precondition, so the fixture covers both
  ;; places a derived fact can be consumed.
  (:action check
    :parameters (?s - switch)
    :precondition (and (ready ?s) (not (checked ?s)))
    :effect (checked ?s)))
