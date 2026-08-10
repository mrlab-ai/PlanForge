;; A derived predicate whose body is a numeric comparison over a compound
;; expression, feeding a second, purely propositional derived predicate.
;;
;; This is the shape our port has and mainline Fast Downward does not, and it is
;; the one that ties the three axiom blocks together in a single task:
;;
;;   * `(+ (level ?b) (boost ?b))` is normalised into a derived function with an
;;     arithmetic assignment axiom, which occupies the lowest axiom layers;
;;   * `(>= ... 3)` becomes a three-valued comparison variable on the layer
;;     directly above the last arithmetic layer;
;;   * `charged` and `ready` are propositional derived variables, which the
;;     translator shifts above the comparison layer.
;;
;; So the fixture fails if the comparison is evaluated before the sum it reads,
;; or if `charged` is evaluated before its comparison, as well as if the
;; comparison itself is wrong.
(define (domain derived-numeric-body)
  (:requirements :strips :typing :fluents :derived-predicates)
  (:types battery)
  (:predicates
    (armed ?b - battery)
    (overdriven ?b - battery)
    (charged ?b - battery)
    (ready ?b - battery)
    (launched ?b - battery))
  (:functions
    (level ?b - battery)
    (boost ?b - battery))

  (:derived (charged ?b - battery)
    (>= (+ (level ?b) (boost ?b)) 3))

  (:derived (ready ?b - battery)
    (and (charged ?b) (armed ?b)))

  (:action recharge
    :parameters (?b - battery)
    :effect (increase (level ?b) 1))

  ;; Worth two `recharge`s, and available once, so the optimum uses it.
  (:action overdrive
    :parameters (?b - battery)
    :precondition (not (overdriven ?b))
    :effect (and (overdriven ?b) (increase (boost ?b) 2)))

  (:action arm
    :parameters (?b - battery)
    :precondition (not (armed ?b))
    :effect (armed ?b))

  (:action launch
    :parameters (?b - battery)
    :precondition (and (ready ?b) (not (launched ?b)))
    :effect (launched ?b)))
