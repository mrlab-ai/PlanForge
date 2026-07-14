;; Simplified sailing: x/y coordinates, exact-position rescues.
;;
;; A workhorse for reasoning about numeric domain abstractions and cost
;; partitioning. Unlike the original sailing domain, save_person requires the
;; boat to match the person's target position *exactly* (equality on both
;; axes), and moves are unit along the axes / 0.5 along the diagonals. This
;; keeps every optimal cost hand-computable while preserving the sailing
;; structure (a boat accumulating additive moves to reach rescue targets).
;;
;; tx/ty are per-person target functions, set in :init and never modified, so
;; they behave as constants (singleton numeric domains) exactly like d(?t) in
;; the original sailing domain.
(define (domain sailing-simple)
    (:requirements :typing :numeric-fluents)
    (:types boat person - object)
    (:predicates
        (saved ?p - person)
    )
    (:functions
        (x ?b - boat)
        (y ?b - boat)
        (tx ?p - person)
        (ty ?p - person)
    )

    (:action go_east
        :parameters (?b - boat)
        :effect (increase (x ?b) 1))
    (:action go_west
        :parameters (?b - boat)
        :effect (decrease (x ?b) 1))
    (:action go_north
        :parameters (?b - boat)
        :effect (increase (y ?b) 1))
    (:action go_south
        :parameters (?b - boat)
        :effect (decrease (y ?b) 1))

    (:action go_north_east
        :parameters (?b - boat)
        :effect (and (increase (x ?b) 0.5) (increase (y ?b) 0.5)))
    (:action go_north_west
        :parameters (?b - boat)
        :effect (and (decrease (x ?b) 0.5) (increase (y ?b) 0.5)))
    (:action go_south_east
        :parameters (?b - boat)
        :effect (and (increase (x ?b) 0.5) (decrease (y ?b) 0.5)))
    (:action go_south_west
        :parameters (?b - boat)
        :effect (and (decrease (x ?b) 0.5) (decrease (y ?b) 0.5)))

    (:action save_person
        :parameters (?b - boat ?t - person)
        :precondition (and (= (x ?b) (tx ?t)) (= (y ?b) (ty ?t)))
        :effect (saved ?t))
)
