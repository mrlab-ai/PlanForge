;; A chain of derived predicates that genuinely stratifies into three layers.
;;
;; A layer is charged only when an axiom reads a derived atom at the value that
;; atom *defaults* to, because that reading is negation by failure and is only
;; sound once the layer below has reached its fixpoint. Reading a derived atom at
;; the value an axiom can prove needs no layer of its own: the fixpoint inside
;; one layer already propagates it.
;;
;; That is why `check-dirty` and `check-flawed` are here. They are never part of
;; an optimal plan, but they read `dirty` and `flawed` *positively*, which makes
;; both atoms provable rather than only refutable, so both default to false and
;; the `(not (dirty ?m))` and `(not (flawed ?m))` readings above them each cost
;; a layer:
;;
;;     dirty  <  clean, flawed  <  perfect
;;
;; The chain is also arranged so that every layer has to be established by an
;; action. `flawed` needs `(not (dirty ?m))` as well, so the dust cannot simply
;; be left in place to refute it - the crack has to be welded.
(define (domain derived-layered-chain)
  (:requirements :strips :typing :derived-predicates)
  (:types machine)
  (:predicates
    (dusty ?m - machine)
    (washed ?m - machine)
    (cracked ?m - machine)
    (dirty ?m - machine)
    (clean ?m - machine)
    (flawed ?m - machine)
    (perfect ?m - machine)
    (checked-dirty ?m - machine)
    (checked-flawed ?m - machine)
    (shipped ?m - machine))

  (:derived (dirty ?m - machine)
    (dusty ?m))

  (:derived (clean ?m - machine)
    (and (washed ?m) (not (dirty ?m))))

  (:derived (flawed ?m - machine)
    (and (cracked ?m) (not (dirty ?m))))

  (:derived (perfect ?m - machine)
    (and (clean ?m) (not (flawed ?m))))

  (:action vacuum
    :parameters (?m - machine)
    :precondition (dusty ?m)
    :effect (not (dusty ?m)))

  (:action wash
    :parameters (?m - machine)
    :precondition (not (washed ?m))
    :effect (washed ?m))

  (:action weld
    :parameters (?m - machine)
    :precondition (cracked ?m)
    :effect (not (cracked ?m)))

  (:action ship
    :parameters (?m - machine)
    :precondition (and (perfect ?m) (not (shipped ?m)))
    :effect (shipped ?m))

  ;; The two positive readings that turn the negations above them into layers.
  (:action check-dirty
    :parameters (?m - machine)
    :precondition (and (dirty ?m) (not (checked-dirty ?m)))
    :effect (checked-dirty ?m))

  (:action check-flawed
    :parameters (?m - machine)
    :precondition (and (flawed ?m) (not (checked-flawed ?m)))
    :effect (checked-flawed ?m)))
