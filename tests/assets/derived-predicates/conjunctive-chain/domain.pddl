;; A chain of derived predicates whose bodies are conjunctions of the previous
;; link and one more ordinary fact.
;;
;; All three axioms are positive, which is deliberate: axiom layers are charged
;; for sign flips, so a positive chain sits on a *single* layer and has to be
;; closed by the fixpoint inside that layer, in dependency order, from a queue
;; that starts out holding only ordinary facts. `layered-chain` and
;; `negated-dependency` are the fixtures that produce several layers; this one
;; pins that one layer is closed transitively.
;;
;; Each link adds exactly one required ordinary fact, so a lost conjunct saves
;; exactly one action and a chain that is not closed transitively leaves
;; `stage3` unreachable.
(define (domain derived-conjunctive-chain)
  (:requirements :strips :typing :derived-predicates)
  (:types machine)
  (:predicates
    (bolted ?m - machine)
    (wired ?m - machine)
    (sealed ?m - machine)
    (stage1 ?m - machine)
    (stage2 ?m - machine)
    (stage3 ?m - machine)
    (shipped ?m - machine))

  (:derived (stage1 ?m - machine)
    (bolted ?m))

  (:derived (stage2 ?m - machine)
    (and (stage1 ?m) (wired ?m)))

  (:derived (stage3 ?m - machine)
    (and (stage2 ?m) (sealed ?m)))

  (:action bolt
    :parameters (?m - machine)
    :precondition (not (bolted ?m))
    :effect (bolted ?m))

  (:action wire
    :parameters (?m - machine)
    :precondition (not (wired ?m))
    :effect (wired ?m))

  (:action seal
    :parameters (?m - machine)
    :precondition (not (sealed ?m))
    :effect (sealed ?m))

  (:action ship
    :parameters (?m - machine)
    :precondition (and (stage3 ?m) (not (shipped ?m)))
    :effect (shipped ?m)))
