;; One derived predicate supported by *two* axioms, which is how PDDL spells a
;; disjunctive body.
;;
;; A room is `safe` either because it is guarded, or because it is both lit and
;; patrolled. The two supports cost a different number of actions, and the
;; problem gives one room that can use the cheap support and one that cannot, so
;; both axioms are load-bearing: dropping either one changes the optimum.
;;
;; `axioms_by_atom` keeps a *list* per head, and `simplify` prunes only axioms
;; whose body is dominated by another. Neither body here dominates the other, so
;; a pipeline that keeps just one axiom per head fails this fixture.
(define (domain derived-disjunctive-support)
  (:requirements :strips :typing :derived-predicates)
  (:types room)
  (:predicates
    (has-post ?r - room)
    (has-lamp ?r - room)
    (guarded ?r - room)
    (lit ?r - room)
    (patrolled ?r - room)
    (safe ?r - room)
    (secured ?r - room))

  (:derived (safe ?r - room)
    (guarded ?r))

  (:derived (safe ?r - room)
    (and (lit ?r) (patrolled ?r)))

  (:action guard
    :parameters (?r - room)
    :precondition (and (has-post ?r) (not (guarded ?r)))
    :effect (guarded ?r))

  (:action light
    :parameters (?r - room)
    :precondition (and (has-lamp ?r) (not (lit ?r)))
    :effect (lit ?r))

  (:action patrol
    :parameters (?r - room)
    :precondition (not (patrolled ?r))
    :effect (patrolled ?r))

  (:action enter
    :parameters (?r - room)
    :precondition (and (safe ?r) (not (secured ?r)))
    :effect (secured ?r)))
