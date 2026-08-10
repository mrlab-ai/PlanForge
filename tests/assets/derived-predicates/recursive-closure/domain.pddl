;; The transitive closure of a graph, which is the reason derived predicates
;; exist: `path` is recursive, so no finite set of ordinary operators computes
;; it.
;;
;; Three things about this domain are load-bearing.
;;
;; First, the recursion is *positive*, so every `path` atom sits on the same
;; axiom layer and the closure has to be reached by the fixpoint inside that
;; layer; a single pass over the rules derives only the one-hop paths.
;;
;; Second, on a cyclic graph the grounded axioms are cyclic too. The links of the
;; 4-cycle are static, so the paths between those four nodes come out of the
;; translator as unconditional rules and their cycles collapse. `link n4 n5` is
;; the one link an operator writes, and the paths into `n5` therefore keep the
;; cycle: `path(n4,n5)` supports `path(n1,n5)` supports `path(n2,n5)` supports
;; `path(n3,n5)` supports `path(n4,n5)` again - one strongly connected component
;; of four variables, one of them with two axioms.
;;
;; Third, `build-link` changes the graph, so the closure is not a property of the
;; task but of the state: a planner that computes the derived atoms once, or
;; inherits them from the predecessor state, gets a different optimum.
;;
;; `walk` takes one link at a time and `teleport` takes any path in one action,
;; so the optimum measures how much of the closure was actually derived.
(define (domain derived-recursive-closure)
  (:requirements :strips :typing :derived-predicates)
  (:types node)
  (:predicates
    (link ?a - node ?b - node)
    (buildable ?a - node ?b - node)
    (path ?a - node ?b - node)
    (at ?a - node)
    (visited ?a - node))

  (:derived (path ?a - node ?b - node)
    (link ?a ?b))

  (:derived (path ?a - node ?b - node)
    (exists (?via - node) (and (link ?a ?via) (path ?via ?b))))

  (:action walk
    :parameters (?a - node ?b - node)
    :precondition (and (at ?a) (link ?a ?b))
    :effect (and (not (at ?a)) (at ?b)))

  ;; The only action that marks a node visited, and the only consumer of `path`.
  (:action teleport
    :parameters (?a - node ?b - node)
    :precondition (and (at ?a) (path ?a ?b))
    :effect (and (not (at ?a)) (at ?b) (visited ?b)))

  ;; Restricted to the pairs listed as `buildable`, so the fixture stays small.
  (:action build-link
    :parameters (?a - node ?b - node)
    :precondition (and (buildable ?a ?b) (not (link ?a ?b)))
    :effect (link ?a ?b)))
