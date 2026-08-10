;; A directed 4-cycle n1->n2->n3->n4->n1, plus an isolated `n5` that only
;; `build-link n4 n5` can attach. The goal is to have *visited* both `n4` and
;; `n5`, and only `teleport` marks a node visited:
;;
;;     teleport n1 n4, build-link n4 n5, teleport n4 n5         -> 3 actions
;;
;; The cost pins the closure from both sides.
;;
;; Too little derived. `path(n1,n4)` is three hops away, so it only exists once
;; the fixpoint has derived `path(n3,n4)`, then `path(n2,n4)`, then
;; `path(n1,n4)`. Without the recursive axiom, or with the layer closed by a
;; single pass, the plan has to walk the graph instead:
;;
;;   * one-hop paths only -> walk n1 n2, walk n2 n3, teleport n3 n4,
;;     build-link n4 n5, teleport n4 n5, cost 5;
;;   * two-hop paths only -> walk n1 n2, teleport n2 n4, build-link n4 n5,
;;     teleport n4 n5, cost 4.
;;
;; Too much derived. `n5` has no incoming link until `build-link` runs, so no
;; `path(?a,n5)` may hold before then - and those are exactly the four variables
;; that form the cyclic component. If the derived atoms defaulted to true, or a
;; cyclic component were proven wholesale, `teleport n1 n4, teleport n4 n5` would
;; solve the task in 2 actions.
;;
;; The optimum also depends on the closure being recomputed per state: `n5`
;; becomes reachable only in the state after `build-link`.
(define (problem derived-recursive-closure-1)
  (:domain derived-recursive-closure)
  (:objects n1 n2 n3 n4 n5 - node)
  (:init
    (at n1)
    (link n1 n2)
    (link n2 n3)
    (link n3 n4)
    (link n4 n1)
    (buildable n4 n5))
  (:goal (and (visited n4) (visited n5))))
