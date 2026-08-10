;; Both nodes start seeded, so both are `live` and therefore `charged`. One has to
;; be sealed, which needs it *not* charged, and the other has to be used, which
;; needs it charged:
;;
;;     remove-seed n1, seal n1, use n2                          -> 3 actions
;;
;; `charged` is only ever proven through the cycle - nothing asserts it - and it
;; is only refutable by removing the seed, so the cost reads both directions of
;; the cyclic component:
;;
;;   * if the refutation were exact but the seed ignored, `seal n1` would apply
;;     at once and the optimum would fall to 2;
;;   * if the component could not be refuted at all, `sealed n1` would be
;;     unreachable and there would be no plan;
;;   * if the component could not be proven, `used n2` would be unreachable and
;;     again there would be no plan.
;;
;; The blind optimum stays 3 whichever way the *negated* axioms are computed,
;; because the evaluator does not read them. What they decide is whether an
;; axiom-aware heuristic can see a plan at all, which is pinned separately.
(define (problem derived-cyclic-negation-1)
  (:domain derived-cyclic-negation)
  (:objects n1 n2 - node)
  (:init
    (seed n1)
    (seed n2))
  (:goal (and (sealed n1) (used n2))))
