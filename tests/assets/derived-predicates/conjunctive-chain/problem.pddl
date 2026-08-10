;; One machine, nothing done to it yet. `ship` needs the far end of the chain,
;; which needs all three ordinary facts:
;;
;;     bolt m1, wire m1, seal m1, ship m1                      -> 4 actions
;;
;; The three setup actions are independent, so the optimum is exactly "one
;; action per required fact, plus the ship". That makes the cost count the body
;; conjuncts:
;;
;;   * a chain that is not closed transitively (`stage3` never derived, because
;;     `stage2` was still unproven when its rule was tried) leaves no plan;
;;   * dropping the ordinary conjunct of any link removes its setup action and
;;     the optimum falls to 3;
;;   * a `true` default for the derived predicates makes `ship m1` applicable at
;;     once, cost 1.
(define (problem derived-conjunctive-chain-1)
  (:domain derived-conjunctive-chain)
  (:objects m1 - machine)
  (:init)
  (:goal (shipped m1)))
