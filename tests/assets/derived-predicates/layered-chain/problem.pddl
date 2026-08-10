;; One machine that starts dusty and cracked. `ship` needs `perfect`, at the top
;; of the chain, and each of the three layers below it has to be established:
;;
;;     vacuum m1, wash m1, weld m1, ship m1                    -> 4 actions
;;
;; Every setup action is necessary, and for a different reason:
;;
;;   * without `wash`, `clean` fails on its positive conjunct;
;;   * without `vacuum`, `dirty` holds and `clean` fails on its negative one;
;;   * without `weld`, `cracked` still holds and - because `vacuum` has already
;;     refuted `dirty` - `flawed` is derivable, so `perfect` fails.
;;
;; So each way of getting the layering wrong moves the cost:
;;
;;   * evaluating `perfect` before `flawed`'s layer has settled reads `flawed` at
;;     its unproven default, `weld` becomes unnecessary and the optimum falls
;;     to 3; the same holds one layer down for `clean` and `dirty`;
;;   * putting the whole chain on one layer does both, cost 2;
;;   * a `true` default for the derived predicates makes `ship m1` applicable at
;;     once, cost 1;
;;   * refusing to derive `clean` at all leaves no plan.
(define (problem derived-layered-chain-1)
  (:domain derived-layered-chain)
  (:objects m1 - machine)
  (:init
    (dusty m1)
    (cracked m1))
  (:goal (shipped m1)))
