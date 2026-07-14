;; 1 boat, 2 persons on the main diagonal at (5,5) and (10,10). h* = 22
;; (10 go_north_east to (5,5), save p0, 10 more to (10,10), save p1:
;; 20 moves + 2 unit-cost saves).
;; Additive CP across persons via residual segments: the leg for p0 charges
;; moves in the region reaching (5,5), the leg for p1 charges the residual
;; segment (5,5)->(10,10). This mirrors the real sailing prob_2_2 chain route.
(define (problem sailing-simple-1b2p-diag)
    (:domain sailing-simple)
    (:objects
        b0 - boat
        p0 p1 - person
    )
    (:init
        (= (x b0) 0)
        (= (y b0) 0)
        (= (tx p0) 5)
        (= (ty p0) 5)
        (= (tx p1) 10)
        (= (ty p1) 10)
    )
    (:goal (and (saved p0) (saved p1)))
)
