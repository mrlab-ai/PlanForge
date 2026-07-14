;; 1 boat, 4 persons at the tips of the axes: E(10,0), W(-10,0), N(0,10), S(0,-10).
;; h* = 74 = 70 moves + 4 unit-cost saves. The boat must visit all four tips
;; from the centre; on a plus-shaped star with four rays of length 10 an optimal
;; open walk costs 2*10*3 + 10 = 70 moves (e.g. E->(10,0), across to
;; W(-10,0)=+20, up to N(0,10)=+20, down to S(0,-10)=+20).
;; This is the "8 abstractions" target: 4 persons x {progression, regression}
;; route abstractions, combined by region CP. It is also the
;; strict-dominance-over-LMc demonstrator: hmax/LM-cut-style counting evaluates
;; every target from the CURRENT position and cannot see return legs
;; (~ 4x10 moves + 4 saves = 44), while chain abstractions
;; (x * saved(pe) * saved(pw)) + (y * saved(pn) * saved(ps)) count the
;; round trips exactly (32 + 32 = 64 under region CP).
(define (problem sailing-simple-1b4p-axes)
    (:domain sailing-simple)
    (:objects
        b0 - boat
        pe pw pn ps - person
    )
    (:init
        (= (x b0) 0)
        (= (y b0) 0)
        (= (tx pe) 10)
        (= (ty pe) 0)
        (= (tx pw) -10)
        (= (ty pw) 0)
        (= (tx pn) 0)
        (= (ty pn) 10)
        (= (tx ps) 0)
        (= (ty ps) -10)
    )
    (:goal (and (saved pe) (saved pw) (saved pn) (saved ps)))
)
