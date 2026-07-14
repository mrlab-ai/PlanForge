;; 2 boats, 2 persons on a line. b0 at x=0, b1 at x=30; p0 at x=10, p1 at x=40.
;; h* = 22: each boat saves its near person independently (10 moves + save) twice.
(define (problem sailing-simple-2b2p-x)
    (:domain sailing-simple)
    (:objects
        b0 b1 - boat
        p0 p1 - person
    )
    (:init
        (= (x b0) 0)
        (= (y b0) 0)
        (= (x b1) 30)
        (= (y b1) 0)
        (= (tx p0) 10)
        (= (ty p0) 0)
        (= (tx p1) 40)
        (= (ty p1) 0)
    )
    (:goal (and (saved p0) (saved p1)))
)
