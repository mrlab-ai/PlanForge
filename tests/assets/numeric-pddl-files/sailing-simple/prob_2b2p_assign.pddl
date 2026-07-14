;; 2 boats, 2 persons on a line. b0 at x=0, b1 at x=100; persons at x=5 and x=-5.
;; h* = 17: b0 saves both persons (5 moves + save + 10 moves + save);
;; b1 is a decoy at least 95 moves away from either person.
(define (problem sailing-simple-2b2p-assign)
    (:domain sailing-simple)
    (:objects
        b0 b1 - boat
        p0 p1 - person
    )
    (:init
        (= (x b0) 0)
        (= (y b0) 0)
        (= (x b1) 100)
        (= (y b1) 0)
        (= (tx p0) 5)
        (= (ty p0) 0)
        (= (tx p1) -5)
        (= (ty p1) 0)
    )
    (:goal (and (saved p0) (saved p1)))
)
