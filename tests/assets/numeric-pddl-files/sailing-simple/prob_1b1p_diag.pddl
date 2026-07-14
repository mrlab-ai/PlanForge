;; 1 boat, 1 person on the main diagonal. h* = 11
;; (10 moves: go_north_east x10 at +0.5/+0.5, or 5 go_east + 5 go_north;
;; plus one unit-cost save_person).
;; Exercises operators that change BOTH x and y, and the "one interval is
;; infinite" footprint case: an x-only abstraction sees go_north_east as a
;; +0.5 x move with a y self-loop on the unbounded y interval.
;; Also the natural place where the u = x + y root becomes the useful variable.
(define (problem sailing-simple-1b1p-diag)
    (:domain sailing-simple)
    (:objects
        b0 - boat
        p0 - person
    )
    (:init
        (= (x b0) 0)
        (= (y b0) 0)
        (= (tx p0) 5)
        (= (ty p0) 5)
    )
    (:goal (and (saved p0)))
)
