;; 1 boat, 1 person far away on the x-axis. h* = 101 (100 go_east + 1 save;
;; all actions unit cost).
;; sailing-ipc scale proxy: a deep 1-D chain abstraction (~100 layers). Stresses
;; (a) the >64-overlap fallback in max_overlap_reduction, (b) the
;; MAX_ABSTRACT_OPERATOR_REDUCTION_PIECES hull fallback (hull-merging destroys
;; the piece disjointness additivity depends on), and (c) the construction
;; budget: abstraction build must stay at millisecond scale with O(#layers)
;; states, or we reproduce the published sailing-ipc failure mode (DAs time out
;; where LMc's arithmetic counting is instant).
(define (problem sailing-simple-1b1p-far)
    (:domain sailing-simple)
    (:objects
        b0 - boat
        p0 - person
    )
    (:init
        (= (x b0) 0)
        (= (y b0) 0)
        (= (tx p0) 100)
        (= (ty p0) 0)
    )
    (:goal (and (saved p0)))
)
