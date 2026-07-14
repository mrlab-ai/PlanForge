;; 2 boats, 1 person. b0 at origin, b1 far away at x=100. h* = 11
;; (b0 sails east 10 moves + one unit-cost save_person).
;; Multi-boat teleport / untracked-achiever admissibility limit:
;;   - A "b0-only + saved(p0)" abstraction lets save_person(b1,p0) fire for free
;;     (b1's position is projected away -> precondition optimistically true),
;;     collapsing h to ~1. Do NOT cost-partition across boats for one person.
;;   - The correct abstraction retains BOTH boats' x-root together so Dijkstra
;;     takes the cheapest boat's route (b0) and reports h = 10.
(define (problem sailing-simple-2b1p)
    (:domain sailing-simple)
    (:objects
        b0 b1 - boat
        p0 - person
    )
    (:init
        (= (x b0) 0)
        (= (y b0) 0)
        (= (x b1) 100)
        (= (y b1) 0)
        (= (tx p0) 10)
        (= (ty p0) 0)
    )
    (:goal (and (saved p0)))
)
