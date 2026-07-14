;; 1 boat, 2 persons on the x-axis at 10 and 15 (nested/prefix targets). h* = 17
;; (sail east to 10, save p0, continue to 15, save p1: 15 moves + 2 saves,
;; all actions unit cost).
;; Overlap reassurance case: build alpha1 = fine on [0,10] (for p0, alone h=11)
;; and alpha2 = fine on [0,15] (for p1, alone h=16). They overlap heavily on
;; [0,10], yet region/transition SCP recovers 16 (alpha1 first) or 17 = h*
;; (alpha2 first; p0's save is a distinct label and survives), because each
;; concrete transition is charged at most once via the residual.
;; A naive independent sum would be 27 (inadmissible).
(define (problem sailing-simple-1b2p-x)
    (:domain sailing-simple)
    (:objects
        b0 - boat
        p0 p1 - person
    )
    (:init
        (= (x b0) 0)
        (= (y b0) 0)
        (= (tx p0) 10)
        (= (ty p0) 0)
        (= (tx p1) 15)
        (= (ty p1) 0)
    )
    (:goal (and (saved p0) (saved p1)))
)
