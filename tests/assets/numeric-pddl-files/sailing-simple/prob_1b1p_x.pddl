;; 1 boat, 1 person, on the x-axis. h* = 11 (ten go_east + one save_person;
;; all actions unit cost).
;; Task-1 workhorse: alpha1 = fine on x in [0,5] / coarse [5,inf);
;;                   alpha2 = fine on x in [5,10] / coarse (-inf,5].
;; Task-2 workhorse: regression/target-centered CEGAR must build alpha2.
(define (problem sailing-simple-1b1p-x)
    (:domain sailing-simple)
    (:objects
        b0 - boat
        p0 - person
    )
    (:init
        (= (x b0) 0)
        (= (y b0) 0)
        (= (tx p0) 10)
        (= (ty p0) 0)
    )
    (:goal (and (saved p0)))
)
