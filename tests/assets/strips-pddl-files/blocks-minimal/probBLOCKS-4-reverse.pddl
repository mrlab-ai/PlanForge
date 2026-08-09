(define (problem BLOCKS-4-REVERSE)
  (:domain BLOCKS)
  (:objects A B C D)
  (:init
    (on A B)
    (on B C)
    (on C D)
    (ontable D)
    (clear A)
    (handempty))
  (:goal (and
    (on D C)
    (on C B)
    (on B A))))
