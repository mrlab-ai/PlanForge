(define (problem BLOCKS-3-REVERSE)
  (:domain BLOCKS)
  (:objects A B C)
  (:init
    (on A B)
    (on B C)
    (ontable C)
    (clear A)
    (handempty))
  (:goal (and
    (on C B)
    (on B A))))
