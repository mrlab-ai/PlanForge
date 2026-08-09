(define (problem BLOCKS-2-REVERSE)
  (:domain BLOCKS)
  (:objects A B)
  (:init
    (on A B)
    (ontable B)
    (clear A)
    (handempty))
  (:goal (and
    (on B A))))
