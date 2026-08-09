(define (problem BLOCKS-3-PRESERVE-MIDDLE)
  (:domain BLOCKS)
  (:objects A B C)
  (:init
    (on A B)
    (on B C)
    (ontable C)
    (clear A)
    (handempty))
  (:goal (and
    (on B C)
    (on C A))))
