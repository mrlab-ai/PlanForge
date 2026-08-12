; The same task with the cost function named `cost` rather than `total-cost`.
; Optimum 8 either way; the pair exists because the two names take different
; paths through the translator and only the pair pins both.
(define (problem cost-name-1)
  (:domain action-costs-cost-name)
  (:init (= (cost) 0))
  (:goal (goal))
  (:metric minimize (cost)))
