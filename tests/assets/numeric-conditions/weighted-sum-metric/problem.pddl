(define (problem weighted-sum-metric-problem)
  (:domain weighted-sum-metric)
  (:requirements :strips :numeric-fluents :action-costs)
  (:init
    (= (fuel) 0)
    (= (time) 0)
    (= (total-cost) 0))
  (:goal (done))
  (:metric minimize (+ (* 10 (fuel)) (time))))
