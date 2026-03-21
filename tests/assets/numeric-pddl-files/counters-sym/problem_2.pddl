;; Enrico Scala (enricos83@gmail.com) and Miquel Ramirez (miquel.ramirez@gmail.com)
(define (problem instance_2)
  (:domain fn-counters-inv)
  (:objects
    c0 c1 - counter
  )

  (:init
    (= (value c0) 2)
    (= (value c1) 0)
    (= (max_int) 4)
    (= (cost) 0)
  )

  (:goal (and 
(<= 7 (+ (value c0) (value c1)))
  ))

  (:metric minimize (cost))
  

  
)
