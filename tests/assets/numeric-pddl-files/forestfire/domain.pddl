;; Alexander Shleyfman (shleyfman.alexander@gmail.com)
(define (domain forestfire)
  (:types location locatable - object
          bushes grass - location
          bot axe water - locatable
  )
   (:predicates (at ?o - locatable ?x - location)
                (connected ?x - location ?y - location)
		 (has ?r - bot ?ax - axe)
		 (pond ?g - grass)
)
		
   (:functions 
	(durability ?ax - axe) 
	(water-capacity ?r - bot) 
	(has-water ?r - bot)
	(tree ?x - location)
	(max-water ?x - bushes)
	(fire ?x - location)
	(cost))

   (:action move-grass
       :parameters (?r - bot ?x - grass ?y - location) 
       :precondition (and (at ?r ?x)
                          (connected ?x ?y)
                          (<= (tree ?x) 0)
                          )
       :effect (and (at ?r ?y)
		     (not (at ?r ?x))
		     (increase (cost) 1)
		     ))

   (:action move-bushes
       :parameters (?r - bot ?x - bushes ?y - location) 
       :precondition (and (at ?r ?x)
                          (connected ?x ?y)
                          (<= (tree ?x) 0)
                          (<= (has-water ?r) (max-water ?x))
                          )
       :effect (and (at ?r ?y)
		     (not (at ?r ?x))
		     (increase (cost) 1)
		     ))
		     
   (:action drop-water
       :parameters (?r - bot) 
       :precondition (>= (has-water ?r) 1)
                          
       :effect (and (decrease (has-water ?r) 1)
		     (increase (cost) 1)
		     ))
		     
   (:action chop-tree
       :parameters (?r - bot ?ax - axe ?l - location) 
       :precondition (and (at ?r ?l)
                          (>= (tree ?l) 1)
                          (has ?r ?ax)
                          (>= (durability ?ax) 1)
                          )
                          
       :effect (and (decrease (tree ?l) 1)
		     (decrease (durability ?ax) 1)
		     (increase (cost) 1)
		     ))

   (:action fill-water
       :parameters (?r - bot ?x - grass) 
       :precondition (and (at ?r ?x)
                          (pond ?x)
                          (< (has-water ?r) (water-capacity ?r)))               
       :effect (and (increase (has-water ?r) 1)
                    (increase (cost) 1))
       
       )		     
		     
   (:action pick-ax
       :parameters (?r - bot ?ax - axe ?l - location) 
       :precondition  (and (at ?r ?l)
                           (at ?ax ?l))
       :effect (and (not (at ?ax ?l)) 
		    (has ?r ?ax)
		    (increase (cost) 1)
		    ))
		    

   (:action put-out-fire
       :parameters (?r - bot ?l - location)
       :precondition (and (at ?r ?l) 
                          (>= (fire ?l) 1)
                          (>= (has-water ?r) 1))
       :effect (and (decrease (fire ?l) 1)
		     (decrease (has-water ?r) 1)
		     (increase (cost) 1)
		     ))
   
)
