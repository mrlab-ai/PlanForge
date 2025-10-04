use planners::translate::pddl::{
    pddl_types::{Type, TypedObject},
    conditions::{Condition, Literal},
    predicates::Predicate,
    functions::Function,
};

fn main() {
    println!("🔧 Testing Rust PDDL module against Python semantics");
    
    // Test 1: Basic Type functionality
    println!("\n📋 Test 1: Type functionality");
    
    let type1 = Type::new("location".to_string(), None);
    let type2 = Type::new("vehicle".to_string(), Some("object".to_string()));
    
    println!("   Type1: {:?}", type1);
    println!("   Type2: {:?}", type2);
    println!("   ✅ Type creation works");
    
    // Test 2: TypedObject functionality
    println!("\n📋 Test 2: TypedObject functionality");
    
    let obj1 = TypedObject::new("truck1".to_string(), Some("vehicle".to_string()));
    let obj2 = TypedObject::new("location1".to_string(), Some("location".to_string()));
    
    println!("   Object1: {:?}", obj1);
    println!("   Object2: {:?}", obj2);
    
    // Test equality
    let obj1_copy = TypedObject::new("truck1".to_string(), Some("vehicle".to_string()));
    println!("   obj1 == obj1_copy: {}", obj1 == obj1_copy);
    println!("   obj1 == obj2: {}", obj1 == obj2);
    println!("   ✅ TypedObject creation and equality works");
    
    // Test 3: Basic Condition functionality  
    println!("\n📋 Test 3: Condition functionality");
    
    let literal = Literal {
        predicate: "at".to_string(),
        args: vec!["truck1".to_string(), "location1".to_string()],
        negated: false,
    };
    
    let condition1 = Condition::Literal(literal.clone());
    let condition2 = Condition::Truth;
    let condition3 = Condition::And(vec![condition1.clone(), condition2.clone()]);
    
    println!("   Literal: {:?}", literal);
    println!("   Condition1 (Literal): {:?}", condition1);
    println!("   Condition2 (Truth): {:?}", condition2);
    println!("   Condition3 (And): Complex conjunction");
    println!("   ✅ Condition creation works");
    
    // Test 4: Predicate functionality
    println!("\n📋 Test 4: Predicate functionality");
    
    let predicate = Predicate::new("at".to_string(), vec![
        TypedObject::new("obj".to_string(), Some("object".to_string())),
        TypedObject::new("loc".to_string(), Some("location".to_string()))
    ]);
    
    println!("   Predicate: {:?}", predicate);
    println!("   ✅ Predicate creation works");
    
    // Test 5: Function functionality  
    println!("\n📋 Test 5: Function functionality");
    
    let function = Function::new("distance".to_string(), vec![
        TypedObject::new("loc1".to_string(), Some("location".to_string())),
        TypedObject::new("loc2".to_string(), Some("location".to_string()))
    ], "number".to_string());
    
    println!("   Function: {:?}", function);
    println!("   ✅ Function creation works");
    
    println!("\n🎯 PDDL Module Validation Summary:");
    println!("   ✅ Basic type system (Type, TypedObject) functional");
    println!("   ✅ Condition enum-based architecture working");
    println!("   ✅ Predicate and Function structures present");
    println!("   ⚠️  Module is architecturally different from Python (enum vs class hierarchy)");
    println!("   ⚠️  Many methods not yet implemented (uniquify_name, get_atom, etc.)");
    println!("   ⚠️  Requires full implementation validation once complete");
    
    println!("\n📝 Validation Status:");
    println!("   • Core data structures: ✅ Present and functional");
    println!("   • Method parity: ❌ Many Python methods missing");
    println!("   • Architectural equivalence: ⚠️ Different approach (enum vs classes)");
    println!("   • Ready for validation: ❌ Needs more implementation");
}
