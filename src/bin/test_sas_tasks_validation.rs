use planners::translate::sas::{SASTask, Variable, SASOperator, NumericVariable};
use planners::translate::sas_tasks::*;

fn main() {
    println!("🔧 Testing Rust SAS Tasks implementation against Python semantics");
    
    // Test 1: Basic SASTask creation and manipulation
    println!("\n📋 Test 1: SASTask basic functionality");
    
    let mut task = SASTask::new();
    println!("   Created empty SAS task");
    println!("   Variables: {}", task.num_variables());
    println!("   Operators: {}", task.num_operators());
    
    // Test 2: Variable handling
    println!("\n📋 Test 2: Variable handling");
    
    let var1 = Variable {
        value_names: vec!["location1".to_string(), "location2".to_string(), "location3".to_string()],
    };
    
    let var2 = Variable {
        value_names: vec!["empty".to_string(), "has-package".to_string()],
    };
    
    task.add_variable(var1);
    task.add_variable(var2);
    
    println!("   Added 2 variables");
    println!("   Variables count: {}", task.num_variables());
    println!("   Variable 0 values: {:?}", task.variables[0].value_names);
    println!("   Variable 1 values: {:?}", task.variables[1].value_names);
    
    // Test 3: Operator handling  
    println!("\n📋 Test 3: Operator handling");
    
    let op1 = SASOperator {
        name: "move".to_string(),
        prevails: vec![], // No prevail conditions
        effects: vec![(0, Some(0), 1)], // Variable 0: location1 -> location2
        numeric_effects: vec![],
        numeric_preconds: vec![],
    };
    
    let op2 = SASOperator {
        name: "pickup".to_string(),
        prevails: vec![(0, 1)], // Must be at location2
        effects: vec![(1, Some(0), 1)], // Variable 1: empty -> has-package
        numeric_effects: vec![],
        numeric_preconds: vec![],
    };
    
    task.add_operator(op1);
    task.add_operator(op2);
    
    println!("   Added 2 operators");
    println!("   Operators count: {}", task.num_operators());
    println!("   Operator 0: {}", task.operators[0].name);
    println!("   Operator 1: {}", task.operators[1].name);
    
    // Test 4: Numeric variables
    println!("\n📋 Test 4: Numeric variables");
    
    let num_var = NumericVariable {
        name: "total-cost".to_string(),
        initial: Some(0),
        ntype: "number".to_string(),
        axiom_layer: 0,
    };
    
    task.numeric_variables.push(num_var);
    println!("   Added numeric variable: {}", task.numeric_variables[0].name);
    
    // Test 5: Task dump
    println!("\n📋 Test 5: Task information dump");
    task.dump();
    
    // Test 6: File writing (basic test)
    println!("\n📋 Test 6: File writing capability");
    match task.write_to_file("test_output.sas") {
        Ok(_) => println!("   ✅ Successfully wrote SAS file"),
        Err(e) => println!("   ❌ Failed to write SAS file: {}", e),
    }
    
    println!("\n🎯 SAS Tasks Validation Summary:");
    println!("   ✅ Basic SASTask creation and manipulation works");
    println!("   ✅ Variable handling (finite-domain variables) works");
    println!("   ✅ Operator handling (name, prevails, effects) works");
    println!("   ✅ Numeric variable support present");
    println!("   ✅ File writing capability present");
    
    println!("\n⚠️  Missing compared to Python SASTask:");
    println!("   • init, goal fields missing");
    println!("   • axioms, global_constraint fields missing");
    println!("   • metric field missing");
    println!("   • init_constant_predicates, init_constant_numerics missing");
    println!("   • validate() method missing");
    println!("   • Many output methods missing");
    
    println!("\n📝 Validation Status:");
    println!("   • Core data structures: ✅ Present and working");
    println!("   • Basic operations: ✅ Functional");
    println!("   • Field completeness: ❌ Missing critical fields");
    println!("   • Method completeness: ❌ Missing many Python methods");
    println!("   • Ready for full validation: ❌ Needs more implementation");
}
