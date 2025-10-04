use planners::translate::constraints::Assignment;

fn main() {
    println!("🔧 Testing complex equivalence classes in Rust");
    
    // Test the same complex case as Python: ?x = ?y, ?y = ?z, ?z = value1, ?a = ?b
    let mut assignment = Assignment::new(vec![
        ("?x".to_string(), "?y".to_string()),
        ("?y".to_string(), "?z".to_string()),
        ("?z".to_string(), "value1".to_string()),
        ("?a".to_string(), "?b".to_string()),
    ]);
    
    println!("Assignment: {:?}", assignment.equalities);
    println!("is_consistent: {}", assignment.is_consistent());
    
    if let Some(mapping) = assignment.get_mapping() {
        println!("Mapping:");
        for (key, value) in &mapping {
            println!("  {} -> {}", key, value);
        }
        
        // Check that we get equivalent results to Python
        // Python gave: {'?z': 'value1', '?x': 'value1', 'value1': 'value1', '?y': 'value1', '?a': '?a', '?b': '?a'}
        println!("\nChecking equivalence with Python results:");
        
        // Check that ?x, ?y, ?z all map to value1
        let x_val = mapping.get("?x").unwrap();
        let y_val = mapping.get("?y").unwrap(); 
        let z_val = mapping.get("?z").unwrap();
        let value1_val = mapping.get("value1").unwrap();
        
        println!("  ?x maps to: {}", x_val);
        println!("  ?y maps to: {}", y_val);
        println!("  ?z maps to: {}", z_val);
        println!("  value1 maps to: {}", value1_val);
        
        let all_to_value1 = x_val == "value1" && y_val == "value1" && z_val == "value1" && value1_val == "value1";
        println!("  ✅ All ?x,?y,?z,value1 -> value1: {}", all_to_value1);
        
        // Check that ?a and ?b map to the same value (either ?a or ?b)
        let a_val = mapping.get("?a").unwrap();
        let b_val = mapping.get("?b").unwrap();
        let ab_consistent = a_val == b_val;
        println!("  ?a maps to: {}", a_val);
        println!("  ?b maps to: {}", b_val);
        println!("  ✅ ?a and ?b map to same value: {}", ab_consistent);
        
        if all_to_value1 && ab_consistent {
            println!("\n🎯 ✅ Rust equivalence classes match Python behavior!");
        } else {
            println!("\n❌ Rust equivalence classes differ from Python!");
        }
    }
}
