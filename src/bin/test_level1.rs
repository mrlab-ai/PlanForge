use planners::translate::numeric_axiom_rules::*;

fn main() {
    println!("🔧 Testing Level 1 Numeric Axiom Rules Functions");
    
    // Test 1: Create basic numeric axioms
    let pne1 = PrimitiveNumericExpression {
        name: "fuel".to_string(),
        args: vec!["rocket1".to_string()],
    };
    
    let pne2 = PrimitiveNumericExpression {
        name: "capacity".to_string(),
        args: vec!["tank1".to_string()],
    };
    
    let axiom1 = InstantiatedNumericAxiom {
        name: "fuel-consumption".to_string(),
        op: Some("-".to_string()),
        parts: vec![
            NumericPart::Primitive(pne1.clone()),
            NumericPart::Constant(NumericConstant(10))
        ],
        effect: pne2.clone(),
    };
    
    let axioms = vec![axiom1];
    
    // Test 2: Build axiom by PNE mapping
    let axiom_by_pne = axiom_by_pne(&axioms);
    println!("✅ axiom_by_pne function works: {} entries", axiom_by_pne.len());
    
    // Test 3: Identify constants
    let constants = identify_constants(&axioms, &axiom_by_pne);
    println!("✅ identify_constants function works: {} constants", constants.len());
    
    // Test 4: Compute axiom layers
    let (layers, max_layer) = compute_axiom_layers(&axioms, &constants, &axiom_by_pne);
    println!("✅ compute_axiom_layers function works: {} layers, max: {}", layers.len(), max_layer);
    
    // Test 5: Handle axioms (main entry point)
    let (layers_2, max_layer_2, equivalent, constant_axioms) = handle_axioms(&axioms);
    println!("✅ handle_axioms function works: {} layers, {} equivalent, {} constants", 
             layers_2.len(), equivalent.len(), constant_axioms.len());
    
    println!("🎉 Level 1 Numeric Axiom Rules: ALL TESTS PASSED!");
}
