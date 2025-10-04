use planners::translate::pddl_parser::{SExpr, parse_sexprs};

fn main() {
    println!("🔧 Testing Rust PDDL Parser Functions");
    
    let test_cases = [
        "(define domain test)",
        "(action move :parameters (?x ?y))",
        "(and (at ?x) (not (at ?y)))",
        "(= (distance ?x ?y) 10)",
        "simple-atom",
        "(nested (list (with atoms)))"
    ];
    
    for test_case in &test_cases {
        println!("\n📋 Testing: '{}'", test_case);
        
        match parse_sexprs(test_case) {
            Ok(result) => {
                println!("✅ Success: {:?}", result);
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }
    
    // Test domain file parsing
    println!("\n🏗️ Testing domain file parsing...");
    
    use std::fs;
    match fs::read_to_string("pddl/domain.pddl") {
        Ok(content) => {
            match parse_sexprs(&content) {
                Ok(forms) => {
                    println!("✅ Parsed {} forms from domain.pddl", forms.len());
                    for (i, form) in forms.iter().take(3).enumerate() {
                        if let SExpr::List(items) = form {
                            if !items.is_empty() {
                                if let SExpr::Atom(first) = &items[0] {
                                    println!("  Form {}: ({} ...)", i+1, first);
                                }
                            }
                        } else if let SExpr::Atom(atom) = form {
                            println!("  Form {}: {}", i+1, atom);
                        }
                    }
                }
                Err(e) => {
                    println!("❌ Failed to parse domain.pddl: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to read domain.pddl: {}", e);
        }
    }
    
    println!("\n🎉 Rust PDDL Parser validation complete!");
}
