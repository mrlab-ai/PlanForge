use planners::translate::pddl_parser::{SExpr, parse_sexprs};

fn main() {
    println!("🔤 Testing PDDL Parser Case Sensitivity");
    println!("{}", "=".repeat(50));

    // Test case sensitivity in parsing
    let test_cases = vec![
        "(DEFINE domain test)",
        "(define DOMAIN test)",  
        "(AND (at ?x) (NOT (at ?y)))",
        "(and (AT ?x) (not (AT ?y)))",
        ":PARAMETERS",
        ":parameters",
        "UPPER-CASE-ATOM",
        "lower-case-atom",
        "Mixed-Case-Atom"
    ];

    for test_case in &test_cases {
        println!("\n📋 Testing: '{}'", test_case);
        
        match parse_sexprs(test_case) {
            Ok(result) => {
                println!("✅ Success: {:?}", result);
                
                // Analyze the first result to see case handling
                if let Some(first) = result.first() {
                    analyze_case_handling(first, test_case);
                }
            }
            Err(e) => {
                println!("❌ Error: {}", e);
            }
        }
    }

    println!("\n{}", "=".repeat(50));
    println!("🎯 Case Sensitivity Analysis Summary");
}

fn analyze_case_handling(sexpr: &SExpr, original: &str) {
    match sexpr {
        SExpr::Atom(atom) => {
            let is_lowercase = atom.to_lowercase() == *atom;
            let is_uppercase = atom.to_uppercase() == *atom;
            
            if is_lowercase && original.to_lowercase() != *original {
                println!("  🔽 Converted to lowercase: '{}' → '{}'", original, atom);
            } else if is_uppercase && original.to_uppercase() != *original {
                println!("  🔼 Converted to uppercase: '{}' → '{}'", original, atom);
            } else if atom == original {
                println!("  ➡️  Preserved case: '{}'", atom);
            } else {
                println!("  🔄 Case transformation: '{}' → '{}'", original, atom);
            }
        }
        SExpr::List(items) => {
            if let Some(SExpr::Atom(first_atom)) = items.first() {
                // Check the first atom in the list
                if original.contains("DEFINE") || original.contains("define") {
                    let expected_lowercase = first_atom == "define";
                    println!("  📝 First atom '{}' is lowercase: {}", first_atom, expected_lowercase);
                }
            }
        }
    }
}
