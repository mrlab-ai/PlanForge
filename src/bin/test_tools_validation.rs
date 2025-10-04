use planners::translate::tools::cartesian_product;

fn main() {
    println!("🔧 Testing Rust cartesian_product with Python-equivalent inputs");
    
    // Test cases that match the Python behavior we observed
    let test_cases = vec![
        // Empty case 
        vec![],
        // Single sequence: [[[1, 2], [3, 4]]] → [[1, 2], [3, 4]]
        vec![vec![vec![1, 2], vec![3, 4]]],
        // Two sequences: [[[1, 2]], [[3, 4]]] → [[1, 2, 3, 4]]
        vec![vec![vec![1, 2]], vec![vec![3, 4]]],
        // Multiple elements: [[[1], [2]], [[3], [4]]] → [[1, 3], [1, 4], [2, 3], [2, 4]]  
        vec![vec![vec![1], vec![2]], vec![vec![3], vec![4]]],
    ];
    
    for (i, case) in test_cases.iter().enumerate() {
        println!("\n📋 Test {}: Input: {:?}", i + 1, case);
        let result = cartesian_product(case);
        println!("   Output: {:?}", result);
    }
    
    println!("\n🎯 Expected Python outputs for comparison:");
    println!("   Test 1: [[]]");
    println!("   Test 2: [[1, 2], [3, 4]]");
    println!("   Test 3: [[1, 2, 3, 4]]");
    println!("   Test 4: [[1, 3], [1, 4], [2, 3], [2, 4]]");
}
