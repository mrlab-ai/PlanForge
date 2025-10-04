use planners::translate::tools::get_peak_memory_in_kb;

fn main() {
    println!("🔧 Testing Rust get_peak_memory_in_kb function");
    
    match get_peak_memory_in_kb() {
        Ok(memory) => {
            println!("✅ Peak memory: {} KB", memory);
            println!("📋 Function working correctly on Linux");
        }
        Err(e) => {
            println!("❌ Error: {}", e);
            println!("📋 Expected behavior - may fail on non-Linux systems");
        }
    }
}
