pub fn cartesian_product<T: Clone>(sequences: &[Vec<Vec<T>>]) -> Vec<Vec<T>> {
    if sequences.is_empty() {
        return vec![Vec::new()];
    }
    let tail = cartesian_product(&sequences[1..]);
    let mut out = Vec::new();
    for item in &sequences[0] {
        for seq in &tail {
            let mut combined = item.clone();
            combined.extend(seq.iter().cloned());
            out.push(combined);
        }
    }
    out
}

pub fn get_peak_memory_in_kb() -> Result<u64, String> {
    let contents = std::fs::read_to_string("/proc/self/status")
        .map_err(|err| format!("warning: could not determine peak memory: {}", err))?;
    for line in contents.lines() {
        let mut parts = line.split_whitespace();
        if let Some(label) = parts.next() {
            if label == "VmPeak:" {
                if let Some(val) = parts.next() {
                    return val
                        .parse::<u64>()
                        .map_err(|err| format!("warning: invalid VmPeak value: {}", err));
                }
            }
        }
    }
    Err("warning: could not determine peak memory".to_string())
}
