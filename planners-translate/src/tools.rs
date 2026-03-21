/// Port of tools.py

/// Python: def cartesian_product(sequences)
/// This isn't actually a proper cartesian product because we
/// concatenate lists, rather than forming sequences of atomic elements.
pub fn cartesian_product<T: Clone>(sequences: &[Vec<Vec<T>>]) -> Vec<Vec<T>> {
    if sequences.is_empty() {
        return vec![vec![]];
    }

    let rest = cartesian_product(&sequences[1..]);
    let mut result = vec![];
    for item in &sequences[0] {
        for sequence in &rest {
            let mut combined = item.clone();
            combined.extend(sequence.iter().cloned());
            result.push(combined);
        }
    }
    result
}

/// Standard cartesian product (itertools.product equivalent)
pub fn product<T: Clone>(sequences: &[Vec<T>]) -> Vec<Vec<T>> {
    if sequences.is_empty() {
        return vec![vec![]];
    }

    let rest = product(&sequences[1..]);
    let mut result = vec![];
    for item in &sequences[0] {
        for sequence in &rest {
            let mut combined = vec![item.clone()];
            combined.extend(sequence.iter().cloned());
            result.push(combined);
        }
    }
    result
}

/// Python: def get_peak_memory_in_kb()
pub fn get_peak_memory_in_kb() -> Option<usize> {
    if let Ok(content) = std::fs::read_to_string("/proc/self/status") {
        for line in content.lines() {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 && parts[0] == "VmPeak:" {
                return parts[1].parse().ok();
            }
        }
    }
    None
}
