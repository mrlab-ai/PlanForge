//! Utility functions for the translator
//! Port of python/translate/tools.py

use std::fs::File;
use std::io::{BufRead, BufReader};

/// Compute a pseudo-cartesian product that concatenates lists
/// rather than forming sequences of atomic elements  
/// This matches the Python implementation which concatenates lists with +
pub fn cartesian_product<T: Clone>(sequences: &[Vec<Vec<T>>]) -> Vec<Vec<T>> {
    if sequences.is_empty() {
        return vec![vec![]];
    }
    
    let mut result = Vec::new();
    let tail_products = cartesian_product(&sequences[1..]);
    
    for item in &sequences[0] {
        for sequence in &tail_products {
            // Concatenate lists like Python: item + sequence
            let mut new_sequence = item.clone();
            new_sequence.extend_from_slice(sequence);
            result.push(new_sequence);
        }
    }
    
    result
}

/// Get peak memory usage in KB (Linux only)
pub fn get_peak_memory_in_kb() -> Result<u64, Box<dyn std::error::Error>> {
    let file = File::open("/proc/self/status")?;
    let reader = BufReader::new(file);
    
    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 2 && parts[0] == "VmPeak:" {
            return Ok(parts[1].parse()?);
        }
    }
    
    Err("Could not determine peak memory".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cartesian_product_empty() {
        let sequences: Vec<Vec<Vec<i32>>> = vec![];
        let result = cartesian_product(&sequences);
        let expected: Vec<Vec<i32>> = vec![vec![]];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_cartesian_product_single() {
        let sequences = vec![vec![vec![1], vec![2], vec![3]]];
        let result = cartesian_product(&sequences);
        assert_eq!(result, vec![vec![1], vec![2], vec![3]]);
    }

    #[test]
    fn test_cartesian_product_two_lists() {
        // Test case: [[[1], [2]], [[3], [4]]] should produce [[1,3], [1,4], [2,3], [2,4]]
        let sequences = vec![vec![vec![1], vec![2]], vec![vec![3], vec![4]]];
        let result = cartesian_product(&sequences);
        assert_eq!(result, vec![
            vec![1, 3], vec![1, 4],
            vec![2, 3], vec![2, 4]
        ]);
    }
}
