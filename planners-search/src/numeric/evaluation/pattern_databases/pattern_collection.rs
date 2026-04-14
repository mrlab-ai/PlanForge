#[cfg(test)]
mod tests;

use super::projected_task::Pattern;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct PatternCollection {
    patterns: Vec<Pattern>,
}

impl PatternCollection {
    pub fn new(patterns: Vec<Pattern>) -> Self {
        let mut collection = Self { patterns };
        collection.normalize_in_place();
        collection
    }

    pub fn empty() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    pub fn as_slice(&self) -> &[Pattern] {
        &self.patterns
    }

    pub fn iter(&self) -> impl Iterator<Item = &Pattern> {
        self.patterns.iter()
    }

    pub fn into_vec(self) -> Vec<Pattern> {
        self.patterns
    }

    pub fn push(&mut self, pattern: Pattern) -> bool {
        let normalized = pattern.normalized();
        match self.patterns.binary_search(&normalized) {
            Ok(_) => false,
            Err(index) => {
                self.patterns.insert(index, normalized);
                true
            }
        }
    }

    pub fn contains(&self, pattern: &Pattern) -> bool {
        self.patterns.binary_search(&pattern.normalized()).is_ok()
    }

    fn normalize_in_place(&mut self) {
        for pattern in &mut self.patterns {
            pattern.normalize_in_place();
        }
        self.patterns.sort();
        self.patterns.dedup();
    }
}
