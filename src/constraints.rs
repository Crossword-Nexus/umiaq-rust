// constraints.rs
use std::collections::{HashMap, HashSet};

/// Map from variable name (e.g., 'A') to its constraint bundle.
#[derive(Debug, Clone, Default)]
pub struct VarConstraints {
    inner: HashMap<char, VarConstraint>,
}

impl VarConstraints {
    pub fn new() -> Self {
        Self { inner: HashMap::new() }
    }

    pub fn insert(&mut self, var: char, constraint: VarConstraint) {
        self.inner.insert(var, constraint);
    }

    pub fn ensure(&mut self, var: char) -> &mut VarConstraint {
        self.inner.entry(var).or_default()
    }

    pub fn get(&self, var: char) -> Option<&VarConstraint> {
        self.inner.get(&var)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&char, &VarConstraint)> {
        self.inner.iter()
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VarConstraint {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub form:    Option<String>,     // e.g. "a*" or "*z*"
    pub not_equal:  HashSet<char>,   // e.g. for "!=AB", A -> {'B'}, B -> {'A'}
}

impl VarConstraint {
    /// Clamp (min,max) for regex prefilter or substring loop.
    /// Falls back to (1, big) when not set.
    pub fn bounds(&self, default_min: usize, default_max: usize) -> (usize, usize) {
        let min = self.min_length.unwrap_or(default_min);
        let max = self.max_length.unwrap_or(default_max);
        (min, max)
    }
}
