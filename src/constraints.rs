// constraints.rs
use std::collections::{HashMap, HashSet};

pub type VarName = char;

/// Map from variable name (e.g., 'A') to its constraint bundle.
pub type VarConstraints = HashMap<VarName, VarConstraint>;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VarConstraint {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub pattern:    Option<String>,     // e.g. "a*" or "*z*"
    pub not_equal:  HashSet<VarName>,   // e.g. for "!=AB", A -> {'B'}, B -> {'A'}
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

/// Convenience to get or create the `VarConstraint` for a var.
pub fn ensure_var(vc: &mut VarConstraints, var: VarName) -> &mut VarConstraint {
    vc.entry(var).or_default()
}
