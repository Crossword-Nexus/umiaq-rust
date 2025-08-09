// constraints.rs
use std::collections::{HashMap, HashSet};

/// A collection of constraints for variables in a pattern-matching equation.
///
/// This wraps a `HashMap<char, VarConstraint>` where:
/// - Each `char` key is a variable name (e.g., 'A').
/// - The associated `VarConstraint` stores rules about what that variable can match.
#[derive(Debug, Clone, Default)]
pub struct VarConstraints {
    inner: HashMap<char, VarConstraint>,
}

impl VarConstraints {
    /// Create an empty `VarConstraints` map.
    pub fn new() -> Self {
        Self { inner: HashMap::new() }
    }

    /// Insert a complete `VarConstraint` for a variable.
    pub fn insert(&mut self, var: char, constraint: VarConstraint) {
        self.inner.insert(var, constraint);
    }

    /// Ensure a variable has an entry; create a default constraint if missing.
    /// Returns a mutable reference so the caller can update it in place.
    pub fn ensure(&mut self, var: char) -> &mut VarConstraint {
        self.inner.entry(var).or_default()
    }

    /// Retrieve a read-only reference to the constraints for a variable, if any.
    pub fn get(&self, var: char) -> Option<&VarConstraint> {
        self.inner.get(&var)
    }

    /// Iterate over `(variable, constraint)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&char, &VarConstraint)> {
        self.inner.iter()
    }
}

/// A set of rules restricting what a single variable can match.
///
/// Fields are optional so that constraints can be partial:
/// - `min_length` / `max_length` limit how many characters the variable can bind to.
/// - `form` is an optional sub-pattern the variable's match must satisfy
///   (e.g., `"a*"` means must start with `a`; `"*z*"` means must contain `z`).
/// - `not_equal` lists variables whose matches must *not* be identical to this one.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct VarConstraint {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub form: Option<String>,     // e.g. "a*" or "*z*"
    pub not_equal: HashSet<char>, // e.g. A's set contains 'B' if `A != B` is required
}

impl VarConstraint {
    /// Return the (min, max) length bounds for this variable.
    ///
    /// - If a bound is missing, falls back to the provided defaults.
    /// - This is often used when generating regex prefilters or substring loops.
    pub fn bounds(&self, default_min: usize, default_max: usize) -> (usize, usize) {
        let min = self.min_length.unwrap_or(default_min);
        let max = self.max_length.unwrap_or(default_max);
        (min, max)
    }
}
