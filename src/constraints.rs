// constraints.rs
use std::collections::{HashMap, HashSet};
use std::fmt;

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
    /// Create a `VarConstraints` map whose internal map is `map`.
    fn of(map: HashMap<char, VarConstraint>) -> Self {
        Self { inner: map }
    }

    /// Insert a complete `VarConstraint` for a variable.
    pub(crate) fn insert(&mut self, var: char, constraint: VarConstraint) {
        self.inner.insert(var, constraint);
    }

    /// Ensure a variable has an entry; create a default constraint if missing.
    /// Returns a mutable reference so the caller can update it in place.
    pub(crate) fn ensure(&mut self, var: char) -> &mut VarConstraint {
        self.inner.entry(var).or_default()
    }

    /// Retrieve a read-only reference to the constraints for a variable, if any.
    pub(crate) fn get(&self, var: char) -> Option<&VarConstraint> {
        self.inner.get(&var)
    }

    /// Iterate over `(variable, constraint)` pairs.
    fn iter(&self) -> impl Iterator<Item = (&char, &VarConstraint)> {
        self.inner.iter()
    }

    /// Convenience: number of variables with constraints.
    fn len(&self) -> usize {
        self.inner.len()
    }

    /// Convenience: true if no constraints are stored.
    fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

/// Pretty, deterministic display (sorted by variable) like:
/// `A: len=[2, 4], form=Some("a*"), not_equal={B,C}`
impl fmt::Display for VarConstraints {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut keys: Vec<char> = self.inner.keys().copied().collect();
        keys.sort_unstable();
        for (i, k) in keys.iter().enumerate() {
            let vc = &self.inner[k];
            if i > 0 { writeln!(f)?; }
            write!(f, "{k}: {vc}")?;
        }
        Ok(())
    }
}

/// A set of rules restricting what a single variable can match.
///
/// Fields are optional so that constraints can be partial:
/// - `min_length` / `max_length` limit how many characters the variable can bind to.
/// - `form` is an optional sub-pattern the variable's match must satisfy
///   (e.g., `"a*"` means "must start with `a`"; `"*z*"` means "must contain `z`").
/// - `not_equal` lists variables whose matches must *not* be identical to this one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VarConstraint {
    pub min_length: usize,
    pub max_length: usize,
    pub form: Option<String>,     // e.g., "a*" or "*z*"
    pub not_equal: HashSet<char>, // e.g., A's set contains 'B' if `A != B` is required
}

impl Default for VarConstraint {
    fn default() -> Self {
        Self {
            min_length: 1,
            max_length: usize::MAX,
            form: None,
            not_equal: HashSet::new(),
        }
    }
}

impl VarConstraint {
    /// Return the (min, max) length bounds for this variable.
    ///
    /// - If a bound is missing, falls back to the provided defaults.
    /// - This is often used when generating regex prefilters or substring loops.
    fn bounds(&self) -> (usize, usize) {
        let min = self.min_length;
        let max = self.max_length;
        (min, max)
    }

    /// Set both min and max to the same exact length.
    pub(crate) fn set_exact_len(&mut self, len: usize) {
        self.min_length = len;
        self.max_length = len;
    }
}

/// Compact human-readable display for a single constraint.
impl fmt::Display for VarConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Show not_equal in sorted order for stability
        let mut ne: Vec<char> = self.not_equal.iter().copied().collect();
        ne.sort_unstable();
        write!(
            f,
            "len=[{:?}, {:?}]; form={:?}; not_equal={{{}}}",
            self.min_length,
            self.max_length,
            self.form.as_deref(),
            ne.into_iter().collect::<String>()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bounds_with_overrides() {
        let vc = VarConstraint {
            min_length: 2,
            max_length: 5,
            form: None,
            not_equal: HashSet::new(),
        };
        assert_eq!((2, 5), vc.bounds());
    }

    #[test]
    fn ensure_creates_default() {
        let mut vcs = VarConstraints::default();
        assert!(vcs.get('A').is_none());
        {
            let a = vcs.ensure('A');
            // default created; tweak it
            a.min_length = 3;
        }
        assert_eq!(3, vcs.get('A').unwrap().min_length);
        assert_eq!(1, vcs.len());
    }

    #[test]
    fn insert_and_get_roundtrip() {
        let mut vcs = VarConstraints::default();
        let mut vc = VarConstraint::default();
        vc.form = Some("*z*".into());
        vc.not_equal.extend(['B', 'C']);
        vcs.insert('A', vc.clone());
        assert_eq!(Some(&vc), vcs.get('A'));
    }

    #[test]
    fn display_var_constraint_is_stable() {
        let mut vc = VarConstraint::default();
        vc.min_length = 2;
        vc.max_length = 4;
        vc.form = Some("a*".into());
        vc.not_equal.extend(['C', 'B']); // out of order on purpose
        let shown = vc.to_string();
        // not_equal should be sorted -> {BC}
        assert_eq!("len=[2, 4]; form=Some(\"a*\"); not_equal={BC}", shown);
    }

    #[test]
    fn display_var_constraints_multiline_sorted() {
        let mut vcs = VarConstraints::default();
        let mut a = VarConstraint::default();
        a.min_length = 1;
        let mut c = VarConstraint::default();
        c.max_length = 2;
        let mut b = VarConstraint::default();
        b.form = Some("*x*".into());
        // Insert out of order to verify deterministic sort in Display
        vcs.insert('C', c);
        vcs.insert('A', a);
        vcs.insert('B', b);

        let s = vcs.to_string();
        let lines: Vec<&str> = s.lines().collect();

        let expected = vec![
            "A: len=[1, 18446744073709551615]; form=None; not_equal={}",
            "B: len=[1, 18446744073709551615]; form=Some(\"*x*\"); not_equal={}",
            "C: len=[1, 2]; form=None; not_equal={}"
        ];

        assert_eq!(expected, lines);
    }
}
