// constraints.rs
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use crate::parser::{parse_form, ParsedForm};

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
    // Create a `VarConstraints` map whose internal map is `map`.
    //fn of(map: HashMap<char, VarConstraint>) -> Self { Self { inner: map } }

    /// Insert a complete `VarConstraint` for a variable.
    #[cfg(test)]
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

    // Iterate over `(variable, constraint)` pairs.
    //fn iter(&self) -> impl Iterator<Item = (&char, &VarConstraint)> { self.inner.iter() }

    // Convenience: number of variables with constraints.
    #[cfg(test)]
    fn len(&self) -> usize {
        self.inner.len()
    }

    // Convenience: true if no constraints are stored.
    // fn is_empty(&self) -> bool { self.inner.is_empty() }
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
#[derive(Debug, Clone, Default)]
pub struct VarConstraint {
    pub min_length: Option<usize>,
    pub max_length: Option<usize>,
    pub form: Option<String>,
    pub not_equal: HashSet<char>,
    pub parsed_form: OnceCell<ParsedForm>, // Default::default() → OnceCell::new()
}

impl VarConstraint {
    /// Set both min and max to the same exact length.
    pub(crate) fn set_exact_len(&mut self, len: usize) {
        self.min_length = Some(len);
        self.max_length = Some(len);
    }
    /// Get the parsed form
    pub(crate) fn get_parsed_form(&self) -> Option<&ParsedForm> {
        self.form.as_deref().map(|f| self.parsed_form.get_or_init(|| parse_form(f).unwrap()))
    }
    /// Get bounds
    pub fn bounds(&self) -> (Option<usize>, Option<usize>) {
        (self.min_length, self.max_length)
    }
}

// Implement equality for VarConstraint
impl PartialEq for VarConstraint {
    fn eq(&self, other: &Self) -> bool {
        self.min_length == other.min_length
            && self.max_length == other.max_length
            && self.form == other.form
            && self.not_equal == other.not_equal
        // ignore parsed_form
    }
}

impl Eq for VarConstraint {}

/// Compact human-readable display for a single `VarConstraint`.
///
/// This is intended for debugging / logs, not for round-tripping.
/// It summarizes:
/// - the allowed length range (e.g. `[3–5]`, `[≥3]`, `[≤5]`, `[*]`)
/// - the optional form string (or `*` if absent)
/// - the set of variables it must not equal, in sorted order
impl fmt::Display for VarConstraint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Format the length range nicely.
        // Handle each case: both bounds, only min, only max, or none.
        let len_str = match (self.min_length, self.max_length) {
            (Some(min), Some(max)) => format!("[{}-{}]", min, max),
            (Some(min), None)      => format!("[≥{}]", min),
            (None, Some(max))      => format!("[≤{}]", max),
            (None, None)           => "[*]".to_string(), // unconstrained
        };

        // Show the "form" string if present, otherwise `-`
        let form_str = self.form.as_deref().unwrap_or("*");

        // Collect the `not_equal` set into a sorted Vec<char> for stable output
        let mut ne: Vec<char> = self.not_equal.iter().copied().collect();
        ne.sort_unstable();
        // Turn it into a string: e.g. ['A','B','C'] → "ABC"
        let ne_str = if ne.is_empty() {
            "{}".to_string() // explicit empty set
        } else {
            ne.into_iter().collect::<String>()
        };

        // Final compact output
        write!(f, "len={}; form={}; not_equal={}", len_str, form_str, ne_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ensure_creates_default() {
        let mut vcs = VarConstraints::default();
        assert!(vcs.get('A').is_none());
        {
            let a = vcs.ensure('A');
            // default created; tweak it
            a.min_length = Some(3);
        }
        assert_eq!(Some(3), vcs.get('A').unwrap().min_length);
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
        let mut vc = VarConstraint {
            min_length: Some(2),
            max_length: Some(4),
            form: Some("a*".into()),
            ..Default::default()
        };
        vc.not_equal.extend(['C', 'B']); // out of order on purpose
        let shown = vc.to_string();
        // not_equal should be sorted -> {BC}
        assert_eq!("len=[2-4]; form=a*; not_equal=BC", shown);
    }

    #[test]
    fn display_var_constraints_multiline_sorted() {
        let mut vcs = VarConstraints::default();
        let mut a = VarConstraint::default();
        a.min_length = Some(1);
        let mut c = VarConstraint::default();
        c.max_length = Some(2);
        let mut b = VarConstraint::default();
        b.form = Some("*x*".into());
        // Insert out of order to verify deterministic sort in Display
        vcs.insert('C', c);
        vcs.insert('A', a);
        vcs.insert('B', b);

        let s = vcs.to_string();
        let lines: Vec<&str> = s.lines().collect();

        let expected = vec![
            "A: len=[≥1]; form=*; not_equal={}",
            "B: len=[*]; form=*x*; not_equal={}",
            "C: len=[≤2]; form=*; not_equal={}"
        ];

        assert_eq!(expected, lines);
    }
}
