use crate::constraints::VarConstraints;
use fancy_regex::Regex;
use std::collections::HashSet;
use std::sync::LazyLock;

/// Matches exact length constraints like `|A|=5`
static LEN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\|([A-Z])\|=(\d+)$").unwrap());

/// Matches inequality constraints like `!=AB`
static NEQ_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^!=([A-Z]+)$").unwrap());

// TODO? disallow accepting one paren and the other; only allow, e.g., 3-5 or 3 (not 3-5-7)
// TODO require colon if both types, require no colon if just one (but support both or just one)
// TODO constrain re to only allow lc letters, '.', '*', '/', '@', '#', etc. instead of "^)" in "[^)]"
/// Matches complex constraints like `A=(3-5:a*)` with optional length and pattern
static COMPLEX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([A-Z])=\(?([\d\-]*):?([^)]*)\)?$").unwrap());

#[derive(Debug, Clone)]
/// A single raw form string plus *solver metadata*; **not tokenized**.
/// Use `parse_equation(&pattern.raw_string)` to get `Vec<FormPart>`.
///
/// ## Solver metadata (what it is and why it exists)
/// - `lookup_keys`: `Option<HashSet<char>>`
///   - **What:** The subset of this form's variables that also appear in forms
///     that have already been placed earlier in `Patterns::ordered_list`.
///   - **When it's set:** Assigned by `Patterns::ordered_partitions()` *after* the
///     forms have been reordered for solving.
///   - **Why it helps:** During the multi-form join, candidate bindings for this
///     form can be bucketed by the concrete values of these variables and then
///     matched in O(1)/O(log N) time against earlier choices, instead of scanning
///     all candidates. In other words, `lookup_keys` is the *join key* that lets
///     you intersect partial solutions cheaply.
///   - **How it's used:** When you collect matches for each form, you can index
///     (e.g., `HashMap<JoinKey, Vec<Bindings>>`) by the values of `lookup_keys`.
///     Then, when recursing, you fetch only the compatible bucket for the next form.
///
/// Note: `Pattern` intentionally keeps `raw_string` (e.g., "AB", "A~A", "/sett")
/// unparsed; tokenization to `Vec<FormPart>` is deferred to matching time.
///
/// Example:
/// - Input: `"ABC;BC;C"`
/// - Reordering picks `"ABC"` first, then `"BC"`, then `"C"`.
/// - `lookup_keys`:
///     * for `"ABC"`: `None` (first form has nothing prior)
///     * for `"BC"`: `Some({'B','C'})` (overlap with already-chosen variables)
///     * for `"C"`:  `Some({'C'})`
pub struct Pattern {
    /// The raw string representation of the pattern, such as "AB" or "/triangle"
    pub raw_string: String,
    /// Set of variable names that this pattern shares with previously processed ones,
    /// used for optimizing lookups in recursive solving
    pub lookup_keys: Option<HashSet<char>>,
}

/// Implementation for the `Pattern` struct, representing a single pattern string
/// and utilities to extract its variable set.
impl Pattern {
    /// Constructs a new `Pattern` from any type that can be converted into a `String`.
    /// The resulting `lookup_keys` is initialized to `None`.
    pub fn of(string: impl Into<String>) -> Self {
        Self {
            raw_string: string.into(),
            lookup_keys: None,
        }
    }

    // TODO? just do this once (lazily) rather than recomputing it each time?
    /// Extracts all uppercase ASCII letters from the pattern string.
    /// These are treated as variable names (e.g., A, B, C).
    pub fn variables(&self) -> HashSet<char> {
        self.raw_string
            .chars()
            .filter(char::is_ascii_uppercase)
            .collect()
    }
}

#[derive(Debug, Default)]
/// The **parsed equation** at a structural level: extracted constraints + collected forms +
/// a solver-friendly order. Forms here are still raw strings; tokenize each with
/// `parse_equation` when matching.
///
/// - `list`: all non-constraint forms in original order
/// - `var_constraints`: per-variable rules parsed from things like `|A|=5`, `!=AB`,
///   `A=(3-5:a*)`
/// - `ordered_list`: `list` reordered so that forms with many variables appear
///   earlier and subsequent forms maximize overlap with already-chosen variables.
///   As part of this step, each later form's `lookup_keys` is set to the overlap
///   with the variables seen so far (its *join key*).
pub struct Patterns {
    /// List of patterns directly extracted from the input string (not constraints)
    pub list: Vec<Pattern>,
    /// Map of variable names (A-Z) to their associated constraints
    pub var_constraints: VarConstraints,
    /// Reordered list of patterns, optimized for solving (most-constrained first)
    pub ordered_list: Vec<Pattern>,
}

impl Patterns {
    pub fn of(input: &str) -> Self {
        let mut patterns = Patterns::default();
        patterns.make_list(input);
        patterns.ordered_list = patterns.ordered_partitions();
        patterns
    }

    /// Parses the input string into constraint entries and pattern entries.
    /// Recognizes:
    /// - exact length (e.g., `|A|=5`)
    /// - inequality constraints (e.g., `!=AB`)
    /// - complex constraints (e.g., length + pattern) (e.g., `A=(3-5:a*)`)
    ///
    /// Non-constraint entries are added to `self.list` as actual patterns.
    fn make_list(&mut self, input: &str) {
        let forms: Vec<&str> = input.split(';').collect(); // TODO make ';' not magic constant
        // Iterate through all parts of the input string, split by `;`
        for form in &forms {
            if let Some(cap) = LEN_RE.captures(form).unwrap() {
                // Extract the variable (e.g., A) and its required length
                let var = cap[1].chars().next().unwrap();
                let len = cap[2].parse::<usize>().unwrap();
                self.var_constraints.ensure(var).set_exact_len(len); // TODO avoid mutability?
            } else if let Some(cap) = NEQ_RE.captures(form).unwrap() {
                // Extract all variables from inequality constraint (e.g., !=AB means A != B) // TODO document !=ABC (etc.) case
                let vars: Vec<char> = cap[1].chars().collect();
                for &v in &vars {
                    let var_constraint = self.var_constraints.ensure(v);
                    var_constraint.not_equal = vars.iter().copied().filter(|&x| x != v).collect();
                }
            } else if let Some(cap) = COMPLEX_RE.captures(form).unwrap() {
                // Extract variable and complex constraint info
                let var = cap[1].chars().next().unwrap();
                let len = &cap[2];
                let patt = cap[3].to_string();
                let var_constraint = self.var_constraints.ensure(var);

                if let Some((min, max)) = parse_length_range(len) {
                    var_constraint.min_length = min.unwrap();
                    var_constraint.max_length = max.unwrap();
                } else {
                    // TODO error here... though also handle the no-length-specified case correctly
                }

                if !patt.is_empty() && patt != "*" {
                    var_constraint.form = Some(patt);
                }
            } else {
                self.list.push(Pattern::of(*form));
            }
        }
    }

    // TODO is this the right way to order things?
    /// Reorders the list of patterns to improve solving efficiency.
    /// First selects the pattern with the most variables,
    /// then repeatedly selects the next pattern with the most overlap with those already chosen.
    /// This ensures early patterns can help prune the solution space. // TODO is "ensures" correct?
    fn ordered_partitions(&self) -> Vec<Pattern> {
        let mut patt_list = self.list.clone();
        let mut ordered = Vec::new();

        // Find the index of the pattern with the most variables
        let first_ix = patt_list
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| p.variables().len())
            .map(|(i, _)| i)
            .unwrap();

        // Start the ordered list with that most-variable-rich pattern
        let first = patt_list.remove(first_ix);
        ordered.push(first);

        while !patt_list.is_empty() {
            // Collect all variables used in the ordered patterns so far
            let found_vars: HashSet<char> = ordered.iter().flat_map(Pattern::variables).collect();

            // Find the pattern that shares the most variables with `found_vars`
            let (ix, mut next) = patt_list
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    let overlap = p.variables().intersection(&found_vars).count();
                    (i, overlap)
                })
                .max_by_key(|&(_, count)| count)
                .map(|(i, _)| (i, patt_list[i].clone()))
                .unwrap();

            let lookup_keys = next
                .variables()
                .intersection(&found_vars)
                .copied()
                .collect();
            next.lookup_keys = Some(lookup_keys);
            patt_list.remove(ix);
            ordered.push(next);
        }

        ordered
    }

    /// Number of forms (from `ordered_list`)
    pub fn len(&self) -> usize {
        self.ordered_list.len()
    }

    /// Convenience (often handy with `len`)
    pub fn is_empty(&self) -> bool {
        self.ordered_list.is_empty()
    }

    /// Iterate over forms in solver-friendly order
    pub fn iter(&self) -> std::slice::Iter<'_, Pattern> {
        self.ordered_list.iter()
    }
}

/// Enable `for p in &patterns { ... }`.
///
/// Why `&Patterns` and not `Patterns`?
/// - `for x in collection` desugars to `IntoIterator::into_iter(collection)`.
/// - If we implement `IntoIterator` for **`Patterns`**, iteration would *consume* (move) the
///   whole `Patterns`, which we don't want here.
/// - Implementing it for **`&Patterns`** lets you iterate **by reference** without moving.
impl<'a> IntoIterator for &'a Patterns {
    type Item = &'a Pattern;
    type IntoIter = std::slice::Iter<'a, Pattern>;

    fn into_iter(self) -> Self::IntoIter {
        // Delegate to the slice iterator over the underlying Vec
        self.ordered_list.iter()
    }
}

/// Parses a string like "3-5", "-5", "3-", or "3" into min and max length values.
/// Returns `Some((min, max))` where each is an `Option<usize>` unless the input is empty, in which
/// case it returns `None`.
fn parse_length_range(input: &str) -> Option<(Option<usize>, Option<usize>)> {
    if input.is_empty() {
        return None;
    }
    let parts: Vec<&str> = input.split('-').collect();
    if parts.len() > 2 { // TODO? return error instead?
        return None;
    }
    let min = parts.first().and_then(|s| s.parse::<usize>().ok());
    let max = parts.last().and_then(|s| s.parse::<usize>().ok());
    Some((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_pattern_and_constraints() {
        let input = "AB;|A|=3;!=AB;B=(2:b*)";
        let patterns = Patterns::of(input);

        println!("{:?}", patterns);

        // Test raw pattern list
        assert_eq!(1, patterns.list.len());
        assert_eq!("AB", patterns.list[0].raw_string);

        // Test constraints
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(3, a.min_length);
        assert_eq!(3, a.max_length);
        let set_1: HashSet<char> = ['B'].into_iter().collect();
        assert_eq!(set_1, a.not_equal);

        let b = patterns.var_constraints.get('B').unwrap();
        assert_eq!(2, b.min_length);
        assert_eq!(2, b.max_length);
        assert_eq!(Some("b*"), b.form.as_deref());
        let set_2: HashSet<char> = ['A'].into_iter().collect();
        assert_eq!(set_2, b.not_equal);
    }

    #[test]
    fn test_ordered_partitioning() {
        let input = "ABC;BC;C";
        let patterns = Patterns::of(input);

        let vars0 = patterns.ordered_list[0].variables();
        let vars1 = patterns.ordered_list[1].variables();
        let vars2 = patterns.ordered_list[2].variables();

        assert!(vars0.len() >= vars1.len());
        assert!(vars1.intersection(&vars0).count() >= vars2.intersection(&vars0).count());
    }

    #[test]
    fn test_parse_length_range() {
        assert_eq!(Some((Some(2), Some(3))), parse_length_range("2-3"));
        assert_eq!(Some((None, Some(3))), parse_length_range("-3"));
        assert_eq!(Some((Some(1), None)), parse_length_range("1-"));
        assert_eq!(Some((Some(7), Some(7))), parse_length_range("7"));
        assert_eq!(None, parse_length_range(""));
        assert_eq!(None, parse_length_range("1-2-3"));
    }
}
