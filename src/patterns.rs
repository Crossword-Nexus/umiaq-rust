use regex::Regex;
use std::collections::{HashMap, HashSet};
use std::sync::LazyLock;
use crate::constraints::VarConstraint;

/// Matches exact length constraints like `|A|=5`
static LEN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\|([A-Z])\|=(\d+)$").unwrap());

/// Matches inequality constraints like `!=AB`
static NEQ_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^!=([A-Z]+)$").unwrap());

/// Matches complex constraints like `A=(3-5:a*)` with optional length and pattern
static COMPLEX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^([A-Z])=\(?([\d\-]*):?([^)]*)\)?$").unwrap());

#[derive(Debug, Clone)]
/// Represents a single pattern (e.g., "AB", "A=(3:a*)") extracted from input
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
    pub fn new(string: impl Into<String>) -> Self {
        Self {
            raw_string: string.into(),
            lookup_keys: None,
        }
    }

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
/// A container for all parsed patterns and variable constraints
pub struct Patterns {
    /// List of patterns directly extracted from the input string (not constraints)
    pub list: Vec<Pattern>,
    /// Map of variable names (A-Z) to their associated constraints
    pub var_constraints: HashMap<char, VarConstraint>,
    /// Reordered list of patterns, optimized for solving (most-constrained first)
    pub ordered_list: Vec<Pattern>,
}

impl Patterns {
    pub fn new(input: &str) -> Self {
        let mut patterns = Patterns::default();
        patterns.make_list(input);
        patterns.ordered_list = patterns.ordered_partitions();
        patterns
    }

    /// Parses the input string into constraint entries and pattern entries.
    /// Recognizes:
    /// - exact length (e.g., `|A|=5`)
    /// - inequality constraints (e.g., `!=AB`)
    /// - complex constraints (length + pattern) (e.g., `A=(3-5:a*)`)
    /// Non-constraint entries are added to `self.list` as actual patterns.
    fn make_list(&mut self, input: &str) {
        let parts: Vec<&str> = input.split(';').collect();
        // Iterate through all parts of the input string, split by `;`
        for part in &parts {
            if let Some(cap) = LEN_RE.captures(part) {
                // Extract the variable (e.g., A) and its required length
                let var = cap[1].chars().next().unwrap();
                let len = cap[2].parse::<usize>().unwrap();
                let entry = self
                    .var_constraints
                    .entry(var)
                    .or_insert_with(VarConstraint::default);
                entry.min_length = Some(len);
                entry.max_length = Some(len);
            } else if let Some(cap) = NEQ_RE.captures(part) {
                // Extract all variables from inequality constraint (e.g., !=AB means A != B)
                let vars: Vec<char> = cap[1].chars().collect();
                for &v in &vars {
                    let entry = self
                        .var_constraints
                        .entry(v)
                        .or_insert_with(VarConstraint::default);
                    entry.not_equal = vars.iter().copied().filter(|&x| x != v).collect();
                }
            } else if let Some(cap) = COMPLEX_RE.captures(part) {
                // Extract variable and complex constraint info
                let var = cap[1].chars().next().unwrap();
                let len = &cap[2];
                let patt = cap[3].to_string();
                let entry = self
                    .var_constraints
                    .entry(var)
                    .or_insert_with(VarConstraint::default);

                if let Some((min, max)) = parse_length_range(len) {
                    entry.min_length = min;
                    entry.max_length = max;
                }

                if !patt.is_empty() && patt != "*" {
                    entry.form = Some(patt);
                }
            } else {
                self.list.push(Pattern::new(*part));
            }
        }
    }

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
        let patterns = Patterns::new(input);

        println!("{:?}", patterns);

        // Test raw pattern list
        assert_eq!(patterns.list.len(), 1);
        assert_eq!(patterns.list[0].raw_string, "AB");

        // Test constraints
        let a = patterns.var_constraints.get(&'A').unwrap();
        assert_eq!(a.min_length, Some(3));
        assert_eq!(a.max_length, Some(3));
        let set_1: HashSet<char> = ['B'].into_iter().collect();
        assert_eq!(a.not_equal, set_1);

        let b = patterns.var_constraints.get(&'B').unwrap();
        assert_eq!(b.min_length, Some(2));
        assert_eq!(b.max_length, Some(2));
        assert_eq!(b.form.as_deref(), Some("b*"));
        let set_2: HashSet<char> = ['A'].into_iter().collect();
        assert_eq!(b.not_equal, set_2);
    }

    #[test]
    fn test_ordered_partitioning() {
        let input = "ABC;BC;C";
        let patterns = Patterns::new(input);

        let vars0 = patterns.ordered_list[0].variables();
        let vars1 = patterns.ordered_list[1].variables();
        let vars2 = patterns.ordered_list[2].variables();

        assert!(vars0.len() >= vars1.len());
        assert!(vars1.intersection(&vars0).count() >= vars2.intersection(&vars0).count());
    }

    #[test]
    fn test_parse_length_range() {
        assert_eq!(parse_length_range("2-3"), Some((Some(2), Some(3))));
        assert_eq!(parse_length_range("-3"), Some((None, Some(3))));
        assert_eq!(parse_length_range("1-"), Some((Some(1), None)));
        assert_eq!(parse_length_range("7"), Some((Some(7), Some(7))));
        assert_eq!(parse_length_range(""), None);
        assert_eq!(parse_length_range("1-2-3"), None);
    }
}
