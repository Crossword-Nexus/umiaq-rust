use crate::constraints::{VarConstraint, VarConstraints};
use crate::parser::{parse_form, ParseError};
use fancy_regex::Regex;
use std::cmp::Reverse;
use std::collections::HashSet;
use std::sync::LazyLock;

/// The character that separates forms, in an equation
pub const FORM_SEPARATOR: char = ';';

/// Matches exact length constraints like `|A|=5`
static LEN_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\|([A-Z])\|=(\d+)$").unwrap());

/// Matches inequality constraints like `!=AB`
static NEQ_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^!=([A-Z]+)$").unwrap());

// TODO? disallow accepting one paren and not the other
// TODO require colon if both types, require no colon if just one (but support both or just one)
/// Matches complex constraints like `A=(3-5:a*)` with length and/or pattern
// syntax:
//
// complex constraint expression = {variable name}={constraint}
// variable name = A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z
// constraint = ({inner constraint})
//            | inner_constraint
// inner_constraint = {length range}:{literal string}
//                  | {length range}
//                  | {literal string}
// length range = {number}-{number}
//              | {number}-
//              | -{number}
//              | {number}
// number = {digit}
//        | {digit}{number}
// digit = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
// literal string = {literal string component}
//                | {literal string component}{literal string}
// literal string component = {literal string character}
//                          | {dot char}
//                          | {star char}
//                          | {vowel char}
//                          | {consonant char}
//                          | {charset string}
//                          | {anagram string}
// literal string char = a | b | c | d | e | f | g | h | i | j | k | l | m | n | o | p | q | r | s | t | u | v | w | x | y | z
// dot char = .
// star char = *
// vowel char = @
// consonant char = #
// charset string = [{one or more literal string chars}]
// one or more literal string chars = {literal string char}
//               | {literal string char}{charset chars}
// anagram string = /{one or more literal string chars}
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
    /// Position of this form among *forms only* in the original (display) order.
    /// This is stable and survives reordering/cloning.
    pub original_index: usize,
    /// Determine whether the string is deterministic. Created on init.
    pub(crate) is_deterministic: bool,
    /// The set of variables present in the pattern
    pub(crate) variables: HashSet<char>,
}

/// Implementation for the `Pattern` struct, representing a single pattern string
/// and utilities to extract its variable set.
impl Pattern {
    /// Constructs a new `Pattern` from any type that can be converted into a `String`.
    /// The resulting `lookup_keys` is initialized to `None`.
    /// `original_index` is the index for this `Pattern`'s position in the original equation.
    fn create(string: impl Into<String>, original_index: usize) -> Self {
        let raw_string = string.into();
        // Determine if the pattern is deterministic
        let deterministic = parse_form(&raw_string)
            .map(|parts| { parts.iter().all(|p| { p.is_deterministic() }) })
            .unwrap_or(false);

        // Get the variables involved
        let vars = raw_string
            .chars()
            .filter(char::is_ascii_uppercase)
            .collect();

        Self {
            raw_string,
            lookup_keys: None,
            original_index,
            is_deterministic: deterministic,
            variables: vars,
        }
    }

    /// True iff every variable this pattern uses is included in its `lookup_keys`.
    /// (If `lookup_keys` is `None`, only patterns with zero variables return true.)
    pub fn all_vars_in_lookup_keys(&self) -> bool {
        match &self.lookup_keys {
            Some(keys) => self.variables.is_subset(keys),
            None => self.variables.is_empty(),
        }
    }

    /// Get the "constraint score" (name?) of a pattern
    /// The more literals and @# it has, the more constrained it is
    fn constraint_score(&self) -> usize {
        let s = &self.raw_string;
        s.chars()
            .map(|c| {
                if c.is_ascii_lowercase() {
                    3 // TODO avoid magic constants (i.e., name this)
                } else if c == '@' || c == '#' {
                    1 // TODO avoid magic constants (i.e., name this)
                } else {
                    0 // TODO avoid magic constants (i.e., name this)
                }
            })
            .sum()
    }

    /// Set the lookup keys
    pub fn set_lookup_keys(&mut self, keys: HashSet<char>) {
        self.lookup_keys = Some(keys);
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
    // TODO should we keep Vec<Pattern> for each order or just one (likely ordered_list) and use map
    //      (original_to_ordered) when other is needed?
    pub list: Vec<Pattern>,
    /// Map of variable names (A-Z) to their associated constraints
    pub var_constraints: VarConstraints,
    /// Reordered list of patterns, optimized for solving (most-constrained first)
    pub ordered_list: Vec<Pattern>,        // solver order
    /// ordered index -> original index
    pub ordered_to_original: Vec<usize>,
    /// original index -> ordered index
    pub original_to_ordered: Vec<usize>,
}

impl Patterns {
    pub(crate) fn of(input: &str) -> Self {
        let mut patterns = Patterns::default();
        patterns.make_list(input);
        patterns.ordered_list = patterns.ordered_partitions();
        // populate original_to_ordered and ordered_to_original
        patterns.build_order_maps();
        patterns
    }

    fn build_order_maps(&mut self) {
        let n = self.list.len();
        self.ordered_to_original = self
            .ordered_list
            .iter()
            .map(|p| p.original_index)
            .collect();

        self.original_to_ordered = vec![usize::MAX; n];
        for (ordered_ix, &orig_ix) in self.ordered_to_original.iter().enumerate() {
            self.original_to_ordered[orig_ix] = ordered_ix;
        }
    }

    /// Parses the input string into constraint entries and pattern entries.
    /// Recognizes:
    /// - exact length (e.g., `|A|=5`)
    /// - inequality constraints (e.g., `!=AB`)
    /// - complex constraints (e.g., length + pattern) (e.g., `A=(3-5:a*)`)
    ///
    /// Non-constraint entries are added to `self.list` as actual patterns.
    fn make_list(&mut self, input: &str) {
        let forms: Vec<&str> = input.split(FORM_SEPARATOR).collect();
        // Iterate through all parts of the input string, split by `;`

        let mut next_form_ix = 0; // counts only *forms* we accept

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
            } else if let Ok((var, cc_vc)) = get_complex_constraint(form) {
                let var_constraint = self.var_constraints.ensure(var);

                // TODO is there a better way to do this?
                var_constraint.min_length = cc_vc.min_length;
                var_constraint.max_length = cc_vc.max_length;
                var_constraint.form = cc_vc.form;
            } else { // TODO? avoid swallowing error?
                // We only want to add a form if it is parseable
                // Specifically, things like |AB|=7 should not be picked up here
                // TODO do we check for those separately?
                // TODO avoid calling parse_form twice on the same form? (here and in solve_equation)
                if let Ok(_parsed) = parse_form(form) {
                    self.list.push(Pattern::create(*form, next_form_ix));
                    next_form_ix += 1;
                } else {
                    // TODO throw exception
                }
            }
        }
    }

    // TODO is this the right way to order things?
    /// Reorders the list of patterns to improve solving efficiency.
    /// First selects the pattern with the most variables,
    /// then repeatedly selects the next pattern with the most overlap with those already chosen.
    /// This helps early patterns prune the solution space.
    fn ordered_partitions(&self) -> Vec<Pattern> {
        let mut patt_list = self.list.clone();
        let mut ordered = Vec::with_capacity(patt_list.len());

        // Reusable tiebreak tail: (constraint_score desc, deterministic asc, original_index desc)
        // Note: Reverse(bool) makes false > true under max_by_key, i.e., ascending by bool.
        let tie_tail = |p: &Pattern| (p.constraint_score(), Reverse(p.is_deterministic), Reverse(p.original_index));

        // First pick: most variables; tiebreak by tail.
        let first_ix = patt_list
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| (p.variables.len(), tie_tail(p)))
            .map(|(i, _)| i)
            .unwrap();

        let first = patt_list.remove(first_ix);
        ordered.push(first);

        while !patt_list.is_empty() {
            // Vars already “seen”
            let found_vars: HashSet<char> = ordered
                .iter()
                .flat_map(|p| p.variables.iter().copied())
                .collect();

            // Next pick: maximize overlap; tiebreak by tail.
            let (ix, mut next) = patt_list
                .iter()
                .enumerate()
                .max_by_key(|(_, p)| {
                    let overlap = p.variables.intersection(&found_vars).count();
                    (overlap, tie_tail(p))
                })
                .map(|(i, p)| (i, p.clone()))
                .unwrap();

            // Assign join keys for the chosen pattern
            let lookup_keys: HashSet<char> = next.variables
                .intersection(&found_vars)
                .copied()
                .collect();
            next.set_lookup_keys(lookup_keys);

            patt_list.remove(ix);
            ordered.push(next);
        }

        ordered
    }

    /// Number of forms (from `ordered_list`)
    pub(crate) fn len(&self) -> usize {
        self.ordered_list.len()
    }

    /// Iterate over forms in solver-friendly order
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, Pattern> {
        self.ordered_list.iter()
    }

    /* -- Several unused functions but maybe some day?
    /// Convenience (often handy with `len`)
    fn is_empty(&self) -> bool {
        self.ordered_list.is_empty()
    }

    /// Iterate in original (display) order
    pub(crate) fn iter_original(&self) -> std::slice::Iter<'_, Pattern> {
        self.list.iter()
    }

    /// Map a solver index to the original index
    pub(crate) fn original_ix(&self, ordered_ix: usize) -> usize {
        self.ordered_to_original[ordered_ix]
    }

    /// Map an original index to the solver index (if it was placed)
    pub(crate) fn ordered_ix(&self, original_ix: usize) -> Option<usize> {
        let ix = self.original_to_ordered.get(original_ix).copied()?;
        (ix != usize::MAX).then_some(ix)
    }
    */
}

// TODO? do this via regex?
// e.g., A=(3-:x*)
fn get_complex_constraint(form: &&str) -> Result<(char, VarConstraint), ParseError> {
    let top_parts = form.split('=').collect::<Vec<_>>();
    if top_parts.len() != 2 {
        return Err(ParseError::InvalidComplexConstraint { str: format!("expected 1 equals sign (not {})", top_parts.len()) });
    }

    let var_str = top_parts[0];
    if var_str.len() != 1 {
        return Err(ParseError::InvalidComplexConstraint { str: format!("expected 1 character (as the variable) to the left of \"=\" (not {})", var_str.len()) });
    }

    let var = var_str.chars().next().unwrap();

    let constraint_str = top_parts[1].to_string();

    // remove outer parentheses if they are there
    let inner_constraint_str = if constraint_str.starts_with('(') && constraint_str.ends_with(')') {
        let mut chars = constraint_str.chars();
        chars.next();
        chars.next_back();
        chars.as_str()
    } else {
        constraint_str.as_str()
    };

    let constraint_halves = inner_constraint_str.split(':').collect::<Vec<_>>();
    let (len_range, literal_constraint_str) = match constraint_halves.len() {
        2 => {
            let len_range = parse_length_range(constraint_halves[0])?;
            (Some(len_range), Some(constraint_halves[1]))
        },
        1 => {
            match parse_length_range(constraint_halves[0]) {
                Ok(len_range) => (Some(len_range), None),
                Err(_) => (None, Some(constraint_halves[0]))
            }
        }
        _ => return Err(ParseError::InvalidComplexConstraint { str: format!("too many colons--0 or 1 expected (not {})", constraint_halves.len() - 1) })
    };

    let vc = VarConstraint {
        min_length: len_range.and_then(|lr| lr.0),
        max_length: len_range.and_then(|lr| lr.1),
        form: literal_constraint_str.map(ToString::to_string),
        not_equal: HashSet::default(),
    };

    Ok((var, vc))
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
/// Returns `((min, max))` where each is an `Option<usize>`.
fn parse_length_range(input: &str) -> Result<(Option<usize>, Option<usize>), ParseError> {
    let parts: Vec<_> = input.split('-').map(|part| part.parse::<usize>().ok()).collect();
    if (parts.len() == 1 && parts[0].is_none()) || parts.len() > 2 {
        return Err(ParseError::InvalidLengthRange { input: input.to_string() })
    }
    let min = *parts.first().unwrap();
    let max = *parts.last().unwrap();
    Ok((min, max))
}

#[cfg(test)]
mod tests {
    use crate::constraints::VarConstraint;
    use super::*;

    #[test]
    fn test_basic_pattern_and_constraints() {
        let patterns = Patterns::of("AB;|A|=3;!=AB;B=(2:b*)");

        // Test raw pattern list
        assert_eq!(vec!["AB".to_string()], patterns.list.iter().map(|p| p.raw_string.clone()).collect::<Vec<_>>());

        // Test constraints
        let a = patterns.var_constraints.get('A').unwrap();

        let expected_a = VarConstraint {
            min_length: Some(3),
            max_length: Some(3),
            form: None,
            not_equal: ['B'].into_iter().collect(),
        };
        assert_eq!(expected_a, a.clone());

        let b = patterns.var_constraints.get('B').unwrap();
        let expected_b = VarConstraint {
            min_length: Some(2),
            max_length: Some(2),
            form: Some("b*".to_string()),
            not_equal: ['A'].into_iter().collect(),
        };
        assert_eq!(expected_b, b.clone());
    }

    #[test]
    fn test_complex_re_len_only() {
        let patterns = Patterns::of("A;A=(6)");

        let expected = VarConstraint {
            min_length: Some(6),
            max_length: Some(6),
            form: None,
            not_equal: Default::default(),
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_lit_only() {
        let patterns = Patterns::of("A;A=(g*)");

        let expected = VarConstraint {
            min_length: None,
            max_length: None,
            form: Some("g*".to_string()),
            not_equal: Default::default(),
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }
    #[test]
    fn test_complex_re() {
        let patterns = Patterns::of("A;A=(3-4:x*)");

        let expected = VarConstraint {
            min_length: Some(3),
            max_length: Some(4),
            form: Some("x*".to_string()),
            not_equal: Default::default(),
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_unbounded_max_len() {
        let patterns = Patterns::of("A;A=(3-:x*)");

        let expected = VarConstraint {
            min_length: Some(3),
            max_length: None,
            form: Some("x*".to_string()),
            not_equal: Default::default(),
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_unbounded_min_len() {
        let patterns = Patterns::of("A;A=(-4:x*)");

        let expected = VarConstraint {
            min_length: None,
            max_length: Some(4),
            form: Some("x*".to_string()),
            not_equal: Default::default(),
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_complex_re_exact_len() {
        let patterns = Patterns::of("A;A=(6:x*)");

        let expected = VarConstraint {
            min_length: Some(6),
            max_length: Some(6),
            form: Some("x*".to_string()),
            not_equal: Default::default(),
        };
        assert_eq!(expected, patterns.var_constraints.get('A').unwrap().clone());
    }

    #[test]
    fn test_ordered_partitioning() {
        let input = "ABC;BC;C";
        let patterns = Patterns::of(input);

        let vars: Vec<HashSet<char>> = patterns.ordered_list.iter().map(|p| p.variables.clone()).collect();

        assert!((&vars[0]).len() >= (&vars[1]).len());
        assert!((&vars[1]).intersection(&vars[0]).count() >= (&vars[2]).intersection(&vars[0]).count());
    }

    #[test]
    fn test_parse_length_range() {
        assert_eq!((Some(2), Some(3)), parse_length_range("2-3").unwrap());
        assert_eq!((None, Some(3)), parse_length_range("-3").unwrap());
        assert_eq!((Some(1), None), parse_length_range("1-").unwrap());
        assert_eq!((Some(7), Some(7)), parse_length_range("7").unwrap());
        // TODO replace "_" with a more specific check (next two lines--and elsewhere... as appropriate)
        assert!(matches!(parse_length_range("").unwrap_err(), ParseError::InvalidLengthRange { input: _ }));
        assert!(matches!(parse_length_range("1-2-3").unwrap_err(), ParseError::InvalidLengthRange { input: _ }));
    }
}
