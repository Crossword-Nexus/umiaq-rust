use crate::constraints::VarConstraints;
use crate::parser::{parse_form, FormPart, ParseError};
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

static VAR_RE_STR: &str = "([A-Z])";
static LENGTH_RE_STR: &str = "(\\d+(-\\d+)?)";
// TODO constrain re to only allow lc letters, '.', '*', '/', '@', '#', etc. instead of "^)" in "[^)]"
static LIT_PATTERN_RE_STR: &str = "([^)]+)";

// TODO? disallow accepting one paren and not the other
// TODO require colon if both types, require no colon if just one (but support both or just one)
/// Matches complex constraints like `A=(3-5:a*)` with optional length and pattern
// syntax:
//
// complex constraint expression = {variable name}={constraint}
// variable name = A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V | W | X | Y | Z
// constraint = ({inner constraint})
//            | inner_constraint
// inner_constraint = {length range}:{literal string}
//                  | {length range}
//                  | {literal string}
// length range = {number}
//              | {number}-{number}
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
//
// group 1: var
// group 2: length constraint
// group 3 (ignored): hyphen plus end of length range (when present)
// group 4: literal pattern
static COMPLEX_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(&format!("^{VAR_RE_STR}=\\(?{LENGTH_RE_STR}:?{LIT_PATTERN_RE_STR}\\)?$")).unwrap());
// "^([A-Z])=(\\d+(-\\d+)?):?[^)]+$"
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
    deterministic: bool,
    /// The set of variables present in the pattern
    _variables: HashSet<char>,
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
            .map(|parts| {
                parts.iter().all(|p| {
                    matches!(p, FormPart::Var(_)
                                | FormPart::RevVar(_)
                                | FormPart::Lit(_))
                })
            })
            .unwrap_or(false);

        // Get the variables involved
        let _vars = raw_string
            .chars()
            .filter(char::is_ascii_uppercase)
            .collect();

        Self {
            raw_string,
            lookup_keys: None,
            original_index,
            deterministic,
            _variables: _vars,
        }
    }

    /// True iff every variable this pattern uses is included in its lookup_keys.
    /// (If lookup_keys is None, only patterns with zero variables return true.)
    pub fn all_vars_in_lookup_keys(&self) -> bool {
        match &self.lookup_keys {
            Some(keys) => self.variables().is_subset(keys),
            None => self.variables().is_empty(),
        }
    }

    /// Get the "constraint score" (name?) of a pattern
    /// The more literals and @# it has, the more constrained it is
    fn constraint_score(&self) -> usize {
        let s = &self.raw_string;
        s.chars()
            .map(|c| {
                if c.is_ascii_lowercase() {
                    3
                } else if c == '@' || c == '#' {
                    1
                } else {
                    0
                }
            })
            .sum()
    }


    /// Return the variables present in the pattern
    pub fn variables(&self) -> &HashSet<char> {
        &self._variables
    }

    /// A flag to tell us if the pattern is deterministic
    pub fn is_deterministic(&self) -> bool {
        self.deterministic
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
            } else if let Some(cap) = COMPLEX_RE.captures(form).unwrap() {
                // Extract variable and complex constraint info
                let var = cap[1].chars().next().unwrap();
                let len = &cap[2];
                let patt = cap[4].to_string();
                let var_constraint = self.var_constraints.ensure(var);

                if let Ok((min, max)) = parse_length_range(len) {
                    var_constraint.min_length = min.unwrap();
                    var_constraint.max_length = max.unwrap();
                } else {
                    // TODO error here... though also handle the no-length-specified case correctly
                }

                if !patt.is_empty() && patt != "*" {
                    var_constraint.form = Some(patt);
                }
            } else {
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

        // Reusable tie-break tail: (constraint_score desc, deterministic asc, original_index desc)
        // Note: Reverse(bool) makes false > true under max_by_key, i.e., ascending by bool.
        let tie_tail = |p: &Pattern| (p.constraint_score(), Reverse(p.deterministic), Reverse(p.original_index));

        // First pick: most variables; tiebreak by tail.
        let first_ix = patt_list
            .iter()
            .enumerate()
            .max_by_key(|(_, p)| (p.variables().len(), tie_tail(p)))
            .map(|(i, _)| i)
            .unwrap();

        let first = patt_list.remove(first_ix);
        ordered.push(first);

        while !patt_list.is_empty() {
            // Vars already “seen”
            let found_vars: HashSet<char> = ordered
                .iter()
                .flat_map(|p| p.variables().iter().copied())
                .collect();

            // Next pick: most overlap; tiebreak by tail.
            let (ix, mut next) = patt_list
                .iter()
                .enumerate()
                .max_by_key(|(_, p)| {
                    let overlap = p.variables().intersection(&found_vars).count();
                    (overlap, tie_tail(p))
                })
                .map(|(i, _)| (i, patt_list[i].clone()))
                .unwrap();

            // Assign join keys for the chosen pattern
            let lookup_keys: HashSet<char> = next
                .variables()
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
    let parts: Vec<&str> = input.split('-').collect();
    if (parts.len() == 1 && parts[0].is_empty()) || parts.len() > 2 {
        return Err(ParseError::InvalidLengthRange { input: input.parse().unwrap() })
    }
    let min = parts.first().and_then(|s| s.parse::<usize>().ok());
    let max = parts.last().and_then(|s| s.parse::<usize>().ok());
    Ok((min, max))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_pattern_and_constraints() {
        let patterns = Patterns::of("AB;|A|=3;!=AB;B=(2:b*)");

        // Test raw pattern list
        assert_eq!(vec!["AB".to_string()], patterns.list.iter().map(|p| p.raw_string.clone()).collect::<Vec<_>>());

        // Test constraints
        let a = patterns.var_constraints.get('A').unwrap();
        assert_eq!(3, a.min_length);
        assert_eq!(3, a.max_length);
        assert_eq!(['B'].into_iter().collect::<HashSet<_>>(), a.not_equal);

        let b = patterns.var_constraints.get('B').unwrap();
        assert_eq!(2, b.min_length);
        assert_eq!(2, b.max_length);
        assert_eq!(Some("b*"), b.form.as_deref());
        assert_eq!(['A'].into_iter().collect::<HashSet<_>>(), b.not_equal);
    }

    #[test]
    fn test_complex_re() {
        let patterns = Patterns::of("A;A=(3-4:x*)");

        let var_constraint = patterns.var_constraints.get('A').unwrap();
        assert_eq!(3, var_constraint.min_length);
        assert_eq!(4, var_constraint.max_length);
        assert_eq!(Some("x*"), var_constraint.form.as_deref());
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
        assert_eq!((Some(2), Some(3)), parse_length_range("2-3").unwrap());
        assert_eq!((None, Some(3)), parse_length_range("-3").unwrap());
        assert_eq!((Some(1), None), parse_length_range("1-").unwrap());
        assert_eq!((Some(7), Some(7)), parse_length_range("7").unwrap());
        // TODO replace "_" with a more specific check (next two lines--and elsewhere... as appropriate)
        assert!(matches!(parse_length_range("").unwrap_err(), ParseError::InvalidLengthRange { input: _ }));
        assert!(matches!(parse_length_range("1-2-3").unwrap_err(), ParseError::InvalidLengthRange { input: _ }));
    }
}
