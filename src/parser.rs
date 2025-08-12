use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::one_of,
    combinator::map,
    multi::many1,
    sequence::preceded,
    IResult,
    Parser,
};

use crate::bindings::Bindings;
use crate::constraints::{VarConstraint, VarConstraints};
use regex::Regex;
use std::fmt::Write as _;

// Character-set constants
const VOWELS: &str = "AEIOUY";
const CONSONANTS: &str = "BCDFGHJKLMNPQRSTVWXZ";
const UPPERCASE_ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const LOWERCASE_ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

/// Represents a single parsed token (component) from a "form" string.
///
/// Examples of forms:
/// - `l@#A*` (literal + vowel + consonant + variable + wildcard)
/// - `ABc/abc` (two variables + lowercase literal + an anagram)
///
/// Variants correspond to different token types:
#[derive(Debug, Clone, PartialEq)]
pub enum FormPart {
    Var(char),          // 'A': uppercase A–Z variable reference
    RevVar(char),       // '~A': reversed variable reference
    Lit(String),        // 'abc': literal lowercase sequence (will be uppercased internally)
    Dot,                // '.' wildcard: exactly one letter
    Star,               // '*' wildcard: zero or more letters
    Vowel,              // '@' wildcard: any vowel (AEIOUY)
    Consonant,          // '#' wildcard: any consonant (BCDF...XZ)
    Charset(Vec<char>), // '[abc]': any of the given letters
    Anagram(String),    // '/abc': any permutation of the given letters
}

/// Validate whether a candidate binding value is allowed under a `VarConstraint`.
///
/// Checks:
/// 1. If `form` is present, the value must itself match that form.
/// 2. The value must not equal any variable listed in `not_equal` that is already bound.
pub fn is_valid_binding(val: &str, constraints: &VarConstraint, bindings: &Bindings) -> bool {
    // 1. Apply nested form constraint if present
    if let Some(form_str) = &constraints.form {
        match parse_form(form_str) {
            Ok(p) => {
                if !match_equation_exists(val, &p, None) {
                    return false;
                }
            }
            Err(_) => return false, // If the nested form is invalid, reject
        }
    }

    // 2. Check "not equal" constraints
    for &other in &constraints.not_equal {
        if let Some(existing) = bindings.get(other) {
            if existing == val {
                return false;
            }
        }
    }

    true
}

/// Return the first binding set that satisfies the equation, or `None` if none match.
pub fn match_equation(
    word: &str,
    parts: &[FormPart],
    constraints: Option<&VarConstraints>,
) -> Option<Bindings> {
    let mut results = Vec::new();
    match_equation_internal(word, parts, false, &mut results, constraints);
    results.into_iter().next()
}

/// Return `true` if at least one binding satisfies the equation.
pub fn match_equation_exists(
    word: &str,
    parts: &[FormPart],
    constraints: Option<&VarConstraints>,
) -> bool {
    let mut results: Vec<Bindings> = Vec::new();
    match_equation_internal(word, parts, false, &mut results, constraints);
    results.into_iter().next().is_some()
}

/// Return all bindings that satisfy the equation.
pub fn match_equation_all(
    word: &str,
    parts: &[FormPart],
    constraints: Option<&VarConstraints>,
) -> Vec<Bindings> {
    let mut results: Vec<Bindings> = Vec::new();
    match_equation_internal(word, parts, true, &mut results, constraints);
    results
}

/// Core backtracking search that tries to match `word` against `parts`.
///
/// - Does an initial regex prefilter to skip impossible words quickly.
/// - Recursively attempts to bind variables and match literals/wildcards.
/// - Stops early if `all_matches` is false and a single match is found.
fn match_equation_internal(
    word: &str,
    parts: &[FormPart],
    all_matches: bool,
    results: &mut Vec<Bindings>,
    constraints: Option<&VarConstraints>,
) {
    /// Helper to reverse a bound value if the part is `RevVar`.
    fn get_reversed_or_not(first: &FormPart, val: &str) -> String {
        if matches!(first, FormPart::RevVar(_)) {
            val.chars().rev().collect::<String>()
        } else {
            val.to_owned()
        }
    }

    /// Recursive matching helper.
    ///
    /// `chars`       – remaining characters of the word
    /// `parts`       – remaining pattern parts
    /// `bindings`    – current variable assignments
    /// `results`     – collection of successful bindings
    /// `all_matches` – whether to collect all or stop at first
    fn helper(
        chars: &[char],
        parts: &[FormPart],
        bindings: &mut Bindings,
        results: &mut Vec<Bindings>,
        all_matches: bool,
        word: &str,
        constraints: Option<&VarConstraints>,
    ) -> bool {
        // Base case: no parts left
        if parts.is_empty() {
            if chars.is_empty() {
                let mut full_result = bindings.clone();
                full_result.set_word(word);
                results.push(full_result);
                return !all_matches; // Stop early if only one match needed
            }
            return false;
        }

        let (first, rest) = (&parts[0], &parts[1..]);

        match first {
            FormPart::Lit(s) => {
                // Literal match (case-insensitive, stored uppercased)
                let s = s.to_uppercase();
                if chars.starts_with(&s.chars().collect::<Vec<_>>()) {
                    return helper(&chars[s.len()..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            FormPart::Dot => {
                // Single-char wildcard
                if !chars.is_empty() {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            FormPart::Star => {
                // Zero-or-more wildcard; try all possible splits
                for i in 0..=chars.len() {
                    if helper(&chars[i..], rest, bindings, results, all_matches, word, constraints)
                        && !all_matches
                    {
                        return true;
                    }
                }
            }
            FormPart::Vowel => {
                if matches!(chars.first(), Some(c) if VOWELS.contains(*c)) {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            FormPart::Consonant => {
                if matches!(chars.first(), Some(c) if CONSONANTS.contains(*c)) {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            FormPart::Charset(set) => {
                if matches!(chars.first(), Some(c) if set.contains(&c.to_ascii_lowercase())) {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            FormPart::Anagram(s) => {
                // Match if the next N chars are an anagram of target
                let len = s.len();
                if chars.len() >= len {
                    let window: String = chars[..len].iter().collect();
                    let mut sorted_window: Vec<char> = window.chars().collect();
                    sorted_window.sort_unstable();
                    let mut sorted_target: Vec<char> = s.to_uppercase().chars().collect();
                    sorted_target.sort_unstable();
                    if sorted_window == sorted_target {
                        return helper(&chars[len..], rest, bindings, results, all_matches, word, constraints);
                    }
                }
            }
            FormPart::Var(name) | FormPart::RevVar(name) => {
                if let Some(bound_val) = bindings.get(*name) {
                    // Already bound: must match exactly
                    let val = get_reversed_or_not(first, bound_val);
                    if chars.starts_with(&val.chars().collect::<Vec<_>>()) {
                        return helper(&chars[val.len()..], rest, bindings, results, all_matches, word, constraints);
                    }
                } else {
                    // Not bound yet: try binding to all possible lengths
                    for l in 1..=chars.len() {
                        let candidate: String = chars[..l].iter().collect();
                        let bound_val = get_reversed_or_not(first, &candidate);

                        // Apply variable-specific constraints
                        let valid = if let Some(all_c) = constraints {
                            if let Some(c) = all_c.get(*name) {
                                is_valid_binding(&bound_val, c, bindings)
                            } else {
                                true
                            }
                        } else {
                            true
                        };

                        if !valid {
                            continue;
                        }

                        bindings.set(*name, bound_val);
                        if helper(&chars[l..], rest, bindings, results, all_matches, word, constraints) && !all_matches {
                            return true;
                        }
                        bindings.remove(*name);
                    }
                }
            }
        }

        false
    }

    // === PREFILTER STEP ===
    // Convert pattern to a regex and check before attempting backtracking.
    let regex_str_with_ends = format!("^{}$", form_to_regex_str(parts));
    if let Ok(regex) = Regex::new(&regex_str_with_ends) {
        if !regex.is_match(word) {
            return; // Fail fast
        }
    }

    // Normalize word and start recursive matching
    let word = word.to_uppercase();
    let chars: Vec<char> = word.chars().collect();
    let mut bindings = Bindings::default();
    helper(&chars, parts, &mut bindings, results, all_matches, &word, constraints);
}

/// Convert a parsed `FormPart` sequence into a regex string.
///
/// Used for the initial fast prefilter in `match_equation_internal`.
pub fn form_to_regex_str(parts: &[FormPart]) -> String {
    let mut regex_str = String::new();
    for part in parts {
        match part {
            FormPart::Var(_) | FormPart::RevVar(_) => regex_str.push_str(".+"), // Variable: one or more chars
            FormPart::Lit(s) => regex_str.push_str(&regex::escape(&s.to_uppercase())),
            FormPart::Dot => regex_str.push('.'),
            FormPart::Star => regex_str.push_str(".*"),
            FormPart::Vowel => regex_str.push_str(&format!("[{VOWELS}]")),
            FormPart::Consonant => regex_str.push_str(&format!("[{CONSONANTS}]")),
            FormPart::Charset(chars) => {
                regex_str.push('[');
                for c in chars {
                    regex_str.push(c.to_ascii_uppercase());
                }
                regex_str.push(']');
            }
            FormPart::Anagram(s) => {
                let len = s.len();
                let class = regex::escape(&s.to_uppercase());
                let _ = write!(regex_str, "[{class}]{{{len}}}");
            }
        }
    }
    regex_str
}

/// Parse a form string into a `Vec<FormPart>` sequence.
///
/// Walks the input, consuming tokens one at a time with `equation_part`.
pub fn parse_form(input: &str) -> Result<Vec<FormPart>, String> {
    let mut rest = input;
    let mut parts = Vec::new();

    while !rest.is_empty() {
        match equation_part(rest) {
            Ok((next, part)) => {
                parts.push(part);
                rest = next;
            }
            Err(_) => return Err(format!("Could not parse at: {rest}")),
        }
    }

    Ok(parts)
}

// === Token parsers ===
// These small functions use `nom` combinators to recognize individual token types.

fn varref(input: &str) -> IResult<&str, FormPart> {
    map(one_of(UPPERCASE_ALPHABET), FormPart::Var).parse(input)
}

fn revref(input: &str) -> IResult<&str, FormPart> {
    map(preceded(tag("~"), one_of(UPPERCASE_ALPHABET)), FormPart::RevVar).parse(input)
}

fn literal(input: &str) -> IResult<&str, FormPart> {
    map(many1(one_of(LOWERCASE_ALPHABET)), |chars| {
        FormPart::Lit(chars.into_iter().collect())
    }).parse(input)
}

fn dot(input: &str) -> IResult<&str, FormPart> {
    map(tag("."), |_| FormPart::Dot).parse(input)
}

fn star(input: &str) -> IResult<&str, FormPart> {
    map(tag("*"), |_| FormPart::Star).parse(input)
}

fn vowel(input: &str) -> IResult<&str, FormPart> {
    map(tag("@"), |_| FormPart::Vowel).parse(input)
}

fn consonant(input: &str) -> IResult<&str, FormPart> {
    map(tag("#"), |_| FormPart::Consonant).parse(input)
}

fn charset(input: &str) -> IResult<&str, FormPart> {
    let (input, _) = tag("[")(input)?;
    let (input, chars) = many1(one_of(LOWERCASE_ALPHABET)).parse(input)?;
    let (input, _) = tag("]")(input)?;
    Ok((input, FormPart::Charset(chars)))
}

fn anagram(input: &str) -> IResult<&str, FormPart> {
    let (input, _) = tag("/")(input)?;
    let (input, chars) = many1(one_of(LOWERCASE_ALPHABET)).parse(input)?;
    Ok((input, FormPart::Anagram(chars.into_iter().collect())))
}

/// Try parsing any valid token from the input.
fn equation_part(input: &str) -> IResult<&str, FormPart> {
    alt((revref, varref, anagram, charset, literal, dot, star, vowel, consonant))
    .parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_equation_exists() {
        let patt1 = parse_form("A~A[rstlne]/jon@#.*").unwrap();
        assert!(match_equation_exists("AARONJUDGE", &patt1, None));
        assert!(!match_equation_exists("NOON", &patt1, None));
        assert!(!match_equation_exists("TOON", &patt1, None));
    }

    #[test]
    fn test_valid_binding_not_equal_pass() {
        // A must be != B; current B is bound to "TEST"
        let mut vc = VarConstraint::default();
        vc.not_equal.insert('B');

        let mut b = Bindings::default();
        b.set('B', "TEST".to_string());

        // "OTHER" != "TEST" so this should pass
        assert!(is_valid_binding("OTHER", &vc, &b));
    }

    #[test]
    fn test_valid_binding_simple_pass() {
        let mut vc = VarConstraint::default();
        vc.form = Option::from("abc*".to_string());

        let b = Bindings::default();
        assert!(is_valid_binding("ABCAT", &vc, &b));
    }

    #[test]
    fn test_valid_binding_fail() {
        let mut vc = VarConstraint::default();
        vc.form = Option::from("abc*".to_string());

        let b = Bindings::default();
        assert!(!is_valid_binding("XYZ", &vc, &b));
    }

    #[test]
    fn test_valid_binding_not_equal_fail() {
        let mut vc = VarConstraint::default();
        vc.not_equal.insert('B');

        let mut b = Bindings::default();
        b.set('B', "TEST".to_string());
        assert!(!is_valid_binding("TEST", &vc, &b));
    }

    #[test]
    fn test_match_equation_with_constraints() {
        let patt = parse_form("AB").unwrap();
        // create a constraints holder
        let mut var_constraints = VarConstraints::default();
        // add !=AB
        // first, add it for A
        let mut vc_a = VarConstraint::default();
        vc_a.not_equal.insert('B');
        var_constraints.insert('A', vc_a);
        // now add it for B
        let mut vc_b = VarConstraint::default();
        vc_b.not_equal.insert('A');
        var_constraints.insert('B', vc_b);
        let result = match_equation("INCH", &patt, Some(&var_constraints));
        assert!(result.is_some());
        let m = result.unwrap();
        assert_ne!(m.get('A'), m.get('B'));
    }

    #[test]
    fn test_match_equation_all_with_constraints() {
        let patt = parse_form("AA").unwrap();
        // We add length constraints
        let mut var_constraints = VarConstraints::default();
        let mut vc = VarConstraint::default();
        const MIN_LENGTH: usize = 2;
        const MAX_LENGTH: usize = 3;
        // min length 2, max_length 3
        vc.min_length = MIN_LENGTH;
        vc.max_length = MAX_LENGTH;
        // associate this constraint with variable 'A'
        var_constraints.insert('A', vc);

        let matches = match_equation_all("HOTHOT", &patt, Some(&var_constraints));
        println!("{matches:?}");
        for m in matches.iter() {
            let val = m.get('A').unwrap();
            assert!(val.len() >= MIN_LENGTH && val.len() <= MAX_LENGTH);
        }
    }

    /// Test AB;|A|=2;|B|=2;!=AB on INCH
    #[test]
    fn test_match_equation_all_with_constraints2() {
        // Pattern
        let patt = parse_form("AB").unwrap();
        // Constraints
        let mut var_constraints = VarConstraints::default();
        // add !=AB
        // first, add it for A
        let mut vc_a = VarConstraint::default();
        vc_a.not_equal.insert('B');
        vc_a.min_length = 2;
        vc_a.max_length = 2;
        var_constraints.insert('A', vc_a);
        // now add it for B
        let mut vc_b = VarConstraint::default();
        vc_b.not_equal.insert('A');
        vc_b.min_length = 2;
        vc_b.max_length = 2;
        var_constraints.insert('B', vc_b);

        let matches = match_equation_all("INCH", &patt, Some(&var_constraints));
        println!("{matches:?}");
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_match_equation_exists_with_constraints() {
        let patt = parse_form("AB").unwrap();
        // First constraint: A=(*i.*)
        let mut var_constraints1 = VarConstraints::default();
        let mut vc = VarConstraint::default();
        vc.form = Option::from("*i.*".to_string());
        var_constraints1.insert('A', vc.clone());

        assert!(match_equation_exists("INCH", &patt, Some(&var_constraints1)));

        // Second constraint: A=(*z*)
        let mut var_constraints2 = VarConstraints::default();
        let mut vc2 = VarConstraint::default();
        vc2.form = Option::from("*z*".to_string());
        var_constraints2.insert('A', vc2.clone());

        assert!(!match_equation_exists("INCH", &patt, Some(&var_constraints2)));
    }

    #[test]
    fn test_parse_form_basic() {
        let result = parse_form("abc");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0], FormPart::Lit(ref s) if s == "abc"));
    }

    #[test]
    fn test_parse_form_variable() {
        let result = parse_form("A");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0], FormPart::Var('A')));
    }

    #[test]
    fn test_parse_form_reversed_variable() {
        let result = parse_form("~A");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0], FormPart::RevVar('A')));
    }

    #[test]
    fn test_parse_form_wildcards() {
        let result = parse_form(".*@#");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.len(), 4);
        assert!(matches!(parts[0], FormPart::Dot));
        assert!(matches!(parts[1], FormPart::Star));
        assert!(matches!(parts[2], FormPart::Vowel));
        assert!(matches!(parts[3], FormPart::Consonant));
    }

    #[test]
    fn test_parse_form_charset() {
        let result = parse_form("[abc]");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0], FormPart::Charset(ref chars) if chars == &['a', 'b', 'c']));
    }

    #[test]
    fn test_parse_form_anagram() {
        let result = parse_form("/abc");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.len(), 1);
        assert!(matches!(parts[0], FormPart::Anagram(ref s) if s == "abc"));
    }

    #[test]
    fn test_parse_form_complex() {
        let result = parse_form("A~A[rstlne]/jon@#.*");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.len(), 8);
        assert!(matches!(parts[0], FormPart::Var('A')));
        assert!(matches!(parts[1], FormPart::RevVar('A')));
        assert!(matches!(parts[2], FormPart::Charset(_))); // TODO check _ value
        assert!(matches!(parts[3], FormPart::Anagram(_))); // TODO check _ value
        assert!(matches!(parts[4], FormPart::Vowel));
        assert!(matches!(parts[5], FormPart::Consonant));
        assert!(matches!(parts[6], FormPart::Dot));
        assert!(matches!(parts[7], FormPart::Star));
    }

    #[test]
    fn test_form_to_regex_str() {
        let parts = parse_form("l.x").unwrap();
        let regex_str = form_to_regex_str(&parts);
        assert_eq!(regex_str, "L.X");
    }

    #[test]
    fn test_form_to_regex_str_with_variables() {
        let parts = parse_form("AB").unwrap();
        let regex_str = form_to_regex_str(&parts);
        assert_eq!(regex_str, ".+.+");
    }

    #[test]
    fn test_form_to_regex_str_with_wildcards() {
        let parts = parse_form(".*@#").unwrap();
        let regex_str = form_to_regex_str(&parts);
        assert_eq!(regex_str, "..*[AEIOUY][BCDFGHJKLMNPQRSTVWXZ]");
    }

    #[test]
    fn test_palindrome_matching() {
        let patt = parse_form("A~A").unwrap();
        assert!(match_equation_exists("NOON", &patt, None));
        assert!(!match_equation_exists("RADAR", &patt, None));
        assert!(!match_equation_exists("TEST", &patt, None));
    }

    #[test]
    fn test_anagram_matching() {
        let patt = parse_form("/triangle").unwrap();
        assert!(match_equation_exists("INTEGRAL", &patt, None));
        assert!(!match_equation_exists("SQUARE", &patt, None));
    }

    #[test]
    fn test_variable_binding() {
        let patt = parse_form("AB").unwrap();
        let result = match_equation("INCH", &patt, None);
        assert!(result.is_some());
        let binding = result.unwrap();
        // TODO allow for IN/CH or INC/H
        assert_eq!(binding.get('A'), Some(&"I".to_string()));
        assert_eq!(binding.get('B'), Some(&"NCH".to_string()));
    }

    #[test]
    fn test_reversed_variable_binding() {
        let patt = parse_form("A~A").unwrap();
        let result = match_equation("NOON", &patt, None);
        assert!(result.is_some());
        let binding = result.unwrap();
        assert_eq!(binding.get('A'), Some(&"NO".to_string()));
    }

    #[test]
    fn test_literal_matching() {
        let patt = parse_form("abc").unwrap();
        assert!(match_equation_exists("ABC", &patt, None));
        assert!(!match_equation_exists("XYZ", &patt, None));
    }

    #[test]
    fn test_dot_wildcard() {
        let patt = parse_form("a.z").unwrap();
        assert!(match_equation_exists("ABZ", &patt, None));
        assert!(match_equation_exists("AZZ", &patt, None));
        assert!(!match_equation_exists("AZ", &patt, None));
        assert!(!match_equation_exists("ABBZ", &patt, None));
    }

    #[test]
    fn test_star_wildcard() {
        let patt = parse_form("a*z").unwrap();
        assert!(match_equation_exists("AZ", &patt, None));
        assert!(match_equation_exists("ABZ", &patt, None));
        assert!(match_equation_exists("ABBBZ", &patt, None));
        assert!(!match_equation_exists("AY", &patt, None));
    }

    #[test]
    fn test_vowel_wildcard() {
        let patt = parse_form("A@Z").unwrap();
        assert!(match_equation_exists("AAZ", &patt, None));
        assert!(match_equation_exists("AEZ", &patt, None));
        assert!(!match_equation_exists("ABZ", &patt, None));
    }

    #[test]
    fn test_consonant_wildcard() {
        let patt = parse_form("A#Z").unwrap();
        assert!(match_equation_exists("ABZ", &patt, None));
        assert!(match_equation_exists("ACZ", &patt, None));
        assert!(!match_equation_exists("AAZ", &patt, None));
    }

    #[test]
    fn test_charset_matching() {
        let patt = parse_form("[abc]").unwrap();
        assert!(match_equation_exists("A", &patt, None));
        assert!(match_equation_exists("B", &patt, None));
        assert!(match_equation_exists("C", &patt, None));
        assert!(!match_equation_exists("D", &patt, None));
    }

    #[test]
    fn test_empty_pattern() {
        let result = parse_form("");
        assert!(result.is_ok());
        let parts = result.unwrap();
        assert_eq!(parts.len(), 0);
    }

    #[test]
    fn test_invalid_pattern() {
        let result = parse_form("[");
        assert!(result.is_err());
    }
}
