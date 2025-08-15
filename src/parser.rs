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
use fancy_regex::Regex;
use std::collections::HashSet;
use std::fmt::Write as _;
use std::sync::{LazyLock, OnceLock};

// Character-set constants
const VOWELS: &str = "AEIOUY";
const CONSONANTS: &str = "BCDFGHJKLMNPQRSTVWXZ";
const UPPERCASE_ALPHABET: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
const LOWERCASE_ALPHABET: &str = "abcdefghijklmnopqrstuvwxyz";

const NUM_POSSIBLE_VARIABLES: usize = 26; // TODO rename? TODO derive?

static VOWEL_SET: LazyLock<HashSet<char>> = LazyLock::new(|| VOWELS.chars().collect());
static CONSONANT_SET: LazyLock<HashSet<char>> = LazyLock::new(|| CONSONANTS.chars().collect());

static REGEX_CACHE: OnceLock<std::collections::HashMap<String, Regex>> = OnceLock::new();

fn get_regex(pattern: &str) -> Result<Regex, fancy_regex::Error> {
    let cache = REGEX_CACHE.get_or_init(std::collections::HashMap::new);

    if let Some(regex) = cache.get(pattern) {
        Ok(regex.clone())
    } else {
        Regex::new(pattern)
    }
}

/// Custom error type for parsing operations
#[derive(Debug, thiserror::Error)]
pub enum ParseError {
    #[error("Form parsing failed @ pos. {position}; remaining input: \"{remaining}\"")]
    ParseFailure { position: usize, remaining: String },
    #[error("Invalid regex pattern: {0}")]
    RegexError(#[from] fancy_regex::Error),
    #[error("Empty form string")]
    EmptyForm,
}

// TODO test this
// TODO is this the right way to handle things? (i.e., eventually convert `ParserError`s into `Error`s)
impl From<ParseError> for std::io::Error {
    fn from(pe: ParseError) -> Self {
        // TODO is this right??
        std::io::Error::new(std::io::ErrorKind::InvalidInput, pe.to_string())
    }
}

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

/// A `Vec` of `FormPart`s along with a compiled regex prefilter
#[derive(Debug)]
pub struct ParsedForm {
    pub parts: Vec<FormPart>,
    pub prefilter: Regex,
}

impl ParsedForm {
    fn of(parts: Vec<FormPart>) -> Result<Self, ParseError> {
        // Build the regex string
        let regex_str = form_to_regex_str(&parts);
        let anchored = format!("^{regex_str}$");
        let prefilter = get_regex(&anchored)?;

        Ok(ParsedForm { parts, prefilter })
    }
}

/// Validate whether a candidate binding value is allowed under a `VarConstraint`.
///
/// Checks:
/// 0. If "length" constraints are present, enforce them
/// 1. If `form` is present, the value must itself match that form.
/// 2. The value must not equal any variable listed in `not_equal` that is already bound.
fn is_valid_binding(val: &str, constraints: &VarConstraint, bindings: &Bindings) -> bool {
    // 0) Length checks (if configured)
    if constraints.min_length > 0 && val.len() < constraints.min_length {
        return false;
    }
    if constraints.max_length > 0 && val.len() > constraints.max_length {
        return false;
    }

    // 1) Apply nested form constraint if present
    if let Some(form_str) = &constraints.form {
        match parse_form(form_str) {
            Ok(p) => {
                if !match_equation_exists(val, &p, None) {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }

    // 2) Check "not equal" constraints
    for &other in &constraints.not_equal {
        if let Some(existing) = bindings.get(other) && existing == val {
            return false;
        }
    }

    true
}

/// Return `true` if at least one binding satisfies the equation.
fn match_equation_exists(
    word: &str,
    parts: &ParsedForm,
    constraints: Option<&VarConstraints>,
) -> bool {
    let mut results: Vec<Bindings> = Vec::new();
    match_equation_internal(word, parts, false, &mut results, constraints);
    results.into_iter().next().is_some()
}

/// Return all bindings that satisfy the equation.
pub(crate) fn match_equation_all(
    word: &str,
    parts: &ParsedForm,
    constraints: Option<&VarConstraints>,
) -> Vec<Bindings> {
    let mut results: Vec<Bindings> = Vec::new(); // TODO avoid mutability? sim. elsewhere
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
    parsed_form: &ParsedForm,
    all_matches: bool,
    results: &mut Vec<Bindings>,
    constraints: Option<&VarConstraints>,
) {
    /// Helper to reverse a bound value if the part is `RevVar`.
    fn get_reversed_or_not(first: &FormPart, val: &str) -> String {
        if matches!(first, FormPart::RevVar(_)) {
            val.chars().rev().collect()
        } else {
            val.to_owned()
        }
    }

    // TODO WTF does return value do (also: perhaps it should (always) be used)...
    // TODO maybe instead use a 3-way enum (e.g., can't continue, continue, done)
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
                return is_prefix(&s.to_ascii_uppercase(), &chars, bindings, results, all_matches, word, constraints, rest)
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
                if VOWEL_SET.contains(chars.first().unwrap()) {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            FormPart::Consonant => {
                if CONSONANT_SET.contains(chars.first().unwrap()) {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            FormPart::Charset(set) => {
                if set.contains(&chars.first().unwrap().to_ascii_lowercase()) { // TODO? avoid to_ascii_lowercase here (and elsewhere) by uppercasing things early
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            FormPart::Anagram(s) => {
                // Match if the next len chars are an anagram of target
                let len = s.len();
                if chars.len() >= len && are_anagrams(&chars[..len], s) {
                    return helper(&chars[len..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            FormPart::Var(var_name) | FormPart::RevVar(var_name) => {
                if let Some(bound_val) = bindings.get(*var_name) {
                    // Already bound: must match exactly
                    return is_prefix(&get_reversed_or_not(first, bound_val), &chars, bindings, results, all_matches, word, constraints, rest)
                }

                // Not bound yet: try binding to all possible lengths
                // To prune the search space, apply length constraints up front
                let mut min_len = 1usize;
                let mut max_len = chars.len(); // cannot take more than what’s left

                if let Some(all_c) = constraints && let Some(c) = all_c.get(*var_name) {
                    if c.min_length > 0 { min_len = min_len.max(c.min_length); }
                    if c.max_length > 0 { max_len = max_len.min(c.max_length); }
                }
                if min_len > max_len { return false; }

                for l in min_len..=max_len {
                    let candidate_chars = &chars[..l];

                    let bound_val = if matches!(first, FormPart::RevVar(_)) {
                        candidate_chars.iter().rev().collect::<String>()
                    } else {
                        candidate_chars.iter().collect::<String>()
                    };

                    // Apply variable-specific constraints
                    let valid = if let Some(all_c) = constraints {
                        if let Some(c) = all_c.get(*var_name) {
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

                    bindings.set(*var_name, bound_val);
                    if helper(&chars[l..], rest, bindings, results, all_matches, word, constraints) && !all_matches {
                        return true;
                    }
                    bindings.remove(*var_name);
                }
            }
        }

        false
    }

    /// Returns true if `prefix` is a prefix of `chars`
    fn is_prefix(prefix: &str, chars: &&[char], bindings: &mut Bindings, results: &mut Vec<Bindings>, all_matches: bool, word: &str, constraints: Option<&VarConstraints>, rest: &[FormPart]) -> bool {
        let n = prefix.len();

        if chars.len() >= n && chars[..n].iter().copied().zip(prefix.chars()).all(|(a, b)| a == b) {
            helper(&chars[n..], rest, bindings, results, all_matches, word, constraints)
        } else {
            false
        }
    }

    // TODO? support beyond 0-127 (i.e., beyond ASCII)? (at least document behavior (here and in general)!)
    // TODO are we actually guaranteed to have uppercase_word be upperase?
    fn are_anagrams(uppercase_word: &[char], other_word: &str) -> bool {
        if uppercase_word.len() != other_word.len() {
            return false;
        }

        let mut char_counts = [0u8; 128];

        for &c in uppercase_word {
            if (c as usize) < 128 {
                char_counts[c as usize] += 1;
            }
            // TODO? handle characters outside of 0-127 differently? (e.g., error vs. ignore, etc.)
        }

        for c in other_word.chars() {
            let c_upper = c.to_ascii_uppercase();
            if (c_upper as usize) < 128 {
                if char_counts[c_upper as usize] == 0 {
                    return false;
                }
                char_counts[c_upper as usize] -= 1;
            }
        }


        char_counts.iter().all(|&count| count == 0)
    }

    // === PREFILTER STEP ===
    if !parsed_form.prefilter.is_match(word).unwrap() {
        return;
    }

    // Normalize word and start recursive matching
    let word = word.to_uppercase(); // TODO perform uppercasing as early as possible
    let chars: Vec<char> = word.chars().collect();
    let mut bindings = Bindings::default();
    helper(&chars, &parsed_form.parts, &mut bindings, results, all_matches, &word, constraints);
}

/// Convert a parsed `FormPart` sequence into a regex string.
///
/// Used for the initial fast prefilter in `match_equation_internal`.
fn form_to_regex_str(parts: &[FormPart]) -> String {
    let (var_counts, rev_var_counts) = get_var_and_rev_var_counts(parts);
    let mut var_to_backreference_num = [0usize; NUM_POSSIBLE_VARIABLES];
    let mut rev_var_to_backreference_num = [0usize; NUM_POSSIBLE_VARIABLES];
    let mut backreference_index = 0usize;

    let mut regex_str = String::new();
    for part in parts {
        match part {
            FormPart::Var(c) => regex_str.push_str(&get_regex_str_segment(var_counts, &mut var_to_backreference_num, &mut backreference_index, *c)),
            FormPart::RevVar(c) => regex_str.push_str(&get_regex_str_segment(rev_var_counts, &mut rev_var_to_backreference_num, &mut backreference_index, *c)),
            FormPart::Lit(s) => regex_str.push_str(&fancy_regex::escape(&s.to_uppercase())),
            FormPart::Dot => regex_str.push('.'),
            FormPart::Star => regex_str.push_str(".*"),
            FormPart::Vowel => {
                let _ = write!(regex_str, "[{VOWELS}]");
            },
            FormPart::Consonant => {
                let _ = write!(regex_str, "[{CONSONANTS}]");
            },
            FormPart::Charset(chars) => {
                regex_str.push('[');
                for c in chars {
                    regex_str.push(c.to_ascii_uppercase());
                }
                regex_str.push(']');
            }
            FormPart::Anagram(s) => {
                let len = s.len();
                let s_upper = s.to_uppercase();
                let class = fancy_regex::escape(&s_upper);
                let _ = write!(regex_str, "[{class}]{{{len}}}");
            }
        }
    }

    regex_str
}

fn get_regex_str_segment(var_counts: [usize; NUM_POSSIBLE_VARIABLES], var_to_backreference_num: &mut [usize; NUM_POSSIBLE_VARIABLES], backreference_index: &mut usize, c: char) -> String {
    let char_as_num = char_to_num(c);
    let pushed_str = if var_to_backreference_num[char_as_num] != 0 {
        &format!("\\{}", var_to_backreference_num[char_as_num])
    } else if var_counts[char_as_num] > 1 {
        *backreference_index += 1;
        var_to_backreference_num[char_as_num] = *backreference_index;
        "(.+)"
    } else {
        ".+"
    };

    pushed_str.to_string()
}

// TODO doesn't really need to count--really only need to the return values to distinguish between
//      one and many (NB: in the zero case callers won't use what's returned here)
fn get_var_and_rev_var_counts(parts: &[FormPart]) -> ([usize; NUM_POSSIBLE_VARIABLES], [usize; NUM_POSSIBLE_VARIABLES]) {
    let mut var_counts = [0usize; NUM_POSSIBLE_VARIABLES];
    let mut rev_var_counts = [0usize; NUM_POSSIBLE_VARIABLES];
    for part in parts {
        match part {
            FormPart::Var(c) => var_counts[char_to_num(*c)] += 1,
            FormPart::RevVar(c) => rev_var_counts[char_to_num(*c)] += 1,
            _ => ()
        }
    }

    (var_counts, rev_var_counts)
}

// 'A' -> 0, 'B' -> 1, ..., 'Z' -> 25
fn char_to_num(c: char) -> usize {
    c as usize - 'A' as usize
}

/// Parse a form string into a `ParsedForm` object
///
/// Walks the input, consuming tokens one at a time with `equation_part`.
pub(crate) fn parse_form(raw_form: &str) -> Result<ParsedForm, ParseError> {
    let mut rest = raw_form;
    let mut parts = Vec::new(); // TODO? avoid mutability

    while !rest.is_empty() {
        match equation_part(rest) {
            Ok((next, part)) => { // TODO why not just replace "next" with "rest"
                parts.push(part);
                rest = next;
            }
            Err(_) => return Err(ParseError::ParseFailure { position: raw_form.len() - rest.len(), remaining: rest.to_string() }),
        }
    }

    if parts.is_empty() {
        return Err(ParseError::EmptyForm);
    }

    ParsedForm::of(parts)
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
    fn test_empty_form_error() {
        let actual = parse_form("");
        assert!(matches!(actual.unwrap_err(), ParseError::EmptyForm));
    }

    #[test]
    fn test_parse_failure_error() {
        let actual = parse_form("[");
        assert!(matches!(actual.unwrap_err(), ParseError::ParseFailure { .. }));
    }

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
        let result = match_equation_all("INCH", &patt, Some(&var_constraints)).into_iter().next();
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
        println!("{}", var_constraints);
        assert_eq!(1, matches.len());
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
        let parsed_form = result.unwrap();
        let parts = parsed_form.parts;
        assert_eq!(1, parts.len());
        assert!(matches!(parts[0], FormPart::Lit(ref s) if s == "abc"));
    }

    #[test]
    fn test_parse_form_variable() {
        let result = parse_form("A");
        assert!(result.is_ok());
        let parsed_form = result.unwrap();
        let parts = parsed_form.parts;
        assert_eq!(1, parts.len());
        assert!(matches!(parts[0], FormPart::Var('A')));
    }

    #[test]
    fn test_parse_form_reversed_variable() {
        let result = parse_form("~A");
        assert!(result.is_ok());
        let parsed_form = result.unwrap();
        let parts = parsed_form.parts;
        assert_eq!(1, parts.len());
        assert!(matches!(parts[0], FormPart::RevVar('A')));
    }

    #[test]
    fn test_parse_form_wildcards() {
        let result = parse_form(".*@#");
        assert!(result.is_ok());
        let parsed_form = result.unwrap();
        let parts = parsed_form.parts;
        assert_eq!(4, parts.len());
        assert!(matches!(parts[0], FormPart::Dot));
        assert!(matches!(parts[1], FormPart::Star));
        assert!(matches!(parts[2], FormPart::Vowel));
        assert!(matches!(parts[3], FormPart::Consonant));
    }

    #[test]
    fn test_parse_form_charset() {
        let result = parse_form("[abc]");
        assert!(result.is_ok());
        let parsed_form = result.unwrap();
        let parts = parsed_form.parts;
        assert_eq!(1, parts.len());
        assert!(matches!(parts[0], FormPart::Charset(ref chars) if chars == &['a', 'b', 'c']));
    }

    #[test]
    fn test_parse_form_anagram() {
        let result = parse_form("/abc");
        assert!(result.is_ok());
        let parsed_form = result.unwrap();
        let parts = parsed_form.parts;
        assert_eq!(1, parts.len());
        assert!(matches!(parts[0], FormPart::Anagram(ref s) if s == "abc"));
    }

    #[test]
    fn test_parse_form_complex() {
        let result = parse_form("A~A[rstlne]/jon@#.*");
        assert!(result.is_ok());
        let parsed_form = result.unwrap();
        let parts = parsed_form.parts;
        assert_eq!(8, parts.len());
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
        let parsed_form = parse_form("l.x").unwrap();
        let parts = parsed_form.parts;
        let regex_str = form_to_regex_str(&parts);
        assert_eq!("L.X", regex_str);
    }

    #[test]
    fn test_form_to_regex_str_with_variables() {
        let parts = parse_form("AB").unwrap().parts;
        let regex_str = form_to_regex_str(&parts);
        assert_eq!(".+.+", regex_str);
    }

    #[test]
    fn test_form_to_regex_str_with_variables_with_backref() {
        let parts = parse_form("AA").unwrap().parts;
        let regex_str = form_to_regex_str(&parts);
        assert_eq!("(.+)\\1", regex_str);
    }

    #[test]
    fn test_form_to_regex_str_with_variables_with_backref_complex() {
        let parts = parse_form("ABABBCA").unwrap().parts;
        let regex_str = form_to_regex_str(&parts);
        assert_eq!("(.+)(.+)\\1\\2\\2.+\\1", regex_str);
    }

    #[test]
    fn test_form_to_regex_str_with_variables_with_rev_and_backref_complex() {
        let parts = parse_form("AB~AABBDC~AA~C").unwrap().parts;
        let regex_str = form_to_regex_str(&parts);
        assert_eq!("(.+)(.+)(.+)\\1\\2\\2.+.+\\3\\1.+", regex_str);
    }

    #[test]
    fn test_form_to_regex_str_with_wildcards() {
        let parts = parse_form(".*@#").unwrap().parts;
        let regex_str = form_to_regex_str(&parts);
        assert_eq!("..*[AEIOUY][BCDFGHJKLMNPQRSTVWXZ]", regex_str);
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
        let result = match_equation_all("INCH", &patt, None).into_iter().next();
        assert!(result.is_some());
        let binding = result.unwrap();
        // TODO allow for IN/CH or INC/H
        assert_eq!(Some(&"I".to_string()), binding.get('A'));
        assert_eq!(Some(&"NCH".to_string()), binding.get('B'));
    }

    #[test]
    fn test_reversed_variable_binding() {
        let patt = parse_form("A~A").unwrap();
        let result = match_equation_all("NOON", &patt, None).into_iter().next();
        assert!(result.is_some());
        let binding = result.unwrap();
        assert_eq!(Some(&"NO".to_string()), binding.get('A'));
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
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ParseError::EmptyForm));
    }

    #[test]
    fn test_invalid_pattern() {
        let result = parse_form("[");
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), ParseError::ParseFailure { position: 0, remaining: _ }));
    }
}
