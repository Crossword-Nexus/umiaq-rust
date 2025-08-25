use crate::umiaq_char::{CONSONANTS, LITERAL_CHARS, NUM_POSSIBLE_VARIABLES, UmiaqChar, VARIABLE_CHARS, VOWELS};
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
use std::cmp::min;
use std::collections::HashMap;
use crate::bindings::Bindings;
use crate::constraints::{VarConstraint, VarConstraints};
use crate::joint_constraints::JointConstraints;
use fancy_regex::Regex;
use std::fmt::Write as _;
use std::sync::{Mutex, OnceLock};

static REGEX_CACHE: OnceLock<Mutex<HashMap<String, Regex>>> = OnceLock::new();

/// Return a compiled `Regex` for `pattern`, caching the result.
///
/// Behavior:
/// 1. Try to fetch a cached `Regex` under the `pattern` key.
/// 2. If missing, compile it and insert into the cache.
/// 3. Return a **clone** of the cached/compiled `Regex`. Cloning is cheap.
///
/// Locking strategy:
/// - We hold the `Mutex` only while accessing the map (lookups/inserts).
/// - We accept that two threads might compile the same pattern simultaneously
///   in rare races; the second will simply overwrite the same value. This keeps
///   the lock hold-time minimal. If you need to avoid duplicate compilation,
///   see the "double-check" note below.
pub(crate) fn get_regex(pattern: &str) -> Result<Regex, fancy_regex::Error> {
    // Initialize the cache on first use.
    let cache = REGEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Fast path: check if we already have it.
    if let Some(re) = cache.lock().unwrap().get(pattern).cloned() {
        return Ok(re);
    }

    // Miss: compile outside the locked critical section to keep contention low.
    let compiled = Regex::new(pattern)?;

    // Insert the compiled regex, then return a clone.
    let mut guard = cache.lock().unwrap();

    // Optional "double-check" to avoid duplicate compilation:
    // If another thread inserted while we were compiling, prefer the existing one.
    if let Some(existing) = guard.get(pattern).cloned() {
        return Ok(existing);
    }

    guard.insert(pattern.to_string(), compiled.clone());
    Ok(compiled)
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
    #[error("Invalid length range \"{input}\"")]
    InvalidLengthRange { input: String },
    #[error("{str}")]
    InvalidComplexConstraint { str: String },
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
    Lit(String),        // 'abc': literal lowercase sequence (lowercase)
    Dot,                // '.' wildcard: exactly one letter
    Star,               // '*' wildcard: zero or more letters
    Vowel,              // '@' wildcard: any vowel (aeiouy)
    Consonant,          // '#' wildcard: any consonant (bcdf...xz)
    Charset(Vec<char>), // '[abc]': any of the given letters
    Anagram(String),    // '/abc': any permutation of the given letters
}

impl FormPart {
    pub(crate) fn is_deterministic(&self) -> bool {
        matches!(self, FormPart::Var(_) | FormPart::RevVar(_) | FormPart::Lit(_))
    }
}

/// A `Vec` of `FormPart`s along with a compiled regex prefilter
#[derive(Debug, Clone)]
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

    // Return an iterator over the form parts
    pub(crate) fn iter(&self) -> std::slice::Iter<'_, FormPart> {
        self.parts.iter()
    }

    /// If this form is deterministic, build the concrete word using `env`.
    /// Returns `None` if any required var is unbound or if a nondeterministic part is present.
    pub(crate) fn materialize_deterministic_with_env(
        &self,
        env: &HashMap<char, String>,
    ) -> Option<String> {
        let mut out = String::new();
        for part in &self.parts {
            match part {
                FormPart::Lit(s) => out.push_str(s),
                FormPart::Var(v) => out.push_str(env.get(v)?),
                FormPart::RevVar(v) => out.extend(env.get(v)?.chars().rev()),
                _ => return None, // any nondeterministic token → not materializable
            }
        }
        Some(out)
    }

}

// Enable `for part in &parsed_form { ... }`
impl<'a> IntoIterator for &'a ParsedForm {
    type Item = &'a FormPart;
    type IntoIter = std::slice::Iter<'a, FormPart>;

    fn into_iter(self) -> Self::IntoIter {
        self.parts.iter()
    }
}

/// Validate whether a candidate binding value is allowed under a `VarConstraint`.
///
/// Checks:
/// 1. If length constraints are present, enforce them
/// 2. If `form` is present, the value must itself match that form.
/// 3. The value must not equal any variable listed in `not_equal` that is already bound.
fn is_valid_binding(val: &str, constraints: &VarConstraint, bindings: &Bindings) -> bool {
    // 1. Length checks (if configured)
    if constraints.min_length.is_some_and(|min_len| val.len() < min_len) ||
        constraints.max_length.is_some_and(|max_len| val.len() > max_len) {
        return false;
    }

    // 2. Apply nested-form constraint if present (use cached parse)
    if let Some(parsed) = constraints.get_parsed_form() && !match_equation_exists(val, parsed, None, None) {
        return false;
    }

    // 3. Check "not equal" constraints
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
    joint_constraints: Option<&JointConstraints>,
) -> bool {
    let mut results: Vec<Bindings> = vec![];
    match_equation_internal(word, parts, false, &mut results, constraints, joint_constraints);
    results.into_iter().next().is_some()
}

/// Return all bindings that satisfy the equation.
pub(crate) fn match_equation_all(
    word: &str,
    parts: &ParsedForm,
    constraints: Option<&VarConstraints>,
    joint_constraints: Option<&JointConstraints>,
) -> Vec<Bindings> {
    // Using a mutable Vec here is intentional and idiomatic:
    // - We accumulate matches in place and pass `&mut results` down the recursion.
    // - This avoids repeated allocations or copies.
    let mut results: Vec<Bindings> = vec![];
    match_equation_internal(word, parts, true, &mut results, constraints, joint_constraints);
    results
}

// TODO better name
struct HelperParams<'a> {
    bindings: &'a mut Bindings,
    results: &'a mut Vec<Bindings>,
    all_matches: bool,
    word: &'a str,
    constraints: Option<&'a VarConstraints>,
    joint_constraints: Option<&'a JointConstraints>,
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
    joint_constraints: Option<&JointConstraints>,
) {
    /// Helper to reverse a bound value if the part is `RevVar`.
    fn get_reversed_or_not(first: &FormPart, val: &str) -> String {
        if matches!(first, FormPart::RevVar(_)) {
            val.chars().rev().collect()
        } else {
            val.to_owned()
        }
    }

    /// Try to consume exactly one char if present, and apply `pred` to it.
    /// - If the predicate passes, recurse with the rest of the chars.
    /// - Otherwise return false (dead end).
    ///
    /// Covers Dot (any char), Vowel, Consonant, Charset, etc.
    fn take_if(chars: &[char], rest: &[FormPart], hp: &mut HelperParams, pred: impl Fn(&char) -> bool) -> bool {
        chars.split_first().is_some_and(|(c, rest_chars)| pred(c) && helper(rest_chars, rest, hp))
    }

    /// Recursive matching helper.
    ///
    /// `chars`       – remaining characters of the word
    /// `parts`       – remaining pattern parts
    /// `bindings`    – current variable assignments
    /// `results`     – collection of successful bindings
    /// `all_matches` – whether to collect all or stop at first
    ///
    /// Return value:
    /// - `true`  → at least one successful match was found along this path
    /// - `false` → no matches were found
    ///
    /// Note that the caller is responsible for interpreting this:
    /// if `hp.all_matches` is `false` (stop at first match),
    /// then the caller should short-circuit and bubble `true` up immediately.
    /// Otherwise (`all_matches == true`), recursion will continue exploring
    /// other possibilities even after one match succeeds.
    fn helper(chars: &[char], parts: &[FormPart], hp: &mut HelperParams) -> bool {
        // Base case: no parts left
        if parts.is_empty() {
            if chars.is_empty() {
                // Joint constraint check here uses `all_satisfied` (not strict):
                // - In the scan phase, we may only have a *partial* binding (e.g., ABC without D).
                // - `all_satisfied` means "nothing inconsistent so far" — so we keep it.
                // - The *strict* check (`all_strictly_satisfied_for_parts`) runs later in
                //   `recursive_join`, once all patterns are combined and every variable is bound.
                if hp.joint_constraints.is_none_or(|jc| jc.all_satisfied(hp.bindings)) {
                    let mut full_result = hp.bindings.clone();
                    full_result.set_word(hp.word);
                    hp.results.push(full_result);
                    return !hp.all_matches;
                }
            }
            return false;
        }

        let (first, rest) = (&parts[0], &parts[1..]);

        match first {
            FormPart::Lit(s) => {
                // Literal match (case-insensitive, stored lowercase)
                is_prefix(s, &chars, rest, hp)
            }

            FormPart::Star => {
                // Zero-or-more wildcard; try all possible splits
                (0..=chars.len()).any(|i| helper(&chars[i..], rest, hp))
            }

            // Combined vowel, consonant, charset, dot cases
            FormPart::Dot => take_if(chars, rest, hp, |_| true),
            FormPart::Vowel => take_if(chars, rest, hp, char::is_vowel),
            FormPart::Consonant => take_if(chars, rest, hp, char::is_consonant),
            FormPart::Charset(s) => take_if(chars, rest, hp, |c| s.contains(c)),

            FormPart::Anagram(s) => {
                // Match if the next len chars are an anagram of target
                let len = s.len();

                chars.len() >= len && are_anagrams(&chars[..len], s) && helper(&chars[len..], rest, hp)
            }
            FormPart::Var(var_name) | FormPart::RevVar(var_name) => {
                if let Some(bound_val) = hp.bindings.get(*var_name) {
                    // Already bound: must match exactly
                    is_prefix(&get_reversed_or_not(first, bound_val), &chars, rest, hp)
                } else {
                    // Not bound yet: try binding to all possible lengths
                    // To prune the search space, apply length constraints up front
                    let min_len = hp.constraints.and_then(|constraints_inner|
                        constraints_inner.get(*var_name).map(|vc| vc.min_length)
                    ).flatten().unwrap_or(1usize);
                    let max_len_cfg = hp.constraints.and_then(|constraints_inner|
                        constraints_inner.get(*var_name).map(|vc| vc.max_length)
                    ).flatten().unwrap_or(chars.len());

                    let avail = chars.len();

                    // If the minimum exceeds what we have left, this path can't work
                    if min_len > avail {
                        false
                    } else {
                        // Never try to slice past what's actually available
                        let capped_max = min(max_len_cfg, avail);

                        (min_len..=capped_max).into_iter().any(|l| {
                            let candidate_chars = &chars[..l];

                            let bound_val: String = if matches!(first, FormPart::RevVar(_)) {
                                candidate_chars.iter().rev().collect()
                            } else {
                                candidate_chars.iter().collect()
                            };

                            // Apply variable-specific constraints
                            let valid = hp.constraints.is_none_or(|all_c| {
                                all_c.get(*var_name).is_none_or(|c| is_valid_binding(&bound_val, c, hp.bindings))
                            });

                            if valid {
                                hp.bindings.set(*var_name, bound_val);

                                // Recurse to try this choice
                                let found = helper(&chars[l..], rest, hp);

                                // Always undo our tentative binding before returning/continuing
                                hp.bindings.remove(*var_name);

                                // If we only need one match and we found one, bubble up early.
                                if found && !hp.all_matches {
                                    return true;
                                }

                                // Otherwise continue searching other lengths.
                                found
                            } else {
                                false
                            }
                        })
                    }
                }
            }
        }
    }

    /// Returns true if `prefix` is a prefix of `chars`
    fn is_prefix(prefix: &str, chars: &&[char], rest: &[FormPart], helper_params: &mut HelperParams) -> bool {
        let prefix_len = prefix.len();

        if chars.len() >= prefix_len && chars[..prefix_len].iter().copied().zip(prefix.chars()).all(|(a, b)| a == b) {
            helper(&chars[prefix_len..], rest, helper_params)
        } else {
            false
        }
    }

    /// Returns `true` if `lowercase_word` is an anagram of `other_word`.
    ///
    /// Implementation details / limitations:
    /// - Only ASCII characters in the range 0–127 are counted. Any character
    ///   ≥128 is ignored in the frequency table, which may cause false positives.
    /// - Intended use is for lowercase a–z inputs; callers should ensure both
    ///   arguments are normalized (e.g., to lowercase) before calling.
    /// - Runs in O(n) time with O(1) memory (fixed 128-entry table).
    ///
    /// If you need full Unicode or mixed-case support, replace this with a
    /// `HashMap<char, usize>` count comparison.
    fn are_anagrams(lowercase_word: &[char], other_word: &str) -> bool {
        if lowercase_word.len() != other_word.len() {
            return false;
        }

        let mut char_counts = [0u8; 128];

        for &c in lowercase_word {
            if (c as usize) < 128 {
                char_counts[c as usize] += 1;
            }
        }

        for c in other_word.chars() {
            if (c as usize) < 128 {
                if char_counts[c as usize] == 0 {
                    return false;
                }
                char_counts[c as usize] -= 1;
            }
        }


        char_counts.iter().all(|&count| count == 0)
    }

    // === PREFILTER STEP ===
    if !parsed_form.prefilter.is_match(word).unwrap() {
        return;
    }

    // Normalize word and start recursive matching
    let mut hp = HelperParams {
        bindings: &mut Bindings::default(),
        results,
        all_matches,
        word,
        constraints,
        joint_constraints,
    };

    helper(&word.chars().collect::<Vec<_>>(), &parsed_form.parts, &mut hp);
}

/// Convert a parsed `FormPart` sequence into a regex string.
///
/// Used for the initial fast prefilter in `match_equation_internal`.
fn form_to_regex_str(parts: &[FormPart]) -> String {
    let (var_counts, rev_var_counts) = get_var_and_rev_var_counts(parts);
    let mut var_to_backreference_num = [0; NUM_POSSIBLE_VARIABLES];
    let mut rev_var_to_backreference_num = [0; NUM_POSSIBLE_VARIABLES];
    let mut backreference_index = 0;

    let mut regex_str = String::new();
    for part in parts {
        match part {
            FormPart::Var(c) => regex_str.push_str(&get_regex_str_segment(var_counts, &mut var_to_backreference_num, &mut backreference_index, *c)),
            FormPart::RevVar(c) => regex_str.push_str(&get_regex_str_segment(rev_var_counts, &mut rev_var_to_backreference_num, &mut backreference_index, *c)),
            FormPart::Lit(s) => regex_str.push_str(&fancy_regex::escape(s)),
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
                    regex_str.push(*c);
                }
                regex_str.push(']');
            }
            FormPart::Anagram(s) => {
                let len = s.len();
                let class = fancy_regex::escape(s);
                let _ = write!(regex_str, "[{class}]{{{len}}}");
            }
        }
    }

    regex_str
}

// TODO DRY w/form_to_regex_str
/// Convert a parsed `FormPart` sequence into a regex string,
/// taking variable constraints into account when possible.
///
/// Differences from `form_to_regex_str`:
/// - For the **first occurrence** of a variable with an attached
///   simple form constraint (e.g., `A=(x*a)`), we inject a
///   lookahead `(?=x.*a)` so that the regex prefilter enforces it
///   early. This prunes the candidate list before recursion.
/// - For **multi-use variables**, we still capture the first
///   occurrence (`(.+)`) and backreference later ones (`\1`, etc.).
///   The lookahead becomes `(?=x.*a)(.+)` so numbering is preserved.
/// - For single-use variables, we emit `(?=x.*a).+` instead of just `.+`.
///
/// Notes:
/// - Constraint forms are assumed to contain only literals and
///   wildcards, never other variables. That guarantee makes it safe
///   to inline them into regex directly.
/// - Reversed variables (`~A`) are left unchanged: enforcing a
///   reversed constraint at regex level would require reversing
///   arbitrary sub-regexes, which isn’t practical here.
/// - If no constraint exists for a variable, or no form is present,
///   behavior falls back to the original `.++` / `(.+)` scheme.
pub(crate) fn form_to_regex_str_with_constraints(
    parts: &[FormPart],
    constraints: Option<&VarConstraints>,
) -> String {
    use std::fmt::Write;

    // Count how many times each var/revvar occurs in this form
    let (var_counts, rev_var_counts) = get_var_and_rev_var_counts(parts);

    // Track capture group assignment for vars and revvars
    let mut var_to_backreference_num = [0; NUM_POSSIBLE_VARIABLES];
    let mut rev_var_to_backreference_num = [0; NUM_POSSIBLE_VARIABLES];
    let mut backreference_index = 0;

    let mut regex_str = String::new();

    for part in parts {
        match part {
            FormPart::Var(c) => {
                let idx = char_to_num(*c);
                let occurs_many = var_counts[idx] > 1;
                let already_has_group = var_to_backreference_num[idx] != 0;

                // Extract a simple nested regex if a constraint exists
                let lookahead = constraints
                    .and_then(|all| all.get(*c))
                    .and_then(|vc| vc.get_parsed_form())
                    .map(|pf| form_to_regex_str(&pf.parts));

                if already_has_group {
                    // Subsequent occurrence: always a backref (\1, \2, …)
                    let _ = write!(regex_str, "\\{}", var_to_backreference_num[idx]);
                } else if occurs_many {
                    // First of multiple occurrences: capture group
                    backreference_index += 1;
                    var_to_backreference_num[idx] = backreference_index;

                    if let Some(nested) = lookahead {
                        // Add lookahead before the group to enforce constraint
                        let _ = write!(regex_str, "(?={nested})(.+)");
                    } else {
                        regex_str.push_str("(.+)");
                    }
                } else {
                    // Single-use variable (no backrefs needed)
                    if let Some(nested) = lookahead {
                        let _ = write!(regex_str, "(?={nested}).+");
                    } else {
                        regex_str.push_str(".+");
                    }
                }
            }

            FormPart::RevVar(c) => {
                // Reverse vars behave as before (no lookahead injection)
                let idx = char_to_num(*c);
                let occurs_many = rev_var_counts[idx] > 1;
                let already_has_group = rev_var_to_backreference_num[idx] != 0;

                if already_has_group {
                    let _ = write!(regex_str, "\\{}", rev_var_to_backreference_num[idx]);
                } else if occurs_many {
                    backreference_index += 1;
                    rev_var_to_backreference_num[idx] = backreference_index;
                    regex_str.push_str("(.+)");
                } else {
                    regex_str.push_str(".+");
                }
            }

            // All other token types follow the original scheme
            FormPart::Lit(s) => regex_str.push_str(&fancy_regex::escape(s)),
            FormPart::Dot => regex_str.push('.'),
            FormPart::Star => regex_str.push_str(".*"),
            FormPart::Vowel => { let _ = write!(regex_str, "[{VOWELS}]"); }
            FormPart::Consonant => { let _ = write!(regex_str, "[{CONSONANTS}]"); }
            FormPart::Charset(chars) => {
                regex_str.push('[');
                for c in chars { regex_str.push(*c); }
                regex_str.push(']');
            }
            FormPart::Anagram(s) => {
                let len = s.len();
                let class = fancy_regex::escape(s);
                let _ = write!(regex_str, "[{class}]{{{len}}}");
            }
        }
    }

    regex_str
}

// Tiny detector so we don’t build a fancy regex when it can’t help
pub(crate) fn has_inlineable_var_form(parts: &[FormPart], constraints: &VarConstraints) -> bool {
    parts.iter().any(|p| match p {
        FormPart::Var(c) => constraints.get(*c).and_then(|vc| vc.get_parsed_form()).is_some(),
        _ => false,
    })
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

/// Count the number of times each variable (and reversed variable) appears
/// in a sequence of `FormPart`s.
///
/// Returns two parallel arrays (length = `NUM_POSSIBLE_VARIABLES`):
/// - `var_counts[i]`    = number of times variable 'A'+i appears
/// - `rev_var_counts[i]` = number of times reversed variable '~(A+i)' appears
///
/// Usage: in `form_to_regex_str`, we only care about distinguishing
/// between "appears once" vs. "appears multiple times" in order to decide
/// whether to emit a backreference. In the zero case, callers don't consult
/// the arrays at all. So while the exact counts aren’t strictly necessary,
/// tracking them is cheap and keeps the code simple/clear.
fn get_var_and_rev_var_counts(parts: &[FormPart]) -> ([usize; NUM_POSSIBLE_VARIABLES], [usize; NUM_POSSIBLE_VARIABLES]) {
    let mut var_counts = [0; NUM_POSSIBLE_VARIABLES];
    let mut rev_var_counts = [0; NUM_POSSIBLE_VARIABLES];
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
    // this mutability isn't so bad -- it's local and efficient
    let mut parts = vec![];

    while !rest.is_empty() {
        match equation_part(rest) {
            // Note: we can't write `Ok((rest, part))` here because that would
            // shadow the outer `rest` instead of updating it. We bind to a new
            // name (`next`) and then assign it back to the outer `rest`.
            Ok((next, part)) => {
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
    map(one_of(VARIABLE_CHARS), FormPart::Var).parse(input)
}

fn revref(input: &str) -> IResult<&str, FormPart> {
    map(preceded(tag("~"), one_of(VARIABLE_CHARS)), FormPart::RevVar).parse(input)
}

fn literal(input: &str) -> IResult<&str, FormPart> {
    map(many1(one_of(LITERAL_CHARS)), |chars| {
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
    let (input, chars) = many1(one_of(LITERAL_CHARS)).parse(input)?;
    let (input, _) = tag("]")(input)?;
    Ok((input, FormPart::Charset(chars)))
}

fn anagram(input: &str) -> IResult<&str, FormPart> {
    let (input, _) = tag("/")(input)?;
    let (input, chars) = many1(one_of(LITERAL_CHARS)).parse(input)?;
    Ok((input, FormPart::Anagram(chars.into_iter().collect())))
}

/// Try parsing any valid token from the input.
fn equation_part(input: &str) -> IResult<&str, FormPart> {
    alt((revref, varref, anagram, charset, literal, dot, star, vowel, consonant))
    .parse(input)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use super::*;

    #[test]
    fn test_empty_form_error() {
        assert!(matches!(parse_form("").unwrap_err(), ParseError::EmptyForm));
    }

    #[test]
    fn test_parse_failure_error() {
        assert!(matches!(parse_form("[").unwrap_err(), ParseError::ParseFailure { .. }));
    }

    #[test]
    fn test_match_equation_exists() {
        let patt = parse_form("A~A[rstlne]/jon@#.*").unwrap();
        assert!(match_equation_exists("aaronjudge", &patt, None, None));
        assert!(!match_equation_exists("noon", &patt, None, None));
        assert!(!match_equation_exists("toon", &patt, None, None));
    }

    #[test]
    fn test_valid_binding_not_equal_pass() {
        // A must be != B; current B is bound to "test"
        let mut vc = VarConstraint::default();
        vc.not_equal.insert('B');

        let mut b = Bindings::default();
        b.set('B', "test".to_string());

        // "OTHER" != "TEST", so this should pass
        assert!(is_valid_binding("other", &vc, &b));
    }

    #[test]
    fn test_valid_binding_simple_pass() {
        let mut vc = VarConstraint::default();
        vc.form = Option::from("abc*".to_string());

        let b = Bindings::default();
        assert!(is_valid_binding("abcat", &vc, &b));
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
        let m = match_equation_all("INCH", &patt, Some(&var_constraints), None).into_iter().next().unwrap();
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
        vc.min_length = Some(MIN_LENGTH);
        vc.max_length = Some(MAX_LENGTH);
        // associate this constraint with variable 'A'
        var_constraints.insert('A', vc);

        let matches = match_equation_all("HOTHOT", &patt, Some(&var_constraints), None);
        println!("{matches:?}");
        for bindings in matches.iter() {
            assert!((MIN_LENGTH..=MAX_LENGTH).contains(&bindings.get('A').unwrap().len()));
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
        vc_a.min_length = Some(2);
        vc_a.max_length = Some(2);
        var_constraints.insert('A', vc_a);
        // now add it for B
        let mut vc_b = VarConstraint::default();
        vc_b.not_equal.insert('A');
        vc_b.min_length = Some(2);
        vc_b.max_length = Some(2);
        var_constraints.insert('B', vc_b);

        let matches = match_equation_all("INCH", &patt, Some(&var_constraints), None);
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

        assert!(match_equation_exists("inch", &patt, Some(&var_constraints1), None));

        // Second constraint: A=(*z*)
        let mut var_constraints2 = VarConstraints::default();
        let mut vc2 = VarConstraint::default();
        vc2.form = Option::from("*z*".to_string());
        var_constraints2.insert('A', vc2.clone());

        assert!(!match_equation_exists("inch", &patt, Some(&var_constraints2), None));
    }

    #[test]
    fn test_parse_form_basic() {
        let result = parse_form("abc");
        assert!(result.is_ok());
        let parsed_form = result.unwrap();
        let parts = parsed_form.parts;
        assert_eq!(vec![FormPart::Lit("abc".parse().unwrap())], parts);
    }

    #[test]
    fn test_parse_form_variable() {
        assert_eq!(vec![FormPart::Var('A')], parse_form("A").unwrap().parts);
    }

    #[test]
    fn test_parse_form_reversed_variable() {
        assert_eq!(vec![FormPart::RevVar('A')], parse_form("~A").unwrap().parts);
    }

    #[test]
    fn test_parse_form_wildcards() {
        let result = parse_form(".*@#");
        assert!(result.is_ok());
        let parsed_form = result.unwrap();
        let parts = parsed_form.parts;
        assert_eq!(vec![FormPart::Dot, FormPart::Star, FormPart::Vowel, FormPart::Consonant], parts);
    }

    #[test]
    fn test_parse_form_charset() {
        assert_eq!(vec![FormPart::Charset(vec!['a', 'b', 'c'])], parse_form("[abc]").unwrap().parts);
    }

    #[test]
    fn test_parse_form_anagram() {
        assert_eq!(vec![FormPart::Anagram("abc".parse().unwrap())], parse_form("/abc").unwrap().parts);
    }

    #[test]
    fn test_parse_form_complex() {
        let expected = vec![
            FormPart::Var('A'),
            FormPart::RevVar('A'),
            FormPart::Charset(vec!['r', 's', 't', 'l', 'n', 'e']),
            FormPart::Anagram("jon".parse().unwrap()),
            FormPart::Vowel,
            FormPart::Consonant,
            FormPart::Dot,
            FormPart::Star
        ];
        assert_eq!(expected, parse_form("A~A[rstlne]/jon@#.*").unwrap().parts);
    }

    #[test]
    fn test_form_to_regex_str() {
        assert_eq!("l.x", form_to_regex_str(&parse_form("l.x").unwrap().parts));
    }

    #[test]
    fn test_form_to_regex_str_with_variables() {
        assert_eq!(".+.+", form_to_regex_str(&parse_form("AB").unwrap().parts));
    }

    #[test]
    fn test_form_to_regex_str_with_variables_with_backref() {
        assert_eq!("(.+)\\1", form_to_regex_str(&parse_form("AA").unwrap().parts));
    }

    #[test]
    fn test_form_to_regex_str_with_variables_with_backref_complex() {
        assert_eq!("(.+)(.+)\\1\\2\\2.+\\1", form_to_regex_str(&parse_form("ABABBCA").unwrap().parts));
    }

    #[test]
    fn test_form_to_regex_str_with_variables_with_rev_and_backref_complex() {
        assert_eq!("(.+)(.+)(.+)\\1\\2\\2.+.+\\3\\1.+", form_to_regex_str(&parse_form("AB~AABBDC~AA~C").unwrap().parts));
    }

    #[test]
    fn test_form_to_regex_str_with_wildcards() {
        assert_eq!("..*[aeiouy][bcdfghjklmnpqrstvwxz]", form_to_regex_str(&parse_form(".*@#").unwrap().parts));
    }

    #[test]
    fn test_palindrome_matching() {
        let patt = parse_form("A~A").unwrap();
        assert!(match_equation_exists("noon", &patt, None, None));
        assert!(!match_equation_exists("radar", &patt, None, None));
        assert!(!match_equation_exists("test", &patt, None, None));
    }

    #[test]
    fn test_anagram_matching() {
        let patt = parse_form("/triangle").unwrap();
        assert!(match_equation_exists("integral", &patt, None, None));
        assert!(!match_equation_exists("square", &patt, None, None));
    }

    #[test]
    fn test_variable_binding() {
        let patt = parse_form("AB").unwrap();
        let results = match_equation_all("inch", &patt, None, None);

        // Collect the observed (A,B) pairs
        let observed: HashSet<(String, String)> = results
            .into_iter()
            .map(|b| {
                (
                    b.get('A').unwrap().clone(),
                    b.get('B').unwrap().clone(),
                )
            })
            .collect();

        // All valid splits of "inch" into two nonempty pieces
        let expected: HashSet<(String, String)> = [
            ("i".to_string(), "nch".to_string()),
            ("in".to_string(), "ch".to_string()),
            ("inc".to_string(), "h".to_string()),
        ]
            .into_iter()
            .collect();

        assert_eq!(observed, expected);
    }

    #[test]
    fn test_reversed_variable_binding() {
        let patt = parse_form("A~A").unwrap();
        let bindings = match_equation_all("noon", &patt, None, None).into_iter().next().unwrap();
        assert_eq!(Some(&"no".to_string()), bindings.get('A'));
    }

    #[test]
    fn test_literal_matching() {
        let patt = parse_form("abc").unwrap();
        assert!(match_equation_exists("abc", &patt, None, None));
        assert!(!match_equation_exists("xyz", &patt, None, None));
    }

    #[test]
    fn test_dot_wildcard() {
        let patt = parse_form("a.z").unwrap();
        assert!(match_equation_exists("abz", &patt, None, None));
        assert!(match_equation_exists("azz", &patt, None, None));
        assert!(!match_equation_exists("az", &patt, None, None));
        assert!(!match_equation_exists("abbz", &patt, None, None));
    }

    #[test]
    fn test_star_wildcard() {
        let patt = parse_form("a*z").unwrap();
        assert!(match_equation_exists("az", &patt, None, None));
        assert!(match_equation_exists("abz", &patt, None, None));
        assert!(match_equation_exists("abbbz", &patt, None, None));
        assert!(!match_equation_exists("ay", &patt, None, None));
    }

    #[test]
    fn test_vowel_wildcard() {
        let patt = parse_form("A@Z").unwrap();
        assert!(match_equation_exists("aaz", &patt, None, None));
        assert!(match_equation_exists("aez", &patt, None, None));
        assert!(!match_equation_exists("abz", &patt, None, None));
    }

    #[test]
    fn test_consonant_wildcard() {
        let patt = parse_form("A#Z").unwrap();
        assert!(match_equation_exists("abz", &patt, None, None));
        assert!(match_equation_exists("acz", &patt, None, None));
        assert!(!match_equation_exists("aaz", &patt, None, None));
    }

    #[test]
    fn test_charset_matching() {
        let patt = parse_form("[abc]").unwrap();
        assert!(match_equation_exists("a", &patt, None, None));
        assert!(match_equation_exists("b", &patt, None, None));
        assert!(match_equation_exists("c", &patt, None, None));
        assert!(!match_equation_exists("d", &patt, None, None));
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

    #[test]
    fn test_var_vowel_var_no_panic_and_matches() {
        let patt = parse_form("A@B").unwrap();
        assert!(match_equation_exists("cab", &patt, None, None)); // 'A'='C', '@'='A', 'B'='B
        assert!(!match_equation_exists("c", &patt, None, None));  // too short
        assert!(!match_equation_exists("ca", &patt, None, None));  // too short
    }

    /// Test that a pattern with a star works
    #[test]
    fn test_star() {
        // Pattern
        let patt = parse_form("l*x").unwrap();
        let matches = match_equation_all("lox", &patt, None, None);
        println!("{matches:?}");
        assert_eq!(1, matches.len());
    }

    #[test]
    fn materialize_deterministic_rejects_nondeterministic_parts() {
        // Pattern has a Dot (.) → nondeterministic
        let pf = parse_form("A.B").unwrap();

        // Supply values for A and B
        let env = HashMap::from([('A', "x".to_string()), ('B', "y".to_string())]);

        // Because of the Dot, this form cannot be fully materialized → expect None
        assert!(pf.materialize_deterministic_with_env(&env).is_none());
    }

    #[test]
    fn materialize_deterministic_requires_all_vars_bound() {
        // Pattern has two variables
        let pf = parse_form("AB").unwrap();

        // Only provide a value for A; leave B unbound
        let env = HashMap::from([('A', "hi".to_string())]);

        // Since B is missing, materialization must fail → expect None
        assert!(pf.materialize_deterministic_with_env(&env).is_none());
    }

    #[test]
    fn materialize_deterministic_succeeds_with_only_lits_and_vars() {
        // Pattern is fully deterministic: literals + vars + a reversed var
        let pf = parse_form("preAB~Apost").unwrap();

        // Provide bindings for A and B
        let env = HashMap::from([('A', "no".to_string()), ('B', "de".to_string())]);

        // Expect the literal "pre", then "B"="de", then "A"="no",
        // then "~A"="on", then the literal "post"
        assert_eq!(
            Some("prenodeonpost".to_string()),
            pf.materialize_deterministic_with_env(&env)
        );
    }
}
