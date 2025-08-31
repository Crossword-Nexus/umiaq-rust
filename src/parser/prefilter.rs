use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use fancy_regex::Regex;

use crate::constraints::VarConstraints;
use crate::umiaq_char::{CONSONANTS, NUM_POSSIBLE_VARIABLES, VOWELS};

use super::form::{FormPart, ParsedForm};

/// Global, lazily initialized cache of compiled regexes.
///
/// - `OnceLock` ensures the cache is created at most once, on first use.
/// - We wrap the `HashMap` in a `Mutex` to provide **interior mutability** and
///   **thread safety**. A plain `HashMap` isn’t thread-safe and cannot be
///   mutated through a shared reference; `Mutex` gives us a safe, exclusive
///   handle when inserting or reading.
///
/// Locking strategy:
/// - We hold the `Mutex` only while accessing the map (lookups/inserts).
/// - We compile outside the lock to keep contention low, with a “double-check”
///   before insert to avoid duplicate work in rare races.
/// - `Regex` clones are cheap (internally ref-counted), so we release the lock quickly.
static REGEX_CACHE: OnceLock<Mutex<HashMap<String, Regex>>> = OnceLock::new();

/// Return a compiled `Regex` for `pattern`, caching the result.
pub(crate) fn get_regex(pattern: &str) -> Result<Regex, fancy_regex::Error> {
    let cache = REGEX_CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    if let Some(re) = cache.lock().unwrap().get(pattern).cloned() {
        return Ok(re);
    }

    // Compile outside the lock.
    let compiled = Regex::new(pattern)?;

    // Insert with a double-check in case another thread inserted it meanwhile.
    let mut guard = cache.lock().unwrap();
    if let Some(existing) = guard.get(pattern).cloned() {
        return Ok(existing);
    }
    guard.insert(pattern.to_string(), compiled.clone());
    Ok(compiled)
}

// TODO DRY w/form_to_regex_str_with_constraints
/// Convert a parsed `FormPart` sequence into a regex string (no constraints).
///
/// Used for the initial fast prefilter.
pub(crate) fn form_to_regex_str(parts: &[FormPart]) -> String {
    let (var_counts, rev_var_counts) = get_var_and_rev_var_counts(parts);
    let mut var_to_backreference_num = [0; NUM_POSSIBLE_VARIABLES];
    let mut rev_var_to_backreference_num = [0; NUM_POSSIBLE_VARIABLES];
    let mut backreference_index = 0;

    let mut regex_str = String::new();
    for part in parts {
        match part {
            FormPart::Var(c) => {
                regex_str.push_str(&get_regex_str_segment(
                    var_counts,
                    &mut var_to_backreference_num,
                    &mut backreference_index,
                    *c,
                ));
            }
            FormPart::RevVar(c) => {
                regex_str.push_str(&get_regex_str_segment(
                    rev_var_counts,
                    &mut rev_var_to_backreference_num,
                    &mut backreference_index,
                    *c,
                ));
            }
            FormPart::Lit(s) => regex_str.push_str(&fancy_regex::escape(s)),
            FormPart::Dot => regex_str.push('.'),
            FormPart::Star => regex_str.push_str(".*"),
            FormPart::Vowel => {
                use std::fmt::Write;
                let _ = write!(regex_str, "[{VOWELS}]");
            }
            FormPart::Consonant => {
                use std::fmt::Write;
                let _ = write!(regex_str, "[{CONSONANTS}]");
            }
            FormPart::Charset(chars) => {
                regex_str.push('[');
                for c in chars {
                    regex_str.push(*c);
                }
                regex_str.push(']');
            }
            FormPart::Anagram(ag) => {
                use std::fmt::Write;
                let len = ag.len;
                let class = fancy_regex::escape(ag.as_string.as_str());
                let _ = write!(regex_str, "[{class}]{{{len}}}");
            }
        }
    }

    regex_str
}

fn get_regex_str_segment(
    var_counts: [usize; NUM_POSSIBLE_VARIABLES],
    var_to_backreference_num: &mut [usize; NUM_POSSIBLE_VARIABLES],
    backreference_index: &mut usize,
    c: char,
) -> String {
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

// 'A' -> 0, 'B' -> 1, ..., 'Z' -> 25
fn char_to_num(c: char) -> usize { c as usize - 'A' as usize }

// Count occurrences of vars and revvars to decide capture/backref scheme.
fn get_var_and_rev_var_counts(
    parts: &[FormPart],
) -> ([usize; NUM_POSSIBLE_VARIABLES], [usize; NUM_POSSIBLE_VARIABLES]) {
    let mut var_counts = [0; NUM_POSSIBLE_VARIABLES];
    let mut rev_var_counts = [0; NUM_POSSIBLE_VARIABLES];
    for part in parts {
        match part {
            FormPart::Var(c) => var_counts[char_to_num(*c)] += 1,
            FormPart::RevVar(c) => rev_var_counts[char_to_num(*c)] += 1,
            _ => (),
        }
    }
    (var_counts, rev_var_counts)
}

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

    let (var_counts, rev_var_counts) = get_var_and_rev_var_counts(parts);
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

                // If A has a nested form, turn that into regex once
                let lookahead = constraints
                    .and_then(|all| all.get(*c))
                    .and_then(|vc| vc.get_parsed_form())
                    .map(|pf| form_to_regex_str(&pf.parts)); // constraint forms are var-free

                if already_has_group {
                    let _ = write!(regex_str, "\\{}", var_to_backreference_num[idx]);
                } else if occurs_many {
                    backreference_index += 1;
                    var_to_backreference_num[idx] = backreference_index;
                    if let Some(nested) = lookahead {
                        let _ = write!(regex_str, "(?={nested})(.+)");
                    } else {
                        regex_str.push_str("(.+)");
                    }
                } else if let Some(nested) = lookahead {
                    let _ = write!(regex_str, "(?={nested}).+");
                } else {
                    regex_str.push_str(".+");
                }
            }

            FormPart::RevVar(c) => {
                // Keep existing behavior for ~A
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
            FormPart::Anagram(ag) => {
                let len = ag.len;
                let class = fancy_regex::escape(ag.as_string.as_str());
                let _ = write!(regex_str, "[{class}]{{{len}}}");
            }
        }
    }

    regex_str
}

/// True if any `Var` in `parts` has a `.form` constraint we can inline.
pub(crate) fn has_inlineable_var_form(parts: &[FormPart], constraints: &VarConstraints) -> bool {
    parts.iter().any(|p| match p {
        FormPart::Var(c) => constraints.get(*c).and_then(|vc| vc.get_parsed_form()).is_some(),
        _ => false,
    })
}

/// Build the best prefilter for this (form, constraints) pair:
/// - If a var has a simple `.form`, build the constraint-aware regex (with lookaheads).
/// - Otherwise, reuse the already-cached `ParsedForm.prefilter`.
pub(crate) fn build_prefilter_regex(
    parsed_form: &ParsedForm,
    constraints: Option<&VarConstraints>,
) -> Regex {
    if let Some(vcs) = constraints && has_inlineable_var_form(&parsed_form.parts, vcs) {
        let anchored = format!("^{}$", form_to_regex_str_with_constraints(&parsed_form.parts, Some(vcs)));
        return get_regex(&anchored).unwrap_or_else(|_| parsed_form.prefilter.clone());
    }
    parsed_form.prefilter.clone()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constraints::{VarConstraint, VarConstraints};
    use crate::parser::form::parse_form;

    #[test]
    fn test_constraint_prefilter_string_single_use() {
        let pf = parse_form("A").unwrap();
        let mut vcs = VarConstraints::default();
        let mut vc = VarConstraint::default();
        vc.form = Some("x*a".to_string());
        vcs.insert('A', vc);
        let re_str = form_to_regex_str_with_constraints(&pf.parts, Some(&vcs));
        assert_eq!(re_str, "(?=x.*a).+");
    }

    #[test]
    fn test_prefilter_upgrade_prunes_nonmatching_words() {
        let mut pf = parse_form("A").unwrap();
        let mut vcs = VarConstraints::default();
        let mut vc = VarConstraint::default();
        vc.form = Some("x*a".to_string());
        vcs.insert('A', vc);

        assert!(pf.prefilter.is_match("abba").unwrap());
        let upgraded = build_prefilter_regex(&pf, Some(&vcs));
        pf.prefilter = upgraded;

        assert!(pf.prefilter.is_match("xya").unwrap());
        assert!(!pf.prefilter.is_match("abba").unwrap());
    }
}
