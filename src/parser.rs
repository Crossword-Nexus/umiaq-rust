// src/lib.rs
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

use regex::Regex;
use std::fmt::Write as _;
use crate::bindings::Bindings;
use crate::constraints::{VarConstraint, VarConstraints};

#[derive(Debug, Clone, PartialEq)]
pub enum PatternPart {
    Var(char),           // A-Z variable reference
    RevVar(char),        // ~A reversed variable reference
    Lit(String),         // literal lowercase string
    Dot,                 // . wildcard for any single character
    Star,                // * wildcard for any number of characters
    Vowel,               // @ vowel character
    Consonant,           // # consonant character
    Charset(Vec<char>),  // [abc] character set
    Anagram(String),     // /abc indicates an anagram of given letters
}

/// Check if a binding is valid, given a value
pub fn is_valid_binding(
    val: &str,
    constraints: &VarConstraint,
    bindings: &Bindings,
) -> bool {
    // 1) nested pattern constraint
    if let Some(pattern_str) = &constraints.pattern {
        match parse_pattern(pattern_str) {
            Ok(p) => {
                if !match_pattern_exists(val, &p, None) {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }

    // 2) not_equal is a HashSet<char>
    for &other in &constraints.not_equal {
        if let Some(existing) = bindings.get(&other) {
            if existing == val {
                return false;
            }
        }
    }

    true
}

/// Returns the first successful binding (if any)
pub fn match_pattern(
    word: &str,
    parts: &[PatternPart],
    constraints: Option<&VarConstraints>,
) -> Option<Bindings> {
    let mut results = Vec::new();
    match_pattern_internal(word, parts, false, &mut results, constraints);
    results.into_iter().next()
}

/// Returns a boolean
pub fn match_pattern_exists(
    word: &str,
    parts: &[PatternPart],
    constraints: Option<&VarConstraints>,
) -> bool {
    let mut results: Vec<Bindings> = Vec::new();
    match_pattern_internal(word, parts, false, &mut results, constraints);
    results.into_iter().next().is_some()
}

/// Returns all successful bindings
pub fn match_pattern_all(
    word: &str,
    parts: &[PatternPart],
    constraints: Option<&VarConstraints>,
) -> Vec<Bindings> {
    let mut results: Vec<Bindings> = Vec::new();
    match_pattern_internal(word, parts, true, &mut results, constraints);
    results
}

fn match_pattern_internal(
    word: &str,
    parts: &[PatternPart],
    all_matches: bool,
    results: &mut Vec<Bindings>,
    constraints: Option<&VarConstraints>,
) {
    fn get_reversed_or_not(first: &PatternPart, val: &str) -> String {
        if matches!(first, PatternPart::RevVar(_)) {
            val.chars().rev().collect::<String>()
        } else {
            val.to_owned()
        }
    }

    fn helper(
        chars: &[char],
        parts: &[PatternPart],
        bindings: &mut Bindings,
        results: &mut Vec<Bindings>,
        all_matches: bool,
        word: &str,
        constraints: Option<&VarConstraints>,
    ) -> bool {
        if parts.is_empty() {
            if chars.is_empty() {
                let mut full_result = bindings.clone();
                full_result.set_word(word);
                results.push(full_result);
                return !all_matches;
            }
            return false;
        }

        let (first, rest) = (&parts[0], &parts[1..]);

        match first {
            PatternPart::Lit(s) => {
                let s = s.to_uppercase();
                if chars.starts_with(&s.chars().collect::<Vec<_>>()) {
                    return helper(&chars[s.len()..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            PatternPart::Dot => {
                if !chars.is_empty() {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            PatternPart::Star => {
                for i in 0..=chars.len() {
                    if helper(&chars[i..], rest, bindings, results, all_matches, word, constraints)
                        && !all_matches
                    {
                        return true;
                    }
                }
            }
            PatternPart::Vowel => {
                if matches!(chars.first(), Some(c) if "AEIOUY".contains(*c)) {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            PatternPart::Consonant => {
                if matches!(chars.first(), Some(c) if "BCDFGHJKLMNPQRSTVWXZ".contains(*c)) {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            PatternPart::Charset(set) => {
                if matches!(chars.first(), Some(c) if set.contains(&c.to_ascii_lowercase())) {
                    return helper(&chars[1..], rest, bindings, results, all_matches, word, constraints);
                }
            }
            PatternPart::Anagram(s) => {
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
            PatternPart::Var(name) | PatternPart::RevVar(name) => {
                if let Some(bound_val) = bindings.get(&name) {
                    let val = get_reversed_or_not(first, bound_val);
                    if chars.starts_with(&val.chars().collect::<Vec<_>>()) {
                        return helper(&chars[val.len()..], rest, bindings, results, all_matches, word, constraints);
                    }
                } else {
                    for l in 1..=chars.len() {
                        let candidate: String = chars[..l].iter().collect();
                        let bound_val = get_reversed_or_not(first, &candidate);

                        // Apply constraints here if present
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

    // Before we do anything else, do a regex filter
    let regex_str = format!("^{}$", pattern_to_regex(parts));
    if let Ok(regex) = Regex::new(&regex_str) {
        if !regex.is_match(word) {
            return;
        }
    }

    let word = word.to_uppercase();
    let chars: Vec<char> = word.chars().collect();

    let mut bindings = Bindings::new();
    helper(&chars, parts, &mut bindings, results, all_matches, &word, constraints);
}


pub fn pattern_to_regex(parts: &[PatternPart]) -> String {
    let mut regex = String::new();
    for part in parts {
        match part {
            PatternPart::Var(_) | PatternPart::RevVar(_) => {
                regex.push_str(".+");
            },
            PatternPart::Lit(s) => {
                regex.push_str(&regex::escape(&s.to_uppercase()));
            },
            PatternPart::Dot => regex.push('.'),
            PatternPart::Star => regex.push_str(".*"),
            PatternPart::Vowel => regex.push_str("[AEIOUY]"),
            PatternPart::Consonant => regex.push_str("[BCDFGHJKLMNPQRSTVWXZ]"),
            PatternPart::Charset(chars) => {
                regex.push('[');
                for c in chars {
                    regex.push(c.to_ascii_uppercase());
                }
                regex.push(']');
            },
            PatternPart::Anagram(s) => {
                let len = s.len();
                let class = regex::escape(&s.to_uppercase());
                // TODO? do something if there's an error?
                let _ = write!(regex, "[{class}]{{{len}}}");
            },
        }
    }
    regex
}

pub fn parse_pattern(input: &str) -> Result<Vec<PatternPart>, String> {
    let mut rest = input;
    let mut parts = Vec::new();

    while !rest.is_empty() {
        match pattern_part(rest) {
            Ok((next, part)) => {
                parts.push(part);
                rest = next;
            }
            Err(_) => return Err(format!("Could not parse at: {rest}")), // TODO? avoid swallowing error?
        }
    }

    Ok(parts)
}

fn varref(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = map(one_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), PatternPart::Var);
    parser.parse(input)
}

fn revref(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = map(preceded(tag("~"), one_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ")), PatternPart::RevVar);
    parser.parse(input)
}

fn literal(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = map(many1(one_of("abcdefghijklmnopqrstuvwxyz")), |chars| {
        PatternPart::Lit(chars.into_iter().collect())
    });
    parser.parse(input)
}

fn dot(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = map(tag("."), |_| PatternPart::Dot);
    parser.parse(input)
}

fn star(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = map(tag("*"), |_| PatternPart::Star);
    parser.parse(input)
}

fn vowel(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = map(tag("@"), |_| PatternPart::Vowel);
    parser.parse(input)
}

fn consonant(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = map(tag("#"), |_| PatternPart::Consonant);
    parser.parse(input)
}

fn charset(input: &str) -> IResult<&str, PatternPart> {
    let (input, _) = tag("[")(input)?;
    let mut parser = many1(one_of("abcdefghijklmnopqrstuvwxyz"));
    let (input, chars) = parser.parse(input)?;
    let (input, _) = tag("]")(input)?;
    Ok((input, PatternPart::Charset(chars)))
}

fn anagram(input: &str) -> IResult<&str, PatternPart> {
    let (input, _) = tag("/")(input)?;
    let mut parser = many1(one_of("abcdefghijklmnopqrstuvwxyz"));
    let (input, chars) = parser.parse(input)?;
    Ok((input, PatternPart::Anagram(chars.into_iter().collect())))
}

fn pattern_part(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = alt((
        revref,
        varref,
        anagram,
        charset,
        literal,
        dot,
        star,
        vowel,
        consonant,
    ));
    parser.parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_match_pattern_exists() {
        let patt1 = parse_pattern("A~A[rstlne]/jon@#.*").unwrap();
        assert!(match_pattern_exists("AARONJUDGE", &patt1, None));
        assert!(!match_pattern_exists("NOON", &patt1, None));
        assert!(!match_pattern_exists("TOON", &patt1, None));
    }

    #[test]
    fn test_valid_binding_not_equal_pass() {
        // A must be != B; current B is bound to "TEST"
        let mut vc = VarConstraint::default();
        vc.not_equal.insert('B');

        let mut b = Bindings::new();
        b.set('B', "TEST".to_string());

        // "OTHER" != "TEST" so this should pass
        assert!(is_valid_binding("OTHER", &vc, &b));
    }

    #[test]
    fn test_valid_binding_simple_pass() {
        let mut vc = VarConstraint::default();
        vc.pattern = Option::from("abc*".to_string());

        let b = Bindings::new();
        assert!(is_valid_binding("ABCAT", &vc, &b));
    }

    #[test]
    fn test_valid_binding_pattern_fail() {
        let mut vc = VarConstraint::default();
        vc.pattern = Option::from("abc*".to_string());

        let b = Bindings::new();
        assert!(!is_valid_binding("XYZ", &vc, &b));
    }

    #[test]
    fn test_valid_binding_not_equal_fail() {
        let mut vc = VarConstraint::default();
        vc.not_equal.insert('B');

        let mut b = Bindings::new();
        b.set('B', "TEST".to_string());
        assert!(!is_valid_binding("TEST", &vc, &b));
    }

    #[test]
    fn test_match_pattern_with_constraints() {
        let patt = parse_pattern("AB").unwrap();
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
        let result = match_pattern("INCH", &patt, Some(&var_constraints));
        assert!(result.is_some());
        let m = result.unwrap();
        assert_ne!(m.get(&'A'), m.get(&'B'));
    }

    #[test]
    fn test_match_pattern_all_with_constraints() {
        let patt = parse_pattern("AA").unwrap();
        // We add length constraints
        let mut var_constraints = VarConstraints::default();
        let mut vc = VarConstraint::default();
        const MIN_LENGTH: Option<usize> = Some(2);
        const MAX_LENGTH: Option<usize> = Some(2);
        // min length 2, max_length 3
        vc.min_length = MIN_LENGTH;
        vc.max_length = MAX_LENGTH;
        // associate this constraint with variable 'A'
        var_constraints.insert('A', vc);

        let matches = match_pattern_all("INCHIN", &patt, Some(&var_constraints));
        for m in matches.iter() {
            let val = m.get(&'A').unwrap();
            assert!(val.len() >= MIN_LENGTH.unwrap() && val.len() <= MAX_LENGTH.unwrap());
        }
    }

    #[test]
    fn test_match_pattern_exists_with_constraints() {
        let patt = parse_pattern("AB").unwrap();
        // First constraint: A=(*i.*)
        let mut var_constraints1 = VarConstraints::default();
        let mut vc = VarConstraint::default();
        vc.pattern = Option::from("*i.*".to_string());
        var_constraints1.insert('A', vc.clone());

        assert!(match_pattern_exists("INCH", &patt, Some(&var_constraints1)));

        // Second constraint: A=(*z*)
        let mut var_constraints2 = VarConstraints::default();
        let mut vc2 = VarConstraint::default();
        vc2.pattern = Option::from("*z*".to_string());
        var_constraints2.insert('A', vc2.clone());

        assert!(!match_pattern_exists("INCH", &patt, Some(&var_constraints2)));
    }
}

