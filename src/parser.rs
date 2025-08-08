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
use std::collections::HashMap;

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

// Reinserted full implementations

pub fn is_valid_binding(
    val: &str,
    constraints: &HashMap<String, String>,
    bindings: &HashMap<String, String>,
) -> bool {
    if let Some(pattern_str) = constraints.get("pattern") {
        match parse_pattern(pattern_str) {
            Ok(p) => {
                if !match_pattern_exists(val, &p, None) {
                    return false;
                }
            }
            Err(_) => return false,
        }
    }

    if let Some(not_eq) = constraints.get("not_equal") {
        for other in not_eq.chars() {
            if let Some(existing) = bindings.get(&other.to_string()) {
                if existing == val {
                    return false;
                }
            }
        }
    }

    true
}

/// Returns the first successful binding (if any)
pub fn match_pattern(
    word: &str,
    parts: &[PatternPart],
    constraints: Option<&HashMap<String, HashMap<String, String>>>,
) -> Option<HashMap<String, String>> {
    let mut results = Vec::new();
    match_pattern_internal(word, parts, false, &mut results, constraints);
    results.into_iter().next()
}

/// Returns a boolean
pub fn match_pattern_exists(
    word: &str,
    parts: &[PatternPart],
    constraints: Option<&HashMap<String, HashMap<String, String>>>,
) -> bool {
    match_pattern(word, parts, constraints).is_some()
}

/// Returns all successful bindings
pub fn match_pattern_all(
    word: &str,
    parts: &[PatternPart],
    constraints: Option<&HashMap<String, HashMap<String, String>>>,
) -> Vec<HashMap<String, String>> {
    let mut results = Vec::new();
    match_pattern_internal(word, parts, true, &mut results, constraints);
    results
}

fn match_pattern_internal(
    word: &str,
    parts: &[PatternPart],
    all_matches: bool,
    results: &mut Vec<HashMap<String, String>>,
    constraints: Option<&HashMap<String, HashMap<String, String>>>,
) {

    // Before we do anything else, do a regex filter
    let regex_str = format!("^{}$", pattern_to_regex(parts));
    if let Ok(regex) = Regex::new(&regex_str) {
        if !regex.is_match(word) {
            return;
        }
    }

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
        bindings: &mut HashMap<String, String>,
        results: &mut Vec<HashMap<String, String>>,
        all_matches: bool,
        word: &str,
        constraints: Option<&HashMap<String, HashMap<String, String>>>,
    ) -> bool {
        if parts.is_empty() {
            if chars.is_empty() {
                let mut full_result = bindings.clone();
                full_result.insert("word".to_string(), word.to_string());
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
                let name_str = name.to_string();
                if let Some(bound_val) = bindings.get(&name_str) {
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
                            if let Some(c) = all_c.get(&name_str) {
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

                        bindings.insert(name_str.clone(), bound_val);
                        if helper(&chars[l..], rest, bindings, results, all_matches, word, constraints) && !all_matches {
                            return true;
                        }
                        bindings.remove(&name_str);
                    }
                }
            }
        }

        false
    }

    let word = word.to_uppercase();
    let chars: Vec<char> = word.chars().collect();

    let mut bindings = HashMap::new();
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
            PatternPart::Consonant => regex.push_str("[B-DF-HJ-NP-TV-Z]"),
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
                regex.push_str(&format!("[{class}]{{{len}}}"));
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
            Err(_) => return Err(format!("Could not parse at: {rest}")),
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
    fn test_valid_binding_simple_pass() {
        let constraints = HashMap::from([("pattern".to_string(), "abc*".to_string())]);
        let bindings = HashMap::new();
        assert!(is_valid_binding("ABCAT", &constraints, &bindings));
    }

    #[test]
    fn test_valid_binding_pattern_fail() {
        let constraints = HashMap::from([("pattern".to_string(), "abc*".to_string())]);
        let bindings = HashMap::new();
        assert!(!is_valid_binding("XYZ", &constraints, &bindings));
    }

    #[test]
    fn test_valid_binding_not_equal_fail() {
        let constraints = HashMap::from([("not_equal".to_string(), "B".to_string())]);
        let bindings = HashMap::from([("B".to_string(), "TEST".to_string())]);
        assert!(!is_valid_binding("TEST", &constraints, &bindings));
    }

    #[test]
    fn test_valid_binding_not_equal_pass() {
        let constraints = HashMap::from([("not_equal".to_string(), "B".to_string())]);
        let bindings = HashMap::from([("B".to_string(), "TEST".to_string())]);
        assert!(is_valid_binding("OTHER", &constraints, &bindings));
    }

    #[test]
    fn test_match_pattern_with_constraints() {
        let patt = parse_pattern("AB").unwrap();
        let constraints = HashMap::from([
            ("A".to_string(), HashMap::from([
                ("not_equal".to_string(), "B".to_string())
            ])),
            ("B".to_string(), HashMap::new())
        ]);
        let result = match_pattern("INCH", &patt, Some(&constraints));
        assert!(result.is_some());
        let m = result.unwrap();
        assert_ne!(m.get("A"), m.get("B"));
    }

    #[test]
    fn test_match_pattern_all_with_constraints() {
        let patt = parse_pattern("AA").unwrap();
        let constraints = HashMap::from([
            ("A".to_string(), HashMap::from([
                ("min_length".to_string(), "2".to_string()),
                ("max_length".to_string(), "3".to_string())
            ]))
        ]);
        let matches = match_pattern_all("INCHIN", &patt, Some(&constraints));
        for m in matches.iter() {
            let val = m.get("A").unwrap();
            assert!(val.len() >= 2 && val.len() <= 3);
        }
    }

    #[test]
    fn test_match_pattern_exists_with_constraints() {
        let patt = parse_pattern("AB").unwrap();
        let constraints = HashMap::from([
            ("A".to_string(), HashMap::from([("pattern".to_string(), "*i.*".to_string())]))
        ]);
        assert!(match_pattern_exists("INCH", &patt, Some(&constraints)));
        let constraints2 = HashMap::from([
            ("A".to_string(), HashMap::from([("pattern".to_string(), "*z*".to_string())]))
        ]);
        assert!(!match_pattern_exists("INCH", &patt, Some(&constraints2)));
    }
}

