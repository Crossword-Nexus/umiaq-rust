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

fn is_valid_binding(
    val: &str,
    constraints: &HashMap<String, String>,
    bindings: &HashMap<String, String>,
) -> bool {
    if let Some(pattern_str) = constraints.get("pattern") {
        if !match_pattern_exists(val, &parse_pattern(pattern_str).unwrap()) {
            return false;
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
pub fn match_pattern(word: &str, parts: &[PatternPart]) -> Option<HashMap<String, String>> {
    let mut results = Vec::new();
    match_pattern_internal(word, parts, false, &mut results);
    results.into_iter().next()
}

/// Returns a boolean
pub fn match_pattern_exists(word: &str, parts: &[PatternPart]) -> bool {
    match_pattern(word, parts).is_some()
}


/// Returns all successful bindings
pub fn match_pattern_all(word: &str, parts: &[PatternPart]) -> Vec<HashMap<String, String>> {
    let mut results = Vec::new();
    match_pattern_internal(word, parts, true, &mut results);
    results
}

fn match_pattern_internal(
    word: &str,
    parts: &[PatternPart],
    all_matches: bool,
    results: &mut Vec<HashMap<String, String>>,
) {
    let word = word.to_uppercase();
    let chars: Vec<char> = word.chars().collect();

    fn helper(
        chars: &[char],
        parts: &[PatternPart],
        bindings: &mut HashMap<String, String>,
        results: &mut Vec<HashMap<String, String>>,
        all_matches: bool,
    ) -> bool {
        if parts.is_empty() {
            if chars.is_empty() {
                results.push(bindings.clone());
                return !all_matches; // Stop if we only want the first match
            }
            return false;
        }

        let (first, rest) = (&parts[0], &parts[1..]);

        match first {
            PatternPart::Lit(s) => {
                let s = s.to_uppercase();
                if chars.starts_with(&s.chars().collect::<Vec<_>>()) {
                    return helper(&chars[s.len()..], rest, bindings, results, all_matches);
                }
            }
            PatternPart::Dot => {
                if !chars.is_empty() {
                    return helper(&chars[1..], rest, bindings, results, all_matches);
                }
            }
            PatternPart::Star => {
                for i in 0..=chars.len() {
                    if helper(&chars[i..], rest, bindings, results, all_matches) && !all_matches {
                        return true;
                    }
                }
            }
            PatternPart::Vowel => {
                if matches!(chars.first(), Some(c) if "AEIOUY".contains(*c)) {
                    return helper(&chars[1..], rest, bindings, results, all_matches);
                }
            }
            PatternPart::Consonant => {
                if matches!(chars.first(), Some(c) if "BCDFGHJKLMNPQRSTVWXZ".contains(*c)) {
                    return helper(&chars[1..], rest, bindings, results, all_matches);
                }
            }
            PatternPart::Charset(set) => {
                if matches!(chars.first(), Some(c) if set.contains(&c.to_ascii_lowercase())) {
                    return helper(&chars[1..], rest, bindings, results, all_matches);
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
                        return helper(&chars[len..], rest, bindings, results, all_matches);
                    }
                }
            }
            PatternPart::Var(name) | PatternPart::RevVar(name) => {
                let name_str = name.to_string();
                if let Some(bound_val) = bindings.get(&name_str) {
                    let val = if matches!(first, PatternPart::RevVar(_)) {
                        bound_val.chars().rev().collect::<String>()
                    } else {
                        bound_val.clone()
                    };
                    if chars.starts_with(&val.chars().collect::<Vec<_>>()) {
                        return helper(&chars[val.len()..], rest, bindings, results, all_matches);
                    }
                } else {
                    for l in 1..=chars.len() {
                        let candidate: String = chars[..l].iter().collect();
                        let bound_val = if matches!(first, PatternPart::RevVar(_)) {
                            candidate.chars().rev().collect::<String>()
                        } else {
                            candidate.clone()
                        };
                        bindings.insert(name_str.clone(), bound_val);
                        if helper(&chars[l..], rest, bindings, results, all_matches) && !all_matches {
                            return true;
                        }
                        bindings.remove(&name_str);
                    }
                }
            }
        }

        false
    }

    let mut bindings = HashMap::new();
    helper(&chars, parts, &mut bindings, results, all_matches);
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
                regex.push_str(&format!("[{}]{{{}}}", class, len));
            },
        }
    }
    regex
}

fn parse_pattern(input: &str) -> Result<Vec<PatternPart>, String> {
    let mut rest = input;
    let mut parts = Vec::new();

    while !rest.is_empty() {
        match pattern_part(rest) {
            Ok((next, part)) => {
                parts.push(part);
                rest = next;
            }
            Err(_) => return Err(format!("Could not parse at: {}", rest)),
        }
    }

    Ok(parts)
}

fn varref(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = map(one_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), |c| PatternPart::Var(c));
    parser.parse(input)
}

fn revref(input: &str) -> IResult<&str, PatternPart> {
    let mut parser = map(preceded(tag("~"), one_of("ABCDEFGHIJKLMNOPQRSTUVWXYZ")), |c| PatternPart::RevVar(c));
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

// Checks if a word matches the pattern
pub fn word_matches_pattern(word: &str, pattern: &str) -> Result<bool, String> {
    let parts = parse_pattern(pattern)?;

    // First, check fast regex filter
    let regex_str = pattern_to_regex(&parts);
    let re = Regex::new(&format!("^{}$", regex_str)).map_err(|e| e.to_string())?;
    if !re.is_match(&word.to_uppercase()) {
        return Ok(false);
    }

    // Then run recursive matcher
    Ok(match_pattern_exists(word, &parts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_binding_simple_pass() {
        let mut constraints = HashMap::new();
        constraints.insert("pattern".to_string(), "abc*".to_string());
        let bindings = HashMap::new();
        assert!(is_valid_binding("abcat", &constraints, &bindings));
    }

    #[test]
    fn test_valid_binding_pattern_fail() {
        let mut constraints = HashMap::new();
        constraints.insert("pattern".to_string(), "abc*".to_string());
        let bindings = HashMap::new();
        assert!(!is_valid_binding("xyz", &constraints, &bindings));
    }

    #[test]
    fn test_valid_binding_not_equal_fail() {
        let mut constraints = HashMap::new();
        constraints.insert("not_equal".to_string(), "B".to_string());
        let mut bindings = HashMap::new();
        bindings.insert("B".to_string(), "TEST".to_string());
        assert!(!is_valid_binding("TEST", &constraints, &bindings));
    }

    #[test]
    fn test_valid_binding_not_equal_pass() {
        let mut constraints = HashMap::new();
        constraints.insert("not_equal".to_string(), "B".to_string());
        let mut bindings = HashMap::new();
        bindings.insert("B".to_string(), "TEST".to_string());
        assert!(is_valid_binding("OTHER", &constraints, &bindings));
    }

    #[test]
    fn test_match_literal_and_dot() {
        let patt = parse_pattern("a.b").unwrap();
        assert!(match_pattern_exists("acb", &patt));
        assert!(!match_pattern_exists("ab", &patt));
    }

    #[test]
    fn test_match_star_and_vowel() {
        let patt = parse_pattern("*a@b").unwrap();
        assert!(match_pattern_exists("zzzaob", &patt));
        assert!(!match_pattern_exists("zzzbb", &patt));
    }

    #[test]
    fn test_match_charset_and_anagram() {
        let patt = parse_pattern("[abc]/cat").unwrap();
        assert!(match_pattern_exists("acat", &patt));
        assert!(match_pattern_exists("bcta", &patt));
        assert!(!match_pattern_exists("bxyz", &patt));
    }

    #[test]
    fn test_match_variable_and_reverse() {
        let patt = parse_pattern("A.~A").unwrap();
        assert!(!match_pattern_exists("DEED", &patt));
        assert!(match_pattern_exists("RACECAR", &patt));
        assert!(!match_pattern_exists("TEST", &patt));
    }

    #[test]
    fn test_word_matches_pattern_wrapper() {
        assert!(word_matches_pattern("DEED", "A~A").unwrap());
        assert!(!word_matches_pattern("TEST", "A~A").unwrap());
        assert!(word_matches_pattern("AARONJUDGE", "A~A[rstlne]/jon@#.*").unwrap());
    }

    #[test]
    fn test_match_pattern_all_examples() {
        // INCH against AB (3 matches)
        let patt = parse_pattern("AB").unwrap();
        let matches = match_pattern_all("INCH", &patt);
        assert_eq!(matches.len(), 3, "Expected 3 matches for INCH with AB");
        for (i, m) in matches.iter().enumerate() {
            println!("INCH vs AB Match {}: {:?}", i + 1, m);
        }

        // INCH against AA (should be empty)
        let patt = parse_pattern("AA").unwrap();
        let matches = match_pattern_all("INCH", &patt);
        assert_eq!(matches.len(), 0, "Expected 0 matches for INCH with AA");

        // DEEMED against A*~A
        let patt = parse_pattern("A*~A").unwrap();
        let matches = match_pattern_all("DEEMED", &patt);
        assert_eq!(matches.len(), 2, "Expected 2 matches for DEEMED with A*~A");
        for (i, m) in matches.iter().enumerate() {
            println!("DEEMED vs A*~A Match {}: {:?}", i + 1, m);
        }
    }
}
