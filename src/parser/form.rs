use crate::umiaq_char::{LITERAL_CHARS, VARIABLE_CHARS};
use fancy_regex::Regex;
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
use crate::errors::ParseError;
use crate::errors::ParseError::ParseFailure;
use crate::parser::utils::char_to_num;
use super::prefilter::{form_to_regex_str, get_regex};

const ALPHABET_SIZE:usize = 26;

/// Represents a single parsed token (component) from a "form" string.
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
    Anagram(Alphagram), // '/abc': any permutation of the given letters
}

impl FormPart {
    pub(crate) fn is_deterministic(&self) -> bool {
        matches!(self, FormPart::Var(_) | FormPart::RevVar(_) | FormPart::Lit(_))
    }

    fn get_tag_string(&self) -> Option<&str> {
        match self {
            FormPart::Dot => Some("."),
            FormPart::Star => Some("*"),
            FormPart::Vowel => Some("@"),
            FormPart::Consonant => Some("#"),
            _ => None // Only the single-char tokens have tags
        }
    }

    pub(crate) fn anagram_of(s: String) -> Result<FormPart, crate::errors::ParseError> {
        Ok(FormPart::Anagram(Alphagram::of(s)?))
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Alphagram {
    char_counts: [u8; ALPHABET_SIZE],
    pub(crate) as_string: String, // for regexes, pretty printing
    pub(crate) len: usize
}

// 'a' -> 0, 'b' -> 1, ..., 'z' -> 25
fn lc_char_to_num(c: char) -> Result<usize, ParseError> { char_to_num(c, 'a' as usize) }

impl Alphagram {
    // TODO? don't assume lowercase?
    fn of(lowercase_word: String) -> Result<Alphagram, crate::errors::ParseError> {
        let mut len = 0;
        let mut char_counts = [0u8; ALPHABET_SIZE];
        for c in lowercase_word.chars() {
            let c_as_num = lc_char_to_num(c)?;
            if c_as_num < ALPHABET_SIZE {
                char_counts[c_as_num] += 1;
                len += 1;
            } else {
                return Err(ParseFailure{ s: lowercase_word })
            }
        }

        Ok(Alphagram { char_counts, as_string: lowercase_word, len })
    }

    pub(crate) fn is_anagram(&self, other_word: &[char]) -> Result<bool, crate::errors::ParseError> {
        if self.len != other_word.len() {
            return Ok(false);
        }

        let mut char_counts = self.char_counts;
        for &c in other_word {
            let c_as_num = lc_char_to_num(c)?;
            if c_as_num < ALPHABET_SIZE {
                if char_counts[c_as_num] == 0 {
                    return Ok(false);
                }
                char_counts[c as usize] -= 1;
            } else {
                return Err(ParseFailure{ s: other_word.iter().collect() })
            }
        }

        Ok(char_counts.iter().all(|&count| count == 0))
    }
}

/// A `Vec` of `FormPart`s along with a compiled regex prefilter.
#[derive(Debug, Clone)]
pub struct ParsedForm {
    pub parts: Vec<FormPart>,
    pub prefilter: Regex,
}

impl ParsedForm {
    fn of(parts: Vec<FormPart>) -> Result<Self, ParseError> {
        // Build the base regex string from tokens only (no var-constraints).
        let regex_str = form_to_regex_str(&parts)?;
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
        env: &std::collections::HashMap<char, String>,
    ) -> Option<String> {
        self.iter()
            .map(|part| match part {
                FormPart::Lit(s) => Some(s.clone()),
                FormPart::Var(v) => Some(env.get(v)?.clone()),
                FormPart::RevVar(v) => Some(env.get(v)?.chars().rev().collect()),
                _ => None, // stop at first nondeterministic token
            })
            .collect::<Option<String>>()
    }
}

// Enable `for part in &parsed_form { ... }`
impl<'a> IntoIterator for &'a ParsedForm {
    type Item = &'a FormPart;
    type IntoIter = std::slice::Iter<'a, FormPart>;
    fn into_iter(self) -> Self::IntoIter { self.parts.iter() }
}

/// Parse a form string into a `ParsedForm` object.
///
/// Walks the input, consuming tokens one at a time with `equation_part`.
pub fn parse_form(raw_form: &str) -> Result<ParsedForm, crate::errors::ParseError> {
    let mut rest = raw_form;
    let mut parts = Vec::new();

    while !rest.is_empty() {
        match equation_part(rest) {
            Ok((next, part)) => {
                parts.push(part);
                rest = next;
            }
            Err(_) => return Err(ParseFailure { s: rest.to_string() }),
        }
    }

    if parts.is_empty() {
        return Err(crate::errors::ParseError::EmptyForm);
    }

    ParsedForm::of(parts)
}

// === Token parsers ===

fn varref(input: &str) -> IResult<&str, FormPart> {
    map(one_of(VARIABLE_CHARS), FormPart::Var).parse(input)
}
fn revref(input: &str) -> IResult<&str, FormPart> {
    map(preceded(tag("~"), one_of(VARIABLE_CHARS)), FormPart::RevVar).parse(input)
}
fn literal(input: &str) -> IResult<&str, FormPart> {
    map(many1(one_of(LITERAL_CHARS)), |chars| {
        FormPart::Lit(chars.into_iter().collect())
    })
    .parse(input)
}
fn dot(input: &str) -> IResult<&str, FormPart> { parser_one_char_inner(input, &FormPart::Dot) }
fn star(input: &str) -> IResult<&str, FormPart> { parser_one_char_inner(input, &FormPart::Star) }
fn vowel(input: &str) -> IResult<&str, FormPart> { parser_one_char_inner(input, &FormPart::Vowel) }
fn consonant(input: &str) -> IResult<&str, FormPart> { parser_one_char_inner(input, &FormPart::Consonant) }

// single-char tokens share the same shape
fn parser_one_char_inner<'a>(input: &'a str, form_part: &FormPart) -> IResult<&'a str, FormPart> {
    map(tag(form_part.get_tag_string().unwrap()), |_| form_part.clone()).parse(input)
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
    Ok((input, FormPart::anagram_of(chars.into_iter().collect()).unwrap())) // TODO handle error (better than unwrap)
}

fn equation_part(input: &str) -> IResult<&str, FormPart> {
    alt((revref, varref, anagram, charset, literal, dot, star, vowel, consonant)).parse(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_empty_form_error() {
        assert!(matches!(parse_form("").unwrap_err(), crate::errors::ParseError::EmptyForm));
    }

    #[test] fn test_parse_failure_error() {
        assert!(matches!(parse_form("[").unwrap_err(), ParseFailure { .. }));
    }

    #[test] fn test_parse_form_basic() {
        let parsed_form = parse_form("abc").unwrap();
        assert_eq!(vec![FormPart::Lit("abc".to_string())], parsed_form.parts);
    }

    #[test] fn test_parse_form_variable() {
        assert_eq!(vec![FormPart::Var('A')], parse_form("A").unwrap().parts);
    }

    #[test] fn test_parse_form_reversed_variable() {
        assert_eq!(vec![FormPart::RevVar('A')], parse_form("~A").unwrap().parts);
    }

    #[test] fn test_parse_form_wildcards() {
        let parts = parse_form(".*@#").unwrap().parts;
        assert_eq!(vec![FormPart::Dot, FormPart::Star, FormPart::Vowel, FormPart::Consonant], parts);
    }

    #[test] fn test_parse_form_charset() {
        assert_eq!(vec![FormPart::Charset(vec!['a','b','c'])], parse_form("[abc]").unwrap().parts);
    }

    #[test] fn test_parse_form_anagram() {
        assert_eq!(vec![FormPart::anagram_of("abc".to_string()).unwrap()], parse_form("/abc").unwrap().parts);
    }

    // only lowercase is allowed
    #[test] fn test_parse_form_anagram_bad_char() {
        assert!(FormPart::anagram_of("aBc".to_string()).is_err());
    }
}
