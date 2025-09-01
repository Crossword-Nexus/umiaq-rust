use crate::errors::ParseError;
use crate::errors::ParseError::ParseFailure;
use crate::umiaq_char::ALPHABET_SIZE;

pub(crate) fn letter_to_num(c: char, base_char_as_usize: usize) -> Result<usize, Box<ParseError>> {
    (c as usize).checked_sub(base_char_as_usize).and_then(|diff| {
        if diff < ALPHABET_SIZE { Some(diff) } else { None }
    }).ok_or_else(|| Box::new(ParseFailure { s : format!("Illegal char: '{c}'") }))
}
