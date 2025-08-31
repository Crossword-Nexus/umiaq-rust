use crate::errors::ParseError;
use crate::errors::ParseError::ParseFailure;

pub(crate) fn char_to_num(c: char, base_char_as_usize: usize) -> Result<usize, ParseError> {
    (c as usize).checked_sub(base_char_as_usize).ok_or_else(|| ParseFailure { s : format!("illegal char: {c}") })
}
