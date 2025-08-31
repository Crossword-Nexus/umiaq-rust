use crate::errors::ParseError;
use crate::errors::ParseError::ParseFailure;

pub(crate) fn char_to_num(c: char, base_char_as_usize: usize, num_values: usize) -> Result<usize, ParseError> {
    (c as usize).checked_sub(base_char_as_usize).and_then(|diff| {
        if diff < num_values { Some(diff) } else { None }
    }).ok_or_else(|| ParseFailure { s : format!("illegal char: {c}") })
}
