use std::io;

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
    #[error("int-parsing error: {0}")]
    ParseIntError(#[from] std::num::ParseIntError),
}

impl From<ParseError> for io::Error {
    fn from(pe: ParseError) -> Self {
        // String version is the least fragile (no Send/Sync bounds issues)
        io::Error::new(io::ErrorKind::InvalidInput, pe.to_string())
    }
}