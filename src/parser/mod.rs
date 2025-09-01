pub mod form;
pub mod prefilter;
pub mod matcher;
mod utils;

// Re-export the public API so existing call sites keep working.
pub use form::{FormPart, ParsedForm};
pub use matcher::{match_equation_all, match_equation_exists};
