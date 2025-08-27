pub mod form;
pub mod prefilter;
pub mod matcher;

// Re-export the public API so existing call sites keep working.
pub use form::{parse_form, ParsedForm, FormPart};
pub use matcher::{match_equation_all, match_equation_exists};
