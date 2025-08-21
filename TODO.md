* populate this file (`TODO.md`) with TODOs found in code comments
* organize this file (at least sort... somehow)
* implement better prefilters for Vars with forms (e.g., `A;A=(x*a)` currently has a prefilter of `.+`)
* (?) add methods on `char` (and `String`?): `is_variable` and `is_literal` (just sugar for `is_ascii_uppercase`, `is_ascii_lowercase`)
* return `None` vs `Err`
* avoid duplicating work (e.g., `parse_form` call in `make_list`)
* consistency in using `usize::MAX` vs. `None` for unbounded-above lengths(?)
* add detailed error messages
* add progress indicators for long-running operations
* implement parallel processing for large word lists
  * possible concern with web interface?
* add integration tests with real word lists
* create a `struct` for parameters for certain methods \(e.g., `helper`\)?
* "binding" vs. "bindings" in variable names, comments, etc.
* use `Range`s for length ranges?
* allow constraints like |A|<4, |B|>=5, etc.
* consider adding a struct when faced with long argument lists (for methods)
* throw exception if cannot parse form in `make_list` (else case)
* misc. TODOs in code
