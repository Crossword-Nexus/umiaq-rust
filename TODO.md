* populate this file (`TODO.md`) with TODOs found in code comments
* organize this file (at least sort... somehow)
* fix "2-", "-4", etc. in literal constraints
* allow for "complex" constraints like `A=(g*)`
* add detailed error messages
* add progress indicators for long-running operations
* implement parallel processing for large word lists
* add integration tests with real word lists
* literals and word-list entries to LC ASAP (and solutions to UC at last moment)
  * remove some unnecessary `to_uppercase()` statements
* allow for constraints like `|AB|=5`
* improve prefilter for, e.g., `AB;|A|=2;|B|=2` (`.{2}.{2}`)
* create a `struct` for parameters for certain methods \(e.g., `helper`\)?
* add support for `<` and `>` like in Qat
* "binding" vs. "bindings" in variable names, comments, etc.
* use `Range`s for length ranges?
* add length constraints to individual variables when there is a joint constraint
* optimize joint constraints
* allow constraints like |A|<4, |B|>=5, etc.
* consider adding a struct when faced with long argument lists (for methods)
* throw exception if cannot parse form in `make_list` (else case)
* misc. TODOs in code
