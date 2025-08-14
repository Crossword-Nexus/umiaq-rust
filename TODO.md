* avoid swallowing errors
* add detailed error messages
* avoid `#![allow(dead_code)]`
* avoid `#![allow(unused_variables)]`
* support alternate word lists
* add progress indicators for long-running operations
* implement parallel processing for large word lists
* add integration tests with real word lists
* remove some unnecessary `to_uppercase()` statements
* allow for constraints like `|AB|=5`
* improve prefilter for, e.g., `AB;|A|=2;|B|=2` (`.{2}.{2}`) and `AA` (`(.+)\1`)
* create a `struct` for parameters for certain methods \(e.g., `helper`\)?
* return results in original order (e.g., for `C;BC;ABC`)
* actually show the "loading" text in web interface
* add support for `<` and `>` like in Qat
* misc. TODOs in code
