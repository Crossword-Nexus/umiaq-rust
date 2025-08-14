* add detailed error messages
* add progress indicators for long-running operations
* implement parallel processing for large word lists
* add integration tests with real word lists
* remove some unnecessary `to_uppercase()` statements
* allow for constraints like `|AB|=5`
* improve prefilter for, e.g., `AB;|A|=2;|B|=2` (`.{2}.{2}`) and `AA` (`(.+)\1`)
* create a `struct` for parameters for certain methods \(e.g., `helper`\)?
* return results in original order (e.g., for `C;BC;ABC`)
* add support for `<` and `>` like in Qat
* misc. TODOs in code
