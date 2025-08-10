use std::collections::HashMap;
use crate::patterns::Patterns;
use crate::parser::{match_equation_all, parse_form, FormPart};

/// The max number of matches to grab during our initial pass through the word list
const MAX_INITIAL_MATCHES: usize = 50_000;

/// A single solution’s variable bindings:
/// maps a variable name (e.g., 'A') to the concrete substring it was bound to.
/// - We use `String` because bindings are slices of candidate words and may be reused;
///   if cloning shows up in profiles later, we can switch to `Arc<str>`.
pub type Binding = HashMap<char, String>;

/// Bucket key for indexing candidates by the subset of variables that must agree.
/// - `None` means “no lookup constraints for this pattern” (Python’s `words[i][None]`).
/// - When present, we store a *sorted* `(var, value)` list so the key is deterministic
///   and implements `Eq`/`Hash` naturally. This mirrors Python’s
///   `frozenset(dict(...).items())`, but with a stable order.
/// - The sort happens once when we construct the key, not on hash/compare.
pub type LookupKey = Option<Vec<(char, String)>>;

/// All candidates for one pattern (“bucketed” by `LookupKey`).
/// - `buckets`: groups candidate bindings that share the same values for the
///   pattern’s `lookup_keys` (variables that must align with previously chosen patterns).
/// - `count`: mirrors Python’s `word_counts[i]` and is used to stop early when a global cap
///   per-pattern is reached (e.g., `MAX_WORD_COUNT`). We track it here to avoid recomputing.
#[derive(Debug, Default)]
pub struct CandidateBuckets {
    /// Mapping from lookup key -> all bindings that fit that key
    pub buckets: HashMap<LookupKey, Vec<Binding>>,
    /// Total number of bindings added for this pattern (across all keys)
    pub count: usize,
}

/// Read in an equation string and return results from the word list
pub fn solve_equation(
    input: &str,
    word_list: &[&str],
    num_results: usize
) -> Vec<Vec<Binding>> {
    // 1. Build "patterns" from the input
    let pattern_obj = Patterns::new(input);

    // 2. Prepare per-pattern candidate buckets
    let mut words: Vec<CandidateBuckets> = Vec::with_capacity(pattern_obj.len());
    for _ in &pattern_obj {
        words.push(CandidateBuckets::default());
    }

    // 2. Parse each pattern once; keep index-aligned vectors
    let parsed_patterns: Vec<Vec<FormPart>> = pattern_obj
        .iter()
        .map(|p| {
            parse_form(&p.raw_string).unwrap()
        })
        .collect();

    // 3. Grab our constraints
    let var_constraints = &pattern_obj.var_constraints;

    // 4. Take an initial pass through the word list.
    // For each of our "patterns", find words that match, and add them appropriately.

    // loop through words
    'words_loop: for word in word_list.iter() {
        // loop through patterns
        for (i, patt) in pattern_obj.iter().enumerate() {
            let matches = match_equation_all(&word, &parsed_patterns[i], Some(&pattern_obj.var_constraints));
            for binding in matches {
                // Determine the hash key based on shared variable bindings
                //let key = patt.lookup_keys.as_ref().map(|keys| {
                //    keys.iter().map(|k| (k.clone(), binding.get(*k).unwrap().clone())).collect()
                //});

                //words[i].get(key).or_default().push(binding);
                //word_counts[i] += 1;
            }

            //if word_counts[i] >= MAX_INITIAL_MATCHES {
            //    break 'words_loop;
            //}
        }
    }


    // for debugging purposes
    println!("{pattern_obj:?}");
    println!("{words:?}");
    println!("{parsed_patterns:?}");
    println!("{var_constraints:?}");

    // Return an empty vec for now
    Vec::new()
}

#[test]
fn test_solve_equation() {
    let word_list: Vec<&str> = vec!["LAX", "TAX", "LOX"];
    let input = "l.x".to_string();
    solve_equation(&input, &word_list, 5);
}

#[test]
fn test_solve_equation2() {
    let word_list: Vec<&str> = vec!["INCH", "CHIN", "DADA"];
    let input = "AB;BA;|A|=2;|B|=2;!=AB".to_string();
    solve_equation(&input, &word_list, 5);
}