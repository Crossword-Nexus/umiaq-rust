use crate::parser::{match_equation_all, parse_form, FormPart};
use crate::patterns::Patterns;
use std::collections::HashMap;

/// The max number of matches to grab during our initial pass through the word list
const MAX_INITIAL_MATCHES: usize = 50_000;

/// A single solution's variable bindings:
/// maps a variable name (e.g., 'A') to the concrete substring it was bound to.
/// - We use `String` because bindings are slices of candidate words and may be reused;
///   if cloning shows up in profiles later, we can switch to `Arc<str>`.
pub type Binding = HashMap<char, String>;

/// Bucket key for indexing candidates by the subset of variables that must agree.
/// - `None` means "no lookup constraints for this pattern" (Python's `words[i][None]`).
/// - When present, we store a *sorted* `(var, value)` list so the key is deterministic
///   and implements `Eq`/`Hash` naturally. This mirrors Python's
///   `frozenset(dict(...).items())`, but with a stable order.
/// - The sort happens once when we construct the key, not on hash/compare.
pub type LookupKey = Option<Vec<(char, String)>>;

/// All candidates for one pattern ("bucketed" by `LookupKey`).
/// - `buckets`: groups candidate bindings that share the same values for the
///   pattern's `lookup_keys` (variables that must align with previously chosen patterns).
/// - `count`: mirrors Python's `word_counts[i]` and is used to stop early when a global cap
///   per-pattern is reached (e.g., `MAX_WORD_COUNT`). We track it here to avoid recomputing.
#[derive(Debug, Default)]
pub struct CandidateBuckets {
    /// Mapping from lookup key -> all bindings that fit that key
    pub buckets: HashMap<LookupKey, Vec<Binding>>,
    /// Total number of bindings added for this pattern (across all keys)
    pub count: usize,
}

/// Read in an equation string and return results from the word list
///
/// - `input`: equation in our pattern syntax (e.g., `"AB;BA;|A|=2;..."`)
/// - `word_list`: list of candidate words to test
/// - `num_results`: maximum number of *final* results to return
///
/// Returns:
/// - A `Vec` of solutions, each solution being a `Vec<Binding>` where each `Binding`
///   maps variable names (chars) to concrete substrings they were bound to in that solution.
pub fn solve_equation(input: &str, word_list: &[&str], num_results: usize) -> Vec<Vec<Binding>> {
    // 1. Parse the input equation string into our `Patterns` struct.
    //    This holds each pattern string, its parsed form, and its `lookup_keys` (shared vars).
    let pattern_obj = Patterns::new(input);

    // 2. Prepare storage for candidate buckets, one per pattern.
    //    `CandidateBuckets` tracks (a) the bindings bucketed by shared variable values, and
    //    (b) a count so we can stop early if a pattern gets too many matches.
    let mut words: Vec<CandidateBuckets> = Vec::with_capacity(pattern_obj.len());
    for _ in &pattern_obj {
        words.push(CandidateBuckets::default());
    }

    // 3. Parse each pattern's string form once into a vector of `FormPart`s.
    //    These are index-aligned with `pattern_obj`.
    let parsed_patterns: Vec<Vec<FormPart>> = pattern_obj
        .iter()
        .map(|p| parse_form(&p.raw_string).unwrap())
        .collect();

    // 4. Pull out the per-variable constraints collected from the equation.
    let var_constraints = &pattern_obj.var_constraints;

    // 5. Iterate through every candidate word.
    'words_loop: for &word in word_list {
        // Check each pattern against this word
        for (i, patt) in pattern_obj.iter().enumerate() {
            // Skip this pattern if we already have too many matches for it
            if words[i].count >= MAX_INITIAL_MATCHES {
                continue;
            }

            // Try matching the word against the parsed pattern.
            // `match_equation_all` returns a list of `Bindings` (variable→string maps)
            // that satisfy the pattern given the current constraints.
            let matches = match_equation_all(word, &parsed_patterns[i], Some(var_constraints));

            // 6. For each binding produced for this pattern/word:
            for binding in matches {
                // ---- Build the lookup key for bucketing ----
                // `LookupKey` is:
                //   None => no shared variables with previous patterns
                //   Some(Vec<(char, String)>) => specific values for shared variables,
                //                                sorted for deterministic equality/hash.
                let key: LookupKey = match patt.lookup_keys.as_ref() {
                    None => None,
                    Some(keys) => {
                        let mut pairs: Vec<(char, String)> = Vec::with_capacity(keys.len());
                        for &var in keys {
                            if let Some(val) = binding.get(var) {
                                pairs.push((var, val.clone()));
                            } else {
                                // If any shared var is missing, this binding can't be used
                                pairs.clear();
                                break;
                            }
                        }
                        if pairs.is_empty() && !keys.is_empty() {
                            continue; // skip binding entirely
                        }
                        // Sort by variable name so the key is deterministic
                        pairs.sort_unstable_by_key(|(c, _)| *c);
                        Some(pairs)
                    }
                };

                // ---- Store the binding in the correct bucket ----
                // Clone the inner `HashMap<char, String>` from the `Bindings` wrapper
                let this_binding: Binding = binding.get_map().clone();

                // Insert into the appropriate bucket (creating a new Vec if needed)
                words[i].buckets.entry(key).or_default().push(this_binding);

                // Track how many bindings we've stored for this pattern
                words[i].count += 1;

                // Stop scanning more words entirely if this pattern hit the cap
                if words[i].count >= MAX_INITIAL_MATCHES {
                    continue 'words_loop;
                }
            }
        }
    }

    // ---- Debug: dump internal state ----
    println!("{pattern_obj:?}");
    println!("{words:?}");
    println!("{parsed_patterns:?}");
    println!("{var_constraints:?}");

    // TODO: Implement recursive join logic to combine per-pattern matches
    //       into complete solutions. For now, return an empty Vec.
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
