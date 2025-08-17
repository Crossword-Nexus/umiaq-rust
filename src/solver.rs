use crate::bindings::{Bindings, WORD_SENTINEL};
use crate::joint_constraints::parse_joint_constraints;
use crate::parser::{match_equation_all, parse_form, ParseError, ParsedForm};
use crate::patterns::Patterns;

use std::collections::{HashMap, HashSet};

/// The max number of matches to grab during our initial pass through the word list
const MAX_INITIAL_MATCHES: usize = 50_000;

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
    pub buckets: HashMap<LookupKey, Vec<Bindings>>,
    /// Total number of bindings added for this pattern (across all keys)
    pub count: usize,
}

/// Depth-first recursive join of per-pattern candidate buckets into full solutions.
///
/// This mirrors `recursive_filter` from `umiaq.py`. We walk patterns in order
/// (index `idx`) and at each step select only the bucket of candidates whose
/// shared variables agree with what we’ve already chosen (`env`).
///
/// Parameters:
/// - `idx`: which pattern we’re placing now (0-based).
/// - `words`: per-pattern candidate buckets (what you built during scanning).
/// - `lookup_keys`: for each pattern, which variables must agree with previously
///   chosen patterns. `None` means “no lookup constraint” (use the `None` bucket).
///   `Some(vec)` means we must look up a concrete `Some(sorted_pairs)` key—even if
///   `vec` is empty.
/// - `selected`: the partial solution (one chosen Binding per pattern so far).
/// - `env`: the accumulated variable → value environment from earlier choices.
/// - `results`: completed solutions (each is a Vec<Binding>, one per pattern).
/// - `num_results`: cap on how many full solutions to collect.
///
/// Return:
/// - This function mutates `results` and stops early once it has `num_results`.
fn recursive_join(
    idx: usize,
    words: &Vec<CandidateBuckets>,
    lookup_keys: &Vec<Option<HashSet<char>>>,
    selected: &mut Vec<Bindings>,
    env: &mut HashMap<char, String>,
    results: &mut Vec<Vec<Bindings>>,
    num_results: usize,
    patterns: &Patterns,                 // for patt.deterministic / vars / lookup_keys
    parsed_forms: &Vec<ParsedForm>,      // same order as `words` / `patterns.ordered_list`
    word_list_as_set: &HashSet<&str>,
) {
    // Stop if we’ve met the requested quota of full solutions.
    if results.len() >= num_results {
        return;
    }

    // Base case: if we’ve placed all patterns, `selected` is a full solution.
    if idx == words.len() {
        results.push(selected.clone());
        return;
    }

    // ---- FAST PATH: deterministic + fully keyed ----------------------------
    let patt = &patterns.ordered_list[idx];
    if patt.is_deterministic && patt.all_vars_in_lookup_keys() {
        // The word is fully determined by literals + already-bound vars in `env`.
        let pf = &parsed_forms[idx];
        if let Some(expected) = pf.materialize_deterministic_with_env(env) {
            if !word_list_as_set.contains(expected.as_str()) {
                // This branch cannot succeed — prune immediately.
                return;
            }

            // Build a minimal Bindings for this pattern:
            // - include WORD_SENTINEL (whole word)
            // - include only vars that belong to this pattern (they must already be in env)
            let mut binding = Bindings::default();
            binding.set_word(&expected);
            for &v in &patt.variables {
                // safe to unwrap because all vars are in lookup_keys ⇒ must be in env
                if let Some(val) = env.get(&v) {
                    binding.set(v, val.clone());
                }
            }

            selected.push(binding);
            recursive_join(
                idx + 1, words, lookup_keys, selected, env, results, num_results,
                patterns, parsed_forms, word_list_as_set,
            );
            selected.pop();
            return; // IMPORTANT: skip normal enumeration path
        } else {
            // Not actually materializable (shouldn't happen if patt.deterministic is correct)
            // TODO throw error?
            return;
        }
    }
    // ------------------------------------------------------------------------

    // Decide which bucket of candidates to iterate for pattern `idx`.
    //
    // - If this pattern has `None` lookup_keys, we use the `None` bucket.
    // - If it has `Some(keys)`, we must create the deterministic key
    //   `Some(sorted_pairs)` using the current `env` and fetch that bucket.
    //   (This includes the case keys.is_empty() → key is `Some([])`.)
    let bucket_candidates_opt: Option<&Vec<Bindings>> = match &lookup_keys[idx] {
        None => {
            // No shared vars for this pattern → use the None bucket.
            words[idx].buckets.get(&None)
        }
        Some(keys) => {
            // Build (var, value) pairs from env using the set of shared vars.
            // NOTE: HashSet iteration order is arbitrary — we sort the pairs below
            // so the final key is stable/deterministic.
            let mut pairs: Vec<(char, String)> = Vec::with_capacity(keys.len());
            for &var in keys {
                if let Some(v) = env.get(&var) {
                    pairs.push((var, v.clone()));
                } else {
                    // If any required var isn’t bound yet, there can be no matches for this branch.
                    return;
                }
            }
            // Deterministic key: sort by the variable name.
            pairs.sort_unstable_by_key(|(c, _)| *c);

            // IMPORTANT: if `keys` is empty, `pairs` is empty → we intentionally
            // look up the `Some([])` bucket (not `None`). This matches the way you
            // bucketed candidates during the scan phase.
            words[idx].buckets.get(&Some(pairs))
        }
    };

    // If there are no candidates in that bucket, dead-end this branch.
    let Some(bucket_candidates) = bucket_candidates_opt else {
        return;
    };

    // Try each candidate binding for this pattern.
    for cand in bucket_candidates {
        if results.len() >= num_results {
            break; // stop early if we’ve already met the quota
        }

        // Defensive compatibility check: if a variable is already in `env`,
        // its value must match the candidate. This *should* already be true
        // because we selected the bucket using the shared vars—but keep this
        // in case upstream bucketing logic ever changes.
        let mut compatible = true;
        for (k, v) in cand.iter() {
            if *k == WORD_SENTINEL {
                continue; // ignore the “whole word” sentinel binding
            }
            if let Some(prev) = env.get(k) && prev != v {
                compatible = false;
                break;
            }
        }
        if !compatible {
            continue;
        }

        // Extend `env` with any *new* bindings from this candidate (don’t overwrite).
        // Track what we added so we can backtrack cleanly.
        let mut added_vars: Vec<char> = Vec::new();
        for (k, v) in cand.iter() {
            if *k == WORD_SENTINEL {
                continue;
            }
            if !env.contains_key(k) {
                env.insert(*k, v.clone());
                added_vars.push(*k);
            }
        }

        // Choose this candidate for pattern `idx` and recurse for `idx + 1`.
        selected.push(cand.clone());
        recursive_join(idx + 1, words, lookup_keys, selected, env, results, num_results,
                       patterns, parsed_forms, word_list_as_set);
        selected.pop();

        // Backtrack: remove only what we added at this level.
        for k in added_vars {
            env.remove(&k);
        }
    }
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
///
/// # Errors
///
/// Will return a `ParseError` if a form cannot be parsed.
// TODO? add more detail in Errors section
pub fn solve_equation(input: &str, word_list: &[&str], num_results: usize) -> Result<Vec<Vec<Bindings>>, ParseError> {
    // 0. Make a hash set version of our word list
    let word_list_as_set: HashSet<&str> = word_list.iter().copied().collect();

    // 1. Parse the input equation string into our `Patterns` struct.
    //    This holds each pattern string, its parsed form, and its `lookup_keys` (shared vars).
    let patterns = Patterns::of(input);

    // 2. Prepare storage for candidate buckets, one per pattern.
    //    `CandidateBuckets` tracks (a) the bindings bucketed by shared variable values, and
    //    (b) a count so we can stop early if a pattern gets too many matches.
    let mut words: Vec<CandidateBuckets> = Vec::with_capacity(patterns.len()); // TODO why mutable?
    for _ in &patterns {
        words.push(CandidateBuckets::default());
    }

    // 3. Parse each pattern's string form once into a vector of `FormPart`s.
    //    These are index-aligned with `patterns`.
    // TODO inline parsed_forms_result
    let parsed_forms_result: Result<Vec<_>, _> = patterns
        .iter()
        .map(|p| parse_form(&p.raw_string))
        .collect();
    let parsed_forms = parsed_forms_result?;


    // 4. Pull out the per-variable constraints collected from the equation.
    let var_constraints = &patterns.var_constraints;

    // 4a. Get the joint constraints
    let joint_constraints = parse_joint_constraints(input);

    // 5. Iterate through every candidate word.
    'words_loop: for &word in word_list {
        // Check each pattern against this word
        for (i, patt) in patterns.iter().enumerate() {
            // Skip this pattern if we already have too many matches for it
            if words[i].count >= MAX_INITIAL_MATCHES { // TODO is there a better way to handle this? could lead to 0 final outputs when there are some...
                continue;
            }

            // Skip this pattern if it is deterministic and fully bound
            if patt.is_deterministic && patt.all_vars_in_lookup_keys() {
                continue;
            }

            // Try matching the word against the parsed pattern.
            // `match_equation_all` returns a list of `Bindings` (variable→string maps)
            // that satisfy the pattern given the current constraints.
            let matches = match_equation_all(word, &parsed_forms[i], Some(var_constraints), joint_constraints.as_ref());

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
                // Clone the "Bindings"
                let this_binding: Bindings = binding.clone();

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

    // ---- Recursive join (like umiaq.py’s recursive_filter) ----
    //
    // Build an index-aligned vector of `lookup_keys` (one per pattern). Each entry:
    //   - None  → this pattern contributes to the `None` bucket (no shared vars)
    //   - Some(vec_of_vars) → this pattern is bucketed by a deterministic
    //                         `Some(sorted (var,value) pairs)` key
    //
    // NOTE: We clone since `p.lookup_keys` is an Option<Vec<char>>. If you’d rather
    // borrow, you can restructure `recursive_join` to accept slices instead.
    let lookup_keys: Vec<Option<HashSet<char>>> = patterns
        .iter()
        .map(|p| p.lookup_keys.clone())
        .collect();

    // Accumulators for the DFS:
    // - `results`: finished solutions (Vec<Binding> per pattern)
    // - `selected`: the current partial solution down this branch
    // - `env`: running map of variable → concrete string, used to enforce joins
    let mut results: Vec<Vec<Bindings>> = Vec::new();
    let mut selected: Vec<Bindings> = Vec::new();
    let mut env: HashMap<char, String> = HashMap::new();

    // Kick off the depth-first join from pattern 0.
    recursive_join(
        0,              // start at first pattern
        &words,         // per-pattern buckets you built above
        &lookup_keys,   // which variables must agree with previous choices
        &mut selected,  // current partial solution (initially empty)
        &mut env,       // current variable environment (initially empty)
        &mut results,   // collect final solutions here
        num_results,    // stop once we have this many solutions
        &patterns,
        &parsed_forms,
        &word_list_as_set,
    );

    // ---- Reorder solutions back to original form order ----
    let reordered = results.iter().map(|solution| {
        (0..solution.len()).map(|original_i| {
            (solution.clone())[patterns.original_to_ordered[original_i]].clone()
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    // Return up to `num_results` reordered solutions
    Ok(reordered)
}

#[test]
fn test_solve_equation() {
    let word_list: Vec<&str> = vec!["LAX", "TAX", "LOX"];
    let input = "l.x".to_string();
    let results = solve_equation(&input, &word_list, 5).unwrap();
    println!("{:?}", results);
    assert_eq!(2, results.len());
}

#[test]
fn test_solve_equation2() {
    let word_list: Vec<&str> = vec!["INCH", "CHIN", "DADA", "TEST", "AB"];
    let input = "AB;BA;|A|=2;|B|=2;!=AB".to_string();
    let results = solve_equation(&input, &word_list, 5).unwrap();
    println!("{:?}", results);
    assert_eq!(2, results.len());
}

#[test]
fn test_solve_equation3() {
    let word_list = vec!["INCH", "CHIN", "DADA", "TEST", "SKY", "SLY"];
    let input = "AkB;AlB".to_string();
    let results = solve_equation(&input, &word_list, 5).unwrap();

    let sky_bindings = Bindings { map: HashMap::from([('*', "SKY".to_string()), ('A', "S".to_string()), ('B', "Y".to_string())]) };
    let sly_bindings = Bindings { map: HashMap::from([('*', "SLY".to_string()), ('A', "S".to_string()), ('B', "Y".to_string())]) };
    // NB: this could give a false negative if SLY comes out before SKY (since we presumably shouldn't care about the order), so...
    // TODO allow order independence for equality... perhaps create a richer struct than just Vec<Bindings> that has a notion of order-independent equality
    let expected = Vec::from([Vec::from([sky_bindings, sly_bindings])]);
    assert_eq!(expected, results);
}
