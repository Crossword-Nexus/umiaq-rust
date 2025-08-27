use crate::bindings::{Bindings, WORD_SENTINEL};
use crate::joint_constraints::{parse_joint_constraints, propagate_joint_to_var_bounds, JointConstraints};
use crate::parser::{
    match_equation_all,
    parse_form,
    ParseError,
    ParsedForm,
    has_inlineable_var_form,
    form_to_regex_str_with_constraints,
    get_regex,
};
use crate::patterns::Patterns;
use crate::scan_hints::{form_len_hints_pf, PatternLenHints};

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::collections::hash_map::DefaultHasher;
use instant::Instant;
use std::time::Duration;

// The amount of time (in seconds) we allow the query to run
const TIME_BUDGET: u64 = 30;
// The initial number of words from the word list we look through
const BATCH_SIZE: usize = 10_000;
// A constant to split up items in our hashes
const HASH_SPLIT: u16 = 0xFFFFu16;

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

/// Build a stable key for a full solution (bindings in **pattern order**).
/// Prefer the whole word if present (WORD_SENTINEL). Fall back to sorted (var,val) pairs.
fn solution_key(solution: &[Bindings]) -> u64 {
    let mut hasher = DefaultHasher::new();

    for b in solution {
        // Try whole-word first (fast + canonical)
        if let Some(w) = b.get_word() {
            w.hash(&mut hasher);
        } else {
            // this should never happen
            /*
            // Fall back: hash all (var,val) pairs sorted by var
            let mut pairs: Vec<(char, String)> =
                b.iter().map(|(k, v)| (*k, v.clone())).collect();
            pairs.sort_unstable_by_key(|(k, _)| *k);
            for (k, v) in pairs {
                k.hash(&mut hasher);
                v.hash(&mut hasher);
            }
            */
        }
        // Separator between patterns to avoid ambiguity like ["ab","c"] vs ["a","bc"]
        HASH_SPLIT.hash(&mut hasher);
    }

    hasher.finish()
}

/// Simple helper to enforce a wall-clock time limit.
///
/// Usage:
///   let budget = TimeBudget::new(Duration::from_secs(30));
///   while !budget.expired() {
///       // do some work
///   }
///
/// You can also query how much time is left (`remaining()`).
/// TODO: consider using a countdown timer with one "time left" parameter
struct TimeBudget {
    start: Instant,   // when the budget began
    limit: Duration,  // maximum allowed elapsed time
}

impl TimeBudget {
    /// Create a new budget that lasts for `limit` (e.g., 30 seconds).
    fn new(limit: Duration) -> Self {
        Self { start: Instant::now(), limit }
    }

    /// Returns true if the allowed time has fully elapsed.
    fn expired(&self) -> bool {
        self.start.elapsed() >= self.limit
    }

    // Returns the remaining time before expiration, or zero if the budget is already used up.
    // Unused for now but it may be useful later
    // fn remaining(&self) -> Duration {self.limit.saturating_sub(self.start.elapsed())}
}


/// Build the deterministic lookup key for a binding given the pattern's lookup vars.
/// Returns:
///   - None: pattern has no lookup constraints (unkeyed bucket)
///   - Some(vec): concrete key (sorted by var char)
///   - Some(empty vec): sentinel meaning "required key missing" → caller should skip
fn lookup_key_for_binding(
    binding: &Bindings,
    keys_opt: Option<&HashSet<char>>,
) -> LookupKey {
    let keys = match keys_opt {
        None => return None, // unkeyed
        Some(k) => k,
    };

    // Collect (var, value) for all required keys; bail out immediately if any is missing.
    let mut pairs: Vec<(char, String)> = Vec::with_capacity(keys.len());
    for &var in keys {
        match binding.get(var) {
            Some(val) => pairs.push((var, val.clone())),
            None => return Some(Vec::new()), // "impossible" sentinel; caller will skip
        }
    }

    // Normalize key order for stable hashing/equality
    pairs.sort_unstable_by_key(|(c, _)| *c);
    Some(pairs)
}


/// Push a binding into the appropriate bucket and bump the count.
fn push_binding(words: &mut [CandidateBuckets], i: usize, key: LookupKey, binding: Bindings) {
    words[i].buckets.entry(key).or_default().push(binding);
    words[i].count += 1;
}

/// Scan a slice of the word list and incrementally fill candidate buckets.
/// Returns whether we hit the per-pattern cap and the last index scanned (exclusive).
/// Scan a slice of the word list and incrementally fill candidate buckets.
/// Returns (new_scan_pos, time_up).
fn scan_batch(
    word_list: &[&str],
    start_idx: usize,
    batch_size: usize,
    patterns: &Patterns,
    parsed_forms: &[ParsedForm],
    scan_hints: &[PatternLenHints],
    var_constraints: &crate::constraints::VarConstraints,
    joint_constraints: Option<&JointConstraints>,
    words: &mut [CandidateBuckets],
    budget: Option<&TimeBudget>,
) -> (usize, bool) {
    let mut i_word = start_idx;
    let end = start_idx.saturating_add(batch_size).min(word_list.len());

    while i_word < end {
        if let Some(b) = budget {
            // TODO: have this timeout bubble all the way up
            if b.expired() { return (i_word, true); }
        }

        let word = word_list[i_word];

        for (i, patt) in patterns.iter().enumerate() {
            // No per-pattern cap anymore

            // Skip deterministic fully-keyed forms
            if patt.is_deterministic && patt.all_vars_in_lookup_keys() {
                continue;
            }
            // Cheap length prefilter
            if !scan_hints[i].is_word_len_possible(word.len()) {
                continue;
            }

            let matches = match_equation_all(
                word,
                &parsed_forms[i],
                Some(var_constraints),
                joint_constraints,
            );

            for binding in matches {
                let key = lookup_key_for_binding(&binding, patt.lookup_keys.as_ref());

                // If a required key is missing, skip
                if key.as_ref().is_some_and(|v| v.is_empty() && patt.lookup_keys.as_ref().is_some_and(|ks| !ks.is_empty())) {
                    continue;
                }

                push_binding(words, i, key, binding.clone());
            }
        }

        i_word += 1;
    }

    (i_word, false)
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
///   chosen patterns. `None` means "no lookup constraint" (use the `None` bucket).
///   `Some(vec)` means we must look up a concrete `Some(sorted_pairs)` key—even if
///   `vec` is empty.
/// - `selected`: the partial solution (one chosen Binding per pattern so far).
/// - `env`: the accumulated variable → value environment from earlier choices.
/// - `results`: completed solutions (each is a Vec<Binding>, one per pattern).
/// - `num_results_requested`: cap on how many full solutions to collect.
///
/// Return:
/// - This function mutates `results` and stops early once it has `num_results_requested`.
fn recursive_join(
    idx: usize,
    words: &Vec<CandidateBuckets>,
    lookup_keys: &Vec<Option<HashSet<char>>>,
    selected: &mut Vec<Bindings>,
    env: &mut HashMap<char, String>,
    results: &mut Vec<Vec<Bindings>>,
    num_results_requested: usize,
    patterns: &Patterns,                 // for patt.deterministic / vars / lookup_keys
    parsed_forms: &Vec<ParsedForm>,      // same order as `words` / `patterns.ordered_list`
    word_list_as_set: &HashSet<&str>,
    joint_constraints: Option<&JointConstraints>,
    seen: &mut HashSet<u64>,
) {
    // Stop if we’ve met the requested quota of full solutions.
    if results.len() >= num_results_requested {
        return;
    }

    // Base case: if we’ve placed all patterns, `selected` is a full solution.
    if idx == words.len() {
        if joint_constraints
            .as_ref()
            .is_none_or(|jcs| jcs.all_strictly_satisfied_for_parts(selected))
        {
            let key = solution_key(selected);
            if seen.insert(key) {
                results.push(selected.clone());
            }
        }
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
                idx + 1, words, lookup_keys, selected, env, results, num_results_requested,
                patterns, parsed_forms, word_list_as_set, joint_constraints, seen,
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
        if results.len() >= num_results_requested {
            break; // stop early if we’ve already met the quota
        }

        // Defensive compatibility check: if a variable is already in `env`,
        // its value must match the candidate. This *should* already be true
        // because we selected the bucket using the shared vars—but keep this
        // in case upstream bucketing logic ever changes.
        if cand.iter().filter(|(k, _)| **k != WORD_SENTINEL).any(|(k, v)| env.get(k).is_some_and(|prev| prev != v)) {
            continue;
        }

        // Extend `env` with any *new* bindings from this candidate (don’t overwrite).
        // Track what we added so we can backtrack cleanly.
        let mut added_vars: Vec<char> = vec![];
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
        recursive_join(idx + 1, words, lookup_keys, selected, env, results, num_results_requested,
                       patterns, parsed_forms, word_list_as_set, joint_constraints, seen,);
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
/// - `word_list`: list of candidate words to test.
///   Note that we require (but do not enforce!) that all words be lowercase.
///   TODO: should we enforce this?
/// - `num_results_requested`: maximum number of *final* results to return
///
/// Returns:
/// - A `Vec` of solutions, each solution being a `Vec<Binding>` where each `Binding`
///   maps variable names (chars) to concrete substrings they were bound to in that solution.
///
/// # Errors
///
/// Will return a `ParseError` if a form cannot be parsed.
// TODO? add more detail in Errors section
pub fn solve_equation(input: &str, word_list: &[&str], num_results_requested: usize) -> Result<Vec<Vec<Bindings>>, ParseError> {
    // 0. Make a hash set version of our word list
    let word_list_as_set: HashSet<&str> = word_list.iter().copied().collect();

    // 1. Parse the input equation string into our `Patterns` struct.
    //    This holds each pattern string, its parsed form, and its `lookup_keys` (shared vars).
    let patterns = Patterns::of(input);

    // 2. Build per-pattern lookup key specs (shared vars) for the join
    let lookup_keys: Vec<Option<HashSet<char>>> =
        patterns.iter().map(|p| p.lookup_keys.clone()).collect();

    // 3. Prepare storage for candidate buckets, one per pattern.
    //    `CandidateBuckets` tracks (a) the bindings bucketed by shared variable values, and
    //    (b) a count so we can stop early if a pattern gets too many matches.
    // Mutable because we fill buckets/counts during the scan phase.
    let mut words: Vec<CandidateBuckets> = Vec::with_capacity(patterns.len());
    for _ in &patterns {
        words.push(CandidateBuckets::default());
    }

    // 4. Parse each pattern's string form once into a vector of `FormPart`s.
    //    These are index-aligned with `patterns`.
    let mut parsed_forms: Vec<_> = patterns
        .iter()
        .map(|p| parse_form(&p.raw_string))
        .collect::<Result<_, _>>()?;

    // 5. Pull out the per-variable constraints collected from the equation.
    let mut var_constraints = patterns.var_constraints.clone();

    // 6. Upgrade prefilters once per form (only if it helps)
    for pf in &mut parsed_forms {
        if has_inlineable_var_form(&pf.parts, &var_constraints) {
            // Build the anchored, constraint-aware pattern string
            let anchored = format!(
                "^{}$",
                form_to_regex_str_with_constraints(&pf.parts, Some(&var_constraints))
            );

            // Compile via the shared cache; fall back to the existing prefilter on error
            if let Ok(re) = get_regex(&anchored) {
                pf.prefilter = re;
            }
        }
    }


    // 7. Get the joint constraints and use them to tighten per-variable constraints
    let joint_constraints = parse_joint_constraints(input);

    if let Some(jcs) = joint_constraints.as_ref() {
        propagate_joint_to_var_bounds(&mut var_constraints, jcs);
    }

    // 8. Build cheap, per-form length hints once (index-aligned with patterns/parsed_forms)
    let scan_hints: Vec<PatternLenHints> = parsed_forms
        .iter()
        .map(|pf| form_len_hints_pf(pf, &patterns.var_constraints, joint_constraints.as_ref()))
        .collect();

    // 9. Iterate through every candidate word.
    let budget = TimeBudget::new(Duration::from_secs(TIME_BUDGET));

    let mut results: Vec<Vec<Bindings>> = vec![];
    let mut selected: Vec<Bindings> = vec![];
    let mut env: HashMap<char, String> = HashMap::new();

    // scan_pos tracks how far we've scanned into the word list.
    let mut scan_pos: usize = 0;

    // Global set of fingerprints for already-emitted solutions.
    // Ensures we don't return duplicate solutions across scan/join rounds.
    let mut seen: HashSet<u64> = HashSet::new();

    // batch_size controls how many words to scan this round (adaptive).
    let mut batch_size: usize = BATCH_SIZE;

    // High-level solver driver. Alternates between:
    //   (1) scanning more words from the dictionary into candidate buckets
    //   (2) recursively joining those buckets into full solutions
    // Continues until either we have enough results, the word list is exhausted,
    // or the time budget expires.
    while results.len() < num_results_requested
        && scan_pos < word_list.len()
        && !budget.expired()
    {
        // 1) Scan the next batch_size words into candidate buckets.
        // Each candidate binding is grouped by its lookup key so later joins are fast.
        let (new_pos, _time_up) = scan_batch(
            word_list,
            scan_pos,
            batch_size,
            &patterns,
            &parsed_forms,
            &scan_hints,
            &var_constraints,
            joint_constraints.as_ref(),
            &mut words,
            Some(&budget),
        );
        scan_pos = new_pos;

        // Respect the TimeBudget
        if budget.expired() { break; }

        // 2) Attempt to build full solutions from the candidates accumulated so far.
        // This may rediscover old partials, so we use `seen` at the base case
        // to ensure only truly new solutions are added to `results`.
        recursive_join(
            0,
            &words,
            &lookup_keys,
            &mut selected,
            &mut env,
            &mut results,
            num_results_requested,
            &patterns,
            &parsed_forms,
            &word_list_as_set,
            joint_constraints.as_ref(),
            &mut seen,
        );

        if results.len() >= num_results_requested || budget.expired() {
            break;
        }

        // Optional early-exit when we’re out of input or not progressing
        // TODO: magic number
        if scan_pos >= word_list.len() {
            break;
        }

        // Grow the batch size for the next round
        // TODO: magic number, maybe adaptive resizing?
        batch_size = batch_size.saturating_mul(2);
    }

    // ---- Reorder solutions back to original form order ----
    let reordered = results.iter().map(|solution| {
        (0..solution.len()).map(|original_i| {
            solution.clone()[patterns.original_to_ordered[original_i]].clone()
        }).collect::<Vec<_>>()
    }).collect::<Vec<_>>();

    // Return up to `num_results_requested` reordered solutions
    Ok(reordered)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_solve_equation() {
        let word_list: Vec<&str> = vec!["lax", "tax", "lox"];
        let input = "l.x".to_string();
        let results = solve_equation(&input, &word_list, 5).unwrap();
        println!("{:?}", results);
        assert_eq!(2, results.len());
    }

    #[test]
    fn test_solve_equation2() {
        let word_list: Vec<&str> = vec!["inch", "chin", "dada", "test", "ab"];
        let input = "AB;BA;|A|=2;|B|=2;!=AB".to_string();
        let results = solve_equation(&input, &word_list, 5).unwrap();
        println!("{:?}", results);
        assert_eq!(2, results.len());
    }

    #[test]
    fn test_solve_equation3() {
        let word_list = vec!["inch", "chin", "dada", "test", "sky", "sly"];
        let input = "AkB;AlB".to_string();
        let results = solve_equation(&input, &word_list, 5).unwrap();

        let mut sky_bindings = Bindings::default();
        sky_bindings.set('A', "s".to_string());
        sky_bindings.set('B', "y".to_string());
        sky_bindings.set_word("sky".to_string().as_ref());

        let mut sly_bindings = Bindings::default();
        sly_bindings.set('A', "s".to_string());
        sly_bindings.set('B', "y".to_string());
        sly_bindings.set_word("sly".to_string().as_ref());
        // NB: this could give a false negative if SLY comes out before SKY (since we presumably shouldn't care about the order), so...
        // TODO allow order independence for equality... perhaps create a richer struct than just Vec<Bindings> that has a notion of order-independent equality
        let expected = vec![vec![sky_bindings, sly_bindings]];
        assert_eq!(expected, results);
    }

    #[test]
    fn test_solve_equation_joint_constraints() {
        let word_list = vec!["inch", "chin", "chess", "chortle"];
        let input = "ABC;CD;|ABCD|=7".to_string();
        let results = solve_equation(&input, &word_list, 5).unwrap();
        println!("{:?}", results);
        let mut inch_bindings = Bindings::default();
        inch_bindings.set('A', "i".to_string());
        inch_bindings.set('B', "n".to_string());
        inch_bindings.set('C', "ch".to_string());
        inch_bindings.set_word("inch".to_string().as_ref());

        let mut chess_bindings = Bindings::default();
        chess_bindings.set('C', "ch".to_string());
        chess_bindings.set('D', "ess".to_string());
        chess_bindings.set_word("chess".to_string().as_ref());
        let expected = vec![vec![inch_bindings, chess_bindings]];
        assert_eq!(expected, results);
    }
}
