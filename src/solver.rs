// solver.rs
use crate::patterns::{Patterns, Pattern};
use crate::parser::{parse_pattern, match_pattern};
use std::collections::{HashMap, HashSet, BTreeMap};
use std::fs::File;
use std::io::{BufReader, BufRead};

const NUM_RESULTS: usize = 100;
const MIN_SCORE: u32 = 50;
const MAX_WORD_LENGTH: usize = 21;
const WORD_LIST_FILE: &str = "xwordlist_sorted_trimmed.txt";

/// Solve a multi-pattern matching equation
/// Returns a list of vectors of match dictionaries
pub fn solve_equation(input: &str, num_results: usize) -> Vec<Vec<HashMap<String, String>>> {
    // Parse the pattern string into a Patterns object
    let patterns = Patterns::new(input);

    // Map each Pattern to its parsed internal representation
    let mut parsed_patterns = BTreeMap::new();
    for patt in patterns.iter() {
        parsed_patterns.insert(patt.clone(), parse_pattern(&patt.string).unwrap());
    }

    // Vector of dictionaries where:
    // words[i][key] = Vec of bindings matching the i-th pattern
    let mut words: Vec<BTreeMap<Option<BTreeMap<String, String>>, Vec<HashMap<String, String>>>> =
        vec![BTreeMap::new(); patterns.len()];

    let mut word_counts = vec![0; patterns.len()];
    let mut entry_to_score: HashMap<String, u32> = HashMap::new();

    // Read the word list file
    let file = File::open(WORD_LIST_FILE).expect("Word list file not found");
    let reader = BufReader::new(file);

    // Process each line in the word list
    'line_loop: for line in reader.lines().flatten() {
        let mut parts = line.trim().split(';');
        let word = parts.next().unwrap_or("").to_uppercase();
        let score = parts.next().unwrap_or("0").parse::<u32>().unwrap_or(0);

        // Skip words that don't meet basic criteria
        if score < MIN_SCORE || word.len() > MAX_WORD_LENGTH {
            continue;
        }

        entry_to_score.insert(word.clone(), score);

        // Check the word against each pattern
        for (i, patt) in patterns.iter().enumerate() {
            let parsed = parsed_patterns.get(patt).unwrap();
            let matches = match_pattern(&word, parsed, true, Some(&patterns.var_constraints));
            for binding in matches {
                // Determine the hash key based on shared variable bindings
                let key = patt.lookup_keys.as_ref().map(|keys| {
                    keys.iter().map(|k| (k.clone(), binding.get(k).unwrap().clone())).collect()
                });

                words[i].entry(key).or_default().push(binding);
                word_counts[i] += 1;
            }

            if word_counts[i] >= 50_000 {
                break 'line_loop;
            }
        }
    }

    let mut results = vec![];

    // Recursive function to build valid match sequences
    fn recurse(
        words: &[BTreeMap<Option<BTreeMap<String, String>>, Vec<HashMap<String, String>>>],
        patterns: &Patterns,
        idx: usize,
        selected: Vec<HashMap<String, String>>,
        current_dict: BTreeMap<String, String>,
        num_results: usize,
        results: &mut Vec<Vec<HashMap<String, String>>>,
    ) {
        if results.len() >= num_results {
            return;
        }

        if idx == words.len() {
            results.push(selected);
            return;
        }

        // Choose candidates from next pattern
        let candidates = match patterns.ordered_list[idx].lookup_keys {
            None => words[idx].get(&None).cloned().unwrap_or_default(),
            Some(ref keys) => {
                let lookup_key: BTreeMap<String, String> = keys
                    .iter()
                    .filter_map(|k| current_dict.get(k).map(|v| (k.clone(), v.clone())))
                    .collect();
                words[idx].get(&Some(lookup_key)).cloned().unwrap_or_default()
            }
        };

        // Recurse over each candidate
        for cand in candidates {
            let mut new_dict = current_dict.clone();
            for (k, v) in &cand {
                if k != "word" {
                    new_dict.insert(k.clone(), v.clone());
                }
            }
            let mut new_sel = selected.clone();
            new_sel.push(cand);
            recurse(words, patterns, idx + 1, new_sel, new_dict, num_results, results);
            if results.len() >= num_results {
                break;
            }
        }
    }

    // Start with first pattern, which has no lookup keys
    let first_list = words[0].get(&None).cloned().unwrap_or_default();
    for cand in first_list {
        let mut dict = BTreeMap::new();
        for (k, v) in &cand {
            if k != "word" {
                dict.insert(k.clone(), v.clone());
            }
        }
        recurse(&words, &patterns, 1, vec![cand], dict, num_results, &mut results);
        if results.len() >= num_results {
            break;
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_equation() {
        let input = "AB;BA;|A|=2;|B|=2;!=AB";
        let result = solve_equation(input, 10);
        assert!(result.len() > 0);
        for tuple in result {
            assert_eq!(tuple.len(), 2);
            let word1 = &tuple[0]["word"];
            let word2 = &tuple[1]["word"];
            assert_ne!(word1, word2);
        }
    }

    #[test]
    fn test_symmetric_vars() {
        let input = "A~A";
        let result = solve_equation(input, 5);
        assert!(result.iter().all(|group| group[0]["A"].chars().eq(group[0]["A"].chars().rev())));
    }
}
