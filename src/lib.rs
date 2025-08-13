// src/lib.rs
mod wordlist;
mod solver;
mod bindings;
mod parser;
mod patterns;
mod constraints;

// ── WASM glue only when compiling for wasm32 ───────────────────────────────────
#[cfg(target_arch = "wasm32")]
mod wasm_api {
    use super::bindings::Bindings;
    use super::solver::solve_equation;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn init_panic_hook() {
        console_error_panic_hook::set_once();
    }

    // Pull just the bound word ("*") out of a Bindings
    fn binding_to_word(b: &Bindings) -> Option<String> {
        b.get_word().cloned()
    }

    /// JS entry: (input: string, word_list: string[], num_results: number)
    /// returns Array<Array<string>> — only the bound words
    #[wasm_bindgen]
    pub fn solve_equation_wasm(
        input: &str,
        word_list: JsValue,
        num_results: usize,
    ) -> Result<JsValue, JsValue> {
        // word_list: string[] -> Vec<String>
        let words: Vec<String> = serde_wasm_bindgen::from_value(word_list)
            .map_err(|e| JsValue::from_str(&format!("word_list must be string[]: {e}")))?;

        // Borrow as &[&str] for the solver
        let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

        // Call the existing solver
        let raw = solve_equation(input, &refs, num_results); // Vec<Vec<Bindings>>

        // Keep only the "*" word from each Bindings
        // If any Binding lacks a word, it's skipped (filter_map).
        let js_ready: Vec<Vec<String>> = raw
            .into_iter()
            .map(|row| row.into_iter().filter_map(|b| binding_to_word(&b)).collect())
            .collect();

        serde_wasm_bindgen::to_value(&js_ready)
            .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
    }
}
