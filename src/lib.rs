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
    use std::collections::HashMap;
    use wasm_bindgen::prelude::*;

    #[wasm_bindgen(start)]
    pub fn init_panic_hook() {
        console_error_panic_hook::set_once();
    }

    // Convert a Bindings into a JS-ish map (string keys, including "*")
    fn bindings_to_map(b: &Bindings) -> HashMap<String, String> {
        b.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
    }

    /// JS entry: (input: string, word_list: string[], num_results: number)
    /// returns Array<Array<Record<string,string>>>
    #[wasm_bindgen]
    pub fn solve_equation_wasm(
        input: &str,
        word_list: JsValue,
        num_results: usize,
    ) -> Result<JsValue, JsValue> {
        let words: Vec<String> = serde_wasm_bindgen::from_value(word_list)
            .map_err(|e| JsValue::from_str(&format!("word_list must be string[]: {e}")))?;

        let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

        let raw = solve_equation(input, &refs, num_results);

        let js_ready: Vec<Vec<HashMap<String, String>>> = raw
            .iter()
            .map(|row| row.iter().map(bindings_to_map).collect())
            .collect();

        serde_wasm_bindgen::to_value(&js_ready)
            .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
    }
}