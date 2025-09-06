use wasm_bindgen::prelude::*;
use crate::bindings::Bindings;
use crate::errors::ParseError;
use crate::solver::solve_equation;

/// Implement `Box<ParseError>` for `JsValue`s
impl From<Box<ParseError>> for JsValue {
    fn from(e: Box<ParseError>) -> JsValue { JsValue::from_str(format!("parse error: {}", *e).as_str()) }
}

#[wasm_bindgen(start)]
fn init_panic_hook() {
    console_error_panic_hook::set_once();
}

// Pull just the bound word ("*") out of a Bindings
fn binding_to_word(b: &Bindings) -> Option<String> {
    b.get_word().cloned()
}

/// JS entry: (input: string, word_list: string[], num_results_requested: number)
/// returns Array<Array<string>> — only the bound words
#[wasm_bindgen]
pub fn solve_equation_wasm(
    input: &str,
    word_list: JsValue,
    num_results_requested: usize,
) -> JsValue {
    // Try to parse the JS word list into Vec<String>.
    let words: Vec<String> = match serde_wasm_bindgen::from_value(word_list) {
        Ok(w) => w,
        Err(_) => wasm_bindgen::throw_str("word_list must be string[]"),
    };
    let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

    // Call solver. If it panics, wasm will trap, but the panic hook will log details.
    let raw = match solve_equation(input, &refs, num_results_requested) {
        Ok(r) => r,
        Err(_) => wasm_bindgen::throw_str("Input parsing failed — see browser console for details"),
    };

    // Convert Vec<Vec<Bindings>> → Vec<Vec<String>>.
    let js_ready: Vec<Vec<String>> = raw
        .into_iter()
        .map(|row| row.into_iter().filter_map(|b| binding_to_word(&b)).collect())
        .collect();

    // Serialize into JsValue for JS side.
    match serde_wasm_bindgen::to_value(&js_ready) {
        Ok(val) => val,
        Err(_) => wasm_bindgen::throw_str("Serialization failed — see browser console for details"),
    }
}





