use wasm_bindgen::prelude::*;
use crate::bindings::Bindings;
use crate::errors::ParseError;
use crate::solver::solve_equation;

use serde::Serialize;
use web_sys::console;

#[derive(Serialize)]
struct WasmSolveReport {
    results: Vec<Vec<String>>,
    timed_out: bool,
}

/// Convert parse errors into JS values for `?` propagation.
impl From<Box<ParseError>> for JsValue {
    fn from(e: Box<ParseError>) -> JsValue {
        JsValue::from_str(format!("parse error: {}", *e).as_str())
    }
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
/// returns { results: string[][], timed_out: boolean }
#[wasm_bindgen]
pub fn solve_equation_wasm(
    input: &str,
    word_list: JsValue,
    num_results_requested: usize,
) -> Result<JsValue, JsValue> {
    console::log_1(&JsValue::from_str("solve_equation_wasm: start"));

    // Convert the JS array to Vec<String>.
    let words: Vec<String> = serde_wasm_bindgen::from_value(word_list)
        .map_err(|e| JsValue::from_str(&format!("word_list must be string[]: {e}")))?;
    let refs: Vec<&str> = words.iter().map(|s| s.as_str()).collect();

    console::log_1(&JsValue::from_str("solve_equation_wasm: calling solver"));
    let report = solve_equation(input, &refs, num_results_requested)?; // SolveReport

    console::log_1(&JsValue::from_str("solve_equation_wasm: mapping bindings → words"));
    let results: Vec<Vec<String>> = report
        .results
        .into_iter()
        .map(|row| row.into_iter().filter_map(|b| binding_to_word(&b)).collect())
        .collect();

    let out = WasmSolveReport {
        results,
        timed_out: report.timed_out,
    };

    console::log_1(&JsValue::from_str("solve_equation_wasm: serializing to JsValue"));
    serde_wasm_bindgen::to_value(&out)
        .map_err(|e| JsValue::from_str(&format!("serialization failed: {e}")))
}
