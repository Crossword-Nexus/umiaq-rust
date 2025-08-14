// Reusable library API — visible to both CLI and WASM builds
pub mod wordlist;
pub mod solver;
pub mod bindings;
pub mod parser;
pub mod patterns;
pub mod constraints;

// Compile the wasm glue only when targeting wasm32.
#[cfg(target_arch = "wasm32")]
pub mod wasm; // this points to src/wasm.rs
