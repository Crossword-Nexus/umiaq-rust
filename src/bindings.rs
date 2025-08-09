use std::collections::HashMap;

const WORD_SENTINEL: char = '*';

/// `Bindings` maps a variable name (char) to the string it’s bound to.
/// Special variable `'*'` is reserved for the bound word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Bindings {
    map: HashMap<char, String>,
}

impl Bindings {
    /// Create a new, empty set of bindings
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    /// Assign the word binding to '*'
    pub fn set_word(&mut self, word: &str) {
        self.map.insert(WORD_SENTINEL, word.to_string());
    }

    /// Retrieve the bound word, if any
    pub fn get_word(&self) -> Option<&String> {
        self.map.get(&WORD_SENTINEL)
    }

    /// Iterate over the bindings
    pub fn iter(&self) -> impl Iterator<Item = (&char, &String)> {
        self.map.iter()
    }
}
