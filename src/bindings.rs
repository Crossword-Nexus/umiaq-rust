use std::collections::HashMap;

pub(crate) const WORD_SENTINEL: char = '*';

/// `Bindings` maps a variable name (char) to the string it's bound to.
/// Special variable `'*'` is reserved for the bound word.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Bindings {
    map: HashMap<char, String>,
}

impl Bindings {
    /// Bind a variable to a value
    pub fn set(&mut self, var: char, val: String) {
        self.map.insert(var, val);
    }

    /// Retrieve the binding for a variable
    pub fn get(&self, var: char) -> Option<&String> {
        self.map.get(&var)
    }

    /// Remove a binding for the given variable (if it exists)
    pub fn remove(&mut self, var: char) {
        self.map.remove(&var);
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

    // Get the map (currently unused)
    pub fn get_map(&self) -> &HashMap<char, String> {
        &self.map
    }
}
