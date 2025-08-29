use std::collections::HashSet;
use std::sync::LazyLock;

// Character-set constants
pub(crate) const VOWELS: &str = "aeiouy";
pub(crate) const CONSONANTS: &str = "bcdfghjklmnpqrstvwxz";
pub(crate) const VARIABLE_CHARS: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
pub(crate) const LITERAL_CHARS: &str = "abcdefghijklmnopqrstuvwxyz";

pub(crate) const NUM_POSSIBLE_VARIABLES: usize = VARIABLE_CHARS.len();

static VOWEL_SET: LazyLock<HashSet<char>> = LazyLock::new(|| VOWELS.chars().collect());
static CONSONANT_SET: LazyLock<HashSet<char>> = LazyLock::new(|| CONSONANTS.chars().collect());

pub(crate) trait UmiaqChar {
    fn is_vowel(&self) -> bool;
    fn is_consonant(&self) -> bool;
    fn is_variable(&self) -> bool;
    fn is_literal(&self) -> bool;
}

impl UmiaqChar for char {
    fn is_vowel(&self) -> bool {
        VOWEL_SET.contains(self)
    }
    fn is_consonant(&self) -> bool {
        CONSONANT_SET.contains(self)
    }
    fn is_variable(&self) -> bool {
        self.is_ascii_uppercase()
    }
    fn is_literal(&self) -> bool {
        self.is_ascii_lowercase()
    }
}
