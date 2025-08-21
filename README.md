# Umiaq

## Usage
[A web interface is available](web/index.html)

## License
MIT License.

Spread the Word List is released under CC BY-NC-SA 4.0

## Features

- **pattern matching**: match words against complex patterns using variables, wildcards, and constraints
- **variable binding**: use uppercase letters \(A-Z\) as variables that can bind to substrings
- **wildcards**: support for various wildcard types:
    - `.`: exactly one character
    - `*`: zero or more characters
    - `@`: any vowel (AEIOUY)
    - `#`: any consonant (BCDFGHJKLMNPQRSTVWXZ)
    - `[abc]`: any of the specified characters
    - `/abc`: anagram of the specified characters
- **constraints**: Apply constraints to variables:
    - length constraints: `|A|=3`
    - inequality constraints: `!=AB` (A must not equal B)
    - complex constraints: `A=(3-5:a*)` (length 3-5, must match pattern "a*")
    - joint constrints: `|ABC|=10` (the lengths of A, B, C sum to 10)
- **reversed variables**: `~A`

### Pattern Examples

- `"l.x"`: matches words like "LAX", "LOX"
- `"A~A"`: matches palindromes like "NOON", "REDDER"
- `"AB;|A|=2;|B|=2;!=AB"`: matches words where A and B are distinct 2-letter substrings
- `"A@#A"`: matches words with some nonempty string, a vowel, a consonant, and the first nonempty string again
- `"/triangle"`: matches anagrams of "triangle"
- `"A;AB;|AB|=7;A=(3-4:g*)"` matches words of length 7 that start with a word of length 3-4 whose first letter is "g"
