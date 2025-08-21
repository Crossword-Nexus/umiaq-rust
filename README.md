# Umiaq

**Umiaq** is an open-source solver and word-pattern matching tool.  
It’s designed for crossword constructors, wordplay enthusiasts, and puzzle makers who want to search large word lists using expressive patterns, variables, and constraints.

👉 [Try the web interface](web/index.html)

---

## Features

- **Expressive pattern matching**  
  Match words against patterns with variables, wildcards, and constraints.

- **Variable binding**  
  Use uppercase letters (`A`–`Z`) as variables that can bind to substrings and be reused.

- **Wildcards**
  - `.` — any single character
  - `*` — zero or more characters
  - `@` — any vowel (`AEIOUY`)
  - `#` — any consonant (`BCDFGHJKLMNPQRSTVWXZ`)
  - `[abc]` — any of the listed characters
  - `/abc` — any anagram of the listed letters

- **Constraints**  
  Add conditions on variables or groups of variables:
  - Length: `|A|=3`
  - Inequality: `!=AB` (A must not equal B)
  - Complex: `A=(3-5:a*)` (length 3–5, must match pattern `a*`)
  - Joint: `|ABC|=10` (the lengths of A, B, C sum to 10)

- **Reversed variables**  
  `~A` matches the reverse of variable `A`.

---

## Examples

- `l.x` → matches words like **LAX**, **LOX**
- `A~A` → palindromes like **NOON**, **REDDER**
- `AB;|A|=2;|B|=2;!=AB` → words with two distinct 2-letter substrings
- `A@#A` → words with some string, then a vowel, then a consonant, then the original string again
- `/triangle` → any anagram of “triangle”
- `A;AB;|AB|=7;A=(3-4:g*)` → 7-letter words starting with a 3–4 letter string beginning with **g**

---

## License

- **Code**: MIT License
- **Word list (Spread the Word List)**: CC BY-NC-SA 4.0  

