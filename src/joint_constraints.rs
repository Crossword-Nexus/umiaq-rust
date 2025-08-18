use crate::bindings::Bindings;
use crate::patterns::FORM_SEPARATOR;
use std::cmp::Ordering;

/// Compact representation of the relation between (sum) and (target).
///
/// We encode three mutually-exclusive outcomes as bits:
/// - LT (sum < target)  -> 0b001
/// - EQ (sum == target) -> 0b010
/// - GT (sum > target)  -> 0b100
///
/// Compound operators (<=, >=, !=) are unions of these bits.
/// Evaluation is then: `rel.allows(total.cmp(&target))`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelMask {
    mask: u8,
}

impl RelMask {
    pub const LT: Self = Self { mask: 0b001 };
    pub const EQ: Self = Self { mask: 0b010 };
    pub const GT: Self = Self { mask: 0b100 };

    pub const LE: Self = Self { mask: Self::LT.mask | Self::EQ.mask }; // <=
    pub const GE: Self = Self { mask: Self::GT.mask | Self::EQ.mask }; // >=
    pub const NE: Self = Self { mask: Self::LT.mask | Self::GT.mask }; // !=

    /// Return true if this mask allows the given ordering outcome.
    #[inline]
    pub fn allows(self, ord: Ordering) -> bool {
        let bit = match ord {
            Ordering::Less    => 0b001,
            Ordering::Equal   => 0b010,
            Ordering::Greater => 0b100,
        };
        (self.mask & bit) != 0
    }

    /// Parse an operator token into a mask.
    /// Accepted: "==", "=", "!=", "<=", ">=", "<", ">".
    pub fn from_str(op: &str) -> Option<Self> {
        Some(match op {
            "==" | "=" => Self::EQ,
            "!="       => Self::NE,
            "<="       => Self::LE,
            ">="       => Self::GE,
            "<"        => Self::LT,
            ">"        => Self::GT,
            _          => return None,
        })
    }
}

/// Joint length constraint like `|ABC| <= 7`.
///
/// - `vars`  : the participating variable names (A–Z). Duplicates **do** count toward the sum.
/// - `target`: RHS integer to compare against.
/// - `rel`   : operator, stored as a relation mask (see `RelMask`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JointConstraint {
    pub vars: Vec<char>,   // e.g., ['A','B','C']
    pub target: usize,     // e.g., 7
    pub rel: RelMask,      // operator as data
}

impl JointConstraint {
    /// Check satisfaction against current `bindings`.
    ///
    /// **Mid-search semantics** (by design): if **any** referenced var is unbound,
    /// we return `true` (no opinion yet). This keeps partial assignments alive.
    ///
    /// If you need a *final* strict check, run this only after all vars are bound,
    /// or add a separate strict method that returns `false` when some vars are unbound.
    #[inline]
    pub fn is_satisfied_by(&self, bindings: &Bindings) -> bool {
        // If not all vars are bound, skip this check for now.
        if bindings.contains_all_vars(&self.vars) {
            // Sum the lengths of the bound strings for the referenced vars.
            let total: usize = self.vars.iter().map(|v| bindings.get(*v).unwrap().len()).sum();

            // Compare once via Ordering -> mask test.
            self.rel.allows(total.cmp(&self.target))
        } else {
            true
        }
    }

    // --- Test-only convenience for asserting behavior without needing real `Bindings`.
    //     This keeps tests independent of crate::bindings internals.
    #[cfg(test)]
    fn is_satisfied_by_map(&self, map: &std::collections::HashMap<char, String>) -> bool {
        if !self.vars.iter().all(|v| map.contains_key(v)) {
            return true;
        }
        let total: usize = self.vars.iter().map(|v| map.get(v).unwrap().len()).sum();
        self.rel.allows(total.cmp(&self.target))
    }
}

/// Parse a single joint-length expression that **starts at** a `'|'`.
///
/// Shape: `|VARS| OP NUMBER`
///  - `VARS`  : at least **two** ASCII uppercase letters (A–Z).
///  - `OP`    : one of `<=`, `>=`, `==`, `!=`, `<`, `>`, `=` (two-char ops matched first).
///  - `NUMBER`: one or more ASCII digits (base 10).
///
/// Notes:
///  - Any trailing content after the number is currently **ignored**. If you need
///    strictness here, add a trailing-whitespace check and reject junk.
fn parse_joint_len(expr: &str) -> Option<JointConstraint> {
    let s = expr.trim();
    if !s.starts_with('|') { return None; }

    // Locate the closing bar.
    let end_bar_rel = s[1..].find('|')?;
    let end_bar_idx = 1 + end_bar_rel;
    let vars_str = &s[1..end_bar_idx];

    // Enforce A–Z only and at least two variables (true "joint" constraint).
    if !vars_str.chars().all(|c| c.is_ascii_uppercase()) { return None; }
    if vars_str.chars().count() < 2 { return None; }

    // Remainder like "=7", "<= 10", etc.
    let rhs = s[end_bar_idx + 1..].trim_start();

    // Recognize operators (two-char first to avoid "<" grabbing from "<=").
    let (op_tok, rest) = ["<=", ">=", "==", "!=", "<", ">", "="]
        .iter()
        .find_map(|&tok| rhs.strip_prefix(tok).map(|r| (tok, r.trim_start())))?;

    // Parse integer (digits only).
    let digits_len = rest.chars().take_while(char::is_ascii_digit).count();
    if digits_len == 0 { return None; }
    let target: usize = rest[..digits_len].parse().ok()?;

    let rel = RelMask::from_str(op_tok)?;
    let vars = vars_str.chars().collect::<Vec<char>>(); // duplicates are kept

    Some(JointConstraint { vars, target, rel })
}

/// Container for many joint constraints (useful as a field on your puzzle/parse).
#[derive(Debug, Default, Clone)]
pub struct JointConstraints {
    pub as_vec: Vec<JointConstraint>, // TODO? avoid using this directly
}

impl JointConstraints {
    /// Return true iff **every** joint constraint is satisfied w.r.t. `bindings`.
    ///
    /// Mid-search semantics: a constraint with unbound vars returns `true`
    /// (see `JointConstraint::is_satisfied_by`), so this is safe to call
    /// during search as a "non-pruning check".
    pub fn all_satisfied(&self, bindings: &Bindings) -> bool {
        self.as_vec.iter().all(|jc| jc.is_satisfied_by(bindings))
    }

    // Test-only helper mirroring `all_satisfied` over a plain map.
    #[cfg(test)]
    fn all_satisfied_map(
        &self,
        map: &std::collections::HashMap<char, String>
    ) -> bool {
        self.as_vec.iter().all(|jc| jc.is_satisfied_by_map(map))
    }
}

/// Parse all joint constraints from an equation string by splitting on your
/// `FORM_SEPARATOR` (e.g., ';'), feeding each part through `parse_joint_len`.
///
/// Returns `None` if no joint constraints are found.
pub fn parse_joint_constraints(equation: &str) -> Option<JointConstraints> {
    let mut v = Vec::new();
    for part in equation.split(FORM_SEPARATOR) {
        if let Some(jc) = parse_joint_len(part.trim()) {
            v.push(jc);
        }
    }
    if v.is_empty() { None } else { Some(JointConstraints { as_vec: v }) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use crate::patterns::FORM_SEPARATOR;

    #[test]
    fn relmask_from_str_and_allows() {
        assert_eq!(RelMask::from_str("="),  Some(RelMask::EQ));
        assert_eq!(RelMask::from_str("=="), Some(RelMask::EQ));
        assert_eq!(RelMask::from_str("<="), Some(RelMask::LE));
        assert_eq!(RelMask::from_str(">="), Some(RelMask::GE));
        assert_eq!(RelMask::from_str("!="), Some(RelMask::NE));
        assert_eq!(RelMask::from_str("<"),  Some(RelMask::LT));
        assert_eq!(RelMask::from_str(">"),  Some(RelMask::GT));
        assert!(RelMask::LE.allows(Ordering::Less));
        assert!(RelMask::LE.allows(Ordering::Equal));
        assert!(!RelMask::LE.allows(Ordering::Greater));
        assert!(RelMask::NE.allows(Ordering::Less));
        assert!(RelMask::NE.allows(Ordering::Greater));
        assert!(!RelMask::NE.allows(Ordering::Equal));
    }

    #[test]
    fn parse_joint_len_basic_variants() {
        // Basic equality
        let jc = parse_joint_len("|AB|=7").expect("should parse");
        assert_eq!(jc.vars, vec!['A','B']);
        assert_eq!(jc.target, 7);
        assert_eq!(jc.rel, RelMask::EQ);

        // Whitespace tolerated; two-char op
        let jc2 = parse_joint_len("|ABC|  <=   10").expect("should parse");
        assert_eq!(jc2.vars, vec!['A','B','C']);
        assert_eq!(jc2.target, 10);
        assert_eq!(jc2.rel, RelMask::LE);

        // Reject single-var
        assert!(parse_joint_len("|A|=3").is_none());

        // Reject lowercase
        assert!(parse_joint_len("|Ab|=3").is_none());

        // Must start at '|' (strict)
        assert!(parse_joint_len("foo |AB|=3").is_none());
    }

    #[test]
    fn parse_joint_constraints_from_equation() {
        let sep = FORM_SEPARATOR; // could be ';' or some other separator

        // Build an equation with two constraints and a non-constraint chunk.
        let equation = format!("|AB|=3{sep}foo{sep}|BC|<=5");

        let parsed = parse_joint_constraints(&equation).expect("should find constraints");
        assert_eq!(parsed.as_vec.len(), 2);

        assert_eq!(parsed.as_vec[0].vars, vec!['A', 'B']);
        assert_eq!(parsed.as_vec[0].target, 3);
        assert_eq!(parsed.as_vec[0].rel, RelMask::EQ);

        assert_eq!(parsed.as_vec[1].vars, vec!['B', 'C']);
        assert_eq!(parsed.as_vec[1].target, 5);
        assert_eq!(parsed.as_vec[1].rel, RelMask::LE);
    }

    #[test]
    fn is_satisfied_mid_search_semantics() {
        // |AB| = 5
        let jc = JointConstraint { vars: vec!['A','B'], target: 5, rel: RelMask::EQ };

        let mut map = HashMap::new();
        map.insert('A', "HI".to_string()); // len 2
        // 'B' unbound -> should return true (skip mid-search)
        assert!(jc.is_satisfied_by_map(&map));

        // Bind B (len 3) => total 5 -> satisfied
        map.insert('B', "YOU".to_string());
        assert!(jc.is_satisfied_by_map(&map));

        // Change B to length 4 => total 6 -> violates
        map.insert('B', "YOUR".to_string());
        assert!(!jc.is_satisfied_by_map(&map));
    }

    #[test]
    fn jointconstraints_all_satisfied_map_variant() {
        let jcs = JointConstraints {
            as_vec: vec![
                JointConstraint { vars: vec!['A', 'B'], target: 6, rel: RelMask::LE }, // len(A)+len(B) <= 6
                JointConstraint { vars: vec!['B', 'C'], target: 3, rel: RelMask::GE }, // len(B)+len(C) >= 3
            ]
        };

        let mut map = HashMap::new();
        map.insert('A', "NO".to_string());     // 2
        map.insert('B', "YES".to_string());    // 3
        map.insert('C', "X".to_string());      // 1

        // (2+3) <= 6  AND  (3+1) >= 3  => true
        assert!(jcs.all_satisfied_map(&map));

        // Make B longer → first constraint fails
        map.insert('B', "LONGER".to_string()); // 4
        // (2+6) <= 6  is false  → overall false
        assert!(!jcs.all_satisfied_map(&map));
    }
}
