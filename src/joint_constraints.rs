use crate::bindings::Bindings;
use crate::patterns::FORM_SEPARATOR;
use std::cmp::Ordering;
use crate::constraints::{VarConstraint, VarConstraints};
use crate::umiaq_char::UmiaqChar;

/// Compact representation of the relation between (sum) and (target).
///
/// We encode three mutually exclusive outcomes as bits:
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
    pub(crate) fn allows(self, ord: Ordering) -> bool {
        let bit = match ord {
            Ordering::Less    => 0b001,
            Ordering::Equal   => 0b010,
            Ordering::Greater => 0b100,
        };
        (self.mask & bit) != 0
    }

    /// Parse an operator token into a mask.
    /// Accepted: "==", "=", "!=", "<=", ">=", "<", ">".
    pub(crate) fn from_str(op: &str) -> Option<Self> {
        match op {
            // TODO: Jeremy's OCD
            "==" | "=" => Some(Self::EQ),
            "!=" => Some(Self::NE),
            "<=" => Some(Self::LE),
            ">=" => Some(Self::GE),
            "<" => Some(Self::LT),
            ">" => Some(Self::GT),
            _ => None,
        }
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
    pub(crate) fn is_satisfied_by(&self, bindings: &Bindings) -> bool {
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

    /// Check this constraint against a *solution row* represented as a slice of Bindings
    /// (one Bindings per form/pattern). Duplicates in `vars` count toward the sum.
    /// Returns `false` if *any* referenced var is unbound across `parts`.
    pub fn is_strictly_satisfied_by_parts(&self, parts: &[Bindings]) -> bool {
        let mut total: usize = 0;
        for var_len in self.vars.iter().map(|var| resolve_var_len(parts, *var)) {
            let Some(len) = var_len else { return false };
            total += len;
        }
        self.rel.allows(total.cmp(&self.target))
    }

    // --- Test-only convenience for asserting behavior without needing real `Bindings`.
    //     This keeps tests independent of crate::bindings internals.
    #[cfg(test)]
    fn is_satisfied_by_map(&self, map: &std::collections::HashMap<char, String>) -> bool {
        if !self.vars.iter().all(|v| map.contains_key(v)) {
            true
        } else {
            let total: usize = self.vars.iter().map(|v| map.get(v).unwrap().len()).sum();
            self.rel.allows(total.cmp(&self.target))
        }
    }
}

/// Helper: find the current length of variable `v` across a slice of Bindings.
/// Returns None if `v` is unbound in all Bindings in `parts`.
#[inline]
fn resolve_var_len(parts: &[Bindings], v: char) -> Option<usize> {
    parts.iter().find_map(|bindings| bindings.get(v).map(String::len))
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
    if !vars_str.chars().all(|c| c.is_variable()) || vars_str.chars().count() < 2 {
        None
    } else {
        // Remainder like "=7", "<= 10", etc.
        let rhs = s[end_bar_idx + 1..].trim_start();

        // Recognize operators (two-char first to avoid "<" grabbing from "<=").
        let (op_tok, rest) = ["<=", ">=", "==", "!=", "<", ">", "="]
            .iter()
            .find_map(|&tok| rhs.strip_prefix(tok).map(|r| (tok, r.trim_start())))?;

        // Parse integer (digits only).
        let digits_len = rest.chars().take_while(char::is_ascii_digit).count();
        if digits_len == 0 {
            None
        } else {
            let target: usize = rest[..digits_len].parse().ok()?;

            let rel = RelMask::from_str(op_tok)?;
            let vars = vars_str.chars().collect::<Vec<char>>(); // duplicates are kept

            Some(JointConstraint { vars, target, rel })
        }
    }
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
    pub(crate) fn all_satisfied(&self, bindings: &Bindings) -> bool {
        self.as_vec.iter().all(|jc| jc.is_satisfied_by(bindings))
    }

    /// True iff **every** joint constraint is satisfied w.r.t. a slice of `Bindings`.
    /// Requires all referenced variables to be bound.
    pub fn all_strictly_satisfied_for_parts(&self, parts: &[Bindings]) -> bool {
        self.as_vec.iter().all(|jc| jc.is_strictly_satisfied_by_parts(parts))
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
/// `FORM_SEPARATOR` (i.e., ';'), feeding each part through `parse_joint_len`.
///
/// Returns `None` if no joint constraints are found.
pub(crate) fn parse_joint_constraints(equation: &str) -> Option<JointConstraints> {
    let mut v = vec![];
    for part in equation.split(FORM_SEPARATOR) {
        if let Some(jc) = parse_joint_len(part.trim()) {
            v.push(jc);
        }
    }
    if v.is_empty() { None } else { Some(JointConstraints { as_vec: v }) }
}

/// Attempt to tighten per-variable length bounds using information from joint constraints.
///
/// This is a simple propagation step that converts some equalities over groups of variables
/// into stronger individual bounds.  Example:
///   • Joint constraint: `|ABCDEFGHIJKLMN| = 14`
///   • Default per-var bounds: each ≥ 1
///   • Since sum(mins) = 14, every variable must be exactly length 1.
/// This allows the solver to avoid exploring longer assignments unnecessarily.
///
/// Algorithm outline:
///   1. For each joint constraint with relation `= T`:
///      - Collect current min/max bounds for the vars in the group.
///      - If sum(mins) == T, then all vars are fixed at their minimum length.
///      - Else if sum(maxes) == T (and all maxes are finite), then all vars are fixed at their maximum.
///      - Else, perform generic interval tightening:
///        • New min for Vi = max(current min, T - Σ other maxes)
///        • New max for Vi = min(current max, T - Σ other mins)
///
/// This propagation is *sound* (never removes feasible solutions) and often
/// eliminates huge amounts of search, especially for long chains of unconstrained vars.
/// TODO: does this optimally account for, e.g., |AB|=3; |BC|=6?
pub fn propagate_joint_to_var_bounds(vcs: &mut VarConstraints, jcs: &JointConstraints) {
    for jc in &jcs.as_vec {
        if jc.rel != RelMask::EQ { continue; }

        // Cache per-var (min,max) and aggregate sums
        let mut sum_min = 0usize;
        let mut sum_max_opt: Option<usize> = Some(0);

        let mut mins: Vec<(char, usize)> = Vec::with_capacity(jc.vars.len());
        let mut maxs: Vec<(char, usize)> = Vec::with_capacity(jc.vars.len());

        for &v in &jc.vars {
            let (li, ui) = vcs.bounds(v);
            sum_min += li;

            // Track finite sum of maxes; if any is ∞, the group max is unbounded.
            if ui == VarConstraint::DEFAULT_MAX {
                sum_max_opt = None; // TODO? feels dirty
            } else {
                sum_max_opt = sum_max_opt.map(|a| a + ui);
            }

            mins.push((v, li));
            maxs.push((v, ui));
        }

        // Case 1: exact by mins
        if sum_min == jc.target {
            for (v, li) in mins {
                vcs.ensure_entry_mut(v).set_exact_len(li);
            }
            continue;
        }

        // Case 2: exact by finite maxes
        if let Some(sum_max) = sum_max_opt && sum_max == jc.target {
            for (v, ui) in maxs {
                vcs.ensure_entry_mut(v).set_exact_len(ui);
            }
            continue;
        }

        // Case 3: generic tightening
        for &v in &jc.vars {
            let (li, ui) = vcs.bounds(v);

            // Σ other mins
            let sum_other_min: usize = jc.vars
                .iter()
                .filter(|&&w| w != v)
                .map(|&w| vcs.bounds(w).0)
                .sum();

            // Σ other finite maxes (None if any is ∞)
            let mut sum_other_max_opt: Option<usize> = Some(0);
            for &w in jc.vars.iter().filter(|&&w| w != v) {
                let (_, w_ui) = vcs.bounds(w);
                if w_ui == VarConstraint::DEFAULT_MAX {
                    sum_other_max_opt = None;
                    break;
                }
                sum_other_max_opt = sum_other_max_opt.map(|a| a + w_ui);
            }

            let lower_from_joint = match sum_other_max_opt {
                Some(s) => jc.target.saturating_sub(s),
                None => 0, // others can stretch arbitrarily
            };
            let upper_from_joint = jc.target.saturating_sub(sum_other_min);

            // Tighten and store
            let new_min = li.max(lower_from_joint);
            let new_max = ui.min(upper_from_joint);

            let e = vcs.ensure_entry_mut(v);
            e.min_length = Some(new_min);
            e.max_length = Some(new_max);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patterns::FORM_SEPARATOR;

    #[test]
    fn rel_mask_from_str_and_allows() {
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

        let mut map = std::collections::HashMap::from([('A', "HI".to_string())]); // len 2
        // 'B' unbound -> should return true (skip mid-search)
        assert!(jc.is_satisfied_by_map(&map));

        // Bind B (len 3) => total 5 -> satisfied
        map.insert('B', "YOU".to_string());
        assert!(jc.is_satisfied_by_map(&map));

        // Change B to length 4 => total 6 -> violated
        map.insert('B', "YOUR".to_string());
        assert!(!jc.is_satisfied_by_map(&map));
    }

    #[test]
    fn joint_constraints_all_satisfied_map_variant() {
        let jcs = JointConstraints {
            as_vec: vec![
                JointConstraint { vars: vec!['A', 'B'], target: 6, rel: RelMask::LE }, // len(A)+len(B) <= 6
                JointConstraint { vars: vec!['B', 'C'], target: 3, rel: RelMask::GE }, // len(B)+len(C) >= 3
            ]
        };

        let mut map = std::collections::HashMap::from([
            ('A', "NO".to_string()), // 2
            ('B', "YES".to_string()), // 3
            ('C', "X".to_string())] // 1
        );

        // (2+3) <= 6  AND  (3+1) >= 3  => true
        assert!(jcs.all_satisfied_map(&map));

        // Make B longer → first constraint fails
        map.insert('B', "LONGER".to_string()); // 6
        // (2+6) <= 6  is false  → overall false
        assert!(!jcs.all_satisfied_map(&map));
    }
}
