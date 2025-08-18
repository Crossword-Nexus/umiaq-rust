// src/scan_hints.rs
// -----------------------------------------------------------------------------
// Fast, form-local length hints for prefiltering during the scan phase.
//
// This module computes (min_len, max_len?) for a single ParsedForm, taking into account:
//   • fixed tokens in the form (i.e., literals, '.', '@', '#', [charset], /anagram)
//   • frequencies of variables that appear in the form (Var, RevVar)
//   • unary per-variable bounds from VarConstraints
//   • joint group constraints from JointConstraints that refer ONLY to vars
//     present in THIS form (e.g., "|AB|=6").
//
// The result can be used as a cheap prefilter: if a candidate word’s length does
// not satisfy these bounds, you can skip calling the heavy matcher altogether.
//
// Design notes:
// - We intentionally do *not* attempt global propagation across multiple forms.
//   These hints are computed per-form, once per equation.
// - We do not try to detect infeasibility of the full constraint set; if min>max
//   emerges it simply means "no candidates" for that form.
// - "!=" (not-equal) group relations are ignored for tightening because they do
//   not produce a contiguous interval--we conservatively skip tightening on them.
// - Presence of '*' in the form makes the form’s max length unbounded for the
//   hint’s purposes (even if unary-var maxima are finite), because '*' can soak
//   an arbitrary number of extra characters.
// -----------------------------------------------------------------------------

use std::collections::{HashMap, HashSet};

use crate::constraints::{VarConstraint, VarConstraints};
use crate::joint_constraints::{JointConstraints, JointConstraint, RelMask};
use crate::parser::{FormPart, ParsedForm};

/// Resulting per-form hints.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PatternLenHints {
    /// Lower bound on the form’s length. `None` = unbounded below.
    pub min_len: Option<usize>,
    /// Upper bound on the form’s length. `None` = unbounded above.
    pub max_len: Option<usize>,
}

impl PatternLenHints {
    /// Quick check for a candidate word length against this hint.
    pub fn is_word_len_possible(&self, len: usize) -> bool {
        self.min_len.is_none_or(|min_len| len >= min_len) && self.max_len.is_none_or(|max_len| len <= max_len)
    }
}

/// A joint constraint over variables *restricted to this form* considered
/// as a contiguous total length bound for their raw lengths (not weighted).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupLenConstraint {
    pub vars: Vec<char>,          // e.g., ['A','B'] for |AB|
    pub total_min: usize,         // inclusive
    pub total_max: Option<usize>, // inclusive, None => unbounded
}

/// Convert a crate-level `JointConstraint` to a `GroupLenConstraint` interval.
/// Returns `None` for incompatible/non-interval relations (e.g., NE) or empty.
fn group_from_joint(jc: &JointConstraint) -> Option<GroupLenConstraint> {
    // Map RelMask to [min,max] on the target.
    // Note: For LT/GT we avoid underflow/overflow.
    let (tmin, tmax_opt) = if jc.rel == RelMask::EQ {
        (jc.target, Some(jc.target))
    } else if jc.rel == RelMask::LE {
        (0, Some(jc.target))
    } else if jc.rel == RelMask::LT {
        if jc.target == 0 { return Some(GroupLenConstraint { vars: jc.vars.clone(), total_min: 0, total_max: Some(0) }); }
        (0, Some(jc.target - 1))
    } else if jc.rel == RelMask::GE {
        (jc.target, None)
    } else if jc.rel == RelMask::GT {
        (jc.target.saturating_add(1), None)
    } else {
        // NE (or unusual mask combos) don't give a single interval — skip tightening.
        return None;
    };

    // Basic sanity: empty interval ⇒ None
    if let Some(tmax) = tmax_opt && tmin > tmax { return None; }

    Some(GroupLenConstraint { vars: jc.vars.clone(), total_min: tmin, total_max: tmax_opt })
}

/// Compute per-form hints from a `ParsedForm` *and* the equation’s constraints.
///
/// - `vcs`: the full equation’s `VarConstraints` (we’ll only read vars present in form)
/// - `jcs`: the equation’s `JointConstraints` (we’ll filter to constraints whose
///   variable set is a subset of form’s variables)
pub fn form_len_hints_pf(
    form: &ParsedForm,
    vcs: &VarConstraints,
    jcs: Option<&JointConstraints>,
) -> PatternLenHints {
    form_len_hints_iter(form, |c| var_bounds_from_vcs(vcs, c), &group_constraints_for_form(form, jcs))
}

/// Core generic implementation — accepts any iterator yielding `&FormPart`
/// (e.g., a `&ParsedForm` thanks to its `IntoIterator` impl or a slice).
pub fn form_len_hints_iter<'a, I, F>(
    parts: I,
    mut get_var_bounds: F,
    form_groups: &[GroupLenConstraint],
) -> PatternLenHints
where
    I: IntoIterator<Item = &'a FormPart>,
    F: FnMut(char) -> (Option<usize>, Option<usize>), // (min, max)
{
    #[derive(Clone, Copy, Debug)]
    struct Bounds { li: Option<usize>, ui: Option<usize> }

    // 1. Scan tokens: accumulate fixed_base, detect '*', and count var frequencies
    let mut fixed_base = 0;
    let mut has_star = false;
    let mut var_frequency = HashMap::new();

    for p in parts {
        match p {
            FormPart::Star => { has_star = true; }
            FormPart::Dot | FormPart::Vowel | FormPart::Consonant | FormPart::Charset(_) => {
                fixed_base += 1;
            }
            FormPart::Lit(s) | FormPart::Anagram(s) => { fixed_base += s.len(); }
            FormPart::Var(v) | FormPart::RevVar(v) => { *var_frequency.entry(*v).or_insert(0) += 1; }
        }
    }

    // Exact case (exit early)): no variables and no star ⇒ exact fixed length
    if var_frequency.is_empty() && !has_star {
        return PatternLenHints { min_len: Some(fixed_base), max_len: Some(fixed_base) };
    }

    // 2. Pull unary bounds just for vars in this form
    let mut vars: Vec<char> = var_frequency.keys().copied().collect();
    vars.sort_unstable();

    let mut bounds: HashMap<char, Bounds> = HashMap::new();
    for &v in &vars {
        let (li, ui) = get_var_bounds(v);
        bounds.insert(v, Bounds { li, ui });
    }

    let get_weight = |v: char| *var_frequency.get(&v).unwrap_or(&0);

    // Baseline min/max ignoring groups
    let mut weighted_min =
        // TODO! do not do "unwrap_or(1)"
        Some(fixed_base + vars.iter().map(|&v| get_weight(v) * bounds[&v].li.unwrap_or(1)).sum::<usize>());

    let mut weighted_max =
        if has_star || vars.iter().any(|&v| bounds[&v].ui.is_none()) {
            None
        } else {
            Some(fixed_base + vars.iter().map(|&v| get_weight(v) * bounds[&v].ui.unwrap()).sum::<usize>())
        };

    // 3. Tighten with group constraints valid for this form
    for g in form_groups {
        #[derive(Clone, Copy)]
        struct Row { w: usize, li: Option<usize>, ui: Option<usize> }

        /// Compute the weighted extremal sum at fixed total `t`.
        /// If `minimize` is true, distribute to cheapest weights first (minimization).
        /// If false, distribute to most expensive weights first (maximization).
        fn weighted_extreme_for_t(rows: &[Row], sum_li: usize, sum_ui: Option<usize>, t: usize, minimize: bool) -> Option<usize> {
            if t < sum_li { return None; }
            if let Some(su) = sum_ui && t > su { return None; }

            // TODO! do not do "unwrap_or(1)"
            let base = rows.iter().map(|r| r.w * r.li.unwrap_or(1)).sum::<usize>();
            let mut rem = t - sum_li;

            let mut rows_sorted = rows.to_vec();
            if minimize {
                rows_sorted.sort_by_key(|r| r.w); // cheapest first
            } else {
                rows_sorted.sort_by_key(|r| std::cmp::Reverse(r.w)); // most expensive first
            }

            let mut extra = 0;
            for r in &rows_sorted {
                if rem == 0 { break; }
                // TODO! do not do "unwrap_or(1)"
                let cap = r.ui.map_or(rem, |u| u.saturating_sub(r.li.unwrap_or(1)));
                let take = cap.min(rem);
                extra += r.w * take;
                rem -= take;
            }

            debug_assert_eq!(rem, 0);
            Some(base + extra)
        }

        if g.vars.is_empty() { continue; }
        // Intersect with the form’s variables
        let mut gvars: Vec<char> = g.vars.iter().copied().filter(|v| var_frequency.contains_key(v)).collect();
        gvars.sort_unstable();
        if gvars.is_empty() { continue; }

        let mut rows: Vec<Row> = Vec::with_capacity(gvars.len());
        let mut sum_li: usize = 0;
        let mut sum_ui: Option<usize> = Some(0);
        for &v in &gvars {
            let b = bounds[&v];
            rows.push(Row { w: get_weight(v), li: b.li, ui: b.ui });
            // TODO! do not do "unwrap_or(1)"
            sum_li += b.li.unwrap_or(1);
            sum_ui = match (sum_ui, b.ui) { (Some(a), Some(bu)) => Some(a + bu), _ => None };
        }

        let weighted_min_for_t = |t: usize| weighted_extreme_for_t(&rows, sum_li, sum_ui, t, true);
        let weighted_max_for_t = |t: usize| weighted_extreme_for_t(&rows, sum_li, sum_ui, t, false);


        // Evaluate endpoints of the group’s allowed total interval
        // ---- Account for group vars that are NOT in this form ------------------
        // They eat into the group's total before we allocate to in-form vars.
        let mut outside_form_min = Some(0);
        let mut outside_form_max = Some(0);

        for v in g.vars.iter().copied().filter(|v| !var_frequency.contains_key(v)) {
            let (li, ui) = get_var_bounds(v);
            outside_form_min = match (outside_form_min, li) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None, // unbounded outside ⇒ no finite lower bound for in-form part
            };
            outside_form_max = match (outside_form_max, ui) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None, // unbounded outside ⇒ no finite upper bound for in-form part
            };
        }

        // Effective totals for the in-form part of this group:
        // - For the LOWER bound, outside takes as much as possible (use outside_form_max if finite).
        // - For the UPPER bound, outside takes as little as possible (use outside_form_min).
        let tmin_eff_opt = outside_form_max.map(|of_max| g.total_min.saturating_sub(of_max));
        // TODO! do not do "unwrap_or(1)"
        let tmax_eff_opt = g.total_max.map(|tmax| tmax.saturating_sub(outside_form_min.unwrap_or(1)));

        // Evaluate endpoints of the adjusted interval for in-form vars.
        let gmin_w = match tmin_eff_opt {
            Some(tmin_eff) => weighted_min_for_t(tmin_eff),
            None => None, // no finite lower bound if outside unbounded
        };
        let gmax_w = match tmax_eff_opt {
            Some(tmax_eff) => weighted_max_for_t(tmax_eff),
            None => None,
        };

        // Combine with outside-of-group contributions
        let outside: Vec<char> = vars.iter().copied().filter(|v| !gvars.contains(v)).collect();
        // TODO! do not do "unwrap_or(1)"
        let outside_min = outside.iter().map(|&v| get_weight(v) * bounds[&v].li.unwrap_or(1)).sum::<usize>();
        let outside_max: Option<usize> = if has_star || outside.iter().any(|&v| bounds[&v].ui.is_none()) {
            None
        } else {
            Some(outside.iter().map(|&v| get_weight(v) * bounds[&v].ui.unwrap()).sum::<usize>())
        };

        if let Some(gmin) = gmin_w {
            weighted_min = weighted_min.map(|wm| wm.max(fixed_base + gmin + outside_min));
        }
        // Compute a candidate upper bound from this group + outside.
        let candidate_upper = match (gmax_w, outside_max) {
            (Some(gm), Some(om)) => Some(fixed_base + gm + om),
            _ => None,
        };

        // Combine with the running upper bound:
        weighted_max = match (weighted_max, candidate_upper) {
            (Some(cur), Some(cand)) => Some(cur.min(cand)),
            (None, Some(cand))      => Some(cand),
            (Some(cur), None)       => Some(cur),
            (None, None)            => None,
        };
    }

    PatternLenHints { min_len: weighted_min, max_len: weighted_max }
}

/// Helper: extract (min,max?) for a variable from `VarConstraints`.
fn var_bounds_from_vcs(vcs: &VarConstraints, var: char) -> (Option<usize>, Option<usize>) {
    // TODO is there are a better/canonical way to do this?
    let vc = if let Some(vc) = vcs.get(var) { vc } else { &VarConstraint::default() };

    (vc.min_length, vc.max_length)
}

/// Build the list of group constraints (as contiguous intervals) that are *scoped
/// to this form*: every referenced variable must appear in the form.
fn group_constraints_for_form(form: &ParsedForm, jcso: Option<&JointConstraints>) -> Vec<GroupLenConstraint> {
    let Some(jcs) = jcso else { return Vec::new(); };

    // Collect variable set present in this form
    // TODO is there a better way to do this?
    let present: HashSet<_> = form.iter().filter_map(|p|
        {
            if let FormPart::Var(v) | FormPart::RevVar(v) = p { Some(v) } else { None }
        }).collect();

    // Filter joint constraints to those whose vars ⊆ present, and convert to intervals
    jcs.as_vec.iter()
        .filter(|jc| jc.vars.iter().any(|v| present.contains(v)))
        .filter_map(group_from_joint)
        .collect()
}

// -----------------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::FormPart;

    // Minimal helper ParsedForm for tests without pulling regex machinery.
    fn pf(parts: Vec<FormPart>) -> ParsedForm {
        // We don’t care about the prefilter for these tests; build a dummy.
        // ParsedForm::of requires a Regex; use a simple anchored ".*" via a helper
        // from your parser module if available. If not, expose a test-only ctor.
        // Here, we fake it by reusing ParsedForm::of through a small form_to_regex_str.
        // If that’s not accessible, replace with a test-only constructor in your codebase.
        ParsedForm { parts, prefilter: fancy_regex::Regex::new("^.*$").unwrap() }
    }

    #[test]
    fn no_vars_no_star_exact() {
        let form = pf(vec![FormPart::Lit("AB".into()), FormPart::Dot, FormPart::Anagram("XY".into())]);
        let vcs = VarConstraints::default();
        let hints = form_len_hints_pf(&form, &vcs, None);

        let expected = PatternLenHints { min_len: Some(5), max_len: Some(5) };
        assert_eq!(expected, hints);
        assert!(hints.is_word_len_possible(5));
        assert!(!hints.is_word_len_possible(4));
    }

    #[test]
    fn star_unbounded_max() {
        let form = pf(vec![FormPart::Lit("HEL".into()), FormPart::Star, FormPart::Var('A')]);
        let mut vcs = VarConstraints::default();

        // A in [2,4]
        let a = vcs.ensure('A');
        a.min_length = Some(2);
        a.max_length = Some(4);

        let hints = form_len_hints_pf(&form, &vcs, None);

        let expected = PatternLenHints { min_len: Some(5), max_len: None };
        assert_eq!(expected, hints);
    }

    #[test]
    fn unary_bounds_only() {
        // A . B ; base=1
        let form = pf(vec![FormPart::Var('A'), FormPart::Dot, FormPart::Var('B')]);
        let mut vcs = VarConstraints::default();

        let a = vcs.ensure('A');
        a.min_length = Some(2);
        a.max_length = Some(3);

        let b = vcs.ensure('B');
        b.min_length = Some(1);
        b.max_length = Some(5);

        let hints = form_len_hints_pf(&form, &vcs, None);

        let expected = PatternLenHints { min_len: Some(4), max_len: Some(9) };
        assert_eq!(expected, hints);
    }

    #[test]
    fn group_eq_on_ab_with_weights() {
        // Form: A B A   weights wA=2, wB=1
        let form = pf(vec![FormPart::Var('A'), FormPart::Var('B'), FormPart::Var('A')]);
        let vcs = VarConstraints::default();

        // Build a JointConstraints equivalent to |AB|=6
        let jc = JointConstraint { vars: vec!['A','B'], target: 6, rel: RelMask::EQ };
        let jcs = JointConstraints { as_vec: vec![jc] };

        let hints = form_len_hints_pf(&form, &vcs, Some(&jcs));
        // Min puts as much as possible on cheaper (B): |AB| + 1 = 6 + 1 = 7
        // Max puts all on A: |AB| + A = 6 + 5 = 11

        let expected = PatternLenHints { min_len: Some(7), max_len: Some(11) };
        assert_eq!(expected, hints);
    }

    #[test]
    fn group_eq_plus_unary_forces_exact() {
        // . A B . ; base=2 ; |AB|=6 ; |A| fixed to 2
        let form = pf(vec![FormPart::Dot, FormPart::Var('A'), FormPart::Var('B'), FormPart::Dot]);
        let mut vcs = VarConstraints::default();

        let a = vcs.ensure('A');
        a.min_length = Some(2);
        a.max_length = Some(2);

        let jc = JointConstraint { vars: vec!['A','B'], target: 6, rel: RelMask::EQ };
        let jcs = JointConstraints { as_vec: vec![jc] };
        let hints = form_len_hints_pf(&form, &vcs, Some(&jcs));

        let expected = PatternLenHints { min_len: Some(8), max_len: Some(8) };
        assert_eq!(expected, hints);
    }

    #[test]
    fn ranged_group_bounds() {
        // A B with unary: A in [1,5], B in [0,10]; group |AB| in [4,6]
        let form = pf(vec![FormPart::Var('A'), FormPart::Var('B')]);
        let mut vcs = VarConstraints::default();

        let a = vcs.ensure('A');
        a.min_length = Some(1);
        a.max_length = Some(5);

        let b = vcs.ensure('B');
        b.min_length = Some(0);
        b.max_length = Some(10);

        let g1 = JointConstraint { vars: vec!['A','B'], target: 4, rel: RelMask::GE };
        let g2 = JointConstraint { vars: vec!['A','B'], target: 6, rel: RelMask::LE };
        let jcs = JointConstraints { as_vec: vec![g1, g2] };
        let hints = form_len_hints_pf(&form, &vcs, Some(&jcs));


        let expected = PatternLenHints { min_len: Some(4), max_len: Some(6) };
        assert_eq!(expected, hints);
    }

    #[test]
    fn star_blocks_exact_even_with_exact_groups() {
        // A*B ; |AB|=6
        let form = pf(vec![FormPart::Var('A'), FormPart::Star, FormPart::Var('B')]);
        let jc = JointConstraint { vars: vec!['A','B'], target: 6, rel: RelMask::EQ };
        let jcs = JointConstraints { as_vec: vec![jc] };
        let vcs = VarConstraints::default();
        let hints = form_len_hints_pf(&form, &vcs, Some(&jcs));

        // star contributes 0 to min_len
        let expected = PatternLenHints { min_len: Some(6), max_len: None };
        assert_eq!(expected, hints);
    }

    #[test]
    fn group_hints_apply_to_single_var() {
        let form = pf(vec![FormPart::Var('A')]);
        let jc = JointConstraint { vars: vec!['A','B'], target: 6, rel: RelMask::EQ };
        let jcs = JointConstraints { as_vec: vec![jc] };
        let vcs = VarConstraints::default();
        let hints = form_len_hints_pf(&form, &vcs, Some(&jcs));

        // TODO should this be Some(1) or None ?
        let expected = PatternLenHints { min_len: Some(1), max_len: Some(5) };
        assert_eq!(expected, hints);
    }
}
