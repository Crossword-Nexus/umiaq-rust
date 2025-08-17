// src/scan_hints.rs
//
// Form-local length bounds for fast prefiltering.
// - Directly supports group constraints like |AB| = 6 on the variables
//   that appear in THIS form.
// - Computes (exact_len?, min_len, max_len?) for a single ParsedForm.
//
// Use in solver.rs:
//   1) For each form, collect the subset of joint constraints that involve
//      only variables present in that form, as `GroupLenConstraint`s.
//   2) Call `form_len_hints(parts, get_var_bounds, &groups)` to get bounds.
//   3) Prefilter words by length against these bounds before calling the matcher.

use crate::parser::FormPart;

/// Resulting per-form hints.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PatternLenHints {
    pub exact_len: Option<usize>,
    pub min_len: usize,
    pub max_len: Option<usize>,
}

/// A joint constraint over a set (or multiset) of variables, applied ONLY
/// to variables present in the current form:
///   - `vars`: e.g., ['A','B'] for |AB|, or ['A','A','B'] if multiplicity intended.
///     (Usually 1 each; multiplicity rarely used in constraints, but allowed.)
///   - `total_min`, `total_max`: bounds on the sum of **raw lengths**
///     (not weighted by occurrences in the form).
///
/// Examples:
///   |AB| = 6        => vars=['A','B'], total_min=6, total_max=Some(6)
///   |A| >= 3        => vars=['A'],     total_min=3, total_max=None
///   |BCD| ∈ [4, 8]  => vars=['B','C','D'], total_min=4, total_max=Some(8)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GroupLenConstraint {
    pub vars: Vec<char>,
    pub total_min: usize,
    pub total_max: Option<usize>,
}

/// Compute per-form length bounds (min/max/exact) for a sequence of FormPart,
/// given:
///   - a way to query unary var bounds (from your var_constraints),
///   - a list of group constraints that involve ONLY the vars present in this form.
///
/// The function:
///   1) Computes the base fixed length from literals/dots/vowel/consonant/charset/anagram.
///   2) Detects presence of Star: Star ⇒ `max_len=None` unless other parts force exact.
///   3) Builds a tiny LP on var lengths L_v with:
///        li ≤ L_v ≤ ui,  and  Σ_{v∈G} L_v ∈ [Tmin, Tmax]  for each group G
///      and objective Σ w_v * L_v, where w_v is the number of occurrences of v in the form.
///      We bound this weighted sum between min and max using greedy water-filling.
///
/// NOTE: This is a *bounds* computation — it does not try to detect infeasibility.
///       If groups conflict, you may get min > max; treat as “no matches”.
pub fn form_len_hints<F>(
    parts: &[FormPart],
    mut get_var_bounds: F,
    form_groups: &[GroupLenConstraint],
) -> PatternLenHints
where
    F: FnMut(char) -> (usize, Option<usize>), // returns (min, max)
{
    // 1) Scan tokens to compute:
    //    - fixed_base: fixed contribution from non-var, non-star tokens
    //    - has_star: whether '*' appears
    //    - multiplicities: how many times each var occurs in this form
    use std::collections::HashMap;
    let mut fixed_base = 0usize;
    let mut has_star = false;
    let mut multiplicity: HashMap<char, usize> = HashMap::new();

    for p in parts {
        match p {
            FormPart::Star => { has_star = true; }
            FormPart::Dot | FormPart::Vowel | FormPart::Consonant | FormPart::Charset(_) => {
                fixed_base += 1;
            }
            FormPart::Lit(s) | FormPart::Anagram(s) => {
                fixed_base += s.len();
            }
            FormPart::Var(v) | FormPart::RevVar(v) => {
                *multiplicity.entry(*v).or_insert(0) += 1;
            }
        }
    }

    // Early exit: If there are no variables and no Star, the length is exact already.
    if multiplicity.is_empty() && !has_star {
        return PatternLenHints {
            exact_len: Some(fixed_base),
            min_len: fixed_base,
            max_len: Some(fixed_base),
        };
    }

    // 2) Get unary var bounds (li, ui) for vars in this form.
    #[derive(Clone, Copy, Debug)]
    struct Bounds { li: usize, ui: Option<usize> }
    let mut vars: Vec<char> = multiplicity.keys().copied().collect();
    vars.sort_unstable();
    let mut bounds: HashMap<char, Bounds> = HashMap::new();
    for &v in &vars {
        let (min_v, max_v) = get_var_bounds(v);
        bounds.insert(v, Bounds { li: min_v, ui: max_v });
    }

    // Helper closures to sum weighted contributions for subsets.
    let weight = |v: char| -> usize { multiplicity.get(&v).copied().unwrap_or(0) };

    // Lower bound (without groups): base + Σ w_v * li
    let mut weighted_min = fixed_base
        + vars.iter().map(|&v| weight(v) * bounds[&v].li).sum::<usize>();

    // Upper bound (without groups): if any ui is None OR has Star => unbounded
    let mut weighted_max: Option<usize> = if has_star || vars.iter().any(|&v| bounds[&v].ui.is_none()) {
        None
    } else {
        Some(fixed_base + vars.iter().map(|&v| weight(v) * bounds[&v].ui.unwrap()).sum::<usize>())
    };

    // 3) Tighten with each group constraint (applies only to vars in this form).
    // For each group S, with Σ L_v ∈ [Tmin, Tmax]:
    //   Let weights be w_v for v∈S, and li/ui from unary bounds.
    //   Bound the weighted sum over S given the total T:
    //     - For min at fixed T: start at li, distribute extra to smallest weights.
    //     - For max at fixed T: start at li, distribute extra to largest weights.
    //   With T in [Tmin, Tmax], min occurs at Tmin, max at Tmax (weights ≥ 0).
    //
    // We combine with outside vars by adding their standalone contributions.
    for g in form_groups {
        if g.vars.is_empty() { continue; }

        // Extract this group's variables intersected with vars present.
        let mut gvars: Vec<char> = g.vars.iter().copied().filter(|v| multiplicity.contains_key(v)).collect();
        gvars.sort_unstable();
        if gvars.is_empty() { continue; }

        // Precompute li/ui and weights for group vars.
        #[derive(Clone, Copy)]
        struct Row { w: usize, li: usize, ui: Option<usize> }
        let mut rows: Vec<Row> = Vec::with_capacity(gvars.len());
        let mut sum_li: usize = 0;
        let mut sum_ui: Option<usize> = Some(0);
        for &v in &gvars {
            let b = bounds[&v];
            rows.push(Row { w: weight(v), li: b.li, ui: b.ui });
            sum_li += b.li;
            sum_ui = match (sum_ui, b.ui) {
                (Some(a), Some(bu)) => Some(a + bu),
                _ => None, // unbounded if any ui is None
            };
        }

        // Helper: min/max weighted sum on group for fixed total T.
        let mut weighted_min_for_T = |T: usize| -> Option<usize> {
            if T < sum_li { return None; }
            if let Some(sum_ui_fin) = sum_ui {
                if T > sum_ui_fin { return None; }
            }
            // Start from li
            let base = rows.iter().map(|r| r.w * r.li).sum::<usize>();
            let mut rem = T - sum_li;

            // Distribute to the *cheapest* weights first
            let mut rows_sorted = rows.clone();
            rows_sorted.sort_by_key(|r| r.w);

            let mut extra = 0usize;
            for r in rows_sorted.iter() {
                if rem == 0 { break; }
                let cap = match r.ui {
                    Some(ui) => ui.saturating_sub(r.li),
                    None => rem, // no upper bound
                };
                let take = cap.min(rem);
                extra += r.w * take;
                rem -= take;
            }
            debug_assert_eq!(rem, 0, "distribution must fill T");
            Some(base + extra)
        };

        let mut weighted_max_for_T = |T: usize| -> Option<usize> {
            if T < sum_li { return None; }
            if let Some(sum_ui_fin) = sum_ui {
                if T > sum_ui_fin { return None; }
            }
            let base = rows.iter().map(|r| r.w * r.li).sum::<usize>();
            let mut rem = T - sum_li;

            // Distribute to the *most expensive* weights first
            let mut rows_sorted = rows.clone();
            rows_sorted.sort_by_key(|r| std::cmp::Reverse(r.w));

            let mut extra = 0usize;
            for r in rows_sorted.iter() {
                if rem == 0 { break; }
                let cap = match r.ui {
                    Some(ui) => ui.saturating_sub(r.li),
                    None => rem,
                };
                let take = cap.min(rem);
                extra += r.w * take;
                rem -= take;
            }
            debug_assert_eq!(rem, 0, "distribution must fill T");
            Some(base + extra)
        };

        // The group’s weighted sum bounds across T in [Tmin, Tmax]
        // are attained at the endpoints because weights are nonnegative.
        // If Tmax is None (unbounded), then the group's weighted max is unbounded.
        let g_min_weighted = weighted_min_for_T(g.total_min);

        let g_max_weighted = match g.total_max {
            Some(tmax) => weighted_max_for_T(tmax),
            None => None,
        };

        // Combine with outside-of-group vars contributions.
        // Outside vars are independent of this group, so:
        let outside_vars: Vec<char> = vars.iter().copied().filter(|v| !gvars.contains(v)).collect();

        // Outside min = Σ w * li
        let outside_min = outside_vars.iter().map(|&v| weight(v) * bounds[&v].li).sum::<usize>();

        // Outside max = Σ w * ui if all bounded, else unbounded
        let outside_max: Option<usize> = if has_star || outside_vars.iter().any(|&v| bounds[&v].ui.is_none()) {
            None
        } else {
            Some(outside_vars.iter().map(|&v| weight(v) * bounds[&v].ui.unwrap()).sum::<usize>())
        };

        // Now update overall [weighted_min, weighted_max] with this group's tighter info.
        if let Some(gmin) = g_min_weighted {
            weighted_min = weighted_min.max(fixed_base + gmin + outside_min);
        }
        match (weighted_max, g_max_weighted, outside_max) {
            // Any unbounded piece ⇒ remains unbounded
            (None, _, _) | (_, None, _) | (_, _, None) => { weighted_max = None; }
            (Some(cur), Some(gmax), Some(omax)) => {
                let new_max = fixed_base + gmax + omax;
                weighted_max = Some(cur.min(new_max));
            }
        }
    }

    // 4) Possibly determine exact length.
    // Exact length iff:
    //  - There is NO Star
    //  - weighted_max is Some(M) and equals weighted_min
    let exact_len = if !has_star {
        match weighted_max {
            Some(mx) if mx == weighted_min => Some(mx),
            _ => None,
        }
    } else {
        None
    };

    PatternLenHints {
        exact_len,
        min_len: weighted_min,
        max_len: weighted_max,
    }
}

/* =========================
   Unit tests
   ========================= */

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: constant bounds provider
    fn bounds_fixed<A: IntoIterator<Item = (char, (usize, Option<usize>))>>(pairs: A)
                                                                            -> impl FnMut(char) -> (usize, Option<usize>)
    {
        use std::collections::HashMap;
        let mut m: HashMap<char, (usize, Option<usize>)> = HashMap::new();
        for (k, v) in pairs { m.insert(k, v); }
        move |c: char| m.get(&c).cloned().unwrap_or((0, None))
    }

    #[test]
    fn no_vars_no_star_exact() {
        let parts = vec![
            FormPart::Lit("AB".into()),
            FormPart::Dot,
            FormPart::Anagram("XY".into()),
        ];
        let hints = form_len_hints(&parts, bounds_fixed(std::iter::empty()), &[]);
        assert_eq!(hints.exact_len, Some(2 + 1 + 2));
        assert_eq!(hints.min_len, 5);
        assert_eq!(hints.max_len, Some(5));
    }

    #[test]
    fn star_makes_unbounded_max() {
        let parts = vec![
            FormPart::Lit("HEL".into()), // 3
            FormPart::Star,              // *
            FormPart::Var('A'),
        ];
        let mut get = bounds_fixed([('A', (2, Some(4)))]);
        let hints = form_len_hints(&parts, &mut get, &[]);
        // base min = 3 + 0 + 2 = 5; max is unbounded due to Star
        assert_eq!(hints.exact_len, None);
        assert_eq!(hints.min_len, 5);
        assert_eq!(hints.max_len, None);
    }

    #[test]
    fn simple_unary_bounds() {
        // parts: A . B  (weights: wA=1, wB=1), base=1
        let parts = vec![
            FormPart::Var('A'),
            FormPart::Dot,
            FormPart::Var('B'),
        ];
        let mut get = bounds_fixed([('A', (2, Some(3))), ('B', (1, Some(5)))]);
        let hints = form_len_hints(&parts, &mut get, &[]);
        // min = base + 1*2 + 1*1 = 1 + 3 = 4
        // max = base + 1*3 + 1*5 = 1 + 8 = 9
        assert_eq!(hints.min_len, 4);
        assert_eq!(hints.max_len, Some(9));
        assert_eq!(hints.exact_len, None);
    }

    #[test]
    fn group_sum_exact_min_max_with_weights() {
        // Form: A B A   => weights: wA=2, wB=1, base=0
        // Group: |AB| = 6  => L_A + L_B = 6
        // Unary: no bounds (defaults 0..∞)
        //
        // Min weighted sum = minimize 2*L_A + 1*L_B subject to L_A + L_B = 6.
        // Put as much as possible on B (cheaper weight): L_B=6, L_A=0 => 6
        // Max weighted sum: put all on A: L_A=6, L_B=0 => 12
        let parts = vec![
            FormPart::Var('A'),
            FormPart::Var('B'),
            FormPart::Var('A'),
        ];
        let mut get = bounds_fixed(std::iter::empty()); // no unary bounds
        let groups = vec![
            GroupLenConstraint { vars: vec!['A','B'], total_min: 6, total_max: Some(6) }
        ];
        let hints = form_len_hints(&parts, &mut get, &groups);
        assert_eq!(hints.min_len, 6);
        assert_eq!(hints.max_len, Some(12));
        assert_eq!(hints.exact_len, None); // not forced since weights differ
    }

    #[test]
    fn group_plus_unary_can_force_exact() {
        // Form: A B, base=2 (two dots)
        // - weights: wA=1, wB=1
        // - group: |AB| = 6  -> L_A + L_B = 6
        // - unary: A in [2,2]  (|A| fixed), B unrestricted
        //
        // Then L_B = 4; weighted sum = 1*2 + 1*4 = 6; base=2 ⇒ exact total 8
        let parts = vec![
            FormPart::Dot,
            FormPart::Var('A'),
            FormPart::Var('B'),
            FormPart::Dot,
        ];
        let mut get = bounds_fixed([('A', (2, Some(2)))]);
        let groups = vec![
            GroupLenConstraint { vars: vec!['A','B'], total_min: 6, total_max: Some(6) }
        ];
        let hints = form_len_hints(&parts, &mut get, &groups);
        assert_eq!(hints.min_len, 8);
        assert_eq!(hints.max_len, Some(8));
        assert_eq!(hints.exact_len, Some(8));
    }

    #[test]
    fn ranged_group_bounds() {
        // Form: A B, weights wA=wB=1, base=0
        // Unary: A in [1, 5], B in [0, 10]
        // Group: |AB| in [4, 6]
        //
        // - Min occurs at T=4, push to cheapest (equal weights => any distribution),
        //   but also respect unary mins: A>=1, B>=0  ⇒ possible min = 4
        // - Max occurs at T=6, push to largest weights (same), but bounded by A<=5,B<=10 ⇒ 6
        let parts = vec![FormPart::Var('A'), FormPart::Var('B')];
        let mut get = bounds_fixed([('A', (1, Some(5))), ('B', (0, Some(10)))]);
        let groups = vec![
            GroupLenConstraint { vars: vec!['A','B'], total_min: 4, total_max: Some(6) }
        ];
        let hints = form_len_hints(&parts, &mut get, &groups);
        assert_eq!(hints.min_len, 4);
        assert_eq!(hints.max_len, Some(6));
    }

    #[test]
    fn star_still_blocks_exact_even_with_exact_groups() {
        // Form: A * B ; |AB|=6; base=0
        // Star ⇒ max is unbounded, exact_len stays None.
        let parts = vec![FormPart::Var('A'), FormPart::Star, FormPart::Var('B')];
        let mut get = bounds_fixed(std::iter::empty());
        let groups = vec![
            GroupLenConstraint { vars: vec!['A','B'], total_min: 6, total_max: Some(6) }
        ];
        let hints = form_len_hints(&parts, &mut get, &groups);
        assert_eq!(hints.min_len, 6);   // star contributes 0 to min
        assert_eq!(hints.max_len, None);
        assert_eq!(hints.exact_len, None);
    }
}
