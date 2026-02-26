use pale_ale_rotor::{
    embed_simple29_to_even128, even_masks_v1, grade_of_mask, inner,
    left_fold_mul_time_reversed_normalize_once, mul_even128, normalize, normalize_vec8,
    simple_rotor29_doc_to_ans, Even128, EvenError, RotorConfig, RotorError, RotorStep, ALGEBRA_ID,
    BLADE_SIGN_ID, COMPOSITION_ID, EMBED_ID, NORMALIZE_ID, REVERSE_ID, ROOT_DIM,
};
use std::collections::BTreeMap;
use std::fmt;

pub const GATE2_SPEC_VERSION: &str = "v4.1.0-ssot.3";
pub const GATE2_METHOD_ID: &str = "rotor_holonomy_telemetry_v1";
pub const H3_NAME_ID: &str = "higher_grade_energy_ratio_v1";

pub const GATE2_ROTOR_CONSTRUCTION_ID: &str = "simple_rotor29_uv_v1";
pub const GATE2_THETA_SOURCE_ID: &str = "theta_uv_atan2_v1";
pub const GATE2_BIVECTOR_BASIS_ID: &str = "lex_i_lt_j_v1";
pub const GATE2_ANTIPODAL_POLICY_ID: &str =
    "antipodal_split_v1(angle_only_for_theta,drop_on_nonfinite)";

#[derive(Clone, Debug, PartialEq)]
pub struct Gate2TelemetryInput {
    pub sample_id: u64,
    pub ans_vec8: Vec<[f64; ROOT_DIM]>,
    pub answer_length: Option<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gate2TelemetryIds {
    pub spec_version: &'static str,
    pub method_id: &'static str,
    pub algebra_id: &'static str,
    pub blade_sign_id: &'static str,
    pub reverse_id: &'static str,
    pub normalize_id: &'static str,
    pub composition_id: &'static str,
    pub embed_id: &'static str,
    pub h3_name_id: &'static str,
    pub rotor_construction_id: &'static str,
    pub theta_source_id: &'static str,
    pub bivector_basis_id: &'static str,
    pub antipodal_policy_id: &'static str,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Gate2SampleAbortReason {
    NonFiniteAnswerVector,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate2MetricMissingReason {
    NotEnoughAnswerUnits,
    MissingEvenRotorStep,
    NormalizeFailure,
    NoUsableTriangleLoops,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MissingEvenRotorReason {
    AntipodalAngleOnly,
    Vec8NormalizationFailure,
    RotorConstructionFailure,
}

impl fmt::Display for MissingEvenRotorReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AntipodalAngleOnly => write!(f, "antipodal_angle_only"),
            Self::Vec8NormalizationFailure => write!(f, "vec8_normalization_failure"),
            Self::RotorConstructionFailure => write!(f, "rotor_construction_failure"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gate2TelemetryCounts {
    pub count_missing_even_rotor_steps: usize,
    pub count_loops_considered: usize,
    pub count_loops_used: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gate2StatTriple {
    pub max: f64,
    pub mean: f64,
    pub p90: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate2SampleTelemetry {
    pub sample_id: u64,
    pub n_ans_units: usize,
    pub counts: Gate2TelemetryCounts,
    pub sample_abort_reason: Option<Gate2SampleAbortReason>,
    pub missing_even_rotor_reasons: Vec<MissingEvenRotorReason>,
    pub h1b_closure_error: Option<f64>,
    pub h1b_missing_reason: Option<Gate2MetricMissingReason>,
    pub h2_loop_stats: Option<Gate2StatTriple>,
    pub h2_missing_reason: Option<Gate2MetricMissingReason>,
    pub h3_ratio_total_product: Option<f64>,
    pub h3_total_missing_reason: Option<Gate2MetricMissingReason>,
    pub h3_ratio_triangle_loop_stats: Option<Gate2StatTriple>,
    pub h3_loop_missing_reason: Option<Gate2MetricMissingReason>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate2TelemetryResult {
    pub ids: Gate2TelemetryIds,
    pub per_sample: Gate2SampleTelemetry,
}

#[derive(Clone, Debug, PartialEq)]
enum RotorBuild {
    Materialized(Box<Even128>),
    Missing(MissingEvenRotorReason),
}

#[derive(Clone, Debug)]
struct RotorBuilder<'a> {
    ans_vec8: &'a [[f64; ROOT_DIM]],
    cache: BTreeMap<(usize, usize), RotorBuild>,
    missing_count: usize,
    missing_reasons: Vec<MissingEvenRotorReason>,
}

impl<'a> RotorBuilder<'a> {
    fn new(ans_vec8: &'a [[f64; ROOT_DIM]]) -> Self {
        Self {
            ans_vec8,
            cache: BTreeMap::new(),
            missing_count: 0,
            missing_reasons: Vec::new(),
        }
    }

    fn get(&mut self, from_idx: usize, to_idx: usize) -> RotorBuild {
        if let Some(cached) = self.cache.get(&(from_idx, to_idx)) {
            return cached.clone();
        }

        let built = self.build_uncached(from_idx, to_idx);
        if let RotorBuild::Missing(reason) = &built {
            self.missing_count += 1;
            self.missing_reasons.push(reason.clone());
        }
        self.cache.insert((from_idx, to_idx), built.clone());
        built
    }

    fn build_uncached(&self, from_idx: usize, to_idx: usize) -> RotorBuild {
        let from = self.ans_vec8[from_idx];
        let to = self.ans_vec8[to_idx];
        let from_norm = match normalize_vec8(from) {
            Ok(v) => v,
            Err(_) => return RotorBuild::Missing(MissingEvenRotorReason::Vec8NormalizationFailure),
        };
        let to_norm = match normalize_vec8(to) {
            Ok(v) => v,
            Err(_) => return RotorBuild::Missing(MissingEvenRotorReason::Vec8NormalizationFailure),
        };
        match simple_rotor29_doc_to_ans(from_norm, to_norm, RotorConfig::default()) {
            Ok(RotorStep::Materialized { r29, .. }) => {
                RotorBuild::Materialized(Box::new(embed_simple29_to_even128(&r29)))
            }
            Ok(RotorStep::AntipodalAngleOnly { .. }) => {
                RotorBuild::Missing(MissingEvenRotorReason::AntipodalAngleOnly)
            }
            Err(RotorError::Vec8(_) | RotorError::NonFiniteTheta | RotorError::RenormFailure) => {
                RotorBuild::Missing(MissingEvenRotorReason::RotorConstructionFailure)
            }
        }
    }
}

pub fn compute_gate2_telemetry(input: &Gate2TelemetryInput) -> Gate2TelemetryResult {
    let ids = Gate2TelemetryIds {
        spec_version: GATE2_SPEC_VERSION,
        method_id: GATE2_METHOD_ID,
        algebra_id: ALGEBRA_ID,
        blade_sign_id: BLADE_SIGN_ID,
        reverse_id: REVERSE_ID,
        normalize_id: NORMALIZE_ID,
        composition_id: COMPOSITION_ID,
        embed_id: EMBED_ID,
        h3_name_id: H3_NAME_ID,
        rotor_construction_id: GATE2_ROTOR_CONSTRUCTION_ID,
        theta_source_id: GATE2_THETA_SOURCE_ID,
        bivector_basis_id: GATE2_BIVECTOR_BASIS_ID,
        antipodal_policy_id: GATE2_ANTIPODAL_POLICY_ID,
    };

    if has_non_finite_ans_vec8(&input.ans_vec8) {
        return Gate2TelemetryResult {
            ids,
            per_sample: Gate2SampleTelemetry {
                sample_id: input.sample_id,
                n_ans_units: input.ans_vec8.len(),
                counts: Gate2TelemetryCounts {
                    count_missing_even_rotor_steps: 0,
                    count_loops_considered: 0,
                    count_loops_used: 0,
                },
                sample_abort_reason: Some(Gate2SampleAbortReason::NonFiniteAnswerVector),
                missing_even_rotor_reasons: Vec::new(),
                h1b_closure_error: None,
                h1b_missing_reason: Some(Gate2MetricMissingReason::MissingEvenRotorStep),
                h2_loop_stats: None,
                h2_missing_reason: Some(Gate2MetricMissingReason::MissingEvenRotorStep),
                h3_ratio_total_product: None,
                h3_total_missing_reason: Some(Gate2MetricMissingReason::MissingEvenRotorStep),
                h3_ratio_triangle_loop_stats: None,
                h3_loop_missing_reason: Some(Gate2MetricMissingReason::MissingEvenRotorStep),
            },
        };
    }

    let n_ans = input.ans_vec8.len();
    let mut builder = RotorBuilder::new(&input.ans_vec8);

    let mut h1b_closure_error = None;
    let mut h1b_missing_reason = None;
    let mut h1_raw = None;

    if n_ans < 2 {
        h1b_missing_reason = Some(Gate2MetricMissingReason::NotEnoughAnswerUnits);
    } else {
        let mut adjacent = Vec::with_capacity(n_ans - 1);
        let mut has_missing_adjacent = false;
        for idx in 0..(n_ans - 1) {
            match builder.get(idx, idx + 1) {
                RotorBuild::Materialized(rotor) => adjacent.push(*rotor),
                RotorBuild::Missing(_) => {
                    has_missing_adjacent = true;
                    break;
                }
            }
        }

        if has_missing_adjacent {
            h1b_missing_reason = Some(Gate2MetricMissingReason::MissingEvenRotorStep);
        } else {
            h1_raw = left_fold_time_reversed_raw(&adjacent);
            match left_fold_mul_time_reversed_normalize_once(&adjacent) {
                Ok(r_total) => match builder.get(0, n_ans - 1) {
                    RotorBuild::Materialized(r_direct) => {
                        h1b_closure_error =
                            Some(projective_chordal_distance(&r_total, r_direct.as_ref()));
                    }
                    RotorBuild::Missing(_) => {
                        h1b_missing_reason = Some(Gate2MetricMissingReason::MissingEvenRotorStep);
                        h1_raw = None;
                    }
                },
                Err(EvenError::NonFiniteNormSquared | EvenError::NonPositiveNormSquared) => {
                    h1b_missing_reason = Some(Gate2MetricMissingReason::NormalizeFailure);
                    h1_raw = None;
                }
            }
        }
    }

    let mut loops_considered = 0usize;
    let mut loops_used = 0usize;
    let mut h2_distances = Vec::new();
    let mut h3_loop_ratios = Vec::new();
    if n_ans >= 3 {
        for idx in 0..(n_ans - 2) {
            loops_considered += 1;
            let r01 = builder.get(idx, idx + 1);
            let r12 = builder.get(idx + 1, idx + 2);
            let r20 = builder.get(idx + 2, idx);
            let (r01, r12, r20) = match (r01, r12, r20) {
                (
                    RotorBuild::Materialized(r01),
                    RotorBuild::Materialized(r12),
                    RotorBuild::Materialized(r20),
                ) => (*r01, *r12, *r20),
                _ => continue,
            };
            let l_raw = mul_even128(&mul_even128(&r20, &r12), &r01);
            let l = match normalize(&l_raw) {
                Ok(v) => v,
                Err(EvenError::NonFiniteNormSquared | EvenError::NonPositiveNormSquared) => {
                    continue
                }
            };
            loops_used += 1;
            let inn = inner(&l, &Even128::identity());
            let a = inn.abs().min(1.0);
            let d = (2.0 * (1.0 - a)).max(0.0).sqrt();
            h2_distances.push(d);
            h3_loop_ratios.push(higher_grade_energy_ratio(&l_raw));
        }
    }

    let h2_loop_stats = if h2_distances.is_empty() {
        None
    } else {
        Some(stat_triple(&h2_distances))
    };
    let h2_missing_reason = if h2_loop_stats.is_some() {
        None
    } else if n_ans < 3 {
        Some(Gate2MetricMissingReason::NotEnoughAnswerUnits)
    } else {
        Some(Gate2MetricMissingReason::NoUsableTriangleLoops)
    };

    let h3_ratio_total_product = h1_raw.as_ref().map(higher_grade_energy_ratio);
    let h3_total_missing_reason = if h3_ratio_total_product.is_some() {
        None
    } else {
        h1b_missing_reason
    };

    let h3_ratio_triangle_loop_stats = if h3_loop_ratios.is_empty() {
        None
    } else {
        Some(stat_triple(&h3_loop_ratios))
    };
    let h3_loop_missing_reason = if h3_ratio_triangle_loop_stats.is_some() {
        None
    } else {
        h2_missing_reason
    };

    Gate2TelemetryResult {
        ids,
        per_sample: Gate2SampleTelemetry {
            sample_id: input.sample_id,
            n_ans_units: n_ans,
            counts: Gate2TelemetryCounts {
                count_missing_even_rotor_steps: builder.missing_count,
                count_loops_considered: loops_considered,
                count_loops_used: loops_used,
            },
            sample_abort_reason: None,
            missing_even_rotor_reasons: builder.missing_reasons,
            h1b_closure_error,
            h1b_missing_reason,
            h2_loop_stats,
            h2_missing_reason,
            h3_ratio_total_product,
            h3_total_missing_reason,
            h3_ratio_triangle_loop_stats,
            h3_loop_missing_reason,
        },
    }
}

fn has_non_finite_ans_vec8(ans_vec8: &[[f64; ROOT_DIM]]) -> bool {
    ans_vec8
        .iter()
        .any(|vec8| vec8.iter().any(|value| !value.is_finite()))
}

fn left_fold_time_reversed_raw(rotors_time_order: &[Even128]) -> Option<Even128> {
    if rotors_time_order.is_empty() {
        return None;
    }
    let mut acc = rotors_time_order[rotors_time_order.len() - 1];
    for rotor in rotors_time_order[..rotors_time_order.len() - 1]
        .iter()
        .rev()
    {
        acc = mul_even128(&acc, rotor);
    }
    Some(acc)
}

fn projective_chordal_distance(left: &Even128, right: &Even128) -> f64 {
    let inn = inner(left, right);
    let a = inn.abs().min(1.0);
    (2.0 * (1.0 - a)).max(0.0).sqrt()
}

fn higher_grade_energy_ratio(raw: &Even128) -> f64 {
    let masks = even_masks_v1();
    let mut e_total = 0.0;
    let mut e_high = 0.0;
    for (idx, &coeff) in raw.coeffs.iter().enumerate() {
        let energy = coeff * coeff;
        e_total += energy;
        let grade = grade_of_mask(masks[idx]);
        if grade >= 4 {
            e_high += energy;
        }
    }
    if e_total > 0.0 {
        e_high / e_total
    } else {
        0.0
    }
}

fn stat_triple(values: &[f64]) -> Gate2StatTriple {
    debug_assert!(!values.is_empty());
    let max = values
        .iter()
        .copied()
        .max_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0);
    let mean = values.iter().sum::<f64>() / (values.len() as f64);
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    Gate2StatTriple {
        max,
        mean,
        p90: nearest_rank(&sorted, 0.90),
    }
}

fn nearest_rank(sorted: &[f64], p: f64) -> f64 {
    debug_assert!(!sorted.is_empty());
    let n = sorted.len();
    let p = p.clamp(0.0, 1.0);
    let idx_f = (p * (n as f64)).ceil() - 1.0;
    let idx = if idx_f.is_nan() || idx_f < 0.0 {
        0
    } else if idx_f >= (n as f64) {
        n - 1
    } else {
        idx_f as usize
    };
    sorted[idx]
}

#[cfg(test)]
mod tests {
    use super::*;
    use pale_ale_rotor::gate1_lex_bivector_index_for_pair;

    fn e(i: usize) -> [f64; ROOT_DIM] {
        let mut out = [0.0; ROOT_DIM];
        out[i] = 1.0;
        out
    }

    fn unit_sum(i: usize, j: usize) -> [f64; ROOT_DIM] {
        let mut out = [0.0; ROOT_DIM];
        let inv = (0.5_f64).sqrt();
        out[i] = inv;
        out[j] = inv;
        out
    }

    fn plane_rotor_simple29(i: usize, j: usize, theta: f64) -> [f64; 29] {
        let mut out = [0.0; 29];
        let idx = gate1_lex_bivector_index_for_pair(i, j).expect("pair");
        out[0] = (0.5 * theta).cos();
        out[1 + idx] = (0.5 * theta).sin();
        out
    }

    #[test]
    fn h3_total_ratio_is_zero_for_single_embedded_rotor_chain() {
        let input = Gate2TelemetryInput {
            sample_id: 1,
            ans_vec8: vec![e(0), e(1)],
            answer_length: None,
        };
        let out = compute_gate2_telemetry(&input);
        let ratio = out
            .per_sample
            .h3_ratio_total_product
            .expect("h3 total available");
        assert!((ratio - 0.0).abs() < 1e-12, "ratio={ratio}");
    }

    #[test]
    fn h3_total_ratio_is_positive_for_non_coplanar_composition() {
        let r01 = embed_simple29_to_even128(&plane_rotor_simple29(0, 1, 0.41));
        let r23 = embed_simple29_to_even128(&plane_rotor_simple29(2, 3, -0.33));
        let raw = mul_even128(&r23, &r01);
        let ratio = higher_grade_energy_ratio(&raw);
        assert!(ratio > 1e-8, "expected >0, got {ratio}");
    }

    #[test]
    fn h1b_closure_near_zero_for_backtracking_chain() {
        let input = Gate2TelemetryInput {
            sample_id: 3,
            ans_vec8: vec![e(0), e(1), e(0)],
            answer_length: None,
        };
        let out = compute_gate2_telemetry(&input);
        let d = out.per_sample.h1b_closure_error.expect("h1b available");
        assert!(d.abs() < 1e-10, "closure distance {d}");
    }

    #[test]
    fn h2_matches_direct_r20_loop_formula() {
        let input = Gate2TelemetryInput {
            sample_id: 4,
            ans_vec8: vec![e(0), e(1), e(2)],
            answer_length: None,
        };
        let out = compute_gate2_telemetry(&input);
        let stats = out.per_sample.h2_loop_stats.expect("h2 available");

        let r01 = match simple_rotor29_doc_to_ans(e(0), e(1), RotorConfig::default()).expect("r01")
        {
            RotorStep::Materialized { r29, .. } => embed_simple29_to_even128(&r29),
            RotorStep::AntipodalAngleOnly { .. } => panic!("unexpected antipodal"),
        };
        let r12 = match simple_rotor29_doc_to_ans(e(1), e(2), RotorConfig::default()).expect("r12")
        {
            RotorStep::Materialized { r29, .. } => embed_simple29_to_even128(&r29),
            RotorStep::AntipodalAngleOnly { .. } => panic!("unexpected antipodal"),
        };
        let r20 = match simple_rotor29_doc_to_ans(e(2), e(0), RotorConfig::default()).expect("r20")
        {
            RotorStep::Materialized { r29, .. } => embed_simple29_to_even128(&r29),
            RotorStep::AntipodalAngleOnly { .. } => panic!("unexpected antipodal"),
        };

        let l_raw = mul_even128(&mul_even128(&r20, &r12), &r01);
        let l = normalize(&l_raw).expect("normalize");
        let inn = inner(&l, &Even128::identity()).abs().min(1.0);
        let expected = (2.0 * (1.0 - inn)).max(0.0).sqrt();
        assert!((stats.max - expected).abs() < 1e-12);
        assert!((stats.mean - expected).abs() < 1e-12);
        assert!((stats.p90 - expected).abs() < 1e-12);
    }

    #[test]
    fn antipodal_transition_marks_missing_and_no_fabricated_even_rotor() {
        let mut neg_e0 = e(0);
        neg_e0[0] = -1.0;
        let input = Gate2TelemetryInput {
            sample_id: 5,
            ans_vec8: vec![e(0), neg_e0, e(1)],
            answer_length: None,
        };
        let out = compute_gate2_telemetry(&input);
        assert!(out.per_sample.h1b_closure_error.is_none());
        assert_eq!(
            out.per_sample.h1b_missing_reason,
            Some(Gate2MetricMissingReason::MissingEvenRotorStep)
        );
        assert!(out.per_sample.counts.count_missing_even_rotor_steps >= 1);
        assert!(out
            .per_sample
            .missing_even_rotor_reasons
            .iter()
            .any(|reason| matches!(reason, MissingEvenRotorReason::AntipodalAngleOnly)));
    }

    #[test]
    fn gate2_telemetry_is_deterministic_for_identical_input() {
        let input = Gate2TelemetryInput {
            sample_id: 6,
            ans_vec8: vec![e(0), unit_sum(1, 2), unit_sum(2, 3), e(3)],
            answer_length: Some(42),
        };
        let left = compute_gate2_telemetry(&input);
        let right = compute_gate2_telemetry(&input);
        assert_eq!(left, right);
    }
}
