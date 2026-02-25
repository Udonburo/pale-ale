use crate::linking::LinkSanityResult;
use crate::rotor_diagnostics::{DegeneratePathRateCounts, RotorDiagnosticsResult, TAU_WEDGE};
use serde::{Deserialize, Serialize};
use std::fmt;

pub const QUANTILE_ID: &str = "nearest_rank_total_cmp_v1";
pub const RANK_METHOD_ID: &str = "average_rank_total_cmp_v1";
pub const AUC_ALGORITHM_ID: &str = "mann_whitney_rank_sum_v1";

pub const QUANTILE_P01: f64 = 0.01;
pub const QUANTILE_P50: f64 = 0.50;
pub const QUANTILE_P90: f64 = 0.90;
pub const QUANTILE_P99: f64 = 0.99;

pub const COLLAPSE_RATE_COLLINEAR_THRESHOLD: f64 = 0.80;
pub const COLLAPSE_RATE_ANTIPODAL_DROP_THRESHOLD: f64 = 0.20;
pub const PRIMARY_EXCLUSION_CEILING: f64 = 0.10;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RunEvalInput {
    pub samples: Vec<RunEvalSample>,
    pub link_sanity: LinkSanityResult,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RunEvalSample {
    pub sample_id: u64,
    pub sample_label: Option<u8>,
    pub answer_length: Option<usize>,
    pub diagnostics: RotorDiagnosticsResult,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RunInvalidReason {
    LinkSanityFail,
    RandomLikeLinkCollapse,
    DominantLinkCollapse,
    ExcessExclusionsPrimary,
    EmptyQuantilePopulationPrimary,
    NoSupervisedEligibleSamples,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CollapseInvalidReason {
    RateCollinearExceeded,
    RateAntipodalDropExceeded,
    WedgeNormP99BelowTauWedge,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum AucUndefinedReason {
    SingleClassAfterExclusions,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Quantiles4 {
    pub p01: f64,
    pub p50: f64,
    pub p90: f64,
    pub p99: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CollapseGateStatus {
    pub collinear_exceeds_threshold: bool,
    pub antipodal_drop_exceeds_threshold: bool,
    pub wedge_norm_p99_below_tau_wedge: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RunEvalResult {
    pub run_valid: bool,
    pub run_invalid_reason: Option<RunInvalidReason>,
    pub collapse_invalid_reason: Option<CollapseInvalidReason>,
    pub quantile_reference_only: bool,

    pub quantile_id: String,
    pub rank_method_id: String,
    pub auc_algorithm_id: String,

    pub n_supervised_eligible: usize,
    pub n_supervised_used_primary: usize,
    pub n_supervised_excluded_primary: usize,
    pub primary_exclusion_rate: Option<f64>,
    pub label_missing_rate: f64,

    pub primary_auc: Option<f64>,
    pub auc_undefined_reason: Option<AucUndefinedReason>,
    pub primary_auc_n_pos: usize,
    pub primary_auc_n_neg: usize,

    pub top1_dot_population_size: usize,
    pub top1_wedge_population_size: usize,
    pub top1_dot_quantiles: Option<Quantiles4>,
    pub top1_wedge_norm_quantiles: Option<Quantiles4>,

    pub rate_collinear: f64,
    pub rate_antipodal_drop: f64,
    pub collapse_gate_status: CollapseGateStatus,

    pub degenerate_path_rate_counts: DegeneratePathRateCounts,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RunEvalError {
    SampleIdMismatch {
        sample_id: u64,
        diagnostics_sample_id: u64,
    },
    InvalidSampleLabel {
        sample_id: u64,
        label: u8,
    },
}

impl fmt::Display for RunEvalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SampleIdMismatch {
                sample_id,
                diagnostics_sample_id,
            } => write!(
                f,
                "sample_id mismatch: sample {}, diagnostics {}",
                sample_id, diagnostics_sample_id
            ),
            Self::InvalidSampleLabel { sample_id, label } => write!(
                f,
                "invalid sample label {} for sample {} (expected 0/1)",
                label, sample_id
            ),
        }
    }
}

impl std::error::Error for RunEvalError {}

pub fn compute_run_eval(input: &RunEvalInput) -> Result<RunEvalResult, RunEvalError> {
    let mut dot_population = Vec::new();
    let mut wedge_population = Vec::new();
    let mut count_collinear = 0usize;
    let mut count_antipodal_drop = 0usize;

    let mut n_label_missing = 0usize;
    let mut n_supervised_eligible = 0usize;
    let mut n_supervised_excluded_primary = 0usize;
    let mut primary_points = Vec::new();

    let mut degenerate_counts = DegeneratePathRateCounts {
        top1_numerator: 0,
        top1_denominator: 0,
        trimmed_numerator: 0,
        trimmed_denominator: 0,
    };

    for sample in &input.samples {
        if sample.sample_id != sample.diagnostics.sample_id {
            return Err(RunEvalError::SampleIdMismatch {
                sample_id: sample.sample_id,
                diagnostics_sample_id: sample.diagnostics.sample_id,
            });
        }

        for step in &sample.diagnostics.top1_gate_steps {
            // Defensive: dot should always be finite per Top1GateStep contract
            // (see rotor_diagnostics::compute_top1_track debug_assert).
            // Kept as a safety net against future plumbing errors.
            if !step.dot.is_finite() {
                continue;
            }
            dot_population.push(step.dot);
            if step.is_collinear {
                count_collinear += 1;
            }
            if step.is_antipodal_drop {
                count_antipodal_drop += 1;
            }
            if let Some(wedge_norm) = step.wedge_norm {
                if wedge_norm.is_finite() {
                    wedge_population.push(wedge_norm);
                }
            }
        }

        degenerate_counts.top1_numerator += sample
            .diagnostics
            .degenerate_path_rate_counts
            .top1_numerator;
        degenerate_counts.top1_denominator += sample
            .diagnostics
            .degenerate_path_rate_counts
            .top1_denominator;
        degenerate_counts.trimmed_numerator += sample
            .diagnostics
            .degenerate_path_rate_counts
            .trimmed_numerator;
        degenerate_counts.trimmed_denominator += sample
            .diagnostics
            .degenerate_path_rate_counts
            .trimmed_denominator;

        match sample.sample_label {
            None => {
                n_label_missing += 1;
            }
            Some(label) => {
                if label > 1 {
                    return Err(RunEvalError::InvalidSampleLabel {
                        sample_id: sample.sample_id,
                        label,
                    });
                }

                n_supervised_eligible += 1;
                let primary_score = sample.diagnostics.top1.max_theta.value;
                let is_primary_used = sample.diagnostics.top1.excluded_reason.is_none()
                    && primary_score.map(|v| v.is_finite()).unwrap_or(false);
                if is_primary_used {
                    primary_points.push((primary_score.expect("checked Some"), label == 1));
                } else {
                    n_supervised_excluded_primary += 1;
                }
            }
        }
    }

    let total_samples = input.samples.len();
    let n_supervised_used_primary =
        n_supervised_eligible.saturating_sub(n_supervised_excluded_primary);
    let primary_exclusion_rate = if n_supervised_eligible > 0 {
        Some(ratio(n_supervised_excluded_primary, n_supervised_eligible))
    } else {
        None
    };
    let label_missing_rate = ratio(n_label_missing, total_samples);

    let auc_stats = compute_auc_mann_whitney(&primary_points);

    let top1_dot_quantiles = quantiles_for_values(&dot_population);
    let top1_wedge_norm_quantiles = quantiles_for_values(&wedge_population);
    let rate_collinear = ratio(count_collinear, dot_population.len());
    let rate_antipodal_drop = ratio(count_antipodal_drop, dot_population.len());
    let wedge_norm_p99 = top1_wedge_norm_quantiles.as_ref().map(|q| q.p99);

    let collapse_gate_status = CollapseGateStatus {
        collinear_exceeds_threshold: rate_collinear > COLLAPSE_RATE_COLLINEAR_THRESHOLD,
        antipodal_drop_exceeds_threshold: rate_antipodal_drop
            > COLLAPSE_RATE_ANTIPODAL_DROP_THRESHOLD,
        wedge_norm_p99_below_tau_wedge: wedge_norm_p99.map(|v| v < TAU_WEDGE).unwrap_or(false),
    };

    let (mut run_valid, mut run_invalid_reason, quantile_reference_only) =
        link_sanity_invalidation(&input.link_sanity);
    let mut collapse_invalid_reason = None;

    if run_valid {
        if dot_population.is_empty() {
            run_valid = false;
            run_invalid_reason = Some(RunInvalidReason::EmptyQuantilePopulationPrimary);
        } else if collapse_gate_status.collinear_exceeds_threshold {
            run_valid = false;
            collapse_invalid_reason = Some(CollapseInvalidReason::RateCollinearExceeded);
        } else if collapse_gate_status.antipodal_drop_exceeds_threshold {
            run_valid = false;
            collapse_invalid_reason = Some(CollapseInvalidReason::RateAntipodalDropExceeded);
        } else if collapse_gate_status.wedge_norm_p99_below_tau_wedge {
            run_valid = false;
            collapse_invalid_reason = Some(CollapseInvalidReason::WedgeNormP99BelowTauWedge);
        }
    }

    if run_valid {
        if n_supervised_eligible == 0 {
            run_valid = false;
            run_invalid_reason = Some(RunInvalidReason::NoSupervisedEligibleSamples);
        } else if primary_exclusion_rate.unwrap_or(0.0) > PRIMARY_EXCLUSION_CEILING {
            run_valid = false;
            run_invalid_reason = Some(RunInvalidReason::ExcessExclusionsPrimary);
        }
    }

    Ok(RunEvalResult {
        run_valid,
        run_invalid_reason,
        collapse_invalid_reason,
        quantile_reference_only,

        quantile_id: QUANTILE_ID.to_string(),
        rank_method_id: RANK_METHOD_ID.to_string(),
        auc_algorithm_id: AUC_ALGORITHM_ID.to_string(),

        n_supervised_eligible,
        n_supervised_used_primary,
        n_supervised_excluded_primary,
        primary_exclusion_rate,
        label_missing_rate,

        primary_auc: auc_stats.auc,
        auc_undefined_reason: auc_stats.undefined_reason,
        primary_auc_n_pos: auc_stats.n_pos,
        primary_auc_n_neg: auc_stats.n_neg,

        top1_dot_population_size: dot_population.len(),
        top1_wedge_population_size: wedge_population.len(),
        top1_dot_quantiles,
        top1_wedge_norm_quantiles,

        rate_collinear,
        rate_antipodal_drop,
        collapse_gate_status,

        degenerate_path_rate_counts: degenerate_counts,
    })
}

fn link_sanity_invalidation(sanity: &LinkSanityResult) -> (bool, Option<RunInvalidReason>, bool) {
    if sanity.link_sanity_fail {
        return (false, Some(RunInvalidReason::LinkSanityFail), true);
    }
    if sanity.random_like_link_collapse {
        return (false, Some(RunInvalidReason::RandomLikeLinkCollapse), true);
    }
    if sanity.dominant_link_collapse {
        return (false, Some(RunInvalidReason::DominantLinkCollapse), true);
    }
    (true, None, false)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct AucStats {
    pub auc: Option<f64>,
    pub undefined_reason: Option<AucUndefinedReason>,
    pub n_pos: usize,
    pub n_neg: usize,
}

pub(crate) fn compute_auc_mann_whitney(points: &[(f64, bool)]) -> AucStats {
    let n_pos = points.iter().filter(|(_, is_pos)| *is_pos).count();
    let n_neg = points.len().saturating_sub(n_pos);
    if n_pos == 0 || n_neg == 0 {
        return AucStats {
            auc: None,
            undefined_reason: Some(AucUndefinedReason::SingleClassAfterExclusions),
            n_pos,
            n_neg,
        };
    }

    let mut ranked: Vec<(usize, f64, bool)> = points
        .iter()
        .enumerate()
        .map(|(idx, (score, is_pos))| (idx, *score, *is_pos))
        .collect();
    ranked.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut sum_rank_pos = 0.0_f64;
    let mut start = 0usize;
    while start < ranked.len() {
        let mut end = start + 1;
        while end < ranked.len() && ranked[end].1 == ranked[start].1 {
            end += 1;
        }
        let avg_rank = ((start + 1 + end) as f64) * 0.5;
        for (_, _, is_pos) in &ranked[start..end] {
            if *is_pos {
                sum_rank_pos += avg_rank;
            }
        }
        start = end;
    }

    let n_pos_f = n_pos as f64;
    let n_neg_f = n_neg as f64;
    let auc = (sum_rank_pos - (n_pos_f * (n_pos_f + 1.0) * 0.5)) / (n_pos_f * n_neg_f);
    AucStats {
        auc: Some(auc),
        undefined_reason: None,
        n_pos,
        n_neg,
    }
}

pub(crate) fn quantiles_for_values(values: &[f64]) -> Option<Quantiles4> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    Some(Quantiles4 {
        p01: nearest_rank(&sorted, QUANTILE_P01),
        p50: nearest_rank(&sorted, QUANTILE_P50),
        p90: nearest_rank(&sorted, QUANTILE_P90),
        p99: nearest_rank(&sorted, QUANTILE_P99),
    })
}

pub(crate) fn nearest_rank(sorted: &[f64], p: f64) -> f64 {
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

pub(crate) fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        (num as f64) / (den as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linking::{LinkSanityRecord, SanityCategory, SanityJudgment};
    use crate::rotor_diagnostics::{
        AlignmentMetric, ExcludedReason, MetricField, MetricMissingReason, PlaneTurnMetric,
        RateMetrics, Top1GateStep, TrackCounts, TrackDiagnostics, TrimmedStabilityDiagnostics,
        WanderingMetric, DISTANCE_ID, METHOD_ID, THETA_SOURCE_ID, TRIMMED_BEST_ID,
    };

    fn sanity_result(
        link_sanity_fail: bool,
        random_like_link_collapse: bool,
        dominant_link_collapse: bool,
    ) -> LinkSanityResult {
        LinkSanityResult {
            link_sanity_id: "sanity16_single_judgment_v1".to_string(),
            rng_id: "splitmix64_v1".to_string(),
            seed: 0,
            sampling_id: "hash_sort_without_replacement_v1".to_string(),
            selected_sample_ids: vec![],
            k_eff: 0,
            records: vec![LinkSanityRecord {
                sample_id: 1,
                representative_ans_unit_id: None,
                selected_doc_unit_id: None,
                category: SanityCategory::NoLink,
                judgment: SanityJudgment::Unreviewed,
            }],
            unrelated_count: 0,
            link_sanity_fail,
            h_norm: 0.0,
            max_share: 0.0,
            random_like_link_collapse,
            dominant_link_collapse,
        }
    }

    fn metric(value: Option<f64>, missing_reason: Option<MetricMissingReason>) -> MetricField<f64> {
        MetricField {
            value,
            metric_missing_reason: missing_reason,
        }
    }

    fn base_track(track_id: &str) -> TrackDiagnostics {
        TrackDiagnostics {
            track_id: track_id.to_string(),
            excluded_reason: None,
            max_theta: metric(Some(0.0), None),
            plane_turn: PlaneTurnMetric {
                mean: metric(Some(0.0), None),
                max: metric(Some(0.0), None),
                var: metric(Some(0.0), None),
            },
            alignment: AlignmentMetric {
                mean: metric(Some(0.0), None),
                var: metric(Some(0.0), None),
            },
            wandering: WanderingMetric {
                ratio: metric(Some(1.0), None),
                degenerate_path: Some(false),
                degenerate_path_rate_numerator: 0,
                degenerate_path_rate_denominator: 1,
            },
            rates: RateMetrics {
                rate_collinear: 0.0,
                rate_antipodal_angle_only: 0.0,
                rate_antipodal_drop: 0.0,
                rate_missing_link_steps: 0.0,
                rate_missing_top1_steps: 0.0,
                normalized_rate: 0.0,
            },
            counts: TrackCounts {
                steps_total: 1,
                vec8_total: 2,
                normalized_count: 0,
                max_norm_err: 0.0,
                count_collinear: 0,
                count_antipodal_angle_only: 0,
                count_antipodal_drop: 0,
                count_missing_link_steps: 0,
                count_missing_top1_steps: 0,
                n_theta_valid: 1,
                n_rotors_valid: 1,
                n_planes_valid: 1,
                missing_link_step_rate: 0.0,
                missing_top1_step_rate: 0.0,
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sample(
        sample_id: u64,
        sample_label: Option<u8>,
        answer_length: Option<usize>,
        top1_score: Option<f64>,
        top1_missing_reason: Option<MetricMissingReason>,
        top1_excluded_reason: Option<ExcludedReason>,
        top1_gate_steps: Vec<Top1GateStep>,
        degenerate_counts: DegeneratePathRateCounts,
    ) -> RunEvalSample {
        let mut top1 = base_track("top1");
        top1.max_theta = metric(top1_score, top1_missing_reason);
        top1.excluded_reason = top1_excluded_reason;

        let trimmed = base_track("trimmed_best");
        RunEvalSample {
            sample_id,
            sample_label,
            answer_length,
            diagnostics: RotorDiagnosticsResult {
                sample_id,
                method_id: METHOD_ID.to_string(),
                distance_id: DISTANCE_ID.to_string(),
                theta_source_id: THETA_SOURCE_ID.to_string(),
                trimmed_best_id: TRIMMED_BEST_ID.to_string(),
                top1,
                trimmed,
                top1_gate_steps,
                trimmed_rbar_norm_pre_values: Vec::new(),
                trimmed_stability: TrimmedStabilityDiagnostics {
                    trimmed_rbar_norm_pre_p50: None,
                    trimmed_rbar_norm_pre_p10: None,
                    trimmed_rbar_norm_pre_p01: None,
                    trimmed_failure_rate: 0.0,
                    trimmed_failure_steps: 0,
                    trimmed_attempted_steps: 0,
                },
                degenerate_path_rate_counts: degenerate_counts,
            },
        }
    }

    fn gate_step(
        dot: f64,
        wedge_norm: Option<f64>,
        is_collinear: bool,
        is_antipodal_drop: bool,
    ) -> Top1GateStep {
        Top1GateStep {
            ans_unit_id: 0,
            doc_unit_id: 0,
            dot,
            wedge_norm,
            is_collinear,
            is_antipodal_angle_only: false,
            is_antipodal_drop,
        }
    }

    fn approx_eq(left: f64, right: f64) {
        let diff = (left - right).abs();
        assert!(
            diff < 1e-12,
            "left={}, right={}, diff={}",
            left,
            right,
            diff
        );
    }

    #[test]
    fn auc_no_ties_matches_hand_calculation() {
        let input = RunEvalInput {
            samples: vec![
                sample(
                    1,
                    Some(0),
                    Some(10),
                    Some(0.1),
                    None,
                    None,
                    vec![gate_step(0.1, Some(0.4), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    2,
                    Some(1),
                    Some(20),
                    Some(0.4),
                    None,
                    None,
                    vec![gate_step(0.2, Some(0.5), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    3,
                    Some(0),
                    Some(30),
                    Some(0.5),
                    None,
                    None,
                    vec![gate_step(0.3, Some(0.6), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    4,
                    Some(1),
                    Some(40),
                    Some(0.8),
                    None,
                    None,
                    vec![gate_step(0.4, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
            ],
            link_sanity: sanity_result(false, false, false),
        };

        let out = compute_run_eval(&input).expect("run eval");
        approx_eq(out.primary_auc.expect("auc"), 0.75);
        assert_eq!(out.auc_undefined_reason, None);
    }

    #[test]
    fn auc_ties_use_average_rank() {
        let stats = compute_auc_mann_whitney(&[(0.5, true), (0.5, false), (0.1, false)]);
        approx_eq(stats.auc.expect("auc"), 0.75);
        assert_eq!(stats.undefined_reason, None);
        assert_eq!(stats.n_pos, 1);
        assert_eq!(stats.n_neg, 2);
    }

    #[test]
    fn auc_single_class_after_exclusions_is_undefined() {
        let input = RunEvalInput {
            samples: vec![
                sample(
                    1,
                    Some(1),
                    Some(10),
                    Some(0.9),
                    None,
                    None,
                    vec![gate_step(0.1, Some(0.4), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    2,
                    Some(0),
                    Some(20),
                    None,
                    Some(MetricMissingReason::MissingTheta),
                    None,
                    vec![gate_step(0.2, Some(0.5), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
            ],
            link_sanity: sanity_result(false, false, false),
        };

        let out = compute_run_eval(&input).expect("run eval");
        assert_eq!(out.primary_auc, None);
        assert_eq!(
            out.auc_undefined_reason,
            Some(AucUndefinedReason::SingleClassAfterExclusions)
        );
    }

    #[test]
    fn auc_label_flip_is_one_minus_auc() {
        let base =
            compute_auc_mann_whitney(&[(0.1, false), (0.4, true), (0.5, false), (0.8, true)]);
        let flipped =
            compute_auc_mann_whitney(&[(0.1, true), (0.4, false), (0.5, true), (0.8, false)]);
        let auc = base.auc.expect("auc");
        let auc_flipped = flipped.auc.expect("auc");
        approx_eq(auc_flipped, 1.0 - auc);
    }

    #[test]
    fn quantiles_follow_nearest_rank_contract() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let q = quantiles_for_values(&values).expect("quantiles");
        approx_eq(q.p01, 1.0);
        approx_eq(q.p50, 5.0);
        approx_eq(q.p90, 9.0);
        approx_eq(q.p99, 10.0);
    }

    #[test]
    fn empty_dot_population_invalidates_run() {
        let input = RunEvalInput {
            samples: vec![sample(
                1,
                Some(1),
                Some(10),
                Some(0.8),
                None,
                None,
                vec![],
                DegeneratePathRateCounts {
                    top1_numerator: 0,
                    top1_denominator: 0,
                    trimmed_numerator: 0,
                    trimmed_denominator: 0,
                },
            )],
            link_sanity: sanity_result(false, false, false),
        };

        let out = compute_run_eval(&input).expect("run eval");
        assert!(!out.run_valid);
        assert_eq!(
            out.run_invalid_reason,
            Some(RunInvalidReason::EmptyQuantilePopulationPrimary)
        );
        assert!(!out.quantile_reference_only);
    }

    #[test]
    fn link_sanity_fail_has_priority_and_sets_reference_only() {
        let input = RunEvalInput {
            samples: vec![sample(
                1,
                Some(1),
                Some(10),
                Some(0.8),
                None,
                None,
                vec![gate_step(0.1, Some(0.0), true, true)],
                DegeneratePathRateCounts {
                    top1_numerator: 0,
                    top1_denominator: 1,
                    trimmed_numerator: 0,
                    trimmed_denominator: 1,
                },
            )],
            link_sanity: sanity_result(true, false, false),
        };

        let out = compute_run_eval(&input).expect("run eval");
        assert!(!out.run_valid);
        assert_eq!(
            out.run_invalid_reason,
            Some(RunInvalidReason::LinkSanityFail)
        );
        assert!(out.quantile_reference_only);
    }

    #[test]
    fn collapse_gate_triggers_only_when_link_sanity_passes() {
        let input = RunEvalInput {
            samples: vec![sample(
                1,
                Some(1),
                Some(10),
                Some(0.8),
                None,
                None,
                vec![
                    gate_step(0.1, Some(0.3), true, false),
                    gate_step(0.2, Some(0.3), true, false),
                    gate_step(0.3, Some(0.3), true, false),
                    gate_step(0.4, Some(0.3), true, false),
                    gate_step(0.5, Some(0.3), true, false),
                ],
                DegeneratePathRateCounts {
                    top1_numerator: 0,
                    top1_denominator: 1,
                    trimmed_numerator: 0,
                    trimmed_denominator: 1,
                },
            )],
            link_sanity: sanity_result(false, false, false),
        };

        let out = compute_run_eval(&input).expect("run eval");
        assert!(!out.run_valid);
        assert_eq!(out.run_invalid_reason, None);
        assert_eq!(
            out.collapse_invalid_reason,
            Some(CollapseInvalidReason::RateCollinearExceeded)
        );
        assert!(!out.quantile_reference_only);
    }

    #[test]
    fn exclusion_ceiling_applies_after_prior_gates_pass() {
        let input = RunEvalInput {
            samples: vec![
                sample(
                    1,
                    Some(1),
                    Some(10),
                    Some(0.8),
                    None,
                    None,
                    vec![gate_step(0.1, Some(0.6), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    2,
                    Some(0),
                    Some(20),
                    None,
                    Some(MetricMissingReason::MissingTheta),
                    None,
                    vec![gate_step(0.2, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
            ],
            link_sanity: sanity_result(false, false, false),
        };

        let out = compute_run_eval(&input).expect("run eval");
        assert!(!out.run_valid);
        assert_eq!(
            out.run_invalid_reason,
            Some(RunInvalidReason::ExcessExclusionsPrimary)
        );
    }

    #[test]
    fn supervised_accounting_counts_match_contract() {
        let input = RunEvalInput {
            samples: vec![
                sample(
                    1,
                    Some(1),
                    Some(10),
                    Some(0.9),
                    None,
                    None,
                    vec![gate_step(0.1, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    2,
                    Some(0),
                    Some(20),
                    Some(0.2),
                    None,
                    None,
                    vec![gate_step(0.2, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    3,
                    Some(1),
                    Some(30),
                    None,
                    Some(MetricMissingReason::MissingTheta),
                    None,
                    vec![gate_step(0.3, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    4,
                    Some(0),
                    Some(40),
                    Some(0.1),
                    None,
                    Some(ExcludedReason::RotorRenormFailure),
                    vec![gate_step(0.4, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    5,
                    None,
                    None,
                    Some(0.5),
                    None,
                    None,
                    vec![gate_step(0.5, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
            ],
            link_sanity: sanity_result(false, false, false),
        };

        let out = compute_run_eval(&input).expect("run eval");
        assert_eq!(out.n_supervised_eligible, 4);
        assert_eq!(out.n_supervised_excluded_primary, 2);
        assert_eq!(out.n_supervised_used_primary, 2);
        approx_eq(out.primary_exclusion_rate.expect("rate"), 0.5);
        approx_eq(out.label_missing_rate, 0.2);
    }

    #[test]
    fn degenerate_path_counts_use_track_specific_numerators_and_denominators() {
        let input = RunEvalInput {
            samples: vec![
                sample(
                    1,
                    Some(1),
                    Some(10),
                    Some(0.8),
                    None,
                    None,
                    vec![gate_step(0.1, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 1,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    2,
                    Some(0),
                    Some(20),
                    Some(0.2),
                    None,
                    None,
                    vec![gate_step(0.2, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 1,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    3,
                    Some(1),
                    Some(30),
                    Some(0.4),
                    None,
                    None,
                    vec![gate_step(0.3, Some(0.7), false, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 0,
                        trimmed_numerator: 0,
                        trimmed_denominator: 0,
                    },
                ),
            ],
            link_sanity: sanity_result(false, false, false),
        };

        let out = compute_run_eval(&input).expect("run eval");
        assert_eq!(out.degenerate_path_rate_counts.top1_numerator, 1);
        assert_eq!(out.degenerate_path_rate_counts.top1_denominator, 2);
        assert_eq!(out.degenerate_path_rate_counts.trimmed_numerator, 1);
        assert_eq!(out.degenerate_path_rate_counts.trimmed_denominator, 2);
    }

    #[test]
    fn collapse_rates_are_computed_over_d1_population_size() {
        let input = RunEvalInput {
            samples: vec![
                sample(
                    1,
                    Some(1),
                    Some(10),
                    Some(0.8),
                    None,
                    None,
                    vec![gate_step(0.1, Some(0.7), true, false)],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
                sample(
                    2,
                    Some(0),
                    Some(20),
                    Some(0.2),
                    None,
                    None,
                    vec![
                        gate_step(0.2, Some(0.7), false, false),
                        gate_step(0.3, Some(0.7), false, true),
                        gate_step(0.4, Some(0.7), false, false),
                    ],
                    DegeneratePathRateCounts {
                        top1_numerator: 0,
                        top1_denominator: 1,
                        trimmed_numerator: 0,
                        trimmed_denominator: 1,
                    },
                ),
            ],
            link_sanity: sanity_result(false, false, false),
        };

        let out = compute_run_eval(&input).expect("run eval");
        assert_eq!(out.top1_dot_population_size, 4);
        approx_eq(out.rate_collinear, 0.25);
        approx_eq(out.rate_antipodal_drop, 0.25);
    }
}
