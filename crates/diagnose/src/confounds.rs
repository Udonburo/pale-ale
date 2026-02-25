use crate::rotor_diagnostics::ExcludedReason;
use crate::run_eval::{compute_auc_mann_whitney, quantiles_for_values, ratio, RunEvalSample};
use serde::{Deserialize, Serialize};

pub const RUN_WARNING_ANTIPODAL_ANGLE_ONLY_HIGH: &str = "antipodal_angle_only_high";

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ConfoundStatus {
    Ok,
    Unavailable,
    InsufficientData,
}

impl ConfoundStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Unavailable => "unavailable",
            Self::InsufficientData => "insufficient_data",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ConfoundOutputs {
    pub confound_status: ConfoundStatus,
    pub rho_len_max_theta: Option<f64>,
    pub auc_len_tertile_short: Option<f64>,
    pub auc_len_tertile_medium: Option<f64>,
    pub auc_len_tertile_long: Option<f64>,
    pub n_len_tertile_short: usize,
    pub n_len_tertile_medium: usize,
    pub n_len_tertile_long: usize,
    pub exclusion_rate_short: Option<f64>,
    pub exclusion_rate_medium: Option<f64>,
    pub exclusion_rate_long: Option<f64>,
    pub length_confound_warning: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AntipodalWarningOutputs {
    pub rate_antipodal_angle_only_p50: f64,
    pub rate_antipodal_angle_only_p90: f64,
    pub share_samples_antipodal_angle_only_gt_0_50: f64,
    pub run_warning: Option<String>,
}

#[derive(Clone, Debug)]
struct PrimaryRow {
    sample_id: u64,
    answer_length: usize,
    label_is_pos: bool,
    is_primary_used: bool,
    primary_score: Option<f64>,
}

pub fn compute_confounds(samples: &[RunEvalSample]) -> ConfoundOutputs {
    let mut rows = Vec::new();
    for sample in samples {
        let label = match sample.sample_label {
            Some(value) if value <= 1 => value == 1,
            _ => continue,
        };
        let answer_length = match sample.answer_length {
            Some(value) => value,
            None => continue,
        };
        let primary_score = sample.diagnostics.top1.max_theta.value;
        let is_primary_used = sample.diagnostics.top1.excluded_reason.is_none()
            && primary_score.map(|v| v.is_finite()).unwrap_or(false);
        rows.push(PrimaryRow {
            sample_id: sample.sample_id,
            answer_length,
            label_is_pos: label,
            is_primary_used,
            primary_score: if is_primary_used { primary_score } else { None },
        });
    }

    if rows.is_empty() {
        return ConfoundOutputs {
            confound_status: ConfoundStatus::Unavailable,
            rho_len_max_theta: None,
            auc_len_tertile_short: None,
            auc_len_tertile_medium: None,
            auc_len_tertile_long: None,
            n_len_tertile_short: 0,
            n_len_tertile_medium: 0,
            n_len_tertile_long: 0,
            exclusion_rate_short: None,
            exclusion_rate_medium: None,
            exclusion_rate_long: None,
            length_confound_warning: false,
        };
    }

    rows.sort_by(|left, right| {
        left.answer_length
            .cmp(&right.answer_length)
            .then_with(|| left.sample_id.cmp(&right.sample_id))
    });

    let (short_rows, medium_rows, long_rows) = split_into_tertiles(&rows);
    let short = compute_tertile_stats(short_rows);
    let medium = compute_tertile_stats(medium_rows);
    let long = compute_tertile_stats(long_rows);

    let rho_rows: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|row| {
            row.primary_score
                .map(|score| (row.answer_length as f64, score))
        })
        .collect();

    let rho = spearman_rho(&rho_rows);
    let confound_status = if rho_rows.len() < 2 || rho.is_none() {
        ConfoundStatus::InsufficientData
    } else {
        ConfoundStatus::Ok
    };
    let length_confound_warning =
        confound_status == ConfoundStatus::Ok && rho.map(|v| v.abs() > 0.70).unwrap_or(false);

    ConfoundOutputs {
        confound_status,
        rho_len_max_theta: if confound_status == ConfoundStatus::Ok {
            rho
        } else {
            None
        },
        auc_len_tertile_short: short.auc,
        auc_len_tertile_medium: medium.auc,
        auc_len_tertile_long: long.auc,
        n_len_tertile_short: short.n_rows,
        n_len_tertile_medium: medium.n_rows,
        n_len_tertile_long: long.n_rows,
        exclusion_rate_short: Some(short.exclusion_rate),
        exclusion_rate_medium: Some(medium.exclusion_rate),
        exclusion_rate_long: Some(long.exclusion_rate),
        length_confound_warning,
    }
}

pub fn compute_antipodal_warning(samples: &[RunEvalSample]) -> AntipodalWarningOutputs {
    let mut per_sample_rates = Vec::new();
    for sample in samples {
        if sample.sample_label.is_none() {
            continue;
        }
        if sample.diagnostics.top1.counts.steps_total == 0 {
            continue;
        }
        if is_aborted(sample.diagnostics.top1.excluded_reason) {
            continue;
        }
        let rate = ratio(
            sample.diagnostics.top1.counts.count_antipodal_angle_only,
            sample.diagnostics.top1.counts.steps_total,
        );
        per_sample_rates.push(rate);
    }

    let quantiles = quantiles_for_values(&per_sample_rates);
    let rate_antipodal_angle_only_p50 = quantiles.as_ref().map(|q| q.p50).unwrap_or(0.0);
    let rate_antipodal_angle_only_p90 = quantiles.as_ref().map(|q| q.p90).unwrap_or(0.0);
    let share_samples_antipodal_angle_only_gt_0_50 = ratio(
        per_sample_rates.iter().filter(|rate| **rate > 0.50).count(),
        per_sample_rates.len(),
    );

    let run_warning = if share_samples_antipodal_angle_only_gt_0_50 > 0.20 {
        Some(RUN_WARNING_ANTIPODAL_ANGLE_ONLY_HIGH.to_string())
    } else {
        None
    };

    AntipodalWarningOutputs {
        rate_antipodal_angle_only_p50,
        rate_antipodal_angle_only_p90,
        share_samples_antipodal_angle_only_gt_0_50,
        run_warning,
    }
}

fn is_aborted(reason: Option<ExcludedReason>) -> bool {
    reason.is_some()
}

fn split_into_tertiles(rows: &[PrimaryRow]) -> (&[PrimaryRow], &[PrimaryRow], &[PrimaryRow]) {
    let n = rows.len();
    let cut1 = ((n as f64) / 3.0).ceil() as usize;
    let cut2 = ((2.0 * n as f64) / 3.0).ceil() as usize;
    let cut1 = cut1.min(n);
    let cut2 = cut2.min(n).max(cut1);
    (&rows[..cut1], &rows[cut1..cut2], &rows[cut2..])
}

struct TertileStats {
    auc: Option<f64>,
    n_rows: usize,
    exclusion_rate: f64,
}

fn compute_tertile_stats(rows: &[PrimaryRow]) -> TertileStats {
    let points: Vec<(f64, bool)> = rows
        .iter()
        .filter_map(|row| row.primary_score.map(|score| (score, row.label_is_pos)))
        .collect();
    let auc = compute_auc_mann_whitney(&points).auc;
    let excluded = rows.iter().filter(|row| !row.is_primary_used).count();
    TertileStats {
        auc,
        n_rows: rows.len(),
        exclusion_rate: ratio(excluded, rows.len()),
    }
}

fn spearman_rho(rows: &[(f64, f64)]) -> Option<f64> {
    if rows.len() < 2 {
        return None;
    }

    let xs: Vec<f64> = rows.iter().map(|(x, _)| *x).collect();
    let ys: Vec<f64> = rows.iter().map(|(_, y)| *y).collect();
    let rx = average_ranks(&xs);
    let ry = average_ranks(&ys);
    pearson(&rx, &ry)
}

fn average_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx, *value))
        .collect();
    indexed.sort_by(|left, right| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    });

    let mut out = vec![0.0_f64; values.len()];
    let mut start = 0usize;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == indexed[start].1 {
            end += 1;
        }
        let avg_rank = ((start + 1 + end) as f64) * 0.5;
        for i in start..end {
            out[indexed[i].0] = avg_rank;
        }
        start = end;
    }
    out
}

fn pearson(xs: &[f64], ys: &[f64]) -> Option<f64> {
    if xs.len() != ys.len() || xs.len() < 2 {
        return None;
    }
    let n = xs.len() as f64;
    let mean_x = xs.iter().sum::<f64>() / n;
    let mean_y = ys.iter().sum::<f64>() / n;

    let mut num = 0.0_f64;
    let mut den_x = 0.0_f64;
    let mut den_y = 0.0_f64;
    for idx in 0..xs.len() {
        let dx = xs[idx] - mean_x;
        let dy = ys[idx] - mean_y;
        num += dx * dy;
        den_x += dx * dx;
        den_y += dy * dy;
    }
    if den_x == 0.0 || den_y == 0.0 {
        return None;
    }
    Some(num / (den_x.sqrt() * den_y.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotor_diagnostics::{
        AlignmentMetric, MetricField, MetricMissingReason, PlaneTurnMetric, RateMetrics,
        RotorDiagnosticsResult, Top1GateStep, TrackCounts, TrackDiagnostics,
        TrimmedStabilityDiagnostics, WanderingMetric, DISTANCE_ID, METHOD_ID, THETA_SOURCE_ID,
        TRIMMED_BEST_ID,
    };
    use crate::{DegeneratePathRateCounts, RunEvalSample};

    fn metric(value: Option<f64>, reason: Option<MetricMissingReason>) -> MetricField<f64> {
        MetricField {
            value,
            metric_missing_reason: reason,
        }
    }

    fn base_track() -> TrackDiagnostics {
        TrackDiagnostics {
            track_id: "top1".to_string(),
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

    fn sample(
        sample_id: u64,
        sample_label: Option<u8>,
        answer_length: Option<usize>,
        score: Option<f64>,
        score_missing_reason: Option<MetricMissingReason>,
    ) -> RunEvalSample {
        let mut top1 = base_track();
        top1.max_theta = metric(score, score_missing_reason);
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
                trimmed: base_track(),
                top1_gate_steps: Vec::<Top1GateStep>::new(),
                trimmed_rbar_norm_pre_values: Vec::new(),
                trimmed_stability: TrimmedStabilityDiagnostics {
                    trimmed_rbar_norm_pre_p50: None,
                    trimmed_rbar_norm_pre_p10: None,
                    trimmed_rbar_norm_pre_p01: None,
                    trimmed_failure_rate: 0.0,
                    trimmed_failure_steps: 0,
                    trimmed_attempted_steps: 0,
                },
                degenerate_path_rate_counts: DegeneratePathRateCounts {
                    top1_numerator: 0,
                    top1_denominator: 1,
                    trimmed_numerator: 0,
                    trimmed_denominator: 1,
                },
            },
        }
    }

    #[test]
    fn confounds_unavailable_when_answer_length_missing() {
        let samples = vec![sample(1, Some(1), None, Some(0.8), None)];
        let out = compute_confounds(&samples);
        assert_eq!(out.confound_status, ConfoundStatus::Unavailable);
        assert!(!out.length_confound_warning);
    }

    #[test]
    fn confounds_compute_rho_and_tertiles() {
        let samples = vec![
            sample(1, Some(0), Some(10), Some(0.1), None),
            sample(2, Some(1), Some(20), Some(0.3), None),
            sample(3, Some(1), Some(30), Some(0.8), None),
        ];
        let out = compute_confounds(&samples);
        assert_eq!(out.confound_status, ConfoundStatus::Ok);
        assert!(out.rho_len_max_theta.expect("rho").is_finite());
        assert_eq!(
            out.n_len_tertile_short + out.n_len_tertile_medium + out.n_len_tertile_long,
            3
        );
    }

    #[test]
    fn confounds_insufficient_data_when_no_valid_score_rows() {
        let samples = vec![sample(
            1,
            Some(1),
            Some(10),
            None,
            Some(MetricMissingReason::MissingTheta),
        )];
        let out = compute_confounds(&samples);
        assert_eq!(out.confound_status, ConfoundStatus::InsufficientData);
        assert_eq!(out.rho_len_max_theta, None);
    }
}
