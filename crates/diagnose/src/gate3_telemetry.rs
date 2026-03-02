use crate::metrics_common::{higher_grade_energy_ratio, projective_chordal_distance};
use pale_ale_rotor::{
    embed_simple29_to_even128, mul_even128, normalize_vec8, reverse, simple_rotor29_doc_to_ans,
    Even128, RotorConfig, RotorError, RotorStep, ROOT_DIM,
};

const EPS_RATIO: f64 = 1e-12;

#[derive(Clone, Debug, PartialEq)]
pub struct Gate3TelemetryInput {
    pub sample_id: u64,
    pub ans_vec8: Vec<Vec<f64>>,
    pub sample_label: Option<u8>,
    pub answer_length: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gate3MissingReason {
    TooFewSteps,
    InvalidVec8,
    AllStepsMissing,
    InsufficientAdjacentRotors,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate3SampleTelemetry {
    pub sample_id: u64,
    pub sample_label: Option<u8>,
    pub answer_length: Option<usize>,
    pub count_steps_total: usize,
    pub count_rotors_total: usize,
    pub count_rotors_valid: usize,
    pub count_missing_even_rotor_steps: usize,
    pub kappa_count: usize,
    pub tau_count: usize,
    pub l3_kappa_max: f64,
    pub l3_kappa_mean: f64,
    pub l3_kappa_std: f64,
    pub l3_kappa_ratio: f64,
    pub l4_tau_max: f64,
    pub l4_tau_mean: f64,
    pub l4_tau_std: f64,
    pub l4_tau_p90: f64,
    pub missing_reason: Option<Gate3MissingReason>,
}

#[derive(Clone, Debug, PartialEq)]
enum StepRotor {
    Materialized(Box<Even128>),
    Missing,
}

pub fn compute_gate3_telemetry(input: &Gate3TelemetryInput) -> Gate3SampleTelemetry {
    let count_steps_total = input.ans_vec8.len();
    let count_rotors_total = count_steps_total.saturating_sub(1);

    let ans_rows = match validate_and_convert_ans_vec8(&input.ans_vec8) {
        Ok(rows) => rows,
        Err(reason) => {
            return missing_telemetry(
                input,
                count_steps_total,
                count_rotors_total,
                0,
                0,
                0,
                reason,
            );
        }
    };

    if count_steps_total < 4 {
        return missing_telemetry(
            input,
            count_steps_total,
            count_rotors_total,
            0,
            0,
            0,
            Gate3MissingReason::TooFewSteps,
        );
    }

    let mut step_rotors = Vec::with_capacity(count_rotors_total);
    let mut count_missing_even_rotor_steps = 0usize;
    for idx in 0..count_rotors_total {
        let rotor = build_step_rotor(&ans_rows[idx], &ans_rows[idx + 1]);
        if matches!(rotor, StepRotor::Missing) {
            count_missing_even_rotor_steps += 1;
        }
        step_rotors.push(rotor);
    }
    let count_rotors_valid = step_rotors
        .iter()
        .filter(|rotor| matches!(rotor, StepRotor::Materialized(_)))
        .count();
    if count_rotors_valid == 0 {
        return missing_telemetry(
            input,
            count_steps_total,
            count_rotors_total,
            count_rotors_valid,
            count_missing_even_rotor_steps,
            0,
            Gate3MissingReason::AllStepsMissing,
        );
    }

    let mut kappa_values = Vec::new();
    let mut tau_values = Vec::new();
    if count_rotors_total >= 2 {
        for idx in 0..(count_rotors_total - 1) {
            let (left, right) = match (&step_rotors[idx], &step_rotors[idx + 1]) {
                (StepRotor::Materialized(left), StepRotor::Materialized(right)) => {
                    (left.as_ref(), right.as_ref())
                }
                _ => continue,
            };
            let kappa = projective_chordal_distance(left, right);
            let relative = mul_even128(right, &reverse(left));
            let tau = higher_grade_energy_ratio(&relative);
            kappa_values.push(kappa);
            tau_values.push(tau);
        }
    }

    let kappa_count = kappa_values.len();
    let tau_count = tau_values.len();
    if kappa_count == 0 || tau_count == 0 {
        return missing_telemetry(
            input,
            count_steps_total,
            count_rotors_total,
            count_rotors_valid,
            count_missing_even_rotor_steps,
            kappa_count,
            Gate3MissingReason::InsufficientAdjacentRotors,
        );
    }

    let l3_kappa_max = max_value(&kappa_values);
    let l3_kappa_mean = mean_value(&kappa_values);
    let l3_kappa_std = std_population(&kappa_values, l3_kappa_mean);
    let l3_kappa_ratio = l3_kappa_max / l3_kappa_mean.max(EPS_RATIO);

    let l4_tau_max = max_value(&tau_values);
    let l4_tau_mean = mean_value(&tau_values);
    let l4_tau_std = std_population(&tau_values, l4_tau_mean);
    let l4_tau_p90 = percentile_nearest_rank(&tau_values, 0.90);

    Gate3SampleTelemetry {
        sample_id: input.sample_id,
        sample_label: input.sample_label,
        answer_length: input.answer_length,
        count_steps_total,
        count_rotors_total,
        count_rotors_valid,
        count_missing_even_rotor_steps,
        kappa_count,
        tau_count,
        l3_kappa_max,
        l3_kappa_mean,
        l3_kappa_std,
        l3_kappa_ratio,
        l4_tau_max,
        l4_tau_mean,
        l4_tau_std,
        l4_tau_p90,
        missing_reason: None,
    }
}

fn validate_and_convert_ans_vec8(
    rows: &[Vec<f64>],
) -> Result<Vec<[f64; ROOT_DIM]>, Gate3MissingReason> {
    let mut converted = Vec::with_capacity(rows.len());
    for row in rows {
        if row.len() != ROOT_DIM {
            return Err(Gate3MissingReason::InvalidVec8);
        }
        let mut out = [0.0; ROOT_DIM];
        for (idx, value) in row.iter().enumerate() {
            if !value.is_finite() {
                return Err(Gate3MissingReason::InvalidVec8);
            }
            out[idx] = *value;
        }
        converted.push(out);
    }
    Ok(converted)
}

fn build_step_rotor(from: &[f64; ROOT_DIM], to: &[f64; ROOT_DIM]) -> StepRotor {
    let from_norm = match normalize_vec8(*from) {
        Ok(v) => v,
        Err(_) => return StepRotor::Missing,
    };
    let to_norm = match normalize_vec8(*to) {
        Ok(v) => v,
        Err(_) => return StepRotor::Missing,
    };
    match simple_rotor29_doc_to_ans(from_norm, to_norm, RotorConfig::default()) {
        Ok(RotorStep::Materialized { r29, .. }) => {
            StepRotor::Materialized(Box::new(embed_simple29_to_even128(&r29)))
        }
        Ok(RotorStep::AntipodalAngleOnly { .. }) => StepRotor::Missing,
        Err(RotorError::Vec8(_) | RotorError::NonFiniteTheta | RotorError::RenormFailure) => {
            StepRotor::Missing
        }
    }
}

fn missing_telemetry(
    input: &Gate3TelemetryInput,
    count_steps_total: usize,
    count_rotors_total: usize,
    count_rotors_valid: usize,
    count_missing_even_rotor_steps: usize,
    kappa_count: usize,
    reason: Gate3MissingReason,
) -> Gate3SampleTelemetry {
    Gate3SampleTelemetry {
        sample_id: input.sample_id,
        sample_label: input.sample_label,
        answer_length: input.answer_length,
        count_steps_total,
        count_rotors_total,
        count_rotors_valid,
        count_missing_even_rotor_steps,
        kappa_count,
        tau_count: kappa_count,
        l3_kappa_max: 0.0,
        l3_kappa_mean: 0.0,
        l3_kappa_std: 0.0,
        l3_kappa_ratio: 0.0,
        l4_tau_max: 0.0,
        l4_tau_mean: 0.0,
        l4_tau_std: 0.0,
        l4_tau_p90: 0.0,
        missing_reason: Some(reason),
    }
}

fn max_value(values: &[f64]) -> f64 {
    values
        .iter()
        .copied()
        .max_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0)
}

fn mean_value(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / (values.len() as f64)
}

fn std_population(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = *value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (values.len() as f64);
    variance.sqrt()
}

fn percentile_nearest_rank(values: &[f64], p: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
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
    use std::f64::consts::PI;

    fn e(i: usize) -> Vec<f64> {
        let mut out = vec![0.0; ROOT_DIM];
        out[i] = 1.0;
        out
    }

    fn normalize_vec(mut row: Vec<f64>) -> Vec<f64> {
        let mut norm_sq = 0.0;
        for value in &row {
            norm_sq += value * value;
        }
        let norm = norm_sq.sqrt();
        if norm > 0.0 {
            for value in &mut row {
                *value /= norm;
            }
        }
        row
    }

    fn smooth_trajectory(steps: usize) -> Vec<Vec<f64>> {
        let mut out = Vec::with_capacity(steps);
        for idx in 0..steps {
            let angle = (2.0 * PI * (idx as f64)) / (steps as f64);
            let mut row = vec![0.0; ROOT_DIM];
            row[0] = angle.cos();
            row[1] = angle.sin();
            for (dim, value) in row.iter_mut().enumerate().skip(2) {
                *value = 0.01 * (0.17 * ((idx + 1) as f64) * ((dim + 1) as f64)).sin();
            }
            out.push(normalize_vec(row));
        }
        out
    }

    fn kink_trajectory(steps: usize, kink_at: usize) -> Vec<Vec<f64>> {
        let mut out = smooth_trajectory(steps);
        for value in &mut out[kink_at] {
            *value = -*value;
        }
        out
    }

    fn assert_f64_bits_eq(label: &str, left: f64, right: f64) {
        assert_eq!(left.to_bits(), right.to_bits(), "{label}");
    }

    fn assert_bits_equal(left: &Gate3SampleTelemetry, right: &Gate3SampleTelemetry) {
        assert_eq!(left.sample_id, right.sample_id);
        assert_eq!(left.sample_label, right.sample_label);
        assert_eq!(left.answer_length, right.answer_length);
        assert_eq!(left.count_steps_total, right.count_steps_total);
        assert_eq!(left.count_rotors_total, right.count_rotors_total);
        assert_eq!(left.count_rotors_valid, right.count_rotors_valid);
        assert_eq!(
            left.count_missing_even_rotor_steps,
            right.count_missing_even_rotor_steps
        );
        assert_eq!(left.kappa_count, right.kappa_count);
        assert_eq!(left.tau_count, right.tau_count);
        assert_f64_bits_eq("l3_kappa_max", left.l3_kappa_max, right.l3_kappa_max);
        assert_f64_bits_eq("l3_kappa_mean", left.l3_kappa_mean, right.l3_kappa_mean);
        assert_f64_bits_eq("l3_kappa_std", left.l3_kappa_std, right.l3_kappa_std);
        assert_f64_bits_eq("l3_kappa_ratio", left.l3_kappa_ratio, right.l3_kappa_ratio);
        assert_f64_bits_eq("l4_tau_max", left.l4_tau_max, right.l4_tau_max);
        assert_f64_bits_eq("l4_tau_mean", left.l4_tau_mean, right.l4_tau_mean);
        assert_f64_bits_eq("l4_tau_std", left.l4_tau_std, right.l4_tau_std);
        assert_f64_bits_eq("l4_tau_p90", left.l4_tau_p90, right.l4_tau_p90);
        assert_eq!(left.missing_reason, right.missing_reason);
    }

    #[test]
    fn too_few_steps_returns_missing_reason() {
        let input = Gate3TelemetryInput {
            sample_id: 1,
            ans_vec8: vec![e(0), e(1), e(2)],
            sample_label: None,
            answer_length: Some(3),
        };
        let out = compute_gate3_telemetry(&input);
        assert_eq!(out.missing_reason, Some(Gate3MissingReason::TooFewSteps));
    }

    #[test]
    fn invalid_vec8_returns_missing_reason() {
        let mut bad = e(0);
        bad[0] = f64::NAN;
        let input_nan = Gate3TelemetryInput {
            sample_id: 2,
            ans_vec8: vec![e(0), e(1), bad, e(2)],
            sample_label: Some(1),
            answer_length: Some(4),
        };
        let out_nan = compute_gate3_telemetry(&input_nan);
        assert_eq!(
            out_nan.missing_reason,
            Some(Gate3MissingReason::InvalidVec8)
        );

        let input_dim = Gate3TelemetryInput {
            sample_id: 3,
            ans_vec8: vec![e(0), vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], e(1), e(2)],
            sample_label: Some(0),
            answer_length: Some(4),
        };
        let out_dim = compute_gate3_telemetry(&input_dim);
        assert_eq!(
            out_dim.missing_reason,
            Some(Gate3MissingReason::InvalidVec8)
        );
    }

    #[test]
    fn antipodal_transition_marks_missing_step_count() {
        let mut neg_e0 = e(0);
        neg_e0[0] = -1.0;
        let input = Gate3TelemetryInput {
            sample_id: 4,
            ans_vec8: vec![e(0), neg_e0, e(1), e(2)],
            sample_label: None,
            answer_length: Some(4),
        };
        let out = compute_gate3_telemetry(&input);
        assert!(out.count_missing_even_rotor_steps >= 1);
        assert!(out.count_rotors_valid < out.count_rotors_total);
    }

    #[test]
    fn smooth_vs_kink_has_higher_tau_for_kink() {
        let steps = 24;
        let smooth = Gate3TelemetryInput {
            sample_id: 10,
            ans_vec8: smooth_trajectory(steps),
            sample_label: None,
            answer_length: Some(steps),
        };
        let kink = Gate3TelemetryInput {
            sample_id: 11,
            ans_vec8: kink_trajectory(steps, 12),
            sample_label: None,
            answer_length: Some(steps),
        };

        let smooth_out = compute_gate3_telemetry(&smooth);
        let kink_out = compute_gate3_telemetry(&kink);
        assert_eq!(smooth_out.missing_reason, None);
        assert_eq!(kink_out.missing_reason, None);
        assert!(
            kink_out.l4_tau_mean > smooth_out.l4_tau_mean,
            "expected kink tau_mean > smooth tau_mean, got {} <= {}",
            kink_out.l4_tau_mean,
            smooth_out.l4_tau_mean
        );
    }

    #[test]
    fn gate3_telemetry_is_bitwise_deterministic() {
        let input = Gate3TelemetryInput {
            sample_id: 12,
            ans_vec8: smooth_trajectory(24),
            sample_label: Some(1),
            answer_length: Some(24),
        };
        let left = compute_gate3_telemetry(&input);
        let right = compute_gate3_telemetry(&input);
        assert_bits_equal(&left, &right);
    }
}
