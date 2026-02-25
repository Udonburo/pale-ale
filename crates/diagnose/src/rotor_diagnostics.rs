use crate::linking::{
    CanonicalLink, SampleLinkReport, Top1Step, GATE1_TOPK, MAX_MISSING_LINK_STEP_RATE,
};
use pale_ale_rotor::{
    proj_chordal_v1, simple_rotor29_doc_to_ans, RotorConfig, RotorError, RotorStep, BIV_DIM,
    ROOT_DIM, ROTOR_DIM,
};
use serde::{Deserialize, Serialize};
use std::fmt;

pub const METHOD_ID: &str = "rotor_diagnostics_v1";
pub const DISTANCE_ID: &str = "proj_chordal_v1";
pub const THETA_SOURCE_ID: &str = "theta_uv_atan2_v1";
pub const TRIMMED_BEST_ID: &str = "trimmed_best_v1(k=8,p=0.5,key=rank,tie=doc_unit_id)";

pub const EPS_NORM: f64 = 1e-6;
pub const EPS_DIST: f64 = 1e-9;
pub const TAU_PLANE: f64 = 1e-5;
pub const TAU_WEDGE: f64 = 1e-6;
pub const MAX_ANTIPODAL_DROP_RATE: f64 = 0.20;
pub const MIN_ROTORS: usize = 3;
pub const MIN_PLANES: usize = 2;
pub const TRIMMED_BEST_P: f64 = 0.5;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ExcludedReason {
    NonFiniteVec8,
    ZeroOrNonfiniteNorm,
    NoAnswerUnits,
    MissingHallucUnitLabel,
    ExcessAntipodalDropRate,
    RotorRenormFailure,
    TrimmedBestZeroOrNonfiniteNorm,
    LinkSanitySampleFail,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum MetricMissingReason {
    MissingTop1Link,
    MissingLinksForTheta,
    MissingTheta,
    MissingPlanes,
    TooFewRotors,
    TooManyMissingLinkSteps,
    TooFewRotorStepsForWandering,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RotorDiagnosticsInput {
    pub sample_id: u64,
    pub links: SampleLinkReport,
    pub doc_vec8: Vec<[f64; ROOT_DIM]>,
    pub ans_vec8: Vec<[f64; ROOT_DIM]>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RotorDiagnosticsResult {
    pub sample_id: u64,
    pub method_id: String,
    pub distance_id: String,
    pub theta_source_id: String,
    pub trimmed_best_id: String,
    pub top1: TrackDiagnostics,
    pub trimmed: TrackDiagnostics,
    pub top1_gate_steps: Vec<Top1GateStep>,
    pub trimmed_rbar_norm_pre_values: Vec<f64>,
    pub trimmed_stability: TrimmedStabilityDiagnostics,
    pub degenerate_path_rate_counts: DegeneratePathRateCounts,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Top1GateStep {
    pub ans_unit_id: u32,
    pub doc_unit_id: u32,
    pub dot: f64,
    pub wedge_norm: Option<f64>,
    pub is_collinear: bool,
    pub is_antipodal_angle_only: bool,
    pub is_antipodal_drop: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct DegeneratePathRateCounts {
    pub top1_numerator: usize,
    pub top1_denominator: usize,
    pub trimmed_numerator: usize,
    pub trimmed_denominator: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrackDiagnostics {
    pub track_id: String,
    pub excluded_reason: Option<ExcludedReason>,
    pub max_theta: MetricField<f64>,
    pub plane_turn: PlaneTurnMetric,
    pub alignment: AlignmentMetric,
    pub wandering: WanderingMetric,
    pub rates: RateMetrics,
    pub counts: TrackCounts,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MetricField<T> {
    pub value: Option<T>,
    pub metric_missing_reason: Option<MetricMissingReason>,
}

impl<T> MetricField<T> {
    fn present(value: T) -> Self {
        Self {
            value: Some(value),
            metric_missing_reason: None,
        }
    }

    fn missing(reason: MetricMissingReason) -> Self {
        Self {
            value: None,
            metric_missing_reason: Some(reason),
        }
    }

    fn excluded() -> Self {
        Self {
            value: None,
            metric_missing_reason: None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PlaneTurnMetric {
    pub mean: MetricField<f64>,
    pub max: MetricField<f64>,
    pub var: MetricField<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AlignmentMetric {
    pub mean: MetricField<f64>,
    pub var: MetricField<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct WanderingMetric {
    pub ratio: MetricField<f64>,
    pub degenerate_path: Option<bool>,
    pub degenerate_path_rate_numerator: usize,
    pub degenerate_path_rate_denominator: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RateMetrics {
    pub rate_collinear: f64,
    pub rate_antipodal_angle_only: f64,
    pub rate_antipodal_drop: f64,
    pub rate_missing_link_steps: f64,
    pub rate_missing_top1_steps: f64,
    pub normalized_rate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrackCounts {
    pub steps_total: usize,
    pub vec8_total: usize,
    pub normalized_count: usize,
    pub max_norm_err: f64,
    pub count_collinear: usize,
    pub count_antipodal_angle_only: usize,
    pub count_antipodal_drop: usize,
    pub count_missing_link_steps: usize,
    pub count_missing_top1_steps: usize,
    pub n_theta_valid: usize,
    pub n_rotors_valid: usize,
    pub n_planes_valid: usize,
    pub missing_link_step_rate: f64,
    pub missing_top1_step_rate: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct TrimmedStabilityDiagnostics {
    pub trimmed_rbar_norm_pre_p50: Option<f64>,
    pub trimmed_rbar_norm_pre_p10: Option<f64>,
    pub trimmed_rbar_norm_pre_p01: Option<f64>,
    pub trimmed_failure_rate: f64,
    pub trimmed_failure_steps: usize,
    pub trimmed_attempted_steps: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RotorDiagnosticsError {
    UnitCountMismatch {
        sample_id: u64,
        expected_doc_unit_count: usize,
        actual_doc_unit_count: usize,
        expected_ans_unit_count: usize,
        actual_ans_unit_count: usize,
    },
    Top1StepCountMismatch {
        sample_id: u64,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for RotorDiagnosticsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnitCountMismatch {
                sample_id,
                expected_doc_unit_count,
                actual_doc_unit_count,
                expected_ans_unit_count,
                actual_ans_unit_count,
            } => write!(
                f,
                "unit count mismatch for sample {}: docs expected {}, got {}; answers expected {}, got {}",
                sample_id,
                expected_doc_unit_count,
                actual_doc_unit_count,
                expected_ans_unit_count,
                actual_ans_unit_count
            ),
            Self::Top1StepCountMismatch {
                sample_id,
                expected,
                actual,
            } => write!(
                f,
                "top1 step count mismatch for sample {}: expected {}, got {}",
                sample_id, expected, actual
            ),
        }
    }
}

impl std::error::Error for RotorDiagnosticsError {}

pub fn compute_rotor_diagnostics(
    input: &RotorDiagnosticsInput,
) -> Result<RotorDiagnosticsResult, RotorDiagnosticsError> {
    let expected_doc = input.links.canonicalized.doc_unit_count;
    let expected_ans = input.links.canonicalized.ans_unit_count;
    if input.doc_vec8.len() != expected_doc || input.ans_vec8.len() != expected_ans {
        return Err(RotorDiagnosticsError::UnitCountMismatch {
            sample_id: input.sample_id,
            expected_doc_unit_count: expected_doc,
            actual_doc_unit_count: input.doc_vec8.len(),
            expected_ans_unit_count: expected_ans,
            actual_ans_unit_count: input.ans_vec8.len(),
        });
    }

    if input.links.top1.steps.len() != expected_ans {
        return Err(RotorDiagnosticsError::Top1StepCountMismatch {
            sample_id: input.sample_id,
            expected: expected_ans,
            actual: input.links.top1.steps.len(),
        });
    }

    let vec8_summary = normalize_vec8_inputs(&input.doc_vec8, &input.ans_vec8);
    let top1 = compute_top1_track(input, &vec8_summary);
    let trimmed = compute_trimmed_track(input, &vec8_summary);
    let trimmed_stability = compute_trimmed_stability(
        &trimmed.trimmed_rbar_norm_pre,
        trimmed.trimmed_failure_steps,
        expected_ans,
    );

    let degenerate_path_rate_counts = DegeneratePathRateCounts {
        top1_numerator: top1.diagnostics.wandering.degenerate_path_rate_numerator,
        top1_denominator: top1.diagnostics.wandering.degenerate_path_rate_denominator,
        trimmed_numerator: trimmed.diagnostics.wandering.degenerate_path_rate_numerator,
        trimmed_denominator: trimmed
            .diagnostics
            .wandering
            .degenerate_path_rate_denominator,
    };

    Ok(RotorDiagnosticsResult {
        sample_id: input.sample_id,
        method_id: METHOD_ID.to_string(),
        distance_id: DISTANCE_ID.to_string(),
        theta_source_id: THETA_SOURCE_ID.to_string(),
        trimmed_best_id: TRIMMED_BEST_ID.to_string(),
        top1_gate_steps: top1.top1_gate_steps,
        trimmed_rbar_norm_pre_values: trimmed.trimmed_rbar_norm_pre.clone(),
        top1: top1.diagnostics,
        trimmed: trimmed.diagnostics,
        trimmed_stability,
        degenerate_path_rate_counts,
    })
}

#[derive(Clone, Debug)]
struct NormalizedVec8Summary {
    normalized_doc: Vec<[f64; ROOT_DIM]>,
    normalized_ans: Vec<[f64; ROOT_DIM]>,
    normalized_count: usize,
    max_norm_err: f64,
    vec8_total: usize,
    excluded_reason: Option<ExcludedReason>,
}

fn normalize_vec8_inputs(
    doc_vec8: &[[f64; ROOT_DIM]],
    ans_vec8: &[[f64; ROOT_DIM]],
) -> NormalizedVec8Summary {
    let mut normalized_count = 0usize;
    let mut max_norm_err = 0.0_f64;

    let mut normalized_doc = Vec::with_capacity(doc_vec8.len());
    for vec8 in doc_vec8 {
        match normalize_vec8_with_stats(vec8, &mut normalized_count, &mut max_norm_err) {
            Ok(value) => normalized_doc.push(value),
            Err(reason) => {
                return NormalizedVec8Summary {
                    normalized_doc: Vec::new(),
                    normalized_ans: Vec::new(),
                    normalized_count: 0,
                    max_norm_err: 0.0,
                    vec8_total: doc_vec8.len() + ans_vec8.len(),
                    excluded_reason: Some(reason),
                };
            }
        }
    }

    let mut normalized_ans = Vec::with_capacity(ans_vec8.len());
    for vec8 in ans_vec8 {
        match normalize_vec8_with_stats(vec8, &mut normalized_count, &mut max_norm_err) {
            Ok(value) => normalized_ans.push(value),
            Err(reason) => {
                return NormalizedVec8Summary {
                    normalized_doc: Vec::new(),
                    normalized_ans: Vec::new(),
                    normalized_count: 0,
                    max_norm_err: 0.0,
                    vec8_total: doc_vec8.len() + ans_vec8.len(),
                    excluded_reason: Some(reason),
                };
            }
        }
    }

    NormalizedVec8Summary {
        normalized_doc,
        normalized_ans,
        normalized_count,
        max_norm_err,
        vec8_total: doc_vec8.len() + ans_vec8.len(),
        excluded_reason: None,
    }
}

fn normalize_vec8_with_stats(
    vec8: &[f64; ROOT_DIM],
    normalized_count: &mut usize,
    max_norm_err: &mut f64,
) -> Result<[f64; ROOT_DIM], ExcludedReason> {
    if vec8.iter().any(|value| !value.is_finite()) {
        return Err(ExcludedReason::NonFiniteVec8);
    }

    let norm_sq: f64 = vec8.iter().map(|value| value * value).sum();
    let norm = norm_sq.sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return Err(ExcludedReason::ZeroOrNonfiniteNorm);
    }

    let norm_err = (norm - 1.0).abs();
    if norm_err > EPS_NORM {
        *normalized_count += 1;
    }
    if norm_err > *max_norm_err {
        *max_norm_err = norm_err;
    }

    let inv = 1.0 / norm;
    let mut out = [0.0_f64; ROOT_DIM];
    for idx in 0..ROOT_DIM {
        out[idx] = vec8[idx] * inv;
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TrackKind {
    Top1,
    Trimmed,
}

impl TrackKind {
    fn id(self) -> &'static str {
        match self {
            Self::Top1 => "top1",
            Self::Trimmed => "trimmed_best",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct StepSignal {
    theta: Option<f64>,
    rotor: Option<[f64; ROTOR_DIM]>,
    bhat: Option<[f64; BIV_DIM]>,
    is_collinear: bool,
    antipodal_angle_only: bool,
    antipodal_drop: bool,
}

impl StepSignal {
    fn missing() -> Self {
        Self {
            theta: None,
            rotor: None,
            bhat: None,
            is_collinear: false,
            antipodal_angle_only: false,
            antipodal_drop: false,
        }
    }
}

#[allow(clippy::large_enum_variant)]
#[derive(Clone, Copy, Debug)]
enum CandidateSignal {
    Materialized {
        theta: f64,
        r29: [f64; ROTOR_DIM],
        bhat: Option<[f64; BIV_DIM]>,
        is_collinear: bool,
    },
    AntipodalAngleOnly {
        theta: f64,
    },
    AntipodalDrop,
    RotorRenormFailure,
}

#[derive(Clone, Debug)]
struct TrackBuild {
    diagnostics: TrackDiagnostics,
    trimmed_rbar_norm_pre: Vec<f64>,
    trimmed_failure_steps: usize,
    top1_gate_steps: Vec<Top1GateStep>,
}

fn compute_top1_track(
    input: &RotorDiagnosticsInput,
    vec8_summary: &NormalizedVec8Summary,
) -> TrackBuild {
    if let Some(reason) = vec8_summary.excluded_reason {
        return build_excluded_track(TrackKind::Top1, &input.links, vec8_summary, reason);
    }

    let steps_total = input.links.canonicalized.ans_unit_count;
    if steps_total == 0 {
        return build_excluded_track(
            TrackKind::Top1,
            &input.links,
            vec8_summary,
            ExcludedReason::NoAnswerUnits,
        );
    }

    let mut steps = Vec::with_capacity(steps_total);
    let mut excluded_reason = None;
    let mut top1_gate_steps = Vec::new();
    for step in &input.links.top1.steps {
        let signal = match *step {
            Top1Step::MissingLink { .. } => StepSignal::missing(),
            Top1Step::MissingTop1 { .. } => StepSignal::missing(),
            Top1Step::Selected {
                ans_unit_id,
                doc_unit_id,
            } => {
                let doc_vec8 = &vec8_summary.normalized_doc[doc_unit_id as usize];
                let ans_vec8 = &vec8_summary.normalized_ans[ans_unit_id as usize];
                let (dot, wedge_norm) = dot_wedge_from_normalized(doc_vec8, ans_vec8);
                // dot is always finite: Vec8 contract rejects non-finite inputs,
                // normalize_vec8_with_stats ensures unit norm, and dot_wedge_from_normalized
                // clamps the inner product to [-1, 1].
                let candidate = compute_candidate_signal(doc_vec8, ans_vec8);
                match candidate {
                    CandidateSignal::Materialized {
                        theta,
                        r29,
                        bhat,
                        is_collinear,
                    } => {
                        top1_gate_steps.push(Top1GateStep {
                            ans_unit_id,
                            doc_unit_id,
                            dot,
                            wedge_norm: Some(wedge_norm),
                            is_collinear,
                            is_antipodal_angle_only: false,
                            is_antipodal_drop: false,
                        });
                        debug_assert!(top1_gate_steps
                            .last()
                            .map(|s| s.dot.is_finite())
                            .unwrap_or(false));
                        StepSignal {
                            theta: Some(theta),
                            rotor: Some(r29),
                            bhat,
                            is_collinear,
                            antipodal_angle_only: false,
                            antipodal_drop: false,
                        }
                    }
                    CandidateSignal::AntipodalAngleOnly { theta } => {
                        top1_gate_steps.push(Top1GateStep {
                            ans_unit_id,
                            doc_unit_id,
                            dot,
                            wedge_norm: None,
                            is_collinear: false,
                            is_antipodal_angle_only: true,
                            is_antipodal_drop: false,
                        });
                        debug_assert!(top1_gate_steps
                            .last()
                            .map(|s| s.dot.is_finite())
                            .unwrap_or(false));
                        StepSignal {
                            theta: Some(theta),
                            rotor: None,
                            bhat: None,
                            is_collinear: false,
                            antipodal_angle_only: true,
                            antipodal_drop: false,
                        }
                    }
                    CandidateSignal::AntipodalDrop => {
                        top1_gate_steps.push(Top1GateStep {
                            ans_unit_id,
                            doc_unit_id,
                            dot,
                            wedge_norm: None,
                            is_collinear: false,
                            is_antipodal_angle_only: false,
                            is_antipodal_drop: true,
                        });
                        debug_assert!(top1_gate_steps
                            .last()
                            .map(|s| s.dot.is_finite())
                            .unwrap_or(false));
                        StepSignal {
                            theta: None,
                            rotor: None,
                            bhat: None,
                            is_collinear: false,
                            antipodal_angle_only: false,
                            antipodal_drop: true,
                        }
                    }
                    CandidateSignal::RotorRenormFailure => {
                        top1_gate_steps.push(Top1GateStep {
                            ans_unit_id,
                            doc_unit_id,
                            dot,
                            wedge_norm: Some(wedge_norm),
                            is_collinear: false,
                            is_antipodal_angle_only: false,
                            is_antipodal_drop: false,
                        });
                        debug_assert!(top1_gate_steps
                            .last()
                            .map(|s| s.dot.is_finite())
                            .unwrap_or(false));
                        excluded_reason = Some(ExcludedReason::RotorRenormFailure);
                        StepSignal::missing()
                    }
                }
            }
        };
        steps.push(signal);
    }

    let diagnostics = finalize_track(
        TrackKind::Top1,
        &steps,
        input.links.top1.count_missing_link_steps,
        input.links.top1.count_missing_top1_steps,
        vec8_summary,
        excluded_reason,
    );

    TrackBuild {
        diagnostics,
        trimmed_rbar_norm_pre: Vec::new(),
        trimmed_failure_steps: 0,
        top1_gate_steps,
    }
}

fn compute_trimmed_track(
    input: &RotorDiagnosticsInput,
    vec8_summary: &NormalizedVec8Summary,
) -> TrackBuild {
    if let Some(reason) = vec8_summary.excluded_reason {
        return build_excluded_track(TrackKind::Trimmed, &input.links, vec8_summary, reason);
    }

    let steps_total = input.links.canonicalized.ans_unit_count;
    if steps_total == 0 {
        return build_excluded_track(
            TrackKind::Trimmed,
            &input.links,
            vec8_summary,
            ExcludedReason::NoAnswerUnits,
        );
    }

    let mut steps = Vec::with_capacity(steps_total);
    let mut count_missing_link_steps = 0usize;
    let count_missing_top1_steps = 0usize;
    let mut trimmed_rbar_norm_pre = Vec::new();
    let mut trimmed_failure_steps = 0usize;

    for ans_unit_idx in 0..steps_total {
        let ans_unit_id = ans_unit_idx as u32;
        let links = &input.links.canonicalized.links_by_answer[ans_unit_idx];
        if links.is_empty() {
            count_missing_link_steps += 1;
            steps.push(StepSignal::missing());
            continue;
        }

        let selected = trimmed_candidates(links);
        let mut theta_candidates = Vec::new();
        let mut materialized = Vec::new();
        let mut antipodal_drop_for_step = false;
        for link in selected {
            let candidate = compute_candidate_signal(
                &vec8_summary.normalized_doc[link.doc_unit_id as usize],
                &vec8_summary.normalized_ans[ans_unit_id as usize],
            );
            match candidate {
                CandidateSignal::Materialized { theta, r29, .. } => {
                    theta_candidates.push(theta);
                    materialized.push(r29);
                }
                CandidateSignal::AntipodalAngleOnly { theta } => {
                    theta_candidates.push(theta);
                }
                CandidateSignal::AntipodalDrop => {
                    antipodal_drop_for_step = true;
                }
                CandidateSignal::RotorRenormFailure => {
                    // Candidate-level renorm failure is dropped from Trimmed averaging
                    // (same exclusion lane as unusable candidates, not a track-wide abort).
                }
            }
        }

        if materialized.is_empty() {
            let theta = theta_candidates
                .into_iter()
                .max_by(|left, right| left.total_cmp(right));
            steps.push(StepSignal {
                theta,
                rotor: None,
                bhat: None,
                is_collinear: false,
                antipodal_angle_only: theta.is_some(),
                antipodal_drop: antipodal_drop_for_step && theta.is_none(),
            });
            continue;
        }

        let (mean_r, pre_norm) = mean_rotor29(&materialized);
        if pre_norm.is_finite() {
            trimmed_rbar_norm_pre.push(pre_norm);
        }
        if !pre_norm.is_finite() || pre_norm == 0.0 {
            trimmed_failure_steps += 1;
            steps.push(StepSignal::missing());
            continue;
        }
        match build_trimmed_step_signal(&mean_r, &theta_candidates) {
            Ok(signal) => steps.push(signal),
            Err(reason) => {
                if reason == ExcludedReason::TrimmedBestZeroOrNonfiniteNorm {
                    trimmed_failure_steps += 1;
                }
                steps.push(StepSignal::missing());
            }
        }
    }

    let excluded_reason = if trimmed_failure_steps > 0 {
        Some(ExcludedReason::TrimmedBestZeroOrNonfiniteNorm)
    } else {
        None
    };

    let diagnostics = finalize_track(
        TrackKind::Trimmed,
        &steps,
        count_missing_link_steps,
        count_missing_top1_steps,
        vec8_summary,
        excluded_reason,
    );

    TrackBuild {
        diagnostics,
        trimmed_rbar_norm_pre,
        trimmed_failure_steps,
        top1_gate_steps: Vec::new(),
    }
}

fn build_excluded_track(
    track: TrackKind,
    links: &SampleLinkReport,
    vec8_summary: &NormalizedVec8Summary,
    reason: ExcludedReason,
) -> TrackBuild {
    let steps_total = links.canonicalized.ans_unit_count;
    let count_missing_link_steps = if track == TrackKind::Top1 {
        links.top1.count_missing_link_steps
    } else {
        links
            .canonicalized
            .links_by_answer
            .iter()
            .filter(|links| links.is_empty())
            .count()
    };
    let count_missing_top1_steps = if track == TrackKind::Top1 {
        links.top1.count_missing_top1_steps
    } else {
        0
    };

    let missing_link_step_rate = ratio(count_missing_link_steps, steps_total);
    let missing_top1_step_rate = ratio(count_missing_top1_steps, steps_total);
    let counts = TrackCounts {
        steps_total,
        vec8_total: vec8_summary.vec8_total,
        normalized_count: vec8_summary.normalized_count,
        max_norm_err: vec8_summary.max_norm_err,
        count_collinear: 0,
        count_antipodal_angle_only: 0,
        count_antipodal_drop: 0,
        count_missing_link_steps,
        count_missing_top1_steps,
        n_theta_valid: 0,
        n_rotors_valid: 0,
        n_planes_valid: 0,
        missing_link_step_rate,
        missing_top1_step_rate,
    };

    let diagnostics = TrackDiagnostics {
        track_id: track.id().to_string(),
        excluded_reason: Some(reason),
        max_theta: MetricField::excluded(),
        plane_turn: PlaneTurnMetric {
            mean: MetricField::excluded(),
            max: MetricField::excluded(),
            var: MetricField::excluded(),
        },
        alignment: AlignmentMetric {
            mean: MetricField::excluded(),
            var: MetricField::excluded(),
        },
        wandering: WanderingMetric {
            ratio: MetricField::excluded(),
            degenerate_path: None,
            degenerate_path_rate_numerator: 0,
            degenerate_path_rate_denominator: 0,
        },
        rates: RateMetrics {
            rate_collinear: 0.0,
            rate_antipodal_angle_only: 0.0,
            rate_antipodal_drop: 0.0,
            rate_missing_link_steps: missing_link_step_rate,
            rate_missing_top1_steps: missing_top1_step_rate,
            normalized_rate: ratio(vec8_summary.normalized_count, vec8_summary.vec8_total),
        },
        counts,
    };

    TrackBuild {
        diagnostics,
        trimmed_rbar_norm_pre: Vec::new(),
        trimmed_failure_steps: 0,
        top1_gate_steps: Vec::new(),
    }
}

fn finalize_track(
    track: TrackKind,
    steps: &[StepSignal],
    count_missing_link_steps: usize,
    count_missing_top1_steps: usize,
    vec8_summary: &NormalizedVec8Summary,
    explicit_excluded_reason: Option<ExcludedReason>,
) -> TrackDiagnostics {
    let steps_total = steps.len();
    let missing_link_step_rate = ratio(count_missing_link_steps, steps_total);
    let missing_top1_step_rate = ratio(count_missing_top1_steps, steps_total);

    let mut count_collinear = 0usize;
    let mut count_antipodal_angle_only = 0usize;
    let mut count_antipodal_drop = 0usize;
    let mut theta_values = Vec::new();
    let mut rotor_values = Vec::new();
    let mut plane_values = Vec::new();
    for step in steps {
        if let Some(theta) = step.theta {
            if theta.is_finite() {
                theta_values.push(theta);
            }
        }
        if let Some(rotor) = step.rotor {
            rotor_values.push(rotor);
        }
        if let Some(plane) = step.bhat {
            plane_values.push(plane);
        }
        if step.is_collinear {
            count_collinear += 1;
        }
        if step.antipodal_angle_only {
            count_antipodal_angle_only += 1;
        }
        if step.antipodal_drop {
            count_antipodal_drop += 1;
        }
    }

    let counts = TrackCounts {
        steps_total,
        vec8_total: vec8_summary.vec8_total,
        normalized_count: vec8_summary.normalized_count,
        max_norm_err: vec8_summary.max_norm_err,
        count_collinear,
        count_antipodal_angle_only,
        count_antipodal_drop,
        count_missing_link_steps,
        count_missing_top1_steps,
        n_theta_valid: theta_values.len(),
        n_rotors_valid: rotor_values.len(),
        n_planes_valid: plane_values.len(),
        missing_link_step_rate,
        missing_top1_step_rate,
    };

    let rates = RateMetrics {
        rate_collinear: ratio(count_collinear, steps_total),
        rate_antipodal_angle_only: ratio(count_antipodal_angle_only, steps_total),
        rate_antipodal_drop: ratio(count_antipodal_drop, steps_total),
        rate_missing_link_steps: missing_link_step_rate,
        rate_missing_top1_steps: missing_top1_step_rate,
        normalized_rate: ratio(vec8_summary.normalized_count, vec8_summary.vec8_total),
    };

    let excluded_reason = if let Some(reason) = explicit_excluded_reason {
        Some(reason)
    } else if ratio(count_antipodal_drop, steps_total) > MAX_ANTIPODAL_DROP_RATE {
        Some(ExcludedReason::ExcessAntipodalDropRate)
    } else {
        None
    };

    if excluded_reason.is_some() {
        return TrackDiagnostics {
            track_id: track.id().to_string(),
            excluded_reason,
            max_theta: MetricField::excluded(),
            plane_turn: PlaneTurnMetric {
                mean: MetricField::excluded(),
                max: MetricField::excluded(),
                var: MetricField::excluded(),
            },
            alignment: AlignmentMetric {
                mean: MetricField::excluded(),
                var: MetricField::excluded(),
            },
            wandering: WanderingMetric {
                ratio: MetricField::excluded(),
                degenerate_path: None,
                degenerate_path_rate_numerator: 0,
                degenerate_path_rate_denominator: 0,
            },
            rates,
            counts,
        };
    }

    let max_theta = compute_m1_max_theta(track, &counts, &theta_values);
    let rotor_missing_reason = rotor_metric_missing_reason(&counts);

    let plane_turn = compute_m2_plane_turn(&counts, &plane_values, rotor_missing_reason);
    let alignment = compute_m3_alignment(&counts, &plane_values, rotor_missing_reason);
    let wandering = compute_m4_wandering(&rotor_values, rotor_missing_reason);

    TrackDiagnostics {
        track_id: track.id().to_string(),
        excluded_reason: None,
        max_theta,
        plane_turn,
        alignment,
        wandering,
        rates,
        counts,
    }
}

fn compute_m1_max_theta(
    track: TrackKind,
    counts: &TrackCounts,
    theta_values: &[f64],
) -> MetricField<f64> {
    if let Some(value) = theta_values
        .iter()
        .copied()
        .max_by(|left, right| left.total_cmp(right))
    {
        return MetricField::present(value);
    }

    match track {
        TrackKind::Top1 => {
            if counts.count_missing_top1_steps > 0 {
                MetricField::missing(MetricMissingReason::MissingTop1Link)
            } else if counts.count_missing_link_steps > 0 {
                MetricField::missing(MetricMissingReason::MissingLinksForTheta)
            } else {
                MetricField::missing(MetricMissingReason::MissingTheta)
            }
        }
        TrackKind::Trimmed => {
            if counts.count_missing_link_steps > 0 {
                MetricField::missing(MetricMissingReason::MissingLinksForTheta)
            } else {
                MetricField::missing(MetricMissingReason::MissingTheta)
            }
        }
    }
}

fn rotor_metric_missing_reason(counts: &TrackCounts) -> Option<MetricMissingReason> {
    if counts.n_rotors_valid < MIN_ROTORS {
        return Some(MetricMissingReason::TooFewRotors);
    }
    if counts.missing_link_step_rate > MAX_MISSING_LINK_STEP_RATE {
        return Some(MetricMissingReason::TooManyMissingLinkSteps);
    }
    None
}

fn compute_m2_plane_turn(
    counts: &TrackCounts,
    plane_values: &[[f64; BIV_DIM]],
    rotor_missing_reason: Option<MetricMissingReason>,
) -> PlaneTurnMetric {
    if let Some(reason) = rotor_missing_reason {
        return PlaneTurnMetric {
            mean: MetricField::missing(reason),
            max: MetricField::missing(reason),
            var: MetricField::missing(reason),
        };
    }
    if counts.n_planes_valid < MIN_PLANES {
        return PlaneTurnMetric {
            mean: MetricField::missing(MetricMissingReason::MissingPlanes),
            max: MetricField::missing(MetricMissingReason::MissingPlanes),
            var: MetricField::missing(MetricMissingReason::MissingPlanes),
        };
    }

    let mut turns = Vec::with_capacity(plane_values.len().saturating_sub(1));
    for idx in 0..plane_values.len().saturating_sub(1) {
        let dot = plane_dot(&plane_values[idx], &plane_values[idx + 1]).clamp(-1.0, 1.0);
        turns.push(1.0 - dot.abs());
    }

    let mean_value = mean(&turns);
    let max_value = turns
        .iter()
        .copied()
        .max_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0);
    let var_value = variance_population(&turns, mean_value);
    PlaneTurnMetric {
        mean: MetricField::present(mean_value),
        max: MetricField::present(max_value),
        var: MetricField::present(var_value),
    }
}

fn compute_m3_alignment(
    counts: &TrackCounts,
    plane_values: &[[f64; BIV_DIM]],
    rotor_missing_reason: Option<MetricMissingReason>,
) -> AlignmentMetric {
    if let Some(reason) = rotor_missing_reason {
        return AlignmentMetric {
            mean: MetricField::missing(reason),
            var: MetricField::missing(reason),
        };
    }
    if counts.n_planes_valid < MIN_PLANES {
        return AlignmentMetric {
            mean: MetricField::missing(MetricMissingReason::MissingPlanes),
            var: MetricField::missing(MetricMissingReason::MissingPlanes),
        };
    }

    let b_global = compute_global_plane(plane_values);
    let mut aligned = Vec::with_capacity(plane_values.len());
    for bhat in plane_values {
        let dot = plane_dot(bhat, &b_global).clamp(-1.0, 1.0);
        aligned.push(1.0 - dot.abs());
    }

    let mean_value = mean(&aligned);
    let var_value = variance_population(&aligned, mean_value);
    AlignmentMetric {
        mean: MetricField::present(mean_value),
        var: MetricField::present(var_value),
    }
}

fn compute_global_plane(planes: &[[f64; BIV_DIM]]) -> [f64; BIV_DIM] {
    let first = planes.first().expect("must have at least one plane");
    let last = planes.last().expect("must have at least one plane");
    // SSOT asks for a start->end plane preference. With available primitives in Gate 1,
    // we approximate this by normalized endpoint plane mixing before mean-plane fallback.
    let mut endpoint_mix = [0.0_f64; BIV_DIM];
    for idx in 0..BIV_DIM {
        endpoint_mix[idx] = first[idx] + last[idx];
    }
    if let Some(unit) = normalize_plane(&endpoint_mix) {
        return unit;
    }

    let mut mean_plane = [0.0_f64; BIV_DIM];
    for plane in planes {
        for idx in 0..BIV_DIM {
            mean_plane[idx] += plane[idx];
        }
    }
    if let Some(unit) = normalize_plane(&mean_plane) {
        return unit;
    }
    *first
}

fn compute_m4_wandering(
    rotor_values: &[[f64; ROTOR_DIM]],
    rotor_missing_reason: Option<MetricMissingReason>,
) -> WanderingMetric {
    if let Some(reason) = rotor_missing_reason {
        return WanderingMetric {
            ratio: MetricField::missing(reason),
            degenerate_path: None,
            degenerate_path_rate_numerator: 0,
            degenerate_path_rate_denominator: 0,
        };
    }
    if rotor_values.len() < 2 {
        return WanderingMetric {
            ratio: MetricField::missing(MetricMissingReason::TooFewRotorStepsForWandering),
            degenerate_path: None,
            degenerate_path_rate_numerator: 0,
            degenerate_path_rate_denominator: 0,
        };
    }

    let mut length = 0.0_f64;
    for idx in 0..(rotor_values.len() - 1) {
        length += proj_chordal_v1(&rotor_values[idx], &rotor_values[idx + 1]);
    }
    let direct = proj_chordal_v1(
        rotor_values.first().expect("len>=2"),
        rotor_values.last().expect("len>=2"),
    );
    if direct < EPS_DIST {
        return WanderingMetric {
            ratio: MetricField::present(0.0),
            degenerate_path: Some(true),
            degenerate_path_rate_numerator: 1,
            degenerate_path_rate_denominator: 1,
        };
    }

    WanderingMetric {
        ratio: MetricField::present(length / direct),
        degenerate_path: Some(false),
        degenerate_path_rate_numerator: 0,
        degenerate_path_rate_denominator: 1,
    }
}

fn compute_trimmed_stability(
    norms_pre: &[f64],
    trimmed_failure_steps: usize,
    steps_total: usize,
) -> TrimmedStabilityDiagnostics {
    TrimmedStabilityDiagnostics {
        trimmed_rbar_norm_pre_p50: quantile_nearest_rank(norms_pre, 0.50),
        trimmed_rbar_norm_pre_p10: quantile_nearest_rank(norms_pre, 0.10),
        trimmed_rbar_norm_pre_p01: quantile_nearest_rank(norms_pre, 0.01),
        trimmed_failure_rate: ratio(trimmed_failure_steps, steps_total),
        trimmed_failure_steps,
        trimmed_attempted_steps: norms_pre.len(),
    }
}

fn quantile_nearest_rank(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let n = sorted.len();
    let rank_f64 = (q.clamp(0.0, 1.0) * n as f64).ceil();
    let rank = if rank_f64 < 1.0 {
        1
    } else if rank_f64 > n as f64 {
        n
    } else {
        rank_f64 as usize
    };
    Some(sorted[rank - 1])
}

fn trimmed_candidates(links: &[CanonicalLink]) -> &[CanonicalLink] {
    let k_eff = links.len().min(GATE1_TOPK as usize);
    let m = ((k_eff as f64) * TRIMMED_BEST_P).ceil() as usize;
    &links[..m]
}

fn dot_wedge_from_normalized(doc_vec8: &[f64; ROOT_DIM], ans_vec8: &[f64; ROOT_DIM]) -> (f64, f64) {
    let mut dot = 0.0_f64;
    for idx in 0..ROOT_DIM {
        dot += doc_vec8[idx] * ans_vec8[idx];
    }
    let dot = dot.clamp(-1.0, 1.0);

    let mut wedge_sq = 0.0_f64;
    for i in 0..ROOT_DIM {
        for j in (i + 1)..ROOT_DIM {
            let coeff = (doc_vec8[i] * ans_vec8[j]) - (doc_vec8[j] * ans_vec8[i]);
            wedge_sq += coeff * coeff;
        }
    }
    (dot, wedge_sq.sqrt())
}

fn compute_candidate_signal(
    doc_vec8: &[f64; ROOT_DIM],
    ans_vec8: &[f64; ROOT_DIM],
) -> CandidateSignal {
    match simple_rotor29_doc_to_ans(*doc_vec8, *ans_vec8, RotorConfig::default()) {
        Ok(RotorStep::AntipodalAngleOnly { theta }) => {
            if theta.is_finite() {
                CandidateSignal::AntipodalAngleOnly { theta }
            } else {
                CandidateSignal::AntipodalDrop
            }
        }
        Ok(RotorStep::Materialized {
            r29,
            theta,
            is_collinear,
        }) => {
            if !theta.is_finite() {
                return CandidateSignal::AntipodalDrop;
            }
            match renormalize_rotor29(&r29) {
                Some(r29_unit) => {
                    let b_norm = rotor_b_norm(&r29_unit);
                    let bhat = if b_norm > TAU_PLANE {
                        let mut out = [0.0_f64; BIV_DIM];
                        let inv = 1.0 / b_norm;
                        for idx in 0..BIV_DIM {
                            out[idx] = r29_unit[1 + idx] * inv;
                        }
                        Some(out)
                    } else {
                        None
                    };
                    CandidateSignal::Materialized {
                        theta,
                        r29: r29_unit,
                        bhat,
                        is_collinear,
                    }
                }
                None => CandidateSignal::RotorRenormFailure,
            }
        }
        Err(RotorError::NonFiniteTheta) => CandidateSignal::AntipodalDrop,
        Err(RotorError::RenormFailure) => CandidateSignal::RotorRenormFailure,
        Err(RotorError::Vec8(_)) => CandidateSignal::RotorRenormFailure,
    }
}

fn mean_rotor29(materialized: &[[f64; ROTOR_DIM]]) -> ([f64; ROTOR_DIM], f64) {
    let mut mean_r = [0.0_f64; ROTOR_DIM];
    for rotor in materialized {
        for idx in 0..ROTOR_DIM {
            mean_r[idx] += rotor[idx];
        }
    }
    let inv_len = 1.0 / (materialized.len() as f64);
    for value in mean_r.iter_mut().take(ROTOR_DIM) {
        *value *= inv_len;
    }
    let pre_norm = rotor_norm(&mean_r);
    (mean_r, pre_norm)
}

fn build_trimmed_step_signal(
    mean_r: &[f64; ROTOR_DIM],
    theta_candidates: &[f64],
) -> Result<StepSignal, ExcludedReason> {
    let unit = renormalize_rotor29(mean_r).ok_or(ExcludedReason::TrimmedBestZeroOrNonfiniteNorm)?;
    let b_norm = rotor_b_norm(&unit);
    let bhat = if b_norm > TAU_PLANE {
        let mut out = [0.0_f64; BIV_DIM];
        let inv = 1.0 / b_norm;
        for idx in 0..BIV_DIM {
            out[idx] = unit[1 + idx] * inv;
        }
        Some(out)
    } else {
        None
    };
    let theta = theta_candidates
        .iter()
        .copied()
        .max_by(|left, right| left.total_cmp(right));
    // Trimmed step aggregates multiple doc->ans candidates; using max(theta) keeps
    // "higher = more suspicious" monotonicity for the Gate 1 suspicion channel.
    Ok(StepSignal {
        theta,
        rotor: Some(unit),
        bhat,
        is_collinear: b_norm <= TAU_WEDGE,
        antipodal_angle_only: false,
        antipodal_drop: false,
    })
}

fn renormalize_rotor29(r29: &[f64; ROTOR_DIM]) -> Option<[f64; ROTOR_DIM]> {
    let norm = rotor_norm(r29);
    if !norm.is_finite() || norm == 0.0 {
        return None;
    }
    let mut out = [0.0_f64; ROTOR_DIM];
    let inv = 1.0 / norm;
    for idx in 0..ROTOR_DIM {
        out[idx] = r29[idx] * inv;
    }
    Some(out)
}

fn rotor_norm(r29: &[f64; ROTOR_DIM]) -> f64 {
    let mut sum = 0.0_f64;
    for value in r29 {
        sum += value * value;
    }
    sum.sqrt()
}

fn rotor_b_norm(r29: &[f64; ROTOR_DIM]) -> f64 {
    let mut sum = 0.0_f64;
    for idx in 0..BIV_DIM {
        sum += r29[1 + idx] * r29[1 + idx];
    }
    sum.sqrt()
}

fn plane_dot(left: &[f64; BIV_DIM], right: &[f64; BIV_DIM]) -> f64 {
    let mut sum = 0.0_f64;
    for idx in 0..BIV_DIM {
        sum += left[idx] * right[idx];
    }
    sum
}

fn normalize_plane(plane: &[f64; BIV_DIM]) -> Option<[f64; BIV_DIM]> {
    let norm = plane_dot(plane, plane).sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return None;
    }
    let mut out = [0.0_f64; BIV_DIM];
    let inv = 1.0 / norm;
    for idx in 0..BIV_DIM {
        out[idx] = plane[idx] * inv;
    }
    Some(out)
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        (num as f64) / (den as f64)
    }
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / (values.len() as f64)
    }
}

fn variance_population(values: &[f64], mean_value: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values
        .iter()
        .map(|value| {
            let d = *value - mean_value;
            d * d
        })
        .sum::<f64>()
        / (values.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linking::{canonicalize_links, compute_top1_accounting, LinkRow, SampleLinksInput};

    fn sample_links(
        sample_id: u64,
        ans_unit_count: usize,
        doc_unit_count: usize,
        rows: &[(u32, u32, u16)],
    ) -> SampleLinkReport {
        let input = SampleLinksInput {
            sample_id,
            ans_unit_count,
            doc_unit_count,
            links_topk: rows
                .iter()
                .map(|(ans_unit_id, doc_unit_id, rank)| LinkRow {
                    ans_unit_id: *ans_unit_id,
                    doc_unit_id: *doc_unit_id,
                    rank: *rank,
                })
                .collect(),
        };
        let canonicalized = canonicalize_links(&input);
        let top1 = compute_top1_accounting(&canonicalized);
        SampleLinkReport {
            sample_id,
            canonicalized,
            top1,
        }
    }

    fn e(idx: usize) -> [f64; ROOT_DIM] {
        let mut out = [0.0_f64; ROOT_DIM];
        out[idx] = 1.0;
        out
    }

    #[test]
    fn antipodal_angle_only_contributes_to_m1_only() {
        let input = RotorDiagnosticsInput {
            sample_id: 1,
            links: sample_links(1, 1, 1, &[(0, 0, 1)]),
            doc_vec8: vec![{
                let mut v = e(0);
                v[0] = -1.0;
                v
            }],
            ans_vec8: vec![e(0)],
        };

        let result = compute_rotor_diagnostics(&input).expect("diagnostics");
        assert_eq!(result.top1.excluded_reason, None);
        assert_eq!(result.top1.max_theta.value, Some(std::f64::consts::PI));
        assert_eq!(
            result.top1.plane_turn.mean.metric_missing_reason,
            Some(MetricMissingReason::TooFewRotors)
        );
        assert_eq!(
            result.top1.alignment.mean.metric_missing_reason,
            Some(MetricMissingReason::TooFewRotors)
        );
        assert_eq!(
            result.top1.wandering.ratio.metric_missing_reason,
            Some(MetricMissingReason::TooFewRotors)
        );
        assert_eq!(result.top1.counts.count_antipodal_angle_only, 1);
        assert_eq!(result.top1.counts.n_rotors_valid, 0);
    }

    #[test]
    fn trimmed_zero_materialized_candidates_is_metric_missing_not_excluded() {
        let input = RotorDiagnosticsInput {
            sample_id: 2,
            links: sample_links(2, 1, 2, &[(0, 0, 1), (0, 1, 2)]),
            doc_vec8: vec![
                {
                    let mut v = e(0);
                    v[0] = -1.0;
                    v
                },
                {
                    let mut v = e(0);
                    v[0] = -1.0;
                    v
                },
            ],
            ans_vec8: vec![e(0)],
        };

        let result = compute_rotor_diagnostics(&input).expect("diagnostics");
        assert_eq!(result.trimmed.excluded_reason, None);
        assert_eq!(
            result.trimmed.wandering.ratio.metric_missing_reason,
            Some(MetricMissingReason::TooFewRotors)
        );
        assert_eq!(result.trimmed.counts.n_rotors_valid, 0);
    }

    #[test]
    fn compressed_sequence_ordering_is_preserved_for_wandering() {
        let step_a = StepSignal {
            theta: Some(1.0),
            rotor: Some({
                let mut out = [0.0_f64; ROTOR_DIM];
                out[0] = 1.0;
                out
            }),
            bhat: None,
            is_collinear: false,
            antipodal_angle_only: false,
            antipodal_drop: false,
        };
        let step_b = StepSignal::missing();
        let step_c = StepSignal {
            theta: Some(1.0),
            rotor: Some({
                let mut out = [0.0_f64; ROTOR_DIM];
                out[1] = 1.0;
                out
            }),
            bhat: None,
            is_collinear: false,
            antipodal_angle_only: false,
            antipodal_drop: false,
        };
        let step_d = StepSignal {
            theta: Some(1.0),
            rotor: Some({
                let mut out = [0.0_f64; ROTOR_DIM];
                out[2] = 1.0;
                out
            }),
            bhat: None,
            is_collinear: false,
            antipodal_angle_only: false,
            antipodal_drop: false,
        };

        let diagnostics = finalize_track(
            TrackKind::Top1,
            &[step_a, step_b, step_c, step_d],
            0,
            0,
            &NormalizedVec8Summary {
                normalized_doc: vec![],
                normalized_ans: vec![],
                normalized_count: 0,
                max_norm_err: 0.0,
                vec8_total: 8,
                excluded_reason: None,
            },
            None,
        );

        assert_eq!(diagnostics.wandering.ratio.metric_missing_reason, None);
        assert!(diagnostics.wandering.ratio.value.expect("wandering") > 1.0);
    }

    #[test]
    fn fail_fast_excess_antipodal_drop_rate_excludes_track() {
        let dropped = StepSignal {
            theta: None,
            rotor: None,
            bhat: None,
            is_collinear: false,
            antipodal_angle_only: false,
            antipodal_drop: true,
        };
        let diagnostics = finalize_track(
            TrackKind::Top1,
            &[
                dropped,
                dropped,
                StepSignal::missing(),
                StepSignal::missing(),
            ],
            0,
            0,
            &NormalizedVec8Summary {
                normalized_doc: vec![],
                normalized_ans: vec![],
                normalized_count: 0,
                max_norm_err: 0.0,
                vec8_total: 8,
                excluded_reason: None,
            },
            None,
        );

        assert_eq!(
            diagnostics.excluded_reason,
            Some(ExcludedReason::ExcessAntipodalDropRate)
        );
    }

    #[test]
    fn degenerate_path_rate_counts_follow_denominator_rule() {
        let step = StepSignal {
            theta: Some(1.0),
            rotor: Some({
                let mut out = [0.0_f64; ROTOR_DIM];
                out[0] = 1.0;
                out
            }),
            bhat: None,
            is_collinear: true,
            antipodal_angle_only: false,
            antipodal_drop: false,
        };

        let diagnostics = finalize_track(
            TrackKind::Top1,
            &[step, step, step],
            0,
            0,
            &NormalizedVec8Summary {
                normalized_doc: vec![],
                normalized_ans: vec![],
                normalized_count: 0,
                max_norm_err: 0.0,
                vec8_total: 6,
                excluded_reason: None,
            },
            None,
        );

        assert_eq!(diagnostics.wandering.ratio.value, Some(0.0));
        assert_eq!(diagnostics.wandering.degenerate_path, Some(true));
        assert_eq!(diagnostics.wandering.degenerate_path_rate_numerator, 1);
        assert_eq!(diagnostics.wandering.degenerate_path_rate_denominator, 1);
    }

    #[test]
    fn trimmed_renorm_failure_path_sets_expected_reason() {
        let candidates = [
            {
                let mut out = [0.0_f64; ROTOR_DIM];
                out[0] = 1.0;
                out
            },
            {
                let mut out = [0.0_f64; ROTOR_DIM];
                out[0] = -1.0;
                out
            },
        ];
        let (mean_r, pre_norm) = mean_rotor29(&candidates);
        assert_eq!(pre_norm, 0.0);
        let result = build_trimmed_step_signal(&mean_r, &[1.0]);
        assert_eq!(result, Err(ExcludedReason::TrimmedBestZeroOrNonfiniteNorm));
    }

    #[test]
    fn trimmed_failure_does_not_change_top1_output() {
        let top1 = finalize_track(
            TrackKind::Top1,
            &[StepSignal {
                theta: Some(1.0),
                rotor: Some({
                    let mut out = [0.0_f64; ROTOR_DIM];
                    out[0] = 1.0;
                    out
                }),
                bhat: None,
                is_collinear: false,
                antipodal_angle_only: false,
                antipodal_drop: false,
            }],
            0,
            0,
            &NormalizedVec8Summary {
                normalized_doc: vec![],
                normalized_ans: vec![],
                normalized_count: 0,
                max_norm_err: 0.0,
                vec8_total: 2,
                excluded_reason: None,
            },
            None,
        );
        assert_eq!(top1.excluded_reason, None);
    }

    #[test]
    fn repeated_runs_are_deterministic() {
        let input = RotorDiagnosticsInput {
            sample_id: 4,
            links: sample_links(4, 3, 3, &[(0, 0, 1), (1, 1, 1), (2, 2, 1)]),
            doc_vec8: vec![e(0), e(1), e(2)],
            ans_vec8: vec![e(1), e(2), e(0)],
        };

        let a = compute_rotor_diagnostics(&input).expect("A");
        let b = compute_rotor_diagnostics(&input).expect("B");
        assert_eq!(a, b);
    }

    #[test]
    fn m1_missing_reason_precedence_for_top1() {
        let missing_top1 = RotorDiagnosticsInput {
            sample_id: 5,
            links: sample_links(5, 1, 1, &[(0, 0, 2)]),
            doc_vec8: vec![e(0)],
            ans_vec8: vec![e(0)],
        };
        let out = compute_rotor_diagnostics(&missing_top1).expect("missing_top1");
        assert_eq!(
            out.top1.max_theta.metric_missing_reason,
            Some(MetricMissingReason::MissingTop1Link)
        );

        let missing_link = RotorDiagnosticsInput {
            sample_id: 6,
            links: sample_links(6, 1, 1, &[]),
            doc_vec8: vec![e(0)],
            ans_vec8: vec![e(0)],
        };
        let out = compute_rotor_diagnostics(&missing_link).expect("missing_link");
        assert_eq!(
            out.top1.max_theta.metric_missing_reason,
            Some(MetricMissingReason::MissingLinksForTheta)
        );

        let synthetic = finalize_track(
            TrackKind::Top1,
            &[StepSignal::missing()],
            0,
            0,
            &NormalizedVec8Summary {
                normalized_doc: vec![],
                normalized_ans: vec![],
                normalized_count: 0,
                max_norm_err: 0.0,
                vec8_total: 2,
                excluded_reason: None,
            },
            None,
        );
        assert_eq!(
            synthetic.max_theta.metric_missing_reason,
            Some(MetricMissingReason::MissingTheta)
        );
    }
}
