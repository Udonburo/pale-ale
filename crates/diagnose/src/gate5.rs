use crate::gate4::{Gate4MetadataInputV1, Gate4RunInputV1, Gate4SampleInputV1, Gate4Variant};
use pale_ale_rotor::{
    embed_simple29_to_even128, inner, left_fold_mul_time_reversed_normalize_once, normalize_vec8,
    simple_rotor29_doc_to_ans, Even128, EvenError, RotorConfig, RotorError, RotorStep, Vec8Error,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

const SPEC_HASH_RAW_INPUT_ID: &str = "spec_text_raw_utf8_v1";
const SPEC_HASH_INPUT_ID: &str = "spec_text_utf8_lf_v1";
const FLOAT_FORMAT_ID: &str = "sci_17e_v1";
const TOKEN_TELEMETRY_SCHEMA_ID: &str = "gate5_token_telemetry_csv_v1";
const SAMPLE_SUMMARY_SCHEMA_ID: &str = "gate5_sample_summary_csv_v1";
const TRANSITION_LABEL_MODE_ID: &str = "max_pair_v1";
const EDGE_OUTCOME_ENUM_ID: &str = "gate5_edge_outcome_v1";
const LOOP_OUTCOME_ENUM_ID: &str = "gate5_loop_outcome_v1";
const SCORE_MISSING_SENTINEL_ID: &str = "empty_string_v1";
const TAU_WEDGE_V0: f64 = 1e-6;
const TAU_ANTIPODAL_DOT_V0: f64 = 1.0 - 1e-6;

pub const GATE5_SPEC_VERSION: &str = "v0.1.0-ssot.draft.0";
pub const GATE5_METHOD_ID: &str = "transport_loop_residual_experiment_v1";
pub const GATE5_PRIMARY_METRIC_ID: &str = "rotor_loop_chordal_v1";
pub const GATE5_TOKEN_TELEMETRY_CSV_COLUMNS_V1: &[&str] = &[
    "run_id",
    "sample_id",
    "variant",
    "world_type",
    "step",
    "absolute_pos",
    "token_id",
    "token_text",
    "answer_char_start",
    "answer_char_end",
    "label_token",
    "label_transition",
    "defect_span_id",
    "label_coverage_ratio",
    "exact_token_match_ratio",
    "transition_missing_reason",
    "edge_outcome_r1_v_to_splus",
    "edge_outcome_r2_splus_to_sminus",
    "edge_outcome_r3_sminus_to_v",
    "loop_outcome",
    "score_A_logprob",
    "score_B_entropy",
    "score_E_v_sminus_vnext",
    "score_F_loop",
    "rotor_loop_chordal_v1",
    "rotor_loop_nonscalar_norm_v1",
];
pub const GATE5_SAMPLE_SUMMARY_CSV_COLUMNS_V1: &[&str] = &[
    "run_id",
    "sample_id",
    "variant",
    "world_type",
    "n_token_steps",
    "n_transition_steps",
    "n_loop_steps_valid",
    "n_loop_steps_missing",
    "positive_token_count",
    "positive_transition_count",
    "label_coverage_ratio",
    "exact_token_match_ratio",
    "triplets_sha256",
    "labels_sha256",
    "auprc_A",
    "auprc_B",
    "auprc_E",
    "auprc_F",
    "auprc_rotor_loop_chordal_v1",
    "best_token_baseline_name",
    "delta_auprc_rotor_loop_chordal_v1_vs_F",
    "hit_at_10_F",
    "hit_at_10_rotor_loop_chordal_v1",
];
pub const GATE5_DIAGNOSTIC_TOKEN_CSV_COLUMNS_V1: &[&str] = &[
    "sample_id",
    "step",
    "absolute_pos",
    "token_id",
    "token_text",
    "label_token",
    "norm_status_v",
    "norm_status_splus",
    "norm_status_sminus",
    "input_norm_v",
    "input_norm_splus",
    "input_norm_sminus",
    "dot_v_splus",
    "dot_splus_sminus",
    "dot_sminus_v",
    "chordal_v_splus",
    "chordal_splus_sminus",
    "chordal_sminus_v",
    "edge_outcome_r1_v_to_splus",
    "edge_outcome_r2_splus_to_sminus",
    "edge_outcome_r3_sminus_to_v",
    "edge_chordal_r1_v_to_splus",
    "edge_chordal_r2_splus_to_sminus",
    "edge_chordal_r3_sminus_to_v",
    "loop_outcome",
    "rotor_loop_chordal_v1",
    "rotor_loop_nonscalar_norm_v1",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Gate5IdentityInput {
    pub run_id: String,
    pub dataset_revision_id: String,
    pub dataset_hash_blake3: String,
    pub spec_hash_raw_blake3: String,
    pub spec_hash_blake3: String,
    pub evaluation_mode_id: String,
    pub code_git_commit: String,
    pub build_target_triple: String,
    pub rustc_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gate5ArtifactPaths {
    pub manifest_json: PathBuf,
    pub token_telemetry_csv: PathBuf,
    pub sample_summary_csv: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gate5RunSummary {
    pub n_samples_total: usize,
    pub n_token_rows_total: usize,
    pub n_transition_rows_total: usize,
    pub n_loop_rows_valid: usize,
    pub n_loop_rows_missing: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate5RunOutput {
    pub run_id: String,
    pub spec_version: String,
    pub summary: Gate5RunSummary,
    pub artifact_paths: Gate5ArtifactPaths,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gate5DiagnosticOutput {
    pub diagnostic_csv_path: PathBuf,
    pub n_rows: usize,
}

#[derive(Debug)]
pub enum Gate5OrchestratorError {
    JsonParse(serde_json::Error),
    DuplicateSampleId {
        sample_id: u64,
    },
    MissingTokenSteps {
        sample_id: u64,
    },
    DuplicateStep {
        sample_id: u64,
        step: usize,
    },
    NonContiguousStep {
        sample_id: u64,
        expected: usize,
        actual: usize,
    },
    InvalidLabel {
        sample_id: u64,
        step: usize,
        label: u8,
    },
    InvalidRange {
        sample_id: u64,
        field: &'static str,
        min_inclusive: f64,
        max_inclusive: Option<f64>,
        value: f64,
    },
    InvalidEvaluationMode(String),
    InvalidFloat {
        sample_id: Option<u64>,
        step: Option<usize>,
        field: &'static str,
        value: f64,
    },
    InvalidVec8Dim {
        sample_id: u64,
        step: usize,
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    Io(std::io::Error),
    JsonWrite(serde_json::Error),
    ManifestRead(std::io::Error),
    ManifestValidation(Gate5ManifestValidationError),
}

impl fmt::Display for Gate5OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JsonParse(err) => write!(f, "failed to parse Gate5 JSON input: {}", err),
            Self::DuplicateSampleId { sample_id } => {
                write!(f, "duplicate Gate5 sample_id {}", sample_id)
            }
            Self::MissingTokenSteps { sample_id } => {
                write!(f, "sample {} has no token_steps", sample_id)
            }
            Self::DuplicateStep { sample_id, step } => {
                write!(f, "sample {} has duplicate step {}", sample_id, step)
            }
            Self::NonContiguousStep {
                sample_id,
                expected,
                actual,
            } => write!(
                f,
                "sample {} has non-contiguous step sequence: expected {}, got {}",
                sample_id, expected, actual
            ),
            Self::InvalidLabel {
                sample_id,
                step,
                label,
            } => write!(
                f,
                "sample {} step {} has invalid label_token {} (expected 0 or 1)",
                sample_id, step, label
            ),
            Self::InvalidRange {
                sample_id,
                field,
                min_inclusive,
                max_inclusive,
                value,
            } => {
                if let Some(max) = max_inclusive {
                    write!(
                        f,
                        "sample {} {} out of range [{}, {}]: {}",
                        sample_id, field, min_inclusive, max, value
                    )
                } else {
                    write!(
                        f,
                        "sample {} {} below minimum {}: {}",
                        sample_id, field, min_inclusive, value
                    )
                }
            }
            Self::InvalidEvaluationMode(value) => write!(
                f,
                "invalid evaluation_mode_id '{}': expected supervised_v1 or unsupervised_v1",
                value
            ),
            Self::InvalidFloat {
                sample_id,
                step,
                field,
                value,
            } => write!(
                f,
                "non-finite float for {} at sample {:?} step {:?}: {}",
                field, sample_id, step, value
            ),
            Self::InvalidVec8Dim {
                sample_id,
                step,
                field,
                expected,
                actual,
            } => write!(
                f,
                "sample {} step {} {} has invalid dimension: expected {}, got {}",
                sample_id, step, field, expected, actual
            ),
            Self::Io(err) => write!(f, "io error: {}", err),
            Self::JsonWrite(err) => write!(f, "json serialization error: {}", err),
            Self::ManifestRead(err) => write!(f, "failed to read manifest.json: {}", err),
            Self::ManifestValidation(err) => write!(f, "manifest validation error: {}", err),
        }
    }
}

impl std::error::Error for Gate5OrchestratorError {}

impl From<std::io::Error> for Gate5OrchestratorError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for Gate5OrchestratorError {
    fn from(value: serde_json::Error) -> Self {
        Self::JsonWrite(value)
    }
}

#[derive(Debug)]
pub enum Gate5ManifestValidationError {
    Utf8(String),
    Json(serde_json::Error),
    NotObject,
    MissingKey(&'static str),
    InvalidFixedString {
        key: &'static str,
        expected: &'static str,
        actual: Option<String>,
    },
}

impl fmt::Display for Gate5ManifestValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utf8(err) => write!(f, "manifest is not valid utf-8: {}", err),
            Self::Json(err) => write!(f, "manifest is not valid json: {}", err),
            Self::NotObject => write!(f, "manifest root must be a JSON object"),
            Self::MissingKey(key) => write!(f, "manifest missing required key '{}'", key),
            Self::InvalidFixedString {
                key,
                expected,
                actual,
            } => write!(
                f,
                "manifest key '{}' must equal '{}' but got {:?}",
                key, expected, actual
            ),
        }
    }
}

impl std::error::Error for Gate5ManifestValidationError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransitionMissingReason {
    None,
    FinalStepNoSuccessor,
}

impl TransitionMissingReason {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FinalStepNoSuccessor => "final_step_no_successor",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EdgeOutcome {
    Materialized,
    CollinearIdentity,
    AntipodalAngleOnly,
    Vec8NonFiniteComponent,
    Vec8ZeroOrNonFiniteNorm,
    RotorNonFiniteTheta,
    RotorRenormFailure,
}

impl EdgeOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::Materialized => "materialized",
            Self::CollinearIdentity => "collinear_identity",
            Self::AntipodalAngleOnly => "antipodal_angle_only",
            Self::Vec8NonFiniteComponent => "vec8_nonfinite_component",
            Self::Vec8ZeroOrNonFiniteNorm => "vec8_zero_or_nonfinite_norm",
            Self::RotorNonFiniteTheta => "rotor_nonfinite_theta",
            Self::RotorRenormFailure => "rotor_renorm_failure",
        }
    }

    fn is_materialized(self) -> bool {
        matches!(self, Self::Materialized | Self::CollinearIdentity)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LoopOutcome {
    None,
    PartialLoopMissing,
    InvalidLoopProduct,
}

impl LoopOutcome {
    fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::PartialLoopMissing => "partial_loop_missing",
            Self::InvalidLoopProduct => "invalid_loop_product",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct ValidatedSample {
    sample_id: u64,
    variant: Gate4Variant,
    world_type: Option<String>,
    exact_token_match_ratio: f64,
    label_coverage_ratio: f64,
    triplets_sha256: String,
    labels_sha256: String,
    token_steps: Vec<ValidatedTokenStep>,
}

#[derive(Clone, Debug, PartialEq)]
struct ValidatedTokenStep {
    step: usize,
    absolute_pos: usize,
    answer_char_start: Option<usize>,
    answer_char_end: Option<usize>,
    token_id: u64,
    token_text: String,
    label_token: u8,
    defect_span_id: Option<String>,
    v_8d: [f64; 8],
    splus_8d: [f64; 8],
    sminus_8d: [f64; 8],
    baseline_logprob: f64,
    baseline_entropy: f64,
}

#[derive(Clone, Debug, PartialEq)]
struct Gate5TokenTelemetryRow {
    sample_id: u64,
    variant: Gate4Variant,
    world_type: Option<String>,
    step: usize,
    absolute_pos: usize,
    token_id: u64,
    token_text: String,
    answer_char_start: Option<usize>,
    answer_char_end: Option<usize>,
    label_token: u8,
    label_transition: u8,
    defect_span_id: Option<String>,
    label_coverage_ratio: f64,
    exact_token_match_ratio: f64,
    transition_missing_reason: TransitionMissingReason,
    edge_outcome_r1: EdgeOutcome,
    edge_outcome_r2: EdgeOutcome,
    edge_outcome_r3: EdgeOutcome,
    loop_outcome: LoopOutcome,
    score_a: f64,
    score_b: f64,
    score_e: Option<f64>,
    score_f: f64,
    rotor_loop_chordal: Option<f64>,
    rotor_loop_nonscalar_norm: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
struct Gate5SampleSummaryRow {
    sample_id: u64,
    variant: Gate4Variant,
    world_type: Option<String>,
    n_token_steps: usize,
    n_transition_steps: usize,
    n_loop_steps_valid: usize,
    n_loop_steps_missing: usize,
    positive_token_count: usize,
    positive_transition_count: usize,
    label_coverage_ratio: f64,
    exact_token_match_ratio: f64,
    triplets_sha256: String,
    labels_sha256: String,
    auprc_a: Option<f64>,
    auprc_b: Option<f64>,
    auprc_e: Option<f64>,
    auprc_f: Option<f64>,
    auprc_rotor_loop_chordal: Option<f64>,
    best_token_baseline_name: Option<&'static str>,
    delta_auprc_rotor_loop_chordal_vs_f: Option<f64>,
    hit_at_10_f: Option<usize>,
    hit_at_10_rotor_loop_chordal: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
struct Gate5ManifestJson {
    spec_version: String,
    method_id: String,
    spec_hash_raw_blake3: String,
    spec_hash_raw_input_id: String,
    spec_hash_blake3: String,
    spec_hash_input_id: String,
    dataset_revision_id: String,
    dataset_hash_blake3: String,
    code_git_commit: String,
    build_target_triple: String,
    rustc_version: String,
    evaluation_mode_id: String,
    run_id: String,
    n_samples_total: usize,
    n_token_rows_total: usize,
    n_transition_rows_total: usize,
    n_loop_rows_valid: usize,
    n_loop_rows_missing: usize,
    model_id: String,
    model_revision: String,
    seed: u64,
    perm_r: Option<u64>,
    primary_score: Option<String>,
    proj_id: String,
    splus_def_id: String,
    sminus_def_id: String,
    token_telemetry_schema_id: String,
    sample_summary_schema_id: String,
    float_format_id: String,
    transition_label_mode_id: String,
    edge_outcome_enum_id: String,
    loop_outcome_enum_id: String,
    score_missing_sentinel_id: String,
    input_json_sha256: String,
    token_telemetry_sha256: String,
    sample_summary_sha256: String,
}

#[derive(Clone, Copy, Debug)]
struct EdgeComputation {
    outcome: EdgeOutcome,
    rotor: Option<Even128>,
    edge_chordal_identity: Option<f64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ComparatorNormStatus {
    Materialized,
    NonFiniteComponent,
    ZeroOrNonFiniteNorm,
}

impl ComparatorNormStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Materialized => "materialized",
            Self::NonFiniteComponent => "nonfinite_component",
            Self::ZeroOrNonFiniteNorm => "zero_or_nonfinite_norm",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
struct Gate5DiagnosticTokenRow {
    sample_id: u64,
    step: usize,
    absolute_pos: usize,
    token_id: u64,
    token_text: String,
    label_token: u8,
    norm_status_v: ComparatorNormStatus,
    norm_status_splus: ComparatorNormStatus,
    norm_status_sminus: ComparatorNormStatus,
    input_norm_v: f64,
    input_norm_splus: f64,
    input_norm_sminus: f64,
    dot_v_splus: Option<f64>,
    dot_splus_sminus: Option<f64>,
    dot_sminus_v: Option<f64>,
    chordal_v_splus: Option<f64>,
    chordal_splus_sminus: Option<f64>,
    chordal_sminus_v: Option<f64>,
    edge_outcome_r1: EdgeOutcome,
    edge_outcome_r2: EdgeOutcome,
    edge_outcome_r3: EdgeOutcome,
    edge_chordal_r1: Option<f64>,
    edge_chordal_r2: Option<f64>,
    edge_chordal_r3: Option<f64>,
    loop_outcome: LoopOutcome,
    rotor_loop_chordal: Option<f64>,
    rotor_loop_nonscalar_norm: Option<f64>,
}

pub fn run_gate5_and_write<P: AsRef<Path>>(
    out_dir: P,
    input_json_bytes: &[u8],
    identity: &Gate5IdentityInput,
) -> Result<Gate5RunOutput, Gate5OrchestratorError> {
    if identity.evaluation_mode_id != "supervised_v1"
        && identity.evaluation_mode_id != "unsupervised_v1"
    {
        return Err(Gate5OrchestratorError::InvalidEvaluationMode(
            identity.evaluation_mode_id.clone(),
        ));
    }

    let parsed: Gate4RunInputV1 =
        serde_json::from_slice(input_json_bytes).map_err(Gate5OrchestratorError::JsonParse)?;
    let samples = validate_samples(parsed.samples)?;

    let mut token_rows = Vec::new();
    let mut sample_rows = Vec::new();
    let mut n_loop_rows_valid = 0usize;
    let mut n_loop_rows_missing = 0usize;

    for sample in &samples {
        let (sample_token_rows, sample_summary) = compute_sample_outputs(sample);
        n_loop_rows_valid += sample_summary.n_loop_steps_valid;
        n_loop_rows_missing += sample_summary.n_loop_steps_missing;
        token_rows.extend(sample_token_rows);
        sample_rows.push(sample_summary);
    }

    token_rows.sort_by(|left, right| {
        left.sample_id
            .cmp(&right.sample_id)
            .then(left.step.cmp(&right.step))
    });
    sample_rows.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));

    let summary = Gate5RunSummary {
        n_samples_total: sample_rows.len(),
        n_token_rows_total: token_rows.len(),
        n_transition_rows_total: token_rows
            .iter()
            .filter(|row| row.transition_missing_reason == TransitionMissingReason::None)
            .count(),
        n_loop_rows_valid,
        n_loop_rows_missing,
    };

    let token_csv = build_token_telemetry_csv(&identity.run_id, &token_rows)?;
    let sample_csv = build_sample_summary_csv(&identity.run_id, &sample_rows)?;

    let input_json_sha256 = sha256_hex(input_json_bytes);
    let token_telemetry_sha256 = sha256_hex(token_csv.as_bytes());
    let sample_summary_sha256 = sha256_hex(sample_csv.as_bytes());

    let manifest = build_manifest(
        identity,
        &parsed.metadata,
        &summary,
        &input_json_sha256,
        &token_telemetry_sha256,
        &sample_summary_sha256,
    );

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    validate_gate5_manifest_json(&manifest_bytes)
        .map_err(Gate5OrchestratorError::ManifestValidation)?;

    let manifest_path = out_dir.join("manifest.json");
    let token_csv_path = out_dir.join("gate5_token_telemetry.csv");
    let sample_csv_path = out_dir.join("gate5_sample_summary.csv");

    write_bytes_lf(&manifest_path, &manifest_bytes)?;
    write_string_lf(&token_csv_path, &token_csv)?;
    write_string_lf(&sample_csv_path, &sample_csv)?;

    let manifest_read = fs::read(&manifest_path).map_err(Gate5OrchestratorError::ManifestRead)?;
    validate_gate5_manifest_json(&manifest_read)
        .map_err(Gate5OrchestratorError::ManifestValidation)?;

    Ok(Gate5RunOutput {
        run_id: identity.run_id.clone(),
        spec_version: GATE5_SPEC_VERSION.to_string(),
        summary,
        artifact_paths: Gate5ArtifactPaths {
            manifest_json: manifest_path,
            token_telemetry_csv: token_csv_path,
            sample_summary_csv: sample_csv_path,
        },
    })
}

pub fn run_gate5_diagnostics_and_write<P: AsRef<Path>>(
    out_csv_path: P,
    input_json_bytes: &[u8],
) -> Result<Gate5DiagnosticOutput, Gate5OrchestratorError> {
    let parsed: Gate4RunInputV1 =
        serde_json::from_slice(input_json_bytes).map_err(Gate5OrchestratorError::JsonParse)?;
    let samples = validate_samples(parsed.samples)?;
    let mut rows = Vec::new();
    for sample in &samples {
        rows.extend(compute_diagnostic_rows(sample));
    }
    rows.sort_by(|left, right| {
        left.sample_id
            .cmp(&right.sample_id)
            .then(left.step.cmp(&right.step))
    });
    let csv = build_diagnostic_token_csv(&rows)?;
    let out_csv_path = out_csv_path.as_ref();
    write_string_lf(out_csv_path, &csv)?;
    Ok(Gate5DiagnosticOutput {
        diagnostic_csv_path: out_csv_path.to_path_buf(),
        n_rows: rows.len(),
    })
}

fn validate_samples(
    mut samples: Vec<Gate4SampleInputV1>,
) -> Result<Vec<ValidatedSample>, Gate5OrchestratorError> {
    samples.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let mut out = Vec::with_capacity(samples.len());
    let mut previous_sample_id: Option<u64> = None;
    for sample in samples {
        if previous_sample_id == Some(sample.sample_id) {
            return Err(Gate5OrchestratorError::DuplicateSampleId {
                sample_id: sample.sample_id,
            });
        }
        previous_sample_id = Some(sample.sample_id);
        if sample.token_steps.is_empty() {
            return Err(Gate5OrchestratorError::MissingTokenSteps {
                sample_id: sample.sample_id,
            });
        }
        ensure_finite(
            sample.exact_token_match_ratio,
            Some(sample.sample_id),
            None,
            "exact_token_match_ratio",
        )?;
        ensure_range(
            sample.exact_token_match_ratio,
            sample.sample_id,
            "exact_token_match_ratio",
            0.0,
            None,
        )?;
        ensure_finite(
            sample.label_coverage_ratio,
            Some(sample.sample_id),
            None,
            "label_coverage_ratio",
        )?;
        ensure_range(
            sample.label_coverage_ratio,
            sample.sample_id,
            "label_coverage_ratio",
            0.0,
            Some(1.0),
        )?;

        let mut token_steps = sample.token_steps;
        token_steps.sort_by(|left, right| left.step.cmp(&right.step));
        let mut previous_step: Option<usize> = None;
        let mut validated_steps = Vec::with_capacity(token_steps.len());
        for step in token_steps {
            if previous_step == Some(step.step) {
                return Err(Gate5OrchestratorError::DuplicateStep {
                    sample_id: sample.sample_id,
                    step: step.step,
                });
            }
            if let Some(prev) = previous_step {
                if step.step != prev + 1 {
                    return Err(Gate5OrchestratorError::NonContiguousStep {
                        sample_id: sample.sample_id,
                        expected: prev + 1,
                        actual: step.step,
                    });
                }
            }
            previous_step = Some(step.step);
            if step.label_token > 1 {
                return Err(Gate5OrchestratorError::InvalidLabel {
                    sample_id: sample.sample_id,
                    step: step.step,
                    label: step.label_token,
                });
            }
            ensure_finite(
                step.baseline_logprob,
                Some(sample.sample_id),
                Some(step.step),
                "baseline_logprob",
            )?;
            ensure_finite(
                step.baseline_entropy,
                Some(sample.sample_id),
                Some(step.step),
                "baseline_entropy",
            )?;
            validated_steps.push(ValidatedTokenStep {
                step: step.step,
                absolute_pos: step.absolute_pos,
                answer_char_start: step.answer_char_start,
                answer_char_end: step.answer_char_end,
                token_id: step.token_id,
                token_text: step.token_str,
                label_token: step.label_token,
                defect_span_id: step.defect_span_id.filter(|value| !value.is_empty()),
                v_8d: vec8_to_array(sample.sample_id, step.step, "V_8d", &step.v_8d)?,
                splus_8d: vec8_to_array(sample.sample_id, step.step, "Splus_8d", &step.splus_8d)?,
                sminus_8d: vec8_to_array(
                    sample.sample_id,
                    step.step,
                    "Sminus_8d",
                    &step.sminus_8d,
                )?,
                baseline_logprob: step.baseline_logprob,
                baseline_entropy: step.baseline_entropy,
            });
        }

        out.push(ValidatedSample {
            sample_id: sample.sample_id,
            variant: sample.variant,
            world_type: sample.world_type,
            exact_token_match_ratio: sample.exact_token_match_ratio,
            label_coverage_ratio: sample.label_coverage_ratio,
            triplets_sha256: sample.triplets_sha256,
            labels_sha256: sample.labels_sha256,
            token_steps: validated_steps,
        });
    }
    Ok(out)
}

fn vec8_to_array(
    sample_id: u64,
    step: usize,
    field: &'static str,
    values: &[f64],
) -> Result<[f64; 8], Gate5OrchestratorError> {
    if values.len() != 8 {
        return Err(Gate5OrchestratorError::InvalidVec8Dim {
            sample_id,
            step,
            field,
            expected: 8,
            actual: values.len(),
        });
    }
    let mut out = [0.0_f64; 8];
    for (idx, value) in values.iter().copied().enumerate() {
        ensure_finite(value, Some(sample_id), Some(step), field)?;
        out[idx] = value;
    }
    Ok(out)
}

fn ensure_finite(
    value: f64,
    sample_id: Option<u64>,
    step: Option<usize>,
    field: &'static str,
) -> Result<(), Gate5OrchestratorError> {
    if !value.is_finite() {
        return Err(Gate5OrchestratorError::InvalidFloat {
            sample_id,
            step,
            field,
            value,
        });
    }
    Ok(())
}

fn ensure_range(
    value: f64,
    sample_id: u64,
    field: &'static str,
    min_inclusive: f64,
    max_inclusive: Option<f64>,
) -> Result<(), Gate5OrchestratorError> {
    if value < min_inclusive {
        return Err(Gate5OrchestratorError::InvalidRange {
            sample_id,
            field,
            min_inclusive,
            max_inclusive,
            value,
        });
    }
    if let Some(max) = max_inclusive {
        if value > max {
            return Err(Gate5OrchestratorError::InvalidRange {
                sample_id,
                field,
                min_inclusive,
                max_inclusive,
                value,
            });
        }
    }
    Ok(())
}

fn compute_sample_outputs(
    sample: &ValidatedSample,
) -> (Vec<Gate5TokenTelemetryRow>, Gate5SampleSummaryRow) {
    let n = sample.token_steps.len();
    let mut labels_token = Vec::with_capacity(n);
    let mut score_a = Vec::with_capacity(n);
    let mut score_b = Vec::with_capacity(n);
    let mut score_f = Vec::with_capacity(n);
    let mut score_chordal = Vec::with_capacity(n);
    let mut labels_transition = Vec::with_capacity(n.saturating_sub(1));
    let mut score_e = Vec::with_capacity(n.saturating_sub(1));
    let mut rows = Vec::with_capacity(n);
    let mut n_loop_steps_valid = 0usize;
    let mut n_loop_steps_missing = 0usize;

    for token in &sample.token_steps {
        labels_token.push(token.label_token);
        score_a.push(-token.baseline_logprob);
        score_b.push(token.baseline_entropy);
        score_f.push(
            d_proj(&token.v_8d, &token.splus_8d)
                + d_proj(&token.splus_8d, &token.sminus_8d)
                + d_proj(&token.sminus_8d, &token.v_8d),
        );
    }

    for t in 0..n.saturating_sub(1) {
        let current = &sample.token_steps[t];
        let next = &sample.token_steps[t + 1];
        labels_transition.push(current.label_token.max(next.label_token));
        score_e.push(
            d_proj(&current.v_8d, &current.sminus_8d) + d_proj(&current.sminus_8d, &next.v_8d),
        );
    }

    for (idx, token) in sample.token_steps.iter().enumerate() {
        let r1 = compute_edge(token.v_8d, token.splus_8d);
        let r2 = compute_edge(token.splus_8d, token.sminus_8d);
        let r3 = compute_edge(token.sminus_8d, token.v_8d);
        let (loop_outcome, chordal, nonscalar_norm) = compute_loop_metrics(&r1, &r2, &r3);
        if loop_outcome == LoopOutcome::None {
            n_loop_steps_valid += 1;
        } else {
            n_loop_steps_missing += 1;
        }
        score_chordal.push(chordal);

        let (label_transition, transition_missing_reason, e_value) = if idx + 1 < n {
            (
                labels_transition[idx],
                TransitionMissingReason::None,
                Some(score_e[idx]),
            )
        } else {
            (0, TransitionMissingReason::FinalStepNoSuccessor, None)
        };

        rows.push(Gate5TokenTelemetryRow {
            sample_id: sample.sample_id,
            variant: sample.variant,
            world_type: sample.world_type.clone(),
            step: token.step,
            absolute_pos: token.absolute_pos,
            token_id: token.token_id,
            token_text: token.token_text.clone(),
            answer_char_start: token.answer_char_start,
            answer_char_end: token.answer_char_end,
            label_token: token.label_token,
            label_transition,
            defect_span_id: token.defect_span_id.clone(),
            label_coverage_ratio: sample.label_coverage_ratio,
            exact_token_match_ratio: sample.exact_token_match_ratio,
            transition_missing_reason,
            edge_outcome_r1: r1.outcome,
            edge_outcome_r2: r2.outcome,
            edge_outcome_r3: r3.outcome,
            loop_outcome,
            score_a: score_a[idx],
            score_b: score_b[idx],
            score_e: e_value,
            score_f: score_f[idx],
            rotor_loop_chordal: chordal,
            rotor_loop_nonscalar_norm: nonscalar_norm,
        });
    }

    let auprc_a = average_precision(&labels_token, &score_a);
    let auprc_b = average_precision(&labels_token, &score_b);
    let auprc_e = average_precision(&labels_transition, &score_e);
    let auprc_f = average_precision(&labels_token, &score_f);
    let auprc_rotor_loop_chordal = average_precision_opt(&labels_token, &score_chordal);
    let best_token_baseline_name = match (auprc_a, auprc_b) {
        (Some(left), Some(right)) => Some(if left >= right { "A" } else { "B" }),
        (Some(_), None) => Some("A"),
        (None, Some(_)) => Some("B"),
        (None, None) => None,
    };
    let delta_auprc_rotor_loop_chordal_vs_f = match (auprc_rotor_loop_chordal, auprc_f) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    };
    let hit_at_10_f = hit_at_k_optional(&labels_token, &score_f, 10);
    let hit_at_10_rotor_loop_chordal = hit_at_k_opt(&labels_token, &score_chordal, 10);

    let summary = Gate5SampleSummaryRow {
        sample_id: sample.sample_id,
        variant: sample.variant,
        world_type: sample.world_type.clone(),
        n_token_steps: n,
        n_transition_steps: labels_transition.len(),
        n_loop_steps_valid,
        n_loop_steps_missing,
        positive_token_count: labels_token.iter().filter(|&&y| y == 1).count(),
        positive_transition_count: labels_transition.iter().filter(|&&y| y == 1).count(),
        label_coverage_ratio: sample.label_coverage_ratio,
        exact_token_match_ratio: sample.exact_token_match_ratio,
        triplets_sha256: sample.triplets_sha256.clone(),
        labels_sha256: sample.labels_sha256.clone(),
        auprc_a,
        auprc_b,
        auprc_e,
        auprc_f,
        auprc_rotor_loop_chordal,
        best_token_baseline_name,
        delta_auprc_rotor_loop_chordal_vs_f,
        hit_at_10_f,
        hit_at_10_rotor_loop_chordal,
    };

    (rows, summary)
}

fn compute_diagnostic_rows(sample: &ValidatedSample) -> Vec<Gate5DiagnosticTokenRow> {
    let mut rows = Vec::with_capacity(sample.token_steps.len());
    for token in &sample.token_steps {
        let (norm_status_v, normalized_v, input_norm_v) = normalize_for_comparator(token.v_8d);
        let (norm_status_splus, normalized_splus, input_norm_splus) =
            normalize_for_comparator(token.splus_8d);
        let (norm_status_sminus, normalized_sminus, input_norm_sminus) =
            normalize_for_comparator(token.sminus_8d);

        let dot_v_splus = normalized_pair_dot(&normalized_v, &normalized_splus);
        let dot_splus_sminus = normalized_pair_dot(&normalized_splus, &normalized_sminus);
        let dot_sminus_v = normalized_pair_dot(&normalized_sminus, &normalized_v);
        let chordal_v_splus = normalized_pair_chordal(&normalized_v, &normalized_splus);
        let chordal_splus_sminus = normalized_pair_chordal(&normalized_splus, &normalized_sminus);
        let chordal_sminus_v = normalized_pair_chordal(&normalized_sminus, &normalized_v);

        let r1 = compute_edge(token.v_8d, token.splus_8d);
        let r2 = compute_edge(token.splus_8d, token.sminus_8d);
        let r3 = compute_edge(token.sminus_8d, token.v_8d);
        let (loop_outcome, chordal, nonscalar_norm) = compute_loop_metrics(&r1, &r2, &r3);

        rows.push(Gate5DiagnosticTokenRow {
            sample_id: sample.sample_id,
            step: token.step,
            absolute_pos: token.absolute_pos,
            token_id: token.token_id,
            token_text: token.token_text.clone(),
            label_token: token.label_token,
            norm_status_v,
            norm_status_splus,
            norm_status_sminus,
            input_norm_v,
            input_norm_splus,
            input_norm_sminus,
            dot_v_splus,
            dot_splus_sminus,
            dot_sminus_v,
            chordal_v_splus,
            chordal_splus_sminus,
            chordal_sminus_v,
            edge_outcome_r1: r1.outcome,
            edge_outcome_r2: r2.outcome,
            edge_outcome_r3: r3.outcome,
            edge_chordal_r1: r1.edge_chordal_identity,
            edge_chordal_r2: r2.edge_chordal_identity,
            edge_chordal_r3: r3.edge_chordal_identity,
            loop_outcome,
            rotor_loop_chordal: chordal,
            rotor_loop_nonscalar_norm: nonscalar_norm,
        });
    }
    rows
}

fn compute_edge(doc_vec8: [f64; 8], ans_vec8: [f64; 8]) -> EdgeComputation {
    let config = RotorConfig {
        tau_wedge: TAU_WEDGE_V0,
        tau_antipodal_dot: TAU_ANTIPODAL_DOT_V0,
    };
    match simple_rotor29_doc_to_ans(doc_vec8, ans_vec8, config) {
        Ok(RotorStep::Materialized {
            r29, is_collinear, ..
        }) => EdgeComputation {
            outcome: if is_collinear {
                EdgeOutcome::CollinearIdentity
            } else {
                EdgeOutcome::Materialized
            },
            rotor: Some(embed_simple29_to_even128(&r29)),
            edge_chordal_identity: Some((2.0 * (1.0 - r29[0].abs().min(1.0))).max(0.0).sqrt()),
        },
        Ok(RotorStep::AntipodalAngleOnly { .. }) => EdgeComputation {
            outcome: EdgeOutcome::AntipodalAngleOnly,
            rotor: None,
            edge_chordal_identity: None,
        },
        Err(RotorError::Vec8(Vec8Error::NonFiniteComponent)) => EdgeComputation {
            outcome: EdgeOutcome::Vec8NonFiniteComponent,
            rotor: None,
            edge_chordal_identity: None,
        },
        Err(RotorError::Vec8(Vec8Error::ZeroOrNonFiniteNorm)) => EdgeComputation {
            outcome: EdgeOutcome::Vec8ZeroOrNonFiniteNorm,
            rotor: None,
            edge_chordal_identity: None,
        },
        Err(RotorError::NonFiniteTheta) => EdgeComputation {
            outcome: EdgeOutcome::RotorNonFiniteTheta,
            rotor: None,
            edge_chordal_identity: None,
        },
        Err(RotorError::RenormFailure) => EdgeComputation {
            outcome: EdgeOutcome::RotorRenormFailure,
            rotor: None,
            edge_chordal_identity: None,
        },
    }
}

fn normalize_for_comparator(input: [f64; 8]) -> (ComparatorNormStatus, Option<[f64; 8]>, f64) {
    let norm = input.iter().map(|value| value * value).sum::<f64>().sqrt();
    match normalize_vec8(input) {
        Ok(value) => (ComparatorNormStatus::Materialized, Some(value), norm),
        Err(Vec8Error::NonFiniteComponent) => {
            (ComparatorNormStatus::NonFiniteComponent, None, norm)
        }
        Err(Vec8Error::ZeroOrNonFiniteNorm) => {
            (ComparatorNormStatus::ZeroOrNonFiniteNorm, None, norm)
        }
    }
}

fn normalized_pair_dot(left: &Option<[f64; 8]>, right: &Option<[f64; 8]>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()),
        _ => None,
    }
}

fn normalized_pair_chordal(left: &Option<[f64; 8]>, right: &Option<[f64; 8]>) -> Option<f64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(d_proj(left, right)),
        _ => None,
    }
}

fn compute_loop_metrics(
    r1: &EdgeComputation,
    r2: &EdgeComputation,
    r3: &EdgeComputation,
) -> (LoopOutcome, Option<f64>, Option<f64>) {
    if !(r1.outcome.is_materialized()
        && r2.outcome.is_materialized()
        && r3.outcome.is_materialized())
    {
        return (LoopOutcome::PartialLoopMissing, None, None);
    }

    let composed = match left_fold_mul_time_reversed_normalize_once(&[
        r1.rotor.expect("materialized edge rotor"),
        r2.rotor.expect("materialized edge rotor"),
        r3.rotor.expect("materialized edge rotor"),
    ]) {
        Ok(value) => value,
        Err(EvenError::NonFiniteNormSquared) | Err(EvenError::NonPositiveNormSquared) => {
            return (LoopOutcome::InvalidLoopProduct, None, None)
        }
    };

    let identity = Even128::identity();
    let a = inner(&composed, &identity).abs().min(1.0);
    let chordal = (2.0 * (1.0 - a)).max(0.0).sqrt();
    let nonscalar_norm = composed.coeffs[1..]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if !chordal.is_finite() || !nonscalar_norm.is_finite() {
        return (LoopOutcome::InvalidLoopProduct, None, None);
    }
    (LoopOutcome::None, Some(chordal), Some(nonscalar_norm))
}

fn average_precision(labels: &[u8], scores: &[f64]) -> Option<f64> {
    let n_pos = labels.iter().filter(|&&y| y == 1).count();
    if n_pos == 0 {
        return None;
    }
    let mut indexed: Vec<usize> = (0..scores.len()).collect();
    indexed.sort_by(|left, right| {
        scores[*right]
            .total_cmp(&scores[*left])
            .then(left.cmp(right))
    });

    let mut tp = 0usize;
    let mut fp = 0usize;
    let mut prev_recall = 0.0_f64;
    let mut ap = 0.0_f64;
    for idx in indexed {
        if labels[idx] == 1 {
            tp += 1;
        } else {
            fp += 1;
        }
        let recall = tp as f64 / n_pos as f64;
        let precision = tp as f64 / (tp + fp) as f64;
        ap += (recall - prev_recall) * precision;
        prev_recall = recall;
    }
    Some(ap)
}

fn average_precision_opt(labels: &[u8], scores: &[Option<f64>]) -> Option<f64> {
    let mut filtered_labels = Vec::new();
    let mut filtered_scores = Vec::new();
    for (label, score) in labels.iter().copied().zip(scores.iter().copied()) {
        if let Some(value) = score {
            filtered_labels.push(label);
            filtered_scores.push(value);
        }
    }
    average_precision(&filtered_labels, &filtered_scores)
}

fn hit_at_k(labels: &[u8], scores: &[f64], k: usize) -> usize {
    let mut indexed: Vec<usize> = (0..scores.len()).collect();
    indexed.sort_by(|left, right| {
        scores[*right]
            .total_cmp(&scores[*left])
            .then(left.cmp(right))
    });
    indexed
        .into_iter()
        .take(k)
        .filter(|&idx| labels[idx] == 1)
        .count()
}

fn hit_at_k_optional(labels: &[u8], scores: &[f64], k: usize) -> Option<usize> {
    if labels.iter().all(|&label| label == 0) {
        None
    } else {
        Some(hit_at_k(labels, scores, k))
    }
}

fn hit_at_k_opt(labels: &[u8], scores: &[Option<f64>], k: usize) -> Option<usize> {
    let mut indexed = Vec::new();
    for (idx, score) in scores.iter().copied().enumerate() {
        if let Some(value) = score {
            indexed.push((idx, value));
        }
    }
    if indexed.is_empty() || labels.iter().all(|&label| label == 0) {
        return None;
    }
    indexed.sort_by(|left, right| right.1.total_cmp(&left.1).then(left.0.cmp(&right.0)));
    Some(
        indexed
            .into_iter()
            .take(k)
            .filter(|(idx, _)| labels[*idx] == 1)
            .count(),
    )
}

fn dot_abs_clamped(left: &[f64; 8], right: &[f64; 8]) -> f64 {
    let inner: f64 = left.iter().zip(right.iter()).map(|(a, b)| *a * *b).sum();
    inner.abs().min(1.0)
}

fn d_proj(left: &[f64; 8], right: &[f64; 8]) -> f64 {
    (2.0 * (1.0 - dot_abs_clamped(left, right))).max(0.0).sqrt()
}

fn build_diagnostic_token_csv(
    rows: &[Gate5DiagnosticTokenRow],
) -> Result<String, Gate5OrchestratorError> {
    let mut out = String::new();
    out.push_str(&GATE5_DIAGNOSTIC_TOKEN_CSV_COLUMNS_V1.join(","));
    out.push('\n');
    for row in rows {
        let line = format!(
            "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n",
            row.sample_id,
            row.step,
            row.absolute_pos,
            row.token_id,
            csv_escape(&row.token_text),
            row.label_token,
            csv_escape(row.norm_status_v.as_str()),
            csv_escape(row.norm_status_splus.as_str()),
            csv_escape(row.norm_status_sminus.as_str()),
            fmt_float_csv(row.input_norm_v),
            fmt_float_csv(row.input_norm_splus),
            fmt_float_csv(row.input_norm_sminus),
            fmt_option_float_csv(row.dot_v_splus),
            fmt_option_float_csv(row.dot_splus_sminus),
            fmt_option_float_csv(row.dot_sminus_v),
            fmt_option_float_csv(row.chordal_v_splus),
            fmt_option_float_csv(row.chordal_splus_sminus),
            fmt_option_float_csv(row.chordal_sminus_v),
            csv_escape(row.edge_outcome_r1.as_str()),
            csv_escape(row.edge_outcome_r2.as_str()),
            csv_escape(row.edge_outcome_r3.as_str()),
            fmt_option_float_csv(row.edge_chordal_r1),
            fmt_option_float_csv(row.edge_chordal_r2),
            fmt_option_float_csv(row.edge_chordal_r3),
            csv_escape(row.loop_outcome.as_str()),
            fmt_option_float_csv(row.rotor_loop_chordal),
            fmt_option_float_csv(row.rotor_loop_nonscalar_norm),
        );
        out.push_str(&line);
    }
    Ok(out)
}

fn build_token_telemetry_csv(
    run_id: &str,
    rows: &[Gate5TokenTelemetryRow],
) -> Result<String, Gate5OrchestratorError> {
    let mut out = String::new();
    out.push_str(&GATE5_TOKEN_TELEMETRY_CSV_COLUMNS_V1.join(","));
    out.push('\n');
    for row in rows {
        let record = [
            csv_escape(run_id),
            row.sample_id.to_string(),
            csv_escape(gate4_variant_str(row.variant)),
            csv_escape(row.world_type.as_deref().unwrap_or("")),
            row.step.to_string(),
            row.absolute_pos.to_string(),
            row.token_id.to_string(),
            csv_escape(&row.token_text),
            opt_usize_to_string(row.answer_char_start),
            opt_usize_to_string(row.answer_char_end),
            row.label_token.to_string(),
            row.label_transition.to_string(),
            csv_escape(row.defect_span_id.as_deref().unwrap_or("")),
            fmt_float_csv(row.label_coverage_ratio),
            fmt_float_csv(row.exact_token_match_ratio),
            csv_escape(row.transition_missing_reason.as_str()),
            csv_escape(row.edge_outcome_r1.as_str()),
            csv_escape(row.edge_outcome_r2.as_str()),
            csv_escape(row.edge_outcome_r3.as_str()),
            csv_escape(row.loop_outcome.as_str()),
            fmt_float_csv(row.score_a),
            fmt_float_csv(row.score_b),
            fmt_option_float_csv(row.score_e),
            fmt_float_csv(row.score_f),
            fmt_option_float_csv(row.rotor_loop_chordal),
            fmt_option_float_csv(row.rotor_loop_nonscalar_norm),
        ];
        out.push_str(&record.join(","));
        out.push('\n');
    }
    Ok(out)
}

fn build_sample_summary_csv(
    run_id: &str,
    rows: &[Gate5SampleSummaryRow],
) -> Result<String, Gate5OrchestratorError> {
    let mut out = String::new();
    out.push_str(&GATE5_SAMPLE_SUMMARY_CSV_COLUMNS_V1.join(","));
    out.push('\n');
    for row in rows {
        let record = [
            csv_escape(run_id),
            row.sample_id.to_string(),
            csv_escape(gate4_variant_str(row.variant)),
            csv_escape(row.world_type.as_deref().unwrap_or("")),
            row.n_token_steps.to_string(),
            row.n_transition_steps.to_string(),
            row.n_loop_steps_valid.to_string(),
            row.n_loop_steps_missing.to_string(),
            row.positive_token_count.to_string(),
            row.positive_transition_count.to_string(),
            fmt_float_csv(row.label_coverage_ratio),
            fmt_float_csv(row.exact_token_match_ratio),
            csv_escape(&row.triplets_sha256),
            csv_escape(&row.labels_sha256),
            fmt_option_float_csv(row.auprc_a),
            fmt_option_float_csv(row.auprc_b),
            fmt_option_float_csv(row.auprc_e),
            fmt_option_float_csv(row.auprc_f),
            fmt_option_float_csv(row.auprc_rotor_loop_chordal),
            csv_escape(row.best_token_baseline_name.unwrap_or("")),
            fmt_option_float_csv(row.delta_auprc_rotor_loop_chordal_vs_f),
            fmt_option_usize_csv(row.hit_at_10_f),
            fmt_option_usize_csv(row.hit_at_10_rotor_loop_chordal),
        ];
        out.push_str(&record.join(","));
        out.push('\n');
    }
    Ok(out)
}

fn build_manifest(
    identity: &Gate5IdentityInput,
    metadata: &Gate4MetadataInputV1,
    summary: &Gate5RunSummary,
    input_json_sha256: &str,
    token_telemetry_sha256: &str,
    sample_summary_sha256: &str,
) -> Gate5ManifestJson {
    Gate5ManifestJson {
        spec_version: GATE5_SPEC_VERSION.to_string(),
        method_id: GATE5_METHOD_ID.to_string(),
        spec_hash_raw_blake3: identity.spec_hash_raw_blake3.clone(),
        spec_hash_raw_input_id: SPEC_HASH_RAW_INPUT_ID.to_string(),
        spec_hash_blake3: identity.spec_hash_blake3.clone(),
        spec_hash_input_id: SPEC_HASH_INPUT_ID.to_string(),
        dataset_revision_id: identity.dataset_revision_id.clone(),
        dataset_hash_blake3: identity.dataset_hash_blake3.clone(),
        code_git_commit: identity.code_git_commit.clone(),
        build_target_triple: identity.build_target_triple.clone(),
        rustc_version: identity.rustc_version.clone(),
        evaluation_mode_id: identity.evaluation_mode_id.clone(),
        run_id: identity.run_id.clone(),
        n_samples_total: summary.n_samples_total,
        n_token_rows_total: summary.n_token_rows_total,
        n_transition_rows_total: summary.n_transition_rows_total,
        n_loop_rows_valid: summary.n_loop_rows_valid,
        n_loop_rows_missing: summary.n_loop_rows_missing,
        model_id: metadata.model_id.clone(),
        model_revision: metadata.model_revision.clone(),
        seed: metadata.seed,
        perm_r: metadata.perm_r,
        primary_score: Some(GATE5_PRIMARY_METRIC_ID.to_string()),
        proj_id: metadata.proj_id.clone(),
        splus_def_id: metadata.splus_def_id.clone(),
        sminus_def_id: metadata.sminus_def_id.clone(),
        token_telemetry_schema_id: TOKEN_TELEMETRY_SCHEMA_ID.to_string(),
        sample_summary_schema_id: SAMPLE_SUMMARY_SCHEMA_ID.to_string(),
        float_format_id: FLOAT_FORMAT_ID.to_string(),
        transition_label_mode_id: TRANSITION_LABEL_MODE_ID.to_string(),
        edge_outcome_enum_id: EDGE_OUTCOME_ENUM_ID.to_string(),
        loop_outcome_enum_id: LOOP_OUTCOME_ENUM_ID.to_string(),
        score_missing_sentinel_id: SCORE_MISSING_SENTINEL_ID.to_string(),
        input_json_sha256: input_json_sha256.to_string(),
        token_telemetry_sha256: token_telemetry_sha256.to_string(),
        sample_summary_sha256: sample_summary_sha256.to_string(),
    }
}

fn write_string_lf(path: &Path, content: &str) -> Result<(), Gate5OrchestratorError> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    fs::write(path, normalized.as_bytes()).map_err(Gate5OrchestratorError::Io)
}

fn write_bytes_lf(path: &Path, content: &[u8]) -> Result<(), Gate5OrchestratorError> {
    fs::write(path, content).map_err(Gate5OrchestratorError::Io)
}

fn gate4_variant_str(variant: Gate4Variant) -> &'static str {
    match variant {
        Gate4Variant::Consistent => "consistent",
        Gate4Variant::Frustrated => "frustrated",
        Gate4Variant::Unknown => "unknown",
    }
}

fn fmt_float_csv(value: f64) -> String {
    format!("{:.17e}", value)
}

fn fmt_option_float_csv(value: Option<f64>) -> String {
    value.map(fmt_float_csv).unwrap_or_default()
}

fn fmt_option_usize_csv(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn opt_usize_to_string(value: Option<usize>) -> String {
    value.map(|v| v.to_string()).unwrap_or_default()
}

fn csv_escape(value: &str) -> String {
    let needs_quote =
        value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r');
    if !needs_quote {
        return value.to_string();
    }
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn validate_gate5_manifest_json(bytes: &[u8]) -> Result<(), Gate5ManifestValidationError> {
    std::str::from_utf8(bytes)
        .map_err(|err| Gate5ManifestValidationError::Utf8(err.to_string()))?;
    let value: Value = serde_json::from_slice(bytes).map_err(Gate5ManifestValidationError::Json)?;
    let object = value
        .as_object()
        .ok_or(Gate5ManifestValidationError::NotObject)?;

    for key in [
        "spec_version",
        "method_id",
        "dataset_revision_id",
        "dataset_hash_blake3",
        "spec_hash_raw_blake3",
        "spec_hash_blake3",
        "code_git_commit",
        "build_target_triple",
        "rustc_version",
        "evaluation_mode_id",
        "run_id",
        "n_samples_total",
        "n_token_rows_total",
        "n_transition_rows_total",
        "n_loop_rows_valid",
        "n_loop_rows_missing",
        "model_id",
        "model_revision",
        "seed",
        "perm_r",
        "primary_score",
        "proj_id",
        "splus_def_id",
        "sminus_def_id",
        "token_telemetry_schema_id",
        "sample_summary_schema_id",
        "float_format_id",
        "transition_label_mode_id",
        "edge_outcome_enum_id",
        "loop_outcome_enum_id",
        "score_missing_sentinel_id",
        "input_json_sha256",
        "token_telemetry_sha256",
        "sample_summary_sha256",
    ] {
        if !object.contains_key(key) {
            return Err(Gate5ManifestValidationError::MissingKey(key));
        }
    }

    validate_fixed_string(object, "spec_version", GATE5_SPEC_VERSION)?;
    validate_fixed_string(object, "method_id", GATE5_METHOD_ID)?;
    validate_fixed_string(
        object,
        "token_telemetry_schema_id",
        TOKEN_TELEMETRY_SCHEMA_ID,
    )?;
    validate_fixed_string(object, "sample_summary_schema_id", SAMPLE_SUMMARY_SCHEMA_ID)?;
    validate_fixed_string(object, "float_format_id", FLOAT_FORMAT_ID)?;
    validate_fixed_string(object, "transition_label_mode_id", TRANSITION_LABEL_MODE_ID)?;
    validate_fixed_string(object, "edge_outcome_enum_id", EDGE_OUTCOME_ENUM_ID)?;
    validate_fixed_string(object, "loop_outcome_enum_id", LOOP_OUTCOME_ENUM_ID)?;
    validate_fixed_string(
        object,
        "score_missing_sentinel_id",
        SCORE_MISSING_SENTINEL_ID,
    )?;
    Ok(())
}

fn validate_fixed_string(
    object: &Map<String, Value>,
    key: &'static str,
    expected: &'static str,
) -> Result<(), Gate5ManifestValidationError> {
    let actual = object
        .get(key)
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());
    if actual.as_deref() != Some(expected) {
        return Err(Gate5ManifestValidationError::InvalidFixedString {
            key,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn identity_fixture() -> Gate5IdentityInput {
        Gate5IdentityInput {
            run_id: "gate5_fixture_run".to_string(),
            dataset_revision_id: "cfa_v1".to_string(),
            dataset_hash_blake3: "0000000000000000000000000000000000000000000000000000000000000000"
                .to_string(),
            spec_hash_raw_blake3:
                "1111111111111111111111111111111111111111111111111111111111111111".to_string(),
            spec_hash_blake3: "2222222222222222222222222222222222222222222222222222222222222222"
                .to_string(),
            evaluation_mode_id: "supervised_v1".to_string(),
            code_git_commit: "deadbeef".to_string(),
            build_target_triple: "x86_64-pc-windows-msvc".to_string(),
            rustc_version: "rustc 1.81.0".to_string(),
        }
    }

    fn input_json_fixture() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "metadata": {
                "model_id": "Qwen/Qwen2.5-1.5B",
                "model_revision": "main",
                "seed": 7,
                "perm_r": 2000,
                "primary_score": "E",
                "proj_id": "fwht_pad_pow2_take8_v1",
                "splus_def_id": "attn_lastlayer_weighted_hidden_v1",
                "sminus_def_id": "lm_head_row_expectation_topk128_v1",
                "script_sha256_extract": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "script_sha256_eval": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
            },
            "samples": [
                {
                    "sample_id": 2,
                    "variant": "frustrated",
                    "world_type": "genealogy",
                    "exact_token_match_ratio": 1.0,
                    "label_coverage_ratio": 1.0,
                    "triplets_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
                    "labels_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
                    "token_steps": [
                        {
                            "step": 0,
                            "absolute_pos": 10,
                            "answer_char_start": 0,
                            "answer_char_end": 5,
                            "token_id": 101,
                            "token_str": "Beryl",
                            "label_token": 0,
                            "defect_span_id": "",
                            "V_8d": [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "Splus_8d": [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "Sminus_8d": [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "baseline_logprob": -0.1,
                            "baseline_entropy": 0.2
                        },
                        {
                            "step": 1,
                            "absolute_pos": 11,
                            "answer_char_start": 5,
                            "answer_char_end": 8,
                            "token_id": 102,
                            "token_str": " is",
                            "label_token": 1,
                            "defect_span_id": "span-1",
                            "V_8d": [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "Splus_8d": [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "Sminus_8d": [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
                            "baseline_logprob": -0.2,
                            "baseline_entropy": 0.3
                        },
                        {
                            "step": 2,
                            "absolute_pos": 12,
                            "answer_char_start": 8,
                            "answer_char_end": 10,
                            "token_id": 103,
                            "token_str": ".",
                            "label_token": 0,
                            "defect_span_id": "",
                            "V_8d": [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
                            "Splus_8d": [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "Sminus_8d": [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "baseline_logprob": -0.3,
                            "baseline_entropy": 0.4
                        }
                    ]
                },
                {
                    "sample_id": 1,
                    "variant": "consistent",
                    "world_type": "temporal",
                    "exact_token_match_ratio": 1.0,
                    "label_coverage_ratio": 1.0,
                    "triplets_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
                    "labels_sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
                    "token_steps": [
                        {
                            "step": 0,
                            "absolute_pos": 3,
                            "answer_char_start": 0,
                            "answer_char_end": 4,
                            "token_id": 201,
                            "token_str": "Noble",
                            "label_token": 0,
                            "V_8d": [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "Splus_8d": [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "Sminus_8d": [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "baseline_logprob": -0.5,
                            "baseline_entropy": 0.6
                        },
                        {
                            "step": 1,
                            "absolute_pos": 4,
                            "answer_char_start": 4,
                            "answer_char_end": 5,
                            "token_id": 202,
                            "token_str": ".",
                            "label_token": 0,
                            "V_8d": [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "Splus_8d": [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                            "Sminus_8d": [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
                            "baseline_logprob": -0.7,
                            "baseline_entropy": 0.8
                        }
                    ]
                }
            ]
        }))
        .expect("fixture json")
    }

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "pale-ale-diagnose-gate5-{}-{}-{}",
            prefix,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn gate5_orchestrator_writes_artifacts_and_valid_manifest() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("e2e");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate5_and_write(&out_dir, &input, &identity).expect("orchestrator");
        assert_eq!(output.run_id, "gate5_fixture_run");
        assert_eq!(output.spec_version, GATE5_SPEC_VERSION);
        assert!(output.artifact_paths.manifest_json.exists());
        assert!(output.artifact_paths.token_telemetry_csv.exists());
        assert!(output.artifact_paths.sample_summary_csv.exists());

        let manifest_bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest");
        validate_gate5_manifest_json(&manifest_bytes).expect("manifest valid");
        let manifest: serde_json::Value = serde_json::from_slice(&manifest_bytes).expect("json");
        assert_eq!(
            manifest["primary_score"].as_str(),
            Some(GATE5_PRIMARY_METRIC_ID)
        );

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate5_orchestrator_is_deterministic_for_identical_input() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir_a = temp_dir("det-a");
        let out_dir_b = temp_dir("det-b");
        fs::create_dir_all(&out_dir_a).expect("mkdir a");
        fs::create_dir_all(&out_dir_b).expect("mkdir b");

        let out_a = run_gate5_and_write(&out_dir_a, &input, &identity).expect("run a");
        let out_b = run_gate5_and_write(&out_dir_b, &input, &identity).expect("run b");

        let manifest_a = fs::read(out_a.artifact_paths.manifest_json).expect("manifest a");
        let manifest_b = fs::read(out_b.artifact_paths.manifest_json).expect("manifest b");
        assert_eq!(manifest_a, manifest_b);

        let tokens_a = fs::read(out_a.artifact_paths.token_telemetry_csv).expect("tokens a");
        let tokens_b = fs::read(out_b.artifact_paths.token_telemetry_csv).expect("tokens b");
        assert_eq!(tokens_a, tokens_b);

        let summary_a = fs::read(out_a.artifact_paths.sample_summary_csv).expect("summary a");
        let summary_b = fs::read(out_b.artifact_paths.sample_summary_csv).expect("summary b");
        assert_eq!(summary_a, summary_b);

        let _ = fs::remove_dir_all(&out_dir_a);
        let _ = fs::remove_dir_all(&out_dir_b);
    }

    #[test]
    fn token_columns_are_hard_locked() {
        let expected = [
            "run_id",
            "sample_id",
            "variant",
            "world_type",
            "step",
            "absolute_pos",
            "token_id",
            "token_text",
            "answer_char_start",
            "answer_char_end",
            "label_token",
            "label_transition",
            "defect_span_id",
            "label_coverage_ratio",
            "exact_token_match_ratio",
            "transition_missing_reason",
            "edge_outcome_r1_v_to_splus",
            "edge_outcome_r2_splus_to_sminus",
            "edge_outcome_r3_sminus_to_v",
            "loop_outcome",
            "score_A_logprob",
            "score_B_entropy",
            "score_E_v_sminus_vnext",
            "score_F_loop",
            "rotor_loop_chordal_v1",
            "rotor_loop_nonscalar_norm_v1",
        ];
        assert_eq!(GATE5_TOKEN_TELEMETRY_CSV_COLUMNS_V1, &expected);
    }

    #[test]
    fn sample_summary_columns_are_hard_locked() {
        let expected = [
            "run_id",
            "sample_id",
            "variant",
            "world_type",
            "n_token_steps",
            "n_transition_steps",
            "n_loop_steps_valid",
            "n_loop_steps_missing",
            "positive_token_count",
            "positive_transition_count",
            "label_coverage_ratio",
            "exact_token_match_ratio",
            "triplets_sha256",
            "labels_sha256",
            "auprc_A",
            "auprc_B",
            "auprc_E",
            "auprc_F",
            "auprc_rotor_loop_chordal_v1",
            "best_token_baseline_name",
            "delta_auprc_rotor_loop_chordal_v1_vs_F",
            "hit_at_10_F",
            "hit_at_10_rotor_loop_chordal_v1",
        ];
        assert_eq!(GATE5_SAMPLE_SUMMARY_CSV_COLUMNS_V1, &expected);
    }

    #[test]
    fn antipodal_edge_marks_partial_loop_missing() {
        let edge = compute_edge(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        assert_eq!(edge.outcome, EdgeOutcome::AntipodalAngleOnly);
        let identity_edge = compute_edge(
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        );
        let (loop_outcome, chordal, nonscalar) =
            compute_loop_metrics(&edge, &identity_edge, &identity_edge);
        assert_eq!(loop_outcome, LoopOutcome::PartialLoopMissing);
        assert_eq!(chordal, None);
        assert_eq!(nonscalar, None);
    }
}
