use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

const SPEC_HASH_RAW_INPUT_ID: &str = "spec_text_raw_utf8_v1";
const SPEC_HASH_INPUT_ID: &str = "spec_text_utf8_lf_v1";
const FLOAT_FORMAT_ID: &str = "sci_17e_v1";
const TOKEN_FEATURES_SCHEMA_ID: &str = "gate4_token_features_csv_v1";
const SAMPLE_SUMMARY_SCHEMA_ID: &str = "gate4_sample_summary_csv_v1";
const TRANSITION_LABEL_MODE_ID: &str = "max_pair_v1";
const TRANSITION_MISSING_ENUM_ID: &str = "gate4_transition_missing_reason_v1";
const SCORE_MISSING_SENTINEL_ID: &str = "empty_string_v1";

pub const GATE4_SPEC_VERSION: &str = "v0.1.0-ssot.draft.0";
pub const GATE4_METHOD_ID: &str = "proxy_observable_feature_sink_v1";

pub const GATE4_TOKEN_FEATURES_CSV_COLUMNS_V1: &[&str] = &[
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
    "score_A_logprob",
    "score_B_entropy",
    "score_C_v_curvature",
    "score_D_v_splus_vnext",
    "score_E_v_sminus_vnext",
    "score_F_loop",
];

pub const GATE4_SAMPLE_SUMMARY_CSV_COLUMNS_V1: &[&str] = &[
    "run_id",
    "sample_id",
    "variant",
    "world_type",
    "n_token_steps",
    "n_transition_steps",
    "positive_token_count",
    "positive_transition_count",
    "label_coverage_ratio",
    "exact_token_match_ratio",
    "triplets_sha256",
    "labels_sha256",
    "auprc_A",
    "auprc_B",
    "auprc_C",
    "auprc_D",
    "auprc_E",
    "auprc_F",
    "best_baseline_name",
    "delta_auprc_E_vs_best_baseline",
    "hit_at_10_E",
];

const REQUIRED_MANIFEST_KEYS: &[&str] = &[
    "spec_version",
    "method_id",
    "spec_hash_raw_blake3",
    "spec_hash_raw_input_id",
    "spec_hash_blake3",
    "spec_hash_input_id",
    "dataset_revision_id",
    "dataset_hash_blake3",
    "code_git_commit",
    "build_target_triple",
    "rustc_version",
    "evaluation_mode_id",
    "run_id",
    "n_samples_total",
    "n_token_rows_total",
    "n_transition_rows_total",
    "n_samples_with_positive_tokens",
    "n_samples_with_positive_transitions",
    "model_id",
    "model_revision",
    "seed",
    "proj_id",
    "splus_def_id",
    "sminus_def_id",
    "script_sha256_extract",
    "script_sha256_featuregen",
    "token_features_schema_id",
    "sample_summary_schema_id",
    "float_format_id",
    "transition_label_mode_id",
    "transition_missing_enum_id",
    "score_missing_sentinel_id",
    "input_json_sha256",
    "token_features_sha256",
    "sample_summary_sha256",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate4RunInputV1 {
    pub metadata: Gate4MetadataInputV1,
    pub samples: Vec<Gate4SampleInputV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate4MetadataInputV1 {
    pub model_id: String,
    pub model_revision: String,
    pub seed: u64,
    #[serde(default)]
    pub perm_r: Option<u64>,
    #[serde(default)]
    pub primary_score: Option<String>,
    pub proj_id: String,
    pub splus_def_id: String,
    pub sminus_def_id: String,
    pub script_sha256_extract: String,
    #[serde(default)]
    pub script_sha256_eval: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate4SampleInputV1 {
    pub sample_id: u64,
    #[serde(default)]
    pub variant: Gate4Variant,
    #[serde(default)]
    pub world_type: Option<String>,
    pub exact_token_match_ratio: f64,
    pub label_coverage_ratio: f64,
    pub triplets_sha256: String,
    pub labels_sha256: String,
    pub token_steps: Vec<Gate4TokenStepInputV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate4TokenStepInputV1 {
    pub step: usize,
    pub absolute_pos: usize,
    #[serde(default)]
    pub answer_char_start: Option<usize>,
    #[serde(default)]
    pub answer_char_end: Option<usize>,
    pub token_id: u64,
    #[serde(alias = "token_text")]
    pub token_str: String,
    pub label_token: u8,
    #[serde(default)]
    pub defect_span_id: Option<String>,
    #[serde(rename = "V_8d")]
    pub v_8d: Vec<f64>,
    #[serde(rename = "Splus_8d")]
    pub splus_8d: Vec<f64>,
    #[serde(rename = "Sminus_8d")]
    pub sminus_8d: Vec<f64>,
    pub baseline_logprob: f64,
    pub baseline_entropy: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Gate4Variant {
    Consistent,
    Frustrated,
    Unknown,
}

impl Default for Gate4Variant {
    fn default() -> Self {
        Self::Unknown
    }
}

impl Gate4Variant {
    fn as_str(self) -> &'static str {
        match self {
            Self::Consistent => "consistent",
            Self::Frustrated => "frustrated",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Gate4IdentityInput {
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
pub struct Gate4ArtifactPaths {
    pub manifest_json: PathBuf,
    pub token_features_csv: PathBuf,
    pub sample_summary_csv: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gate4RunSummary {
    pub n_samples_total: usize,
    pub n_token_rows_total: usize,
    pub n_transition_rows_total: usize,
    pub n_samples_with_positive_tokens: usize,
    pub n_samples_with_positive_transitions: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate4RunOutput {
    pub run_id: String,
    pub spec_version: String,
    pub summary: Gate4RunSummary,
    pub artifact_paths: Gate4ArtifactPaths,
}

#[derive(Debug)]
pub enum Gate4OrchestratorError {
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
    ManifestValidation(Gate4ManifestValidationError),
}

impl fmt::Display for Gate4OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JsonParse(err) => write!(f, "failed to parse Gate4 JSON v1: {}", err),
            Self::DuplicateSampleId { sample_id } => {
                write!(f, "duplicate Gate4 sample_id {}", sample_id)
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
                if let Some(max_inclusive) = max_inclusive {
                    write!(
                        f,
                        "sample {} {} out of range [{}, {}]: {}",
                        sample_id, field, min_inclusive, max_inclusive, value
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
                "sample {} step {} {} dimension mismatch: expected {}, got {}",
                sample_id, step, field, expected, actual
            ),
            Self::Io(err) => write!(f, "io error: {}", err),
            Self::JsonWrite(err) => write!(f, "json serialization error: {}", err),
            Self::ManifestRead(err) => write!(f, "failed to read manifest.json: {}", err),
            Self::ManifestValidation(err) => write!(f, "manifest validation error: {}", err),
        }
    }
}

impl std::error::Error for Gate4OrchestratorError {}

impl From<std::io::Error> for Gate4OrchestratorError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for Gate4OrchestratorError {
    fn from(value: serde_json::Error) -> Self {
        Self::JsonWrite(value)
    }
}

#[derive(Debug)]
pub enum Gate4ManifestValidationError {
    Utf8(String),
    Json(serde_json::Error),
    RootNotObject,
    MissingKey(&'static str),
    InvalidFixedString {
        key: &'static str,
        expected: &'static str,
        actual: Option<String>,
    },
    ForbiddenToken(&'static str),
}

impl fmt::Display for Gate4ManifestValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Utf8(err) => write!(f, "manifest is not UTF-8: {}", err),
            Self::Json(err) => write!(f, "manifest JSON parse failed: {}", err),
            Self::RootNotObject => write!(f, "manifest root must be a JSON object"),
            Self::MissingKey(key) => write!(f, "manifest missing required key '{}'", key),
            Self::InvalidFixedString {
                key,
                expected,
                actual,
            } => write!(
                f,
                "manifest '{}' must be '{}', got {}",
                key,
                expected,
                actual.as_deref().unwrap_or("<non-string>")
            ),
            Self::ForbiddenToken(token) => {
                write!(f, "manifest contains forbidden token '{}'", token)
            }
        }
    }
}

impl std::error::Error for Gate4ManifestValidationError {}

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
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

#[derive(Clone, Debug)]
struct Gate4TokenFeatureRow {
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
    score_a: f64,
    score_b: f64,
    score_c: Option<f64>,
    score_d: Option<f64>,
    score_e: Option<f64>,
    score_f: f64,
}

#[derive(Clone, Debug)]
struct Gate4SampleSummaryRow {
    sample_id: u64,
    variant: Gate4Variant,
    world_type: Option<String>,
    n_token_steps: usize,
    n_transition_steps: usize,
    positive_token_count: usize,
    positive_transition_count: usize,
    label_coverage_ratio: f64,
    exact_token_match_ratio: f64,
    triplets_sha256: String,
    labels_sha256: String,
    auprc_a: Option<f64>,
    auprc_b: Option<f64>,
    auprc_c: Option<f64>,
    auprc_d: Option<f64>,
    auprc_e: Option<f64>,
    auprc_f: Option<f64>,
    best_baseline_name: &'static str,
    delta_auprc_e_vs_best_baseline: Option<f64>,
    hit_at_10_e: usize,
}

#[derive(Serialize)]
struct Gate4ManifestJson {
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
    n_samples_with_positive_tokens: usize,
    n_samples_with_positive_transitions: usize,
    model_id: String,
    model_revision: String,
    seed: u64,
    perm_r: Option<u64>,
    primary_score: Option<String>,
    proj_id: String,
    splus_def_id: String,
    sminus_def_id: String,
    script_sha256_extract: String,
    script_sha256_eval: Option<String>,
    script_sha256_featuregen: String,
    token_features_schema_id: String,
    sample_summary_schema_id: String,
    float_format_id: String,
    transition_label_mode_id: String,
    transition_missing_enum_id: String,
    score_missing_sentinel_id: String,
    input_json_sha256: String,
    token_features_sha256: String,
    sample_summary_sha256: String,
}

pub fn run_gate4_and_write<P: AsRef<Path>>(
    out_dir: P,
    input_json_bytes: &[u8],
    identity: &Gate4IdentityInput,
) -> Result<Gate4RunOutput, Gate4OrchestratorError> {
    if identity.evaluation_mode_id != "supervised_v1"
        && identity.evaluation_mode_id != "unsupervised_v1"
    {
        return Err(Gate4OrchestratorError::InvalidEvaluationMode(
            identity.evaluation_mode_id.clone(),
        ));
    }

    let parsed: Gate4RunInputV1 =
        serde_json::from_slice(input_json_bytes).map_err(Gate4OrchestratorError::JsonParse)?;
    let samples = validate_samples(parsed.samples)?;

    let mut token_rows = Vec::new();
    let mut sample_rows = Vec::new();
    let mut n_samples_with_positive_tokens = 0usize;
    let mut n_samples_with_positive_transitions = 0usize;

    for sample in &samples {
        let (sample_token_rows, sample_summary) = compute_sample_outputs(sample);
        if sample_summary.positive_token_count > 0 {
            n_samples_with_positive_tokens += 1;
        }
        if sample_summary.positive_transition_count > 0 {
            n_samples_with_positive_transitions += 1;
        }
        token_rows.extend(sample_token_rows);
        sample_rows.push(sample_summary);
    }

    token_rows.sort_by(|left, right| {
        left.sample_id
            .cmp(&right.sample_id)
            .then(left.step.cmp(&right.step))
    });
    sample_rows.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));

    let token_csv = build_token_features_csv(&identity.run_id, &token_rows)?;
    let sample_csv = build_sample_summary_csv(&identity.run_id, &sample_rows)?;

    let input_json_sha256 = sha256_hex(input_json_bytes);
    let token_features_sha256 = sha256_hex(token_csv.as_bytes());
    let sample_summary_sha256 = sha256_hex(sample_csv.as_bytes());
    let summary = Gate4RunSummary {
        n_samples_total: sample_rows.len(),
        n_token_rows_total: token_rows.len(),
        n_transition_rows_total: token_rows
            .iter()
            .filter(|row| row.transition_missing_reason == TransitionMissingReason::None)
            .count(),
        n_samples_with_positive_tokens,
        n_samples_with_positive_transitions,
    };

    let manifest = build_manifest(
        identity,
        &parsed.metadata,
        &summary,
        &input_json_sha256,
        &token_features_sha256,
        &sample_summary_sha256,
    );

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    validate_gate4_manifest_json(&manifest_bytes)
        .map_err(Gate4OrchestratorError::ManifestValidation)?;

    let manifest_path = out_dir.join("manifest.json");
    let token_csv_path = out_dir.join("gate4_token_features.csv");
    let sample_csv_path = out_dir.join("gate4_sample_summary.csv");

    write_bytes_lf(&manifest_path, &manifest_bytes)?;
    write_string_lf(&token_csv_path, &token_csv)?;
    write_string_lf(&sample_csv_path, &sample_csv)?;

    let manifest_read = fs::read(&manifest_path).map_err(Gate4OrchestratorError::ManifestRead)?;
    validate_gate4_manifest_json(&manifest_read)
        .map_err(Gate4OrchestratorError::ManifestValidation)?;

    Ok(Gate4RunOutput {
        run_id: identity.run_id.clone(),
        spec_version: GATE4_SPEC_VERSION.to_string(),
        summary,
        artifact_paths: Gate4ArtifactPaths {
            manifest_json: manifest_path,
            token_features_csv: token_csv_path,
            sample_summary_csv: sample_csv_path,
        },
    })
}
fn validate_samples(
    mut samples: Vec<Gate4SampleInputV1>,
) -> Result<Vec<ValidatedSample>, Gate4OrchestratorError> {
    samples.sort_by(|left, right| left.sample_id.cmp(&right.sample_id));
    let mut out = Vec::with_capacity(samples.len());
    let mut previous_sample_id: Option<u64> = None;
    for sample in samples {
        if previous_sample_id == Some(sample.sample_id) {
            return Err(Gate4OrchestratorError::DuplicateSampleId {
                sample_id: sample.sample_id,
            });
        }
        previous_sample_id = Some(sample.sample_id);
        if sample.token_steps.is_empty() {
            return Err(Gate4OrchestratorError::MissingTokenSteps {
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
                return Err(Gate4OrchestratorError::DuplicateStep {
                    sample_id: sample.sample_id,
                    step: step.step,
                });
            }
            if let Some(prev) = previous_step {
                if step.step != prev + 1 {
                    return Err(Gate4OrchestratorError::NonContiguousStep {
                        sample_id: sample.sample_id,
                        expected: prev + 1,
                        actual: step.step,
                    });
                }
            }
            previous_step = Some(step.step);
            if step.label_token > 1 {
                return Err(Gate4OrchestratorError::InvalidLabel {
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
                defect_span_id: step.defect_span_id,
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
) -> Result<[f64; 8], Gate4OrchestratorError> {
    if values.len() != 8 {
        return Err(Gate4OrchestratorError::InvalidVec8Dim {
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
) -> Result<(), Gate4OrchestratorError> {
    if !value.is_finite() {
        return Err(Gate4OrchestratorError::InvalidFloat {
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
) -> Result<(), Gate4OrchestratorError> {
    if value < min_inclusive {
        return Err(Gate4OrchestratorError::InvalidRange {
            sample_id,
            field,
            min_inclusive,
            max_inclusive,
            value,
        });
    }
    if let Some(max_inclusive) = max_inclusive {
        if value > max_inclusive {
            return Err(Gate4OrchestratorError::InvalidRange {
                sample_id,
                field,
                min_inclusive,
                max_inclusive: Some(max_inclusive),
                value,
            });
        }
    }
    Ok(())
}

fn compute_sample_outputs(
    sample: &ValidatedSample,
) -> (Vec<Gate4TokenFeatureRow>, Gate4SampleSummaryRow) {
    let n = sample.token_steps.len();
    let mut labels_token = Vec::with_capacity(n);
    let mut score_a = Vec::with_capacity(n);
    let mut score_b = Vec::with_capacity(n);
    let mut score_f = Vec::with_capacity(n);
    let mut labels_transition = Vec::with_capacity(n.saturating_sub(1));
    let mut score_c = Vec::with_capacity(n.saturating_sub(1));
    let mut score_d = Vec::with_capacity(n.saturating_sub(1));
    let mut score_e = Vec::with_capacity(n.saturating_sub(1));

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
        score_c.push(d_proj(&current.v_8d, &next.v_8d));
        score_d
            .push(d_proj(&current.v_8d, &current.splus_8d) + d_proj(&current.splus_8d, &next.v_8d));
        score_e.push(
            d_proj(&current.v_8d, &current.sminus_8d) + d_proj(&current.sminus_8d, &next.v_8d),
        );
    }

    let auprc_a = average_precision(&labels_token, &score_a);
    let auprc_b = average_precision(&labels_token, &score_b);
    let (best_baseline_auprc, best_baseline_name) = match (auprc_a, auprc_b) {
        (Some(a), Some(b)) => {
            if a >= b {
                (Some(a), "A")
            } else {
                (Some(b), "B")
            }
        }
        (Some(a), None) => (Some(a), "A"),
        (None, Some(b)) => (Some(b), "B"),
        (None, None) => (None, "none"),
    };
    let auprc_c = average_precision(&labels_transition, &score_c);
    let auprc_d = average_precision(&labels_transition, &score_d);
    let auprc_e = average_precision(&labels_transition, &score_e);
    let auprc_f = average_precision(&labels_token, &score_f);
    let delta_auprc_e_vs_best_baseline = match (auprc_e, best_baseline_auprc) {
        (Some(left), Some(right)) => Some(left - right),
        _ => None,
    };
    let hit_at_10_e = hit_at_k(&labels_transition, &score_e, 10);

    let mut rows = Vec::with_capacity(n);
    for (idx, token) in sample.token_steps.iter().enumerate() {
        let (label_transition, transition_missing_reason, c, d, e) = if idx + 1 < n {
            (
                labels_transition[idx],
                TransitionMissingReason::None,
                Some(score_c[idx]),
                Some(score_d[idx]),
                Some(score_e[idx]),
            )
        } else {
            (
                0,
                TransitionMissingReason::FinalStepNoSuccessor,
                None,
                None,
                None,
            )
        };
        rows.push(Gate4TokenFeatureRow {
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
            score_a: score_a[idx],
            score_b: score_b[idx],
            score_c: c,
            score_d: d,
            score_e: e,
            score_f: score_f[idx],
        });
    }

    let summary = Gate4SampleSummaryRow {
        sample_id: sample.sample_id,
        variant: sample.variant,
        world_type: sample.world_type.clone(),
        n_token_steps: n,
        n_transition_steps: labels_transition.len(),
        positive_token_count: labels_token.iter().filter(|&&y| y == 1).count(),
        positive_transition_count: labels_transition.iter().filter(|&&y| y == 1).count(),
        label_coverage_ratio: sample.label_coverage_ratio,
        exact_token_match_ratio: sample.exact_token_match_ratio,
        triplets_sha256: sample.triplets_sha256.clone(),
        labels_sha256: sample.labels_sha256.clone(),
        auprc_a,
        auprc_b,
        auprc_c,
        auprc_d,
        auprc_e,
        auprc_f,
        best_baseline_name,
        delta_auprc_e_vs_best_baseline,
        hit_at_10_e,
    };
    (rows, summary)
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

fn dot_abs_clamped(left: &[f64; 8], right: &[f64; 8]) -> f64 {
    let inner: f64 = left.iter().zip(right.iter()).map(|(a, b)| *a * *b).sum();
    inner.abs().min(1.0)
}

fn d_proj(left: &[f64; 8], right: &[f64; 8]) -> f64 {
    (2.0 * (1.0 - dot_abs_clamped(left, right))).max(0.0).sqrt()
}
fn build_token_features_csv(
    run_id: &str,
    rows: &[Gate4TokenFeatureRow],
) -> Result<String, Gate4OrchestratorError> {
    let mut out = String::new();
    out.push_str(&GATE4_TOKEN_FEATURES_CSV_COLUMNS_V1.join(","));
    out.push('\n');
    for row in rows {
        let record = [
            csv_escape(run_id),
            row.sample_id.to_string(),
            csv_escape(row.variant.as_str()),
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
            fmt_float_csv(row.score_a),
            fmt_float_csv(row.score_b),
            fmt_option_float_csv(row.score_c),
            fmt_option_float_csv(row.score_d),
            fmt_option_float_csv(row.score_e),
            fmt_float_csv(row.score_f),
        ];
        out.push_str(&record.join(","));
        out.push('\n');
    }
    Ok(out)
}

fn build_sample_summary_csv(
    run_id: &str,
    rows: &[Gate4SampleSummaryRow],
) -> Result<String, Gate4OrchestratorError> {
    let mut out = String::new();
    out.push_str(&GATE4_SAMPLE_SUMMARY_CSV_COLUMNS_V1.join(","));
    out.push('\n');
    for row in rows {
        let record = [
            csv_escape(run_id),
            row.sample_id.to_string(),
            csv_escape(row.variant.as_str()),
            csv_escape(row.world_type.as_deref().unwrap_or("")),
            row.n_token_steps.to_string(),
            row.n_transition_steps.to_string(),
            row.positive_token_count.to_string(),
            row.positive_transition_count.to_string(),
            fmt_float_csv(row.label_coverage_ratio),
            fmt_float_csv(row.exact_token_match_ratio),
            csv_escape(&row.triplets_sha256),
            csv_escape(&row.labels_sha256),
            fmt_option_float_csv(row.auprc_a),
            fmt_option_float_csv(row.auprc_b),
            fmt_option_float_csv(row.auprc_c),
            fmt_option_float_csv(row.auprc_d),
            fmt_option_float_csv(row.auprc_e),
            fmt_option_float_csv(row.auprc_f),
            csv_escape(row.best_baseline_name),
            fmt_option_float_csv(row.delta_auprc_e_vs_best_baseline),
            row.hit_at_10_e.to_string(),
        ];
        out.push_str(&record.join(","));
        out.push('\n');
    }
    Ok(out)
}

fn build_manifest(
    identity: &Gate4IdentityInput,
    metadata: &Gate4MetadataInputV1,
    summary: &Gate4RunSummary,
    input_json_sha256: &str,
    token_features_sha256: &str,
    sample_summary_sha256: &str,
) -> Gate4ManifestJson {
    Gate4ManifestJson {
        spec_version: GATE4_SPEC_VERSION.to_string(),
        method_id: GATE4_METHOD_ID.to_string(),
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
        n_samples_with_positive_tokens: summary.n_samples_with_positive_tokens,
        n_samples_with_positive_transitions: summary.n_samples_with_positive_transitions,
        model_id: metadata.model_id.clone(),
        model_revision: metadata.model_revision.clone(),
        seed: metadata.seed,
        perm_r: metadata.perm_r,
        primary_score: metadata.primary_score.clone(),
        proj_id: metadata.proj_id.clone(),
        splus_def_id: metadata.splus_def_id.clone(),
        sminus_def_id: metadata.sminus_def_id.clone(),
        script_sha256_extract: metadata.script_sha256_extract.clone(),
        script_sha256_eval: metadata.script_sha256_eval.clone(),
        script_sha256_featuregen: gate4_source_sha256(),
        token_features_schema_id: TOKEN_FEATURES_SCHEMA_ID.to_string(),
        sample_summary_schema_id: SAMPLE_SUMMARY_SCHEMA_ID.to_string(),
        float_format_id: FLOAT_FORMAT_ID.to_string(),
        transition_label_mode_id: TRANSITION_LABEL_MODE_ID.to_string(),
        transition_missing_enum_id: TRANSITION_MISSING_ENUM_ID.to_string(),
        score_missing_sentinel_id: SCORE_MISSING_SENTINEL_ID.to_string(),
        input_json_sha256: input_json_sha256.to_string(),
        token_features_sha256: token_features_sha256.to_string(),
        sample_summary_sha256: sample_summary_sha256.to_string(),
    }
}

fn gate4_source_sha256() -> String {
    sha256_hex(include_str!("gate4.rs").as_bytes())
}

fn write_string_lf(path: &Path, content: &str) -> Result<(), Gate4OrchestratorError> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    fs::write(path, normalized.as_bytes()).map_err(Gate4OrchestratorError::Io)
}

fn write_bytes_lf(path: &Path, content: &[u8]) -> Result<(), Gate4OrchestratorError> {
    fs::write(path, content).map_err(Gate4OrchestratorError::Io)
}

fn fmt_float_csv(value: f64) -> String {
    format!("{:.17e}", value)
}

fn fmt_option_float_csv(value: Option<f64>) -> String {
    value.map(fmt_float_csv).unwrap_or_default()
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

pub fn validate_gate4_manifest_json(bytes: &[u8]) -> Result<(), Gate4ManifestValidationError> {
    std::str::from_utf8(bytes)
        .map_err(|err| Gate4ManifestValidationError::Utf8(err.to_string()))?;

    let value: Value = serde_json::from_slice(bytes).map_err(Gate4ManifestValidationError::Json)?;
    for (_path, string_value) in iter_string_values(String::new(), &value) {
        if let Some(token) = forbidden_manifest_string(&string_value) {
            return Err(Gate4ManifestValidationError::ForbiddenToken(token));
        }
    }
    let object = value
        .as_object()
        .ok_or(Gate4ManifestValidationError::RootNotObject)?;
    for key in REQUIRED_MANIFEST_KEYS {
        if !object.contains_key(*key) {
            return Err(Gate4ManifestValidationError::MissingKey(key));
        }
    }
    validate_fixed_string(object, "spec_version", GATE4_SPEC_VERSION)?;
    validate_fixed_string(object, "method_id", GATE4_METHOD_ID)?;
    validate_fixed_string(object, "spec_hash_raw_input_id", SPEC_HASH_RAW_INPUT_ID)?;
    validate_fixed_string(object, "spec_hash_input_id", SPEC_HASH_INPUT_ID)?;
    validate_fixed_string(object, "token_features_schema_id", TOKEN_FEATURES_SCHEMA_ID)?;
    validate_fixed_string(object, "sample_summary_schema_id", SAMPLE_SUMMARY_SCHEMA_ID)?;
    validate_fixed_string(object, "float_format_id", FLOAT_FORMAT_ID)?;
    validate_fixed_string(object, "transition_label_mode_id", TRANSITION_LABEL_MODE_ID)?;
    validate_fixed_string(
        object,
        "transition_missing_enum_id",
        TRANSITION_MISSING_ENUM_ID,
    )?;
    validate_fixed_string(
        object,
        "score_missing_sentinel_id",
        SCORE_MISSING_SENTINEL_ID,
    )?;
    Ok(())
}

fn forbidden_manifest_string(value: &str) -> Option<&'static str> {
    match value {
        "NaN" => Some("NaN"),
        "nan" => Some("nan"),
        "inf" => Some("inf"),
        "-inf" => Some("-inf"),
        _ => None,
    }
}

fn iter_string_values(path: String, value: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    match value {
        Value::String(text) => out.push((path, text.clone())),
        Value::Array(items) => {
            for (idx, item) in items.iter().enumerate() {
                let next = if path.is_empty() {
                    format!("[{}]", idx)
                } else {
                    format!("{}[{}]", path, idx)
                };
                out.extend(iter_string_values(next, item));
            }
        }
        Value::Object(map) => {
            for (key, item) in map {
                let next = if path.is_empty() {
                    key.clone()
                } else {
                    format!("{}.{}", path, key)
                };
                out.extend(iter_string_values(next, item));
            }
        }
        _ => {}
    }
    out
}

fn validate_fixed_string(
    object: &Map<String, Value>,
    key: &'static str,
    expected: &'static str,
) -> Result<(), Gate4ManifestValidationError> {
    let actual = object
        .get(key)
        .and_then(|value| value.as_str())
        .map(|value| value.to_string());
    if actual.as_deref() != Some(expected) {
        return Err(Gate4ManifestValidationError::InvalidFixedString {
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

    fn identity_fixture() -> Gate4IdentityInput {
        Gate4IdentityInput {
            run_id: "gate4_fixture_run".to_string(),
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
        })).expect("fixture json")
    }

    fn temp_dir(prefix: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "pale-ale-diagnose-gate4-{}-{}-{}",
            prefix,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn gate4_orchestrator_writes_artifacts_and_valid_manifest() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("e2e");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate4_and_write(&out_dir, &input, &identity).expect("orchestrator");
        assert_eq!(output.run_id, "gate4_fixture_run");
        assert_eq!(output.spec_version, GATE4_SPEC_VERSION);
        assert!(output.artifact_paths.manifest_json.exists());
        assert!(output.artifact_paths.token_features_csv.exists());
        assert!(output.artifact_paths.sample_summary_csv.exists());

        let manifest_bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest");
        validate_gate4_manifest_json(&manifest_bytes).expect("manifest valid");

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate4_orchestrator_is_deterministic_for_identical_input() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir_a = temp_dir("det-a");
        let out_dir_b = temp_dir("det-b");
        fs::create_dir_all(&out_dir_a).expect("mkdir a");
        fs::create_dir_all(&out_dir_b).expect("mkdir b");

        let out_a = run_gate4_and_write(&out_dir_a, &input, &identity).expect("run a");
        let out_b = run_gate4_and_write(&out_dir_b, &input, &identity).expect("run b");

        let manifest_a = fs::read(out_a.artifact_paths.manifest_json).expect("manifest a");
        let manifest_b = fs::read(out_b.artifact_paths.manifest_json).expect("manifest b");
        assert_eq!(manifest_a, manifest_b);

        let tokens_a = fs::read(out_a.artifact_paths.token_features_csv).expect("tokens a");
        let tokens_b = fs::read(out_b.artifact_paths.token_features_csv).expect("tokens b");
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
            "score_A_logprob",
            "score_B_entropy",
            "score_C_v_curvature",
            "score_D_v_splus_vnext",
            "score_E_v_sminus_vnext",
            "score_F_loop",
        ];
        assert_eq!(GATE4_TOKEN_FEATURES_CSV_COLUMNS_V1, expected.as_slice());
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
            "positive_token_count",
            "positive_transition_count",
            "label_coverage_ratio",
            "exact_token_match_ratio",
            "triplets_sha256",
            "labels_sha256",
            "auprc_A",
            "auprc_B",
            "auprc_C",
            "auprc_D",
            "auprc_E",
            "auprc_F",
            "best_baseline_name",
            "delta_auprc_E_vs_best_baseline",
            "hit_at_10_E",
        ];
        assert_eq!(GATE4_SAMPLE_SUMMARY_CSV_COLUMNS_V1, expected.as_slice());
    }

    #[test]
    fn tokens_csv_is_sorted_by_sample_id_then_step() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("sorted");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate4_and_write(&out_dir, &input, &identity).expect("run");
        let rows = fs::read_to_string(output.artifact_paths.token_features_csv).expect("rows");
        let lines: Vec<&str> = rows.lines().collect();
        assert!(lines.len() >= 4);
        assert!(lines[1].starts_with("gate4_fixture_run,1,"));
        assert!(lines[2].starts_with("gate4_fixture_run,1,"));
        assert!(lines[3].starts_with("gate4_fixture_run,2,"));

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn undefined_baselines_emit_none_sentinel_in_summary() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("baseline-none");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate4_and_write(&out_dir, &input, &identity).expect("run");
        let summary =
            fs::read_to_string(output.artifact_paths.sample_summary_csv).expect("summary");
        let lines: Vec<&str> = summary.lines().collect();
        assert!(lines.len() >= 2);
        assert!(lines[1].contains(",none,"));

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn final_step_emits_missing_transition_fields() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("final-step-missing");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate4_and_write(&out_dir, &input, &identity).expect("run");
        let csv = fs::read_to_string(output.artifact_paths.token_features_csv).expect("csv");
        let lines: Vec<&str> = csv.lines().collect();
        assert!(lines.len() >= 6);

        let header: Vec<&str> = lines[0].split(',').collect();
        let idx_sample_id = header
            .iter()
            .position(|col| *col == "sample_id")
            .expect("sample_id index");
        let idx_step = header
            .iter()
            .position(|col| *col == "step")
            .expect("step index");
        let last_frustrated_row: Vec<&str> = lines[1..]
            .iter()
            .map(|line| line.split(',').collect::<Vec<_>>())
            .find(|cols| cols[idx_sample_id] == "2" && cols[idx_step] == "2")
            .expect("sample 2 step 2 row");

        let idx_reason = header
            .iter()
            .position(|col| *col == "transition_missing_reason")
            .expect("transition_missing_reason index");
        let idx_c = header
            .iter()
            .position(|col| *col == "score_C_v_curvature")
            .expect("score_C index");
        let idx_d = header
            .iter()
            .position(|col| *col == "score_D_v_splus_vnext")
            .expect("score_D index");
        let idx_e = header
            .iter()
            .position(|col| *col == "score_E_v_sminus_vnext")
            .expect("score_E index");

        assert_eq!(last_frustrated_row[idx_reason], "final_step_no_successor");
        assert_eq!(last_frustrated_row[idx_c], "");
        assert_eq!(last_frustrated_row[idx_d], "");
        assert_eq!(last_frustrated_row[idx_e], "");

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn duplicate_step_is_rejected() {
        let identity = identity_fixture();
        let mut value: Value = serde_json::from_slice(&input_json_fixture()).expect("json");
        let token_steps = value["samples"][0]["token_steps"]
            .as_array_mut()
            .expect("token steps");
        let dup = token_steps[0].clone();
        token_steps.push(dup);
        let broken = serde_json::to_vec(&value).expect("to vec");
        let err = run_gate4_and_write(temp_dir("dup-step"), &broken, &identity)
            .expect_err("expected duplicate step error");
        assert!(matches!(
            err,
            Gate4OrchestratorError::DuplicateStep {
                sample_id: 2,
                step: 0
            }
        ));
    }

    #[test]
    fn non_contiguous_step_is_rejected() {
        let identity = identity_fixture();
        let mut value: Value = serde_json::from_slice(&input_json_fixture()).expect("json");
        value["samples"][0]["token_steps"][1]["step"] = serde_json::json!(2);
        let broken = serde_json::to_vec(&value).expect("to vec");
        let err = run_gate4_and_write(temp_dir("non-contig-step"), &broken, &identity)
            .expect_err("expected non-contiguous step error");
        assert!(matches!(
            err,
            Gate4OrchestratorError::NonContiguousStep {
                sample_id: 2,
                expected: 1,
                actual: 2
            }
        ));
    }

    #[test]
    fn invalid_vec8_dim_is_rejected() {
        let identity = identity_fixture();
        let mut value: Value = serde_json::from_slice(&input_json_fixture()).expect("json");
        value["samples"][0]["token_steps"][0]["V_8d"] = serde_json::json!([1.0, 0.0]);
        let broken = serde_json::to_vec(&value).expect("to vec");
        let err = run_gate4_and_write(temp_dir("bad-dim"), &broken, &identity)
            .expect_err("expected vec8 dim error");
        assert!(matches!(
            err,
            Gate4OrchestratorError::InvalidVec8Dim {
                sample_id: 2,
                step: 0,
                field: "V_8d",
                expected: 8,
                actual: 2
            }
        ));
    }

    #[test]
    fn invalid_label_coverage_ratio_is_rejected() {
        let identity = identity_fixture();
        let mut value: Value = serde_json::from_slice(&input_json_fixture()).expect("json");
        value["samples"][0]["label_coverage_ratio"] = serde_json::json!(1.2);
        let broken = serde_json::to_vec(&value).expect("to vec");
        let err = run_gate4_and_write(temp_dir("bad-coverage"), &broken, &identity)
            .expect_err("expected invalid range error");
        assert!(matches!(
            err,
            Gate4OrchestratorError::InvalidRange {
                sample_id: 2,
                field: "label_coverage_ratio",
                min_inclusive: 0.0,
                max_inclusive: Some(1.0),
                value
            } if (value - 1.2).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn negative_exact_token_match_ratio_is_rejected() {
        let identity = identity_fixture();
        let mut value: Value = serde_json::from_slice(&input_json_fixture()).expect("json");
        value["samples"][0]["exact_token_match_ratio"] = serde_json::json!(-0.1);
        let broken = serde_json::to_vec(&value).expect("to vec");
        let err = run_gate4_and_write(temp_dir("bad-match-ratio"), &broken, &identity)
            .expect_err("expected invalid range error");
        assert!(matches!(
            err,
            Gate4OrchestratorError::InvalidRange {
                sample_id: 2,
                field: "exact_token_match_ratio",
                min_inclusive: 0.0,
                max_inclusive: None,
                value
            } if (value + 0.1).abs() < f64::EPSILON
        ));
    }

    #[test]
    fn gate4_manifest_validator_rejects_missing_key() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("manifest-missing");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate4_and_write(&out_dir, &input, &identity).expect("run");
        let bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest read");
        let mut value: Value = serde_json::from_slice(&bytes).expect("json");
        value.as_object_mut().expect("object").remove("method_id");
        let broken = serde_json::to_vec(&value).expect("to vec");
        let err = validate_gate4_manifest_json(&broken).expect_err("expected failure");
        assert!(matches!(
            err,
            Gate4ManifestValidationError::MissingKey("method_id")
        ));

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate4_manifest_validator_allows_safe_substrings_in_strings() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("manifest-safe-substrings");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate4_and_write(&out_dir, &input, &identity).expect("run");
        let bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest read");
        let mut value: Value = serde_json::from_slice(&bytes).expect("json");
        let object = value.as_object_mut().expect("object");
        object.insert(
            "run_id".to_string(),
            Value::String("infinite_loop_test".to_string()),
        );
        object.insert(
            "dataset_revision_id".to_string(),
            Value::String("finance_reports".to_string()),
        );
        let safe = serde_json::to_vec(&value).expect("to vec");
        validate_gate4_manifest_json(&safe).expect("substring-safe manifest");

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate4_manifest_validator_rejects_exact_forbidden_string_token() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("manifest-forbidden-token");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate4_and_write(&out_dir, &input, &identity).expect("run");
        let bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest read");
        let mut value: Value = serde_json::from_slice(&bytes).expect("json");
        value
            .as_object_mut()
            .expect("object")
            .insert("run_id".to_string(), Value::String("NaN".to_string()));
        let broken = serde_json::to_vec(&value).expect("to vec");
        let err = validate_gate4_manifest_json(&broken).expect_err("expected forbidden token");
        assert!(matches!(
            err,
            Gate4ManifestValidationError::ForbiddenToken("NaN")
        ));

        let _ = fs::remove_dir_all(&out_dir);
    }
}
