use crate::gate2_telemetry::{
    compute_gate2_telemetry, Gate2SampleTelemetry, Gate2StatTriple, Gate2TelemetryInput,
    GATE2_ANTIPODAL_POLICY_ID, GATE2_BIVECTOR_BASIS_ID, GATE2_METHOD_ID,
    GATE2_ROTOR_CONSTRUCTION_ID, GATE2_SPEC_VERSION, GATE2_THETA_SOURCE_ID, H3_NAME_ID,
};
use pale_ale_rotor::{
    ALGEBRA_ID, BLADE_SIGN_ID, COMPOSITION_ID, EMBED_ID, NORMALIZE_ID, REVERSE_ID,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

const VEC8_DIM: usize = 8;
const SPEC_HASH_RAW_INPUT_ID: &str = "spec_text_raw_utf8_v1";
const SPEC_HASH_INPUT_ID: &str = "spec_text_utf8_lf_v1";

pub const GATE2_SUMMARY_CSV_COLUMNS_V1: &[&str] = &[
    "spec_version",
    "method_id",
    "composition_id",
    "dataset_revision_id",
    "code_git_commit",
    "build_target_triple",
    "rustc_version",
    "n_samples_total",
    "n_samples_h1b_present",
    "n_samples_h2_present",
    "n_samples_h3_total_present",
    "h1b_mean",
    "h1b_p90",
    "h1b_max",
    "h2_mean",
    "h2_p90",
    "h2_max",
    "h3_total_mean",
    "h3_total_p90",
    "h3_total_max",
    "h3_loop_mean",
    "h3_loop_p90",
    "h3_loop_max",
];

pub const GATE2_SAMPLES_CSV_COLUMNS_V1: &[&str] = &[
    "sample_id",
    "n_ans_units",
    "sample_label",
    "answer_length",
    "h1b_closure_error",
    "h2_loop_max",
    "h2_loop_mean",
    "h2_loop_p90",
    "h3_ratio_total_product",
    "h3_ratio_triangle_loop_max",
    "h3_ratio_triangle_loop_mean",
    "h3_ratio_triangle_loop_p90",
    "missing_even_rotor_steps",
    "loops_considered",
    "loops_used",
];

const REQUIRED_MANIFEST_KEYS: &[&str] = &[
    "spec_version",
    "method_id",
    "algebra_id",
    "blade_sign_id",
    "reverse_id",
    "normalize_id",
    "composition_id",
    "embed_id",
    "h3_name_id",
    "rotor_construction_id",
    "theta_source_id",
    "bivector_basis_id",
    "antipodal_policy_id",
    "spec_hash_raw_blake3",
    "spec_hash_raw_input_id",
    "spec_hash_blake3",
    "spec_hash_input_id",
    "dataset_revision_id",
    "dataset_hash_blake3",
    "code_git_commit",
    "build_target_triple",
    "rustc_version",
    "unitization_id",
    "rotor_encoder_id",
    "rotor_encoder_preproc_id",
    "vec8_postproc_id",
    "evaluation_mode_id",
    "n_samples_total",
    "n_samples_h1b_present",
    "n_samples_h2_present",
    "n_samples_h3_total_present",
    "h1b_mean",
    "h1b_p90",
    "h1b_max",
    "h2_mean",
    "h2_p90",
    "h2_max",
    "h3_total_mean",
    "h3_total_p90",
    "h3_total_max",
    "h3_loop_mean",
    "h3_loop_p90",
    "h3_loop_max",
];

const OPTIONAL_FLOAT_MANIFEST_KEYS: &[&str] = &[
    "h1b_mean",
    "h1b_p90",
    "h1b_max",
    "h2_mean",
    "h2_p90",
    "h2_max",
    "h3_total_mean",
    "h3_total_p90",
    "h3_total_max",
    "h3_loop_mean",
    "h3_loop_p90",
    "h3_loop_max",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate2RunInputV1 {
    #[serde(default = "default_gate2_run_id")]
    pub run_id: String,
    #[serde(default)]
    pub explicitly_unrelated_sample_ids: Vec<u64>,
    pub samples: Vec<Gate2SampleInputV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate2SampleInputV1 {
    pub sample_id: u64,
    pub ans_vec8: Vec<Vec<f64>>,
    #[serde(default)]
    pub sample_label: Option<u8>,
    #[serde(default)]
    pub answer_length: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Gate2IdentityInput {
    pub dataset_revision_id: String,
    pub dataset_hash_blake3: String,
    pub spec_hash_raw_blake3: String,
    pub spec_hash_blake3: String,
    pub unitization_id: String,
    pub rotor_encoder_id: String,
    pub rotor_encoder_preproc_id: String,
    pub vec8_postproc_id: String,
    pub evaluation_mode_id: String,
    pub code_git_commit: String,
    pub build_target_triple: String,
    pub rustc_version: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Gate2ArtifactPaths {
    pub manifest_json: PathBuf,
    pub summary_csv: PathBuf,
    pub samples_csv: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gate2AggregateStat {
    pub mean: f64,
    pub p90: f64,
    pub max: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate2RunSummary {
    pub n_samples_total: usize,
    pub n_samples_h1b_present: usize,
    pub n_samples_h2_present: usize,
    pub n_samples_h3_total_present: usize,
    pub h1b: Option<Gate2AggregateStat>,
    pub h2: Option<Gate2AggregateStat>,
    pub h3_total: Option<Gate2AggregateStat>,
    pub h3_loop: Option<Gate2AggregateStat>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate2RunOutput {
    pub run_id: String,
    pub spec_version: String,
    pub summary: Gate2RunSummary,
    pub artifact_paths: Gate2ArtifactPaths,
}

#[derive(Debug)]
pub enum Gate2OrchestratorError {
    JsonParse(serde_json::Error),
    InvalidVec8Dim {
        sample_id: u64,
        row_index: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteVec8 {
        sample_id: u64,
        row_index: usize,
        col_index: usize,
        value: f64,
    },
    InvalidSampleLabel {
        sample_id: u64,
        label: u8,
    },
    InvalidEvaluationMode(String),
    InvalidFloat {
        field: &'static str,
        value: f64,
    },
    Io(std::io::Error),
    JsonWrite(serde_json::Error),
    ManifestRead(std::io::Error),
    ManifestValidation(Gate2ManifestValidationError),
}

impl fmt::Display for Gate2OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JsonParse(err) => write!(f, "failed to parse Gate2 JSON v1: {}", err),
            Self::InvalidVec8Dim {
                sample_id,
                row_index,
                expected,
                actual,
            } => write!(
                f,
                "sample {} ans_vec8[{}] dimension mismatch: expected {}, got {}",
                sample_id, row_index, expected, actual
            ),
            Self::NonFiniteVec8 {
                sample_id,
                row_index,
                col_index,
                value,
            } => write!(
                f,
                "sample {} ans_vec8[{}][{}] is non-finite: {}",
                sample_id, row_index, col_index, value
            ),
            Self::InvalidSampleLabel { sample_id, label } => write!(
                f,
                "invalid sample_label for sample {}: {} (expected 0, 1, or null)",
                sample_id, label
            ),
            Self::InvalidEvaluationMode(value) => write!(
                f,
                "invalid evaluation_mode_id '{}': expected supervised_v1 or unsupervised_v1",
                value
            ),
            Self::InvalidFloat { field, value } => {
                write!(f, "non-finite float for {}: {}", field, value)
            }
            Self::Io(err) => write!(f, "io error: {}", err),
            Self::JsonWrite(err) => write!(f, "json serialization error: {}", err),
            Self::ManifestRead(err) => write!(f, "failed to read manifest.json: {}", err),
            Self::ManifestValidation(err) => write!(f, "manifest validation error: {}", err),
        }
    }
}

impl std::error::Error for Gate2OrchestratorError {}

impl From<std::io::Error> for Gate2OrchestratorError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for Gate2OrchestratorError {
    fn from(value: serde_json::Error) -> Self {
        Self::JsonWrite(value)
    }
}
#[derive(Debug)]
pub enum Gate2ManifestValidationError {
    Utf8(String),
    Json(serde_json::Error),
    RootNotObject,
    MissingKey(&'static str),
    InvalidFixedString {
        key: &'static str,
        expected: &'static str,
        actual: Option<String>,
    },
    InvalidFloatString {
        key: &'static str,
        value: String,
    },
    InvalidFloatType {
        key: &'static str,
    },
    ForbiddenToken(&'static str),
}

impl fmt::Display for Gate2ManifestValidationError {
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
            Self::InvalidFloatString { key, value } => {
                write!(
                    f,
                    "manifest '{}' has invalid sci_17e float string: {}",
                    key, value
                )
            }
            Self::InvalidFloatType { key } => write!(
                f,
                "manifest '{}' must be either a sci_17e string or null",
                key
            ),
            Self::ForbiddenToken(token) => {
                write!(f, "manifest contains forbidden token '{}'", token)
            }
        }
    }
}

impl std::error::Error for Gate2ManifestValidationError {}

#[derive(Clone, Debug)]
struct RunTelemetrySample {
    input_order: usize,
    sample_id: u64,
    sample_label: Option<u8>,
    answer_length: Option<usize>,
    telemetry: Gate2SampleTelemetry,
}

#[derive(Serialize)]
struct Gate2ManifestJson {
    spec_version: String,
    method_id: String,
    algebra_id: String,
    blade_sign_id: String,
    reverse_id: String,
    normalize_id: String,
    composition_id: String,
    embed_id: String,
    h3_name_id: String,
    rotor_construction_id: String,
    theta_source_id: String,
    bivector_basis_id: String,
    antipodal_policy_id: String,
    spec_hash_raw_blake3: String,
    spec_hash_raw_input_id: String,
    spec_hash_blake3: String,
    spec_hash_input_id: String,
    dataset_revision_id: String,
    dataset_hash_blake3: String,
    code_git_commit: String,
    build_target_triple: String,
    rustc_version: String,
    unitization_id: String,
    rotor_encoder_id: String,
    rotor_encoder_preproc_id: String,
    vec8_postproc_id: String,
    evaluation_mode_id: String,
    n_samples_total: usize,
    n_samples_h1b_present: usize,
    n_samples_h2_present: usize,
    n_samples_h3_total_present: usize,
    h1b_mean: Option<String>,
    h1b_p90: Option<String>,
    h1b_max: Option<String>,
    h2_mean: Option<String>,
    h2_p90: Option<String>,
    h2_max: Option<String>,
    h3_total_mean: Option<String>,
    h3_total_p90: Option<String>,
    h3_total_max: Option<String>,
    h3_loop_mean: Option<String>,
    h3_loop_p90: Option<String>,
    h3_loop_max: Option<String>,
}

fn default_gate2_run_id() -> String {
    "gate2_run".to_string()
}

pub fn run_gate2_and_write<P: AsRef<Path>>(
    out_dir: P,
    input_json_bytes: &[u8],
    identity: &Gate2IdentityInput,
) -> Result<Gate2RunOutput, Gate2OrchestratorError> {
    if identity.evaluation_mode_id != "supervised_v1"
        && identity.evaluation_mode_id != "unsupervised_v1"
    {
        return Err(Gate2OrchestratorError::InvalidEvaluationMode(
            identity.evaluation_mode_id.clone(),
        ));
    }

    let parsed: Gate2RunInputV1 =
        serde_json::from_slice(input_json_bytes).map_err(Gate2OrchestratorError::JsonParse)?;

    let mut samples = Vec::with_capacity(parsed.samples.len());
    for (input_order, sample) in parsed.samples.into_iter().enumerate() {
        if let Some(label) = sample.sample_label {
            if label > 1 {
                return Err(Gate2OrchestratorError::InvalidSampleLabel {
                    sample_id: sample.sample_id,
                    label,
                });
            }
        }

        let ans_vec8 = convert_vec8_rows(sample.sample_id, sample.ans_vec8)?;
        let telemetry = compute_gate2_telemetry(&Gate2TelemetryInput {
            sample_id: sample.sample_id,
            ans_vec8,
            answer_length: sample.answer_length,
        });

        samples.push(RunTelemetrySample {
            input_order,
            sample_id: sample.sample_id,
            sample_label: sample.sample_label,
            answer_length: sample.answer_length,
            telemetry: telemetry.per_sample,
        });
    }

    let summary = aggregate_summary(&samples);

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let manifest = build_manifest(identity, &summary)?;
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).map_err(Gate2OrchestratorError::JsonWrite)?;
    validate_gate2_manifest_json(&manifest_bytes)
        .map_err(Gate2OrchestratorError::ManifestValidation)?;

    let manifest_path = out_dir.join("manifest.json");
    write_bytes_lf(&manifest_path, &manifest_bytes)?;

    let summary_csv = build_summary_csv(identity, &summary)?;
    let summary_path = out_dir.join("summary.csv");
    write_string_lf(&summary_path, &summary_csv)?;

    let samples_csv = build_samples_csv(&samples)?;
    let samples_path = out_dir.join("samples.csv");
    write_string_lf(&samples_path, &samples_csv)?;

    let manifest_readback =
        fs::read(&manifest_path).map_err(Gate2OrchestratorError::ManifestRead)?;
    validate_gate2_manifest_json(&manifest_readback)
        .map_err(Gate2OrchestratorError::ManifestValidation)?;

    Ok(Gate2RunOutput {
        run_id: parsed.run_id,
        spec_version: GATE2_SPEC_VERSION.to_string(),
        summary,
        artifact_paths: Gate2ArtifactPaths {
            manifest_json: manifest_path,
            summary_csv: summary_path,
            samples_csv: samples_path,
        },
    })
}

pub fn validate_gate2_manifest_json(bytes: &[u8]) -> Result<(), Gate2ManifestValidationError> {
    std::str::from_utf8(bytes)
        .map_err(|err| Gate2ManifestValidationError::Utf8(err.to_string()))?;

    let root: Value = serde_json::from_slice(bytes).map_err(Gate2ManifestValidationError::Json)?;
    for (_path, value) in iter_string_values(String::new(), &root) {
        if value.contains("NaN") {
            return Err(Gate2ManifestValidationError::ForbiddenToken("NaN"));
        }
        if value.contains("Inf") {
            return Err(Gate2ManifestValidationError::ForbiddenToken("Inf"));
        }
    }
    let obj = root
        .as_object()
        .ok_or(Gate2ManifestValidationError::RootNotObject)?;

    for key in REQUIRED_MANIFEST_KEYS {
        if !obj.contains_key(*key) {
            return Err(Gate2ManifestValidationError::MissingKey(key));
        }
    }

    check_fixed_string(obj, "spec_version", GATE2_SPEC_VERSION)?;
    check_fixed_string(obj, "method_id", GATE2_METHOD_ID)?;
    check_fixed_string(obj, "algebra_id", ALGEBRA_ID)?;
    check_fixed_string(obj, "blade_sign_id", BLADE_SIGN_ID)?;
    check_fixed_string(obj, "reverse_id", REVERSE_ID)?;
    check_fixed_string(obj, "normalize_id", NORMALIZE_ID)?;
    check_fixed_string(obj, "composition_id", COMPOSITION_ID)?;
    check_fixed_string(obj, "embed_id", EMBED_ID)?;
    check_fixed_string(obj, "h3_name_id", H3_NAME_ID)?;
    check_fixed_string(obj, "rotor_construction_id", GATE2_ROTOR_CONSTRUCTION_ID)?;
    check_fixed_string(obj, "theta_source_id", GATE2_THETA_SOURCE_ID)?;
    check_fixed_string(obj, "bivector_basis_id", GATE2_BIVECTOR_BASIS_ID)?;
    check_fixed_string(obj, "antipodal_policy_id", GATE2_ANTIPODAL_POLICY_ID)?;
    check_fixed_string(obj, "spec_hash_raw_input_id", SPEC_HASH_RAW_INPUT_ID)?;
    check_fixed_string(obj, "spec_hash_input_id", SPEC_HASH_INPUT_ID)?;

    for key in OPTIONAL_FLOAT_MANIFEST_KEYS {
        let value = obj
            .get(*key)
            .ok_or(Gate2ManifestValidationError::MissingKey(key))?;
        if value.is_null() {
            continue;
        }
        let value_str = value
            .as_str()
            .ok_or(Gate2ManifestValidationError::InvalidFloatType { key })?;
        if !is_sci_17e(value_str) {
            return Err(Gate2ManifestValidationError::InvalidFloatString {
                key,
                value: value_str.to_string(),
            });
        }
    }

    Ok(())
}

fn check_fixed_string(
    obj: &Map<String, Value>,
    key: &'static str,
    expected: &'static str,
) -> Result<(), Gate2ManifestValidationError> {
    let actual = obj.get(key).and_then(Value::as_str).map(str::to_string);
    if actual.as_deref() == Some(expected) {
        Ok(())
    } else {
        Err(Gate2ManifestValidationError::InvalidFixedString {
            key,
            expected,
            actual,
        })
    }
}
fn convert_vec8_rows(
    sample_id: u64,
    rows: Vec<Vec<f64>>,
) -> Result<Vec<[f64; VEC8_DIM]>, Gate2OrchestratorError> {
    let mut out = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.into_iter().enumerate() {
        if row.len() != VEC8_DIM {
            return Err(Gate2OrchestratorError::InvalidVec8Dim {
                sample_id,
                row_index,
                expected: VEC8_DIM,
                actual: row.len(),
            });
        }
        let mut arr = [0.0_f64; VEC8_DIM];
        for (col_index, value) in row.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(Gate2OrchestratorError::NonFiniteVec8 {
                    sample_id,
                    row_index,
                    col_index,
                    value,
                });
            }
            arr[col_index] = value;
        }
        out.push(arr);
    }
    Ok(out)
}

fn aggregate_summary(samples: &[RunTelemetrySample]) -> Gate2RunSummary {
    let mut h1_values = Vec::new();
    let mut h2_values = Vec::new();
    let mut h3_total_values = Vec::new();
    let mut h3_loop_values = Vec::new();

    for sample in samples {
        if let Some(value) = sample.telemetry.h1b_closure_error {
            h1_values.push(value);
        }
        if let Some(stats) = sample.telemetry.h2_loop_stats {
            h2_values.push(stats.mean);
        }
        if let Some(value) = sample.telemetry.h3_ratio_total_product {
            h3_total_values.push(value);
        }
        if let Some(stats) = sample.telemetry.h3_ratio_triangle_loop_stats {
            h3_loop_values.push(stats.mean);
        }
    }

    Gate2RunSummary {
        n_samples_total: samples.len(),
        n_samples_h1b_present: h1_values.len(),
        n_samples_h2_present: h2_values.len(),
        n_samples_h3_total_present: h3_total_values.len(),
        h1b: aggregate_stat(&h1_values),
        h2: aggregate_stat(&h2_values),
        h3_total: aggregate_stat(&h3_total_values),
        h3_loop: aggregate_stat(&h3_loop_values),
    }
}

fn aggregate_stat(values: &[f64]) -> Option<Gate2AggregateStat> {
    if values.is_empty() {
        return None;
    }
    let max = values
        .iter()
        .copied()
        .max_by(|left, right| left.total_cmp(right))
        .unwrap_or(0.0);
    let mean = values.iter().sum::<f64>() / (values.len() as f64);
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.total_cmp(right));
    let p90 = nearest_rank(&sorted, 0.90);
    Some(Gate2AggregateStat { mean, p90, max })
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

fn build_manifest(
    identity: &Gate2IdentityInput,
    summary: &Gate2RunSummary,
) -> Result<Gate2ManifestJson, Gate2OrchestratorError> {
    Ok(Gate2ManifestJson {
        spec_version: GATE2_SPEC_VERSION.to_string(),
        method_id: GATE2_METHOD_ID.to_string(),
        algebra_id: ALGEBRA_ID.to_string(),
        blade_sign_id: BLADE_SIGN_ID.to_string(),
        reverse_id: REVERSE_ID.to_string(),
        normalize_id: NORMALIZE_ID.to_string(),
        composition_id: COMPOSITION_ID.to_string(),
        embed_id: EMBED_ID.to_string(),
        h3_name_id: H3_NAME_ID.to_string(),
        rotor_construction_id: GATE2_ROTOR_CONSTRUCTION_ID.to_string(),
        theta_source_id: GATE2_THETA_SOURCE_ID.to_string(),
        bivector_basis_id: GATE2_BIVECTOR_BASIS_ID.to_string(),
        antipodal_policy_id: GATE2_ANTIPODAL_POLICY_ID.to_string(),
        spec_hash_raw_blake3: identity.spec_hash_raw_blake3.clone(),
        spec_hash_raw_input_id: SPEC_HASH_RAW_INPUT_ID.to_string(),
        spec_hash_blake3: identity.spec_hash_blake3.clone(),
        spec_hash_input_id: SPEC_HASH_INPUT_ID.to_string(),
        dataset_revision_id: identity.dataset_revision_id.clone(),
        dataset_hash_blake3: identity.dataset_hash_blake3.clone(),
        code_git_commit: identity.code_git_commit.clone(),
        build_target_triple: identity.build_target_triple.clone(),
        rustc_version: identity.rustc_version.clone(),
        unitization_id: identity.unitization_id.clone(),
        rotor_encoder_id: identity.rotor_encoder_id.clone(),
        rotor_encoder_preproc_id: identity.rotor_encoder_preproc_id.clone(),
        vec8_postproc_id: identity.vec8_postproc_id.clone(),
        evaluation_mode_id: identity.evaluation_mode_id.clone(),
        n_samples_total: summary.n_samples_total,
        n_samples_h1b_present: summary.n_samples_h1b_present,
        n_samples_h2_present: summary.n_samples_h2_present,
        n_samples_h3_total_present: summary.n_samples_h3_total_present,
        h1b_mean: format_optional_stat(summary.h1b, |stat| stat.mean, "h1b_mean")?,
        h1b_p90: format_optional_stat(summary.h1b, |stat| stat.p90, "h1b_p90")?,
        h1b_max: format_optional_stat(summary.h1b, |stat| stat.max, "h1b_max")?,
        h2_mean: format_optional_stat(summary.h2, |stat| stat.mean, "h2_mean")?,
        h2_p90: format_optional_stat(summary.h2, |stat| stat.p90, "h2_p90")?,
        h2_max: format_optional_stat(summary.h2, |stat| stat.max, "h2_max")?,
        h3_total_mean: format_optional_stat(summary.h3_total, |stat| stat.mean, "h3_total_mean")?,
        h3_total_p90: format_optional_stat(summary.h3_total, |stat| stat.p90, "h3_total_p90")?,
        h3_total_max: format_optional_stat(summary.h3_total, |stat| stat.max, "h3_total_max")?,
        h3_loop_mean: format_optional_stat(summary.h3_loop, |stat| stat.mean, "h3_loop_mean")?,
        h3_loop_p90: format_optional_stat(summary.h3_loop, |stat| stat.p90, "h3_loop_p90")?,
        h3_loop_max: format_optional_stat(summary.h3_loop, |stat| stat.max, "h3_loop_max")?,
    })
}

fn format_optional_stat(
    stat: Option<Gate2AggregateStat>,
    select: impl Fn(Gate2AggregateStat) -> f64,
    field: &'static str,
) -> Result<Option<String>, Gate2OrchestratorError> {
    stat.map(|value| format_float_17e(field, select(value)))
        .transpose()
}

fn build_summary_csv(
    identity: &Gate2IdentityInput,
    summary: &Gate2RunSummary,
) -> Result<String, Gate2OrchestratorError> {
    let mut out = String::new();
    out.push_str(&GATE2_SUMMARY_CSV_COLUMNS_V1.join(","));
    out.push('\n');

    let values: Vec<String> = GATE2_SUMMARY_CSV_COLUMNS_V1
        .iter()
        .map(|column| summary_column_value(column, identity, summary))
        .collect::<Result<Vec<_>, _>>()?;

    out.push_str(&values.join(","));
    out.push('\n');
    Ok(out)
}

fn summary_column_value(
    column: &str,
    identity: &Gate2IdentityInput,
    summary: &Gate2RunSummary,
) -> Result<String, Gate2OrchestratorError> {
    match column {
        "spec_version" => Ok(GATE2_SPEC_VERSION.to_string()),
        "method_id" => Ok(GATE2_METHOD_ID.to_string()),
        "composition_id" => Ok(COMPOSITION_ID.to_string()),
        "dataset_revision_id" => Ok(identity.dataset_revision_id.clone()),
        "code_git_commit" => Ok(identity.code_git_commit.clone()),
        "build_target_triple" => Ok(identity.build_target_triple.clone()),
        "rustc_version" => Ok(identity.rustc_version.clone()),
        "n_samples_total" => Ok(summary.n_samples_total.to_string()),
        "n_samples_h1b_present" => Ok(summary.n_samples_h1b_present.to_string()),
        "n_samples_h2_present" => Ok(summary.n_samples_h2_present.to_string()),
        "n_samples_h3_total_present" => Ok(summary.n_samples_h3_total_present.to_string()),
        "h1b_mean" => format_csv_optional(summary.h1b.map(|stat| ("h1b_mean", stat.mean))),
        "h1b_p90" => format_csv_optional(summary.h1b.map(|stat| ("h1b_p90", stat.p90))),
        "h1b_max" => format_csv_optional(summary.h1b.map(|stat| ("h1b_max", stat.max))),
        "h2_mean" => format_csv_optional(summary.h2.map(|stat| ("h2_mean", stat.mean))),
        "h2_p90" => format_csv_optional(summary.h2.map(|stat| ("h2_p90", stat.p90))),
        "h2_max" => format_csv_optional(summary.h2.map(|stat| ("h2_max", stat.max))),
        "h3_total_mean" => {
            format_csv_optional(summary.h3_total.map(|stat| ("h3_total_mean", stat.mean)))
        }
        "h3_total_p90" => {
            format_csv_optional(summary.h3_total.map(|stat| ("h3_total_p90", stat.p90)))
        }
        "h3_total_max" => {
            format_csv_optional(summary.h3_total.map(|stat| ("h3_total_max", stat.max)))
        }
        "h3_loop_mean" => {
            format_csv_optional(summary.h3_loop.map(|stat| ("h3_loop_mean", stat.mean)))
        }
        "h3_loop_p90" => format_csv_optional(summary.h3_loop.map(|stat| ("h3_loop_p90", stat.p90))),
        "h3_loop_max" => format_csv_optional(summary.h3_loop.map(|stat| ("h3_loop_max", stat.max))),
        _ => Ok(String::new()),
    }
}
fn format_csv_optional(
    value: Option<(&'static str, f64)>,
) -> Result<String, Gate2OrchestratorError> {
    match value {
        Some((field, v)) => format_float_17e(field, v),
        None => Ok(String::new()),
    }
}

fn build_samples_csv(samples: &[RunTelemetrySample]) -> Result<String, Gate2OrchestratorError> {
    let mut out = String::new();
    out.push_str(&GATE2_SAMPLES_CSV_COLUMNS_V1.join(","));
    out.push('\n');

    let mut ordered: Vec<&RunTelemetrySample> = samples.iter().collect();
    ordered.sort_by(|left, right| {
        left.sample_id
            .cmp(&right.sample_id)
            .then(left.input_order.cmp(&right.input_order))
    });

    for sample in ordered {
        let mut row = Vec::with_capacity(GATE2_SAMPLES_CSV_COLUMNS_V1.len());
        row.push(sample.sample_id.to_string());
        row.push(sample.telemetry.n_ans_units.to_string());
        row.push(
            sample
                .sample_label
                .map(|v| v.to_string())
                .unwrap_or_default(),
        );
        row.push(
            sample
                .answer_length
                .map(|v| v.to_string())
                .unwrap_or_default(),
        );
        row.push(optional_float_csv(
            "h1b_closure_error",
            sample.telemetry.h1b_closure_error,
        )?);
        row.push(optional_stat_field_csv(
            "h2_loop_max",
            sample.telemetry.h2_loop_stats,
            |stat| stat.max,
        )?);
        row.push(optional_stat_field_csv(
            "h2_loop_mean",
            sample.telemetry.h2_loop_stats,
            |stat| stat.mean,
        )?);
        row.push(optional_stat_field_csv(
            "h2_loop_p90",
            sample.telemetry.h2_loop_stats,
            |stat| stat.p90,
        )?);
        row.push(optional_float_csv(
            "h3_ratio_total_product",
            sample.telemetry.h3_ratio_total_product,
        )?);
        row.push(optional_stat_field_csv(
            "h3_ratio_triangle_loop_max",
            sample.telemetry.h3_ratio_triangle_loop_stats,
            |stat| stat.max,
        )?);
        row.push(optional_stat_field_csv(
            "h3_ratio_triangle_loop_mean",
            sample.telemetry.h3_ratio_triangle_loop_stats,
            |stat| stat.mean,
        )?);
        row.push(optional_stat_field_csv(
            "h3_ratio_triangle_loop_p90",
            sample.telemetry.h3_ratio_triangle_loop_stats,
            |stat| stat.p90,
        )?);
        row.push(
            sample
                .telemetry
                .counts
                .count_missing_even_rotor_steps
                .to_string(),
        );
        row.push(sample.telemetry.counts.count_loops_considered.to_string());
        row.push(sample.telemetry.counts.count_loops_used.to_string());

        out.push_str(&row.join(","));
        out.push('\n');
    }

    Ok(out)
}

fn optional_float_csv(
    field: &'static str,
    value: Option<f64>,
) -> Result<String, Gate2OrchestratorError> {
    match value {
        Some(v) => format_float_17e(field, v),
        None => Ok(String::new()),
    }
}

fn optional_stat_field_csv(
    field: &'static str,
    stat: Option<Gate2StatTriple>,
    select: impl Fn(Gate2StatTriple) -> f64,
) -> Result<String, Gate2OrchestratorError> {
    match stat {
        Some(s) => format_float_17e(field, select(s)),
        None => Ok(String::new()),
    }
}

fn format_float_17e(field: &'static str, value: f64) -> Result<String, Gate2OrchestratorError> {
    if !value.is_finite() {
        return Err(Gate2OrchestratorError::InvalidFloat { field, value });
    }
    Ok(format!("{:.17e}", value))
}

fn write_string_lf(path: &Path, content: &str) -> Result<(), Gate2OrchestratorError> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    fs::write(path, normalized.as_bytes())?;
    Ok(())
}

fn write_bytes_lf(path: &Path, bytes: &[u8]) -> Result<(), Gate2OrchestratorError> {
    let content = std::str::from_utf8(bytes).map_err(|err| {
        Gate2OrchestratorError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid UTF-8 bytes: {}", err),
        ))
    })?;
    write_string_lf(path, content)
}

fn is_sci_17e(value: &str) -> bool {
    let Some(e_idx) = value.find('e') else {
        return false;
    };
    let (mantissa, exponent) = value.split_at(e_idx);
    let exponent = &exponent[1..];
    if mantissa.is_empty() || exponent.is_empty() {
        return false;
    }

    let mantissa = if let Some(rest) = mantissa.strip_prefix('-') {
        rest
    } else {
        mantissa
    };

    let mut mantissa_parts = mantissa.split('.');
    let int_part = match mantissa_parts.next() {
        Some(v) => v,
        None => return false,
    };
    let frac_part = match mantissa_parts.next() {
        Some(v) => v,
        None => return false,
    };

    if mantissa_parts.next().is_some() {
        return false;
    }

    if int_part.len() != 1 || !int_part.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }

    if frac_part.len() != 17 || !frac_part.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }

    let exp_body = if exponent.starts_with('-') || exponent.starts_with('+') {
        &exponent[1..]
    } else {
        exponent
    };
    !exp_body.is_empty() && exp_body.chars().all(|c| c.is_ascii_digit())
}

fn iter_string_values(path: String, value: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    match value {
        Value::String(s) => out.push((path, s.clone())),
        Value::Array(values) => {
            for (idx, item) in values.iter().enumerate() {
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
                    key.to_string()
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn identity_fixture() -> Gate2IdentityInput {
        Gate2IdentityInput {
            dataset_revision_id: "dataset_rev".to_string(),
            dataset_hash_blake3: "dataset_hash".to_string(),
            spec_hash_raw_blake3: "spec_raw".to_string(),
            spec_hash_blake3: "spec_lf".to_string(),
            unitization_id: "sentence_split_v1".to_string(),
            rotor_encoder_id: "encoder@rev".to_string(),
            rotor_encoder_preproc_id: "preproc_v1".to_string(),
            vec8_postproc_id: "vec8_postproc_v1".to_string(),
            evaluation_mode_id: "supervised_v1".to_string(),
            code_git_commit: "deadbeef".to_string(),
            build_target_triple: "x86_64-unknown-linux-gnu".to_string(),
            rustc_version: "rustc 1.75.0".to_string(),
        }
    }

    fn input_json_fixture() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "run_id": "gate2_fixture_run",
            "samples": [
                {
                    "sample_id": 1,
                    "ans_vec8": [
                        [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]
                    ],
                    "sample_label": 1,
                    "answer_length": 12
                },
                {
                    "sample_id": 2,
                    "ans_vec8": [
                        [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0]
                    ],
                    "sample_label": null,
                    "answer_length": 7
                },
                {
                    "sample_id": 3,
                    "ans_vec8": [
                        [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
                    ],
                    "sample_label": 0,
                    "answer_length": 20
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
            "pale-ale-diagnose-gate2-{}-{}-{}",
            prefix,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn gate2_orchestrator_writes_artifacts_and_valid_manifest() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("e2e");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate2_and_write(&out_dir, &input, &identity).expect("orchestrator");
        assert_eq!(output.run_id, "gate2_fixture_run");
        assert_eq!(output.spec_version, GATE2_SPEC_VERSION);
        assert!(output.artifact_paths.manifest_json.exists());
        assert!(output.artifact_paths.summary_csv.exists());
        assert!(output.artifact_paths.samples_csv.exists());

        let manifest_bytes =
            fs::read(&output.artifact_paths.manifest_json).expect("manifest bytes");
        validate_gate2_manifest_json(&manifest_bytes).expect("manifest valid");

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate2_orchestrator_is_deterministic_for_identical_input() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir_a = temp_dir("det-a");
        let out_dir_b = temp_dir("det-b");
        fs::create_dir_all(&out_dir_a).expect("mkdir a");
        fs::create_dir_all(&out_dir_b).expect("mkdir b");

        let out_a = run_gate2_and_write(&out_dir_a, &input, &identity).expect("run a");
        let out_b = run_gate2_and_write(&out_dir_b, &input, &identity).expect("run b");

        let manifest_a = fs::read(out_a.artifact_paths.manifest_json).expect("manifest a");
        let manifest_b = fs::read(out_b.artifact_paths.manifest_json).expect("manifest b");
        assert_eq!(manifest_a, manifest_b);

        let summary_a = fs::read(out_a.artifact_paths.summary_csv).expect("summary a");
        let summary_b = fs::read(out_b.artifact_paths.summary_csv).expect("summary b");
        assert_eq!(summary_a, summary_b);

        let samples_a = fs::read(out_a.artifact_paths.samples_csv).expect("samples a");
        let samples_b = fs::read(out_b.artifact_paths.samples_csv).expect("samples b");
        assert_eq!(samples_a, samples_b);

        let _ = fs::remove_dir_all(&out_dir_a);
        let _ = fs::remove_dir_all(&out_dir_b);
    }

    #[test]
    fn summary_columns_are_hard_locked() {
        let expected = [
            "spec_version",
            "method_id",
            "composition_id",
            "dataset_revision_id",
            "code_git_commit",
            "build_target_triple",
            "rustc_version",
            "n_samples_total",
            "n_samples_h1b_present",
            "n_samples_h2_present",
            "n_samples_h3_total_present",
            "h1b_mean",
            "h1b_p90",
            "h1b_max",
            "h2_mean",
            "h2_p90",
            "h2_max",
            "h3_total_mean",
            "h3_total_p90",
            "h3_total_max",
            "h3_loop_mean",
            "h3_loop_p90",
            "h3_loop_max",
        ];
        assert_eq!(GATE2_SUMMARY_CSV_COLUMNS_V1, expected.as_slice());
    }

    #[test]
    fn samples_columns_are_hard_locked() {
        let expected = [
            "sample_id",
            "n_ans_units",
            "sample_label",
            "answer_length",
            "h1b_closure_error",
            "h2_loop_max",
            "h2_loop_mean",
            "h2_loop_p90",
            "h3_ratio_total_product",
            "h3_ratio_triangle_loop_max",
            "h3_ratio_triangle_loop_mean",
            "h3_ratio_triangle_loop_p90",
            "missing_even_rotor_steps",
            "loops_considered",
            "loops_used",
        ];
        assert_eq!(GATE2_SAMPLES_CSV_COLUMNS_V1, expected.as_slice());
    }

    #[test]
    fn gate2_manifest_validator_rejects_missing_key() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("manifest-missing");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate2_and_write(&out_dir, &input, &identity).expect("run");
        let bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest read");
        let mut value: Value = serde_json::from_slice(&bytes).expect("json");
        value
            .as_object_mut()
            .expect("object")
            .remove("composition_id");
        let broken = serde_json::to_vec(&value).expect("to vec");

        let err = validate_gate2_manifest_json(&broken).expect_err("expected failure");
        assert!(matches!(
            err,
            Gate2ManifestValidationError::MissingKey("composition_id")
        ));

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate2_manifest_validator_rejects_bad_float_string() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("manifest-float");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate2_and_write(&out_dir, &input, &identity).expect("run");
        let bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest read");
        let mut value: Value = serde_json::from_slice(&bytes).expect("json");
        value
            .as_object_mut()
            .expect("object")
            .insert("h1b_mean".to_string(), Value::String("0.1".to_string()));
        let broken = serde_json::to_vec(&value).expect("to vec");

        let err = validate_gate2_manifest_json(&broken).expect_err("expected failure");
        assert!(matches!(
            err,
            Gate2ManifestValidationError::InvalidFloatString {
                key: "h1b_mean",
                ..
            }
        ));

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate2_manifest_validator_rejects_wrong_fixed_id() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("manifest-fixed");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate2_and_write(&out_dir, &input, &identity).expect("run");
        let bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest read");
        let mut value: Value = serde_json::from_slice(&bytes).expect("json");
        value
            .as_object_mut()
            .expect("object")
            .insert("method_id".to_string(), Value::String("wrong".to_string()));
        let broken = serde_json::to_vec(&value).expect("to vec");

        let err = validate_gate2_manifest_json(&broken).expect_err("expected failure");
        assert!(matches!(
            err,
            Gate2ManifestValidationError::InvalidFixedString {
                key: "method_id",
                ..
            }
        ));

        let _ = fs::remove_dir_all(&out_dir);
    }
}
