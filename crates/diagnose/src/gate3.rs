use crate::gate2_telemetry::{
    GATE2_ANTIPODAL_POLICY_ID, GATE2_BIVECTOR_BASIS_ID, GATE2_ROTOR_CONSTRUCTION_ID,
    GATE2_THETA_SOURCE_ID,
};
use crate::gate3_telemetry::{
    compute_gate3_telemetry, Gate3MissingReason, Gate3SampleTelemetry, Gate3TelemetryInput,
};
use pale_ale_rotor::{
    ALGEBRA_ID, BLADE_SIGN_ID, COMPOSITION_ID, EMBED_ID, NORMALIZE_ID, REVERSE_ID,
};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

const SPEC_HASH_RAW_INPUT_ID: &str = "spec_text_raw_utf8_v1";
const SPEC_HASH_INPUT_ID: &str = "spec_text_utf8_lf_v1";
const FLOAT_FORMAT_ID: &str = "sci_17e_v1";
const SUMMARY_SCHEMA_ID: &str = "gate3_summary_csv_v1";
const SAMPLES_SCHEMA_ID: &str = "gate3_samples_csv_v1";
const EPS_RATIO_VALUE: f64 = 1e-12;

pub const GATE3_SPEC_VERSION: &str = "v4.2.0-draft.1";
pub const GATE3_METHOD_ID: &str = "rotor_local_geometry_telemetry_v1";
pub const GATE3_CURVATURE_ID: &str = "local_curvature_projective_chordal_v1";
pub const GATE3_TORSION_ID: &str = "local_torsion_higher_grade_ratio_v1";

pub const GATE3_SUMMARY_CSV_COLUMNS_V1: &[&str] = &[
    "run_id",
    "n_samples_total",
    "n_samples_valid",
    "n_samples_missing",
    "kappa_count_total",
    "tau_count_total",
    "kappa_global_mean",
    "kappa_global_p90",
    "kappa_global_max",
    "tau_global_mean",
    "tau_global_p90",
    "tau_global_max",
];

pub const GATE3_SAMPLES_CSV_COLUMNS_V1: &[&str] = &[
    "sample_id",
    "sample_label",
    "answer_length",
    "steps_total",
    "rotors_total",
    "rotors_valid",
    "missing_even_rotor_steps",
    "kappa_count",
    "tau_count",
    "l3_kappa_max",
    "l3_kappa_mean",
    "l3_kappa_std",
    "l3_kappa_ratio",
    "l4_tau_max",
    "l4_tau_mean",
    "l4_tau_std",
    "l4_tau_p90",
    "missing_reason",
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
    "curvature_id",
    "torsion_id",
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
    "run_id",
    "n_explicitly_unrelated_sample_ids",
    "n_samples_total",
    "n_samples_valid",
    "n_samples_missing",
    "kappa_count_total",
    "tau_count_total",
    "missing_even_rotor_steps_total",
    "kappa_global_mean",
    "kappa_global_p90",
    "kappa_global_max",
    "tau_global_mean",
    "tau_global_p90",
    "tau_global_max",
    "float_format_id",
    "summary_schema_id",
    "samples_schema_id",
    "eps_ratio",
];

const OPTIONAL_FLOAT_MANIFEST_KEYS: &[&str] = &[
    "kappa_global_mean",
    "kappa_global_p90",
    "kappa_global_max",
    "tau_global_mean",
    "tau_global_p90",
    "tau_global_max",
];

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate3RunInputV1 {
    #[serde(default = "default_gate3_run_id")]
    pub run_id: String,
    #[serde(default)]
    pub explicitly_unrelated_sample_ids: Vec<u64>,
    pub samples: Vec<Gate3SampleInputV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate3SampleInputV1 {
    pub sample_id: u64,
    pub ans_vec8: Vec<Vec<f64>>,
    #[serde(default)]
    pub sample_label: Option<u8>,
    #[serde(default)]
    pub answer_length: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Gate3IdentityInput {
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
pub struct Gate3ArtifactPaths {
    pub manifest_json: PathBuf,
    pub summary_csv: PathBuf,
    pub samples_csv: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gate3AggregateStat {
    pub mean: f64,
    pub p90: f64,
    pub max: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate3RunSummary {
    pub n_samples_total: usize,
    pub n_samples_valid: usize,
    pub n_samples_missing: usize,
    pub kappa_count_total: usize,
    pub tau_count_total: usize,
    pub missing_even_rotor_steps_total: usize,
    pub kappa_global: Option<Gate3AggregateStat>,
    pub tau_global: Option<Gate3AggregateStat>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate3RunOutput {
    pub run_id: String,
    pub spec_version: String,
    pub summary: Gate3RunSummary,
    pub artifact_paths: Gate3ArtifactPaths,
}

#[derive(Debug)]
pub enum Gate3OrchestratorError {
    JsonParse(serde_json::Error),
    InvalidSampleLabel { sample_id: u64, label: u8 },
    InvalidEvaluationMode(String),
    InvalidFloat { field: &'static str, value: f64 },
    Io(std::io::Error),
    JsonWrite(serde_json::Error),
    ManifestRead(std::io::Error),
    ManifestValidation(Gate3ManifestValidationError),
}

impl fmt::Display for Gate3OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JsonParse(err) => write!(f, "failed to parse Gate3 JSON v1: {}", err),
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

impl std::error::Error for Gate3OrchestratorError {}

impl From<std::io::Error> for Gate3OrchestratorError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for Gate3OrchestratorError {
    fn from(value: serde_json::Error) -> Self {
        Self::JsonWrite(value)
    }
}

#[derive(Debug)]
pub enum Gate3ManifestValidationError {
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

impl fmt::Display for Gate3ManifestValidationError {
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

impl std::error::Error for Gate3ManifestValidationError {}

#[derive(Clone, Debug)]
struct RunTelemetrySample {
    input_order: usize,
    sample_id: u64,
    sample_label: Option<u8>,
    answer_length: Option<usize>,
    telemetry: Gate3SampleTelemetry,
}

#[derive(Serialize)]
struct Gate3ManifestJson {
    spec_version: String,
    method_id: String,
    algebra_id: String,
    blade_sign_id: String,
    reverse_id: String,
    normalize_id: String,
    composition_id: String,
    embed_id: String,
    curvature_id: String,
    torsion_id: String,
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
    run_id: String,
    n_explicitly_unrelated_sample_ids: usize,
    n_samples_total: usize,
    n_samples_valid: usize,
    n_samples_missing: usize,
    kappa_count_total: usize,
    tau_count_total: usize,
    missing_even_rotor_steps_total: usize,
    kappa_global_mean: Option<String>,
    kappa_global_p90: Option<String>,
    kappa_global_max: Option<String>,
    tau_global_mean: Option<String>,
    tau_global_p90: Option<String>,
    tau_global_max: Option<String>,
    float_format_id: String,
    summary_schema_id: String,
    samples_schema_id: String,
    eps_ratio: String,
}

fn default_gate3_run_id() -> String {
    "gate3_run".to_string()
}

pub fn run_gate3_and_write<P: AsRef<Path>>(
    out_dir: P,
    input_json_bytes: &[u8],
    identity: &Gate3IdentityInput,
) -> Result<Gate3RunOutput, Gate3OrchestratorError> {
    if identity.evaluation_mode_id != "supervised_v1"
        && identity.evaluation_mode_id != "unsupervised_v1"
    {
        return Err(Gate3OrchestratorError::InvalidEvaluationMode(
            identity.evaluation_mode_id.clone(),
        ));
    }

    let parsed: Gate3RunInputV1 =
        serde_json::from_slice(input_json_bytes).map_err(Gate3OrchestratorError::JsonParse)?;

    let mut samples_plan: Vec<(usize, Gate3SampleInputV1)> =
        parsed.samples.into_iter().enumerate().collect();
    samples_plan.sort_by(|left, right| {
        left.1
            .sample_id
            .cmp(&right.1.sample_id)
            .then(left.0.cmp(&right.0))
    });

    let mut samples = Vec::with_capacity(samples_plan.len());
    for (input_order, sample) in samples_plan {
        if let Some(label) = sample.sample_label {
            if label > 1 {
                return Err(Gate3OrchestratorError::InvalidSampleLabel {
                    sample_id: sample.sample_id,
                    label,
                });
            }
        }

        let telemetry = compute_gate3_telemetry(&Gate3TelemetryInput {
            sample_id: sample.sample_id,
            ans_vec8: sample.ans_vec8,
            sample_label: sample.sample_label,
            answer_length: sample.answer_length,
        });

        samples.push(RunTelemetrySample {
            input_order,
            sample_id: sample.sample_id,
            sample_label: sample.sample_label,
            answer_length: sample.answer_length,
            telemetry,
        });
    }

    let summary = aggregate_summary(&samples);
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let manifest = build_manifest(
        identity,
        &parsed.run_id,
        parsed.explicitly_unrelated_sample_ids.len(),
        &summary,
    )?;
    let manifest_bytes =
        serde_json::to_vec_pretty(&manifest).map_err(Gate3OrchestratorError::JsonWrite)?;
    validate_gate3_manifest_json(&manifest_bytes)
        .map_err(Gate3OrchestratorError::ManifestValidation)?;

    let manifest_path = out_dir.join("manifest.json");
    write_bytes_lf(&manifest_path, &manifest_bytes)?;

    let summary_csv = build_summary_csv(&parsed.run_id, &summary)?;
    let summary_path = out_dir.join("summary.csv");
    write_string_lf(&summary_path, &summary_csv)?;

    let samples_csv = build_samples_csv(&samples)?;
    let samples_path = out_dir.join("samples.csv");
    write_string_lf(&samples_path, &samples_csv)?;

    let manifest_readback =
        fs::read(&manifest_path).map_err(Gate3OrchestratorError::ManifestRead)?;
    validate_gate3_manifest_json(&manifest_readback)
        .map_err(Gate3OrchestratorError::ManifestValidation)?;

    Ok(Gate3RunOutput {
        run_id: parsed.run_id,
        spec_version: GATE3_SPEC_VERSION.to_string(),
        summary,
        artifact_paths: Gate3ArtifactPaths {
            manifest_json: manifest_path,
            summary_csv: summary_path,
            samples_csv: samples_path,
        },
    })
}

pub fn validate_gate3_manifest_json(bytes: &[u8]) -> Result<(), Gate3ManifestValidationError> {
    std::str::from_utf8(bytes)
        .map_err(|err| Gate3ManifestValidationError::Utf8(err.to_string()))?;

    let root: Value = serde_json::from_slice(bytes).map_err(Gate3ManifestValidationError::Json)?;
    for (_path, value) in iter_string_values(String::new(), &root) {
        if value.contains("NaN") {
            return Err(Gate3ManifestValidationError::ForbiddenToken("NaN"));
        }
        if value.contains("Inf") {
            return Err(Gate3ManifestValidationError::ForbiddenToken("Inf"));
        }
    }

    let obj = root
        .as_object()
        .ok_or(Gate3ManifestValidationError::RootNotObject)?;
    for key in REQUIRED_MANIFEST_KEYS {
        if !obj.contains_key(*key) {
            return Err(Gate3ManifestValidationError::MissingKey(key));
        }
    }

    check_fixed_string(obj, "spec_version", GATE3_SPEC_VERSION)?;
    check_fixed_string(obj, "method_id", GATE3_METHOD_ID)?;
    check_fixed_string(obj, "algebra_id", ALGEBRA_ID)?;
    check_fixed_string(obj, "blade_sign_id", BLADE_SIGN_ID)?;
    check_fixed_string(obj, "reverse_id", REVERSE_ID)?;
    check_fixed_string(obj, "normalize_id", NORMALIZE_ID)?;
    check_fixed_string(obj, "composition_id", COMPOSITION_ID)?;
    check_fixed_string(obj, "embed_id", EMBED_ID)?;
    check_fixed_string(obj, "curvature_id", GATE3_CURVATURE_ID)?;
    check_fixed_string(obj, "torsion_id", GATE3_TORSION_ID)?;
    check_fixed_string(obj, "rotor_construction_id", GATE2_ROTOR_CONSTRUCTION_ID)?;
    check_fixed_string(obj, "theta_source_id", GATE2_THETA_SOURCE_ID)?;
    check_fixed_string(obj, "bivector_basis_id", GATE2_BIVECTOR_BASIS_ID)?;
    check_fixed_string(obj, "antipodal_policy_id", GATE2_ANTIPODAL_POLICY_ID)?;
    check_fixed_string(obj, "spec_hash_raw_input_id", SPEC_HASH_RAW_INPUT_ID)?;
    check_fixed_string(obj, "spec_hash_input_id", SPEC_HASH_INPUT_ID)?;
    check_fixed_string(obj, "float_format_id", FLOAT_FORMAT_ID)?;
    check_fixed_string(obj, "summary_schema_id", SUMMARY_SCHEMA_ID)?;
    check_fixed_string(obj, "samples_schema_id", SAMPLES_SCHEMA_ID)?;

    for key in OPTIONAL_FLOAT_MANIFEST_KEYS {
        let value = obj
            .get(*key)
            .ok_or(Gate3ManifestValidationError::MissingKey(key))?;
        if value.is_null() {
            continue;
        }
        let value_str = value
            .as_str()
            .ok_or(Gate3ManifestValidationError::InvalidFloatType { key })?;
        if !is_sci_17e(value_str) {
            return Err(Gate3ManifestValidationError::InvalidFloatString {
                key,
                value: value_str.to_string(),
            });
        }
    }

    let eps_ratio = obj
        .get("eps_ratio")
        .ok_or(Gate3ManifestValidationError::MissingKey("eps_ratio"))?
        .as_str()
        .ok_or(Gate3ManifestValidationError::InvalidFloatType { key: "eps_ratio" })?;
    if !is_sci_17e(eps_ratio) {
        return Err(Gate3ManifestValidationError::InvalidFloatString {
            key: "eps_ratio",
            value: eps_ratio.to_string(),
        });
    }

    Ok(())
}

fn check_fixed_string(
    obj: &Map<String, Value>,
    key: &'static str,
    expected: &'static str,
) -> Result<(), Gate3ManifestValidationError> {
    let actual = obj.get(key).and_then(Value::as_str).map(str::to_string);
    if actual.as_deref() == Some(expected) {
        Ok(())
    } else {
        Err(Gate3ManifestValidationError::InvalidFixedString {
            key,
            expected,
            actual,
        })
    }
}

fn aggregate_summary(samples: &[RunTelemetrySample]) -> Gate3RunSummary {
    let mut kappa_values = Vec::new();
    let mut tau_values = Vec::new();
    let mut kappa_count_total = 0usize;
    let mut tau_count_total = 0usize;
    let mut missing_even_rotor_steps_total = 0usize;
    let mut n_samples_valid = 0usize;

    for sample in samples {
        kappa_count_total += sample.telemetry.kappa_count;
        tau_count_total += sample.telemetry.tau_count;
        missing_even_rotor_steps_total += sample.telemetry.count_missing_even_rotor_steps;

        if sample.telemetry.missing_reason.is_none() {
            n_samples_valid += 1;
            kappa_values.push(sample.telemetry.l3_kappa_mean);
            tau_values.push(sample.telemetry.l4_tau_mean);
        }
    }

    let n_samples_total = samples.len();
    Gate3RunSummary {
        n_samples_total,
        n_samples_valid,
        n_samples_missing: n_samples_total.saturating_sub(n_samples_valid),
        kappa_count_total,
        tau_count_total,
        missing_even_rotor_steps_total,
        kappa_global: aggregate_stat(&kappa_values),
        tau_global: aggregate_stat(&tau_values),
    }
}

fn aggregate_stat(values: &[f64]) -> Option<Gate3AggregateStat> {
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
    let idx = nearest_rank_index(sorted.len(), 0.90);
    let p90 = sorted[idx];
    Some(Gate3AggregateStat { mean, p90, max })
}

fn nearest_rank_index(n: usize, p: f64) -> usize {
    if n == 0 {
        return 0;
    }
    let p = p.clamp(0.0, 1.0);
    let idx_f = (p * (n as f64)).ceil() - 1.0;
    if idx_f.is_nan() || idx_f < 0.0 {
        0
    } else if idx_f >= (n as f64) {
        n - 1
    } else {
        idx_f as usize
    }
}

fn build_manifest(
    identity: &Gate3IdentityInput,
    run_id: &str,
    n_explicitly_unrelated_sample_ids: usize,
    summary: &Gate3RunSummary,
) -> Result<Gate3ManifestJson, Gate3OrchestratorError> {
    Ok(Gate3ManifestJson {
        spec_version: GATE3_SPEC_VERSION.to_string(),
        method_id: GATE3_METHOD_ID.to_string(),
        algebra_id: ALGEBRA_ID.to_string(),
        blade_sign_id: BLADE_SIGN_ID.to_string(),
        reverse_id: REVERSE_ID.to_string(),
        normalize_id: NORMALIZE_ID.to_string(),
        composition_id: COMPOSITION_ID.to_string(),
        embed_id: EMBED_ID.to_string(),
        curvature_id: GATE3_CURVATURE_ID.to_string(),
        torsion_id: GATE3_TORSION_ID.to_string(),
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
        run_id: run_id.to_string(),
        n_explicitly_unrelated_sample_ids,
        n_samples_total: summary.n_samples_total,
        n_samples_valid: summary.n_samples_valid,
        n_samples_missing: summary.n_samples_missing,
        kappa_count_total: summary.kappa_count_total,
        tau_count_total: summary.tau_count_total,
        missing_even_rotor_steps_total: summary.missing_even_rotor_steps_total,
        kappa_global_mean: format_optional_stat(
            summary.kappa_global,
            |stat| stat.mean,
            "kappa_global_mean",
        )?,
        kappa_global_p90: format_optional_stat(
            summary.kappa_global,
            |stat| stat.p90,
            "kappa_global_p90",
        )?,
        kappa_global_max: format_optional_stat(
            summary.kappa_global,
            |stat| stat.max,
            "kappa_global_max",
        )?,
        tau_global_mean: format_optional_stat(
            summary.tau_global,
            |stat| stat.mean,
            "tau_global_mean",
        )?,
        tau_global_p90: format_optional_stat(
            summary.tau_global,
            |stat| stat.p90,
            "tau_global_p90",
        )?,
        tau_global_max: format_optional_stat(
            summary.tau_global,
            |stat| stat.max,
            "tau_global_max",
        )?,
        float_format_id: FLOAT_FORMAT_ID.to_string(),
        summary_schema_id: SUMMARY_SCHEMA_ID.to_string(),
        samples_schema_id: SAMPLES_SCHEMA_ID.to_string(),
        eps_ratio: format_float_17e("eps_ratio", EPS_RATIO_VALUE)?,
    })
}

fn format_optional_stat(
    stat: Option<Gate3AggregateStat>,
    select: impl Fn(Gate3AggregateStat) -> f64,
    field: &'static str,
) -> Result<Option<String>, Gate3OrchestratorError> {
    stat.map(|value| format_float_17e(field, select(value)))
        .transpose()
}

fn build_summary_csv(
    run_id: &str,
    summary: &Gate3RunSummary,
) -> Result<String, Gate3OrchestratorError> {
    let mut out = String::new();
    out.push_str(&GATE3_SUMMARY_CSV_COLUMNS_V1.join(","));
    out.push('\n');

    let values: Vec<String> = GATE3_SUMMARY_CSV_COLUMNS_V1
        .iter()
        .map(|column| summary_column_value(column, run_id, summary))
        .collect::<Result<Vec<_>, _>>()?;

    out.push_str(&values.join(","));
    out.push('\n');
    Ok(out)
}

fn summary_column_value(
    column: &str,
    run_id: &str,
    summary: &Gate3RunSummary,
) -> Result<String, Gate3OrchestratorError> {
    match column {
        "run_id" => Ok(run_id.to_string()),
        "n_samples_total" => Ok(summary.n_samples_total.to_string()),
        "n_samples_valid" => Ok(summary.n_samples_valid.to_string()),
        "n_samples_missing" => Ok(summary.n_samples_missing.to_string()),
        "kappa_count_total" => Ok(summary.kappa_count_total.to_string()),
        "tau_count_total" => Ok(summary.tau_count_total.to_string()),
        "kappa_global_mean" => {
            format_csv_optional(summary.kappa_global.map(|v| ("kappa_global_mean", v.mean)))
        }
        "kappa_global_p90" => {
            format_csv_optional(summary.kappa_global.map(|v| ("kappa_global_p90", v.p90)))
        }
        "kappa_global_max" => {
            format_csv_optional(summary.kappa_global.map(|v| ("kappa_global_max", v.max)))
        }
        "tau_global_mean" => {
            format_csv_optional(summary.tau_global.map(|v| ("tau_global_mean", v.mean)))
        }
        "tau_global_p90" => {
            format_csv_optional(summary.tau_global.map(|v| ("tau_global_p90", v.p90)))
        }
        "tau_global_max" => {
            format_csv_optional(summary.tau_global.map(|v| ("tau_global_max", v.max)))
        }
        _ => Ok(String::new()),
    }
}

fn format_csv_optional(
    value: Option<(&'static str, f64)>,
) -> Result<String, Gate3OrchestratorError> {
    match value {
        Some((field, v)) => format_float_17e(field, v),
        None => Ok(String::new()),
    }
}

fn build_samples_csv(samples: &[RunTelemetrySample]) -> Result<String, Gate3OrchestratorError> {
    let mut out = String::new();
    out.push_str(&GATE3_SAMPLES_CSV_COLUMNS_V1.join(","));
    out.push('\n');

    let mut ordered: Vec<&RunTelemetrySample> = samples.iter().collect();
    ordered.sort_by(|left, right| {
        left.sample_id
            .cmp(&right.sample_id)
            .then(left.input_order.cmp(&right.input_order))
    });

    for sample in ordered {
        let mut row = Vec::with_capacity(GATE3_SAMPLES_CSV_COLUMNS_V1.len());
        row.push(sample.sample_id.to_string());
        row.push(
            sample
                .sample_label
                .map(|value| value.to_string())
                .unwrap_or_default(),
        );
        row.push(
            sample
                .answer_length
                .map(|value| value.to_string())
                .unwrap_or_default(),
        );
        row.push(sample.telemetry.count_steps_total.to_string());
        row.push(sample.telemetry.count_rotors_total.to_string());
        row.push(sample.telemetry.count_rotors_valid.to_string());
        row.push(sample.telemetry.count_missing_even_rotor_steps.to_string());
        row.push(sample.telemetry.kappa_count.to_string());
        row.push(sample.telemetry.tau_count.to_string());
        row.push(sample_metric_field_csv(
            "l3_kappa_max",
            sample.telemetry.missing_reason,
            sample.telemetry.l3_kappa_max,
        )?);
        row.push(sample_metric_field_csv(
            "l3_kappa_mean",
            sample.telemetry.missing_reason,
            sample.telemetry.l3_kappa_mean,
        )?);
        row.push(sample_metric_field_csv(
            "l3_kappa_std",
            sample.telemetry.missing_reason,
            sample.telemetry.l3_kappa_std,
        )?);
        row.push(sample_metric_field_csv(
            "l3_kappa_ratio",
            sample.telemetry.missing_reason,
            sample.telemetry.l3_kappa_ratio,
        )?);
        row.push(sample_metric_field_csv(
            "l4_tau_max",
            sample.telemetry.missing_reason,
            sample.telemetry.l4_tau_max,
        )?);
        row.push(sample_metric_field_csv(
            "l4_tau_mean",
            sample.telemetry.missing_reason,
            sample.telemetry.l4_tau_mean,
        )?);
        row.push(sample_metric_field_csv(
            "l4_tau_std",
            sample.telemetry.missing_reason,
            sample.telemetry.l4_tau_std,
        )?);
        row.push(sample_metric_field_csv(
            "l4_tau_p90",
            sample.telemetry.missing_reason,
            sample.telemetry.l4_tau_p90,
        )?);
        row.push(
            sample
                .telemetry
                .missing_reason
                .map(gate3_missing_reason_str)
                .unwrap_or_default()
                .to_string(),
        );

        out.push_str(&row.join(","));
        out.push('\n');
    }

    Ok(out)
}

fn gate3_missing_reason_str(reason: Gate3MissingReason) -> &'static str {
    match reason {
        Gate3MissingReason::TooFewSteps => "too_few_steps",
        Gate3MissingReason::InvalidVec8 => "invalid_vec8",
        Gate3MissingReason::AllStepsMissing => "all_steps_missing",
        Gate3MissingReason::InsufficientAdjacentRotors => "insufficient_adjacent_rotors",
    }
}

fn sample_metric_field_csv(
    field: &'static str,
    missing_reason: Option<Gate3MissingReason>,
    value: f64,
) -> Result<String, Gate3OrchestratorError> {
    if missing_reason.is_some() {
        Ok(String::new())
    } else {
        format_float_17e(field, value)
    }
}

fn format_float_17e(field: &'static str, value: f64) -> Result<String, Gate3OrchestratorError> {
    if !value.is_finite() {
        return Err(Gate3OrchestratorError::InvalidFloat { field, value });
    }
    Ok(format!("{:.17e}", value))
}

fn write_string_lf(path: &Path, content: &str) -> Result<(), Gate3OrchestratorError> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    fs::write(path, normalized.as_bytes())?;
    Ok(())
}

fn write_bytes_lf(path: &Path, bytes: &[u8]) -> Result<(), Gate3OrchestratorError> {
    let content = std::str::from_utf8(bytes).map_err(|err| {
        Gate3OrchestratorError::Io(std::io::Error::new(
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

    fn identity_fixture() -> Gate3IdentityInput {
        Gate3IdentityInput {
            dataset_revision_id: "dataset_rev".to_string(),
            dataset_hash_blake3: "dataset_hash".to_string(),
            spec_hash_raw_blake3: "spec_raw".to_string(),
            spec_hash_blake3: "spec_lf".to_string(),
            unitization_id: "sentence_split_v1".to_string(),
            rotor_encoder_id: "encoder@rev".to_string(),
            rotor_encoder_preproc_id: "preproc_v1".to_string(),
            vec8_postproc_id: "vec8_postproc_v1".to_string(),
            evaluation_mode_id: "unsupervised_v1".to_string(),
            code_git_commit: "deadbeef".to_string(),
            build_target_triple: "x86_64-unknown-linux-gnu".to_string(),
            rustc_version: "rustc 1.75.0".to_string(),
        }
    }

    fn input_json_fixture() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "run_id": "gate3_fixture_run",
            "explicitly_unrelated_sample_ids": [42],
            "samples": [
                {
                    "sample_id": 3,
                    "ans_vec8": [
                        [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
                    ],
                    "sample_label": 1,
                    "answer_length": 12
                },
                {
                    "sample_id": 1,
                    "ans_vec8": [
                        [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [-1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0]
                    ],
                    "sample_label": null,
                    "answer_length": 7
                },
                {
                    "sample_id": 2,
                    "ans_vec8": [
                        [1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,1.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,1.0,0.0,0.0,0.0,0.0,0.0],
                        [0.0,0.0,0.0,1.0,0.0,0.0,0.0,0.0]
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
            "pale-ale-diagnose-gate3-{}-{}-{}",
            prefix,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn gate3_orchestrator_writes_artifacts_and_valid_manifest() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("e2e");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate3_and_write(&out_dir, &input, &identity).expect("orchestrator");
        assert_eq!(output.run_id, "gate3_fixture_run");
        assert_eq!(output.spec_version, GATE3_SPEC_VERSION);
        assert!(output.artifact_paths.manifest_json.exists());
        assert!(output.artifact_paths.summary_csv.exists());
        assert!(output.artifact_paths.samples_csv.exists());

        let manifest_bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest");
        validate_gate3_manifest_json(&manifest_bytes).expect("manifest valid");

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate3_orchestrator_is_deterministic_for_identical_input() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir_a = temp_dir("det-a");
        let out_dir_b = temp_dir("det-b");
        fs::create_dir_all(&out_dir_a).expect("mkdir a");
        fs::create_dir_all(&out_dir_b).expect("mkdir b");

        let out_a = run_gate3_and_write(&out_dir_a, &input, &identity).expect("run a");
        let out_b = run_gate3_and_write(&out_dir_b, &input, &identity).expect("run b");

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
            "run_id",
            "n_samples_total",
            "n_samples_valid",
            "n_samples_missing",
            "kappa_count_total",
            "tau_count_total",
            "kappa_global_mean",
            "kappa_global_p90",
            "kappa_global_max",
            "tau_global_mean",
            "tau_global_p90",
            "tau_global_max",
        ];
        assert_eq!(GATE3_SUMMARY_CSV_COLUMNS_V1, expected.as_slice());
    }

    #[test]
    fn samples_columns_are_hard_locked() {
        let expected = [
            "sample_id",
            "sample_label",
            "answer_length",
            "steps_total",
            "rotors_total",
            "rotors_valid",
            "missing_even_rotor_steps",
            "kappa_count",
            "tau_count",
            "l3_kappa_max",
            "l3_kappa_mean",
            "l3_kappa_std",
            "l3_kappa_ratio",
            "l4_tau_max",
            "l4_tau_mean",
            "l4_tau_std",
            "l4_tau_p90",
            "missing_reason",
        ];
        assert_eq!(GATE3_SAMPLES_CSV_COLUMNS_V1, expected.as_slice());
    }

    #[test]
    fn samples_csv_is_sorted_by_sample_id() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("sorted");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate3_and_write(&out_dir, &input, &identity).expect("run");
        let rows = fs::read_to_string(output.artifact_paths.samples_csv).expect("rows");
        let lines: Vec<&str> = rows.lines().collect();
        assert!(lines.len() >= 4);
        assert!(lines[1].starts_with("1,"));
        assert!(lines[2].starts_with("2,"));
        assert!(lines[3].starts_with("3,"));

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate3_manifest_validator_rejects_missing_key() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("manifest-missing");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate3_and_write(&out_dir, &input, &identity).expect("run");
        let bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest read");
        let mut value: Value = serde_json::from_slice(&bytes).expect("json");
        value
            .as_object_mut()
            .expect("object")
            .remove("curvature_id");
        let broken = serde_json::to_vec(&value).expect("to vec");

        let err = validate_gate3_manifest_json(&broken).expect_err("expected failure");
        assert!(matches!(
            err,
            Gate3ManifestValidationError::MissingKey("curvature_id")
        ));

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate3_manifest_validator_rejects_wrong_fixed_id() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("manifest-fixed");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate3_and_write(&out_dir, &input, &identity).expect("run");
        let bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest read");
        let mut value: Value = serde_json::from_slice(&bytes).expect("json");
        value
            .as_object_mut()
            .expect("object")
            .insert("method_id".to_string(), Value::String("wrong".to_string()));
        let broken = serde_json::to_vec(&value).expect("to vec");

        let err = validate_gate3_manifest_json(&broken).expect_err("expected failure");
        assert!(matches!(
            err,
            Gate3ManifestValidationError::InvalidFixedString {
                key: "method_id",
                ..
            }
        ));

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn gate3_manifest_validator_rejects_bad_float_string() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("manifest-float");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate3_and_write(&out_dir, &input, &identity).expect("run");
        let bytes = fs::read(&output.artifact_paths.manifest_json).expect("manifest read");
        let mut value: Value = serde_json::from_slice(&bytes).expect("json");
        value
            .as_object_mut()
            .expect("object")
            .insert("eps_ratio".to_string(), Value::String("0.1".to_string()));
        let broken = serde_json::to_vec(&value).expect("to vec");

        let err = validate_gate3_manifest_json(&broken).expect_err("expected failure");
        assert!(matches!(
            err,
            Gate3ManifestValidationError::InvalidFloatString {
                key: "eps_ratio",
                ..
            }
        ));

        let _ = fs::remove_dir_all(&out_dir);
    }
}
