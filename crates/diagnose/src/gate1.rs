use crate::linking::{
    evaluate_link_sanity, process_sample_links, LinkRow, LinkSanityError, SampleLinkReport,
    SampleLinksInput,
};
use crate::manifest_validator::{validate_manifest_json, ValidationError};
use crate::rotor_diagnostics::{
    compute_rotor_diagnostics, RotorDiagnosticsError, RotorDiagnosticsInput,
};
use crate::run_eval::{compute_run_eval, RunEvalError, RunEvalInput, RunEvalResult, RunEvalSample};
use crate::writer::{
    write_gate1_artifacts, ArtifactPaths, Gate1WriterInput, WriteError, SPEC_VERSION,
};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;

const VEC8_DIM: usize = 8;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate1RunInputV1 {
    #[serde(default)]
    pub run_id: Option<String>,
    #[serde(default)]
    pub explicitly_unrelated_sample_ids: Vec<u64>,
    pub samples: Vec<Gate1SampleInputV1>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Gate1SampleInputV1 {
    pub sample_id: u64,
    pub doc_vec8: Vec<Vec<f64>>,
    pub ans_vec8: Vec<Vec<f64>>,
    pub links_topk: Vec<LinkRow>,
    pub sample_label: Option<u8>,
    pub answer_length: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct Gate1IdentityInput {
    pub spec_hash_raw_blake3: String,
    pub spec_hash_blake3: String,
    pub dataset_revision_id: String,
    pub dataset_hash_blake3: String,
    pub code_git_commit: String,
    pub build_target_triple: String,
    pub rustc_version: String,
    pub unitization_id: String,
    pub rotor_encoder_id: String,
    pub rotor_encoder_preproc_id: String,
    pub vec8_postproc_id: String,
    pub evaluation_mode_id: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Gate1RunOutput {
    pub run_id: String,
    pub spec_version: String,
    pub run_eval_result: RunEvalResult,
    pub artifact_paths: ArtifactPaths,
}

#[derive(Debug)]
pub enum Gate1OrchestratorError {
    JsonParse(serde_json::Error),
    InvalidVec8Dim {
        sample_id: u64,
        field: &'static str,
        row_index: usize,
        expected: usize,
        actual: usize,
    },
    NonFiniteVec8 {
        sample_id: u64,
        field: &'static str,
        row_index: usize,
        col_index: usize,
        value: f64,
    },
    InvalidSampleLabel {
        sample_id: u64,
        label: u8,
    },
    LinkSanity(LinkSanityError),
    RotorDiagnostics(RotorDiagnosticsError),
    RunEval(RunEvalError),
    Write(WriteError),
    ManifestRead(std::io::Error),
    ManifestValidation(ValidationError),
}

impl fmt::Display for Gate1OrchestratorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::JsonParse(err) => write!(f, "failed to parse Gate1 JSON v1: {}", err),
            Self::InvalidVec8Dim {
                sample_id,
                field,
                row_index,
                expected,
                actual,
            } => write!(
                f,
                "sample {} {}[{}] dimension mismatch: expected {}, got {}",
                sample_id, field, row_index, expected, actual
            ),
            Self::NonFiniteVec8 {
                sample_id,
                field,
                row_index,
                col_index,
                value,
            } => write!(
                f,
                "sample {} {}[{}][{}] is non-finite: {}",
                sample_id, field, row_index, col_index, value
            ),
            Self::InvalidSampleLabel { sample_id, label } => write!(
                f,
                "invalid sample_label for sample {}: {} (expected 0, 1, or null)",
                sample_id, label
            ),
            Self::LinkSanity(err) => write!(f, "link sanity error: {}", err),
            Self::RotorDiagnostics(err) => write!(f, "rotor diagnostics error: {}", err),
            Self::RunEval(err) => write!(f, "run eval error: {}", err),
            Self::Write(err) => write!(f, "artifact write error: {}", err),
            Self::ManifestRead(err) => write!(f, "failed to read manifest.json: {}", err),
            Self::ManifestValidation(err) => write!(f, "manifest validation error: {}", err),
        }
    }
}

impl std::error::Error for Gate1OrchestratorError {}

pub fn run_gate1_and_write<P: AsRef<Path>>(
    out_dir: P,
    input_json_bytes: &[u8],
    identity: &Gate1IdentityInput,
) -> Result<Gate1RunOutput, Gate1OrchestratorError> {
    let parsed: Gate1RunInputV1 =
        serde_json::from_slice(input_json_bytes).map_err(Gate1OrchestratorError::JsonParse)?;
    let run_id = parsed.run_id.unwrap_or_else(|| "gate1_run".to_string());

    let mut sample_reports = Vec::<SampleLinkReport>::with_capacity(parsed.samples.len());
    let mut run_eval_samples = Vec::<RunEvalSample>::with_capacity(parsed.samples.len());

    for sample in parsed.samples {
        if let Some(label) = sample.sample_label {
            if label > 1 {
                return Err(Gate1OrchestratorError::InvalidSampleLabel {
                    sample_id: sample.sample_id,
                    label,
                });
            }
        }

        let doc_vec8 = convert_vec8_rows(sample.sample_id, "doc_vec8", sample.doc_vec8)?;
        let ans_vec8 = convert_vec8_rows(sample.sample_id, "ans_vec8", sample.ans_vec8)?;

        let link_input = SampleLinksInput {
            sample_id: sample.sample_id,
            ans_unit_count: ans_vec8.len(),
            doc_unit_count: doc_vec8.len(),
            links_topk: sample.links_topk,
        };
        let sample_report = process_sample_links(&link_input);

        let rotor_input = RotorDiagnosticsInput {
            sample_id: sample.sample_id,
            links: sample_report.clone(),
            doc_vec8,
            ans_vec8,
        };
        let diagnostics = compute_rotor_diagnostics(&rotor_input)
            .map_err(Gate1OrchestratorError::RotorDiagnostics)?;

        sample_reports.push(sample_report);
        run_eval_samples.push(RunEvalSample {
            sample_id: sample.sample_id,
            sample_label: sample.sample_label,
            answer_length: sample.answer_length,
            diagnostics,
        });
    }

    let link_sanity =
        evaluate_link_sanity(&sample_reports, &parsed.explicitly_unrelated_sample_ids)
            .map_err(Gate1OrchestratorError::LinkSanity)?;

    let run_eval_input = RunEvalInput {
        samples: run_eval_samples,
        link_sanity,
    };
    let run_eval_result =
        compute_run_eval(&run_eval_input).map_err(Gate1OrchestratorError::RunEval)?;

    let writer_input = Gate1WriterInput {
        run_eval_input,
        run_eval_result: run_eval_result.clone(),
        sample_links: sample_reports,
        spec_hash_raw_blake3: identity.spec_hash_raw_blake3.clone(),
        spec_hash_blake3: identity.spec_hash_blake3.clone(),
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
    };

    let artifact_paths =
        write_gate1_artifacts(out_dir, &writer_input).map_err(Gate1OrchestratorError::Write)?;

    let manifest_bytes =
        fs::read(&artifact_paths.manifest_json).map_err(Gate1OrchestratorError::ManifestRead)?;
    validate_manifest_json(&manifest_bytes).map_err(Gate1OrchestratorError::ManifestValidation)?;

    Ok(Gate1RunOutput {
        run_id,
        spec_version: SPEC_VERSION.to_string(),
        run_eval_result,
        artifact_paths,
    })
}

fn convert_vec8_rows(
    sample_id: u64,
    field: &'static str,
    rows: Vec<Vec<f64>>,
) -> Result<Vec<[f64; VEC8_DIM]>, Gate1OrchestratorError> {
    let mut out = Vec::with_capacity(rows.len());
    for (row_index, row) in rows.into_iter().enumerate() {
        if row.len() != VEC8_DIM {
            return Err(Gate1OrchestratorError::InvalidVec8Dim {
                sample_id,
                field,
                row_index,
                expected: VEC8_DIM,
                actual: row.len(),
            });
        }
        let mut arr = [0.0_f64; VEC8_DIM];
        for (col_index, value) in row.into_iter().enumerate() {
            if !value.is_finite() {
                return Err(Gate1OrchestratorError::NonFiniteVec8 {
                    sample_id,
                    field,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn identity_fixture() -> Gate1IdentityInput {
        Gate1IdentityInput {
            spec_hash_raw_blake3: "spec_raw_hash".to_string(),
            spec_hash_blake3: "spec_lf_hash".to_string(),
            dataset_revision_id: "dataset_rev".to_string(),
            dataset_hash_blake3: "dataset_hash".to_string(),
            code_git_commit: "deadbeef".to_string(),
            build_target_triple: "x86_64-unknown-linux-gnu".to_string(),
            rustc_version: "rustc 1.75.0".to_string(),
            unitization_id: "sentence_split_v1".to_string(),
            rotor_encoder_id: "encoder@rev".to_string(),
            rotor_encoder_preproc_id: "preproc_v1".to_string(),
            vec8_postproc_id: "vec8_postproc_v1".to_string(),
            evaluation_mode_id: "supervised_v1".to_string(),
        }
    }

    fn input_json_fixture() -> Vec<u8> {
        serde_json::to_vec(&serde_json::json!({
            "run_id": "fixture_run",
            "explicitly_unrelated_sample_ids": [],
            "samples": [
                {
                    "sample_id": 1,
                    "doc_vec8": [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    "ans_vec8": [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    "links_topk": [
                        {"ans_unit_id": 0, "doc_unit_id": 0, "rank": 1},
                        {"ans_unit_id": 1, "doc_unit_id": 1, "rank": 1}
                    ],
                    "sample_label": 1,
                    "answer_length": 12
                },
                {
                    "sample_id": 2,
                    "doc_vec8": [
                        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    "ans_vec8": [
                        [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    ],
                    "links_topk": [
                        {"ans_unit_id": 0, "doc_unit_id": 0, "rank": 1}
                    ],
                    "sample_label": 0,
                    "answer_length": 24
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
            "pale-ale-diagnose-gate1-{}-{}-{}",
            prefix,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn orchestrator_writes_all_artifacts_and_validates_manifest() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir = temp_dir("e2e");
        fs::create_dir_all(&out_dir).expect("mkdir");

        let output = run_gate1_and_write(&out_dir, &input, &identity).expect("orchestrator");
        assert_eq!(output.run_id, "fixture_run");
        assert_eq!(output.spec_version, SPEC_VERSION);
        assert!(output.artifact_paths.manifest_json.exists());
        assert!(output.artifact_paths.summary_csv.exists());
        assert!(output.artifact_paths.link_topk_csv.exists());
        assert!(output.artifact_paths.link_sanity_md.exists());

        let manifest_bytes =
            fs::read(&output.artifact_paths.manifest_json).expect("read manifest bytes");
        validate_manifest_json(&manifest_bytes).expect("manifest valid");

        let _ = fs::remove_dir_all(&out_dir);
    }

    #[test]
    fn orchestrator_is_deterministic_for_identical_input() {
        let identity = identity_fixture();
        let input = input_json_fixture();
        let out_dir_a = temp_dir("det-a");
        let out_dir_b = temp_dir("det-b");
        fs::create_dir_all(&out_dir_a).expect("mkdir a");
        fs::create_dir_all(&out_dir_b).expect("mkdir b");

        let out_a = run_gate1_and_write(&out_dir_a, &input, &identity).expect("run a");
        let out_b = run_gate1_and_write(&out_dir_b, &input, &identity).expect("run b");

        let manifest_a = fs::read(out_a.artifact_paths.manifest_json).expect("manifest a");
        let manifest_b = fs::read(out_b.artifact_paths.manifest_json).expect("manifest b");
        assert_eq!(manifest_a, manifest_b);

        let summary_a = fs::read(out_a.artifact_paths.summary_csv).expect("summary a");
        let summary_b = fs::read(out_b.artifact_paths.summary_csv).expect("summary b");
        assert_eq!(summary_a, summary_b);

        let topk_a = fs::read(out_a.artifact_paths.link_topk_csv).expect("topk a");
        let topk_b = fs::read(out_b.artifact_paths.link_topk_csv).expect("topk b");
        assert_eq!(topk_a, topk_b);

        let sanity_a = fs::read(out_a.artifact_paths.link_sanity_md).expect("sanity a");
        let sanity_b = fs::read(out_b.artifact_paths.link_sanity_md).expect("sanity b");
        assert_eq!(sanity_a, sanity_b);

        let _ = fs::remove_dir_all(&out_dir_a);
        let _ = fs::remove_dir_all(&out_dir_b);
    }
}
