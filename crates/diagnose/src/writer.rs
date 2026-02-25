use crate::confounds::{
    compute_antipodal_warning, compute_confounds, AntipodalWarningOutputs, ConfoundOutputs,
};
use crate::linking::{
    LinkSanityResult, SampleLinkReport, LINKS_TOPK_CANONICALIZATION_ID, LINK_SANITY_ID,
    TOP1_POLICY_ID,
};
use crate::manifest_validator::{validate_manifest_json, ValidationError};
use crate::rotor_diagnostics::{
    DISTANCE_ID, EPS_DIST, EPS_NORM, MAX_ANTIPODAL_DROP_RATE, METHOD_ID, MIN_PLANES, MIN_ROTORS,
    TAU_WEDGE, THETA_SOURCE_ID, TRIMMED_BEST_ID,
};
use crate::run_eval::{
    nearest_rank, ratio, AucUndefinedReason, CollapseInvalidReason, RunEvalInput, RunEvalResult,
    RunInvalidReason, AUC_ALGORITHM_ID, COLLAPSE_RATE_ANTIPODAL_DROP_THRESHOLD,
    COLLAPSE_RATE_COLLINEAR_THRESHOLD, PRIMARY_EXCLUSION_CEILING, QUANTILE_ID, RANK_METHOD_ID,
};
use pale_ale_rotor::RotorConfig;
use serde::Serialize;
use std::collections::BTreeMap;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

pub const SPEC_VERSION: &str = "v4.0.0-ssot.9";
pub const SPEC_HASH_RAW_INPUT_ID: &str = "spec_text_raw_utf8_v1";
pub const SPEC_HASH_INPUT_ID: &str = "spec_text_utf8_lf_v1";
pub const STATS_ID: &str = "none";
pub const LABEL_MISSING_POLICY_ID: &str = "exclude_sample_on_missing_halluc_unit_v1";
pub const QUANTILE_TRACK_SCOPE: &str = "top1_only_primary_v1";
pub const SUMMARY_SCHEMA_ID: &str = "summary_csv_schema_v1";
pub const LINK_TOPK_SCHEMA_ID: &str = "link_topk_csv_schema_v1";
pub const REASON_ENUM_ID: &str = "reason_ids_v1";
pub const METRIC_MISSING_ENUM_ID: &str = "metric_missing_reason_ids_v1";
pub const FLOAT_FORMAT_ID: &str = "sci_17e_v1";
pub const DETERMINISM_SCOPE: &str = "same_binary_same_target_same_dataset";
pub const ROTOR_CONSTRUCTION_ID: &str = "simple_rotor29_uv_v1";
pub const BIVECTOR_BASIS_ID: &str = "lex_i_lt_j_v1";
pub const ANTIPODAL_POLICY_ID: &str = "antipodal_split_v1(angle_only_for_theta,drop_on_nonfinite)";

pub const SUMMARY_CSV_COLUMNS_V1: &[&str] = &[
    "run_valid",
    "run_invalid_reason",
    "collapse_invalid_reason",
    "primary_auc",
    "primary_auc_n_pos",
    "primary_auc_n_neg",
    "auc_undefined_reason",
    "n_supervised_eligible",
    "n_supervised_used_primary",
    "n_supervised_excluded_primary",
    "primary_exclusion_rate",
    "label_missing_rate",
    "link_sanity_pass",
    "link_sanity_unrelated_count",
    "h_norm",
    "max_share",
    "dot_p01",
    "dot_p50",
    "dot_p90",
    "dot_p99",
    "wedge_norm_p01",
    "wedge_norm_p50",
    "wedge_norm_p90",
    "wedge_norm_p99",
    "rate_collinear",
    "rate_antipodal_angle_only",
    "rate_antipodal_drop",
    "rate_missing_link_steps",
    "rate_missing_top1_steps",
    "normalized_rate",
    "collapse_collinear_exceeded",
    "collapse_antipodal_drop_exceeded",
    "collapse_wedge_norm_p99_below_tau_wedge",
    "quantile_reference_only",
    "run_warning",
    "length_confound_warning",
    "confound_status",
    "rho_len_max_theta",
    "auc_len_tertile_short",
    "auc_len_tertile_medium",
    "auc_len_tertile_long",
    "n_len_tertile_short",
    "n_len_tertile_medium",
    "n_len_tertile_long",
    "exclusion_rate_short",
    "exclusion_rate_medium",
    "exclusion_rate_long",
    "degenerate_path_rate_top1",
    "trimmed_degenerate_path_rate",
    "rate_antipodal_angle_only_p50",
    "rate_antipodal_angle_only_p90",
    "share_samples_antipodal_angle_only_gt_0_50",
    "vec8_eff_dim_pr",
    "vec8_eff_dim_pr_status",
];

#[derive(Clone, Debug)]
pub struct Gate1WriterInput {
    pub run_eval_input: RunEvalInput,
    pub run_eval_result: RunEvalResult,
    pub sample_links: Vec<SampleLinkReport>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ArtifactPaths {
    pub manifest_json: PathBuf,
    pub summary_csv: PathBuf,
    pub link_topk_csv: PathBuf,
    pub link_sanity_md: PathBuf,
}

#[derive(Debug)]
pub enum WriteError {
    Io(std::io::Error),
    Json(serde_json::Error),
    Validation(ValidationError),
    InvalidFloat { field: &'static str, value: f64 },
    InvalidEvaluationMode(String),
}

impl fmt::Display for WriteError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "io error: {}", err),
            Self::Json(err) => write!(f, "json error: {}", err),
            Self::Validation(err) => write!(f, "manifest validation error: {}", err),
            Self::InvalidFloat { field, value } => {
                write!(f, "non-finite float for {}: {}", field, value)
            }
            Self::InvalidEvaluationMode(value) => write!(
                f,
                "invalid evaluation_mode_id '{}': expected supervised_v1 or unsupervised_v1",
                value
            ),
        }
    }
}

impl std::error::Error for WriteError {}

impl From<std::io::Error> for WriteError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

impl From<serde_json::Error> for WriteError {
    fn from(value: serde_json::Error) -> Self {
        Self::Json(value)
    }
}

#[derive(Clone, Debug)]
struct RunRates {
    rate_antipodal_angle_only: f64,
    rate_missing_link_steps: f64,
    rate_missing_top1_steps: f64,
    normalized_rate: f64,
}

#[derive(Clone, Debug)]
struct TrimmedDiagnosticsAggregate {
    trimmed_rbar_norm_pre_p50: Option<f64>,
    trimmed_rbar_norm_pre_p10: Option<f64>,
    trimmed_rbar_norm_pre_p01: Option<f64>,
    trimmed_failure_rate: Option<f64>,
}

#[derive(Clone, Debug)]
struct WriterContext {
    confounds: ConfoundOutputs,
    antipodal_warning: AntipodalWarningOutputs,
    run_rates: RunRates,
    trimmed_diag: TrimmedDiagnosticsAggregate,
}

pub fn write_gate1_artifacts<P: AsRef<Path>>(
    out_dir: P,
    input: &Gate1WriterInput,
) -> Result<ArtifactPaths, WriteError> {
    if input.evaluation_mode_id != "supervised_v1" && input.evaluation_mode_id != "unsupervised_v1"
    {
        return Err(WriteError::InvalidEvaluationMode(
            input.evaluation_mode_id.clone(),
        ));
    }

    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;
    let context = build_context(input);

    let manifest = build_manifest(input, &context)?;
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    validate_manifest_json(&manifest_bytes).map_err(WriteError::Validation)?;
    let manifest_path = out_dir.join("manifest.json");
    write_bytes_lf(&manifest_path, &manifest_bytes)?;

    let summary = build_summary_csv(input, &context)?;
    let summary_path = out_dir.join("summary.csv");
    write_string_lf(&summary_path, &summary)?;

    let link_topk = build_link_topk_csv(&input.sample_links)?;
    let link_topk_path = out_dir.join("link_topk.csv");
    write_string_lf(&link_topk_path, &link_topk)?;

    let link_sanity_md = build_link_sanity_md(&input.run_eval_input.link_sanity)?;
    let link_sanity_path = out_dir.join("link_sanity.md");
    write_string_lf(&link_sanity_path, &link_sanity_md)?;

    Ok(ArtifactPaths {
        manifest_json: manifest_path,
        summary_csv: summary_path,
        link_topk_csv: link_topk_path,
        link_sanity_md: link_sanity_path,
    })
}

fn build_context(input: &Gate1WriterInput) -> WriterContext {
    let confounds = compute_confounds(&input.run_eval_input.samples);
    let antipodal_warning = compute_antipodal_warning(&input.run_eval_input.samples);
    let run_rates = compute_run_rates(&input.run_eval_input.samples);
    let trimmed_diag = aggregate_trimmed_diagnostics(&input.run_eval_input.samples);
    WriterContext {
        confounds,
        antipodal_warning,
        run_rates,
        trimmed_diag,
    }
}

fn compute_run_rates(samples: &[crate::run_eval::RunEvalSample]) -> RunRates {
    let mut count_antipodal_angle_only = 0usize;
    let mut count_missing_link_steps = 0usize;
    let mut count_missing_top1_steps = 0usize;
    let mut steps_total = 0usize;
    let mut normalized_count = 0usize;
    let mut vec8_total = 0usize;

    for sample in samples {
        let counts = &sample.diagnostics.top1.counts;
        count_antipodal_angle_only += counts.count_antipodal_angle_only;
        count_missing_link_steps += counts.count_missing_link_steps;
        count_missing_top1_steps += counts.count_missing_top1_steps;
        steps_total += counts.steps_total;
        normalized_count += counts.normalized_count;
        vec8_total += counts.vec8_total;
    }

    RunRates {
        rate_antipodal_angle_only: ratio(count_antipodal_angle_only, steps_total),
        rate_missing_link_steps: ratio(count_missing_link_steps, steps_total),
        rate_missing_top1_steps: ratio(count_missing_top1_steps, steps_total),
        normalized_rate: ratio(normalized_count, vec8_total),
    }
}

fn aggregate_trimmed_diagnostics(
    samples: &[crate::run_eval::RunEvalSample],
) -> TrimmedDiagnosticsAggregate {
    let mut norms = Vec::new();
    let mut total_failure_steps = 0usize;
    let mut total_steps = 0usize;
    for sample in samples {
        norms.extend(
            sample
                .diagnostics
                .trimmed_rbar_norm_pre_values
                .iter()
                .copied()
                .filter(|value| value.is_finite()),
        );
        total_failure_steps += sample.diagnostics.trimmed_stability.trimmed_failure_steps;
        total_steps += sample.diagnostics.trimmed.counts.steps_total;
    }
    norms.sort_by(|left, right| left.total_cmp(right));
    let trimmed_rbar_norm_pre_p50 = if norms.is_empty() {
        None
    } else {
        Some(nearest_rank(&norms, 0.50))
    };
    let trimmed_rbar_norm_pre_p10 = if norms.is_empty() {
        None
    } else {
        Some(nearest_rank(&norms, 0.10))
    };
    let trimmed_rbar_norm_pre_p01 = if norms.is_empty() {
        None
    } else {
        Some(nearest_rank(&norms, 0.01))
    };
    let trimmed_failure_rate = if total_steps == 0 {
        None
    } else {
        Some(ratio(total_failure_steps, total_steps))
    };

    TrimmedDiagnosticsAggregate {
        trimmed_rbar_norm_pre_p50,
        trimmed_rbar_norm_pre_p10,
        trimmed_rbar_norm_pre_p01,
        trimmed_failure_rate,
    }
}

fn write_string_lf(path: &Path, content: &str) -> Result<(), WriteError> {
    let normalized = content.replace("\r\n", "\n").replace('\r', "\n");
    fs::write(path, normalized.as_bytes())?;
    Ok(())
}

fn write_bytes_lf(path: &Path, bytes: &[u8]) -> Result<(), WriteError> {
    let content = std::str::from_utf8(bytes).map_err(|err| {
        WriteError::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("invalid UTF-8 bytes: {}", err),
        ))
    })?;
    write_string_lf(path, content)
}

#[derive(Serialize)]
struct ManifestJson {
    spec_version: String,
    spec_hash_raw_blake3: String,
    spec_hash_raw_input_id: String,
    spec_hash_blake3: String,
    spec_hash_input_id: String,
    dataset_revision_id: String,
    dataset_hash_blake3: String,
    code_git_commit: String,
    build_target_triple: String,
    rustc_version: String,
    method_id: String,
    distance_id: String,
    theta_source_id: String,
    trimmed_best_id: String,
    rotor_construction_id: String,
    bivector_basis_id: String,
    antipodal_policy_id: String,
    top1_policy_id: String,
    unitization_id: String,
    rotor_encoder_id: String,
    rotor_encoder_preproc_id: String,
    vec8_postproc_id: String,
    evaluation_mode_id: String,
    auc_algorithm_id: String,
    rank_method_id: String,
    stats_id: String,
    label_missing_policy_id: String,
    quantile_id: String,
    quantile_track_scope: String,
    quantiles_missing_ok: bool,
    quantile_reference_only: bool,
    summary_schema_id: String,
    link_topk_schema_id: String,
    link_sanity_id: String,
    links_topk_canonicalization_id: String,
    link_sanity_rng_id: String,
    link_sanity_seed: u64,
    link_sanity_sampling_id: String,
    link_sanity_selected_sample_ids: Vec<u64>,
    link_sanity_unrelated_count: usize,
    link_sanity_pass: bool,
    h_norm: String,
    max_share: String,
    reason_enum_id: String,
    metric_missing_enum_id: String,
    float_format_id: String,
    determinism_scope: String,
    run_valid: bool,
    run_invalid_reason: Option<String>,
    collapse_invalid_reason: Option<String>,
    run_warning: Option<String>,
    length_confound_warning: bool,
    rate_antipodal_angle_only_p50: String,
    rate_antipodal_angle_only_p90: String,
    share_samples_antipodal_angle_only_gt_0_50: String,
    n_supervised_eligible: usize,
    n_supervised_used_primary: usize,
    n_supervised_excluded_primary: usize,
    primary_exclusion_rate: String,
    label_missing_rate: String,
    degenerate_path_rate_top1_numerator: usize,
    degenerate_path_rate_top1_denominator: usize,
    degenerate_path_rate_top1: String,
    trimmed_degenerate_path_rate_numerator: usize,
    trimmed_degenerate_path_rate_denominator: usize,
    trimmed_degenerate_path_rate: String,
    trimmed_rbar_norm_pre_p50: Option<String>,
    trimmed_rbar_norm_pre_p10: Option<String>,
    trimmed_rbar_norm_pre_p01: Option<String>,
    trimmed_failure_rate: Option<String>,
    vec8_eff_dim_pr: Option<String>,
    vec8_eff_dim_pr_status: String,
    confound_status: String,
    rho_len_max_theta: Option<String>,
    auc_len_tertile_short: Option<String>,
    auc_len_tertile_medium: Option<String>,
    auc_len_tertile_long: Option<String>,
    n_len_tertile_short: usize,
    n_len_tertile_medium: usize,
    n_len_tertile_long: usize,
    exclusion_rate_short: Option<String>,
    exclusion_rate_medium: Option<String>,
    exclusion_rate_long: Option<String>,
    tau_wedge: String,
    tau_antipodal_dot: String,
    eps_norm: String,
    eps_dist: String,
    max_antipodal_drop_rate: String,
    collapse_rate_collinear_threshold: String,
    collapse_rate_antipodal_drop_threshold: String,
    primary_exclusion_ceiling: String,
    min_rotors: usize,
    min_planes: usize,
    dot_p01: Option<String>,
    dot_p50: Option<String>,
    dot_p90: Option<String>,
    dot_p99: Option<String>,
    wedge_norm_p01: Option<String>,
    wedge_norm_p50: Option<String>,
    wedge_norm_p90: Option<String>,
    wedge_norm_p99: Option<String>,
    rate_collinear: String,
    rate_antipodal_angle_only: String,
    rate_antipodal_drop: String,
    rate_missing_link_steps: String,
    rate_missing_top1_steps: String,
    normalized_rate: String,
}

fn build_manifest(
    input: &Gate1WriterInput,
    context: &WriterContext,
) -> Result<ManifestJson, WriteError> {
    let top1_deg_rate = ratio(
        input
            .run_eval_result
            .degenerate_path_rate_counts
            .top1_numerator,
        input
            .run_eval_result
            .degenerate_path_rate_counts
            .top1_denominator,
    );
    let trimmed_deg_rate = ratio(
        input
            .run_eval_result
            .degenerate_path_rate_counts
            .trimmed_numerator,
        input
            .run_eval_result
            .degenerate_path_rate_counts
            .trimmed_denominator,
    );
    let top1_dot_quantiles = input.run_eval_result.top1_dot_quantiles.as_ref();
    let top1_wedge_quantiles = input.run_eval_result.top1_wedge_norm_quantiles.as_ref();
    let primary_exclusion_rate_value = if input.run_eval_result.n_supervised_eligible == 0 {
        0.0
    } else {
        input.run_eval_result.primary_exclusion_rate.unwrap_or(0.0)
    };

    Ok(ManifestJson {
        spec_version: SPEC_VERSION.to_string(),
        spec_hash_raw_blake3: input.spec_hash_raw_blake3.clone(),
        spec_hash_raw_input_id: SPEC_HASH_RAW_INPUT_ID.to_string(),
        spec_hash_blake3: input.spec_hash_blake3.clone(),
        spec_hash_input_id: SPEC_HASH_INPUT_ID.to_string(),
        dataset_revision_id: input.dataset_revision_id.clone(),
        dataset_hash_blake3: input.dataset_hash_blake3.clone(),
        code_git_commit: input.code_git_commit.clone(),
        build_target_triple: input.build_target_triple.clone(),
        rustc_version: input.rustc_version.clone(),
        method_id: METHOD_ID.to_string(),
        distance_id: DISTANCE_ID.to_string(),
        theta_source_id: THETA_SOURCE_ID.to_string(),
        trimmed_best_id: TRIMMED_BEST_ID.to_string(),
        rotor_construction_id: ROTOR_CONSTRUCTION_ID.to_string(),
        bivector_basis_id: BIVECTOR_BASIS_ID.to_string(),
        antipodal_policy_id: ANTIPODAL_POLICY_ID.to_string(),
        top1_policy_id: TOP1_POLICY_ID.to_string(),
        unitization_id: input.unitization_id.clone(),
        rotor_encoder_id: input.rotor_encoder_id.clone(),
        rotor_encoder_preproc_id: input.rotor_encoder_preproc_id.clone(),
        vec8_postproc_id: input.vec8_postproc_id.clone(),
        evaluation_mode_id: input.evaluation_mode_id.clone(),
        auc_algorithm_id: AUC_ALGORITHM_ID.to_string(),
        rank_method_id: RANK_METHOD_ID.to_string(),
        stats_id: STATS_ID.to_string(),
        label_missing_policy_id: LABEL_MISSING_POLICY_ID.to_string(),
        quantile_id: QUANTILE_ID.to_string(),
        quantile_track_scope: QUANTILE_TRACK_SCOPE.to_string(),
        quantiles_missing_ok: true,
        quantile_reference_only: input.run_eval_result.quantile_reference_only,
        summary_schema_id: SUMMARY_SCHEMA_ID.to_string(),
        link_topk_schema_id: LINK_TOPK_SCHEMA_ID.to_string(),
        link_sanity_id: LINK_SANITY_ID.to_string(),
        links_topk_canonicalization_id: LINKS_TOPK_CANONICALIZATION_ID.to_string(),
        link_sanity_rng_id: input.run_eval_input.link_sanity.rng_id.clone(),
        link_sanity_seed: input.run_eval_input.link_sanity.seed,
        link_sanity_sampling_id: input.run_eval_input.link_sanity.sampling_id.clone(),
        link_sanity_selected_sample_ids: input
            .run_eval_input
            .link_sanity
            .selected_sample_ids
            .clone(),
        link_sanity_unrelated_count: input.run_eval_input.link_sanity.unrelated_count,
        link_sanity_pass: !input.run_eval_input.link_sanity.link_sanity_fail,
        h_norm: fmt_f64("h_norm", input.run_eval_input.link_sanity.h_norm)?,
        max_share: fmt_f64("max_share", input.run_eval_input.link_sanity.max_share)?,
        reason_enum_id: REASON_ENUM_ID.to_string(),
        metric_missing_enum_id: METRIC_MISSING_ENUM_ID.to_string(),
        float_format_id: FLOAT_FORMAT_ID.to_string(),
        determinism_scope: DETERMINISM_SCOPE.to_string(),
        run_valid: input.run_eval_result.run_valid,
        run_invalid_reason: input
            .run_eval_result
            .run_invalid_reason
            .map(run_invalid_reason_str),
        collapse_invalid_reason: input
            .run_eval_result
            .collapse_invalid_reason
            .map(collapse_invalid_reason_str),
        run_warning: context.antipodal_warning.run_warning.clone(),
        length_confound_warning: context.confounds.length_confound_warning,
        rate_antipodal_angle_only_p50: fmt_f64(
            "rate_antipodal_angle_only_p50",
            context.antipodal_warning.rate_antipodal_angle_only_p50,
        )?,
        rate_antipodal_angle_only_p90: fmt_f64(
            "rate_antipodal_angle_only_p90",
            context.antipodal_warning.rate_antipodal_angle_only_p90,
        )?,
        share_samples_antipodal_angle_only_gt_0_50: fmt_f64(
            "share_samples_antipodal_angle_only_gt_0_50",
            context
                .antipodal_warning
                .share_samples_antipodal_angle_only_gt_0_50,
        )?,
        n_supervised_eligible: input.run_eval_result.n_supervised_eligible,
        n_supervised_used_primary: input.run_eval_result.n_supervised_used_primary,
        n_supervised_excluded_primary: input.run_eval_result.n_supervised_excluded_primary,
        primary_exclusion_rate: fmt_f64("primary_exclusion_rate", primary_exclusion_rate_value)?,
        label_missing_rate: fmt_f64(
            "label_missing_rate",
            input.run_eval_result.label_missing_rate,
        )?,
        degenerate_path_rate_top1_numerator: input
            .run_eval_result
            .degenerate_path_rate_counts
            .top1_numerator,
        degenerate_path_rate_top1_denominator: input
            .run_eval_result
            .degenerate_path_rate_counts
            .top1_denominator,
        degenerate_path_rate_top1: fmt_f64("degenerate_path_rate_top1", top1_deg_rate)?,
        trimmed_degenerate_path_rate_numerator: input
            .run_eval_result
            .degenerate_path_rate_counts
            .trimmed_numerator,
        trimmed_degenerate_path_rate_denominator: input
            .run_eval_result
            .degenerate_path_rate_counts
            .trimmed_denominator,
        trimmed_degenerate_path_rate: fmt_f64("trimmed_degenerate_path_rate", trimmed_deg_rate)?,
        trimmed_rbar_norm_pre_p50: fmt_opt_f64(
            "trimmed_rbar_norm_pre_p50",
            context.trimmed_diag.trimmed_rbar_norm_pre_p50,
        )?,
        trimmed_rbar_norm_pre_p10: fmt_opt_f64(
            "trimmed_rbar_norm_pre_p10",
            context.trimmed_diag.trimmed_rbar_norm_pre_p10,
        )?,
        trimmed_rbar_norm_pre_p01: fmt_opt_f64(
            "trimmed_rbar_norm_pre_p01",
            context.trimmed_diag.trimmed_rbar_norm_pre_p01,
        )?,
        trimmed_failure_rate: fmt_opt_f64(
            "trimmed_failure_rate",
            context.trimmed_diag.trimmed_failure_rate,
        )?,
        vec8_eff_dim_pr: None,
        vec8_eff_dim_pr_status: "unavailable".to_string(),
        confound_status: context.confounds.confound_status.as_str().to_string(),
        rho_len_max_theta: fmt_opt_f64("rho_len_max_theta", context.confounds.rho_len_max_theta)?,
        auc_len_tertile_short: fmt_opt_f64(
            "auc_len_tertile_short",
            context.confounds.auc_len_tertile_short,
        )?,
        auc_len_tertile_medium: fmt_opt_f64(
            "auc_len_tertile_medium",
            context.confounds.auc_len_tertile_medium,
        )?,
        auc_len_tertile_long: fmt_opt_f64(
            "auc_len_tertile_long",
            context.confounds.auc_len_tertile_long,
        )?,
        n_len_tertile_short: context.confounds.n_len_tertile_short,
        n_len_tertile_medium: context.confounds.n_len_tertile_medium,
        n_len_tertile_long: context.confounds.n_len_tertile_long,
        exclusion_rate_short: fmt_opt_f64(
            "exclusion_rate_short",
            context.confounds.exclusion_rate_short,
        )?,
        exclusion_rate_medium: fmt_opt_f64(
            "exclusion_rate_medium",
            context.confounds.exclusion_rate_medium,
        )?,
        exclusion_rate_long: fmt_opt_f64(
            "exclusion_rate_long",
            context.confounds.exclusion_rate_long,
        )?,
        tau_wedge: fmt_f64("tau_wedge", TAU_WEDGE)?,
        tau_antipodal_dot: fmt_f64(
            "tau_antipodal_dot",
            RotorConfig::default().tau_antipodal_dot,
        )?,
        eps_norm: fmt_f64("eps_norm", EPS_NORM)?,
        eps_dist: fmt_f64("eps_dist", EPS_DIST)?,
        max_antipodal_drop_rate: fmt_f64("max_antipodal_drop_rate", MAX_ANTIPODAL_DROP_RATE)?,
        collapse_rate_collinear_threshold: fmt_f64(
            "collapse_rate_collinear_threshold",
            COLLAPSE_RATE_COLLINEAR_THRESHOLD,
        )?,
        collapse_rate_antipodal_drop_threshold: fmt_f64(
            "collapse_rate_antipodal_drop_threshold",
            COLLAPSE_RATE_ANTIPODAL_DROP_THRESHOLD,
        )?,
        primary_exclusion_ceiling: fmt_f64("primary_exclusion_ceiling", PRIMARY_EXCLUSION_CEILING)?,
        min_rotors: MIN_ROTORS,
        min_planes: MIN_PLANES,
        dot_p01: fmt_opt_f64("dot_p01", top1_dot_quantiles.map(|q| q.p01))?,
        dot_p50: fmt_opt_f64("dot_p50", top1_dot_quantiles.map(|q| q.p50))?,
        dot_p90: fmt_opt_f64("dot_p90", top1_dot_quantiles.map(|q| q.p90))?,
        dot_p99: fmt_opt_f64("dot_p99", top1_dot_quantiles.map(|q| q.p99))?,
        wedge_norm_p01: fmt_opt_f64("wedge_norm_p01", top1_wedge_quantiles.map(|q| q.p01))?,
        wedge_norm_p50: fmt_opt_f64("wedge_norm_p50", top1_wedge_quantiles.map(|q| q.p50))?,
        wedge_norm_p90: fmt_opt_f64("wedge_norm_p90", top1_wedge_quantiles.map(|q| q.p90))?,
        wedge_norm_p99: fmt_opt_f64("wedge_norm_p99", top1_wedge_quantiles.map(|q| q.p99))?,
        rate_collinear: fmt_f64("rate_collinear", input.run_eval_result.rate_collinear)?,
        rate_antipodal_angle_only: fmt_f64(
            "rate_antipodal_angle_only",
            context.run_rates.rate_antipodal_angle_only,
        )?,
        rate_antipodal_drop: fmt_f64(
            "rate_antipodal_drop",
            input.run_eval_result.rate_antipodal_drop,
        )?,
        rate_missing_link_steps: fmt_f64(
            "rate_missing_link_steps",
            context.run_rates.rate_missing_link_steps,
        )?,
        rate_missing_top1_steps: fmt_f64(
            "rate_missing_top1_steps",
            context.run_rates.rate_missing_top1_steps,
        )?,
        normalized_rate: fmt_f64("normalized_rate", context.run_rates.normalized_rate)?,
    })
}

fn run_invalid_reason_str(value: RunInvalidReason) -> String {
    match value {
        RunInvalidReason::LinkSanityFail => "link_sanity_fail".to_string(),
        RunInvalidReason::RandomLikeLinkCollapse => "random_like_link_collapse".to_string(),
        RunInvalidReason::DominantLinkCollapse => "dominant_link_collapse".to_string(),
        RunInvalidReason::ExcessExclusionsPrimary => "excess_exclusions_primary".to_string(),
        RunInvalidReason::EmptyQuantilePopulationPrimary => {
            "empty_quantile_population_primary".to_string()
        }
        RunInvalidReason::NoSupervisedEligibleSamples => {
            "no_supervised_eligible_samples".to_string()
        }
    }
}

fn collapse_invalid_reason_str(value: CollapseInvalidReason) -> String {
    match value {
        CollapseInvalidReason::RateCollinearExceeded => "rate_collinear_exceeded".to_string(),
        CollapseInvalidReason::RateAntipodalDropExceeded => {
            "rate_antipodal_drop_exceeded".to_string()
        }
        CollapseInvalidReason::WedgeNormP99BelowTauWedge => {
            "wedge_norm_p99_below_tau_wedge".to_string()
        }
    }
}

fn auc_undefined_reason_str(value: AucUndefinedReason) -> String {
    match value {
        AucUndefinedReason::SingleClassAfterExclusions => {
            "single_class_after_exclusions".to_string()
        }
    }
}

fn bool_str(value: bool) -> String {
    if value {
        "true".to_string()
    } else {
        "false".to_string()
    }
}

fn fmt_f64(field: &'static str, value: f64) -> Result<String, WriteError> {
    if !value.is_finite() {
        return Err(WriteError::InvalidFloat { field, value });
    }
    Ok(format!("{:.17e}", value))
}

fn fmt_opt_f64(field: &'static str, value: Option<f64>) -> Result<Option<String>, WriteError> {
    match value {
        Some(v) => Ok(Some(fmt_f64(field, v)?)),
        None => Ok(None),
    }
}

fn csv_escape(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\n') || value.contains('\r') {
        let escaped = value.replace('"', "\"\"");
        format!("\"{}\"", escaped)
    } else {
        value.to_string()
    }
}

fn build_summary_csv(
    input: &Gate1WriterInput,
    context: &WriterContext,
) -> Result<String, WriteError> {
    let mut row = BTreeMap::new();
    let top1_dot = input.run_eval_result.top1_dot_quantiles.as_ref();
    let top1_wedge = input.run_eval_result.top1_wedge_norm_quantiles.as_ref();
    let top1_deg = ratio(
        input
            .run_eval_result
            .degenerate_path_rate_counts
            .top1_numerator,
        input
            .run_eval_result
            .degenerate_path_rate_counts
            .top1_denominator,
    );
    let trimmed_deg = ratio(
        input
            .run_eval_result
            .degenerate_path_rate_counts
            .trimmed_numerator,
        input
            .run_eval_result
            .degenerate_path_rate_counts
            .trimmed_denominator,
    );
    let primary_exclusion_rate_value = if input.run_eval_result.n_supervised_eligible == 0 {
        0.0
    } else {
        input.run_eval_result.primary_exclusion_rate.unwrap_or(0.0)
    };

    row.insert(
        "run_valid",
        if input.run_eval_result.run_valid {
            "true".to_string()
        } else {
            "false".to_string()
        },
    );
    row.insert(
        "run_invalid_reason",
        input
            .run_eval_result
            .run_invalid_reason
            .map(run_invalid_reason_str)
            .unwrap_or_default(),
    );
    row.insert(
        "collapse_invalid_reason",
        input
            .run_eval_result
            .collapse_invalid_reason
            .map(collapse_invalid_reason_str)
            .unwrap_or_default(),
    );
    row.insert(
        "primary_auc",
        fmt_opt_f64("primary_auc", input.run_eval_result.primary_auc)?.unwrap_or_default(),
    );
    row.insert(
        "primary_auc_n_pos",
        input.run_eval_result.primary_auc_n_pos.to_string(),
    );
    row.insert(
        "primary_auc_n_neg",
        input.run_eval_result.primary_auc_n_neg.to_string(),
    );
    row.insert(
        "auc_undefined_reason",
        input
            .run_eval_result
            .auc_undefined_reason
            .map(auc_undefined_reason_str)
            .unwrap_or_default(),
    );
    row.insert(
        "n_supervised_eligible",
        input.run_eval_result.n_supervised_eligible.to_string(),
    );
    row.insert(
        "n_supervised_used_primary",
        input.run_eval_result.n_supervised_used_primary.to_string(),
    );
    row.insert(
        "n_supervised_excluded_primary",
        input
            .run_eval_result
            .n_supervised_excluded_primary
            .to_string(),
    );
    row.insert(
        "primary_exclusion_rate",
        fmt_f64("primary_exclusion_rate", primary_exclusion_rate_value)?,
    );
    row.insert(
        "label_missing_rate",
        fmt_f64(
            "label_missing_rate",
            input.run_eval_result.label_missing_rate,
        )?,
    );
    row.insert(
        "link_sanity_pass",
        bool_str(!input.run_eval_input.link_sanity.link_sanity_fail),
    );
    row.insert(
        "link_sanity_unrelated_count",
        input.run_eval_input.link_sanity.unrelated_count.to_string(),
    );
    row.insert(
        "h_norm",
        fmt_f64("h_norm", input.run_eval_input.link_sanity.h_norm)?,
    );
    row.insert(
        "max_share",
        fmt_f64("max_share", input.run_eval_input.link_sanity.max_share)?,
    );
    row.insert(
        "dot_p01",
        fmt_opt_f64("dot_p01", top1_dot.map(|q| q.p01))?.unwrap_or_default(),
    );
    row.insert(
        "dot_p50",
        fmt_opt_f64("dot_p50", top1_dot.map(|q| q.p50))?.unwrap_or_default(),
    );
    row.insert(
        "dot_p90",
        fmt_opt_f64("dot_p90", top1_dot.map(|q| q.p90))?.unwrap_or_default(),
    );
    row.insert(
        "dot_p99",
        fmt_opt_f64("dot_p99", top1_dot.map(|q| q.p99))?.unwrap_or_default(),
    );
    row.insert(
        "wedge_norm_p01",
        fmt_opt_f64("wedge_norm_p01", top1_wedge.map(|q| q.p01))?.unwrap_or_default(),
    );
    row.insert(
        "wedge_norm_p50",
        fmt_opt_f64("wedge_norm_p50", top1_wedge.map(|q| q.p50))?.unwrap_or_default(),
    );
    row.insert(
        "wedge_norm_p90",
        fmt_opt_f64("wedge_norm_p90", top1_wedge.map(|q| q.p90))?.unwrap_or_default(),
    );
    row.insert(
        "wedge_norm_p99",
        fmt_opt_f64("wedge_norm_p99", top1_wedge.map(|q| q.p99))?.unwrap_or_default(),
    );
    row.insert(
        "rate_collinear",
        fmt_f64("rate_collinear", input.run_eval_result.rate_collinear)?,
    );
    row.insert(
        "rate_antipodal_angle_only",
        fmt_f64(
            "rate_antipodal_angle_only",
            context.run_rates.rate_antipodal_angle_only,
        )?,
    );
    row.insert(
        "rate_antipodal_drop",
        fmt_f64(
            "rate_antipodal_drop",
            input.run_eval_result.rate_antipodal_drop,
        )?,
    );
    row.insert(
        "rate_missing_link_steps",
        fmt_f64(
            "rate_missing_link_steps",
            context.run_rates.rate_missing_link_steps,
        )?,
    );
    row.insert(
        "rate_missing_top1_steps",
        fmt_f64(
            "rate_missing_top1_steps",
            context.run_rates.rate_missing_top1_steps,
        )?,
    );
    row.insert(
        "normalized_rate",
        fmt_f64("normalized_rate", context.run_rates.normalized_rate)?,
    );
    row.insert(
        "collapse_collinear_exceeded",
        bool_str(
            input
                .run_eval_result
                .collapse_gate_status
                .collinear_exceeds_threshold,
        ),
    );
    row.insert(
        "collapse_antipodal_drop_exceeded",
        bool_str(
            input
                .run_eval_result
                .collapse_gate_status
                .antipodal_drop_exceeds_threshold,
        ),
    );
    row.insert(
        "collapse_wedge_norm_p99_below_tau_wedge",
        bool_str(
            input
                .run_eval_result
                .collapse_gate_status
                .wedge_norm_p99_below_tau_wedge,
        ),
    );
    row.insert(
        "quantile_reference_only",
        bool_str(input.run_eval_result.quantile_reference_only),
    );
    row.insert(
        "run_warning",
        context
            .antipodal_warning
            .run_warning
            .clone()
            .unwrap_or_default(),
    );
    row.insert(
        "length_confound_warning",
        bool_str(context.confounds.length_confound_warning),
    );
    row.insert(
        "confound_status",
        context.confounds.confound_status.as_str().to_string(),
    );
    row.insert(
        "rho_len_max_theta",
        fmt_opt_f64("rho_len_max_theta", context.confounds.rho_len_max_theta)?.unwrap_or_default(),
    );
    row.insert(
        "auc_len_tertile_short",
        fmt_opt_f64(
            "auc_len_tertile_short",
            context.confounds.auc_len_tertile_short,
        )?
        .unwrap_or_default(),
    );
    row.insert(
        "auc_len_tertile_medium",
        fmt_opt_f64(
            "auc_len_tertile_medium",
            context.confounds.auc_len_tertile_medium,
        )?
        .unwrap_or_default(),
    );
    row.insert(
        "auc_len_tertile_long",
        fmt_opt_f64(
            "auc_len_tertile_long",
            context.confounds.auc_len_tertile_long,
        )?
        .unwrap_or_default(),
    );
    row.insert(
        "n_len_tertile_short",
        context.confounds.n_len_tertile_short.to_string(),
    );
    row.insert(
        "n_len_tertile_medium",
        context.confounds.n_len_tertile_medium.to_string(),
    );
    row.insert(
        "n_len_tertile_long",
        context.confounds.n_len_tertile_long.to_string(),
    );
    row.insert(
        "exclusion_rate_short",
        fmt_opt_f64(
            "exclusion_rate_short",
            context.confounds.exclusion_rate_short,
        )?
        .unwrap_or_default(),
    );
    row.insert(
        "exclusion_rate_medium",
        fmt_opt_f64(
            "exclusion_rate_medium",
            context.confounds.exclusion_rate_medium,
        )?
        .unwrap_or_default(),
    );
    row.insert(
        "exclusion_rate_long",
        fmt_opt_f64("exclusion_rate_long", context.confounds.exclusion_rate_long)?
            .unwrap_or_default(),
    );
    row.insert(
        "degenerate_path_rate_top1",
        fmt_f64("degenerate_path_rate_top1", top1_deg)?,
    );
    row.insert(
        "trimmed_degenerate_path_rate",
        fmt_f64("trimmed_degenerate_path_rate", trimmed_deg)?,
    );
    row.insert(
        "rate_antipodal_angle_only_p50",
        fmt_f64(
            "rate_antipodal_angle_only_p50",
            context.antipodal_warning.rate_antipodal_angle_only_p50,
        )?,
    );
    row.insert(
        "rate_antipodal_angle_only_p90",
        fmt_f64(
            "rate_antipodal_angle_only_p90",
            context.antipodal_warning.rate_antipodal_angle_only_p90,
        )?,
    );
    row.insert(
        "share_samples_antipodal_angle_only_gt_0_50",
        fmt_f64(
            "share_samples_antipodal_angle_only_gt_0_50",
            context
                .antipodal_warning
                .share_samples_antipodal_angle_only_gt_0_50,
        )?,
    );
    row.insert("vec8_eff_dim_pr", String::new());
    row.insert("vec8_eff_dim_pr_status", "unavailable".to_string());

    let mut out = String::new();
    out.push_str(&SUMMARY_CSV_COLUMNS_V1.join(","));
    out.push('\n');
    let values: Vec<String> = SUMMARY_CSV_COLUMNS_V1
        .iter()
        .map(|column| csv_escape(row.get(column).map(String::as_str).unwrap_or("")))
        .collect();
    out.push_str(&values.join(","));
    out.push('\n');
    Ok(out)
}

fn build_link_topk_csv(sample_links: &[SampleLinkReport]) -> Result<String, WriteError> {
    let mut out = String::new();
    out.push_str("sample_id,ans_unit_id,rank,doc_unit_id,indexer_score_optional\n");

    let mut rows = Vec::new();
    for report in sample_links {
        for (ans_idx, links) in report.canonicalized.links_by_answer.iter().enumerate() {
            for link in links {
                rows.push((
                    report.sample_id,
                    ans_idx as u32,
                    link.rank,
                    link.doc_unit_id,
                    String::new(),
                ));
            }
        }
    }

    rows.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then_with(|| left.1.cmp(&right.1))
            .then_with(|| left.2.cmp(&right.2))
            .then_with(|| left.3.cmp(&right.3))
    });
    for row in rows {
        out.push_str(&format!(
            "{},{},{},{},{}\n",
            row.0, row.1, row.2, row.3, row.4
        ));
    }
    Ok(out)
}

fn build_link_sanity_md(sanity: &LinkSanityResult) -> Result<String, WriteError> {
    let mut out = String::new();
    out.push_str("# Link Sanity (Gate1)\n\n");
    out.push_str(&format!("spec_version: {}\n", SPEC_VERSION));
    out.push_str("K: 16\n");
    out.push_str(&format!("rng_id: {}\n", sanity.rng_id));
    out.push_str(&format!("seed: {}\n", sanity.seed));
    out.push_str(&format!("sampling_id: {}\n", sanity.sampling_id));
    out.push_str(&format!(
        "selected_sample_ids: [{}]\n\n",
        sanity
            .selected_sample_ids
            .iter()
            .map(|id| id.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    ));

    out.push_str("| sample_id | ans_unit_id | doc_unit_id | category | counted_as_unrelated |\n");
    out.push_str("|---:|---:|---:|---|---|\n");
    for record in &sanity.records {
        let ans = record
            .representative_ans_unit_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "NO_LINK".to_string());
        let doc = record
            .selected_doc_unit_id
            .map(|value| value.to_string())
            .unwrap_or_else(|| "NO_LINK".to_string());
        let category = match record.category {
            crate::SanityCategory::DocUnitId(value) => format!("doc_unit_id:{}", value),
            crate::SanityCategory::NoLink => "NO_LINK".to_string(),
        };
        let counted = if matches!(record.judgment, crate::SanityJudgment::Unrelated)
            || matches!(record.category, crate::SanityCategory::NoLink)
        {
            "true"
        } else {
            "false"
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            record.sample_id, ans, doc, category, counted
        ));
    }

    out.push('\n');
    out.push_str(&format!("unrelated_count: {}\n", sanity.unrelated_count));
    out.push_str("rule: FAIL if unrelated > 6\n");
    out.push_str(&format!(
        "result: {}\n",
        if sanity.link_sanity_fail {
            "FAIL"
        } else {
            "PASS"
        }
    ));
    out.push_str(&format!("H_norm: {}\n", fmt_f64("h_norm", sanity.h_norm)?));
    out.push_str(&format!(
        "max_share: {}\n",
        fmt_f64("max_share", sanity.max_share)?
    ));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linking::{
        CanonicalLink, CanonicalizationCounters, CanonicalizedSampleLinks, LinkSanityRecord,
        LinkSanityResult, SampleLinkReport, SanityCategory, SanityJudgment, Top1Accounting,
        Top1Step, LINKS_TOPK_CANONICALIZATION_ID, TOP1_POLICY_ID,
    };
    use crate::rotor_diagnostics::{
        AlignmentMetric, DegeneratePathRateCounts, MetricField, PlaneTurnMetric, RateMetrics,
        RotorDiagnosticsResult, Top1GateStep, TrackCounts, TrackDiagnostics,
        TrimmedStabilityDiagnostics, WanderingMetric, DISTANCE_ID, METHOD_ID, THETA_SOURCE_ID,
        TRIMMED_BEST_ID,
    };
    use crate::run_eval::{compute_run_eval, RunEvalInput, RunEvalSample};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn metric(value: Option<f64>) -> MetricField<f64> {
        MetricField {
            value,
            metric_missing_reason: None,
        }
    }

    fn base_track(track_id: &str, max_theta: Option<f64>) -> TrackDiagnostics {
        TrackDiagnostics {
            track_id: track_id.to_string(),
            excluded_reason: None,
            max_theta: metric(max_theta),
            plane_turn: PlaneTurnMetric {
                mean: metric(Some(0.0)),
                max: metric(Some(0.0)),
                var: metric(Some(0.0)),
            },
            alignment: AlignmentMetric {
                mean: metric(Some(0.0)),
                var: metric(Some(0.0)),
            },
            wandering: WanderingMetric {
                ratio: metric(Some(1.0)),
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

    fn run_eval_sample(
        sample_id: u64,
        sample_label: Option<u8>,
        answer_length: Option<usize>,
        top1_score: f64,
        dot: f64,
        wedge_norm: f64,
    ) -> RunEvalSample {
        let top1 = base_track("top1", Some(top1_score));
        let trimmed = base_track("trimmed_best", Some(top1_score * 0.5));
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
                top1_gate_steps: vec![Top1GateStep {
                    ans_unit_id: 0,
                    doc_unit_id: 0,
                    dot,
                    wedge_norm: Some(wedge_norm),
                    is_collinear: false,
                    is_antipodal_angle_only: false,
                    is_antipodal_drop: false,
                }],
                trimmed_rbar_norm_pre_values: vec![0.8, 0.9],
                trimmed_stability: TrimmedStabilityDiagnostics {
                    trimmed_rbar_norm_pre_p50: Some(0.85),
                    trimmed_rbar_norm_pre_p10: Some(0.8),
                    trimmed_rbar_norm_pre_p01: Some(0.8),
                    trimmed_failure_rate: 0.0,
                    trimmed_failure_steps: 0,
                    trimmed_attempted_steps: 1,
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

    fn sample_link_report(
        sample_id: u64,
        links_by_answer: Vec<Vec<CanonicalLink>>,
    ) -> SampleLinkReport {
        let steps_total = links_by_answer.len();
        let top1_steps = (0..steps_total)
            .map(|ans_unit_id| Top1Step::MissingTop1 {
                ans_unit_id: ans_unit_id as u32,
            })
            .collect();
        SampleLinkReport {
            sample_id,
            canonicalized: CanonicalizedSampleLinks {
                sample_id,
                links_topk_canonicalization_id: LINKS_TOPK_CANONICALIZATION_ID.to_string(),
                ans_unit_count: steps_total,
                doc_unit_count: 8,
                links_by_answer,
                counters: CanonicalizationCounters::default(),
            },
            top1: Top1Accounting {
                top1_policy_id: TOP1_POLICY_ID.to_string(),
                steps_total,
                count_missing_link_steps: 0,
                count_missing_top1_steps: steps_total,
                missing_link_step_rate: 0.0,
                missing_top1_step_rate: 1.0,
                max_missing_link_step_rate: 0.20,
                missing_link_step_rate_exceeds_threshold: false,
                steps: top1_steps,
            },
        }
    }

    fn link_sanity_result() -> LinkSanityResult {
        LinkSanityResult {
            link_sanity_id: crate::linking::LINK_SANITY_ID.to_string(),
            rng_id: crate::linking::LINK_SANITY_RNG_ID.to_string(),
            seed: crate::linking::LINK_SANITY_SEED,
            sampling_id: crate::linking::LINK_SANITY_SAMPLING_ID.to_string(),
            selected_sample_ids: vec![1, 2],
            k_eff: 2,
            records: vec![
                LinkSanityRecord {
                    sample_id: 1,
                    representative_ans_unit_id: Some(0),
                    selected_doc_unit_id: Some(1),
                    category: SanityCategory::DocUnitId(1),
                    judgment: SanityJudgment::Unreviewed,
                },
                LinkSanityRecord {
                    sample_id: 2,
                    representative_ans_unit_id: None,
                    selected_doc_unit_id: None,
                    category: SanityCategory::NoLink,
                    judgment: SanityJudgment::Unreviewed,
                },
            ],
            unrelated_count: 1,
            link_sanity_fail: false,
            h_norm: 0.25,
            max_share: 0.50,
            random_like_link_collapse: false,
            dominant_link_collapse: false,
        }
    }

    fn writer_input_fixture() -> Gate1WriterInput {
        let run_eval_input = RunEvalInput {
            samples: vec![
                run_eval_sample(1, Some(1), Some(12), 0.8, 0.2, 0.3),
                run_eval_sample(2, Some(0), Some(24), 0.2, 0.4, 0.6),
            ],
            link_sanity: link_sanity_result(),
        };
        let run_eval_result = compute_run_eval(&run_eval_input).expect("run eval");
        let sample_links = vec![
            sample_link_report(
                2,
                vec![vec![
                    CanonicalLink {
                        doc_unit_id: 3,
                        rank: 2,
                    },
                    CanonicalLink {
                        doc_unit_id: 1,
                        rank: 1,
                    },
                ]],
            ),
            sample_link_report(
                1,
                vec![vec![
                    CanonicalLink {
                        doc_unit_id: 2,
                        rank: 2,
                    },
                    CanonicalLink {
                        doc_unit_id: 0,
                        rank: 1,
                    },
                ]],
            ),
        ];
        Gate1WriterInput {
            run_eval_input,
            run_eval_result,
            sample_links,
            spec_hash_raw_blake3: "spec-raw-hash".to_string(),
            spec_hash_blake3: "spec-lf-hash".to_string(),
            dataset_revision_id: "dataset-rev".to_string(),
            dataset_hash_blake3: "dataset-hash".to_string(),
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

    fn temp_dir(prefix: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        let mut path = std::env::temp_dir();
        path.push(format!(
            "pale-ale-{}-{}-{}",
            prefix,
            std::process::id(),
            nanos
        ));
        path
    }

    #[test]
    fn summary_csv_header_matches_constant_order() {
        let input = writer_input_fixture();
        let context = build_context(&input);
        let csv = build_summary_csv(&input, &context).expect("summary csv");
        let header = csv.lines().next().expect("header line");
        assert_eq!(header, SUMMARY_CSV_COLUMNS_V1.join(","));
    }

    #[test]
    fn link_topk_rows_are_sorted_deterministically() {
        let input = writer_input_fixture();
        let csv = build_link_topk_csv(&input.sample_links).expect("csv");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(
            lines[0],
            "sample_id,ans_unit_id,rank,doc_unit_id,indexer_score_optional"
        );
        assert_eq!(lines[1], "1,0,1,0,");
        assert_eq!(lines[2], "1,0,2,2,");
        assert_eq!(lines[3], "2,0,1,1,");
        assert_eq!(lines[4], "2,0,2,3,");
    }

    #[test]
    fn manifest_json_serialization_is_stable() {
        let input = writer_input_fixture();
        let context = build_context(&input);
        let manifest = build_manifest(&input, &context).expect("manifest");
        let s1 =
            String::from_utf8(serde_json::to_vec_pretty(&manifest).expect("json")).expect("utf8");
        let s2 =
            String::from_utf8(serde_json::to_vec_pretty(&manifest).expect("json")).expect("utf8");
        assert_eq!(s1, s2);
        assert!(
            s1.find("\"spec_version\"").expect("spec_version")
                < s1.find("\"spec_hash_raw_blake3\"")
                    .expect("spec_hash_raw_blake3")
        );
        assert!(
            s1.find("\"summary_schema_id\"").expect("summary_schema_id")
                < s1.find("\"link_topk_schema_id\"")
                    .expect("link_topk_schema_id")
        );
    }

    #[test]
    fn writing_artifacts_is_byte_deterministic() {
        let input = writer_input_fixture();
        let dir_a = temp_dir("writer-a");
        let dir_b = temp_dir("writer-b");
        fs::create_dir_all(&dir_a).expect("dir_a");
        fs::create_dir_all(&dir_b).expect("dir_b");

        let paths_a = write_gate1_artifacts(&dir_a, &input).expect("write a");
        let paths_b = write_gate1_artifacts(&dir_b, &input).expect("write b");

        let manifest_a = fs::read(paths_a.manifest_json).expect("manifest a");
        let manifest_b = fs::read(paths_b.manifest_json).expect("manifest b");
        assert_eq!(manifest_a, manifest_b);

        let summary_a = fs::read(paths_a.summary_csv).expect("summary a");
        let summary_b = fs::read(paths_b.summary_csv).expect("summary b");
        assert_eq!(summary_a, summary_b);

        let topk_a = fs::read(paths_a.link_topk_csv).expect("topk a");
        let topk_b = fs::read(paths_b.link_topk_csv).expect("topk b");
        assert_eq!(topk_a, topk_b);

        let sanity_a = fs::read(paths_a.link_sanity_md).expect("sanity a");
        let sanity_b = fs::read(paths_b.link_sanity_md).expect("sanity b");
        assert_eq!(sanity_a, sanity_b);

        let _ = fs::remove_dir_all(&dir_a);
        let _ = fs::remove_dir_all(&dir_b);
    }

    #[test]
    fn writer_rejects_invalid_evaluation_mode() {
        let mut input = writer_input_fixture();
        input.evaluation_mode_id = "invalid_mode".to_string();
        let dir = temp_dir("writer-invalid");
        fs::create_dir_all(&dir).expect("dir");
        let err = write_gate1_artifacts(&dir, &input).expect_err("invalid mode");
        assert!(matches!(err, WriteError::InvalidEvaluationMode(_)));
        let _ = fs::remove_dir_all(&dir);
    }
}
