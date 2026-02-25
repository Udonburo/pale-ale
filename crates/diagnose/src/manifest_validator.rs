use serde_json::Value;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    InvalidJson,
    MissingField(String),
    InvalidFieldValue {
        field: String,
        expected: String,
        actual: String,
    },
    InvalidFieldType {
        field: String,
        expected: String,
    },
    InvalidFloatFormat {
        field: String,
        value: String,
    },
    NonFiniteStringValue {
        field: String,
        value: String,
    },
    InvalidRunReasonState,
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson => write!(f, "invalid json"),
            Self::MissingField(field) => write!(f, "missing field: {}", field),
            Self::InvalidFieldValue {
                field,
                expected,
                actual,
            } => write!(
                f,
                "invalid field value for {}: expected {}, got {}",
                field, expected, actual
            ),
            Self::InvalidFieldType { field, expected } => {
                write!(f, "invalid field type for {}: expected {}", field, expected)
            }
            Self::InvalidFloatFormat { field, value } => {
                write!(f, "invalid sci_17e float for {}: {}", field, value)
            }
            Self::NonFiniteStringValue { field, value } => {
                write!(f, "non-finite token in string field {}: {}", field, value)
            }
            Self::InvalidRunReasonState => write!(
                f,
                "run_valid=false requires run_invalid_reason or collapse_invalid_reason"
            ),
        }
    }
}

impl std::error::Error for ValidationError {}

pub fn validate_manifest_json(bytes: &[u8]) -> Result<(), ValidationError> {
    let value: Value = serde_json::from_slice(bytes).map_err(|_| ValidationError::InvalidJson)?;
    let obj = value.as_object().ok_or(ValidationError::InvalidJson)?;

    for key in REQUIRED_KEYS {
        if !obj.contains_key(*key) {
            return Err(ValidationError::MissingField((*key).to_string()));
        }
    }

    check_fixed_string(obj, "summary_schema_id", "summary_csv_schema_v1")?;
    check_fixed_string(obj, "link_topk_schema_id", "link_topk_csv_schema_v1")?;
    check_fixed_string(obj, "float_format_id", "sci_17e_v1")?;
    check_fixed_string(obj, "quantile_id", "nearest_rank_total_cmp_v1")?;
    check_fixed_string(obj, "rank_method_id", "average_rank_total_cmp_v1")?;
    check_fixed_string(obj, "auc_algorithm_id", "mann_whitney_rank_sum_v1")?;

    check_bool_field(obj, "quantiles_missing_ok")?;
    check_bool_field(obj, "quantile_reference_only")?;
    check_bool_field(obj, "run_valid")?;

    for (field, value) in iter_string_values(String::new(), &value) {
        if value.contains("NaN") || value.contains("Inf") {
            return Err(ValidationError::NonFiniteStringValue { field, value });
        }
    }

    for key in REQUIRED_FLOAT_STRING_KEYS {
        check_float_string_key(obj, key)?;
    }
    for key in OPTIONAL_FLOAT_STRING_KEYS {
        if let Some(v) = obj.get(*key) {
            if !v.is_null() {
                check_float_string_key(obj, key)?;
            }
        }
    }

    if !get_bool(obj, "run_valid")? {
        let has_run_reason = !is_null_or_missing(obj.get("run_invalid_reason"));
        let has_collapse_reason = !is_null_or_missing(obj.get("collapse_invalid_reason"));
        if !has_run_reason && !has_collapse_reason {
            return Err(ValidationError::InvalidRunReasonState);
        }
    }

    let confound_status = get_string(obj, "confound_status")?;
    if confound_status == "ok" {
        for key in CONFOUND_REQUIRED_WHEN_OK_KEYS {
            if !obj.contains_key(*key) {
                return Err(ValidationError::MissingField((*key).to_string()));
            }
        }
    }

    Ok(())
}

fn check_fixed_string(
    obj: &serde_json::Map<String, Value>,
    key: &str,
    expected: &str,
) -> Result<(), ValidationError> {
    let actual = get_string(obj, key)?;
    if actual != expected {
        return Err(ValidationError::InvalidFieldValue {
            field: key.to_string(),
            expected: expected.to_string(),
            actual: actual.to_string(),
        });
    }
    Ok(())
}

fn check_bool_field(
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<(), ValidationError> {
    if obj.get(key).and_then(|v| v.as_bool()).is_none() {
        return Err(ValidationError::InvalidFieldType {
            field: key.to_string(),
            expected: "bool".to_string(),
        });
    }
    Ok(())
}

fn get_bool(obj: &serde_json::Map<String, Value>, key: &str) -> Result<bool, ValidationError> {
    obj.get(key)
        .and_then(|v| v.as_bool())
        .ok_or_else(|| ValidationError::InvalidFieldType {
            field: key.to_string(),
            expected: "bool".to_string(),
        })
}

fn get_string<'a>(
    obj: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a str, ValidationError> {
    obj.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| ValidationError::InvalidFieldType {
            field: key.to_string(),
            expected: "string".to_string(),
        })
}

fn is_null_or_missing(value: Option<&Value>) -> bool {
    value.map(|v| v.is_null()).unwrap_or(true)
}

fn check_float_string_key(
    obj: &serde_json::Map<String, Value>,
    key: &str,
) -> Result<(), ValidationError> {
    let value = get_string(obj, key)?;
    if !is_sci_17e(value) {
        return Err(ValidationError::InvalidFloatFormat {
            field: key.to_string(),
            value: value.to_string(),
        });
    }
    Ok(())
}

fn is_sci_17e(value: &str) -> bool {
    let Some(e_idx) = value.find('e') else {
        return false;
    };
    let (mantissa, exp) = value.split_at(e_idx);
    let exp = &exp[1..];
    if mantissa.is_empty() || exp.is_empty() {
        return false;
    }

    let mut chars = mantissa.chars();
    if mantissa.starts_with('-') {
        chars.next();
    }
    let remaining: String = chars.collect();
    let mut parts = remaining.split('.');
    let int_part = parts.next().unwrap_or("");
    let frac_part = parts.next().unwrap_or("");
    if parts.next().is_some() {
        return false;
    }
    if int_part.len() != 1 || !int_part.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }
    if frac_part.len() != 17 || !frac_part.chars().all(|c| c.is_ascii_digit()) {
        return false;
    }

    let exp_body = if exp.starts_with('-') || exp.starts_with('+') {
        &exp[1..]
    } else {
        exp
    };
    !exp_body.is_empty() && exp_body.chars().all(|c| c.is_ascii_digit())
}

fn iter_string_values(path: String, value: &Value) -> Vec<(String, String)> {
    let mut out = Vec::new();
    match value {
        Value::String(s) => {
            out.push((path, s.clone()));
        }
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

const REQUIRED_KEYS: &[&str] = &[
    "spec_version",
    "spec_hash_raw_blake3",
    "spec_hash_raw_input_id",
    "spec_hash_blake3",
    "spec_hash_input_id",
    "dataset_revision_id",
    "dataset_hash_blake3",
    "code_git_commit",
    "build_target_triple",
    "rustc_version",
    "method_id",
    "distance_id",
    "theta_source_id",
    "trimmed_best_id",
    "rotor_construction_id",
    "bivector_basis_id",
    "antipodal_policy_id",
    "top1_policy_id",
    "unitization_id",
    "rotor_encoder_id",
    "rotor_encoder_preproc_id",
    "vec8_postproc_id",
    "evaluation_mode_id",
    "auc_algorithm_id",
    "rank_method_id",
    "stats_id",
    "label_missing_policy_id",
    "quantile_id",
    "quantile_track_scope",
    "quantiles_missing_ok",
    "quantile_reference_only",
    "summary_schema_id",
    "link_topk_schema_id",
    "link_sanity_id",
    "links_topk_canonicalization_id",
    "link_sanity_rng_id",
    "link_sanity_seed",
    "link_sanity_sampling_id",
    "link_sanity_selected_sample_ids",
    "link_sanity_unrelated_count",
    "link_sanity_pass",
    "h_norm",
    "max_share",
    "reason_enum_id",
    "metric_missing_enum_id",
    "float_format_id",
    "determinism_scope",
    "run_valid",
    "run_invalid_reason",
    "collapse_invalid_reason",
    "run_warning",
    "length_confound_warning",
    "rate_antipodal_angle_only_p50",
    "rate_antipodal_angle_only_p90",
    "share_samples_antipodal_angle_only_gt_0_50",
    "n_supervised_eligible",
    "n_supervised_used_primary",
    "n_supervised_excluded_primary",
    "primary_exclusion_rate",
    "label_missing_rate",
    "degenerate_path_rate_top1_numerator",
    "degenerate_path_rate_top1_denominator",
    "degenerate_path_rate_top1",
    "trimmed_degenerate_path_rate_numerator",
    "trimmed_degenerate_path_rate_denominator",
    "trimmed_degenerate_path_rate",
    "vec8_eff_dim_pr_status",
    "confound_status",
    "n_len_tertile_short",
    "n_len_tertile_medium",
    "n_len_tertile_long",
    "tau_wedge",
    "tau_antipodal_dot",
    "eps_norm",
    "eps_dist",
    "max_antipodal_drop_rate",
    "collapse_rate_collinear_threshold",
    "collapse_rate_antipodal_drop_threshold",
    "primary_exclusion_ceiling",
    "min_rotors",
    "min_planes",
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
];

const REQUIRED_FLOAT_STRING_KEYS: &[&str] = &[
    "h_norm",
    "max_share",
    "rate_antipodal_angle_only_p50",
    "rate_antipodal_angle_only_p90",
    "share_samples_antipodal_angle_only_gt_0_50",
    "primary_exclusion_rate",
    "label_missing_rate",
    "degenerate_path_rate_top1",
    "trimmed_degenerate_path_rate",
    "tau_wedge",
    "tau_antipodal_dot",
    "eps_norm",
    "eps_dist",
    "max_antipodal_drop_rate",
    "collapse_rate_collinear_threshold",
    "collapse_rate_antipodal_drop_threshold",
    "primary_exclusion_ceiling",
    "rate_collinear",
    "rate_antipodal_angle_only",
    "rate_antipodal_drop",
    "rate_missing_link_steps",
    "rate_missing_top1_steps",
    "normalized_rate",
];

const OPTIONAL_FLOAT_STRING_KEYS: &[&str] = &[
    "trimmed_rbar_norm_pre_p50",
    "trimmed_rbar_norm_pre_p10",
    "trimmed_rbar_norm_pre_p01",
    "trimmed_failure_rate",
    "vec8_eff_dim_pr",
    "rho_len_max_theta",
    "auc_len_tertile_short",
    "auc_len_tertile_medium",
    "auc_len_tertile_long",
    "exclusion_rate_short",
    "exclusion_rate_medium",
    "exclusion_rate_long",
    "dot_p01",
    "dot_p50",
    "dot_p90",
    "dot_p99",
    "wedge_norm_p01",
    "wedge_norm_p50",
    "wedge_norm_p90",
    "wedge_norm_p99",
];

const CONFOUND_REQUIRED_WHEN_OK_KEYS: &[&str] = &[
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
];

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_manifest_value() -> Value {
        serde_json::from_str(
            r#"{
              "spec_version":"v4.0.0-ssot.9",
              "spec_hash_raw_blake3":"raw",
              "spec_hash_raw_input_id":"spec_text_raw_utf8_v1",
              "spec_hash_blake3":"lf",
              "spec_hash_input_id":"spec_text_utf8_lf_v1",
              "dataset_revision_id":"ds",
              "dataset_hash_blake3":"dsh",
              "code_git_commit":"abc",
              "build_target_triple":"x86_64-unknown-linux-gnu",
              "rustc_version":"rustc",
              "method_id":"rotor_diagnostics_v1",
              "distance_id":"proj_chordal_v1",
              "theta_source_id":"theta_uv_atan2_v1",
              "trimmed_best_id":"trimmed_best_v1(k=8,p=0.5,key=rank,tie=doc_unit_id)",
              "rotor_construction_id":"simple_rotor29_uv_v1",
              "bivector_basis_id":"lex_i_lt_j_v1",
              "antipodal_policy_id":"antipodal_split_v1(angle_only_for_theta,drop_on_nonfinite)",
              "top1_policy_id":"strict_rank1_or_missing_v1",
              "unitization_id":"sentence_split_v1",
              "rotor_encoder_id":"enc",
              "rotor_encoder_preproc_id":"pre",
              "vec8_postproc_id":"post",
              "evaluation_mode_id":"supervised_v1",
              "auc_algorithm_id":"mann_whitney_rank_sum_v1",
              "rank_method_id":"average_rank_total_cmp_v1",
              "stats_id":"none",
              "label_missing_policy_id":"exclude_sample_on_missing_halluc_unit_v1",
              "quantile_id":"nearest_rank_total_cmp_v1",
              "quantile_track_scope":"top1_only_primary_v1",
              "quantiles_missing_ok":true,
              "quantile_reference_only":false,
              "summary_schema_id":"summary_csv_schema_v1",
              "link_topk_schema_id":"link_topk_csv_schema_v1",
              "link_sanity_id":"sanity16_single_judgment_v1",
              "links_topk_canonicalization_id":"link_rank_doc_canon_v1",
              "link_sanity_rng_id":"splitmix64_v1",
              "link_sanity_seed":0,
              "link_sanity_sampling_id":"hash_sort_without_replacement_v1",
              "link_sanity_selected_sample_ids":[1,2],
              "link_sanity_unrelated_count":0,
              "link_sanity_pass":true,
              "h_norm":"0.00000000000000000e0",
              "max_share":"1.00000000000000000e0",
              "reason_enum_id":"reason_ids_v1",
              "metric_missing_enum_id":"metric_missing_reason_ids_v1",
              "float_format_id":"sci_17e_v1",
              "determinism_scope":"same_binary_same_target_same_dataset",
              "run_valid":true,
              "run_invalid_reason":null,
              "collapse_invalid_reason":null,
              "run_warning":null,
              "length_confound_warning":false,
              "rate_antipodal_angle_only_p50":"0.00000000000000000e0",
              "rate_antipodal_angle_only_p90":"0.00000000000000000e0",
              "share_samples_antipodal_angle_only_gt_0_50":"0.00000000000000000e0",
              "n_supervised_eligible":1,
              "n_supervised_used_primary":1,
              "n_supervised_excluded_primary":0,
              "primary_exclusion_rate":"0.00000000000000000e0",
              "label_missing_rate":"0.00000000000000000e0",
              "degenerate_path_rate_top1_numerator":0,
              "degenerate_path_rate_top1_denominator":1,
              "degenerate_path_rate_top1":"0.00000000000000000e0",
              "trimmed_degenerate_path_rate_numerator":0,
              "trimmed_degenerate_path_rate_denominator":1,
              "trimmed_degenerate_path_rate":"0.00000000000000000e0",
              "trimmed_rbar_norm_pre_p50":null,
              "trimmed_rbar_norm_pre_p10":null,
              "trimmed_rbar_norm_pre_p01":null,
              "trimmed_failure_rate":null,
              "vec8_eff_dim_pr":null,
              "vec8_eff_dim_pr_status":"unavailable",
              "confound_status":"unavailable",
              "rho_len_max_theta":null,
              "auc_len_tertile_short":null,
              "auc_len_tertile_medium":null,
              "auc_len_tertile_long":null,
              "n_len_tertile_short":0,
              "n_len_tertile_medium":0,
              "n_len_tertile_long":0,
              "exclusion_rate_short":null,
              "exclusion_rate_medium":null,
              "exclusion_rate_long":null,
              "tau_wedge":"1.00000000000000000e-6",
              "tau_antipodal_dot":"9.99999000000000027e-1",
              "eps_norm":"1.00000000000000000e-6",
              "eps_dist":"1.00000000000000006e-9",
              "max_antipodal_drop_rate":"2.00000000000000011e-1",
              "collapse_rate_collinear_threshold":"8.00000000000000044e-1",
              "collapse_rate_antipodal_drop_threshold":"2.00000000000000011e-1",
              "primary_exclusion_ceiling":"1.00000000000000006e-1",
              "min_rotors":3,
              "min_planes":2,
              "dot_p01":"0.00000000000000000e0",
              "dot_p50":"0.00000000000000000e0",
              "dot_p90":"0.00000000000000000e0",
              "dot_p99":"0.00000000000000000e0",
              "wedge_norm_p01":"0.00000000000000000e0",
              "wedge_norm_p50":"0.00000000000000000e0",
              "wedge_norm_p90":"0.00000000000000000e0",
              "wedge_norm_p99":"0.00000000000000000e0",
              "rate_collinear":"0.00000000000000000e0",
              "rate_antipodal_angle_only":"0.00000000000000000e0",
              "rate_antipodal_drop":"0.00000000000000000e0",
              "rate_missing_link_steps":"0.00000000000000000e0",
              "rate_missing_top1_steps":"0.00000000000000000e0",
              "normalized_rate":"0.00000000000000000e0"
            }"#,
        )
        .expect("valid json fixture")
    }

    #[test]
    fn validator_passes_on_valid_manifest() {
        let bytes = serde_json::to_vec(&valid_manifest_value()).expect("json");
        validate_manifest_json(&bytes).expect("valid manifest");
    }

    #[test]
    fn validator_fails_on_missing_required_field() {
        let mut value = valid_manifest_value();
        value.as_object_mut().expect("obj").remove("spec_version");
        let bytes = serde_json::to_vec(&value).expect("json");
        let err = validate_manifest_json(&bytes).expect_err("missing field");
        assert!(matches!(err, ValidationError::MissingField(_)));
    }

    #[test]
    fn validator_fails_on_invalid_float_format() {
        let mut value = valid_manifest_value();
        value
            .as_object_mut()
            .expect("obj")
            .insert("h_norm".to_string(), Value::String("0.123".to_string()));
        let bytes = serde_json::to_vec(&value).expect("json");
        let err = validate_manifest_json(&bytes).expect_err("invalid float format");
        assert!(matches!(err, ValidationError::InvalidFloatFormat { .. }));
    }

    #[test]
    fn validator_fails_on_schema_mismatch() {
        let mut value = valid_manifest_value();
        value.as_object_mut().expect("obj").insert(
            "summary_schema_id".to_string(),
            Value::String("wrong".to_string()),
        );
        let bytes = serde_json::to_vec(&value).expect("json");
        let err = validate_manifest_json(&bytes).expect_err("schema mismatch");
        assert!(matches!(err, ValidationError::InvalidFieldValue { .. }));
    }

    #[test]
    fn validator_fails_when_run_invalid_without_reasons() {
        let mut value = valid_manifest_value();
        value
            .as_object_mut()
            .expect("obj")
            .insert("run_valid".to_string(), Value::Bool(false));
        value
            .as_object_mut()
            .expect("obj")
            .insert("run_invalid_reason".to_string(), Value::Null);
        value
            .as_object_mut()
            .expect("obj")
            .insert("collapse_invalid_reason".to_string(), Value::Null);
        let bytes = serde_json::to_vec(&value).expect("json");
        let err = validate_manifest_json(&bytes).expect_err("invalid run reason state");
        assert_eq!(err, ValidationError::InvalidRunReasonState);
    }
}
