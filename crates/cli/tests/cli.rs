use assert_cmd::cargo::cargo_bin_cmd;
use predicates::str::contains;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn read_ndjson(path: &Path) -> Vec<Value> {
    let content = fs::read_to_string(path).expect("read ndjson");
    content
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("valid ndjson row"))
        .collect()
}

#[test]
fn json_usage_error_when_missing_subcommand() {
    let output = cargo_bin_cmd!("pale-ale").arg("--json").output().unwrap();

    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "CLI_USAGE");
}

#[test]
fn json_offline_model_download() {
    let temp = TempDir::new().unwrap();
    let output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["model", "download", "--offline", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "OFFLINE_FORBIDS_DOWNLOAD");
}

#[test]
fn json_doctor_success() {
    let output = cargo_bin_cmd!("pale-ale")
        .args(["doctor", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert!(value["error"].is_null());
    assert_eq!(value["status"], "OK");
}

#[test]
fn json_eval_usage_error() {
    let output = cargo_bin_cmd!("pale-ale")
        .args(["eval", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "CLI_USAGE");
    assert_eq!(
        value["audit_trace"]["hashes"]["inputs_hash"], "UNAVAILABLE",
        "inputs_hash must be UNAVAILABLE when eval inputs are incomplete"
    );
}

#[test]
fn non_json_usage_error() {
    let mut cmd = cargo_bin_cmd!("pale-ale");
    cmd.assert()
        .failure()
        .code(1)
        .stdout(predicates::str::is_empty())
        .stderr(contains("Usage"));
}

#[test]
fn help_exits_zero() {
    let output = cargo_bin_cmd!("pale-ale").arg("--help").output().unwrap();
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn version_exits_zero() {
    let output = cargo_bin_cmd!("pale-ale")
        .arg("--version")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));
}

#[test]
fn json_model_status_cache_missing() {
    let temp = TempDir::new().unwrap();
    let output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["model", "status", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["data"]["cache_present"], Value::Bool(false));
}

#[test]
fn json_model_verify_missing_cache() {
    let temp = TempDir::new().unwrap();
    let output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["model", "verify", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "MODEL_MISSING");
}

#[test]
fn json_model_verify_returns_file_details() {
    let temp = TempDir::new().unwrap();

    let path_output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["model", "path", "--json"])
        .output()
        .unwrap();
    assert_eq!(path_output.status.code(), Some(0));
    let path_stdout = String::from_utf8(path_output.stdout).unwrap();
    let path_json: Value = serde_json::from_str(&path_stdout).expect("path output JSON");
    let model_path = PathBuf::from(path_json["data"]["path"].as_str().unwrap());

    fs::create_dir_all(&model_path).unwrap();
    fs::write(model_path.join("model.safetensors"), b"bad-model").unwrap();
    fs::write(model_path.join("tokenizer.json"), b"bad-tokenizer").unwrap();

    let verify_output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["model", "verify", "--json"])
        .output()
        .unwrap();

    assert_eq!(verify_output.status.code(), Some(2));
    let verify_stdout = String::from_utf8(verify_output.stdout).unwrap();
    let verify_json: Value = serde_json::from_str(&verify_stdout).expect("verify output JSON");
    assert_eq!(verify_json["error"]["code"], "MODEL_FILE_MISSING");
    let details = verify_json["data"]["details"]
        .as_array()
        .expect("details array");
    assert!(!details.is_empty());
    assert!(details.iter().any(|item| item["state"] == "MISMATCH"));
    assert!(details.iter().any(|item| item["state"] == "MISSING"));
}

#[test]
fn json_model_print_hashes_missing_cache() {
    let temp = TempDir::new().unwrap();
    let output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["model", "print-hashes", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "MODEL_MISSING");
    assert!(value["data"]["warning"].is_string());
}

#[test]
fn json_model_clear_cache_requires_yes() {
    let temp = TempDir::new().unwrap();
    let output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["model", "clear-cache", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "CLI_USAGE");
}

#[test]
fn json_model_clear_cache_yes_deletes_model_dir() {
    let temp = TempDir::new().unwrap();

    let path_output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["model", "path", "--json"])
        .output()
        .unwrap();
    assert_eq!(path_output.status.code(), Some(0));
    let path_stdout = String::from_utf8(path_output.stdout).unwrap();
    let path_json: Value = serde_json::from_str(&path_stdout).expect("path output JSON");
    let model_path = PathBuf::from(path_json["data"]["path"].as_str().unwrap());

    fs::create_dir_all(&model_path).unwrap();
    fs::write(model_path.join("dummy.bin"), b"abc").unwrap();
    assert!(model_path.exists());

    let clear_output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["model", "clear-cache", "--yes", "--json"])
        .output()
        .unwrap();
    assert_eq!(clear_output.status.code(), Some(0));
    let clear_stdout = String::from_utf8(clear_output.stdout).unwrap();
    let clear_json: Value = serde_json::from_str(&clear_stdout).expect("clear output JSON");
    assert_eq!(clear_json["data"]["deleted"], Value::Bool(true));
    assert_eq!(
        clear_json["data"]["path"].as_str().unwrap(),
        model_path.to_string_lossy()
    );
    assert!(!model_path.exists());
}

#[test]
fn json_embed_offline_missing_cache() {
    let temp = TempDir::new().unwrap();
    let output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["embed", "hello", "--offline", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "MODEL_MISSING_OFFLINE");
}

#[test]
fn json_eval_offline_missing_cache() {
    let temp = TempDir::new().unwrap();
    let output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", temp.path())
        .args(["eval", "q", "c", "a", "--offline", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "MODEL_MISSING_OFFLINE");
    let inputs_hash = value["audit_trace"]["hashes"]["inputs_hash"]
        .as_str()
        .expect("inputs_hash string");
    assert!(is_lower_hex_64(inputs_hash), "inputs_hash format invalid");
}

#[test]
fn json_eval_success_has_verdict_and_evidence() {
    let verify_output = cargo_bin_cmd!("pale-ale")
        .args(["model", "verify", "--json"])
        .output()
        .unwrap();
    if verify_output.status.code() != Some(0) {
        return;
    }

    let output = cargo_bin_cmd!("pale-ale")
        .args([
            "eval",
            "query",
            "context sentence.",
            "answer sentence.",
            "--offline",
            "--json",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert!(value["error"].is_null());

    let status = value["status"].as_str().expect("status string");
    assert!(matches!(status, "LUCID" | "HAZY" | "DELIRIUM"));
    assert!(
        value["data"].get("status").is_none(),
        "data.status should not exist"
    );
    assert!(value["data"]["evidence"].is_array());
}

#[test]
fn json_eval_inputs_hash_present() {
    let verify_output = cargo_bin_cmd!("pale-ale")
        .args(["model", "verify", "--json"])
        .output()
        .unwrap();
    if verify_output.status.code() != Some(0) {
        return;
    }

    let output = cargo_bin_cmd!("pale-ale")
        .args(["eval", "q", "c.", "a.", "--offline", "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    let inputs_hash = value["audit_trace"]["hashes"]["inputs_hash"]
        .as_str()
        .expect("inputs_hash string");
    assert!(is_lower_hex_64(inputs_hash), "inputs_hash format invalid");
}

#[test]
fn json_eval_emits_truncation_warning_for_long_input() {
    let verify_output = cargo_bin_cmd!("pale-ale")
        .args(["model", "verify", "--json"])
        .output()
        .unwrap();
    if verify_output.status.code() != Some(0) {
        return;
    }

    let long_context = vec!["token"; 1800].join(" ");
    let output = cargo_bin_cmd!("pale-ale")
        .args(["eval", "q", &long_context, "a", "--offline", "--json"])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(0));

    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    let warnings = value["audit_trace"]["warnings"]
        .as_array()
        .expect("warnings array");
    assert!(
        warnings.iter().any(|w| w["type"] == "EMBED_TRUNCATED"),
        "expected EMBED_TRUNCATED warning in audit_trace.warnings"
    );
}

#[test]
fn json_embed_vector_invariants() {
    let output = cargo_bin_cmd!("pale-ale")
        .args(["embed", "Hello world", "--json"])
        .output()
        .unwrap();

    assert_eq!(
        output.status.code(),
        Some(0),
        "embed failed with stderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");

    assert_eq!(value["status"], "OK");
    assert_eq!(value["data"]["dim"], 384);

    let vector = value["data"]["vector"]
        .as_array()
        .expect("vector array missing");
    assert_eq!(vector.len(), 384, "expected 384-d embedding");

    let mut sq_sum = 0.0_f64;
    for (idx, item) in vector.iter().enumerate() {
        let v = item
            .as_f64()
            .unwrap_or_else(|| panic!("non-float embedding value at index {}", idx));
        assert!(v.is_finite(), "non-finite embedding value at index {}", idx);
        sq_sum += v * v;
    }
    let norm = sq_sum.sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-3,
        "embedding norm out of bounds: {}",
        norm
    );
}

#[test]
fn json_batch_offline_missing_cache() {
    let temp = TempDir::new().unwrap();
    let model_dir = temp.path().join("empty-model-cache");
    let input_path = temp.path().join("input.jsonl");
    let out_path = temp.path().join("out.ndjson");

    fs::write(&input_path, r#"{"query":"q","context":"c","answer":"a"}"#).unwrap();

    let output = cargo_bin_cmd!("pale-ale")
        .env("PA_MEASURE_MODEL_DIR", &model_dir)
        .args([
            "batch",
            input_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
            "--offline",
            "--json",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "MODEL_MISSING_OFFLINE");
}

#[test]
fn json_batch_dry_run_non_strict_row_errors_and_ordering() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("input.jsonl");
    let out_path = temp.path().join("out.ndjson");

    fs::write(
        &input_path,
        concat!(
            "{\"id\":\"bad\",\"query\":\"q1\",\"context\":\"c1\"}\n",
            "{\"id\":\"ok\",\"query\":\"q2\",\"context\":\"c2\",\"answer\":\"a2\"}\n"
        ),
    )
    .unwrap();

    let output = cargo_bin_cmd!("pale-ale")
        .args([
            "batch",
            input_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
            "--dry-run",
            "--threads",
            "4",
            "--json",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["status"], "OK");
    assert_eq!(value["data"]["rows_total"], 2);
    assert_eq!(value["data"]["rows_ok"], 1);
    assert_eq!(value["data"]["rows_err"], 1);

    let rows = read_ndjson(&out_path);
    assert_eq!(rows.len(), 2);

    assert_eq!(rows[0]["row_index"], 0);
    assert_eq!(rows[0]["status"], "UNKNOWN");
    assert_eq!(rows[0]["error"]["code"], "BATCH_INPUT_MISSING_FIELDS");

    assert_eq!(rows[1]["row_index"], 1);
    assert!(rows[1]["error"].is_null());
    assert_eq!(rows[1]["status"], "UNKNOWN");
    let hash = rows[1]["inputs_hash"].as_str().expect("inputs_hash");
    assert!(is_lower_hex_64(hash));
}

#[test]
fn json_batch_dry_run_hash_determinism_and_stable_output() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("input.jsonl");
    let out_path_a = temp.path().join("out-a.ndjson");
    let out_path_b = temp.path().join("out-b.ndjson");

    fs::write(
        &input_path,
        concat!(
            "{\"id\":\"r0\",\"query\":\"q0\",\"context\":\"c0\",\"answer\":\"a0\"}\n",
            "{\"id\":\"r1\",\"query\":\"q1\",\"context\":\"c1\"}\n",
            "{\"id\":\"r2\",\"query\":\"q2\",\"context\":\"c2\",\"answer\":\"a2\"}\n",
            "{\"id\":\"r3\",\"query\":\"q3\",\"context\":\"c3\",\"answer\":\"a3\"}\n"
        ),
    )
    .unwrap();

    let first = cargo_bin_cmd!("pale-ale")
        .args([
            "batch",
            input_path.to_str().unwrap(),
            "--out",
            out_path_a.to_str().unwrap(),
            "--dry-run",
            "--threads",
            "4",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(first.status.code(), Some(0));

    let second = cargo_bin_cmd!("pale-ale")
        .args([
            "batch",
            input_path.to_str().unwrap(),
            "--out",
            out_path_b.to_str().unwrap(),
            "--dry-run",
            "--threads",
            "4",
            "--json",
        ])
        .output()
        .unwrap();
    assert_eq!(second.status.code(), Some(0));

    let out_a = fs::read_to_string(&out_path_a).unwrap();
    let out_b = fs::read_to_string(&out_path_b).unwrap();
    assert_eq!(out_a, out_b, "dry-run NDJSON output must be deterministic");

    let rows = read_ndjson(&out_path_a);
    for (idx, row) in rows.iter().enumerate() {
        assert_eq!(row["row_index"], idx as u64);
        let hash = row["inputs_hash"].as_str().expect("inputs_hash");
        assert!(is_lower_hex_64(hash));
    }
}

#[test]
fn json_batch_dry_run_strict_failure_exit_code_2() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("input.jsonl");
    let out_path = temp.path().join("out.ndjson");

    fs::write(
        &input_path,
        concat!(
            "{\"id\":\"bad\",\"query\":\"q1\",\"context\":\"c1\"}\n",
            "{\"id\":\"ok\",\"query\":\"q2\",\"context\":\"c2\",\"answer\":\"a2\"}\n"
        ),
    )
    .unwrap();

    let output = cargo_bin_cmd!("pale-ale")
        .args([
            "batch",
            input_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
            "--dry-run",
            "--strict",
            "--json",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "BATCH_STRICT_FAILURE");
    assert_eq!(value["data"]["rows_total"], 2);
    assert_eq!(value["data"]["rows_err"], 1);

    let rows = read_ndjson(&out_path);
    assert_eq!(rows.len(), 2);
}

#[test]
fn json_batch_dry_run_accepts_bom_and_skips_blank_lines() {
    let temp = TempDir::new().unwrap();
    let input_path = temp.path().join("input.jsonl");
    let out_path = temp.path().join("out.ndjson");

    fs::write(
        &input_path,
        concat!(
            "\u{FEFF}{\"id\":\"r0\",\"query\":\"q0\",\"context\":\"c0\",\"answer\":\"a0\"}\n",
            "\n",
            "   \n",
            "{\"id\":\"r1\",\"query\":\"q1\",\"context\":\"c1\",\"answer\":\"a1\"}\n"
        ),
    )
    .unwrap();

    let output = cargo_bin_cmd!("pale-ale")
        .args([
            "batch",
            input_path.to_str().unwrap(),
            "--out",
            out_path.to_str().unwrap(),
            "--dry-run",
            "--json",
        ])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(0));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["data"]["rows_total"], 2);
    assert_eq!(value["data"]["rows_ok"], 2);
    assert_eq!(value["data"]["rows_err"], 0);

    let rows = read_ndjson(&out_path);
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0]["row_index"], 0);
    assert_eq!(rows[1]["row_index"], 1);
    assert!(rows[0]["error"].is_null());
    assert!(rows[1]["error"].is_null());
}
