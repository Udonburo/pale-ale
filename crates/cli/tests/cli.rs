use assert_cmd::cargo::cargo_bin_cmd;
use predicates::str::contains;
use serde_json::Value;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

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
