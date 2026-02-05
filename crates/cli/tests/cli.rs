use assert_cmd::cargo::cargo_bin_cmd;
use predicates::str::contains;
use serde_json::Value;

#[test]
fn json_usage_error_when_missing_subcommand() {
    let output = cargo_bin_cmd!("pale-ale")
        .arg("--json")
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(1));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "CLI_USAGE");
}

#[test]
fn json_offline_model_download() {
    let output = cargo_bin_cmd!("pale-ale")
        .args(["model", "download", "--offline", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "OFFLINE_MODE");
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
fn json_eval_not_implemented() {
    let output = cargo_bin_cmd!("pale-ale")
        .args(["eval", "q", "c", "a", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "NOT_IMPLEMENTED");
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
fn json_model_verify_not_implemented_exits_two() {
    let output = cargo_bin_cmd!("pale-ale")
        .args(["model", "verify", "--json"])
        .output()
        .unwrap();

    assert_eq!(output.status.code(), Some(2));
    let stdout = String::from_utf8(output.stdout).unwrap();
    let value: Value = serde_json::from_str(&stdout).expect("stdout must be JSON");
    assert_eq!(value["error"]["code"], "NOT_IMPLEMENTED");
}
