use blake3::Hasher;
use directories::ProjectDirs;
use pale_ale_modelspec::{ModelFileSpec, ModelSpec};
use reqwest::header::ETAG;
use serde::{Deserialize, Serialize};
use std::env;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StatusReport {
    pub cache_dir: PathBuf,
    pub cache_present: bool,
    pub missing_files: Vec<String>,
    pub note: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyReport {
    pub cache_dir: PathBuf,
    pub files: Vec<VerifyFile>,
    pub details: Vec<VerifyDetail>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyFile {
    pub path: String,
    pub expected_blake3: String,
    pub actual_blake3: String,
    pub size_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VerifyDetail {
    pub path: String,
    pub expected_blake3: String,
    pub actual_blake3: Option<String>,
    pub size_bytes: Option<u64>,
    pub state: VerifyState,
    pub final_url: Option<String>,
    pub etag: Option<String>,
    pub content_length: Option<u64>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerifyState {
    Match,
    Missing,
    Mismatch,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PrintHashesReport {
    pub model_id: String,
    pub revision: String,
    pub details: Vec<VerifyDetail>,
    pub rust_constants: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, Default)]
struct FileProvenance {
    pub final_url: Option<String>,
    pub etag: Option<String>,
    pub content_length: Option<u64>,
}

#[derive(Clone, Debug)]
pub enum EmbedError {
    OfflineForbidden,
    ModelMissing {
        dir: PathBuf,
    },
    ModelFileMissing {
        path: PathBuf,
        details: Vec<VerifyDetail>,
    },
    HashMismatch {
        path: PathBuf,
        expected: String,
        actual: String,
        details: Vec<VerifyDetail>,
    },
    InvalidHashFormat {
        path: String,
        hash: String,
    },
    InvalidPayloadHtml {
        path: PathBuf,
        snippet_len: usize,
        final_url: Option<String>,
    },
    InvalidPayloadJson {
        path: PathBuf,
        message: String,
        final_url: Option<String>,
    },
    DownloadFailed {
        url: String,
        message: String,
        final_url: Option<String>,
        etag: Option<String>,
        content_length: Option<u64>,
    },
    InvalidUrl {
        url: String,
    },
    ConfigLoad {
        path: PathBuf,
        message: String,
    },
    ModelLoad {
        path: PathBuf,
        message: String,
    },
    TokenizerLoad {
        path: PathBuf,
        message: String,
    },
    Tokenization {
        message: String,
    },
    Tensor {
        message: String,
    },
    Inference {
        message: String,
    },
    Io {
        path: Option<PathBuf>,
        message: String,
    },
}

impl EmbedError {
    fn io(path: Option<PathBuf>, error: io::Error) -> Self {
        Self::Io {
            path,
            message: error.to_string(),
        }
    }
}

pub struct ModelManager {
    spec: ModelSpec,
    base_dir: PathBuf,
}

impl ModelManager {
    pub fn new(spec: ModelSpec) -> Self {
        let base_dir = resolve_base_dir();
        Self { spec, base_dir }
    }

    pub fn new_with_base_dir(spec: ModelSpec, base_dir: PathBuf) -> Self {
        Self { spec, base_dir }
    }

    pub fn spec(&self) -> &ModelSpec {
        &self.spec
    }

    pub fn resolved_dir(&self) -> PathBuf {
        let mut path = self.base_dir.clone();
        for part in self.spec.model_id.split('/') {
            path.push(part);
        }
        path.push(&self.spec.revision);
        path
    }

    pub fn status(&self) -> StatusReport {
        let dir = self.resolved_dir();
        let manifest_valid = validate_manifest_hashes(&self.spec).is_ok();

        if !dir.exists() {
            return StatusReport {
                cache_dir: dir,
                cache_present: false,
                missing_files: Vec::new(),
                note: if manifest_valid {
                    "cache not found".to_string()
                } else {
                    "cache not found; manifest hash format invalid".to_string()
                },
            };
        }

        let mut missing = Vec::new();
        for file in &self.spec.required_files {
            let path = dir.join(&file.path);
            if !path.exists() {
                missing.push(file.path.clone());
            }
        }

        StatusReport {
            cache_dir: dir,
            cache_present: true,
            missing_files: missing,
            note: if manifest_valid {
                "cache present".to_string()
            } else {
                "cache present; manifest hash format invalid".to_string()
            },
        }
    }

    pub fn download(&self, offline: bool) -> Result<VerifyReport, EmbedError> {
        if offline {
            return Err(EmbedError::OfflineForbidden);
        }
        validate_manifest_hashes(&self.spec)?;

        let dir = self.resolved_dir();
        fs::create_dir_all(&dir).map_err(|err| EmbedError::io(Some(dir.clone()), err))?;
        let client = reqwest::blocking::Client::builder()
            .user_agent("pale-ale/1.0.1")
            .build()
            .map_err(|err| EmbedError::DownloadFailed {
                url: "client".to_string(),
                message: err.to_string(),
                final_url: None,
                etag: None,
                content_length: None,
            })?;

        for file in &self.spec.required_files {
            let url = self.spec.download_url(file);
            ensure_https(&url)?;
            let dest = dir.join(&file.path);
            download_file(&client, &url, &dest, file)?;
        }

        self.verify()
    }

    pub fn verify(&self) -> Result<VerifyReport, EmbedError> {
        validate_manifest_hashes(&self.spec)?;

        let dir = self.resolved_dir();
        if !dir.exists() {
            return Err(EmbedError::ModelMissing { dir });
        }

        let mut files = Vec::new();
        let mut details = Vec::new();

        for file in &self.spec.required_files {
            let path = dir.join(&file.path);
            if !path.exists() {
                details.push(VerifyDetail {
                    path: file.path.clone(),
                    expected_blake3: file.blake3.clone(),
                    actual_blake3: None,
                    size_bytes: None,
                    state: VerifyState::Missing,
                    final_url: None,
                    etag: None,
                    content_length: None,
                });
                continue;
            }

            let (actual, size) = file_blake3(&path)?;
            let provenance = read_provenance(&path);
            if actual == file.blake3 {
                files.push(VerifyFile {
                    path: file.path.clone(),
                    expected_blake3: file.blake3.clone(),
                    actual_blake3: actual.clone(),
                    size_bytes: size,
                });
                details.push(VerifyDetail {
                    path: file.path.clone(),
                    expected_blake3: file.blake3.clone(),
                    actual_blake3: Some(actual),
                    size_bytes: Some(size),
                    state: VerifyState::Match,
                    final_url: provenance.final_url,
                    etag: provenance.etag,
                    content_length: provenance.content_length,
                });
            } else {
                details.push(VerifyDetail {
                    path: file.path.clone(),
                    expected_blake3: file.blake3.clone(),
                    actual_blake3: Some(actual),
                    size_bytes: Some(size),
                    state: VerifyState::Mismatch,
                    final_url: provenance.final_url,
                    etag: provenance.etag,
                    content_length: provenance.content_length,
                });
            }
        }

        if let Some(first_missing) = details.iter().find(|d| d.state == VerifyState::Missing) {
            return Err(EmbedError::ModelFileMissing {
                path: dir.join(&first_missing.path),
                details,
            });
        }

        if let Some(first_mismatch) = details.iter().find(|d| d.state == VerifyState::Mismatch) {
            return Err(EmbedError::HashMismatch {
                path: dir.join(&first_mismatch.path),
                expected: first_mismatch.expected_blake3.clone(),
                actual: first_mismatch.actual_blake3.clone().unwrap_or_default(),
                details,
            });
        }

        Ok(VerifyReport {
            cache_dir: dir,
            files,
            details,
        })
    }

    pub fn print_hashes(&self) -> Result<PrintHashesReport, EmbedError> {
        validate_manifest_hashes(&self.spec)?;
        let status = self.status();
        if !status.cache_present || !status.missing_files.is_empty() {
            return Err(EmbedError::ModelMissing {
                dir: status.cache_dir,
            });
        }

        let dir = self.resolved_dir();
        let mut details = Vec::new();
        for file in &self.spec.required_files {
            let path = dir.join(&file.path);
            let (actual, size) = file_blake3(&path)?;
            let provenance = read_provenance(&path);
            let state = if actual == file.blake3 {
                VerifyState::Match
            } else {
                VerifyState::Mismatch
            };
            details.push(VerifyDetail {
                path: file.path.clone(),
                expected_blake3: file.blake3.clone(),
                actual_blake3: Some(actual),
                size_bytes: Some(size),
                state,
                final_url: provenance.final_url,
                etag: provenance.etag,
                content_length: provenance.content_length,
            });
        }

        Ok(PrintHashesReport {
            model_id: self.spec.model_id.clone(),
            revision: self.spec.revision.clone(),
            rust_constants: render_rust_constants(&self.spec.revision, &details),
            details,
        })
    }

    pub fn clear_cache(&self) -> Result<bool, EmbedError> {
        let dir = self.resolved_dir();
        if !dir.exists() {
            return Ok(false);
        }
        fs::remove_dir_all(&dir).map_err(|err| EmbedError::io(Some(dir.clone()), err))?;
        Ok(true)
    }
}

fn resolve_base_dir() -> PathBuf {
    if let Ok(path) = env::var("PA_MEASURE_MODEL_DIR") {
        return PathBuf::from(path);
    }
    if let Ok(path) = env::var("PALE_ALE_MODEL_DIR") {
        return PathBuf::from(path);
    }

    if let Some(project) = ProjectDirs::from("dev", "pale-ale", "pale-ale") {
        return project.cache_dir().join("models");
    }

    PathBuf::from("models")
}

fn ensure_https(url: &str) -> Result<(), EmbedError> {
    let parsed = reqwest::Url::parse(url).map_err(|_| EmbedError::InvalidUrl {
        url: url.to_string(),
    })?;
    if parsed.scheme() != "https" {
        return Err(EmbedError::InvalidUrl {
            url: url.to_string(),
        });
    }
    Ok(())
}

fn validate_manifest_hashes(spec: &ModelSpec) -> Result<(), EmbedError> {
    for file in &spec.required_files {
        if !is_lower_hex_64(&file.blake3) {
            return Err(EmbedError::InvalidHashFormat {
                path: file.path.clone(),
                hash: file.blake3.clone(),
            });
        }
    }
    Ok(())
}

fn is_lower_hex_64(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn download_file(
    client: &reqwest::blocking::Client,
    url: &str,
    dest: &Path,
    file_spec: &ModelFileSpec,
) -> Result<(), EmbedError> {
    let mut response = client
        .get(url)
        .send()
        .map_err(|err| EmbedError::DownloadFailed {
            url: url.to_string(),
            message: err.to_string(),
            final_url: None,
            etag: None,
            content_length: None,
        })?;

    let final_url = Some(response.url().to_string());
    let etag = response
        .headers()
        .get(ETAG)
        .and_then(|v| v.to_str().ok())
        .map(ToString::to_string);
    let content_length = response.content_length();

    if !response.status().is_success() {
        return Err(EmbedError::DownloadFailed {
            url: url.to_string(),
            message: format!("http status {}", response.status()),
            final_url,
            etag,
            content_length,
        });
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .map_err(|err| EmbedError::io(Some(parent.to_path_buf()), err))?;
    }

    let tmp_path = temp_path(dest);
    let write_result = (|| -> Result<(), EmbedError> {
        let mut file =
            File::create(&tmp_path).map_err(|err| EmbedError::io(Some(tmp_path.clone()), err))?;
        io::copy(&mut response, &mut file)
            .map_err(|err| EmbedError::io(Some(tmp_path.clone()), err))?;
        file.flush()
            .map_err(|err| EmbedError::io(Some(tmp_path.clone()), err))?;
        file.sync_all()
            .map_err(|err| EmbedError::io(Some(tmp_path.clone()), err))?;

        validate_not_html_payload(&tmp_path, dest, final_url.clone())?;
        validate_json_payload_if_required(file_spec, &tmp_path, dest, final_url.clone())?;

        let (actual_hash, size_bytes) = file_blake3(&tmp_path)?;
        if actual_hash != file_spec.blake3 {
            return Err(EmbedError::HashMismatch {
                path: dest.to_path_buf(),
                expected: file_spec.blake3.clone(),
                actual: actual_hash.clone(),
                details: vec![VerifyDetail {
                    path: file_spec.path.clone(),
                    expected_blake3: file_spec.blake3.clone(),
                    actual_blake3: Some(actual_hash),
                    size_bytes: Some(size_bytes),
                    state: VerifyState::Mismatch,
                    final_url: final_url.clone(),
                    etag: etag.clone(),
                    content_length,
                }],
            });
        }

        if dest.exists() {
            fs::remove_file(dest).map_err(|err| EmbedError::io(Some(dest.to_path_buf()), err))?;
        }
        fs::rename(&tmp_path, dest).map_err(|err| EmbedError::io(Some(dest.to_path_buf()), err))?;

        let provenance = FileProvenance {
            final_url,
            etag,
            content_length,
        };
        write_provenance(dest, &provenance)?;
        Ok(())
    })();

    if write_result.is_err() {
        cleanup_temp_artifacts(&tmp_path, &provenance_temp_path(dest));
    }

    write_result
}

fn validate_not_html_payload(
    sniff_path: &Path,
    file_path: &Path,
    final_url: Option<String>,
) -> Result<(), EmbedError> {
    let mut file = File::open(sniff_path)
        .map_err(|err| EmbedError::io(Some(sniff_path.to_path_buf()), err))?;
    let mut buf = [0_u8; 512];
    let read = file
        .read(&mut buf)
        .map_err(|err| EmbedError::io(Some(sniff_path.to_path_buf()), err))?;
    let snippet = &buf[..read];
    let lower: Vec<u8> = snippet.iter().map(|b| b.to_ascii_lowercase()).collect();
    let has_html =
        lower.windows(5).any(|w| w == b"<html") || lower.windows(9).any(|w| w == b"<!doctype");

    if has_html {
        return Err(EmbedError::InvalidPayloadHtml {
            path: file_path.to_path_buf(),
            snippet_len: read,
            final_url,
        });
    }

    Ok(())
}

fn validate_json_payload_if_required(
    file_spec: &ModelFileSpec,
    sniff_path: &Path,
    file_path: &Path,
    final_url: Option<String>,
) -> Result<(), EmbedError> {
    if !is_json_manifest_file(&file_spec.path) {
        return Ok(());
    }

    let bytes =
        fs::read(sniff_path).map_err(|err| EmbedError::io(Some(sniff_path.to_path_buf()), err))?;
    serde_json::from_slice::<serde_json::Value>(&bytes).map_err(|err| {
        EmbedError::InvalidPayloadJson {
            path: file_path.to_path_buf(),
            message: err.to_string(),
            final_url,
        }
    })?;
    Ok(())
}

fn is_json_manifest_file(path: &str) -> bool {
    matches!(path, "config.json" | "tokenizer.json")
}

fn cleanup_temp_artifacts(payload_tmp_path: &Path, meta_tmp_path: &Path) {
    let _ = fs::remove_file(payload_tmp_path);
    let _ = fs::remove_file(meta_tmp_path);
}

fn provenance_path(dest: &Path) -> PathBuf {
    let name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model-file");
    dest.with_file_name(format!("{name}.meta.json"))
}

fn write_provenance(dest: &Path, provenance: &FileProvenance) -> Result<(), EmbedError> {
    let metadata_path = provenance_path(dest);
    let metadata_tmp_path = provenance_temp_path(dest);
    let bytes = serde_json::to_vec(provenance).map_err(|err| EmbedError::Io {
        path: Some(metadata_path.clone()),
        message: err.to_string(),
    })?;

    let write_result = (|| -> Result<(), EmbedError> {
        let mut file = File::create(&metadata_tmp_path)
            .map_err(|err| EmbedError::io(Some(metadata_tmp_path.clone()), err))?;
        file.write_all(&bytes)
            .map_err(|err| EmbedError::io(Some(metadata_tmp_path.clone()), err))?;
        file.flush()
            .map_err(|err| EmbedError::io(Some(metadata_tmp_path.clone()), err))?;
        let _ = file.sync_all();

        if metadata_path.exists() {
            fs::remove_file(&metadata_path)
                .map_err(|err| EmbedError::io(Some(metadata_path.clone()), err))?;
        }
        fs::rename(&metadata_tmp_path, &metadata_path)
            .map_err(|err| EmbedError::io(Some(metadata_path.clone()), err))?;
        Ok(())
    })();

    if write_result.is_err() {
        let _ = fs::remove_file(&metadata_tmp_path);
    }

    write_result
}

fn read_provenance(dest: &Path) -> FileProvenance {
    let metadata_path = provenance_path(dest);
    let bytes = match fs::read(&metadata_path) {
        Ok(v) => v,
        Err(_) => return FileProvenance::default(),
    };

    serde_json::from_slice::<FileProvenance>(&bytes).unwrap_or_default()
}

fn provenance_temp_path(dest: &Path) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let name = dest
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("model-file");
    dest.with_file_name(format!(".{name}.meta.part-{pid}-{nanos}"))
}

fn temp_path(dest: &Path) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    let pid = std::process::id();
    let file_name = dest
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("download");
    let tmp_name = format!(".{file_name}.part-{pid}-{nanos}");
    dest.with_file_name(tmp_name)
}

fn file_blake3(path: &Path) -> Result<(String, u64), EmbedError> {
    let mut file = File::open(path).map_err(|err| EmbedError::io(Some(path.to_path_buf()), err))?;
    let mut hasher = Hasher::new();
    let mut buf = [0_u8; 8192];
    let mut total = 0_u64;
    loop {
        let read = file
            .read(&mut buf)
            .map_err(|err| EmbedError::io(Some(path.to_path_buf()), err))?;
        if read == 0 {
            break;
        }
        total += read as u64;
        hasher.update(&buf[..read]);
    }
    Ok((hasher.finalize().to_hex().to_string(), total))
}

fn render_rust_constants(revision: &str, details: &[VerifyDetail]) -> String {
    let mut out = format!("const CLASSIC_REVISION: &str = \"{}\";\n", revision);

    for detail in details {
        if let Some(actual) = &detail.actual_blake3 {
            let const_name = match detail.path.as_str() {
                "model.safetensors" => "HASH_MODEL_SAFETENSORS".to_string(),
                "tokenizer.json" => "HASH_TOKENIZER_JSON".to_string(),
                "config.json" => "HASH_CONFIG_JSON".to_string(),
                _ => format!("HASH_{}", sanitize_const_name(&detail.path)),
            };
            out.push_str(&format!("const {}: &str = \"{}\";\n", const_name, actual));
        }
    }

    out
}

fn sanitize_const_name(path: &str) -> String {
    let mut out = String::new();
    for c in path.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_uppercase());
        } else {
            out.push('_');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use once_cell::sync::Lazy;
    use std::sync::Mutex;
    use tempfile::TempDir;

    static ENV_LOCK: Lazy<Mutex<()>> = Lazy::new(|| Mutex::new(()));

    struct EnvGuard {
        key: &'static str,
        value: Option<String>,
    }

    impl EnvGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = env::var(key).ok();
            env::set_var(key, value);
            Self { key, value: prev }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(prev) = &self.value {
                env::set_var(self.key, prev);
            } else {
                env::remove_var(self.key);
            }
        }
    }

    #[test]
    fn resolves_dir_from_env_override() {
        let _guard = ENV_LOCK.lock().expect("env lock");
        let temp = TempDir::new().expect("tempdir");
        let _env = EnvGuard::set(
            "PA_MEASURE_MODEL_DIR",
            temp.path().to_str().expect("utf8 path"),
        );
        let manager = ModelManager::new(ModelSpec::classic());
        let dir = manager.resolved_dir();
        assert!(dir.starts_with(temp.path()));
    }

    #[test]
    fn verify_with_fake_manifest() {
        let temp = TempDir::new().expect("tempdir");
        let spec = ModelSpec {
            model_id: "test/model".to_string(),
            revision: "rev1".to_string(),
            base_url: "https://example.com".to_string(),
            required_files: vec![ModelFileSpec {
                path: "file.txt".to_string(),
                blake3: String::new(),
                size_bytes: None,
            }],
        };

        let manager = ModelManager::new_with_base_dir(spec.clone(), temp.path().to_path_buf());
        let dir = manager.resolved_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let file_path = dir.join("file.txt");
        fs::write(&file_path, b"hello").expect("write");

        let (hash, _) = file_blake3(&file_path).expect("hash");
        let mut spec = spec;
        spec.required_files[0].blake3 = hash.clone();
        let manager = ModelManager::new_with_base_dir(spec, temp.path().to_path_buf());

        let report = manager.verify().expect("verify");
        assert_eq!(report.files.len(), 1);
        assert_eq!(report.files[0].actual_blake3, hash);
        assert_eq!(report.details[0].state, VerifyState::Match);
    }

    #[test]
    fn verify_rejects_invalid_manifest_hash_format() {
        let temp = TempDir::new().expect("tempdir");
        let spec = ModelSpec {
            model_id: "test/model".to_string(),
            revision: "rev1".to_string(),
            base_url: "https://example.com".to_string(),
            required_files: vec![ModelFileSpec {
                path: "file.txt".to_string(),
                blake3: "zz".to_string(),
                size_bytes: None,
            }],
        };
        let manager = ModelManager::new_with_base_dir(spec, temp.path().to_path_buf());

        let err = manager.verify().expect_err("must fail");
        match err {
            EmbedError::InvalidHashFormat { path, hash } => {
                assert_eq!(path, "file.txt");
                assert_eq!(hash, "zz");
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    fn download_url_includes_model_id_and_revision() {
        let spec = ModelSpec::classic();
        let file = &spec.required_files[0];
        let url = spec.download_url(file);
        assert!(url.contains(&spec.model_id));
        assert!(url.contains(&spec.revision));
        assert!(url.ends_with(&file.path));
    }

    #[test]
    fn temp_path_is_part_file_in_same_directory() {
        let temp = TempDir::new().expect("tempdir");
        let dest = temp.path().join("model.safetensors");
        let tmp = temp_path(&dest);
        assert_eq!(tmp.parent(), dest.parent());
        let tmp_name = tmp
            .file_name()
            .and_then(|s| s.to_str())
            .expect("tmp file name");
        assert!(tmp_name.starts_with(".model.safetensors.part-"));
    }

    #[test]
    fn html_payload_guard_rejects_html() {
        let temp = TempDir::new().expect("tempdir");
        for file_name in ["model.safetensors", "tokenizer.json", "config.json"] {
            let sniff_path = temp.path().join(format!("{file_name}.part"));
            let final_path = temp.path().join(file_name);
            fs::write(&sniff_path, "<!DOCTYPE html><html>blocked</html>").expect("write html");
            let err = validate_not_html_payload(
                &sniff_path,
                &final_path,
                Some("https://example.com".to_string()),
            )
            .expect_err("must fail");
            match err {
                EmbedError::InvalidPayloadHtml {
                    path, snippet_len, ..
                } => {
                    assert_eq!(path, final_path);
                    assert!(snippet_len > 0);
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn json_payload_guard_rejects_invalid_json_for_config_and_tokenizer() {
        let temp = TempDir::new().expect("tempdir");
        for file_name in ["tokenizer.json", "config.json"] {
            let sniff_path = temp.path().join(format!("{file_name}.part"));
            let final_path = temp.path().join(file_name);
            fs::write(&sniff_path, "{\"broken\":").expect("write invalid json");

            let file_spec = ModelFileSpec {
                path: file_name.to_string(),
                blake3: String::new(),
                size_bytes: None,
            };
            let err = validate_json_payload_if_required(
                &file_spec,
                &sniff_path,
                &final_path,
                Some("https://example.com".to_string()),
            )
            .expect_err("must fail");

            match err {
                EmbedError::InvalidPayloadJson { path, .. } => {
                    assert_eq!(path, final_path);
                }
                other => panic!("unexpected error: {other:?}"),
            }
        }
    }

    #[test]
    fn cleanup_temp_artifacts_removes_payload_and_meta_parts() {
        let temp = TempDir::new().expect("tempdir");
        let payload_tmp = temp.path().join(".config.json.part-123");
        let meta_tmp = temp.path().join(".config.json.meta.part-123");
        fs::write(&payload_tmp, b"temp").expect("write payload temp");
        fs::write(&meta_tmp, b"temp").expect("write meta temp");

        cleanup_temp_artifacts(&payload_tmp, &meta_tmp);

        assert!(!payload_tmp.exists());
        assert!(!meta_tmp.exists());
    }

    #[test]
    fn provenance_sidecar_roundtrip() {
        let temp = TempDir::new().expect("tempdir");
        let dest = temp.path().join("tokenizer.json");
        fs::write(&dest, b"{}").expect("write");

        let expected = FileProvenance {
            final_url: Some("https://huggingface.co/example".to_string()),
            etag: Some("etag-value".to_string()),
            content_length: Some(123),
        };
        write_provenance(&dest, &expected).expect("write provenance");
        let actual = read_provenance(&dest);

        assert_eq!(actual.final_url, expected.final_url);
        assert_eq!(actual.etag, expected.etag);
        assert_eq!(actual.content_length, expected.content_length);
        assert!(provenance_path(&dest).exists());

        let temp_sidecars: Vec<PathBuf> = fs::read_dir(temp.path())
            .expect("read dir")
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|name| name.starts_with(".tokenizer.json.meta.part-"))
                    .unwrap_or(false)
            })
            .collect();
        assert!(temp_sidecars.is_empty(), "temporary sidecar files remain");
    }

    #[test]
    fn provenance_sidecar_overwrite_is_atomic_on_existing_meta() {
        let temp = TempDir::new().expect("tempdir");
        let dest = temp.path().join("config.json");
        fs::write(&dest, b"{}").expect("write");

        let first = FileProvenance {
            final_url: Some("https://example.com/first".to_string()),
            etag: Some("etag-1".to_string()),
            content_length: Some(111),
        };
        write_provenance(&dest, &first).expect("first provenance write");

        let second = FileProvenance {
            final_url: Some("https://example.com/second".to_string()),
            etag: Some("etag-2".to_string()),
            content_length: Some(222),
        };
        write_provenance(&dest, &second).expect("second provenance write");

        let actual = read_provenance(&dest);
        assert_eq!(actual.final_url, second.final_url);
        assert_eq!(actual.etag, second.etag);
        assert_eq!(actual.content_length, second.content_length);
    }

    #[test]
    fn verify_ignores_invalid_provenance_sidecar() {
        let temp = TempDir::new().expect("tempdir");
        let spec = ModelSpec {
            model_id: "test/model".to_string(),
            revision: "rev1".to_string(),
            base_url: "https://example.com".to_string(),
            required_files: vec![ModelFileSpec {
                path: "file.txt".to_string(),
                blake3: String::new(),
                size_bytes: None,
            }],
        };

        let manager = ModelManager::new_with_base_dir(spec.clone(), temp.path().to_path_buf());
        let dir = manager.resolved_dir();
        fs::create_dir_all(&dir).expect("mkdir");
        let file_path = dir.join("file.txt");
        fs::write(&file_path, b"hello").expect("write");

        let (hash, _) = file_blake3(&file_path).expect("hash");
        let mut spec = spec;
        spec.required_files[0].blake3 = hash;
        let manager = ModelManager::new_with_base_dir(spec, temp.path().to_path_buf());

        fs::write(provenance_path(&file_path), b"{not-json").expect("write invalid sidecar");
        let report = manager.verify().expect("verify");
        assert_eq!(report.details[0].state, VerifyState::Match);
        assert!(report.details[0].final_url.is_none());
        assert!(report.details[0].etag.is_none());
        assert!(report.details[0].content_length.is_none());
    }
}
