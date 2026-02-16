use serde_json::{Map, Value};
use std::env;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

const TARGET_ENV_KEY: &str = "PALE_ALE_TARGET";
const STATE_FILE_NAME: &str = "state.json";
const MAX_RECENT_TARGETS: usize = 5;
const NDJSON_PREVIEW_MAX_ROWS: usize = 200;
const NDJSON_PREVIEW_MAX_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug)]
pub(super) struct ResolvedTarget {
    pub target: PathBuf,
    pub ndjson_path: PathBuf,
    pub source: ResolveSource,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ResolveSource {
    Explicit,
    Env,
    State,
    CwdRunsManifest,
    CwdDefaultReport,
}

impl ResolveSource {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Explicit => "explicit",
            Self::Env => "env",
            Self::State => "state",
            Self::CwdRunsManifest => "cwd:runs/*/manifest.json",
            Self::CwdDefaultReport => "cwd:report_out.ndjson",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct ResolveRequest {
    pub explicit_target: Option<String>,
}

#[derive(Debug)]
pub(super) enum ResolveError {
    Unresolved(String),
    InvalidTarget(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TargetKind {
    Ndjson,
    Manifest,
    Unknown,
}

impl TargetKind {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Ndjson => "ndjson",
            Self::Manifest => "manifest",
            Self::Unknown => "unknown",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TargetCheck {
    Ok,
    Missing,
    Unreadable,
    Invalid,
}

impl TargetCheck {
    pub(super) fn as_str(self) -> &'static str {
        match self {
            Self::Ok => "OK",
            Self::Missing => "MISSING",
            Self::Unreadable => "UNREADABLE",
            Self::Invalid => "INVALID",
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct StatusCountsPreview {
    pub lucid: usize,
    pub hazy: usize,
    pub delirium: usize,
    pub unknown: usize,
}

#[derive(Clone, Debug, Default)]
pub(super) struct TargetPreview {
    pub modified: Option<SystemTime>,
    pub size_bytes: Option<u64>,
    pub sampled_rows: usize,
    pub sampled_bytes: usize,
    pub truncated: bool,
    pub counts: Option<StatusCountsPreview>,
    pub policy_hazy: Option<f64>,
    pub policy_delirium: Option<f64>,
}

#[derive(Clone, Debug)]
pub(super) struct TargetInspection {
    pub requested: PathBuf,
    pub canonical: Option<PathBuf>,
    pub check: TargetCheck,
    pub kind: TargetKind,
    pub hint: Option<String>,
    pub preview: Option<TargetPreview>,
}

pub(super) struct TargetResolver {
    cwd: PathBuf,
    env_target: Option<String>,
    state_path: Option<PathBuf>,
}

impl TargetResolver {
    pub(super) fn from_environment() -> Self {
        let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let env_target = env::var(TARGET_ENV_KEY).ok();
        let state_path = default_state_path();
        Self {
            cwd,
            env_target,
            state_path,
        }
    }

    pub(super) fn resolve(&self, request: ResolveRequest) -> Result<ResolvedTarget, ResolveError> {
        if let Some(explicit) = request.explicit_target {
            return self.resolve_target_str(&explicit, ResolveSource::Explicit);
        }

        if let Some(env_target) = self.env_target.as_deref() {
            if let Ok(resolved) = self.resolve_target_str(env_target, ResolveSource::Env) {
                return Ok(resolved);
            }
        }

        if let Some(state_target) = self.read_last_target() {
            if let Ok(resolved) = self.resolve_target_str(&state_target, ResolveSource::State) {
                return Ok(resolved);
            }
        }

        if let Some((target, source)) = self.discover_from_cwd() {
            if let Ok(resolved) = self.resolve_path(target, source) {
                return Ok(resolved);
            }
        }

        Err(ResolveError::Unresolved(
            "target was not resolved; tried PALE_ALE_TARGET, state last_target, and CWD discovery"
                .to_string(),
        ))
    }

    pub(super) fn inspect_path(&self, target: &Path) -> TargetInspection {
        let requested = self.normalize_target_path(target);
        let canonical = fs::canonicalize(&requested).ok();

        if !requested.exists() {
            return TargetInspection {
                requested,
                canonical,
                check: TargetCheck::Missing,
                kind: TargetKind::Unknown,
                hint: Some("missing file".to_string()),
                preview: None,
            };
        }

        if requested.is_file() {
            let kind = if has_ndjson_extension(&requested) {
                TargetKind::Ndjson
            } else {
                TargetKind::Unknown
            };
            let readable = File::open(&requested).is_ok();
            if !readable {
                return TargetInspection {
                    requested,
                    canonical,
                    check: TargetCheck::Unreadable,
                    kind,
                    hint: Some("permission denied or unreadable file".to_string()),
                    preview: None,
                };
            }

            if kind != TargetKind::Ndjson {
                return TargetInspection {
                    requested,
                    canonical,
                    check: TargetCheck::Invalid,
                    kind,
                    hint: Some("unsupported extension (expected .ndjson)".to_string()),
                    preview: None,
                };
            }

            return TargetInspection {
                requested: requested.clone(),
                canonical,
                check: TargetCheck::Ok,
                kind,
                hint: None,
                preview: build_ndjson_preview(&requested),
            };
        }

        if requested.is_dir() {
            let manifest_path = requested.join("manifest.json");
            if !manifest_path.is_file() {
                return TargetInspection {
                    requested,
                    canonical,
                    check: TargetCheck::Invalid,
                    kind: TargetKind::Unknown,
                    hint: Some("run bundle is missing manifest.json".to_string()),
                    preview: None,
                };
            }

            let manifest_value = match read_manifest_value(&manifest_path) {
                Ok(value) => value,
                Err(message) => {
                    return TargetInspection {
                        requested,
                        canonical,
                        check: TargetCheck::Unreadable,
                        kind: TargetKind::Manifest,
                        hint: Some(message),
                        preview: None,
                    };
                }
            };

            match self.resolve_run_bundle_dir(&requested) {
                Ok(report_path) => {
                    let mut preview = build_ndjson_preview(&report_path).unwrap_or_default();
                    let (hazy, delirium) = extract_policy_thresholds(&manifest_value);
                    preview.policy_hazy = hazy;
                    preview.policy_delirium = delirium;
                    TargetInspection {
                        requested,
                        canonical,
                        check: TargetCheck::Ok,
                        kind: TargetKind::Manifest,
                        hint: None,
                        preview: Some(preview),
                    }
                }
                Err(ResolveError::InvalidTarget(message))
                | Err(ResolveError::Unresolved(message)) => TargetInspection {
                    requested,
                    canonical,
                    check: TargetCheck::Invalid,
                    kind: TargetKind::Manifest,
                    hint: Some(message),
                    preview: None,
                },
            }
        } else {
            TargetInspection {
                requested,
                canonical,
                check: TargetCheck::Invalid,
                kind: TargetKind::Unknown,
                hint: Some("unsupported target type".to_string()),
                preview: None,
            }
        }
    }

    pub(super) fn recent_targets(&self) -> Vec<PathBuf> {
        let Ok(object) = self.read_state_object() else {
            return Vec::new();
        };
        object
            .get("recent_targets")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(PathBuf::from)
                    .collect::<Vec<PathBuf>>()
            })
            .unwrap_or_default()
    }

    pub(super) fn last_theme(&self) -> Option<String> {
        let object = self.read_state_object().ok()?;
        object
            .get("last_theme")
            .and_then(Value::as_str)
            .map(|value| value.to_string())
    }

    pub(super) fn persist_last_target(&self, target: &Path) -> Result<(), String> {
        let Some(state_path) = &self.state_path else {
            return Ok(());
        };
        let Some(state_dir) = state_path.parent() else {
            return Err("invalid state file path".to_string());
        };

        fs::create_dir_all(state_dir).map_err(|err| {
            format!(
                "failed to create state directory {}: {}",
                state_dir.display(),
                err
            )
        })?;

        let mut object = self.read_state_object()?;

        let absolute_target = if target.is_absolute() {
            target.to_path_buf()
        } else {
            self.cwd.join(target)
        };
        let normalized_target = fs::canonicalize(&absolute_target).unwrap_or(absolute_target);
        let normalized_value = normalized_target.display().to_string();
        object.insert(
            "last_target".to_string(),
            Value::String(normalized_value.clone()),
        );
        let mut recents = object
            .get("recent_targets")
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(Value::as_str)
                    .map(|item| item.to_string())
                    .collect::<Vec<String>>()
            })
            .unwrap_or_default();
        recents.retain(|item| item != &normalized_value);
        recents.insert(0, normalized_value);
        recents.truncate(MAX_RECENT_TARGETS);
        object.insert(
            "recent_targets".to_string(),
            Value::Array(recents.into_iter().map(Value::String).collect()),
        );

        let serialized = serde_json::to_string_pretty(&Value::Object(object))
            .map_err(|err| format!("failed to serialize state: {}", err))?;
        fs::write(state_path, serialized).map_err(|err| {
            format!(
                "failed to write state file {}: {}",
                state_path.display(),
                err
            )
        })?;
        Ok(())
    }

    pub(super) fn persist_last_theme(&self, theme: &str) -> Result<(), String> {
        let Some(state_path) = &self.state_path else {
            return Ok(());
        };
        let Some(state_dir) = state_path.parent() else {
            return Err("invalid state file path".to_string());
        };
        fs::create_dir_all(state_dir).map_err(|err| {
            format!(
                "failed to create state directory {}: {}",
                state_dir.display(),
                err
            )
        })?;

        let mut object = self.read_state_object()?;
        object.insert("last_theme".to_string(), Value::String(theme.to_string()));
        let serialized = serde_json::to_string_pretty(&Value::Object(object))
            .map_err(|err| format!("failed to serialize state: {}", err))?;
        fs::write(state_path, serialized).map_err(|err| {
            format!(
                "failed to write state file {}: {}",
                state_path.display(),
                err
            )
        })?;
        Ok(())
    }

    fn resolve_target_str(
        &self,
        raw_target: &str,
        source: ResolveSource,
    ) -> Result<ResolvedTarget, ResolveError> {
        let trimmed = raw_target.trim();
        if trimmed.is_empty() {
            return Err(ResolveError::InvalidTarget(
                "target must not be empty".to_string(),
            ));
        }
        let path = self.normalize_target_path(Path::new(trimmed));
        self.resolve_path(path, source)
    }

    fn normalize_target_path(&self, path: &Path) -> PathBuf {
        if path.is_absolute() {
            path.to_path_buf()
        } else {
            self.cwd.join(path)
        }
    }

    fn resolve_path(
        &self,
        target_path: PathBuf,
        source: ResolveSource,
    ) -> Result<ResolvedTarget, ResolveError> {
        if target_path.is_file() {
            self.validate_ndjson_file(&target_path)?;
            return Ok(ResolvedTarget {
                target: target_path.clone(),
                ndjson_path: target_path,
                source,
            });
        }

        if target_path.is_dir() {
            let ndjson_path = self.resolve_run_bundle_dir(&target_path)?;
            return Ok(ResolvedTarget {
                target: target_path,
                ndjson_path,
                source,
            });
        }

        Err(ResolveError::InvalidTarget(format!(
            "target does not exist: {}",
            target_path.display()
        )))
    }

    fn resolve_run_bundle_dir(&self, dir: &Path) -> Result<PathBuf, ResolveError> {
        let manifest_path = dir.join("manifest.json");
        if !manifest_path.is_file() {
            return Err(ResolveError::InvalidTarget(format!(
                "run bundle target must contain manifest.json: {}",
                dir.display()
            )));
        }

        let mut manifest_raw = String::new();
        File::open(&manifest_path)
            .and_then(|mut file| file.read_to_string(&mut manifest_raw))
            .map_err(|err| {
                ResolveError::InvalidTarget(format!(
                    "failed to read manifest {}: {}",
                    manifest_path.display(),
                    err
                ))
            })?;

        let manifest_value: Value = serde_json::from_str(&manifest_raw).map_err(|err| {
            ResolveError::InvalidTarget(format!(
                "failed to parse manifest {}: {}",
                manifest_path.display(),
                err
            ))
        })?;

        let mut candidates = manifest_report_candidates(dir, &manifest_value);
        candidates.push(dir.join("report_out.ndjson"));
        candidates.push(dir.join("pale-ale.batch.ndjson"));
        candidates.push(dir.join("report.ndjson"));

        for candidate in candidates {
            if !candidate.is_file() {
                continue;
            }
            self.validate_ndjson_file(&candidate)?;
            return Ok(candidate);
        }

        Err(ResolveError::InvalidTarget(format!(
            "manifest exists but no readable NDJSON report was found under {}",
            dir.display()
        )))
    }

    fn validate_ndjson_file(&self, path: &Path) -> Result<(), ResolveError> {
        let is_ndjson = path
            .extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.eq_ignore_ascii_case("ndjson"))
            .unwrap_or(false);
        if !is_ndjson {
            return Err(ResolveError::InvalidTarget(format!(
                "target file must be .ndjson: {}",
                path.display()
            )));
        }

        File::open(path).map_err(|err| {
            ResolveError::InvalidTarget(format!(
                "target is not readable {}: {}",
                path.display(),
                err
            ))
        })?;
        Ok(())
    }

    fn read_last_target(&self) -> Option<String> {
        let object = self.read_state_object().ok()?;
        object
            .get("last_target")
            .and_then(Value::as_str)
            .map(|value| value.to_string())
    }

    fn read_state_object(&self) -> Result<Map<String, Value>, String> {
        let Some(state_path) = &self.state_path else {
            return Ok(Map::new());
        };
        match fs::read_to_string(state_path) {
            Ok(raw) => match serde_json::from_str::<Value>(&raw) {
                Ok(Value::Object(map)) => Ok(map),
                Ok(_) => Ok(Map::new()),
                Err(err) => Err(format!(
                    "failed to parse state file {}: {}",
                    state_path.display(),
                    err
                )),
            },
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(Map::new()),
            Err(err) => Err(format!(
                "failed to read state file {}: {}",
                state_path.display(),
                err
            )),
        }
    }

    fn discover_from_cwd(&self) -> Option<(PathBuf, ResolveSource)> {
        if let Some(run_dir) = discover_latest_run_dir(&self.cwd) {
            return Some((run_dir, ResolveSource::CwdRunsManifest));
        }

        let report_path = self.cwd.join("report_out.ndjson");
        if report_path.is_file() {
            return Some((report_path, ResolveSource::CwdDefaultReport));
        }

        None
    }
}

fn manifest_report_candidates(base_dir: &Path, manifest: &Value) -> Vec<PathBuf> {
    let mut out = Vec::new();

    let push_candidate = |out: &mut Vec<PathBuf>, raw: &str| {
        let path = Path::new(raw);
        if path.is_absolute() {
            out.push(path.to_path_buf());
        } else {
            out.push(base_dir.join(path));
        }
    };

    let Some(root) = manifest.as_object() else {
        return out;
    };

    for key in ["report_ndjson", "report_path", "report", "ndjson"] {
        if let Some(value) = root.get(key).and_then(Value::as_str) {
            push_candidate(&mut out, value);
        }
    }

    if let Some(artifacts) = root.get("artifacts").and_then(Value::as_object) {
        for key in ["report_ndjson", "report_path", "report", "ndjson"] {
            if let Some(value) = artifacts.get(key).and_then(Value::as_str) {
                push_candidate(&mut out, value);
            }
        }
    }

    out
}

fn has_ndjson_extension(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| ext.eq_ignore_ascii_case("ndjson"))
        .unwrap_or(false)
}

fn read_manifest_value(path: &Path) -> Result<Value, String> {
    let mut raw = String::new();
    File::open(path)
        .and_then(|mut file| file.read_to_string(&mut raw))
        .map_err(|err| format!("failed to read manifest {}: {}", path.display(), err))?;
    serde_json::from_str::<Value>(&raw)
        .map_err(|err| format!("failed to parse manifest {}: {}", path.display(), err))
}

fn extract_policy_thresholds(manifest: &Value) -> (Option<f64>, Option<f64>) {
    let policy = manifest.get("policy").and_then(Value::as_object);
    let hazy = policy
        .and_then(|map| map.get("th_ratio_hazy"))
        .and_then(Value::as_f64);
    let delirium = policy
        .and_then(|map| map.get("th_ratio_delirium"))
        .and_then(Value::as_f64);
    (hazy, delirium)
}

fn build_ndjson_preview(path: &Path) -> Option<TargetPreview> {
    let metadata = fs::metadata(path).ok();
    let modified = metadata.as_ref().and_then(|meta| meta.modified().ok());
    let size_bytes = metadata.as_ref().map(|meta| meta.len());

    let file = File::open(path).ok()?;
    let reader = BufReader::new(file);
    let mut preview = TargetPreview {
        modified,
        size_bytes,
        ..TargetPreview::default()
    };
    let mut counts = StatusCountsPreview::default();
    let mut parsed_any_status = false;

    for line_result in reader.lines() {
        if preview.sampled_rows >= NDJSON_PREVIEW_MAX_ROWS {
            preview.truncated = true;
            break;
        }
        let Ok(raw_line) = line_result else {
            break;
        };
        let line_bytes = raw_line.len().saturating_add(1);
        if preview.sampled_bytes.saturating_add(line_bytes) > NDJSON_PREVIEW_MAX_BYTES {
            preview.truncated = true;
            break;
        }
        preview.sampled_bytes = preview.sampled_bytes.saturating_add(line_bytes);

        let trimmed = raw_line.trim();
        if trimmed.is_empty() {
            continue;
        }
        preview.sampled_rows = preview.sampled_rows.saturating_add(1);

        let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
            continue;
        };
        let status = value
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or("UNKNOWN");
        parsed_any_status = true;
        match status {
            "LUCID" => counts.lucid = counts.lucid.saturating_add(1),
            "HAZY" => counts.hazy = counts.hazy.saturating_add(1),
            "DELIRIUM" => counts.delirium = counts.delirium.saturating_add(1),
            _ => counts.unknown = counts.unknown.saturating_add(1),
        }
    }

    if parsed_any_status {
        preview.counts = Some(counts);
    }
    Some(preview)
}

fn discover_latest_run_dir(cwd: &Path) -> Option<PathBuf> {
    let runs_dir = cwd.join("runs");
    if !runs_dir.is_dir() {
        return None;
    }

    let mut candidates = Vec::new();
    let entries = fs::read_dir(&runs_dir).ok()?;
    for entry in entries.flatten() {
        let run_dir = entry.path();
        if !run_dir.is_dir() {
            continue;
        }

        let manifest_path = run_dir.join("manifest.json");
        if !manifest_path.is_file() {
            continue;
        }

        let modified = fs::metadata(&manifest_path)
            .and_then(|meta| meta.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        candidates.push((run_dir, modified));
    }

    if candidates.is_empty() {
        return None;
    }

    candidates.sort_by(|left, right| {
        right.1.cmp(&left.1).then_with(|| {
            left.0
                .display()
                .to_string()
                .cmp(&right.0.display().to_string())
        })
    });
    candidates.into_iter().next().map(|item| item.0)
}

fn default_state_path() -> Option<PathBuf> {
    let base = if cfg!(windows) {
        env::var_os("APPDATA").map(PathBuf::from)
    } else if let Some(value) = env::var_os("XDG_CONFIG_HOME") {
        Some(PathBuf::from(value))
    } else {
        env::var_os("HOME")
            .map(PathBuf::from)
            .map(|home| home.join(".config"))
    }?;

    Some(base.join("pale-ale").join(STATE_FILE_NAME))
}
