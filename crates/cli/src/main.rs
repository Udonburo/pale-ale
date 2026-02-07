mod batch;

use clap::{error::ErrorKind, Parser, Subcommand, ValueEnum};
use pale_ale_diagnose::{
    compute_inputs_hash, default_measurement_config, default_policy_config, diagnose_eval,
    measure_eval, measurement_hash, policy_hash, AttestationLevel, AuditTrace, ConfigSource,
    EvalReport, HashesTrace, MeasureError, ModelFile, ModelTrace, VerdictStatus,
};
use pale_ale_embed::{
    EmbedError, Embedder, ModelManager, ModelSpec, PrintHashesReport, VerifyDetail, VerifyReport,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

const PRINT_HASHES_WARNING: &str =
    "hashes are computed from local cache bytes; treat as canonical only after pinned revision verification";

#[derive(Parser, Debug)]
#[command(name = "pale-ale", version, about = "Pale-Ale Classic CLI")]
struct Cli {
    #[command(subcommand)]
    command: Commands,

    #[arg(long, short = 'j', global = true)]
    json: bool,

    #[arg(long, global = true)]
    offline: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Doctor,
    Model {
        #[command(subcommand)]
        command: ModelCommand,
    },
    Embed {
        text: String,
    },
    Eval {
        query: String,
        context: String,
        answer: String,
    },
    Batch {
        input: String,
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "jsonl")]
        format: BatchFormat,
        #[arg(long)]
        threads: Option<usize>,
        #[arg(long)]
        max_rows: Option<usize>,
        #[arg(long)]
        strict: bool,
        #[arg(long)]
        dry_run: bool,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BatchFormat {
    Jsonl,
    Tsv,
}

#[derive(Subcommand, Debug)]
enum ModelCommand {
    Status,
    Download,
    Verify,
    Path,
    PrintHashes,
    ClearCache {
        #[arg(long)]
        yes: bool,
    },
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum AppErrorKind {
    Usage,
    NotImplemented,
    ModelMissingOffline,
    Dependency,
    Internal,
}

#[derive(Clone, Debug)]
struct AppError {
    kind: AppErrorKind,
    code: &'static str,
    message: String,
    details: Box<Value>,
    data: Option<Box<Value>>,
    inputs_hash: Option<String>,
}

impl AppError {
    fn usage(message: String) -> Self {
        Self {
            kind: AppErrorKind::Usage,
            code: "CLI_USAGE",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn model_missing_offline(message: String) -> Self {
        Self {
            kind: AppErrorKind::ModelMissingOffline,
            code: "MODEL_MISSING_OFFLINE",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn offline_forbids_download(message: String) -> Self {
        Self {
            kind: AppErrorKind::Dependency,
            code: "OFFLINE_FORBIDS_DOWNLOAD",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn model_missing(message: String) -> Self {
        Self {
            kind: AppErrorKind::Dependency,
            code: "MODEL_MISSING",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn model_file_missing(message: String) -> Self {
        Self {
            kind: AppErrorKind::Dependency,
            code: "MODEL_FILE_MISSING",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn model_hash_mismatch(message: String) -> Self {
        Self {
            kind: AppErrorKind::Dependency,
            code: "MODEL_HASH_MISMATCH",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn model_hash_format_invalid(message: String) -> Self {
        Self {
            kind: AppErrorKind::Dependency,
            code: "MODEL_HASH_FORMAT_INVALID",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn model_invalid_payload(message: String) -> Self {
        Self {
            kind: AppErrorKind::Dependency,
            code: "MODEL_INVALID_PAYLOAD",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn dependency(message: String) -> Self {
        Self {
            kind: AppErrorKind::Dependency,
            code: "DEPENDENCY_ERROR",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn internal(message: String) -> Self {
        Self {
            kind: AppErrorKind::Internal,
            code: "INTERNAL_ERROR",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn strict_failure(message: String) -> Self {
        Self {
            kind: AppErrorKind::Dependency,
            code: "BATCH_STRICT_FAILURE",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn exit_code(&self) -> i32 {
        match self.kind {
            AppErrorKind::Usage => 1,
            AppErrorKind::ModelMissingOffline
            | AppErrorKind::Dependency
            | AppErrorKind::NotImplemented
            | AppErrorKind::Internal => 2,
        }
    }

    fn with_details_and_data(mut self, details: Value) -> Self {
        self.details = Box::new(details.clone());
        self.data = Some(Box::new(json!({ "details": details })));
        self
    }

    fn with_data(mut self, data: Value) -> Self {
        self.data = Some(Box::new(data));
        self
    }

    fn with_inputs_hash(mut self, inputs_hash: String) -> Self {
        self.inputs_hash = Some(inputs_hash);
        self
    }
}

#[derive(Serialize)]
struct JsonEnvelope {
    status: String,
    error: Option<ErrorEnvelope>,
    audit_trace: AuditTrace,
    data: Option<Value>,
}

#[derive(Serialize)]
struct ErrorEnvelope {
    code: String,
    message: String,
    details: Value,
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let wants_json = args.iter().any(|arg| arg == "--json" || arg == "-j");

    match Cli::try_parse_from(&args) {
        Ok(cli) => {
            let json = cli.json || wants_json;
            match run(cli, json) {
                Ok(envelope) => {
                    if json {
                        print_json(&envelope);
                    }
                    std::process::exit(0);
                }
                Err(err) => {
                    let exit_code = err.exit_code();
                    if json {
                        let envelope = error_envelope(&err);
                        print_json(&envelope);
                    } else {
                        eprintln!("{}", err.message);
                    }
                    std::process::exit(exit_code);
                }
            }
        }
        Err(err) => match err.kind() {
            ErrorKind::DisplayHelp | ErrorKind::DisplayVersion => {
                print!("{err}");
                std::process::exit(0);
            }
            _ => {
                if wants_json {
                    let usage = AppError::usage(err.to_string());
                    let envelope = error_envelope(&usage);
                    print_json(&envelope);
                } else {
                    let _ = err.print();
                }
                std::process::exit(1);
            }
        },
    }
}

fn run(cli: Cli, json: bool) -> Result<JsonEnvelope, AppError> {
    match cli.command {
        Commands::Doctor => doctor(cli.offline),
        Commands::Model { command } => model(command, cli.offline, json),
        Commands::Embed { text } => embed(text, cli.offline, json),
        Commands::Eval {
            query,
            context,
            answer,
        } => eval(query, context, answer, cli.offline, json),
        Commands::Batch {
            input,
            out,
            format,
            threads,
            max_rows,
            strict,
            dry_run,
        } => batch::run(batch::BatchCommand {
            input,
            out,
            format,
            threads,
            max_rows,
            offline: cli.offline,
            json_output: json,
            strict,
            dry_run,
        }),
    }
}

fn doctor(offline: bool) -> Result<JsonEnvelope, AppError> {
    Ok(JsonEnvelope {
        status: "OK".to_string(),
        error: None,
        audit_trace: audit_trace_with_model(default_model_trace()),
        data: Some(json!({ "mode": "doctor", "offline": offline })),
    })
}

fn model(command: ModelCommand, offline: bool, json: bool) -> Result<JsonEnvelope, AppError> {
    let manager = ModelManager::new(ModelSpec::classic());
    match command {
        ModelCommand::Download => {
            let report = manager.download(offline).map_err(map_embed_error)?;
            Ok(JsonEnvelope {
                status: "OK".to_string(),
                error: None,
                audit_trace: audit_trace_with_model(model_trace_from_verify(
                    manager.spec(),
                    &report,
                )),
                data: Some(json!(report)),
            })
        }
        ModelCommand::Verify => {
            let report = manager.verify().map_err(map_embed_error)?;
            Ok(JsonEnvelope {
                status: "OK".to_string(),
                error: None,
                audit_trace: audit_trace_with_model(model_trace_from_verify(
                    manager.spec(),
                    &report,
                )),
                data: Some(json!(report)),
            })
        }
        ModelCommand::Path => Ok(JsonEnvelope {
            status: "OK".to_string(),
            error: None,
            audit_trace: audit_trace_with_model(model_trace_from_spec(manager.spec())),
            data: Some(json!({ "path": manager.resolved_dir() })),
        }),
        ModelCommand::Status => {
            let report = manager.status();
            Ok(JsonEnvelope {
                status: "OK".to_string(),
                error: None,
                audit_trace: audit_trace_with_model(model_trace_from_spec(manager.spec())),
                data: Some(json!(report)),
            })
        }
        ModelCommand::PrintHashes => {
            let status = manager.status();
            if !status.cache_present || !status.missing_files.is_empty() {
                return Err(AppError::model_missing(format!(
                    "model cache missing or incomplete: {}",
                    status.cache_dir.display()
                ))
                .with_data(json!({ "warning": PRINT_HASHES_WARNING })));
            }
            let report = manager.print_hashes().map_err(map_embed_error)?;
            if !json {
                print!("{}", report.rust_constants);
                eprintln!("WARNING: {}", PRINT_HASHES_WARNING);
            }
            Ok(JsonEnvelope {
                status: "OK".to_string(),
                error: None,
                audit_trace: audit_trace_with_model(model_trace_from_details(
                    manager.spec(),
                    &report.details,
                )),
                data: Some(print_hashes_data_with_warning(&report)),
            })
        }
        ModelCommand::ClearCache { yes } => {
            if !yes {
                return Err(AppError::usage(
                    "model clear-cache requires --yes to perform deletion".to_string(),
                ));
            }
            let path = manager.resolved_dir();
            let bytes_freed = if path.exists() {
                dir_size_bytes(&path).ok()
            } else {
                None
            };
            let deleted = manager.clear_cache().map_err(map_embed_error)?;
            Ok(JsonEnvelope {
                status: "OK".to_string(),
                error: None,
                audit_trace: audit_trace_with_model(model_trace_from_spec(manager.spec())),
                data: Some(json!({
                    "deleted": deleted,
                    "path": path,
                    "bytes_freed": bytes_freed,
                })),
            })
        }
    }
}

fn embed(text: String, offline: bool, json: bool) -> Result<JsonEnvelope, AppError> {
    let manager = ModelManager::new(ModelSpec::classic());
    let verify_report = if offline {
        let status = manager.status();
        if !status.cache_present || !status.missing_files.is_empty() {
            return Err(AppError::model_missing_offline(format!(
                "model cache missing in offline mode: {}",
                status.cache_dir.display()
            )));
        }
        manager.verify().map_err(map_embed_error)?
    } else {
        manager.download(false).map_err(map_embed_error)?
    };

    let embedder = Embedder::new(&manager).map_err(map_embed_error)?;
    let vector = embedder.embed(&text).map_err(map_embed_error)?;
    let dim = vector.len();

    if !json {
        print_vector_summary(&vector);
    }

    Ok(JsonEnvelope {
        status: "OK".to_string(),
        error: None,
        audit_trace: audit_trace_with_model(model_trace_from_verify(
            manager.spec(),
            &verify_report,
        )),
        data: Some(json!({
            "dim": dim,
            "vector": vector,
        })),
    })
}

fn eval(
    query: String,
    context: String,
    answer: String,
    offline: bool,
    json: bool,
) -> Result<JsonEnvelope, AppError> {
    let inputs_hash_hex = compute_inputs_hash(&query, &context, &answer);
    let manager = ModelManager::new(ModelSpec::classic());
    let verify_report = if offline {
        let status = manager.status();
        if !status.cache_present || !status.missing_files.is_empty() {
            return Err(AppError::model_missing_offline(format!(
                "model cache missing in offline mode: {}",
                status.cache_dir.display()
            ))
            .with_inputs_hash(inputs_hash_hex.clone()));
        }
        manager
            .verify()
            .map_err(map_embed_error)
            .map_err(|err| err.with_inputs_hash(inputs_hash_hex.clone()))?
    } else {
        manager
            .download(false)
            .map_err(map_embed_error)
            .map_err(|err| err.with_inputs_hash(inputs_hash_hex.clone()))?
    };

    let embedder = Embedder::new(&manager)
        .map_err(map_embed_error)
        .map_err(|err| err.with_inputs_hash(inputs_hash_hex.clone()))?;
    let measurement = measure_eval(
        &|sentence: &str| {
            embedder
                .embed(sentence)
                .map_err(|err| format!("embed failure: {:?}", err))
        },
        &query,
        &context,
        &answer,
    )
    .map_err(map_measure_error)
    .map_err(|err| err.with_inputs_hash(inputs_hash_hex.clone()))?;
    let policy = default_policy_config();
    let diagnose_result = diagnose_eval(measurement, &policy);

    if !json {
        print_eval_report(diagnose_result.status, &diagnose_result.report);
    }

    Ok(JsonEnvelope {
        status: verdict_status_str(diagnose_result.status).to_string(),
        error: None,
        audit_trace: audit_trace_with_model_and_inputs(
            model_trace_from_verify(manager.spec(), &verify_report),
            Some(inputs_hash_hex),
        ),
        data: Some(json!(diagnose_result.report)),
    })
}

fn print_vector_summary(vector: &[f32]) {
    let preview: Vec<String> = vector.iter().take(8).map(|v| format!("{v:.6}")).collect();
    let suffix = if vector.len() > 8 { ", ..." } else { "" };
    println!(
        "embedding dim={} [{}{}]",
        vector.len(),
        preview.join(", "),
        suffix
    );
}

fn print_eval_report(status: VerdictStatus, report: &EvalReport) {
    println!("verdict: {}", verdict_status_str(status));
    println!(
        "scores: ctx_n={} ans_n={} pairs_n={} max_ratio={:.6} max_struct={:.6} min_sem={:.6}",
        report.scores.ctx_n,
        report.scores.ans_n,
        report.scores.pairs_n,
        report.scores.max_score_ratio,
        report.scores.max_score_struct,
        report.scores.min_score_sem,
    );
    for item in report.evidence.iter().take(3) {
        println!(
            "ans[{}] -> ctx[{}] ratio={:.6} struct={:.6} sem={:.6} tags={}",
            item.ans_sentence_index,
            item.ctx_sentence_index,
            item.score_ratio,
            item.score_struct,
            item.score_sem,
            item.tags.join("|")
        );
    }
}

fn verdict_status_str(status: VerdictStatus) -> &'static str {
    match status {
        VerdictStatus::Lucid => "LUCID",
        VerdictStatus::Hazy => "HAZY",
        VerdictStatus::Delirium => "DELIRIUM",
    }
}

fn audit_trace_with_model(model: ModelTrace) -> AuditTrace {
    audit_trace_with_model_and_inputs(model, None)
}

fn audit_trace_with_model_and_inputs(model: ModelTrace, inputs_hash: Option<String>) -> AuditTrace {
    let measurement = measurement_hash(&default_measurement_config());
    let policy = policy_hash(&default_policy_config());

    AuditTrace {
        model,
        hashes: HashesTrace {
            measurement_hash: measurement,
            policy_hash: policy,
            inputs_hash: inputs_hash.unwrap_or_else(|| "UNAVAILABLE".to_string()),
        },
        config_source: ConfigSource::Default,
        attestation_level: AttestationLevel::Unattested,
        invalid_block_rate: 0.0,
        comparability: Value::Null,
        error: None,
        build: None,
    }
}

fn error_envelope(err: &AppError) -> JsonEnvelope {
    JsonEnvelope {
        status: "UNKNOWN".to_string(),
        error: Some(ErrorEnvelope {
            code: err.code.to_string(),
            message: err.message.clone(),
            details: (*err.details).clone(),
        }),
        audit_trace: audit_trace_with_model_and_inputs(
            default_model_trace(),
            err.inputs_hash.clone(),
        ),
        data: err.data.as_deref().cloned(),
    }
}

fn print_json(envelope: &JsonEnvelope) {
    let json = serde_json::to_string(envelope).expect("failed to serialize json");
    println!("{json}");
}

fn default_model_trace() -> ModelTrace {
    let spec = ModelSpec::classic();
    model_trace_from_spec(&spec)
}

fn model_trace_from_spec(spec: &ModelSpec) -> ModelTrace {
    ModelTrace {
        model_id: spec.model_id.clone(),
        revision: spec.revision.clone(),
        files: spec
            .required_files
            .iter()
            .map(|file| ModelFile {
                path: file.path.clone(),
                blake3: Some(file.blake3.clone()),
                size_bytes: file.size_bytes,
            })
            .collect(),
    }
}

fn model_trace_from_verify(spec: &ModelSpec, report: &VerifyReport) -> ModelTrace {
    ModelTrace {
        model_id: spec.model_id.clone(),
        revision: spec.revision.clone(),
        files: report
            .files
            .iter()
            .map(|file| ModelFile {
                path: file.path.clone(),
                blake3: Some(file.actual_blake3.clone()),
                size_bytes: Some(file.size_bytes),
            })
            .collect(),
    }
}

fn model_trace_from_details(spec: &ModelSpec, details: &[VerifyDetail]) -> ModelTrace {
    ModelTrace {
        model_id: spec.model_id.clone(),
        revision: spec.revision.clone(),
        files: details
            .iter()
            .map(|detail| ModelFile {
                path: detail.path.clone(),
                blake3: detail
                    .actual_blake3
                    .clone()
                    .or_else(|| Some(detail.expected_blake3.clone())),
                size_bytes: detail.size_bytes,
            })
            .collect(),
    }
}

fn print_hashes_data_with_warning(report: &PrintHashesReport) -> Value {
    let mut value =
        serde_json::to_value(report).expect("failed to serialize print hashes report to json");
    if let Some(obj) = value.as_object_mut() {
        obj.insert(
            "warning".to_string(),
            Value::String(PRINT_HASHES_WARNING.to_string()),
        );
    }
    value
}

fn dir_size_bytes(path: &Path) -> std::io::Result<u64> {
    if path.is_file() {
        return fs::metadata(path).map(|m| m.len());
    }
    let mut total = 0_u64;
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            total = total.saturating_add(dir_size_bytes(&entry_path)?);
        } else {
            total = total.saturating_add(entry.metadata()?.len());
        }
    }
    Ok(total)
}

fn map_embed_error(err: EmbedError) -> AppError {
    match err {
        EmbedError::OfflineForbidden => {
            AppError::offline_forbids_download("offline mode forbids model download".to_string())
        }
        EmbedError::ModelMissing { dir } => {
            AppError::model_missing(format!("model cache missing: {}", dir.display()))
        }
        EmbedError::ModelFileMissing { path, details } => {
            AppError::model_file_missing(format!("model file missing: {}", path.display()))
                .with_details_and_data(json!(details))
        }
        EmbedError::HashMismatch {
            path,
            expected,
            actual,
            details,
        } => AppError::model_hash_mismatch(format!(
            "model hash mismatch: {} (expected {}, got {})",
            path.display(),
            expected,
            actual
        ))
        .with_details_and_data(json!(details)),
        EmbedError::InvalidHashFormat { path, hash } => AppError::model_hash_format_invalid(
            format!("invalid manifest hash for {}: {}", path, hash),
        )
        .with_details_and_data(json!({ "path": path, "hash": hash })),
        EmbedError::InvalidPayloadHtml {
            path,
            snippet_len,
            final_url,
        } => {
            AppError::model_invalid_payload(format!("invalid html payload for {}", path.display()))
                .with_details_and_data(
                    json!({ "path": path, "snippet_len": snippet_len, "final_url": final_url }),
                )
        }
        EmbedError::InvalidPayloadJson {
            path,
            message,
            final_url,
        } => {
            let detail_message = message.clone();
            AppError::model_invalid_payload(format!(
                "invalid json payload for {}: {}",
                path.display(),
                detail_message
            ))
            .with_details_and_data(json!({
                "path": path,
                "message": message,
                "final_url": final_url
            }))
        }
        EmbedError::DownloadFailed {
            url,
            message,
            final_url,
            etag,
            content_length,
        } => AppError::dependency(format!("model download failed: {} ({})", url, message))
            .with_details_and_data(json!({
                "request_url": url,
                "final_url": final_url,
                "etag": etag,
                "content_length": content_length,
            })),
        EmbedError::InvalidUrl { url } => AppError::internal(format!("invalid model URL: {}", url)),
        EmbedError::ConfigLoad { path, message } => AppError::dependency(format!(
            "invalid model config at {}: {}",
            path.display(),
            message
        )),
        EmbedError::ModelLoad { path, message } => AppError::dependency(format!(
            "failed to load model at {}: {}",
            path.display(),
            message
        )),
        EmbedError::TokenizerLoad { path, message } => AppError::dependency(format!(
            "failed to load tokenizer at {}: {}",
            path.display(),
            message
        )),
        EmbedError::Tokenization { message } => {
            AppError::internal(format!("tokenization failed: {}", message))
        }
        EmbedError::Tensor { message } => {
            AppError::internal(format!("tensor operation failed: {}", message))
        }
        EmbedError::Inference { message } => {
            AppError::internal(format!("embedding inference failed: {}", message))
        }
        EmbedError::Io { path, message } => {
            let detail = path
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            AppError::dependency(format!("I/O error at {}: {}", detail, message))
        }
    }
}

fn map_measure_error(err: MeasureError) -> AppError {
    AppError::internal(format!("measurement failed: {}", err))
}
