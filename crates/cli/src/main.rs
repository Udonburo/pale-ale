mod batch;
mod calibrate;
#[cfg(feature = "cli-tui")]
mod launcher;
mod report;
#[cfg(feature = "cli-tui")]
mod target_resolver;

use clap::{error::ErrorKind, Parser, Subcommand, ValueEnum};
use pale_ale_diagnose::{
    compute_inputs_hash, default_measurement_config, default_policy_config, diagnose_eval,
    measure_eval, measurement_hash, policy_hash, AttestationLevel, AuditTrace, AuditWarning,
    ConfigSource, EvalReport, HashesTrace, MeasureError, MeasurementConfig, ModelFile, ModelTrace,
    SentenceEmbedding, VerdictStatus,
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
const TARGET_UNRESOLVED_NON_TTY_EXIT: i32 = 20;
const LAUNCHER_REQUIRES_TTY_EXIT: i32 = 21;
const TARGET_INVALID_EXIT: i32 = 22;

#[derive(Parser, Debug)]
#[command(name = "pale-ale", version, about = "Pale-Ale Classic CLI")]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(long, short = 'j', global = true)]
    json: bool,

    #[arg(long, global = true)]
    offline: bool,
}

#[derive(Subcommand, Debug)]
enum Commands {
    Tui {
        target: Option<String>,
        #[arg(long = "target", short = 't', value_name = "TARGET")]
        target_flag: Option<String>,
        #[arg(long, value_enum, default_value = "classic")]
        theme: ReportThemeArg,
        #[arg(long, value_enum, default_value = "auto")]
        color: ReportColorArg,
        #[arg(long)]
        ascii: bool,
    },
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
    Report {
        input_ndjson: String,
        #[arg(long)]
        summary: bool,
        #[arg(long, default_value = "10")]
        top: usize,
        #[arg(long = "filter")]
        filters: Vec<String>,
        #[arg(long)]
        find: Option<String>,
    },
    Calibrate {
        input_ndjson: String,
        #[arg(long, value_enum, default_value = "quantile")]
        method: CalibrateMethod,
        #[arg(long, default_value = "0.90")]
        hazy_q: f64,
        #[arg(long, default_value = "0.98")]
        delirium_q: f64,
        #[arg(long, default_value = "50")]
        min_rows: usize,
        #[arg(long)]
        out: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BatchFormat {
    Jsonl,
    Tsv,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum CalibrateMethod {
    Quantile,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ReportThemeArg {
    Classic,
    Term,
    Cyber,
}

impl From<ReportThemeArg> for report::ReportTheme {
    fn from(value: ReportThemeArg) -> Self {
        match value {
            ReportThemeArg::Classic => report::ReportTheme::Classic,
            ReportThemeArg::Term => report::ReportTheme::Term,
            ReportThemeArg::Cyber => report::ReportTheme::Cyber,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ReportColorArg {
    Auto,
    Always,
    Never,
}

impl From<ReportColorArg> for report::ReportColor {
    fn from(value: ReportColorArg) -> Self {
        match value {
            ReportColorArg::Auto => report::ReportColor::Auto,
            ReportColorArg::Always => report::ReportColor::Always,
            ReportColorArg::Never => report::ReportColor::Never,
        }
    }
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

    fn calibration_insufficient_data(message: String) -> Self {
        Self {
            kind: AppErrorKind::Dependency,
            code: "CALIBRATION_INSUFFICIENT_DATA",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn target_unresolved_non_tty(message: String) -> Self {
        Self {
            kind: AppErrorKind::Usage,
            code: "TARGET_UNRESOLVED_NON_TTY",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn launcher_requires_tty(message: String) -> Self {
        Self {
            kind: AppErrorKind::Usage,
            code: "LAUNCHER_REQUIRES_TTY",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn target_invalid(message: String) -> Self {
        Self {
            kind: AppErrorKind::Usage,
            code: "TARGET_INVALID",
            message,
            details: Box::new(Value::Null),
            data: None,
            inputs_hash: None,
        }
    }

    fn exit_code(&self) -> i32 {
        match self.code {
            "TARGET_UNRESOLVED_NON_TTY" => return TARGET_UNRESOLVED_NON_TTY_EXIT,
            "LAUNCHER_REQUIRES_TTY" => return LAUNCHER_REQUIRES_TTY_EXIT,
            "TARGET_INVALID" => return TARGET_INVALID_EXIT,
            _ => {}
        }

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
        None => run_launcher(json),
        Some(Commands::Tui {
            target,
            target_flag,
            theme,
            color,
            ascii,
        }) => run_tui_command(target, target_flag, theme.into(), color.into(), ascii),
        Some(Commands::Doctor) => doctor(cli.offline),
        Some(Commands::Model { command }) => model(command, cli.offline, json),
        Some(Commands::Embed { text }) => embed(text, cli.offline, json),
        Some(Commands::Eval {
            query,
            context,
            answer,
        }) => eval(query, context, answer, cli.offline, json),
        Some(Commands::Batch {
            input,
            out,
            format,
            threads,
            max_rows,
            strict,
            dry_run,
        }) => batch::run(batch::BatchCommand {
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
        Some(Commands::Report {
            input_ndjson,
            summary,
            top,
            filters,
            find,
        }) => report::run(
            report::ReportCommand {
                input: input_ndjson,
                summary,
                top,
                filters,
                find,
                tui: false,
                theme: report::ReportTheme::Classic,
                color: report::ReportColor::Auto,
                ascii: false,
            },
            json,
        ),
        Some(Commands::Calibrate {
            input_ndjson,
            method,
            hazy_q,
            delirium_q,
            min_rows,
            out,
        }) => calibrate::run(
            calibrate::CalibrateCommand {
                input: input_ndjson,
                method,
                hazy_q,
                delirium_q,
                min_rows,
                out,
            },
            json,
        ),
    }
}

fn ok_cli_envelope(data: Option<Value>) -> JsonEnvelope {
    JsonEnvelope {
        status: "OK".to_string(),
        error: None,
        audit_trace: audit_trace_with_model(default_model_trace()),
        data,
    }
}

fn run_launcher(json: bool) -> Result<JsonEnvelope, AppError> {
    #[cfg(not(feature = "cli-tui"))]
    {
        let _ = json;
        return Err(AppError::usage(
            "launcher is unavailable in this build; recompile with --features cli-tui".to_string(),
        ));
    }

    #[cfg(feature = "cli-tui")]
    {
        use launcher::LauncherAction;
        use target_resolver::TargetResolver;

        if !launcher::stdio_is_tty() {
            return Err(AppError::launcher_requires_tty(
                "launcher requires an interactive TTY; use `pale-ale tui --target <TARGET>`"
                    .to_string(),
            ));
        }
        if json {
            return Err(AppError::usage(
                "launcher is interactive; use `pale-ale tui [TARGET]` for scripted usage"
                    .to_string(),
            ));
        }

        let resolver = TargetResolver::from_environment();
        match launcher::run_launcher(&resolver)
            .map_err(|message| AppError::internal(format!("launcher failed: {}", message)))?
        {
            LauncherAction::Quit => Ok(ok_cli_envelope(Some(json!({
                "mode": "launcher",
                "action": "quit"
            })))),
            LauncherAction::Launch { target, theme } => run_resolved_tui(
                &resolver,
                target,
                report::TuiOptions {
                    theme,
                    color: report::ReportColor::Auto,
                    ascii: false,
                },
            ),
        }
    }
}

fn run_tui_command(
    target_positional: Option<String>,
    target_flag: Option<String>,
    theme: report::ReportTheme,
    color: report::ReportColor,
    ascii: bool,
) -> Result<JsonEnvelope, AppError> {
    if target_positional.is_some() && target_flag.is_some() {
        return Err(AppError::usage(
            "provide TARGET either positionally or via --target/-t, not both".to_string(),
        ));
    }

    #[cfg(not(feature = "cli-tui"))]
    {
        let _ = (target_positional, target_flag, theme, color, ascii);
        return Err(AppError::usage(
            "tui is unavailable in this build; recompile with --features cli-tui".to_string(),
        ));
    }

    #[cfg(feature = "cli-tui")]
    {
        use target_resolver::{ResolveError, ResolveRequest, TargetResolver};

        let resolver = TargetResolver::from_environment();
        let requested = target_flag.or(target_positional);

        let resolved = match resolver.resolve(ResolveRequest {
            explicit_target: requested,
        }) {
            Ok(target) => target,
            Err(ResolveError::InvalidTarget(message)) => {
                return Err(AppError::target_invalid(message));
            }
            Err(ResolveError::Unresolved(message)) => {
                if !launcher::stdio_is_tty() {
                    return Err(AppError::target_unresolved_non_tty(message));
                }

                match launcher::prompt_for_target(&resolver)
                    .map_err(|err| AppError::internal(format!("launcher prompt failed: {}", err)))?
                {
                    Some(target) => target,
                    None => {
                        return Ok(ok_cli_envelope(Some(json!({
                            "mode": "tui",
                            "action": "cancelled"
                        }))));
                    }
                }
            }
        };

        run_resolved_tui(
            &resolver,
            resolved,
            report::TuiOptions {
                theme,
                color,
                ascii,
            },
        )
    }
}

#[cfg(feature = "cli-tui")]
fn run_resolved_tui(
    resolver: &target_resolver::TargetResolver,
    resolved: target_resolver::ResolvedTarget,
    options: report::TuiOptions,
) -> Result<JsonEnvelope, AppError> {
    report::run_tui_from_path(&resolved.ndjson_path, options)?;

    if let Err(err) = resolver.persist_last_target(&resolved.target) {
        eprintln!("warning: failed to persist last_target: {}", err);
    }
    let theme_name = match options.theme {
        report::ReportTheme::Classic => "classic",
        report::ReportTheme::Term => "term",
        report::ReportTheme::Cyber => "cyber",
    };
    if let Err(err) = resolver.persist_last_theme(theme_name) {
        eprintln!("warning: failed to persist last_theme: {}", err);
    }

    Ok(ok_cli_envelope(Some(json!({
        "mode": "tui",
        "source": resolved.source.as_str(),
        "target": resolved.target.display().to_string(),
        "input": resolved.ndjson_path.display().to_string(),
    }))))
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
    let output = embedder.embed(&text).map_err(map_embed_error)?;
    let dim = output.vector.len();

    if !json {
        print_vector_summary(&output.vector);
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
            "vector": output.vector,
            "truncated": output.truncated,
            "seq_len": output.seq_len,
            "max_seq_len": output.max_seq_len,
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
    let measurement_cfg = default_measurement_config();
    let measurement_hash_hex = measurement_hash(&measurement_cfg);
    debug_assert_eq!(measurement_hash(&measurement_cfg), measurement_hash_hex);
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
            let embedded = embedder
                .embed(sentence)
                .map_err(|err| format!("embed failure: {:?}", err))?;
            Ok(SentenceEmbedding {
                vector: embedded.vector,
                truncated: embedded.truncated,
                seq_len: embedded.seq_len,
                max_seq_len: embedded.max_seq_len,
            })
        },
        &query,
        &context,
        &answer,
        &measurement_cfg,
    )
    .map_err(map_measure_error)
    .map_err(|err| err.with_inputs_hash(inputs_hash_hex.clone()))?;
    let warnings = measurement.warnings.clone();
    let policy = default_policy_config();
    let diagnose_result = diagnose_eval(measurement, &policy);

    if !json {
        print_eval_report(diagnose_result.status, &diagnose_result.report);
    }

    Ok(JsonEnvelope {
        status: verdict_status_str(diagnose_result.status).to_string(),
        error: None,
        audit_trace: audit_trace_with_model_and_inputs_cfg(
            model_trace_from_verify(manager.spec(), &verify_report),
            Some(inputs_hash_hex),
            &measurement_cfg,
            Some(measurement_hash_hex),
            warnings,
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
    let measurement_cfg = default_measurement_config();
    audit_trace_with_model_and_inputs_cfg(model, inputs_hash, &measurement_cfg, None, Vec::new())
}

fn audit_trace_with_model_and_inputs_cfg(
    model: ModelTrace,
    inputs_hash: Option<String>,
    measurement_cfg: &MeasurementConfig,
    precomputed_measurement_hash: Option<String>,
    warnings: Vec<AuditWarning>,
) -> AuditTrace {
    let measurement =
        precomputed_measurement_hash.unwrap_or_else(|| measurement_hash(measurement_cfg));
    let policy = policy_hash(&default_policy_config());

    AuditTrace {
        model,
        hashes: HashesTrace {
            measurement_hash: measurement,
            policy_hash: policy,
            inputs_hash: inputs_hash.unwrap_or_else(|| "UNAVAILABLE".to_string()),
        },
        warnings,
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
