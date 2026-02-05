use clap::{Parser, Subcommand};
use pale_ale_diagnose::{
    default_measurement_config, default_policy_config, measurement_hash, policy_hash,
    AttestationLevel, AuditTrace, ConfigSource, HashesTrace, ModelTrace,
};
use serde::Serialize;
use serde_json::{json, Value};
use std::env;
use std::path::PathBuf;

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
    Eval {
        query: String,
        context: String,
        answer: String,
    },
}

#[derive(Subcommand, Debug)]
enum ModelCommand {
    Status,
    Download,
    Verify,
    Path,
}

#[allow(dead_code)]
#[derive(Clone, Copy, Debug)]
enum AppErrorKind {
    Usage,
    NotImplemented,
    OfflineMode,
    ModelMissingOffline,
    Dependency,
    Internal,
}

#[derive(Clone, Debug)]
struct AppError {
    kind: AppErrorKind,
    code: &'static str,
    message: String,
}

impl AppError {
    fn usage(message: String) -> Self {
        Self {
            kind: AppErrorKind::Usage,
            code: "CLI_USAGE",
            message,
        }
    }

    fn not_implemented(message: String) -> Self {
        Self {
            kind: AppErrorKind::NotImplemented,
            code: "NOT_IMPLEMENTED",
            message,
        }
    }

    fn offline_mode(message: String) -> Self {
        Self {
            kind: AppErrorKind::OfflineMode,
            code: "OFFLINE_MODE",
            message,
        }
    }

    fn model_missing_offline(message: String) -> Self {
        Self {
            kind: AppErrorKind::ModelMissingOffline,
            code: "MODEL_MISSING_OFFLINE",
            message,
        }
    }

    fn exit_code(&self) -> i32 {
        match self.kind {
            AppErrorKind::Usage => 1,
            AppErrorKind::OfflineMode | AppErrorKind::ModelMissingOffline | AppErrorKind::Dependency => 2,
            AppErrorKind::NotImplemented | AppErrorKind::Internal => 3,
        }
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
            match run(cli) {
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
        Err(err) => {
            if wants_json {
                let usage = AppError::usage(err.to_string());
                let envelope = error_envelope(&usage);
                print_json(&envelope);
            } else {
                let _ = err.print();
            }
            std::process::exit(1);
        }
    }
}

fn run(cli: Cli) -> Result<JsonEnvelope, AppError> {
    match cli.command {
        Commands::Doctor => doctor(cli.offline),
        Commands::Model { command } => model(command, cli.offline),
        Commands::Eval {
            query: _,
            context: _,
            answer: _,
        } => eval(cli.offline),
    }
}

fn doctor(_offline: bool) -> Result<JsonEnvelope, AppError> {
    Ok(JsonEnvelope {
        status: "OK".to_string(),
        error: None,
        audit_trace: stub_audit_trace(),
        data: Some(json!({ "mode": "doctor", "offline": _offline })),
    })
}

fn model(command: ModelCommand, offline: bool) -> Result<JsonEnvelope, AppError> {
    match command {
        ModelCommand::Download => {
            if offline {
                return Err(AppError::offline_mode(
                    "offline mode forbids model download".to_string(),
                ));
            }
            Err(AppError::not_implemented(
                "model download not implemented".to_string(),
            ))
        }
        ModelCommand::Verify => Err(AppError::not_implemented(
            "model verify not implemented".to_string(),
        )),
        ModelCommand::Path => Ok(JsonEnvelope {
            status: "OK".to_string(),
            error: None,
            audit_trace: stub_audit_trace(),
            data: Some(json!({ "path": model_dir().to_string_lossy() })),
        }),
        ModelCommand::Status => {
            let present = model_cache_present();
            Ok(JsonEnvelope {
                status: "OK".to_string(),
                error: None,
                audit_trace: stub_audit_trace(),
                data: Some(json!({
                    "cache_present": present,
                    "note": "status check is stubbed; missing is non-fatal in PR2"
                })),
            })
        }
    }
}

fn eval(offline: bool) -> Result<JsonEnvelope, AppError> {
    if offline && !model_cache_present() {
        return Err(AppError::model_missing_offline(
            "model cache missing in offline mode".to_string(),
        ));
    }
    Err(AppError::not_implemented("eval not implemented".to_string()))
}

fn model_dir() -> PathBuf {
    if let Ok(path) = env::var("PA_MODEL_DIR") {
        return PathBuf::from(path);
    }
    PathBuf::from("models")
}

fn model_cache_present() -> bool {
    model_dir().exists()
}

fn stub_audit_trace() -> AuditTrace {
    let measurement = measurement_hash(&default_measurement_config());
    let policy = policy_hash(&default_policy_config());

    AuditTrace {
        model: ModelTrace {
            model_id: "UNAVAILABLE".to_string(),
            revision: "UNAVAILABLE".to_string(),
            files: Vec::new(),
        },
        hashes: HashesTrace {
            measurement_hash: measurement,
            policy_hash: policy,
            inputs_hash: "UNAVAILABLE".to_string(),
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
            details: Value::Null,
        }),
        audit_trace: stub_audit_trace(),
        data: None,
    }
}

fn print_json(envelope: &JsonEnvelope) {
    let json = serde_json::to_string(envelope).expect("failed to serialize json");
    println!("{json}");
}
