use crate::{
    audit_trace_with_model_and_inputs, audit_trace_with_model_and_inputs_cfg, map_embed_error,
    map_measure_error, model_trace_from_spec, model_trace_from_verify, verdict_status_str,
    AppError, BatchFormat, ErrorEnvelope, JsonEnvelope,
};
use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use indicatif::{ProgressBar, ProgressStyle};
use pale_ale_diagnose::{
    compute_inputs_hash, default_measurement_config, default_policy_config, diagnose_eval,
    measure_eval, measurement_hash, AuditTrace, AuditWarning, EvalReport, MeasurementConfig,
    ModelTrace, SentenceEmbedding,
};
use pale_ale_embed::{Embedder, ModelManager, ModelSpec};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

const DEFAULT_BATCH_OUT_PATH: &str = "pale-ale.batch.ndjson";
const ROW_WORK_QUEUE: usize = 256;
const WORST_K_LIMIT: usize = 10;

pub(super) struct BatchCommand {
    pub input: String,
    pub out: Option<PathBuf>,
    pub format: BatchFormat,
    pub threads: Option<usize>,
    pub max_rows: Option<usize>,
    pub offline: bool,
    pub json_output: bool,
    pub strict: bool,
    pub dry_run: bool,
}

#[derive(Clone, Debug)]
struct BatchTask {
    row_index: usize,
    id: Option<String>,
    query: String,
    context: String,
    answer: String,
    inputs_hash: String,
}

struct ParseFailure {
    id: Option<String>,
    inputs_hash: String,
    error: ErrorEnvelope,
}

#[derive(Clone, Debug)]
struct WorkerConfig {
    dry_run: bool,
    model_spec: ModelSpec,
    model_trace: ModelTrace,
    measurement_cfg: MeasurementConfig,
    measurement_hash: String,
}

struct WorkerState {
    config: WorkerConfig,
    embedder: Option<Embedder>,
}

#[derive(Serialize)]
struct BatchRowResult {
    row_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    inputs_hash: String,
    status: String,
    error: Option<ErrorEnvelope>,
    data: Option<EvalReport>,
    audit_trace: AuditTrace,
    #[serde(skip_serializing)]
    max_score_ratio: Option<f32>,
}

#[derive(Serialize)]
struct BatchSummary {
    input_path: String,
    out_path: String,
    rows_total: usize,
    rows_ok: usize,
    rows_err: usize,
    duration_ms: u64,
    worst_k: Vec<BatchWorstRow>,
}

#[derive(Serialize)]
struct BatchWorstRow {
    row_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    inputs_hash: String,
    max_score_ratio: f32,
    status: String,
}

struct WriterStats {
    rows_total: usize,
    rows_ok: usize,
    rows_err: usize,
    worst_k: Vec<BatchWorstRow>,
}

#[derive(Deserialize)]
struct JsonlRow {
    id: Option<String>,
    query: Option<String>,
    context: Option<String>,
    answer: Option<String>,
}

pub(super) fn run(command: BatchCommand) -> Result<JsonEnvelope, AppError> {
    let BatchCommand {
        input,
        out,
        format,
        threads,
        max_rows,
        offline,
        json_output,
        strict,
        dry_run,
    } = command;

    if matches!(threads, Some(0)) {
        return Err(AppError::usage("--threads must be >= 1".to_string()));
    }
    if matches!(max_rows, Some(0)) {
        return Err(AppError::usage("--max-rows must be >= 1".to_string()));
    }

    let started = Instant::now();
    let out_path = out.unwrap_or_else(|| PathBuf::from(DEFAULT_BATCH_OUT_PATH));
    let out_path_display = out_path.display().to_string();
    let measurement_cfg = default_measurement_config();
    let measurement_hash_hex = measurement_hash(&measurement_cfg);

    let manager = ModelManager::new(ModelSpec::classic());
    let model_trace = prepare_model_trace(&manager, offline, dry_run)?;
    let worker_config = WorkerConfig {
        dry_run,
        model_spec: manager.spec().clone(),
        model_trace: model_trace.clone(),
        measurement_cfg: measurement_cfg.clone(),
        measurement_hash: measurement_hash_hex.clone(),
    };

    let progress = build_progress_bar(&input, max_rows)?;
    let out_file = File::create(&out_path).map_err(|err| {
        AppError::dependency(format!(
            "failed to open output file {}: {}",
            out_path.display(),
            err
        ))
    })?;

    let pool = build_thread_pool(threads)?;
    let worker_count = pool.current_num_threads().max(1);

    let (task_tx, task_rx) = bounded::<BatchTask>(ROW_WORK_QUEUE);
    let (result_tx, result_rx) = unbounded::<(usize, BatchRowResult)>();

    let writer_handle = thread::spawn(move || write_rows_in_order(out_file, result_rx));

    let mut dispatch_error: Option<AppError> = None;
    pool.scope(|scope| {
        for _ in 0..worker_count {
            let worker_task_rx = task_rx.clone();
            let worker_result_tx = result_tx.clone();
            let worker_progress = progress.clone();
            let config = worker_config.clone();
            scope.spawn(move |_| {
                run_worker(worker_task_rx, worker_result_tx, worker_progress, config);
            });
        }

        let dispatch_result = if input == "-" {
            let stdin = io::stdin();
            let mut locked = stdin.lock();
            dispatch_lines(
                &mut locked,
                format,
                max_rows,
                &task_tx,
                &result_tx,
                &model_trace,
                &progress,
            )
        } else {
            let file = match File::open(&input) {
                Ok(file) => file,
                Err(err) => {
                    dispatch_error = Some(AppError::dependency(format!(
                        "failed to open input file {}: {}",
                        input, err
                    )));
                    return;
                }
            };
            let mut reader = BufReader::new(file);
            dispatch_lines(
                &mut reader,
                format,
                max_rows,
                &task_tx,
                &result_tx,
                &model_trace,
                &progress,
            )
        };

        if let Err(err) = dispatch_result {
            dispatch_error = Some(err);
        }
        drop(task_tx);
    });

    drop(result_tx);

    let writer_result = writer_handle
        .join()
        .map_err(|_| AppError::internal("batch writer thread panicked".to_string()))?;
    let writer_stats = writer_result?;

    finish_progress(&progress, writer_stats.rows_total);

    if let Some(err) = dispatch_error {
        return Err(err);
    }

    let duration_ms = duration_ms(started);
    let summary = BatchSummary {
        input_path: input.clone(),
        out_path: out_path_display,
        rows_total: writer_stats.rows_total,
        rows_ok: writer_stats.rows_ok,
        rows_err: writer_stats.rows_err,
        duration_ms,
        worst_k: writer_stats.worst_k,
    };

    if !json_output {
        print_batch_summary(&summary);
    }

    if strict && summary.rows_err > 0 {
        return Err(AppError::strict_failure(format!(
            "strict mode failed: {} row(s) had errors",
            summary.rows_err
        ))
        .with_data(json!(summary)));
    }

    Ok(JsonEnvelope {
        status: "OK".to_string(),
        error: None,
        audit_trace: audit_trace_with_model_and_inputs_cfg(
            model_trace,
            None,
            &measurement_cfg,
            Some(measurement_hash_hex),
            Vec::new(),
        ),
        data: Some(json!(summary)),
    })
}

fn prepare_model_trace(
    manager: &ModelManager,
    offline: bool,
    dry_run: bool,
) -> Result<ModelTrace, AppError> {
    if dry_run {
        return Ok(model_trace_from_spec(manager.spec()));
    }

    if offline {
        let status = manager.status();
        if !status.cache_present || !status.missing_files.is_empty() {
            return Err(AppError::model_missing_offline(format!(
                "model cache missing in offline mode: {}",
                status.cache_dir.display()
            )));
        }
        let verify = manager.verify().map_err(map_embed_error)?;
        return Ok(model_trace_from_verify(manager.spec(), &verify));
    }

    let verify = manager.download(false).map_err(map_embed_error)?;
    Ok(model_trace_from_verify(manager.spec(), &verify))
}

fn build_thread_pool(threads: Option<usize>) -> Result<rayon::ThreadPool, AppError> {
    let builder = rayon::ThreadPoolBuilder::new();
    let builder = if let Some(n) = threads {
        builder.num_threads(n)
    } else {
        builder
    };
    builder
        .build()
        .map_err(|err| AppError::internal(format!("failed to build batch thread pool: {}", err)))
}

fn build_progress_bar(input: &str, max_rows: Option<usize>) -> Result<ProgressBar, AppError> {
    if input == "-" {
        let spinner = ProgressBar::new_spinner();
        if let Ok(style) =
            ProgressStyle::with_template("[{elapsed_precise}] {spinner} {pos} rows processed")
        {
            spinner.set_style(style.tick_chars("/|\\- "));
        }
        spinner.enable_steady_tick(Duration::from_millis(120));
        return Ok(spinner);
    }

    let total = count_lines(Path::new(input), max_rows)?;
    let bar = ProgressBar::new(total as u64);
    if let Ok(style) = ProgressStyle::with_template(
        "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} rows ({percent}%)",
    ) {
        bar.set_style(style.progress_chars("=> "));
    }
    Ok(bar)
}

fn count_lines(path: &Path, max_rows: Option<usize>) -> Result<usize, AppError> {
    let file = File::open(path).map_err(|err| {
        AppError::dependency(format!(
            "failed to open input file {}: {}",
            path.display(),
            err
        ))
    })?;
    let mut reader = BufReader::new(file);
    let mut count = 0usize;
    let mut line = String::new();
    let mut saw_first_non_empty_line = false;

    loop {
        line.clear();
        let read = reader.read_line(&mut line).map_err(|err| {
            AppError::dependency(format!(
                "failed to read input file {}: {}",
                path.display(),
                err
            ))
        })?;
        if read == 0 {
            break;
        }
        trim_line_ending(&mut line);

        if line.trim().is_empty() {
            continue;
        }

        if !saw_first_non_empty_line {
            strip_utf8_bom(&mut line);
            saw_first_non_empty_line = true;
            if line.trim().is_empty() {
                continue;
            }
        }

        if let Some(limit) = max_rows {
            if count >= limit {
                break;
            }
        }
        count = count.saturating_add(1);
    }

    Ok(count)
}

fn dispatch_lines<R: BufRead>(
    reader: &mut R,
    format: BatchFormat,
    max_rows: Option<usize>,
    task_tx: &Sender<BatchTask>,
    result_tx: &Sender<(usize, BatchRowResult)>,
    model_trace: &ModelTrace,
    progress: &ProgressBar,
) -> Result<(), AppError> {
    let mut row_index = 0usize;
    let mut line = String::new();
    let mut saw_first_non_empty_line = false;

    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .map_err(|err| AppError::dependency(format!("failed to read input line: {}", err)))?;
        if read == 0 {
            break;
        }
        trim_line_ending(&mut line);

        if line.trim().is_empty() {
            continue;
        }

        if !saw_first_non_empty_line {
            strip_utf8_bom(&mut line);
            saw_first_non_empty_line = true;
            if line.trim().is_empty() {
                continue;
            }
        }

        if let Some(limit) = max_rows {
            if row_index >= limit {
                break;
            }
        }

        match parse_input_line(row_index, &line, format) {
            Ok(task) => {
                task_tx.send(task).map_err(|_| {
                    AppError::internal("batch worker queue closed unexpectedly".to_string())
                })?;
            }
            Err(failure) => {
                let row = BatchRowResult::input_failure(
                    row_index,
                    failure.id,
                    failure.inputs_hash,
                    failure.error,
                    model_trace,
                );
                result_tx.send((row_index, row)).map_err(|_| {
                    AppError::internal("batch writer queue closed unexpectedly".to_string())
                })?;
                progress.inc(1);
            }
        }

        row_index = row_index.saturating_add(1);
    }

    Ok(())
}

fn trim_line_ending(line: &mut String) {
    while line.ends_with('\n') || line.ends_with('\r') {
        line.pop();
    }
}

fn strip_utf8_bom(line: &mut String) {
    if let Some(stripped) = line.strip_prefix('\u{feff}') {
        *line = stripped.to_string();
    }
}

fn parse_input_line(
    row_index: usize,
    line: &str,
    format: BatchFormat,
) -> Result<BatchTask, Box<ParseFailure>> {
    match format {
        BatchFormat::Jsonl => parse_jsonl_line(row_index, line),
        BatchFormat::Tsv => parse_tsv_line(row_index, line),
    }
}

fn parse_jsonl_line(row_index: usize, line: &str) -> Result<BatchTask, Box<ParseFailure>> {
    let parsed = serde_json::from_str::<JsonlRow>(line).map_err(|err| {
        Box::new(ParseFailure {
            id: None,
            inputs_hash: compute_inputs_hash(line, "", ""),
            error: ErrorEnvelope {
                code: "BATCH_INPUT_PARSE".to_string(),
                message: format!("row {} is not valid JSONL", row_index),
                details: json!({
                "format": "jsonl",
                    "reason": err.to_string(),
                }),
            },
        })
    })?;

    let missing = missing_fields(&parsed);
    if !missing.is_empty() {
        let inputs_hash = compute_inputs_hash(
            parsed.query.as_deref().unwrap_or(""),
            parsed.context.as_deref().unwrap_or(""),
            parsed.answer.as_deref().unwrap_or(""),
        );
        return Err(Box::new(ParseFailure {
            id: parsed.id,
            inputs_hash,
            error: ErrorEnvelope {
                code: "BATCH_INPUT_MISSING_FIELDS".to_string(),
                message: format!("row {} is missing required fields", row_index),
                details: json!({
                    "missing": missing,
                    "format": "jsonl",
                }),
            },
        }));
    }

    let query = parsed.query.unwrap_or_default();
    let context = parsed.context.unwrap_or_default();
    let answer = parsed.answer.unwrap_or_default();
    Ok(BatchTask {
        row_index,
        id: parsed.id,
        query: query.clone(),
        context: context.clone(),
        answer: answer.clone(),
        inputs_hash: compute_inputs_hash(&query, &context, &answer),
    })
}

fn missing_fields(row: &JsonlRow) -> Vec<&'static str> {
    let mut missing = Vec::new();
    if row.query.is_none() {
        missing.push("query");
    }
    if row.context.is_none() {
        missing.push("context");
    }
    if row.answer.is_none() {
        missing.push("answer");
    }
    missing
}

fn parse_tsv_line(row_index: usize, line: &str) -> Result<BatchTask, Box<ParseFailure>> {
    let parts: Vec<&str> = line.split('\t').collect();
    if parts.len() != 3 {
        return Err(Box::new(ParseFailure {
            id: None,
            inputs_hash: compute_inputs_hash(
                parts.first().copied().unwrap_or(""),
                parts.get(1).copied().unwrap_or(""),
                parts.get(2).copied().unwrap_or(""),
            ),
            error: ErrorEnvelope {
                code: "BATCH_INPUT_TSV_COLUMNS".to_string(),
                message: format!(
                    "row {} must contain exactly 3 tab-separated fields",
                    row_index
                ),
                details: json!({
                    "format": "tsv",
                    "columns": parts.len(),
                    "assumption": "TSV fields are not escaped; tabs inside fields are unsupported",
                }),
            },
        }));
    }

    let query = parts[0].to_string();
    let context = parts[1].to_string();
    let answer = parts[2].to_string();
    Ok(BatchTask {
        row_index,
        id: None,
        query: query.clone(),
        context: context.clone(),
        answer: answer.clone(),
        inputs_hash: compute_inputs_hash(&query, &context, &answer),
    })
}

fn run_worker(
    task_rx: Receiver<BatchTask>,
    result_tx: Sender<(usize, BatchRowResult)>,
    progress: ProgressBar,
    config: WorkerConfig,
) {
    let mut state = WorkerState::new(config);
    while let Ok(task) = task_rx.recv() {
        let row_index = task.row_index;
        let row = state.process(task);
        if result_tx.send((row_index, row)).is_err() {
            break;
        }
        progress.inc(1);
    }
}

impl WorkerState {
    fn new(config: WorkerConfig) -> Self {
        Self {
            config,
            embedder: None,
        }
    }

    fn process(&mut self, task: BatchTask) -> BatchRowResult {
        if self.config.dry_run {
            return BatchRowResult::dry_run(task, &self.config.model_trace);
        }
        let model_trace = self.config.model_trace.clone();
        let measurement_cfg = self.config.measurement_cfg.clone();
        let measurement_hash = self.config.measurement_hash.clone();

        let embedder = match self.ensure_embedder() {
            Ok(embedder) => embedder,
            Err(err) => return BatchRowResult::runtime_failure(task, err, &model_trace),
        };

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
            &task.query,
            &task.context,
            &task.answer,
            &measurement_cfg,
        )
        .map_err(map_measure_error);

        let measurement = match measurement {
            Ok(measurement) => measurement,
            Err(err) => return BatchRowResult::runtime_failure(task, err, &model_trace),
        };

        let warnings = measurement.warnings.clone();
        let policy = default_policy_config();
        let diagnosed = diagnose_eval(measurement, &policy);
        BatchRowResult::success(
            task,
            diagnosed.status,
            diagnosed.report,
            &model_trace,
            warnings,
            &measurement_cfg,
            &measurement_hash,
        )
    }

    fn ensure_embedder(&mut self) -> Result<&Embedder, AppError> {
        if self.embedder.is_none() {
            let manager = ModelManager::new(self.config.model_spec.clone());
            let embedder = Embedder::new(&manager).map_err(map_embed_error)?;
            self.embedder = Some(embedder);
        }
        self.embedder
            .as_ref()
            .ok_or_else(|| AppError::internal("failed to initialize embedder".to_string()))
    }
}

impl BatchRowResult {
    fn success(
        task: BatchTask,
        status: pale_ale_diagnose::VerdictStatus,
        report: EvalReport,
        model_trace: &ModelTrace,
        warnings: Vec<AuditWarning>,
        measurement_cfg: &MeasurementConfig,
        measurement_hash: &str,
    ) -> Self {
        let max_score_ratio = report.scores.max_score_ratio;
        let inputs_hash = task.inputs_hash;
        Self {
            row_index: task.row_index,
            id: task.id,
            inputs_hash: inputs_hash.clone(),
            status: verdict_status_str(status).to_string(),
            error: None,
            data: Some(report),
            audit_trace: audit_trace_with_model_and_inputs_cfg(
                model_trace.clone(),
                Some(inputs_hash),
                measurement_cfg,
                Some(measurement_hash.to_string()),
                warnings,
            ),
            max_score_ratio: Some(max_score_ratio),
        }
    }

    fn dry_run(task: BatchTask, model_trace: &ModelTrace) -> Self {
        let inputs_hash = task.inputs_hash;
        Self {
            row_index: task.row_index,
            id: task.id,
            inputs_hash: inputs_hash.clone(),
            status: "UNKNOWN".to_string(),
            error: None,
            data: None,
            audit_trace: audit_trace_with_model_and_inputs(model_trace.clone(), Some(inputs_hash)),
            max_score_ratio: None,
        }
    }

    fn runtime_failure(task: BatchTask, err: AppError, model_trace: &ModelTrace) -> Self {
        let inputs_hash = task.inputs_hash;
        let details = (*err.details).clone();
        Self {
            row_index: task.row_index,
            id: task.id,
            inputs_hash: inputs_hash.clone(),
            status: "UNKNOWN".to_string(),
            error: Some(ErrorEnvelope {
                code: err.code.to_string(),
                message: err.message,
                details,
            }),
            data: None,
            audit_trace: audit_trace_with_model_and_inputs(model_trace.clone(), Some(inputs_hash)),
            max_score_ratio: None,
        }
    }

    fn input_failure(
        row_index: usize,
        id: Option<String>,
        inputs_hash: String,
        error: ErrorEnvelope,
        model_trace: &ModelTrace,
    ) -> Self {
        Self {
            row_index,
            id,
            inputs_hash: inputs_hash.clone(),
            status: "UNKNOWN".to_string(),
            error: Some(error),
            data: None,
            audit_trace: audit_trace_with_model_and_inputs(model_trace.clone(), Some(inputs_hash)),
            max_score_ratio: None,
        }
    }
}

fn write_rows_in_order(
    out_file: File,
    result_rx: Receiver<(usize, BatchRowResult)>,
) -> Result<WriterStats, AppError> {
    let mut writer = BufWriter::new(out_file);
    let mut next_expected = 0usize;
    let mut buffer = BTreeMap::<usize, BatchRowResult>::new();
    let mut rows_total = 0usize;
    let mut rows_ok = 0usize;
    let mut rows_err = 0usize;
    let mut worst_k = Vec::<BatchWorstRow>::new();

    while let Ok((row_index, row)) = result_rx.recv() {
        buffer.insert(row_index, row);
        while let Some(row) = buffer.remove(&next_expected) {
            let row_has_error = row.error.is_some();
            if row_has_error {
                rows_err = rows_err.saturating_add(1);
            } else {
                rows_ok = rows_ok.saturating_add(1);
            }

            if let Some(max_score_ratio) = row.max_score_ratio {
                worst_k.push(BatchWorstRow {
                    row_index: row.row_index,
                    id: row.id.clone(),
                    inputs_hash: row.inputs_hash.clone(),
                    max_score_ratio,
                    status: row.status.clone(),
                });
            }

            serde_json::to_writer(&mut writer, &row).map_err(|err| {
                AppError::internal(format!(
                    "failed to serialize batch row {}: {}",
                    row_index, err
                ))
            })?;
            writer.write_all(b"\n").map_err(|err| {
                AppError::dependency(format!(
                    "failed to write batch output row {}: {}",
                    row_index, err
                ))
            })?;

            rows_total = rows_total.saturating_add(1);
            next_expected = next_expected.saturating_add(1);
        }
    }

    if !buffer.is_empty() {
        return Err(AppError::internal(
            "writer stopped before all rows were flushed".to_string(),
        ));
    }

    writer
        .flush()
        .map_err(|err| AppError::dependency(format!("failed to flush batch output: {}", err)))?;

    worst_k.sort_by(|left, right| {
        right
            .max_score_ratio
            .total_cmp(&left.max_score_ratio)
            .then_with(|| left.row_index.cmp(&right.row_index))
            .then_with(|| left.inputs_hash.cmp(&right.inputs_hash))
    });
    worst_k.truncate(WORST_K_LIMIT);

    Ok(WriterStats {
        rows_total,
        rows_ok,
        rows_err,
        worst_k,
    })
}

fn finish_progress(progress: &ProgressBar, rows_total: usize) {
    progress.set_position(rows_total as u64);
    progress.finish_with_message(format!("processed {} rows", rows_total));
}

fn duration_ms(started: Instant) -> u64 {
    let elapsed = started.elapsed();
    let millis = elapsed.as_millis();
    if millis > u128::from(u64::MAX) {
        u64::MAX
    } else {
        millis as u64
    }
}

fn print_batch_summary(summary: &BatchSummary) {
    println!(
        "batch complete: total={} ok={} err={} out={}",
        summary.rows_total, summary.rows_ok, summary.rows_err, summary.out_path
    );
    if summary.worst_k.is_empty() {
        return;
    }
    println!(
        "worst_k (top {} by max_score_ratio):",
        summary.worst_k.len()
    );
    for item in &summary.worst_k {
        if let Some(id) = &item.id {
            println!(
                "row={} id={} ratio={:.6} status={} hash={}",
                item.row_index, id, item.max_score_ratio, item.status, item.inputs_hash
            );
        } else {
            println!(
                "row={} ratio={:.6} status={} hash={}",
                item.row_index, item.max_score_ratio, item.status, item.inputs_hash
            );
        }
    }
}
