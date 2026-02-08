use crate::{audit_trace_with_model, default_model_trace, AppError, CalibrateMethod, JsonEnvelope};
use serde::Serialize;
use serde_json::{json, Value};
use std::fs;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const METHOD_QUANTILE_NEAREST_RANK: &str = "quantile_nearest_rank";

pub(super) struct CalibrateCommand {
    pub input: String,
    pub method: CalibrateMethod,
    pub hazy_q: f64,
    pub delirium_q: f64,
    pub min_rows: usize,
    pub out: Option<PathBuf>,
}

#[derive(Serialize)]
struct CalibrationData {
    method: &'static str,
    hazy_q: f64,
    delirium_q: f64,
    rows_total: u64,
    rows_used: u64,
    rows_err: u64,
    min_ratio: f64,
    max_ratio: f64,
    th_ratio_hazy: f64,
    th_ratio_delirium: f64,
}

pub(super) fn run(command: CalibrateCommand, json_output: bool) -> Result<JsonEnvelope, AppError> {
    let CalibrateCommand {
        input,
        method,
        hazy_q,
        delirium_q,
        min_rows,
        out,
    } = command;

    if !hazy_q.is_finite() || !(0.0..=1.0).contains(&hazy_q) {
        return Err(AppError::usage(
            "--hazy-q must be a finite number in [0,1]".to_string(),
        ));
    }
    if !delirium_q.is_finite() || !(0.0..=1.0).contains(&delirium_q) {
        return Err(AppError::usage(
            "--delirium-q must be a finite number in [0,1]".to_string(),
        ));
    }

    let method_name = match method {
        CalibrateMethod::Quantile => METHOD_QUANTILE_NEAREST_RANK,
    };

    let ratios = collect_usable_ratios(Path::new(&input))?;
    let rows_total = ratios.rows_total;
    let mut values = ratios.ratios;
    values.sort_by(|left, right| left.total_cmp(right));

    let rows_used = values.len();
    let rows_err = rows_total.saturating_sub(rows_used);
    if rows_used < min_rows {
        let details = json!({
            "rows_total": usize_to_u64(rows_total),
            "rows_used": usize_to_u64(rows_used),
            "rows_err": usize_to_u64(rows_err),
            "min_rows": usize_to_u64(min_rows),
        });
        return Err(AppError::calibration_insufficient_data(format!(
            "calibration needs at least {} usable rows, found {}",
            min_rows, rows_used
        ))
        .with_details_and_data(details));
    }

    let min_ratio = values[0];
    let max_ratio = values[values.len() - 1];
    let th_ratio_hazy = quantile_nearest_rank(&values, hazy_q);
    let mut th_ratio_delirium = quantile_nearest_rank(&values, delirium_q);
    if th_ratio_delirium < th_ratio_hazy {
        th_ratio_delirium = th_ratio_hazy + 1e-6_f64;
    }

    let data = CalibrationData {
        method: method_name,
        hazy_q,
        delirium_q,
        rows_total: usize_to_u64(rows_total),
        rows_used: usize_to_u64(rows_used),
        rows_err: usize_to_u64(rows_err),
        min_ratio,
        max_ratio,
        th_ratio_hazy,
        th_ratio_delirium,
    };

    let snippet = snippet_from_data(&data);
    if let Some(path) = out {
        fs::write(&path, snippet.as_bytes()).map_err(|err| {
            AppError::dependency(format!(
                "failed to write calibration snippet {}: {}",
                path.display(),
                err
            ))
        })?;
    }
    if !json_output {
        print!("{snippet}");
    }

    Ok(JsonEnvelope {
        status: "OK".to_string(),
        error: None,
        audit_trace: audit_trace_with_model(default_model_trace()),
        data: Some(
            serde_json::to_value(&data)
                .map_err(|err| AppError::internal(format!("failed to serialize data: {}", err)))?,
        ),
    })
}

struct RatioScan {
    rows_total: usize,
    ratios: Vec<f64>,
}

fn collect_usable_ratios(path: &Path) -> Result<RatioScan, AppError> {
    let file = File::open(path).map_err(|err| {
        AppError::dependency(format!(
            "failed to open calibration input {}: {}",
            path.display(),
            err
        ))
    })?;
    let mut reader = BufReader::new(file);
    let mut rows_total = 0usize;
    let mut ratios = Vec::new();
    let mut line = String::new();
    let mut saw_first_non_empty_line = false;

    loop {
        line.clear();
        let read = reader.read_line(&mut line).map_err(|err| {
            AppError::dependency(format!("failed to read calibration input: {}", err))
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

        rows_total = rows_total.saturating_add(1);

        let parsed: Value = match serde_json::from_str(&line) {
            Ok(value) => value,
            Err(_) => continue,
        };
        let Some(row) = parsed.as_object() else {
            continue;
        };

        if !matches!(row.get("error"), Some(value) if value.is_null()) {
            continue;
        }

        let Some(ratio) = row
            .get("data")
            .and_then(|value| value.get("scores"))
            .and_then(|value| value.get("max_score_ratio"))
            .and_then(Value::as_f64)
        else {
            continue;
        };

        if ratio.is_finite() {
            ratios.push(ratio);
        }
    }

    Ok(RatioScan { rows_total, ratios })
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

// Nearest-rank quantile: rank = ceil(q * n), clamped to [1, n], then pick sorted[rank - 1].
fn quantile_nearest_rank(sorted_values: &[f64], q: f64) -> f64 {
    let n = sorted_values.len();
    debug_assert!(n > 0, "quantile_nearest_rank requires non-empty input");
    debug_assert!((0.0..=1.0).contains(&q), "q must be in [0,1]");

    if n == 1 {
        return sorted_values[0];
    }

    let rank_f64 = (q * n as f64).ceil();
    let mut rank = if rank_f64 < 1.0_f64 {
        1usize
    } else if rank_f64 > n as f64 {
        n
    } else {
        rank_f64 as usize
    };
    if rank == 0 {
        rank = 1;
    }
    if rank > n {
        rank = n;
    }

    sorted_values[rank - 1]
}

fn snippet_from_data(data: &CalibrationData) -> String {
    format!(
        concat!(
            "---\n",
            "policy:\n",
            "  th_ratio_hazy: {}\n",
            "  th_ratio_delirium: {}\n",
            "calibration:\n",
            "  method: {}\n",
            "  hazy_q: {}\n",
            "  delirium_q: {}\n",
            "  rows_total: {}\n",
            "  rows_used: {}\n",
            "  rows_err: {}\n",
            "  min_ratio: {}\n",
            "  max_ratio: {}\n",
            "---\n"
        ),
        fmt_f64(data.th_ratio_hazy),
        fmt_f64(data.th_ratio_delirium),
        data.method,
        fmt_f64(data.hazy_q),
        fmt_f64(data.delirium_q),
        data.rows_total,
        data.rows_used,
        data.rows_err,
        fmt_f64(data.min_ratio),
        fmt_f64(data.max_ratio),
    )
}

fn fmt_f64(value: f64) -> String {
    let mut formatted = format!("{:.6}", value);
    while formatted.contains('.') && formatted.ends_with('0') {
        formatted.pop();
    }
    if formatted.ends_with('.') {
        formatted.push('0');
    }
    formatted
}

fn usize_to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests {
    use super::quantile_nearest_rank;

    #[test]
    fn quantile_nearest_rank_is_deterministic() {
        let sorted = [1.0_f64, 2.0_f64, 10.0_f64, 20.0_f64];

        assert_eq!(quantile_nearest_rank(&sorted, 0.0), 1.0);
        assert_eq!(quantile_nearest_rank(&sorted, 0.25), 1.0);
        assert_eq!(quantile_nearest_rank(&sorted, 0.50), 2.0);
        assert_eq!(quantile_nearest_rank(&sorted, 0.51), 10.0);
        assert_eq!(quantile_nearest_rank(&sorted, 0.90), 20.0);
        assert_eq!(quantile_nearest_rank(&sorted, 1.0), 20.0);
    }
}
