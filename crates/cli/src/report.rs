use crate::{audit_trace_with_model, default_model_trace, AppError, JsonEnvelope};
use serde::Serialize;
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

pub(super) struct ReportCommand {
    pub input: String,
    pub summary: bool,
    pub top: usize,
    pub filters: Vec<String>,
    pub find: Option<String>,
    pub tui: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ReportStatus {
    Lucid,
    Hazy,
    Delirium,
    Unknown,
}

impl ReportStatus {
    fn from_row(value: Option<&str>) -> Self {
        match value {
            Some("LUCID") => Self::Lucid,
            Some("HAZY") => Self::Hazy,
            Some("DELIRIUM") => Self::Delirium,
            _ => Self::Unknown,
        }
    }

    fn from_filter(value: &str) -> Option<Self> {
        match value.to_ascii_uppercase().as_str() {
            "LUCID" => Some(Self::Lucid),
            "HAZY" => Some(Self::Hazy),
            "DELIRIUM" => Some(Self::Delirium),
            "UNKNOWN" => Some(Self::Unknown),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Lucid => "LUCID",
            Self::Hazy => "HAZY",
            Self::Delirium => "DELIRIUM",
            Self::Unknown => "UNKNOWN",
        }
    }
}

#[derive(Clone, Debug, Default)]
struct ReportFilters {
    status: Option<ReportStatus>,
    has_warning: bool,
    has_error: bool,
    find: Option<String>,
}

#[derive(Clone, Debug)]
struct ReportError {
    code: Option<String>,
    #[cfg(feature = "cli-tui")]
    message: Option<String>,
}

#[derive(Clone, Debug)]
struct ReportRow {
    row_index: usize,
    id: Option<String>,
    inputs_hash: String,
    status: ReportStatus,
    error: Option<ReportError>,
    warnings: Vec<Value>,
    max_score_ratio: Option<f64>,
    #[cfg(feature = "cli-tui")]
    raw: Value,
}

#[derive(Serialize)]
struct ReportSummary {
    input_path: String,
    rows_total: usize,
    rows_ok: usize,
    rows_err: usize,
    counts_by_status: StatusCounts,
    rows_with_warnings: usize,
    worst_k: Vec<ReportWorstRow>,
}

#[derive(Default, Serialize)]
struct StatusCounts {
    #[serde(rename = "LUCID")]
    lucid: usize,
    #[serde(rename = "HAZY")]
    hazy: usize,
    #[serde(rename = "DELIRIUM")]
    delirium: usize,
    #[serde(rename = "UNKNOWN")]
    unknown: usize,
}

#[derive(Clone, Serialize)]
struct ReportWorstRow {
    row_index: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    id: Option<String>,
    status: String,
    inputs_hash: String,
    max_score_ratio: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    err_code: Option<String>,
}

pub(super) fn run(command: ReportCommand, json_output: bool) -> Result<JsonEnvelope, AppError> {
    let summary_enabled = command.summary || !command.tui;
    if json_output && !summary_enabled {
        return Err(AppError::usage(
            "--json requires --summary when --tui is used".to_string(),
        ));
    }
    if command.top == 0 {
        return Err(AppError::usage("--top must be >= 1".to_string()));
    }
    if command.tui {
        #[cfg(not(feature = "cli-tui"))]
        {
            return Err(AppError::usage(
                "--tui is unavailable in this build; recompile with --features cli-tui".to_string(),
            ));
        }
    }

    let filters = parse_filters(&command.filters, command.find)?;
    let rows = read_rows_from_path(Path::new(&command.input))?;
    let filtered_rows = filtered_rows(&rows, &filters);
    let summary = summarize_rows(&command.input, &filtered_rows, command.top);

    #[cfg(feature = "cli-tui")]
    let mut print_summary_mode = !json_output;
    #[cfg(not(feature = "cli-tui"))]
    let print_summary_mode = !json_output;
    if command.tui && !json_output {
        #[cfg(feature = "cli-tui")]
        {
            match tui::run(rows, filters) {
                Ok(()) => {
                    print_summary_mode = false;
                }
                Err(tui::TuiError::Init(message)) => {
                    eprintln!(
                        "report: failed to initialize TUI ({}), falling back to --summary",
                        message
                    );
                }
                Err(tui::TuiError::Runtime(message)) => {
                    return Err(AppError::internal(format!(
                        "report TUI failed: {}",
                        message
                    )));
                }
            }
        }
    }

    if print_summary_mode {
        print_summary(&summary);
    }

    let data = serde_json::to_value(&summary).map_err(|err| {
        AppError::internal(format!("failed to serialize report summary: {}", err))
    })?;

    Ok(JsonEnvelope {
        status: "OK".to_string(),
        error: None,
        audit_trace: audit_trace_with_model(default_model_trace()),
        data: Some(data),
    })
}

fn parse_filters(raw_filters: &[String], find: Option<String>) -> Result<ReportFilters, AppError> {
    let mut filters = ReportFilters {
        find,
        ..ReportFilters::default()
    };

    for raw in raw_filters {
        if raw.eq_ignore_ascii_case("has_warning") {
            filters.has_warning = true;
            continue;
        }
        if raw.eq_ignore_ascii_case("has_error") {
            filters.has_error = true;
            continue;
        }

        if let Some((key, value)) = raw.split_once('=') {
            if key.eq_ignore_ascii_case("status") {
                let parsed = ReportStatus::from_filter(value).ok_or_else(|| {
                    AppError::usage(format!(
                        "invalid --filter status value: {} (expected LUCID|HAZY|DELIRIUM|UNKNOWN)",
                        value
                    ))
                })?;
                if let Some(existing) = filters.status {
                    if existing != parsed {
                        return Err(AppError::usage(
                            "multiple status filters are not supported".to_string(),
                        ));
                    }
                } else {
                    filters.status = Some(parsed);
                }
                continue;
            }
        }

        return Err(AppError::usage(format!(
            "invalid --filter value: {} (allowed: status=<LUCID|HAZY|DELIRIUM|UNKNOWN>, has_warning, has_error)",
            raw
        )));
    }

    Ok(filters)
}

fn read_rows_from_path(path: &Path) -> Result<Vec<ReportRow>, AppError> {
    let file = File::open(path).map_err(|err| {
        AppError::dependency(format!(
            "failed to open report input {}: {}",
            path.display(),
            err
        ))
    })?;

    let mut reader = BufReader::new(file);
    read_rows_from_reader(&mut reader)
}

fn read_rows_from_reader<R: BufRead>(reader: &mut R) -> Result<Vec<ReportRow>, AppError> {
    let mut rows = Vec::new();
    let mut line = String::new();
    let mut line_no = 0usize;
    let mut saw_first_non_empty_line = false;

    loop {
        line.clear();
        let read = reader
            .read_line(&mut line)
            .map_err(|err| AppError::dependency(format!("failed to read report input: {}", err)))?;
        if read == 0 {
            break;
        }

        line_no = line_no.saturating_add(1);
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

        let parsed: Value = serde_json::from_str(&line).map_err(|err| {
            AppError::dependency(format!(
                "failed to parse NDJSON row at line {}: {}",
                line_no, err
            ))
        })?;

        if !parsed.is_object() {
            return Err(AppError::dependency(format!(
                "NDJSON row at line {} must be a JSON object",
                line_no
            )));
        }

        let fallback_row_index = rows.len();
        rows.push(ReportRow::from_value(parsed, fallback_row_index));
    }

    Ok(rows)
}

impl ReportRow {
    fn from_value(value: Value, fallback_row_index: usize) -> Self {
        let row_index = value
            .get("row_index")
            .and_then(value_to_usize)
            .unwrap_or(fallback_row_index);
        let id = value
            .get("id")
            .and_then(Value::as_str)
            .map(|s| s.to_string());
        let inputs_hash = value
            .get("inputs_hash")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let status = ReportStatus::from_row(value.get("status").and_then(Value::as_str));

        let error = match value.get("error") {
            Some(error_value) if !error_value.is_null() => Some(ReportError {
                code: error_value
                    .get("code")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
                #[cfg(feature = "cli-tui")]
                message: error_value
                    .get("message")
                    .and_then(Value::as_str)
                    .map(|s| s.to_string()),
            }),
            _ => None,
        };

        let warnings = value
            .get("audit_trace")
            .and_then(|audit| audit.get("warnings"))
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();

        let max_score_ratio = value
            .get("data")
            .and_then(|data| data.get("scores"))
            .and_then(|scores| scores.get("max_score_ratio"))
            .and_then(Value::as_f64);

        Self {
            row_index,
            id,
            inputs_hash,
            status,
            error,
            warnings,
            max_score_ratio,
            #[cfg(feature = "cli-tui")]
            raw: value,
        }
    }
}

fn value_to_usize(value: &Value) -> Option<usize> {
    let as_u64 = value.as_u64()?;
    usize::try_from(as_u64).ok()
}

fn filtered_rows<'a>(rows: &'a [ReportRow], filters: &ReportFilters) -> Vec<&'a ReportRow> {
    rows.iter()
        .filter(|row| row_matches_filters(row, filters))
        .collect()
}

fn row_matches_filters(row: &ReportRow, filters: &ReportFilters) -> bool {
    if let Some(status) = filters.status {
        if row.status != status {
            return false;
        }
    }

    if filters.has_warning && row.warnings.is_empty() {
        return false;
    }

    if filters.has_error && row.error.is_none() {
        return false;
    }

    if let Some(find) = &filters.find {
        let mut matched = row.inputs_hash.contains(find);
        if !matched {
            if let Some(id) = &row.id {
                matched = id.contains(find);
            }
        }
        if !matched {
            return false;
        }
    }

    true
}

fn summarize_rows(input_path: &str, rows: &[&ReportRow], top: usize) -> ReportSummary {
    let mut rows_ok = 0usize;
    let mut rows_err = 0usize;
    let mut rows_with_warnings = 0usize;
    let mut counts_by_status = StatusCounts::default();

    for row in rows {
        if row.error.is_some() {
            rows_err = rows_err.saturating_add(1);
        } else {
            rows_ok = rows_ok.saturating_add(1);
        }

        if !row.warnings.is_empty() {
            rows_with_warnings = rows_with_warnings.saturating_add(1);
        }

        match row.status {
            ReportStatus::Lucid => {
                counts_by_status.lucid = counts_by_status.lucid.saturating_add(1)
            }
            ReportStatus::Hazy => counts_by_status.hazy = counts_by_status.hazy.saturating_add(1),
            ReportStatus::Delirium => {
                counts_by_status.delirium = counts_by_status.delirium.saturating_add(1)
            }
            ReportStatus::Unknown => {
                counts_by_status.unknown = counts_by_status.unknown.saturating_add(1)
            }
        }
    }

    ReportSummary {
        input_path: input_path.to_string(),
        rows_total: rows.len(),
        rows_ok,
        rows_err,
        counts_by_status,
        rows_with_warnings,
        worst_k: worst_k(rows, top),
    }
}

fn worst_k(rows: &[&ReportRow], top: usize) -> Vec<ReportWorstRow> {
    let mut items: Vec<ReportWorstRow> = rows
        .iter()
        .filter_map(|row| {
            let ratio = row.max_score_ratio?;
            Some(ReportWorstRow {
                row_index: row.row_index,
                id: row.id.clone(),
                status: row.status.as_str().to_string(),
                inputs_hash: row.inputs_hash.clone(),
                max_score_ratio: ratio,
                err_code: row.error.as_ref().and_then(|error| error.code.clone()),
            })
        })
        .collect();

    items.sort_by(|left, right| {
        right
            .max_score_ratio
            .total_cmp(&left.max_score_ratio)
            .then_with(|| left.row_index.cmp(&right.row_index))
    });

    items.truncate(top);
    items
}

fn print_summary(summary: &ReportSummary) {
    println!(
        "report summary: input={} total={} ok={} err={} warnings={}",
        summary.input_path,
        summary.rows_total,
        summary.rows_ok,
        summary.rows_err,
        summary.rows_with_warnings
    );
    println!(
        "status counts: LUCID={} HAZY={} DELIRIUM={} UNKNOWN={}",
        summary.counts_by_status.lucid,
        summary.counts_by_status.hazy,
        summary.counts_by_status.delirium,
        summary.counts_by_status.unknown
    );

    if summary.worst_k.is_empty() {
        println!("worst_k: none");
        return;
    }

    println!(
        "worst_k (top {} by max_score_ratio among filtered rows):",
        summary.worst_k.len()
    );
    for row in &summary.worst_k {
        if let Some(id) = &row.id {
            if let Some(err_code) = &row.err_code {
                println!(
                    "row={} id={} ratio={:.6} status={} hash={} err_code={}",
                    row.row_index, id, row.max_score_ratio, row.status, row.inputs_hash, err_code
                );
            } else {
                println!(
                    "row={} id={} ratio={:.6} status={} hash={}",
                    row.row_index, id, row.max_score_ratio, row.status, row.inputs_hash
                );
            }
        } else if let Some(err_code) = &row.err_code {
            println!(
                "row={} ratio={:.6} status={} hash={} err_code={}",
                row.row_index, row.max_score_ratio, row.status, row.inputs_hash, err_code
            );
        } else {
            println!(
                "row={} ratio={:.6} status={} hash={}",
                row.row_index, row.max_score_ratio, row.status, row.inputs_hash
            );
        }
    }
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

#[cfg(feature = "cli-tui")]
mod tui {
    use super::{row_matches_filters, value_to_usize, ReportFilters, ReportRow, ReportStatus};
    use crossterm::event::{self, Event, KeyCode, KeyEventKind};
    use crossterm::execute;
    use crossterm::terminal::{
        disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
    };
    use ratatui::backend::CrosstermBackend;
    use ratatui::layout::{Constraint, Direction, Layout};
    use ratatui::style::{Color, Modifier, Style};
    use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, TableState, Wrap};
    use ratatui::Terminal;
    use serde_json::Value;
    use std::fmt::Write as _;
    use std::io;
    use std::time::Duration;

    pub(super) enum TuiError {
        Init(String),
        Runtime(String),
    }

    #[derive(Clone, Copy, Eq, PartialEq)]
    enum InputMode {
        Normal,
        Search,
    }

    struct ViewState {
        rows: Vec<ReportRow>,
        status_filter: Option<ReportStatus>,
        require_warning: bool,
        require_error: bool,
        search_query: String,
        search_edit: String,
        input_mode: InputMode,
        selected: usize,
    }

    impl ViewState {
        fn new(rows: Vec<ReportRow>, base_filters: ReportFilters) -> Self {
            Self {
                rows,
                status_filter: base_filters.status,
                require_warning: base_filters.has_warning,
                require_error: base_filters.has_error,
                search_query: base_filters.find.unwrap_or_default(),
                search_edit: String::new(),
                input_mode: InputMode::Normal,
                selected: 0,
            }
        }

        fn effective_filters(&self) -> ReportFilters {
            ReportFilters {
                status: self.status_filter,
                has_warning: self.require_warning,
                has_error: self.require_error,
                find: if self.search_query.is_empty() {
                    None
                } else {
                    Some(self.search_query.clone())
                },
            }
        }

        fn filtered_indices(&self) -> Vec<usize> {
            let filters = self.effective_filters();
            self.rows
                .iter()
                .enumerate()
                .filter_map(|(index, row)| {
                    if row_matches_filters(row, &filters) {
                        Some(index)
                    } else {
                        None
                    }
                })
                .collect()
        }

        fn clamp_selection(&mut self) {
            let len = self.filtered_indices().len();
            if len == 0 {
                self.selected = 0;
            } else if self.selected >= len {
                self.selected = len - 1;
            }
        }

        fn move_selection_up(&mut self) {
            let len = self.filtered_indices().len();
            if len == 0 {
                self.selected = 0;
                return;
            }
            if self.selected == 0 {
                self.selected = len - 1;
            } else {
                self.selected -= 1;
            }
        }

        fn move_selection_down(&mut self) {
            let len = self.filtered_indices().len();
            if len == 0 {
                self.selected = 0;
                return;
            }
            self.selected = (self.selected + 1) % len;
        }

        fn cycle_error_warning_filters(&mut self) {
            match (self.require_error, self.require_warning) {
                (false, false) => {
                    self.require_error = true;
                    self.require_warning = false;
                }
                (true, false) => {
                    self.require_error = false;
                    self.require_warning = true;
                }
                (false, true) => {
                    self.require_error = true;
                    self.require_warning = true;
                }
                (true, true) => {
                    self.require_error = false;
                    self.require_warning = false;
                }
            }
            self.selected = 0;
        }

        fn filter_mode_label(&self) -> &'static str {
            match (self.require_error, self.require_warning) {
                (false, false) => "none",
                (true, false) => "has_error",
                (false, true) => "has_warning",
                (true, true) => "has_error+has_warning",
            }
        }
    }

    pub(super) fn run(rows: Vec<ReportRow>, base_filters: ReportFilters) -> Result<(), TuiError> {
        enable_raw_mode()
            .map_err(|err| TuiError::Init(format!("failed to enable raw mode: {}", err)))?;

        let mut stdout = io::stdout();
        if let Err(err) = execute!(stdout, EnterAlternateScreen) {
            let _ = disable_raw_mode();
            return Err(TuiError::Init(format!(
                "failed to enter alternate screen: {}",
                err
            )));
        }

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(err) => {
                let _ = disable_raw_mode();
                let mut cleanup_stdout = io::stdout();
                let _ = execute!(cleanup_stdout, LeaveAlternateScreen);
                return Err(TuiError::Init(format!(
                    "failed to initialize terminal: {}",
                    err
                )));
            }
        };

        let mut state = ViewState::new(rows, base_filters);
        let run_result = run_loop(&mut terminal, &mut state);

        let mut cleanup_error: Option<String> = None;
        if let Err(err) = disable_raw_mode() {
            cleanup_error = Some(format!("failed to disable raw mode: {}", err));
        }
        if let Err(err) = execute!(terminal.backend_mut(), LeaveAlternateScreen) {
            cleanup_error = Some(format!("failed to leave alternate screen: {}", err));
        }
        if let Err(err) = terminal.show_cursor() {
            cleanup_error = Some(format!("failed to restore terminal cursor: {}", err));
        }

        if let Some(cleanup_error) = cleanup_error {
            return Err(TuiError::Runtime(cleanup_error));
        }

        run_result
    }

    fn run_loop(
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
        state: &mut ViewState,
    ) -> Result<(), TuiError> {
        loop {
            state.clamp_selection();
            terminal.draw(|frame| render(frame, state)).map_err(|err| {
                TuiError::Runtime(format!("failed to draw report viewer: {}", err))
            })?;

            if !event::poll(Duration::from_millis(120))
                .map_err(|err| TuiError::Runtime(format!("failed to poll TUI events: {}", err)))?
            {
                continue;
            }

            let event = event::read()
                .map_err(|err| TuiError::Runtime(format!("failed to read TUI event: {}", err)))?;

            if let Event::Key(key) = event {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                match state.input_mode {
                    InputMode::Normal => match key.code {
                        KeyCode::Char('q') => return Ok(()),
                        KeyCode::Up => state.move_selection_up(),
                        KeyCode::Down => state.move_selection_down(),
                        KeyCode::Char('/') => {
                            state.input_mode = InputMode::Search;
                            state.search_edit = state.search_query.clone();
                        }
                        KeyCode::Char('f') => state.cycle_error_warning_filters(),
                        _ => {}
                    },
                    InputMode::Search => match key.code {
                        KeyCode::Enter => {
                            state.search_query = state.search_edit.clone();
                            state.input_mode = InputMode::Normal;
                            state.selected = 0;
                        }
                        KeyCode::Esc => {
                            state.search_edit.clear();
                            state.input_mode = InputMode::Normal;
                        }
                        KeyCode::Backspace => {
                            state.search_edit.pop();
                        }
                        KeyCode::Char(ch) => {
                            state.search_edit.push(ch);
                        }
                        _ => {}
                    },
                }
            }
        }
    }

    fn render(frame: &mut ratatui::Frame<'_>, state: &ViewState) {
        let outer = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(1), Constraint::Length(2)])
            .split(frame.area());

        let body = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(54), Constraint::Percentage(46)])
            .split(outer[0]);

        let filtered = state.filtered_indices();
        let mut table_state = TableState::default();
        if !filtered.is_empty() {
            table_state.select(Some(state.selected));
        }

        let header = Row::new(vec![
            Cell::from("row_index"),
            Cell::from("id"),
            Cell::from("status"),
            Cell::from("max_ratio"),
            Cell::from("warn_count"),
            Cell::from("err_code"),
        ])
        .style(
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        );

        let table_rows: Vec<Row<'_>> = filtered
            .iter()
            .map(|index| {
                let row = &state.rows[*index];
                let warn_count = row.warnings.len().to_string();
                let max_ratio = row
                    .max_score_ratio
                    .map(|ratio| format!("{ratio:.6}"))
                    .unwrap_or_else(|| "-".to_string());
                let err_code = row
                    .error
                    .as_ref()
                    .and_then(|error| error.code.clone())
                    .unwrap_or_else(|| "-".to_string());
                Row::new(vec![
                    Cell::from(row.row_index.to_string()),
                    Cell::from(row.id.clone().unwrap_or_else(|| "-".to_string())),
                    Cell::from(row.status.as_str()),
                    Cell::from(max_ratio),
                    Cell::from(warn_count),
                    Cell::from(err_code),
                ])
            })
            .collect();

        let table = Table::new(
            table_rows,
            [
                Constraint::Length(9),
                Constraint::Length(20),
                Constraint::Length(10),
                Constraint::Length(12),
                Constraint::Length(10),
                Constraint::Min(10),
            ],
        )
        .header(header)
        .row_highlight_style(Style::default().add_modifier(Modifier::REVERSED))
        .highlight_symbol("> ")
        .block(
            Block::default()
                .borders(Borders::ALL)
                .title("Rows (Up/Down select, / search, f toggle, q quit)"),
        );
        frame.render_stateful_widget(table, body[0], &mut table_state);

        let selected_row = filtered
            .get(state.selected)
            .and_then(|index| state.rows.get(*index));
        let detail_text = render_details(selected_row);

        let details = Paragraph::new(detail_text)
            .block(Block::default().title("Details").borders(Borders::ALL))
            .wrap(Wrap { trim: false });
        frame.render_widget(details, body[1]);

        let help_text = if state.input_mode == InputMode::Search {
            format!("search: /{} (Enter apply, Esc cancel)", state.search_edit)
        } else {
            format!(
                "status={} filter={} search={} matches={}",
                state
                    .status_filter
                    .map(|status| status.as_str().to_string())
                    .unwrap_or_else(|| "ALL".to_string()),
                state.filter_mode_label(),
                if state.search_query.is_empty() {
                    "-".to_string()
                } else {
                    state.search_query.clone()
                },
                filtered.len()
            )
        };

        let help =
            Paragraph::new(help_text).block(Block::default().borders(Borders::ALL).title("State"));
        frame.render_widget(help, outer[1]);
    }

    fn render_details(row: Option<&ReportRow>) -> String {
        let Some(row) = row else {
            return "No rows match current filters.".to_string();
        };

        let mut out = String::new();
        let _ = writeln!(out, "row_index: {}", row.row_index);
        let _ = writeln!(
            out,
            "id: {}",
            row.id.clone().unwrap_or_else(|| "-".to_string())
        );
        let _ = writeln!(out, "inputs_hash: {}", row.inputs_hash);
        let _ = writeln!(out, "status: {}", row.status.as_str());

        if let Some(error) = &row.error {
            let _ = writeln!(
                out,
                "error: code={} message={}",
                error.code.clone().unwrap_or_else(|| "-".to_string()),
                error.message.clone().unwrap_or_else(|| "-".to_string())
            );
        } else {
            let _ = writeln!(out, "error: none");
        }

        if let Some(hashes) = row
            .raw
            .get("audit_trace")
            .and_then(|audit| audit.get("hashes"))
            .and_then(Value::as_object)
        {
            let _ = writeln!(
                out,
                "audit hashes: inputs={} measurement={} policy={}",
                hashes
                    .get("inputs_hash")
                    .and_then(Value::as_str)
                    .unwrap_or("-"),
                hashes
                    .get("measurement_hash")
                    .and_then(Value::as_str)
                    .unwrap_or("-"),
                hashes
                    .get("policy_hash")
                    .and_then(Value::as_str)
                    .unwrap_or("-")
            );
        }

        if row.warnings.is_empty() {
            let _ = writeln!(out, "warnings: none");
        } else {
            let _ = writeln!(out, "warnings ({}):", row.warnings.len());
            for warning in row.warnings.iter().take(8) {
                let warning_type = warning.get("type").and_then(Value::as_str).unwrap_or("-");
                let field = warning.get("field").and_then(Value::as_str).unwrap_or("-");
                let sentence_index = warning
                    .get("sentence_index")
                    .and_then(value_to_usize)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".to_string());
                let max_seq_len = warning
                    .get("max_seq_len")
                    .and_then(value_to_usize)
                    .map(|v| v.to_string())
                    .unwrap_or_else(|| "-".to_string());
                let _ = writeln!(
                    out,
                    "- type={} field={} sentence_index={} max_seq_len={}",
                    warning_type, field, sentence_index, max_seq_len
                );
            }
            if row.warnings.len() > 8 {
                let _ = writeln!(out, "- ... ({} more)", row.warnings.len() - 8);
            }
        }

        if let Some(evidence) = row
            .raw
            .get("data")
            .and_then(|data| data.get("evidence"))
            .and_then(Value::as_array)
        {
            if evidence.is_empty() {
                let _ = writeln!(out, "top evidence: none");
            } else {
                let _ = writeln!(out, "top evidence ({} shown):", evidence.len().min(6));
                for item in evidence.iter().take(6) {
                    let ans_idx = item
                        .get("ans_sentence_index")
                        .and_then(value_to_usize)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let ctx_idx = item
                        .get("ctx_sentence_index")
                        .and_then(value_to_usize)
                        .map(|v| v.to_string())
                        .unwrap_or_else(|| "-".to_string());
                    let ratio = item
                        .get("score_ratio")
                        .and_then(Value::as_f64)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "-".to_string());
                    let sem = item
                        .get("score_sem")
                        .or_else(|| item.get("score_sem_raw"))
                        .and_then(Value::as_f64)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "-".to_string());
                    let structural = item
                        .get("score_struct")
                        .and_then(Value::as_f64)
                        .map(|v| format!("{v:.6}"))
                        .unwrap_or_else(|| "-".to_string());
                    let tags = item
                        .get("tags")
                        .and_then(Value::as_array)
                        .map(|tags| {
                            tags.iter()
                                .map(|tag| {
                                    tag.as_str()
                                        .map(|v| v.to_string())
                                        .unwrap_or_else(|| tag.to_string())
                                })
                                .collect::<Vec<String>>()
                                .join("|")
                        })
                        .filter(|joined| !joined.is_empty())
                        .unwrap_or_else(|| "-".to_string());

                    let _ = writeln!(
                        out,
                        "- ans[{}] ctx[{}] ratio={} sem={} struct={} tags={}",
                        ans_idx, ctx_idx, ratio, sem, structural, tags
                    );

                    if let Some(rule_trace) = item.get("rule_trace") {
                        let trace_text = rule_trace
                            .as_str()
                            .map(|v| v.to_string())
                            .unwrap_or_else(|| rule_trace.to_string());
                        let _ = writeln!(out, "  rule_trace: {}", trace_text);
                    }
                }
            }
        } else {
            let _ = writeln!(out, "top evidence: unavailable");
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::{read_rows_from_reader, summarize_rows, ReportRow};
    use serde_json::json;
    use std::io::Cursor;

    #[test]
    fn ndjson_reader_strips_bom_on_first_non_empty_and_skips_blank_lines() {
        let text = concat!(
            "\n",
            "   \n",
            "\u{feff}{\"row_index\":0,\"inputs_hash\":\"h0\",\"status\":\"LUCID\",\"error\":null,\"data\":{\"scores\":{\"max_score_ratio\":1.0}},\"audit_trace\":{\"warnings\":[]}}\n",
            "\r\n",
            "{\"row_index\":1,\"id\":\"r1\",\"inputs_hash\":\"h1\",\"status\":\"HAZY\",\"error\":null,\"data\":{\"scores\":{\"max_score_ratio\":2.0}},\"audit_trace\":{\"warnings\":[]}}\n"
        );

        let mut cursor = Cursor::new(text.as_bytes());
        let rows = read_rows_from_reader(&mut cursor).expect("rows should parse");

        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].row_index, 0);
        assert_eq!(rows[1].row_index, 1);
        assert_eq!(rows[1].id.as_deref(), Some("r1"));
    }

    #[test]
    fn worst_k_uses_ratio_desc_then_row_index_asc() {
        let rows = [
            row(3, Some("a"), "h3", "HAZY", None, 2.4, false),
            row(1, Some("b"), "h1", "DELIRIUM", None, 2.4, false),
            row(2, Some("c"), "h2", "LUCID", None, 3.1, false),
            row(0, Some("d"), "h0", "UNKNOWN", Some("ROW_ERR"), 1.5, false),
        ];

        let refs: Vec<&ReportRow> = rows.iter().collect();
        let summary = summarize_rows("fixture.ndjson", &refs, 3);
        let worst = summary.worst_k;

        assert_eq!(worst.len(), 3);
        assert_eq!(worst[0].row_index, 2);
        assert_eq!(worst[1].row_index, 1);
        assert_eq!(worst[2].row_index, 3);
        assert_eq!(worst[2].status, "HAZY");
    }

    fn row(
        row_index: usize,
        id: Option<&str>,
        inputs_hash: &str,
        status: &str,
        err_code: Option<&str>,
        max_score_ratio: f64,
        has_warning: bool,
    ) -> ReportRow {
        let warning_list = if has_warning {
            vec![json!({ "type": "EMBED_TRUNCATED" })]
        } else {
            Vec::new()
        };

        let error_value = err_code.map(|code| {
            json!({
                "code": code,
                "message": "boom"
            })
        });

        let row = json!({
            "row_index": row_index,
            "id": id,
            "inputs_hash": inputs_hash,
            "status": status,
            "error": error_value,
            "data": { "scores": { "max_score_ratio": max_score_ratio } },
            "audit_trace": { "warnings": warning_list }
        });

        ReportRow::from_value(row, row_index)
    }
}
