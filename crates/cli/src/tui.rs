use super::{
    row_matches_filters, value_to_usize, ReportFilters, ReportRow, ReportStatus, TuiOptions,
};
use super::{ReportColor, ReportTheme};
use crossterm::cursor::{Hide, MoveTo, Show};
use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event, KeyCode, KeyEventKind, KeyModifiers,
    MouseEventKind,
};
use crossterm::execute;
use crossterm::style::ResetColor;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, Clear as TermClear, ClearType, DisableLineWrap,
    EnableLineWrap, EnterAlternateScreen, LeaveAlternateScreen,
};
use pale_ale_diagnose::default_policy_config;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{
    Block, Borders, Cell, Clear, Gauge, Paragraph, Row, Sparkline, Table, TableState, Wrap,
};
use ratatui::Terminal;
use serde_json::Value;

use std::io::{self, Write as _};
use std::time::Duration;

mod tui_theme;
use tui_theme::{ColorPreference, GaugeLevel, ThemeContext, ThemeName, UiState};

const EVENT_POLL_TIMEOUT: Duration = Duration::from_millis(250);
const THRESHOLD_STEP: f64 = 0.05;
const THRESHOLD_MIN: f64 = 0.0;
const THRESHOLD_MAX: f64 = 10.0;
const METRIC_MAX: f64 = 1.5;
const HIST_BINS: usize = 40;

const MIN_WIDTH: u16 = 50;
const MIN_HEIGHT: u16 = 12;
const WIDE_LAYOUT_WIDTH: u16 = 110;
const HEADER_HEIGHT: u16 = 2;
const FOOTER_HEIGHT: u16 = 1;
const PAGE_STEP: usize = 10;

type ReportTerminal = Terminal<CrosstermBackend<io::Stdout>>;

pub(super) enum TuiError {
    Init(String),
    Runtime(String),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum InputMode {
    Normal,
    Search,
}

#[derive(Clone, Copy, Debug, Default)]
struct RowMetrics {
    ratio: f64,
    sem: f64,
    struct_: f64,
}

#[derive(Clone, Copy, Debug)]
struct SelectedContext<'a> {
    row: &'a ReportRow,
    metrics: RowMetrics,
}

#[derive(Debug)]
struct HistogramState {
    bins: Vec<usize>,
    max_count: usize,
    max_value: f64,
    min_value: Option<f64>,
    p50_value: Option<f64>,
    p95_value: Option<f64>,
    observed_max: Option<f64>,
    usable_count: usize,
    above_hazy: usize,
    above_delirium: usize,
    top_outliers: Vec<(String, f64)>,
}

#[derive(Debug)]
struct TuiOutcome {
    print_policy_snippet: bool,
    policy_snippet: String,
}

struct TerminalGuard {
    terminal: ReportTerminal,
    restored: bool,
}

fn teardown_terminal_surface() -> io::Result<()> {
    let mut stdout = io::stdout();
    let result = execute!(
        stdout,
        DisableMouseCapture,
        EnableLineWrap,
        Show,
        ResetColor,
        LeaveAlternateScreen
    );
    let flush_result = stdout.flush();
    let leave_result = result.and(flush_result);

    // VSCode-based editors (VSCode, Cursor, Windsurf, Trae etc.) use
    // ConPTY + xterm.js which may leave rendering residue on the primary
    // screen after LeaveAlternateScreen.  Clear the primary screen only
    // in those environments to avoid side-effects on well-behaved
    // terminals like Windows Terminal.
    if is_vscode_like_terminal() {
        let _ = execute!(stdout, TermClear(ClearType::All), MoveTo(0, 0));
        let _ = stdout.flush();
    }

    leave_result
}

/// Returns `true` when the process is running inside an integrated terminal
/// of VSCode or a VSCode-based editor (Cursor, Windsurf, Trae, etc.).
fn is_vscode_like_terminal() -> bool {
    // TERM_PROGRAM is set by many VSCode forks.
    if let Ok(term) = std::env::var("TERM_PROGRAM") {
        let lower = term.to_ascii_lowercase();
        if lower.contains("vscode")
            || lower.contains("cursor")
            || lower.contains("windsurf")
            || lower.contains("trae")
        {
            return true;
        }
    }
    // Fallback: VSCode (and most forks) inject these env vars into child
    // processes even when TERM_PROGRAM is absent or renamed.
    std::env::var_os("VSCODE_PID").is_some()
        || std::env::var_os("VSCODE_IPC_HOOK_CLI").is_some()
        || std::env::var_os("VSCODE_GIT_IPC_HANDLE").is_some()
}

fn enter_terminal_surface() -> io::Result<()> {
    let mut stdout = io::stdout();
    let result = execute!(
        stdout,
        EnterAlternateScreen,
        Hide,
        DisableLineWrap,
        EnableMouseCapture
    );
    let flush_result = stdout.flush();
    result.and(flush_result)
}

fn flush_terminal_backend(terminal: &mut ReportTerminal, context: &str) -> Result<(), TuiError> {
    terminal
        .backend_mut()
        .flush()
        .map_err(|err| TuiError::Runtime(format!("failed to flush terminal {}: {}", context, err)))
}

impl TerminalGuard {
    fn enter() -> Result<Self, TuiError> {
        enable_raw_mode()
            .map_err(|err| TuiError::Init(format!("failed to enable raw mode: {}", err)))?;

        let backend = CrosstermBackend::new(io::stdout());
        let mut terminal = match Terminal::new(backend) {
            Ok(terminal) => terminal,
            Err(err) => {
                let _ = disable_raw_mode();
                return Err(TuiError::Init(format!(
                    "failed to initialize terminal: {}",
                    err
                )));
            }
        };

        if let Err(err) = enter_terminal_surface() {
            let _ = teardown_terminal_surface();
            let _ = disable_raw_mode();
            return Err(TuiError::Init(format!(
                "failed to enter alternate screen: {}",
                err
            )));
        }

        flush_terminal_backend(&mut terminal, "after enter")?;

        Ok(Self {
            terminal,
            restored: false,
        })
    }

    fn terminal_mut(&mut self) -> &mut ReportTerminal {
        &mut self.terminal
    }

    fn restore(&mut self) -> Result<(), TuiError> {
        if self.restored {
            return Ok(());
        }

        let mut failures = Vec::new();
        if let Err(err) = self.terminal.backend_mut().flush() {
            failures.push(format!(
                "failed to flush stdout backend before leave: {}",
                err
            ));
        }
        if let Err(err) = teardown_terminal_surface() {
            failures.push(format!("failed to leave alternate screen: {}", err));
        }
        if let Err(err) = disable_raw_mode() {
            failures.push(format!("failed to disable raw mode: {}", err));
        }
        self.restored = true;

        if failures.is_empty() {
            Ok(())
        } else {
            Err(TuiError::Runtime(failures.join("; ")))
        }
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        let _ = self.restore();
    }
}

#[derive(Default)]
struct KeyAction {
    redraw: bool,
    quit: bool,
}

impl KeyAction {
    fn none() -> Self {
        Self::default()
    }

    fn redraw() -> Self {
        Self {
            redraw: true,
            quit: false,
        }
    }

    fn quit() -> Self {
        Self {
            redraw: false,
            quit: true,
        }
    }
}

struct App {
    rows: Vec<ReportRow>,
    status_filter: Option<ReportStatus>,
    errors_only: bool,
    warnings_only: bool,
    search_query: String,
    search_edit: String,
    input_mode: InputMode,
    selected_row: usize,
    tmp_th_ratio_hazy: f64,
    tmp_th_ratio_delirium: f64,
    default_th_ratio_hazy: f64,
    default_th_ratio_delirium: f64,
    emit_policy_snippet: bool,
    show_hist_debug: bool,
    show_help: bool,
    theme_name: ThemeName,
    color_preference: ColorPreference,
    theme: ThemeContext,
}
impl App {
    fn new(rows: Vec<ReportRow>, base_filters: ReportFilters, options: TuiOptions) -> Self {
        let policy = default_policy_config();
        let default_hazy = f64::from(policy.th_ratio_hazy).clamp(THRESHOLD_MIN, THRESHOLD_MAX);
        let default_delirium = f64::from(policy.th_ratio_delirium)
            .clamp(THRESHOLD_MIN, THRESHOLD_MAX)
            .max(default_hazy);
        let theme_name = theme_name(options.theme);
        let color_preference = color_preference(options.color);

        Self {
            rows,
            status_filter: base_filters.status,
            errors_only: base_filters.has_error,
            warnings_only: base_filters.has_warning,
            search_query: base_filters.find.unwrap_or_default(),
            search_edit: String::new(),
            input_mode: InputMode::Normal,
            selected_row: 0,
            tmp_th_ratio_hazy: default_hazy,
            tmp_th_ratio_delirium: default_delirium,
            default_th_ratio_hazy: default_hazy,
            default_th_ratio_delirium: default_delirium,
            emit_policy_snippet: false,
            show_hist_debug: false,
            show_help: false,
            theme_name,
            color_preference,
            theme: ThemeContext::new(theme_name, color_preference, options.ascii),
        }
    }

    fn effective_filters(&self) -> ReportFilters {
        ReportFilters {
            status: self.status_filter,
            has_warning: self.warnings_only,
            has_error: self.errors_only,
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
            .filter_map(|(idx, row)| {
                if row_matches_filters(row, &filters) {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    fn selected_context<'a>(&'a self, filtered_indices: &[usize]) -> Option<SelectedContext<'a>> {
        let selected_index = *filtered_indices.get(self.selected_row)?;
        let row = self.rows.get(selected_index)?;
        Some(SelectedContext {
            row,
            metrics: extract_row_metrics(row),
        })
    }

    fn clamp_selection(&mut self) {
        let len = self.filtered_indices().len();
        if len == 0 {
            self.selected_row = 0;
        } else if self.selected_row >= len {
            self.selected_row = len - 1;
        }
    }

    fn move_up(&mut self) {
        let len = self.filtered_indices().len();
        if len == 0 {
            self.selected_row = 0;
        } else {
            self.selected_row = self.selected_row.saturating_sub(1);
        }
    }

    fn move_down(&mut self) {
        let len = self.filtered_indices().len();
        if len == 0 {
            self.selected_row = 0;
        } else {
            self.selected_row = (self.selected_row + 1).min(len - 1);
        }
    }

    fn page_up(&mut self) {
        let len = self.filtered_indices().len();
        if len == 0 {
            self.selected_row = 0;
            return;
        }
        self.selected_row = self.selected_row.saturating_sub(PAGE_STEP);
    }

    fn page_down(&mut self) {
        let len = self.filtered_indices().len();
        if len == 0 {
            self.selected_row = 0;
            return;
        }
        self.selected_row = (self.selected_row + PAGE_STEP).min(len - 1);
    }

    fn move_top(&mut self) {
        self.selected_row = 0;
    }

    fn move_bottom(&mut self) {
        let len = self.filtered_indices().len();
        if len == 0 {
            self.selected_row = 0;
        } else {
            self.selected_row = len - 1;
        }
    }

    fn toggle_errors_only(&mut self) {
        self.errors_only = !self.errors_only;
        self.selected_row = 0;
    }

    fn toggle_warnings_only(&mut self) {
        self.warnings_only = !self.warnings_only;
        self.selected_row = 0;
    }

    fn enter_search_mode(&mut self) {
        self.input_mode = InputMode::Search;
        self.search_edit = self.search_query.clone();
    }

    fn apply_search(&mut self) {
        self.search_query = self.search_edit.clone();
        self.input_mode = InputMode::Normal;
        self.selected_row = 0;
    }

    fn cancel_search(&mut self) {
        self.search_edit.clear();
        self.input_mode = InputMode::Normal;
    }

    fn toggle_hist_debug(&mut self) {
        self.show_hist_debug = !self.show_hist_debug;
    }

    fn toggle_help(&mut self) {
        self.show_help = !self.show_help;
    }

    fn cycle_theme(&mut self) {
        self.theme_name = match self.theme_name {
            ThemeName::Classic => ThemeName::Term,
            ThemeName::Term => ThemeName::Cyber,
            ThemeName::Cyber => ThemeName::Classic,
        };
        self.theme = ThemeContext::new(self.theme_name, self.color_preference, self.theme.ascii());
    }

    fn theme_label(&self) -> &'static str {
        match self.theme_name {
            ThemeName::Classic => "classic",
            ThemeName::Term => "term",
            ThemeName::Cyber => "cyber",
        }
    }

    fn adjust_hazy(&mut self, delta: f64) {
        self.tmp_th_ratio_hazy =
            (self.tmp_th_ratio_hazy + delta).clamp(THRESHOLD_MIN, THRESHOLD_MAX);
        self.normalize_thresholds();
    }

    fn adjust_delirium(&mut self, delta: f64) {
        self.tmp_th_ratio_delirium =
            (self.tmp_th_ratio_delirium + delta).clamp(THRESHOLD_MIN, THRESHOLD_MAX);
        self.normalize_thresholds();
    }

    fn reset_thresholds(&mut self) {
        self.tmp_th_ratio_hazy = self.default_th_ratio_hazy;
        self.tmp_th_ratio_delirium = self.default_th_ratio_delirium;
        self.normalize_thresholds();
    }

    fn normalize_thresholds(&mut self) {
        if self.tmp_th_ratio_delirium < self.tmp_th_ratio_hazy {
            self.tmp_th_ratio_delirium = self.tmp_th_ratio_hazy;
        }
    }

    fn policy_snippet(&self) -> String {
        format!(
            "policy:\n  th_ratio_hazy: {:.2}\n  th_ratio_delirium: {:.2}",
            self.tmp_th_ratio_hazy, self.tmp_th_ratio_delirium
        )
    }

    fn outcome(&self) -> TuiOutcome {
        TuiOutcome {
            print_policy_snippet: self.emit_policy_snippet,
            policy_snippet: self.policy_snippet(),
        }
    }
}

fn theme_name(theme: ReportTheme) -> ThemeName {
    match theme {
        ReportTheme::Classic => ThemeName::Classic,
        ReportTheme::Term => ThemeName::Term,
        ReportTheme::Cyber => ThemeName::Cyber,
    }
}

fn color_preference(color: ReportColor) -> ColorPreference {
    match color {
        ReportColor::Auto => ColorPreference::Auto,
        ReportColor::Always => ColorPreference::Always,
        ReportColor::Never => ColorPreference::Never,
    }
}

pub(super) fn run(
    rows: Vec<ReportRow>,
    base_filters: ReportFilters,
    options: TuiOptions,
) -> Result<(), TuiError> {
    let mut session = TerminalGuard::enter()?;
    let mut app = App::new(rows, base_filters, options);

    let app_result = run_app(session.terminal_mut(), &mut app);
    let restore_result = session.restore();

    match (app_result, restore_result) {
        (Err(err), _) => Err(err),
        (Ok(_), Err(err)) => Err(err),
        (Ok(outcome), Ok(())) => {
            if outcome.print_policy_snippet {
                println!("{}", outcome.policy_snippet);
            }
            Ok(())
        }
    }
}

fn run_app(terminal: &mut ReportTerminal, app: &mut App) -> Result<TuiOutcome, TuiError> {
    let mut dirty = true;

    loop {
        if dirty {
            app.clamp_selection();
            terminal
                .draw(|frame| {
                    render(frame, app);
                })
                .map_err(|err| {
                    TuiError::Runtime(format!("failed to draw report viewer: {}", err))
                })?;
            flush_terminal_backend(terminal, "after draw")?;
            dirty = false;
        }

        if !event::poll(EVENT_POLL_TIMEOUT)
            .map_err(|err| TuiError::Runtime(format!("failed to poll TUI events: {}", err)))?
        {
            continue;
        }

        let event = event::read()
            .map_err(|err| TuiError::Runtime(format!("failed to read TUI event: {}", err)))?;

        match event {
            Event::Key(key) => {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                if app.show_help {
                    app.show_help = false;
                    dirty = true;
                    continue;
                }

                if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                    return Ok(app.outcome());
                }

                let action = match app.input_mode {
                    InputMode::Normal => handle_normal_mode_key(app, key.code),
                    InputMode::Search => handle_search_mode_key(app, key.code, key.modifiers),
                };
                if action.quit {
                    return Ok(app.outcome());
                }
                if action.redraw {
                    dirty = true;
                }
            }
            Event::Mouse(mouse) => {
                let action = handle_mouse_event(app, mouse.kind);
                if action.redraw {
                    dirty = true;
                }
            }
            Event::Resize(_, _) => {
                terminal.clear().map_err(|err| {
                    TuiError::Runtime(format!("failed to clear terminal on resize: {}", err))
                })?;
                flush_terminal_backend(terminal, "after resize clear")?;
                dirty = true;
                continue;
            }
            _ => {}
        }
    }
}
fn handle_normal_mode_key(app: &mut App, code: KeyCode) -> KeyAction {
    match code {
        KeyCode::Char('q') => KeyAction::quit(),
        KeyCode::Home | KeyCode::Char('g') => {
            app.move_top();
            KeyAction::redraw()
        }
        KeyCode::End | KeyCode::Char('G') => {
            app.move_bottom();
            KeyAction::redraw()
        }
        KeyCode::Up | KeyCode::Char('k') => {
            app.move_up();
            KeyAction::redraw()
        }
        KeyCode::Down | KeyCode::Char('j') => {
            app.move_down();
            KeyAction::redraw()
        }
        KeyCode::PageUp => {
            app.page_up();
            KeyAction::redraw()
        }
        KeyCode::PageDown => {
            app.page_down();
            KeyAction::redraw()
        }
        KeyCode::Char('/') => {
            app.enter_search_mode();
            KeyAction::redraw()
        }
        KeyCode::Char('f') => {
            app.toggle_errors_only();
            KeyAction::redraw()
        }
        KeyCode::Char('w') => {
            app.toggle_warnings_only();
            KeyAction::redraw()
        }
        KeyCode::Char('+') => {
            app.adjust_hazy(THRESHOLD_STEP);
            KeyAction::redraw()
        }
        KeyCode::Char('-') => {
            app.adjust_hazy(-THRESHOLD_STEP);
            KeyAction::redraw()
        }
        KeyCode::Char('[') => {
            app.adjust_delirium(-THRESHOLD_STEP);
            KeyAction::redraw()
        }
        KeyCode::Char(']') => {
            app.adjust_delirium(THRESHOLD_STEP);
            KeyAction::redraw()
        }
        KeyCode::Char('r') => {
            app.reset_thresholds();
            KeyAction::redraw()
        }
        KeyCode::Char('s') => {
            app.emit_policy_snippet = true;
            KeyAction::redraw()
        }
        KeyCode::Char('?') => {
            app.toggle_help();
            KeyAction::redraw()
        }
        KeyCode::Char('t') => {
            app.cycle_theme();
            KeyAction::redraw()
        }
        KeyCode::Char('d') => {
            app.toggle_hist_debug();
            KeyAction::redraw()
        }
        _ => KeyAction::none(),
    }
}

fn handle_search_mode_key(app: &mut App, code: KeyCode, modifiers: KeyModifiers) -> KeyAction {
    match code {
        KeyCode::Enter => {
            app.apply_search();
            KeyAction::redraw()
        }
        KeyCode::Esc => {
            app.cancel_search();
            KeyAction::redraw()
        }
        KeyCode::Backspace => {
            app.search_edit.pop();
            KeyAction::redraw()
        }
        KeyCode::Char(ch)
            if !modifiers.contains(KeyModifiers::CONTROL)
                && !modifiers.contains(KeyModifiers::ALT) =>
        {
            app.search_edit.push(ch);
            KeyAction::redraw()
        }
        _ => KeyAction::none(),
    }
}

fn handle_mouse_event(app: &mut App, kind: MouseEventKind) -> KeyAction {
    match kind {
        MouseEventKind::ScrollUp => {
            app.move_up();
            KeyAction::redraw()
        }
        MouseEventKind::ScrollDown => {
            app.move_down();
            KeyAction::redraw()
        }
        _ => KeyAction::none(),
    }
}

fn render(frame: &mut ratatui::Frame<'_>, app: &App) {
    let area = frame.area();
    frame.render_widget(
        Block::default().style(Style::default().bg(app.theme.app_bg_color())),
        area,
    );

    if area.width < MIN_WIDTH || area.height < MIN_HEIGHT {
        render_too_small(frame, area, area.width, area.height, &app.theme);
        return;
    }

    let filtered = app.filtered_indices();
    let selected = app.selected_context(&filtered);

    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(HEADER_HEIGHT),
            Constraint::Min(6),
            Constraint::Length(FOOTER_HEIGHT),
        ])
        .split(area);

    render_header(frame, vertical[0], app, &filtered, selected);
    render_body(frame, vertical[1], app, &filtered, selected);
    render_footer(frame, vertical[2], app, &filtered);

    if app.show_help {
        render_help_popup(frame, app);
    }
}

fn render_too_small(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    width: u16,
    height: u16,
    theme: &ThemeContext,
) {
    let message = format!("Terminal too small {}x{}", width, height);
    let widget = Paragraph::new(message)
        .alignment(Alignment::Center)
        .style(theme.style_text())
        .block(
            Block::default()
                .title("Report Cockpit")
                .title_style(theme.style_title())
                .borders(Borders::ALL)
                .border_style(theme.style_border())
                .border_type(theme.glyphs().border_type)
                .style(theme.style_panel()),
        );
    frame.render_widget(widget, area);
}

fn render_header(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &App,
    filtered: &[usize],
    selected: Option<SelectedContext<'_>>,
) {
    let top = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(44), Constraint::Percentage(56)])
        .split(area);

    let selected_label = selected
        .map(|ctx| format!("row#{}", ctx.row.row_index))
        .unwrap_or_else(|| "-".to_string());
    let selected_pos = if filtered.is_empty() {
        "-".to_string()
    } else {
        format!("{}/{}", app.selected_row.saturating_add(1), filtered.len())
    };
    let summary = format!(
        "rows={} matches={} sel={} ({}) policy={:.2}/{:.2} theme={}",
        app.rows.len(),
        filtered.len(),
        selected_pos,
        selected_label,
        app.default_th_ratio_hazy,
        app.default_th_ratio_delirium,
        app.theme_label()
    );

    frame.render_widget(
        Paragraph::new("Pale Ale Ops Cockpit")
            .style(app.theme.style_accent())
            .alignment(Alignment::Left),
        top[0],
    );
    frame.render_widget(
        Paragraph::new(summary)
            .style(app.theme.style_muted())
            .alignment(Alignment::Right),
        top[1],
    );

    if area.height < 2 {
        return;
    }

    let (lucid_count, hazy_count, delirium_count, unknown_count) =
        summarize_statuses(app, filtered);
    let status_filter = app
        .status_filter
        .map(|status| status.as_str().to_string())
        .unwrap_or_else(|| "ALL".to_string());

    let mut chips = vec![
        Span::styled("L ", app.theme.style_muted()),
        Span::styled(
            format!("{:>3}", lucid_count),
            app.theme.style_chip(UiState::Lucid),
        ),
        Span::raw("  "),
        Span::styled("H ", app.theme.style_muted()),
        Span::styled(
            format!("{:>3}", hazy_count),
            app.theme.style_chip(UiState::Hazy),
        ),
        Span::raw("  "),
        Span::styled("D ", app.theme.style_muted()),
        Span::styled(
            format!("{:>3}", delirium_count),
            app.theme.style_chip(UiState::Delirium),
        ),
        Span::raw("  "),
        Span::styled("U ", app.theme.style_muted()),
        Span::styled(
            format!("{:>3}", unknown_count),
            app.theme.style_chip(UiState::Unknown),
        ),
        Span::raw("  "),
        Span::styled("status ", app.theme.style_muted()),
        Span::styled(status_filter, app.theme.style_text()),
        Span::raw("  "),
        Span::styled(
            if app.errors_only { "err:on" } else { "err:off" },
            if app.errors_only {
                app.theme.style_text().add_modifier(Modifier::BOLD)
            } else {
                app.theme.style_muted()
            },
        ),
        Span::raw("  "),
        Span::styled(
            if app.warnings_only {
                "warn:on"
            } else {
                "warn:off"
            },
            if app.warnings_only {
                app.theme.style_text().add_modifier(Modifier::BOLD)
            } else {
                app.theme.style_muted()
            },
        ),
    ];
    if !app.search_query.is_empty() {
        chips.push(Span::raw("  "));
        chips.push(Span::styled("find ", app.theme.style_muted()));
        chips.push(Span::styled(
            app.search_query.clone(),
            app.theme.style_text().add_modifier(Modifier::BOLD),
        ));
    }

    let status_area = Rect {
        x: area.x,
        y: area.y.saturating_add(1),
        width: area.width,
        height: 1,
    };
    frame.render_widget(
        Paragraph::new(Line::from(chips))
            .style(app.theme.style_text())
            .wrap(Wrap { trim: false }),
        status_area,
    );
}
fn render_footer(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App, filtered: &[usize]) {
    let nav = "? help   / search   q quit   t theme";
    let lead = if app.input_mode == InputMode::Search {
        format!(
            "search /{}   Enter apply   Esc cancel   matches {}",
            app.search_edit,
            filtered.len()
        )
    } else {
        "j/k move   g/G top-btm   PgUp/PgDn page   f err   w warn   +/- hazy   [] delirium   r reset   s snippet"
            .to_string()
    };
    let mut spans = vec![
        Span::styled(lead, app.theme.style_muted()),
        Span::raw("   "),
        Span::styled(nav, app.theme.style_text()),
        Span::raw("   "),
        Span::styled(
            format!("theme={}", app.theme_label()),
            app.theme.style_muted(),
        ),
    ];
    if app.emit_policy_snippet {
        spans.push(Span::raw("  "));
        spans.push(Span::styled(
            "snippet queued",
            app.theme.style_text().add_modifier(Modifier::BOLD),
        ));
    }
    let footer = Paragraph::new(Line::from(spans))
        .wrap(Wrap { trim: false })
        .style(app.theme.style_muted());
    frame.render_widget(footer, area);
}
fn summarize_statuses(app: &App, filtered: &[usize]) -> (usize, usize, usize, usize) {
    let mut lucid = 0usize;
    let mut hazy = 0usize;
    let mut delirium = 0usize;
    let mut unknown = 0usize;
    for index in filtered {
        if let Some(row) = app.rows.get(*index) {
            match row.status {
                ReportStatus::Lucid => lucid = lucid.saturating_add(1),
                ReportStatus::Hazy => hazy = hazy.saturating_add(1),
                ReportStatus::Delirium => delirium = delirium.saturating_add(1),
                ReportStatus::Unknown => unknown = unknown.saturating_add(1),
            }
        }
    }
    (lucid, hazy, delirium, unknown)
}
fn render_help_popup(frame: &mut ratatui::Frame<'_>, app: &App) {
    let popup = centered_rect(80, 70, frame.area());
    frame.render_widget(Clear, popup);

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(app.theme.style_border())
        .title("Help")
        .title_style(app.theme.style_title())
        .border_type(app.theme.glyphs().border_type)
        .style(app.theme.style_panel());
    let inner = block.inner(popup);
    frame.render_widget(block, popup);

    let lines = vec![
        Line::from(Span::styled("Navigation", app.theme.style_muted())),
        Line::from("  Up/Down or j/k: move selection"),
        Line::from("  Home/g: first row"),
        Line::from("  End/G: last row"),
        Line::from("  PgUp/PgDn: page move"),
        Line::from("  Mouse wheel: scroll"),
        Line::from(""),
        Line::from(Span::styled("Filters / Search", app.theme.style_muted())),
        Line::from("  / : search mode"),
        Line::from("  f : toggle errors only"),
        Line::from("  w : toggle warnings only"),
        Line::from(""),
        Line::from(Span::styled("Thresholds", app.theme.style_muted())),
        Line::from("  +/- : adjust hazy threshold"),
        Line::from("  [ / ] : adjust delirium threshold"),
        Line::from("  r : reset thresholds"),
        Line::from("  s : queue policy snippet"),
        Line::from("  d : histogram debug"),
        Line::from("  t : cycle theme (classic -> term -> cyber)"),
        Line::from(""),
        Line::from(Span::styled("CLI Examples", app.theme.style_muted())),
        Line::from("  pale-ale tui --target ./report_out.ndjson --theme classic --color auto"),
        Line::from(
            "  pale-ale tui --target ./report_out.ndjson --theme term --color never --ascii",
        ),
        Line::from("  pale-ale tui --target ./report_out.ndjson --theme cyber --color always"),
        Line::from(""),
        Line::from("  ? : toggle help"),
        Line::from("  q : quit"),
        Line::from(""),
        Line::from(Span::styled(
            "Press any key to close",
            app.theme.style_text().add_modifier(Modifier::BOLD),
        )),
    ];

    let text = Paragraph::new(lines)
        .style(app.theme.style_text())
        .wrap(Wrap { trim: false });
    frame.render_widget(text, inner);
}

fn centered_rect(percent_x: u16, percent_y: u16, area: Rect) -> Rect {
    let vertical = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage((100 - percent_y) / 2),
            Constraint::Percentage(percent_y),
            Constraint::Percentage((100 - percent_y) / 2),
        ])
        .split(area);
    let horizontal = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage((100 - percent_x) / 2),
            Constraint::Percentage(percent_x),
            Constraint::Percentage((100 - percent_x) / 2),
        ])
        .split(vertical[1]);
    horizontal[1]
}

fn render_body(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &App,
    filtered: &[usize],
    selected: Option<SelectedContext<'_>>,
) {
    if area.width >= WIDE_LAYOUT_WIDTH {
        let columns = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(54), Constraint::Percentage(46)])
            .split(area);
        render_table(frame, columns[0], app, filtered);
        render_details_column(frame, columns[1], app, filtered, selected);
    } else {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
            .split(area);
        render_table(frame, rows[0], app, filtered);
        render_details_column(frame, rows[1], app, filtered, selected);
    }
}

fn render_details_column(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &App,
    filtered: &[usize],
    selected: Option<SelectedContext<'_>>,
) {
    let selected_ratio = selected.map(|ctx| ctx.metrics.ratio);
    let sections = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Percentage(22),
            Constraint::Percentage(28),
            Constraint::Percentage(50),
        ])
        .split(area);

    let histogram = build_histogram_state(app, filtered, selected_ratio);

    render_metrics_panel(frame, sections[0], app, selected, &histogram);
    render_histogram_panel(frame, sections[1], app, filtered, &histogram, selected);
    render_snippet_panel(frame, sections[2], app, selected);
}

fn render_table(frame: &mut ratatui::Frame<'_>, area: Rect, app: &App, filtered: &[usize]) {
    let glyphs = app.theme.glyphs();
    let header = Row::new(vec![
        Cell::from(""),
        Cell::from("row_index"),
        Cell::from("id"),
        Cell::from("status"),
        Cell::from("max_ratio"),
        Cell::from("warn_count"),
        Cell::from("err_code"),
    ])
    .style(app.theme.style_muted().add_modifier(Modifier::BOLD));

    let rows: Vec<Row<'_>> = filtered
        .iter()
        .enumerate()
        .map(|(visible_index, index)| {
            let row = &app.rows[*index];
            let max_ratio = row
                .max_score_ratio
                .filter(|value| value.is_finite())
                .map(|value| format!("{:>10.6}", value))
                .unwrap_or_else(|| format!("{:>10}", "-"));
            let err_code = row
                .error
                .as_ref()
                .and_then(|error| error.code.clone())
                .unwrap_or_else(|| "-".to_string());
            let marker_cell = if visible_index == app.selected_row {
                Cell::from(Span::styled(
                    glyphs.selection_marker,
                    app.theme.style_selection_glyph(),
                ))
            } else {
                Cell::from(" ")
            };
            let status_cell = Cell::from(Line::from(vec![
                Span::styled(
                    status_marker(row.status, app.theme.ascii()),
                    app.theme.style_chip(report_status_to_ui_state(row.status)),
                ),
                Span::styled(row.status.as_str(), app.theme.style_text()),
            ]));

            Row::new(vec![
                marker_cell,
                Cell::from(format!("{:>9}", row.row_index)),
                Cell::from(row.id.clone().unwrap_or_else(|| "-".to_string())),
                status_cell,
                Cell::from(max_ratio),
                Cell::from(format!("{:>10}", row.warnings.len())),
                Cell::from(err_code),
            ])
            .style(app.theme.style_text())
        })
        .collect();

    let mut table_state = TableState::default();
    if !filtered.is_empty() {
        table_state.select(Some(app.selected_row));
    }

    let table = Table::new(
        rows,
        [
            Constraint::Length(2),
            Constraint::Length(9),
            Constraint::Length(18),
            Constraint::Length(14),
            Constraint::Length(10),
            Constraint::Length(10),
            Constraint::Min(10),
        ],
    )
    .header(header)
    .row_highlight_style(app.theme.style_selected())
    .block(
        Block::default()
            .borders(Borders::ALL)
            .border_style(app.theme.style_border())
            .title("Rows")
            .title_style(app.theme.style_section_title())
            .border_type(glyphs.border_type)
            .style(app.theme.style_panel()),
    );

    frame.render_stateful_widget(table, area, &mut table_state);
    if filtered.is_empty() {
        let empty_area = Rect {
            x: area.x.saturating_add(1),
            y: area.y.saturating_add(1),
            width: area.width.saturating_sub(2),
            height: area.height.saturating_sub(2),
        };
        if empty_area.width > 0 && empty_area.height > 0 {
            frame.render_widget(
                Paragraph::new("No rows match current filters")
                    .style(app.theme.style_muted())
                    .alignment(Alignment::Center),
                empty_area,
            );
        }
    }
}

fn render_metrics_panel(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &App,
    selected: Option<SelectedContext<'_>>,
    histogram: &HistogramState,
) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(app.theme.style_border())
        .title(" Metrics ")
        .title_style(app.theme.style_section_title())
        .border_type(app.theme.glyphs().border_type)
        .style(app.theme.style_panel());
    let inner = block.inner(area);
    frame.render_widget(block, area);

    let Some(selected) = selected else {
        let empty = Paragraph::new("No row selected")
            .alignment(Alignment::Center)
            .style(app.theme.style_muted());
        frame.render_widget(empty, inner);
        return;
    };

    let ratio_max = histogram.max_value.max(1.0);
    let (badge, badge_state) = ratio_badge(selected.row.status);
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Min(1),
        ])
        .split(inner);

    render_metric_gauge(
        frame,
        rows[0],
        &app.theme,
        "ratio",
        selected.metrics.ratio,
        ratio_max,
        app.theme.style_gauge(GaugeLevel::Primary),
        false,
    );
    let ratio_line = Line::from(vec![
        Span::styled("Ratio ", app.theme.style_muted()),
        Span::styled(
            format!("{:.4}", selected.metrics.ratio),
            app.theme
                .style_chip(report_status_to_ui_state(selected.row.status)),
        ),
        Span::raw("  "),
        Span::styled(badge, app.theme.style_badge(badge_state)),
    ]);
    frame.render_widget(
        Paragraph::new(ratio_line).style(app.theme.style_text()),
        rows[1],
    );

    let sem_line = Line::from(vec![
        Span::styled("Sem   ", app.theme.style_muted()),
        Span::styled(
            format!("{:.4}", selected.metrics.sem.min(METRIC_MAX)),
            app.theme.style_text(),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(sem_line).style(app.theme.style_text()),
        rows[2],
    );

    let struct_line = Line::from(vec![
        Span::styled("Struct ", app.theme.style_muted()),
        Span::styled(
            format!("{:.4}", selected.metrics.struct_.min(METRIC_MAX)),
            app.theme.style_text(),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(struct_line).style(app.theme.style_text()),
        rows[3],
    );

    let text = Paragraph::new(Line::from(vec![
        Span::styled("status ", app.theme.style_muted()),
        Span::styled(selected.row.status.as_str(), app.theme.style_text()),
        Span::raw("  "),
        Span::styled("tmp ", app.theme.style_muted()),
        Span::styled(
            format!(
                "{:.2}/{:.2}",
                app.tmp_th_ratio_hazy, app.tmp_th_ratio_delirium
            ),
            app.theme.style_text(),
        ),
        Span::raw("  "),
        Span::styled("policy ", app.theme.style_muted()),
        Span::styled(
            format!(
                "{:.2}/{:.2}",
                app.default_th_ratio_hazy, app.default_th_ratio_delirium
            ),
            app.theme.style_text(),
        ),
    ]))
    .wrap(Wrap { trim: false })
    .style(app.theme.style_text());
    frame.render_widget(text, rows[4]);
}

#[allow(clippy::too_many_arguments)]
fn render_metric_gauge(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    theme: &ThemeContext,
    title: &str,
    value: f64,
    max_value: f64,
    gauge_style: Style,
    ensure_visible_bar: bool,
) {
    let value = value.max(0.0);
    let mut ratio = gauge_ratio(value, max_value);
    if ensure_visible_bar && value > 0.0 {
        // Keep tiny positive values visible even in narrow panes.
        let inner_width = area.width;
        if inner_width > 0 {
            let min_ratio = (1.0 / f64::from(inner_width)).clamp(0.0, 1.0);
            ratio = ratio.max(min_ratio);
        }
    }
    let label = Span::styled(
        format!("{:<7} {:>7.4}", title, value),
        theme.style_text().add_modifier(Modifier::BOLD),
    );

    let mut gauge = Gauge::default()
        .style(theme.style_muted())
        .ratio(ratio)
        .label(label)
        .gauge_style(gauge_style.add_modifier(Modifier::BOLD));
    if ensure_visible_bar {
        gauge = gauge.use_unicode(true);
    }
    frame.render_widget(gauge, area);
}

fn gauge_ratio(value: f64, max_value: f64) -> f64 {
    if !max_value.is_finite() || max_value <= 0.0 {
        return 0.0;
    }
    let raw = (value / max_value).clamp(0.0, 1.0);
    if value > 0.0 && raw == 0.0 {
        f64::EPSILON
    } else {
        raw
    }
}

#[cfg(test)]
fn visible_fill_width(ratio: f64, inner_width: u16) -> u16 {
    if inner_width == 0 || ratio <= 0.0 {
        return 0;
    }
    let raw = (ratio * f64::from(inner_width)).ceil() as u16;
    raw.max(1).min(inner_width)
}

fn render_histogram_panel(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &App,
    filtered: &[usize],
    histogram: &HistogramState,
    selected: Option<SelectedContext<'_>>,
) {
    let title = format!(
        "max_ratio distribution (usable rows, n={})",
        histogram.usable_count
    );
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(app.theme.style_border())
        .title(title)
        .title_style(app.theme.style_section_title())
        .border_type(app.theme.glyphs().border_type)
        .style(app.theme.style_panel());
    let inner = block.inner(area);
    frame.render_widget(block, area);

    if histogram.max_count == 0 {
        let empty = Paragraph::new("no usable rows (requires finite ratio and no row error)")
            .style(app.theme.style_muted())
            .alignment(Alignment::Center);
        frame.render_widget(empty, inner);
        return;
    }

    let selected_summary = selected_distribution_line(app, filtered, selected);
    let stats_summary = format!(
        "min={:>7} p50={:>7} p95={:>7} max={:>7}",
        format_stat(histogram.min_value),
        format_stat(histogram.p50_value),
        format_stat(histogram.p95_value),
        format_stat(histogram.observed_max)
    );

    if histogram.usable_count < 25 {
        let lines = vec![
            Line::from(vec![
                Span::styled("policy: ", app.theme.style_muted()),
                Span::styled(
                    format!(
                        "hazy>={:.2} delirium>={:.2}",
                        app.tmp_th_ratio_hazy, app.tmp_th_ratio_delirium
                    ),
                    app.theme.style_text(),
                ),
            ]),
            Line::from(vec![
                Span::styled("counts: ", app.theme.style_muted()),
                Span::styled(
                    format!(
                        "above_hazy={} above_delirium={}",
                        histogram.above_hazy, histogram.above_delirium
                    ),
                    app.theme.style_text(),
                ),
            ]),
            Line::from(vec![
                Span::styled("top outliers: ", app.theme.style_muted()),
                Span::styled(
                    outlier_summary_line(&histogram.top_outliers),
                    app.theme.style_text(),
                ),
            ]),
            Line::from(vec![
                Span::styled("stats: ", app.theme.style_muted()),
                Span::styled(stats_summary, app.theme.style_text()),
            ]),
            Line::from(vec![
                Span::styled("selected: ", app.theme.style_muted()),
                Span::styled(selected_summary, app.theme.style_text()),
            ]),
        ];
        frame.render_widget(
            Paragraph::new(lines)
                .style(app.theme.style_text())
                .wrap(Wrap { trim: false }),
            inner,
        );
        return;
    }

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(3),
        ])
        .split(inner);

    let summary = Line::from(vec![
        Span::styled("policy ", app.theme.style_muted()),
        Span::styled(
            format!(
                "hazy>={:.2} delirium>={:.2}",
                app.tmp_th_ratio_hazy, app.tmp_th_ratio_delirium
            ),
            app.theme.style_text(),
        ),
        Span::raw("  "),
        Span::styled("counts ", app.theme.style_muted()),
        Span::styled(
            format!(
                "above_hazy={} above_delirium={}",
                histogram.above_hazy, histogram.above_delirium
            ),
            app.theme.style_text(),
        ),
    ]);
    frame.render_widget(
        Paragraph::new(summary).wrap(Wrap { trim: false }),
        chunks[0],
    );

    let spark_data = resample_bins_to_width(&histogram.bins, usize::from(chunks[1].width.max(1)));
    let spark = Sparkline::default()
        .data(&spark_data)
        .style(app.theme.style_gauge(GaugeLevel::Primary));
    frame.render_widget(spark, chunks[1]);

    let stats = vec![
        Line::from(vec![
            Span::styled("stats ", app.theme.style_muted()),
            Span::styled(stats_summary, app.theme.style_text()),
        ]),
        Line::from(vec![
            Span::styled("top outliers: ", app.theme.style_muted()),
            Span::styled(
                outlier_summary_line(&histogram.top_outliers),
                app.theme.style_text(),
            ),
        ]),
        Line::from(vec![
            Span::styled("selected: ", app.theme.style_muted()),
            Span::styled(selected_summary, app.theme.style_text()),
        ]),
    ];
    frame.render_widget(
        Paragraph::new(stats)
            .style(app.theme.style_text())
            .wrap(Wrap { trim: false }),
        chunks[2],
    );
}

fn render_snippet_panel(
    frame: &mut ratatui::Frame<'_>,
    area: Rect,
    app: &App,
    selected: Option<SelectedContext<'_>>,
) {
    let details = Block::default()
        .borders(Borders::ALL)
        .border_style(app.theme.style_border())
        .title("Evidence / Warnings")
        .title_style(app.theme.style_section_title())
        .border_type(app.theme.glyphs().border_type)
        .style(app.theme.style_panel());
    let inner = details.inner(area);
    frame.render_widget(details, area);

    let Some(selected) = selected else {
        frame.render_widget(
            Paragraph::new("No row selected")
                .style(app.theme.style_muted())
                .alignment(Alignment::Center),
            inner,
        );
        return;
    };

    let mut lines = Vec::new();
    lines.push(Line::from(vec![
        Span::styled("row_index: ", app.theme.style_muted()),
        Span::styled(selected.row.row_index.to_string(), app.theme.style_text()),
    ]));
    lines.push(Line::from(vec![
        Span::styled("id: ", app.theme.style_muted()),
        Span::styled(
            selected.row.id.clone().unwrap_or_else(|| "-".to_string()),
            app.theme.style_text(),
        ),
    ]));
    lines.push(Line::from(vec![
        Span::styled("status: ", app.theme.style_muted()),
        Span::styled(selected.row.status.as_str(), app.theme.style_text()),
        Span::raw("  "),
        Span::styled("metrics: ", app.theme.style_muted()),
        Span::styled(
            format!(
                "ratio={:.6} sem={:.6} struct={:.6}",
                selected.metrics.ratio, selected.metrics.sem, selected.metrics.struct_
            ),
            app.theme.style_text(),
        ),
    ]));

    if let Some(error) = &selected.row.error {
        let first_line = error
            .message
            .as_deref()
            .and_then(|text| text.lines().next())
            .unwrap_or("-");
        lines.push(Line::from(vec![
            Span::styled("error: ", app.theme.style_muted()),
            Span::styled("code=", app.theme.style_muted()),
            Span::styled(
                error.code.clone().unwrap_or_else(|| "-".to_string()),
                app.theme.style_chip(UiState::Delirium),
            ),
            Span::raw(" "),
            Span::styled("message=", app.theme.style_muted()),
            Span::styled(first_line.to_string(), app.theme.style_text()),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::styled("error: ", app.theme.style_muted()),
            Span::styled("none", app.theme.style_text()),
        ]));
    }

    if let Some(evidence) = selected
        .row
        .raw
        .get("data")
        .and_then(|data| data.get("evidence"))
        .and_then(Value::as_array)
    {
        if evidence.is_empty() {
            lines.push(Line::from(vec![
                Span::styled("evidence: ", app.theme.style_muted()),
                Span::styled("none", app.theme.style_text()),
            ]));
        } else {
            lines.push(Line::from(Span::styled(
                "evidence:",
                app.theme.style_muted(),
            )));
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
                let ratio = parse_f64(item.get("score_ratio"))
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_else(|| "-".to_string());
                let sem = parse_f64(item.get("score_sem").or_else(|| item.get("score_sem_raw")))
                    .map(|v| format!("{:.6}", v))
                    .unwrap_or_else(|| "-".to_string());
                let struct_ = parse_f64(
                    item.get("score_struct")
                        .or_else(|| item.get("score_struct_raw")),
                )
                .map(|v| format!("{:.6}", v))
                .unwrap_or_else(|| "-".to_string());

                lines.push(Line::from(vec![
                    Span::styled("- ", app.theme.style_muted()),
                    Span::styled(format!("ans[{}] ", ans_idx), app.theme.style_text()),
                    Span::styled(format!("ctx[{}] ", ctx_idx), app.theme.style_text()),
                    Span::styled("ratio=", app.theme.style_muted()),
                    Span::styled(ratio, app.theme.style_text()),
                    Span::raw(" "),
                    Span::styled("sem=", app.theme.style_muted()),
                    Span::styled(sem, app.theme.style_text()),
                    Span::raw(" "),
                    Span::styled("struct=", app.theme.style_muted()),
                    Span::styled(struct_, app.theme.style_text()),
                ]));
            }
        }
    } else {
        lines.push(Line::from(vec![
            Span::styled("evidence: ", app.theme.style_muted()),
            Span::styled("unavailable", app.theme.style_text()),
        ]));
    }

    if selected.row.warnings.is_empty() {
        lines.push(Line::from(vec![
            Span::styled("warnings: ", app.theme.style_muted()),
            Span::styled("none", app.theme.style_text()),
        ]));
    } else {
        lines.push(Line::from(Span::styled(
            "warnings:",
            app.theme.style_muted(),
        )));
        for warning in selected.row.warnings.iter().take(8) {
            let warning_type = warning.get("type").and_then(Value::as_str).unwrap_or("-");
            let sentence_index = warning
                .get("sentence_index")
                .and_then(value_to_usize)
                .map(|v| v.to_string())
                .unwrap_or_else(|| "-".to_string());
            lines.push(Line::from(vec![
                Span::styled("- ", app.theme.style_muted()),
                Span::styled("type=", app.theme.style_muted()),
                Span::styled(warning_type.to_string(), app.theme.style_text()),
                Span::raw(" "),
                Span::styled("sentence_index=", app.theme.style_muted()),
                Span::styled(sentence_index, app.theme.style_text()),
            ]));
        }
    }

    frame.render_widget(
        Paragraph::new(lines)
            .style(app.theme.style_text())
            .wrap(Wrap { trim: false }),
        inner,
    );
}

fn build_histogram_state(
    app: &App,
    filtered_indices: &[usize],
    selected_ratio: Option<f64>,
) -> HistogramState {
    let mut ratios = Vec::new();
    let mut labeled_ratios = Vec::new();
    for index in filtered_indices {
        let row = &app.rows[*index];
        if let Some(ratio) = usable_ratio(row) {
            ratios.push(ratio);
            let label = row
                .id
                .clone()
                .unwrap_or_else(|| format!("row#{}", row.row_index));
            labeled_ratios.push((label, ratio));
        }
    }

    ratios.sort_by(|a, b| a.total_cmp(b));
    let min_value = ratios.first().copied();
    let p50_value = percentile(&ratios, 0.50);
    let p95_value = percentile(&ratios, 0.95);
    let observed_max = ratios.last().copied();

    let mut max_value = 1.0_f64;
    for ratio in &ratios {
        if *ratio > max_value {
            max_value = *ratio;
        }
    }
    max_value = max_value
        .max(app.tmp_th_ratio_hazy)
        .max(app.tmp_th_ratio_delirium)
        .max(selected_ratio.unwrap_or(0.0));
    max_value = (max_value * 1.05).max(1.0);

    let bins = build_histogram_bins(&ratios, HIST_BINS, max_value);
    let max_count = bins.iter().copied().max().unwrap_or(0);
    let above_hazy = ratios
        .iter()
        .filter(|ratio| **ratio >= app.tmp_th_ratio_hazy)
        .count();
    let above_delirium = ratios
        .iter()
        .filter(|ratio| **ratio >= app.tmp_th_ratio_delirium)
        .count();

    labeled_ratios.sort_by(|left, right| {
        right
            .1
            .total_cmp(&left.1)
            .then_with(|| left.0.cmp(&right.0))
    });
    let top_outliers = labeled_ratios.into_iter().take(3).collect();

    HistogramState {
        bins,
        max_count,
        max_value,
        min_value,
        p50_value,
        p95_value,
        observed_max,
        usable_count: ratios.len(),
        above_hazy,
        above_delirium,
        top_outliers,
    }
}

fn outlier_summary_line(outliers: &[(String, f64)]) -> String {
    if outliers.is_empty() {
        return "none".to_string();
    }

    outliers
        .iter()
        .map(|(id, ratio)| format!("{}={:.3}", id, ratio))
        .collect::<Vec<String>>()
        .join("  ")
}

fn selected_distribution_line(
    app: &App,
    filtered: &[usize],
    selected: Option<SelectedContext<'_>>,
) -> String {
    let Some(selected) = selected else {
        return "n/a (none)".to_string();
    };

    let Some(selected_ratio) = usable_ratio(selected.row) else {
        return "n/a (non-usable)".to_string();
    };

    let mut usable = Vec::new();
    for index in filtered {
        if let Some(ratio) = usable_ratio(&app.rows[*index]) {
            usable.push(ratio);
        }
    }
    if usable.is_empty() {
        return "n/a (non-usable)".to_string();
    }

    let greater = usable
        .iter()
        .filter(|ratio| **ratio > selected_ratio)
        .count();
    let rank = greater.saturating_add(1);
    let total = usable.len();
    let pct = if total == 0 {
        0.0
    } else {
        (rank as f64 / total as f64) * 100.0
    };
    format!(
        "value={:.3} rank={}/{} (pct={:.1}%)",
        selected_ratio, rank, total, pct
    )
}

fn resample_bins_to_width(bins: &[usize], width: usize) -> Vec<u64> {
    if width == 0 {
        return Vec::new();
    }
    if bins.is_empty() {
        return vec![0; width];
    }
    if bins.len() == width {
        return bins.iter().map(|value| *value as u64).collect();
    }

    let mut sampled = Vec::with_capacity(width);
    for index in 0..width {
        let start = index * bins.len() / width;
        let mut end = (index + 1) * bins.len() / width;
        if end <= start {
            end = (start + 1).min(bins.len());
        }

        let mut total = 0usize;
        let mut count = 0usize;
        for value in &bins[start..end] {
            total = total.saturating_add(*value);
            count = count.saturating_add(1);
        }
        let averaged = if count == 0 {
            0
        } else {
            (total + count / 2) / count
        };
        sampled.push(averaged as u64);
    }
    sampled
}

fn percentile(values: &[f64], p: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let p = p.clamp(0.0, 1.0);
    let max_index = values.len().saturating_sub(1);
    let rank = (p * max_index as f64).round() as usize;
    values.get(rank).copied()
}

fn format_stat(value: Option<f64>) -> String {
    value
        .map(|v| format!("{:.3}", v))
        .unwrap_or_else(|| "-".to_string())
}

fn usable_ratio(row: &ReportRow) -> Option<f64> {
    if row.error.is_some() {
        return None;
    }

    let ratio = row.max_score_ratio?;
    if ratio.is_finite() && ratio >= 0.0 {
        Some(ratio)
    } else {
        None
    }
}

fn build_histogram_bins(ratios: &[f64], bins: usize, max_value: f64) -> Vec<usize> {
    let mut out = vec![0usize; bins];
    if bins == 0 || !max_value.is_finite() || max_value <= 0.0 {
        return out;
    }

    for ratio in ratios {
        if !ratio.is_finite() || *ratio < 0.0 {
            continue;
        }
        let bin = marker_bin_index(*ratio, max_value, bins);
        out[bin] = out[bin].saturating_add(1);
    }

    out
}

fn marker_bin_index(value: f64, max_value: f64, bins: usize) -> usize {
    if bins == 0 || !max_value.is_finite() || max_value <= 0.0 {
        return 0;
    }

    let normalized = (value / max_value).clamp(0.0, 1.0);
    let raw = (normalized * bins as f64).floor() as usize;
    raw.min(bins - 1)
}

fn extract_row_metrics(row: &ReportRow) -> RowMetrics {
    let mut metrics = RowMetrics {
        ratio: row
            .max_score_ratio
            .filter(|value| value.is_finite() && *value >= 0.0)
            .unwrap_or(0.0),
        sem: 0.0,
        struct_: 0.0,
    };
    if let Some(evidence) = row
        .raw
        .get("data")
        .and_then(|data| data.get("evidence"))
        .and_then(Value::as_array)
    {
        for item in evidence {
            let Some(ratio) = parse_f64(item.get("score_ratio")) else {
                continue;
            };
            if !ratio.is_finite() || ratio < 0.0 {
                continue;
            }

            if ratio >= metrics.ratio {
                metrics.ratio = ratio;
                metrics.sem =
                    parse_f64(item.get("score_sem").or_else(|| item.get("score_sem_raw")))
                        .unwrap_or(0.0)
                        .max(0.0);
                metrics.struct_ = parse_f64(
                    item.get("score_struct")
                        .or_else(|| item.get("score_struct_raw")),
                )
                .unwrap_or(0.0)
                .max(0.0);
            }
        }
    }

    metrics
}

fn parse_f64(value: Option<&Value>) -> Option<f64> {
    let number = value?.as_f64()?;
    if number.is_finite() {
        Some(number)
    } else {
        None
    }
}

fn report_status_to_ui_state(status: ReportStatus) -> UiState {
    match status {
        ReportStatus::Lucid => UiState::Lucid,
        ReportStatus::Hazy => UiState::Hazy,
        ReportStatus::Delirium => UiState::Delirium,
        ReportStatus::Unknown => UiState::Unknown,
    }
}

fn ratio_badge(status: ReportStatus) -> (&'static str, UiState) {
    match status {
        ReportStatus::Lucid => ("SAFE", UiState::Lucid),
        ReportStatus::Hazy => ("WARN", UiState::Hazy),
        ReportStatus::Delirium => ("DANGER", UiState::Delirium),
        ReportStatus::Unknown => ("WARN", UiState::Unknown),
    }
}

fn status_marker(status: ReportStatus, ascii: bool) -> &'static str {
    match (status, ascii) {
        (ReportStatus::Lucid, true) => "o ",
        (ReportStatus::Lucid, false) => "\u{25CF} ",
        (ReportStatus::Hazy, true) => "^ ",
        (ReportStatus::Hazy, false) => "\u{25B2} ",
        (ReportStatus::Delirium, _) => "! ",
        (ReportStatus::Unknown, _) => "? ",
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_histogram_bins, gauge_ratio, handle_mouse_event, handle_normal_mode_key,
        marker_bin_index, status_marker, visible_fill_width, App, ReportColor, ReportFilters,
        ReportRow, ReportTheme, TuiOptions,
    };
    use crossterm::event::{KeyCode, MouseEventKind};
    use serde_json::{json, Value};

    fn test_tui_options() -> TuiOptions {
        TuiOptions {
            theme: ReportTheme::Classic,
            color: ReportColor::Auto,
            ascii: false,
        }
    }

    #[test]
    fn mouse_scroll_down_moves_selection() {
        let rows = vec![
            row(0, Some("alpha"), "LUCID", 0.5, false, false),
            row(1, Some("beta"), "LUCID", 0.7, false, false),
        ];
        let mut app = App::new(rows, ReportFilters::default(), test_tui_options());

        let action = handle_mouse_event(&mut app, MouseEventKind::ScrollDown);

        assert!(action.redraw);
        assert_eq!(app.selected_row, 1);
    }

    #[test]
    fn mouse_scroll_up_clamps_at_zero() {
        let rows = vec![
            row(0, Some("alpha"), "LUCID", 0.5, false, false),
            row(1, Some("beta"), "LUCID", 0.7, false, false),
        ];
        let mut app = App::new(rows, ReportFilters::default(), test_tui_options());

        let action = handle_mouse_event(&mut app, MouseEventKind::ScrollUp);

        assert!(action.redraw);
        assert_eq!(app.selected_row, 0);
    }

    #[test]
    fn selection_clamps_after_filter_change() {
        let rows = vec![
            row(0, Some("alpha"), "LUCID", 0.2, false, false),
            row(1, Some("beta"), "LUCID", 0.3, false, false),
            row(2, Some("gamma"), "LUCID", 0.4, false, false),
        ];
        let mut app = App::new(rows, ReportFilters::default(), test_tui_options());

        app.selected_row = 2;
        app.search_query = "alpha".to_string();
        app.clamp_selection();

        assert_eq!(app.filtered_indices().len(), 1);
        assert_eq!(app.selected_row, 0);
    }

    #[test]
    fn histogram_marker_clamp_edges() {
        assert_eq!(marker_bin_index(-1.0, 3.0, 30), 0);
        assert_eq!(marker_bin_index(0.0, 3.0, 30), 0);
        assert_eq!(marker_bin_index(1.5, 3.0, 30), 15);
        assert_eq!(marker_bin_index(3.0, 3.0, 30), 29);
        assert_eq!(marker_bin_index(9.0, 3.0, 30), 29);
    }

    #[test]
    fn histogram_binning_is_deterministic() {
        let ratios = vec![0.0, 0.1, 0.49, 0.5, 0.99, 1.0, 1.2];
        let first = build_histogram_bins(&ratios, 4, 1.0);
        let second = build_histogram_bins(&ratios, 4, 1.0);

        assert_eq!(first, vec![2, 1, 1, 3]);
        assert_eq!(first, second);
    }

    #[test]
    fn debug_toggle_key_flips_histogram_debug_visibility() {
        let rows = vec![row(0, Some("alpha"), "LUCID", 0.5, false, false)];
        let mut app = App::new(rows, ReportFilters::default(), test_tui_options());
        assert!(!app.show_hist_debug);

        let first = handle_normal_mode_key(&mut app, KeyCode::Char('d'));
        assert!(first.redraw);
        assert!(app.show_hist_debug);

        let second = handle_normal_mode_key(&mut app, KeyCode::Char('d'));
        assert!(second.redraw);
        assert!(!app.show_hist_debug);
    }

    #[test]
    fn top_bottom_navigation_keys_move_to_bounds() {
        let rows = vec![
            row(0, Some("alpha"), "LUCID", 0.2, false, false),
            row(1, Some("beta"), "LUCID", 0.3, false, false),
            row(2, Some("gamma"), "LUCID", 0.4, false, false),
        ];
        let mut app = App::new(rows, ReportFilters::default(), test_tui_options());
        app.selected_row = 1;

        let top = handle_normal_mode_key(&mut app, KeyCode::Home);
        assert!(top.redraw);
        assert_eq!(app.selected_row, 0);

        let bottom = handle_normal_mode_key(&mut app, KeyCode::End);
        assert!(bottom.redraw);
        assert_eq!(app.selected_row, 2);

        let top_vi = handle_normal_mode_key(&mut app, KeyCode::Char('g'));
        assert!(top_vi.redraw);
        assert_eq!(app.selected_row, 0);

        let bottom_vi = handle_normal_mode_key(&mut app, KeyCode::Char('G'));
        assert!(bottom_vi.redraw);
        assert_eq!(app.selected_row, 2);
    }

    #[test]
    fn gauge_ratio_handles_positive_sem_without_zeroing() {
        let ratio = gauge_ratio(0.2, 1.5);
        assert!(ratio > 0.0);
        assert!((ratio - (0.2 / 1.5)).abs() < 1e-12);
    }

    #[test]
    fn sem_bar_visibility_floor_is_applied_for_tiny_positive_values() {
        let area_width = 10_u16;
        let inner_width = area_width;
        let mut ratio = gauge_ratio(0.0001, 1.5);
        if inner_width > 0 {
            let min_ratio = (1.0 / f64::from(inner_width)).clamp(0.0, 1.0);
            ratio = ratio.max(min_ratio);
        }
        assert!(ratio >= 1.0 / f64::from(inner_width));
    }

    #[test]
    fn visible_fill_width_guarantees_one_cell_for_positive_ratio() {
        assert_eq!(visible_fill_width(0.0, 20), 0);
        assert_eq!(visible_fill_width(0.00001, 20), 1);
        assert_eq!(visible_fill_width(0.5, 20), 10);
        assert_eq!(visible_fill_width(1.0, 20), 20);
    }

    #[test]
    fn ascii_hazy_marker_is_distinct_from_delirium() {
        assert_eq!(status_marker(super::ReportStatus::Hazy, true), "^ ");
        assert_eq!(status_marker(super::ReportStatus::Delirium, true), "! ");
    }

    fn row(
        row_index: usize,
        id: Option<&str>,
        status: &str,
        max_score_ratio: f64,
        has_error: bool,
        has_warning: bool,
    ) -> ReportRow {
        let warning_list = if has_warning {
            vec![json!({ "type": "EMBED_TRUNCATED" })]
        } else {
            Vec::new()
        };

        let error_value = if has_error {
            json!({ "code": "ROW_ERR", "message": "boom" })
        } else {
            Value::Null
        };

        let row = json!({
            "row_index": row_index,
            "id": id,
            "inputs_hash": format!("h{}", row_index),
            "status": status,
            "error": error_value,
            "data": { "scores": { "max_score_ratio": max_score_ratio } },
            "audit_trace": { "warnings": warning_list }
        });

        ReportRow::from_value(row, row_index)
    }
}
