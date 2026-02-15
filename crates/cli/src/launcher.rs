use crate::report::ReportTheme;
use crate::target_resolver::{
    ResolveError, ResolveRequest, ResolvedTarget, StatusCountsPreview, TargetCheck, TargetResolver,
};
use chrono::{DateTime, Local};
use crossterm::cursor::MoveTo;
use crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use crossterm::style::{style, Attribute, Color, Stylize};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode, Clear, ClearType};
use crossterm::tty::IsTty;
use crossterm::ExecutableCommand;
use pale_ale_diagnose::default_policy_config;
use std::env;
use std::io::{self, Write};
use std::path::Path;
use std::time::SystemTime;

const LAUNCHER_AA: &[&str] = &[
    "eeeee eeeee e     eeee      eeeee e     eeee",
    "8   8 8   8 8     8         8   8 8     8",
    "8eee8 8eee8 8e    8eee eeee 8eee8 8e    8eee",
    "88    88  8 88    88        88  8 88    88",
    "88    88  8 88eee 88ee      88  8 88eee 88ee",
];
const THEME_ENV_KEY: &str = "PALE_ALE_THEME";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LauncherTheme {
    Classic,
    Term,
    Cyber,
}

impl LauncherTheme {
    fn cycle(self) -> Self {
        match self {
            Self::Classic => Self::Term,
            Self::Term => Self::Cyber,
            Self::Cyber => Self::Classic,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Classic => "classic",
            Self::Term => "term",
            Self::Cyber => "cyber",
        }
    }

    fn to_report_theme(self) -> ReportTheme {
        match self {
            Self::Classic => ReportTheme::Classic,
            Self::Term => ReportTheme::Term,
            Self::Cyber => ReportTheme::Cyber,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LauncherColorMode {
    TrueColor,
    Ansi16,
    None,
}

#[derive(Clone, Copy, Debug)]
struct LauncherStyles {
    theme: LauncherTheme,
    color_mode: LauncherColorMode,
}

impl LauncherStyles {
    fn new(theme: LauncherTheme, color_mode: LauncherColorMode) -> Self {
        Self { theme, color_mode }
    }

    fn title(&self, text: &str) -> String {
        if self.theme == LauncherTheme::Term {
            self.paint(text, self.value_color(), true, false)
        } else {
            self.paint(text, self.accent_color(), true, false)
        }
    }

    fn section(&self, text: &str) -> String {
        if self.theme == LauncherTheme::Term {
            self.paint(text, self.value_color(), true, false)
        } else {
            self.paint(text, self.accent_color(), true, false)
        }
    }

    fn section_label(&self, text: &str) -> String {
        match self.theme {
            LauncherTheme::Classic => format!("{} \u{00B7}", text),
            LauncherTheme::Term => text.to_string(),
            LauncherTheme::Cyber => format!("[{}]", text.to_ascii_uppercase()),
        }
    }

    fn key(&self, text: &str) -> String {
        self.paint(text, self.key_color(), false, true)
    }

    fn value(&self, text: &str) -> String {
        self.paint(text, self.value_color(), false, false)
    }

    fn aa_line(&self, text: &str) -> String {
        match self.theme {
            LauncherTheme::Term => {
                if self.color_mode == LauncherColorMode::None {
                    self.paint(text, self.value_color(), true, false)
                } else {
                    self.paint(text, self.term_aa_color(), true, false)
                }
            }
            _ => self.paint(text, self.value_color(), true, false),
        }
    }

    fn critical(&self, text: &str) -> String {
        self.paint(text, self.value_color(), true, false)
    }

    fn ok_word(&self, text: &str) -> String {
        if self.theme == LauncherTheme::Term {
            self.value(text)
        } else if self.color_mode == LauncherColorMode::None {
            self.critical(text)
        } else {
            self.paint(text, self.accent_color(), true, false)
        }
    }

    fn danger_word(&self, text: &str) -> String {
        if self.theme == LauncherTheme::Term || self.color_mode == LauncherColorMode::None {
            self.value(text)
        } else {
            self.paint(text, self.danger_color(), false, false)
        }
    }

    fn danger_token_reversed(&self, text: &str) -> String {
        let mut styled = style(text.to_string())
            .attribute(Attribute::Bold)
            .attribute(Attribute::Reverse);
        if self.theme != LauncherTheme::Term && self.color_mode != LauncherColorMode::None {
            if let Some(color) = self.danger_color() {
                styled = styled.with(color);
            }
        }
        format!("{}", styled)
    }

    fn theme_label(&self) -> &'static str {
        self.theme.as_str()
    }

    fn theme_tag(&self) -> String {
        let token = format!("[theme={}]", self.theme_label());
        match self.theme {
            LauncherTheme::Term => self.critical(&token),
            _ => self.section(&token),
        }
    }

    fn signal_line(&self) -> Option<String> {
        let line = "\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}\u{2500}";
        match self.theme {
            LauncherTheme::Classic => Some(self.key(line)),
            LauncherTheme::Term => Some(self.key(line)),
            LauncherTheme::Cyber => {
                let dot = self.danger_word("\u{25C9}");
                let right = self.title(line);
                Some(format!("{}{}", dot, right))
            }
        }
    }

    fn paint(&self, text: &str, color: Option<Color>, bold: bool, dim: bool) -> String {
        let mut styled = style(text.to_string());
        if let Some(color) = color {
            styled = styled.with(color);
        }
        if bold {
            styled = styled.attribute(Attribute::Bold);
        }
        if dim {
            styled = styled.attribute(Attribute::Dim);
        }
        format!("{}", styled)
    }

    fn accent_color(&self) -> Option<Color> {
        match self.color_mode {
            LauncherColorMode::None => None,
            LauncherColorMode::Ansi16 => Some(match self.theme {
                LauncherTheme::Classic => Color::Yellow,
                LauncherTheme::Term => Color::Cyan,
                LauncherTheme::Cyber => Color::Cyan,
            }),
            LauncherColorMode::TrueColor => Some(match self.theme {
                LauncherTheme::Classic => Color::Rgb {
                    r: 0xD6,
                    g: 0xB4,
                    b: 0x6A,
                },
                LauncherTheme::Term => Color::Rgb {
                    r: 0x87,
                    g: 0xAA,
                    b: 0xCF,
                },
                LauncherTheme::Cyber => Color::Rgb {
                    r: 0x52,
                    g: 0xD1,
                    b: 0xFF,
                },
            }),
        }
    }

    fn danger_color(&self) -> Option<Color> {
        match self.color_mode {
            LauncherColorMode::None => None,
            LauncherColorMode::Ansi16 => Some(match self.theme {
                LauncherTheme::Cyber => Color::Magenta,
                _ => Color::Red,
            }),
            LauncherColorMode::TrueColor => Some(match self.theme {
                LauncherTheme::Classic => Color::Rgb {
                    r: 0xBD,
                    g: 0x7B,
                    b: 0x82,
                },
                LauncherTheme::Term => Color::Rgb {
                    r: 0xB8,
                    g: 0x80,
                    b: 0x86,
                },
                LauncherTheme::Cyber => Color::Rgb {
                    r: 0xE0,
                    g: 0x67,
                    b: 0xD0,
                },
            }),
        }
    }

    fn value_color(&self) -> Option<Color> {
        match self.color_mode {
            LauncherColorMode::None => None,
            LauncherColorMode::Ansi16 => Some(match self.theme {
                LauncherTheme::Term => Color::White,
                _ => Color::Grey,
            }),
            LauncherColorMode::TrueColor => Some(match self.theme {
                LauncherTheme::Term => Color::Rgb {
                    r: 0xE2,
                    g: 0xE8,
                    b: 0xF0,
                },
                _ => Color::Rgb {
                    r: 0xCF,
                    g: 0xD6,
                    b: 0xE1,
                },
            }),
        }
    }

    fn key_color(&self) -> Option<Color> {
        match self.color_mode {
            LauncherColorMode::None => None,
            LauncherColorMode::Ansi16 => Some(Color::DarkGrey),
            LauncherColorMode::TrueColor => Some(Color::Rgb {
                r: 0x8D,
                g: 0x98,
                b: 0xAA,
            }),
        }
    }

    fn term_aa_color(&self) -> Option<Color> {
        match self.color_mode {
            LauncherColorMode::None => None,
            LauncherColorMode::Ansi16 => Some(Color::DarkBlue),
            LauncherColorMode::TrueColor => Some(Color::Rgb {
                r: 0x5A,
                g: 0x6E,
                b: 0x94,
            }),
        }
    }
}

pub(super) enum LauncherAction {
    Launch {
        target: ResolvedTarget,
        theme: ReportTheme,
    },
    Quit,
}

pub(super) fn stdio_is_tty() -> bool {
    io::stdin().is_tty() && io::stdout().is_tty()
}

pub(super) fn run_launcher(resolver: &TargetResolver) -> Result<LauncherAction, String> {
    let mut theme = resolve_theme(resolver);
    let color_mode = detect_color_mode();
    let mut styles = LauncherStyles::new(theme, color_mode);

    let detected = match resolver.resolve(ResolveRequest::default()) {
        Ok(target) => Some(target),
        Err(ResolveError::Unresolved(_)) => None,
        Err(ResolveError::InvalidTarget(_)) => None,
    };
    render_launcher_screen(&styles, resolver, detected.as_ref())?;

    loop {
        let prompt = if detected.is_some() {
            "launcher> [Enter=open, /=target, r=recent, t=theme, ?=help, q=quit] "
        } else {
            "launcher> [/=target, r=recent, t=theme, ?=help, q=quit] "
        };
        let input = read_line(prompt)?;
        let trimmed = input.trim();

        if trimmed.eq_ignore_ascii_case("q") || trimmed.eq_ignore_ascii_case("quit") {
            return Ok(LauncherAction::Quit);
        }

        if trimmed.is_empty() {
            if let Some(target) = detected.clone() {
                return Ok(LauncherAction::Launch {
                    target,
                    theme: theme.to_report_theme(),
                });
            }
            match prompt_for_target_with_styles(resolver, &styles)? {
                Some(target) => {
                    return Ok(LauncherAction::Launch {
                        target,
                        theme: theme.to_report_theme(),
                    });
                }
                None => return Ok(LauncherAction::Quit),
            }
        }

        if trimmed == "/" {
            match prompt_for_target_with_styles(resolver, &styles)? {
                Some(target) => {
                    return Ok(LauncherAction::Launch {
                        target,
                        theme: theme.to_report_theme(),
                    });
                }
                None => continue,
            }
        }

        if trimmed.eq_ignore_ascii_case("r") {
            match prompt_recent_target(resolver, &styles)? {
                Some(target) => {
                    return Ok(LauncherAction::Launch {
                        target,
                        theme: theme.to_report_theme(),
                    });
                }
                None => continue,
            }
        }

        if trimmed.eq_ignore_ascii_case("t") {
            theme = theme.cycle();
            styles = LauncherStyles::new(theme, color_mode);
            if let Err(err) = resolver.persist_last_theme(theme.as_str()) {
                eprintln!(
                    "{} {}",
                    styles.key("warning: failed to persist last_theme:"),
                    styles.value(&err)
                );
            }
            render_launcher_screen(&styles, resolver, detected.as_ref())?;
            continue;
        }

        if trimmed == "?" {
            print_help(&styles);
            continue;
        }

        println!(
            "{} {}",
            styles.key("unknown command:"),
            styles.value(&format!("{:?}", trimmed))
        );
        println!(
            "{}",
            styles.key("This is not a shell. Press '/' to enter target path.")
        );
    }
}

pub(super) fn prompt_for_target(
    resolver: &TargetResolver,
) -> Result<Option<ResolvedTarget>, String> {
    let styles = LauncherStyles::new(resolve_theme(resolver), detect_color_mode());
    prompt_for_target_with_styles(resolver, &styles)
}

fn prompt_for_target_with_styles(
    resolver: &TargetResolver,
    styles: &LauncherStyles,
) -> Result<Option<ResolvedTarget>, String> {
    println!(
        "{}",
        styles.key("Enter target path (run bundle dir or .ndjson). Esc/q to cancel.")
    );
    loop {
        let Some(input) = read_target_line_modal("target> ")? else {
            return Ok(None);
        };
        let trimmed = input.trim();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed.eq_ignore_ascii_case("q") || trimmed.eq_ignore_ascii_case("quit") {
            return Ok(None);
        }

        match resolver.resolve(ResolveRequest {
            explicit_target: Some(trimmed.to_string()),
        }) {
            Ok(target) => return Ok(Some(target)),
            Err(ResolveError::InvalidTarget(message)) => {
                eprintln!(
                    "{} {}",
                    styles.danger_word("invalid target:"),
                    styles.value(&message)
                );
            }
            Err(ResolveError::Unresolved(message)) => {
                eprintln!("{}", styles.value(&message));
            }
        }
    }
}

fn prompt_recent_target(
    resolver: &TargetResolver,
    styles: &LauncherStyles,
) -> Result<Option<ResolvedTarget>, String> {
    let recents = resolver.recent_targets();
    if recents.is_empty() {
        println!("{} {}", styles.key("recent:"), styles.value("none"));
        return Ok(None);
    }

    println!(
        "{}",
        styles.section(&styles.section_label("Recent Targets"))
    );
    for (index, path) in recents.iter().enumerate() {
        let inspection = resolver.inspect_path(path);
        let path_text = shorten_text(&display_path(path), 72);
        let check = render_check_word(styles, inspection.check);
        println!(
            "  {} {} [{}]",
            styles.key(&format!("{}.", index + 1)),
            styles.value(&path_text),
            check
        );
    }
    println!(
        "{} {}",
        styles.key("Choose index:"),
        styles.value(&format!("1-{} (q to cancel)", recents.len()))
    );

    loop {
        let input = read_line("recent> ")?;
        let trimmed = input.trim();
        if trimmed.eq_ignore_ascii_case("q") || trimmed.eq_ignore_ascii_case("quit") {
            return Ok(None);
        }
        let Ok(index) = trimmed.parse::<usize>() else {
            println!(
                "{} {}",
                styles.danger_word("invalid choice:"),
                styles.value(trimmed)
            );
            continue;
        };
        if index == 0 || index > recents.len() {
            println!(
                "{} {}",
                styles.danger_word("out of range:"),
                styles.value(&index.to_string())
            );
            continue;
        }

        let target_text = recents[index - 1].display().to_string();
        match resolver.resolve(ResolveRequest {
            explicit_target: Some(target_text),
        }) {
            Ok(target) => return Ok(Some(target)),
            Err(ResolveError::InvalidTarget(message)) => {
                println!(
                    "{} {}",
                    styles.danger_word("invalid target:"),
                    styles.value(&message)
                );
                return Ok(None);
            }
            Err(ResolveError::Unresolved(message)) => {
                println!("{}", styles.value(&message));
                return Ok(None);
            }
        }
    }
}

fn print_banner(styles: &LauncherStyles) {
    for line in LAUNCHER_AA {
        println!(" {}", styles.aa_line(line));
    }
    println!();
    let title = styles.title(&format!(
        "Pale Ale Ops Cockpit Launcher  v{}",
        env!("CARGO_PKG_VERSION")
    ));
    println!("{} {}", title, styles.theme_tag());
    if let Some(signal) = styles.signal_line() {
        println!("{}", signal);
    }
    println!(
        "{}",
        styles.key("This is not a shell. Use '/' to enter a target path.")
    );
}

fn render_launcher_screen(
    styles: &LauncherStyles,
    resolver: &TargetResolver,
    target: Option<&ResolvedTarget>,
) -> Result<(), String> {
    clear_and_home()?;
    print_banner(styles);
    print_target_card(styles, resolver, target);
    print_actions(styles, resolver, target);
    Ok(())
}

fn clear_and_home() -> Result<(), String> {
    let mut stdout = io::stdout();
    stdout
        .execute(Clear(ClearType::All))
        .and_then(|stream| stream.execute(MoveTo(0, 0)))
        .map_err(|err| format!("failed to clear launcher screen: {}", err))?;
    stdout
        .flush()
        .map_err(|err| format!("failed to flush launcher screen: {}", err))?;
    Ok(())
}

fn print_target_card(
    styles: &LauncherStyles,
    resolver: &TargetResolver,
    target: Option<&ResolvedTarget>,
) {
    match target {
        Some(resolved) => {
            let inspection = resolver.inspect_path(&resolved.target);
            let canonical = inspection
                .canonical
                .as_ref()
                .unwrap_or(&inspection.requested);
            let canonical_display = display_path(canonical);
            let short_display = shorten_text(&canonical_display, 72);
            println!(
                "{}",
                styles.section(&styles.section_label("Detected Target"))
            );
            let row_hint = inspection
                .preview
                .as_ref()
                .map(|preview| {
                    if preview.sampled_rows > 0 {
                        if preview.truncated {
                            format!(
                                " (sample {} rows / {} B, truncated)",
                                preview.sampled_rows, preview.sampled_bytes
                            )
                        } else {
                            format!(
                                " (sample {} rows / {} B)",
                                preview.sampled_rows, preview.sampled_bytes
                            )
                        }
                    } else {
                        String::new()
                    }
                })
                .unwrap_or_default();
            println!(
                "  {} {}",
                styles.key("target:"),
                styles.critical(&format!("{}{}", path_tail(canonical), row_hint))
            );
            println!("  {} {}", styles.key("path:"), styles.value(&short_display));
            println!(
                "  {} {}   {} {}   {} {}",
                styles.key("source:"),
                styles.value(resolved.source.as_str()),
                styles.key("kind:"),
                styles.value(inspection.kind.as_str()),
                styles.key("check:"),
                render_check_word(styles, inspection.check)
            );
            let mut metadata_line = format!("  {} ", styles.key("meta:"));
            if let Some(preview) = inspection.preview.as_ref() {
                let mtime = preview
                    .modified
                    .map(format_local_time)
                    .unwrap_or_else(|| "unknown".to_string());
                let size = preview
                    .size_bytes
                    .map(format_size)
                    .unwrap_or_else(|| "unknown".to_string());
                metadata_line.push_str(&format!(
                    "{} {}   {} {}",
                    styles.key("mtime"),
                    styles.value(&mtime),
                    styles.key("size"),
                    styles.value(&size)
                ));
            } else {
                metadata_line.push_str(&format!(
                    "{} {}   {} {}",
                    styles.key("mtime"),
                    styles.value("unknown"),
                    styles.key("size"),
                    styles.value("unknown")
                ));
            }
            println!("{}", metadata_line);

            if let Some(preview) = inspection.preview.as_ref() {
                if let Some(counts) = preview.counts.as_ref() {
                    print_status_counts(styles, counts);
                }
            }
            let (policy_hazy, policy_delirium) =
                launcher_policy_thresholds(inspection.preview.as_ref());
            println!(
                "  {} {}",
                styles.key("policy:"),
                styles.value(&format!(
                    "hazy>={:.2} delirium>={:.2}",
                    policy_hazy, policy_delirium
                ))
            );

            if let Some(hint) = inspection.hint {
                println!(
                    "  {} {}",
                    styles.key("hint:"),
                    if inspection.check == TargetCheck::Ok {
                        styles.value(&hint)
                    } else {
                        styles.danger_word(&hint)
                    }
                );
            }
        }
        None => {
            println!(
                "{}",
                styles.section(&styles.section_label("Detected Target"))
            );
            println!("  {} {}", styles.key("path:"), styles.value("none"));
            println!(
                "  {} {}",
                styles.key("check:"),
                render_check_word(styles, TargetCheck::Missing)
            );
        }
    }
    println!();
}

fn print_status_counts(styles: &LauncherStyles, counts: &StatusCountsPreview) {
    let long_counts = format!(
        "LUCID {} HAZY {} DELIRIUM {} UNKNOWN {}",
        counts.lucid, counts.hazy, counts.delirium, counts.unknown
    );
    let compact = [
        render_count_badge(styles, '\u{25CF}', "L", counts.lucid),
        render_count_badge(styles, '\u{25B2}', "H", counts.hazy),
        render_count_badge(styles, '!', "D", counts.delirium),
        render_count_badge(styles, '?', "U", counts.unknown),
    ]
    .join(" ");
    println!(
        "  {} {}   {}",
        styles.key("counts:"),
        styles.value(&long_counts),
        compact
    );
    println!(
        "  {}",
        styles.key(&format!(
            "legend: {}=LUCID {}=HAZY !=DELIRIUM ?=UNKNOWN",
            '\u{25CF}', '\u{25B2}'
        ))
    );
}

fn render_count_badge(
    styles: &LauncherStyles,
    marker: char,
    short_label: &str,
    count: usize,
) -> String {
    format!(
        "{}{}{}",
        styles.value(&marker.to_string()),
        styles.value(short_label),
        styles.value(&count.to_string())
    )
}

fn render_check_word(styles: &LauncherStyles, check: TargetCheck) -> String {
    if check == TargetCheck::Ok {
        styles.ok_word(check.as_str())
    } else {
        styles.danger_token_reversed(check.as_str())
    }
}

fn print_actions(
    styles: &LauncherStyles,
    resolver: &TargetResolver,
    target: Option<&ResolvedTarget>,
) {
    println!("{}", styles.section(&styles.section_label("Actions")));
    println!(
        "  {}",
        styles.key("Enter open   / target   r recent   t theme   ? help   q quit")
    );
    if let Some(resolved) = target {
        let run_path = display_path(&resolved.ndjson_path);
        println!(
            "  {}",
            styles.critical(&format!(
                "Enter = open TUI (theme: {})",
                styles.theme_label()
            ))
        );
        println!(
            "  {} {} {}",
            styles.key("Will run:"),
            styles.critical("pale-ale tui"),
            styles.value(&format!("\"{}\"", run_path.replace('\"', "\\\"")))
        );
    }
    let recents = resolver.recent_targets();
    if recents.is_empty() {
        println!(
            "  {} {}",
            styles.key("tip:"),
            styles.value("press '/' to paste a path; Esc cancels")
        );
    } else {
        let recent_line = recents
            .iter()
            .take(3)
            .map(|path| path_tail(path))
            .collect::<Vec<String>>()
            .join(", ");
        println!("  {} {}", styles.key("recent:"), styles.value(&recent_line));
    }
    println!();
}

fn print_help(styles: &LauncherStyles) {
    println!("{}", styles.section(&styles.section_label("Help")));
    println!("{}", styles.key("  Enter : open detected target"));
    println!("{}", styles.key("  /     : target input mode"));
    println!("{}", styles.key("  r     : recent targets"));
    println!(
        "{}",
        styles.key("  t     : cycle theme (classic -> term -> cyber)")
    );
    println!("{}", styles.key("  q     : quit launcher"));
    println!("{}", styles.key("  This is not a shell."));
    println!();
}

fn resolve_theme(resolver: &TargetResolver) -> LauncherTheme {
    if let Ok(value) = env::var(THEME_ENV_KEY) {
        if let Some(theme) = parse_theme_name(&value) {
            return theme;
        }
    }

    if let Some(value) = resolver.last_theme() {
        if let Some(theme) = parse_theme_name(&value) {
            return theme;
        }
    }

    LauncherTheme::Classic
}

fn parse_theme_name(raw: &str) -> Option<LauncherTheme> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "classic" => Some(LauncherTheme::Classic),
        "term" => Some(LauncherTheme::Term),
        "cyber" => Some(LauncherTheme::Cyber),
        _ => None,
    }
}

fn detect_color_mode() -> LauncherColorMode {
    if env::var_os("NO_COLOR").is_some() {
        return LauncherColorMode::None;
    }

    let term = env::var("TERM")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if term == "dumb" {
        return LauncherColorMode::None;
    }

    let colorterm = env::var("COLORTERM")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if colorterm.contains("truecolor") || colorterm.contains("24bit") {
        return LauncherColorMode::TrueColor;
    }

    if env::var_os("WT_SESSION").is_some() {
        return LauncherColorMode::TrueColor;
    }

    let term_program = env::var("TERM_PROGRAM")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if term_program.contains("windows_terminal") {
        return LauncherColorMode::TrueColor;
    }

    LauncherColorMode::Ansi16
}

fn display_path(path: &Path) -> String {
    let mut text = path.display().to_string();
    if cfg!(windows) {
        if let Some(stripped) = text.strip_prefix("\\\\?\\UNC\\") {
            text = format!("\\\\{}", stripped);
        } else if let Some(stripped) = text.strip_prefix("\\\\?\\") {
            text = stripped.to_string();
        }
    }
    text
}

fn shorten_text(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_string();
    }
    let tail: String = text
        .chars()
        .rev()
        .take(max_chars.saturating_sub(3))
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    format!("...{}", tail)
}

fn path_tail(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| name.to_string())
        .unwrap_or_else(|| display_path(path))
}

fn format_local_time(time: SystemTime) -> String {
    let dt: DateTime<Local> = DateTime::<Local>::from(time);
    dt.format("%Y-%m-%d %H:%M").to_string()
}

fn format_size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

    let value = bytes as f64;
    if value < KIB {
        format!("{} B", bytes)
    } else if value < MIB {
        format!("{:.1} KiB", value / KIB)
    } else if value < GIB {
        format!("{:.1} MiB", value / MIB)
    } else {
        format!("{:.1} GiB", value / GIB)
    }
}

fn launcher_policy_thresholds(
    preview: Option<&crate::target_resolver::TargetPreview>,
) -> (f64, f64) {
    let default_policy = default_policy_config();
    let default_hazy = f64::from(default_policy.th_ratio_hazy);
    let default_delirium = f64::from(default_policy.th_ratio_delirium);
    let hazy = preview
        .and_then(|item| item.policy_hazy)
        .unwrap_or(default_hazy);
    let delirium = preview
        .and_then(|item| item.policy_delirium)
        .unwrap_or(default_delirium);
    (hazy, delirium)
}

fn read_line(prompt: &str) -> Result<String, String> {
    print!("{}", prompt);
    io::stdout()
        .flush()
        .map_err(|err| format!("failed to flush stdout: {}", err))?;
    let mut line = String::new();
    io::stdin()
        .read_line(&mut line)
        .map_err(|err| format!("failed to read stdin: {}", err))?;
    Ok(line)
}

struct RawModeGuard {
    enabled: bool,
}

impl RawModeGuard {
    fn enable() -> Result<Self, String> {
        enable_raw_mode().map_err(|err| format!("failed to enable raw mode: {}", err))?;
        Ok(Self { enabled: true })
    }
}

impl Drop for RawModeGuard {
    fn drop(&mut self) {
        if self.enabled {
            let _ = disable_raw_mode();
        }
    }
}

fn read_target_line_modal(prompt: &str) -> Result<Option<String>, String> {
    let _raw_guard = RawModeGuard::enable()?;
    print!("{}", prompt);
    io::stdout()
        .flush()
        .map_err(|err| format!("failed to flush stdout: {}", err))?;

    let mut buffer = String::new();
    loop {
        let event = event::read().map_err(|err| format!("failed to read key event: {}", err))?;
        let Event::Key(key) = event else {
            continue;
        };
        if key.kind != KeyEventKind::Press {
            continue;
        }

        match key.code {
            KeyCode::Esc => {
                println!();
                return Ok(None);
            }
            KeyCode::Enter => {
                println!();
                return Ok(Some(buffer));
            }
            KeyCode::Backspace => {
                if buffer.pop().is_some() {
                    print!("\u{8} \u{8}");
                    io::stdout()
                        .flush()
                        .map_err(|err| format!("failed to flush stdout: {}", err))?;
                }
            }
            KeyCode::Char(ch)
                if !key.modifiers.contains(KeyModifiers::CONTROL)
                    && !key.modifiers.contains(KeyModifiers::ALT) =>
            {
                buffer.push(ch);
                print!("{}", ch);
                io::stdout()
                    .flush()
                    .map_err(|err| format!("failed to flush stdout: {}", err))?;
            }
            _ => {}
        }
    }
}
