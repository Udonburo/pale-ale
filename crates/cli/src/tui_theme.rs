use ratatui::style::{Color, Modifier, Style};
use ratatui::widgets::BorderType;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum UiState {
    Lucid,
    Hazy,
    Delirium,
    Unknown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ThemeName {
    Classic,
    Term,
    Cyber,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ColorPreference {
    Auto,
    Always,
    Never,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ColorMode {
    TrueColor,
    Ansi256,
    Ansi16,
    None,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum GaugeLevel {
    Primary,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct Glyphs {
    pub selection_marker: &'static str,
    pub border_type: BorderType,
}

#[derive(Clone, Copy, Debug)]
struct Rgb {
    r: u8,
    g: u8,
    b: u8,
}

impl Rgb {
    const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Theme {
    bg: Rgb,
    panel_bg: Rgb,
    border: Rgb,
    text: Rgb,
    muted: Rgb,
    accent: Rgb,
    lucid: Rgb,
    hazy: Rgb,
    delirium: Rgb,
    unknown: Rgb,
    selected_bg: Rgb,
    gauge_primary: Rgb,
}

impl Theme {
    fn classic() -> Self {
        Self {
            bg: Rgb::new(0x0B, 0x0E, 0x14),
            panel_bg: Rgb::new(0x11, 0x16, 0x1F),
            border: Rgb::new(0x26, 0x30, 0x41),
            text: Rgb::new(0xC7, 0xCE, 0xD8),
            muted: Rgb::new(0x7F, 0x8A, 0x9B),
            accent: Rgb::new(0x7E, 0xA6, 0xD9),
            lucid: Rgb::new(0x84, 0xB7, 0x9C),
            hazy: Rgb::new(0xC3, 0xA7, 0x7A),
            delirium: Rgb::new(0xBD, 0x7B, 0x82),
            unknown: Rgb::new(0x8D, 0x99, 0xAB),
            selected_bg: Rgb::new(0x1D, 0x26, 0x34),
            gauge_primary: Rgb::new(0x7E, 0xA6, 0xD9),
        }
    }

    fn term() -> Self {
        Self {
            // Term mode keeps chroma low and relies on terminal defaults for surfaces.
            bg: Rgb::new(0x00, 0x00, 0x00),
            panel_bg: Rgb::new(0x00, 0x00, 0x00),
            border: Rgb::new(0x80, 0x80, 0x80),
            text: Rgb::new(0xDD, 0xE1, 0xE8),
            muted: Rgb::new(0x97, 0xA1, 0xAF),
            accent: Rgb::new(0x87, 0xAA, 0xCF),
            lucid: Rgb::new(0x89, 0xB8, 0x9F),
            hazy: Rgb::new(0xC6, 0xAE, 0x82),
            delirium: Rgb::new(0xB8, 0x80, 0x86),
            unknown: Rgb::new(0x95, 0xA2, 0xB5),
            selected_bg: Rgb::new(0x00, 0x00, 0x00),
            gauge_primary: Rgb::new(0x87, 0xAA, 0xCF),
        }
    }

    fn cyber() -> Self {
        // Crisp cyberpunk-ish palette with restrained area color usage.
        Self {
            bg: Rgb::new(0x09, 0x0B, 0x0F),
            panel_bg: Rgb::new(0x0D, 0x12, 0x1A),
            border: Rgb::new(0x2D, 0x3A, 0x4E),
            text: Rgb::new(0xD3, 0xDA, 0xE4),
            muted: Rgb::new(0x8B, 0x98, 0xAC),
            accent: Rgb::new(0x52, 0xD1, 0xFF), // cyan-ish
            lucid: Rgb::new(0x66, 0xD2, 0xB4),
            hazy: Rgb::new(0xD6, 0xB1, 0x74),
            delirium: Rgb::new(0xE0, 0x67, 0xD0), // magenta-ish
            unknown: Rgb::new(0x99, 0xA6, 0xBB),
            selected_bg: Rgb::new(0x1A, 0x2A, 0x3B),
            gauge_primary: Rgb::new(0x52, 0xD1, 0xFF),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Token {
    Bg,
    PanelBg,
    Border,
    Text,
    Muted,
    Accent,
    Lucid,
    Hazy,
    Delirium,
    Unknown,
    SelectedBg,
    GaugePrimary,
}

#[derive(Clone, Copy, Debug)]
pub(super) struct ThemeContext {
    theme_name: ThemeName,
    theme: Theme,
    color_mode: ColorMode,
    ascii: bool,
}

impl ThemeContext {
    pub(super) fn new(theme_name: ThemeName, preference: ColorPreference, ascii: bool) -> Self {
        let theme = match theme_name {
            ThemeName::Classic => Theme::classic(),
            ThemeName::Term => Theme::term(),
            ThemeName::Cyber => Theme::cyber(),
        };

        let color_mode = match preference {
            ColorPreference::Never => ColorMode::None,
            ColorPreference::Always => ColorMode::TrueColor,
            ColorPreference::Auto => detect_auto_mode(),
        };

        Self {
            theme_name,
            theme,
            color_mode,
            ascii,
        }
    }

    pub(super) fn ascii(&self) -> bool {
        self.ascii
    }

    pub(super) fn glyphs(&self) -> Glyphs {
        if self.ascii {
            Glyphs {
                selection_marker: ">",
                border_type: BorderType::Plain,
            }
        } else if self.theme_name == ThemeName::Cyber {
            Glyphs {
                selection_marker: "\u{25B8}",
                border_type: BorderType::Plain,
            }
        } else {
            Glyphs {
                selection_marker: "\u{00BB}",
                border_type: BorderType::Rounded,
            }
        }
    }

    pub(super) fn app_bg_color(&self) -> Color {
        if self.backgrounds_disabled() {
            Color::Reset
        } else {
            self.color(Token::Bg)
        }
    }

    pub(super) fn style_panel(&self) -> Style {
        if self.backgrounds_disabled() {
            Style::default()
        } else {
            Style::default().bg(self.color(Token::PanelBg))
        }
    }

    pub(super) fn style_text(&self) -> Style {
        Style::default().fg(self.color(Token::Text))
    }

    pub(super) fn style_muted(&self) -> Style {
        if matches!(self.color_mode, ColorMode::None) {
            Style::default().add_modifier(Modifier::DIM)
        } else {
            Style::default().fg(self.color(Token::Muted))
        }
    }

    pub(super) fn style_border(&self) -> Style {
        Style::default().fg(self.color(Token::Border))
    }

    pub(super) fn style_title(&self) -> Style {
        if matches!(self.color_mode, ColorMode::None) {
            self.style_text().add_modifier(Modifier::BOLD)
        } else {
            self.style_muted().add_modifier(Modifier::BOLD)
        }
    }

    pub(super) fn style_section_title(&self) -> Style {
        if matches!(self.color_mode, ColorMode::None) {
            self.style_text().add_modifier(Modifier::BOLD)
        } else if self.theme_name == ThemeName::Cyber {
            self.style_text()
                .fg(self.color(Token::Accent))
                .add_modifier(Modifier::BOLD)
        } else {
            self.style_muted().add_modifier(Modifier::BOLD)
        }
    }

    pub(super) fn style_selected(&self) -> Style {
        if self.backgrounds_disabled() {
            self.style_text().add_modifier(Modifier::REVERSED)
        } else {
            self.style_text().bg(self.color(Token::SelectedBg))
        }
    }

    pub(super) fn style_chip(&self, state: UiState) -> Style {
        let token = match state {
            UiState::Lucid => Token::Lucid,
            UiState::Hazy => Token::Hazy,
            UiState::Delirium => Token::Delirium,
            UiState::Unknown => Token::Unknown,
        };
        self.style_text().fg(self.color(token))
    }

    pub(super) fn style_badge(&self, state: UiState) -> Style {
        if matches!(self.color_mode, ColorMode::None) {
            self.style_text()
                .add_modifier(Modifier::REVERSED | Modifier::BOLD)
        } else {
            self.style_chip(state).add_modifier(Modifier::BOLD)
        }
    }

    pub(super) fn style_selection_glyph(&self) -> Style {
        if matches!(self.color_mode, ColorMode::None) {
            self.style_text().add_modifier(Modifier::BOLD)
        } else if self.theme_name == ThemeName::Cyber {
            self.style_text()
                .fg(self.color(Token::Accent))
                .add_modifier(Modifier::BOLD)
        } else {
            self.style_muted().add_modifier(Modifier::BOLD)
        }
    }

    pub(super) fn style_accent(&self) -> Style {
        self.style_text()
            .fg(self.color(Token::Accent))
            .add_modifier(Modifier::BOLD)
    }

    pub(super) fn style_gauge(&self, level: GaugeLevel) -> Style {
        let token = match level {
            GaugeLevel::Primary => Token::GaugePrimary,
        };
        self.style_text().fg(self.color(token))
    }

    fn color(&self, token: Token) -> Color {
        match self.color_mode {
            ColorMode::TrueColor => {
                let rgb = self.token_rgb(token);
                Color::Rgb(rgb.r, rgb.g, rgb.b)
            }
            ColorMode::Ansi256 => self.color_ansi256(token),
            ColorMode::Ansi16 => self.color_ansi16(token),
            ColorMode::None => Color::Reset,
        }
    }

    fn backgrounds_disabled(&self) -> bool {
        self.theme_name == ThemeName::Term
            || self.theme_name == ThemeName::Cyber
            || matches!(self.color_mode, ColorMode::Ansi16 | ColorMode::None)
    }

    fn token_rgb(&self, token: Token) -> Rgb {
        match token {
            Token::Bg => self.theme.bg,
            Token::PanelBg => self.theme.panel_bg,
            Token::Border => self.theme.border,
            Token::Text => self.theme.text,
            Token::Muted => self.theme.muted,
            Token::Accent => self.theme.accent,
            Token::Lucid => self.theme.lucid,
            Token::Hazy => self.theme.hazy,
            Token::Delirium => self.theme.delirium,
            Token::Unknown => self.theme.unknown,
            Token::SelectedBg => self.theme.selected_bg,
            Token::GaugePrimary => self.theme.gauge_primary,
        }
    }

    fn color_ansi256(&self, token: Token) -> Color {
        match token {
            Token::Bg => Color::Indexed(234),
            Token::PanelBg => Color::Indexed(235),
            Token::Border => Color::Indexed(239),
            Token::Text => Color::Indexed(251),
            Token::Muted => Color::Indexed(244),
            Token::Accent => match self.theme_name {
                ThemeName::Cyber => Color::Indexed(45),
                _ => Color::Indexed(110),
            },
            Token::Lucid => Color::Indexed(108),
            Token::Hazy => Color::Indexed(180),
            Token::Delirium => match self.theme_name {
                ThemeName::Cyber => Color::Indexed(201),
                _ => Color::Indexed(174),
            },
            Token::Unknown => Color::Indexed(145),
            Token::SelectedBg => Color::Indexed(237),
            Token::GaugePrimary => match self.theme_name {
                ThemeName::Cyber => Color::Indexed(45),
                _ => Color::Indexed(110),
            },
        }
    }

    fn color_ansi16(&self, token: Token) -> Color {
        match token {
            Token::Bg | Token::PanelBg => Color::Reset,
            Token::Border => Color::DarkGray,
            Token::Text => Color::White,
            Token::Muted => Color::DarkGray,
            Token::Accent => match self.theme_name {
                ThemeName::Classic => Color::Blue,
                ThemeName::Term => Color::Cyan,
                ThemeName::Cyber => Color::Cyan,
            },
            Token::Lucid => Color::Green,
            Token::Hazy => Color::Yellow,
            Token::Delirium => match self.theme_name {
                ThemeName::Cyber => Color::Magenta,
                _ => Color::Red,
            },
            Token::Unknown => Color::Gray,
            Token::SelectedBg => Color::DarkGray,
            Token::GaugePrimary => match self.theme_name {
                ThemeName::Classic => Color::Blue,
                ThemeName::Term => Color::Cyan,
                ThemeName::Cyber => Color::Cyan,
            },
        }
    }
}

fn detect_auto_mode() -> ColorMode {
    let colorterm = std::env::var("COLORTERM")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if colorterm.contains("truecolor") || colorterm.contains("24bit") {
        return ColorMode::TrueColor;
    }

    if std::env::var_os("WT_SESSION").is_some() {
        return ColorMode::TrueColor;
    }

    let term_program = std::env::var("TERM_PROGRAM")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if term_program.contains("windows_terminal") {
        return ColorMode::TrueColor;
    }

    let term = std::env::var("TERM")
        .map(|value| value.to_ascii_lowercase())
        .unwrap_or_default();
    if term.contains("256color") {
        ColorMode::Ansi256
    } else {
        ColorMode::Ansi16
    }
}
