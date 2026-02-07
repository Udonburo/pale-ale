use pale_ale_modelspec::ModelSpec;
use serde::{Deserialize, Serialize};

mod binding;
mod diagnose;
mod measure;

pub use binding::compute_inputs_hash;
pub use diagnose::{
    diagnose_eval, DiagnoseResult, EvalReport, EvidenceItem, ScoresSummary, VerdictStatus,
};
pub use measure::{
    measure_eval, EvalResult, EvalSummary, MeasureError, PairScore, SentenceEmbedder,
};

pub fn jcs_bytes<T: Serialize>(value: &T) -> Vec<u8> {
    serde_jcs::to_vec(value).expect("JCS serialization failed")
}

pub fn blake3_hex(bytes: &[u8]) -> String {
    blake3::hash(bytes).to_hex().to_string()
}

pub fn measurement_hash(config: &MeasurementConfig) -> String {
    blake3_hex(&jcs_bytes(config))
}

pub fn policy_hash(config: &PolicyConfig) -> String {
    blake3_hex(&jcs_bytes(config))
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct MeasurementConfig {
    #[serde(flatten)]
    pub sentence_split: SentenceSplitConfig,
    pub embed: EmbedConfig,
    pub core: CoreConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SentenceSplitConfig {
    pub sentence_split_version: String,
    pub sentence_split_max_sentences: usize,
    pub sentence_split_normalize_newlines: bool,
    pub sentence_split_per_line: bool,
    pub sentence_split_boundary_chars: Vec<String>,
    pub sentence_split_closing_chars: Vec<String>,
    pub sentence_split_keep_boundary: bool,
    pub sentence_split_trim_ascii_ws: bool,
    pub sentence_split_overflow_strategy: String,
    pub sentence_split_overflow_joiner: String,
    pub sentence_split_unicode_normalize: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbedConfig {
    pub model_id: String,
    pub revision: String,
    pub required_files: Vec<EmbedRequiredFile>,
    pub dtype: String,
    pub pooling: String,
    pub l2_norm: bool,
    pub rounding: RoundingConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EmbedRequiredFile {
    pub path: String,
    pub blake3: Option<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct RoundingConfig {
    pub enabled: bool,
    pub decimals: Option<u32>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CoreConfig {
    pub e8_roots: u32,
    pub k: f32,
    pub beta: f32,
    pub aggregation_weights: AggregationWeights,
    pub semantic_distance_def: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AggregationWeights {
    pub d_intra: f32,
    pub d_inter: f32,
    pub d_hct: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PolicyConfig {
    pub policy_version: String,
    pub policy_profile: String,
    pub policy_defaults_rev: String,

    pub status_levels: Vec<StatusLevel>,
    pub status_ratio_lucid_min: f32,
    pub status_ratio_hazy_min: f32,
    pub status_sem_raw_min: f32,
    pub status_struct_min: f32,
    pub th_ratio_hazy: f32,
    pub th_ratio_delirium: f32,

    pub evidence_max_items: u32,
    pub evidence_max_per_ctx_sentence: u32,
    pub evidence_max_per_ans_sentence: u32,
    pub max_evidence: usize,
    pub max_evidence_per_answer: usize,
    pub evidence_min_score_ratio: f32,
    pub evidence_min_score_sem_raw: f32,
    pub evidence_min_score_struct: f32,
    pub evidence_sort_key: EvidenceSortKey,
    pub evidence_sort_desc: bool,
    pub evidence_context_window_sentences_before: u32,
    pub evidence_context_window_sentences_after: u32,

    pub rule_format_or_order_enabled: bool,
    pub rule_format_or_order_min_ratio: f32,
    pub rule_format_or_order_min_struct: f32,
    pub rule_invalid_block_rate_high_enabled: bool,
    pub rule_invalid_block_rate_high_threshold: f32,
    pub rule_invalid_block_rate_high_min_status: StatusLevel,

    pub report_round_digits: u32,
    pub report_include_rule_trace: bool,
    pub report_include_evidence_text: bool,
    pub report_include_raw_scores: bool,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum StatusLevel {
    Lucid,
    Hazy,
    Delirium,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvidenceSortKey {
    ScoreRatio,
    ScoreSemRaw,
    ScoreStruct,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AttestationLevel {
    Attested,
    AttestedWithPolicy,
    Unattested,
}

pub fn attestation_level(
    canonical_measurement_hash: &str,
    canonical_policy_hash: &str,
    actual_measurement_hash: &str,
    actual_policy_hash: &str,
    use_internal_embed: bool,
    external_vectors: bool,
    build_modified: bool,
) -> AttestationLevel {
    if canonical_measurement_hash != actual_measurement_hash
        || !use_internal_embed
        || external_vectors
        || build_modified
    {
        return AttestationLevel::Unattested;
    }

    if canonical_policy_hash == actual_policy_hash {
        AttestationLevel::Attested
    } else {
        AttestationLevel::AttestedWithPolicy
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuditTrace {
    pub model: ModelTrace,
    pub hashes: HashesTrace,
    pub config_source: ConfigSource,
    pub attestation_level: AttestationLevel,
    pub invalid_block_rate: f32,
    pub comparability: serde_json::Value,
    pub error: Option<AuditError>,
    pub build: Option<BuildTrace>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelTrace {
    pub model_id: String,
    pub revision: String,
    pub files: Vec<ModelFile>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelFile {
    pub path: String,
    pub blake3: Option<String>,
    pub size_bytes: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct HashesTrace {
    pub measurement_hash: String,
    pub policy_hash: String,
    pub inputs_hash: String,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ConfigSource {
    Default,
    File,
    Env,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AuditError {
    pub code: String,
    pub message: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct BuildTrace {
    pub git_commit: Option<String>,
    pub git_dirty: Option<bool>,
    pub rustc_version: Option<String>,
    pub profile: Option<String>,
    pub cargo_lock_hash: Option<String>,
}

pub fn default_measurement_config() -> MeasurementConfig {
    let model_spec = ModelSpec::classic();
    MeasurementConfig {
        sentence_split: SentenceSplitConfig {
            sentence_split_version: "v1".to_string(),
            sentence_split_max_sentences: 64,
            sentence_split_normalize_newlines: true,
            sentence_split_per_line: true,
            sentence_split_boundary_chars: vec![
                ".".to_string(),
                "!".to_string(),
                "?".to_string(),
                "。".to_string(),
                "！".to_string(),
                "？".to_string(),
                "．".to_string(),
                "…".to_string(),
            ],
            sentence_split_closing_chars: vec![
                ")".to_string(),
                "]".to_string(),
                "}".to_string(),
                "」".to_string(),
                "』".to_string(),
                "】".to_string(),
                "》".to_string(),
                "〉".to_string(),
                "\"".to_string(),
                "'".to_string(),
            ],
            sentence_split_keep_boundary: true,
            sentence_split_trim_ascii_ws: true,
            sentence_split_overflow_strategy: "merge_tail".to_string(),
            sentence_split_overflow_joiner: " ".to_string(),
            sentence_split_unicode_normalize: false,
        },
        embed: EmbedConfig {
            model_id: model_spec.model_id.clone(),
            revision: model_spec.revision.clone(),
            required_files: model_spec
                .required_files
                .iter()
                .map(|f| EmbedRequiredFile {
                    path: f.path.clone(),
                    blake3: Some(f.blake3.clone()),
                })
                .collect(),
            dtype: "f32".to_string(),
            pooling: "masked_mean".to_string(),
            l2_norm: true,
            rounding: RoundingConfig {
                enabled: true,
                decimals: Some(6),
            },
        },
        core: CoreConfig {
            e8_roots: 240,
            k: 8.0,
            beta: 1.0,
            aggregation_weights: AggregationWeights {
                d_intra: 0.5,
                d_inter: 0.3,
                d_hct: 0.2,
            },
            semantic_distance_def: "0.5*(1-cos)+eps".to_string(),
        },
    }
}

pub fn default_policy_config() -> PolicyConfig {
    PolicyConfig {
        policy_version: "v1".to_string(),
        policy_profile: "classic".to_string(),
        policy_defaults_rev: "classic-1.0.1".to_string(),
        status_levels: vec![StatusLevel::Lucid, StatusLevel::Hazy, StatusLevel::Delirium],
        status_ratio_lucid_min: 0.8,
        status_ratio_hazy_min: 0.6,
        status_sem_raw_min: 0.5,
        status_struct_min: 0.5,
        th_ratio_hazy: 1.5,
        th_ratio_delirium: 2.2,
        evidence_max_items: 20,
        evidence_max_per_ctx_sentence: 3,
        evidence_max_per_ans_sentence: 3,
        max_evidence: 6,
        max_evidence_per_answer: 3,
        evidence_min_score_ratio: 0.2,
        evidence_min_score_sem_raw: 0.2,
        evidence_min_score_struct: 0.2,
        evidence_sort_key: EvidenceSortKey::ScoreRatio,
        evidence_sort_desc: true,
        evidence_context_window_sentences_before: 1,
        evidence_context_window_sentences_after: 1,
        rule_format_or_order_enabled: true,
        rule_format_or_order_min_ratio: 0.7,
        rule_format_or_order_min_struct: 0.2,
        rule_invalid_block_rate_high_enabled: true,
        rule_invalid_block_rate_high_threshold: 0.1,
        rule_invalid_block_rate_high_min_status: StatusLevel::Hazy,
        report_round_digits: 6,
        report_include_rule_trace: true,
        report_include_evidence_text: true,
        report_include_raw_scores: false,
    }
}

pub fn split_sentences_v1(input: &str) -> Vec<String> {
    const MAX_SENTENCES: usize = 64;

    let normalized = normalize_newlines(input);
    let mut sentences = Vec::new();

    for line in normalized.split('\n') {
        let mut buf = String::new();
        let chars: Vec<char> = line.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            let ch = chars[i];
            buf.push(ch);

            if is_boundary(ch) {
                i += 1;
                while i < chars.len() && is_boundary(chars[i]) {
                    buf.push(chars[i]);
                    i += 1;
                }
                while i < chars.len() && is_closer(chars[i]) {
                    buf.push(chars[i]);
                    i += 1;
                }
                push_sentence(&mut buf, &mut sentences);
                continue;
            }

            i += 1;
        }

        push_sentence(&mut buf, &mut sentences);
    }

    if sentences.len() > MAX_SENTENCES {
        let mut merged = sentences[..(MAX_SENTENCES - 1)].to_vec();
        let tail = sentences[(MAX_SENTENCES - 1)..].join(" ");
        merged.push(tail);
        return merged;
    }

    sentences
}

fn normalize_newlines(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut iter = input.chars().peekable();

    while let Some(ch) = iter.next() {
        if ch == '\r' {
            if let Some('\n') = iter.peek().copied() {
                iter.next();
            }
            out.push('\n');
        } else {
            out.push(ch);
        }
    }

    out
}

fn is_boundary(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | '。' | '！' | '？' | '．' | '…')
}

fn is_closer(ch: char) -> bool {
    matches!(
        ch,
        ')' | ']' | '}' | '」' | '』' | '】' | '》' | '〉' | '"' | '\''
    )
}

fn push_sentence(buffer: &mut String, out: &mut Vec<String>) {
    if buffer.is_empty() {
        return;
    }

    let trimmed = trim_ascii_ws(buffer);
    if !trimmed.is_empty() {
        out.push(trimmed);
    }
    buffer.clear();
}

fn trim_ascii_ws(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut start = 0;
    let mut end = bytes.len();

    while start < end && (bytes[start] == b' ' || bytes[start] == b'\t') {
        start += 1;
    }

    while end > start && (bytes[end - 1] == b' ' || bytes[end - 1] == b'\t') {
        end -= 1;
    }

    if start == 0 && end == bytes.len() {
        value.to_string()
    } else {
        value[start..end].to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn jcs_hash_stable_for_key_order() {
        let v1 = json!({"b": 1, "a": 2});
        let v2 = json!({"a": 2, "b": 1});
        let h1 = blake3_hex(&jcs_bytes(&v1));
        let h2 = blake3_hex(&jcs_bytes(&v2));
        assert_eq!(h1, h2);
    }

    #[test]
    fn split_normalizes_newlines() {
        let input = "a.\r\nb.\rc.";
        let out = split_sentences_v1(input);
        assert_eq!(out, vec!["a.", "b.", "c."]);
    }

    #[test]
    fn split_boundary_runs() {
        let input = "hi...wow?!";
        let out = split_sentences_v1(input);
        assert_eq!(out, vec!["hi...", "wow?!"]);
    }

    #[test]
    fn split_closers_are_kept() {
        let input = "hi.)」next";
        let out = split_sentences_v1(input);
        assert_eq!(out, vec!["hi.)」", "next"]);
    }

    #[test]
    fn split_trims_ascii_whitespace() {
        let input = " \thi.\t ";
        let out = split_sentences_v1(input);
        assert_eq!(out, vec!["hi."]);
    }

    #[test]
    fn split_merges_tail_overflow() {
        let input = (0..65).map(|_| "a.").collect::<Vec<_>>().join(" ");
        let out = split_sentences_v1(&input);
        assert_eq!(out.len(), 64);
        assert_eq!(out[63], "a. a.");
    }

    #[test]
    fn split_drops_empty_sentences() {
        let input = "\n \t \n";
        let out = split_sentences_v1(input);
        assert!(out.is_empty());
    }
}
