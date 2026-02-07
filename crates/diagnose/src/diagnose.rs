use crate::{EvalResult, PairScore, PolicyConfig};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeMap;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum VerdictStatus {
    Lucid,
    Hazy,
    Delirium,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvidenceItem {
    pub ctx_sentence_index: usize,
    pub ans_sentence_index: usize,
    pub ctx_text: String,
    pub ans_text: String,
    pub score_struct: f32,
    pub score_sem: f32,
    pub score_ratio: f32,
    pub tags: Vec<String>,
    pub rule_trace: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ScoresSummary {
    pub max_score_ratio: f32,
    pub max_score_struct: f32,
    pub min_score_sem: f32,
    pub pairs_n: usize,
    pub ctx_n: usize,
    pub ans_n: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvalReport {
    pub query_sentences: Vec<String>,
    pub ctx_sentences: Vec<String>,
    pub ans_sentences: Vec<String>,
    pub scores: ScoresSummary,
    pub evidence: Vec<EvidenceItem>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiagnoseResult {
    pub status: VerdictStatus,
    pub report: EvalReport,
}

pub fn diagnose_eval(measurement: EvalResult, policy: &PolicyConfig) -> DiagnoseResult {
    let EvalResult {
        query_sentences,
        ctx_sentences,
        ans_sentences,
        pairs,
        summary: _,
        warnings: _,
    } = measurement;

    let max_score_ratio = pairs
        .iter()
        .map(|pair| pair.score_ratio)
        .fold(0.0, f32::max);
    let max_score_struct = pairs
        .iter()
        .map(|pair| pair.score_struct)
        .fold(0.0, f32::max);
    let min_score_sem = if pairs.is_empty() {
        0.0
    } else {
        pairs
            .iter()
            .map(|pair| pair.score_sem)
            .fold(f32::INFINITY, f32::min)
    };

    let status = verdict_from_ratio(
        max_score_ratio,
        policy.th_ratio_hazy,
        policy.th_ratio_delirium,
    );
    let evidence = select_evidence(&pairs, &ctx_sentences, &ans_sentences, policy);

    DiagnoseResult {
        status,
        report: EvalReport {
            query_sentences,
            ctx_sentences: ctx_sentences.clone(),
            ans_sentences: ans_sentences.clone(),
            scores: ScoresSummary {
                max_score_ratio,
                max_score_struct,
                min_score_sem,
                pairs_n: pairs.len(),
                ctx_n: ctx_sentences.len(),
                ans_n: ans_sentences.len(),
            },
            evidence,
        },
    }
}

fn verdict_from_ratio(ratio: f32, th_hazy: f32, th_delirium: f32) -> VerdictStatus {
    if ratio >= th_delirium {
        VerdictStatus::Delirium
    } else if ratio >= th_hazy {
        VerdictStatus::Hazy
    } else {
        VerdictStatus::Lucid
    }
}

fn select_evidence(
    pairs: &[PairScore],
    ctx_sentences: &[String],
    ans_sentences: &[String],
    policy: &PolicyConfig,
) -> Vec<EvidenceItem> {
    let mut grouped: BTreeMap<usize, Vec<&PairScore>> = BTreeMap::new();
    for pair in pairs {
        grouped.entry(pair.ans_idx).or_default().push(pair);
    }

    let mut selected = Vec::new();
    for group in grouped.values_mut() {
        group.sort_by(compare_pairs_for_answer_group);
        selected.extend(group.iter().take(policy.max_evidence_per_answer).copied());
    }

    selected.sort_by(compare_pairs_global);
    selected.truncate(policy.max_evidence);

    selected
        .into_iter()
        .map(|pair| to_evidence_item(pair, ctx_sentences, ans_sentences, policy))
        .collect()
}

fn to_evidence_item(
    pair: &PairScore,
    ctx_sentences: &[String],
    ans_sentences: &[String],
    policy: &PolicyConfig,
) -> EvidenceItem {
    let (tags, rule_trace) = tags_and_trace(pair.score_ratio, policy);
    EvidenceItem {
        ctx_sentence_index: pair.ctx_idx,
        ans_sentence_index: pair.ans_idx,
        ctx_text: ctx_sentences.get(pair.ctx_idx).cloned().unwrap_or_default(),
        ans_text: ans_sentences.get(pair.ans_idx).cloned().unwrap_or_default(),
        score_struct: pair.score_struct,
        score_sem: pair.score_sem,
        score_ratio: pair.score_ratio,
        tags,
        rule_trace,
    }
}

fn tags_and_trace(ratio: f32, policy: &PolicyConfig) -> (Vec<String>, Vec<String>) {
    if ratio >= policy.th_ratio_delirium {
        (
            vec!["RATIO_DELIRIUM".to_string()],
            vec![
                "RATIO>=TH_HAZY".to_string(),
                "RATIO>=TH_DELIRIUM".to_string(),
            ],
        )
    } else if ratio >= policy.th_ratio_hazy {
        (
            vec!["RATIO_HAZY".to_string()],
            vec!["RATIO>=TH_HAZY".to_string()],
        )
    } else {
        (
            vec!["RATIO_LUCID".to_string()],
            vec!["RATIO<TH_HAZY".to_string()],
        )
    }
}

fn compare_pairs_for_answer_group(left: &&PairScore, right: &&PairScore) -> Ordering {
    right
        .score_ratio
        .total_cmp(&left.score_ratio)
        .then_with(|| left.ctx_idx.cmp(&right.ctx_idx))
        .then_with(|| left.ans_idx.cmp(&right.ans_idx))
}

fn compare_pairs_global(left: &&PairScore, right: &&PairScore) -> Ordering {
    right
        .score_ratio
        .total_cmp(&left.score_ratio)
        .then_with(|| left.ans_idx.cmp(&right.ans_idx))
        .then_with(|| left.ctx_idx.cmp(&right.ctx_idx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{default_policy_config, EvalSummary};

    fn sample_measurement_with_ratios(ratios: &[f32]) -> EvalResult {
        EvalResult {
            query_sentences: vec!["query".to_string()],
            ctx_sentences: vec!["ctx0".to_string(), "ctx1".to_string(), "ctx2".to_string()],
            ans_sentences: vec!["ans0".to_string()],
            pairs: ratios
                .iter()
                .enumerate()
                .map(|(idx, ratio)| PairScore {
                    ans_idx: 0,
                    ctx_idx: idx,
                    score_struct: *ratio,
                    score_sem: 1.0,
                    score_ratio: *ratio,
                })
                .collect(),
            summary: EvalSummary {
                ctx_n: 3,
                ans_n: 1,
                pairs_n: ratios.len(),
                max_score_ratio: ratios.iter().copied().fold(0.0, f32::max),
            },
            warnings: Vec::new(),
        }
    }

    #[test]
    fn verdict_thresholds_work() {
        let policy = default_policy_config();

        let lucid_result = diagnose_eval(sample_measurement_with_ratios(&[1.49]), &policy);
        assert_eq!(lucid_result.status, VerdictStatus::Lucid);

        let hazy_result = diagnose_eval(sample_measurement_with_ratios(&[1.50]), &policy);
        assert_eq!(hazy_result.status, VerdictStatus::Hazy);

        let delirium_result = diagnose_eval(sample_measurement_with_ratios(&[2.20]), &policy);
        assert_eq!(delirium_result.status, VerdictStatus::Delirium);
    }

    #[test]
    fn evidence_selection_is_deterministic() {
        let mut policy = default_policy_config();
        policy.max_evidence = 4;
        policy.max_evidence_per_answer = 2;

        let measurement = EvalResult {
            query_sentences: vec!["query".to_string()],
            ctx_sentences: vec![
                "ctx0".to_string(),
                "ctx1".to_string(),
                "ctx2".to_string(),
                "ctx3".to_string(),
            ],
            ans_sentences: vec!["ans0".to_string(), "ans1".to_string()],
            pairs: vec![
                PairScore {
                    ans_idx: 0,
                    ctx_idx: 2,
                    score_struct: 2.0,
                    score_sem: 1.0,
                    score_ratio: 2.0,
                },
                PairScore {
                    ans_idx: 0,
                    ctx_idx: 1,
                    score_struct: 2.0,
                    score_sem: 1.0,
                    score_ratio: 2.0,
                },
                PairScore {
                    ans_idx: 0,
                    ctx_idx: 0,
                    score_struct: 1.9,
                    score_sem: 1.0,
                    score_ratio: 1.9,
                },
                PairScore {
                    ans_idx: 1,
                    ctx_idx: 1,
                    score_struct: 2.0,
                    score_sem: 1.0,
                    score_ratio: 2.0,
                },
                PairScore {
                    ans_idx: 1,
                    ctx_idx: 0,
                    score_struct: 2.0,
                    score_sem: 1.0,
                    score_ratio: 2.0,
                },
                PairScore {
                    ans_idx: 1,
                    ctx_idx: 2,
                    score_struct: 1.8,
                    score_sem: 1.0,
                    score_ratio: 1.8,
                },
            ],
            summary: EvalSummary {
                ctx_n: 4,
                ans_n: 2,
                pairs_n: 6,
                max_score_ratio: 2.0,
            },
            warnings: Vec::new(),
        };

        let result_a = diagnose_eval(measurement.clone(), &policy);
        let result_b = diagnose_eval(measurement, &policy);
        assert_eq!(result_a, result_b);
        assert_eq!(
            result_a
                .report
                .evidence
                .iter()
                .map(|item| (item.ans_sentence_index, item.ctx_sentence_index))
                .collect::<Vec<_>>(),
            vec![(0, 1), (0, 2), (1, 0), (1, 1)]
        );
    }
}
