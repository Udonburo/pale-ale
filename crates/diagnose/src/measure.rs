use crate::split_sentences_v1;
use pale_ale_core::spin3_struct_distance;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt;

const EMBED_DIM: usize = 384;
const TOP_K: usize = 3;
const SCORE_RATIO_EPS: f32 = 1e-6;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvalResult {
    pub query_sentences: Vec<String>,
    pub ctx_sentences: Vec<String>,
    pub ans_sentences: Vec<String>,
    pub pairs: Vec<PairScore>,
    pub summary: EvalSummary,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PairScore {
    pub ans_idx: usize,
    pub ctx_idx: usize,
    pub score_struct: f32,
    pub score_sem: f32,
    pub score_ratio: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct EvalSummary {
    pub ctx_n: usize,
    pub ans_n: usize,
    pub pairs_n: usize,
    pub max_score_ratio: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeasureError {
    EmbedError {
        sentence: String,
        message: String,
    },
    EmbeddingDimMismatch {
        sentence: String,
        expected: usize,
        actual: usize,
    },
    VectorLenMismatch {
        left: usize,
        right: usize,
    },
    StructuralDistanceError {
        ans_idx: usize,
        ctx_idx: usize,
        message: String,
    },
}

impl fmt::Display for MeasureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmbedError { sentence, message } => {
                write!(
                    f,
                    "embedding failed for sentence {:?}: {}",
                    sentence, message
                )
            }
            Self::EmbeddingDimMismatch {
                sentence,
                expected,
                actual,
            } => write!(
                f,
                "embedding dimension mismatch for sentence {:?}: expected {}, got {}",
                sentence, expected, actual
            ),
            Self::VectorLenMismatch { left, right } => {
                write!(f, "vector length mismatch: {} vs {}", left, right)
            }
            Self::StructuralDistanceError {
                ans_idx,
                ctx_idx,
                message,
            } => write!(
                f,
                "structural distance failed for ans_idx={} ctx_idx={}: {}",
                ans_idx, ctx_idx, message
            ),
        }
    }
}

impl std::error::Error for MeasureError {}

pub trait SentenceEmbedder {
    fn embed_sentence(&self, sentence: &str) -> Result<Vec<f32>, String>;
}

impl<F> SentenceEmbedder for F
where
    F: Fn(&str) -> Result<Vec<f32>, String>,
{
    fn embed_sentence(&self, sentence: &str) -> Result<Vec<f32>, String> {
        self(sentence)
    }
}

#[derive(Clone, Debug)]
struct Embedding {
    f32_values: Vec<f32>,
    f64_values: Vec<f64>,
}

pub fn measure_eval<E: SentenceEmbedder>(
    embedder: &E,
    query: &str,
    context: &str,
    answer: &str,
) -> Result<EvalResult, MeasureError> {
    let query_sentence = query.trim().to_string();
    let query_sentences = vec![query_sentence];
    let ctx_sentences = split_sentences_v1(context);
    let ans_sentences = split_sentences_v1(answer);

    let _query_embeddings = embed_sentences(embedder, &query_sentences)?;
    let ctx_embeddings = embed_sentences(embedder, &ctx_sentences)?;
    let ans_embeddings = embed_sentences(embedder, &ans_sentences)?;

    let mut pairs = Vec::new();
    for (ans_idx, ans_embedding) in ans_embeddings.iter().enumerate() {
        let mut candidates = Vec::with_capacity(ctx_embeddings.len());
        for (ctx_idx, ctx_embedding) in ctx_embeddings.iter().enumerate() {
            let score_sem =
                semantic_distance(&ans_embedding.f32_values, &ctx_embedding.f32_values)?;
            let score_struct =
                spin3_struct_distance(&ans_embedding.f64_values, &ctx_embedding.f64_values)
                    .map_err(|message| MeasureError::StructuralDistanceError {
                        ans_idx,
                        ctx_idx,
                        message,
                    })? as f32;
            let score_ratio = score_struct / (score_sem + SCORE_RATIO_EPS);
            candidates.push(PairScore {
                ans_idx,
                ctx_idx,
                score_struct,
                score_sem,
                score_ratio,
            });
        }

        candidates.sort_by(compare_pair_scores);
        candidates.truncate(TOP_K.min(candidates.len()));
        pairs.extend(candidates);
    }

    let max_score_ratio = pairs
        .iter()
        .map(|pair| pair.score_ratio)
        .fold(0.0_f32, f32::max);

    Ok(EvalResult {
        query_sentences,
        ctx_sentences,
        ans_sentences,
        summary: EvalSummary {
            ctx_n: ctx_embeddings.len(),
            ans_n: ans_embeddings.len(),
            pairs_n: pairs.len(),
            max_score_ratio,
        },
        pairs,
    })
}

fn embed_sentences<E: SentenceEmbedder>(
    embedder: &E,
    sentences: &[String],
) -> Result<Vec<Embedding>, MeasureError> {
    let mut out = Vec::with_capacity(sentences.len());
    for sentence in sentences {
        let values =
            embedder
                .embed_sentence(sentence)
                .map_err(|message| MeasureError::EmbedError {
                    sentence: sentence.clone(),
                    message,
                })?;
        if values.len() != EMBED_DIM {
            return Err(MeasureError::EmbeddingDimMismatch {
                sentence: sentence.clone(),
                expected: EMBED_DIM,
                actual: values.len(),
            });
        }
        let f64_values = values.iter().map(|v| *v as f64).collect();
        out.push(Embedding {
            f32_values: values,
            f64_values,
        });
    }
    Ok(out)
}

fn semantic_distance(left: &[f32], right: &[f32]) -> Result<f32, MeasureError> {
    if left.len() != right.len() {
        return Err(MeasureError::VectorLenMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    let dot = left
        .iter()
        .zip(right.iter())
        .map(|(l, r)| (*l as f64) * (*r as f64))
        .sum::<f64>();
    Ok((1.0_f64 - dot.clamp(-1.0, 1.0)) as f32)
}

fn compare_pair_scores(left: &PairScore, right: &PairScore) -> Ordering {
    right
        .score_ratio
        .total_cmp(&left.score_ratio)
        .then_with(|| left.ctx_idx.cmp(&right.ctx_idx))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn semantic_distance_invariants() {
        let same = semantic_distance(&[1.0_f32, 0.0], &[1.0_f32, 0.0]).expect("same");
        let opposite = semantic_distance(&[1.0_f32, 0.0], &[-1.0_f32, 0.0]).expect("opposite");
        assert!(same.abs() < 1e-6, "same={}", same);
        assert!((opposite - 2.0).abs() < 1e-6, "opposite={}", opposite);
    }

    #[test]
    fn pair_selection_is_deterministic() {
        let fake_embedder = |_sentence: &str| -> Result<Vec<f32>, String> {
            let mut vector = vec![0.0_f32; EMBED_DIM];
            vector[0] = 1.0;
            Ok(vector)
        };

        let query = "query";
        let context = "c0. c1. c2. c3.";
        let answer = "a0.";

        let result_a = measure_eval(&fake_embedder, query, context, answer).expect("measure A");
        let result_b = measure_eval(&fake_embedder, query, context, answer).expect("measure B");

        assert_eq!(result_a, result_b);
        assert_eq!(result_a.summary.ctx_n, 4);
        assert_eq!(result_a.summary.ans_n, 1);
        assert_eq!(result_a.summary.pairs_n, 3);
        assert_eq!(
            result_a.pairs.iter().map(|p| p.ctx_idx).collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
    }
}
