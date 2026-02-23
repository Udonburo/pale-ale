use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const GATE1_TOPK: u16 = 8;
pub const LINKS_TOPK_CANONICALIZATION_ID: &str = "link_rank_doc_canon_v1";
pub const TOP1_POLICY_ID: &str = "strict_rank1_or_missing_v1";
pub const MAX_MISSING_LINK_STEP_RATE: f64 = 0.20;

pub const LINK_SANITY_ID: &str = "sanity16_single_judgment_v1";
pub const LINK_SANITY_K: usize = 16;
pub const LINK_SANITY_RNG_ID: &str = "splitmix64_v1";
pub const LINK_SANITY_SAMPLING_ID: &str = "hash_sort_without_replacement_v1";
pub const LINK_SANITY_SEED: u64 = 0x5341_4E49_5459_3136;
pub const LINK_SANITY_FAIL_UNRELATED_MAX: usize = 6;
pub const LINK_SANITY_RANDOM_LIKE_H_NORM_THRESHOLD: f64 = 0.95;
pub const LINK_SANITY_DOMINANT_SHARE_THRESHOLD: f64 = 0.50;

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LinkRow {
    pub ans_unit_id: u32,
    pub doc_unit_id: u32,
    pub rank: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SampleLinksInput {
    pub sample_id: u64,
    pub ans_unit_count: usize,
    pub doc_unit_count: usize,
    pub links_topk: Vec<LinkRow>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalLink {
    pub doc_unit_id: u32,
    pub rank: u16,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalizationCounters {
    pub count_invalid_rank_links: usize,
    pub count_invalid_ans_unit_id: usize,
    pub count_invalid_doc_unit_id: usize,
    pub count_link_dedup: usize,
    pub count_multi_rank1_links: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonicalizedSampleLinks {
    pub sample_id: u64,
    pub links_topk_canonicalization_id: String,
    pub ans_unit_count: usize,
    pub doc_unit_count: usize,
    pub links_by_answer: Vec<Vec<CanonicalLink>>,
    pub counters: CanonicalizationCounters,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Top1Step {
    Selected { ans_unit_id: u32, doc_unit_id: u32 },
    MissingLink { ans_unit_id: u32 },
    MissingTop1 { ans_unit_id: u32 },
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Top1Accounting {
    pub top1_policy_id: String,
    pub steps_total: usize,
    pub count_missing_link_steps: usize,
    pub count_missing_top1_steps: usize,
    pub missing_link_step_rate: f64,
    pub missing_top1_step_rate: f64,
    pub max_missing_link_step_rate: f64,
    pub missing_link_step_rate_exceeds_threshold: bool,
    pub steps: Vec<Top1Step>,
}

impl Top1Accounting {
    pub fn representative_top1(&self) -> Option<(u32, u32)> {
        // Steps are generated in ans_unit_id ascending order (0..ans_unit_count),
        // so the first selected step is the minimum ans_unit_id with top-1 available.
        for step in &self.steps {
            if let Top1Step::Selected {
                ans_unit_id,
                doc_unit_id,
            } = step
            {
                return Some((*ans_unit_id, *doc_unit_id));
            }
        }
        None
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SampleLinkReport {
    pub sample_id: u64,
    pub canonicalized: CanonicalizedSampleLinks,
    pub top1: Top1Accounting,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(rename_all = "snake_case")]
pub enum SanityCategory {
    DocUnitId(u32),
    NoLink,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SanityJudgment {
    Unreviewed,
    Related,
    Unrelated,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct LinkSanityRecord {
    pub sample_id: u64,
    pub representative_ans_unit_id: Option<u32>,
    pub selected_doc_unit_id: Option<u32>,
    pub category: SanityCategory,
    pub judgment: SanityJudgment,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct LinkSanityResult {
    pub link_sanity_id: String,
    pub rng_id: String,
    pub seed: u64,
    pub sampling_id: String,
    pub selected_sample_ids: Vec<u64>,
    pub k_eff: usize,
    pub records: Vec<LinkSanityRecord>,
    pub unrelated_count: usize,
    pub link_sanity_fail: bool,
    pub h_norm: f64,
    pub max_share: f64,
    pub random_like_link_collapse: bool,
    pub dominant_link_collapse: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinkSanityError {
    DuplicateSampleId { sample_id: u64 },
}

impl fmt::Display for LinkSanityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DuplicateSampleId { sample_id } => {
                write!(f, "duplicate sample_id in sanity input: {}", sample_id)
            }
        }
    }
}

impl std::error::Error for LinkSanityError {}

#[allow(dead_code)]
pub fn process_batch_links(inputs: &[SampleLinksInput]) -> Vec<SampleLinkReport> {
    inputs.iter().map(process_sample_links).collect()
}

#[allow(dead_code)]
pub fn process_sample_links(input: &SampleLinksInput) -> SampleLinkReport {
    let canonicalized = canonicalize_links(input);
    let top1 = compute_top1_accounting(&canonicalized);
    SampleLinkReport {
        sample_id: input.sample_id,
        canonicalized,
        top1,
    }
}

pub fn canonicalize_links(input: &SampleLinksInput) -> CanonicalizedSampleLinks {
    let mut counters = CanonicalizationCounters::default();
    let mut dedup_rank_by_pair: BTreeMap<(u32, u32), u16> = BTreeMap::new();

    for link in &input.links_topk {
        if link.rank == 0 || link.rank > GATE1_TOPK {
            counters.count_invalid_rank_links += 1;
            continue;
        }

        if (link.ans_unit_id as usize) >= input.ans_unit_count {
            counters.count_invalid_ans_unit_id += 1;
            continue;
        }

        if (link.doc_unit_id as usize) >= input.doc_unit_count {
            counters.count_invalid_doc_unit_id += 1;
            continue;
        }

        let key = (link.ans_unit_id, link.doc_unit_id);
        match dedup_rank_by_pair.get_mut(&key) {
            Some(existing_rank) => {
                counters.count_link_dedup += 1;
                if link.rank < *existing_rank {
                    *existing_rank = link.rank;
                }
            }
            None => {
                dedup_rank_by_pair.insert(key, link.rank);
            }
        }
    }

    let mut links_by_answer = vec![Vec::new(); input.ans_unit_count];
    for ((ans_unit_id, doc_unit_id), rank) in dedup_rank_by_pair {
        links_by_answer[ans_unit_id as usize].push(CanonicalLink { doc_unit_id, rank });
    }

    for links in links_by_answer.iter_mut() {
        links.sort_by(|left, right| {
            left.rank
                .cmp(&right.rank)
                .then_with(|| left.doc_unit_id.cmp(&right.doc_unit_id))
        });
        let rank1_count = links.iter().filter(|link| link.rank == 1).count();
        if rank1_count > 1 {
            counters.count_multi_rank1_links += 1;
        }
    }

    CanonicalizedSampleLinks {
        sample_id: input.sample_id,
        links_topk_canonicalization_id: LINKS_TOPK_CANONICALIZATION_ID.to_string(),
        ans_unit_count: input.ans_unit_count,
        doc_unit_count: input.doc_unit_count,
        links_by_answer,
        counters,
    }
}

pub fn compute_top1_accounting(canonicalized: &CanonicalizedSampleLinks) -> Top1Accounting {
    let steps_total = canonicalized.ans_unit_count;
    let mut steps = Vec::with_capacity(steps_total);
    let mut count_missing_link_steps = 0usize;
    let mut count_missing_top1_steps = 0usize;

    for ans_unit_idx in 0..canonicalized.ans_unit_count {
        let ans_unit_id = ans_unit_idx as u32;
        let links = &canonicalized.links_by_answer[ans_unit_idx];
        if links.is_empty() {
            count_missing_link_steps += 1;
            steps.push(Top1Step::MissingLink { ans_unit_id });
            continue;
        }

        if let Some(rank1) = links.iter().find(|link| link.rank == 1) {
            steps.push(Top1Step::Selected {
                ans_unit_id,
                doc_unit_id: rank1.doc_unit_id,
            });
        } else {
            count_missing_top1_steps += 1;
            steps.push(Top1Step::MissingTop1 { ans_unit_id });
        }
    }

    let missing_link_step_rate = if steps_total == 0 {
        0.0
    } else {
        (count_missing_link_steps as f64) / (steps_total as f64)
    };
    let missing_top1_step_rate = if steps_total == 0 {
        0.0
    } else {
        (count_missing_top1_steps as f64) / (steps_total as f64)
    };

    Top1Accounting {
        top1_policy_id: TOP1_POLICY_ID.to_string(),
        steps_total,
        count_missing_link_steps,
        count_missing_top1_steps,
        missing_link_step_rate,
        missing_top1_step_rate,
        max_missing_link_step_rate: MAX_MISSING_LINK_STEP_RATE,
        missing_link_step_rate_exceeds_threshold: missing_link_step_rate
            > MAX_MISSING_LINK_STEP_RATE,
        steps,
    }
}

pub fn evaluate_link_sanity(
    sample_reports: &[SampleLinkReport],
    explicitly_unrelated_sample_ids: &[u64],
) -> Result<LinkSanityResult, LinkSanityError> {
    evaluate_link_sanity_with_seed(
        sample_reports,
        explicitly_unrelated_sample_ids,
        LINK_SANITY_SEED,
    )
}

pub fn evaluate_link_sanity_with_seed(
    sample_reports: &[SampleLinkReport],
    explicitly_unrelated_sample_ids: &[u64],
    seed: u64,
) -> Result<LinkSanityResult, LinkSanityError> {
    let sample_ids: Vec<u64> = sample_reports
        .iter()
        .map(|report| report.sample_id)
        .collect();
    let selected_sample_ids = select_sanity_sample_ids(&sample_ids, seed, LINK_SANITY_K)?;
    let k_eff = selected_sample_ids.len();

    let report_by_id: BTreeMap<u64, &SampleLinkReport> = sample_reports
        .iter()
        .map(|report| (report.sample_id, report))
        .collect();
    let explicitly_unrelated: BTreeSet<u64> =
        explicitly_unrelated_sample_ids.iter().copied().collect();

    let mut records = Vec::with_capacity(selected_sample_ids.len());
    let mut unrelated_count = 0usize;
    let mut category_counts: BTreeMap<SanityCategory, usize> = BTreeMap::new();

    for sample_id in &selected_sample_ids {
        let report = report_by_id
            .get(sample_id)
            .expect("selected sample id must exist");
        let representative = report.top1.representative_top1();
        let (representative_ans_unit_id, selected_doc_unit_id, category, judgment) =
            match representative {
                Some((ans_unit_id, doc_unit_id)) => {
                    let judgment = if explicitly_unrelated.contains(sample_id) {
                        unrelated_count += 1;
                        SanityJudgment::Unrelated
                    } else {
                        SanityJudgment::Unreviewed
                    };
                    (
                        Some(ans_unit_id),
                        Some(doc_unit_id),
                        SanityCategory::DocUnitId(doc_unit_id),
                        judgment,
                    )
                }
                None => {
                    unrelated_count += 1;
                    (
                        None,
                        None,
                        SanityCategory::NoLink,
                        SanityJudgment::Unrelated,
                    )
                }
            };

        *category_counts.entry(category).or_insert(0) += 1;
        records.push(LinkSanityRecord {
            sample_id: *sample_id,
            representative_ans_unit_id,
            selected_doc_unit_id,
            category,
            judgment,
        });
    }

    let total = records.len() as f64;
    let observed_categories = category_counts.len();
    let mut entropy = 0.0_f64;
    let mut max_share = 0.0_f64;
    for count in category_counts.values() {
        let p = (*count as f64) / total;
        if p > 0.0 {
            entropy -= p * p.ln();
        }
        if p > max_share {
            max_share = p;
        }
    }
    let h_norm = if observed_categories > 1 {
        let h_max = (observed_categories as f64).ln();
        if h_max > 0.0 {
            entropy / h_max
        } else {
            0.0
        }
    } else {
        0.0
    };

    let link_sanity_fail = unrelated_count > LINK_SANITY_FAIL_UNRELATED_MAX;
    let random_like_link_collapse = h_norm > LINK_SANITY_RANDOM_LIKE_H_NORM_THRESHOLD;
    let dominant_link_collapse = max_share > LINK_SANITY_DOMINANT_SHARE_THRESHOLD;

    Ok(LinkSanityResult {
        link_sanity_id: LINK_SANITY_ID.to_string(),
        rng_id: LINK_SANITY_RNG_ID.to_string(),
        seed,
        sampling_id: LINK_SANITY_SAMPLING_ID.to_string(),
        selected_sample_ids,
        k_eff,
        records,
        unrelated_count,
        link_sanity_fail,
        h_norm,
        max_share,
        random_like_link_collapse,
        dominant_link_collapse,
    })
}

pub fn select_sanity_sample_ids(
    sample_ids: &[u64],
    seed: u64,
    k: usize,
) -> Result<Vec<u64>, LinkSanityError> {
    let mut unique_ids = BTreeSet::new();
    for sample_id in sample_ids {
        if !unique_ids.insert(*sample_id) {
            return Err(LinkSanityError::DuplicateSampleId {
                sample_id: *sample_id,
            });
        }
    }

    let mut scored: Vec<(u64, u64)> = unique_ids
        .into_iter()
        .map(|sample_id| (splitmix64(seed ^ sample_id), sample_id))
        .collect();
    scored.sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
    let k_eff = k.min(scored.len());
    Ok(scored
        .into_iter()
        .take(k_eff)
        .map(|(_, sample_id)| sample_id)
        .collect())
}

fn splitmix64(mut state: u64) -> u64 {
    state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(
        sample_id: u64,
        ans_unit_count: usize,
        doc_unit_count: usize,
        links_topk: &[(u32, u32, u16)],
    ) -> SampleLinksInput {
        SampleLinksInput {
            sample_id,
            ans_unit_count,
            doc_unit_count,
            links_topk: links_topk
                .iter()
                .map(|(ans_unit_id, doc_unit_id, rank)| LinkRow {
                    ans_unit_id: *ans_unit_id,
                    doc_unit_id: *doc_unit_id,
                    rank: *rank,
                })
                .collect(),
        }
    }

    fn link_report_with_top1(sample_id: u64, doc_unit_id: u32) -> SampleLinkReport {
        process_sample_links(&sample(
            sample_id,
            1,
            128,
            &[(0, doc_unit_id, 1), (0, (doc_unit_id + 1) % 128, 2)],
        ))
    }

    fn link_report_with_no_link(sample_id: u64) -> SampleLinkReport {
        process_sample_links(&sample(sample_id, 1, 128, &[]))
    }

    #[test]
    fn canonicalization_drops_invalid_ids_ranks_and_dedups() {
        let canonicalized = canonicalize_links(&sample(
            10,
            3,
            4,
            &[
                (1, 3, 2),
                (1, 1, 1),
                (1, 1, 3),
                (1, 2, 4),
                (1, 2, 2),
                (2, 0, 0),
                (2, 1, 9),
                (3, 1, 1),
                (2, 4, 1),
            ],
        ));

        assert_eq!(canonicalized.counters.count_invalid_rank_links, 2);
        assert_eq!(canonicalized.counters.count_invalid_ans_unit_id, 1);
        assert_eq!(canonicalized.counters.count_invalid_doc_unit_id, 1);
        assert_eq!(canonicalized.counters.count_link_dedup, 2);

        let ans1 = canonicalized.links_by_answer[1]
            .iter()
            .map(|link| (link.rank, link.doc_unit_id))
            .collect::<Vec<_>>();
        assert_eq!(ans1, vec![(1, 1), (2, 2), (2, 3)]);
    }

    #[test]
    fn canonicalization_dedup_keeps_smallest_rank_for_same_pair() {
        let canonicalized = canonicalize_links(&sample(11, 1, 4, &[(0, 2, 2), (0, 2, 1)]));
        assert_eq!(canonicalized.counters.count_link_dedup, 1);
        assert_eq!(
            canonicalized.links_by_answer[0],
            vec![CanonicalLink {
                doc_unit_id: 2,
                rank: 1
            }]
        );
    }

    #[test]
    fn top1_separates_missing_link_from_missing_rank1() {
        let report = process_sample_links(&sample(
            20,
            3,
            3,
            &[(0, 0, 2), (0, 1, 3), (2, 2, 1), (2, 1, 2)],
        ));

        assert_eq!(report.top1.steps_total, 3);
        assert_eq!(report.top1.count_missing_link_steps, 1);
        assert_eq!(report.top1.count_missing_top1_steps, 1);
        assert_eq!(
            report.top1.steps,
            vec![
                Top1Step::MissingTop1 { ans_unit_id: 0 },
                Top1Step::MissingLink { ans_unit_id: 1 },
                Top1Step::Selected {
                    ans_unit_id: 2,
                    doc_unit_id: 2
                },
            ]
        );
        assert!((report.top1.missing_link_step_rate - (1.0 / 3.0)).abs() < 1e-12);
        assert!((report.top1.missing_top1_step_rate - (1.0 / 3.0)).abs() < 1e-12);
        assert!(report.top1.missing_link_step_rate_exceeds_threshold);
    }

    #[test]
    fn top1_accounts_missing_links_across_full_answer_unit_range() {
        let report = process_sample_links(&sample(21, 3, 3, &[(2, 1, 1)]));
        assert_eq!(report.top1.steps_total, 3);
        assert_eq!(report.top1.count_missing_link_steps, 2);
        assert_eq!(report.top1.count_missing_top1_steps, 0);
        assert_eq!(
            report.top1.steps,
            vec![
                Top1Step::MissingLink { ans_unit_id: 0 },
                Top1Step::MissingLink { ans_unit_id: 1 },
                Top1Step::Selected {
                    ans_unit_id: 2,
                    doc_unit_id: 1
                }
            ]
        );
    }

    #[test]
    fn top1_rank1_tie_breaks_on_min_doc_unit_id() {
        let report = process_sample_links(&sample(30, 1, 8, &[(0, 3, 1), (0, 1, 1), (0, 2, 2)]));
        assert_eq!(report.canonicalized.counters.count_multi_rank1_links, 1);
        assert_eq!(
            report.top1.steps,
            vec![Top1Step::Selected {
                ans_unit_id: 0,
                doc_unit_id: 1
            }]
        );
        assert_eq!(report.top1.representative_top1(), Some((0, 1)));
    }

    #[test]
    fn sanity_sampler_snapshot_for_fixed_n32_k16() {
        let ids: Vec<u64> = (0..32_u64).collect();
        let selected = select_sanity_sample_ids(&ids, LINK_SANITY_SEED, LINK_SANITY_K).unwrap();

        assert_eq!(
            selected,
            vec![5, 18, 26, 25, 1, 12, 21, 17, 8, 16, 9, 4, 30, 0, 3, 10]
        );

        let mut reversed = ids.clone();
        reversed.reverse();
        let selected_reversed =
            select_sanity_sample_ids(&reversed, LINK_SANITY_SEED, LINK_SANITY_K).unwrap();
        assert_eq!(selected_reversed, selected);
    }

    #[test]
    fn sanity_sampler_uses_all_samples_when_n_below_k() {
        let ids: Vec<u64> = (0..10_u64).collect();
        let selected = select_sanity_sample_ids(&ids, LINK_SANITY_SEED, LINK_SANITY_K).unwrap();
        assert_eq!(selected.len(), 10);

        let reports: Vec<SampleLinkReport> = ids
            .iter()
            .map(|sample_id| link_report_with_top1(*sample_id, *sample_id as u32))
            .collect();
        let sanity = evaluate_link_sanity(&reports, &[]).unwrap();
        assert_eq!(sanity.k_eff, 10);
        assert_eq!(sanity.selected_sample_ids.len(), 10);
    }

    #[test]
    fn link_sanity_fail_triggers_on_unrelated_over_threshold() {
        let mut reports = Vec::new();
        for sample_id in 0..9_u64 {
            reports.push(link_report_with_no_link(sample_id));
        }
        for sample_id in 9..16_u64 {
            reports.push(link_report_with_top1(sample_id, sample_id as u32));
        }

        let sanity = evaluate_link_sanity(&reports, &[]).unwrap();
        assert_eq!(sanity.selected_sample_ids.len(), LINK_SANITY_K);
        assert_eq!(sanity.k_eff, LINK_SANITY_K);
        assert_eq!(sanity.unrelated_count, 9);
        assert!(sanity.link_sanity_fail);
    }

    #[test]
    fn random_like_collapse_triggers_on_high_normalized_entropy() {
        let reports: Vec<SampleLinkReport> = (0..16_u64)
            .map(|sample_id| link_report_with_top1(sample_id, sample_id as u32))
            .collect();

        let sanity = evaluate_link_sanity(&reports, &[]).unwrap();
        assert_eq!(sanity.unrelated_count, 0);
        assert!(sanity
            .records
            .iter()
            .all(|record| record.judgment == SanityJudgment::Unreviewed));
        assert!(sanity.h_norm > LINK_SANITY_RANDOM_LIKE_H_NORM_THRESHOLD);
        assert!(sanity.random_like_link_collapse);
        assert!(!sanity.dominant_link_collapse);
    }

    #[test]
    fn dominant_collapse_triggers_on_max_share_over_half() {
        let mut reports: Vec<SampleLinkReport> = (0..9_u64)
            .map(|sample_id| link_report_with_top1(sample_id, 7))
            .collect();
        reports.extend(
            (9..16_u64).map(|sample_id| link_report_with_top1(sample_id, sample_id as u32)),
        );

        let sanity = evaluate_link_sanity(&reports, &[]).unwrap();
        assert!(sanity.max_share > LINK_SANITY_DOMINANT_SHARE_THRESHOLD);
        assert!(sanity.dominant_link_collapse);
        assert!(!sanity.random_like_link_collapse);
    }

    #[test]
    fn entropy_single_category_has_zero_h_norm_without_nan() {
        let reports: Vec<SampleLinkReport> = (0..16_u64)
            .map(|sample_id| link_report_with_top1(sample_id, 7))
            .collect();

        let sanity = evaluate_link_sanity(&reports, &[]).unwrap();
        assert!(sanity.h_norm.is_finite());
        assert_eq!(sanity.h_norm, 0.0);
        assert_eq!(sanity.max_share, 1.0);
        assert!(sanity.dominant_link_collapse);
        assert!(!sanity.random_like_link_collapse);
    }
}
