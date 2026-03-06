# CFA Case Study Index

- date: `2026-03-05`
- dataset: `data/cfa/cfa_v1.jsonl`
- dataset_sha256: `76658a4ee2230460ae7525ffc5c488fe600b1b9ae51df3cc77d1b868e874e7d2`
- batch_report: [2026-03-05_cfa_batch_primaryE_report.txt](attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt)

## Commands
```bash
python tools/run_cfa_case_representative.py --results-jsonl runs/cfa_batch_primaryE/results.jsonl --cfa-jsonl data/cfa/cfa_v1.jsonl --batch-report attestations/triality/2026-03-05_cfa_batch_primaryE_report.txt --group-size 5 --seed 7 --model-id Qwen/Qwen2.5-1.5B --device auto --topk 128 --perm-R 0 --min-coverage 0.3
```

## Selected Samples
- top: `127,141,153,75,117`
- median: `199,79,139,31,187`
- bottom: `135,109,155,17,173`

## Script SHA256
- `C:/Users/aoika/Documents/GitHub/pale-ale/tools/eval_local_span.py`: `2a29a8914ca1163b95ff9669df70dd1b4d9f1ee0375a0b0101741ec8a9ee2b29`
- `C:/Users/aoika/Documents/GitHub/pale-ale/tools/eval_triality_token.py`: `efc648f44a5c5f04fcf68e95f85864f5b2bbb8436e0d1ad3e4b71790f6607cf3`
- `C:/Users/aoika/Documents/GitHub/pale-ale/tools/extract_triality_triplets.py`: `d30c2415762aa89fb000e45dc62d5a9824500676dbd5ae94568524c38e8d0035`
- `C:/Users/aoika/Documents/GitHub/pale-ale/tools/labels_from_cfa_spans.py`: `e9d451669334dab1dfaeebdda48d378941af50bd1c73ef5625b9f3f8c66557eb`
- `C:/Users/aoika/Documents/GitHub/pale-ale/tools/plot_cfa_case_pair.py`: `df236deea84c493851c088b77092c8b33643e3369315657789afb5b80e91e5a6`
- `C:/Users/aoika/Documents/GitHub/pale-ale/tools/run_cfa_case_representative.py`: `06edf7593acefddaf1173bc7768229050a09ba4a668a6bed84aa70e2e51775e9`
