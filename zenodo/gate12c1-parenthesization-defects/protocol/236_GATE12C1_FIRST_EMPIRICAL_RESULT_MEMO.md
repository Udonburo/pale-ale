# Gate12C-1 First Empirical Result Memo

Status: tracked empirical result memo
Role: first blind Gate12C-1 equal-rank alpha empirical execution result over the canonical twelve Gate12A source runs; not a Gate12B overlay, not a Type-III claim, and not a physical nonassociativity claim
Date: 2026-06-26

## 0. Execution Root Provenance

```text
abandoned_root_path = C:\Users\aoika\Documents\GitHub\pale-ale-gate12c1-empirical\runs\gate12c1_first_empirical_v1
abandoned_root_reason = interrupted prior Codex thread; root existed before authorized clean execution
abandoned_root_inspected = false
abandoned_root_used = false
clean_execution_root = C:\Users\aoika\Documents\GitHub\pale-ale-gate12c1-empirical\runs\gate12c1_first_empirical_v1_clean_001
```

The abandoned root was not opened, listed, parsed, moved, deleted, reused, summarized, or committed. This memo records only the clean explicitly versioned execution root above.

## 1. Result

| field | value |
| --- | --- |
| execution_status | complete |
| grid_outcome | no_directional_support |
| supporting_run_count | 0 |
| q_discordant_run_count | 0 |
| coverage_limited_endpoint_count | 0 |
| primary_zero_tolerance | 1e-12 |
| coverage_threshold | 0.9 |
| holm_alpha | 0.05 |
| holm_endpoint_count | 24 |

The summary tool classification is used as emitted. This memo does not reclassify the grid manually.

## 2. Provenance

| artifact | value |
| --- | --- |
| frozen_runner_worktree | C:\Users\aoika\Documents\GitHub\pale-ale-gate12c1-frozen-runner-lf |
| frozen_runner_commit | 8d5613bffe5b6c91d0956c812404072eb76e98c6 |
| frozen_runner_script_sha256 | b363fd874a0538dc548853e97e8ec17c0eb84be5658f6e2f01f60d2a12789c3e |
| summary_worktree | C:\Users\aoika\Documents\GitHub\pale-ale-gate12c1-empirical |
| summary_tool_commit | 6b66f49710f4b7cbdae7e5a282d14d1d81723b30 |
| summary_tool_script_sha256 | 9e527c0363875de912f9abf9ba38acb462a13b0b62da6e14be35d4c3526df708 |
| case_manifest_sha256 | dae1f2690fc707a166277dac143aafcfa5e51e23934428861ddc48cac97ad704 |
| case_manifest_path | C:\Users\aoika\Documents\GitHub\pale-ale-gate12c1-empirical\runs\gate12c1_first_empirical_v1_clean_001\case_manifest.json |
| summary_output_dir | C:\Users\aoika\Documents\GitHub\pale-ale-gate12c1-empirical\runs\gate12c1_first_empirical_v1_clean_001\summary |
| generated_runs_committed | false |

## 3. Frozen Settings

| setting | value |
| --- | --- |
| orientation_null_seed | gate12c1_first_empirical_orientation_null_v1 |
| orientation_null_requested_draw_count | 255 |
| orientation_null_max_attempt_count | 1024 |
| tau_overlap_sv_min | 1e-8 |
| tau_overlap_singular_value_abs_error | 1e-8 |
| tau_transport_reconstruction_fro | 1e-8 |
| tau_ordinary_associator_fro | 1e-10 |
| tau_no_compression_associator_fro | 1e-10 |
| tau_split_rel | 1e-3 |
| tau_gauge_operator_covariance_fro | 1e-8 |
| tau_gauge_scalar_delta_abs | 1e-10 |
| epsilon | 1e-12 |

## 4. Case Inventory

| case | model | family | source_run_id | gate12c1_run_id | preflight_cycles | derived_cycles | expected_blocks | mixed_expected_cycles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_01 | qwen_qwen2_5_0_5b | transcript_128r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_transcript_128r_gate9k | case_01 | 320 | 320 | 128 | 0 |
| case_02 | qwen_qwen2_5_0_5b | briefing_200r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_briefing_200r_gate9k | case_02 | 500 | 500 | 200 | 0 |
| case_03 | qwen_qwen2_5_0_5b | archive_128r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_0_5b_archive_128r_gate9k | case_03 | 320 | 320 | 128 | 0 |
| case_04 | qwen_qwen2_5_3b_instruct | transcript_128r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_transcript_128r_gate9k | case_04 | 320 | 320 | 128 | 0 |
| case_05 | qwen_qwen2_5_3b_instruct | briefing_200r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_briefing_200r_gate9k | case_05 | 500 | 500 | 200 | 0 |
| case_06 | qwen_qwen2_5_3b_instruct | archive_128r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen2_5_3b_instruct_archive_128r_gate9k | case_06 | 320 | 320 | 128 | 0 |
| case_07 | meta_llama_llama_3_2_3b_instruct | transcript_128r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_transcript_128r_gate9k | case_07 | 320 | 320 | 128 | 0 |
| case_08 | meta_llama_llama_3_2_3b_instruct | briefing_200r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_briefing_200r_gate9k | case_08 | 500 | 500 | 200 | 0 |
| case_09 | meta_llama_llama_3_2_3b_instruct | archive_128r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_meta_llama_llama_3_2_3b_instruct_archive_128r_gate9k | case_09 | 320 | 320 | 128 | 0 |
| case_10 | qwen_qwen3_4b | transcript_128r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_transcript_128r_gate9k | case_10 | 320 | 320 | 128 | 0 |
| case_11 | qwen_qwen3_4b | briefing_200r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_briefing_200r_gate9k | case_11 | 500 | 500 | 200 | 0 |
| case_12 | qwen_qwen3_4b | archive_128r | gate12a_discrete_connection_recheck_from_gate12a_upstream_gate8cm_qwen_qwen3_4b_archive_128r_gate9k | case_12 | 320 | 320 | 128 | 0 |

## 5. Mechanical Contract Status

| field | value |
| --- | --- |
| intended_case_count | 12 |
| found_case_count | 12 |
| executed_case_count | 12 |
| source_grid_preflight_status | pass |
| source_run_id_status | pass |
| post_grid_source_immutability_status | pass |
| gate12c1_output_checksum_status | pass |
| summary_checksum_status | pass |

| case | exit | files | manifest/settings | process_status | checksums | source_immutability | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| case_01 | 0 | true | pass | pass | pass | pass | 229.691 |
| case_02 | 0 | true | pass | pass | pass | pass | 387.91 |
| case_03 | 0 | true | pass | pass | pass | pass | 228.55 |
| case_04 | 0 | true | pass | pass | pass | pass | 307.268 |
| case_05 | 0 | true | pass | pass | pass | pass | 422.953 |
| case_06 | 0 | true | pass | pass | pass | pass | 219.042 |
| case_07 | 0 | true | pass | pass | pass | pass | 190.911 |
| case_08 | 0 | true | pass | pass | pass | pass | 376.746 |
| case_09 | 0 | true | pass | pass | pass | pass | 221.954 |
| case_10 | 0 | true | pass | pass | pass | pass | 230.056 |
| case_11 | 0 | true | pass | pass | pass | pass | 294.534 |
| case_12 | 0 | true | pass | pass | pass | pass | 184.964 |

## 6. Primary Run/q Endpoints

| case | q | expected_cycles | represented_cycles | cycle_cov | expected_blocks | represented_blocks | block_cov | pos | neg | tie | test_status | run_q_median | raw_p | holm_p | q_support |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_01 | 1 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -3.5740928050950265 | 1.0 | 1.0 | false |
| case_01 | 2 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -2.001353884374879 | 1.0 | 1.0 | false |
| case_02 | 1 | 500 | 500 | 1.0 | 200 | 200 | 1.0 | 0 | 200 | 0 | informative | -3.2684813641791877 | 1.0 | 1.0 | false |
| case_02 | 2 | 500 | 500 | 1.0 | 200 | 200 | 1.0 | 0 | 200 | 0 | informative | -2.221681568725521 | 1.0 | 1.0 | false |
| case_03 | 1 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -3.2122596810225494 | 1.0 | 1.0 | false |
| case_03 | 2 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 7 | 121 | 0 | informative | -2.1161873154346025 | 1.0 | 1.0 | false |
| case_04 | 1 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -3.5549637614382585 | 1.0 | 1.0 | false |
| case_04 | 2 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -2.526255967842035 | 1.0 | 1.0 | false |
| case_05 | 1 | 500 | 500 | 1.0 | 200 | 200 | 1.0 | 0 | 200 | 0 | informative | -3.080007754937126 | 1.0 | 1.0 | false |
| case_05 | 2 | 500 | 500 | 1.0 | 200 | 200 | 1.0 | 0 | 200 | 0 | informative | -2.6479499637931636 | 1.0 | 1.0 | false |
| case_06 | 1 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -2.8845077502868004 | 1.0 | 1.0 | false |
| case_06 | 2 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -2.1779887614208797 | 1.0 | 1.0 | false |
| case_07 | 1 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -2.798489931764285 | 1.0 | 1.0 | false |
| case_07 | 2 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -2.298431632289879 | 1.0 | 1.0 | false |
| case_08 | 1 | 500 | 500 | 1.0 | 200 | 200 | 1.0 | 0 | 200 | 0 | informative | -2.8926261334664254 | 1.0 | 1.0 | false |
| case_08 | 2 | 500 | 500 | 1.0 | 200 | 200 | 1.0 | 0 | 200 | 0 | informative | -2.2744405016358136 | 1.0 | 1.0 | false |
| case_09 | 1 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -2.8693987965738312 | 1.0 | 1.0 | false |
| case_09 | 2 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -2.2478015310493333 | 1.0 | 1.0 | false |
| case_10 | 1 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -3.1795927316313013 | 1.0 | 1.0 | false |
| case_10 | 2 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 1 | 127 | 0 | informative | -1.8074924473552518 | 1.0 | 1.0 | false |
| case_11 | 1 | 500 | 500 | 1.0 | 200 | 200 | 1.0 | 0 | 200 | 0 | informative | -3.1565415910695913 | 1.0 | 1.0 | false |
| case_11 | 2 | 500 | 500 | 1.0 | 200 | 200 | 1.0 | 9 | 191 | 0 | informative | -1.6461179757112587 | 1.0 | 1.0 | false |
| case_12 | 1 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -3.2721566391421177 | 1.0 | 1.0 | false |
| case_12 | 2 | 320 | 320 | 1.0 | 128 | 128 | 1.0 | 0 | 128 | 0 | informative | -1.443656902121825 | 1.0 | 1.0 | false |

## 7. Run Support

| case | order | model | family | q1_support | q2_support | run_support | q_discordant_run |
| --- | --- | --- | --- | --- | --- | --- | --- |
| case_01 | 1 | qwen_qwen2_5_0_5b | transcript_128r | false | false | false | false |
| case_02 | 2 | qwen_qwen2_5_0_5b | briefing_200r | false | false | false | false |
| case_03 | 3 | qwen_qwen2_5_0_5b | archive_128r | false | false | false | false |
| case_04 | 4 | qwen_qwen2_5_3b_instruct | transcript_128r | false | false | false | false |
| case_05 | 5 | qwen_qwen2_5_3b_instruct | briefing_200r | false | false | false | false |
| case_06 | 6 | qwen_qwen2_5_3b_instruct | archive_128r | false | false | false | false |
| case_07 | 7 | meta_llama_llama_3_2_3b_instruct | transcript_128r | false | false | false | false |
| case_08 | 8 | meta_llama_llama_3_2_3b_instruct | briefing_200r | false | false | false | false |
| case_09 | 9 | meta_llama_llama_3_2_3b_instruct | archive_128r | false | false | false | false |
| case_10 | 10 | qwen_qwen3_4b | transcript_128r | false | false | false | false |
| case_11 | 11 | qwen_qwen3_4b | briefing_200r | false | false | false | false |
| case_12 | 12 | qwen_qwen3_4b | archive_128r | false | false | false | false |

## 8. Secondary Descriptive Telemetry

Secondary telemetry is descriptive only and is not Holm-corrected.

### 8.1 Run/q Secondary Telemetry

| case | q | robust_z | p_min | p25 | p50 | p75 | p_max | assoc_rel_median | root_spread_median | scale_deg_rate | incomplete_null_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_01 | 1 | -1.4792726598962314 | 0.8828125 | 0.99609375 | 1.0 | 1.0 | 1.0 | 0.004120098818475101 | 0.9634808759643512 | 0.0 | 0.0 |
| case_01 | 2 | -1.9299937926343975 | 0.66015625 | 0.9921875 | 1.0 | 1.0 | 1.0 | 0.0027028438087318392 | 0.7476921368277261 | 0.0 | 0.0 |
| case_02 | 1 | -1.515410380291522 | 0.85546875 | 0.99609375 | 1.0 | 1.0 | 1.0 | 0.006574396383569684 | 0.9846080567338882 | 0.0 | 0.0 |
| case_02 | 2 | -2.045038371146461 | 0.51953125 | 0.99609375 | 1.0 | 1.0 | 1.0 | 0.0026899320803970145 | 0.7908065744437176 | 0.0 | 0.0 |
| case_03 | 1 | -1.4571909758770851 | 0.90234375 | 0.99609375 | 1.0 | 1.0 | 1.0 | 0.006296288674522178 | 1.020122728224502 | 0.0 | 0.0 |
| case_03 | 2 | -2.0247595267341043 | 0.03515625 | 0.97265625 | 1.0 | 1.0 | 1.0 | 0.002455014224441366 | 0.6960455097927779 | 0.0 | 0.0 |
| case_04 | 1 | -1.4972371446839832 | 0.94140625 | 1.0 | 1.0 | 1.0 | 1.0 | 0.00451208202969641 | 0.9442832866347934 | 0.0 | 0.0 |
| case_04 | 2 | -2.0676231735107002 | 0.81640625 | 0.99609375 | 1.0 | 1.0 | 1.0 | 0.0017390317427412123 | 0.8645025776248207 | 0.0 | 0.0 |
| case_05 | 1 | -1.4717675887280501 | 0.87109375 | 0.99609375 | 1.0 | 1.0 | 1.0 | 0.007853936181070702 | 1.0207082420311242 | 0.0 | 0.0 |
| case_05 | 2 | -2.3021216961730833 | 0.609375 | 1.0 | 1.0 | 1.0 | 1.0 | 0.0028473218640593196 | 0.6968959177478433 | 0.0 | 0.0 |
| case_06 | 1 | -1.494856209820473 | 0.87890625 | 0.9921875 | 1.0 | 1.0 | 1.0 | 0.009708789431298902 | 0.8263040256368033 | 0.0 | 0.0 |
| case_06 | 2 | -2.3196704760868436 | 0.16015625 | 0.9765625 | 0.99609375 | 1.0 | 1.0 | 0.0037946255042643406 | 0.7940447185984594 | 0.0 | 0.0 |
| case_07 | 1 | -1.7783997345686795 | 0.9296875 | 1.0 | 1.0 | 1.0 | 1.0 | 0.013629658442039647 | 1.024182271657787 | 0.0 | 0.0 |
| case_07 | 2 | -1.8832677006008123 | 0.49609375 | 0.98828125 | 0.99609375 | 1.0 | 1.0 | 0.01411976146542138 | 0.6975351753740908 | 0.0 | 0.0 |
| case_08 | 1 | -1.5889865236352547 | 0.91796875 | 1.0 | 1.0 | 1.0 | 1.0 | 0.015295179533558505 | 0.9531515997691704 | 0.0 | 0.0 |
| case_08 | 2 | -2.278648116876827 | 0.62109375 | 0.984375 | 0.99609375 | 1.0 | 1.0 | 0.01655629346121433 | 0.8387664862795335 | 0.0 | 0.0 |
| case_09 | 1 | -1.6130098847309535 | 0.8515625 | 0.99609375 | 1.0 | 1.0 | 1.0 | 0.014696159539552773 | 0.978467121735233 | 0.0 | 0.0 |
| case_09 | 2 | -2.265560776160343 | 0.6015625 | 0.9765625 | 0.99609375 | 1.0 | 1.0 | 0.01307910686409611 | 0.9910708199849448 | 0.0 | 0.0 |
| case_10 | 1 | -2.0732736794013435 | 0.9609375 | 1.0 | 1.0 | 1.0 | 1.0 | 0.011388427612326911 | 1.2031774613824109 | 0.0 | 0.0 |
| case_10 | 2 | -1.6212503854573626 | 0.23828125 | 0.87109375 | 0.9921875 | 1.0 | 1.0 | 0.04753600732534585 | 1.0314648391494012 | 0.0 | 0.0 |
| case_11 | 1 | -1.8963385738815683 | 0.97265625 | 1.0 | 1.0 | 1.0 | 1.0 | 0.010603658254180408 | 1.399051077588513 | 0.0 | 0.0 |
| case_11 | 2 | -1.6564995740043695 | 0.21875 | 0.9140625 | 0.98828125 | 1.0 | 1.0 | 0.03562654041831466 | 1.2568644537161688 | 0.0 | 0.0 |
| case_12 | 1 | -2.1157193003077586 | 0.9140625 | 1.0 | 1.0 | 1.0 | 1.0 | 0.012188372835881551 | 1.327790088189977 | 0.0 | 0.0 |
| case_12 | 2 | -1.5935349256917246 | 0.38671875 | 0.87890625 | 0.984375 | 1.0 | 1.0 | 0.042664376830252655 | 1.2535266552399429 | 0.0 | 0.0 |

### 8.2 q Difference

| case | order | q2_minus_q1_median |
| --- | --- | --- |
| case_01 | 1 | 1.5727389207201474 |
| case_02 | 2 | 1.0467997954536665 |
| case_03 | 3 | 1.096072365587947 |
| case_04 | 4 | 1.0287077935962237 |
| case_05 | 5 | 0.43205779114396226 |
| case_06 | 6 | 0.7065189888659207 |
| case_07 | 7 | 0.5000582994744058 |
| case_08 | 8 | 0.6181856318306118 |
| case_09 | 9 | 0.6215972655244979 |
| case_10 | 10 | 1.3721002842760495 |
| case_11 | 11 | 1.5104236153583326 |
| case_12 | 12 | 1.8284997370202927 |

### 8.3 Low-Holonomy Secondary Surface

| case | q | selected_expected_cycles | selected_valid_cycles | cycle_cov | selected_expected_blocks | selected_blocks | block_cov | mixed_selected | low_holonomy_run_q_median |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| case_01 | 1 | 80 | 80 | 1.0 | 30 | 30 | 1.0 | 0 | -2.8971485163740436 |
| case_01 | 2 | 80 | 80 | 1.0 | 30 | 30 | 1.0 | 0 | -2.791634788533033 |
| case_02 | 1 | 125 | 125 | 1.0 | 55 | 55 | 1.0 | 0 | -3.2310563283805447 |
| case_02 | 2 | 125 | 125 | 1.0 | 55 | 55 | 1.0 | 0 | -2.5251195555623567 |
| case_03 | 1 | 80 | 80 | 1.0 | 36 | 36 | 1.0 | 0 | -3.3108899950986146 |
| case_03 | 2 | 80 | 80 | 1.0 | 36 | 36 | 1.0 | 0 | -2.309693274456104 |
| case_04 | 1 | 80 | 80 | 1.0 | 27 | 27 | 1.0 | 0 | -3.4392455391431866 |
| case_04 | 2 | 80 | 80 | 1.0 | 27 | 27 | 1.0 | 0 | -2.6529874750567206 |
| case_05 | 1 | 125 | 125 | 1.0 | 52 | 52 | 1.0 | 0 | -3.088455673519549 |
| case_05 | 2 | 125 | 125 | 1.0 | 52 | 52 | 1.0 | 0 | -2.918634001127014 |
| case_06 | 1 | 80 | 80 | 1.0 | 37 | 37 | 1.0 | 0 | -2.861483687809085 |
| case_06 | 2 | 80 | 80 | 1.0 | 37 | 37 | 1.0 | 0 | -2.3771069830123683 |
| case_07 | 1 | 80 | 80 | 1.0 | 32 | 32 | 1.0 | 0 | -2.9813909270675074 |
| case_07 | 2 | 80 | 80 | 1.0 | 32 | 32 | 1.0 | 0 | -2.538379446664572 |
| case_08 | 1 | 125 | 125 | 1.0 | 63 | 63 | 1.0 | 0 | -3.1458899536433114 |
| case_08 | 2 | 125 | 125 | 1.0 | 63 | 63 | 1.0 | 0 | -2.4821399590598743 |
| case_09 | 1 | 80 | 80 | 1.0 | 38 | 38 | 1.0 | 0 | -3.134004622969755 |
| case_09 | 2 | 80 | 80 | 1.0 | 38 | 38 | 1.0 | 0 | -2.516077078508278 |
| case_10 | 1 | 80 | 80 | 1.0 | 39 | 39 | 1.0 | 0 | -3.287730900182171 |
| case_10 | 2 | 80 | 80 | 1.0 | 39 | 39 | 1.0 | 0 | -1.666645099271142 |
| case_11 | 1 | 125 | 125 | 1.0 | 53 | 53 | 1.0 | 0 | -3.3594679975511585 |
| case_11 | 2 | 125 | 125 | 1.0 | 53 | 53 | 1.0 | 0 | -2.1602587947778265 |
| case_12 | 1 | 80 | 80 | 1.0 | 36 | 36 | 1.0 | 0 | -3.4313369036207257 |
| case_12 | 2 | 80 | 80 | 1.0 | 36 | 36 | 1.0 | 0 | -1.212703720792157 |

### 8.4 Spearman Correlations

| case | q | predictor | status | rho | cycle_q_count |
| --- | --- | --- | --- | --- | --- |
| case_01 | 1 | gate12a_holonomy_residual_fro | defined | -0.41565235797621997 | 320 |
| case_01 | 1 | edge_compatibility_gap_max | defined | -0.6005475340820472 | 320 |
| case_01 | 2 | gate12a_holonomy_residual_fro | defined | 0.652215884465568 | 320 |
| case_01 | 2 | edge_compatibility_gap_max | defined | 0.7269051766152961 | 320 |
| case_02 | 1 | gate12a_holonomy_residual_fro | defined | 0.07085431912786234 | 500 |
| case_02 | 1 | edge_compatibility_gap_max | defined | -0.002024274688131627 | 500 |
| case_02 | 2 | gate12a_holonomy_residual_fro | defined | 0.177761888639569 | 500 |
| case_02 | 2 | edge_compatibility_gap_max | defined | -0.025667804965526755 | 500 |
| case_03 | 1 | gate12a_holonomy_residual_fro | defined | -0.06815943327471236 | 320 |
| case_03 | 1 | edge_compatibility_gap_max | defined | -0.018886451399055933 | 320 |
| case_03 | 2 | gate12a_holonomy_residual_fro | defined | 0.3535125261187311 | 320 |
| case_03 | 2 | edge_compatibility_gap_max | defined | 0.4239374139729684 | 320 |
| case_04 | 1 | gate12a_holonomy_residual_fro | defined | -0.021280590998390834 | 320 |
| case_04 | 1 | edge_compatibility_gap_max | defined | -0.15195687035754035 | 320 |
| case_04 | 2 | gate12a_holonomy_residual_fro | defined | 0.05190333562689293 | 320 |
| case_04 | 2 | edge_compatibility_gap_max | defined | 0.4518894402414604 | 320 |
| case_05 | 1 | gate12a_holonomy_residual_fro | defined | -0.1589577025357064 | 500 |
| case_05 | 1 | edge_compatibility_gap_max | defined | -0.30019110372525337 | 500 |
| case_05 | 2 | gate12a_holonomy_residual_fro | defined | 0.21163009807425176 | 500 |
| case_05 | 2 | edge_compatibility_gap_max | defined | 0.16162614374661669 | 500 |
| case_06 | 1 | gate12a_holonomy_residual_fro | defined | 0.1504605244968847 | 320 |
| case_06 | 1 | edge_compatibility_gap_max | defined | 0.025927841803332585 | 320 |
| case_06 | 2 | gate12a_holonomy_residual_fro | defined | 0.5458121210938272 | 320 |
| case_06 | 2 | edge_compatibility_gap_max | defined | 0.4727816897790206 | 320 |
| case_07 | 1 | gate12a_holonomy_residual_fro | defined | 0.4880718886480743 | 320 |
| case_07 | 1 | edge_compatibility_gap_max | defined | 0.6261949170003959 | 320 |
| case_07 | 2 | gate12a_holonomy_residual_fro | defined | 0.42294795569426713 | 320 |
| case_07 | 2 | edge_compatibility_gap_max | defined | 0.505507888479614 | 320 |
| case_08 | 1 | gate12a_holonomy_residual_fro | defined | 0.8044133969595095 | 500 |
| case_08 | 1 | edge_compatibility_gap_max | defined | 0.15850995488158612 | 500 |
| case_08 | 2 | gate12a_holonomy_residual_fro | defined | 0.6208492822975121 | 500 |
| case_08 | 2 | edge_compatibility_gap_max | defined | 0.45336939368732954 | 500 |
| case_09 | 1 | gate12a_holonomy_residual_fro | defined | 0.7889729870079013 | 320 |
| case_09 | 1 | edge_compatibility_gap_max | defined | 0.7921054359168233 | 320 |
| case_09 | 2 | gate12a_holonomy_residual_fro | defined | 0.6418208407022593 | 320 |
| case_09 | 2 | edge_compatibility_gap_max | defined | 0.5734687369813761 | 320 |
| case_10 | 1 | gate12a_holonomy_residual_fro | defined | 0.48836586524567477 | 320 |
| case_10 | 1 | edge_compatibility_gap_max | defined | 0.6189189466345544 | 320 |
| case_10 | 2 | gate12a_holonomy_residual_fro | defined | -0.191446380183695 | 320 |
| case_10 | 2 | edge_compatibility_gap_max | defined | -0.5772698125325445 | 320 |
| case_11 | 1 | gate12a_holonomy_residual_fro | defined | 0.5152404573276004 | 500 |
| case_11 | 1 | edge_compatibility_gap_max | defined | 0.5768210025319584 | 500 |
| case_11 | 2 | gate12a_holonomy_residual_fro | defined | 0.23116254619808593 | 500 |
| case_11 | 2 | edge_compatibility_gap_max | defined | -0.270475205419742 | 500 |
| case_12 | 1 | gate12a_holonomy_residual_fro | defined | 0.7067777798224786 | 320 |
| case_12 | 1 | edge_compatibility_gap_max | defined | 0.7159725180615971 | 320 |
| case_12 | 2 | gate12a_holonomy_residual_fro | defined | -0.31052613220034003 | 320 |
| case_12 | 2 | edge_compatibility_gap_max | defined | -0.6896899313645726 | 320 |

## 9. Explicit Non-Claims

- No Type-III claim is made.
- No physical nonassociativity claim is made.
- No model-quality ranking is made.
- No correctness classifier is defined or implied.
- No Gate12B overlay was consumed.
- No rectangular-rank result is reported.
- Row p-values are descriptive only.
- Secondary telemetry is descriptive only.
- No weight-level causal claim is made.

## 10. Boundary

The generated run root remains untracked and is not part of this commit. This tracked memo records the frozen blind 12-case execution, deterministic summary classification, and explicit claim boundaries only.
