# Revision Analysis Results — Manifest

Generated: 2026-06-05 (after advisor meeting on the same date).

## SI Figures (manuscript-ready PDFs in `[word-color-association]final_draft/`)

| SI | 파일명 | 분석 (소스 csv·md) | 핵심 결과 |
|---|---|---|---|
| S5 | `S5 Fig.pdf` | R1.1 weighting sensitivity (`R1_1_weighting_sensitivity_*`) | 10색·α∈[0.1,1.0]·y±0.03; 8/10 sign 유지 |
| S6 | `S6 Fig.pdf` | R2.2 NRC validation (`R2_2_auc_per_color.*`) | median AUC 0.62, top-3 chromatic 0.604 |
| S7 | `S7 Fig.pdf` | R2.4 permutation null (`R2_4_permutation_null.*`) | Black z=−10.0, p<0.0001 등 |
| S8 | `S8 Fig.pdf` | R2.6 crop robustness (`R2_6_crop_robustness.*`) | 30단어, mean cos 0.955 @50% crop |
| S9 | `S9 Fig.pdf` | R2.3 polysemy demo (`R2_3_polysemy.*`) | 4브랜드+nail/chip bare↔disambiguated |
| **S10** | `S10 Fig.pdf` | **R2.6b page 1 vs page 10** (`R2_6b_page_compare.*`) | **28/30 page-10 도달, mean cos 0.947** |

## 분석 CSV·요약·로그 (`csv/revision_results/`)

| 분석 | 파일들 |
|---|---|
| R1.1 weighting sensitivity | `R1_1_weighting_sensitivity.csv`, `_summary.md`, `_figure.{png,pdf}` |
| R2.2 NRC validation | `R2_2_auc_per_color.{png,pdf}`, `R2_2_confusion_matrix.png` |
| R2.3 polysemy demo | `R2_3_polysemy.{png,pdf}` |
| R2.4 permutation null | `R2_4_permutation_null.{png,pdf}`, `_summary.md` |
| R2.6 crop robustness | `R2_6_crop_robustness.{png,pdf}`, `_summary.md` |
| R2.6 v2 (확장 sample) | `R2_6_crop_robustness_v2.{png,pdf}`, `R2_6_crop_robustness_v2_summary.md` |
| **R2.6b page 1 vs 10** | `R2_6b_page_compare.{csv,png,pdf}`, `R2_6b_page_compare_summary.md`, `R2_6b_page_metadata.csv`, `R2_6b_scrape.log` |

## 스크린샷 (`Images/`)

| 분석 | 폴더 | 내용 |
|---|---|---|
| R2.6 v1 | `Images/r26_raw/` | 20단어 screenshots |
| R2.6 v2 | `Images/r26_raw_v2/` | 30단어 screenshots (broad filter) |
| R2.6b | `Images/r26b_pages/{word}/page_{1..N}.png` | 30단어 × 페이지별 viewport screenshot |

## 분석 스크립트 (`/Users/qgroup/Desktop/word_color_association-main/`)

| 분석 | 스크립트 |
|---|---|
| R1.1 | `R1_1_weighting_sensitivity.py` (bootstrap) + `R1_1_report.py` (post-process·figure) |
| R2.2 | `R2_2_mohammad_validation.py` |
| R2.3 | `R2_3_scrape.py` → `R2_3_polysemy_demo.py` |
| R2.4 | `R2_4_permutation_null.py` |
| R2.5 (date filter feasibility) | `R2_5_date_filter_test.py` |
| R2.6 v1/v2 | `R2_6_sample_words[_v2].py` → `R2_6_scrape[_v2].py` → `R2_6_crop_robustness[_v2].py` |
| **R2.6b** | `R2_6b_page_compare_scrape.py` → `R2_6b_page_compare_analyze.py` |

## 추적 문서

- `PLOS_Sungpil_Revision.xlsx` (작업 마스터, 시트 1-5)
- `PLOS_Sungpil_Revision_BACKUP_2026-06-05_1556.xlsx` (교수님 미팅 후속 갱신 직전 백업)

---

## 외부 검증 (교수님 미팅 후속, 2026-06-05)

| 비교 대상 | 분석 | 결과 파일 | 핵심 결과 |
|---|---|---|---|
| **Rathore 2019** (12 fruits, 54 human raters × 58 UW colors) | per-concept 10-d cosine + Pearson/Spearman | `R_rathore_compare.{csv,pdf,png,_summary.md}` | mean cos 0.842, chrom 0.776, achrom 0.909, Pearson 0.618 |
| **Guilbeault/comp-syn** (39,949 단어 8-d JzAzBz + RGB) | RSA (5,220 overlap, 2k sampled) | `R_compsyn_compare.{csv,pdf,png,_summary.md}` + `_seeds.csv` | RSA Pearson r = 0.451, z = 43σ above null |

### 외부 데이터 위치

- `external_compare/rathore2019/` — Rathore GitHub clone (이미지+CSV+노트북, 2617 파일)
- `external_compare/compsyn/compsyn_vectors.json` — comp-syn 사전계산 embedding (29 MB)
- `Images/rathore_raw/` — 12 fruit Google Image screenshots

### v2 갱신 (Rathore 18 concepts + comp-syn 결합 vocab)

| 비교 대상 | 분석 | 결과 파일 | 핵심 결과 |
|---|---|---|---|
| **Rathore 2019 (18 concepts)** | 12 fruits + 6 materials, UW-58 + BCP-37 매핑 | `R_rathore_compare_v2.{csv,pdf,png,_summary.md}` | mean cos 0.821 (fruits 0.842, materials 0.781) |
| **comp-syn v2** (9,244 단어 vocab) | RSA 5,449 overlap, 2k sampled, 50회 null | `R_compsyn_compare_v2.{csv,pdf,png,_summary.md}` + `_seeds.csv` | Pearson r = 0.470, z = 44σ above null |

v1 파일들은 reference로 유지.
