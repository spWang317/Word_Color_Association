# External Validation — 세 비교의 통합 정리

세 외부 자원(NRC · Rathore 2019 · comp-syn / Guilbeault 2020)과 우리 framework의 비교를 *같은 7-항목 구조* 로 정리.

각 비교마다:
1. **그쪽 방법** — 데이터 출처·생성 방식
2. **우리 방법** — 같은 단어에 대한 우리 framework 출력
3. **차이** — 입력·출력·차원·기준
4. **정렬 (Alignment)** — 두 표현을 비교 가능하게 만든 처리
5. **미정렬 (Residual)** — 통일 불가능한 부분과 처리 방식
6. **지표 (Metrics)** — 무엇을 측정하나
7. **결과·해석** — 숫자가 의미하는 것

---

## (A) NRC Word–Colour Association Lexicon (Mohammad 2011 EMNLP / 2013 LREC)

### 1. 그쪽 방법
- **Crowdsourcing on Amazon Mechanical Turk** — 단어 → 색 *introspective* 선택
- 각 단어마다 11개 *basic color* (red, orange, yellow, green, blue, purple, pink, brown, black, grey, white) 중 *하나*를 forced choice
- 다수결 → 단어당 *단일 categorical label*
- **Output**: word → 1-of-11 color label (8,889 단어)

### 2. 우리 방법
- 단어 → Google Image search → 결과 페이지 screenshot → 픽셀 색 추출 (CIELch)
- **Output**: word → 10-d 색 비율 벡터 (7 chromatic by hue + 3 achromatic by lightness, two-block normalization)

### 3. 차이
| | NRC | Ours |
|---|---|---|
| 입력 | 인간 *introspection* (생각으로 떠올리는 색) | *시각 web 콘텐츠* (실제 이미지 픽셀) |
| 출력 | 단일 categorical label (one-hot) | 연속 10-d 분포 |
| 색 기준 | 11 basic colors (brown 포함) | 10 base colors (brown 미포함) |
| 데이터 단위 | 단어당 1 label | 단어당 10 ratio |

### 4. 정렬
- 우리 vocab에서 *NRC 라벨이 있는* 500개 단어 무작위 sample
- NRC 11색 중 *우리 10색에 매칭되는 10개*만 사용 (brown 제외)
- *각 색 c마다 binary classification problem 구성*: NRC-label == c 인 단어 (positive) vs NRC-label != c 인 단어 (negative)
- 우리 framework 의 *c 점수* (10-d 중 c 차원 값) 가 positive 를 negative 보다 위로 ranking 하는지 평가

### 5. 미정렬 잔여
- **Brown (NRC 11번째)** — 우리 10색에 없음. 두 가지 처리:
  - `chromatic_unified` 분석: brown 제외, 7 chromatic 색만으로 평가
  - Top-1 hit rate에서는 brown 라벨 단어는 별도 처리
- **단일 라벨 vs 분포 비대칭** — NRC 는 강제 1색 선택, 우리는 분포. 직접 1-d 비교 대신 *ranking 평가* (AUC) 로 우회

### 6. 지표
- **one-vs-rest AUC** (per color, 10개) — *threshold-free, base-rate-invariant* (Bradley 1997, Hand & Till 2001)
  - 정의: P(우리_점수(NRC-c 단어) > 우리_점수(NRC-not-c 단어))
- **Top-1 hit rate** — NRC 라벨이 우리 framework 상위 1색인가? (chance = 1/k, k는 가능 라벨 수)
- **Top-3 hit rate** — 상위 3색 안에 NRC 라벨 있나? (chance = 3/k)

### 7. 결과·해석
- **Median AUC (10색) = 0.62** — 우연(0.5) 대비 +24%, "10 binary problem 평균적으로 우리 점수가 NRC-라벨된 단어를 그렇지 않은 단어보다 상위에 ranking"
- **Chromatic top-1 = 0.343** (chance 0.143, 2.4× chance) — 7 chromatic 색 중 우리 top-1이 NRC top-1과 일치할 확률
- **Chromatic top-3 = 0.604** (chance 0.429, 1.4× chance) — 우리 top-3 색 안에 NRC label 있을 확률
- 100% 가 아닌 이유: NRC 는 *연상* (블루 → "ocean") 이지만 우리는 *이미지의 실제 색* (ocean 사진은 blue + grey 하늘 + white 거품). 두 신호가 정렬되지만 동일하지 않음 — *예상되는 차이*

---

## (B) Rathore et al. 2019 (IEEE TVCG, Schloss Lab)

### 1. 그쪽 방법
- **온라인 설문 (54명 참가자)**
- 각 (concept, color) 쌍마다 *연관 강도* 를 슬라이더 (0-1)로 평가 — graded rating
- 두 sub-experiment:
  - 12 fruit concepts × **UW-58 colors** (CIE Lab 기반 58개 고정 팔레트)
  - 6 material/recycling concepts × **BCP-37 colors** (Berkeley Color Project, 37개)
- **Output**: 각 concept → graded 58-d (또는 37-d) human rating distribution

### 2. 우리 방법
- 동일 concepts 에 대해 우리 framework 출력 = 10-d 색 비율
- 단어가 *우리 코퍼스 안에 있으면* 매뉴스크립트 파이프라인 *기존 per-word 값* 사용 — *재현성·일관성 보장*
- 코퍼스에 없는 concept (6 fruits + 3 materials) 만 새로 Google Image screenshot 처리

### 3. 차이
| | Rathore | Ours |
|---|---|---|
| 입력 | 인간 *introspective* slider rating | *시각 web 콘텐츠* |
| 출력 | graded 분포 (58 or 37-d) | 10-d 분포 |
| 색 기준 | UW-58 / BCP-37 (CIE Lab 좌표 고정 팔레트) | 10 base colors (CIELch bin) |
| 단위 | concept (구체적 명사) | 임의 단어 |

### 4. 정렬

(a) **색 기준 통일** (UW-58·BCP-37 → 10 base):
- UW58_Colors.csv, BCP37_Colors.csv 가 *이미 (L, c, h)* 제공
- 우리 framework 의 *동일한 분류 규칙*을 각 팔레트 색에 적용:
  ```
  if c ≥ 15.93 (우리 chroma threshold):
      → chromatic. h 가 들어가는 hue 구간 1개 (7색 중)
  else:
      → achromatic. L 이 들어가는 lightness 구간 1개 (3색 중)
  ```
- 각 UW/BCP 색은 우리 10색 중 *정확히 1개*에 categorical 할당
- UW-58 매핑 분포: red 3, orange 7, yellow 9, green 9, blue 6, purple 11, pink 8, black 2, grey 2, white 1

(b) **Human rating → 10-d 집계**:
- 각 concept × 우리 base color c 마다: c 에 매핑된 UW/BCP 색들의 인간 평가 *합산*
- *Block-wise 정규화*: chromatic 7색 sum = 1, achromatic 3색 sum = 1 — 우리 framework 의 two-block 출력 구조와 정합

(c) **우리 측 벡터**:
- 코퍼스 *기존 값* 우선 (lemon · lime · mango · orange · raspberry · strawberry · glass · metal · paper — 9개)
- 코퍼스 미존재 시 새 Google Image screenshot (avocado · blueberry · cantaloupe · grapefruit · honeydew · watermelon · compost · plastic · trash — 9개, 필요 시 disambiguating modifier: 'orange fruit', 'glass material' 등)

### 5. 미정렬 잔여
- **Granularity 손실** — Rathore 58색 → 우리 10색 collapsing 시 *bin 내부 변동 정보 손실*. 단, 우리 framework 가 10색 단위로 작동하므로 *해당 단위에서의 비교는 정합*
- **Top-1 의 비대칭성** — 인간 rating 분포와 우리 분포의 *block-wise 정규화* 가 다른 색을 top-1으로 만들 수 있음 (예: 인간 grey 0.50 ↔ 우리 grey 0.40 + yellow 0.30, top-1 다름)
- **Achromatic 비대칭** — 인간은 "강한 색 연관 없음"을 black/grey 슬라이더 중간값으로 표현하는 경향; 사진은 시각적으로 색이 풍부. Block 정규화로 *방향* 일치는 보존되나 *강도* 는 차이남

### 6. 지표
- **10-d cosine** — 두 분포 벡터의 angle. *방향* 정합 (scale 무관)
- **Chromatic 7-d cosine** — chromatic block 만 (각각 sum 1 정규화 후)
- **Achromatic 3-d cosine** — achromatic block 만
- **Pearson r (10-d)** — 두 분포의 *선형 상관*
- **Spearman ρ (10-d)** — 두 분포의 *rank 상관* (선형성 가정 없음)
- **Top-1 match** — 양 측 top-1 색 일치 여부

### 7. 결과·해석
- **18 concept 평균 cos_10d = 0.853** (fruits 0.852, materials 0.855) — 거의 같은 방향
- **Chromatic 7-d 0.777** — "어떤 *유색*"에 대해 강한 일치
- **Achromatic 3-d 0.920** — "light vs dark vs grey" 거의 완벽 일치
- **Pearson r 0.656** — 분포 강도까지 선형 정합
- **Spearman ρ 0.664** — rank 정합도 강함
- **Top-1 6/18** — fine-grained top color 일치는 1/3 정도. *예상* — 두 분포의 *방향*이 정합하지만 *대표 1색*은 정규화 방식 차이로 달라질 수 있음
- 18 concept 중 *최저 cos_10d = 0.754 (Avocado)* — 전부 *방향 일치*. 가장 약한 경우도 r=0.51, ρ=0.61
- **의미**: 우리 framework 가 *독립 인간 평가의 색 분포*를 강하게 회복함. NRC (categorical) + Rathore (graded distribution) 두 종류 검증 통과

---

## (C) comp-syn (Srinivasa Desikan et al. 2020 COLING / Guilbeault et al. 2020 Cognition)

### 1. 그쪽 방법
- 각 단어 → Google Image search → **top ~100 이미지 다운로드**
- 픽셀들을 **JzAzBz** *perceptually uniform 색공간* (Safdar et al. 2017) 으로 변환
- JzAzBz 3D 공간을 **2×2×2 = 8 cube octant** 로 분할 → 각 cube 픽셀 카운트 → 8-d histogram
- 이미지 100장 평균 → 단어당 8-d 벡터
- 39,949 단어 사전계산 후 공개
- **Output**: word → 8-d cube-octant histogram (JzAzBz + RGB 두 버전)

### 2. 우리 방법
- 우리 framework + 코퍼스 처리 → 9,244 English content words 의 per-word 10-d 색 벡터 사전 (manuscript 파이프라인의 기존 값)
- **Output**: word → 10-d (7 chromatic + 3 achromatic) 비율 벡터

### 3. 차이
| | comp-syn | Ours |
|---|---|---|
| 색 공간 | **JzAzBz** (perceptually uniform, HDR 호환, modern) | **CIELch** (LCH 변환) |
| 분할 방식 | 3D 공간을 2×2×2 *cube octant* 8개 — *unnamed 영역* | hue (7) + lightness (3) *named 색* |
| 차원 | 8 (각 octant 픽셀 카운트) | 10 (named bin 비율) |
| 이미지 sampling | 100개 *개별 이미지 다운로드* 평균 | 1개 *검색 결과 페이지 screenshot* (≥30 이미지 thumbnail 모자이크) |
| 해석성 | 낮음 (8 octant 이름 없음) | 높음 (각 차원 = named color) |
| 가산성 | 제한적 | 단어 색 벡터 합산 가능 (코퍼스 수준 확장) |

### 4. 정렬
- **차원·기준 통일 불가** — 8 cube octant 와 10 named bin 은 *공간 분할 방식이 근본적으로 다름*. 개별 차원 매칭 자의적
- **해결: Representational Similarity Analysis (RSA)**
  - *개별 차원이 아니라 단어들 사이 *거리 구조*를 비교*
  - 절차:
    1. Overlap 단어 5,449개 중 무작위 2,000개 추출 (seed 0)
    2. 우리 공간: N×N pairwise cosine 매트릭스 C_ours
    3. comp-syn 공간: 같은 N×N pairwise cosine 매트릭스 C_cs
    4. 두 매트릭스 upper triangle 추출 → 1,999,000 개 pair 값 두 벡터
    5. Pearson r, Spearman ρ
- *dimension-agnostic · scale-invariant* — 두 representation 의 *글로벌 구조 alignment* 측정
- 신경과학·cognitive science 표준 (Kriegeskorte et al. 2008)

### 5. 미정렬 잔여
- **개별 차원 매칭 영구 불가** — 8개 unnamed octant 와 10개 named bin 은 *meaning 자체가 다름*
- **Top-N nearest neighbor 불일치 (정상)**:
  - 'blue' top-8: 우리 ↔ comp-syn Jaccard 0/8
  - 'sea' top-8: 1/8
  - r=0.47 은 *글로벌* 일치 의미, *국소* 정확성 아님
  - 다른 partitioning 이 fine-grained ranking 을 달리 만들지만, *전체 거리 패턴*은 보존
- **comp-syn 벡터는 sum=1 아님** (raw histogram counts). Cosine 은 scale 무관이라 문제 없음

### 6. 지표
- **RSA Pearson r** — 두 거리매트릭스 upper triangle 의 선형 상관
- **RSA Spearman ρ** — rank 상관 (단조 관계만 가정)
- **Permutation null** — 한쪽 매트릭스 행 셔플 후 RSA 재계산, n=50 회. *우연으로 이 정도 r 이 나올 빈도* 의 분포
- **z-score above null** — 관측 r 이 null 평균에서 SD 단위로 얼마나 떨어졌나

### 7. 결과·해석
- **RSA Pearson r (JzAzBz) = 0.470, Spearman ρ = 0.424**
- **RSA Pearson r (RGB) = 0.453**
- **Permutation null** (50 shuffles): mean −0.001, SD 0.011, max 0.024 → 관측값 0.470 → **z ≈ 44σ above null**
- **r 0.47 의 의미**:
  - 두 *완전 독립* 색 추출 방법 (다른 algorithm · 다른 색공간 · 다른 partitioning · 다른 차원) 이 *동일한 단어-단어 색 유사 구조*에 수렴
  - RSA benchmark 비교:
    - 같은 family LLM 두 개: r 0.7-0.9
    - LLM vs 인간 의미 판단: r 0.4-0.6
    - 뇌 fMRI RSA: r 0.1-0.3 (의미 있음)
    - vision vs language encoder: r 0.2-0.4
    - **두 독립 이미지-색 추출 알고리즘 (우리 case): 0.3-0.5 가 의미 있는 수준 → 0.47 = "moderate-to-strong"**
- **z = 44σ** → 우연 확률 천문학적으로 작음 (p << 10⁻¹⁰⁰). 실험과학 표준에서 *확실한 효과*
- **Top-N 불일치도 함께 보고**: 글로벌 정합 + 국소 차이 (다른 표현)이라는 *균형 잡힌* 그림

---

## 비교 한눈에 (응답서용 한 단락 후보)

> Our framework's color predictions are externally validated against three independent resources spanning categorical lexicon labels, graded human concept-color ratings, and an independent image-based color embedding. Against the NRC Word--Colour Association Lexicon (Mohammad 2013, 500 word sample), per-color one-vs-rest AUC has median 0.62 across 10 base colors, with chromatic top-3 accuracy 0.604 (chance 0.429) [bradley1997auc, handTill2001auc]. Against Rathore et al. (2019)'s graded slider-based ratings of 18 fruit and material concepts on the UW-58 / BCP-37 palettes (mapped to our 10 base colors via the same chromatic-threshold + hue/lightness rule, with per-block normalization to mirror our framework's two-block output), mean cosine similarity is 0.853 (chromatic block 0.777, achromatic block 0.920) and Pearson r = 0.656; for nine concepts that appear in our corpus vocabulary we use the manuscript pipeline's existing per-word vectors to ensure full consistency. Against comp-syn (Srinivasa Desikan et al. 2020; Guilbeault et al. 2020) — an independent project that encodes each of 39,949 English words as an 8-bin JzAzBz color-cube histogram — Representational Similarity Analysis on a random 2,000-word subset of the 5,449-word vocabulary overlap yields a pairwise-cosine Pearson r = 0.470 (Spearman ρ = 0.424), or ≈ 44 standard deviations above the row-shuffled null. The two embeddings differ in geometry (10 categorical CIELch bins vs 8 JzAzBz octants), so individual nearest neighbors differ, but the global similarity structure is strongly aligned.
