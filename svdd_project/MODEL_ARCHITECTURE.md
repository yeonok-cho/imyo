# Deep SVDD 이상 탐지 모델 — 기본 구조 기술문서

이 문서는 데이터셋(profile_A/B, 합성 RESI/TEMP 등)과 무관하게, 지금까지
`svdd_project`에서 사용해 온 **모델 자체의 기본 구조**만 정리한다. 실제
raw 데이터로 새로 개발을 시도할 때(`train_raw.py`) 이 구조를 그대로
재사용하되, 데이터에 종속적인 가정(zone 위치, 정규화 범위 등)만 데이터에서
다시 뽑아 쓴다.

---

## 1. 문제 정의

- **One-class 이상 탐지**: 학습 시 정상 데이터만 사용, 이상 데이터는 불필요.
- **입력**: 동일 시점에 함께 측정되는 신호 2개를 페어로 사용. 각 신호는
  길이 300의 시계열 1개 (`(2, 300)` shape). 신호가 1개뿐이면 동일 신호를
  두 채널에 복제해서 넣어도 된다.
- **출력**: 이상 점수(scalar) + 임계값 기반 정상/이상 라벨.

## 2. 전체 파이프라인

```
raw profile (2, 300)
      │  min-max 정규화 (채널별 min/max)
      ▼
 ┌─────────────┐        ┌───────────────────┐
 │  1D-CNN      │──────▶│  zone 기반 통계     │
 │  Encoder     │        │  feature (mean/    │
 │ (2,300)→(32,)│        │  std/diff_std)     │
 └─────────────┘        └───────────────────┘
      │                          │
      └──────────┬───────────────┘
                  ▼
            concat → Linear
                  ▼
            latent z (32,)
                  ▼
     ┌─────────────────────────┐
     │  중심 c (또는 c_0..c_K)   │
     │  거리 기반 이상 점수      │
     └─────────────────────────┘
```

두 갈래(1D-CNN 인코더 / zone 통계)를 합쳐서 latent를 만드는 것이 핵심이며,
이를 "Feature-augmented SVDD"라 부른다. 인코더만 써도(Approach 없는 순수
Deep SVDD, `model.py`의 `DeepSVDD`) 동작은 하지만, 국소적/구간적 이상에는
zone 통계가 있는 편이 훨씬 민감하다 (RESI/TEMP 실험에서 확인).

## 3. 모듈별 구조

### 3.1 Encoder (`model.py`, `model_v2.py` 공통)

```
Conv1d(2→16, k=7, s=2) + BN + LeakyReLU   # 300 → 150
Conv1d(16→32, k=5, s=2) + BN + LeakyReLU  #  150 → 75
Conv1d(32→64, k=3, s=2) + BN + LeakyReLU  #   75 → 38
Conv1d(64→64, k=3, s=2) + BN + LeakyReLU  #   38 → 19
AdaptiveAvgPool1d(1) → Flatten            #   19 → 1
Linear(64 → latent_dim, bias=False)
```

stride-2 conv를 4번 거치므로 시간 해상도가 16배 줄어든다. **국소적으로
매우 좁은(수 포인트) 이상은 이 단계에서 평균에 묻혀 사라질 수 있음** —
zone 통계 feature가 이를 보완하는 이유.

### 3.2 Decoder (pretrain 전용)

Encoder의 대칭 구조 (ConvTranspose1d 4단 + AdaptiveAvgPool1d(300)).
Autoencoder pretrain에만 쓰이고, 최종 inference에는 사용하지 않는다
(Feature-augmented / Multi-center 방식 기준).

### 3.3 Zone 기반 통계 feature

```
채널마다 zone을 나누고, zone마다 [mean, std, diff_std] 3개 추출
n_feats = 3 × (channel별 zone 개수 합)
```

- `diff_std` = 구간 내 1차 차분의 표준편차 = "roughness"(국소 떨림) 지표.
  실제로 hold/유지 구간에서 발생하는 미세 진동성 이상은 이 feature가 가장
  민감하게 반응한다 (RESI/TEMP 실험 결과).
- **zone은 반드시 데이터 자체의 구조(ramp/hold/peak 위치)에서 유도해야 한다.**
  다른 파형(다른 recipe, 다른 신호)에서 가져온 고정 인덱스를 재사용하면,
  "flat"이라 부르는 구간이 실제로는 가장 크게 변하는 구간이 되는 등
  이름과 실제 위치가 어긋나 feature가 무의미해질 수 있다 (실제로 한 번
  겪은 실패 사례 — §5 참조).

### 3.4 FeatEncoder

```python
z_cnn  = Encoder(x)                    # (32,)
z_feat = Linear(n_feats→16) + LeakyReLU # (16,)
z      = Linear(32+16 → 32, bias=False)(concat(z_cnn, z_feat))
```

### 3.5 SVDD 헤드 — 두 가지 변형

**(a) FeatSVDD — 단일 중심 (정상 타입이 1가지일 때)**
```
score(x) = ||z - c||²
threshold = R²  (nu 분위수로 초기화)
```

**(b) MultiCenterSVDD — K개 중심 (정상 타입이 여러 종류일 때, 예: K=2)**
```
score(x) = min_k( ||z - c_k||² / R_k² )
threshold = 1.0 (정규화 거리이므로 고정)
```
중심은 k-means로 초기화하고, 각 클러스터의 (1-nu) 분위수로 반경 R_k를 정한다.

## 4. 학습 절차 (2단계)

1. **Autoencoder pretrain** (`Encoder` + `Decoder`, MSE reconstruction loss)
   — collapse(모든 입력이 동일 latent로 매핑되는 현상) 방지용 초기화.
2. **SVDD fine-tune** (pretrain된 encoder 가중치로 시작)
   - loss: `R² + (1/(nu·N)) · Σ max(0, dist² - R²)` (nu = 허용 outlier 비율)
   - Multi-center는 클러스터별로 위 loss를 평균.
   - 중심 c는 buffer(고정), 반경 R만 학습 파라미터로 두는 것이 일반적
     (원 논문 방식). 이 프로젝트에서는 R도 학습되도록 `nn.Parameter`로 둠.

## 5. 알아둬야 할 실패 사례 (RESI/TEMP 합성 데이터 실험)

- 원래 `ZONES`(`flat/transition/zero/spike/tail`)는 profile_A/B 템플릿
  전용으로 설계된 고정 인덱스였다.
- RESI/TEMP 실제 구조에 이 인덱스를 그대로 적용하니, "flat"이라 이름
  붙은 구간이 실제로는 가장 크게 변화하는 구간이었고, "spike" 구간은
  오히려 완만한 구간이었다 (`zones_vs_shape.png` 참조).
- 결과적으로 zone 통계가 이상 신호보다 profile 간 자연 변동을 더 크게
  반영해서(`|z-shift| < 1`), 이상 탐지 민감도가 크게 떨어졌다.
- 요약 통계(mean/std shape)만으로 raw 데이터를 재구성하는 합성 데이터
  생성 방식 자체도, 개별 profile의 실제 hold 구간 타이밍 편차(time-warp)를
  평균 내는 과정에서 실제 구조를 뭉갠다는 한계가 있었다 (peak 직후
  hold 구간이 population 평균에서는 완만한 경사처럼 보이는 문제).

**결론**: 이 모델 구조(Encoder + zone feature + SVDD) 자체는 유지하되,
zone은 **실제 raw 데이터에서 직접 계산**해야 한다. `train_raw.py`는 이
원칙을 반영해서 zone을 하드코딩하지 않고 데이터 기반으로 자동 검출한다.

## 6. Inference 인터페이스 (참고)

```python
engine = SVDDInference(model_path, norm_path, feat_norm_path, device='cpu')
result = engine.predict(channel_A, channel_B)
# result: scores, labels, threshold, anomaly_ratio, is_batch_anomaly
```
