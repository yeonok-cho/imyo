# SVDD Anomaly Detection — Feat-augmented Deep SVDD

실시간 공정 이상 감지 모델.  
동일 Recipe에서 생성되는 **Profile A / Profile B** (각 300 타점) 쌍을 입력받아  
정상 / 이상을 판정한다.

---

## 모델 개요

```
입력: Profile A (300,) + Profile B (300,)
        ↓ 두 채널을 동시에 학습하는 Joint Encoder
   1D-CNN Encoder  →  latent vector (32,)
        +
   구간별 통계 features (30,)   ← flat/transition/spike 구간의 mean, std, diff_std
        ↓ concat → Linear → latent (32,)
   SVDD 거리:  ||z - c||²  vs  R²
        ↓
   score > R²  →  이상 판정
```

**학습 2단계:**
1. Autoencoder pretrain (encoder 초기화, collapse 방지)
2. SVDD fine-tune (정상 데이터를 초구 안에 압축, R 고정)

**추론 시 Decoder 불필요** → encoder forward pass + 거리 계산만 수행

---

## 프로젝트 구조

```
svdd_project/
├── README.md                  ← 이 파일
├── ipi                        ← 패키지 설치 래퍼 (pip + environment.yml 자동 업데이트)
├── environment.yml            ← 의존 패키지 목록
├── src/
│   ├── inference.py           ← ★ 메인 진입점 (서버 테스트용)
│   ├── model_v2.py            ← Feat-augmented SVDD 모델 정의
│   ├── train_v2.py            ← 학습 스크립트 (Approach 1 & 2)
│   ├── anomaly_generator.py   ← 가짜 이상 데이터 생성기
│   ├── evaluate_v2.py         ← 평가 + 벤치마크
│   ├── model.py               ← 기본 Encoder/Decoder 정의 (train_v2.py가 참조)
│   └── benchmark.py           ← inference 속도/메모리 측정
└── .gitignore
```

**학습 후 생성되는 파일 (git 제외, 서버에서 직접 생성):**
```
svdd_project/
├── svdd_model_feat.pt         ← 학습된 모델 가중치
├── norm_params.npy            ← 정규화 파라미터 (A_min, A_max, B_min, B_max)
└── feat_norm.npy              ← feature 정규화 파라미터 (mean, std)
```

---

## 환경 설치

```bash
# 가상환경 생성
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 패키지 설치
pip install torch numpy scipy matplotlib scikit-learn
```

또는 ipi 스크립트 사용:
```bash
chmod +x ipi
./ipi torch numpy scipy matplotlib scikit-learn
```

---

## 데이터 형식

### 입력 (학습 / 추론 공통)

| 항목 | 형식 | 설명 |
|---|---|---|
| Profile A | `np.ndarray (N, 300)` float32 | Recipe의 첫 번째 시계열 |
| Profile B | `np.ndarray (N, 300)` float32 | Recipe의 두 번째 시계열 |
| 타점 수 | **300** (고정) | 원본 242 타점을 선형 보간한 값 |
| 값 범위 | 원본 스케일 그대로 입력 | 내부에서 min-max 정규화 수행 |

**단일 프로파일도 가능:**
```python
profile_A = np.array([...])   # shape (300,)  or  (1, 300)
profile_B = np.array([...])   # shape (300,)  or  (1, 300)
```

**CSV 파일 형식:**
```
# profile_A.csv  — N행 × 300열, 콤마 구분
5.08,5.07,4.98,...   ← 1번째 프로파일
5.10,5.05,4.95,...   ← 2번째 프로파일
...
```

### 출력

```python
{
  'scores'           : np.ndarray (N,),   # 이상 점수 (높을수록 이상)
  'labels'           : np.ndarray (N,),   # 0=정상, 1=이상
  'threshold'        : float,             # 판정 기준 R²
  'anomaly_ratio'    : float,             # 배치 내 이상 비율
  'is_batch_anomaly' : bool,             # True이면 배치 경보 발령
}
```

---

## 학습

```bash
cd src

# Approach 2 (Feat-augmented SVDD) 학습
python train_v2.py
```

학습 데이터 경로 (train_v2.py 내 수정):
```python
# train_v2.py  load_data() 함수
A = np.loadtxt('/path/to/profile_A.csv', delimiter=',')
B = np.loadtxt('/path/to/profile_B.csv', delimiter=',')
```

학습 완료 후 자동 생성:
- `svdd_model_feat.pt`
- `norm_params.npy`
- `feat_norm.npy`

---

## 추론 (Inference)

### Python API

```python
import numpy as np
from src.inference import SVDDInference

# 초기화 (모델 로드, 1회만)
engine = SVDDInference(
    model_path  = 'svdd_model_feat.pt',
    norm_path   = 'norm_params.npy',
    feat_norm_path = 'feat_norm.npy',
    device      = 'cpu',              # 'cuda' 가능
)

# 단일 판정
profile_A = np.array([...])   # (300,)
profile_B = np.array([...])   # (300,)
result = engine.predict(profile_A, profile_B)

print(result['labels'])           # [0] = 정상, [1] = 이상
print(result['is_batch_anomaly']) # False

# 배치 판정 (500개)
A_batch = np.loadtxt('profile_A.csv', delimiter=',')   # (500, 300)
B_batch = np.loadtxt('profile_B.csv', delimiter=',')   # (500, 300)
result  = engine.predict(A_batch, B_batch)

print(f"이상 비율: {result['anomaly_ratio']:.1%}")
print(f"배치 경보: {result['is_batch_anomaly']}")
```

### CLI

```bash
# 단일 또는 배치 CSV 파일로 직접 실행
python src/inference.py \
    --a data/profile_A.csv \
    --b data/profile_B.csv \
    --model svdd_model_feat.pt \
    --norm  norm_params.npy \
    --fnorm feat_norm.npy
```

---

## 하이퍼파라미터

### train_v2.py

| 파라미터 | 기본값 | 설명 | 영향 |
|---|---|---|---|
| `LATENT_DIM` | 32 | latent 공간 차원 | 낮추면 미세 이상 탐지↑, 속도↑ |
| `BATCH_SIZE` | 64 | 학습 mini-batch 크기 | 크면 안정, 작으면 빠름 |
| `LR` | 1e-3 | pretrain 학습률 | 너무 크면 불안정 |
| `pretrain epochs` | 30 | Autoencoder 학습 횟수 | 늘리면 더 좋은 초기화 |
| `finetune epochs` | 50 | SVDD fine-tune 횟수 | 많을수록 구 촘촘 |

### model_v2.py — FeatSVDD / HybridSVDD

| 파라미터 | 기본값 | 설명 | 올리면 | 내리면 |
|---|---|---|---|---|
| `nu` | 0.05 | 정상 데이터 중 허용 outlier 비율 | R 커짐 → 관대 (FN↑) | R 작아짐 → 엄격 (FP↑) |
| `alpha` (Hybrid only) | 0.6 | SVDD 거리 vs 재현 오차 가중치 | SVDD 거리 비중↑ | 재현 오차 비중↑ |

### inference.py — SVDDInference.predict()

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `is_batch_anomaly` 기준 | 10% | 배치 내 이상 비율이 이 값 초과 시 경보 |

이 기준 변경 방법:
```python
result = engine.predict(A, B)
# 직접 anomaly_ratio로 커스텀 판정
my_alert = result['anomaly_ratio'] > 0.15   # 15%로 변경
```

---

## 구간 정의 (300 타점 기준)

feature 추출 시 사용하는 구간 (model_v2.py `ZONES`):

| 구간 | 인덱스 | 설명 |
|---|---|---|
| flat | 35 ~ 160 | 안정 구간 |
| transition | 160 ~ 200 | 상승 전환 |
| zero | 200 ~ 215 | 0 근처 구간 |
| spike | 215 ~ 245 | 피크 구간 |
| tail | 245 ~ 300 | 마지막 안정 |

실제 데이터의 구간이 다르다면 `model_v2.py`의 `ZONES` dict를 수정 후 재학습 필요.

---

## 이상 탐지 성능 (synthetic 데이터 기준)

| Anomaly Type | 탐지율 | 설명 |
|---|---|---|
| excess_noise | ✅ 100% | flat 구간 노이즈 과다 |
| spike_low/high | ✅ 100% | spike 크기 이상 |
| spike_missing | ✅ 100% | spike 없음 |
| mean_shift | ✅ 100% | 전체 수준 이동 |
| local_burst | ✅ 100% | 변곡점 noise 폭발 |
| shape_distortion | ✅ 100% | 전환 구간 기울기 왜곡 |

**ROC-AUC: 0.94+** (30% anomaly ratio 혼합 배치 기준)

> ⚠️ 위 수치는 synthetic 데이터 기준. 실제 설비 데이터로 검증 필수.

---

## 추론 속도 / 리소스 (CPU 기준)

| 항목 | 값 |
|---|---|
| 1개 프로파일 판정 | **0.21 ms** |
| 500개 배치 판정 | **~10 ms** |
| 피크 메모리 | **1.9 KB** |
| 모델 크기 | **0.10 MB** |
| 파라미터 수 | 25,633 |

---

## 실제 데이터 적용 시 체크리스트

- [ ] 학습 데이터를 실제 정상 프로파일로 교체 후 재학습
- [ ] `ZONES` 구간이 실제 데이터 패턴과 맞는지 확인
- [ ] `nu` 값 조정 (FP rate 모니터링 후 결정)
- [ ] 설비 노후화 drift 발생 시 `norm_params.npy` 및 모델 주기적 재학습 고려
- [ ] 배치 경보 기준 (`anomaly_ratio > 0.10`) 현장 상황에 맞게 조정
