# SVDD Anomaly Detection — Multi-center Feat-augmented Deep SVDD

실시간 공정 이상 감지 모델.  
동일 Recipe에서 생성되는 **단일 프로파일 (300 타점)** 을 입력받아 정상 / 이상을 판정한다.  
정상 프로파일이 **2가지 타입(A형, B형)** 으로 존재하며, 입력 시 타입을 명시할 필요 없다.

---

## 모델 개요

```
입력: 프로파일 1개 (300,)
        ↓
   1D-CNN Encoder  →  latent vector (32,)
        +
   구간별 통계 features (30,)  ← flat/transition/spike 구간의 mean, std, diff_std
        ↓ concat → Linear → latent z (32,)
        ↓
   K=2 중심까지 정규화 거리 계산
   score = min( ||z-c_0||²/R_0²,  ||z-c_1||²/R_1² )
        ↓
   score ≥ 1.0  →  이상   (threshold 고정)
   score <  1.0  →  정상
```

**핵심 특징:**
- **K=2 중심**: A형/B형 각각의 정상 클러스터를 별도 초구(hypersphere)로 학습
- **k-means 초기화**: 학습 시 자동으로 두 타입을 분리
- **threshold = 1.0 고정**: 정규화 거리이므로 별도 조정 불필요
- **Decoder 불필요**: inference = encoder forward pass + 거리 계산만

**학습 2단계:**
1. Autoencoder pretrain — encoder가 의미있는 표현 먼저 학습 (collapse 방지)
2. Multi-center SVDD fine-tune — K=2 중심 주변으로 정상 데이터 압축

---

## 프로젝트 구조

```
svdd_project/
├── README.md                  ← 이 파일
├── ipi                        ← 패키지 설치 래퍼 (pip + environment.yml 자동 업데이트)
├── environment.yml            ← 의존 패키지 목록
├── src/
│   ├── inference.py           ← ★ 메인 진입점 (서버 테스트용)
│   ├── model_v2.py            ← MultiCenterSVDD / FeatSVDD 모델 정의
│   ├── train_v2.py            ← 학습 스크립트 (Multi-center SVDD)
│   ├── evaluate_mc.py         ← Multi-center vs Single 비교 평가
│   ├── anomaly_generator.py   ← 가짜 이상 데이터 생성기 (테스트용)
│   ├── evaluate_v2.py         ← 단일 중심 FeatSVDD 평가
│   ├── model.py               ← 기본 Encoder/Decoder (train_v2.py 참조)
│   └── benchmark.py           ← inference 속도/메모리 측정
└── .gitignore
```

**학습 후 생성되는 파일 (git 제외, 서버에서 직접 생성):**
```
svdd_project/
├── multicenter_model.pt       ← ★ 학습된 모델 가중치 (메인)
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

### 입력

| 항목 | 형식 | 설명 |
|---|---|---|
| 프로파일 | `np.ndarray (N, 300)` float32 | 시계열 1개 또는 배치 |
| 타점 수 | **300** (고정) | 원본 타점 수가 다르면 선형 보간하여 300으로 맞출 것 |
| 값 범위 | 원본 스케일 그대로 입력 | 내부에서 min-max 정규화 수행 |
| 타입 구분 | **불필요** | A형/B형 자동 판별 |

> ⚠️ **주의**: 학습 시 사용한 `norm_params.npy`의 min/max로 정규화됨.  
> 실제 데이터 범위가 학습 데이터와 크게 다르면 재학습 필요.

**단일 프로파일:**
```python
profile = np.array([...])   # shape (300,)  or  (1, 300)
```

**배치:**
```python
batch = np.loadtxt('profiles.csv', delimiter=',')   # shape (N, 300)
```

**CSV 파일 형식:**
```
# profiles.csv — N행 × 300열, 콤마 구분
5.08,5.07,4.98,...   ← 1번째 프로파일 (300개 값)
3.00,2.95,2.90,...   ← 2번째 프로파일
...
```

### 출력

```python
{
  'scores'           : np.ndarray (N,),  # 이상 점수 (≥1.0 이면 이상)
  'labels'           : np.ndarray (N,),  # 0=정상, 1=이상
  'threshold'        : float,            # 1.0 (고정)
  'anomaly_ratio'    : float,            # 배치 내 이상 비율
  'is_batch_anomaly' : bool,             # True이면 배치 경보 (기준: 비율 > 10%)
}
```

---

## 학습

```bash
cd src

# Multi-center SVDD 학습 (메인)
python train_v2.py
```

학습 데이터 경로 수정 (`train_v2.py` `load_data()` 함수):
```python
A = np.loadtxt('/path/to/profile_A.csv', delimiter=',')   # A형 정상 프로파일
B = np.loadtxt('/path/to/profile_B.csv', delimiter=',')   # B형 정상 프로파일
```

> 학습 데이터는 **정상 프로파일만** 사용. 이상 데이터 불필요.

학습 완료 후 자동 생성:
- `multicenter_model.pt` — 메인 모델
- `norm_params.npy` — 정규화 파라미터
- `feat_norm.npy` — feature 정규화 파라미터

---

## 추론 (Inference)

### Python API

```python
import numpy as np
from src.inference import SVDDInference

# 초기화 (모델 로드, 1회만 수행)
engine = SVDDInference(
    model_path     = 'multicenter_model.pt',
    norm_path      = 'norm_params.npy',
    feat_norm_path = 'feat_norm.npy',
    device         = 'cpu',              # 'cuda' 가능
)

# 단일 판정
profile = np.array([...])              # (300,)
result  = engine.predict(profile, profile)   # A, B 동일 입력 가능 (단일 타입)
print(result['labels'])                # [0]=정상, [1]=이상

# 배치 판정 (500개)
batch  = np.loadtxt('profiles.csv', delimiter=',')   # (500, 300)
result = engine.predict(batch, batch)
print(f"이상 비율: {result['anomaly_ratio']:.1%}")
print(f"배치 경보: {result['is_batch_anomaly']}")
```

### CLI

```bash
python src/inference.py \
    --a data/profiles.csv \
    --b data/profiles.csv \
    --model multicenter_model.pt \
    --norm  norm_params.npy \
    --fnorm feat_norm.npy
```

---

## 하이퍼파라미터

### train_v2.py

| 파라미터 | 기본값 | 설명 | 올리면 | 내리면 |
|---|---|---|---|---|
| `LATENT_DIM` | 32 | latent 공간 차원 | 표현력↑, 속도↓ | 빠름, 미세이상 탐지↑ |
| `BATCH_SIZE` | 64 | mini-batch 크기 | 안정적 학습 | 빠른 학습 |
| `LR` | 1e-3 | pretrain 학습률 | 빠른 수렴 (불안정 위험) | 안정적 (느림) |
| `pretrain epochs` | 30 | Autoencoder 학습 횟수 | 더 좋은 초기화 | 빠른 시작 |
| `finetune epochs` | 50 | SVDD fine-tune 횟수 | 구 더 촘촘 | 느슨한 경계 |

### model_v2.py — MultiCenterSVDD

| 파라미터 | 기본값 | 설명 | 올리면 | 내리면 |
|---|---|---|---|---|
| `K` | 2 | 정상 클러스터 수 | 더 많은 타입 지원 | 단순 (K=1이면 기존 SVDD) |
| `nu` | 0.05 | 허용 outlier 비율 | 구 커짐 → 관대 (FN↑) | 구 작아짐 → 엄격 (FP↑) |

### 배치 경보 기준 조정

```python
result   = engine.predict(batch, batch)
my_alert = result['anomaly_ratio'] > 0.15   # 기본 10% → 15%로 변경
```

---

## 구간 정의 (300 타점 기준)

feature 추출 시 사용하는 구간 (`model_v2.py` `ZONES`):

| 구간 | 인덱스 | 설명 |
|---|---|---|
| flat | 35 ~ 160 | 안정 구간 |
| transition | 160 ~ 200 | 상승 전환 |
| zero | 200 ~ 215 | 0 근처 구간 |
| spike | 215 ~ 245 | 피크 구간 |
| tail | 245 ~ 300 | 마지막 안정 |

실제 데이터의 구간이 다르면 `model_v2.py`의 `ZONES` dict 수정 후 재학습 필요.

---

## 이상 탐지 성능 (synthetic 데이터 기준)

| Anomaly Type | 탐지율 | 설명 |
|---|---|---|
| excess_noise | ✅ 100% | flat 구간 노이즈 과다 |
| spike_low/high | ✅ 100% | spike 크기 이상 |
| spike_missing | ✅ 100% | spike 없음 |
| mean_shift | ✅ 100% | 전체 수준 이동 |
| local_burst | ✅ 100% | 변곡점 noise 폭발 |
| shape_distortion | ✅ 97% | 전환 구간 기울기 왜곡 |

**ROC-AUC: 0.94+**

> ⚠️ 위 수치는 synthetic 데이터 기준. 실제 설비 데이터로 검증 필수.

---

## 추론 속도 / 리소스 (CPU 기준)

| 항목 | 값 |
|---|---|
| 1개 프로파일 판정 | **0.29 ms** |
| 500개 배치 판정 | **~15 ms** |
| 피크 메모리 | **2.0 KB** |
| 모델 크기 | **0.10 MB** |
| 파라미터 수 | 25,634 |
| threshold | **1.0 (고정)** |

---

## 실제 데이터 적용 시 체크리스트

- [ ] 학습 데이터를 실제 정상 프로파일로 교체 후 재학습
- [ ] `ZONES` 구간이 실제 데이터 패턴과 맞는지 확인
- [ ] k-means 초기화 후 Center 0/1이 A형/B형으로 잘 분리됐는지 확인 (학습 로그 참조)
- [ ] `nu` 값 조정 (FP rate 모니터링 후 결정)
- [ ] 설비 노후화 drift 발생 시 `norm_params.npy` 및 모델 주기적 재학습 고려
- [ ] 배치 경보 기준 (`anomaly_ratio > 0.10`) 현장 상황에 맞게 조정
