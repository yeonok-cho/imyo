"""
Feat-augmented SVDD  –  Inference Interface
============================================
서버에서 실제 프로파일을 받아 정상/이상을 판정하는 단일 진입점.

사용법 (Python API):
    from inference import SVDDInference
    model = SVDDInference()
    result = model.predict(profile_A, profile_B)

사용법 (CLI):
    python inference.py --a profile_A.csv --b profile_B.csv
"""

import os, sys, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from model_v2 import FeatSVDD, MultiCenterSVDD, extract_features

# 기본 경로 (프로젝트 루트 기준)
_ROOT      = os.path.dirname(os.path.dirname(__file__))
_MODEL_PT  = os.path.join(_ROOT, 'multicenter_model.pt')   # ← Multi-center SVDD
_NORM_NPY  = os.path.join(_ROOT, 'norm_params.npy')
_FNORM_NPY = os.path.join(_ROOT, 'feat_norm.npy')


class SVDDInference:
    """
    Parameters
    ----------
    model_path  : str  학습된 모델 경로 (.pt)
    norm_path   : str  정규화 파라미터 경로 (.npy)
    feat_norm_path : str  feature 정규화 파라미터 경로 (.npy)
    device      : str  'cpu' or 'cuda'
    """

    def __init__(self,
                 model_path=_MODEL_PT,
                 norm_path=_NORM_NPY,
                 feat_norm_path=_FNORM_NPY,
                 device='cpu'):

        self.device = torch.device(device)
        norm        = np.load(norm_path,      allow_pickle=True).item()
        fnorm       = np.load(feat_norm_path, allow_pickle=True).item()

        self.A_min, self.A_max = norm['A_min'], norm['A_max']
        self.B_min, self.B_max = norm['B_min'], norm['B_max']
        self.f_mean = fnorm['mean']
        self.f_std  = fnorm['std']

        ck = torch.load(model_path, map_location=self.device)
        if 'K' in ck:
            # Multi-center SVDD
            self.model = MultiCenterSVDD(ck['latent_dim'], ck['n_feats'], K=ck['K']).to(self.device)
            self.model.encoder.load_state_dict(ck['encoder'])
            for k in range(ck['K']):
                getattr(self.model, f'c_{k}').copy_(ck[f'c_{k}'])
                getattr(self.model, f'R_{k}').data = torch.tensor(ck[f'R_{k}'])
            self.threshold = 1.0          # 정규화 거리이므로 고정
        else:
            # 단일 중심 FeatSVDD (하위 호환)
            self.model = FeatSVDD(ck['latent_dim'], ck['n_feats']).to(self.device)
            self.model.encoder.load_state_dict(ck['encoder'])
            self.model.c.copy_(ck['c'])
            self.model.R.data = torch.tensor(ck['R'])
            self.threshold = ck['R'] ** 2
        self.model.eval()
        self.R2 = self.threshold

    # ── 전처리 ────────────────────────────────────────────────────────────────
    def _preprocess(self, A: np.ndarray, B: np.ndarray):
        """
        A, B: (N, 300) float32  or  (300,) for single profile
        returns: X tensor (N,2,300), F tensor (N, n_feats)
        """
        A = np.atleast_2d(A).astype(np.float32)
        B = np.atleast_2d(B).astype(np.float32)
        assert A.shape[1] == 300 and B.shape[1] == 300, \
            f"각 프로파일은 300 타점이어야 합니다. 현재: A={A.shape}, B={B.shape}"
        assert A.shape[0] == B.shape[0], "A, B 샘플 수가 같아야 합니다."

        A_n = (A - self.A_min) / (self.A_max - self.A_min)
        B_n = (B - self.B_min) / (self.B_max - self.B_min)
        X   = np.stack([A_n, B_n], axis=1)

        F   = extract_features(X)
        F_n = (F - self.f_mean) / self.f_std

        return torch.tensor(X).to(self.device), torch.tensor(F_n).to(self.device)

    # ── 단일/배치 판정 ─────────────────────────────────────────────────────────
    @torch.no_grad()
    def predict(self, profile_A: np.ndarray, profile_B: np.ndarray) -> dict:
        """
        개별 프로파일 판정.

        Parameters
        ----------
        profile_A : np.ndarray  shape (300,) 또는 (N, 300)
        profile_B : np.ndarray  shape (300,) 또는 (N, 300)

        Returns
        -------
        dict:
            scores   : np.ndarray (N,)  – 이상 점수 (높을수록 이상)
            labels   : np.ndarray (N,)  – 0=정상, 1=이상
            threshold: float            – 판정 기준 R²
            is_batch_anomaly: bool      – 배치 이상 여부 (이상 비율 > 10%)
        """
        X, F   = self._preprocess(profile_A, profile_B)
        scores = self.model.anomaly_score(X, F).cpu().numpy()
        labels = (scores > self.threshold).astype(int)

        return {
            'scores'           : scores,
            'labels'           : labels,
            'threshold'        : float(self.R2),
            'anomaly_ratio'    : float(labels.mean()),
            'is_batch_anomaly' : bool(labels.mean() > 0.10),
        }

    @torch.no_grad()
    def score_single(self, profile_A: np.ndarray, profile_B: np.ndarray) -> float:
        """단일 프로파일 쌍의 이상 점수만 반환 (가장 빠른 경로)"""
        X, F = self._preprocess(profile_A, profile_B)
        return float(self.model.anomaly_score(X, F).cpu().item())


# ── CLI 진입점 ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SVDD Anomaly Detection Inference')
    parser.add_argument('--a',        required=True, help='Profile A CSV (N×300 또는 300,)')
    parser.add_argument('--b',        required=True, help='Profile B CSV (N×300 또는 300,)')
    parser.add_argument('--model',    default=_MODEL_PT,  help='모델 경로 (.pt)')
    parser.add_argument('--norm',     default=_NORM_NPY,  help='정규화 파라미터 (.npy)')
    parser.add_argument('--fnorm',    default=_FNORM_NPY, help='Feature 정규화 파라미터 (.npy)')
    parser.add_argument('--device',   default='cpu',      help='cpu 또는 cuda')
    args = parser.parse_args()

    A = np.loadtxt(args.a, delimiter=',').astype(np.float32)
    B = np.loadtxt(args.b, delimiter=',').astype(np.float32)

    engine = SVDDInference(args.model, args.norm, args.fnorm, args.device)
    result = engine.predict(A, B)

    print(f"\n── 판정 결과 ───────────────────────")
    print(f"  샘플 수        : {len(result['scores'])}")
    print(f"  이상 비율      : {result['anomaly_ratio']:.1%}")
    print(f"  배치 경보      : {'🚨 이상' if result['is_batch_anomaly'] else '✅ 정상'}")
    print(f"  threshold (R²) : {result['threshold']:.4f}")
    print(f"  scores (처음 5): {result['scores'][:5].round(4)}")
    print(f"  labels (처음 5): {result['labels'][:5]}")
