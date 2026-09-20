import sys
sys.path.insert(0, '/home/patch/predict')
import numpy as np
from src.features import build as features
from src.models import dynamic as D, hierarchical as H
from src import evaluate

VALID = ['2021/22', '2022/23', '2023/24', '2024/25', '2025/26', '2026/27']
matches = features.load()
rng = np.random.default_rng(0)


def probs_for(model, design_kwargs, fit_kwargs):
    P, A = [], []
    for season in VALID:
        r = evaluate.fold(matches, season, model, fit_kwargs, design_kwargs)
        if r:
            P.append(r[0]); A.append(r[1])
    return np.vstack(P), np.concatenate(A)


variants = {
    'static goals': (H, {'xg_weight': 0.0}, {}),
    'static +xG': (H, {'xg_weight': 0.5}, {}),
    'dynamic +xG': (D, {'xg_weight': 0.5, 'decay': False},
                    {'innovation': 0.10, 'persistence': 0.99}),
}
losses, actual = {}, None
for name, (model, dk, fk) in variants.items():
    p, a = probs_for(model, dk, fk)
    actual = a
    losses[name] = -np.log(np.clip(p[np.arange(len(a)), a], 1e-12, 1))
    print(f"{name:14} log_loss={losses[name].mean():.4f}", flush=True)

print(f"\npaired bootstrap, n={len(actual)}, 10k resamples")
names = list(losses)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        diff = losses[names[j]] - losses[names[i]]
        idx = rng.integers(0, len(diff), size=(10000, len(diff)))
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        verdict = "significant" if lo > 0 or hi < 0 else "NOT significant"
        print(f"  {names[j]:14} - {names[i]:14} = {diff.mean():+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  {verdict}")
