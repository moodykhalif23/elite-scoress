import sys
sys.path.insert(0, '/home/patch/predict')
import numpy as np
from src.features import build as features
from src.models import hierarchical as H
from src import evaluate

TUNE = ['2015/16', '2016/17', '2017/18', '2018/19', '2019/20', '2020/21']
VALID = ['2021/22', '2022/23', '2023/24', '2024/25', '2025/26', '2026/27']

matches = features.load()
rng = np.random.default_rng(0)

VARIANTS = {
    'baseline': {},
    '+derby': {'derby': True},
    '+pair 0.03': {'pair_sigma': 0.03},
    '+pair 0.08': {'pair_sigma': 0.08},
    '+pair 0.15': {'pair_sigma': 0.15},
    '+pair 0.08 +derby': {'pair_sigma': 0.08, 'derby': True},
}

print('=== tune window ===', flush=True)
tuned = {}
for name, kw in VARIANTS.items():
    s, _ = evaluate.walk_forward(matches, TUNE, H, fit_kwargs=kw)
    tuned[name] = s
    print(f"{name:20} log_loss={s['log_loss']:.4f} rps={s['rps']:.4f} acc={s['accuracy']:.1%}",
          flush=True)

print('\n=== held-out window ===', flush=True)
losses, derby_losses = {}, {}
for name, kw in VARIANTS.items():
    probs, actual, frames = [], [], []
    for season in VALID:
        r = evaluate.fold(matches, season, H, kw, {})
        if r:
            probs.append(r[0]); actual.append(r[1]); frames.append(r[2])
    p = np.vstack(probs); a = np.concatenate(actual)
    import pandas as pd
    from src.features.rivalry import tag_derbies
    test = pd.concat(frames, ignore_index=True)
    is_derby = tag_derbies(test).notna().to_numpy()
    losses[name] = -np.log(np.clip(p[np.arange(len(a)), a], 1e-12, 1))
    derby_losses[name] = losses[name][is_derby]
    print(f"{name:20} log_loss={losses[name].mean():.4f} | "
          f"derby-only={derby_losses[name].mean():.4f} (n={is_derby.sum()})", flush=True)

print('\n=== paired bootstrap vs baseline (10k) ===', flush=True)
for name in VARIANTS:
    if name == 'baseline':
        continue
    for label, pool in (('all', losses), ('derbies', derby_losses)):
        diff = pool[name] - pool['baseline']
        idx = rng.integers(0, len(diff), size=(10000, len(diff)))
        boot = diff[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        flag = 'significant' if lo > 0 or hi < 0 else 'not significant'
        print(f"  {name:20} [{label:7}] {diff.mean():+.4f} [{lo:+.4f}, {hi:+.4f}]  {flag}",
              flush=True)
