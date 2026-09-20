import sys
sys.path.insert(0, '/home/patch/predict')
import numpy as np
from src.features import build as features
from src.models import hierarchical as H
from src import evaluate

VALID = ['2021/22', '2022/23', '2023/24', '2024/25', '2025/26', '2026/27']
DOMESTIC = ('E0', 'SP1', 'I1', 'D1')
matches = features.load()
rng = np.random.default_rng(0)


def run(with_europe: bool):
    pool = matches if with_europe else matches[matches['league'].isin(DOMESTIC)]
    probs, actual = [], []
    for season in VALID:
        train = pool[pool['season'] < season]
        test = matches[(matches['season'] == season)
                       & (matches['league'].isin(DOMESTIC))]
        if train.empty or test.empty:
            continue
        design = H.build_design(train, as_of=train['date'].max(),
                                cross_league=with_europe)
        params = H.fit_map(design)
        idx = design.index_fixtures(test)
        usable = idx['usable']
        test = test[usable]
        idx = {k: (v[usable] if hasattr(v, '__len__') else v) for k, v in idx.items()}
        lam, mu, rho = H.rates(params, design, idx)
        probs.append(evaluate.outcome_probabilities(lam, mu, rho))
        actual.append(test['result'].map(evaluate.OUTCOME_INDEX).to_numpy())
    return np.vstack(probs), np.concatenate(actual)


losses = {}
for label, flag in (('domestic only', False), ('with UCL/UEL', True)):
    p, a = run(flag)
    losses[label] = -np.log(np.clip(p[np.arange(len(a)), a], 1e-12, 1))
    s = evaluate.score(p, a)
    print(f"{label:15} log_loss={s['log_loss']:.4f} rps={s['rps']:.4f} "
          f"acc={s['accuracy']:.1%} n={s['n']}", flush=True)

diff = losses['with UCL/UEL'] - losses['domestic only']
idx = rng.integers(0, len(diff), size=(10000, len(diff)))
boot = diff[idx].mean(axis=1)
lo, hi = np.percentile(boot, [2.5, 97.5])
verdict = 'significant' if lo > 0 or hi < 0 else 'NOT significant'
print(f"\ndifference {diff.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {verdict}", flush=True)
