import sys, time
sys.path.insert(0, '/home/patch/predict')
from src.features import build as features
from src.models import hierarchical as H
from src import evaluate

TUNE = ['2015/16', '2016/17', '2017/18', '2018/19', '2019/20', '2020/21']
VALID = ['2021/22', '2022/23', '2023/24', '2024/25', '2025/26', '2026/27']

matches = features.load()
rows = []
for w in (0.0, 0.25, 0.5, 0.75, 1.0):
    t0 = time.time()
    s, _ = evaluate.walk_forward(matches, TUNE, H, design_kwargs={'xg_weight': w})
    rows.append((w, s))
    print(f"xg_weight={w:.2f} -> log_loss={s['log_loss']:.4f} rps={s['rps']:.4f} "
          f"acc={s['accuracy']:.1%} ({time.time()-t0:.0f}s)", flush=True)

best = min(rows, key=lambda r: r[1]['log_loss'])
print(f"\nBEST xg_weight={best[0]:.2f} (tune log_loss={best[1]['log_loss']:.4f})", flush=True)
print('held-out 2021/22 ->:', flush=True)
for label, w in [('goals only  w=0.00', 0.0), (f'tuned       w={best[0]:.2f}', best[0])]:
    v, _ = evaluate.walk_forward(matches, VALID, H, design_kwargs={'xg_weight': w})
    print(f"  {label}  log_loss={v['log_loss']:.4f} rps={v['rps']:.4f} "
          f"acc={v['accuracy']:.1%} n={v['n']}", flush=True)
