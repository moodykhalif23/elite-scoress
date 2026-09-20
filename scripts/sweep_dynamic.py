import sys, time, itertools
sys.path.insert(0, '/home/patch/predict')
from src.features import build as features
from src.models import dynamic as D
from src import evaluate

TUNE = ['2015/16', '2016/17', '2017/18', '2018/19', '2019/20', '2020/21']
VALID = ['2021/22', '2022/23', '2023/24', '2024/25', '2025/26', '2026/27']
DESIGN = {'decay': False, 'xg_weight': 0.5}

matches = features.load()
rows = []
for innovation, persistence in itertools.product((0.06, 0.10, 0.15), (0.97, 0.99, 1.00)):
    t0 = time.time()
    s, _ = evaluate.walk_forward(matches, TUNE, D, design_kwargs=DESIGN,
                                 fit_kwargs={'innovation': innovation,
                                             'persistence': persistence})
    rows.append(((innovation, persistence), s))
    print(f"innov={innovation:.2f} phi={persistence:.2f} -> log_loss={s['log_loss']:.4f} "
          f"rps={s['rps']:.4f} acc={s['accuracy']:.1%} ({time.time()-t0:.0f}s)", flush=True)

(innov, phi), best = min(rows, key=lambda r: r[1]['log_loss'])
print(f"\nBEST innovation={innov} persistence={phi} tune_log_loss={best['log_loss']:.4f}",
      flush=True)
print('held-out 2021/22 onward:', flush=True)
v, _ = evaluate.walk_forward(matches, VALID, D, design_kwargs=DESIGN,
                             fit_kwargs={'innovation': innov, 'persistence': phi})
print(f"  dynamic  log_loss={v['log_loss']:.4f} rps={v['rps']:.4f} "
      f"acc={v['accuracy']:.1%} n={v['n']}", flush=True)
