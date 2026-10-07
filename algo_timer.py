"""以 main.py 之姿執行,額外回報純演算法均速(所有方法通用)。

用法與 main.py 完全相同:
    ./py310/bin/python algo_timer.py --method baro --dataset re2-tt ...

不修改 main.py(論文 baseline 紅線)。做法:
1. 先把 pandas.read_csv 換成計時版,累計讀檔秒數(trace 檔等都經過它)。
2. 把 RCAEval.e2e 上的 method 函式包一層計時;main.py 稍後 import 時拿到的
   就是包過的版本。純演算法時間 = 函式 wall time - 呼叫期間的讀檔時間。
   (main.py 讀 data.csv 在函式之外,天然不計入。)
3. process 結束時印一行「Algo speed: <每 case 平均秒數>」,由 run123.py 解析。

ponytail: 只攔 pandas.read_csv;若某方法用 open()/np.loadtxt 讀大檔,那部分
不會被扣除,屆時再加對應的攔截。
"""
import atexit
import os
import runpy
import sys
from time import perf_counter

import pandas as pd

_io_seconds = 0.0
_real_read_csv = pd.read_csv


def _timed_read_csv(*args, **kwargs):
    global _io_seconds
    t0 = perf_counter()
    out = _real_read_csv(*args, **kwargs)
    _io_seconds += perf_counter() - t0
    return out


pd.read_csv = _timed_read_csv

# 必須在 patch read_csv 之後才 import,讓方法模組內的
# `from pandas import read_csv` 也綁到計時版。
import RCAEval.e2e as e2e  # noqa: E402

_algo_times = []


def _wrap(fn):
    def timed(*args, **kwargs):
        io_before = _io_seconds
        t0 = perf_counter()
        out = fn(*args, **kwargs)
        _algo_times.append(perf_counter() - t0 - (_io_seconds - io_before))
        return out

    return timed


@atexit.register
def _report():
    if _algo_times:
        print("Algo speed:", round(sum(_algo_times) / len(_algo_times), 2))


try:
    method = sys.argv[sys.argv.index("--method") + 1]
except (ValueError, IndexError):
    method = None
# ponytail: 只包得到 e2e 套件屬性的方法(DEFAULT_METHODS 全在內);
# main.py 從子模組直接 import 的(如 causalrca)不會有 Algo speed 行。
if method and hasattr(e2e, method):
    setattr(e2e, method, _wrap(getattr(e2e, method)))

runpy.run_path(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "main.py"),
    run_name="__main__",
)
