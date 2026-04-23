"""
一键跑单元测试 + 生成 HTML 报告。

VSCode 里打开本文件，点右上角 ▶ Run Python File 即可。

行为：
- 跑 tests/ 下所有 pytest 用例（正确性 + 性能 + 可视化）
- 使用 -s 让性能测试的 [PERF] 耗时行与可视化的 [VIZ] 保存路径都打印出来
- HTML 报告落到 experiments/unit_tests/<YYYYMMDD>/pytest.html
- 可视化 PNG 落到 experiments/unit_tests/<YYYYMMDD>/viz/
- 退出码 0 / 1 让 VSCode 显示绿/红

按需改下面的 PATTERN 聚焦到某个模块，例如：
    PATTERN = "test_env_visualization.py"   # 只看轨迹图
    PATTERN = "test_env_performance.py"     # 只看性能数字
    PATTERN = "test_*.py"                   # （默认）全部
"""
from __future__ import annotations

import os
import sys
import datetime
from pathlib import Path

# ---- sys.path 注入 -------------------------------------------------------
_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---- 配置 ----------------------------------------------------------------
# 只跑 tests/ 下匹配该 glob 的文件
PATTERN = "test_*.py"

# 是否生成 HTML 报告；跑得快可关闭
WRITE_HTML = True

# 是否传 -s（让 print 立刻显示）。性能 / 可视化测试依赖这个
SHOW_PRINTS = True
# --------------------------------------------------------------------------

import pytest

today = datetime.date.today().strftime("%Y%m%d")
report_dir = _ROOT / "experiments" / "unit_tests" / today
report_dir.mkdir(parents=True, exist_ok=True)
html_path = report_dir / "pytest.html"
viz_dir = report_dir / "viz"

args = [f"tests/{PATTERN}" if PATTERN != "test_*.py" else "tests/",
        "-v", "--tb=short"]
if SHOW_PRINTS:
    args.append("-s")
if WRITE_HTML:
    args += [f"--html={html_path}", "--self-contained-html"]

print("=" * 60)
print(f"  pytest  args={args}")
print("=" * 60)

rc = pytest.main(args)

print()
print("=" * 60)
if WRITE_HTML and html_path.exists():
    print(f"  HTML report:")
    print(f"    {html_path}")
if viz_dir.exists() and any(viz_dir.iterdir()):
    print(f"  Visualizations (open in Explorer / VSCode image preview):")
    print(f"    {viz_dir}")
print(f"  exit code: {rc}  ({'OK' if rc == 0 else 'FAILED'})")
print("=" * 60)

sys.exit(int(rc))
