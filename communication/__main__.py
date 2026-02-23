"""python -m communication 入口：运行内部调试测试。"""

import sys
import os

# 直接运行本文件时，脚本所在目录在 sys.path[0]，需把项目根加入 path 才能 import communication
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 导入主程序
from communication.run_tests import main

if __name__ == "__main__":
    main()
