"""P2 测试套件公共配置：把仓库根加入 sys.path，使 cryptoquant_auto 可导入。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
