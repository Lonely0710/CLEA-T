#!/usr/bin/env python3
"""
云端服务器启动脚本

"""

import os
import sys
from pathlib import Path

# 获取项目根目录
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 确保可以导入模块
if __name__ == "__main__":
    # 运行服务器
    import subprocess

    # 使用 -m 模式运行，避免相对导入问题
    cmd = [sys.executable, "-m", "api.start_cloud_server"]

    # 切换到项目目录
    os.chdir(project_root)

    # 运行命令
    subprocess.run(cmd)