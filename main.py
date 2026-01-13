"""
HiAgent Cloud Server 主入口

启动云端LLM评估服务器
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

if __name__ == "__main__":
    # 导入并运行
    from api.start_cloud_server import main
    main()