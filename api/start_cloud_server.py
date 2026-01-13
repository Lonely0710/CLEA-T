"""
云端服务器启动脚本
"""

import os
import sys
from pathlib import Path

# 需要添加到sys.path的是项目根目录的上级目录，以便导入cloud模块
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
from cloud.api.main import initialize_server


def main():
    """主函数"""
    # 加载.env文件 - 修正路径到项目根目录
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"已加载环境配置: {env_path}")
    else:
        print("警告: 未找到.env文件，请确保已配置必要的环境变量")

    # 检查必要的环境变量
    required_env_vars = ["OPENAI_API_KEY"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print(f"错误: 缺少必要的环境变量: {', '.join(missing_vars)}")
        print("请编辑cloud/.env文件并设置以下变量:")
        for var in missing_vars:
            print(f"  {var}=your_value")
        sys.exit(1)

    # 显示配置信息
    print("云端服务器配置:")
    print(f"  监听地址: {os.getenv('CLOUD_HOST', '0.0.0.0')}:{os.getenv('CLOUD_PORT', '8000')}")
    print(f"  使用模型: {os.getenv('CLOUD_MODEL', 'gpt-4')}")
    print(f"  API地址: {os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')}")
    print(f"  日志级别: {os.getenv('LOG_LEVEL', 'INFO')}")
    print(f"  日志目录: logs/ (请定期清理)")

    try:
        # 初始化并启动服务器
        print("\n正在启动云端服务器...")
        server = initialize_server()
        server.run()

    except KeyboardInterrupt:
        print("\n服务器已停止")
    except Exception as e:
        print(f"\n服务器错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()