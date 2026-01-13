#!/usr/bin/env python3
"""
日志清理工具

清理过期的日志文件
"""

import os
import time
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 日志目录
LOG_DIR = Path(__file__).parent.parent / "logs"


def cleanup_old_logs(days_to_keep: int = 30):
    """
    清理指定天数之前的日志文件

    Args:
        days_to_keep: 保留天数，默认30天
    """
    if not LOG_DIR.exists():
        print(f"Log directory does not exist: {LOG_DIR}")
        return

    # 计算截止时间
    cutoff_time = time.time() - (days_to_keep * 24 * 60 * 60)
    deleted_files = []
    total_size = 0

    # 遍历日志文件
    for log_file in LOG_DIR.glob("*.log*"):
        try:
            # 获取文件修改时间
            file_mtime = log_file.stat().st_mtime

            # 如果文件过期，删除它
            if file_mtime < cutoff_time:
                file_size = log_file.stat().st_size
                log_file.unlink()
                deleted_files.append(str(log_file))
                total_size += file_size
        except Exception as e:
            print(f"Error deleting {log_file}: {e}")

    # 输出结果
    if deleted_files:
        print(f"\nDeleted {len(deleted_files)} log files:")
        for file in deleted_files:
            print(f"  - {file}")
        print(f"\nTotal space freed: {total_size / (1024*1024):.2f} MB")
    else:
        print("\nNo log files to delete.")


def list_log_files():
    """
    列出所有日志文件及其大小
    """
    if not LOG_DIR.exists():
        print(f"Log directory does not exist: {LOG_DIR}")
        return

    print(f"\nLog files in {LOG_DIR}:")
    print("-" * 80)
    total_size = 0

    for log_file in sorted(LOG_DIR.glob("*.log*")):
        try:
            file_size = log_file.stat().st_size
            file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
            size_mb = file_size / (1024 * 1024)
            total_size += file_size
            print(f"{log_file.name:<30} {file_mtime.strftime('%Y-%m-%d %H:%M:%S'):<20} {size_mb:>8.2f} MB")
        except Exception as e:
            print(f"Error reading {log_file}: {e}")

    print("-" * 80)
    print(f"{'Total':<30} {'':<20} {total_size / (1024*1024):>8.2f} MB")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Log cleanup utility")
    parser.add_argument("--days", type=int, default=30, help="Days to keep logs (default: 30)")
    parser.add_argument("--list", action="store_true", help="List all log files")
    parser.add_argument("--cleanup", action="store_true", help="Clean up old log files")

    args = parser.parse_args()

    if args.list:
        list_log_files()
    elif args.cleanup:
        print(f"Cleaning up log files older than {args.days} days...")
        cleanup_old_logs(args.days)
    else:
        parser.print_help()