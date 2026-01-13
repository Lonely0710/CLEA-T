"""
增强的边缘侧通信客户端，支持云边协同协议
"""

import os
import websocket
import json
import threading
import time
import logging
from typing import Optional, Callable, Dict, Any
from queue import Queue, Empty
from dotenv import load_dotenv

# Import protocol definitions
from .cloud_protocol import CloudMessage, MessageType, SummaryData


# Configure logging with colors
class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages"""
    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    
    COLORS = {
        'DEBUG': BLUE,
        'INFO': GREEN,
        'WARNING': YELLOW,
        'ERROR': RED,
        'CRITICAL': MAGENTA,
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Colorize specific keywords for better visibility
        msg = record.msg
        if "[Edge]" in str(msg):
            msg = str(msg).replace("[Edge]", f"{self.CYAN}[Edge]{color}")
        
        record.msg = msg
        message = super().format(record)
        return f"{color}{message}{self.RESET}"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Check if handler already exists to avoid duplicates
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False  # Avoid double logging with root logger


class EdgeCloudClient:
    """增强的云边通信客户端"""

    def __init__(self,
                 url: str = None,
                 auto_reconnect: bool = True,
                 reconnect_interval: float = 5.0,
                 timeout: float = 10.0):
        """
        初始化客户端

        Args:
            url: 云端WebSocket URL
            auto_reconnect: 是否自动重连
            reconnect_interval: 重连间隔（秒）
            timeout: 连接超时（秒）
        """
        # 加载环境变量
        env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
        if os.path.exists(env_path):
            load_dotenv(env_path)
        else:
            load_dotenv()

        self.url = url or os.getenv("CLOUD_WEBSOCKET_URL", "ws://localhost:8000/ws")
        self.auto_reconnect = auto_reconnect
        self.reconnect_interval = reconnect_interval
        self.timeout = timeout

        # WebSocket连接
        self.ws = None
        self.is_connected = False
        self.should_run = False

        # 消息处理
        self.message_queue = Queue()
        self.response_handlers = {}  # 请求ID -> 处理函数
        self.summary_handlers = []   # 总结回调函数列表

        # 线程管理
        self.receive_thread = None
        self.process_thread = None

        # 统计信息
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "connection_attempts": 0,
            "last_heartbeat": None
        }

    def connect(self) -> bool:
        """连接到云端服务器"""
        try:
            self.stats["connection_attempts"] += 1
            logger.info(f"[Edge] 尝试连接到云端: {self.url}")

            self.ws = websocket.WebSocket()
            self.ws.connect(self.url, timeout=self.timeout)
            self.is_connected = True
            self.should_run = True

            # 启动接收和处理线程
            self.receive_thread = threading.Thread(target=self._receive_loop)
            self.process_thread = threading.Thread(target=self._process_loop)
            self.receive_thread.daemon = True
            self.process_thread.daemon = True
            self.receive_thread.start()
            self.process_thread.start()

            logger.info(f"[Edge] 成功连接到云端")
            return True

        except Exception as e:
            logger.error(f"[Edge] 连接失败: {e}")
            self.is_connected = False
            return False

    def disconnect(self):
        """断开连接"""
        self.should_run = False
        self.is_connected = False

        if self.ws:
            try:
                self.ws.close()
            except:
                pass

        # 等待线程结束
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1)
        if self.process_thread and self.process_thread.is_alive():
            self.process_thread.join(timeout=1)

        logger.info("[Edge] 已断开连接")

    def send_message(self, message: CloudMessage,
                    response_handler: Optional[Callable] = None) -> bool:
        """
        发送消息

        Args:
            message: 要发送的消息
            response_handler: 可选的响应处理函数

        Returns:
            是否发送成功
        """
        if not self.is_connected or not self.ws:
            logger.warning("[Edge] 未连接，无法发送消息")
            return False

        try:
            # 注册响应处理器
            if response_handler:
                self.response_handlers[message.request_id] = response_handler

            # 发送消息
            json_str = message.to_json()
            self.ws.send(json_str)
            self.stats["messages_sent"] += 1

            logger.debug(f"[Edge] 发送消息: {message.message_type.value}")
            return True

        except Exception as e:
            logger.error(f"[Edge] 发送消息失败: {e}")
            self.is_connected = False
            return False

    def send_trajectory(self, trajectory_data,
                       summary_callback: Optional[Callable] = None) -> bool:
        """
        发送轨迹数据

        Args:
            trajectory_data: 轨迹数据
            summary_callback: 总结结果回调函数

        Returns:
            是否发送成功
        """
        # 注册总结回调
        if summary_callback:
            self.summary_handlers.append(summary_callback)

        # 发送轨迹上传消息
        from .cloud_protocol import CloudMessage
        upload_msg = CloudMessage.create_trajectory_upload(trajectory_data)
        if not self.send_message(upload_msg):
            return False

        # 发送总结请求
        request_msg = CloudMessage.create_summary_request(
            trajectory_data.trajectory_id,
            request_summary=True,
            request_evaluation=True
        )
        return self.send_message(request_msg)

    def _receive_loop(self):
        """接收消息循环"""
        while self.should_run and self.is_connected:
            try:
                # 接收消息
                message = self.ws.recv()
                self.message_queue.put(message)
                self.stats["messages_received"] += 1

            except websocket.WebSocketTimeoutException:
                # 超时，继续
                continue
            except websocket.WebSocketConnectionClosedException:
                logger.warning("[Edge] WebSocket连接已关闭")
                break
            except Exception as e:
                logger.error(f"[Edge] 接收消息错误: {e}")
                break

        self.is_connected = False

        # 自动重连
        if self.auto_reconnect and self.should_run:
            self._reconnect()

    def _process_loop(self):
        """处理消息循环"""
        while self.should_run:
            try:
                # 从队列获取消息
                message = self.message_queue.get(timeout=1.0)
                self._handle_message(message)

            except Empty:
                # 队列为空，继续
                continue
            except Exception as e:
                logger.error(f"[Edge] 处理消息错误: {e}")

    def _handle_message(self, message_str: str):
        """处理接收到的消息"""
        try:
            # 尝试解析为JSON
            import json
            data = json.loads(message_str)

            # 尝试解析为CloudMessage
            try:
                message = CloudMessage.from_json(message_str)
                # 标准消息处理
                if message.message_type == MessageType.HEARTBEAT:
                    self._handle_heartbeat(message)
                elif message.message_type == MessageType.SUMMARY_RESPONSE:
                    self._handle_summary_response(message)
                elif message.message_type == MessageType.ACK:
                    self._handle_ack(message)
                elif message.message_type == MessageType.ERROR:
                    self._handle_error(message)
                else:
                    logger.warning(f"[Edge] 未知消息类型: {message.message_type.value}")

                # 调用注册的响应处理器
                if message.request_id in self.response_handlers:
                    handler = self.response_handlers.pop(message.request_id)
                    try:
                        handler(message)
                    except Exception as e:
                        logger.error(f"[Edge] 响应处理器错误: {e}")

            except Exception as e:
                # 非标准CloudMessage，尝试处理简单响应
                logger.debug(f"[Edge] 接收到非标准消息，尝试简单处理: {e}")

                # 检查是否是简单的状态响应
                if isinstance(data, dict):
                    if "status" in data:
                        logger.info(f"[Edge] 收到状态响应: {data}")
                    elif "type" in data:
                        # 可能是其他格式的消息
                        msg_type = data.get("type", "UNKNOWN")
                        logger.info(f"[Edge] 收到消息类型: {msg_type}")
                else:
                    logger.info(f"[Edge] 收到原始消息: {message_str}")

        except json.JSONDecodeError:
            # 不是JSON格式的消息
            logger.debug(f"[Edge] 收到非JSON消息: {message_str}")
        except Exception as e:
            logger.error(f"[Edge] 消息处理错误: {e}")

    def _handle_heartbeat(self, message: CloudMessage):
        """处理心跳消息"""
        self.stats["last_heartbeat"] = time.time()
        logger.debug("[Edge] 收到心跳")

    def _handle_summary_response(self, message: CloudMessage):
        """处理总结响应"""
        try:
            summary_dict = message.payload["summary"]
            summary_data = SummaryData.from_dict(summary_dict)

            logger.info(f"[Edge] 收到总结: {summary_data.trajectory_id}")

            # 调用所有注册的总结回调
            for handler in self.summary_handlers[:]:
                try:
                    handler(summary_data)
                except Exception as e:
                    logger.error(f"[Edge] 总结回调错误: {e}")
                    # 移除有问题的回调
                    if handler in self.summary_handlers:
                        self.summary_handlers.remove(handler)

        except Exception as e:
            logger.error(f"[Edge] 处理总结响应错误: {e}")

    def _handle_ack(self, message: CloudMessage):
        """处理确认消息"""
        logger.debug(f"[Edge] 收到确认: {message.payload}")

    def _handle_error(self, message: CloudMessage):
        """处理错误消息"""
        error_msg = message.payload.get("error", "未知错误")
        logger.error(f"[Edge] 收到错误: {error_msg}")

    def _reconnect(self):
        """自动重连"""
        logger.info(f"[Edge] 将在 {self.reconnect_interval} 秒后重连...")
        time.sleep(self.reconnect_interval)

        if self.should_run:
            self.connect()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        stats = self.stats.copy()
        stats["is_connected"] = self.is_connected
        stats["queue_size"] = self.message_queue.qsize()
        stats["pending_handlers"] = len(self.response_handlers)
        return stats

    def clear_summary_handlers(self):
        """清除所有总结回调函数"""
        self.summary_handlers.clear()

    def send_output(self, content):
        """发送输出到cloud并接收memory更新"""
        from .cloud_protocol import TrajectoryData, TrajectoryItem

        # 创建轨迹数据 - 注意 TrajectoryData 的 items 期望二维列表
        trajectory_data = TrajectoryData(
            trajectory_id=f"edge_output_{int(time.time())}",
            items=[[TrajectoryItem(
                type="action_output",
                content=content,
                timestamp=time.time()
            )]],  # 二维列表格式
            metadata={
                "node_id": getattr(self, 'node_id', 'edge_node'),
                "output_type": "action"
            }
        )

        # 发送轨迹数据到cloud
        self.send_trajectory(trajectory_data)


# 兼容性包装器，保持与原有CloudConnector接口的兼容
class CloudConnector:
    """兼容性包装器"""

    def __init__(self):
        self.client = EdgeCloudClient()
        self.client.connect()

    def send_output(self, content):
        """兼容性方法"""
        # 创建简单的轨迹数据
        from .cloud_protocol import TrajectoryData, TrajectoryItem
        trajectory_data = TrajectoryData(
            trajectory_id=f"legacy_{int(time.time())}",
            items=[[TrajectoryItem(
                type="legacy_output",
                content=content,
                timestamp=time.time()
            )]],  # 二维列表格式
            metadata={"legacy_content": content}
        )

        # 发送
        self.client.send_trajectory(trajectory_data)

    def send_message(self, message: CloudMessage):
        """新方法：发送消息"""
        return self.client.send_message(message)

    def send_trajectory(self, trajectory_data, callback=None):
        """新方法：发送轨迹数据"""
        return self.client.send_trajectory(trajectory_data, callback)

    def close(self):
        """关闭连接"""
        self.client.disconnect()

    def __del__(self):
        """析构函数"""
        self.close()