"""
云边通信协议定义



定义边缘节点与云端服务之间的通信协议和数据结构
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from pydantic import BaseModel
import json
import time
import uuid
from datetime import datetime


class MessageType(Enum):
    """消息类型枚举"""
    # 评估相关
    TRAJECTORY_EVALUATION = "TRAJECTORY_EVALUATION"
    EVALUATION_RESULT = "EVALUATION_RESULT"
    SUMMARY_RESPONSE = "SUMMARY_RESPONSE"

    # 兼容旧版本
    AGENT_OUTPUT = "AGENT_OUTPUT"

    # 控制消息
    HEARTBEAT = "HEARTBEAT"
    ACK = "ACK"
    ERROR = "ERROR"


class TrajectoryType(str, Enum):
    """轨迹类型"""
    ACTION = "Action"
    OBSERVATION = "Observation"
    STATE = "State"
    SUBGOAL = "Subgoal"


@dataclass
class TrajectoryItem:
    """轨迹项"""
    type: str
    content: str
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TrajectoryItem':
        return cls(
            type=data["type"],
            content=data["content"],
            timestamp=data.get("timestamp", time.time())
        )


class Trajectory(BaseModel):
    """轨迹数据"""
    trajectory_id: str
    items: List[List[TrajectoryItem]]  # 二维列表，每个子列表代表一个时间步
    subgoals: List[TrajectoryItem] = []
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "items": [[item.to_dict() for item in step] for step in self.items],
            "subgoals": [item.to_dict() for item in self.subgoals],
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Trajectory':
        return cls(
            trajectory_id=data["trajectory_id"],
            items=[
                [TrajectoryItem.from_dict(item) for item in step]
                for step in data["items"]
            ],
            subgoals=[
                TrajectoryItem.from_dict(item) for item in data.get("subgoals", [])
            ],
            metadata=data.get("metadata", {})
        )


class EvaluationRequest(BaseModel):
    """评估请求模型"""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str
    trajectories: List[Trajectory]
    evaluation_options: Dict[str, Any] = field(default_factory=dict)

    class Config:
        arbitrary_types_allowed = True


class TrajectorySummary(BaseModel):
    """轨迹总结"""
    trajectory_id: str
    summary: str
    confidence: float = 1.0
    insights: List[str] = []
    suggestions: List[str] = []

    class Config:
        extra = "allow"


class EvaluationResponse(BaseModel):
    """评估响应"""
    request_id: str
    success: bool
    message: str = ""
    summaries: List[TrajectorySummary] = []
    processing_time: float = 0.0
    model_used: str = ""
    metadata: Dict[str, Any] = {}

    class Config:
        extra = "allow"


class CloudMessage(BaseModel):
    """云边通信消息"""
    message_type: MessageType
    data: Dict[str, Any]
    timestamp: str = ""
    source: str = "edge"
    destination: str = "cloud"
    request_id: Optional[str] = None

    def __init__(self, **data):
        if not data.get("timestamp"):
            data["timestamp"] = datetime.now().isoformat()
        super().__init__(**data)

    @property
    def payload(self) -> Dict[str, Any]:
        """兼容性属性：返回data"""
        return self.data

    def to_json(self) -> str:
        """序列化为JSON字符串"""
        return self.json()

    @classmethod
    def from_json(cls, json_str: str) -> 'CloudMessage':
        """从JSON字符串反序列化"""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def create_evaluation_request(cls, request: EvaluationRequest) -> 'CloudMessage':
        """创建评估请求消息"""
        return cls(
            message_type=MessageType.TRAJECTORY_EVALUATION,
            data=request.dict(),
            source="edge",
            destination="cloud",
            request_id=request.request_id
        )

    @classmethod
    def create_evaluation_result(cls, response: EvaluationResponse) -> 'CloudMessage':
        """创建评估结果消息"""
        return cls(
            message_type=MessageType.EVALUATION_RESULT,
            data=response.dict(),
            source="cloud",
            destination="edge",
            request_id=response.request_id
        )


class ProtocolConverter:
    """协议转换器"""

    @staticmethod
    def convert_from_legacy_format(
        trajectory_data: List[List[Any]],
        trajectory_id: Optional[str] = None
    ) -> Trajectory:
        """从旧格式转换轨迹数据

        Args:
            trajectory_data: 旧格式轨迹数据 [[("Action", "xxx"), ("Observation", "yyy")], ...]
            trajectory_id: 轨迹ID

        Returns:
            Trajectory对象
        """
        if trajectory_id is None:
            trajectory_id = str(uuid.uuid4())

        items = []
        subgoals = []

        for step in trajectory_data:
            step_items = []
            has_subgoal = False

            for item in step:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    item_type = item[0]
                    content = str(item[1])

                    trajectory_item = TrajectoryItem(
                        type=item_type,
                        content=content
                    )

                    if item_type == "Subgoal":
                        subgoals.append(trajectory_item)
                        has_subgoal = True

                    step_items.append(trajectory_item)

            if step_items:
                items.append(step_items)

        return Trajectory(
            trajectory_id=trajectory_id,
            items=items,
            subgoals=subgoals
        )

    @staticmethod
    def convert_to_legacy_format(trajectory: Trajectory) -> List[List[Any]]:
        """转换为旧格式

        Returns:
            旧格式轨迹数据
        """
        result = []

        for step in trajectory.items:
            step_data = []
            for item in step:
                step_data.append((item.type, item.content))
            result.append(step_data)

        return result