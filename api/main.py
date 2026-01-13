"""
云端服务器

提供HTTP API和WebSocket接口，接收边缘端的评估请求并返回结果
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, status, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from cloud.src.protocol import (
    CloudMessage, MessageType, ProtocolConverter,
    EvaluationRequest, EvaluationResponse, Trajectory
)
from cloud.core.evaluator import CloudEvaluator, CloudLLMConfig
from cloud.core.evaluator import CloudEvaluator, CloudLLMConfig
from cloud.core.logger import init_cloud_logger, log_with_category

# 初始化日志系统
logger = init_cloud_logger()

# 全局变量
evaluator: CloudEvaluator = None


class RequestData(BaseModel):
    """请求数据模型"""
    message_type: str
    data: Dict[str, Any]
    source: str = "edge"
    destination: str = "cloud"


class CloudServer:
    """云端服务器主类"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8000):
        """初始化服务器

        Args:
            host: 服务器主机地址
            port: 服务器端口
        """
        self.app = FastAPI(
            title="HiAgent Cloud Evaluation Server",
            description="Cloud server for trajectory evaluation and summarization",
            version="1.0.0"
        )
        self.host = host
        self.port = port
        self.setup_middleware()
        self.setup_routes()
        self.active_requests = {}  # 活跃请求跟踪
        self.websocket_connections = []  # WebSocket连接管理
        self.trajectory_store = {}  # 轨迹数据存储

    def setup_middleware(self):
        """设置中间件"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def setup_routes(self):
        """设置API路由"""

        @self.app.get("/")
        async def root():
            """根路径"""
            return {
                "message": "HiAgent Cloud Evaluation Server",
                "status": "running",
                "timestamp": datetime.now().isoformat()
            }

        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "evaluator": evaluator is not None,
                "active_requests": len(self.active_requests),
                "websocket_connections": len(self.websocket_connections)
            }

        @self.app.post("/evaluate")
        async def evaluate_trajectory(request_data: RequestData):
            """评估轨迹端点（HTTP REST API）"""
            try:
                # 创建消息对象
                message = CloudMessage(
                    message_type=MessageType(request_data.message_type),
                    data=request_data.data,
                    timestamp=datetime.now().isoformat(),
                    source=request_data.source,
                    destination=request_data.destination
                )

                # 处理评估请求
                if message.message_type == MessageType.TRAJECTORY_EVALUATION:
                    # 解析评估请求
                    eval_request = EvaluationRequest(**message.data)

                    # 异步处理评估
                    response = await self.process_evaluation(eval_request)

                    return {
                        "success": True,
                        "data": response,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Unsupported message type: {message.message_type}"
                    )

            except Exception as e:
                logger.error(f"Error processing evaluation request: {str(e)}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=str(e)
                )

        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket端点（兼容旧版）"""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            self.websocket_connections.append(websocket)
            log_with_category(logger, "info", "CONNECTION", f"New WebSocket connection accepted (total: {len(self.websocket_connections)})")

            try:
                while True:
                    data = await websocket.receive_text()

                    # 尝试解析 JSON
                    try:
                        message_data = json.loads(data)
                        
                        # 兼容两种字段名
                        msg_type_str = message_data.get("message_type") or message_data.get("type", "UNKNOWN")
                        payload = message_data.get("data") or message_data.get("payload", {})

                        # 标准化处理 AGENT_OUTPUT
                        if msg_type_str == "AGENT_OUTPUT":
                            action = payload.get("action")
                            
                            if action == "upload":
                                # 处理轨迹上传
                                t_data = payload.get("trajectory_data")
                                if t_data:
                                    try:
                                        # 解析并存储轨迹
                                        trajectory = Trajectory.from_dict(t_data)
                                        self.trajectory_store[trajectory.trajectory_id] = trajectory
                                        
                                        log_with_category(
                                            logger, "info", "EDGE_DATA", 
                                            f"Stored trajectory {trajectory.trajectory_id}",
                                            data={"trajectory_id": trajectory.trajectory_id, "items_count": len(trajectory.items)}
                                        )
                                        
                                        # 发送确认
                                        resp = {"status": "success", "message": "Trajectory uploaded"}
                                        await websocket.send_json(resp)
                                        
                                    except Exception as e:
                                        logger.error(f"[Cloud] Failed to parse trajectory: {e}")
                                        await websocket.send_json({"status": "error", "message": str(e)})

                            elif action == "summarize":
                                # 处理总结请求
                                t_id = payload.get("trajectory_id")
                                log_with_category(logger, "info", "PROTOCOL", f"Processing summary request", data={"trajectory_id": t_id})
                                
                                if t_id and t_id in self.trajectory_store:
                                    trajectory = self.trajectory_store[t_id]
                                    
                                    # 创建评估请求
                                    eval_req = EvaluationRequest(
                                        agent_id="agent", 
                                        trajectories=[trajectory]
                                    )
                                    
                                    # 处理评估
                                    response = await self.process_evaluation(eval_req)
                                    
                                    # 构造总结响应
                                    if response.get("summaries"):
                                        summary_obj = response["summaries"][0]
                                        
                                        # 构造响应Payload
                                        resp_payload = {
                                            "summary": {
                                                "trajectory_id": summary_obj["trajectory_id"],
                                                "summary": summary_obj["summary"],
                                                "confidence": summary_obj["confidence"],
                                                "insights": summary_obj["insights"],
                                                "suggestions": summary_obj["suggestions"]
                                            }
                                        }
                                        
                                        # 构造CloudMessage
                                        resp_msg = CloudMessage(
                                            message_type=MessageType.SUMMARY_RESPONSE,
                                            data=resp_payload,
                                            source="cloud",
                                            destination="edge",
                                        )
                                        
                                        await websocket.send_text(resp_msg.to_json())
                                        log_with_category(logger, "info", "PROTOCOL", f"Sent summary response", data={"trajectory_id": t_id})
                                    else:
                                        await websocket.send_json({"status": "error", "message": "No summary generated"})
                                else:
                                    logger.warning(f"[Cloud] Trajectory {t_id} not found")
                                    await websocket.send_json({"status": "error", "message": "Trajectory not found"})

                        # 检查是否是新的评估请求格式
                        elif msg_type_str == "TRAJECTORY_EVALUATION":
                            # 处理新的评估请求
                            eval_request = EvaluationRequest(**payload)
                            response = await self.process_evaluation(eval_request)
                            await websocket.send_json({
                                "type": "EVALUATION_RESULT",
                                "payload": response
                            })
                        else:
                            # 兼容旧版格式
                            raw_content = payload.get("raw_content", str(message_data))
                            log_with_category(logger, "info", "PROTOCOL", f"Received UNKNOWN/LEGACY message type: {msg_type_str}")
                            log_with_category(logger, "info", "EDGE_OUTPUT", "Raw content received", data={"content": raw_content})

                            # 构造结构化回执
                            response = {
                                "status": "success",
                                "instruction": "CONTINUE",  # 指令: 继续
                                "feedback": ""              # 反馈: 目前为空
                            }
                            await websocket.send_json(response)

                    except json.JSONDecodeError:
                        # 兼容旧版纯文本
                        logger.info(f"\n[Cloud] 收到原始文本: {data}")

                        # 构造结构化回执
                        response = {
                            "status": "success",
                            "instruction": "CONTINUE",
                            "feedback": ""
                        }
                        await websocket.send_json(response)

            except WebSocketDisconnect:
                self.websocket_connections.remove(websocket)
                log_with_category(logger, "info", "CONNECTION", f"Client disconnected (remaining: {len(self.websocket_connections)})")
            except Exception as e:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
                logger.error(f"[Cloud] WebSocket error: {e}")

        @self.app.get("/request/{request_id}")
        async def get_request_status(request_id: str):
            """获取请求状态"""
            if request_id in self.active_requests:
                return self.active_requests[request_id]
            else:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Request not found"
                )

        @self.app.get("/history")
        async def get_evaluation_history(limit: int = 100):
            """获取评估历史"""
            if evaluator:
                history = evaluator.get_evaluation_history()
                # 返回最近的记录
                recent_history = dict(list(history.items())[-limit:])
                return {
                    "total_records": len(history),
                    "returned_records": len(recent_history),
                    "data": recent_history
                }
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Evaluator not initialized"
                )

        @self.app.get("/provider/info")
        async def get_provider_info():
            """获取LLM提供者信息"""
            if evaluator:
                return evaluator.get_provider_info()
            else:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Evaluator not initialized"
                )

    async def process_evaluation(self, request: EvaluationRequest) -> Dict:
        """处理评估请求

        Args:
            request: 评估请求

        Returns:
            评估结果字典
        """
        # 记录请求开始
        self.active_requests[request.request_id] = {
            "status": "processing",
            "start_time": datetime.now().isoformat(),
            "agent_id": request.agent_id,
            "trajectory_count": len(request.trajectories)
        }

        try:
            # 执行评估
            response = await evaluator.evaluate_trajectories(request)

            # 更新请求状态
            self.active_requests[request.request_id] = {
                "status": "completed",
                "start_time": self.active_requests[request.request_id]["start_time"],
                "end_time": datetime.now().isoformat(),
                "success": response.success,
                "message": response.message
            }

            # 返回结果
            return {
                "request_id": response.request_id,
                "success": response.success,
                "message": response.message,
                "summaries": [
                    {
                        "trajectory_id": s.trajectory_id,
                        "summary": s.summary,
                        "confidence": s.confidence,
                        "insights": s.insights,
                        "suggestions": s.suggestions
                    }
                    for s in response.summaries
                ],
                "processing_time": response.processing_time,
                "model_used": response.model_used
            }

        except Exception as e:
            # 更新请求状态为失败
            self.active_requests[request.request_id] = {
                "status": "failed",
                "start_time": self.active_requests[request.request_id]["start_time"],
                "end_time": datetime.now().isoformat(),
                "error": str(e)
            }
            raise

    def run(self):
        """运行服务器"""
        logger.info(f"Starting Cloud Server on {self.host}:{self.port}")
        uvicorn.run(
            self.app,
            host=self.host,
            port=self.port,
            log_level="info"
        )


# 全局服务器实例
server_instance = None


def initialize_server():
    """初始化服务器"""
    global evaluator, server_instance

    try:
        # 从环境变量初始化评估器
        llm_config = CloudLLMConfig()

        evaluator = CloudEvaluator(llm_config)

        # 从环境变量初始化服务器
        server_instance = CloudServer(
            host=os.getenv("CLOUD_HOST", "0.0.0.0"),
            port=int(os.getenv("CLOUD_PORT", "8000"))
        )

        log_with_category(logger, "info", "SYSTEM", "Cloud server initialized successfully")
        return server_instance

    except Exception as e:
        logger.error(f"Failed to initialize server: {str(e)}")
        raise


def main():
    """主函数"""
    # 初始化服务器
    server = initialize_server()

    # 运行服务器
    server.run()


if __name__ == "__main__":
    main()