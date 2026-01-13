"""
云端LLM评估器

提供轨迹评估和总结功能，支持OpenAI和GLM模型
"""

import asyncio
import json
import logging
import os
from typing import List, Dict, Any, Optional
import time
from datetime import datetime
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

from cloud.src.protocol import (
    EvaluationRequest, EvaluationResponse, Trajectory,
    TrajectorySummary, TrajectoryItem
)
from cloud.core.llm_providers import (
    LLMProvider, LLMConfig, parse_model_string
)
from cloud.core.logger import get_logger

logger = get_logger(__name__)


class CloudLLMConfig:
    """云端LLM配置"""
    def __init__(
        self,
        model_name: Optional[str] = None,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None
    ):
        # 解析模型字符串，确定提供商
        if model_name:
            self.provider, self.model = parse_model_string(model_name)
        else:
            # 从环境变量读取
            default_model = os.getenv("CLOUD_MODEL", "gpt-4")
            self.provider, self.model = parse_model_string(default_model)

        if provider:
            self.provider = provider.lower()

        # 根据提供商获取对应的API密钥和base_url
        if self.provider == "glm":
            self.api_key = api_key or os.getenv("GLM_API_KEY")
            self.api_base = api_base or os.getenv("GLM_API_BASE", "https://open.bigmodel.cn/api/paas/v4/")
        elif self.provider == "deepseek":
            self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
            self.api_base = api_base or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        else:  # openai
            self.api_key = api_key or os.getenv("OPENAI_API_KEY")
            self.api_base = api_base or os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

        self.temperature = temperature if temperature is not None else float(os.getenv("TEMPERATURE", "0.3"))
        self.max_tokens = max_tokens if max_tokens is not None else int(os.getenv("MAX_TOKENS", "1000"))
        self.timeout = timeout if timeout is not None else int(os.getenv("TIMEOUT", "30"))

        try:
            self.llm_config = LLMConfig(
                provider=self.provider,
                model=self.model,
                api_key=self.api_key,
                base_url=self.api_base
            )
            self.llm_provider = self.llm_config.get_provider()
        except Exception as e:
            logger.error(f"Failed to create LLM provider: {e}")
            raise

    def get_provider_info(self) -> Dict[str, Any]:
        """获取提供者信息"""
        return self.llm_provider.get_model_info()


class CloudEvaluator:
    """云端评估器"""

    def __init__(self, config: CloudLLMConfig):
        """初始化评估器

        Args:
            config: LLM配置
        """
        self.config = config
        self.evaluation_history = {}  # 评估历史记录
        self.cache = {}  # 评估缓存

        # 根据不同的提供者调整提示词
        if config.provider == "glm":
            self.system_prompt = """你是一个专业的任务执行评估助手。请分析提供的任务执行轨迹并生成高质量的总结。

你的任务：
1. 理解任务目标和执行步骤
2. 识别关键的动作和观察
3. 提取重要的模式和洞察
4. 生成简洁准确的总结

请以JSON格式返回结果，包含以下字段：
{
    "summary": "轨迹总结（100-200字）",
    "insights": ["关键洞察1", "关键洞察2", "关键洞察3"],
    "suggestions": ["建议1", "建议2"],
    "confidence": 0.85
}"""
        else:  # openai
            self.system_prompt = """你是一个专业的任务执行评估助手。请分析提供的任务执行轨迹并生成高质量的总结。

你的任务：
1. 理解任务目标和执行步骤
2. 识别关键的动作和观察
3. 提取重要的模式和洞察
4. 生成简洁准确的总结

请以JSON格式返回结果，包含以下字段：
- summary: 轨迹总结（100-200字）
- insights: 关键洞察列表（3-5条）
- suggestions: 改进建议列表（2-3条）
- confidence: 总结的置信度（0-1之间的浮点数）

示例输出格式：
{
    "summary": "任务执行过程的简要总结...",
    "insights": ["洞察1", "洞察2", "洞察3"],
    "suggestions": ["建议1", "建议2"],
    "confidence": 0.85
}"""

    async def evaluate_trajectories(
        self,
        request: EvaluationRequest
    ) -> EvaluationResponse:
        """评估轨迹

        Args:
            request: 评估请求

        Returns:
            评估响应
        """
        start_time = time.time()

        try:
            # 创建响应对象
            response = EvaluationResponse(
                request_id=request.request_id,
                success=True,
                model_used=self.config.model,
                metadata={
                    "provider": self.config.provider,
                    "api_base": self.config.api_base
                }
            )

            # 评估每个轨迹
            for trajectory in request.trajectories:
                # 检查缓存
                cache_key = self._get_cache_key(trajectory)
                if cache_key in self.cache:
                    summary_dict = self.cache[cache_key]
                else:
                    # 生成新的评估
                    summary_dict = await self._evaluate_single_trajectory(trajectory)
                    # 缓存结果
                    self.cache[cache_key] = summary_dict

                # 创建总结对象
                summary = TrajectorySummary(
                    trajectory_id=trajectory.trajectory_id,
                    **summary_dict
                )
                response.summaries.append(summary)

            # 计算处理时间
            response.processing_time = time.time() - start_time

            # 记录历史
            self.evaluation_history[request.request_id] = {
                "timestamp": datetime.now().isoformat(),
                "agent_id": request.agent_id,
                "processing_time": response.processing_time,
                "trajectory_count": len(request.trajectories),
                "success": True,
                "provider": self.config.provider,
                "model": self.config.model
            }

            logger.info(f"Completed evaluation for request {request.request_id}")
            return response

        except Exception as e:
            logger.error(f"Evaluation failed for request {request.request_id}: {str(e)}")

            # 记录失败历史
            self.evaluation_history[request.request_id] = {
                "timestamp": datetime.now().isoformat(),
                "agent_id": request.agent_id,
                "processing_time": time.time() - start_time,
                "trajectory_count": len(request.trajectories),
                "success": False,
                "error": str(e),
                "provider": self.config.provider,
                "model": self.config.model
            }

            # 返回失败响应
            return EvaluationResponse(
                request_id=request.request_id,
                success=False,
                message=str(e),
                processing_time=time.time() - start_time,
                model_used=self.config.model,
                metadata={
                    "provider": self.config.provider
                }
            )

    async def _evaluate_single_trajectory(self, trajectory: Trajectory) -> Dict[str, Any]:
        """评估单个轨迹

        Args:
            trajectory: 轨迹数据

        Returns:
            总结字典
        """
        # 构建轨迹文本
        trajectory_text = self._trajectory_to_text(trajectory)

        # 构建提示词
        user_prompt = f"""请分析以下任务执行轨迹：

{trajectory_text}

请生成总结并提供关键洞察和改进建议。"""

        try:
            # 调用LLM API
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            content = await self.config.llm_provider.generate_completion(
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                model=self.config.model
            )

            # 尝试解析JSON
            try:
                result = json.loads(content)
            except json.JSONDecodeError:
                # 如果JSON解析失败，使用简单总结
                logger.warning("Failed to parse JSON response, using fallback")
                result = self._create_fallback_summary(trajectory_text)

            return result

        except Exception as e:
            logger.error(f"LLM evaluation failed: {str(e)}")
            # 降级到简单总结
            trajectory_text = self._trajectory_to_text(trajectory)
            return self._create_fallback_summary(trajectory_text)

    def _trajectory_to_text(self, trajectory: Trajectory) -> str:
        """将轨迹转换为文本

        Args:
            trajectory: 轨迹对象

        Returns:
            文本表示
        """
        lines = []

        # 添加子目标
        if trajectory.subgoals:
            lines.append("子目标:")
            for i, subgoal in enumerate(trajectory.subgoals):
                lines.append(f"  {i+1}. {subgoal.content}")
            lines.append("")

        # 添加轨迹步骤
        lines.append("执行轨迹:")
        for step_idx, step in enumerate(trajectory.items):
            lines.append(f"\n步骤 {step_idx + 1}:")
            for item in step:
                lines.append(f"  {item.type}: {item.content}")

        return "\n".join(lines)

    def _create_fallback_summary(self, trajectory_text: str) -> Dict[str, Any]:
        """创建降级总结

        Args:
            trajectory_text: 轨迹文本

        Returns:
            总结字典
        """
        # 简单截取文本作为总结
        summary = trajectory_text[:200] + "..." if len(trajectory_text) > 200 else trajectory_text

        return {
            "summary": summary,
            "insights": ["轨迹已记录"],
            "suggestions": ["继续执行任务"],
            "confidence": 0.5
        }

    def _get_cache_key(self, trajectory: Trajectory) -> str:
        """生成缓存键

        Args:
            trajectory: 轨迹对象

        Returns:
            缓存键
        """
        # 使用轨迹内容的前几个步骤生成哈希
        content_str = f"{len(trajectory.items)}_{self.config.provider}_{self.config.model}"
        for step in trajectory.items[:3]:
            for item in step:
                content_str += f"{item.type}:{item.content[:30]}"

        return str(hash(content_str))

    def get_evaluation_history(self) -> Dict[str, Any]:
        """获取评估历史

        Returns:
            历史记录字典
        """
        return self.evaluation_history.copy()

    def clear_cache(self):
        """清除缓存"""
        self.cache.clear()
        logger.info("Evaluation cache cleared")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计

        Returns:
            缓存统计信息
        """
        return {
            "cache_size": len(self.cache),
            "history_size": len(self.evaluation_history)
        }

    def get_provider_info(self) -> Dict[str, Any]:
        """获取当前提供者信息"""
        return self.config.get_provider_info()