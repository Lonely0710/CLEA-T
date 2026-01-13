"""
LLM提供者接口和实现
支持OpenAI和GLM模型（使用OpenAI兼容接口）
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from openai import AsyncOpenAI
from cloud.core.logger import get_logger

logger = get_logger(__name__)


class LLMProvider(ABC):
    """LLM提供者抽象基类"""

    @abstractmethod
    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        **kwargs
    ) -> str:
        """生成文本完成"""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass


class OpenAICompatibleProvider(LLMProvider):
    """OpenAI兼容的模型提供者（支持OpenAI和GLM）"""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1", provider: str = "openai"):
        self.api_key = api_key
        self.base_url = base_url
        self.provider = provider
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url
        )

    async def generate_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 1000,
        model: str = "gpt-4",
        **kwargs
    ) -> str:
        """使用OpenAI兼容API生成文本"""
        try:
            response = await self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"{self.provider} API error: {e}")
            raise

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        if self.provider == "openai":
            return {
                "provider": "openai",
                "models": ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo-preview", "gpt-4o"],
                "base_url": self.base_url
            }
        elif self.provider == "glm":
            return {
                "provider": "glm",
                "models": ["glm-4.6", "glm-4.5-air", "glm-4.6v"],
                "base_url": self.base_url
            }
        elif self.provider == "deepseek":
            return {
                "provider": "deepseek",
                "models": ["deepseek-chat", "deepseek-reasoner"],
                "base_url": self.base_url
            }
        else:
            return {
                "provider": self.provider,
                "models": [],
                "base_url": self.base_url
            }


class LLMConfig:
    """LLM配置管理"""

    def __init__(self, provider: str = "openai", model: str = "gpt-4",
                 api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.provider = provider
        self.model = model
        self.api_key = api_key
        self.base_url = base_url

    def get_provider(self) -> LLMProvider:
        """获取LLM提供者实例"""
        if self.provider == "openai":
            api_key = self.api_key or "your-openai-api-key"
            base_url = self.base_url or "https://api.openai.com/v1"
            return OpenAICompatibleProvider(api_key, base_url, "openai")
        elif self.provider == "glm":
            api_key = self.api_key or "your-glm-api-key"
            base_url = self.base_url or "https://open.bigmodel.cn/api/paas/v4/"
            return OpenAICompatibleProvider(api_key, base_url, "glm")
        elif self.provider == "deepseek":
            api_key = self.api_key or "your-deepseek-api-key"
            base_url = self.base_url or "https://api.deepseek.com"
            return OpenAICompatibleProvider(api_key, base_url, "deepseek")
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")


def parse_model_string(model_str: str) -> tuple:
    """
    解析模型字符串，返回(提供商, 模型名)

    示例:
    "gpt-4" -> ("openai", "gpt-4")
    "glm:glm-4.6" -> ("glm", "glm-4.6")
    """
    if ":" in model_str:
        provider, model = model_str.split(":", 1)
        return provider.lower(), model

    # 默认判断
    if model_str.startswith("glm"):
        return "glm", model_str
    elif model_str.startswith("deepseek"):
        return "deepseek", model_str
    elif model_str.startswith("gpt"):
        return "openai", model_str
    else:
        # 默认使用OpenAI
        return "openai", model_str