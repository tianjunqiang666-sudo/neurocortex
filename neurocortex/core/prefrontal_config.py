"""
NeuroCortex AI — PrefrontalConfig / ModelRouter
=================================================
动态且安全的模型管理。根据脑区名称动态路由到正确的LLM客户端并进行能力契约校验。
"""

from __future__ import annotations
import time
from typing import Any, Optional
import ollama
import openai
from loguru import logger
from neurocortex.core.config_loader import ConfigLoader


class CapabilityMismatchError(Exception):
    """模型能力契约不满足脑区需求"""
    pass

class ModelNotFoundError(Exception):
    """找不到匹配的模型"""
    pass


class LLMClient:
    """统一的LLM客户端接口，封装不同provider的调用差异"""

    def __init__(self, model_id: str, provider: str, model_name: str,
                 endpoint: str | None = None, api_key: str | None = None) -> None:
        self.model_id = model_id
        self.provider = provider
        self.model_name = model_name
        self.endpoint = endpoint
        self.api_key = api_key
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        if self.provider == "ollama":
            self._client = ollama.Client(host=self.endpoint)
        elif self.provider == "openai_compatible":
            self._client = openai.OpenAI(base_url=self.endpoint, api_key=self.api_key or "dummy")
        elif self.provider == "anthropic":
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.warning("anthropic 库未安装")
                self._client = None

    def chat(self, prompt: str, system_prompt: str = "", temperature: float = 0.7,
             max_tokens: int = 2048, json_mode: bool = False, images: list[str] | None = None) -> str:
        """统一对话接口"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if images:
            messages[-1]["images"] = images

        try:
            if self.provider == "ollama":
                return self._chat_ollama(messages, temperature, json_mode)
            elif self.provider == "openai_compatible":
                return self._chat_openai(messages, temperature, max_tokens, json_mode)
            elif self.provider == "anthropic":
                return self._chat_anthropic(messages, temperature, max_tokens)
        except Exception as e:
            logger.error(f"LLM 调用失败 [{self.model_id}]: {e}")
            raise

    def _chat_ollama(self, messages: list[dict], temperature: float, json_mode: bool) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name, "messages": messages,
            "options": {"temperature": temperature},
        }
        if json_mode:
            kwargs["format"] = "json"
        response = self._client.chat(**kwargs)
        return response["message"]["content"]

    def _chat_openai(self, messages: list[dict], temperature: float,
                     max_tokens: int, json_mode: bool) -> str:
        kwargs: dict[str, Any] = {
            "model": self.model_name, "messages": messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _chat_anthropic(self, messages: list[dict], temperature: float, max_tokens: int) -> str:
        if self._client is None:
            raise RuntimeError("Anthropic 客户端未初始化")
        system_msg = ""
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_msg = msg["content"]
            else:
                user_messages.append(msg)
        kwargs: dict[str, Any] = {
            "model": self.model_name, "messages": user_messages,
            "temperature": temperature, "max_tokens": max_tokens,
        }
        if system_msg:
            kwargs["system"] = system_msg
        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    async def chat_stream(self, prompt: str, system_prompt: str = "", temperature: float = 0.7,
                          max_tokens: int = 2048) -> Any:
        """统一流式对话接口 (AsyncGenerator)"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        import asyncio

        if self.provider == "ollama":
            kwargs = {
                "model": self.model_name, "messages": messages,
                "options": {"temperature": temperature}, "stream": True
            }
            # ollama.AsyncClient
            async_client = ollama.AsyncClient(host=self.endpoint)
            async for chunk in await async_client.chat(**kwargs):
                yield chunk["message"]["content"]

        elif self.provider == "openai_compatible":
            import openai
            async_client = openai.AsyncOpenAI(base_url=self.endpoint, api_key=self.api_key or "dummy")
            kwargs = {
                "model": self.model_name, "messages": messages,
                "temperature": temperature, "max_tokens": max_tokens, "stream": True
            }
            stream = await async_client.chat.completions.create(**kwargs)
            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

        elif self.provider == "anthropic":
            import anthropic
            async_client = anthropic.AsyncAnthropic(api_key=self.api_key)
            system_msg = ""
            user_messages = []
            for msg in messages:
                if msg["role"] == "system":
                    system_msg = msg["content"]
                else:
                    user_messages.append(msg)
            kwargs = {
                "model": self.model_name, "messages": user_messages,
                "temperature": temperature, "max_tokens": max_tokens,
            }
            if system_msg:
                kwargs["system"] = system_msg
            async with async_client.messages.stream(**kwargs) as stream:
                async for text in stream.text_stream:
                    yield text
        else:
            raise NotImplementedError(f"不支持的 provider: {self.provider}")

    def __repr__(self) -> str:
        return f"LLMClient(id={self.model_id}, provider={self.provider}, model={self.model_name})"


class ModelRouter:
    """根据脑区名称和任务，动态路由到正确的LLM客户端，并进行契约校验"""

    def __init__(self, config_path: str | None = None) -> None:
        self.config_loader = ConfigLoader(config_path)
        self.active_clients: dict[str, LLMClient] = {}
        self.overrides: dict[str, str] = {}

    def get_client(self, brain_region: str, preferred_model: str | None = None,
                   required_capabilities: dict[str, Any] | None = None) -> LLMClient:
        """获取指定脑区的LLM客户端，按优先级路由并校验能力契约"""
        registry = self.config_loader.get_model_registry()
        candidate_id = self._resolve_model_id(brain_region, preferred_model, registry)
        if required_capabilities:
            if not self.capability_check(required_capabilities, candidate_id):
                raise CapabilityMismatchError(
                    f"模型 '{candidate_id}' 不满足脑区 '{brain_region}' 的能力需求: {required_capabilities}")
        return self._get_or_create_client(candidate_id)

    def switch_model(self, brain_region: str, new_model_id: str) -> LLMClient:
        """运行时热切换指定脑区的模型"""
        registry = self.config_loader.get_model_registry()
        if new_model_id not in registry:
            raise ModelNotFoundError(f"模型 '{new_model_id}' 不在注册表中")
        self.overrides[brain_region] = new_model_id
        logger.info(f"脑区 '{brain_region}' 切换到模型 '{new_model_id}'")
        return self._get_or_create_client(new_model_id)

    def capability_check(self, brain_region_requirements: dict[str, Any], model_id: str) -> bool:
        """验证模型能力契约是否满足脑区需求"""
        registry = self.config_loader.get_model_registry()
        if model_id not in registry:
            return False
        contract = registry[model_id].get("capability_contract", {})

        # 推理深度校验
        depth_order = {"low": 1, "medium": 2, "high": 3}
        required_depth = brain_region_requirements.get("reasoning_depth", "low")
        model_depth = contract.get("reasoning_depth", "low")
        if depth_order.get(model_depth, 0) < depth_order.get(required_depth, 0):
            logger.warning(f"模型 '{model_id}' 推理深度不足: {model_depth} < {required_depth}")
            return False

        # 输出模式兼容性
        req_schema = brain_region_requirements.get("output_schema")
        mod_schema = contract.get("output_schema")
        if req_schema and mod_schema:
            req_type = req_schema.get("type") if isinstance(req_schema, dict) else req_schema
            mod_type = mod_schema.get("type") if isinstance(mod_schema, dict) else mod_schema
            if req_type != mod_type and mod_type != "string":
                logger.warning(f"模型 '{model_id}' output_schema 不兼容: {mod_type} vs {req_type}")
                return False

        logger.debug(f"模型 '{model_id}' 通过能力契约校验")
        return True

    def test_connectivity(self, model_id: str) -> dict[str, Any]:
        """测试指定模型的连接性、延迟和格式校验"""
        result = {"model_id": model_id, "available": False, "latency_ms": -1,
                  "error": None, "schema_valid": False}
        try:
            client = self._get_or_create_client(model_id)
            start = time.time()
            response = client.chat(prompt='回复JSON: {"status":"ok"}',
                                   system_prompt="仅返回请求的JSON。",
                                   temperature=0.0, max_tokens=50)
            elapsed = (time.time() - start) * 1000
            result["available"] = True
            result["latency_ms"] = round(elapsed, 1)
            result["schema_valid"] = "ok" in response.lower()
            logger.info(f"模型 '{model_id}' 可用, 延迟={elapsed:.0f}ms")
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"模型 '{model_id}' 连接测试失败: {e}")
        return result

    def list_models(self) -> dict[str, dict[str, Any]]:
        return self.config_loader.get_model_registry()

    def get_models_for_region(self, brain_region: str) -> list[str]:
        registry = self.config_loader.get_model_registry()
        return [mid for mid, cfg in registry.items() if brain_region in cfg.get("scope", [])]

    def _resolve_model_id(self, brain_region: str, preferred_model: str | None,
                          registry: dict[str, Any]) -> str:
        if preferred_model and preferred_model in registry:
            return preferred_model
        if brain_region in self.overrides and self.overrides[brain_region] in registry:
            return self.overrides[brain_region]
        for model_id, cfg in registry.items():
            if brain_region in cfg.get("scope", []):
                return model_id
        raise ModelNotFoundError(f"找不到适用于脑区 '{brain_region}' 的模型")

    def _get_or_create_client(self, model_id: str) -> LLMClient:
        if model_id in self.active_clients:
            return self.active_clients[model_id]
        model_cfg = self.config_loader.get_model_config(model_id)
        api_key = self.config_loader.resolve_api_key(model_id)
        client = LLMClient(
            model_id=model_id, provider=model_cfg["provider"],
            model_name=model_cfg["model_name"], endpoint=model_cfg.get("endpoint"),
            api_key=api_key)
        self.active_clients[model_id] = client
        logger.info(f"已创建 LLM 客户端: {client}")
        return client
