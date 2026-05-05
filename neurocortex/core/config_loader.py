"""
NeuroCortex AI — 配置加载器
============================
负责 YAML 配置文件的加载、校验与热重载。
所有 API Key 从环境变量安全读取。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger


# 默认配置文件路径
_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "model_config.yaml"


class ConfigValidationError(Exception):
    """配置校验错误"""
    pass


class ConfigLoader:
    """YAML 配置加载与校验器

    Attributes:
        config_path: 配置文件路径
        _config: 已解析的配置字典
        _last_modified: 上次加载时文件的修改时间
    """

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
        self._config: dict[str, Any] = {}
        self._last_modified: float = 0.0
        self._load()

    # ── 公共接口 ──────────────────────────────────────────

    def get_model_registry(self) -> dict[str, Any]:
        """返回 model_registry 字典"""
        self._hot_reload_if_changed()
        return self._config.get("model_registry", {})

    def get_system_settings(self) -> dict[str, Any]:
        """返回 system 全局配置"""
        self._hot_reload_if_changed()
        return self._config.get("system", {})

    def get_model_config(self, model_id: str) -> dict[str, Any]:
        """获取指定模型的完整配置"""
        registry = self.get_model_registry()
        if model_id not in registry:
            raise KeyError(f"模型 '{model_id}' 未在注册表中找到。可用模型: {list(registry.keys())}")
        return registry[model_id]

    def resolve_api_key(self, model_id: str) -> str | None:
        """从环境变量中安全读取 API Key（绝不返回硬编码值）"""
        model_cfg = self.get_model_config(model_id)
        env_var = model_cfg.get("api_key_env")
        if not env_var:
            return None  # 本地模型不需要 API Key
        api_key = os.environ.get(env_var)
        if not api_key:
            logger.warning(f"环境变量 '{env_var}' 未设置，模型 '{model_id}' 可能无法使用")
        return api_key

    def reload(self) -> None:
        """强制重新加载配置文件"""
        self._load()
        logger.info(f"配置文件已重新加载: {self.config_path}")

    def save(self, config: dict[str, Any] | None = None) -> None:
        """将当前配置（或传入的配置）写回 YAML 文件"""
        data = config if config is not None else self._config
        with open(self.config_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        self._last_modified = self.config_path.stat().st_mtime
        logger.info(f"配置已保存到: {self.config_path}")

    # ── 内部方法 ──────────────────────────────────────────

    def _load(self) -> None:
        """加载并校验配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")

        with open(self.config_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)

        if not isinstance(raw, dict):
            raise ConfigValidationError("配置文件顶层必须是字典")

        self._validate(raw)
        self._config = raw
        self._last_modified = self.config_path.stat().st_mtime
        logger.debug(f"配置已加载: {self.config_path}")

    def _hot_reload_if_changed(self) -> None:
        """如果文件有更新则自动重新加载（热重载）"""
        try:
            current_mtime = self.config_path.stat().st_mtime
            if current_mtime > self._last_modified:
                logger.info("检测到配置文件变更，自动重新加载...")
                self._load()
        except OSError:
            pass  # 文件暂时不可访问，使用缓存

    def _validate(self, config: dict[str, Any]) -> None:
        """校验配置结构的完整性"""
        registry = config.get("model_registry")
        if not registry:
            raise ConfigValidationError("配置中缺少 'model_registry' 字段")

        for model_id, model_cfg in registry.items():
            # 必须字段校验
            required_fields = ["provider", "model_name", "scope", "capability_contract"]
            for field in required_fields:
                if field not in model_cfg:
                    raise ConfigValidationError(
                        f"模型 '{model_id}' 缺少必要字段: '{field}'"
                    )

            # Provider 类型校验
            valid_providers = {"ollama", "openai_compatible", "anthropic"}
            if model_cfg["provider"] not in valid_providers:
                raise ConfigValidationError(
                    f"模型 '{model_id}' 的 provider '{model_cfg['provider']}' 无效。"
                    f"有效值: {valid_providers}"
                )

            # Scope 必须是列表
            if not isinstance(model_cfg["scope"], list):
                raise ConfigValidationError(
                    f"模型 '{model_id}' 的 scope 必须是列表"
                )

            # 云端模型必须指定 api_key_env
            if model_cfg["provider"] in {"openai_compatible", "anthropic"}:
                if "api_key_env" not in model_cfg:
                    logger.warning(
                        f"云端模型 '{model_id}' 未指定 api_key_env，调用时可能失败"
                    )

            # 能力契约必须包含 reasoning_depth
            contract = model_cfg["capability_contract"]
            if "reasoning_depth" not in contract:
                raise ConfigValidationError(
                    f"模型 '{model_id}' 的 capability_contract 缺少 'reasoning_depth'"
                )

        logger.debug(f"配置校验通过，共 {len(registry)} 个模型")
