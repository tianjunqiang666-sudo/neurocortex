"""
NeuroCortex AI — Amygdala (杏仁核)
=====================================
快速通路：即时情感评估、风险标注、应急响应。
接收丘脑高显著包裹，输出结构化威胁等级和情绪标签，向脑干发送分级中断。
"""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger

from neurocortex.core.prefrontal_config import ModelRouter


class EvaluatedPacket:
    """情感评估结果"""

    def __init__(self, threat_level: float, emotion_label: str,
                 source_packet_id: str, raw_response: str = "") -> None:
        self.threat_level = max(0.0, min(1.0, threat_level))
        self.emotion_label = emotion_label
        self.source_packet_id = source_packet_id
        self.raw_response = raw_response

    def to_dict(self) -> dict[str, Any]:
        return {
            "threat_level": self.threat_level,
            "emotion_label": self.emotion_label,
            "source_packet_id": self.source_packet_id,
        }

    def __repr__(self) -> str:
        return f"EvaluatedPacket(threat={self.threat_level:.2f}, emotion={self.emotion_label})"


class Amygdala:
    """杏仁核 — 即时情感评估

    能力需求:
      - reasoning_depth: low (快速响应)
      - output_schema: {"type": "object"} (结构化JSON输出)

    通过 ModelRouter 获取轻量 LLM，使用 prompt 模板要求返回结构化 JSON。
    """

    required_capabilities = {
        "reasoning_depth": "low",
        "output_schema": {"type": "object"},
    }

    EVALUATION_PROMPT = """你是一个情感和威胁评估系统。分析以下输入内容，评估其威胁等级和情绪。

你必须且只能返回一个JSON对象，格式如下：
{
    "threat_level": <0.0到1.0之间的浮点数，0=无威胁，1=极度危险>,
    "emotion_label": "<情绪标签，如：neutral/happy/angry/fearful/sad/surprised/disgusted>"
}

输入内容：
"""

    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    def evaluate(self, sensory_packet: Any) -> EvaluatedPacket:
        """评估感知包裹的威胁等级和情绪

        Args:
            sensory_packet: 来自丘脑的 SensoryPacket

        Returns:
            结构化的 EvaluatedPacket
        """
        raw_data = sensory_packet.raw_data if hasattr(sensory_packet, 'raw_data') else str(sensory_packet)
        packet_id = sensory_packet.id if hasattr(sensory_packet, 'id') else "unknown"

        try:
            client = self.router.get_client(
                "amygdala",
                required_capabilities=self.required_capabilities,
            )
            response = client.chat(
                prompt=f"{self.EVALUATION_PROMPT}{raw_data}",
                system_prompt="你是一个精确的情感分析系统。只返回JSON，不要添加任何解释。",
                temperature=0.1,
                max_tokens=100,
                json_mode=True,
            )

            # 解析 JSON 响应
            result = json.loads(response)
            return EvaluatedPacket(
                threat_level=float(result.get("threat_level", 0.5)),
                emotion_label=str(result.get("emotion_label", "neutral")),
                source_packet_id=packet_id,
                raw_response=response,
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"杏仁核 JSON 解析失败: {e}，使用默认值")
            return EvaluatedPacket(
                threat_level=0.5, emotion_label="uncertain",
                source_packet_id=packet_id,
            )
        except Exception as e:
            logger.error(f"杏仁核评估失败: {e}")
            return EvaluatedPacket(
                threat_level=0.3, emotion_label="error",
                source_packet_id=packet_id,
            )

    def should_alert_brainstem(self, evaluated: EvaluatedPacket) -> Optional[float]:
        """判断是否需要向脑干发送警报

        Returns:
            警报级别 (None 表示不需要警报)
        """
        if evaluated.threat_level >= 0.7:
            return 0.8  # 高危，触发紧急中断
        elif evaluated.threat_level >= 0.4:
            return 0.5  # 中等，触发减速
        return None
