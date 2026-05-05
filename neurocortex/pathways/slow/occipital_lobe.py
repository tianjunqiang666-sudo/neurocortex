"""
NeuroCortex AI — OccipitalLobe (枕叶)
========================================
慢速通路：视觉特征提取。
输入视觉 SensoryPacket，输出结构化视觉摘要。
阶段一为骨架实现，阶段二完善。
"""

from __future__ import annotations

from typing import Any

from loguru import logger


class OccipitalLobe:
    """枕叶 — 视觉特征提取

    阶段二将集成 ViT/CLIP 模型进行真正的视觉理解。
    当前为占位实现，返回模拟输出。
    """

    required_capabilities = {
        "reasoning_depth": "low",
        "output_schema": {"type": "object"},
    }

    def __init__(self) -> None:
        logger.info("OccipitalLobe 初始化 (骨架模式)")

    def process(self, visual_packet: Any) -> dict[str, Any]:
        """处理视觉输入，输出结构化视觉摘要

        Args:
            visual_packet: SensoryPacket (modality="visual")

        Returns:
            结构化视觉描述，格式:
            {"objects": [...], "scene": "...", "text": "..."}
        """
        raw_data = visual_packet.raw_data if hasattr(visual_packet, 'raw_data') else str(visual_packet)

        # 阶段一：占位实现
        logger.debug(f"OccipitalLobe 处理视觉输入 (占位): {raw_data[:50]}")
        return {
            "objects": [],
            "scene": "unknown",
            "text": raw_data if isinstance(raw_data, str) else "",
            "confidence": 0.0,
            "_note": "阶段一占位输出，阶段二将集成视觉模型",
        }
