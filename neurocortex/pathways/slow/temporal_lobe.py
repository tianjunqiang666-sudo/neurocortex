"""
NeuroCortex AI — TemporalLobe (颞叶)
========================================
慢速通路：听觉/语言处理、语义标注。
输入听觉/文本 SensoryPacket，输出结构化语义摘要。
阶段一为简化实现（文本关键词提取），阶段二完善。
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger


class TemporalLobe:
    """颞叶 — 听觉/语言处理

    阶段二将集成 Whisper 语音识别和情感识别模型。
    阶段一实现基本的文本关键词提取。
    """

    required_capabilities = {
        "reasoning_depth": "low",
        "output_schema": {"type": "object"},
    }

    def __init__(self) -> None:
        logger.info("TemporalLobe 初始化")

    def process(self, sensory_packet: Any) -> dict[str, Any]:
        """处理听觉/文本输入，输出结构化语义摘要

        Args:
            sensory_packet: SensoryPacket

        Returns:
            结构化描述:
            {"transcript": "...", "keywords": [...], "speaker_emotion": "..."}
        """
        raw_data = sensory_packet.raw_data if hasattr(sensory_packet, 'raw_data') else str(sensory_packet)
        modality = sensory_packet.modality if hasattr(sensory_packet, 'modality') else "text"

        if modality == "text":
            return self._process_text(raw_data)
        else:
            # 阶段二：音频处理占位
            return {
                "transcript": raw_data,
                "keywords": [],
                "speaker_emotion": "unknown",
                "_note": "阶段一占位，阶段二集成语音模型",
            }

    def _process_text(self, text: str) -> dict[str, Any]:
        """文本语义处理（阶段一简化版）"""
        # 简单关键词提取：移除停用词后取高频词
        keywords = self._extract_keywords(text)
        emotion = self._simple_emotion_detect(text)

        return {
            "transcript": text,
            "keywords": keywords,
            "speaker_emotion": emotion,
        }

    def _extract_keywords(self, text: str, max_keywords: int = 5) -> list[str]:
        """简单关键词提取"""
        # 中文分词简化：按标点分割后取较长的片段
        # 英文：按空格分割后过滤短词
        words = re.findall(r'[\u4e00-\u9fff]+|[a-zA-Z]{3,}', text)
        # 去重保序
        seen = set()
        unique = []
        for w in words:
            if w.lower() not in seen:
                seen.add(w.lower())
                unique.append(w)
        return unique[:max_keywords]

    def _simple_emotion_detect(self, text: str) -> str:
        """基于规则的简单情绪检测"""
        text_lower = text.lower()
        emotion_keywords = {
            "angry": ["愤怒", "生气", "投诉", "退款", "angry", "mad", "furious"],
            "happy": ["开心", "满意", "谢谢", "棒", "happy", "great", "thanks"],
            "fearful": ["害怕", "担心", "紧急", "危险", "fear", "worried", "urgent"],
            "sad": ["难过", "失望", "遗憾", "sad", "disappointed"],
        }
        for emotion, keywords in emotion_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return emotion
        return "neutral"
