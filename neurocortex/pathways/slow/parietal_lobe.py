"""
NeuroCortex AI — ParietalLobe (顶叶)
========================================
慢速通路：多模态融合、情境预测与误差计算。
接收枕叶/颞叶输出，融合为事件框架，生成情境预测并计算语义预测误差。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

from neurocortex.core.adaptive_params import get_adaptive_manager

try:
    from sentence_transformers import SentenceTransformer, util as st_util
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


class EpisodeTensor:
    """情景张量 — 顶叶融合后的统一事件表示

    Attributes:
        id: 唯一标识
        timestamp: 生成时间
        event_frame: 事件框架 {who, did_what, to_whom, with_tool, context}
        prediction_error: 语义预测误差 (0~1)
        importance_score: 重要性评分
        emotion_label: 情绪标签
        raw_inputs: 原始输入摘要
    """

    def __init__(self, event_frame: dict[str, str], prediction_error: float = 0.0,
                 importance_score: float = 0.5, emotion_label: str = "neutral",
                 raw_inputs: dict[str, Any] | None = None) -> None:
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.event_frame = event_frame
        self.prediction_error = prediction_error
        self.importance_score = importance_score
        self.emotion_label = emotion_label
        self.raw_inputs = raw_inputs or {}

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "timestamp": self.timestamp,
            "event_frame": self.event_frame,
            "prediction_error": self.prediction_error,
            "importance_score": self.importance_score,
            "emotion_label": self.emotion_label,
        }

    def to_text(self) -> str:
        """将事件框架转为自然语言描述"""
        f = self.event_frame
        parts = []
        if f.get("who"):
            parts.append(f["who"])
        if f.get("did_what"):
            parts.append(f["did_what"])
        if f.get("to_whom"):
            parts.append(f"对{f['to_whom']}")
        if f.get("with_tool"):
            parts.append(f"使用{f['with_tool']}")
        if f.get("context"):
            parts.append(f"({f['context']})")
        return " ".join(parts) if parts else str(self.event_frame)

    def __repr__(self) -> str:
        return (f"EpisodeTensor(id={self.id[:8]}, error={self.prediction_error:.2f}, "
                f"importance={self.importance_score:.2f})")


class ParietalLobe:
    """顶叶 — 多模态融合与情境预测

    职责:
      1. 多模态融合：将枕叶/颞叶描述整合为事件框架
      2. 情境预测：基于历史和知识图谱预测下一事件
      3. 预测误差计算：语义匹配度比对
      4. 重要性标记：高误差 → 高重要性

    Attributes:
        error_threshold: 预测误差阈值 (高于此值标记为高重要性)
        _last_prediction: 上一次的预测文本
        _embedding_model: 句子嵌入模型 (用于误差计算)
    """

    required_capabilities = {
        "reasoning_depth": "low",
        "output_schema": {"type": "object"},
    }

    def __init__(self, error_threshold: float = 0.6) -> None:
        self.default_error_threshold = error_threshold
        self.adaptive_mgr = get_adaptive_manager()
        self._last_prediction: str | None = None
        self._embedding_model: Optional[SentenceTransformer] = None

        if _HAS_ST:
            logger.info("ParietalLobe 将使用 sentence-transformers 计算预测误差")

    def _ensure_embedding_model(self) -> None:
        if self._embedding_model is None and _HAS_ST:
            self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def fuse(self, visual_desc: dict[str, Any] | None = None,
             auditory_desc: dict[str, Any] | None = None,
             text_input: str | None = None,
             emotion_label: str = "neutral") -> EpisodeTensor:
        """多模态融合，生成 EpisodeTensor

        Args:
            visual_desc: 枕叶输出的结构化视觉描述
            auditory_desc: 颞叶输出的结构化听觉/语义描述
            text_input: 原始文本 (快捷路径，跳过皮层处理)
            emotion_label: 情绪标签

        Returns:
            融合后的 EpisodeTensor
        """
        # 构建事件框架
        event_frame = self._build_event_frame(visual_desc, auditory_desc, text_input)

        # 计算预测误差
        current_event_text = self._frame_to_text(event_frame)
        prediction_error = self._compute_prediction_error(current_event_text)

        # 将预测误差记录到自适应管理器
        self.adaptive_mgr.record_error(prediction_error)
        
        # 获取动态自适应阈值 (目前仅作打印，后续可用于动态路由控制)
        current_threshold = self.adaptive_mgr.get_error_threshold()
        logger.debug(f"ParietalLobe 预测误差: {prediction_error:.2f} (当前自适应阈值: {current_threshold:.2f})")

        # 重要性评分 (后续也可以考虑引入动态阈值)
        importance = self._compute_importance(prediction_error, emotion_label)

        # 生成新预测 (简单的上下文延续)
        self._last_prediction = self._generate_prediction(event_frame)

        episode = EpisodeTensor(
            event_frame=event_frame,
            prediction_error=prediction_error,
            importance_score=importance,
            emotion_label=emotion_label,
            raw_inputs={"visual": visual_desc, "auditory": auditory_desc, "text": text_input},
        )

        logger.debug(f"ParietalLobe 融合: {episode}")
        return episode

    def _build_event_frame(self, visual: dict | None, auditory: dict | None,
                           text: str | None) -> dict[str, str]:
        """构建统一事件框架"""
        frame: dict[str, str] = {
            "who": "user",
            "did_what": "",
            "to_whom": "system",
            "with_tool": "",
            "context": "",
        }

        if text:
            frame["did_what"] = f"说了: {text}"
            frame["context"] = text

        if auditory:
            transcript = auditory.get("transcript", "")
            keywords = auditory.get("keywords", [])
            if transcript:
                frame["did_what"] = f"说了: {transcript}"
            if keywords:
                frame["with_tool"] = f"关键词: {', '.join(keywords)}"

        if visual:
            objects = visual.get("objects", [])
            scene = visual.get("scene", "")
            if objects:
                frame["context"] += f" 视觉环境: {', '.join(objects)}"
            if scene:
                frame["context"] += f" 场景: {scene}"

        return frame

    def _frame_to_text(self, frame: dict[str, str]) -> str:
        """将事件框架转为文本描述"""
        parts = [v for v in frame.values() if v]
        return " ".join(parts)

    def _compute_prediction_error(self, current_event: str) -> float:
        """计算语义预测误差 = 1 - 语义相似度"""
        if not self._last_prediction:
            return 0.5  # 首次无预测，返回中等误差

        self._ensure_embedding_model()

        if self._embedding_model is not None:
            emb_pred = self._embedding_model.encode(self._last_prediction, convert_to_tensor=True)
            emb_actual = self._embedding_model.encode(current_event, convert_to_tensor=True)
            similarity = float(st_util.cos_sim(emb_pred, emb_actual)[0][0])
            error = 1.0 - max(0.0, min(1.0, similarity))
            logger.debug(f"预测误差: {error:.3f} (预测='{self._last_prediction[:30]}...')")
            return error
        else:
            # Fallback: 简单的字符重叠率
            pred_set = set(self._last_prediction)
            actual_set = set(current_event)
            if not pred_set or not actual_set:
                return 0.5
            overlap = len(pred_set & actual_set) / max(len(pred_set | actual_set), 1)
            return 1.0 - overlap

    def _compute_importance(self, prediction_error: float, emotion_label: str) -> float:
        """综合计算重要性评分"""
        score = 0.3  # 基础分

        # 预测误差贡献
        dynamic_threshold = self.adaptive_mgr.get_error_threshold()
        if prediction_error > dynamic_threshold:
            score += 0.4  # 高误差 → 高重要性
            logger.info(f"高预测误差 ({prediction_error:.2f})，标记为高重要性")
        else:
            score += prediction_error * 0.3

        # 情绪贡献
        emotional_boost = {
            "angry": 0.25, "fearful": 0.25, "surprised": 0.2,
            "sad": 0.15, "happy": 0.05, "neutral": 0.0,
        }
        score += emotional_boost.get(emotion_label, 0.1)

        return min(score, 1.0)

    def _generate_prediction(self, event_frame: dict[str, str]) -> str:
        """生成下一事件的简单预测"""
        did_what = event_frame.get("did_what", "")
        if "天气" in did_what or "weather" in did_what.lower():
            return "用户可能会询问穿衣建议或出行计划"
        elif "退款" in did_what or "refund" in did_what.lower():
            return "用户可能会追问退款进度或表达不满"
        elif "问" in did_what or "?" in did_what or "？" in did_what:
            return "用户可能会追问更多细节或提出新问题"
        else:
            return "用户可能会继续当前话题或提出新话题"
