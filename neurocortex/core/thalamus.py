"""
NeuroCortex AI — Thalamus (丘脑)
==================================
多模态网关、注意力导向、路由分发。
接收多模态原始输入，转为统一向量，计算显著性，路由到对应皮层区域。
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False
    logger.warning("sentence-transformers 未安装，将使用简单哈希嵌入代替")


class SensoryPacket:
    """感知数据包 — 丘脑处理后的标准化输出

    Attributes:
        id: 唯一标识符
        timestamp: ISO8601 时间戳
        modality: 模态类型 (text/visual/auditory)
        raw_data: 原始数据
        embedding: 嵌入向量
        salience: 显著性分数 (0~1)
        routing: 目标模块列表
    """

    def __init__(self, modality: str, raw_data: Any, embedding: list[float],
                 salience: float, routing: list[str]) -> None:
        self.id = str(uuid.uuid4())
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.modality = modality
        self.raw_data = raw_data
        self.embedding = embedding
        self.salience = salience
        self.routing = routing

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "timestamp": self.timestamp,
            "modality": self.modality, "raw_data": self.raw_data,
            "embedding": self.embedding[:5] if self.embedding else [],  # 仅展示前5维
            "embedding_dim": len(self.embedding) if self.embedding else 0,
            "salience": self.salience, "routing": self.routing,
        }

    def __repr__(self) -> str:
        return (f"SensoryPacket(id={self.id[:8]}, modality={self.modality}, "
                f"salience={self.salience:.2f}, routing={self.routing})")


class Thalamus:
    """丘脑 — 多模态网关

    职责:
      - 接收文本/图像/音频输入
      - 生成嵌入向量
      - 计算显著性分数
      - 路由到对应皮层区域 + 条件路由到杏仁核

    Attributes:
        embedding_model: 句子嵌入模型
        salience_threshold: 显著性阈值 (高于此值触发杏仁核路由)
        context_keywords: 上下文关键词 (用于显著性计算)
    """

    # 高显著性关键词 (用于初始显著性评估)
    HIGH_SALIENCE_KEYWORDS: set[str] = {
        "紧急", "危险", "错误", "失败", "崩溃", "攻击", "告警",
        "urgent", "danger", "error", "fail", "crash", "attack", "alert",
        "emergency", "critical", "fatal", "死", "杀", "骗", "退款",
    }

    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2",
                 salience_threshold: float = 0.6) -> None:
        self.salience_threshold = salience_threshold
        self._embedding_model: Optional[SentenceTransformer] = None
        self._embedding_model_name = embedding_model_name

        # 延迟加载嵌入模型
        if _HAS_SENTENCE_TRANSFORMERS:
            logger.info(f"Thalamus 将延迟加载嵌入模型: {embedding_model_name}")
        else:
            logger.warning("Thalamus 将使用简易嵌入 (安装 sentence-transformers 以获得更好效果)")

    def _ensure_model(self) -> None:
        """确保嵌入模型已加载（延迟初始化）"""
        if self._embedding_model is None and _HAS_SENTENCE_TRANSFORMERS:
            logger.info(f"正在加载嵌入模型: {self._embedding_model_name} ...")
            self._embedding_model = SentenceTransformer(self._embedding_model_name)
            logger.info("嵌入模型加载完成")

    def process_input(self, data: str, modality: str = "text") -> SensoryPacket:
        """处理输入并生成 SensoryPacket

        Args:
            data: 原始输入数据 (文本字符串)
            modality: 模态类型 ("text", "visual", "auditory")

        Returns:
            标准化的 SensoryPacket
        """
        # 1. 生成嵌入
        embedding = self._generate_embedding(data)

        # 2. 计算显著性
        salience = self._compute_salience(data, modality)

        # 3. 决定路由
        routing = self._determine_routing(modality, salience)

        packet = SensoryPacket(
            modality=modality, raw_data=data,
            embedding=embedding, salience=salience, routing=routing,
        )

        logger.debug(f"Thalamus 生成: {packet}")
        return packet

    def _generate_embedding(self, text: str) -> list[float]:
        """生成文本的嵌入向量"""
        self._ensure_model()

        if self._embedding_model is not None:
            embedding = self._embedding_model.encode(text, normalize_embeddings=True)
            return embedding.tolist()
        else:
            # 简单的哈希嵌入 (fallback)
            import hashlib
            text_bytes = text.encode("utf-8", errors="replace")
            h = hashlib.sha256(text_bytes).digest()
            # 扩展到 384 维以匹配常见嵌入模型维度
            expanded = h * 12  # 32 * 12 = 384 bytes
            return [float(b) / 255.0 for b in expanded[:384]]

    def _compute_salience(self, data: str, modality: str) -> float:
        """计算输入的显著性分数

        基于:
          - 关键词匹配 (高权重)
          - 文本长度 (越长可能越重要)
          - 标点符号密度 (感叹号/问号增加显著性)
        """
        score = 0.3  # 基础分

        data_lower = data.lower()

        # 关键词匹配
        keyword_hits = sum(1 for kw in self.HIGH_SALIENCE_KEYWORDS if kw in data_lower)
        score += min(keyword_hits * 0.15, 0.45)

        # 感叹号/问号
        excl_count = data.count("!") + data.count("！")
        quest_count = data.count("?") + data.count("？")
        score += min((excl_count + quest_count) * 0.05, 0.15)

        # 长度因素 (中等长度稍高)
        length = len(data)
        if 20 < length < 200:
            score += 0.05
        elif length >= 200:
            score += 0.1

        return min(score, 1.0)

    def _determine_routing(self, modality: str, salience: float) -> list[str]:
        """基于模态和显著性决定路由目标"""
        routing = []

        # 按模态路由到对应皮层
        if modality == "visual":
            routing.append("OccipitalLobe")
        elif modality == "auditory":
            routing.append("TemporalLobe")
        elif modality == "text":
            routing.append("TemporalLobe")  # 文本走颞叶处理

        # 高显著性同时路由到杏仁核
        if salience >= self.salience_threshold:
            routing.append("Amygdala")
            logger.info(f"高显著性 ({salience:.2f}) 触发杏仁核路由")

        return routing

    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        self._ensure_model()
        if self._embedding_model is not None:
            return self._embedding_model.get_sentence_embedding_dimension()
        return 128  # fallback

    def batch_process(self, inputs: list[tuple[str, str]]) -> list[SensoryPacket]:
        """批量处理多个输入

        Args:
            inputs: [(data, modality), ...] 列表

        Returns:
            SensoryPacket 列表
        """
        return [self.process_input(data, modality) for data, modality in inputs]
