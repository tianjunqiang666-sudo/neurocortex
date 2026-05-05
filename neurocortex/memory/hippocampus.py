"""
NeuroCortex AI — Hippocampus (海马体)
=======================================
情景记忆快速编码、模式分离/完成、记忆重播与抽象提取。
使用 ChromaDB 实现向量存储与语义检索。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import chromadb
from loguru import logger


class Hippocampus:
    """海马体 — 情景记忆存储与检索

    职责:
      - 快速编码 EpisodeTensor (含事件框架、情感、预测误差)
      - 基于语义相似性检索 top-K 记忆
      - 根据重要性评分筛选用于巩固的记忆
      - 修剪低重要性过期记忆

    Attributes:
        collection: ChromaDB 集合
        episode_count: 已编码的情景数量
    """

    def __init__(self, storage_path: str | Path | None = None,
                 collection_name: str = "episodic_memory") -> None:
        storage = Path(storage_path) if storage_path else (
            Path(__file__).parent.parent / "data" / "chromadb_storage"
        )
        storage.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(path=str(storage))
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.episode_count = self.collection.count()
        logger.info(f"Hippocampus 初始化: {self.episode_count} 条记忆 (存储: {storage})")

    def encode(self, episode: Any, embedding: list[float] | None = None) -> str:
        """快速编码情景到海马体

        Args:
            episode: EpisodeTensor 或字典
            embedding: 预计算的嵌入向量 (可选)

        Returns:
            存储的记忆 ID
        """
        # 解析 episode
        if hasattr(episode, 'to_dict'):
            ep_dict = episode.to_dict()
            ep_id = episode.id
            ep_text = episode.to_text() if hasattr(episode, 'to_text') else str(ep_dict)
        elif isinstance(episode, dict):
            ep_dict = episode
            ep_id = episode.get("id", str(uuid.uuid4()))
            ep_text = json.dumps(episode, ensure_ascii=False)
        else:
            ep_dict = {"raw": str(episode)}
            ep_id = str(uuid.uuid4())
            ep_text = str(episode)
            
        ep_text = ep_text.encode('utf-8', 'surrogatepass').decode('utf-8', 'replace')

        # 计算重要性
        importance = ep_dict.get("importance_score", 0.5)
        prediction_error = ep_dict.get("prediction_error", 0.0)
        emotion = ep_dict.get("emotion_label", "neutral")

        # 元数据
        metadata = {
            "importance_score": float(importance),
            "prediction_error": float(prediction_error),
            "emotion_label": str(emotion),
            "timestamp": str(ep_dict.get("timestamp", datetime.now(timezone.utc).isoformat())),
            "event_frame": json.dumps(ep_dict.get("event_frame", {}), ensure_ascii=False).encode('utf-8', 'surrogatepass').decode('utf-8', 'replace'),
        }

        # 存储到 ChromaDB
        add_kwargs: dict[str, Any] = {
            "ids": [ep_id],
            "documents": [ep_text],
            "metadatas": [metadata],
        }
        if embedding:
            add_kwargs["embeddings"] = [embedding]

        try:
            self.collection.add(**add_kwargs)
        except Exception as e:
            logger.error(f"ChromaDB add error. metadata={metadata}, types={{k: type(v) for k, v in metadata.items()}}")
            raise

        self.episode_count += 1

        logger.debug(
            f"Hippocampus 编码: id={ep_id[:8]}, importance={importance:.2f}, "
            f"emotion={emotion}, total={self.episode_count}"
        )
        return ep_id

    def retrieve(self, query: str | list[float], k: int = 5) -> list[dict[str, Any]]:
        """语义检索 top-K 最相似记忆

        Args:
            query: 查询文本或嵌入向量
            k: 返回数量

        Returns:
            记忆列表，每项包含 id, document, metadata, distance
        """
        if self.episode_count == 0:
            logger.debug("Hippocampus 为空，无可检索记忆")
            return []

        k = min(k, self.episode_count)

        try:
            if isinstance(query, str):
                results = self.collection.query(query_texts=[query], n_results=k)
            else:
                # 转换 embedding 以防其为 numpy 数组类型导致错误
                embedding_list = [float(x) for x in query]
                results = self.collection.query(query_embeddings=[embedding_list], n_results=k)

            memories = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]

            for i in range(len(ids)):
                memories.append({
                    "id": ids[i],
                    "document": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dists[i] if i < len(dists) else 0.0,
                })

            logger.debug(f"Hippocampus 检索: query='{query[:30] if isinstance(query, str) else 'embedding'}', 返回 {len(memories)} 条")
            return memories

        except Exception as e:
            logger.error(f"Hippocampus 检索失败: {e}")
            return []

    def get_important_memories(self, threshold: float = 0.8) -> list[dict[str, Any]]:
        """获取高重要性记忆（用于睡眠巩固）

        Args:
            threshold: 重要性阈值

        Returns:
            重要性 > threshold 的记忆列表
        """
        if self.episode_count == 0:
            return []

        try:
            results = self.collection.get(
                where={"importance_score": {"$gte": threshold}},
                include=["documents", "metadatas"],
            )

            memories = []
            for i in range(len(results["ids"])):
                memories.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i] if results["documents"] else "",
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })

            logger.info(f"获取到 {len(memories)} 条高重要性记忆 (阈值={threshold})")
            return memories

        except Exception as e:
            logger.error(f"获取重要记忆失败: {e}")
            return []

    def get_all_memories(self, limit: int = 100) -> list[dict[str, Any]]:
        """获取所有记忆（调试用）"""
        if self.episode_count == 0:
            return []
        try:
            results = self.collection.get(
                limit=min(limit, self.episode_count),
                include=["documents", "metadatas"],
            )
            memories = []
            for i in range(len(results["ids"])):
                memories.append({
                    "id": results["ids"][i],
                    "document": results["documents"][i],
                    "metadata": results["metadatas"][i],
                })
            return memories
        except Exception as e:
            logger.error(f"获取记忆失败: {e}")
            return []

    def prune(self, importance_threshold: float = 0.3) -> int:
        """修剪低重要性记忆

        Args:
            importance_threshold: 低于此值的记忆将被删除

        Returns:
            删除的记忆数量
        """
        if self.episode_count == 0:
            return 0

        try:
            results = self.collection.get(
                where={"importance_score": {"$lt": importance_threshold}},
            )

            ids_to_delete = results["ids"]
            if ids_to_delete:
                self.collection.delete(ids=ids_to_delete)
                self.episode_count -= len(ids_to_delete)
                logger.info(f"修剪 {len(ids_to_delete)} 条低重要性记忆")

            return len(ids_to_delete)

        except Exception as e:
            logger.error(f"记忆修剪失败: {e}")
            return 0

    def get_status(self) -> dict[str, Any]:
        """获取海马体状态"""
        return {
            "total_memories": self.episode_count,
            "collection_name": self.collection.name,
        }
