"""
NeuroCortex AI — KnowledgeBase (知识库)
=========================================
长期知识图谱操作。
使用 NetworkX 维护知识图谱，支持规则的增删改查和 JSON 持久化。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import networkx as nx
from loguru import logger


class KnowledgeBase:
    """长期知识库 — 基于 NetworkX 的知识图谱

    存储从睡眠巩固中蒸馏出的抽象规则、常识和关系。

    Attributes:
        graph: NetworkX 有向图
        storage_path: JSON 持久化路径
    """

    def __init__(self, storage_path: str | Path | None = None) -> None:
        self.storage_path = Path(storage_path) if storage_path else (
            Path(__file__).parent.parent / "data" / "knowledge_graph.json"
        )
        self.graph = nx.DiGraph()
        self._load()

    def _load(self) -> None:
        """从 JSON 文件加载知识图谱"""
        if not self.storage_path.exists():
            logger.info("知识图谱文件不存在，创建空图谱")
            self._save()
            return

        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            for node in data.get("nodes", []):
                self.graph.add_node(node["id"], **{k: v for k, v in node.items() if k != "id"})
            for edge in data.get("edges", []):
                self.graph.add_edge(edge["source"], edge["target"],
                                   **{k: v for k, v in edge.items() if k not in ("source", "target")})

            logger.info(f"知识图谱加载: {self.graph.number_of_nodes()} 节点, "
                        f"{self.graph.number_of_edges()} 条边")
        except Exception as e:
            logger.error(f"知识图谱加载失败: {e}")

    def _save(self) -> None:
        """保存知识图谱到 JSON"""
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

        nodes = []
        for node_id, attrs in self.graph.nodes(data=True):
            nodes.append({"id": node_id, **attrs})

        edges = []
        for src, tgt, attrs in self.graph.edges(data=True):
            edges.append({"source": src, "target": tgt, **attrs})

        data = {
            "nodes": nodes,
            "edges": edges,
            "metadata": {
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "version": "1.0",
                "node_count": len(nodes),
                "edge_count": len(edges),
            }
        }

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.debug(f"知识图谱已保存: {len(nodes)} 节点, {len(edges)} 边")

    def add_rule(self, rule_text: str, source_episode_id: str = "",
                 rule_type: str = "if-then", confidence: float = 0.8) -> str:
        """添加规则节点

        Args:
            rule_text: 规则的自然语言描述
            source_episode_id: 来源情景 ID
            rule_type: 规则类型 (if-then / fact / heuristic)
            confidence: 规则置信度

        Returns:
            规则节点 ID
        """
        import hashlib
        rule_id = f"rule_{hashlib.md5(rule_text.encode()).hexdigest()[:12]}"

        if self.graph.has_node(rule_id):
            # 更新已有规则的置信度
            old_conf = self.graph.nodes[rule_id].get("confidence", 0.5)
            new_conf = min(1.0, old_conf + 0.1)
            self.graph.nodes[rule_id]["confidence"] = new_conf
            self.graph.nodes[rule_id]["reinforcement_count"] = (
                self.graph.nodes[rule_id].get("reinforcement_count", 1) + 1
            )
            logger.info(f"规则强化: {rule_id} 置信度 {old_conf:.2f} → {new_conf:.2f}")
        else:
            self.graph.add_node(rule_id,
                                rule_text=rule_text,
                                rule_type=rule_type,
                                confidence=confidence,
                                source_episode_id=source_episode_id,
                                created_at=datetime.now(timezone.utc).isoformat(),
                                reinforcement_count=1)
            logger.info(f"新规则: {rule_id} = '{rule_text[:50]}...'")

        # 创建与来源情景的关系
        if source_episode_id:
            episode_node = f"episode_{source_episode_id[:12]}"
            if not self.graph.has_node(episode_node):
                self.graph.add_node(episode_node, type="episode")
            self.graph.add_edge(episode_node, rule_id, relation="distilled_from")

        self._save()
        return rule_id

    def query_rules(self, context: str = "", rule_type: str | None = None,
                    min_confidence: float = 0.5) -> list[dict[str, Any]]:
        """查询相关规则

        Args:
            context: 上下文关键词 (简单匹配)
            rule_type: 过滤规则类型
            min_confidence: 最低置信度

        Returns:
            匹配的规则列表
        """
        results = []
        context_lower = context.lower()

        for node_id, attrs in self.graph.nodes(data=True):
            if not node_id.startswith("rule_"):
                continue
            if attrs.get("confidence", 0) < min_confidence:
                continue
            if rule_type and attrs.get("rule_type") != rule_type:
                continue
            rule_text = attrs.get("rule_text", "")
            if context and context_lower not in rule_text.lower():
                # 简单关键词匹配
                context_words = set(context_lower.split())
                rule_words = set(rule_text.lower().split())
                if not context_words & rule_words:
                    continue
            results.append({"id": node_id, **attrs})

        results.sort(key=lambda r: r.get("confidence", 0), reverse=True)
        return results

    def get_all_rules(self) -> list[dict[str, Any]]:
        """获取所有规则"""
        return [
            {"id": nid, **attrs}
            for nid, attrs in self.graph.nodes(data=True)
            if nid.startswith("rule_")
        ]

    def get_status(self) -> dict[str, Any]:
        """获取知识库状态"""
        rules = self.get_all_rules()
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "total_rules": len(rules),
            "storage_path": str(self.storage_path),
        }
