"""
NeuroCortex AI — Association Engine (记忆联想)
==============================================
在系统空闲或睡眠期间，自动分析知识图谱中的孤立节点或潜在关联，
通过 LLM 推理建立新的逻辑边，实现“知识的自我成长”。
"""

import json
import random
from typing import Any, List, Dict
from loguru import logger

from neurocortex.core.prefrontal_config import ModelRouter
from neurocortex.memory.knowledge_base import KnowledgeBase


class AssociationEngine:
    """记忆联想引擎"""

    def __init__(self, router: ModelRouter, knowledge_base: KnowledgeBase):
        self.router = router
        self.kb = knowledge_base

    async def run_association_cycle(self, max_new_edges: int = 3):
        """运行一次联想周期"""
        logger.info("▶ AssociationEngine 开始记忆联想周期...")
        
        # 1. 寻找候选节点对
        # 策略：随机选取一些最近活跃的节点，或寻找没有边相连的节点
        nodes = list(self.kb.graph.nodes(data=True))
        if len(nodes) < 2:
            return
            
        # 选取最近添加的或随机的 10 个节点作为候选
        candidates = random.sample(nodes, min(10, len(nodes)))
        
        new_edges_count = 0
        client = self.router.get_client("frontal")
        
        # 尝试寻找两两之间的关系
        for i in range(len(candidates)):
            if new_edges_count >= max_new_edges:
                break
                
            for j in range(i + 1, len(candidates)):
                node_a_id, node_a_attr = candidates[i]
                node_b_id, node_b_attr = candidates[j]
                
                # 如果已经有直接边了，跳过
                if self.kb.graph.has_edge(node_a_id, node_b_id):
                    continue
                
                # 2. 询问 LLM 是否存在关联
                prompt = (
                    f"作为 NeuroCortex AI 的联想中枢，请分析以下两个知识节点是否存在潜在关联：\n\n"
                    f"【节点 A】: {node_a_id} ({node_a_attr})\n"
                    f"【节点 B】: {node_b_id} ({node_b_attr})\n\n"
                    "如果它们之间存在某种逻辑、因果、隶属或属性关联，请描述这种关联并给出一个关系类型。\n"
                    "请以 JSON 格式输出:\n"
                    "{\n"
                    "  \"associated\": true/false,\n"
                    "  \"relation_type\": \"关系名称\",\n"
                    "  \"reason\": \"关联理由\"\n"
                    "}"
                )
                
                try:
                    import asyncio
                    res_text = await asyncio.to_thread(
                        client.chat, prompt=prompt, temperature=0.1, json_mode=True
                    )
                    
                    data = json.loads(res_text)
                    if data.get("associated", False):
                        rel_type = data.get("relation_type", "associated_with")
                        reason = data.get("reason", "")
                        
                        logger.info(f"🔗 发现新联想: {node_a_id} --({rel_type})--> {node_b_id} | 理由: {reason}")
                        self.kb.add_relation(node_a_id, node_b_id, rel_type, inference_reason=reason)
                        new_edges_count += 1
                        
                        if new_edges_count >= max_new_edges:
                            break
                            
                except Exception as e:
                    logger.error(f"联想推理异常: {e}")
                    continue
                    
        logger.info(f"✅ 联想周期结束，新增了 {new_edges_count} 条关联边。")

