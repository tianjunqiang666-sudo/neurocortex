"""
Tests — KnowledgeBase (知识库)
================================
测试图谱节点/边操作、多跳查询、规则降权。
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestKnowledgeBaseRules:
    """规则 CRUD 测试"""

    def test_add_rule(self, knowledge_base):
        rule_id = knowledge_base.add_rule("IF rain THEN umbrella", rule_type="if-then")
        assert rule_id.startswith("rule_")
        assert knowledge_base.graph.has_node(rule_id)

    def test_duplicate_rule_reinforces(self, knowledge_base):
        """相同规则应被强化而非重复添加"""
        r1 = knowledge_base.add_rule("IF hot THEN AC", confidence=0.7)
        r2 = knowledge_base.add_rule("IF hot THEN AC", confidence=0.7)
        assert r1 == r2
        assert math.isclose(knowledge_base.graph.nodes[r1]["confidence"], 0.8)

    def test_query_rules(self, knowledge_base):
        knowledge_base.add_rule("IF user angry THEN calm down")
        results = knowledge_base.query_rules("angry")
        assert len(results) >= 1

    def test_query_with_confidence_filter(self, knowledge_base):
        knowledge_base.add_rule("low conf rule", confidence=0.3)
        results = knowledge_base.query_rules(min_confidence=0.5)
        assert len(results) == 0


class TestKnowledgeBaseEntities:
    """实体与关系测试"""

    def test_add_entity(self, knowledge_base):
        eid = knowledge_base.add_entity("Python", entity_type="Language")
        assert knowledge_base.graph.has_node(eid)
        assert knowledge_base.graph.nodes[eid]["type"] == "entity"

    def test_add_relation(self, knowledge_base):
        e1 = knowledge_base.add_entity("Cat")
        e2 = knowledge_base.add_entity("Animal")
        knowledge_base.add_relation(e1, e2, "is_a")
        assert knowledge_base.graph.has_edge(e1, e2)

    def test_multi_hop_query(self, knowledge_base):
        """多跳查询测试"""
        e1 = knowledge_base.add_entity("A")
        r1 = knowledge_base.add_rule("Rule connected to A", confidence=0.9)
        knowledge_base.add_relation(e1, r1, "produces")

        related = knowledge_base.query_related_rules(e1, depth=2)
        assert len(related) >= 1
        assert related[0]["id"] == r1


class TestKnowledgeBaseDeprecation:
    """规则降权测试"""

    def test_deprecate_rule(self, knowledge_base):
        rid = knowledge_base.add_rule("wrong rule", confidence=0.8)
        success = knowledge_base.deprecate_rule(rid, penalty=0.5)
        assert success
        assert math.isclose(knowledge_base.graph.nodes[rid]["confidence"], 0.3)

    def test_deprecate_below_threshold(self, knowledge_base):
        rid = knowledge_base.add_rule("very wrong", confidence=0.1)
        knowledge_base.deprecate_rule(rid, penalty=0.5)
        assert knowledge_base.graph.nodes[rid]["confidence"] == 0.0
        assert knowledge_base.graph.nodes[rid]["is_deprecated"] is True

    def test_deprecate_nonexistent(self, knowledge_base):
        result = knowledge_base.deprecate_rule("rule_nonexistent")
        assert result is False


class TestKnowledgeBasePersistence:
    """持久化测试"""

    def test_save_and_reload(self, knowledge_base, tmp_data_dir):
        knowledge_base.add_rule("persistent rule")
        knowledge_base.add_entity("TestEntity")

        from neurocortex.memory.knowledge_base import KnowledgeBase
        kb2 = KnowledgeBase(storage_path=tmp_data_dir / "test_kg.json")
        assert kb2.graph.number_of_nodes() == knowledge_base.graph.number_of_nodes()
