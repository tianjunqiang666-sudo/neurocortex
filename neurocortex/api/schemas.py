"""
NeuroCortex AI — API Schemas
==============================
Pydantic 请求/响应模型，用于 FastAPI 接口校验。
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


# === 请求模型 ===

class ChatRequest(BaseModel):
    """对话请求"""
    message: str = Field(..., min_length=1, description="用户输入文本")
    stream: bool = Field(False, description="是否启用流式输出")


class SleepRequest(BaseModel):
    """睡眠巩固请求"""
    force: bool = Field(True, description="是否强制触发")


# === 响应模型 ===

class ChatResponse(BaseModel):
    """对话响应"""
    response: str = Field(..., description="AI 回复内容")
    pathway: str = Field("slow", description="触发的通路 (fast/slow)")
    latency_ms: float = Field(0, description="处理耗时 (毫秒)")
    matched_habit: Optional[str] = Field(None, description="匹配的习惯 ID (快速通路)")


class SystemStatus(BaseModel):
    """系统状态"""
    state: str
    cycle_count: int
    total_alerts: int
    hippocampus_memories: int
    knowledge_rules: int
    knowledge_nodes: int
    knowledge_edges: int
    conversation_turns: int
    habits_count: int


class ConsolidationResult(BaseModel):
    """巩固结果"""
    sampled_memories: int
    rules_extracted: int
    memories_pruned: int
    rules_added: list[dict[str, Any]]
    errors: list[str]


class KnowledgeGraphData(BaseModel):
    """知识图谱数据"""
    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]
    total_nodes: int
    total_edges: int


class HealthResponse(BaseModel):
    """健康检查"""
    status: str = "ok"
    version: str = "1.0.0"


class TelemetryData(BaseModel):
    """性能追踪数据"""
    module_stats: dict[str, dict[str, Any]]
    recent_traces: list[dict[str, Any]]
