"""
NeuroCortex AI — FastAPI 服务
================================
提供 RESTful + WebSocket 接口，供外部系统和未来的桌面应用接入。

启动方式:
  uvicorn neurocortex.api.server:app --reload --port 8000
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Any
import asyncio
import json

# 确保项目路径
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT.parent))

# === Windows UTF-8 兼容 ===
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from neurocortex.api.schemas import (
    ChatRequest, ChatResponse, SystemStatus, SleepRequest,
    ConsolidationResult, KnowledgeGraphData, HealthResponse, TelemetryData,
)
from neurocortex.core.telemetry import get_telemetry, trace_latency
from neurocortex.core.event_bus import get_event_bus, Event, EventType
from neurocortex.core.prefrontal_config import ModelRouter
from neurocortex.core.brainstem import Brainstem
from neurocortex.core.thalamus import Thalamus
from neurocortex.pathways.fast.amygdala import Amygdala
from neurocortex.pathways.fast.basal_ganglia import BasalGanglia
from neurocortex.pathways.slow.occipital_lobe import OccipitalLobe
from neurocortex.pathways.slow.temporal_lobe import TemporalLobe
from neurocortex.pathways.slow.parietal_lobe import ParietalLobe
from neurocortex.pathways.slow.frontal_lobe import FrontalLobe
from neurocortex.memory.hippocampus import Hippocampus
from neurocortex.memory.consolidation import MemoryConsolidation
from neurocortex.memory.knowledge_base import KnowledgeBase


# === 全局模块容器 ===
_modules: dict[str, Any] = {}


def _init_modules() -> dict[str, Any]:
    """初始化所有脑区模块 (与 main.py 逻辑一致)"""
    config_path = _PROJECT_ROOT / "config" / "model_config.yaml"
    router = ModelRouter(str(config_path))
    system_cfg = router.config_loader.get_system_settings()

    brainstem = Brainstem(
        wake_duration_seconds=system_cfg.get("wake_duration_minutes", 60) * 60,
        sleep_duration_seconds=system_cfg.get("sleep_duration_minutes", 10) * 60,
    )
    thalamus = Thalamus(embedding_model_name=system_cfg.get("embedding_model", "all-MiniLM-L6-v2"))
    amygdala = Amygdala(router)
    basal_ganglia = BasalGanglia()
    occipital_lobe = OccipitalLobe(router)
    temporal_lobe = TemporalLobe(router)
    parietal_lobe = ParietalLobe(error_threshold=system_cfg.get("prediction_error_threshold", 0.6))
    frontal_lobe = FrontalLobe(router)
    hippocampus = Hippocampus()
    knowledge_base = KnowledgeBase()
    consolidation = MemoryConsolidation(
        router=router, hippocampus=hippocampus,
        knowledge_base=knowledge_base, basal_ganglia=basal_ganglia,
        importance_threshold=system_cfg.get("memory_importance_threshold", 0.8),
    )
    brainstem.on_sleep(lambda: consolidation.run_consolidation_cycle())

    return {
        "router": router, "brainstem": brainstem, "thalamus": thalamus,
        "amygdala": amygdala, "basal_ganglia": basal_ganglia,
        "occipital_lobe": occipital_lobe, "temporal_lobe": temporal_lobe,
        "parietal_lobe": parietal_lobe, "frontal_lobe": frontal_lobe,
        "hippocampus": hippocampus, "knowledge_base": knowledge_base,
        "consolidation": consolidation,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    global _modules
    logger.info("🧠 NeuroCortex AI API 正在初始化...")
    _modules = _init_modules()
    
    # 启动事件总线
    event_bus = get_event_bus()
    await event_bus.start()
    
    logger.info("✅ API 就绪")
    yield
    logger.info("🛑 API 关闭")


# === FastAPI 应用 ===
app = FastAPI(
    title="NeuroCortex AI API",
    description="基于大脑解剖学架构的类脑智能系统 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === 端点 ===

@app.get("/visualizer", response_class=HTMLResponse)
async def get_visualizer():
    """返回神经通路可视化页面"""
    html_path = Path(__file__).parent.parent / "ui" / "visualizer.html"
    if not html_path.exists():
        return "<h1>Visualizer HTML not found</h1>"
    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """健康检查"""
    return HealthResponse()


@app.post("/chat", response_model=ChatResponse)
@trace_latency("API")
async def chat(req: ChatRequest):
    """同步对话接口"""
    import asyncio
    start = time.perf_counter()

    text = req.message.strip()
    thalamus = _modules["thalamus"]
    basal_ganglia = _modules["basal_ganglia"]
    frontal_lobe = _modules["frontal_lobe"]

    # 1. Thalamus
    packet = thalamus.process_input(text)

    # 2. BasalGanglia 快速通路
    habit_result = basal_ganglia.lookup(packet)
    if habit_result:
        elapsed = (time.perf_counter() - start) * 1000
        return ChatResponse(
            response=habit_result["response"],
            pathway="fast",
            latency_ms=round(elapsed, 2),
            matched_habit=habit_result.get("habit_id"),
        )

    # 3. 负反馈检测
    feedback_score = frontal_lobe.analyze_feedback(text)
    if feedback_score > 0.5:
        resp = frontal_lobe.trigger_correction(
            _modules["hippocampus"], _modules["knowledge_base"], basal_ganglia
        )
        elapsed = (time.perf_counter() - start) * 1000
        return ChatResponse(response=resp, pathway="correction", latency_ms=round(elapsed, 2))

    # 4. 慢速通路
    temporal_lobe = _modules["temporal_lobe"]
    parietal_lobe = _modules["parietal_lobe"]
    hippocampus = _modules["hippocampus"]
    knowledge_base = _modules["knowledge_base"]

    auditory_desc = temporal_lobe.process(packet)
    emotion_label = auditory_desc.get("speaker_emotion", "neutral") if auditory_desc else "neutral"

    episode = parietal_lobe.fuse(
        visual_desc=None, auditory_desc=auditory_desc,
        text_input=text, emotion_label=emotion_label,
    )
    hippocampus.encode(episode, embedding=packet.embedding)
    memories = hippocampus.retrieve(packet.embedding, k=3)
    rules = knowledge_base.query_rules(text)
    rule_texts = [r.get("rule_text", "") for r in rules[:3]]

    response = await asyncio.to_thread(
        frontal_lobe.generate_response,
        user_input=text, episode=episode,
        memories=memories,
        knowledge_rules=rule_texts if rule_texts else None,
    )

    elapsed = (time.perf_counter() - start) * 1000
    return ChatResponse(response=response, pathway="slow", latency_ms=round(elapsed, 2))


@app.get("/status", response_model=SystemStatus)
async def get_status():
    """获取系统状态"""
    bs = _modules["brainstem"].get_status()
    hp = _modules["hippocampus"].get_status()
    kb = _modules["knowledge_base"].get_status()
    fl = _modules["frontal_lobe"].get_status()
    bg = _modules["basal_ganglia"]

    return SystemStatus(
        state=bs["state"],
        cycle_count=bs["cycle_count"],
        total_alerts=bs["total_alerts"],
        hippocampus_memories=hp["total_memories"],
        knowledge_rules=kb["total_rules"],
        knowledge_nodes=kb["total_nodes"],
        knowledge_edges=kb["total_edges"],
        conversation_turns=fl["conversation_turns"],
        habits_count=len(bg.habits),
    )


@app.post("/sleep", response_model=ConsolidationResult)
async def trigger_sleep(req: SleepRequest = SleepRequest()):
    """触发睡眠巩固"""
    import asyncio
    _modules["brainstem"].trigger_sleep()
    result = await _modules["consolidation"].run_consolidation_cycle()
    return ConsolidationResult(**result)


@app.get("/knowledge", response_model=KnowledgeGraphData)
async def get_knowledge():
    """获取知识图谱"""
    kb = _modules["knowledge_base"]
    nodes = [{"id": nid, **attrs} for nid, attrs in kb.graph.nodes(data=True)]
    edges = [{"source": s, "target": t, **attrs} for s, t, attrs in kb.graph.edges(data=True)]
    return KnowledgeGraphData(
        nodes=nodes, edges=edges,
        total_nodes=len(nodes), total_edges=len(edges),
    )


@app.get("/telemetry", response_model=TelemetryData)
async def get_telemetry_data():
    """获取性能追踪数据"""
    tel = get_telemetry()
    return TelemetryData(
        module_stats=tel.get_module_stats(),
        recent_traces=tel.get_recent(30),
    )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """流式对话 WebSocket 接口"""
    await websocket.accept()
    
    thalamus = _modules["thalamus"]
    basal_ganglia = _modules["basal_ganglia"]
    frontal_lobe = _modules["frontal_lobe"]
    temporal_lobe = _modules["temporal_lobe"]
    parietal_lobe = _modules["parietal_lobe"]
    hippocampus = _modules["hippocampus"]
    knowledge_base = _modules["knowledge_base"]
    event_bus = get_event_bus()

    # 定义订阅回调
    async def event_callback(event: Event):
        try:
            await websocket.send_json({
                "type": "event",
                "event_type": event.type.value,
                "source": event.source,
                "payload": event.payload
            })
        except:
            pass

    # 订阅所有事件
    for et in EventType:
        event_bus.subscribe(et, event_callback)
    
    try:
        while True:
            text = await websocket.receive_text()
            text = text.strip()
            
            event_bus.publish(Event(EventType.INPUT_RECEIVED, "Thalamus", {"text": text}))
            packet = thalamus.process_input(text)
            event_bus.publish(Event(EventType.THALAMUS_ROUTED, "Thalamus", {"routing": packet.routing}))
            
            habit_result = basal_ganglia.lookup(packet)
            if habit_result:
                event_bus.publish(Event(EventType.HABIT_MATCHED, "BasalGanglia", {"pattern": habit_result["pattern"]}))
                await websocket.send_json({"type": "chunk", "content": habit_result["response"]})
                await websocket.send_json({"type": "done", "pathway": "fast"})
                continue
                
            feedback_score = frontal_lobe.analyze_feedback(text)
            if feedback_score > 0.5:
                event_bus.publish(Event(EventType.FEEDBACK_RECEIVED, "FrontalLobe", {"score": feedback_score}))
                resp = frontal_lobe.trigger_correction(hippocampus, knowledge_base, basal_ganglia)
                await websocket.send_json({"type": "chunk", "content": resp})
                await websocket.send_json({"type": "done", "pathway": "correction"})
                continue
                
            auditory_desc = temporal_lobe.process(packet)
            emotion_label = auditory_desc.get("speaker_emotion", "neutral") if auditory_desc else "neutral"

            episode = parietal_lobe.fuse(
                visual_desc=None, auditory_desc=auditory_desc,
                text_input=text, emotion_label=emotion_label,
            )
            event_bus.publish(Event(EventType.EPISODE_CREATED, "ParietalLobe", {"episode_id": episode.id}))
            
            hippocampus.encode(episode, embedding=packet.embedding)
            event_bus.publish(Event(EventType.MEMORY_ENCODED, "Hippocampus", {"episode_id": episode.id}))
            
            memories = hippocampus.retrieve(packet.embedding, k=3)
            rules = knowledge_base.query_rules(text)
            rule_texts = [r.get("rule_text", "") for r in rules[:3]]
            
            async for chunk in frontal_lobe.generate_response_stream(
                user_input=text, episode=episode,
                memories=memories, knowledge_rules=rule_texts if rule_texts else None
            ):
                await websocket.send_json({"type": "chunk", "content": chunk})
                
            await websocket.send_json({"type": "done", "pathway": "slow"})
            
    except WebSocketDisconnect:
        logger.info("WebSocket 客户端已断开")
    except Exception as e:
        logger.error(f"WebSocket 异常: {e}")
        try:
            await websocket.send_json({"type": "error", "content": str(e)})
        except:
            pass
    finally:
        # 取消所有订阅
        for et in EventType:
            event_bus.unsubscribe(et, event_callback)
