"""
NeuroCortex AI — 系统启动入口
================================
演示完整 MVP 闭环：
  输入文本 → Thalamus 处理 → FrontalLobe 检索海马体 → LLM 生成回答
  /sleep  → 触发离线巩固
  /status → 查看系统状态
  /memories → 查看存储的记忆
  /rules  → 查看知识图谱规则
  /quit   → 退出
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# === Windows 控制台 UTF-8 兼容 ===
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from loguru import logger

# 配置 loguru
logger.remove()
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
)

# 确保项目路径
_PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(_PROJECT_ROOT.parent))


def print_banner() -> None:
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ███╗   ██╗███████╗██╗   ██╗██████╗  ██████╗              ║
║     ████╗  ██║██╔════╝██║   ██║██╔══██╗██╔═══██╗             ║
║     ██╔██╗ ██║█████╗  ██║   ██║██████╔╝██║   ██║             ║
║     ██║╚██╗██║██╔══╝  ██║   ██║██╔══██╗██║   ██║             ║
║     ██║ ╚████║███████╗╚██████╔╝██║  ██║╚██████╔╝             ║
║     ╚═╝  ╚═══╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝             ║
║                                                              ║
║          C O R T E X    A I    v0.1.0                        ║
║     基于大脑解剖学架构的类脑智能系统                         ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def initialize_system():
    """初始化所有模块"""
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

    config_path = _PROJECT_ROOT / "config" / "model_config.yaml"
    logger.info("正在初始化 NeuroCortex AI 系统...")

    # 核心模块
    router = ModelRouter(str(config_path))
    system_cfg = router.config_loader.get_system_settings()

    brainstem = Brainstem(
        wake_duration_seconds=system_cfg.get("wake_duration_minutes", 60) * 60,
        sleep_duration_seconds=system_cfg.get("sleep_duration_minutes", 10) * 60,
    )

    thalamus = Thalamus(
        embedding_model_name=system_cfg.get("embedding_model", "all-MiniLM-L6-v2"),
    )

    # 快速通路
    amygdala = Amygdala(router)
    basal_ganglia = BasalGanglia()

    # 慢速通路
    occipital_lobe = OccipitalLobe(router)
    temporal_lobe = TemporalLobe(router)
    parietal_lobe = ParietalLobe(
        error_threshold=system_cfg.get("prediction_error_threshold", 0.6),
    )
    frontal_lobe = FrontalLobe(router)

    # 记忆系统
    hippocampus = Hippocampus()
    knowledge_base = KnowledgeBase()
    consolidation = MemoryConsolidation(
        router=router,
        hippocampus=hippocampus,
        knowledge_base=knowledge_base,
        basal_ganglia=basal_ganglia,
        importance_threshold=system_cfg.get("memory_importance_threshold", 0.8),
    )

    # 注册 Brainstem 回调
    brainstem.on_sleep(lambda: consolidation.run_consolidation_cycle())

    # 注册中断回调
    def on_emergency(alert_event):
        logger.warning("紧急中断！保存额叶工作记忆快照...")
        frontal_lobe.rollback_state()

    brainstem.on_interrupt(on_emergency)

    logger.info("✅ 所有模块初始化完成")

    return {
        "router": router,
        "brainstem": brainstem,
        "thalamus": thalamus,
        "amygdala": amygdala,
        "basal_ganglia": basal_ganglia,
        "occipital_lobe": occipital_lobe,
        "temporal_lobe": temporal_lobe,
        "parietal_lobe": parietal_lobe,
        "frontal_lobe": frontal_lobe,
        "hippocampus": hippocampus,
        "knowledge_base": knowledge_base,
        "consolidation": consolidation,
    }


def process_input(text: str, modules: dict) -> str:
    """处理用户输入的完整流水线

    流程: Thalamus → TemporalLobe → ParietalLobe → Hippocampus检索 → FrontalLobe

    Args:
        text: 用户输入
        modules: 模块字典

    Returns:
        系统回复
    """
    thalamus = modules["thalamus"]
    occipital_lobe = modules["occipital_lobe"]
    temporal_lobe = modules["temporal_lobe"]
    parietal_lobe = modules["parietal_lobe"]
    frontal_lobe = modules["frontal_lobe"]
    hippocampus = modules["hippocampus"]
    knowledge_base = modules["knowledge_base"]
    amygdala = modules["amygdala"]
    basal_ganglia = modules["basal_ganglia"]
    brainstem = modules["brainstem"]

    import os
    modality = "text"
    if os.path.isfile(text):
        ext = text.lower().split('.')[-1]
        if ext in ['png', 'jpg', 'jpeg', 'webp', 'gif']:
            modality = "visual"
        elif ext in ['mp3', 'wav', 'ogg', 'm4a', 'flac']:
            modality = "auditory"

    # 1. Thalamus: 生成 SensoryPacket
    logger.info(f"▶ Thalamus 处理输入... (模态: {modality})")
    packet = thalamus.process_input(text, modality=modality)

    # 2. 快速通路检查 (系统 1)
    
    # 2.1 BasalGanglia: 习惯匹配
    habit_response = basal_ganglia.lookup(packet)
    if habit_response:
        return habit_response

    # 2.2 Amygdala: 若高显著性，经过杏仁核
    if "Amygdala" in packet.routing:
        logger.info("▶ Amygdala 评估威胁...")
        evaluated = amygdala.evaluate(packet)
        alert_level = amygdala.should_alert_brainstem(evaluated)
        if alert_level:
            brainstem.alert(alert_level, "Amygdala",
                           f"威胁等级={evaluated.threat_level:.2f}, 情绪={evaluated.emotion_label}")

    # 3. 慢速通路: 语义处理
    visual_desc = None
    auditory_desc = None
    emotion_label = "neutral"

    if modality == "visual":
        logger.info("▶ OccipitalLobe 视觉处理...")
        visual_desc = occipital_lobe.process(packet)
    elif modality == "auditory":
        logger.info("▶ TemporalLobe 听觉处理...")
        auditory_desc = temporal_lobe.process(packet)
        emotion_label = auditory_desc.get("speaker_emotion", "neutral")
    else:
        logger.info("▶ TemporalLobe 语义处理...")
        auditory_desc = temporal_lobe.process(packet)
        emotion_label = auditory_desc.get("speaker_emotion", "neutral")

    # 4. ParietalLobe: 融合为 EpisodeTensor
    logger.info("▶ ParietalLobe 多模态融合...")
    episode = parietal_lobe.fuse(
        visual_desc=visual_desc,
        auditory_desc=auditory_desc,
        text_input=text if modality == "text" else f"[{modality} file input]",
        emotion_label=emotion_label,
    )

    # 5. Hippocampus: 编码当前情景 & 检索相关记忆
    logger.info("▶ Hippocampus 编码与检索...")
    hippocampus.encode(episode, embedding=packet.embedding)
    memories = hippocampus.retrieve(packet.embedding, k=3)

    # 6. KnowledgeBase: 查询相关规则
    rules = knowledge_base.query_rules(text)
    rule_texts = [r.get("rule_text", "") for r in rules[:3]]

    # 7. FrontalLobe: 生成回复
    logger.info("▶ FrontalLobe 推理生成回复...")
    response = frontal_lobe.generate_response(
        user_input=text,
        episode=episode,
        memories=memories,
        knowledge_rules=rule_texts if rule_texts else None,
    )

    return response


def handle_command(command: str, modules: dict) -> str | None:
    """处理系统命令

    Returns:
        命令输出，或 None 表示退出
    """
    cmd = command.strip().lower()

    if cmd == "/quit" or cmd == "/exit":
        return None

    elif cmd == "/sleep":
        logger.info("手动触发睡眠巩固...")
        modules["brainstem"].trigger_sleep()
        result = modules["consolidation"].run_consolidation_cycle()
        return (
            f"\n💤 睡眠巩固完成:\n"
            f"  采样记忆: {result['sampled_memories']}\n"
            f"  提取规则: {result['rules_extracted']}\n"
            f"  修剪记忆: {result['memories_pruned']}\n"
            + (f"  新规则: {[r['rule'][:60] for r in result['rules_added']]}\n" if result['rules_added'] else "")
            + (f"  错误: {result['errors']}\n" if result['errors'] else "")
        )

    elif cmd == "/status":
        brainstem_status = modules["brainstem"].get_status()
        hippo_status = modules["hippocampus"].get_status()
        kb_status = modules["knowledge_base"].get_status()
        frontal_status = modules["frontal_lobe"].get_status()
        return (
            f"\n📊 系统状态:\n"
            f"  系统状态: {brainstem_status['state']}\n"
            f"  睡眠周期: {brainstem_status['cycle_count']}\n"
            f"  总警报数: {brainstem_status['total_alerts']}\n"
            f"  海马体记忆: {hippo_status['total_memories']}\n"
            f"  知识规则: {kb_status['total_rules']}\n"
            f"  对话轮数: {frontal_status['conversation_turns']}\n"
            f"  快照数: {frontal_status['snapshots_count']}\n"
        )

    elif cmd == "/memories":
        memories = modules["hippocampus"].get_all_memories(limit=10)
        if not memories:
            return "\n📭 海马体为空，暂无记忆"
        lines = ["\n🧠 海马体记忆:"]
        for i, mem in enumerate(memories, 1):
            doc = mem.get("document", "")[:80]
            meta = mem.get("metadata", {})
            importance = meta.get("importance_score", 0)
            lines.append(f"  {i}. [{importance:.2f}] {doc}")
        return "\n".join(lines)

    elif cmd == "/rules":
        rules = modules["knowledge_base"].get_all_rules()
        if not rules:
            return "\n📭 知识库为空，暂无规则 (尝试 /sleep 触发巩固)"
        lines = ["\n📚 知识图谱规则:"]
        for i, rule in enumerate(rules, 1):
            text = rule.get("rule_text", "")[:80]
            conf = rule.get("confidence", 0)
            lines.append(f"  {i}. [{conf:.2f}] {text}")
        return "\n".join(lines)

    elif cmd == "/help":
        return (
            "\n📖 可用命令:\n"
            "  /sleep    — 触发睡眠巩固 (知识蒸馏)\n"
            "  /status   — 查看系统状态\n"
            "  /memories — 查看海马体记忆\n"
            "  /rules    — 查看知识图谱规则\n"
            "  /help     — 显示帮助\n"
            "  /quit     — 退出系统\n"
        )

    else:
        return f"❓ 未知命令: {command}。输入 /help 查看可用命令。"


def main() -> None:
    """主入口"""
    print_banner()

    try:
        modules = initialize_system()
    except Exception as e:
        logger.error(f"系统初始化失败: {e}")
        print(f"\n❌ 初始化失败: {e}")
        print("请确保 Ollama 正在运行且已安装所需模型。")
        return

    print("\n🟢 NeuroCortex AI 已就绪！输入文本开始对话，输入 /help 查看命令。\n")

    while True:
        try:
            user_input = input("🧑 You > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见！")
            break

        if not user_input:
            continue

        # 命令处理
        if user_input.startswith("/"):
            result = handle_command(user_input, modules)
            if result is None:
                print("\n👋 NeuroCortex AI 已关闭。再见！")
                break
            print(result)
            continue

        # 正常对话处理
        print("\n🧠 NeuroCortex AI 正在思考...\n")
        try:
            response = process_input(user_input, modules)
            print(f"🤖 AI > {response}\n")
        except Exception as e:
            logger.error(f"处理失败: {e}")
            print(f"\n❌ 处理出错: {e}\n")


if __name__ == "__main__":
    main()
