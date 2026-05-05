# Phase 3: 快速通路与分级中断

在前两个阶段中，我们已经成功搭建了慢速双通路（多模态感知 -> 顶叶融合 -> 海马体 -> 额叶推理）。在本阶段，我们将彻底点亮 **Fast Pathway（快速通路）**，使得系统不仅能“三思而后行”，还能对特定情境做出“条件反射”。

## Goal
实现基底节（Basal Ganglia）模块，构建“刺激-反应”习惯库，使得系统能够绕过复杂的慢速皮层和 LLM 推理，直接输出低延迟的肌肉记忆反射。同时完善快慢通路在主循环中的协同逻辑。

## Proposed Changes

### `neurocortex/pathways/fast/basal_ganglia.py`
- [NEW] 创建 `basal_ganglia.py`。
- **职责**：维护一个规则表/习惯库。通过语义匹配（基于 Thalamus 生成的 Embedding）或者正则/关键字匹配来判断当前输入是否触发了某个“习惯（Habit）”。
- **输出**：如果匹配度超过极高阈值（例如 0.9），直接返回封装的响应，要求 `main.py` 立即阻断后续执行并返回该响应。

### `neurocortex/memory/consolidation.py`
- [MODIFY] 升级睡眠巩固逻辑。
- 额叶在睡眠期蒸馏出知识规则后，不仅写入知识图谱（KnowledgeBase），若某些规则属于“固定指令响应”（如：用户说“退下” -> 系统回复“好的”），则同步写入 `BasalGanglia` 的习惯库中。

### `neurocortex/main.py`
- [MODIFY] 在 `Thalamus` 处理后，先将数据包传入 `BasalGanglia`。
- 若 `BasalGanglia` 返回了习惯动作，直接跳过后面的 `OccipitalLobe`, `TemporalLobe`, `ParietalLobe`, `Hippocampus` 和 `FrontalLobe`，实现真正的毫秒级“膝跳反射”。

## Open Questions
> [!IMPORTANT]
> **关于习惯库的存储方式**：
> 基底节的习惯库需要持久化存储以在重启后生效。你希望我将其存为独立的 `data/habits.json`，还是与现有的 `data/knowledge_graph.json` 整合，通过节点类型（如 `type="habit"`）进行区分？（默认我将使用单独的 `habits.json` 以保持模块解耦）。
