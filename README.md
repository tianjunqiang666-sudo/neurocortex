# NeuroCortex AI (类脑智能系统)

NeuroCortex AI 是一种基于人类大脑解剖学与认知神经科学原理设计的智能软件架构。它不仅仅是一个调用大模型的对话脚本，而是一个拥有感知、记忆、情绪评估和长期知识巩固能力的“硅基大脑”。系统支持完全本地化部署（基于 Ollama），并且能够在不进行任何模型参数微调（Fine-tuning）的情况下，通过睡眠巩固机制实现知识图谱的提取与持久化。

## 🧠 核心架构与模块

系统完全模拟人类大脑的处理路径，划分为以下核心组件：

- **Thalamus (丘脑)**: 多模态感知网关。负责接收外界输入（文本、视觉、听觉），生成向量嵌入（Embedding），并根据显著性（Salience）将信息路由至快慢双通路。
- **Brainstem (脑干)**: 系统的生命周期与调度中心。管理系统的清醒与睡眠周期，处理高优先级的紧急中断（如触发威胁警报）。
- **Fast Pathway (快速通路)**:
  - **Amygdala (杏仁核)**: 负责实时的情绪与威胁评估，当检测到高度危险或极端情绪（如愤怒）时，可直接阻断慢速推理，触发本能反应。
  - **Basal Ganglia (基底节)**: 习惯与条件反射存储。对于高频重复的模式，系统会形成肌肉记忆直接响应，降低大模型的推理开销。
- **Slow Pathway (慢速通路)**:
  - **Temporal Lobe & Occipital Lobe (颞叶/枕叶)**: 处理听觉/语言语义以及视觉输入，提取特征。
  - **Parietal Lobe (顶叶)**: 多模态数据融合中心。将不同维度的感知融合为统一的“事件框架 (Event Frame)”，并计算预测误差（Prediction Error）以标记记忆的重要性。
  - **Frontal Lobe (额叶)**: 系统的“CEO”。负责工作记忆（Working Memory）的维持、复杂逻辑推理、任务规划以及最终回复的生成。
- **Memory System (记忆系统)**:
  - **Hippocampus (海马体)**: 情景记忆（Episodic Memory）的快速编码与检索，基于 ChromaDB 向量数据库实现。
  - **Memory Consolidation (睡眠巩固)**: 模拟人类睡眠时的记忆重播机制。在系统“睡眠”期间，海马体中高重要性的记忆会被提取，通过大模型进行知识蒸馏，转化为抽象规则存储至知识图谱。
  - **Knowledge Base (知识库)**: 基于 NetworkX 的持久化知识图谱存储长期稳定的规则与世界模型。
- **ModelRouter (模型路由)**: 基于“能力契约”的动态模型调度模块。允许在不同认知任务中自动切换最合适的本地大模型。

## 🚀 快速开始

### 环境依赖
- Windows 11 / macOS / Linux
- Python 3.12+
- [Ollama](https://ollama.com/) (需要至少下载一个基础模型，例如 `deepseek-r1:8b` 或 `llama3`)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-username/NeuroCortex-AI.git
   cd "NeuroCortex AI"
   ```

2. **安装 Python 依赖**
   建议使用虚拟环境：
   ```bash
   python -m venv venv
   # Windows: venv\Scripts\activate
   # Linux/macOS: source venv/bin/activate
   pip install -r neurocortex/requirements.txt
   ```

3. **配置本地 Ollama 模型**
   请确保本地已启动 Ollama 并在后台运行：
   ```bash
   ollama run deepseek-r1:8b  # 或者配置中指定的其他模型
   ```

### 运行系统

- **启动主控 CLI**
  运行交互式类脑终端：
  ```bash
  python neurocortex/main.py
  ```
  在终端中可以直接输入内容与系统交互。系统支持特定指令：
  - `/status`：查看系统与各脑区状态
  - `/sleep`：强制系统进入睡眠模式，执行记忆巩固
  - `/memories`：查看海马体中当前存储的情景记忆
  - `/rules`：查看知识图谱中已沉淀的规则
  - `/help`：查看所有命令

- **启动 Streamlit 控制面板 (WIP)**
  如果需要可视化管理模型路由和系统状态：
  ```bash
  streamlit run neurocortex/ui/config_dashboard.py
  ```

## 🛡️ 设计准则与约束

1. **绝对禁止参数微调 (No Parameter Fine-Tuning)**: 本系统坚守“智能应当通过记忆提取与工作空间重构产生”的理念，所有的记忆、学习与成长均通过向量库（海马体）和知识图谱完成。
2. **解剖学对齐**: 模块之间的信息流向严格遵守大脑神经解剖学回路（如：丘脑不可越级直接操作额叶生成）。
3. **隐私安全**: 基于本地部署优先原则，所有记忆均存储于本地 `data/` 目录中。

## 🤝 贡献与二次开发
本项目处于 MVP 阶段，欢迎在“听觉”、“视觉”感知通道接入真实的音频流（如 Whisper）或视觉模型（如 CLIP）以完善其多模态能力。

## 📄 开源协议
本项目基于 **[CC BY-NC 4.0 (知识共享署名-非商业性使用 4.0 国际许可协议)](LICENSE)** 开源。
- **允许**：您可以自由地共享（复制和重新分发）和演绎（修改、转换或以此为基础进行创作）。
- **限制**：**严禁用于任何商业用途**。任何个人或机构不得将本系统或其衍生产品用于商业营利目的。
