"""
NeuroCortex AI — 配置控制面板
===============================
Streamlit 构建的系统配置与监控面板。
功能: 模型列表、能力契约编辑、连接测试、系统状态概览。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st
import yaml

# 确保项目根目录在 path 中
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT.parent))

from neurocortex.core.config_loader import ConfigLoader
from neurocortex.core.prefrontal_config import ModelRouter
from neurocortex.memory.knowledge_base import KnowledgeBase
from streamlit_agraph import agraph, Node, Edge, Config


def main() -> None:
    """Streamlit 应用入口"""
    st.set_page_config(
        page_title="NeuroCortex AI — 控制面板",
        page_icon="🧠",
        layout="wide",
    )

    st.title("🧠 NeuroCortex AI — 配置控制面板")
    st.markdown("---")

    # 初始化
    config_path = _PROJECT_ROOT / "config" / "model_config.yaml"
    if "router" not in st.session_state:
        st.session_state.router = ModelRouter(str(config_path))
    router: ModelRouter = st.session_state.router

    # 侧边栏
    page = st.sidebar.radio("导航", ["📋 模型管理", "🔧 脑区配置", "📊 系统状态", "🕸️ 知识图谱", "➕ 添加模型"])

    if page == "📋 模型管理":
        _page_model_management(router)
    elif page == "🔧 脑区配置":
        _page_brain_region_config(router)
    elif page == "📊 系统状态":
        _page_system_status(router)
    elif page == "🕸️ 知识图谱":
        _page_knowledge_graph()
    elif page == "➕ 添加模型":
        _page_add_model(router, config_path)


def _page_model_management(router: ModelRouter) -> None:
    """模型管理页面"""
    st.header("📋 模型注册表")

    models = router.list_models()
    if not models:
        st.warning("无已注册模型")
        return

    for model_id, cfg in models.items():
        with st.expander(f"🤖 {model_id} ({cfg['provider']})", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"**Provider:** `{cfg['provider']}`")
                st.markdown(f"**Model:** `{cfg['model_name']}`")
                st.markdown(f"**Endpoint:** `{cfg.get('endpoint', 'N/A')}`")
                st.markdown(f"**Scope:** `{', '.join(cfg.get('scope', []))}`")

            with col2:
                contract = cfg.get("capability_contract", {})
                st.markdown("**能力契约:**")
                st.json(contract)

            # 测试连接按钮
            if st.button(f"🔌 测试连接", key=f"test_{model_id}"):
                with st.spinner("正在测试..."):
                    result = router.test_connectivity(model_id)
                if result["available"]:
                    st.success(f"✅ 连接成功! 延迟: {result['latency_ms']}ms, Schema校验: {'通过' if result['schema_valid'] else '失败'}")
                else:
                    st.error(f"❌ 连接失败: {result.get('error', '未知错误')}")


def _page_brain_region_config(router: ModelRouter) -> None:
    """脑区配置页面"""
    st.header("🔧 脑区模型配置")

    brain_regions = ["amygdala", "frontal", "parietal", "thalamus",
                     "hippocampus-consolidation", "occipital", "temporal"]

    for region in brain_regions:
        available = router.get_models_for_region(region)
        current_override = router.overrides.get(region, "默认 (自动)")

        col1, col2, col3 = st.columns([2, 2, 1])
        with col1:
            st.markdown(f"**{region}**")
        with col2:
            options = ["默认 (自动)"] + available
            selected = st.selectbox(
                "选择模型", options, key=f"select_{region}",
                index=0 if current_override == "默认 (自动)" else (
                    options.index(current_override) if current_override in options else 0
                ),
            )
        with col3:
            if st.button("应用", key=f"apply_{region}"):
                if selected == "默认 (自动)":
                    router.overrides.pop(region, None)
                    st.success(f"{region} 恢复默认")
                else:
                    router.switch_model(region, selected)
                    st.success(f"{region} → {selected}")

    st.markdown("---")
    st.subheader("当前覆盖设置")
    if router.overrides:
        st.json(router.overrides)
    else:
        st.info("无手动覆盖，所有脑区使用默认路由")


def _page_system_status(router: ModelRouter) -> None:
    """系统状态页面"""
    st.header("📊 系统状态概览")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("已注册模型")
        models = router.list_models()
        st.metric("模型总数", len(models))
        for mid in models:
            st.markdown(f"- `{mid}`")

    with col2:
        st.subheader("活跃客户端")
        st.metric("活跃连接数", len(router.active_clients))
        for cid, client in router.active_clients.items():
            st.markdown(f"- `{cid}` → {client.model_name}")

    st.markdown("---")
    st.subheader("系统配置")
    system_cfg = router.config_loader.get_system_settings()
    if system_cfg:
        st.json(system_cfg)
    else:
        st.info("无系统配置")


def _page_knowledge_graph() -> None:
    """知识图谱可视化页面"""
    st.header("🕸️ 长期知识图谱")
    
    kb = KnowledgeBase()
    status = kb.get_status()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("节点数", status["total_nodes"])
    col2.metric("边数", status["total_edges"])
    col3.metric("存储路径", Path(status["storage_path"]).name)

    if status["total_nodes"] == 0:
        st.info("图谱目前为空，请在主系统中通过对话和睡眠巩固来生成知识。")
        return

    # 构建 agraph 节点和边
    nodes = []
    edges = []
    
    for nid, attrs in kb.graph.nodes(data=True):
        node_type = attrs.get("type", "rule")
        label = attrs.get("rule_text", attrs.get("name", nid))[:20]
        
        # 不同类型节点使用不同颜色
        color = "#60A5FA" # blue for rule
        if node_type == "entity":
            color = "#F87171" # red
        elif node_type == "episode":
            color = "#34D399" # green
            
        nodes.append(Node(id=nid, label=label, size=15, color=color))

    for src, tgt, attrs in kb.graph.edges(data=True):
        label = attrs.get("relation", "")
        edges.append(Edge(source=src, target=tgt, label=label))

    config = Config(width=900, height=600, directed=True, nodeHighlightBehavior=True, 
                    highlightColor="#F79767", collapsible=True)

    agraph(nodes=nodes, edges=edges, config=config)

    st.markdown("---")
    st.subheader("节点详情")
    for nid, attrs in kb.graph.nodes(data=True):
        with st.expander(f"节点: {nid}"):
            st.json(attrs)


def _page_add_model(router: ModelRouter, config_path: Path) -> None:
    """添加模型页面"""
    st.header("➕ 添加新模型")

    with st.form("add_model_form"):
        model_id = st.text_input("模型 ID", placeholder="my-new-model")
        provider = st.selectbox("Provider", ["ollama", "openai_compatible", "anthropic"])
        model_name = st.text_input("模型名称", placeholder="qwen2.5:7b")
        endpoint = st.text_input("API Endpoint", value="http://localhost:11434")
        api_key_env = st.text_input("API Key 环境变量 (仅云端模型)", placeholder="MY_API_KEY")
        scope = st.multiselect("Scope (适用脑区)",
                               ["amygdala", "frontal", "parietal", "thalamus",
                                "hippocampus-consolidation", "occipital", "temporal"])
        reasoning_depth = st.selectbox("推理深度", ["low", "medium", "high"])

        st.markdown("**输出 Schema (JSON)**")
        output_schema = st.text_area("output_schema", value='{"type": "string"}', height=80)

        submitted = st.form_submit_button("添加模型")

        if submitted and model_id and model_name and scope:
            try:
                schema = json.loads(output_schema)
                new_model = {
                    "provider": provider,
                    "model_name": model_name,
                    "scope": scope,
                    "capability_contract": {
                        "output_schema": schema,
                        "reasoning_depth": reasoning_depth,
                    },
                }
                if endpoint:
                    new_model["endpoint"] = endpoint
                if api_key_env:
                    new_model["api_key_env"] = api_key_env

                # 加载、修改、保存配置
                with open(config_path, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                config["model_registry"][model_id] = new_model
                with open(config_path, "w", encoding="utf-8") as f:
                    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

                router.config_loader.reload()
                st.success(f"✅ 模型 '{model_id}' 已添加！")
            except json.JSONDecodeError:
                st.error("output_schema 不是有效的 JSON")
            except Exception as e:
                st.error(f"添加失败: {e}")


if __name__ == "__main__":
    main()
