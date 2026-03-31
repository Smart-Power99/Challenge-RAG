import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8222/ask")

st.set_page_config(page_title="🍏 Apple 10-K 财报问答系统", page_icon="🍏")

st.title("🍏 Apple 10-K 财报智能问答系统")
st.markdown("基于 **FAISS** 和 **在线大语言模型 (OpenAI格式)** 构建的 RAG 问答机器人。内部数据涵盖 AAPL 公司年度财报 (10-K)。\n\n*提示：第一次启动时可能需要时间来建立本地 FAISS 索引并下载 Embedding 模型权重。*")

# 初始化 Session 状态
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史聊天记录
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 聊天输入框
if prompt := st.chat_input("请输入您关于苹果财报的问题（如：2025年的主要风险是什么？）"):
    # 用户输入上屏
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 机器人回复
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("正在检索相关财报片段并生成答案...")
        
        try:
            response = requests.post(API_URL, json={"query": prompt})
            response.raise_for_status()
            answer = response.json().get("answer", "产生错误：未获得答案")
            message_placeholder.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            error_msg = f"API 请求失败: 请确认后端已就绪并配置了有效的 OpenAI API Key。错误详情：{str(e)}"
            message_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
