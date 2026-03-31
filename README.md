# AAPL 10-K 智能金融问答系统

这是一个基于 AAPL (Apple Inc.) 年度报告 (10-K) 数据的智能检索增强生成 (RAG) 问答系统项目。主要运用 FastAPI、Streamlit、LangChain 以及 FAISS 进行本地化构建，并通过解耦设计的线上可配置化大语言模型 (OpenAI API 兼容协议)，完成极低资源占用的本地知识挖掘能力。

## 功能介绍

- 自动解析无嵌套关系的标准 `aapl_10k.json` 并进行切块 (Chunks)。
- **智能规避**：底层默认通过注入 `HF_ENDPOINT` 并使用国内镜像源直连下载 `all-MiniLM-L6-v2` 进行轻量化本地特征文本向量处理。
- 后端使用 `FastAPI` 提供高性能异步查询能力，暴露端口为 `:8222/ask`。
- 前端使用 `Streamlit` 搭建交互式聊天气泡窗口，所见即所得。
- **配置化生成策略**：您可在 `.env` 配置文件下自由对接阿里百炼/Kimi/DeepSeek OpenAI 服务端点。

## 环境要求

- Docker 与 Docker-Compose

## 目录结构说明

```text
aapl-qa-system/
├── data/
│   └── aapl_10k.json         # 重要：需要您或测试员手动放入财报源数据文件！
├── app/
│   ├── api.py                # 后端查询接口服务
│   ├── rag_pipeline.py       # RAG 信息流构建与检索枢纽
│   └── ui.py                 # Streamlit 面板界面 
├── Dockerfile                # 项目全局镜像装配清单
├── docker-compose.yml        # 基于 Docker 的微服务编排
├── requirements.txt          # 纯净版 Python 依赖集
├── .env.example              # 调用模型凭证配置范式模板
└── README.md                 # 您正在查看的本文档
```

## 测试官快速启动指南 (Quick Start)

**1. 放置数据**
首先在当前项目的根目录下确认存在 `data` 文件夹（如果没有请提前创建），然后再将 `aapl_10k.json` 放置进去：
```bash
./data/aapl_10k.json
```

**2. 配置密钥与模型**
将项目根目录下的 `.env.example` 重命名为 `.env`。并为其配置标准的模型链接格式（代码会自动读取环境变量以启动线上大语言模型回答组建）。
例如：
```ini
OPENAI_API_KEY=sk-xxxxxxx
OPENAI_API_BASE=https://api.openai.com/v1
MODEL_NAME=gpt-4o-mini
```

**3. 一键启停服务**
请确保您的电脑安装且开启了 Docker Daemon 服务，在工程目录下通过命令行执行：
```bash
docker-compose up -d
```
> **提示**：首次执行将触发 Docker 的 `Build` 流程拉取 Python 依赖。随后当 `aapl_api` 服务容器拉起时系统会自动开始切割十万字以上的 Json 文件，并通过镜像节点缓存下模型组件。所以 API 从挂载到能够通过前台问答可能会需要短暂的数分钟本地编译与数据库构建过程（您可通过 `docker logs -f aapl_api` 追踪索引建立进度），当容器存放好 `/data/faiss_index` 文件后，即为最终就绪，此后即使服务器断电仍可秒连读盘库。

**4. 交互界面访问**
打开您的浏览器，输入如下地址进行功能点验：[http://localhost:8501](http://localhost:8501)

## 核心设计决议

针对基于文本进行知识召回，抛弃重度依赖本地显卡的庞大架构，采用轻重分离的业务逻辑来迎合轻快落地的微服务：
1. **Embedding 服务极简主义化**: 我们采用官方 `langchain-huggingface` 构建的底座并抛弃如魔搭社区等容易出现连环底层冲突的生态件。同时通过内嵌 `hf-mirror` 指向完美利用国内加速带。
2. **FAISS 快照留存**: 我们将 FAISS 做本地快照存储机制。首次启动自动计算的千万级向量结果将被保存在外挂磁盘 `data/` 下。任何时候服务重启均直接检测出并秒级读取索引避免冗杂的多次计算浪费。
3. **彻底的大语言平台无关性**: 因为 `Langchain-OpenAI` 具有顶尖的普适性，整个程序只要注入兼容的 url 即成为不限模型厂商的“完全体生成代理”。
