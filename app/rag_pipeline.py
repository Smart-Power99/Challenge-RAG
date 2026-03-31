import os
# 强行注入国内 HuggingFace 加速镜像代理，解决一切无法下载和 timeout 问题
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

class AAPL10KRAG:
    def __init__(self, data_path: str = "./data/aapl_10k.json", index_dir: str = "./data/faiss_index"):
        self.data_path = data_path
        self.index_dir = index_dir
        
        # 1. 向量模型初始化
        print("初始化 Embedding 模型: 正在通过国内 HF 镜像源直连拉取 all-MiniLM-L6-v2 ...")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # 2. LLM 初始化 (自动读取环境变量 OPENAI_API_KEY, OPENAI_API_BASE 等)
        api_key = os.getenv("OPENAI_API_KEY", "xxx")
        api_base = os.getenv("OPENAI_API_BASE", "xxx")
        model_name = os.getenv("MODEL_NAME", "xxx")
        
        self.llm = ChatOpenAI(
            api_key=api_key,
            base_url=api_base,
            model_name=model_name,
            temperature=0.1
        )
        
        # 3. 检查或构建 FAISS 索引
        if os.path.exists(self.index_dir):
            print(f"检测到已存在的 FAISS 索引: {self.index_dir}，正在加载...")
            self.vector_store = FAISS.load_local(self.index_dir, self.embeddings, allow_dangerous_deserialization=True)
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        else:
            print("未检测到索引，准备从数据源构建...")
            self.retriever = self._build_index()

        # 4. Prompt Template
        template = """基于以下提取出的财报信息，请仅根据所提供的内容来回答用户的问题。如果无法从背景信息中得出答案，请声明你不知道，不要虚构事实。
回答时请使用专业、简洁的中文，并尽可能引用年份或章节信息。

【上下文信息】
{context}

【问题】
{question}

【回答】
"""
        self.prompt = PromptTemplate.from_template(template)
        
        # 5. 构建 RAG 预设工作流 (LCEL)
        self.rag_chain = (
            {"context": self.retriever | self._format_docs, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _format_docs(self, docs: List[Document]) -> str:
        return "\n\n".join(f"[源: 年份:{doc.metadata.get('year')}, 章节:{doc.metadata.get('section')}]\n{doc.page_content}" for doc in docs)

    def _load_json_data(self) -> List[Document]:
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"找不到数据文件: {self.data_path}。请确保把 aapl_10k.json 放到了 data/ 目录下！")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # 根据结构，第一层是一个字典（Key是SQL），Value是列表
        docs_raw = []
        for key, rows in data.items():
            if isinstance(rows, list):
                docs_raw.extend(rows)
            
        documents = []
        for row in docs_raw:
            text = row.get("section_text", "")
            if not text:
                continue
            metadata = {
                "symbol": row.get("symbol"),
                "year": row.get("file_fiscal_year"),
                "section": row.get("section_title"),
                "section_id": row.get("section_id")
            }
            documents.append(Document(page_content=text, metadata=metadata))
            
        return documents

    def _build_index(self):
        print("加载及解析 JSON 数据...")
        documents = self._load_json_data()
        
        # 文本切分器
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(documents)
        
        print(f"总计切分出 {len(splits)} 个文档块进行向量化...")
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        
        #确保存储目录存在
        os.makedirs(os.path.dirname(self.index_dir), exist_ok=True)
        self.vector_store.save_local(self.index_dir)
        print("FAISS 索引已构建并保存至本地文件。")
        
        return self.vector_store.as_retriever(search_kwargs={"k": 5})

    def ask(self, query: str) -> str:
        return self.rag_chain.invoke(query)
