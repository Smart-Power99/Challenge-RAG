FROM python:3.10-slim

WORKDIR /app

# 设置 Python 环境路径以及开启全局的 HuggingFace 国内提速
ENV PYTHONPATH=/app
ENV HF_ENDPOINT=https://hf-mirror.com

# 1. 注入依赖并使用国内清华源加速安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 将核心代码复制到镜像中
COPY ./app ./app

# 3. 创建数据存放的挂载存根路径 (运行时由 docker-compose 提供 aapl_10k.json 的实际挂载)
RUN mkdir -p /app/data
