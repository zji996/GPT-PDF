FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖，包括Poppler（用于PDF处理）
RUN apt-get update && apt-get install -y \
    poppler-utils \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . /app/

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# 设置默认环境变量
ENV OPENAI_API_KEY="your_api_key_here" \
    OPENAI_ENDPOINT="https://api.openai.com/v1/chat/completions" \
    MODEL_NAME="gpt-4-turbo" \
    CONCURRENCY=5 \
    MAX_CONCURRENCY=20 \
    MIN_CONCURRENCY=1 \
    DPI=200 \
    IMAGE_QUALITY=85 \
    BATCH_SIZE=20 \
    USE_MULTIPROCESSING=true \
    CLEAN_CODE_BLOCKS=true \
    MAX_RETRIES=3 \
    RETRY_DELAY=2.0 \
    RETRY_BACKOFF=1.5 \
    RETRY_JITTER=0.1 \
    API_TIMEOUT=300

# 创建日志目录
RUN mkdir -p logs

# 暴露服务端口
EXPOSE 50000

# 使用脚本启动服务，或直接使用gunicorn命令启动
CMD gunicorn -w 4 -t 120 --bind 0.0.0.0:50000 --access-logfile logs/access.log --error-logfile logs/error.log app:app 