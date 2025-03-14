#!/bin/bash

# GPT-PDF 生产环境启动脚本
# 使用Gunicorn作为WSGI服务器，并使用.venv虚拟环境

# 工作进程数量（建议设置为 CPU核心数 * 2 + 1）
# 可根据服务器配置调整此值
WORKERS=4

# 激活虚拟环境
source .venv/bin/activate

# 确保日志目录存在
mkdir -p logs

# 启动Gunicorn服务
gunicorn -w $WORKERS \
         -t 120 \
         --bind 0.0.0.0:50000 \
         --access-logfile logs/access.log \
         --error-logfile logs/error.log \
         app:app 