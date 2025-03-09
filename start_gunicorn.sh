#!/bin/bash

# GPT-PDF 生产环境启动脚本
# 使用Gunicorn作为WSGI服务器

# 工作进程数量（建议设置为 CPU核心数 * 2 + 1）
# 可根据服务器配置调整此值
WORKERS=4

# 启动Gunicorn服务
gunicorn -w $WORKERS \
         -t 120 \
         --bind 0.0.0.0:50000 \
         --access-logfile logs/access.log \
         --error-logfile logs/error.log \
         app:app 