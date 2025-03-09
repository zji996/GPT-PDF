# GPT-PDF 生产环境部署指南

## 概述

本文档提供了将 GPT-PDF 应用部署到生产环境的详细指南。Flask 的内置开发服务器（Werkzeug）仅适用于开发目的，不应在生产环境中使用，因为它存在以下问题：

- **性能限制**：开发服务器设计简单，无法处理高并发请求
- **安全性不足**：缺乏生产级别的安全防护措施
- **稳定性问题**：长时间运行可能出现内存泄漏或其他稳定性问题
- **可扩展性差**：不支持负载均衡和高级部署功能

## 部署方案

### 方案一：Gunicorn + Nginx（推荐用于 Linux/Unix 环境）

#### 1. 安装 Gunicorn

```bash
pip install gunicorn
```

#### 2. 创建 Gunicorn 启动脚本

创建文件 `start_gunicorn.sh`：

```bash
#!/bin/bash
gunicorn -w 4 -t 120 --bind 0.0.0.0:50000 app:app
```

参数说明：
- `-w 4`：启动4个工作进程（建议设置为 CPU 核心数 * 2 + 1）
- `-t 120`：worker 超时时间为 120 秒
- `--bind 0.0.0.0:50000`：绑定到所有网络接口的 50000 端口
- `app:app`：模块名:Flask应用实例名

添加执行权限：

```bash
chmod +x start_gunicorn.sh
```

#### 3. 设置 Nginx 反向代理

安装 Nginx：

```bash
# Ubuntu/Debian
sudo apt-get install nginx

# CentOS/RHEL
sudo yum install nginx
```

创建 Nginx 配置文件 `/etc/nginx/sites-available/gpt-pdf`：

```nginx
server {
    listen 80;
    server_name your-domain.com;  # 替换为您的域名或IP地址

    client_max_body_size 100M;  # 匹配应用中的文件大小限制

    location / {
        proxy_pass http://127.0.0.1:50000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 增加超时设置
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

启用站点配置并重启 Nginx：

```bash
# Ubuntu/Debian
sudo ln -s /etc/nginx/sites-available/gpt-pdf /etc/nginx/sites-enabled/
sudo systemctl restart nginx

# CentOS/RHEL
sudo cp /etc/nginx/sites-available/gpt-pdf /etc/nginx/conf.d/
sudo systemctl restart nginx
```

#### 4. 设置系统服务（Systemd）

创建 systemd 服务文件 `/etc/systemd/system/gpt-pdf.service`：

```ini
[Unit]
Description=GPT-PDF OCR API Service
After=network.target

[Service]
User=www-data  # 或您的服务用户
WorkingDirectory=/path/to/your/app  # 替换为应用目录路径
Environment="PATH=/path/to/your/venv/bin"  # 如果使用虚拟环境
ExecStart=/path/to/your/app/start_gunicorn.sh
Restart=always
RestartSec=5
StartLimitInterval=0

[Install]
WantedBy=multi-user.target
```

启用并启动服务：

```bash
sudo systemctl enable gpt-pdf
sudo systemctl start gpt-pdf
```

### 方案二：uWSGI + Nginx（适用于复杂应用）

#### 1. 安装 uWSGI

```bash
pip install uwsgi
```

#### 2. 创建 uWSGI 配置文件

创建 `uwsgi.ini` 文件：

```ini
[uwsgi]
module = app:app
master = true
processes = 5
threads = 2
socket = 127.0.0.1:50000
chmod-socket = 660
vacuum = true
die-on-term = true
harakiri = 300  # 请求超时时间（秒）
buffer-size = 65535  # 提高缓冲区大小
post-buffering = 8192  # 启用POST缓冲
```

#### 3. 设置 Nginx 配置

与 Gunicorn 方案中的 Nginx 配置类似，但修改 proxy_pass：

```nginx
location / {
    include uwsgi_params;
    uwsgi_pass 127.0.0.1:50000;
    # ... 其他设置与 Gunicorn 方案相同
}
```

#### 4. 创建 Systemd 服务

创建 `/etc/systemd/system/gpt-pdf.service`：

```ini
[Unit]
Description=GPT-PDF OCR API uWSGI Service
After=network.target

[Service]
User=www-data
Group=www-data
WorkingDirectory=/path/to/your/app
ExecStart=/path/to/your/venv/bin/uwsgi --ini uwsgi.ini
Restart=always
RestartSec=5
StartLimitInterval=0

[Install]
WantedBy=multi-user.target
```

### 方案三：Waitress（适用于 Windows 环境）

#### 1. 安装 Waitress

```bash
pip install waitress
```

#### 2. 创建启动脚本

创建 `waitress_server.py` 文件：

```python
from waitress import serve
from app import app

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=50000, threads=8, 
          url_scheme='http', channel_timeout=300)
```

#### 3. 使用 Windows 服务或任务计划程序

可以使用 NSSM（Non-Sucking Service Manager）将应用注册为 Windows 服务。

## 高可用部署设计

### 负载均衡

#### 使用 Nginx 实现负载均衡

```nginx
upstream gpt_pdf_cluster {
    server 127.0.0.1:50001;
    server 127.0.0.1:50002;
    server 127.0.0.1:50003;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://gpt_pdf_cluster;
        # ... 其他配置 ...
    }
}
```

### 容器化部署（Docker）

#### 1. 创建 Dockerfile

```dockerfile
FROM python:3.9-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 暴露端口
EXPOSE 50000

# 启动命令
CMD ["gunicorn", "-w", "4", "-t", "120", "--bind", "0.0.0.0:50000", "app:app"]
```

#### 2. 构建和运行容器

```bash
docker build -t gpt-pdf .
docker run -d -p 50000:50000 --name gpt-pdf-api gpt-pdf
```

#### 3. 使用 Docker Compose（可选）

创建 `docker-compose.yml`：

```yaml
version: '3'

services:
  gpt-pdf:
    build: .
    ports:
      - "50000:50000"
    volumes:
      - ./.env:/app/.env
      - ./prompt.txt:/app/prompt.txt
    restart: always
```

启动服务：

```bash
docker-compose up -d
```

## 安全性配置

### 1. 启用 HTTPS

在 Nginx 配置中添加 SSL 证书：

```nginx
server {
    listen 443 ssl;
    server_name your-domain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # ... 其他配置 ...
}
```

### 2. 添加基本认证

在 Nginx 配置中添加：

```nginx
location / {
    auth_basic "Restricted Access";
    auth_basic_user_file /etc/nginx/.htpasswd;
    # ... 其他配置 ...
}
```

创建密码文件：

```bash
sudo htpasswd -c /etc/nginx/.htpasswd username
```

### 3. 设置访问限制

```nginx
# 限制连接数
limit_conn_zone $binary_remote_addr zone=conn_limit_per_ip:10m;
limit_conn conn_limit_per_ip 10;

# 限制请求率
limit_req_zone $binary_remote_addr zone=req_limit_per_ip:10m rate=5r/s;
```

## 监控和日志管理

### 1. 配置日志轮转

创建 `/etc/logrotate.d/gpt-pdf`：

```
/path/to/your/app/logs/*.log {
    daily
    missingok
    rotate 14
    compress
    delaycompress
    notifempty
    create 0640 www-data www-data
    sharedscripts
    postrotate
        [ -f /var/run/nginx.pid ] && kill -USR1 `cat /var/run/nginx.pid`
    endscript
}
```

### 2. 使用 Prometheus 和 Grafana 监控

安装 Prometheus 客户端：

```bash
pip install prometheus-flask-exporter
```

在应用中添加监控：

```python
from prometheus_flask_exporter import PrometheusMetrics

metrics = PrometheusMetrics(app)
```

## 部署清单

- [ ] 更新所有软件包到最新版本
- [ ] 配置环境变量（.env 文件）
- [ ] 设置适当的用户权限
- [ ] 配置 WSGI 服务器（Gunicorn/uWSGI/Waitress）
- [ ] 设置 Nginx 反向代理
- [ ] 配置 SSL 证书
- [ ] 设置防火墙规则
- [ ] 配置系统服务
- [ ] 设置日志轮转
- [ ] 设置监控
- [ ] 进行负载测试
- [ ] 建立备份策略
- [ ] 制定故障恢复计划

## 问题排查

### WSGI 服务器无法启动

检查日志文件以获取详细错误信息：

```bash
journalctl -u gpt-pdf.service
```

### 502 Bad Gateway 错误

1. 检查 WSGI 服务是否正在运行
2. 验证 Nginx 配置中的端口是否与 WSGI 服务匹配
3. 检查防火墙设置

### 处理超时

增加超时设置：
- Nginx: 修改 proxy_read_timeout 值
- Gunicorn: 修改 -t 参数
- uWSGI: 修改 harakiri 参数

## 结论

将 GPT-PDF 应用部署到生产环境需要使用专业的 WSGI 服务器和反向代理。这种配置不仅提高了性能和安全性，还提供了更好的可扩展性和可靠性。根据您的操作系统和具体需求，可以选择最适合的部署方案。 