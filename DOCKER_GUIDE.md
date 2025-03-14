# GPT-PDF Docker 部署指南

本指南将帮助您使用 Docker 和 Docker Compose 部署 GPT-PDF 项目。

## 前提条件

- 安装 [Docker](https://docs.docker.com/get-docker/)
- 安装 [Docker Compose](https://docs.docker.com/compose/install/)

## 快速开始

### 1. 准备环境变量

首先，拷贝环境变量模板并进行配置：

```bash
cp .env.template .env
```

编辑 `.env` 文件，设置您的 OpenAI API 密钥和其他配置。

### 2. 使用 Docker Compose 启动服务

```bash
docker compose up -d
```

此命令将构建Docker镜像并在后台启动服务。服务将在 50000 端口运行。

### 3. 查看服务日志

```bash
docker compose logs -f gpt-pdf
```

## 配置说明

### 环境变量

您可以通过以下方式配置环境变量：

1. 直接编辑 `.env` 文件
2. 在 `docker-compose.yml` 文件的 `environment` 部分修改配置
3. 启动容器时使用 `-e` 参数设置环境变量

### 使用 Nginx 反向代理（可选）

如果您希望使用 Nginx 作为反向代理，可以取消 `docker-compose.yml` 文件中 Nginx 部分的注释，然后：

1. 确保 `nginx.conf` 文件存在
2. 如果需要配置 HTTPS，创建 `ssl` 目录并放入证书文件
3. 修改 `nginx.conf` 中的 `server_name` 为您的域名或 IP 地址

然后重新启动服务：

```bash
docker compose up -d
```

## 持久化存储

默认情况下，日志文件会通过卷挂载保存在主机的 `./logs` 目录中。如果您需要持久化上传的 PDF 文件，可以取消 `docker-compose.yml` 中相关卷挂载的注释。

## 资源调整

在生产环境中，您可能需要调整 Gunicorn 的工作进程数量。这可以通过修改 `Dockerfile` 中的 CMD 指令实现：

```
# 将 -w 4 更改为适合您服务器的值（通常为 CPU 核心数 * 2 + 1）
CMD gunicorn -w 8 -t 120 --bind 0.0.0.0:50000 --access-logfile logs/access.log --error-logfile logs/error.log app:app
```

## 故障排除

### 容器无法启动

检查日志以获取详细错误信息：

```bash
docker compose logs gpt-pdf
```

### PDF 处理问题

确保 Poppler 工具在容器中正确安装。您可以通过以下命令进入容器并检查：

```bash
docker compose exec gpt-pdf bash
which pdftoppm
```

## 高级配置

### 自定义构建镜像

如果您需要添加额外的依赖项或修改构建过程，可以编辑 `Dockerfile`。

### 使用 Docker Swarm 或 Kubernetes 部署

对于更高级的部署场景，您可以使用 Docker Swarm 或 Kubernetes：

1. Docker Swarm 部署：
   ```bash
   docker stack deploy -c docker-compose.yml gpt-pdf
   ```

2. 对于 Kubernetes 部署，您需要创建对应的 Kubernetes 配置文件（pod、service、deployment 等）。 