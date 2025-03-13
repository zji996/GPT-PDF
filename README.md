# GPT-PDF OCR API 文档

## 概述

GPT-PDF OCR API 是一个强大的工具，能够使用视觉模型对 PDF 文件执行 OCR（光学字符识别）任务，并将结果以 Markdown 格式返回。该 API 特别适用于需要从扫描文档、图像化 PDF 或其他不可直接编辑的 PDF 文件中提取文本内容的场景。

## 主要特点

- 使用先进视觉模型处理图像内容
- 将 PDF 的每一页转换为适当分辨率的图像进行处理
- 支持通过 `prompt.txt` 文件自定义系统提示词，控制 AI 的行为
- 通过并发处理提高处理效率
- 将结果输出为规范的 Markdown 格式
- 支持通过环境变量进行灵活配置
- 支持服务端清理Markdown代码块标记，简化客户端处理
- 内置请求失败自动重试机制，提高服务稳定性
- 支持配置多个备用服务端点，实现服务高可用
- 采用自适应并发控制技术，动态调整并发度以优化资源利用率

## 并发优化

本服务实现了先进的并发控制机制，以最大化资源利用率和API吞吐量：

### 动态滑动窗口并发

不同于传统的批处理模式（等待当前批次所有请求完成后再发起新请求），本系统采用动态滑动窗口模式：
- 任何时候都保持最多N个并发请求（N由配置决定）
- 当任何一个请求完成后，立即启动新的请求以保持并发度
- 这种模式确保了服务器资源的持续利用，不会出现空闲等待

### 自适应并发控制

系统会根据API请求的响应时间动态调整并发度：
- 当响应时间增加（可能表明服务压力增大）时，自动降低并发数
- 当响应时间减少（表明服务有更多处理能力）时，自动增加并发数
- 并发度始终保持在配置的最小值和最大值之间
- 通过`/stats`端点可以实时查看当前并发状态和历史响应时间

### 并发监控

系统提供实时的并发监控功能：
- 通过`/stats`端点获取当前活跃请求数、历史最大并发数等统计信息
- 监控并发请求的各项指标，辅助性能调优

## API 端点

### POST /ocr

将 PDF 文件转换为 Markdown 文本。

#### 请求

- **Content-Type**: `multipart/form-data`
- **Body**:
  - `file`: PDF 文件（必需）

#### 响应

- **Content-Type**: `text/markdown`
- **Body**: Markdown 格式的文本内容
- **状态码**:
  - `200 OK`: 处理成功
  - `400 Bad Request`: 请求错误（如未上传文件、文件格式不是 PDF 等）
  - `500 Internal Server Error`: 服务器内部错误

#### 示例

使用 cURL:

```bash
curl -X POST -F "file=@document.pdf" http://localhost:50000/ocr -o document.md
```

使用 Python requests:

```python
import requests

pdf_path = "document.pdf"
url = "http://localhost:50000/ocr"

with open(pdf_path, 'rb') as f:
    files = {'file': (pdf_path, f, 'application/pdf')}
    response = requests.post(url, files=files)

if response.status_code == 200:
    with open('document.md', 'w', encoding='utf-8') as f:
        f.write(response.text)
    print("转换成功，已保存为 document.md")
else:
    print(f"转换失败，错误码: {response.status_code}")
    print(f"错误信息: {response.text}")
```

### GET /stats

获取当前服务的并发请求统计和性能指标。

#### 请求

无需任何参数。

#### 响应

- **Content-Type**: `application/json`
- **Body**: JSON对象，包含以下字段：
  - `active_requests`: 当前活跃的请求数
  - `max_observed_concurrency`: 历史最大并发数
  - `total_requests`: 自服务启动以来处理的总请求数
  - `completed_requests`: 已完成的请求数
  - `current_concurrency_limit`: 当前使用的并发限制
  - `min_concurrency`: 配置的最小并发数
  - `max_concurrency`: 配置的最大并发数
  - `recent_response_times`: 最近N个请求的响应时间数组（秒）
- **状态码**:
  - `200 OK`: 请求成功

#### 示例

使用 cURL:

```bash
curl http://localhost:50000/stats
```

示例响应:

```json
{
  "active_requests": 3,
  "max_observed_concurrency": 5,
  "total_requests": 42,
  "completed_requests": 39,
  "current_concurrency_limit": 4,
  "min_concurrency": 1,
  "max_concurrency": 20,
  "recent_response_times": [3.24, 3.56, 2.98, 3.12, 3.05]
}
```

## 配置

API 服务通过 `.env` 文件进行配置，支持以下配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OPENAI_ENDPOINT` | OpenAI API 端点 | `https://api.openai.com/v1/chat/completions` |
| `MODEL_NAME` | 使用的 OpenAI 模型 | `gpt-4o` |
| `CONCURRENCY` | 并发处理的页面数 | `5` |
| `DPI` | PDF 转图像的分辨率 | `300` |
| `IMAGE_QUALITY` | 图像质量（1-100），影响OCR准确率和处理速度 | `85` |
| `BATCH_SIZE` | 每批处理的最大PDF页数，用于大文档分批处理 | `20` |
| `USE_MULTIPROCESSING` | 是否使用多进程加速图像处理 | `true` |
| `CLEAN_CODE_BLOCKS` | 是否清理markdown代码块标记 | `true` |
| `MAX_RETRIES` | 同一服务的最大重试次数 | `3` |
| `RETRY_DELAY` | 初始重试延迟（秒） | `2.0` |
| `RETRY_BACKOFF` | 重试延迟的指数增长因子 | `1.5` |
| `RETRY_JITTER` | 随机抖动最大幅度（相对延迟的比例） | `0.1` |
| `API_TIMEOUT` | API请求超时时间（秒） | `300` |
| `FALLBACK_ENDPOINTS` | 备用服务端点列表，多个用逗号分隔 | - |
| `FALLBACK_API_KEYS` | 备用服务API密钥列表，多个用逗号分隔 | - |
| `FALLBACK_MODEL_NAMES` | 备用服务模型名称列表，多个用逗号分隔 | - |

## Markdown 处理

服务默认会自动清理 AI 返回内容中的 Markdown 代码块标记，保留内部内容。这个功能可以通过 `.env` 文件中的 `CLEAN_CODE_BLOCKS` 环境变量进行控制：

```
# 启用清理功能（默认）
CLEAN_CODE_BLOCKS=true

# 禁用清理功能
CLEAN_CODE_BLOCKS=false
```

当设置为 `true` 时，服务器会自动移除类似以下格式的代码块标记：

````
```markdown
实际内容
```
````

处理后只保留 `实际内容` 部分。这个处理是在服务端完成的，客户端无需额外处理。

## 提示词定制

系统提示词存储在 `prompt.txt` 文件中，用于指导 AI 如何处理图像。默认的提示词为：

```
使用 OCR 从此图片中提取文本，并仅将提取的文本以 Markdown 格式输出，不添加任何额外的评论或解释。
```

您可以根据需要修改此文件以调整 AI 的行为，例如：
- 指定特定的输出格式
- 要求 AI 关注文档中的特定内容
- 设置更详细的处理规则

## 使用限制

- 文件大小限制：默认最大 100MB
- 处理时间：处理大型 PDF 文件可能需要较长时间，特别是页数较多时
- API 调用成本：每次处理页面都会调用 OpenAI API，可能产生费用，请注意控制

## 部署建议

- 对于生产环境，建议使用 WSGI 服务器（如 Gunicorn、uWSGI 等）
- 考虑添加用户认证和限流机制，防止滥用
- 对于高并发场景，可考虑使用消息队列进行任务管理

## 健壮性设计

本服务使用多种策略来确保API请求的稳定性和可靠性：

### 自动重试机制

当API请求失败时，系统会自动进行重试：

- 使用指数退避算法逐渐增加重试间隔，减轻对服务的压力
- 在重试间隔中添加随机抖动，防止多个并发请求同时重试导致的"雪崩效应"
- 可通过环境变量控制最大重试次数和重试延迟参数

示例重试配置：
```