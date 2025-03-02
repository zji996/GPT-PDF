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

## 配置

API 服务通过 `.env` 文件进行配置，支持以下配置项：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `OPENAI_API_KEY` | OpenAI API 密钥 | - |
| `OPENAI_ENDPOINT` | OpenAI API 端点 | `https://api.openai.com/v1/chat/completions` |
| `MODEL_NAME` | 使用的 OpenAI 模型 | `gpt-4o` |
| `CONCURRENCY` | 并发处理的页面数 | `5` |
| `DPI` | PDF 转图像的分辨率 | `300` |
| `CLEAN_CODE_BLOCKS` | 是否清理markdown代码块标记 | `true` |
| `MAX_RETRIES` | 同一服务的最大重试次数 | `3` |
| `RETRY_DELAY` | 初始重试延迟（秒） | `2.0` |
| `RETRY_BACKOFF` | 重试延迟的指数增长因子 | `1.5` |
| `RETRY_JITTER` | 随机抖动最大幅度（相对延迟的比例） | `0.1` |
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
MAX_RETRIES=3
RETRY_DELAY=2.0
RETRY_BACKOFF=1.5
RETRY_JITTER=0.1
```

这意味着：
- 第一次重试将在约2秒后进行（加随机抖动）
- 第二次重试将在约3秒后进行（2.0 * 1.5 = 3.0，加随机抖动）
- 第三次重试将在约4.5秒后进行（3.0 * 1.5 = 4.5，加随机抖动）

### 备用服务机制

当主服务多次重试仍然失败时，系统会自动切换到备用服务：

- 支持配置多个备用服务端点，按顺序尝试
- 每个备用服务可配置不同的API密钥和模型名称
- 如果所有配置的服务都失败，才会最终返回错误

备用服务配置示例：
```
FALLBACK_ENDPOINTS=https://api.backup1.com/v1/chat/completions,https://api.backup2.com/v1/chat/completions
FALLBACK_API_KEYS=backup1_api_key,backup2_api_key
FALLBACK_MODEL_NAMES=gpt-4-turbo,claude-3-sonnet
```

备用服务机制特别适用于以下场景：
- 主服务临时不可用时自动切换到备用服务
- 使用多个不同提供商的API服务以提高可用性
- 当一种模型不可用时，可以使用替代模型

### 健壮性工作流程

1. 服务首先尝试使用主要配置（OPENAI_API_KEY、OPENAI_ENDPOINT、MODEL_NAME）发送请求
2. 如果请求失败，会根据重试参数进行最多MAX_RETRIES次重试
3. 如果所有重试都失败，则会切换到第一个备用服务
4. 对每个备用服务也执行相同的重试逻辑
5. 只有当所有服务和重试都失败时，才会向用户返回错误

## 错误处理

API 会返回以下错误：

- `未上传文件`: 请求中没有包含文件
- `未选择文件`: 上传的文件名为空
- `上传的文件不是 PDF 格式`: 文件不是 PDF 格式

## 客户端工具

项目附带了一个命令行客户端工具 `client.py`，使用方法如下：

```bash
python client.py --pdf_path sample.pdf --output result.md
```

参数说明：
- `--pdf_path` 或 `-p`: PDF 文件路径，默认为 `./test.pdf`
- `--url`: API 服务器 URL，默认为 `http://localhost:50000/ocr`
- `--output` 或 `-o`: 输出文件路径，默认为与 PDF 同名但后缀为 .md

**注意**: 客户端不再负责清理Markdown代码块标记，此功能现已移至服务端，并可通过服务端的`CLEAN_CODE_BLOCKS`环境变量进行控制。客户端只负责接收服务端处理过的内容并保存文件，这样便简化了客户端逻辑。

## 错误与故障排除

1. **连接错误**：确保 API 服务正在运行，且端口未被防火墙阻止
2. **处理失败**：检查 PDF 文件是否损坏或受密码保护
3. **内存错误**：处理大型 PDF 时，可能需要增加服务器内存或降低并发数
4. **OpenAI API 错误**：确保 API 密钥有效，且未超出使用限制
5. **重试失败**：如果所有服务都返回错误，检查 API 密钥、端点配置和网络连接
6. **服务切换**：开启详细日志记录来观察重试和服务切换的过程 