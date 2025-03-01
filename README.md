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
curl -X POST -F "file=@document.pdf" http://localhost:5000/ocr -o document.md
```

使用 Python requests:

```python
import requests

pdf_path = "document.pdf"
url = "http://localhost:5000/ocr"

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