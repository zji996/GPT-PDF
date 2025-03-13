# GPT-PDF 安装指南

## 依赖项

本项目依赖于以下组件：

1. Python 3.8+
2. uv 包管理器
3. Poppler (用于 PDF 处理)
4. 其它 Python 依赖 (通过 uv 管理)

## 安装步骤

### 1. 安装 Poppler (Windows)

`pdf2image` 库需要 Poppler 工具来处理 PDF 文件。在 Mac 上通常通过 Homebrew 安装，但在 Windows 上需要单独安装：

1. 下载 Windows 版本的 Poppler:
   - 从 [GitHub Releases](https://github.com/oschwartz10612/poppler-windows/releases/) 下载最新版本
   - 例如：`Release-24.02.0-0.zip` 或更新版本

2. 解压文件:
   - 将文件解压到固定位置，例如 `C:\Program Files\poppler`
   - 确保解压后有一个 `bin` 目录，其中包含多个可执行文件 (如 `pdfinfo.exe`, `pdftoppm.exe` 等)

3. 添加到环境变量:
   - 右键点击"此电脑" > 选择"属性"
   - 点击"高级系统设置" > "环境变量"
   - 在"系统变量"部分找到 `Path` 变量并点击"编辑"
   - 点击"新建"并添加 Poppler 的 bin 目录路径，例如 `C:\Program Files\poppler\bin`
   - 点击"确定"保存更改

4. 验证安装:
   - 打开新的命令提示符或 PowerShell 窗口
   - 运行 `pdfinfo -v` 或 `pdftoppm -v`，如果成功显示版本信息则表示安装正确

### 2. 安装 Python 依赖

使用 uv 安装项目依赖:

```bash
uv pip install -r requirements.txt
```

如果您单独需要安装 pdf2image:

```bash
uv pip install pdf2image
```

## 跨平台注意事项

- **Mac 用户**: 使用 Homebrew 安装 Poppler: `brew install poppler`
- **Linux 用户**: 使用系统包管理器安装 Poppler (例如 `apt install poppler-utils`)
- **Windows 用户**: 按照上述步骤安装 Poppler

## 故障排除

如果遇到以下错误:

```
pdf2image.exceptions.PDFInfoNotInstalledError: Unable to get page count. Is poppler installed and in PATH?
```

请检查:
1. Poppler 是否正确安装
2. Poppler 的 bin 目录是否已添加到系统 PATH 中
3. 在添加 PATH 后是否重启了命令行窗口/终端

## 其他注意事项

- 本项目在 Mac 上开发，但支持在 Windows 上运行，只需确保正确安装 Poppler
- 使用 uv 替代传统的 pip 可提供更快的依赖安装速度 