from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import tempfile
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import base64
from PIL import Image
import io
import re
from utils.utils import send_to_openai

# 加载 .env 文件中的环境变量
load_dotenv()

# 初始化 Flask 应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制上传文件大小为 100m

# 从 .env 文件中读取配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ENDPOINT = os.getenv('OPENAI_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
CONCURRENCY = int(os.getenv('CONCURRENCY', '5'))  # 默认并发数为 5
DPI = int(os.getenv('DPI', '300'))  # 默认分辨率为 300 DPI
CLEAN_CODE_BLOCKS = os.getenv('CLEAN_CODE_BLOCKS', 'true').lower() == 'true'  # 默认清理代码块标记

# 读取提示词文件
with open('prompt.txt', 'r', encoding='utf-8') as f:
    PROMPT = f.read().strip()

def clean_markdown_code_blocks(text):
    """
    清理markdown文本中的代码块标记
    
    处理所有的markdown代码块，保留内部内容
    
    参数:
        text (str): 原始markdown文本
        
    返回:
        str: 处理后的markdown文本
    """
    # 去除首尾空白字符
    text = text.strip()
    
    # 使用正则表达式找到所有的代码块并提取其内容
    pattern = r'```(?:markdown)?\s*([\s\S]*?)```'
    
    def replace_code_block(match):
        # 提取代码块内的内容并返回
        return match.group(1).strip()
    
    # 替换所有符合模式的代码块
    processed_text = re.sub(pattern, replace_code_block, text)
    
    return processed_text

@app.route('/ocr', methods=['POST'])
def ocr():
    # 检查是否上传了文件
    if 'file' not in request.files:
        return jsonify({'error': '未上传文件'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '未选择文件'}), 400
    if not file.filename.lower().endswith('.pdf'):
        return jsonify({'error': '上传的文件不是 PDF 格式'}), 400

    # 使用临时目录处理文件
    with tempfile.TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, 'input.pdf')
        file.save(pdf_path)
        
        # 将 PDF 转换为图片，控制分辨率
        images = convert_from_path(pdf_path, dpi=DPI, fmt='jpeg')

        # 使用并发处理图片
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = []
            for page_num, image in enumerate(images):
                future = executor.submit(process_page, page_num, image, PROMPT)
                futures.append(future)
            # 按页面顺序收集结果
            results = [future.result() for future in futures]

        # 将所有页面结果整合为 Markdown 内容
        md_content = '\n\n'.join(results)
        
        # 根据环境变量设置决定是否清理代码块标记
        if CLEAN_CODE_BLOCKS:
            md_content = clean_markdown_code_blocks(md_content)
            
        return md_content, 200, {'Content-Type': 'text/markdown'}

def process_page(page_num, image, prompt):
    """处理单页图片，调用 OpenAI 视觉模型进行 OCR"""
    base64_image = image_to_base64(image)
    messages = [
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }
    ]
    try:
        response = send_to_openai(messages)
        # 兼容不同的响应格式
        if 'choices' in response and len(response['choices']) > 0:
            if 'message' in response['choices'][0]:
                return response['choices'][0]['message']['content']
            elif 'delta' in response['choices'][0] and 'content' in response['choices'][0]['delta']:
                return response['choices'][0]['delta']['content']
        # 如果找不到预期的格式，返回完整响应用于调试
        print(f"API 响应结构异常: {response}")
        return f"API 响应格式异常，请检查日志"
    except Exception as e:
        print(f"处理图像时出错: {str(e)}")
        return f"处理失败: {str(e)}"

def image_to_base64(image):
    """将 PIL 图片转换为 base64 编码的字符串"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)  # 使用 JPEG 格式，质量 85
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=50000) 