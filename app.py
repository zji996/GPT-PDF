from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv
import tempfile
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import base64
from PIL import Image
import io
import re
import sys
import math
import psutil
import mmap
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
IMAGE_QUALITY = int(os.getenv('IMAGE_QUALITY', '85'))  # 图像质量参数
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '20'))  # PDF分批处理的批次大小
USE_MULTIPROCESSING = os.getenv('USE_MULTIPROCESSING', 'true').lower() == 'true'  # 是否使用多进程

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
        
        # 获取PDF页数以决定是否需要分批处理
        try:
            # 获取系统可用内存（以字节为单位）
            available_memory = psutil.virtual_memory().available
            # 估算每页PDF转图像后的内存占用（DPI越高占用越大）
            # 这是一个粗略估计，可能需要根据实际情况调整
            estimated_memory_per_page = (DPI / 100) ** 2 * 2 * 1024 * 1024  # 假设每页2MB基础内存，随DPI平方增长
            
            # 计算可以一次性处理的最大页数
            max_pages_per_batch = int(available_memory * 0.7 / estimated_memory_per_page)
            # 确保批次大小不超过设置的BATCH_SIZE
            max_pages_per_batch = min(max_pages_per_batch, BATCH_SIZE)
            # 确保至少处理1页
            max_pages_per_batch = max(max_pages_per_batch, 1)
            
            print(f"系统可用内存: {available_memory / (1024 * 1024 * 1024):.2f}GB")
            print(f"估计每页内存占用: {estimated_memory_per_page / (1024 * 1024):.2f}MB")
            print(f"单批最大页数: {max_pages_per_batch}")
            
            # 获取PDF总页数，使用pdf2image的功能
            total_pages = len(convert_from_path(pdf_path, dpi=72, first_page=1, last_page=1)) * 10  # 估算总页数
            # 尝试更精确地获取总页数
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    total_pages = len(pdf.pages)
            except:
                # 如果PyPDF2失败，使用全部转换一次来获取准确页数
                total_pages = len(convert_from_path(pdf_path, dpi=72))
                
            print(f"PDF总页数: {total_pages}")
            
            # 计算需要的批次数
            num_batches = math.ceil(total_pages / max_pages_per_batch)
            print(f"需要处理的批次数: {num_batches}")
            
            all_results = []
            
            # 分批处理PDF
            for batch in range(num_batches):
                start_page = batch * max_pages_per_batch + 1
                end_page = min((batch + 1) * max_pages_per_batch, total_pages)
                print(f"处理批次 {batch+1}/{num_batches}: 页面 {start_page}-{end_page}")
                
                # 转换当前批次的PDF页面为图像
                images = convert_from_path(
                    pdf_path, 
                    dpi=DPI, 
                    fmt='jpeg', 
                    first_page=start_page, 
                    last_page=end_page
                )
                
                # 处理当前批次的图像
                batch_results = process_images_batch(images, start_page)
                all_results.extend(batch_results)
                
                # 主动释放内存
                images = None
                batch_results = None
                import gc
                gc.collect()
            
            # 将所有页面结果整合为 Markdown 内容
            md_content = '\n\n'.join(all_results)
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'处理PDF文件时出错: {str(e)}'}), 500
        
        # 根据环境变量设置决定是否清理代码块标记
        if CLEAN_CODE_BLOCKS:
            md_content = clean_markdown_code_blocks(md_content)
            
        return md_content, 200, {'Content-Type': 'text/markdown'}

def process_images_batch(images, start_page):
    """处理一批图像，返回处理结果列表"""
    # 使用进程池进行并行处理
    if USE_MULTIPROCESSING and len(images) > 1:
        # 首先并行将图像转换为base64
        with ProcessPoolExecutor(max_workers=min(os.cpu_count(), len(images))) as executor:
            # 创建包含页码和图像的元组列表
            page_image_pairs = [(start_page + i - 1, img) for i, img in enumerate(images)]
            # 并行处理图像转base64
            base64_images = list(executor.map(process_image_to_base64, page_image_pairs))
        
        # 然后使用线程池调用API（API调用不适合多进程，因为有网络IO）
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = []
            for page_num, base64_image in base64_images:
                future = executor.submit(process_base64_image, page_num, base64_image, PROMPT)
                futures.append(future)
            # 按页面顺序收集结果
            results = [future.result() for future in futures]
    else:
        # 使用单线程或线程池处理
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            futures = []
            for i, image in enumerate(images):
                page_num = start_page + i - 1
                future = executor.submit(process_page, page_num, image, PROMPT)
                futures.append(future)
            # 按页面顺序收集结果
            results = [future.result() for future in futures]
    
    return results

def process_image_to_base64(page_image_tuple):
    """将图像转换为base64格式，适用于多进程处理"""
    page_num, image = page_image_tuple
    base64_str = image_to_base64_optimized(image)
    return (page_num, base64_str)

def process_base64_image(page_num, base64_image, prompt):
    """处理已经转换为base64的图像，调用API进行OCR"""
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

def process_page(page_num, image, prompt):
    """处理单页图片，调用 OpenAI 视觉模型进行 OCR"""
    base64_image = image_to_base64_optimized(image)
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

def image_to_base64_optimized(image):
    """将 PIL 图片转换为 base64 编码的字符串（优化版）"""
    try:
        # 使用内存映射优化大图像处理
        if image.width * image.height > 2000000:  # 200万像素以上的大图像
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                image.save(tmp_file, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
                tmp_file.flush()
                
                with open(tmp_file.name, 'rb') as f:
                    # 使用内存映射读取文件
                    with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                        img_bytes = mmapped_file.read()
                        
                # 删除临时文件
                os.unlink(tmp_file.name)
        else:
            # 小图像直接在内存中处理
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
            img_bytes = buffered.getvalue()
        
        return base64.b64encode(img_bytes).decode('utf-8')
    except Exception as e:
        print(f"图像转base64出错: {str(e)}")
        # 如果优化方法失败，回退到简单方法
        return image_to_base64(image)

def image_to_base64(image):
    """将 PIL 图片转换为 base64 编码的字符串（原始方法，作为备份）"""
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=IMAGE_QUALITY)
    img_bytes = buffered.getvalue()
    return base64.b64encode(img_bytes).decode('utf-8')

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=50000) 