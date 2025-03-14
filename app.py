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
import concurrent.futures
import threading
import time
import signal
import atexit
import traceback

# 全局线程池和进程池对象，便于在程序退出时关闭
_thread_pools = []
_process_pools = []

# 创建线程池的包装函数
def create_thread_pool(max_workers):
    pool = ThreadPoolExecutor(max_workers=max_workers)
    _thread_pools.append(pool)
    return pool

# 创建进程池的包装函数
def create_process_pool(max_workers):
    pool = ProcessPoolExecutor(max_workers=max_workers)
    _process_pools.append(pool)
    return pool

# 清理所有资源
def cleanup_resources():
    print("正在清理资源...")
    # 关闭所有线程池
    for pool in _thread_pools:
        try:
            pool.shutdown(wait=False)
        except:
            pass
    
    # 关闭所有进程池
    for pool in _process_pools:
        try:
            pool.shutdown(wait=False)
        except:
            pass
    
    print("资源清理完成")

# 注册信号处理函数
def setup_signal_handlers():
    # 注册SIGINT和SIGTERM信号处理
    for sig in [signal.SIGINT, signal.SIGTERM]:
        try:
            signal.signal(sig, lambda s, f: cleanup_and_exit())
        except (ValueError, AttributeError):
            # 某些环境下可能不支持信号处理
            pass

# 清理资源并退出
def cleanup_and_exit():
    print("接收到退出信号，正在安全退出...")
    cleanup_resources()
    sys.exit(0)

# 注册退出处理函数
atexit.register(cleanup_resources)

# 尝试设置信号处理
try:
    setup_signal_handlers()
except:
    print("无法设置信号处理函数，程序退出时可能不会完全清理资源")

# 加载 .env 文件中的环境变量
load_dotenv()

# 从 .env 文件中读取配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ENDPOINT = os.getenv('OPENAI_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')
CONCURRENCY = int(os.getenv('CONCURRENCY', '5'))  # 默认并发数为 5
MIN_CONCURRENCY = int(os.getenv('MIN_CONCURRENCY', '1'))  # 最小并发数
MAX_CONCURRENCY = int(os.getenv('MAX_CONCURRENCY', '20'))  # 最大并发数
DPI = int(os.getenv('DPI', '300'))  # 默认分辨率为 300 DPI
CLEAN_CODE_BLOCKS = os.getenv('CLEAN_CODE_BLOCKS', 'true').lower() == 'true'  # 默认清理代码块标记
IMAGE_QUALITY = int(os.getenv('IMAGE_QUALITY', '85'))  # 图像质量参数
BATCH_SIZE = int(os.getenv('BATCH_SIZE', '20'))  # PDF分批处理的批次大小
USE_MULTIPROCESSING = os.getenv('USE_MULTIPROCESSING', 'true').lower() == 'true'  # 是否使用多进程

# 添加全局计数器，用于监控并发请求
class ConcurrencyMonitor:
    def __init__(self):
        self.active_requests = 0
        self.max_observed = 0
        self.total_requests = 0
        self.completed_requests = 0
        self.lock = threading.Lock()
        
    def request_started(self):
        with self.lock:
            self.active_requests += 1
            self.total_requests += 1
            if self.active_requests > self.max_observed:
                self.max_observed = self.active_requests
                
    def request_completed(self):
        with self.lock:
            self.active_requests -= 1
            self.completed_requests += 1
            
    def get_stats(self):
        with self.lock:
            return {
                "active_requests": self.active_requests,
                "max_observed_concurrency": self.max_observed,
                "total_requests": self.total_requests,
                "completed_requests": self.completed_requests
            }

# 创建全局监控实例
concurrency_monitor = ConcurrencyMonitor()

# 添加自适应并发控制
class AdaptiveConcurrencyControl:
    def __init__(self, initial_concurrency=5, min_concurrency=1, max_concurrency=20):
        self.current_concurrency = initial_concurrency
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.response_times = []  # 存储最近的响应时间
        self.window_size = 10     # 滑动窗口大小
        self.lock = threading.Lock()
        self.last_adjustment_time = time.time()
        self.adjustment_interval = 10.0  # 调整间隔（秒）
        
    def record_response_time(self, response_time):
        """记录响应时间并调整并发度"""
        with self.lock:
            # 添加新的响应时间到窗口
            self.response_times.append(response_time)
            if len(self.response_times) > self.window_size:
                self.response_times.pop(0)  # 移除最旧的响应时间
                
            # 检查是否应该调整并发度
            current_time = time.time()
            if current_time - self.last_adjustment_time >= self.adjustment_interval:
                self._adjust_concurrency()
                self.last_adjustment_time = current_time
    
    def _adjust_concurrency(self):
        """基于响应时间调整并发度"""
        if not self.response_times:
            return  # 没有足够的数据进行调整
            
        avg_response_time = sum(self.response_times) / len(self.response_times)
        
        # 简单策略：如果平均响应时间增加，减少并发；如果响应时间减少，增加并发
        if len(self.response_times) >= 2:
            recent_times = self.response_times[-2:]
            trend = recent_times[1] - recent_times[0]
            
            if trend > 1.0:  # 响应时间增加超过1秒
                # 响应时间增加，减少并发
                self.current_concurrency = max(self.min_concurrency, self.current_concurrency - 1)
                print(f"响应时间增加 ({trend:.2f}s)，降低并发度至 {self.current_concurrency}")
            elif trend < -0.5:  # 响应时间减少超过0.5秒
                # 响应时间减少，增加并发
                self.current_concurrency = min(self.max_concurrency, self.current_concurrency + 1)
                print(f"响应时间减少 ({trend:.2f}s)，提高并发度至 {self.current_concurrency}")
        
        print(f"当前平均响应时间: {avg_response_time:.2f}s, 并发度: {self.current_concurrency}")
    
    def get_concurrency(self):
        """获取当前的并发度"""
        with self.lock:
            return self.current_concurrency

# 创建自适应并发控制实例
adaptive_concurrency = AdaptiveConcurrencyControl(
    initial_concurrency=CONCURRENCY,
    min_concurrency=MIN_CONCURRENCY,
    max_concurrency=MAX_CONCURRENCY
)

# 初始化 Flask 应用
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 限制上传文件大小为 100m

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
    temp_dir = None
    try:
        temp_dir = tempfile.mkdtemp()
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
            
            # 获取PDF总页数，优先使用PyPDF2方法
            total_pages = None
            try:
                import PyPDF2
                with open(pdf_path, 'rb') as f:
                    pdf = PyPDF2.PdfReader(f)
                    total_pages = len(pdf.pages)
                    print(f"使用PyPDF2获取到PDF总页数: {total_pages}")
            except Exception as pdf_error:
                print(f"使用PyPDF2获取PDF页数失败: {str(pdf_error)}，尝试使用pdf2image方法")
                try:
                    # 使用pdf2image的统计页数方法
                    total_pages = len(convert_from_path(pdf_path, dpi=72))
                    print(f"使用pdf2image获取到PDF总页数: {total_pages}")
                except Exception as img_error:
                    print(f"使用pdf2image获取PDF页数失败: {str(img_error)}")
                    return jsonify({'error': f'无法获取PDF页数，请检查文件是否损坏: {str(img_error)}'}), 400
            
            if not total_pages or total_pages <= 0:
                return jsonify({'error': '无效的PDF: 找不到页面'}), 400
                
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
                
                try:
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
                except Exception as batch_error:
                    print(f"处理批次 {batch+1} 时出错: {str(batch_error)}")
                    traceback.print_exc()
                    # 如果已经有一些处理结果，则继续处理；否则返回错误
                    if not all_results:
                        return jsonify({'error': f'处理PDF批次 {batch+1} 时出错: {str(batch_error)}'}), 500
                    print("有部分页面处理成功，继续处理其余页面")
            
            # 将所有页面结果整合为 Markdown 内容
            processed_results = []
            for i, result in enumerate([r for r in all_results if r]):
                page_num = i + 1
                if i > 0:  # Don't add page break before the first page
                    processed_results.append(f"\n\n<!-- PAGE BREAK -->\n\n## Page {page_num}\n\n---\n\n")
                processed_results.append(result)
            
            md_content = ''.join(processed_results)
            
            if not md_content.strip():
                return jsonify({'error': '处理完成，但未能提取有效内容'}), 500
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({'error': f'处理PDF文件时出错: {str(e)}'}), 500
        
        # 根据环境变量设置决定是否清理代码块标记
        if CLEAN_CODE_BLOCKS:
            md_content = clean_markdown_code_blocks(md_content)
            
        return md_content, 200, {'Content-Type': 'text/markdown'}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'处理过程中发生未预期错误: {str(e)}'}), 500
    finally:
        # 确保临时目录被删除
        if temp_dir and os.path.exists(temp_dir):
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except:
                print(f"清理临时目录 {temp_dir} 失败")
        # 强制进行垃圾回收
        import gc
        gc.collect()

def process_images_batch(images, start_page):
    """处理一批图像，返回处理结果列表"""
    # 使用进程池进行并行处理
    if USE_MULTIPROCESSING and len(images) > 1 and os.name != 'nt':  # 在Windows上禁用多进程处理
        # 首先并行将图像转换为base64
        base64_images = []
        # 限制进程数，避免创建过多进程
        max_workers = min(os.cpu_count() or 4, len(images), 4)  # 最多4个进程
        
        with create_process_pool(max_workers=max_workers) as executor:
            # 创建包含页码和图像的元组列表
            page_image_pairs = [(start_page + i - 1, img) for i, img in enumerate(images)]
            # 并行处理图像转base64
            base64_images = list(executor.map(process_image_to_base64, page_image_pairs))
    else:
        # 在Windows上或不使用多进程时，串行处理
        base64_images = []
        for i, image in enumerate(images):
            page_num = start_page + i - 1
            base64_str = image_to_base64_optimized(image)
            base64_images.append((page_num, base64_str))
    
    # API调用部分使用线程池
    total_tasks = len(base64_images)
    results = [None] * total_tasks  # 预分配结果列表
    active_futures = {}  # 跟踪活跃的future对象
    next_task_index = 0  # 下一个要处理的任务索引
    
    # 限制线程池的最大工作线程数
    max_thread_workers = min(adaptive_concurrency.max_concurrency, 10)  # 最多10个线程
    
    with create_thread_pool(max_workers=max_thread_workers) as executor:
        # 初始化并发窗口
        current_concurrency = adaptive_concurrency.get_concurrency()
        while next_task_index < total_tasks and len(active_futures) < current_concurrency:
            page_num, base64_image = base64_images[next_task_index]
            future = executor.submit(process_base64_image, page_num, base64_image, PROMPT)
            active_futures[future] = next_task_index
            next_task_index += 1
            
        # 一旦有任务完成就立即启动新任务，保持并发数
        while active_futures:
            try:
                # 等待任何一个任务完成，添加超时防止无限等待
                done, _ = concurrent.futures.wait(
                    active_futures.keys(),
                    return_when=concurrent.futures.FIRST_COMPLETED,
                    timeout=60  # 60秒超时
                )
                
                # 如果没有任务完成，可能是因为超时
                if not done:
                    print("等待任务完成超时，继续等待...")
                    continue
                
                # 处理完成的任务
                for future in done:
                    task_index = active_futures[future]
                    try:
                        results[task_index] = future.result()
                    except Exception as e:
                        print(f"任务处理异常: {str(e)}")
                        results[task_index] = f"处理出错: {str(e)}"
                    
                    del active_futures[future]
                    
                    # 如果还有未启动的任务，立即启动
                    if next_task_index < total_tasks:
                        # 动态获取当前的并发限制
                        current_concurrency = adaptive_concurrency.get_concurrency()
                        if len(active_futures) < current_concurrency:
                            page_num, base64_image = base64_images[next_task_index]
                            new_future = executor.submit(process_base64_image, page_num, base64_image, PROMPT)
                            active_futures[new_future] = next_task_index
                            next_task_index += 1
            except KeyboardInterrupt:
                print("检测到键盘中断，正在安全退出...")
                # 取消所有未完成的任务
                for future in active_futures:
                    future.cancel()
                # 资源会在退出时由清理函数处理
                raise
    
    return results

def process_image_to_base64(page_image_tuple):
    """将图像转换为base64格式，适用于多进程处理"""
    page_num, image = page_image_tuple
    base64_str = image_to_base64_optimized(image)
    return (page_num, base64_str)

def process_base64_image(page_num, base64_image, prompt):
    """处理已经转换为base64的图像，调用API进行OCR"""
    concurrency_monitor.request_started()
    start_time = time.time()
    try:
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
    finally:
        end_time = time.time()
        response_time = end_time - start_time
        adaptive_concurrency.record_response_time(response_time)
        concurrency_monitor.request_completed()
        print(f"页面 {page_num} 处理完成，耗时: {response_time:.2f}秒")

def process_page(page_num, image, prompt):
    """处理单页图片，调用 OpenAI 视觉模型进行 OCR"""
    concurrency_monitor.request_started()
    start_time = time.time()
    try:
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
    finally:
        end_time = time.time()
        response_time = end_time - start_time
        adaptive_concurrency.record_response_time(response_time)
        concurrency_monitor.request_completed()
        print(f"页面 {page_num} 处理完成，耗时: {response_time:.2f}秒")

def image_to_base64_optimized(image):
    """将 PIL 图片转换为 base64 编码的字符串（优化版）"""
    try:
        # 在Windows上，避免使用临时文件和mmap方式，直接在内存中处理
        if os.name == 'nt':  # Windows系统
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
            img_bytes = buffered.getvalue()
            return base64.b64encode(img_bytes).decode('utf-8')
        
        # 非Windows系统使用原有的优化逻辑
        # 使用内存映射优化大图像处理
        if image.width * image.height > 2000000:  # 200万像素以上的大图像
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                image.save(tmp_file, format="JPEG", quality=IMAGE_QUALITY, optimize=True)
                tmp_file.flush()
                tmp_file.close()  # 确保文件已关闭
                
                try:
                    with open(tmp_file.name, 'rb') as f:
                        # 使用内存映射读取文件
                        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mmapped_file:
                            img_bytes = mmapped_file.read()
                finally:
                    # 确保无论如何都删除临时文件
                    try:
                        os.unlink(tmp_file.name)
                    except:
                        pass  # 如果删除失败，忽略错误
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

# 添加监控API端点
@app.route('/stats', methods=['GET'])
def get_stats():
    """返回当前并发请求的统计信息"""
    stats = concurrency_monitor.get_stats()
    stats.update({
        "current_concurrency_limit": adaptive_concurrency.get_concurrency(),
        "min_concurrency": adaptive_concurrency.min_concurrency,
        "max_concurrency": adaptive_concurrency.max_concurrency,
        "recent_response_times": adaptive_concurrency.response_times
    })
    return jsonify(stats), 200

if __name__ == '__main__':
    app.run(threaded=True, host='0.0.0.0', port=50000) 