import requests
from dotenv import load_dotenv
import os
import time
import random

# 加载环境变量
load_dotenv()

# 获取 OpenAI 配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ENDPOINT = os.getenv('OPENAI_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')

# 获取重试配置和备用服务配置
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))  # 最大重试次数
RETRY_DELAY = float(os.getenv('RETRY_DELAY', '2.0'))  # 初始重试延迟（秒）
RETRY_BACKOFF = float(os.getenv('RETRY_BACKOFF', '1.5'))  # 重试延迟的指数增长因子
RETRY_JITTER = float(os.getenv('RETRY_JITTER', '0.1'))  # 随机抖动最大幅度（相对于延迟的比例）

# 获取备用服务配置
FALLBACK_ENDPOINTS = os.getenv('FALLBACK_ENDPOINTS', '').split(',')  # 备用服务端点列表
FALLBACK_API_KEYS = os.getenv('FALLBACK_API_KEYS', '').split(',')  # 备用API密钥列表
FALLBACK_MODEL_NAMES = os.getenv('FALLBACK_MODEL_NAMES', '').split(',')  # 备用模型名称列表

# 清理空字符串
FALLBACK_ENDPOINTS = [endpoint.strip() for endpoint in FALLBACK_ENDPOINTS if endpoint.strip()]
FALLBACK_API_KEYS = [key.strip() for key in FALLBACK_API_KEYS if key.strip()]
FALLBACK_MODEL_NAMES = [model.strip() for model in FALLBACK_MODEL_NAMES if model.strip()]

def send_to_openai(messages, current_retry=0, fallback_index=0):
    """
    发送请求到 OpenAI API，支持自动重试和备用服务
    
    参数:
        messages: 消息列表
        current_retry: 当前重试次数
        fallback_index: 当前使用的备用服务索引
        
    返回:
        API响应的JSON对象
    """
    # 确定当前使用的配置
    if fallback_index == 0:
        # 使用主要配置
        api_key = OPENAI_API_KEY
        endpoint = OPENAI_ENDPOINT
        model_name = MODEL_NAME
    elif fallback_index <= len(FALLBACK_ENDPOINTS):
        # 使用备用配置
        idx = fallback_index - 1
        api_key = FALLBACK_API_KEYS[idx] if idx < len(FALLBACK_API_KEYS) else OPENAI_API_KEY
        endpoint = FALLBACK_ENDPOINTS[idx]
        model_name = FALLBACK_MODEL_NAMES[idx] if idx < len(FALLBACK_MODEL_NAMES) else MODEL_NAME
        print(f"使用备用服务 #{fallback_index}: {endpoint}")
    else:
        # 所有备用服务都已尝试失败
        raise Exception("所有服务请求均失败，无法完成操作")
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    # 为 API 构建正确的请求格式
    data = {
        'model': model_name,
        'messages': messages,
        'temperature': 0,  # 确保输出一致性
        'max_tokens': 4096
    }
    
    # 检查请求中是否包含图像，这需要特殊处理
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            # 确保 content 列表中的图像 URL 格式正确
            for item in msg.get("content", []):
                if item.get("type") == "image_url" and "image_url" in item:
                    # 图像已经按正确格式处理，无需修改
                    pass
    
    print(f"发送到 API ({endpoint}) 的请求:")
    print(f"模型: {model_name}")
    print(f"消息数量: {len(messages)}")
    print(f"当前重试次数: {current_retry}/{MAX_RETRIES}, 备用服务索引: {fallback_index}/{len(FALLBACK_ENDPOINTS)}")
    
    try:
        response = requests.post(endpoint, headers=headers, json=data, timeout=60)
        
        if response.status_code != 200:
            print(f"API 错误状态码: {response.status_code}")
            print(f"API 错误响应: {response.text}")
            
            # 决定是重试当前服务还是尝试备用服务
            return handle_request_failure(messages, current_retry, fallback_index, response)
            
        return response.json()
    except requests.Timeout:
        print(f"API 请求超时")
        return handle_request_failure(messages, current_retry, fallback_index)
    except requests.RequestException as e:
        print(f"API 请求异常: {str(e)}")
        return handle_request_failure(messages, current_retry, fallback_index)
    except Exception as e:
        print(f"未预期的异常: {str(e)}")
        return handle_request_failure(messages, current_retry, fallback_index)

def handle_request_failure(messages, current_retry, fallback_index, response=None):
    """
    处理请求失败逻辑，决定是重试还是切换到备用服务
    
    参数:
        messages: 消息列表
        current_retry: 当前重试次数
        fallback_index: 当前使用的备用服务索引
        response: 失败的响应对象（如果有）
        
    返回:
        重试或备用服务的API响应
    """
    # 检查是否可以重试当前服务
    if current_retry < MAX_RETRIES:
        # 计算重试延迟时间（指数退避 + 随机抖动）
        delay = RETRY_DELAY * (RETRY_BACKOFF ** current_retry)
        jitter = delay * RETRY_JITTER * random.uniform(-1, 1)
        wait_time = delay + jitter
        
        print(f"将在 {wait_time:.2f} 秒后重试请求...")
        time.sleep(wait_time)
        
        # 重试当前服务
        return send_to_openai(messages, current_retry + 1, fallback_index)
    
    # 如果重试次数已用完，尝试使用备用服务
    if fallback_index < len(FALLBACK_ENDPOINTS):
        print(f"切换到备用服务 #{fallback_index + 1}")
        return send_to_openai(messages, 0, fallback_index + 1)
    
    # 如果所有选项都已尝试，抛出异常
    error_msg = "所有服务请求均失败"
    if response:
        error_msg += f"，最后错误: HTTP {response.status_code} - {response.text}"
    raise Exception(error_msg) 