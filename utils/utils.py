import requests
from dotenv import load_dotenv
import os

# 加载环境变量
load_dotenv()

# 获取 OpenAI 配置
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_ENDPOINT = os.getenv('OPENAI_ENDPOINT', 'https://api.openai.com/v1/chat/completions')
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-4o')

def send_to_openai(messages):
    """发送请求到 OpenAI API"""
    headers = {
        'Authorization': f'Bearer {OPENAI_API_KEY}',
        'Content-Type': 'application/json'
    }
    
    # 为 Siliconflow API 构建正确的请求格式
    data = {
        'model': MODEL_NAME,
        'messages': messages,
        'temperature': 0,  # 确保输出一致性
        'max_tokens': 4096  # 减小 token 数以避免超过限制
    }
    
    # 检查请求中是否包含图像，这需要特殊处理
    for msg in messages:
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            # 确保 content 列表中的图像 URL 格式正确
            for item in msg.get("content", []):
                if item.get("type") == "image_url" and "image_url" in item:
                    # 图像已经按正确格式处理，无需修改
                    pass
    
    print(f"发送到 API ({OPENAI_ENDPOINT}) 的请求:")
    print(f"模型: {MODEL_NAME}")
    print(f"消息数量: {len(messages)}")
    
    try:
        response = requests.post(OPENAI_ENDPOINT, headers=headers, json=data)
        if response.status_code != 200:
            print(f"API 错误状态码: {response.status_code}")
            print(f"API 错误响应: {response.text}")
        response.raise_for_status()  # 如果请求失败，抛出异常
        return response.json()
    except Exception as e:
        print(f"API 请求异常: {str(e)}")
        raise 