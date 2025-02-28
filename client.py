#!/usr/bin/env python
# -*- coding: utf-8 -*-

import requests
import argparse
import os
import sys
import time
import re
import chardet

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

def convert_pdf_to_markdown(pdf_path, server_url="http://localhost:5000/ocr", output_path=None):
    """
    将PDF文件转换为Markdown格式
    
    参数:
        pdf_path (str): PDF文件的路径
        server_url (str): OCR API的URL
        output_path (str, optional): 输出文件的路径，默认为PDF文件名加'.md'后缀
        
    返回:
        str: 保存的Markdown文件路径
    """
    if not os.path.exists(pdf_path):
        print(f"错误: 文件 '{pdf_path}' 不存在")
        sys.exit(1)
        
    if not pdf_path.lower().endswith('.pdf'):
        print(f"错误: 文件 '{pdf_path}' 不是PDF格式")
        sys.exit(1)
    
    # 默认输出文件路径
    if output_path is None:
        base_name = os.path.splitext(pdf_path)[0]
        output_path = f"{base_name}.md"
    
    # 打印开始处理信息
    print(f"正在处理PDF文件: {pdf_path}")
    print(f"输出将保存至: {output_path}")
    
    start_time = time.time()
    
    try:
        # 准备文件上传
        with open(pdf_path, 'rb') as f:
            files = {'file': (os.path.basename(pdf_path), f, 'application/pdf')}
            
            # 发送请求
            print("正在发送请求到OCR服务器，这可能需要一些时间...")
            response = requests.post(server_url, files=files)
            
            # 检查响应状态
            if response.status_code != 200:
                print(f"错误: 服务器返回状态码 {response.status_code}")
                print(f"错误信息: {response.text}")
                sys.exit(1)
            
            # 检测响应内容的编码
            response_content = response.content
            detected = chardet.detect(response_content)
            print(f"检测到的响应编码: {detected['encoding']}, 置信度: {detected['confidence']}")
            
            # 尝试使用检测到的编码解码内容，如果失败则回退到utf-8
            try:
                if detected['encoding'] and detected['confidence'] > 0.5:
                    original_content = response_content.decode(detected['encoding'])
                else:
                    original_content = response.text  # 使用默认编码
            except UnicodeDecodeError:
                print(f"无法使用检测到的编码({detected['encoding']})解码内容，回退到utf-8")
                original_content = response.text
            
            # 处理响应内容，删除markdown代码块标记
            content = clean_markdown_code_blocks(original_content)
            
            # 如果内容被处理了，输出信息
            if content != original_content:
                print("检测到并移除了markdown代码块标记")
            
            # 确保内容是有效的UTF-8
            content = content.encode('utf-8', errors='replace').decode('utf-8')
            
            # 保存Markdown内容
            with open(output_path, 'w', encoding='utf-8') as out_file:
                out_file.write(content)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"处理完成! 用时: {duration:.2f} 秒")
            print(f"Markdown文件已保存至: {output_path}")
            
            return output_path
            
    except requests.exceptions.ConnectionError:
        print(f"错误: 无法连接到服务器 {server_url}")
        print("请确保OCR服务器正在运行.")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {str(e)}")
        sys.exit(1)

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='将PDF文件转换为Markdown格式')
    parser.add_argument('--pdf_path', '-p', default='./test.pdf', help='PDF文件的路径 (默认: ./test.pdf)')
    parser.add_argument('--url', default='http://localhost:50000/ocr', help='OCR API的URL (默认: http://localhost:50000/ocr)')
    parser.add_argument('--output', '-o', help='输出文件的路径 (默认: 与PDF同名但后缀为.md)')
    
    args = parser.parse_args()
    
    # 调用转换函数
    convert_pdf_to_markdown(args.pdf_path, args.url, args.output)

if __name__ == "__main__":
    main() 