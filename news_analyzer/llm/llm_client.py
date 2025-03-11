"""
LLM客户端

优化的LLM客户端，支持流式输出和自动加载。
"""

import os
import json
import logging
import requests
import time
import threading
from typing import Callable, Dict, List, Optional, Union, Any


class LLMClient:
    """LLM客户端类"""
    
    def __init__(self, api_key=None, api_url=None, model=None):
        """初始化LLM客户端
        
        Args:
            api_key: API密钥，如果为None则尝试从环境变量获取
            api_url: API URL，如果为None则使用默认值
            model: 模型名称，如果为None则使用默认值
        """
        self.logger = logging.getLogger('news_analyzer.llm.client')
        
        # 设置API密钥（优先使用参数，其次使用环境变量）
        self.api_key = api_key or os.environ.get('LLM_API_KEY', '')
        
        # 设置API URL
        self.api_url = api_url or os.environ.get(
            'LLM_API_URL', 
            'https://api.openai.com/v1/chat/completions'
        )
        
        # 设置模型
        self.model = model or os.environ.get('LLM_MODEL', 'gpt-3.5-turbo')
        
        # 默认参数
        self.temperature = 0.7
        self.max_tokens = 2048
        self.timeout = 60
        
        # 确定API类型
        self.api_type = self._determine_api_type()
        self.logger.info(f"初始化LLM客户端，API类型: {self.api_type}, 模型: {self.model}")
    
    def _determine_api_type(self):
        """确定API类型"""
        url_lower = self.api_url.lower()
        if "openai.com" in url_lower:
            return "openai"
        elif "anthropic.com" in url_lower:
            return "anthropic"
        elif "localhost" in url_lower or "127.0.0.1" in url_lower:
            return "ollama"
        else:
            return "generic"
    
    def analyze_news(self, news_item, analysis_type='摘要'):
        """分析新闻
        
        Args:
            news_item: 新闻数据字典
            analysis_type: 分析类型，默认为'摘要'
            
        Returns:
            str: 格式化的分析结果HTML
        """
        if not news_item:
            raise ValueError("新闻数据不能为空")
        
        # If API key is not set, return mock analysis
        if not self.api_key:
            return self._mock_analysis(news_item, analysis_type)
        
        # 获取提示词
        prompt = self._get_prompt(analysis_type, news_item)
        
        try:
            # 调用API
            headers = self._get_headers()
            
            # 根据API类型准备请求数据
            if self.api_type == "anthropic":
                data = self._prepare_anthropic_request(prompt)
            elif self.api_type == "ollama":
                data = self._prepare_ollama_request(prompt)
            else:  # OpenAI或通用格式
                data = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': prompt}],
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens
                }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # 提取回复内容
            content = self._extract_content_from_response(result)
            
            if not content:
                raise ValueError("API返回的内容为空")
            
            return self._format_analysis_result(content, analysis_type)
            
        except Exception as e:
            self.logger.error(f"调用LLM API失败: {str(e)}")
            raise
    
    def _prepare_anthropic_request(self, prompt):
        """准备Anthropic API请求
        
        Args:
            prompt: 提示文本
            
        Returns:
            dict: 请求数据
        """
        return {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
    
    def _prepare_ollama_request(self, prompt):
        """准备Ollama API请求
        
        Args:
            prompt: 提示文本
            
        Returns:
            dict: 请求数据
        """
        return {
            'model': self.model,
            'messages': [{'role': 'user', 'content': prompt}],
            'options': {
                'temperature': self.temperature,
                'num_predict': self.max_tokens
            }
        }
    
    def _extract_content_from_response(self, result):
        """从不同API响应中提取内容
        
        Args:
            result: API响应的JSON数据
            
        Returns:
            str: 提取的文本内容
        """
        if self.api_type == "anthropic":
            # 处理各种Claude API格式
            if 'content' in result and isinstance(result['content'], list):
                for item in result['content']:
                    if item.get('type') == 'text':
                        return item.get('text', '')
            return result.get('content', [{}])[0].get('text', '')
        elif self.api_type == "ollama":
            return result.get('response', '')
        else:  # OpenAI或通用格式
            return result.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    def chat(self, messages, context="", stream=True, callback=None):
        """聊天功能
        
        Args:
            messages: 聊天历史消息列表
            context: 上下文文本，默认为空
            stream: 是否使用流式输出，默认为True
            callback: 用于接收流式更新的回调函数
            
        Returns:
            str: 非流式模式下的回复内容，流式模式下为None
        """
        # 处理无API密钥情况
        if not self.api_key:
            mock_response = "API密钥未设置，请在设置中配置有效的API密钥。"
            
            if callback:
                callback(mock_response, True)
            return mock_response
        
        # 准备消息列表
        processed_messages = []
        
        # 添加系统消息
        if context:
            system_msg = {
                'role': 'system',
                'content': f"你是一个新闻分析助手。以下是相关的新闻信息：\n\n{context}"
            }
            processed_messages.append(system_msg)
        else:
            processed_messages.append({
                'role': 'system',
                'content': "你是一个专业的新闻分析助手，可以回答各种问题。"
            })
        
        # 添加历史消息
        processed_messages.extend(messages)
        
        # 如果是流式模式且有回调函数
        if stream and callback:
            # 启动线程进行流式处理
            thread = threading.Thread(
                target=self._simulated_stream_response,
                args=(processed_messages, callback)
            )
            thread.daemon = True
            thread.start()
            return None
        else:
            # 非流式请求
            response = self._send_chat_request(processed_messages)
            if callback:
                callback(response, True)
            return response
    
    def _simulated_stream_response(self, messages, callback):
        """模拟流式响应
        
        此方法先获取完整回复，然后逐字符发送以模拟打字效果
        
        Args:
            messages: 消息列表
            callback: 回调函数，用于接收更新
        """
        try:
            # 先获取完整响应
            full_response = self._send_chat_request(messages)
            
            # 分段发送响应，模拟流式效果
            collected_message = ""
            
            # 定义每次发送的字符数和延迟
            chunk_size = 5  # 每次发送字符数
            delay = 0.05    # 每次发送后的延迟(秒)
            
            # 分段发送
            for i in range(0, len(full_response), chunk_size):
                chunk = full_response[i:i+chunk_size]
                collected_message += chunk
                
                # 发送部分响应（未完成）
                callback(collected_message, False)
                
                # 添加延迟模拟打字效果
                time.sleep(delay)
            
            # 发送最终完整响应
            callback(full_response, True)
        
        except Exception as e:
            self.logger.error(f"流式处理失败: {str(e)}")
            error_message = f"""
            <div style="color: #d32f2f; font-weight: bold;">
                处理失败: {str(e)}
            </div>
            <div>
                请检查API设置和网络连接。可能需要在语言模型设置中重新配置API密钥和URL。
            </div>
            """
            callback(error_message, True)
    
    def _stream_chat_response(self, messages, callback):
        """处理真实流式聊天响应（未使用）
        
        此方法尝试使用API的真实流式响应功能
        但由于兼容性问题，现在使用模拟流式响应代替
        
        Args:
            messages: 消息列表
            callback: 回调函数
        """
        try:
            headers = self._get_headers()
            
            # 根据API类型准备请求数据
            if self.api_type == "anthropic":
                data = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'stream': True
                }
            elif self.api_type == "ollama":
                data = {
                    'model': self.model,
                    'messages': messages,
                    'stream': True,
                    'options': {
                        'temperature': self.temperature,
                        'num_predict': self.max_tokens
                    }
                }
            else:  # OpenAI或通用格式
                data = {
                    'model': self.model,
                    'messages': messages,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'stream': True
                }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                stream=True,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            
            # 处理流式响应
            collected_message = ""
            
            for chunk in response.iter_lines():
                if not chunk:
                    continue
                
                chunk = chunk.decode('utf-8', errors='ignore')
                content = ""
                is_done = False
                
                # 处理各种API格式的流式响应
                try:
                    if self.api_type == "openai":
                        if chunk.startswith("data: "):
                            if chunk == "data: [DONE]":
                                is_done = True
                                continue
                                
                            json_str = chunk[6:]
                            chunk_data = json.loads(json_str)
                            content = chunk_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                    
                    elif self.api_type == "anthropic":
                        if chunk.startswith("data: "):
                            if chunk == "data: [DONE]":
                                is_done = True
                                continue
                                
                            json_str = chunk[6:]
                            chunk_data = json.loads(json_str)
                            
                            if 'delta' in chunk_data:
                                content = chunk_data['delta'].get('text', '')
                            elif 'completion' in chunk_data:
                                content = chunk_data['completion']
                    
                    elif self.api_type == "ollama":
                        chunk_data = json.loads(chunk)
                        content = chunk_data.get('response', '')
                        is_done = chunk_data.get('done', False)
                
                except (json.JSONDecodeError, KeyError) as e:
                    self.logger.error(f"解析流式响应失败: {str(e)}, chunk: {chunk}")
                    continue
                
                # 如果有内容，更新累积消息并回调
                if content:
                    collected_message += content
                    callback(collected_message, False)
                
                # 如果响应完成，发送最终回调
                if is_done:
                    callback(collected_message, True)
                    break
            
            # 确保至少一次完成回调
            callback(collected_message, True)
                
        except Exception as e:
            self.logger.error(f"流式处理失败: {str(e)}")
            error_message = f"""
            <div style="color: #d32f2f; font-weight: bold;">
                处理失败: {str(e)}
            </div>
            <div>
                请检查API设置和网络连接。可能需要在语言模型设置中重新配置API密钥和URL。
            </div>
            """
            callback(error_message, True)
    
    def _send_chat_request(self, messages):
        """发送聊天请求
        
        Args:
            messages: 消息列表
            
        Returns:
            str: 回复内容
        """
        headers = self._get_headers()
        
        # 根据API类型准备请求数据
        if self.api_type == "anthropic":
            data = {
                'model': self.model,
                'messages': messages,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        elif self.api_type == "ollama":
            data = {
                'model': self.model,
                'messages': messages,
                'options': {
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens
                }
            }
        else:  # OpenAI或通用格式
            data = {
                'model': self.model,
                'messages': messages,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens
            }
        
        response = requests.post(
            self.api_url,
            headers=headers,
            json=data,
            timeout=self.timeout
        )
        
        response.raise_for_status()
        result = response.json()
        
        # 提取内容
        content = self._extract_content_from_response(result)
        
        if not content:
            raise ValueError("API返回的内容为空")
        
        return content
    
    def test_connection(self):
        """测试API连接
        
        Returns:
            bool: 连接是否成功
        """
        if not self.api_key:
            return False
        
        try:
            headers = self._get_headers()
            
            # 根据API类型准备简单测试请求
            if self.api_type == "anthropic":
                data = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': '你好'}],
                    'max_tokens': 5
                }
            elif self.api_type == "ollama":
                data = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': '你好'}],
                    'options': {'num_predict': 5}
                }
            else:  # OpenAI或通用格式
                data = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': '你好'}],
                    'max_tokens': 5
                }
            
            response = requests.post(
                self.api_url,
                headers=headers,
                json=data,
                timeout=10
            )
            
            response.raise_for_status()
            
            # 尝试解析响应以确认有效性
            result = response.json()
            
            # 验证响应中是否有预期的字段
            if self.api_type == "anthropic":
                return 'content' in result or ('error' not in result)
            elif self.api_type == "ollama":
                return 'response' in result or ('message' in result)
            else:  # OpenAI或通用格式
                return 'choices' in result and len(result['choices']) > 0
                
        except Exception as e:
            self.logger.error(f"测试连接失败: {str(e)}")
            return False
    
    def _get_headers(self):
        """获取请求头
        
        Returns:
            dict: 请求头
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        # 根据API类型设置认证头
        if self.api_type == "anthropic":
            headers['x-api-key'] = self.api_key
            headers['anthropic-version'] = '2023-06-01'  # 使用适当的API版本
        elif self.api_type == "openai":
            headers['Authorization'] = f'Bearer {self.api_key}'
        else:
            # 通用Bearer认证方式
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    def _get_prompt(self, analysis_type, news_item):
        """获取提示词
        
        Args:
            analysis_type: 分析类型
            news_item: 新闻数据
            
        Returns:
            str: 提示词
        """
        title = news_item.get('title', '无标题')
        source = news_item.get('source_name', '未知来源')
        content = news_item.get('description', '无内容')
        pub_date = news_item.get('pub_date', '未知日期')
        
        if analysis_type == '摘要':
            return f"""
            请对以下新闻进行简明扼要的摘要分析。
            
            新闻标题: {title}
            新闻来源: {source}
            发布日期: {pub_date}
            
            新闻内容:
            {content}
            
            请提供:
            1. 一段200字以内的新闻摘要，包含关键信息点
            2. 3-5个要点列表，提炼新闻中最重要的信息
            """
        elif analysis_type == '深度分析':
            return f"""
            请对以下新闻进行深度分析。
            
            新闻标题: {title}
            新闻来源: {source}
            新闻内容:
            {content}
            
            请提供背景、影响和发展趋势分析。
            """
        elif analysis_type == '关键观点':
            return f"""
            请提取以下新闻中的关键观点和立场。
            
            新闻标题: {title}
            新闻来源: {source}
            新闻内容:
            {content}
            
            请分析:
            1. 新闻中表达的主要观点
            2. 各方立场和态度
            3. 潜在的倾向性或偏见
            """
        elif analysis_type == '事实核查':
            return f"""
            请对以下新闻进行事实核查分析。
            
            新闻标题: {title}
            新闻来源: {source}
            新闻内容:
            {content}
            
            请分析:
            1. 新闻中的主要事实声明
            2. 可能需要核实的关键信息点
            3. 潜在的误导或不准确之处
            """
        else:
            return f"""
            请对以下新闻进行{analysis_type}。
            
            新闻标题: {title}
            新闻来源: {source}
            新闻内容:
            {content}
            """
    
    def _format_analysis_result(self, content, analysis_type):
        """格式化分析结果

        Args:
            content: 原始内容
            analysis_type: 分析类型

        Returns:
            str: 格式化的HTML
        """
        # 先处理 content 中的换行符
        processed_content = content.replace('\n\n', '</p><p>').replace('\n- ', '</p><li>').replace('\n', '<br>')
        # 增强HTML格式化
        html = f'''
        <div style="font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; padding: 15px; line-height: 1.5;">
            <h2 style="color: #1976D2; border-bottom: 1px solid #E0E0E0; padding-bottom: 8px;">{analysis_type}结果</h2>
            <div style="padding: 10px 0;">
                {processed_content}
            </div>
        </div>
        '''
        return html
    
    def _mock_analysis(self, news_item, analysis_type):
        """模拟分析结果
        
        Args:
            news_item: 新闻数据
            analysis_type: 分析类型
            
        Returns:
            str: 模拟的HTML结果
        """
        title = news_item.get('title', '无标题')
        
        return f'''
        <div style="font-family: 'Segoe UI', 'Microsoft YaHei', sans-serif; padding: 15px; line-height: 1.5;">
            <h2 style="color: #1976D2; border-bottom: 1px solid #E0E0E0; padding-bottom: 8px;">{analysis_type}结果</h2>
            <div style="padding: 10px 0;">
                <p>这是对"{title}"的{analysis_type}。</p>
                <p style="color: #F44336;">由于未设置API密钥，这是一个模拟结果。请在设置中配置有效的LLM API密钥以获取真实分析。</p>
                <p>您可以通过顶部菜单栏的 <strong>工具 > 语言模型设置</strong> 来配置API。</p>
            </div>
        </div>
        '''