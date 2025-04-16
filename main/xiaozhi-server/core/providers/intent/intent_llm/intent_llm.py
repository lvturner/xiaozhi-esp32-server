from typing import List, Dict
from ..base import IntentProviderBase
from plugins_func.functions.play_music import initialize_music_handler
from config.logger import setup_logging
import re
import json
import hashlib
import time

TAG = __name__
logger = setup_logging()


class IntentProvider(IntentProviderBase):
    def __init__(self, config):
        super().__init__(config)
        self.llm = None
        self.prompt = self.get_intent_system_prompt()
        # 添加缓存管理
        self.intent_cache = {}  # 缓存意图识别结果
        self.cache_expiry = 600  # 缓存有效期10分钟
        self.cache_max_size = 100  # 最多缓存100个意图

    def get_intent_system_prompt(self) -> str:
        """
        根据配置的意图选项动态生成系统提示词
        Returns:
            格式化后的系统提示词
        """

        prompt = (
            "You are an intent recognition assistant. Analyze the user's last sentence and determine which of the following intent categories it belongs to:\n"
            "<start>"
            f"{str(self.intent_options)}"
            "<end>\n"
            "Processing steps:"
            "1. Think about the intent type and generate a function_call format"
            "\n\n"
            "Example return format:\n"
            '1. Play music intent: {"function_call": {"name": "play_music", "arguments": {"song_name": "song_name"}}}\n'
            '2. End conversation intent: {"function_call": {"name": "handle_exit_intent", "arguments": {"say_goodbye": "goodbye"}}}\n'
            '3. Get current date and time: {"function_call": {"name": "get_time"}}\n'
            '4. Continue chatting intent: {"function_call": {"name": "continue_chat"}}\n'
            "\n"
            "Notes:\n"
            '- Play music: If no song name is provided, set song_name to "random"\n'
            "- If no clear intent is detected, default to the continue chatting intent\n"
            "- Return only pure JSON, without any additional text\n"
            "\n"
            "Example analysis:\n"
            "```\n"
            "User: You're so funny\n"
            'Return: {"function_call": {"name": "continue_chat"}}\n'
            "```\n"
            "```\n"
            "User: What's the date today? What time is it?\n"
            'Return: {"function_call": {"name": "get_time"}}\n'
            "```\n"
            "```\n"
            "User: Let's talk again tomorrow\n"
            'Return: {"function_call": {"name": "handle_exit_intent"}}\n'
            "```\n"
            "```\n"
            "User: Play Mid-Autumn Moon\n"
            'Return: {"function_call": {"name": "play_music", "arguments": {"song_name": "Mid-Autumn Moon"}}}\n'
            "```\n"
            "```\n"
            "Available song names:\n"
            "```\n"
            "```\n"
            "User: Get headlines from hacker news"
            'Return: {"function_call": {"name": "get_stories", "arguments": {"limit": 10}}}\n'
        )
        return prompt

    def clean_cache(self):
        """清理过期缓存"""
        now = time.time()
        # 找出过期键
        expired_keys = [
            k
            for k, v in self.intent_cache.items()
            if now - v["timestamp"] > self.cache_expiry
        ]
        for key in expired_keys:
            del self.intent_cache[key]

        # 如果缓存太大，移除最旧的条目
        if len(self.intent_cache) > self.cache_max_size:
            # 按时间戳排序并保留最新的条目
            sorted_items = sorted(
                self.intent_cache.items(), key=lambda x: x[1]["timestamp"]
            )
            for key, _ in sorted_items[: len(sorted_items) - self.cache_max_size]:
                del self.intent_cache[key]

    async def detect_intent(self, conn, dialogue_history: List[Dict], text: str) -> str:
        if not self.llm:
            raise ValueError("LLM provider not set")

        # 记录整体开始时间
        total_start_time = time.time()

        # 打印使用的模型信息
        model_info = getattr(self.llm, "model_name", str(self.llm.__class__.__name__))
        logger.bind(tag=TAG).debug(f"使用意图识别模型: {model_info}")

        # 计算缓存键
        cache_key = hashlib.md5(text.encode()).hexdigest()

        # 检查缓存
        if cache_key in self.intent_cache:
            cache_entry = self.intent_cache[cache_key]
            # 检查缓存是否过期
            if time.time() - cache_entry["timestamp"] <= self.cache_expiry:
                cache_time = time.time() - total_start_time
                logger.bind(tag=TAG).debug(
                    f"使用缓存的意图: {cache_key} -> {cache_entry['intent']}, 耗时: {cache_time:.4f}秒"
                )
                return cache_entry["intent"]

        # 清理缓存
        self.clean_cache()

        # 构建用户最后一句话的提示
        msgStr = ""

        # 只使用最后两句即可
        if len(dialogue_history) >= 2:
            # 保证最少有两句话的时候处理
            msgStr += f"{dialogue_history[-2].role}: {dialogue_history[-2].content}\n"
        msgStr += f"{dialogue_history[-1].role}: {dialogue_history[-1].content}\n"

        msgStr += f"User: {text}\n"
        user_prompt = f"当前的对话如下：\n{msgStr}"
        music_config = initialize_music_handler(conn)
        music_file_names = music_config["music_file_names"]
        prompt_music = f"{self.prompt}\n<start>{music_file_names}\n<end>"
        logger.bind(tag=TAG).debug(f"User prompt: {prompt_music}")

        # 记录预处理完成时间
        preprocess_time = time.time() - total_start_time
        logger.bind(tag=TAG).debug(f"意图识别预处理耗时: {preprocess_time:.4f}秒")

        # 使用LLM进行意图识别
        llm_start_time = time.time()
        logger.bind(tag=TAG).debug(f"开始LLM意图识别调用, 模型: {model_info}")

        intent = self.llm.response_no_stream(
            system_prompt=prompt_music, user_prompt=user_prompt
        )

        # 记录LLM调用完成时间
        llm_time = time.time() - llm_start_time
        logger.bind(tag=TAG).debug(
            f"LLM意图识别完成, 模型: {model_info}, 调用耗时: {llm_time:.4f}秒"
        )

        # 记录后处理开始时间
        postprocess_start_time = time.time()

        # 清理和解析响应
        intent = intent.strip()
        # 尝试提取JSON部分
        match = re.search(r"\{.*\}", intent, re.DOTALL)
        if match:
            intent = match.group(0)

        # 记录总处理时间
        total_time = time.time() - total_start_time
        logger.bind(tag=TAG).debug(
            f"【意图识别性能】模型: {model_info}, 总耗时: {total_time:.4f}秒, LLM调用: {llm_time:.4f}秒, 查询: '{text[:20]}...'"
        )

        # 尝试解析为JSON
        try:
            intent_data = json.loads(intent)
            # 如果包含function_call，则格式化为适合处理的格式
            if "function_call" in intent_data:
                function_data = intent_data["function_call"]
                function_name = function_data.get("name")
                function_args = function_data.get("arguments", {})

                # 记录识别到的function call
                logger.bind(tag=TAG).info(
                    f"识别到function call: {function_name}, 参数: {function_args}"
                )

                # 添加到缓存
                self.intent_cache[cache_key] = {
                    "intent": intent,
                    "timestamp": time.time(),
                }

                # 后处理时间
                postprocess_time = time.time() - postprocess_start_time
                logger.bind(tag=TAG).debug(f"意图后处理耗时: {postprocess_time:.4f}秒")

                # 确保返回完全序列化的JSON字符串
                return intent
            else:
                # 添加到缓存
                self.intent_cache[cache_key] = {
                    "intent": intent,
                    "timestamp": time.time(),
                }

                # 后处理时间
                postprocess_time = time.time() - postprocess_start_time
                logger.bind(tag=TAG).debug(f"意图后处理耗时: {postprocess_time:.4f}秒")

                # 返回普通意图
                return intent
        except json.JSONDecodeError:
            # 后处理时间
            postprocess_time = time.time() - postprocess_start_time
            logger.bind(tag=TAG).error(
                f"无法解析意图JSON: {intent}, 后处理耗时: {postprocess_time:.4f}秒"
            )
            # 如果解析失败，默认返回继续聊天意图
            return '{"intent": "继续聊天"}'
