import json
import sys
import os
import logging
from dotenv import load_dotenv

# 加载.env文件
load_dotenv(os.path.join(os.path.dirname(__file__), '.env'))

from .src import pipeline, recognizer

from .src.pipeline import call_recognizer, get_recognizer
from .src.recognizer import LLMRecognizer

logger = logging.getLogger(__name__)


def process_tokens_with_pipeline(tokens):
    """
    使用pipeline处理tokens格式的输入
    参数:
        tokens: list[str]，如 ['MRI', '头部', '已', '检', '，', '报告', '如下']
    返回:
        dict: 包含处理后的文本和实体信息
    """
    try:
        # 获取识别器（使用LLM规则）
        recognizer = LLMRecognizer()

        # 调用识别器处理tokens
        entities = call_recognizer(recognizer, tokens)
        #logger.info(f"LLM识别出的实体: {entities}")

        # 拼接文本
        text = "".join(tokens)

        # 构造返回结果
        result = {
            "text": text,
            "tokens": tokens,
            "entities": entities
        }

        #logger.info(f"中间层处理完成: {result}")
        return result
    except Exception as e:
        # 出错时返回原始文本
        text = "".join(tokens) if isinstance(tokens, list) else str(tokens)
        return {
            "text": text,
            "tokens": tokens if isinstance(tokens, list) else None,
            "entities": [],
            "error": str(e)
        }
