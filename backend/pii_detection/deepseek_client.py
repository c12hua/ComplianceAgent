import logging
from openai import OpenAI
import json
import os
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

DEEPSEEK_API_URL = "https://api.deepseek.com"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # 从环境变量读取

# 更有逻辑和合规依据的 prompt（全简体）
DEEPSEEK_PROMPT_TEMPLATE = '''你是一名医疗敏感个人信息合规检测专家，请根据下方知识库规则，对输入文本进行合规性分析，步骤如下：\n\n1. 先判断文本中是否存在敏感个人信息。\n2. 对每一条检出的敏感信息，根据知识库规则，给出具体的合规依据、风险等级（高/中/低/未知），并根据知识库内容说明判断理由，需引用规则内容。\n3. 最后，对整体文本的合规风险进行总结，并给出说明与规则依据。\n\n请严格按照以下 JSON 格式返回（只返回 JSON）：\n\n{{\n  "summary": {{\n    "total_entities": 敏感信息总数（整数）,\n    "risk_level": "高|中|低|未知",\n    "overall_reason": "整体合规风险说明，引用知识库相关内容"\n  }},\n  "details": [\n    {{\n      "entities": ["实体1", "实体2", ...],\n      "risk_level": "高|中|低|未知",\n      "reason": "该条敏感信息的合规判断依据与说明，需引用知识库内容"\n    }}\n    // ...如有多条，继续列出\n  ]\n}}\n\n待检测文本如下：\n{input_text}'''
# 如果有tokens信息，使用这个模板



# 加载 .env 文件
load_dotenv(os.path.join(os.path.dirname(__file__), '../.env'))

def deepseek_detect(text, knowledge_base_prompt=None, tokens=None):
    """
    使用 OpenAI SDK 方式訪問 deepseek API。
    当传入 tokens 时，使用简洁 tokens-only 提示，直接让大模型从 tokens 列表抽取 PII 并只返回严格 JSON：{"text":..., "entities":[{"entity","type"}, ...]}
    knowledge_base_prompt 在 tokens 模式下被忽略（如需知识库推理，请使用 detect_pii_with_deepseek 或传入 knowledge_base_prompt 并不提供 tokens）。
    """
    # 优先使用调用方传入的完整 prompt（knowledge_base_prompt），如果调用方没有提供，则根据 tokens 或文本回退到内部模板
    if knowledge_base_prompt:
        # 直接把 caller 提供的 prompt 原样发送给模型
        prompt = knowledge_base_prompt
    elif tokens:
        # caller 未提供完整 prompt，但传入了 tokens：构造一个最小包装，告诉模型只返回严格 JSON
        try:
            tokens_payload = json.dumps(tokens, ensure_ascii=False)
        except Exception:
            tokens_payload = str(tokens)
        prompt = f'请仅返回严格的 JSON 对象，格式为：{{"text":"<原文>","entities":[{{"entity":"...","type":"..."}}, ...]}}。不要输出任何额外说明或格式化文本。\n\nTokens:\n{tokens_payload}\n\n原文:\n{text}'
    else:
        # 回退到原始全文模板（不含额外知识库）
        prompt = DEEPSEEK_PROMPT_TEMPLATE.format(input_text=text)

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一個医疗敏感个人信息检测助手。"},
                {"role": "user", "content": prompt},
            ],
            stream=False,
            max_tokens=4096
        )
        content = response.choices[0].message.content
        # 只提取JSON部分，兼容 markdown 代码块
        try:
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            content = content.strip('`\n ')
            json_start = content.find('{')
            if json_start == -1:
                # 找不到 JSON，返回原始文本并标记错误
                logger.error(f"Deepseek 返回未包含 JSON: {content}")
                return {"content": content, "error": "未包含 JSON"}
            json_str = content[json_start:]
            result = json.loads(json_str)
            return result
        except Exception as e:
            logger.error(f"Deepseek返回内容非标准JSON: {content}")
            return {"content": content, "error": "非标准JSON", "exception": str(e)}
    except Exception as e:
        logger.error(f"Deepseek API 调用失败: {e}")
        return {"error": str(e)}




def detect_pii_with_deepseek(text, tokens=None):
    """
    使用 Deepseek 进行 PII 检测的包装函数
    返回标准化的检测结果
    """
    try:
        # 如果提供了tokens，将其传递给deepseek_detect
        result = deepseek_detect(text, tokens=tokens)
        logger.info(f"DeepSeek API 返回结果: {result}")

        if "error" in result:
            logger.error(f"Deepseek 检测失败: {result['error']}")
            return {
                "summary": {
                    "total_entities": 0,
                    "risk_level": "未知",
                    "overall_reason": f"检测失败: {result['error']}"
                },
                "details": []
            }

        return result  # 直接返回 deepseek_detect 的结果

    except Exception as e:
        logger.error(f"Deepseek PII 检测出错: {str(e)}", exc_info=True)
        return {
            "summary": {
                "total_entities": 0,
                "risk_level": "未知",
                "overall_reason": f"检测过程出错: {str(e)}"
            },
            "details": []
        }



