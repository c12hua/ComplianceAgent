import json
from typing import Dict, Any
from openai import OpenAI

DEEPSEEK_API_URL = "https://api.deepseek.com"
## DEEPSEEK_API_KEY = "" 

DEEPSEEK_PROMPT_TEMPLATE = '''你是数据合规分析专家，请根据以下信息判断实体是否属于“敏感个人信息”，并说明原因。
严格按照以下 JSON 格式返回:
{{
    "entity": "问题中的实体",
    "is_sensitive": "true|false",
    "category": "知识图谱路径中的所属的领域",
    "legal_basisails": "问题涉及的法律条款",
    "explanation": "判断的依据"
}}
待检测文本如下：\n{input_text}
'''

"""
    prompt拼接+大模型问答函数
    
    输入:
    knowledge_gragh_judge(知识图谱判断)  --缺少实际输入案例
    text(向量库中返回的原始文本) --缺少实际输入案例
    匹配分数？

    返回:
    json格式
    {
        "entity": "降压药",
        "is_sensitive": true,
        "category": "医疗服务数据",
        "legal_basis": "《个人信息保护法》第三条",
        "explanation": "降压药属于处方药，归属医疗服务记录，法律明确定义为敏感信息。"
    }
"""

def deepseek_judge(knowledge_gragh_judge, text):

    prompt = DEEPSEEK_PROMPT_TEMPLATE.format(input_text=text) + "\n知识图谱判断如下：\n" + knowledge_gragh_judge

    try:
        client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_API_URL)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=prompt,
            stream=False,
            max_tokens=2048
        )
        content = response.choices[0].message.content
        # 只提取JSON部分，兼容 markdown 代码块
        try:
            # 去除 markdown 代码块标记
            content = content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.startswith('```'):
                content = content[3:]
            content = content.strip('`\n ')
            json_start = content.find('{')
            json_str = content[json_start:]
            result = json.loads(json_str)
            return result
        except Exception as e:
            return {"content": content, "error": "非标准JSON", "exception": str(e)}
    except Exception as e:
        return {"error": str(e)}

def judge_pii_with_deepseek(text):
    
    try:
        result = deepseek_judge(text)
        
        if "error" in result:
            return {
                "summary": {
                    "total_entities": 0,
                    "risk_level": "未知",
                    "overall_reason": f"检测失败: {result['error']}"
                },
                "details": []
            }
            
        return result  
        
    except Exception as e:
        return {
            "summary": {
                "total_entities": 0,
                "risk_level": "未知",
                "overall_reason": f"检测过程出错: {str(e)}"
            },
            "details": []
        }

