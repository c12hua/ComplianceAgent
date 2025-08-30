from __future__ import annotations
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Optional

# Common Chinese surname list (top 100), for simple name detection
SURNAMES = set(list("赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳酆鲍史唐费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴谈宋茅庞熊纪舒屈项祝董粱杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊胡凌霍虞万支柯咎管卢莫经房裘缪干解应宗丁宣邓郁单杭洪包诸左石崔吉钮龚程嵇邢滑裴陆荣翁荀羊於惠甄曲家封芮羿储靳汲邴糜松井段富巫乌焦巴弓牧隗山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶幸司韶郜黎蓟薄印宿白怀蒲台从鄂索咸籍赖卓蔺屠蒙池乔阴鬱胥能苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿通边扈燕冀郏浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾终暨居衡步都耿满弘匡国文寇广禄阙东欧殳沃利蔚越夔隆师巩厍聂晁勾敖融冷訾辛阚那简饶空曾毋沙乜养鞠须丰巢关蒯相查后荆红游竺权逯盖益桓公仉督岳帅缑亢况郈有琴归海晋楚闫"))

AGE_RE = re.compile(r"(\d{1,3})岁")
SEX_RE = re.compile(r"\b(男|女)\b")
PHONE_RE = re.compile(r"(?<!\d)(1[3-9]\d{9})(?!\d)")  # Mainland China mobile
IDCARD_RE = re.compile(r"(?i)(\d{15}|\d{17}[0-9Xx])")  # CN ID
INSURANCE_RE = re.compile(r"(医保卡号|社保卡号|医保号|参保号)[：: ]?([0-9A-Za-z-]{6,})")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
BANK_RE = re.compile(r"(?<!\d)(?:[1-9]\d{11,18})(?!\d)")  # naive long number

SYMPTOMS = {"头痛","发热","咳嗽","胸痛","腹痛","恶心","呕吐","乏力"}
DISEASES = {"高血压","糖尿病","冠心病","哮喘","抑郁症"}
DRUG_HINTS = {"降压药","阿司匹林","布洛芬","对乙酰氨基酚"}
EXAMS = {"MRI","CT","X光","血常规","核酸检测"}
PRESC = {"处方","电子处方"}

@dataclass
class Span:
    text: str
    type: str
    start: int
    end: int
    confidence: float = 0.6

class HeuristicRecognizer:
    def __init__(self) -> None:
        pass

    def _find_names(self, text: str) -> List[Span]:
        spans: List[Span] = []
        for i in range(len(text) - 1):
            ch = text[i]
            if ch in SURNAMES:
                # patterns like 张某, 王某某, etc.
                if i + 1 < len(text) and text[i+1] in ("某","某某"):
                    spans.append(Span(text[i:i+2], "姓名", i, i+2, 0.8))
                # simple two/three-char name
                if i + 2 <= len(text):
                    cand = text[i:i+2]
                    if all('\u4e00' <= c <= '\u9fff' for c in cand):
                        spans.append(Span(cand, "姓名", i, i+2, 0.5))
                if i + 3 <= len(text):
                    cand3 = text[i:i+3]
                    if all('\u4e00' <= c <= '\u9fff' for c in cand3):
                        spans.append(Span(cand3, "姓名", i, i+3, 0.5))
        return spans

    def _find_by_set(self, text: str, vocab: set, label: str) -> List[Span]:
        spans: List[Span] = []
        for term in sorted(vocab, key=lambda x: -len(x)):
            idx = 0
            while True:
                idx = text.find(term, idx)
                if idx == -1:
                    break
                spans.append(Span(term, label, idx, idx+len(term), 0.7))
                idx += len(term)
        return spans

    def recognize(self, text: str) -> List[Dict[str, Any]]:
        spans: List[Span] = []
        # regex-based
        for m in AGE_RE.finditer(text):
            spans.append(Span(m.group(0), "年龄", m.start(), m.end(), 0.9))
        for m in SEX_RE.finditer(text):
            spans.append(Span(m.group(1), "性别", m.start(), m.end(), 0.9))
        for m in PHONE_RE.finditer(text):
            spans.append(Span(m.group(1), "电话", m.start(), m.end(), 0.9))
        for m in IDCARD_RE.finditer(text):
            spans.append(Span(m.group(0), "身份证号", m.start(), m.end(), 0.9))
        for m in INSURANCE_RE.finditer(text):
            spans.append(Span(m.group(0), "支付信息", m.start(), m.end(), 0.95))
        for m in EMAIL_RE.finditer(text):
            spans.append(Span(m.group(0), "邮箱", m.start(), m.end(), 0.9))
        for m in BANK_RE.finditer(text):
            # Very naive bank card candidate; keep low confidence
            spans.append(Span(m.group(0), "支付信息", m.start(), m.end(), 0.4))

        # dictionary-based
        spans += self._find_by_set(text, SYMPTOMS, "症状")
        spans += self._find_by_set(text, DISEASES, "疾病")
        spans += self._find_by_set(text, DRUG_HINTS, "药物")
        spans += self._find_by_set(text, EXAMS, "检查项目")
        spans += self._find_by_set(text, PRESC, "处方")
        spans += self._find_names(text)

        # deduplicate overlaps (keep longest & highest confidence)
        spans.sort(key=lambda s: (s.start, -(s.end - s.start), -s.confidence))
        kept: List[Span] = []
        for sp in spans:
            if not kept or sp.start >= kept[-1].end:
                kept.append(sp)
            else:
                # overlap: replace if longer
                last = kept[-1]
                if (sp.end - sp.start, sp.confidence) > (last.end - last.start, last.confidence):
                    kept[-1] = sp

        return [{
            "entity": s.text,
            "type": s.type,
            "start": s.start,
            "end": s.end,
            "confidence": round(s.confidence, 3)
        } for s in kept]

class LLMRecognizer:
    """Skeleton for using an OpenAI-compatible API. Requires env vars:
       OPENAI_API_KEY, optional OPENAI_BASE_URL, OPENAI_MODEL.
    """
    def __init__(self) -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package not installed. Please `pip install openai`.") from e
        import os, json  # noqa: F401

        base_url = os.getenv("OPENAI_BASE_URL") or None
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set.")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def recognize(self, text: str) -> List[Dict[str, Any]]:
        prompt = f"""你是一个中文PII抽取器。给定文本，提取其中的个人信息相关实体，输出JSON数组。
要求字段：entity, type, start, end。type 可包含：姓名、性别、年龄、电话、身份证号、邮箱、支付信息、症状、疾病、药物、检查项目、处方等。
文本：{text}
只输出JSON数组。"""
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}],
            temperature=0
        )
        content = resp.choices[0].message.content.strip()
        # best-effort JSON parse
        import json
        try:
            arr = json.loads(content)
            assert isinstance(arr, list)
        except Exception:
            arr = []
        # Ensure required keys
        norm = []
        for it in arr:
            ent = {
                "entity": it.get("entity"),
                "type": it.get("type"),
                "start": it.get("start"),
                "end": it.get("end"),
                "confidence": it.get("confidence", 0.7),
            }
            if ent["entity"] and ent["type"]:
                norm.append(ent)
        return norm
