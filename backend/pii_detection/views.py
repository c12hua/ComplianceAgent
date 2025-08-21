from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import logging
from .langchain_agent import detect_pii_and_risk
from .desensitize import simple_desensitize
from rest_framework.decorators import api_view
import re
import unicodedata

@api_view(["POST"])
def desensitize_view(request):
    """
    脱敏处理API，输入：text, entities（可选，若无则自动检测）
    """
    text = request.data.get("text", None)
    entities = request.data.get("entities", None)
    if not text:
        return Response({"error": "请提供待脱敏文本"}, status=status.HTTP_400_BAD_REQUEST)
    # 若未指定entities，则自动检测
    if not entities:
        from .knowledge_utils import load_knowledge_base
        knowledge_base_prompt = load_knowledge_base()
        detect_result = detect_pii_and_risk(text, knowledge_base_prompt)
        # 收集所有实体
        entities = []
        for detail in detect_result.get("details", []):
            entities.extend(detail.get("entities", []))
    # 脱敏
    desensitized_text = simple_desensitize(text, entities)
    return Response({
        "original_text": text,
        "entities": entities,
        "desensitized_text": desensitized_text
    }, status=status.HTTP_200_OK)
from .models import PiiDetectionRecord
from .serializers import PiiDetectionRecordSerializer


from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from django.core.files.uploadedfile import UploadedFile
import io
import docx
import PyPDF2

logger = logging.getLogger(__name__)

class PiiDetectView(APIView):
    parser_classes = (MultiPartParser, FormParser, JSONParser)

    def extract_text_from_file(self, uploaded_file: UploadedFile) -> str:
        name = uploaded_file.name.lower()
        if name.endswith('.txt'):
            return uploaded_file.read().decode('utf-8', errors='ignore')
        elif name.endswith('.docx'):
            doc = docx.Document(uploaded_file)
            return '\n'.join([p.text for p in doc.paragraphs])
        elif name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text() or ''
            return text
        else:
            raise ValueError('Unsupported file type')

    def normalize_text(self, text: str) -> dict:

        if not text:
            return {
                'normalized_text': '[]',
                'sentences': [],
                'paragraphs': [],
                'original_length': 0,
                'normalized_length': 2
            }
        
        original_text = text

        # 1. 基础字符标准化
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 空白字符统一（保留换行符，仅折叠其它空白）
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        text = text.strip()

        # 2. 标点符号处理（只保留逗号，清理重复标点）
        text = re.sub(r'[！]{2,}', '！', text)
        text = re.sub(r'[？]{2,}', '？', text) 
        text = re.sub(r'[；]{2,}', '；', text) 
        text = re.sub(r'[，、：""''（）【】]{2,}', lambda m: m.group()[0], text)

        # 3. 数字和日期标准化
        text = re.sub(r'(?<=\d)[,，](?=\d)', '', text)

        # 4. 医学专业术语处理
        drug_mappings = {
            '阿司匹林肠溶片': '阿司匹林',
            '布洛芬缓释胶囊': '布洛芬',
            '对乙酰氨基酚片': '对乙酰氨基酚'
        }
        for brand, generic in drug_mappings.items():
            if brand in text:
                text = text.replace(brand, generic)

        disease_mappings = {
            '高血压病': '高血压',
            '糖尿病 mellitus': '糖尿病',
            '冠心病': '冠状动脉粥样硬化性心脏病'
        }
        for synonym, standard in disease_mappings.items():
            if synonym in text:
                text = text.replace(synonym, standard)
        
        # 5. 句子分割和段落处理
        sentences = re.split(r'(?:[。！？；?!;]|…|\.\.\.|\n)+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # 6. 特殊格式处理
        text = re.sub(r'第\s*\d+\s*页', '', text)
        text = re.sub(r'Page\s*\d+', '', text)
        
        # 7. 语言和编码处理
        traditional_to_simplified = {
            '醫': '医', '藥': '药', '診': '诊', '療': '疗',
            '檢': '检', '驗': '验', '報': '报', '告': '告'
        }
        for traditional, simplified in traditional_to_simplified.items():
            if traditional in text:
                text = text.replace(traditional, simplified)

        sentence_end_pattern = r'(?:[。！？；?!;]|…|\.\.\.|\n)+'
        raw_sentences = re.split(sentence_end_pattern, text)
        normalized_sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            # 统一逗号为中文逗号
            s = s.replace(',', '，')

            s = re.sub(r"[、:\"“”'（）()【】\[\]]", '', s)
            # 将多种日期格式统一为 YYYYMMDD
            def _normalize_date(m):
                y, mth, d = m.group(1), m.group(2), m.group(3)
                return f"{y}{int(mth):02d}{int(d):02d}"

            s = re.sub(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*(?:日|号)", _normalize_date, s)
            s = re.sub(r"(\d{4})\s*[-/.]\s*(\d{1,2})\s*[-/.]\s*(\d{1,2})", _normalize_date, s)
            
            # 去除电话号码等中的连字符（不影响已处理的日期）
            s = re.sub(r'(?<=\d)[\-–—]+(?=\d)', '', s)
            s = re.sub(r'\s+', '', s)
            normalized_sentences.append(s)
        standardized_text = '[' + ', '.join([f"'{s}'" for s in normalized_sentences]) + ']'
        
        return {
            'normalized_text': standardized_text,  # 主要输出：标准化文本
            'sentences': sentences,                # 按句分割的文本列表
            'paragraphs': paragraphs,              # 段落列表
            'original_length': len(original_text),
            'normalized_length': len(standardized_text)
        }

    def post(self, request):
        text = request.data.get("text", None)
        model = request.data.get("model", "deepseek")  # 默认使用deepseek
        file = request.FILES.get('file', None)
        extracted_text = None
        
        if file:
            try:
                extracted_text = self.extract_text_from_file(file)
                logger.info(extracted_text)
            except Exception as e:
                logger.error(f"File parse error: {e}")
                return Response({"error": "文件解析失败: " + str(e)}, status=status.HTTP_400_BAD_REQUEST)
        elif text:
            extracted_text = text
        else:
            return Response({"error": "请上传文件或输入文本"}, status=status.HTTP_400_BAD_REQUEST)

        logger.info(f"PII检测请求，模型: {model}, 文本长度: {len(extracted_text)}")
        
        # 文本标准化和清理
        normalized_result = self.normalize_text(extracted_text)
        normalized_text = normalized_result['normalized_text']
        logger.info(f"文本标准化完成，原长度: {normalized_result['original_length']}, 标准化后长度: {normalized_result['normalized_length']}")
        logger.info(f"标准化文本: {normalized_text[:100]}...")  # 记录前100个字符用于调试
        
        # 动态加载知识库内容，转为 prompt
        from .knowledge_utils import load_knowledge_base
        knowledge_base_prompt = load_knowledge_base()
        
        # 根据选择的模型调用不同的检测方法
        # 使用deepseek方法
        from .deepseek_client import detect_pii_with_deepseek
        result = detect_pii_with_deepseek(normalized_text)
        
        logger.info(f"检测结果: {result}")
        record = PiiDetectionRecord.objects.create(
            text=extracted_text,
            detected_entities=result.get("entities", []),
            risk_level=result.get("risk_level", "未知")
        )
        serializer = PiiDetectionRecordSerializer(record)
        # 返回原文内容，便于前端展示
        # 直接返回 deepseek summary 字段内容，全部顶层展开
        from .knowledge_lookup import find_knowledge_explanation
        summary = result.get("summary", {})
        overall_reason = summary.get("overall_reason", "")
        knowledge_explanation = find_knowledge_explanation(overall_reason)
        return Response({
            "summary": summary,
            "details": result.get("details", []),
            "text": extracted_text,
            "normalized_text": normalized_result
        }, status=status.HTTP_201_CREATED)
