import json

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import logging
from .langchain_agent import detect_pii_and_risk
from .desensitize import simple_desensitize
from rest_framework.decorators import api_view

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

    def process_text_with_standardization(self, text: str) -> dict:
        """
        对文本进行标准化、分词和结构化处理
        """
        try:
            from .text_processor import text_processor
            # 调用文本标准化和分词功能
            normalized_result = text_processor.normalize_text(text)
            logger.info(f"文本标准化完成，原长度: {normalized_result['original_length']}, 标准化后长度: {normalized_result['normalized_length']}")
            logger.info(f"标准化文本: {normalized_result['normalized_text'][:100]}...")  # 记录前100个字符用于调试
            
            return normalized_result
        except Exception as e:
            logger.error(f"文本标准化处理失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            # 返回错误结果
            return {
                'normalized_text': text,
                'sentences': [text],
                'paragraphs': [text],
                'original_length': len(text),
                'normalized_length': len(text),
                'structured_text': {
                    'structured_text': text,
                    'tokenized_sentences': [text],
                    'sentence_count': 1,
                    'error': str(e)
                }
            }
    
    def preprocess_input(self, text_data):
        """
        预处理输入数据，将输入直接当作tokens格式处理
        返回处理后的文本和tokens信息
        """
        try:
            # 尝试解析为JSON格式（分词数据）
            data = json.loads(text_data)
            if isinstance(data, list):
                # 纯tokens格式: ["患者", "张某", "，", "男", "，", "45", "岁"]
                if all(isinstance(item, str) for item in data):
                    tokens = data
                    text = "".join(tokens)
                    return text, tokens
            elif isinstance(data, dict):
                # 包含tokens的字典格式: {"tokens": [...], "other_data": ...}
                if "tokens" in data and isinstance(data["tokens"], list):
                    tokens = data["tokens"]
                    text = "".join(tokens) if all(isinstance(item, str) for item in tokens) else str(data)
                    return text, tokens
            # 如果JSON格式不符合要求，仍将输入当作单个token处理
            tokens = [text_data]
            text = text_data
            return text, tokens
        except json.JSONDecodeError:
            # 纯文本格式，将其作为单个token处理
            tokens = [text_data]
            text = text_data
            logger.info(f"输入无法解析为JSON，将其作为单个token处理: {tokens}")
            return text, tokens

    def post(self, request):
        text = request.data.get("text", None)
        model = request.data.get("model", "deepseek")  # 默认使用deepseek
        file = request.FILES.get('file', None)
        extracted_text = None
        tokens = None
        if file:
            try:
                extracted_text = self.extract_text_from_file(file)
                logger.info(extracted_text)
            except Exception as e:
                logger.error(f"File parse error: {e}")
                return Response({"error": "文件解析失败: " + str(e)}, status=status.HTTP_400_BAD_REQUEST)
        elif text:
            # 预处理输入数据
            logger.info(f"原始输入文本: {text}")
            extracted_text, tokens = self.preprocess_input(text)
            #logger.info(f"预处理后文本: {extracted_text}")
            #logger.info(f"预处理后tokens: {tokens}")
        else:
            return Response({"error": "请上传文件或输入文本"}, status=status.HTTP_400_BAD_REQUEST)

        logger.info(f"PII检测请求，模型: {model}, 文本长度: {len(extracted_text)}")
        # 动态加载知识库内容，转为 prompt
        from .knowledge_utils import load_knowledge_base
        knowledge_base_prompt = load_knowledge_base()

        # 根据选择的模型调用不同的检测方法
        # 使用deepseek方法
        from .deepseek_client import detect_pii_with_deepseek

        # 文本标准化和分词处理
        logger.info("进入文本标准化和分词处理流程")
        normalized_result = self.process_text_with_standardization(extracted_text)
        normalized_text = normalized_result['normalized_text']

        # 处理tokens（现在总是有tokens）
        logger.info("进入tokens处理流程")
        # 导入中间层处理函数
        from pii_detection.middleware import process_tokens_with_pipeline
        # 使用中间层处理tokens
        processed_result = process_tokens_with_pipeline(tokens)
        logger.info(f"中间层处理结果: {processed_result}")
        # 将处理后的结果作为文本传递给大模型，同时传递实体信息用于构建更准确的prompt
        result = detect_pii_with_deepseek(processed_result.get("text", normalized_text),
                                          processed_result.get("entities", []))

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
            "tokens": tokens  # 返回tokens信息（如果有）
        }, status=status.HTTP_201_CREATED)