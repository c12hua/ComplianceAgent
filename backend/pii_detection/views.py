from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import logging
from .langchain_agent import detect_pii_and_risk
from .desensitize import simple_desensitize
from rest_framework.decorators import api_view
import re
import unicodedata
from opencc import OpenCC
from dateutil import parser as date_parser
import os
import json
from datetime import datetime
try:
    import jieba
    JIEBA_AVAILABLE = True
    # 初始化jieba
    jieba.initialize()
except ImportError:
    JIEBA_AVAILABLE = False
    logger.warning("jieba 未安装，将使用简单分词")

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
                'normalized_length': 2,
                'structured_text': {
                    'structured_text': '[]',
                    'tokenized_sentences': [],
                    'sentence_count': 0
                }
            }
        
        original_text = text

        # 1. 基础字符标准化
        # 去除HTML标签
        text = re.sub(r'<[^>]+>', '', text)
        
        # 去除噪音字符（###, ***, 等）
        text = re.sub(r'[#*]{3,}', '', text)
        
        # 保护科学计数法格式，使用占位符
        scientific_notations = []
        def protect_scientific_notation(match):
            placeholder = f"__SCIENTIFIC_{len(scientific_notations)}__"
            scientific_notations.append(match.group(0))
            return placeholder

        text = re.sub(r'(\d+(?:\.\d+)?)\s*×\s*10\s*[⁰¹²³⁴⁵⁶⁷⁸⁹]+', protect_scientific_notation, text)
        
        # 标准化数学符号
        math_symbols = {
            '≥': '>=',
            '≤': '<=',
            '≠': '!=', 
            '≈': '≈',
            '±': '±', 
            '∞': '∞', 
            '∑': '∑', 
            '∫': '∫', 
            '√': '√', 
        }
        
        for symbol, replacement in math_symbols.items():
            text = text.replace(symbol, replacement)
        
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')

        # 空白字符统一（保留换行符，仅折叠其它空白）
        text = re.sub(r'[^\S\n]+', ' ', text)
        text = re.sub(r' *\n *', '\n', text)
        text = text.strip()

        # 2. 标点符号处理（只保留逗号，清理重复标点）
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[;]{2,}', ';', text)
        text = re.sub(r'[！]{2,}', '！', text)
        text = re.sub(r'[？]{2,}', '？', text) 
        text = re.sub(r'[；]{2,}', '；', text) 
        text = re.sub(r'[，、：""''（）【】]{2,}', lambda m: m.group()[0], text)

        # 3. 数字和日期标准化
        text = re.sub(r'(?<=\d)[,，](?=\d)', '', text)
        
        # 4. 句子分割和段落处理
        sentences = re.split(r'(?:[。！？；?!;]|…|\.\.\.|\n)+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # 5. 特殊格式处理
        text = re.sub(r'第\s*\d+\s*页', '', text)
        text = re.sub(r'Page\s*\d+', '', text)
        
        # 6. 语言和编码处理
        cc = OpenCC('t2s')  # Traditional to Simplified
        text = cc.convert(text)

        sentence_end_pattern = r'(?:[。！？；?!;]|…|\.\.\.|\n)+'
        raw_sentences = re.split(sentence_end_pattern, text)
        normalized_sentences = []
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            # 统一逗号为中文逗号
            s = s.replace(',', '，')

            # 使用 datetime 处理日期标准化（在删除空格之前）
            def _normalize_date_with_datetime(date_str):
                """使用 dateutil.parser 解析各种日期格式"""
                try:
                    original_str = date_str.strip()

                    try:
                        # 标准解析
                        dt = date_parser.parse(original_str)
                    except:
                        try:
                            dt = date_parser.parse(original_str, dayfirst=True)
                        except:
                            dt = date_parser.parse(original_str, fuzzy=True)
                    
                    return dt.strftime('%Y-%m-%d') 
                except Exception as e:
                    return date_str
            
            # 统一日期匹配模式
            date_pattern = re.compile(
                r'\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*[日号]|'  # 中文格式：2024年1月2日
                r'\b(?:19|20)\d{2}\s*[-/.]\s*(?:0?[1-9]|1[0-2])\s*[-/.]\s*(?:0?[1-9]|[12]\d|3[01])\b|'   # YYYY-MM-DD 格式：2024-01-02，限制年份为19xx或20xx
                r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2}[，,]?\s*\d{4}|' # Mon DD YYYY：Jan 22, 2025 或 Jan 22， 2025
                r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|' # DD Mon YYYY：22 Jan 2025
                r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\d{1,2}\d{4}|' # MonDDYYYY：Jan222025
                r'\d{1,2}(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\d{4}|' # DDMonYYYY：22Aug2025
                r'\b(?:19|20)\d{6}\b', # YYYYMMDD 格式：20241231，限制年份为19xx或20xx
                flags=re.IGNORECASE
            )

            # 保护括号内的内容不被删除
            s = s.replace('（', '(').replace('）', ')')
            
            # 暂时保留括号和冒号，只去除其他标点
            s = re.sub(r"[、\"\"''【】\[\]]", '', s)
            
            # 去除电话号码等中的连字符（不影响已处理的日期）
            s = re.sub(r'(?<=\d)[\-–—]+(?=\d)', '', s)

            s = date_pattern.sub(lambda m: _normalize_date_with_datetime(m.group(0)), s)

            # 最后处理：将多个连续空格替换为单个空格
            s = re.sub(r'\s+', ' ', s)
            normalized_sentences.append(s)
        standardized_text = '[' + ','.join([f"'{s}'" for s in normalized_sentences]) + ']'
        
        # 恢复科学计数法占位符
        for i, scientific_notation in enumerate(scientific_notations):
            placeholder = f"__SCIENTIFIC_{i}__"
            standardized_text = standardized_text.replace(placeholder, scientific_notation)
        
        # 7. 分词分句处理（根据图2中的方法）
        structured_text = self.tokenize_and_segment(standardized_text)
        
        return {
            'normalized_text': standardized_text,  # 主要输出：标准化文本
            'sentences': sentences,                # 按句分割的文本列表
            'paragraphs': paragraphs,              # 段落列表
            'original_length': len(original_text),
            'normalized_length': len(standardized_text),
            'structured_text': structured_text     # 新增：结构化文本（分词分句结果）
        }

    def tokenize_and_segment(self, standardized_text: str) -> dict:
        """
        对标准化文本进行分词分句处理
        使用 spaCy + LTP 工具，按照中文能够理解的方式进行分词
        """
        try:
            # 解析标准化文本（去除外层方括号和引号）
            text_content = standardized_text.strip()
            if text_content.startswith('[') and text_content.endswith(']'):
                text_content = text_content[1:-1]
            
            # 分割句子（按英文逗号分隔的字符串）
            sentences = []
            current_sentence = ""
            in_quotes = False
            
            for char in text_content:
                if char == "'" and (not current_sentence or current_sentence[-1] != '\\'):
                    in_quotes = not in_quotes
                elif char == ',' and not in_quotes:
                    if current_sentence.strip():
                        # 去除引号
                        sentence = current_sentence.strip().strip("'")
                        sentences.append(sentence)
                    current_sentence = ""
                else:
                    current_sentence += char
            
            # 处理最后一个句子
            if current_sentence.strip():
                sentence = current_sentence.strip().strip("'")
                sentences.append(sentence)
            
            # 初始化分词工具
            jieba_seg = None
            
            if JIEBA_AVAILABLE:
                try:
                    jieba_seg = jieba
                    logger.info("✅ jieba 加载成功")
                except Exception as e:
                    logger.warning(f"❌ jieba 加载失败: {e}")
                    jieba_seg = None
            else:
                logger.info("jieba 不可用，使用简单分词")
            
            # 对每个句子进行分词
            tokenized_sentences = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                
                # 分词处理
                if jieba_seg:
                    try:
                        # 使用 jieba 进行分词
                        jieba_tokens = list(jieba_seg.cut(sentence))
                        logger.info(f"jieba分词结果: {jieba_tokens}")
                        final_tokens = jieba_tokens
                    except Exception as e:
                        logger.warning(f"jieba分词失败: {e}")
                        final_tokens = self.simple_tokenize(sentence)  # 回退到简单分词
                else:
                    final_tokens = self.simple_tokenize(sentence)  # 使用简单分词
                
                # 清理分词结果：删除空格、冒号、括号等字符
                cleaned_tokens = []
                for token in final_tokens:
                    # 删除空格、冒号、括号等字符
                    cleaned_token = re.sub(r'[\s:：()（）【】\[\]]', '', token)
                    if cleaned_token:  # 只保留非空token
                        cleaned_tokens.append(cleaned_token)
                
                # 将分词结果格式化为图2中的格式（每个句子只有一个方括号）
                if cleaned_tokens:
                    tokenized_sentence = "['" + "', '".join(cleaned_tokens) + "']"
                    tokenized_sentences.append(tokenized_sentence)
            
            # 创建完整的结构化文本（句子之间用][连接，不添加外层方括号）
            if tokenized_sentences:
                structured_text = "".join(tokenized_sentences)
            else:
                structured_text = ""
            
            logger.info(f"分词分句完成，句子数量: {len(tokenized_sentences)}")
            logger.info(f"结构化文本示例: {structured_text[:200]}...")
            
            # 保存结构化文本到文件
            self.save_structured_text(structured_text)
            
            return {
                'structured_text': structured_text,
                'tokenized_sentences': tokenized_sentences,
                'sentence_count': len(tokenized_sentences)
            }
            
        except Exception as e:
            logger.error(f"分词分句处理失败: {e}")
            import traceback
            logger.error(f"详细错误信息: {traceback.format_exc()}")
            return {
                'structured_text': standardized_text,
                'tokenized_sentences': [],
                'sentence_count': 0,
                'error': str(e)
            }

    def simple_tokenize(self, text: str) -> list:
        """
        简单的分词方法，作为 jieba 的回退方案
        按照中文标点符号和空格进行基本分词
        """
        if not text:
            return []
        
        # 定义中文标点符号
        chinese_punct = '，。！？；：""''（）【】、'
        
        # 按标点符号分割
        tokens = []
        current_token = ""
        
        for char in text:
            if char in chinese_punct:
                if current_token.strip():
                    tokens.append(current_token.strip())
                # 不保留标点符号作为独立token
                current_token = ""
            elif char.isspace():
                if current_token.strip():
                    tokens.append(current_token.strip())
                current_token = ""
            else:
                current_token += char
        
        # 处理最后一个token
        if current_token.strip():
            tokens.append(current_token.strip())
        
        # 过滤空token
        tokens = [token for token in tokens if token.strip()]
        
        # 如果没有分割出任何token，返回原文本
        if not tokens:
            tokens = [text]
        
        logger.info(f"简单分词结果: {tokens}")
        return tokens

    def save_structured_text(self, structured_text: str):
        """
        保存结构化文本到文件
        文件名格式：病例文本_时间戳
        保存路径：/backend/knowledge_base/text/
        """
        try:
            # 创建输出目录
            output_dir = "/backend/knowledge_base/text/"
            os.makedirs(output_dir, exist_ok=True)
            
            # 生成文件名（时间戳格式）
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"病例文本_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)
            
            # 保存结构化文本
            data = {
                'timestamp': timestamp,
                'structured_text': structured_text,
                'created_at': datetime.now().isoformat()
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"结构化文本已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存结构化文本失败: {e}")

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
            "normalized_text": normalized_result,
            "structured_text": normalized_result.get('structured_text', {})
        }, status=status.HTTP_201_CREATED)