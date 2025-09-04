import json
import re

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
            logger.info(f"预处理后文本: {extracted_text}")
            logger.info(f"预处理后tokens: {tokens}")
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

        # 使用标准化后的文本作为检测内容（不使用 tokens）
        logger.info(f"使用标准化文本进行PII检测，长度: {len(normalized_text)}")

        custom_instruction = (
            '请仅返回严格的 JSON 对象，格式为：{"text":"<原文>","entities":[{"entity":"...","type":"..."}, ...]}。'
            '不要输出任何额外说明或格式化文本。Entities 类型示例：姓名、身份证号、联系电话、症状、疾病、药物、检查项目、处方、支付信息、住址、年龄、性别、其他。'
            '基于下面的标准化文本，识别并返回所有个人信息实体及其类型。'
        )

        try:
            model_prompt = custom_instruction + "\n\nText:\n" + normalized_text
        except Exception:
            model_prompt = custom_instruction + "\n\nText:\n" + str(normalized_text)

        # 调用底层 deepseek 接口，仅传入标准化文本和我们构造的 prompt（不传 tokens）
        from .deepseek_client import deepseek_detect
        llm_result = deepseek_detect(normalized_text, knowledge_base_prompt=model_prompt)
        logger.info("llm_result内容：" + str(llm_result))
        # 解析 LLM 返回，抽取 entities
        entities_output = []
        try:
            if isinstance(llm_result, dict):
                # 期望直接返回 {'text':..., 'entities':[...]} 或含有 'entities' 字段
                if 'entities' in llm_result:
                    entities_output = llm_result.get('entities') or []
                elif 'details' in llm_result:
                    # 兼容老结构：从 details 中提取实体字符串或 dict
                    for d in llm_result.get('details', []):
                        ents = d.get('entities', [])
                        if isinstance(ents, list):
                            for e in ents:
                                if isinstance(e, dict) and 'entity' in e and 'type' in e:
                                    entities_output.append(e)
                                elif isinstance(e, str):
                                    entities_output.append({"entity": e, "type": "未知"})
            elif isinstance(llm_result, str):
                # 尝试把字符串解析为 JSON
                try:
                    parsed = json.loads(llm_result)
                    entities_output = parsed.get('entities', []) if isinstance(parsed, dict) else []
                except Exception:
                    entities_output = []
        except Exception as e:
            logger.error(f"解析 LLM 返回失败: {e}")
            entities_output = []

        

        # 解析并设置风险等级（默认 未知）
        risk_level = "未知"
        if isinstance(llm_result, dict):
            summary = llm_result.get('summary', {}) or {}
            risk_level = summary.get('risk_level', '未知')

        # 规范化 entities_output 为 [{'entity':..., 'type':...}, ...]
        normalized_entities = []
        for e in entities_output:
            if isinstance(e, dict):
                ent = e.get('entity') or e.get('text') or e.get('name') or ''
                typ = e.get('type') or e.get('label') or '未知'
                if ent:
                    normalized_entities.append({"entity": ent, "type": typ})
            elif isinstance(e, str):
                normalized_entities.append({"entity": e, "type": "未知"})

        # 将 llm_result 确保为 JSON dict，用于向量搜索的查询构建
        llm_json = None
        if isinstance(llm_result, dict):
            llm_json = llm_result
        else:
            try:
                llm_json = json.loads(llm_result) if isinstance(llm_result, str) else {"content": str(llm_result)}
            except Exception:
                llm_json = {"content": str(llm_result)}

        # 构造向量搜索查询：优先使用 summary.overall_reason，其次使用实体文本拼接，最后回退到标准化文本
        query_text = None
        try:
            summary = llm_json.get('summary', {}) if isinstance(llm_json, dict) else {}
            overall_reason = summary.get('overall_reason') if isinstance(summary, dict) else None
            if overall_reason:
                query_text = overall_reason
        except Exception:
            query_text = None

        if not query_text and normalized_entities:
            # 使用实体文本作为查询（取前 20 个字符的拼接，避免过长）
            try:
                query_text = '；'.join([e['entity'] for e in normalized_entities if e.get('entity')])
            except Exception:
                query_text = None

        if not query_text:
            query_text = normalized_text or json.dumps(llm_json, ensure_ascii=False)

        # 调用向量搜索器进行检索
        vector_matches = []
        try:
            from .vector_store_test import VectorSearcher
            searcher = VectorSearcher()
            results = searcher.search(query_text, top_k=5)
            # 只保留必要字段以序列化返回
            for r in results:
                # 打印每条向量检索命中，便于调试和查看匹配的问答内容
                try:
                    log_obj = {
                        'id': r.get('id'),
                        'score': r.get('score'),
                        'metadata': r.get('metadata'),
                        'text_snippet': (r.get('text') or '')[:400],
                        'related_answers': r.get('related_answers', [])
                    }
                    logger.info("向量检索命中: " + json.dumps(log_obj, ensure_ascii=False))
                except Exception as _e:
                    logger.info(f"向量检索命中（无法序列化）: id={r.get('id')} score={r.get('score')}")

                vector_matches.append({
                    'id': r.get('id'),
                    'text': r.get('text'),
                    'metadata': r.get('metadata'),
                    'score': r.get('score'),
                    'related_answers': r.get('related_answers', [])
                })
        except Exception as e:
            logger.error(f"向量检索失败: {e}")

        # 对每个识别到的实体，分别进行向量检索并打印问题与答案，便于逐实体查看检索结果
        entity_vector_matches = {}
        entity_assessments = {}
        try:
            if 'searcher' in locals():
                # 尝试用实体所在的句子（若有）作为检索查询，并在查询中加入实体类型提示
                sentences = []
                try:
                    sentences = normalized_result.get('sentences', []) if isinstance(normalized_result, dict) else []
                except Exception:
                    sentences = []

                for ent in normalized_entities:
                    ent_text = ent.get('entity') if isinstance(ent, dict) else str(ent)
                    ent_type = ent.get('type') if isinstance(ent, dict) else ''
                    if not ent_text:
                        continue

                    # 找到包含实体的句子作为上下文
                    context_sentence = None
                    try:
                        for s in sentences:
                            if ent_text in s:
                                context_sentence = s
                                break
                    except Exception:
                        context_sentence = None

                    # 构造更有语义信息的查询：优先使用实体类型+句子，其次实体本身
                    if context_sentence:
                        query_for_entity = f"实体类型:{ent_type} 上下文:{context_sentence}"
                    else:
                        query_for_entity = f"实体:{ent_text} 类型:{ent_type}"

                    try:
                        logger.info(f"开始对实体进行检索: {ent_text} -> 使用查询: {query_for_entity[:200]}")
                        ent_results = searcher.search(query_for_entity, top_k=5)
                        entity_matches = []

                        for r in ent_results:
                            md = r.get('metadata') or {}
                            q_text = None
                            answers = []
                            if md.get('type') == 'question':
                                q_text = r.get('text')
                                answers = r.get('related_answers', [])
                            elif md.get('type') == 'answer':
                                answers = [r.get('text')] + r.get('related_answers', [])
                            else:
                                q_text = r.get('text')

                            try:
                                log_entry = {
                                    'entity': ent_text,
                                    'match_id': r.get('id'),
                                    'score': r.get('score'),
                                    'question': q_text,
                                    'answers': answers,
                                    'text_snippet': (r.get('text') or '')[:400],
                                    'metadata': md
                                }
                                logger.info("实体检索命中详情: " + json.dumps(log_entry, ensure_ascii=False))
                            except Exception:
                                logger.info(f"实体检索命中: entity={ent_text} id={r.get('id')} score={r.get('score')}")

                            entity_matches.append({
                                'id': r.get('id'),
                                'score': r.get('score'),
                                'question': q_text,
                                'answers': answers,
                                'text': r.get('text'),
                                'metadata': md
                            })

                        entity_vector_matches[ent_text] = entity_matches
                    except Exception as e:
                        logger.error(f"实体检索失败 for {ent_text}: {e}")

                    # 将检索证据发回大模型，要求返回该实体在跨境传输时是否存在泄露风险并给出理由（严格 JSON）
                    try:
                        from .deepseek_client import deepseek_detect

                        evidence_snippets = "\n".join([(m.get('text') or '')[:1000] for m in entity_matches]) if entity_matches else ''
                        assess_prompt = (
                            '请仅返回严格 JSON，对下面的单条实体与其检索到的证据，给出跨境传输泄露风险判断（risk: 高|中|低|未知）和简短理由。'
                            + '\n\n格式: {"entity":"...","risk":"高|中|低|未知","reason":"简短理由，引用证据或规则"}\n\n'
                            + f'实体: {ent_text}\n证据片段:\n{evidence_snippets}'
                        )

                        assess_result = deepseek_detect(ent_text, knowledge_base_prompt=assess_prompt)
                        logger.info(f"实体评估 llm 返回: {assess_result}")

                        # 解析评估结果
                        if isinstance(assess_result, dict):
                            # 如果直接返回 risk/ reason 字段
                            risk = assess_result.get('risk') or assess_result.get('risk_level') or assess_result.get('level')
                            reason = assess_result.get('reason') or assess_result.get('explanation') or assess_result.get('detail')
                            if not risk and 'summary' in assess_result:
                                # 兼容更复杂结构
                                try:
                                    risk = assess_result.get('summary', {}).get('risk_level')
                                except Exception:
                                    risk = None

                            entity_assessments[ent_text] = {
                                'risk': risk or '未知',
                                'reason': reason or (assess_result.get('content') if isinstance(assess_result.get('content'), str) else str(assess_result))
                            }
                        elif isinstance(assess_result, str):
                            # 尝试解析为 JSON
                            try:
                                parsed = json.loads(assess_result)
                                entity_assessments[ent_text] = {
                                    'risk': parsed.get('risk') or parsed.get('risk_level') or '未知',
                                    'reason': parsed.get('reason') or parsed.get('explanation') or ''
                                }
                            except Exception:
                                entity_assessments[ent_text] = {'risk': '未知', 'reason': str(assess_result)[:1000]}
                    except Exception as e:
                        logger.error(f"实体评估失败 for {ent_text}: {e}")
                        entity_assessments[ent_text] = {'risk': '未知', 'reason': str(e)}
        except Exception as e:
            logger.error(f"逐实体检索过程出错: {e}")



        # 保存检测结果到数据库
        record = PiiDetectionRecord.objects.create(
            text=normalized_text,
            detected_entities=entities_output,
            risk_level=risk_level,
            entity_assessments=entity_assessments
        )
        serializer = PiiDetectionRecordSerializer(record)

        # 返回原文内容及向量检索结果，便于前端展示和调试
        response_text = extracted_text or normalized_text or ""

        # 包含逐实体的向量检索命中结果，便于前端展示每条实体的命中问题/答案
        return Response({
            "text": response_text,
            "entities": normalized_entities,
            "risk_level": risk_level,
            "vector_matches": vector_matches,
            "entity_vector_matches": entity_vector_matches,
            "entity_assessments": entity_assessments
        }, status=status.HTTP_201_CREATED)