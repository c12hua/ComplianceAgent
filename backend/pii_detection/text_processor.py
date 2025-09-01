import re
import unicodedata
from opencc import OpenCC
from dateutil import parser as date_parser
import os
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 文本处理工具类
class TextProcessingTools:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # 初始化LTP和spaCy PhraseMatcher
        try:
            from ltp import LTP
            self.ltp = LTP()
            self.ltp_available = True
            logger.info("✅ LTP 初始化成功")
        except (ImportError, Exception) as e:
            self.ltp = None
            self.ltp_available = False
            logger.warning(f"LTP 初始化失败: {e}")
        
        # 初始化spaCy PhraseMatcher（用于医学符号标准化）
        try:
            import spacy
            from spacy.matcher import PhraseMatcher
 
            self.nlp = spacy.blank("zh")
            self.matcher = PhraseMatcher(self.nlp.vocab, attr="TEXT")
            
            self.symbol_map = {
                # 箭头类
                "↑": "高于正常范围",
                "↓": "低于正常范围",
                "↑↑": "显著高于正常范围",
                "↓↓": "显著低于正常范围",
                "↑↑↑": "极度高于正常范围",
                "↓↓↓": "极度低于正常范围",   
            }
            
            # 医学单位标准化映射表
            self.medical_unit_map = {
                # 浓度
                "mg/L": "mg/L",
                "μg/mL": "μg/mL", "ug/mL": "μg/mL",
                "ng/mL": "ng/mL", "ng/L": "ng/L",
                "pg/mL": "pg/mL",
                "pmol/L": "pmol/L",
                "nmol/L": "nmol/L",
                "μmol/L": "μmol/L", "umol/L": "μmol/L",
                "mmol/L": "mmol/L",
                "mol/L": "mol/L",
                "mg/dL": "mg/dL", "mg/100mL": "mg/dL",
                "g/dL": "g/dL",
                "g/L": "g/L",
                "mEq/L": "mmol/L",
                
                # 活性/酶学
                "U/L": "U/L",
                "IU/L": "IU/L",
                "IU/mL": "IU/mL",
                "mIU/L": "mIU/L",
                "μIU/mL": "μIU/mL",
                "kU/L": "kU/L",
                "U/gHb": "U/gHb",
                
                # 血液学（计数）
                "×10⁹/L": "10^9/L", "10^9/L": "10^9/L",
                "×10¹²/L": "10^12/L", "10^12/L": "10^12/L",
                "fL": "fL",
                "pL": "pL",
                "μL": "μL", "uL": "μL",
                
                # 血气/生理
                "mmHg": "mmHg",
                "mL/min": "mL/min",
                "L/min": "L/min",
                "mL/dL": "mL/dL",
                "vol%": "vol%",
                "mOsm/kg": "mOsm/kg",
                "mmol/kg": "mmol/kg",
                
                # 体格参数
                "°C": "°C",
                "次/分": "次/分",
                "kg/m²": "kg/m²",
                "mg/kg": "mg/kg",
                "mg/m²": "mg/m²",
                "μg/kg": "μg/kg",
                
                # 影像/尺寸/时间
                "mm": "mm",
                "cm": "cm",
                "Hz": "Hz",
                "kHz": "kHz",
                "MHz": "MHz",
                "ms": "ms",
                "s": "s",
                "min": "min",
                "h": "h",
                
                # 微生物学
                "copies/mL": "copies/mL",
                "PFU/mL": "PFU/mL",
                "CFU/mL": "CFU/mL",
                "TCID50/mL": "TCID50/mL",
            }
            
            # 创建符号匹配模式（按长度排序，优先匹配长符号）
            symbols = sorted(self.symbol_map.keys(), key=len, reverse=True)
            patterns = [self.nlp.make_doc(symbol) for symbol in symbols]
            self.matcher.add("MEDICAL_SYMBOLS", patterns)
            
            # 创建医学单位匹配器（用于保护单位不被符号替换）
            self.unit_matcher = PhraseMatcher(self.nlp.vocab, attr="TEXT")
            units = sorted(self.medical_unit_map.keys(), key=len, reverse=True)
            unit_patterns = [self.nlp.make_doc(unit) for unit in units]
            self.unit_matcher.add("MEDICAL_UNITS", unit_patterns)
            
            self.spacy_available = True
            logger.info("✅ spaCy PhraseMatcher 初始化成功")
            
        except ImportError as e:
            self.nlp = None
            self.matcher = None
            self.unit_matcher = None
            self.symbol_map = {}
            self.medical_unit_map = {}
            self.spacy_available = False
            logger.warning(f"spaCy 初始化失败: {e}，医学符号标准化功能不可用")

        self.opencc = OpenCC('t2s')
        
        # 预编译正则表达式
        self._compile_regex_patterns()
        
        self._initialized = True
    
    def _compile_regex_patterns(self):
        """预编译所有正则表达式"""
        # HTML标签（仅匹配以字母开头的有效标签，避免误删如 <文本>）
        self.html_tag_pattern = re.compile(r'</?\s*[a-zA-Z][^>]*>')
        
        # 噪音字符
        self.noise_pattern = re.compile(r'[@#*]{3,}')
        
        # 科学计数法
        self.scientific_pattern = re.compile(r'(\d+(?:\.\d+)?)\s*×\s*10\s*[⁰¹²³⁴⁵⁶⁷⁸⁹]+')
        
        # 空白字符处理
        self.whitespace_pattern = re.compile(r'[^\S\n]+')
        self.newline_space_pattern = re.compile(r' *\n *')
        
        # 重复标点符号
        self.repeat_punct_patterns = {
            'exclamation': re.compile(r'[!]{2,}'),
            'question': re.compile(r'[?]{2,}'),
            'semicolon': re.compile(r'[;]{2,}'),
            'chinese_exclamation': re.compile(r'[！]{2,}'),
            'chinese_question': re.compile(r'[？]{2,}'),
            'chinese_semicolon': re.compile(r'[；]{2,}'),
            'other_punct': re.compile(r'[，、：""''（）【】]{2,}')
        }
        
        # 数字中的逗号
        self.number_comma_pattern = re.compile(r'(?<=\d)[,，](?=\d)')
        
        # 句子分割
        self.sentence_split_pattern = re.compile(r'(?:[。！？；?!;]|…|\.\.\.|\\n|\n)+')
        
        # 页码
        self.page_patterns = {
            'chinese': re.compile(r'第\s*\d+\s*页'),
            'english': re.compile(r'Page\s*\d+')
        }
        
        # 时间格式保护
        self.time_pattern = re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AP]M)?\b', re.IGNORECASE)
        
        # 其他标点删除
        self.punct_remove_pattern = re.compile(r'[、""''【】\[\]]')
        
        # 日期格式
        self.date_pattern = re.compile(
            r'\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*[日号]|'  # 中文格式
            r'\b(?:19|20)\d{2}\s*[-/.]\s*(?:0?[1-9]|1[0-2])\s*[-/.]\s*(?:0?[1-9]|[12]\d|3[01])\b|'   # YYYY-MM-DD
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*\d{1,2}[，,]?\s*\d{4}|' # Mon DD YYYY
            r'\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{4}|' # DD Mon YYYY
            r'(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\d{1,2}\d{4}|' # MonDDYYYY
            r'\d{1,2}(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\d{4}|' # DDMonYYYY
            r'\b(?:19|20)\d{6}\b', # YYYYMMDD
            re.IGNORECASE
        )
        
        # 连字符
        self.hyphen_pattern = re.compile(r'(?<=\d)[\-–—]+(?=\d)')
        
        # 多空格
        self.multi_space_pattern = re.compile(r'\s+')
        
        # 中文标点符号（用于文本清理）
        self.chinese_punct = '，。！？；：""''（）【】、'

# 全局实例
text_tools = TextProcessingTools()

class TextProcessor:
    """文本标准化和分词处理类"""
    
    def __init__(self):
        self.text_tools = text_tools
    
    def normalize_text(self, text: str) -> dict:
        """
        文本标准化主函数
        """
        if not text:
            return self._get_empty_result()
        
        original_text = text

        text, scientific_notations = self._basic_cleanup(text)

        sentences = self.text_tools.sentence_split_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        paragraphs = text.split('\n\n')
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        normalized_sentences = self._process_sentences(text)

        standardized_text = '[' + ','.join([f"'{s}'" for s in normalized_sentences]) + ']'

        standardized_text = self._restore_placeholders(standardized_text, scientific_notations)

        structured_text = self.tokenize_and_segment(standardized_text)
        
        return {
            'normalized_text': standardized_text,
            'sentences': sentences,
            'paragraphs': paragraphs,
            'original_length': len(original_text),
            'normalized_length': len(standardized_text),
            'structured_text': structured_text
        }
    
    def _get_empty_result(self) -> dict:
        """
        返回空文本的标准结果
        """
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
    
    def _basic_cleanup(self, text: str) -> tuple:
        """
        基础文本清理
        """
        # 去除HTML标签和噪音字符
        text = self.text_tools.html_tag_pattern.sub('', text)
        text = self.text_tools.noise_pattern.sub('', text)
        
        # 保护科学计数法
        scientific_notations = []
        def protect_scientific_notation(match):
            placeholder = f"__SCIENTIFIC_{len(scientific_notations)}__"
            scientific_notations.append(match.group(0))
            return placeholder
        text = self.text_tools.scientific_pattern.sub(protect_scientific_notation, text)

        # 性别符号标准化
        text = text.replace('♂', '男').replace('♀', '女')

        # 合并重复乘号，仅保留一个
        text = re.sub(r'×{2,}', '×', text)

        # 保护立方单位，避免 ³ 被归一化为 3
        try:
            text = re.sub(r'(?<=/)\b(mm|cm|m)³\b', r'\1^3', text)
            text = re.sub(r'\b(mm|cm|m)³\b', r'\1^3', text)
        except Exception:
            pass
        
        # 标准化数学符号
        math_symbols = {
            '≥': '>=', '≤': '<=', '≠': '!=', '≈': '≈',
            '±': '±', '∞': '∞', '∑': '∑', '∫': '∫', '√': '√'
        }
        for symbol, replacement in math_symbols.items():
            text = text.replace(symbol, replacement)
        
        # 保护时间，避免后续将冒号替换为空格
        time_placeholders = []
        def protect_time(match):
            placeholder = f"__TIME_{len(time_placeholders)}__"
            time_placeholders.append(match.group(0))
            return placeholder
        text = self.text_tools.time_pattern.sub(protect_time, text)

        # 将具有分词作用的标点替换为空格（不包括分句符号与逗号/句内点）
        # 包括：冒号、单双引号、各类括号
        text = re.sub(r'[\:\："\""\'\'\'\'\(\)（）\[\]【】]', ' ', text)

        # 恢复时间占位符
        for i, t in enumerate(time_placeholders):
            text = text.replace(f"__TIME_{i}__", t)

        # Unicode标准化和换行符处理
        text = unicodedata.normalize('NFKC', text)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 空白字符统一
        text = self.text_tools.whitespace_pattern.sub(' ', text)
        text = self.text_tools.newline_space_pattern.sub('\n', text)
        text = text.strip()
        
        # 标点符号处理
        for pattern_name, pattern in self.text_tools.repeat_punct_patterns.items():
            if pattern_name == 'other_punct':
                text = pattern.sub(lambda m: m.group()[0], text)
            else:
                text = pattern.sub(lambda m: m.group()[0], text)
        
        # 数字中的逗号
        text = self.text_tools.number_comma_pattern.sub('', text)
        
        # 页码处理
        for pattern in self.text_tools.page_patterns.values():
            text = pattern.sub('', text)
        
        # 繁体转简体
        text = self.text_tools.opencc.convert(text)
        
        # 医学符号标准化（先标准化单位，再处理符号）
        if self.text_tools.spacy_available:
            text = self._standardize_medical_units(text)
            text = self._standardize_medical_symbols(text)
        
        return text, scientific_notations
    
    def _process_sentences(self, text: str) -> list:
        """
        处理句子级别的标准化
        """
        raw_sentences = self.text_tools.sentence_split_pattern.split(text)
        normalized_sentences = []
        
        for s in raw_sentences:
            s = s.strip()
            if not s:
                continue
            
            # 统一逗号为中文逗号
            s = s.replace(',', '，')
            
            # 保护括号
            s = s.replace('（', '(').replace('）', ')')
            
            # 时间格式保护
            time_colons = []
            def protect_time_colon(match):
                placeholder = f"__TIME_COLON_{len(time_colons)}__"
                time_colons.append(match.group(0))
                return placeholder
            
            s = self.text_tools.time_pattern.sub(protect_time_colon, s)
            
            # 删除其他标点（在删除前添加空格分隔）
            s = self.text_tools.punct_remove_pattern.sub(' ', s)
            
            # 恢复时间格式
            for i, time_colon in enumerate(time_colons):
                placeholder = f"__TIME_COLON_{i}__"
                s = s.replace(placeholder, time_colon)
            
            # 日期标准化
            s = self.text_tools.date_pattern.sub(lambda m: self._normalize_date_with_datetime(m.group(0)), s)
            
            # 去除连字符
            s = self.text_tools.hyphen_pattern.sub('', s)
            
            # 处理空格
            s = self.text_tools.multi_space_pattern.sub(' ', s)
            normalized_sentences.append(s)
        
        return normalized_sentences
    
    def _normalize_date_with_datetime(self, date_str: str) -> str:
        """
        使用 dateutil.parser 解析各种日期格式
        """
        try:
            original_str = date_str.strip()
            try:
                dt = date_parser.parse(original_str)
            except:
                try:
                    dt = date_parser.parse(original_str, dayfirst=True)
                except:
                    dt = date_parser.parse(original_str, fuzzy=True)
            return dt.strftime('%Y-%m-%d')
        except Exception:
            return date_str
    
    def _standardize_medical_units(self, text: str) -> str:
        """
        标准化医学单位
        """
        try:
            # 创建spaCy文档
            doc = self.text_tools.nlp(text)
            
            # 获取匹配的单位
            matches = self.text_tools.unit_matcher(doc)
            
            if not matches:
                return text
            
            # 按位置排序匹配结果（从后往前替换，避免位置偏移）
            matches = sorted(matches, key=lambda x: x[1], reverse=True)
            
            # 替换匹配的单位
            result_text = text
            for match_id, start, end in matches:
                span = doc[start:end]
                matched_unit = span.text
                replacement = self.text_tools.medical_unit_map.get(matched_unit, matched_unit)
                
                # 计算字符位置并替换
                char_start = span.start_char
                char_end = span.end_char
                result_text = result_text[:char_start] + replacement + result_text[char_end:]
            
            return result_text
            
        except Exception as e:
            logger.warning(f"医学单位标准化失败: {e}")
            return text
    
    def _standardize_medical_symbols(self, text: str) -> str:
        """
        使用spaCy匹配和替换医学符号为标准化含义
        """
        try:
            # 创建spaCy文档
            doc = self.text_tools.nlp(text)
            
            # 先获取医学单位的位置，避免在单位中替换符号
            unit_matches = self.text_tools.unit_matcher(doc)
            protected_ranges = set()
            for match_id, start, end in unit_matches:
                for i in range(start, end):
                    protected_ranges.add(i)
            
            # 获取匹配的符号
            symbol_matches = self.text_tools.matcher(doc)
            
            if not symbol_matches:
                return text
            
            # 按位置排序匹配结果（从后往前替换，避免位置偏移）
            symbol_matches = sorted(symbol_matches, key=lambda x: x[1], reverse=True)
            
            # 替换匹配的符号
            result_text = text
            for match_id, start, end in symbol_matches:
                # 检查是否在受保护的单位范围内
                if any(i in protected_ranges for i in range(start, end)):
                    continue
                    
                span = doc[start:end]
                matched_symbol = span.text
                
                # 检查是否为独立的符号（避免误匹配字母）
                if self._is_valid_symbol_match(doc, start, end, matched_symbol):
                    replacement = self.text_tools.symbol_map.get(matched_symbol, matched_symbol)
                    
                    # 计算字符位置并替换
                    char_start = span.start_char
                    char_end = span.end_char
                    result_text = result_text[:char_start] + replacement + result_text[char_end:]
            
            return result_text
            
        except Exception as e:
            logger.warning(f"医学符号标准化失败: {e}")
            return text
    
    def _is_valid_symbol_match(self, doc, start: int, end: int, symbol: str) -> bool:
        """
        验证符号匹配是否有效，避免误匹配字母
        """
        # 对于单字母符号（H, L, N），需要更严格的上下文检查
        if symbol in ['H', 'L', 'N'] and len(symbol) == 1:
            # 检查前后是否有数字或特定医学上下文
            prev_token = doc[start-1] if start > 0 else None
            next_token = doc[end] if end < len(doc) else None
            
            # 如果前面是数字，后面是空格或标点，可能是检验结果
            if prev_token and prev_token.text.replace('.', '').replace(',', '').isdigit():
                return True
            
            # 如果在医学检验相关的上下文中
            context_words = []
            for i in range(max(0, start-3), min(len(doc), end+3)):
                context_words.append(doc[i].text.lower())
            
            medical_context = ['血', '尿', '检', '验', '结果', '报告', '值', '范围', '正常', '异常']
            if any(word in ''.join(context_words) for word in medical_context):
                return True
            
            # 否则可能是普通字母，不替换
            return False
        
        # 对于符号类（+, -, ↑, ↓等）和多字符缩写（NEG, POS等），直接匹配
        return True
    
    def _restore_placeholders(self, text: str, scientific_notations: list) -> str:
        """
        恢复占位符
        """
        for i, scientific_notation in enumerate(scientific_notations):
            placeholder = f"__SCIENTIFIC_{i}__"
            text = text.replace(placeholder, scientific_notation)
        return text

    def tokenize_and_segment(self, standardized_text: str) -> dict:
        """
        对标准化文本进行LTP分词分句处理
        """
        try:
            # 解析标准化文本
            sentences = self._parse_standardized_text(standardized_text)
            
            # 对每个句子进行LTP分词
            tokenized_sentences = []
            for sentence in sentences:
                if not sentence.strip():
                    continue

                tokens = self._tokenize_sentence(sentence)

                if tokens:
                    tokenized_sentence = "['" + "', '".join(tokens) + "']"
                    tokenized_sentences.append(tokenized_sentence)
            
            # 创建完整的结构化文本（保持原有格式用于返回）
            structured_text = "".join(tokenized_sentences) if tokenized_sentences else ""

            logger.info(f"分词分句完成，句子数量: {len(tokenized_sentences)}")
            logger.info(f"结构化文本示例: {structured_text[:200]}...")

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
    
    def _parse_standardized_text(self, standardized_text: str) -> list:
        """
        解析标准化文本，提取句子列表
        """
        text_content = standardized_text.strip()
        if text_content.startswith('[') and text_content.endswith(']'):
            text_content = text_content[1:-1]
        
        sentences = []
        current_sentence = ""
        in_quotes = False
        
        for char in text_content:
            if char == "'" and (not current_sentence or current_sentence[-1] != '\\'):
                in_quotes = not in_quotes
            elif char == ',' and not in_quotes:
                if current_sentence.strip():
                    sentence = current_sentence.strip().strip("'")
                    sentences.append(sentence)
                current_sentence = ""
            else:
                current_sentence += char
        
        # 处理最后一个句子
        if current_sentence.strip():
            sentence = current_sentence.strip().strip("'")
            sentences.append(sentence)
        
        return sentences
    
    def _tokenize_sentence(self, sentence: str) -> list:
        """
        对单个句子进行分词 - 使用LTP分词
        """
        # 使用空格作为硬切分边界：先按空格切段，再分别用LTP分词
        parts = [p for p in re.split(r'\s+', sentence.strip()) if p]
        if not parts:
            return []

        ltp_result = self.text_tools.ltp.pipeline(parts, tasks=["cws"])
        parts_cws = ltp_result.cws if hasattr(ltp_result, 'cws') else ltp_result['cws']

        tokens = []
        for seg_tokens in parts_cws:
            tokens.extend(seg_tokens)
        
        logger.info(f"LTP分词结果: {tokens}")

        cleaned_tokens = []
        for token in tokens:
            # 若是时间格式，保留冒号
            if self.text_tools.time_pattern.fullmatch(token):
                cleaned_token = re.sub(r'[\s\(\)（）【】\[\]\"\""\'\'\'\']', '', token)
            else:
                # 分词后再清理：冒号、单双引号、括号等具有分词效果的符号
                cleaned_token = re.sub(r'[\s:\：\(\)（）【】\[\]\"\""\'\'\'\']', '', token)
            if cleaned_token:
                cleaned_tokens.append(cleaned_token)

        # 合并医学术语（剂量和单位）
        merged_tokens = self._merge_medical_terms(cleaned_tokens)

        return merged_tokens

    def _merge_medical_terms(self, tokens: list) -> list:
        """
        合并常见医学用药表达（剂量+单位，频次，给药途径）
        例如：['0.2', 'g', 'tid', 'po'] -> ['0.2g', 'tid', 'po']
        """
        if not tokens:
            return tokens

        # 统一使用小写进行集合匹配
        unit_set = {
            'mg', 'g', 'μg', 'ug', 'mcg', 'kg', 'ml', 'l', 'iu', 'u', 'ul', 'μl',
            'gtt', '片', '粒', '袋', '支', '滴'
        }
        freq_set = {
            'qd', 'bid', 'tid', 'qid', 'qod', 'qam', 'qpm', 'qn', 'qhs', 'hs',
            'q1h', 'q2h', 'q3h', 'q4h', 'q6h', 'q8h', 'q12h', 'q24h', 'q48h', 'q72h',
            'prn', 'stat', 'ac', 'pc', 'biw', 'tiw', 'qwk'
        }
        route_set = {
            'po', 'iv', 'im', 'sc', 'sl', 'pr', 'top', 'inh', 'id', 'it', 'ia',
            'oph', 'otic', 'nasal', 'pv', 'buccal', 'transdermal'
        }

        def is_number_token(t: str) -> bool:
            return bool(re.fullmatch(r'\d+(?:\.\d+)?', t))

        merged = []
        i = 0
        n = len(tokens)
        while i < n:
            t = tokens[i]
            tl = t.lower()
            # 合并 数字 + 单位（如 0.2 g -> 0.2g）
            if i + 1 < n and is_number_token(t) and tokens[i+1].lower() in unit_set:
                merged_tok = f"{t}{tokens[i+1].lower()}"
                # 进一步合并如 0.2g / kg -> 0.2g/kg
                if i + 3 < n and tokens[i+2] == '/' and tokens[i+3].lower() in {'kg', 'm2', 'm^2', 'm²', 'd', 'day', 'h', 'hr'}:
                    qual = tokens[i+3].lower().replace('m2', 'm^2').replace('m²', 'm^2').replace('hr', 'h').replace('day', 'd')
                    merged.append(f"{merged_tok}/{qual}")
                    i += 4
                    continue
                merged.append(merged_tok)
                i += 2
                continue
            # 合并 单位 / 限定（如 mg / kg -> mg/kg）
            if i + 2 < n and tl in unit_set and tokens[i+1] == '/' and tokens[i+2].lower() in {'kg', 'm2', 'm^2', 'm²', 'd', 'day', 'h', 'hr'}:
                qual = tokens[i+2].lower().replace('m2', 'm^2').replace('m²', 'm^2').replace('hr', 'h').replace('day', 'd')
                merged.append(f"{tl}/{qual}")
                i += 3
                continue
            # 其它直接加入
            merged.append(t)
            i += 1

        normalized = []
        for t in merged:
            if t.lower() in freq_set or t.lower() in route_set:
                normalized.append(t.lower())
            else:
                normalized.append(t)

        return normalized

    def save_structured_text(self, structured_text: str):
        """
        保存结构化文本到文件
        文件名格式：病例文本_时间戳
        保存路径：/backend/knowledge_base/text/
        """
        try:
            output_dir = "/backend/knowledge_base/text/"
            os.makedirs(output_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"病例文本_{timestamp}.json"
            filepath = os.path.join(output_dir, filename)

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

# 创建全局实例
text_processor = TextProcessor()