import json
import os
import logging
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_knowledge_base() -> List[Dict[str, Any]]:
    """加载知识库文件"""
    try:
        with open('backend/knowledge_base/merge.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"加载知识库文件失败: {e}")
        raise

class KnowledgeUnit:
    def __init__(self, text: str, metadata: Dict[str, Any], id_: str):
        self.text = text
        self.metadata = metadata
        self.id = id_

def process_knowledge_base(knowledge_base: List[Dict[str, Any]]) -> List[KnowledgeUnit]:
    """处理知识库数据，返回KnowledgeUnit列表"""
    units = []
    counter = 0

    try:
        for item in knowledge_base:
            # 提取问题
            units.append(KnowledgeUnit(
                text=item['question'],
                metadata={
                    'type': 'question',
                    'category': item.get('category', ''),
                    'answer_index': str(counter + 1)
                },
                id_=f"q_{counter}"
            ))
            counter += 1

            # 提取答案
            for ans in item['answer']:
                units.append(KnowledgeUnit(
                    text=ans,
                    metadata={
                        'type': 'answer',
                        'category': item.get('category', ''),
                        'question_index': str(counter - 1)
                    },
                    id_=f"a_{counter}"
                ))
                counter += 1

            # 提取相关上下文
            if 'positive_contexts' in item:
                for ctx in item['positive_contexts']:
                    units.append(KnowledgeUnit(
                        text=ctx['content'],
                        metadata={
                            'type': 'context',
                            'category': item.get('category', ''),
                            'source': ctx.get('source', ''),
                            'question_index': str(counter - 2)
                        },
                        id_=f"c_{counter}"
                    ))
                    counter += 1

        return units
    except Exception as e:
        logger.error(f"处理知识库数据失败: {e}")
        raise

def create_vector_store():
    """创建并保存向量数据库"""
    try:
        # 加载并处理知识库
        knowledge_base = load_knowledge_base()
        knowledge_units = process_knowledge_base(knowledge_base)
        logger.info(f"成功处理知识库文件，共 {len(knowledge_units)} 条数据")

        # 初始化模型并生成向量
        model = SentenceTransformer('shibing624/text2vec-base-chinese')
        logger.info("开始生成文本向量...")
        texts = [unit.text for unit in knowledge_units]
        embeddings = model.encode(texts)
        logger.info(f"完成向量生成，共处理 {len(texts)} 条文本")

        # 创建FAISS索引
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype(np.float32))

        # 保存索引和数据
        os.makedirs('backend/knowledge/vector_store', exist_ok=True)
        faiss.write_index(index, 'backend/knowledge/vector_store/knowledge_base.index')

        # 保存元数据
        metadata = {
            'texts': texts,
            'metadatas': [unit.metadata for unit in knowledge_units],
            'ids': [unit.id for unit in knowledge_units]
        }
        with open('backend/knowledge/vector_store/metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        logger.info(f"成功保存向量数据库和元数据!")
        return index, knowledge_units
    except Exception as e:
        logger.error(f"创建向量数据库失败: {e}")
        raise

if __name__ == "__main__":
    try:
        create_vector_store()
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise