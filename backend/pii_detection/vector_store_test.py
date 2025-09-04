import json
import logging
from typing import List, Dict, Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from django.conf import settings

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorSearcher:
    def __init__(self, index_path: str = 'backend/knowledge/vector_store/knowledge_base.index',
                 metadata_path: str = 'backend/knowledge/vector_store/metadata.json',
                 model_name: str = 'shibing624/text2vec-base-chinese'):
        """初始化向量搜索器"""
        self.model = SentenceTransformer(model_name)
        # Resolve relative paths against Django BASE_DIR to get absolute paths inside container
        try:
            base_dir = Path(settings.BASE_DIR)
        except Exception:
            base_dir = Path(__file__).resolve().parents[2]

        # project root is parent of backend (BASE_DIR points to backend/)
        project_root = base_dir.parent

        def resolve_path(p: str) -> Path:
            p = str(p)
            # if caller passed a path starting with 'backend/', strip it because files live at project root
            if p.startswith('backend/'):  # avoid backend/backend/... when joining
                p = p[len('backend/') :]
            path = Path(p)
            if not path.is_absolute():
                candidate = project_root / path
                # if the candidate doesn't exist, try a few common alternative locations
                if not candidate.exists():
                    # search for a file with same name under project_root
                    try:
                        matches = list(project_root.rglob(path.name))
                        if matches:
                            return matches[0]
                    except Exception:
                        pass
                return candidate
            return path

        self.index_path = resolve_path(index_path)
        self.metadata_path = resolve_path(metadata_path)

        self.index = self._load_index(str(self.index_path))
        self.metadata = self._load_metadata(str(self.metadata_path))

    def _load_index(self, index_path: str) -> faiss.Index:
        """加载FAISS索引"""
        try:
            p = Path(index_path)
            if not p.exists():
                raise FileNotFoundError(f"FAISS index not found at: {index_path}")
            return faiss.read_index(str(p))
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            raise

    def _load_metadata(self, metadata_path: str) -> Dict[str, Any]:
        """加载元数据"""
        try:
            p = Path(metadata_path)
            if not p.exists():
                raise FileNotFoundError(f"Metadata file not found at: {metadata_path}")
            with open(str(p), 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载元数据失败: {e}")
            raise

    def search(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """搜索最相似的文本和相关答案"""
        try:
            # 将查询文本转换为向量
            query_vector = self.model.encode([query])
            
            # 使用FAISS进行搜索
            scores, indices = self.index.search(query_vector.astype(np.float32), top_k)
            
            # 整理搜索结果
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.metadata['texts']):  # 确保索引有效
                    text = self.metadata['texts'][idx]
                    metadata = self.metadata['metadatas'][idx]
                    result_id = self.metadata['ids'][idx]
                    
                    # 查找相关答案
                    related_answers = []
                    current_id = result_id.split('_')[1]  # 获取ID的数字部分
                    
                    if metadata['type'] == 'question':
                        logger.info(f"处理问题ID: {current_id}")
                        # 查找问题对应的所有答案
                        for i, (meta, text_i, id_i) in enumerate(zip(
                            self.metadata['metadatas'],
                            self.metadata['texts'],
                            self.metadata['ids']
                        )):
                            if (meta['type'] == 'answer' and 
                                meta.get('question_index', '') == current_id):
                                related_answers.append(f"答案：{text_i}")
                                logger.info(f"找到相关答案 ID: {id_i}")
                    
                    elif metadata['type'] == 'answer':
                        logger.info(f"处理答案ID: {current_id}")
                        # 查找答案对应的问题
                        q_index = metadata.get('question_index', '')
                        if q_index:
                            for i, (meta, text_i, id_i) in enumerate(zip(
                                self.metadata['metadatas'],
                                self.metadata['texts'],
                                self.metadata['ids']
                            )):
                                if (meta['type'] == 'question' and 
                                    f"q_{q_index}" == id_i):
                                    related_answers.append(f"问题：{text_i}")
                                    logger.info(f"找到相关问题 ID: {id_i}")
                                    break
                        related_answers.append(f"答案：{text}")

                    result = {
                        'text': text,
                        'metadata': metadata,
                        'id': result_id,
                        'score': float(score),
                        'related_answers': related_answers
                    }
                    results.append(result)
            
            return results
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            raise

def main():
    """测试向量搜索功能"""
    try:
        # 初始化搜索器
        searcher = VectorSearcher()
        logger.info("向量搜索器初始化成功")

        # 测试查询
        test_queries = [
            "医疗数据安全管理的要求是什么？",
            "如何保护个人隐私数据？",
            "数据跨境传输的规定"
        ]

        # 执行搜索并打印结果
        for query in test_queries:
            logger.info(f"\n查询: {query}")
            results = searcher.search(query)
            
            logger.info(f"\n找到 {len(results)} 条相关结果:")
            for i, result in enumerate(results, 1):
                logger.info(f"\n结果 {i}:")
                logger.info(f"ID: {result['id']}")
                logger.info(f"匹配文本: {result['text']}")
                logger.info(f"类型: {result['metadata']['type']}")
                logger.info(f"分类: {result['metadata']['category']}")
                logger.info(f"相似度得分: {result['score']}")
                
                if result['related_answers']:
                    logger.info("\n关联内容:")
                    for answer in result['related_answers']:
                        logger.info(answer)
                else:
                    logger.info("\n未找到关联内容")
                
                logger.info("-" * 50)

    except Exception as e:
        logger.error(f"测试过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()
