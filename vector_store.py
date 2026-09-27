import os
from typing import List, Dict, Optional
from collections import Counter, defaultdict
import re
import math

import jieba
import nltk
from nltk.tokenize import word_tokenize

import chromadb
from chromadb.config import Settings
from openai import OpenAI
from tqdm import tqdm

from config import (
    VECTOR_DB_PATH,
    COLLECTION_NAME,
    OPENAI_API_BASE,
    OPENAI_EMBEDDING_MODEL,
    TOP_K,
    BM25_K1,
    BM25_B,
    RRF_K,
)


class VectorStore:

    def __init__(
        self,
        db_path: str = VECTOR_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        api_key: Optional[str] = None,
        api_base: str = OPENAI_API_BASE,
    ):
        self.db_path = db_path
        self.collection_name = collection_name

        # 注意：不能使用 `from config import OPENAI_API_KEY`，那是模块加载时按值绑定，
        # 启动时 prompt_for_api_config() 写回的 key 不会同步到此处。
        # 这里改为运行时实时从环境变量读取。
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY", "")
        self.client = OpenAI(api_key=api_key, base_url=api_base)

        # 初始化ChromaDB
        os.makedirs(db_path, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(
            path=db_path, settings=Settings(anonymized_telemetry=False)
        )

        # 获取或创建collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name, metadata={"description": "课程材料向量数据库"}
        )
        #词频
        self._bm25_index = defaultdict(dict)
        #文档长度（词频总和）
        self._bm25_doc_len = {}
        #文档长度均值
        self._bm25_avgdl = 0.0
        #索引是否构建过，初始化为否
        self._bm25_built = False
        #id到content和metadata的映射
        self._id_to_doc = {}

    def get_embedding(self, text: str) -> List[float]:
        """获取文本的向量表示

        TODO: 使用OpenAI API获取文本的embedding向量

        """
        try:
            response = self.client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"获取embedding失败: {e}")
            return []

    def add_documents(self, chunks: List[Dict[str, str]]) -> None:
        """添加文档块到向量数据库
        TODO: 实现文档块添加到向量数据库
        要求：
        1. 遍历文档块
        2. 获取文档块内容
        3. 获取文档块元数据
        5. 打印添加进度
        """
        ids = []
        documents = []
        metadatas = []
        embeddings = []
        for idx, chunk in enumerate(tqdm(chunks, desc="生成embedding并添加到向量库", unit="块")):
            content = chunk.get("content", "")
            if not content:
                continue
            #对每个chunk进行embedding
            embedding = self.get_embedding(content)
            if not embedding:
                continue
            #文档块元数据
            metadata = {
                "filename": chunk.get("filename", "unknown"),
                "filepath": chunk.get("filepath", ""),
                "filetype": chunk.get("filetype", ""),
                "page_number": str(chunk.get("page_number", 0)),
                "chunk_id": str(chunk.get("chunk_id", 0)),
            }
            #文档块ID
            doc_id = f"{chunk.get('filename', 'doc')}_{chunk.get('page_number', 0)}_{chunk.get('chunk_id', idx)}"
            
            ids.append(doc_id)
            documents.append(content)
            metadatas.append(metadata)
            embeddings.append(embedding)
        #添加到向量数据库
        if ids:
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings
            )
            print(f"\n已将 {len(ids)} 个文档块添加到向量数据库")

    def _tokenize(self, text: str) -> List[str]:
        text = (text or "").strip()
        if not text:
            return []
        tokens: List[str] = []
        chinese_parts = re.findall(r"[\u4e00-\u9fff]+", text)
        for part in chinese_parts:
            segs = [w.strip() for w in jieba.lcut(part) if w.strip()]
            tokens.extend(segs)
        english_parts = re.findall(r"[A-Za-z0-9]+(?:[A-Za-z0-9']+)?", text)
        if english_parts:
            eng_tokens = word_tokenize(" ".join(english_parts))
            tokens.extend([w.strip() for w in eng_tokens if w.strip()])
        if not tokens:
            tokens = re.findall(r"[A-Za-z0-9]+|[\u4e00-\u9fff]", text)
        return [t.lower() for t in tokens]

    def build_bm25_index(self) -> None:
        if self._bm25_built:
            return
        try:
            # 注意：chromadb 1.x 起不允许在 include 中传 "ids"，
            # ids 是默认返回字段，传了会抛 ValueError，从而被 except 吞掉，
            # 导致 BM25 索引静默为空、Hybrid 检索退化为只剩 Dense。
            data = self.collection.get(include=["documents", "metadatas"])
        except Exception:
            data = None
        if not data or not data.get("ids"):
            print("未从向量库获取到任何文档或ids，索引将为空")
            self._bm25_built = True
            return
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        total_len = 0
        for i, doc_id in enumerate(ids):
            content = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) else {}
            self._id_to_doc[doc_id] = {"content": content, "metadata": meta}
            tokens = self._tokenize(content)
            tf = Counter(tokens)
            dl = sum(tf.values())
            self._bm25_doc_len[doc_id] = dl
            total_len += dl
            for term, freq in tf.items():
                self._bm25_index[term][doc_id] = freq
        doc_count = max(len(ids), 1)
        self._bm25_avgdl = total_len / doc_count if doc_count > 0 else 0.0
        self._bm25_built = True

    def search_bm25(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        self.build_bm25_index()
        if not self._id_to_doc:
            print("BM25检索：索引为空，返回空结果")
            return []
        k1 = BM25_K1
        b = BM25_B
        tokens = self._tokenize(query)
        if not tokens:
            return []
        N = len(self._bm25_doc_len)
        scores = defaultdict(float)
        seen_docs = set()
        for term in tokens:
            postings = self._bm25_index.get(term)
            if not postings:
                continue
            df = len(postings)
            idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
            for doc_id, tf in postings.items():
                dl = self._bm25_doc_len.get(doc_id, 0) or 0.0
                denom = tf + k1 * (1 - b + b * (dl / (self._bm25_avgdl or 1.0)))
                score = idf * (tf * (k1 + 1)) / (denom if denom != 0 else 1.0)
                scores[doc_id] += score
                seen_docs.add(doc_id)
        ranked = sorted(seen_docs, key=lambda d: scores.get(d, 0.0), reverse=True)
        results = []
        for doc_id in ranked[:top_k]:
            info = self._id_to_doc.get(doc_id, {})
            results.append({
                "id": doc_id,
                "content": info.get("content", ""),
                "metadata": info.get("metadata", {}),
            })
        return results

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """搜索相关文档

        TODO: 实现向量相似度搜索
        要求：
        1. 首先获取查询文本的embedding向量（调用self.get_embedding）
        2. 使用self.collection进行向量搜索, 得到top_k个结果
        3. 格式化返回结果，每个结果包含：
           - content: 文档内容
           - metadata: 元数据（文件名、页码等）
        4. 返回格式化的结果列表
        """
        #获取查询文本的embedding向量
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []
        try:
            #使用向量搜索
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                include=["documents", "metadatas"]
            )
            #格式化返回结果
            formatted_results = []
            docs = results.get("documents") or []
            metas = results.get("metadatas") or []
            ids = results.get("ids") or []
            if docs and len(docs) > 0:
                for i, doc in enumerate(docs[0]):
                    meta = metas[0][i] if metas else {}
                    doc_id = ids[0][i] if ids else None
                    formatted_results.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": meta,
                    })
            return formatted_results
        except Exception as e:
            print(f"搜索失败: {e}")
            return []

    def search_hybrid(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        dense = self.search(query, top_k)
        sparse = self.search_bm25(query, top_k)
        rrf_k = RRF_K
        scores = defaultdict(float)
        by_id = {}
        for rank, item in enumerate(dense, 1):
            doc_id = item.get("id") or "dense_" + str(rank)
            scores[doc_id] += 1.0 / (rrf_k + rank)
            by_id[doc_id] = item
        for rank, item in enumerate(sparse, 1):
            doc_id = item.get("id") or "sparse_" + str(rank)
            scores[doc_id] += 1.0 / (rrf_k + rank)
            if doc_id not in by_id:
                by_id[doc_id] = item
        fused_ids = sorted(scores.keys(), key=lambda d: scores[d], reverse=True)
        results = []
        for doc_id in fused_ids[:top_k]:
            item = by_id.get(doc_id)
            if item:
                results.append({
                    "content": item.get("content", ""),
                    "metadata": item.get("metadata", {}),
                })
        return results

    def clear_collection(self) -> None:
        """清空collection"""
        self.chroma_client.delete_collection(name=self.collection_name)
        self.collection = self.chroma_client.create_collection(
            name=self.collection_name, metadata={"description": "课程向量数据库"}
        )
        print("向量数据库已清空")

    def get_collection_count(self) -> int:
        """获取collection中的文档数量"""
        return self.collection.count()
