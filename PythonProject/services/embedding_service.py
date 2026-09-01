# services/embedding_service.py - 语义向量引擎（纯 numpy 实现，零外部依赖）
# 使用字符级 n-gram TF-IDF + 余弦相似度，完美支持中英文
# 无需 jieba / sklearn / scipy / torch / sentence-transformers
import math
import re
import logging
import json
import numpy as np
from collections import Counter

logger = logging.getLogger(__name__)

# 字符级 n-gram 窗口大小（2-5 字符，覆盖中文词语）
_NGRAM_RANGES = [(2, 3), (3, 4)]
# 默认词汇表最大大小（防止无限增长）
_MAX_VOCAB = 50000
# 停用词（英文）
_ENG_STOP_WORDS = frozenset(
    "a an the of and or in to for is are was were be been being "
    "have has had do does did will would shall should may might can "
    "could not no but if then than that this these those it its "
    "he she they him her them their with from on at by as into over "
    "under through after before between out up down about all any each "
    "every both few more most some such only very too just also already "
    "still now here there when where why how what who whom whose "
    "i me my we our us your they it you he she".split()
)


# ============================================================
# 预处理
# ============================================================
# 通用分隔符：中英文标点
_DELIM_RE = re.compile(
    r"[\s,.\-—–;:;:\!\?？、，。！？：；（）()「」『』""''【】\[\]<>《》\t\n\r/\\|~`@#$%^&*_+=]+",
    re.UNICODE,
)


def _tokenize(text: str) -> list[str]:
    """字符级 n-gram 分词。对中文直接按字符滑动，英文单词整体保留。"""
    if not text:
        return []
    text = text.strip().lower()
    # 替换所有分隔符为单个空格
    text = _DELIM_RE.sub(" ", text)
    # 去除多余空格
    text = " ".join(text.split())

    ngrams = []
    # 英文单词整体保留
    tokens = text.split(" ")
    for token in tokens:
        if not token:
            continue
        # 判断是否为英文（不含中文字符）
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in token)
        if not has_chinese and len(token) >= 2:
            # 英文单词
            if token not in _ENG_STOP_WORDS and len(token) > 2:
                ngrams.append(token)
            continue

        # 中文或混合：字符级 n-gram
        for n_low, n_high in _NGRAM_RANGES:
            for n in range(n_low, n_high + 1):
                if len(token) >= n:
                    for i in range(len(token) - n + 1):
                        ngram = token[i:i + n]
                        # 过滤纯标点或全空白
                        if any(c.isalnum() or '\u4e00' <= c <= '\u9fff' for c in ngram):
                            ngrams.append(ngram)

    return ngrams


# ============================================================
# 词汇表构建
# ============================================================
class _Vocabulary:
    """轻量级词汇表：动态维护 term → id 映射，最多 _MAX_VOCAB 个词"""
    __slots__ = ("_word2id", "_id2word", "_word_freq")

    def __init__(self):
        self._word2id: dict[str, int] = {}
        self._id2word: list[str] = []
        self._word_freq: Counter = Counter()

    @property
    def size(self) -> int:
        return len(self._word2id)

    def add(self, text: str):
        """将一段文本加入词汇表"""
        for token in _tokenize(text):
            if token in self._word2id:
                self._word_freq[token] += 1
            elif self.size < _MAX_VOCAB:
                idx = len(self._id2word)
                self._word2id[token] = idx
                self._id2word.append(token)
                self._word_freq[token] = 1

    def vectorize(self, text: str) -> np.ndarray:
        """将文本转为稀疏向量（TF 形式）"""
        vec = np.zeros(self.size) if self.size > 0 else np.zeros(1)
        tokens = _tokenize(text)
        if not tokens or self.size == 0:
            return vec
        for token in tokens:
            idx = self._word2id.get(token)
            if idx is not None:
                vec[idx] += 1.0
        return vec

    def dump(self) -> dict:
        return {
            "size": self.size,
            "word2id": self._word2id,
            "id2word": self._id2word,
        }

    @classmethod
    def load(cls, data: dict) -> "_Vocabulary":
        v = cls()
        v._word2id = data.get("word2id", {})
        v._id2word = data.get("id2word", [])
        return v


# ============================================================
# TF-IDF 引擎
# ============================================================
class SemanticIndex:
    """
    语义索引：基于字符级 n-gram TF-IDF 的轻量级向量搜索引擎。

    工作流程：
    1. 构建词汇表（遍历所有文档）
    2. 计算每个文档的 TF-IDF 向量
    3. 搜索时计算查询向量与文档向量的余弦相似度
    """

    def __init__(self):
        self.vocab = _Vocabulary()
        self._doc_vectors: list[np.ndarray] = []   # 每个文档的 TF-IDF 向量
        self._doc_ids: list[int] = []               # 对应的 kb_id
        self._doc_texts: list[str] = []              # 原始文本（用于重索引）
        self._idf: np.ndarray = np.array([])
        self._needs_rebuild: bool = True

    @property
    def count(self) -> int:
        return len(self._doc_vectors)

    # ── 文档管理 ──

    def add_document(self, doc_id: int, text: str):
        """添加文档到索引"""
        self._doc_ids.append(doc_id)
        self._doc_texts.append(text)
        self.vocab.add(text)
        self._needs_rebuild = True

    def remove_document(self, doc_id: int):
        """移除文档"""
        idx = self._doc_ids.index(doc_id)
        self._doc_ids.pop(idx)
        self._doc_texts.pop(idx)
        self._doc_vectors.pop(idx)
        self._needs_rebuild = True

    def rebuild(self):
        """重建 TF-IDF 向量（添加/删除后必须调用）"""
        if not self._needs_rebuild:
            return

        n = len(self._doc_texts)
        if n == 0:
            self._doc_vectors = []
            self._idf = np.array([])
            self._needs_rebuild = False
            return

        # 构建矩阵：rows = docs, cols = vocab
        vecs = [self.vocab.vectorize(t) for t in self._doc_texts]
        self._doc_vectors = vecs

        vocab_size = self.vocab.size
        if vocab_size > 0:
            # 计算每列（每个词）的文档频率（DF）
            # df[col] = 有多少文档包含这个词
            doc_freq = np.zeros(vocab_size)
            for v in vecs:
                doc_freq += (v > 0).astype(float)

            # IDF: log((N+1) / (1 + df)) + 1（平滑）
            idf = np.log((n + 1) / (1 + doc_freq)) + 1
            # 将 IDF 广播到每行
            self._doc_vectors = [v * idf for v in vecs]
            self._idf = idf

        self._needs_rebuild = False

    # ── 搜索 ──

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        语义搜索。返回按余弦相似度降序排列的结果。
        每个结果: {"id": kb_id, "score": float, "text": original_text}
        """
        if self._needs_rebuild:
            self.rebuild()

        if self.count == 0:
            return []

        query_vec = self.vocab.vectorize(query)
        if np.sum(query_vec) == 0:
            # 查询没有命中词汇表，降级返回空
            return []

        # 余弦相似度
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        query_unit = query_vec / query_norm

        results = []
        for i, doc_vec in enumerate(self._doc_vectors):
            doc_norm = np.linalg.norm(doc_vec)
            if doc_norm == 0:
                continue
            cosine = float(np.dot(query_unit, doc_vec / doc_norm))
            # 截断到 [0, 1]
            cosine = max(0.0, min(1.0, cosine))
            results.append({
                "id": self._doc_ids[i],
                "score": round(cosine, 4),
                "text": self._doc_texts[i],
            })

        # 按相似度降序排序
        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:top_k]

    # ── 序列化 ──

    def to_dict(self) -> dict:
        """序列化为 JSON 可存格式"""
        self.rebuild()
        return {
            "vocab": self.vocab.dump(),
            "doc_ids": self._doc_ids,
            "doc_texts": self._doc_texts,
            "doc_vectors": [v.tolist() for v in self._doc_vectors],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SemanticIndex":
        """从序列化数据恢复"""
        idx = cls()
        idx.vocab = _Vocabulary.load(data["vocab"])
        idx._doc_ids = data["doc_ids"]
        idx._doc_texts = data["doc_texts"]
        idx._doc_vectors = [np.array(v) for v in data["doc_vectors"]]
        idx._needs_rebuild = False
        return idx
