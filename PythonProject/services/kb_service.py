# services/kb_service.py - 知识库（问题/解决方案对）
# 支持关键词搜索 + 语义搜索（字符级 n-gram TF-IDF + 余弦相似度）
import sqlite3
import threading
import time
import logging
from datetime import datetime

from services.logger_service import get_logger
from services.embedding_service import SemanticIndex

logger = get_logger()

KB_TABLE = "knowledge_base"

# ── 全局语义索引（启动时构建，运行时内存缓存）──
_semantic_index: SemanticIndex | None = None
_index_lock = threading.Lock()
_index_last_update: float = 0.0
_INDEX_CACHE_TTL = 60.0  # 索引缓存 TTL（秒）


def _conn():
    from config import CHECKPOINT_DB
    return sqlite3.connect(CHECKPOINT_DB, timeout=10)


# ============================================================
# 语义索引管理
# ============================================================
def _build_semantic_index() -> SemanticIndex:
    """从数据库构建语义索引（全量重建）"""
    global _semantic_index

    start = time.time()
    idx = SemanticIndex()
    conn = _conn()
    cur = conn.execute(
        f"SELECT id, question, solution FROM {KB_TABLE} WHERE question IS NOT NULL AND solution IS NOT NULL"
    )
    rows = cur.fetchall()
    conn.close()

    for kb_id, question, solution in rows:
        combined = f"{question} {solution}"
        idx.add_document(kb_id, combined)

    idx.rebuild()
    elapsed = time.time() - start
    logger.info(f"📊 语义索引构建完成: {idx.count} 条文档, 词汇表 {idx.vocab.size}, 耗时 {elapsed:.1f}s")

    with _index_lock:
        _semantic_index = idx
        _index_last_update = time.time()

    return idx


def _get_semantic_index() -> SemanticIndex:
    """获取语义索引（如果缓存过期则重建）"""
    global _semantic_index

    now = time.time()
    if now - _index_last_update > _INDEX_CACHE_TTL:
        return _build_semantic_index()

    with _index_lock:
        return _semantic_index


def force_rebuild_index():
    """强制重建语义索引（手动触发）"""
    return _build_semantic_index()


# ============================================================
# 数据库初始化
# ============================================================
def init_kb_db():
    """初始化知识库表（幂等）"""
    conn = _conn()
    conn.execute(
        f"""CREATE TABLE IF NOT EXISTS {KB_TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            solution TEXT NOT NULL,
            tags TEXT,
            created_at TEXT
        )"""
    )
    conn.commit()
    conn.close()
    logger.info("✅ 知识库表已就绪")


# ============================================================
# 语义搜索
# ============================================================
def search_semantic(query: str, limit: int = 10) -> list[dict]:
    """
    语义搜索：基于字符级 n-gram TF-IDF + 余弦相似度。
    返回 {"id", "question", "solution", "tags", "score"}。
    """
    if not query or not query.strip():
        return []

    idx = _get_semantic_index()
    if idx.count == 0:
        return []

    raw_results = idx.search(query.strip(), top_k=limit * 3)

    if not raw_results:
        return []

    conn = _conn()
    results = []
    for r in raw_results[:limit]:
        cur = conn.execute(
            f"SELECT id, question, solution, tags FROM {KB_TABLE} WHERE id=?",
            (r["id"],)
        )
        row = cur.fetchone()
        if row:
            results.append({
                "id": row[0],
                "question": row[1],
                "solution": row[2],
                "tags": row[3],
                "score": r["score"],
            })
    conn.close()
    return results


# ============================================================
# 关键词搜索（原有）
# ============================================================
def search_kb(query, limit=5):
    """关键词检索知识库（LIKE 匹配 question/solution/tags）"""
    if not query or not query.strip():
        return []
    conn = _conn()
    pat = f"%{query}%"
    cur = conn.execute(
        f"SELECT id, question, solution, tags FROM {KB_TABLE} "
        f"WHERE question LIKE ? OR solution LIKE ? OR tags LIKE ? "
        f"ORDER BY id DESC LIMIT ?",
        (pat, pat, pat, limit),
    )
    rows = cur.fetchall()
    conn.close()
    return [{"id": r[0], "question": r[1], "solution": r[2], "tags": r[3]} for r in rows]


# ============================================================
# CRUD
# ============================================================
def list_kb():
    """列出全部知识库条目"""
    conn = _conn()
    cur = conn.execute(
        f"SELECT id, question, solution, tags, created_at FROM {KB_TABLE} ORDER BY id DESC"
    )
    rows = cur.fetchall()
    conn.close()
    return [
        {"id": r[0], "question": r[1], "solution": r[2], "tags": r[3], "created_at": r[4]}
        for r in rows
    ]


def save_kb(question, solution, tags=""):
    """新增知识库条目"""
    conn = _conn()
    conn.execute(
        f"INSERT INTO {KB_TABLE}(question, solution, tags, created_at) VALUES(?,?,?,?)",
        (question, solution, tags, datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
    )
    conn.commit()
    conn.close()

    # 标记索引需要重建
    with _index_lock:
        global _semantic_index, _index_last_update
        _index_last_update = 0.0

    logger.info(f"知识库新增条目: {question[:40]}")


def delete_kb(kb_id):
    """删除知识库条目"""
    conn = _conn()
    conn.execute(f"DELETE FROM {KB_TABLE} WHERE id=?", (kb_id,))
    conn.commit()
    conn.close()

    with _index_lock:
        global _semantic_index, _index_last_update
        _index_last_update = 0.0


def update_kb(kb_id, question, solution, tags=""):
    """更新知识库条目"""
    conn = _conn()
    conn.execute(
        f"UPDATE {KB_TABLE} SET question=?, solution=?, tags=? WHERE id=?",
        (question, solution, tags, kb_id),
    )
    conn.commit()
    conn.close()

    with _index_lock:
        global _semantic_index, _index_last_update
        _index_last_update = 0.0


def get_kb(kb_id):
    """获取单条知识库条目"""
    conn = _conn()
    cur = conn.execute(
        f"SELECT id, question, solution, tags, created_at FROM {KB_TABLE} WHERE id=?",
        (kb_id,)
    )
    row = cur.fetchone()
    conn.close()
    if row:
        return {"id": row[0], "question": row[1], "solution": row[2], "tags": row[3], "created_at": row[4]}
    return None
